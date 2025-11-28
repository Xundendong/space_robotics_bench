import sys
import os
import json
from datetime import datetime
from pathlib import Path

# =========================================================
# 🚀 1. 核心启动器 (必须最先导入，否则报错)
# =========================================================
from srb.core.app import AppLauncher

# =========================================================
# 🛠️ 工具函数：递归修改分辨率
# =========================================================
def force_resolution_recursive(obj, target_w=1280, target_h=720, visited=None):
    """
    遍历配置对象，找到所有分辨率设置并强制修改为高清。
    """
    if visited is None: visited = set()
    if id(obj) in visited: return
    visited.add(id(obj))

    # 检查是否有 width/height 属性
    if hasattr(obj, "width") and hasattr(obj, "height"):
        try:
            if isinstance(obj.width, int) and isinstance(obj.height, int):
                # 如果是低分辨率，强制修改
                if obj.width <= 256: 
                    print(f"   🔧 [Auto-Fix] 升级分辨率: {obj.width}x{obj.height} -> {target_w}x{target_h}")
                    obj.width = target_w
                    obj.height = target_h
        except:
            pass

    # 递归遍历属性
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if not k.startswith("__"):
                force_resolution_recursive(v, target_w, target_h, visited)
    
    # 递归遍历字典/列表
    if isinstance(obj, list):
        for item in obj:
            force_resolution_recursive(item, target_w, target_h, visited)
    elif isinstance(obj, dict):
        for v in obj.values():
            force_resolution_recursive(v, target_w, target_h, visited)

# =========================================================
# 🎬 主函数
# =========================================================
def main():
    # --- 配置区域 ---
    TASK_ID = "srb/sample_collection_visual"
    HEADLESS = True  # 采集数据建议开启
    NUM_EPISODES = 50
    SAVE_DIR = "./dataset_vla_hd"
    MAX_STEPS = 400
    
    # 📷 目标分辨率
    CAM_WIDTH = 2560
    CAM_HEIGHT = 1440
    # ----------------

    print(f"🚀 正在启动仿真内核 (Headless={HEADLESS})...")
    launcher = AppLauncher(headless=HEADLESS, enable_cameras=True)

    # ---------------------------------------------------------
    # 🩹 补丁: 绕过 Isaac Sim 版本检查 (必须在 AppLauncher 之后)
    # ---------------------------------------------------------
    import srb.utils.isaacsim
    srb.utils.isaacsim.is_isaacsim_initialized = lambda: True
    print("🩹 版本检查补丁已应用")

    # ---------------------------------------------------------
    # 📦 延迟导入 SRB 和 Gym (防止崩溃)
    # ---------------------------------------------------------
    import gymnasium
    import torch
    import cv2
    import numpy as np
    import srb.tasks 
    from srb.tasks.manipulation.sample_collection.task_visual import VisualTaskCfg

    print(f"🔄 正在配置环境: {TASK_ID} ...")
    env_cfg = VisualTaskCfg()
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.scene, "procedural_assets"):
        env_cfg.scene.procedural_assets = True

    # 🔥【关键步骤】应用分辨率强制修改
    print("✨ 正在应用高清画质补丁...")
    force_resolution_recursive(env_cfg, CAM_WIDTH, CAM_HEIGHT)
    print("✨ 补丁应用完成。")

    # 创建环境
    env = gymnasium.make(TASK_ID, cfg=env_cfg)

    # 准备保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(SAVE_DIR) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ 环境创建成功！开始采集高清数据...")
    
    # 预热
    print("🔥 正在预热物理引擎...")
    env.reset()
    for _ in range(20):
        # 注意：这里使用 step 的 5 个返回值，虽然预热我们不关心结果
        env.step(torch.zeros((1, 7), device=env.unwrapped.device))

    debug_printed = False

    for episode_idx in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        step_count = 0
        ep_dir = save_dir / f"episode_{episode_idx:05d}"
        ep_dir.mkdir(exist_ok=True)
        meta_data = []
        
        grasp_stage = 0 

        while not done and step_count < MAX_STEPS:
            # -----------------------------------------------------
            # 🔍 1. 获取真值 (含维度修复)
            # -----------------------------------------------------
            ee_pos_batch = env.unwrapped._tf_end_effector.data.target_pos_w
            rock_pos_batch = env.unwrapped.scene["sample"].data.root_pos_w

            # 修复：[1, 3] -> [3]
            ee_pos = ee_pos_batch.view(-1)[:3]
            rock_pos = rock_pos_batch.view(-1)[:3]

            # -----------------------------------------------------
            # 🧠 2. 专家策略
            # -----------------------------------------------------
            target_pos = rock_pos.clone()
            gripper_cmd = 1.0 
            
            error = target_pos - ee_pos
            dist_xy = torch.norm(error[:2])
            dist_z = error[2]

            if grasp_stage == 0: 
                target_pos[2] += 0.20
                if dist_xy < 0.05: grasp_stage = 1
            elif grasp_stage == 1: 
                target_pos[2] += 0.02
                if dist_z < 0.04: grasp_stage = 2
            elif grasp_stage == 2: 
                gripper_cmd = -1.0
                if step_count % 60 > 50: grasp_stage = 3
            elif grasp_stage == 3: 
                target_pos[2] += 0.4
                gripper_cmd = -1.0

            kp = 4.0
            vel_cmd = (target_pos - ee_pos) * kp
            vel_cmd = torch.clamp(vel_cmd, -1.0, 1.0)

            action = torch.zeros((1, 7), device=env.unwrapped.device)
            action[0, :3] = vel_cmd
            action[0, 6] = gripper_cmd

            # -----------------------------------------------------
            # 💾 3. 保存高清数据
            # -----------------------------------------------------
            rgb_tensor = None
            
            # 优先用手腕相机，没有则用基座相机
            if "image_wrist" in obs:
                rgb_tensor = obs["image_wrist"]
            elif "image_base" in obs:
                rgb_tensor = obs["image_base"]

            if rgb_tensor is not None:
                # [DEBUG] 第一次打印确认分辨率
                if not debug_printed:
                    print(f"\n📸 [INFO] 正在采集分辨率: {rgb_tensor.shape} (应包含 {CAM_HEIGHT}x{CAM_WIDTH})")
                    debug_printed = True

                # 1. 转 Numpy [1, H, W, 4] -> [H, W, 4]
                if rgb_tensor.dim() == 4:
                    rgb = rgb_tensor[0].cpu().numpy()
                else:
                    rgb = rgb_tensor.cpu().numpy()
                
                # 2. 处理字典情况 (有些环境返回 dict)
                if isinstance(rgb, dict) and "rgb" in rgb:
                    rgb = rgb["rgb"]

                # 3. 类型处理 Float -> Int
                if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                
                # 4. 颜色空间转换 (处理 RGBA 或 RGB)
                if rgb.shape[-1] == 4:
                    bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                else:
                    bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                img_name = f"{step_count:04d}.jpg"
                
                # 🔥 保存 JPG 质量为 100 (无损)
                cv2.imwrite(
                    str(ep_dir / img_name), 
                    bgr_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
                
                act_list = action[0].cpu().tolist()
                meta_data.append({
                    "image_path": img_name,
                    "action": act_list,
                    "instruction": "Pick up the rock",
                    "state_ee": ee_pos.cpu().tolist(),
                    "state_rock": rock_pos.cpu().tolist()
                })

            # -----------------------------------------------------
            # ⚙️ 4. 步进 (使用 5 个返回值)
            # -----------------------------------------------------
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        # 写入 JSON
        if len(meta_data) > 0:
            with open(ep_dir / "data.json", "w") as f:
                json.dump(meta_data, f, indent=2)
            print(f"Episode {episode_idx} Finished. Steps: {step_count} | Saved {len(meta_data)} HD frames.")
        else:
            print(f"Episode {episode_idx} Finished. ⚠️ No data saved.")

    print("🎉 所有任务完成！")
    env.close()
    launcher.app.close()

if __name__ == "__main__":
    main()
