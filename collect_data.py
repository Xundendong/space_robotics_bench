import sys
import os
import gymnasium
import torch
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# 1. 核心启动器
from srb.core.app import AppLauncher

def main():
    # --- 配置区域 ---
    TASK_ID = "srb/sample_collection_visual"
    HEADLESS = True
    NUM_EPISODES = 50
    SAVE_DIR = "./dataset_vla"
    MAX_STEPS = 400
    # ----------------

    print(f"🚀 正在启动仿真 (Headless={HEADLESS})...")
    launcher = AppLauncher(headless=HEADLESS, enable_cameras=True)

    # ---------------------------------------------------------
    # 补丁
    import srb.utils.isaacsim
    srb.utils.isaacsim.is_isaacsim_initialized = lambda: True
    # ---------------------------------------------------------

    print("🔄 正在加载 SRB 任务模块...")
    import srb.tasks 
    from srb.tasks.manipulation.sample_collection.task_visual import VisualTaskCfg

    print(f"🔄 正在配置环境: {TASK_ID} ...")
    env_cfg = VisualTaskCfg()
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.scene, "procedural_assets"):
        env_cfg.scene.procedural_assets = True

    env = gymnasium.make(TASK_ID, cfg=env_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(SAVE_DIR) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ 环境创建成功！开始采集...")
    
    # 预热一下，防止第一帧卡顿
    print("🔥 正在预热物理引擎...")
    for _ in range(20):
        env.step(torch.zeros((1, 7), device=env.unwrapped.device))

    for episode_idx in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        step_count = 0
        ep_dir = save_dir / f"episode_{episode_idx:05d}"
        ep_dir.mkdir(exist_ok=True)
        meta_data = []
        
        # 状态机
        grasp_stage = 0 

        while not done and step_count < MAX_STEPS:
            # -----------------------------------------------------
            # 🔍 1. 稳健的数据获取 (防御性编程)
            # -----------------------------------------------------
            # 获取末端位置
            ee_tensor = obs["proprio"]["fk_pos_end_effector"]
            
            # 第一次运行时打印形状，方便调试
            if episode_idx == 0 and step_count == 0:
                print(f"📊 [Debug] EE Shape: {ee_tensor.shape}")
            
            # 自动降维处理
            if ee_tensor.dim() == 3: # [N, 1, 3]
                ee_pos = ee_tensor[0, 0]
            elif ee_tensor.dim() == 2: # [N, 3]
                ee_pos = ee_tensor[0]
            else:
                ee_pos = ee_tensor.view(-1)[0:3] # 强行展平取前3个
                
            # 获取岩石位置
            # 注意：直接从 physics state 获取，这是真值
            try:
                rock_tensor = env.unwrapped.scene["sample"].data.root_pos_w
                if episode_idx == 0 and step_count == 0:
                    print(f"📊 [Debug] Rock Shape: {rock_tensor.shape}")
                
                if rock_tensor.dim() >= 2:
                    target_pos = rock_tensor[0].clone()
                else:
                    target_pos = rock_tensor.clone()
            except KeyError:
                # 如果找不到 key，打印 keys 帮你看
                if step_count == 0:
                    print(f"⚠️ 警告: 场景中没找到 'sample'，可用物体: {env.unwrapped.scene.keys()}")
                target_pos = ee_pos.clone() # 找不到就原地不动

            # -----------------------------------------------------
            # 🧠 2. 专家策略逻辑
            # -----------------------------------------------------
            gripper_cmd = 1.0 # 1=Open, -1=Close
            
            # 计算相对位置
            error = target_pos - ee_pos
            dist_xy = torch.norm(error[:2])
            dist_z = error[2]

            if grasp_stage == 0: # 靠近上方
                target_pos[2] += 0.20
                if dist_xy < 0.05: grasp_stage = 1
            
            elif grasp_stage == 1: # 下降
                target_pos[2] += 0.02 # 略高于物体中心
                if dist_z < 0.05: grasp_stage = 2
            
            elif grasp_stage == 2: # 抓取
                gripper_cmd = -1.0
                # 简单的计时逻辑：尝试抓 20 步
                if step_count % 50 > 40: 
                    grasp_stage = 3
            
            elif grasp_stage == 3: # 抬起
                target_pos[2] += 0.4
                gripper_cmd = -1.0

            # 简单的 P 控制器
            kp = 4.0
            vel_cmd = (target_pos - ee_pos) * kp
            vel_cmd = torch.clamp(vel_cmd, -1.0, 1.0)

            # 组装动作 [vx, vy, vz, wx, wy, wz, gripper]
            action = torch.zeros((1, 7), device=env.unwrapped.device)
            action[0, :3] = vel_cmd
            action[0, 6] = gripper_cmd

            # -----------------------------------------------------
            # 💾 3. 保存数据
            # -----------------------------------------------------
            # 图片
            if "visual" in obs and "rgb" in obs["visual"]:
                rgb = obs["visual"]["rgb"][0].cpu().numpy()
                img_name = f"{step_count:04d}.jpg"
                cv2.imwrite(str(ep_dir / img_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                
                # 动作
                act_list = action[0].cpu().tolist()
                meta_data.append({
                    "image_path": img_name,
                    "action": act_list,
                    "instruction": "Pick up the rock",
                    "state_ee": ee_pos.cpu().tolist() # 顺便存一下状态方便debug
                })

            # -----------------------------------------------------
            # ⚙️ 4. 步进
            # -----------------------------------------------------
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        # 保存 JSON
        with open(ep_dir / "data.json", "w") as f:
            json.dump(meta_data, f, indent=2)
            
        print(f"Episode {episode_idx} Finished. Steps: {step_count}")

    print("🎉 采集任务完成！")
    env.close()
    launcher.app.close()

if __name__ == "__main__":
    main()
