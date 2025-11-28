use egui::Color32;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Difficulty {
    Demo,
    Easy,
    Medium,
    Challenging,
}

impl std::fmt::Display for Difficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Demo => write!(f, "Demo"),
            Self::Easy => write!(f, "Easy"),
            Self::Medium => write!(f, "Medium"),
            Self::Challenging => write!(f, "Challenging"),
        }
    }
}

impl Difficulty {
    pub fn get_text_color(self, strong: bool) -> Color32 {
        match self {
            Self::Demo => {
                if strong {
                    Color32::from_rgb(245, 245, 245)
                } else {
                    Color32::from_rgb(170, 170, 170)
                }
            }
            Self::Easy => {
                if strong {
                    Color32::from_rgb(46, 220, 117)
                } else {
                    Color32::from_rgb(77, 171, 35)
                }
            }
            Self::Medium => {
                if strong {
                    Color32::from_rgb(211, 218, 60)
                } else {
                    Color32::from_rgb(211, 100, 0)
                }
            }
            Self::Challenging => {
                if strong {
                    Color32::from_rgb(219, 39, 40)
                } else {
                    Color32::from_rgb(132, 0, 32)
                }
            }
        }
    }

    pub fn set_theme(self, ui: &mut egui::Ui, theme: egui::Theme) {
        match self {
            Self::Demo => match theme {
                egui::Theme::Light => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(245, 245, 245);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(170, 170, 170);
                }
                egui::Theme::Dark => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(170, 170, 170);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(245, 245, 245);
                }
            },
            Self::Easy => match theme {
                egui::Theme::Light => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(46, 220, 117);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(77, 171, 35);
                }
                egui::Theme::Dark => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(77, 171, 35);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(46, 220, 117);
                }
            },
            Self::Medium => match theme {
                egui::Theme::Light => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(211, 218, 60);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(211, 100, 0);
                }
                egui::Theme::Dark => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(211, 100, 0);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(211, 218, 60);
                }
            },
            Self::Challenging => match theme {
                egui::Theme::Light => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(219, 39, 40);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(132, 0, 32);
                }
                egui::Theme::Dark => {
                    ui.style_mut().visuals.widgets.inactive.weak_bg_fill =
                        Color32::from_rgb(132, 0, 32);
                    ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                        Color32::from_rgb(219, 39, 40);
                }
            },
        }
    }
}
