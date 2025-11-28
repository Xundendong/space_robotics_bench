#[must_use = "You should call .show()"]
#[derive(Debug, Clone, Copy, PartialEq, typed_builder::TypedBuilder)]
pub struct ScrollableFramedCentralPanel {
    #[builder(default = 1024.0)]
    pub max_content_width: f32,
    #[builder(default = egui::Margin {left: 0, right: 8, top: 0, bottom: 0})]
    pub min_inner_margin: egui::Margin,
}

impl Default for ScrollableFramedCentralPanel {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ScrollableFramedCentralPanel {
    pub fn show<R>(
        self,
        ctx: &egui::Context,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::InnerResponse<R> {
        egui::CentralPanel::default().show(ctx, |ui| {
            // egui::ScrollArea::vertical()
            //     .show(ui, |ui| {
            let margin_x = (ctx.content_rect().width() - self.max_content_width).max(0.0) / 2.0;
            let inner_margin = egui::Margin {
                left: (margin_x as i8).max(self.min_inner_margin.left),
                right: (margin_x as i8).max(self.min_inner_margin.right),
                ..self.min_inner_margin
            };
            egui::Frame::default()
                .inner_margin(inner_margin)
                .show(ui, |ui| add_contents(ui))
                .inner
            // })
            // .inner
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub fn open_url_on_page(ctx: &egui::Context, page: crate::page::Page, same_tab: bool) {
    let target_url = format!("#{}", page.to_string().to_lowercase());
    ctx.open_url(if same_tab {
        egui::OpenUrl::same_tab(target_url)
    } else {
        egui::OpenUrl::new_tab(target_url)
    });
}

pub fn clickable_url(response: egui::Response, url: impl ToString) -> egui::Response {
    debug_assert!(response.sense.senses_click());

    if response.clicked() {
        response.ctx.open_url(egui::OpenUrl::same_tab(url));
    } else {
        #[cfg(target_arch = "wasm32")]
        if response.middle_clicked() {
            response.ctx.open_url(egui::OpenUrl::new_tab(url));
        }
    }
    response
}

pub fn strong_heading(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    ui.label(egui::RichText::new(text).heading().strong())
}

pub fn centered_strong_heading(
    ui: &mut egui::Ui,
    text: impl Into<String>,
) -> egui::InnerResponse<egui::Response> {
    ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
        strong_heading(ui, text)
    })
}

pub fn heading_sized(ui: &mut egui::Ui, text: impl Into<String>, size: f32) -> egui::Response {
    ui.label(egui::RichText::new(text).heading().size(size))
}

pub fn strong_heading_sized(
    ui: &mut egui::Ui,
    text: impl Into<String>,
    size: f32,
) -> egui::Response {
    ui.label(egui::RichText::new(text).heading().strong().size(size))
}

pub fn centered_strong_heading_sized(
    ui: &mut egui::Ui,
    text: impl Into<String>,
    size: f32,
) -> egui::InnerResponse<egui::Response> {
    ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
        crate::utils::egui::strong_heading_sized(ui, text, size)
    })
}
