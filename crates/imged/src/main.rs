//! imged — a simple layer-based image editor.
//!
//! Usage: `imged [FILE]...` (the first file opens as the document, the rest
//! as layers).

mod actions;
mod app;
mod clipboard;
mod magick;
mod persist;
mod project;
mod renderer;
mod renderer_gpu;
mod state;
mod tools;
mod ui;
mod view;

use std::path::PathBuf;

fn main() -> eframe::Result {
    let files: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("imged")
            .with_app_id("imged")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("imged", options, Box::new(move |cc| Ok(Box::new(app::ImgedApp::new(cc, files)))))
}
