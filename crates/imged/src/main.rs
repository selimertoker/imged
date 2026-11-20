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

/// Asks for everything the GPU supports (eframe's default caps OpenGL at
/// WebGL 2 limits, without compute shaders, and textures at 8192 px): the
/// GPU outline select needs compute shaders, the canvas renderer big textures.
fn wgpu_options() -> eframe::egui_wgpu::WgpuConfiguration {
    use eframe::egui_wgpu::{wgpu, WgpuConfiguration, WgpuSetup};
    let mut c = WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(new) = &mut c.wgpu_setup {
        new.device_descriptor = std::sync::Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("imged"),
            required_limits: adapter.limits(),
            ..Default::default()
        });
    }
    c
}

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
        wgpu_options: wgpu_options(),
        ..Default::default()
    };
    eframe::run_native("imged", options, Box::new(move |cc| Ok(Box::new(app::ImgedApp::new(cc, files)))))
}
