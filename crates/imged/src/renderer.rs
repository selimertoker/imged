//! Turns the document into GPU textures for display.
//!
//! [`CanvasRenderer`] has two implementations: [`CpuTiledRenderer`]
//! composites on the CPU (rayon) and uploads 512px tiles through egui;
//! [`crate::renderer_gpu::GpuRenderer`] keeps the layers in GPU textures and
//! composites them in a shader (the default when wgpu is available).

use eframe::egui::{self, Color32, ColorImage, Painter, Rect, TextureHandle, TextureOptions};
use imged_core::composite::composite_document;
use imged_core::{Document, Rect as IRect};

use crate::state::DisplaySettings;
use crate::view::ViewXform;

pub trait CanvasRenderer {
    /// Brings textures up to date with the document (uses its dirty region).
    fn sync(&mut self, ctx: &egui::Context, doc: &mut Document, display: &DisplaySettings);
    /// Draws the canvas. `clip` is the visible canvas area on screen.
    fn paint(&self, painter: &Painter, view: &ViewXform);
    /// What renders the canvas (for the View menu).
    fn name(&self) -> String {
        "CPU".into()
    }
}

const TILE: u32 = 512;

struct Tile {
    rect: IRect,
    tex: TextureHandle,
}

#[derive(Default)]
pub struct CpuTiledRenderer {
    tiles: Vec<Tile>,
    size: (u32, u32),
    display: DisplaySettings,
}


impl CpuTiledRenderer {
    fn options(&self) -> TextureOptions {
        TextureOptions {
            // Crisp pixels when zoomed in, smooth (mipmapped) when zoomed out.
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Linear,
            mipmap_mode: Some(egui::TextureFilter::Linear),
            ..TextureOptions::LINEAR
        }
    }

    pub(crate) fn render_tile(doc: &Document, rect: IRect, display: &DisplaySettings) -> ColorImage {
        let img = composite_document(doc, rect);
        let gain = 2f32.powf(display.exposure);
        let srgb = display.srgb_encode;
        let pixels: Vec<Color32> = img
            .data
            .chunks_exact(4)
            .map(|p| {
                let enc = |v: f32| {
                    let v = (v * gain).clamp(0.0, 1.0);
                    let v = if srgb { linear_to_srgb(v) } else { v };
                    (v * 255.0 + 0.5) as u8
                };
                let a = (p[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                Color32::from_rgba_unmultiplied(enc(p[0]), enc(p[1]), enc(p[2]), a)
            })
            .collect();
        ColorImage::new([rect.w as usize, rect.h as usize], pixels)
    }
}

impl CanvasRenderer for CpuTiledRenderer {
    fn sync(&mut self, ctx: &egui::Context, doc: &mut Document, display: &DisplaySettings) {
        let size = (doc.width, doc.height);
        let full_rebuild = size != self.size || *display != self.display;
        let dirty = doc.take_dirty();
        if full_rebuild {
            self.size = size;
            self.display = *display;
            self.tiles.clear();
            let opts = self.options();
            let mut y = 0;
            while y < size.1 {
                let mut x = 0;
                while x < size.0 {
                    let rect = IRect::new(x as i32, y as i32, TILE.min(size.0 - x), TILE.min(size.1 - y));
                    let img = Self::render_tile(doc, rect, display);
                    let tex = ctx.load_texture(format!("canvas-{x}-{y}"), img, opts);
                    self.tiles.push(Tile { rect, tex });
                    x += TILE;
                }
                y += TILE;
            }
            return;
        }
        let Some(dirty) = dirty else { return };
        let opts = self.options();
        for t in &mut self.tiles {
            let r = t.rect.intersect(&dirty);
            if r.is_empty() {
                continue;
            }
            let img = Self::render_tile(doc, r, display);
            t.tex.set_partial([(r.x - t.rect.x) as usize, (r.y - t.rect.y) as usize], img, opts);
        }
    }

    fn paint(&self, painter: &Painter, view: &ViewXform) {
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        for t in &self.tiles {
            let r = view.rect_to_screen(t.rect);
            if painter.clip_rect().intersects(r) {
                painter.image(t.tex.id(), r, uv, Color32::WHITE);
            }
        }
    }
}

#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// A small 2x2 checker texture for drawing transparency, repeated via UVs.
pub fn checker_texture(ctx: &egui::Context) -> TextureHandle {
    let a = Color32::from_gray(204);
    let b = Color32::from_gray(153);
    let img = ColorImage::new([2, 2], vec![a, b, b, a]);
    ctx.load_texture(
        "checker",
        img,
        TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Nearest,
            wrap_mode: egui::TextureWrapMode::Repeat,
            ..TextureOptions::NEAREST
        },
    )
}

/// Paints a screen-space checkerboard (cells of `cell` points) into `rect`.
pub fn paint_checker(painter: &Painter, tex: &TextureHandle, rect: Rect, cell: f32) {
    let uv = Rect::from_min_size(
        egui::pos2(rect.min.x / (cell * 2.0), rect.min.y / (cell * 2.0)),
        rect.size() / (cell * 2.0),
    );
    painter.image(tex.id(), rect, uv, Color32::WHITE);
}
