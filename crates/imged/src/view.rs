//! Canvas view transform (zoom and pan) between image and screen coordinates.

use eframe::egui::{Pos2, Rect, Vec2 as EVec2};
use imged_core::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct ViewXform {
    /// Screen points per image pixel.
    pub zoom: f32,
    /// Screen position of the image origin.
    pub origin: Pos2,
}

impl Default for ViewXform {
    fn default() -> Self {
        ViewXform { zoom: 1.0, origin: Pos2::ZERO }
    }
}

pub const ZOOM_STEPS: &[f32] = &[
    0.02, 0.03, 0.05, 0.0625, 0.08, 0.1, 0.125, 0.167, 0.25, 0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    16.0, 24.0, 32.0, 48.0, 64.0,
];

impl ViewXform {
    pub fn to_screen(&self, p: Vec2) -> Pos2 {
        Pos2::new(self.origin.x + p.x * self.zoom, self.origin.y + p.y * self.zoom)
    }

    pub fn to_image(&self, p: Pos2) -> Vec2 {
        Vec2::new((p.x - self.origin.x) / self.zoom, (p.y - self.origin.y) / self.zoom)
    }

    /// Screen rectangle of an image-space rectangle.
    pub fn rect_to_screen(&self, r: imged_core::Rect) -> Rect {
        Rect::from_min_max(
            self.to_screen(Vec2::new(r.x as f32, r.y as f32)),
            self.to_screen(Vec2::new(r.right() as f32, r.bottom() as f32)),
        )
    }

    /// Zooms to `new_zoom`, keeping the image point under `anchor` fixed.
    pub fn zoom_around(&mut self, anchor: Pos2, new_zoom: f32) {
        let new_zoom = new_zoom.clamp(ZOOM_STEPS[0], *ZOOM_STEPS.last().unwrap());
        let img = self.to_image(anchor);
        self.zoom = new_zoom;
        self.origin = Pos2::new(anchor.x - img.x * new_zoom, anchor.y - img.y * new_zoom);
    }

    pub fn step(&self, up: bool) -> f32 {
        if up {
            ZOOM_STEPS.iter().copied().find(|z| *z > self.zoom * 1.001).unwrap_or(self.zoom)
        } else {
            ZOOM_STEPS.iter().rev().copied().find(|z| *z < self.zoom * 0.999).unwrap_or(self.zoom)
        }
    }

    /// Fits an image of `size` into `viewport` (never above 100% unless tiny).
    pub fn fit(&mut self, viewport: Rect, size: (u32, u32)) {
        let margin = 24.0;
        let avail = (viewport.size() - EVec2::splat(margin * 2.0)).max(EVec2::splat(16.0));
        let z = (avail.x / size.0.max(1) as f32).min(avail.y / size.1.max(1) as f32);
        self.zoom = if z >= 1.0 { z.floor().clamp(1.0, 8.0) } else { z };
        self.center(viewport, size);
    }

    pub fn center(&mut self, viewport: Rect, size: (u32, u32)) {
        let s = EVec2::new(size.0 as f32, size.1 as f32) * self.zoom;
        self.origin = (viewport.center() - s * 0.5).round();
    }
}
