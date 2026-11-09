use eframe::egui::{self, Color32, CursorIcon, Painter, Stroke, StrokeKind, Ui};
use imged_core::composite::average_color;
use imged_core::{Color, Document, Rect, Vec2};

use super::{Button, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::state::ColorSlot;
use crate::view::ViewXform;

/// Square sample sizes offered for averaging (1 = single pixel).
const SIZES: [u32; 6] = [1, 3, 5, 11, 31, 101];

pub struct EyedropperTool {
    sample_merged: bool,
    /// Side of the averaged square, in pixels.
    size: u32,
    hover: Option<Vec2>,
}

impl Default for EyedropperTool {
    fn default() -> Self {
        EyedropperTool { sample_merged: true, size: 1, hover: None }
    }
}

impl EyedropperTool {
    fn area(&self, p: Vec2) -> Rect {
        let n = self.size.max(1) as i32;
        Rect::new(p.x.floor() as i32 - (n - 1) / 2, p.y.floor() as i32 - (n - 1) / 2, n as u32, n as u32)
    }

    /// The (averaged) color at `p`, or `None` outside the image.
    fn sample(&self, doc: &Document, p: Vec2) -> Option<Color> {
        let area = self.area(p);
        if !doc.canvas_rect().contains(p.x.floor() as i32, p.y.floor() as i32) && self.sample_merged {
            return None;
        }
        if self.sample_merged {
            return average_color(doc, area).map(Color::from_px);
        }
        // Active layer only, transparency included; averaged premultiplied.
        let l = doc.active_layer();
        let r = area.intersect(&l.bounds());
        if r.is_empty() {
            return None;
        }
        let mut acc = [0.0f64; 4];
        for y in r.y..r.bottom() {
            for x in r.x..r.right() {
                let p = l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1);
                let a = p[3] as f64;
                for c in 0..3 {
                    acc[c] += p[c] as f64 * a;
                }
                acc[3] += a;
            }
        }
        let n = r.area() as f64;
        Some(if acc[3] <= 1e-12 {
            Color::TRANSPARENT
        } else {
            Color::rgba((acc[0] / acc[3]) as f32, (acc[1] / acc[3]) as f32, (acc[2] / acc[3]) as f32, (acc[3] / n) as f32)
        })
    }
}

impl Tool for EyedropperTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Eyedropper
    }

    fn options_ui(&mut self, ui: &mut Ui, _ctx: &mut ToolCtx) {
        ui.checkbox(&mut self.sample_merged, "Sample all layers")
            .on_hover_text("Off: sample only the active layer (including its transparency)");
        ui.horizontal(|ui| {
            ui.label("Average");
            egui::ComboBox::from_id_salt("pick-size")
                .selected_text(if self.size == 1 { "Single pixel".to_string() } else { format!("{0}×{0}", self.size) })
                .show_ui(ui, |ui| {
                    for s in SIZES {
                        let label = if s == 1 { "Single pixel".to_string() } else { format!("{s}×{s} average") };
                        ui.selectable_value(&mut self.size, s, label);
                    }
                });
        });
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        self.hover = Some(ev.pos);
        if !matches!(ev.kind, PointerKind::Press | PointerKind::Drag | PointerKind::Release) {
            return;
        }
        let Some(c) = self.sample(ctx.doc, ev.pos) else { return };
        // Show (and edit) the slot that was just set, so the panel never
        // keeps showing the other one.
        match ev.button {
            Button::Primary => {
                ctx.colors.primary = c;
                ctx.colors.editing = ColorSlot::Primary;
            }
            Button::Secondary => {
                ctx.colors.secondary = c;
                ctx.colors.editing = ColorSlot::Secondary;
            }
        }
    }

    fn hover_color(&self, doc: &Document, pos: Vec2) -> Option<Color> {
        self.sample(doc, pos)
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, _doc: &Document) {
        if let (Some(p), true) = (self.hover, self.size > 1) {
            let r = view.rect_to_screen(self.area(p));
            painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Outside);
            painter.rect_stroke(r.expand(1.0), 0.0, Stroke::new(1.0, Color32::WHITE), StrokeKind::Outside);
        }
    }

    fn cursor(&self) -> CursorIcon {
        CursorIcon::Cell
    }

    fn adjust_size(&mut self, bigger: bool) {
        let i = SIZES.iter().position(|s| *s == self.size).unwrap_or(0);
        let i = if bigger { (i + 1).min(SIZES.len() - 1) } else { i.saturating_sub(1) };
        self.size = SIZES[i];
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "picker.sample_all", &mut self.sample_merged);
    }
}
