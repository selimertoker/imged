use eframe::egui::{self, Color32, Painter, Stroke, Ui};
use imged_core::ops::paint::Brush;
use imged_core::{Document, Mask, Px, Vec2};

use super::{Button, LivePaint, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

pub struct BrushTool {
    size: f32,
    hardness: f32,
    stroke: Option<Stroke_>,
    /// End of the previous stroke (canvas coords) for Shift+click lines.
    last_end: Option<Vec2>,
    hover: Option<Vec2>,
}

struct Stroke_ {
    live: LivePaint,
    mask: Mask,
    last: Vec2,
    color: Px,
    button: Button,
}

impl Default for BrushTool {
    fn default() -> Self {
        BrushTool { size: 6.0, hardness: 1.0, stroke: None, last_end: None, hover: None }
    }
}

impl BrushTool {
    fn brush(&self, aa: bool) -> Brush {
        Brush { size: self.size, hardness: self.hardness, antialias: aa }
    }
}

impl Tool for BrushTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Brush
    }

    fn options_ui(&mut self, ui: &mut Ui, _ctx: &mut ToolCtx) {
        ui.add(egui::Slider::new(&mut self.size, 1.0..=200.0).update_while_editing(false).logarithmic(true).text("Size"));
        let mut h = self.hardness * 100.0;
        if ui.add(egui::Slider::new(&mut h, 0.0..=100.0).update_while_editing(false).suffix("%").text("Hardness")).changed() {
            self.hardness = h / 100.0;
        }
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        self.hover = Some(ev.pos);
        let brush = self.brush(ctx.settings.antialias);
        match ev.kind {
            PointerKind::Press => {
                ctx.begin_drawing(ToolKind::Brush, ev.button);
                let live = LivePaint::begin(ctx.doc, ctx.settings.grow_layers);
                let mut mask = Mask::new(live.layer_rect());
                let p = live.local(ev.pos);
                let color = ctx.color_for(ev.button).px();
                let region = match (ev.shift, self.last_end) {
                    (true, Some(prev)) => brush.segment(&mut mask, live.local(prev), p),
                    _ => {
                        brush.dab(&mut mask, p);
                        brush.dab_bounds(p)
                    }
                };
                let mut s = Stroke_ { live, mask, last: ev.pos, color, button: ev.button };
                s.live.update(ctx.doc, &[(&s.mask, color)], region, ctx.settings.paint_mode);
                self.stroke = Some(s);
            }
            PointerKind::Drag => {
                if let Some(s) = &mut self.stroke {
                    let (a, b) = (s.live.local(s.last), s.live.local(ev.pos));
                    let region = brush.segment(&mut s.mask, a, b);
                    s.last = ev.pos;
                    s.live.update(ctx.doc, &[(&s.mask, s.color)], region, ctx.settings.paint_mode);
                }
            }
            PointerKind::Release => {
                if let Some(s) = self.stroke.take() {
                    let color = ctx.color_for(s.button);
                    if s.live.commit(ctx.doc, "Brush") {
                        ctx.colors.mark_used(color);
                    }
                    self.last_end = Some(ev.pos);
                }
            }
            PointerKind::Hover => {}
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, _doc: &Document) {
        if let Some(p) = self.hover {
            let r = (self.size * 0.5).max(0.5) * view.zoom;
            let c = view.to_screen(p);
            painter.circle_stroke(c, r, Stroke::new(1.0, Color32::BLACK));
            painter.circle_stroke(c, r + 1.0, Stroke::new(1.0, Color32::WHITE));
        }
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        if let Some(s) = self.stroke.take() {
            s.live.cancel(ctx.doc);
        }
    }

    fn adjust_size(&mut self, bigger: bool) {
        self.size = if bigger { (self.size * 1.25).max(self.size + 1.0) } else { (self.size / 1.25).min(self.size - 1.0) }
            .clamp(1.0, 500.0)
            .round();
    }
}
