//! Pen: a square, hard, non-antialiased tip that sets pixels to the exact
//! color (alpha included), regardless of the paint mode.

use eframe::egui::{self, Color32, Painter, Stroke, StrokeKind, Ui};
use imged_core::ops::paint::{PaintMode, Pen};
use imged_core::{Document, Mask, Px, Vec2};

use super::{Button, LivePaint, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

pub struct PenTool {
    size: u32,
    stroke: Option<PenStroke>,
    last_end: Option<Vec2>,
    hover: Option<Vec2>,
}

struct PenStroke {
    live: LivePaint,
    mask: Mask,
    last: Vec2,
    color: Px,
    button: Button,
}

impl Default for PenTool {
    fn default() -> Self {
        PenTool { size: 1, stroke: None, last_end: None, hover: None }
    }
}

impl Tool for PenTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Pen
    }

    fn options_ui(&mut self, ui: &mut Ui, _ctx: &mut ToolCtx) {
        ui.add(egui::Slider::new(&mut self.size, 1..=64).update_while_editing(false).text("Size").suffix(" px"));
        ui.label(egui::RichText::new("Always replaces pixels; ignores the Paint mode and Smooth settings.").small().weak());
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        self.hover = Some(ev.pos);
        let pen = Pen { size: self.size };
        match ev.kind {
            PointerKind::Press => {
                ctx.begin_drawing(ToolKind::Pen, ev.button);
                let live = LivePaint::begin(ctx.doc, ctx.settings.grow_layers);
                let mut mask = Mask::new(live.layer_rect());
                let p = live.local(ev.pos);
                let color = ctx.color_for(ev.button).px();
                let region = match (ev.shift, self.last_end) {
                    (true, Some(prev)) => pen.segment(&mut mask, live.local(prev), p),
                    _ => pen.dab(&mut mask, p),
                };
                let mut s = PenStroke { live, mask, last: ev.pos, color, button: ev.button };
                s.live.update(ctx.doc, &[(&s.mask, color)], region, PaintMode::Replace);
                self.stroke = Some(s);
            }
            PointerKind::Drag => {
                if let Some(s) = &mut self.stroke {
                    let region = pen.segment(&mut s.mask, s.live.local(s.last), s.live.local(ev.pos));
                    s.last = ev.pos;
                    s.live.update(ctx.doc, &[(&s.mask, s.color)], region, PaintMode::Replace);
                }
            }
            PointerKind::Release => {
                if let Some(s) = self.stroke.take() {
                    let color = ctx.color_for(s.button);
                    if s.live.commit(ctx.doc, "Pen") {
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
            let r = view.rect_to_screen(Pen { size: self.size }.cell(p));
            painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Outside);
            painter.rect_stroke(r.expand(1.0), 0.0, Stroke::new(1.0, Color32::WHITE), StrokeKind::Outside);
        }
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        if let Some(s) = self.stroke.take() {
            s.live.cancel(ctx.doc);
        }
    }

    fn adjust_size(&mut self, bigger: bool) {
        self.size = if bigger { self.size + 1 } else { self.size.saturating_sub(1) }.clamp(1, 64);
    }
}
