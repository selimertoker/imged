use eframe::egui::{self, Color32, CursorIcon, Painter, Stroke, StrokeKind, Ui};
use imged_core::layer::LayerId;
use imged_core::ops::edges::{MoveSnapper, RectSnap};
use imged_core::{Command, Document, Rect, Vec2};

use super::{PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

struct MoveDrag {
    start: Vec2,
    layers: Vec<(LayerId, (i32, i32))>,
    bounds: Rect,
    snapper: Option<MoveSnapper>,
}

#[derive(Default)]
pub struct MoveTool {
    drag: Option<MoveDrag>,
    delta: (i32, i32),
    snap: RectSnap,
}

impl Tool for MoveTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Move
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        let l = ctx.doc.active_layer();
        ui.label(format!("Position: {}, {}", l.offset.0 + self.delta.0, l.offset.1 + self.delta.1));
        ui.horizontal(|ui| {
            if ui.button("Center").on_hover_text("Center the layer on the canvas").clicked() {
                let b = ctx.doc.targets_bounds();
                let dx = (ctx.doc.width as i32 - b.w as i32) / 2 - b.x;
                let dy = (ctx.doc.height as i32 - b.h as i32) / 2 - b.y;
                ctx.exec(Command::MoveLayerBy { dx, dy });
            }
            if ui.button("To origin").clicked() {
                let b = ctx.doc.targets_bounds();
                ctx.exec(Command::MoveLayerBy { dx: -b.x, dy: -b.y });
            }
        });
    }

    fn snap_option(&self) -> bool {
        true
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Press => {
                ctx.doc.clear_preview();
                let ids = ctx.doc.targets();
                let layers = ids.iter().filter_map(|id| Some((*id, ctx.doc.layer(*id)?.offset))).collect();
                let snapper = ctx.settings.snapping.then(|| MoveSnapper::new(ctx.doc, &ids, ctx.settings.snap_threshold));
                self.drag = Some(MoveDrag { start: ev.raw, layers, bounds: ctx.doc.targets_bounds(), snapper });
                self.delta = (0, 0);
                self.snap = RectSnap::default();
            }
            PointerKind::Drag => {
                let Some(drag) = &self.drag else { return };
                let d = ev.raw - drag.start;
                let (mut dx, mut dy) = (d.x.round() as i32, d.y.round() as i32);
                if ev.shift {
                    if dx.abs() > dy.abs() {
                        dy = 0;
                    } else {
                        dx = 0;
                    }
                }
                self.snap = RectSnap::default();
                if let (Some(s), false) = (&drag.snapper, ev.alt) {
                    let radius = ctx.settings.snap_params(ctx.zoom).radius;
                    let snap = s.snap(drag.bounds.translate(dx, dy), radius);
                    // With Shift, only the free axis snaps.
                    let free_x = !ev.shift || dx != 0;
                    let free_y = !ev.shift || dy != 0;
                    if free_x {
                        dx += snap.dx;
                        self.snap.guide_x = snap.guide_x;
                    }
                    if free_y {
                        dy += snap.dy;
                        self.snap.guide_y = snap.guide_y;
                    }
                }
                if (dx, dy) == self.delta {
                    return;
                }
                self.delta = (dx, dy);
                for (id, off) in &drag.layers {
                    if let Some(px) = ctx.doc.layer(*id).map(|l| l.pixels.clone()) {
                        ctx.doc.set_preview(*id, px, (off.0 + dx, off.1 + dy));
                    }
                }
            }
            PointerKind::Release => {
                if self.drag.take().is_some() {
                    ctx.doc.clear_preview();
                    let (dx, dy) = std::mem::take(&mut self.delta);
                    self.snap = RectSnap::default();
                    ctx.exec(Command::MoveLayerBy { dx, dy });
                }
            }
            PointerKind::Hover => {}
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        let b = match &self.drag {
            Some(d) => d.bounds.translate(self.delta.0, self.delta.1),
            None => doc.targets_bounds(),
        };
        if !b.is_empty() {
            let r = view.rect_to_screen(b);
            painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::from_rgb(0, 160, 255)), StrokeKind::Outside);
        }
        let clip = painter.clip_rect();
        let guide = Stroke::new(1.0, Color32::from_rgb(255, 0, 200));
        if let Some(x) = self.snap.guide_x {
            let sx = view.to_screen(Vec2::new(x as f32, 0.0)).x;
            painter.line_segment([egui::pos2(sx, clip.top()), egui::pos2(sx, clip.bottom())], guide);
        }
        if let Some(y) = self.snap.guide_y {
            let sy = view.to_screen(Vec2::new(0.0, y as f32)).y;
            painter.line_segment([egui::pos2(clip.left(), sy), egui::pos2(clip.right(), sy)], guide);
        }
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        if self.drag.take().is_some() {
            self.delta = (0, 0);
            self.snap = RectSnap::default();
            ctx.doc.clear_preview();
        }
    }

    fn cursor(&self) -> CursorIcon {
        CursorIcon::Move
    }

    fn status(&self) -> Option<String> {
        let d = self.drag.as_ref()?;
        let b = d.bounds.translate(self.delta.0, self.delta.1);
        Some(format!("{}, {}  (Δ {}, {})", b.x, b.y, self.delta.0, self.delta.1))
    }
}
