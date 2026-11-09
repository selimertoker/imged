//! Rotate tool: turn the active layer/group (or the selection, or the whole
//! image, following "Apply to") by any angle around an anchor. The anchor
//! is drawn on the canvas and can be dragged (it snaps to edges, the middle
//! of the canvas/selection and the corners, edge middles and center of what
//! is rotated). Drag anywhere else to turn (snapping to multiples of 45°), or type the angle; the result is
//! previewed live and applied with Apply / Enter (or when switching tools).

use eframe::egui::{self, Color32, CursorIcon, Painter, Pos2, RichText, Stroke, Ui};
use imged_core::ops::transform::{effective_scope, plan_rotate, rect_center, rect_corners, scope_pivot, Scope};
use imged_core::{Command, Document, Rect, Vec2};

use super::{PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    /// (pointer angle, degrees) when the drag started.
    Turn(f32, f32),
    Anchor,
}

pub struct RotateTool {
    /// Counter-clockwise, in degrees.
    degrees: f32,
    smooth: bool,
    /// The point to turn around (canvas coords); `None` = the middle.
    anchor: Option<Vec2>,
    drag: Option<Drag>,
    hover_anchor: bool,
    previewed: Option<(f32, bool, Option<(u32, u32)>, Scope, Rect)>,
    scope: Scope,
}

impl Default for RotateTool {
    fn default() -> Self {
        RotateTool { degrees: 0.0, smooth: true, anchor: None, drag: None, hover_anchor: false, previewed: None, scope: Scope::Layer }
    }
}

/// Draws a small ship's anchor centered on `c`.
fn paint_anchor(painter: &Painter, c: Pos2, color: Color32, hot: bool) {
    let r = if hot { 11.0 } else { 9.0 };
    for (w, col) in [(3.5, Color32::BLACK), (1.6, color)] {
        let s = Stroke::new(w, col);
        let ring = c + egui::vec2(0.0, -r * 0.75);
        painter.circle_stroke(ring, r * 0.22, s);
        painter.line_segment([ring + egui::vec2(0.0, r * 0.22), c + egui::vec2(0.0, r * 0.95)], s);
        painter.line_segment([c + egui::vec2(-r * 0.45, -r * 0.3), c + egui::vec2(r * 0.45, -r * 0.3)], s);
        let n = 10;
        let arc: Vec<Pos2> = (0..=n)
            .map(|i| {
                let a = std::f32::consts::PI * (0.12 + 0.76 * i as f32 / n as f32);
                c + egui::vec2(a.cos() * r * 0.8, a.sin() * r * 0.8 + r * 0.15)
            })
            .collect();
        let (first, last) = (arc[0], arc[n]);
        painter.add(egui::Shape::line(arc, s));
        painter.line_segment([first, first + egui::vec2(-r * 0.05, -r * 0.35)], s);
        painter.line_segment([last, last + egui::vec2(r * 0.05, -r * 0.35)], s);
    }
}

impl RotateTool {
    fn pivot(&self, doc: &Document) -> Rect {
        scope_pivot(doc, self.scope)
    }

    fn whole_image(&self, doc: &Document) -> bool {
        effective_scope(doc, self.scope) == Scope::Image
    }

    /// The point the rotation turns around (the canvas center for the whole image).
    fn center(&self, doc: &Document) -> Vec2 {
        let pv = self.pivot(doc);
        if self.whole_image(doc) {
            rect_center(pv)
        } else {
            self.anchor.unwrap_or_else(|| rect_center(pv))
        }
    }

    fn preview(&mut self, ctx: &mut ToolCtx) {
        let active = self.degrees.rem_euclid(360.0).abs() > 1e-4;
        if !active {
            if self.previewed.take().is_some() {
                ctx.doc.clear_preview();
            }
            return;
        }
        let a = self.anchor.map(|p| (p.x.to_bits(), p.y.to_bits()));
        let key = (self.degrees, self.smooth, a, self.scope, self.pivot(ctx.doc));
        if self.previewed == Some(key) {
            return;
        }
        self.previewed = Some(key);
        match plan_rotate(ctx.doc, self.degrees, self.smooth, self.scope, self.anchor) {
            Some(p) => p.preview(ctx.doc),
            None => ctx.doc.clear_preview(),
        }
    }

    fn reset(&mut self, ctx: &mut ToolCtx) {
        self.degrees = 0.0;
        self.drag = None;
        if self.previewed.take().is_some() {
            ctx.doc.clear_preview();
        }
    }

    fn apply(&mut self, ctx: &mut ToolCtx) {
        self.drag = None;
        if self.previewed.take().is_some() {
            ctx.doc.clear_preview();
        }
        if self.degrees.rem_euclid(360.0).abs() < 1e-4 {
            self.degrees = 0.0;
            return;
        }
        let cmd = Command::Rotate { degrees: self.degrees, smooth: self.smooth, scope: self.scope, center: self.anchor };
        self.degrees = 0.0;
        ctx.exec(cmd);
    }

    fn pointer_angle(&self, doc: &Document, p: Vec2) -> f32 {
        let d = p - self.center(doc);
        // Screen y points down: counter-clockwise on screen is a negative atan2 change.
        (-d.y).atan2(d.x).to_degrees()
    }

    fn near_anchor(&self, doc: &Document, p: Vec2, zoom: f32) -> bool {
        !self.whole_image(doc) && (p - self.center(doc)).len() <= 12.0 / zoom
    }

    /// With snapping on, the anchor also snaps to the corners, edge middles
    /// and center of what is rotated.
    fn snap_anchor(&self, doc: &Document, raw: Vec2, snapped: Vec2, zoom: f32, snapping: bool) -> Vec2 {
        if !snapping {
            return raw;
        }
        let pv = self.pivot(doc);
        let tol = 10.0 / zoom;
        let pts = [0.0, 0.5, 1.0].iter().flat_map(|fx| {
            [0.0, 0.5, 1.0].iter().map(move |fy| Vec2::new(pv.x as f32 + pv.w as f32 * fx, pv.y as f32 + pv.h as f32 * fy))
        });
        pts.filter(|p| (*p - raw).len() <= tol).min_by(|a, b| (*a - raw).len().total_cmp(&(*b - raw).len())).unwrap_or(snapped)
    }
}

impl Tool for RotateTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Rotate
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        if self.scope != ctx.scope() {
            self.scope = ctx.scope();
            self.anchor = None;
            self.previewed = None;
        }
        ui.add(
            egui::Slider::new(&mut self.degrees, -180.0..=180.0)
                .update_while_editing(false)
                .clamping(egui::SliderClamping::Never)
                .suffix("°")
                .text("Angle"),
        )
        .on_hover_text(
            "Counter-clockwise; negative values turn clockwise. With snapping on, dragging sticks to multiples of 45° (Alt: free); Shift snaps to 15°.",
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.smooth, "Smooth").on_hover_text("Bilinear filtering (off: nearest pixel, for pixel art)");
            let whole = self.whole_image(ctx.doc);
            if ui
                .add_enabled(!whole && self.anchor.is_some(), egui::Button::new("Center anchor").small())
                .on_hover_text("Put the anchor back in the middle")
                .clicked()
            {
                self.anchor = None;
                self.previewed = None;
            }
        });
        if self.whole_image(ctx.doc) {
            ui.label(RichText::new("The whole image turns around its center; the canvas grows to fit.").small().weak());
        } else {
            ui.label(RichText::new("Drag the anchor to choose the point it turns around.").small().weak());
        }
        ui.horizontal(|ui| {
            let active = self.degrees.rem_euclid(360.0).abs() > 1e-4;
            if ui.add_enabled(active, egui::Button::new("Apply")).on_hover_text("Enter").clicked() {
                self.apply(ctx);
            }
            if ui.add_enabled(self.degrees != 0.0, egui::Button::new("Reset")).on_hover_text("Esc").clicked() {
                self.reset(ctx);
            }
        });
        self.preview(ctx);
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Hover => self.hover_anchor = self.near_anchor(ctx.doc, ev.raw, ctx.zoom),
            PointerKind::Press => {
                self.drag = Some(if self.near_anchor(ctx.doc, ev.raw, ctx.zoom) {
                    Drag::Anchor
                } else {
                    Drag::Turn(self.pointer_angle(ctx.doc, ev.raw), self.degrees)
                });
            }
            PointerKind::Drag => match self.drag {
                Some(Drag::Anchor) => {
                    let p = self.snap_anchor(ctx.doc, ev.raw, ev.pos, ctx.zoom, ctx.settings.snapping);
                    if self.anchor != Some(p) {
                        self.anchor = Some(p);
                        self.preview(ctx);
                    }
                }
                Some(Drag::Turn(a0, d0)) => {
                    let mut deg = d0 + (self.pointer_angle(ctx.doc, ev.raw) - a0);
                    deg = (deg + 180.0).rem_euclid(360.0) - 180.0;
                    let near45 = (deg / 45.0).round() * 45.0;
                    deg = if ev.shift {
                        (deg / 15.0).round() * 15.0
                    } else if ctx.settings.snapping && !ev.alt && (deg - near45).abs() <= 4.0 {
                        // Snapping: multiples of 45° pull the angle in.
                        near45
                    } else {
                        (deg * 10.0).round() / 10.0
                    };
                    if deg == -180.0 {
                        deg = 180.0;
                    }
                    if deg != self.degrees {
                        self.degrees = deg;
                        self.preview(ctx);
                    }
                }
                None => {}
            },
            PointerKind::Release => self.drag = None,
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        let pv = self.pivot(doc);
        if pv.is_empty() {
            return;
        }
        let accent = Color32::from_rgb(0, 160, 255);
        let c = self.center(doc);
        let a = -self.degrees.to_radians();
        let q = rect_corners(pv.w as f32, pv.h as f32).map(|p| {
            let p = p + Vec2::new(pv.x as f32, pv.y as f32);
            view.to_screen(c + (p - c).rotate(a))
        });
        let pts: Vec<_> = q.iter().chain(std::iter::once(&q[0])).copied().collect();
        painter.add(egui::Shape::line(pts, Stroke::new(1.5, accent)));
        let hot = self.hover_anchor || self.drag == Some(Drag::Anchor);
        paint_anchor(painter, view.to_screen(c), if hot { accent } else { Color32::WHITE }, hot);
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        self.reset(ctx);
    }

    fn commit(&mut self, ctx: &mut ToolCtx) {
        self.apply(ctx);
    }

    fn deactivate(&mut self, ctx: &mut ToolCtx) {
        self.apply(ctx);
    }

    fn uses_snapping(&self) -> bool {
        // Only the anchor snaps to points; turning snaps to multiples of 45°.
        match self.drag {
            Some(Drag::Anchor) => true,
            Some(Drag::Turn(..)) => false,
            None => self.hover_anchor,
        }
    }

    fn snap_option(&self) -> bool {
        true
    }

    fn cursor(&self) -> CursorIcon {
        match self.drag {
            Some(Drag::Anchor) => CursorIcon::Grabbing,
            Some(Drag::Turn(..)) => CursorIcon::Grabbing,
            None if self.hover_anchor => CursorIcon::Move,
            None => CursorIcon::Grab,
        }
    }

    fn status(&self) -> Option<String> {
        (self.degrees != 0.0).then(|| format!("{:.1}°", self.degrees))
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "rotate.smooth", &mut self.smooth);
    }
}
