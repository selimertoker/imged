//! Tilt (perspective) and Shear tools: drag an edge handle of the active
//! layer's bounds.

use eframe::egui::{self, Color32, CursorIcon, Painter, Stroke, StrokeKind, Ui};
use imged_core::ops::transform::{plan_warp, rect_corners, scope_pivot, shear_quad, tilt_angle_for, tilt_quad, Edge, Scope};
use imged_core::{Command, Document, Rect, Vec2};

use super::{PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::ui::widgets::{icon_button_sized, Icon};
use crate::view::ViewXform;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WarpMode {
    Tilt,
    Shear,
}

pub struct WarpTool {
    mode: WarpMode,
    edge: Edge,
    /// Tilt angle in degrees.
    degrees: f32,
    /// Tilt camera distance as a multiple of the layer's larger side.
    focal_factor: f32,
    /// Shear amount in pixels.
    amount: f32,
    dragging: bool,
    press: Vec2,
    press_amount: f32,
    hover_edge: Option<Edge>,
    /// Bounds the current preview was computed for.
    bounds: Rect,
    previewed: Option<(Edge, f32, Scope)>,
    scope: Scope,
}

/// An arrow pointing toward the edge.
fn edge_icon(e: Edge) -> Icon {
    match e {
        Edge::Top => Icon::Arrow(0, -1),
        Edge::Right => Icon::Arrow(1, 0),
        Edge::Bottom => Icon::Arrow(0, 1),
        Edge::Left => Icon::Arrow(-1, 0),
    }
}

impl WarpTool {
    pub fn new(mode: WarpMode) -> Self {
        WarpTool {
            mode,
            edge: Edge::Top,
            degrees: 0.0,
            focal_factor: 2.0,
            amount: 0.0,
            dragging: false,
            press: Vec2::ZERO,
            press_amount: 0.0,
            hover_edge: None,
            bounds: Rect::default(),
            previewed: None,
            scope: Scope::Layer,
        }
    }

    fn value(&self) -> f32 {
        match self.mode {
            WarpMode::Tilt => self.degrees,
            WarpMode::Shear => self.amount,
        }
    }

    fn focal(&self, b: Rect) -> f32 {
        b.w.max(b.h) as f32 * self.focal_factor
    }

    /// Quad in canvas coords for the current parameters.
    fn quad(&self, b: Rect) -> [Vec2; 4] {
        let (w, h) = (b.w as f32, b.h as f32);
        let q = match self.mode {
            WarpMode::Tilt => tilt_quad(w, h, self.edge, self.degrees, self.focal(b)),
            WarpMode::Shear => shear_quad(w, h, self.edge, self.amount),
        };
        q.map(|p| p + Vec2::new(b.x as f32, b.y as f32))
    }

    fn handles(b: Rect) -> [(Edge, Vec2); 4] {
        let c = rect_corners(b.w as f32, b.h as f32).map(|p| p + Vec2::new(b.x as f32, b.y as f32));
        Edge::ALL.map(|e| {
            let (i, j) = e.corners();
            (e, (c[i] + c[j]) * 0.5)
        })
    }

    fn hit(&self, b: Rect, p: Vec2, zoom: f32) -> Option<Edge> {
        let tol = 10.0 / zoom;
        Self::handles(b)
            .into_iter()
            .map(|(e, h)| (e, (h - p).len()))
            .filter(|(_, d)| *d <= tol)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(e, _)| e)
    }

    fn update_preview(&mut self, ctx: &mut ToolCtx) {
        let b = scope_pivot(ctx.doc, self.scope);
        self.bounds = b;
        let key = (self.edge, self.value(), self.scope);
        if self.previewed == Some(key) {
            return;
        }
        self.previewed = Some(key);
        if self.value().abs() < 1e-3 {
            ctx.doc.clear_preview();
            return;
        }
        match plan_warp(ctx.doc, "Warp", self.quad(b), self.scope) {
            Some(plan) => plan.preview(ctx.doc),
            None => ctx.doc.clear_preview(),
        }
    }

    fn reset(&mut self, ctx: &mut ToolCtx) {
        self.degrees = 0.0;
        self.amount = 0.0;
        self.dragging = false;
        self.previewed = None;
        ctx.doc.clear_preview();
    }

    fn apply(&mut self, ctx: &mut ToolCtx) {
        if self.value().abs() < 1e-3 {
            return;
        }
        ctx.doc.clear_preview();
        let b = scope_pivot(ctx.doc, self.scope);
        let scope = self.scope;
        let cmd = match self.mode {
            WarpMode::Tilt => Command::Tilt { edge: self.edge, degrees: self.degrees, focal: self.focal(b), scope },
            WarpMode::Shear => Command::Shear { edge: self.edge, amount: self.amount, scope },
        };
        ctx.exec(cmd);
        self.degrees = 0.0;
        self.amount = 0.0;
        self.previewed = None;
    }
}

impl Tool for WarpTool {
    fn kind(&self) -> ToolKind {
        match self.mode {
            WarpMode::Tilt => ToolKind::Tilt,
            WarpMode::Shear => ToolKind::Shear,
        }
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        let mut changed = false;
        if self.scope != ctx.scope() {
            self.scope = ctx.scope();
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.label(if self.mode == WarpMode::Tilt { "Tilting edge" } else { "Sliding edge" });
            ui.spacing_mut().item_spacing.x = 2.0;
            for e in Edge::ALL {
                if icon_button_sized(ui, edge_icon(e), self.edge == e, "", 22.0).clicked() && self.edge != e {
                    self.edge = e;
                    changed = true;
                }
            }
        });
        match self.mode {
            WarpMode::Tilt => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.degrees, -80.0..=80.0).update_while_editing(false).suffix("°").text("Angle"))
                    .on_hover_text("Positive tilts the edge away from you, negative toward you")
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.focal_factor, 0.5..=10.0).update_while_editing(false).logarithmic(true).text("Camera distance"))
                    .on_hover_text("Smaller = stronger perspective")
                    .changed();
            }
            WarpMode::Shear => {
                let b = scope_pivot(ctx.doc, self.scope);
                let lim = b.w.max(b.h) as f32;
                changed |= ui.add(egui::Slider::new(&mut self.amount, -lim..=lim).update_while_editing(false).suffix(" px").text("Offset")).changed();
            }
        }
        ui.horizontal(|ui| {
            if ui.button("Apply").on_hover_text("Enter").clicked() {
                self.apply(ctx);
            }
            if ui.button("Reset").on_hover_text("Esc").clicked() {
                self.reset(ctx);
            }
        });
        ui.weak("Dragging a handle applies on release.");
        if changed {
            self.previewed = None;
            self.update_preview(ctx);
        }
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        let b = scope_pivot(ctx.doc, self.scope);
        match ev.kind {
            PointerKind::Hover => self.hover_edge = self.hit(b, ev.raw, ctx.zoom),
            PointerKind::Press => {
                if let Some(e) = self.hit(b, ev.raw, ctx.zoom) {
                    if e != self.edge {
                        self.degrees = 0.0;
                        self.amount = 0.0;
                    }
                    self.edge = e;
                    self.dragging = true;
                    self.press = ev.raw;
                    self.press_amount = self.amount;
                }
            }
            PointerKind::Drag if self.dragging => {
                let local = ev.raw - Vec2::new(b.x as f32, b.y as f32);
                match self.mode {
                    WarpMode::Tilt => {
                        self.degrees = tilt_angle_for(b.w as f32, b.h as f32, self.edge, self.focal(b), local);
                        if ev.shift {
                            self.degrees = (self.degrees / 5.0).round() * 5.0;
                        }
                    }
                    WarpMode::Shear => {
                        self.amount = self.press_amount + (ev.raw - self.press).dot(self.edge.along());
                        self.amount = self.amount.round();
                    }
                }
                self.update_preview(ctx);
            }
            PointerKind::Release if self.dragging => {
                self.dragging = false;
                self.apply(ctx);
            }
            _ => {}
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        let b = scope_pivot(doc, self.scope);
        if b.is_empty() {
            return;
        }
        let accent = Color32::from_rgb(0, 160, 255);
        if self.value().abs() > 1e-3 {
            let q = self.quad(b);
            let pts: Vec<_> = q.iter().chain(std::iter::once(&q[0])).map(|p| view.to_screen(*p)).collect();
            painter.add(egui::Shape::line(pts, Stroke::new(1.5, accent)));
        }
        painter.rect_stroke(view.rect_to_screen(b), 0.0, Stroke::new(1.0, Color32::from_gray(140)), StrokeKind::Outside);
        for (e, h) in Self::handles(b) {
            let c = view.to_screen(h);
            let active = Some(e) == self.hover_edge || (self.dragging && e == self.edge);
            let r = egui::Rect::from_center_size(c, egui::vec2(9.0, 9.0));
            painter.rect_filled(r, 1.0, if active { accent } else { Color32::WHITE });
            painter.rect_stroke(r, 1.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Inside);
        }
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        self.reset(ctx);
    }

    fn commit(&mut self, ctx: &mut ToolCtx) {
        self.apply(ctx);
    }

    fn cursor(&self) -> CursorIcon {
        if self.hover_edge.is_some() || self.dragging {
            CursorIcon::Grab
        } else {
            CursorIcon::Default
        }
    }
}
