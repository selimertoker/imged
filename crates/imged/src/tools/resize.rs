//! Resize tool: drag the corners of the active layer/group (or the selection,
//! or the whole image, following the Transform section's "Apply to").
//!
//! With "Move all corners" (the default) the frame stays a rectangle: a
//! dragged corner scales it from the opposite corner, obeying "Keep ratio",
//! and the size can also be typed. Without it, each corner moves on its own
//! (free warp). Dragging inside moves the frame. Changes are previewed live
//! and applied with Apply / Enter (or when switching tools).

use eframe::egui::{self, Color32, CursorIcon, Painter, RichText, Stroke, StrokeKind, Ui};
use imged_core::ops::transform::{plan_resize_to, plan_warp, rect_corners, scope_pivot, effective_scope, ResizeFilter, Scope};
use imged_core::{Command, Document, Rect, Vec2};

use super::{PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

#[derive(Clone, Copy, PartialEq)]
enum Grab {
    Corner(usize),
    Whole(Vec2),
}

pub struct ResizeTool {
    /// Move every corner (resize) instead of one (free warp).
    resize_mode: bool,
    filter: ResizeFilter,
    /// Current frame (TL, TR, BR, BL) in canvas coords, when changed.
    quad: Option<[Vec2; 4]>,
    /// The pivot rectangle (and scope) the frame started from.
    base: Rect,
    base_scope: Scope,
    grab: Option<Grab>,
    hover: Option<Grab>,
    /// What the shown preview was computed for.
    previewed: Option<([Vec2; 4], bool, ResizeFilter, Scope)>,
    /// The scope as last seen by `options_ui`, for pointer handling.
    scope: Scope,
}

impl Default for ResizeTool {
    fn default() -> Self {
        ResizeTool {
            resize_mode: true,
            filter: ResizeFilter::Bicubic,
            quad: None,
            base: Rect::default(),
            base_scope: Scope::Layer,
            grab: None,
            hover: None,
            previewed: None,
            scope: Scope::Layer,
        }
    }
}

fn corners_of(r: Rect) -> [Vec2; 4] {
    rect_corners(r.w as f32, r.h as f32).map(|p| p + Vec2::new(r.x as f32, r.y as f32))
}

fn bounds(q: &[Vec2; 4]) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in q {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    Rect::from_corners(x0.round() as i32, y0.round() as i32, x1.round() as i32, y1.round() as i32)
}

fn inside(p: Vec2, q: &[Vec2; 4]) -> bool {
    // Works for convex and simple concave quads (even-odd rule).
    let mut c = false;
    let mut j = 3;
    for i in 0..4 {
        let (a, b) = (q[i], q[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            c = !c;
        }
        j = i;
    }
    c
}

impl ResizeTool {
    fn frame(&self, doc: &Document) -> [Vec2; 4] {
        self.quad.unwrap_or_else(|| corners_of(scope_pivot(doc, self.scope)))
    }

    fn whole_image(&self, doc: &Document) -> bool {
        effective_scope(doc, self.scope) == Scope::Image
    }

    fn hit(&self, doc: &Document, p: Vec2, zoom: f32) -> Option<Grab> {
        let q = self.frame(doc);
        let tol = 10.0 / zoom;
        let best = (0..4).map(|i| (i, (q[i] - p).len())).filter(|(_, d)| *d <= tol).min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = best {
            return Some(Grab::Corner(i));
        }
        // Moving the whole frame means nothing when resizing the whole image.
        if self.resize_mode && self.whole_image(doc) {
            return None;
        }
        inside(p, &q).then_some(Grab::Whole(p))
    }

    /// Starts editing from the current pivot rectangle.
    fn begin(&mut self, doc: &Document) {
        if self.quad.is_none() {
            self.base = scope_pivot(doc, self.scope);
            self.base_scope = self.scope;
            self.quad = Some(corners_of(self.base));
        }
    }

    fn preview(&mut self, ctx: &mut ToolCtx) {
        let Some(q) = self.quad.filter(|_| self.changed()) else {
            if self.previewed.take().is_some() {
                ctx.doc.clear_preview();
            }
            return;
        };
        let key = (q, self.resize_mode, self.filter, self.scope);
        if self.previewed == Some(key) {
            return;
        }
        self.previewed = Some(key);
        let plan = if self.resize_mode {
            plan_resize_to(ctx.doc, bounds(&q), self.filter, self.scope)
        } else {
            plan_warp(ctx.doc, "Warp", q, self.scope)
        };
        match plan {
            Some(p) => p.preview(ctx.doc),
            None => ctx.doc.clear_preview(),
        }
    }

    fn changed(&self) -> bool {
        self.quad.is_some_and(|q| q != corners_of(self.base))
    }

    fn reset(&mut self, ctx: &mut ToolCtx) {
        self.quad = None;
        self.grab = None;
        if self.previewed.take().is_some() {
            ctx.doc.clear_preview();
        }
    }

    fn apply(&mut self, ctx: &mut ToolCtx) {
        let changed = self.changed();
        let Some(q) = self.quad.take() else { return };
        self.grab = None;
        if self.previewed.take().is_some() {
            ctx.doc.clear_preview();
        }
        if !changed {
            return;
        }
        let scope = self.base_scope;
        let cmd = if self.resize_mode {
            Command::ResizeTo { rect: bounds(&q), filter: self.filter, scope }
        } else {
            Command::Warp { quad: q, scope }
        };
        ctx.exec(cmd);
    }

    /// Resize mode: corner `i` goes to `p`, the opposite corner stays.
    fn resize_corner(&self, i: usize, p: Vec2, keep_ratio: bool, q: [Vec2; 4]) -> [Vec2; 4] {
        let o = q[(i + 2) % 4];
        let mut d = p - o;
        if keep_ratio && self.base.w > 0 && self.base.h > 0 {
            let ratio = self.base.w as f32 / self.base.h as f32;
            // Follow whichever direction the pointer moved more, keeping signs.
            let (ax, ay) = (d.x.abs(), d.y.abs());
            if ax / ratio > ay {
                d.y = d.y.signum() * ax / ratio;
            } else {
                d.x = d.x.signum() * ay * ratio;
            }
            if d.x == 0.0 || d.y == 0.0 {
                d = Vec2::new(ratio.max(1.0), (1.0 / ratio).max(1.0));
            }
        }
        let p = o + d;
        let (x0, x1) = (o.x.min(p.x), o.x.max(p.x).max(o.x.min(p.x) + 1.0));
        let (y0, y1) = (o.y.min(p.y), o.y.max(p.y).max(o.y.min(p.y) + 1.0));
        [Vec2::new(x0, y0), Vec2::new(x1, y0), Vec2::new(x1, y1), Vec2::new(x0, y1)]
    }

    /// Width/height fields, percentage and presets (resize mode).
    fn size_fields(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        let cur = self.quad.map(|q| bounds(&q)).unwrap_or_else(|| scope_pivot(ctx.doc, self.scope));
        let base = if self.quad.is_some() { self.base } else { cur };
        if base.is_empty() {
            return;
        }
        let (bw, bh) = (base.w as f32, base.h as f32);
        let keep = ctx.settings.keep_ratio;
        let (mut w, mut h) = (cur.w, cur.h);
        let mut set = None;
        ui.horizontal(|ui| {
            let rw = crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut w).range(1..=65536).suffix(" px").update_while_editing(false));
            if rw.changed() {
                let nh = if keep { ((w as f32 * bh / bw).round() as u32).max(1) } else { h };
                set = Some((w, nh));
            }
            ui.label("×");
            let rh = crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut h).range(1..=65536).suffix(" px").update_while_editing(false));
            if rh.changed() {
                let nw = if keep { ((h as f32 * bw / bh).round() as u32).max(1) } else { w };
                set = Some((nw, h));
            }
        });
        ui.horizontal_wrapped(|ui| {
            let mut pct = cur.w as f32 / bw * 100.0;
            let r = crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut pct)
                    .range(0.01..=10000.0)
                    .clamp_existing_to_range(false)
                    .suffix(" %")
                    .speed(1.0)
                    .max_decimals(1)
                    .update_while_editing(false),
            );
            let mut scale = r.changed().then_some(pct);
            for p in [25.0, 50.0, 200.0] {
                if ui.small_button(format!("{p}%")).clicked() {
                    scale = Some(p);
                }
            }
            if let Some(p) = scale {
                set = Some((((bw * p / 100.0).round() as u32).max(1), ((bh * p / 100.0).round() as u32).max(1)));
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut ctx.settings.keep_ratio, "Keep ratio");
            egui::ComboBox::from_id_salt("resize-filter").selected_text(self.filter.label()).show_ui(ui, |ui| {
                for f in ResizeFilter::ALL {
                    ui.selectable_value(&mut self.filter, f, f.label());
                }
            });
        });
        if let Some((nw, nh)) = set {
            self.begin(ctx.doc);
            let r = Rect::new(cur.x, cur.y, nw, nh);
            self.quad = Some(corners_of(r));
        }
    }
}

impl Tool for ResizeTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Resize
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        // Follow the "Apply to" choice and changes made elsewhere (undo, …).
        self.scope = ctx.scope();
        if self.quad.is_some() && self.grab.is_none() {
            let pivot = scope_pivot(ctx.doc, self.scope);
            if self.scope != self.base_scope || pivot != self.base {
                self.reset(ctx);
            }
        }
        let was = self.resize_mode;
        ui.checkbox(&mut self.resize_mode, "Move all corners (resize)")
            .on_hover_text("Off: each corner moves on its own (free warp / perspective)");
        if was != self.resize_mode {
            self.reset(ctx);
        }
        if self.resize_mode {
            self.size_fields(ui, ctx);
        } else if let Some(q) = self.quad {
            let b = bounds(&q);
            ui.label(RichText::new(format!("{} × {} at {}, {}", b.w, b.h, b.x, b.y)).small());
        }
        if self.resize_mode && self.whole_image(ctx.doc) {
            ui.label(RichText::new("Resizes the whole image and the canvas.").small().weak());
        }
        ui.horizontal(|ui| {
            let can = self.changed();
            if ui.add_enabled(can, egui::Button::new("Apply")).on_hover_text("Enter").clicked() {
                self.apply(ctx);
            }
            if ui.add_enabled(self.quad.is_some(), egui::Button::new("Reset")).on_hover_text("Esc").clicked() {
                self.reset(ctx);
            }
        });
        self.preview(ctx);
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Hover => self.hover = self.hit(ctx.doc, ev.raw, ctx.zoom),
            PointerKind::Press => {
                let hit = self.hit(ctx.doc, ev.raw, ctx.zoom);
                if hit.is_some() {
                    self.begin(ctx.doc);
                }
                self.grab = hit;
            }
            PointerKind::Drag => {
                let (Some(grab), Some(q)) = (self.grab, self.quad) else { return };
                let new = match grab {
                    Grab::Corner(i) if self.resize_mode => self.resize_corner(i, ev.pos, ctx.settings.keep_ratio, q),
                    Grab::Corner(i) => {
                        let mut q = q;
                        q[i] = ev.pos;
                        q
                    }
                    Grab::Whole(from) => {
                        let d = ev.raw - from;
                        let d = Vec2::new(d.x.round(), d.y.round());
                        if d == Vec2::ZERO {
                            return;
                        }
                        self.grab = Some(Grab::Whole(from + d));
                        q.map(|p| p + d)
                    }
                };
                if Some(new) != self.quad {
                    self.quad = Some(new);
                    self.preview(ctx);
                }
            }
            PointerKind::Release => self.grab = None,
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        let q = self.frame(doc);
        let accent = Color32::from_rgb(0, 160, 255);
        let pts: Vec<_> = q.iter().chain(std::iter::once(&q[0])).map(|p| view.to_screen(*p)).collect();
        painter.add(egui::Shape::line(pts, Stroke::new(1.5, accent)));
        for (i, p) in q.iter().enumerate() {
            let c = view.to_screen(*p);
            let active = matches!(self.hover, Some(Grab::Corner(j)) if j == i) || matches!(self.grab, Some(Grab::Corner(j)) if j == i);
            let r = egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0));
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

    fn deactivate(&mut self, ctx: &mut ToolCtx) {
        self.apply(ctx);
    }

    fn uses_snapping(&self) -> bool {
        true
    }

    fn cursor(&self) -> CursorIcon {
        match self.grab.or(self.hover) {
            Some(Grab::Corner(_)) => CursorIcon::Crosshair,
            Some(Grab::Whole(_)) => CursorIcon::Move,
            None => CursorIcon::Default,
        }
    }

    fn status(&self) -> Option<String> {
        let b = bounds(&self.quad?);
        Some(format!("{} × {} px", b.w, b.h))
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "resize.all_corners", &mut self.resize_mode);
    }
}
