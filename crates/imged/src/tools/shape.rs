use std::sync::Arc;

use eframe::egui::{self, Color32, Painter, Stroke, Ui};
use imged_core::object::ObjectKind;
use imged_core::ops::paint::{Shape, ShapeFill, ShapeKind};
use imged_core::{Color, Command, Document, LayerId, Vec2};

use super::{Button, LivePaint, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::ui::widgets::{icon_button_sized, Icon, ShapeIcon};
use crate::view::ViewXform;

const CHOICES: [(ShapeIcon, &str); 6] = [
    (ShapeIcon::Line, "Line"),
    (ShapeIcon::Rectangle, "Rectangle"),
    (ShapeIcon::Square, "Square"),
    (ShapeIcon::Ellipse, "Ellipse"),
    (ShapeIcon::Circle, "Circle"),
    (ShapeIcon::Arrow, "Arrow"),
];

pub struct ShapeTool {
    choice: ShapeIcon,
    kind: ShapeKind,
    width: f32,
    fill: ShapeFill,
    head_scale: f32,
    rounded: bool,
    corner_radius: f32,
    pixel_aligned: bool,
    from_center: bool,
    drag: Option<Drag>,
    /// The shape layer whose settings the options show and edit, and its
    /// object as last loaded or applied (a different one means it was
    /// changed elsewhere, e.g. undone: the options reload).
    editing: Option<(LayerId, ObjectKind)>,
    /// Dragging a handle (or the body) of the active shape layer.
    edit: Option<EditDrag>,
}

#[derive(Clone, Copy, PartialEq)]
enum Handle {
    A,
    B,
    Body,
}

struct EditDrag {
    id: LayerId,
    handle: Handle,
    /// Pointer at the press, and the object then.
    start: Vec2,
    orig: ObjectKind,
    now: ObjectKind,
}

/// The active layer's shape object, if it is a shape layer.
fn active_shape(doc: &Document) -> Option<(LayerId, Shape, Color, Color)> {
    let l = doc.active_layer();
    match l.object_now()? {
        ObjectKind::Shape { shape, color, fill_color } => Some((l.id, shape, color, fill_color)),
        _ => None,
    }
}

fn color32(c: Color) -> Color32 {
    let [r, g, b, a] = c.to_rgba8();
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

fn from_color32(c: Color32) -> Color {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    Color::from_rgba8(r, g, b, a)
}

struct Drag {
    start: Vec2,
    end: Vec2,
    button: Button,
    live: LivePaint,
}

impl Default for ShapeTool {
    fn default() -> Self {
        ShapeTool {
            choice: ShapeIcon::Rectangle,
            kind: ShapeKind::Rectangle,
            width: 3.0,
            fill: ShapeFill::Outline,
            head_scale: 4.0,
            rounded: false,
            corner_radius: 8.0,
            pixel_aligned: true,
            from_center: false,
            drag: None,
            editing: None,
            edit: None,
        }
    }
}

/// Shift: squares/circles for boxes, 45° steps for lines.
fn constrain(kind: ShapeKind, a: Vec2, b: Vec2) -> Vec2 {
    let d = b - a;
    match kind {
        ShapeKind::Rectangle | ShapeKind::Ellipse => {
            let s = d.x.abs().max(d.y.abs());
            Vec2::new(a.x + s * d.x.signum(), a.y + s * d.y.signum())
        }
        ShapeKind::Line | ShapeKind::Arrow => {
            let len = d.len();
            let ang = d.y.atan2(d.x);
            let step = std::f32::consts::FRAC_PI_4;
            let snapped = (ang / step).round() * step;
            Vec2::new(a.x + len * snapped.cos(), a.y + len * snapped.sin())
        }
    }
}

impl ShapeTool {
    /// Copies a shape's settings into the options.
    fn load(&mut self, shape: &Shape) {
        self.kind = shape.kind;
        self.choice = match shape.kind {
            ShapeKind::Line => ShapeIcon::Line,
            ShapeKind::Rectangle if self.choice == ShapeIcon::Square => ShapeIcon::Square,
            ShapeKind::Rectangle => ShapeIcon::Rectangle,
            ShapeKind::Ellipse if self.choice == ShapeIcon::Circle => ShapeIcon::Circle,
            ShapeKind::Ellipse => ShapeIcon::Ellipse,
            ShapeKind::Arrow => ShapeIcon::Arrow,
        };
        self.width = shape.width;
        self.fill = shape.fill;
        self.head_scale = shape.head_scale;
        self.rounded = shape.rounded;
        self.corner_radius = shape.corner_radius;
        self.pixel_aligned = shape.pixel_aligned;
    }

    /// `shape` with the options' settings.
    fn with_options(&self, shape: &Shape, aa: bool) -> Shape {
        Shape {
            kind: self.kind,
            width: self.width,
            fill: self.fill,
            head_scale: self.head_scale,
            rounded: self.rounded,
            corner_radius: self.corner_radius,
            pixel_aligned: self.pixel_aligned,
            antialias: aa,
            ..*shape
        }
    }

    /// Which handle of the active shape layer is under `p` (canvas coords).
    fn handle_at(shape: &Shape, p: Vec2, zoom: f32) -> Option<Handle> {
        let r = 8.0 / zoom.max(0.01);
        if (p - shape.b).len() <= r {
            Some(Handle::B)
        } else if (p - shape.a).len() <= r {
            Some(Handle::A)
        } else {
            let (lo, hi) = (Vec2::new(shape.a.x.min(shape.b.x), shape.a.y.min(shape.b.y)), Vec2::new(shape.a.x.max(shape.b.x), shape.a.y.max(shape.b.y)));
            let pad = shape.width * 0.5 + r * 0.5;
            (p.x >= lo.x - pad && p.x <= hi.x + pad && p.y >= lo.y - pad && p.y <= hi.y + pad).then_some(Handle::Body)
        }
    }

    /// Shows `kind` in place of the layer while a handle is dragged.
    fn preview_edit(ctx: &mut ToolCtx, id: LayerId, kind: &ObjectKind) {
        let depth = ctx.doc.layer(id).map_or(ctx.doc.default_depth, |l| l.depth());
        if let Ok((px, off)) = kind.render(depth, None) {
            ctx.doc.set_preview(id, Arc::new(px), off);
        }
    }

    /// The shape for a drag from `start` to `end` (canvas coords).
    fn shape(&self, start: Vec2, end: Vec2, aa: bool) -> Shape {
        let (a, b) = if self.from_center {
            // Mirror the end point around the start; when aligned, around the
            // center of the start pixel so the shape is symmetric in pixels.
            let c = if self.pixel_aligned { Vec2::new(start.x.floor() + 0.5, start.y.floor() + 0.5) } else { start };
            (c * 2.0 - end, end)
        } else {
            (start, end)
        };
        Shape {
            kind: self.kind,
            a,
            b,
            width: self.width,
            fill: self.fill,
            antialias: aa,
            head_scale: self.head_scale,
            rounded: self.rounded,
            corner_radius: self.corner_radius,
            pixel_aligned: self.pixel_aligned,
        }
    }

    fn colors(ctx: &ToolCtx, button: Button) -> (imged_core::Color, imged_core::Color) {
        match button {
            Button::Primary => (ctx.colors.primary, ctx.colors.secondary),
            Button::Secondary => (ctx.colors.secondary, ctx.colors.primary),
        }
    }

    fn redraw(&mut self, ctx: &mut ToolCtx) {
        let Some(d) = &self.drag else { return };
        let off = Vec2::new(d.live.offset.0 as f32, d.live.offset.1 as f32);
        // Shift into the live layer's local frame by whole pixels (keeps alignment).
        let mut shape = self.shape(d.start, d.end, ctx.settings.antialias);
        shape.a = shape.a - off;
        shape.b = shape.b - off;
        let (stroke_c, fill_c) = Self::colors(ctx, d.button);
        let fill_c = if self.fill == ShapeFill::Fill { stroke_c } else { fill_c };
        let d = self.drag.as_mut().unwrap();
        let (outline, fill) = shape.masks(d.live.layer_rect());
        let mut parts = Vec::new();
        if let Some(m) = &fill {
            parts.push((m, fill_c.px()));
        }
        if let Some(m) = &outline {
            parts.push((m, stroke_c.px()));
        }
        let region = shape.bounds().intersect(&d.live.layer_rect());
        d.live.update(ctx.doc, &parts, region, ctx.settings.paint_mode);
    }
}

impl Tool for ShapeTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Shape
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        let active = if self.drag.is_none() { active_shape(ctx.doc) } else { None };
        if let Some((id, shape, color, fill_color)) = active {
            let now = ObjectKind::Shape { shape, color, fill_color };
            if self.editing.as_ref() != Some(&(id, now.clone())) {
                self.editing = Some((id, now));
                self.load(&shape);
                ctx.settings.antialias = shape.antialias;
            }
            ui.label(
                egui::RichText::new("Editing the active shape layer: drag its handles or body, or change it here. Drag elsewhere (or Ctrl+drag) for a new shape.")
                    .small()
                    .color(Color32::from_rgb(0, 160, 255)),
            );
        } else {
            self.editing = None;
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (c, name) in CHOICES {
                if icon_button_sized(ui, Icon::Shape(c), self.choice == c, name, 28.0).clicked() {
                    self.choice = c;
                    self.kind = match c {
                        ShapeIcon::Line => ShapeKind::Line,
                        ShapeIcon::Rectangle | ShapeIcon::Square => ShapeKind::Rectangle,
                        ShapeIcon::Ellipse | ShapeIcon::Circle => ShapeKind::Ellipse,
                        ShapeIcon::Arrow => ShapeKind::Arrow,
                    };
                }
            }
        });
        ui.add(egui::Slider::new(&mut self.width, 1.0..=100.0).update_while_editing(false).logarithmic(true).text("Line width"));
        if matches!(self.kind, ShapeKind::Rectangle | ShapeKind::Ellipse) {
            egui::ComboBox::from_label("Style")
                .selected_text(self.fill.label())
                .show_ui(ui, |ui| {
                    for f in ShapeFill::ALL {
                        ui.selectable_value(&mut self.fill, f, f.label());
                    }
                });
        }
        if self.kind == ShapeKind::Arrow {
            ui.add(egui::Slider::new(&mut self.head_scale, 1.0..=10.0).update_while_editing(false).text("Head size"));
        }
        if self.kind != ShapeKind::Ellipse {
            let label = if self.kind == ShapeKind::Rectangle { "Rounded corners" } else { "Rounded ends" };
            ui.checkbox(&mut self.rounded, label);
            if self.rounded && self.kind == ShapeKind::Rectangle {
                ui.add(egui::Slider::new(&mut self.corner_radius, 1.0..=200.0).update_while_editing(false).logarithmic(true).text("Radius"));
            }
        }
        ui.checkbox(&mut self.pixel_aligned, "Pixel-aligned")
            .on_hover_text("Edges cover whole pixels instead of being smeared between two");
        ui.checkbox(&mut self.from_center, "Draw from center")
            .on_hover_text("The point where you start dragging is the center of the shape");
        if let Some((id, shape, mut color, mut fill_color)) = active {
            ui.horizontal_wrapped(|ui| {
                let mut c = color32(color);
                ui.label("Line");
                if egui::color_picker::color_edit_button_srgba(ui, &mut c, egui::color_picker::Alpha::OnlyBlend).changed() {
                    color = from_color32(c);
                }
                if shape.fill != ShapeFill::Outline || self.fill != ShapeFill::Outline {
                    let mut f = color32(fill_color);
                    ui.label("Fill");
                    if egui::color_picker::color_edit_button_srgba(ui, &mut f, egui::color_picker::Alpha::OnlyBlend).changed() {
                        fill_color = from_color32(f);
                    }
                }
                if ui.small_button("Use current colors").on_hover_text("Line: primary color, fill: secondary").clicked() {
                    color = ctx.colors.primary;
                    fill_color = ctx.colors.secondary;
                }
            });
            if ui.button("Rasterize").on_hover_text("Turn the shape layer into plain pixels (it can't be edited as a shape afterwards)").clicked() {
                ctx.exec(Command::Rasterize { id });
                return;
            }
            let kind = ObjectKind::Shape { shape: self.with_options(&shape, ctx.settings.antialias), color, fill_color };
            if self.editing.as_ref().is_some_and(|(_, k)| *k != kind) {
                self.editing = Some((id, kind.clone()));
                ctx.exec(Command::EditObject { id, kind });
            }
        }
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        // Editing the active shape layer: handles and body.
        match ev.kind {
            PointerKind::Press => {
                if let Some((id, shape, color, fill_color)) = active_shape(ctx.doc).filter(|_| !ev.ctrl) {
                    if let Some(handle) = Self::handle_at(&shape, ev.raw, ctx.zoom) {
                        let orig = ObjectKind::Shape { shape, color, fill_color };
                        self.edit = Some(EditDrag { id, handle, start: ev.pos, orig: orig.clone(), now: orig });
                        return;
                    }
                }
            }
            PointerKind::Drag => {
                if let Some(e) = &mut self.edit {
                    let ObjectKind::Shape { mut shape, color, fill_color } = e.orig.clone() else { return };
                    match e.handle {
                        Handle::Body => {
                            let d = ev.pos - e.start;
                            let d = if shape.pixel_aligned { Vec2::new(d.x.round(), d.y.round()) } else { d };
                            shape.a = shape.a + d;
                            shape.b = shape.b + d;
                        }
                        Handle::A => shape.a = if ev.shift { constrain(shape.kind, shape.b, ev.pos) } else { ev.pos },
                        Handle::B => shape.b = if ev.shift { constrain(shape.kind, shape.a, ev.pos) } else { ev.pos },
                    }
                    e.now = ObjectKind::Shape { shape, color, fill_color };
                    let (id, now) = (e.id, e.now.clone());
                    Self::preview_edit(ctx, id, &now);
                    return;
                }
            }
            PointerKind::Release => {
                if let Some(e) = self.edit.take() {
                    ctx.doc.clear_preview();
                    if e.now != e.orig {
                        self.editing = Some((e.id, e.now.clone()));
                        ctx.exec(Command::EditObject { id: e.id, kind: e.now });
                    }
                    return;
                }
            }
            PointerKind::Hover => {}
        }
        match ev.kind {
            PointerKind::Press => {
                let live = LivePaint::begin(ctx.doc, ctx.settings.grow_layers);
                self.drag = Some(Drag { start: ev.pos, end: ev.pos, button: ev.button, live });
            }
            PointerKind::Drag => {
                if let Some(d) = &mut self.drag {
                    let equal_sides = matches!(self.choice, ShapeIcon::Square | ShapeIcon::Circle);
                    d.end = if ev.shift || equal_sides { constrain(self.kind, d.start, ev.pos) } else { ev.pos };
                    self.redraw(ctx);
                }
            }
            PointerKind::Release => {
                if let Some(d) = self.drag.take() {
                    d.live.discard(ctx.doc);
                    if (d.end - d.start).len() < 0.5 {
                        return;
                    }
                    let shape = self.shape(d.start, d.end, ctx.settings.antialias);
                    let (color, fill_color) = Self::colors(ctx, d.button);
                    let cmd = Command::Shape { shape, color, fill_color };
                    if ctx.exec(cmd).is_some() {
                        ctx.colors.mark_used(color);
                    }
                }
            }
            PointerKind::Hover => {}
        }
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        if let Some(d) = self.drag.take() {
            d.live.cancel(ctx.doc);
        }
        if self.edit.take().is_some() {
            ctx.doc.clear_preview();
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        if self.drag.is_some() {
            return;
        }
        let shape = match &self.edit {
            Some(EditDrag { now: ObjectKind::Shape { shape, .. }, .. }) => *shape,
            _ => match active_shape(doc) {
                Some((_, s, _, _)) => s,
                None => return,
            },
        };
        let blue = Color32::from_rgb(0, 160, 255);
        let (a, b) = (view.to_screen(shape.a), view.to_screen(shape.b));
        if matches!(shape.kind, ShapeKind::Rectangle | ShapeKind::Ellipse) {
            let r = egui::Rect::from_two_pos(a, b);
            let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
            painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1.0, blue), 4.0, 3.0));
        } else {
            painter.extend(egui::Shape::dashed_line(&[a, b], Stroke::new(1.0, blue), 4.0, 3.0));
        }
        for p in [a, b] {
            let r = egui::Rect::from_center_size(p, egui::vec2(9.0, 9.0));
            painter.rect_filled(r, 1.0, Color32::WHITE);
            painter.rect_stroke(r, 1.0, Stroke::new(1.5, blue), egui::StrokeKind::Inside);
        }
    }

    fn uses_snapping(&self) -> bool {
        true
    }

    fn adjust_size(&mut self, bigger: bool) {
        self.width = if bigger { self.width + 1.0 } else { self.width - 1.0 }.clamp(1.0, 100.0);
    }

    fn status(&self) -> Option<String> {
        if let Some(EditDrag { now: ObjectKind::Shape { shape, .. }, .. }) = &self.edit {
            let (w, h) = shape.size();
            return Some(format!("{w:.0} × {h:.0} px"));
        }
        let d = self.drag.as_ref()?;
        let (w, h) = self.shape(d.start, d.end, true).size();
        Some(if self.pixel_aligned || (w.fract() == 0.0 && h.fract() == 0.0) {
            format!("{w:.0} × {h:.0} px")
        } else {
            format!("{w:.1} × {h:.1} px")
        })
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "shape.rounded", &mut self.rounded);
        p.bool(load, "shape.pixel_aligned", &mut self.pixel_aligned);
        p.bool(load, "shape.from_center", &mut self.from_center);
    }
}

