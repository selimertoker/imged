use eframe::egui::{self, Ui};
use imged_core::ops::paint::{Shape, ShapeFill, ShapeKind};
use imged_core::{Command, Vec2};

use super::{Button, LivePaint, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::ui::widgets::{icon_button_sized, Icon, ShapeIcon};

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

    fn options_ui(&mut self, ui: &mut Ui, _ctx: &mut ToolCtx) {
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
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Press => {
                ctx.begin_drawing(ToolKind::Shape, ev.button);
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
                    let cmd = Command::Shape { shape, color, fill_color, mode: ctx.settings.paint_mode };
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
    }

    fn uses_snapping(&self) -> bool {
        true
    }

    fn adjust_size(&mut self, bigger: bool) {
        self.width = if bigger { self.width + 1.0 } else { self.width - 1.0 }.clamp(1.0, 100.0);
    }

    fn status(&self) -> Option<String> {
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
