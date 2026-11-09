//! Reusable widgets: tool buttons with painted icons, color swatches.
//!
//! Every toolbar/filter icon button goes through [`icon_button`], so visual
//! effects for the selected or hovered state (glow, pulse, animated
//! highlight) only need to be added in that one function. `selection_anim`
//! is already a smooth 0→1 value driven by `animate_bool`.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui, Vec2};
use imged_core::Color;

use crate::tools::ToolKind;

pub const ACCENT: Color32 = Color32::from_rgb(0x2F, 0x8C, 0xFF);

/// Icons drawn with the painter, so they don't depend on font coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Tool(ToolKind),
    FlipH,
    FlipV,
    RotateCw,
    RotateCcw,
    /// An arrow pointing in direction (dx, dy) (each -1, 0 or 1); (0, 0) is a dot.
    Arrow(i8, i8),
    Shape(ShapeIcon),
    /// Two curved arrows (swap primary and secondary).
    Swap,
    /// A funnel (filter layer).
    Funnel,
}

/// Shapes offered by the shape tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeIcon {
    Line,
    Rectangle,
    Square,
    Ellipse,
    Circle,
    Arrow,
}

/// A square button with a painted icon. `selected` draws the active state.
pub fn icon_button(ui: &mut Ui, icon: Icon, selected: bool, tooltip: &str) -> Response {
    icon_button_sized(ui, icon, selected, tooltip, 34.0)
}

/// Like [`icon_button`] with a custom (square) size.
pub fn icon_button_sized(ui: &mut Ui, icon: Icon, selected: bool, tooltip: &str, side: f32) -> Response {
    let size = Vec2::splat(side);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let id = resp.id;
    // Smooth 0..1 transition of the selected state; hook effects onto this.
    let selection_anim = ui.ctx().animate_bool_with_time(id.with("sel"), selected, 0.12);
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact_selectable(&resp, selected);
        let painter = ui.painter();
        let bg = if selection_anim > 0.0 {
            lerp_color(visuals.weak_bg_fill, ACCENT.gamma_multiply(0.35), selection_anim)
        } else if resp.hovered() {
            visuals.bg_fill
        } else {
            Color32::TRANSPARENT
        };
        painter.rect_filled(rect, 5.0, bg);
        if selection_anim > 0.0 {
            painter.rect_stroke(
                rect.shrink(0.5),
                5.0,
                Stroke::new(1.0, ACCENT.gamma_multiply(selection_anim)),
                StrokeKind::Inside,
            );
        }
        let fg = if selected { ui.visuals().strong_text_color() } else { visuals.fg_stroke.color };
        paint_icon(painter, rect.shrink((side * 0.235).round()), icon, fg);
    }
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_premultiplied(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()), l(a.a(), b.a()))
}

fn p(r: Rect, x: f32, y: f32) -> Pos2 {
    Pos2::new(r.left() + r.width() * x, r.top() + r.height() * y)
}

fn arrow(painter: &egui::Painter, from: Pos2, to: Pos2, s: Stroke) {
    painter.line_segment([from, to], s);
    let d = (to - from).normalized();
    let n = Vec2::new(-d.y, d.x);
    let h = 4.0;
    painter.line_segment([to, to - d * h + n * h * 0.8], s);
    painter.line_segment([to, to - d * h - n * h * 0.8], s);
}

pub fn paint_icon(painter: &egui::Painter, r: Rect, icon: Icon, c: Color32) {
    let s = Stroke::new(1.6, c);
    let thin = Stroke::new(1.2, c);
    match icon {
        Icon::Tool(t) => match t {
            ToolKind::Move => {
                arrow(painter, r.center(), p(r, 0.5, 0.0), s);
                arrow(painter, r.center(), p(r, 0.5, 1.0), s);
                arrow(painter, r.center(), p(r, 0.0, 0.5), s);
                arrow(painter, r.center(), p(r, 1.0, 0.5), s);
            }
            ToolKind::Brush => {
                painter.line_segment([p(r, 0.95, 0.05), p(r, 0.42, 0.58)], Stroke::new(2.4, c));
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.38, 0.52), p(r, 0.5, 0.64), p(r, 0.3, 0.95), p(r, 0.02, 0.98), p(r, 0.12, 0.72)],
                    c,
                    Stroke::NONE,
                ));
            }
            ToolKind::Eyedropper => {
                painter.line_segment([p(r, 0.08, 0.92), p(r, 0.62, 0.38)], s);
                painter.line_segment([p(r, 0.5, 0.26), p(r, 0.74, 0.5)], Stroke::new(2.4, c));
                painter.circle_filled(p(r, 0.78, 0.22), r.width() * 0.17, c);
            }
            ToolKind::Shape => {
                painter.rect_stroke(Rect::from_min_max(p(r, 0.0, 0.0), p(r, 0.62, 0.62)), 0.0, s, StrokeKind::Middle);
                painter.circle_stroke(p(r, 0.66, 0.66), r.width() * 0.33, s);
            }
            ToolKind::Bucket => {
                // Half-full bucket: the lower triangle is filled white.
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.02, 0.45), p(r, 0.85, 0.48), p(r, 0.45, 0.88)],
                    Color32::WHITE,
                    Stroke::NONE,
                ));
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.42, 0.05), p(r, 0.85, 0.48), p(r, 0.45, 0.88), p(r, 0.02, 0.45)],
                    Color32::TRANSPARENT,
                    s,
                ));
                painter.line_segment([p(r, 0.02, 0.45), p(r, 0.85, 0.48)], thin);
                painter.circle_filled(p(r, 0.9, 0.82), r.width() * 0.11, Color32::WHITE);
            }
            ToolKind::Pen => {
                // Pencil with a single square "pixel" at its tip.
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.935, 0.235), p(r, 0.765, 0.065), p(r, 0.265, 0.565), p(r, 0.435, 0.735)],
                    Color32::TRANSPARENT,
                    s,
                ));
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.435, 0.735), p(r, 0.265, 0.565), p(r, 0.2, 0.8)],
                    c,
                    Stroke::NONE,
                ));
                painter.rect_filled(Rect::from_min_max(p(r, 0.0, 0.82), p(r, 0.18, 1.0)), 0.0, c);
            }
            ToolKind::Text => {
                painter.text(r.center(), Align2::CENTER_CENTER, "T", FontId::proportional(r.height() * 1.25), c);
            }
            ToolKind::BoxSelect => {
                let rr = r.shrink(1.0);
                let pts = [rr.left_top(), rr.right_top(), rr.right_bottom(), rr.left_bottom(), rr.left_top()];
                painter.extend(Shape::dashed_line(&pts, s, 3.0, 2.5));
            }
            ToolKind::MagicWand => {
                painter.line_segment([p(r, 0.05, 0.95), p(r, 0.62, 0.38)], Stroke::new(2.2, c));
                let center = p(r, 0.78, 0.22);
                for i in 0..4 {
                    let a = i as f32 * std::f32::consts::FRAC_PI_4;
                    let d = Vec2::angled(a) * r.width() * 0.2;
                    painter.line_segment([center - d, center + d], thin);
                }
            }
            ToolKind::OutlineSelect => {
                painter.rect_stroke(r, 1.0, Stroke::new(2.2, c), StrokeKind::Inside);
                let inner = r.shrink(r.width() * 0.3);
                let pts = [inner.left_top(), inner.right_top(), inner.right_bottom(), inner.left_bottom(), inner.left_top()];
                painter.extend(Shape::dashed_line(&pts, thin, 2.0, 2.0));
            }
            ToolKind::Resize => {
                let inner = r.shrink(r.width() * 0.12);
                painter.rect_stroke(inner, 0.0, thin, StrokeKind::Middle);
                for pt in [inner.left_top(), inner.right_top(), inner.right_bottom(), inner.left_bottom()] {
                    painter.rect_filled(Rect::from_center_size(pt, Vec2::splat(r.width() * 0.26)), 0.0, c);
                }
                arrow(painter, p(r, 0.32, 0.68), p(r, 0.7, 0.3), thin);
            }
            ToolKind::Rotate => {
                let center = r.center();
                let rad = r.width() * 0.45;
                let n = 20;
                let (a0, a1) = (0.4f32, 5.6f32);
                let pts: Vec<Pos2> =
                    (0..=n).map(|i| center + Vec2::angled(a0 + (a1 - a0) * i as f32 / n as f32) * rad).collect();
                let end = *pts.last().unwrap();
                let prev = pts[pts.len() - 2];
                painter.add(Shape::line(pts, s));
                arrow(painter, prev, end, s);
                painter.circle_filled(center, r.width() * 0.1, c);
            }
            ToolKind::Tilt => {
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.25, 0.1), p(r, 0.75, 0.1), p(r, 1.0, 0.9), p(r, 0.0, 0.9)],
                    Color32::TRANSPARENT,
                    s,
                ));
            }
            ToolKind::Shear => {
                painter.add(Shape::convex_polygon(
                    vec![p(r, 0.3, 0.1), p(r, 1.0, 0.1), p(r, 0.7, 0.9), p(r, 0.0, 0.9)],
                    Color32::TRANSPARENT,
                    s,
                ));
            }
        },
        Icon::FlipH => {
            painter.add(Shape::convex_polygon(vec![p(r, 0.42, 0.1), p(r, 0.42, 0.9), p(r, 0.0, 0.9)], c, Stroke::NONE));
            painter.add(Shape::convex_polygon(
                vec![p(r, 0.58, 0.1), p(r, 1.0, 0.9), p(r, 0.58, 0.9)],
                Color32::TRANSPARENT,
                thin,
            ));
            painter.extend(Shape::dashed_line(&[p(r, 0.5, 0.0), p(r, 0.5, 1.0)], thin, 2.0, 2.0));
        }
        Icon::FlipV => {
            painter.add(Shape::convex_polygon(vec![p(r, 0.1, 0.42), p(r, 0.9, 0.42), p(r, 0.9, 0.0)], c, Stroke::NONE));
            painter.add(Shape::convex_polygon(
                vec![p(r, 0.1, 0.58), p(r, 0.9, 1.0), p(r, 0.9, 0.58)],
                Color32::TRANSPARENT,
                thin,
            ));
            painter.extend(Shape::dashed_line(&[p(r, 0.0, 0.5), p(r, 1.0, 0.5)], thin, 2.0, 2.0));
        }
        Icon::Arrow(0, 0) => {
            painter.circle_filled(r.center(), r.width() * 0.2, c);
        }
        Icon::Arrow(dx, dy) => {
            let d = Vec2::new(dx as f32, dy as f32).normalized();
            let from = r.center() - d * r.width() * 0.42;
            let to = r.center() + d * r.width() * 0.42;
            arrow(painter, from, to, s);
        }
        Icon::Swap => {
            // A quarter circle from the left edge to the bottom edge, arrows at both ends.
            let c0 = r.left_bottom();
            let rad = r.width() * 0.8;
            let n = 12;
            let pts: Vec<Pos2> = (0..=n)
                .map(|i| {
                    let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_2 * i as f32 / n as f32;
                    c0 + Vec2::angled(a) * rad
                })
                .collect();
            let (first, second) = (pts[0], pts[1]);
            let (last, before) = (pts[n], pts[n - 1]);
            painter.add(Shape::line(pts, s));
            arrow(painter, second, first - (second - first) * 0.01, s);
            arrow(painter, before, last + (last - before) * 0.01, s);
        }
        Icon::Funnel => {
            let pts = [p(r, 0.0, 0.05), p(r, 1.0, 0.05), p(r, 0.62, 0.5), p(r, 0.62, 0.95), p(r, 0.38, 0.8), p(r, 0.38, 0.5)];
            painter.add(Shape::convex_polygon(pts[..3].iter().chain(&pts[5..]).copied().collect(), c, Stroke::NONE));
            painter.add(Shape::convex_polygon(pts[2..].to_vec(), c, Stroke::NONE));
        }
        Icon::Shape(sh) => match sh {
            ShapeIcon::Line => {
                painter.line_segment([p(r, 0.05, 0.95), p(r, 0.95, 0.05)], s);
            }
            ShapeIcon::Rectangle => {
                painter.rect_stroke(Rect::from_min_max(p(r, 0.0, 0.18), p(r, 1.0, 0.82)), 0.0, s, StrokeKind::Middle);
            }
            ShapeIcon::Square => {
                painter.rect_stroke(Rect::from_min_max(p(r, 0.08, 0.08), p(r, 0.92, 0.92)), 0.0, s, StrokeKind::Middle);
            }
            ShapeIcon::Ellipse => {
                let n = 32;
                let pts: Vec<Pos2> = (0..n)
                    .map(|i| {
                        let a = std::f32::consts::TAU * i as f32 / n as f32;
                        r.center() + Vec2::new(a.cos() * r.width() * 0.5, a.sin() * r.height() * 0.32)
                    })
                    .collect();
                painter.add(Shape::closed_line(pts, s));
            }
            ShapeIcon::Circle => {
                painter.circle_stroke(r.center(), r.width() * 0.45, s);
            }
            ShapeIcon::Arrow => {
                arrow(painter, p(r, 0.05, 0.95), p(r, 0.95, 0.05), s);
            }
        },
        Icon::RotateCw | Icon::RotateCcw => {
            let cw = icon == Icon::RotateCw;
            let center = r.center();
            let rad = r.width() * 0.42;
            let n = 16;
            let (a0, a1) = (-2.6f32, 1.2f32);
            let pts: Vec<Pos2> = (0..=n)
                .map(|i| {
                    let a = a0 + (a1 - a0) * i as f32 / n as f32;
                    let a = if cw { a } else { std::f32::consts::PI - a };
                    center + Vec2::angled(a) * rad
                })
                .collect();
            let end = *pts.last().unwrap();
            let prev = pts[pts.len() - 2];
            painter.add(Shape::line(pts, s));
            arrow(painter, prev, end, s);
        }
    }
}

/// Paints a checkerboard (for showing transparency) inside `rect`.
pub fn paint_checker_rect(painter: &egui::Painter, rect: Rect, cell: f32) {
    painter.rect_filled(rect, 0.0, Color32::from_gray(204));
    let nx = (rect.width() / cell).ceil() as i32;
    let ny = (rect.height() / cell).ceil() as i32;
    for y in 0..ny {
        for x in 0..nx {
            if (x + y) % 2 == 1 {
                let r = Rect::from_min_size(rect.min + Vec2::new(x as f32 * cell, y as f32 * cell), Vec2::splat(cell))
                    .intersect(rect);
                painter.rect_filled(r, 0.0, Color32::from_gray(153));
            }
        }
    }
}

pub fn to_color32(c: Color) -> Color32 {
    let [r, g, b, a] = c.to_rgba8();
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

/// A color swatch showing transparency. Returns the click response.
pub fn swatch(ui: &mut Ui, color: Option<Color>, size: f32, selected: bool, tooltip: &str) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        match color {
            Some(c) => {
                if c.a < 1.0 {
                    paint_checker_rect(painter, rect, (size / 4.0).max(3.0));
                }
                painter.rect_filled(rect, 2.0, to_color32(c));
                if c.a <= 0.0 {
                    painter.line_segment([rect.left_bottom(), rect.right_top()], Stroke::new(1.5, Color32::RED));
                }
            }
            None => {
                let pts = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom(), rect.left_top()];
                painter.extend(Shape::dashed_line(&pts, Stroke::new(1.0, Color32::from_gray(120)), 3.0, 3.0));
            }
        }
        let stroke = if selected {
            Stroke::new(2.0, ACCENT)
        } else if resp.hovered() {
            Stroke::new(1.0, ui.visuals().strong_text_color())
        } else {
            Stroke::new(1.0, Color32::from_gray(90))
        };
        painter.rect_stroke(rect, 2.0, stroke, StrokeKind::Outside);
    }
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

pub fn section_header(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(text).small().strong().color(ui.visuals().weak_text_color()));
}
