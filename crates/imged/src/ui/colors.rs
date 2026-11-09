//! Color panel:
//!
//! ```text
//! ┌──────┐ H ▬▬▬▬▬▬▬▬▬ [ ]
//! │ prim ⇄ S ▬▬▬▬▬▬▬▬▬ [ ]
//! └──┌──────┐ V ▬▬▬▬▬▬▬ [ ]
//!    │ sec  │
//! R G B A   ╭─────╮
//! ▮ ▮ ▮ ▮   │wheel│
//! [] [] []  ╰─────╯
//! Hex [      ]  Oklab [          ]
//! Recent ……   Palette ……
//! ```
//!
//! While the color picker hovers the canvas, every control shows the color
//! under the cursor.

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Ui, Vec2};
use imged_core::Color;

use super::widgets::{icon_button_sized, paint_checker_rect, swatch, to_color32, Icon};
use crate::state::{ColorSlot, ColorState};

const SWATCH: f32 = 40.0;
/// Width of the swatch block (primary, secondary offset, swap icon).
const BLOCK: f32 = 66.0;
const WHEEL: f32 = 156.0;
const BAR: f32 = 16.0;
const VBAR_W: f32 = 18.0;
/// Width of each RGBA column (bar + its number below).
const COL_W: f32 = 32.0;
const SMALL_SWATCH: f32 = 20.0;

#[derive(Default)]
pub struct ColorPanel {
    wheel: Option<TextureHandle>,
    /// Hue/saturation are remembered so grey or black colors don't lose them.
    hsv: (f32, f32, f32),
    last: Option<(ColorSlot, Color)>,
    hex: String,
    hex_focused: bool,
    oklab: String,
    oklab_focused: bool,
    show_all_recent: bool,
    show_all_palette: bool,
}

fn wheel_image(size: usize) -> ColorImage {
    let r = size as f32 / 2.0;
    let mut px = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - r;
            let dy = y as f32 + 0.5 - r;
            let d = (dx * dx + dy * dy).sqrt() / r;
            let hue = dy.atan2(dx).to_degrees().rem_euclid(360.0);
            let c = Color::from_hsv(hue, d.min(1.0), 1.0, 1.0);
            let a = ((1.0 - d) * r + 0.5).clamp(0.0, 1.0); // anti-aliased rim
            let [rr, g, b, _] = c.to_rgba8();
            px.push(Color32::from_rgba_unmultiplied(rr, g, b, (a * 255.0) as u8));
        }
    }
    ColorImage::new([size, size], px)
}

fn oklab_text(c: Color) -> String {
    let (l, a, b) = c.to_oklab();
    format!("{:.3} {:.3} {:.3}", l, a, b).replace("-0.000", "0.000")
}

/// Parses "L a b", "oklab(L a b)" or "oklab(62% 0.1 0.2)".
fn parse_oklab(s: &str) -> Option<(f32, f32, f32)> {
    let s = s.trim().trim_start_matches("oklab").trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<&str> = s.split(|c: char| c.is_whitespace() || c == ',').filter(|p| !p.is_empty()).collect();
    if parts.len() != 3 {
        return None;
    }
    let num = |p: &str| -> Option<f32> {
        match p.strip_suffix('%') {
            Some(v) => v.parse::<f32>().ok().map(|v| v / 100.0),
            None => p.parse().ok(),
        }
    };
    Some((num(parts[0])?, num(parts[1])?, num(parts[2])?))
}

/// A gradient bar. `color_at(t)` gives the color at position t (0..1, left
/// to right, or bottom to top when `vertical`); `t` is the marker. Returns
/// the new position when clicked or dragged.
fn bar(ui: &mut Ui, size: Vec2, vertical: bool, checker: bool, t: f32, tip: &str, color_at: impl Fn(f32) -> Color) -> Option<f32> {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let painter = ui.painter();
    if checker {
        paint_checker_rect(painter, rect, 4.0);
    }
    let steps = 32;
    for k in 0..steps {
        let (t0, t1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
        let c = to_color32(color_at((t0 + t1) / 2.0));
        let r = if vertical {
            Rect::from_min_max(
                Pos2::new(rect.left(), rect.bottom() - rect.height() * t1),
                Pos2::new(rect.right(), rect.bottom() - rect.height() * t0),
            )
        } else {
            Rect::from_min_max(
                Pos2::new(rect.left() + rect.width() * t0, rect.top()),
                Pos2::new(rect.left() + rect.width() * t1, rect.bottom()),
            )
        };
        painter.rect_filled(r, 0.0, c);
    }
    let m = t.clamp(0.0, 1.0);
    let (a, b) = if vertical {
        let y = rect.bottom() - rect.height() * m;
        (Pos2::new(rect.left() - 2.0, y), Pos2::new(rect.right() + 2.0, y))
    } else {
        let x = rect.left() + rect.width() * m;
        (Pos2::new(x, rect.top() - 2.0), Pos2::new(x, rect.bottom() + 2.0))
    };
    painter.line_segment([a, b], Stroke::new(3.0, Color32::BLACK));
    painter.line_segment([a, b], Stroke::new(1.0, Color32::WHITE));
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::from_gray(90)), egui::StrokeKind::Outside);
    let resp = resp.on_hover_text(tip);
    if resp.dragged() || resp.clicked() {
        let p = resp.interact_pointer_pos()?;
        let t = if vertical { (rect.bottom() - p.y) / rect.height() } else { (p.x - rect.left()) / rect.width() };
        return Some(t.clamp(0.0, 1.0));
    }
    None
}

/// A compact number field (small text, so it stays `width` wide).
fn num(ui: &mut Ui, v: &mut f32, max: f32, decimals: usize, width: f32) -> bool {
    ui.spacing_mut().interact_size.x = width;
    ui.spacing_mut().button_padding.x = 2.0;
    ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
    let r = ui.add_sized(
        [width, 18.0],
        egui::DragValue::new(v).speed(0.005 * max.clamp(0.01, 1.0) * if max > 1.5 { 100.0 } else { 1.0 }).range(0.0..=max).max_decimals(decimals).update_while_editing(false),
    );
    crate::ui::grab::mark(ui, &r);
    r.changed()
}

impl ColorPanel {
    fn sync(&mut self, colors: &ColorState, shown: Color) {
        let c = shown;
        if self.last != Some((colors.editing, c)) {
            let (h, s, v) = c.to_hsv();
            let (oh, os, _) = self.hsv;
            // Keep hue for greys and hue+saturation for black.
            self.hsv = if v <= 1e-6 { (oh, os, v) } else if s <= 1e-6 { (oh, s, v) } else { (h, s, v) };
            self.last = Some((colors.editing, c));
            if !self.hex_focused {
                self.hex = c.to_hex();
            }
            if !self.oklab_focused {
                self.oklab = oklab_text(c);
            }
        }
    }

    /// Sets the edited color; `hsv` keeps the hue/saturation the user chose.
    fn set(&mut self, colors: &mut ColorState, c: Color, hsv: Option<(f32, f32, f32)>) {
        colors.set(colors.editing, c);
        if let Some(h) = hsv {
            self.hsv = h;
        } else {
            self.last = None;
            self.sync(colors, c);
        }
        self.last = Some((colors.editing, c));
        if !self.hex_focused {
            self.hex = c.to_hex();
        }
        if !self.oklab_focused {
            self.oklab = oklab_text(c);
        }
    }

    /// `preview`: a color to display instead of the edited one without
    /// changing it (the color under the cursor while the picker is active).
    pub fn ui(&mut self, ui: &mut Ui, colors: &mut ColorState, preview: Option<Color>) {
        let shown = preview.unwrap_or(colors.get(colors.editing));
        self.sync(colors, shown);
        let full = ui.available_width();

        // Top row: swatches (with the swap icon) and the H, S, V bars.
        ui.horizontal_top(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(BLOCK, BLOCK), Sense::hover());
            let pri_rect = Rect::from_min_size(rect.min, Vec2::splat(SWATCH));
            let sec_rect = Rect::from_min_size(rect.max - Vec2::splat(SWATCH), Vec2::splat(SWATCH));
            let sec = ui.interact(sec_rect, ui.scope_id().with("sec"), Sense::click());
            let pri = ui.interact(pri_rect, ui.scope_id().with("pri"), Sense::click());
            for (r, c, slot) in [(sec_rect, colors.secondary, ColorSlot::Secondary), (pri_rect, colors.primary, ColorSlot::Primary)] {
                let p = ui.painter();
                paint_checker_rect(p, r, 6.0);
                p.rect_filled(r, 2.0, to_color32(c));
                let sel = colors.editing == slot;
                p.rect_stroke(
                    r,
                    2.0,
                    Stroke::new(if sel { 2.0 } else { 1.0 }, if sel { super::widgets::ACCENT } else { Color32::from_gray(90) }),
                    egui::StrokeKind::Outside,
                );
            }
            if pri.on_hover_text("Primary color (left mouse). Click to edit it.").clicked() {
                colors.editing = ColorSlot::Primary;
            }
            if sec.on_hover_text("Secondary color (right mouse). Click to edit it.").clicked() {
                colors.editing = ColorSlot::Secondary;
            }
            // The free corner above the secondary, right of the primary.
            let swap_rect = Rect::from_min_max(Pos2::new(pri_rect.right() + 1.0, rect.top()), Pos2::new(rect.right(), sec_rect.top() - 1.0));
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(swap_rect));
            let side = swap_rect.width().min(swap_rect.height());
            if icon_button_sized(&mut child, Icon::Swap, false, "Swap primary and secondary (X)", side).clicked() {
                colors.swap();
            }

            ui.add_space(6.0);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                let (h, s, v) = self.hsv;
                let a = shown.a;
                let num_w = 44.0;
                let bar_w = (full - BLOCK - 6.0 - 14.0 - num_w - 16.0).max(60.0);
                let mut new_hsv = None;
                ui.horizontal(|ui| {
                    ui.label(RichText::new("H").small());
                    if let Some(t) = bar(ui, Vec2::new(bar_w, BAR), false, false, h / 360.0, "Hue", |t| Color::from_hsv(t * 360.0, 1.0, 1.0, 1.0)) {
                        new_hsv = Some((t * 360.0, s, if v <= 0.0 { 1.0 } else { v }));
                    }
                    let mut hh = h;
                    if num(ui, &mut hh, 360.0, 1, num_w) {
                        new_hsv = Some((hh, s, v));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("S").small());
                    if let Some(t) = bar(ui, Vec2::new(bar_w, BAR), false, false, s, "Saturation (HSV)", |t| Color::from_hsv(h, t, v.clamp(0.25, 1.0), 1.0)) {
                        new_hsv = Some((h, t, if v <= 0.0 { 1.0 } else { v }));
                    }
                    let mut ss = s;
                    if num(ui, &mut ss, 1.0, 3, num_w) {
                        new_hsv = Some((h, ss, v));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("V").small());
                    if let Some(t) = bar(ui, Vec2::new(bar_w, BAR), false, false, v, "Value (brightness)", |t| Color::from_hsv(h, s, t, 1.0)) {
                        new_hsv = Some((h, s, t));
                    }
                    let mut vv = v;
                    if num(ui, &mut vv, 1000.0, 3, num_w) {
                        new_hsv = Some((h, s, vv));
                    }
                });
                if let Some((h, s, v)) = new_hsv {
                    let c = Color::from_hsv(h.rem_euclid(360.0), s, v, a);
                    self.set(colors, c, Some((h.rem_euclid(360.0), s, v)));
                }
            });
        });

        // Middle: R, G, B, A bars (vertical) and the wheel.
        let shown = preview.unwrap_or(colors.get(colors.editing));
        ui.add_space(2.0);
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let mut c = shown;
            let mut changed = false;
            for i in 0..4 {
                ui.vertical(|ui| {
                    ui.set_width(COL_W);
                    ui.spacing_mut().item_spacing.y = 2.0;
                    let label = ["R", "G", "B", "A"][i];
                    ui.vertical_centered(|ui| ui.label(RichText::new(label).small()));
                    let cur = [c.r, c.g, c.b, c.a][i];
                    let base = c;
                    ui.horizontal(|ui| {
                        ui.add_space((COL_W - VBAR_W) / 2.0);
                        let tip = ["Red", "Green", "Blue", "Alpha (opacity)"][i];
                        let at = move |t: f32| {
                            let mut k = base;
                            match i {
                                0 => k.r = t,
                                1 => k.g = t,
                                2 => k.b = t,
                                _ => k.a = t,
                            }
                            if i < 3 {
                                k.a = 1.0;
                            }
                            k
                        };
                        if let Some(t) = bar(ui, Vec2::new(VBAR_W, WHEEL - 44.0), true, i == 3, cur, tip, at) {
                            match i {
                                0 => c.r = t,
                                1 => c.g = t,
                                2 => c.b = t,
                                _ => c.a = t,
                            }
                            changed = true;
                        }
                    });
                    let mut v = cur;
                    let max = if i == 3 { 1.0 } else { 1000.0 };
                    if num(ui, &mut v, max, 2, COL_W - 2.0) {
                        match i {
                            0 => c.r = v,
                            1 => c.g = v,
                            2 => c.b = v,
                            _ => c.a = v,
                        }
                        changed = true;
                    }
                });
            }
            if changed {
                self.set(colors, c, None);
            }

            // Wheel (hue / saturation at full value; V is the bar above).
            let left = 4.0 * COL_W;
            let room = full - left;
            let size = WHEEL.min(room - 4.0);
            ui.add_space(((room - size) / 2.0).max(0.0));
            let tex = self
                .wheel
                .get_or_insert_with(|| ui.ctx().load_texture("color-wheel", wheel_image(256), TextureOptions::LINEAR))
                .clone();
            let (h, s, v) = self.hsv;
            let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
            let center = rect.center();
            let radius = size / 2.0;
            ui.painter().image(tex.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            let marker = center + Vec2::angled(h.to_radians()) * s.min(1.0) * radius;
            ui.painter().circle_stroke(marker, 5.0, Stroke::new(2.0, Color32::BLACK));
            ui.painter().circle_stroke(marker, 6.5, Stroke::new(1.5, Color32::WHITE));
            if resp.dragged() || resp.clicked() {
                if let Some(p) = resp.interact_pointer_pos() {
                    let d = p - center;
                    let nh = d.y.atan2(d.x).to_degrees().rem_euclid(360.0);
                    let ns = (d.length() / radius).min(1.0);
                    // Picking on the wheel at value 0 would be invisible; lift it.
                    let nv = if v <= 0.0 { 1.0 } else { v };
                    self.set(colors, Color::from_hsv(nh, ns, nv, shown.a), Some((nh, ns, nv)));
                }
            }
        });

        // Hex and Oklab codes.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(RichText::new("Hex").small());
            let mono = egui::FontId::monospace(11.5);
            // Fixed sizes: long codes scroll inside the box instead of widening the panel.
            let r = ui.add_sized(
                [72.0, 20.0],
                egui::TextEdit::singleline(&mut self.hex).clip_text(true).font(mono.clone()),
            );
            self.hex_focused = r.has_focus();
            if r.changed() {
                if let Some(c) = Color::from_hex(&self.hex) {
                    colors.set(colors.editing, c);
                    self.last = None;
                    self.sync(colors, c);
                }
            }
            if r.lost_focus() {
                self.hex = colors.get(colors.editing).to_hex();
            }
            ui.add_space(4.0);
            ui.label(RichText::new("Oklab").small()).on_hover_text("Perceptual lightness L (0..1) and the a (green–red) and b (blue–yellow) axes");
            let w = (ui.available_width() - 2.0).max(40.0);
            let r = ui.add_sized([w, 20.0], egui::TextEdit::singleline(&mut self.oklab).clip_text(true).font(mono));
            self.oklab_focused = r.has_focus();
            if r.changed() {
                if let Some((l, a, b)) = parse_oklab(&self.oklab) {
                    let c = Color::from_oklab(l, a, b, colors.get(colors.editing).a);
                    colors.set(colors.editing, c);
                    self.last = None;
                    self.sync(colors, c);
                }
            }
            if r.lost_focus() {
                self.oklab = oklab_text(colors.get(colors.editing));
            }
        });

        // Recent colors and palette, one row each unless expanded.
        let per_row = ((full + 3.0) / (SMALL_SWATCH + 3.0 + 2.0)).floor().max(1.0) as usize;
        ui.add_space(2.0);
        let recent: Vec<Color> = colors.recent.iter().copied().collect();
        self.swatch_list(ui, "Recent", &recent, per_row, false, colors);
        let palette = colors.palette.clone();
        self.swatch_list(ui, "Palette", &palette, per_row, true, colors);
    }

    /// One labeled list of swatches with an expand toggle. Left click picks
    /// the primary color, right click the secondary. For the palette, a "+"
    /// adds the edited color, Ctrl+click replaces a swatch with it and
    /// Shift+click removes it.
    fn swatch_list(
        &mut self,
        ui: &mut Ui,
        label: &str,
        list: &[Color],
        per_row: usize,
        palette: bool,
        colors: &mut ColorState,
    ) {
        let expanded = if palette { self.show_all_palette } else { self.show_all_recent };
        let slots = if palette { list.len() + 1 } else { list.len().max(per_row) };
        let can_expand = slots > per_row;
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).small().weak());
            if can_expand {
                let icon = if expanded { Icon::Arrow(0, -1) } else { Icon::Arrow(0, 1) };
                let tip = if expanded { "Show one row" } else { "Show all" };
                if icon_button_sized(ui, icon, false, tip, 14.0).clicked() {
                    if palette {
                        self.show_all_palette = !expanded;
                    } else {
                        self.show_all_recent = !expanded;
                    }
                }
            }
        });
        let shown = if expanded || !can_expand { slots } else { per_row };
        let mut remove = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(3.0, 3.0);
            for i in 0..shown {
                if palette && i == list.len() {
                    if ui
                        .add_sized([SMALL_SWATCH, SMALL_SWATCH], egui::Button::new("+").small())
                        .on_hover_text("Add the edited color to the palette")
                        .clicked()
                    {
                        colors.palette.push(colors.get(colors.editing));
                    }
                    continue;
                }
                let c = list.get(i).copied();
                let tip = match (c, palette) {
                    (Some(c), true) => format!(
                        "{}\nLeft: primary · Right: secondary\nCtrl+click: replace with the edited color · Shift+click: remove",
                        c.to_hex()
                    ),
                    (Some(c), false) => format!("{}\nLeft: primary · Right: secondary", c.to_hex()),
                    (None, _) => String::new(),
                };
                let r = swatch(ui, c, SMALL_SWATCH, false, &tip);
                let Some(c) = c else { continue };
                let (ctrl, shift) = ui.input(|inp| (inp.modifiers.command, inp.modifiers.shift));
                if r.clicked() {
                    if palette && ctrl {
                        colors.palette[i] = colors.get(colors.editing);
                    } else if palette && shift {
                        remove = Some(i);
                    } else {
                        colors.primary = c;
                    }
                }
                if r.secondary_clicked() {
                    colors.secondary = c;
                }
            }
        });
        if let Some(i) = remove {
            colors.palette.remove(i);
        }
    }
}
