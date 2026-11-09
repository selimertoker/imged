use eframe::egui::{self, Color32, Painter, Stroke, Ui};
use imged_core::ops::text::TextSpec;
use imged_core::{Command, Document, Vec2};

use super::{Button, LivePaint, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

pub struct TextTool {
    text: String,
    family: String,
    family_filter: String,
    /// Also list per-script / per-language variants (e.g. "Noto Sans Thai").
    all_variants: bool,
    size: f32,
    bold: bool,
    italic: bool,
    rotation: f32,
    line_spacing: f32,
    anchor: Option<Vec2>,
    button: Button,
    /// Offset from the anchor to the pointer while dragging the text.
    moving: Option<Vec2>,
    live: Option<LivePaint>,
    corners: Option<[Vec2; 4]>,
    needs_render: bool,
}

impl Default for TextTool {
    fn default() -> Self {
        TextTool {
            text: "Text".into(),
            family: String::new(),
            family_filter: String::new(),
            all_variants: false,
            size: 32.0,
            bold: false,
            italic: false,
            rotation: 0.0,
            line_spacing: 1.0,
            anchor: None,
            button: Button::Primary,
            moving: None,
            live: None,
            corners: None,
            needs_render: false,
        }
    }
}

fn point_in_quad(p: Vec2, q: &[Vec2; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let c = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if c.abs() < 1e-6 {
            continue;
        }
        if sign == 0.0 {
            sign = c.signum();
        } else if c.signum() != sign {
            return false;
        }
    }
    true
}

impl TextTool {
    fn spec(&self, pos: Vec2, aa: bool) -> TextSpec {
        TextSpec {
            text: self.text.clone(),
            family: self.family.clone(),
            size: self.size,
            bold: self.bold,
            italic: self.italic,
            rotation: self.rotation,
            pos,
            antialias: aa,
            line_spacing: self.line_spacing,
        }
    }

    fn render_preview(&mut self, ctx: &mut ToolCtx) {
        self.needs_render = false;
        let Some(anchor) = self.anchor else { return };
        if self.live.is_none() {
            self.live = Some(LivePaint::begin(ctx.doc, ctx.settings.grow_layers));
        }
        let spec = self.spec(anchor, ctx.settings.antialias);
        let color = ctx.color_for(self.button).px();
        let rendered = if self.text.is_empty() { None } else { ctx.fonts().render(&spec) };
        let live = self.live.as_mut().unwrap();
        match rendered {
            Some((mask, corners)) => {
                self.corners = Some(corners);
                let mask = mask.translated(-live.offset.0, -live.offset.1);
                let r = mask.rect;
                live.update(ctx.doc, &[(&mask, color)], r, ctx.settings.paint_mode);
            }
            None => {
                self.corners = None;
                live.update(ctx.doc, &[], imged_core::Rect::default(), ctx.settings.paint_mode);
            }
        }
    }

    fn apply(&mut self, ctx: &mut ToolCtx) {
        let Some(anchor) = self.anchor.take() else { return };
        if let Some(live) = self.live.take() {
            live.discard(ctx.doc);
        }
        self.corners = None;
        if self.text.trim().is_empty() {
            return;
        }
        let color = ctx.color_for(self.button);
        let cmd = Command::Text { spec: self.spec(anchor, ctx.settings.antialias), color, mode: ctx.settings.paint_mode };
        if ctx.exec(cmd).is_some() {
            ctx.colors.mark_used(color);
        }
    }
}

impl Tool for TextTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Text
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        if self.family.is_empty() {
            self.family = ctx.fonts().default_family();
        }
        let mut changed = false;
        // Searching always looks at every family; the plain list hides variants.
        let searching = !self.family_filter.is_empty();
        let families: Vec<String> = if self.all_variants || searching {
            ctx.fonts().families().to_vec()
        } else {
            ctx.fonts().main_families().to_vec()
        };
        let hidden = ctx.fonts().families().len() - ctx.fonts().main_families().len();
        egui::ComboBox::from_id_salt("font-family")
            .selected_text(&self.family)
            .width(ui.available_width() - 8.0)
            .height(320.0)
            .show_ui(ui, |ui| {
                ui.add(egui::TextEdit::singleline(&mut self.family_filter).hint_text("Search fonts…"));
                if hidden > 0 {
                    ui.checkbox(&mut self.all_variants, format!("Show {hidden} script/language variants"))
                        .on_hover_text("Families like \"Noto Sans Thai\" are hidden when there are many variants of the same font.\nSearching finds them anyway.");
                }
                let f = self.family_filter.to_lowercase();
                for fam in families.iter().filter(|n| f.is_empty() || n.to_lowercase().contains(&f)) {
                    if ui.selectable_label(*fam == self.family, fam).clicked() {
                        self.family = fam.clone();
                        changed = true;
                    }
                }
            });
        changed |= ui.add(egui::Slider::new(&mut self.size, 4.0..=400.0).update_while_editing(false).logarithmic(true).text("Size")).changed();
        ui.horizontal(|ui| {
            changed |= ui.toggle_value(&mut self.bold, egui::RichText::new("B").strong()).changed();
            changed |= ui.toggle_value(&mut self.italic, egui::RichText::new("I").italics()).changed();
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.rotation, -180.0..=180.0).update_while_editing(false).suffix("°").text("Rotation"))
            .changed();
        changed |= ui.add(egui::Slider::new(&mut self.line_spacing, 0.5..=3.0).update_while_editing(false).text("Line spacing")).changed();
        let te = ui.add(egui::TextEdit::multiline(&mut self.text).desired_rows(3).desired_width(f32::INFINITY).hint_text("Type text…"));
        changed |= te.changed();
        let ctrl_enter = te.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command);
        ui.horizontal(|ui| {
            let can = self.anchor.is_some();
            if ui.add_enabled(can, egui::Button::new("Apply")).on_hover_text("Ctrl+Enter / Enter").clicked() || (can && ctrl_enter) {
                self.apply(ctx);
            }
            if ui.add_enabled(can, egui::Button::new("Cancel")).clicked() {
                self.cancel(ctx);
            }
        });
        if self.anchor.is_none() {
            ui.weak("Click on the canvas to place the text.");
        }
        if (changed || self.needs_render) && self.anchor.is_some() {
            self.render_preview(ctx);
        }
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Press => {
                if let (Some(anchor), Some(q)) = (self.anchor, &self.corners) {
                    if point_in_quad(ev.raw, q) {
                        self.moving = Some(ev.raw - anchor);
                        return;
                    }
                    // Clicking elsewhere commits the current text and starts a new one.
                    self.apply(ctx);
                }
                ctx.begin_drawing(ToolKind::Text, Button::Primary);
                self.anchor = Some(ev.pos);
                self.button = ev.button;
                self.render_preview(ctx);
            }
            PointerKind::Drag => {
                if let Some(off) = self.moving {
                    self.anchor = Some(ev.raw - off);
                    self.render_preview(ctx);
                }
            }
            PointerKind::Release => self.moving = None,
            PointerKind::Hover => {}
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, _doc: &Document) {
        if let Some(q) = &self.corners {
            let pts: Vec<_> = q.iter().chain(std::iter::once(&q[0])).map(|p| view.to_screen(*p)).collect();
            painter.extend(egui::Shape::dashed_line(&pts, Stroke::new(1.0, Color32::from_rgb(0, 160, 255)), 5.0, 3.0));
        }
        if let Some(a) = self.anchor {
            painter.circle_filled(view.to_screen(a), 3.0, Color32::from_rgb(0, 160, 255));
        }
    }

    fn cancel(&mut self, ctx: &mut ToolCtx) {
        self.anchor = None;
        self.corners = None;
        self.moving = None;
        if let Some(live) = self.live.take() {
            live.cancel(ctx.doc);
        }
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

    fn adjust_size(&mut self, bigger: bool) {
        self.size = if bigger { self.size * 1.15 } else { self.size / 1.15 }.clamp(4.0, 400.0).round();
        self.needs_render = true;
    }

    fn captures_keys(&self) -> bool {
        self.anchor.is_some()
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "text.all_variants", &mut self.all_variants);
    }
}
