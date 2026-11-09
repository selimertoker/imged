//! File → Import raw…: open headerless pixel data by describing it.

use std::path::PathBuf;

use eframe::egui::{self, Color32, ColorImage, Context, RichText, TextureHandle, TextureOptions, Ui};
use imged_core::io::raw::{decode, guess_sizes, RawLayout, RawParams, RawSample};
use imged_core::Pixels;

use super::widgets::paint_checker_rect;

#[derive(Default)]
pub struct RawDialog {
    pub open: bool,
    path: String,
    bytes: Option<(PathBuf, Vec<u8>)>,
    load_error: Option<String>,
    params: RawParams,
    preview: Option<TextureHandle>,
    preview_for: Option<(PathBuf, RawParams)>,
    decode_error: Option<String>,
}

/// What the user asked for.
pub struct RawImport {
    pub pixels: Pixels,
    pub name: String,
    pub as_layer: bool,
}

fn num(v: &mut u32, range: std::ops::RangeInclusive<u32>) -> egui::DragValue<'_> {
    egui::DragValue::new(v).range(range).update_while_editing(false)
}

impl RawDialog {
    /// Opens the dialog, asking for a file first.
    pub fn start(&mut self) {
        self.open = true;
        if let Some(p) = rfd::FileDialog::new().set_title("Import raw data").pick_file() {
            self.path = p.display().to_string();
            self.load();
        }
    }

    fn load(&mut self) {
        let p = PathBuf::from(self.path.trim());
        match std::fs::read(&p) {
            Ok(b) => {
                // Pick the most square size that fits exactly, if any.
                if let Some((w, h)) = guess_sizes(b.len(), &self.params).first() {
                    self.params.width = *w;
                    self.params.height = *h;
                }
                self.bytes = Some((p, b));
                self.load_error = None;
            }
            Err(e) => {
                self.bytes = None;
                self.load_error = Some(e.to_string());
            }
        }
        self.preview_for = None;
    }

    fn update_preview(&mut self, ctx: &Context) {
        let Some((path, bytes)) = &self.bytes else { return };
        let key = (path.clone(), self.params);
        if self.preview_for.as_ref() == Some(&key) {
            return;
        }
        self.preview_for = Some(key);
        let p = self.params;
        if (p.width as u64 * p.height as u64) > 50_000_000 {
            self.decode_error = Some("Too large to preview".into());
            self.preview = None;
            return;
        }
        match decode(bytes, &p) {
            Ok(px) => {
                self.decode_error = None;
                let scale = (256.0 / p.width.max(p.height) as f32).min(1.0);
                let (tw, th) = (((p.width as f32 * scale) as usize).max(1), ((p.height as f32 * scale) as usize).max(1));
                let mut img = Vec::with_capacity(tw * th);
                for y in 0..th {
                    for x in 0..tw {
                        let sx = (((x as f32 + 0.5) / scale) as u32).min(p.width - 1);
                        let sy = (((y as f32 + 0.5) / scale) as u32).min(p.height - 1);
                        let c = px.get(sx, sy);
                        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        img.push(Color32::from_rgba_unmultiplied(q(c[0]), q(c[1]), q(c[2]), q(c[3])));
                    }
                }
                let ci = ColorImage::new([tw, th], img);
                match &mut self.preview {
                    Some(t) => t.set(ci, TextureOptions::NEAREST),
                    None => self.preview = Some(ctx.load_texture("raw-preview", ci, TextureOptions::NEAREST)),
                }
            }
            Err(e) => {
                self.decode_error = Some(e);
                self.preview = None;
            }
        }
    }

    fn form(&mut self, ui: &mut Ui) {
        let p = &mut self.params;
        egui::Grid::new("raw-grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Channels");
            egui::ComboBox::from_id_salt("raw-layout").selected_text(p.layout.label()).show_ui(ui, |ui| {
                for l in RawLayout::ALL {
                    ui.selectable_value(&mut p.layout, l, l.label());
                }
            });
            ui.end_row();
            ui.label("Sample type");
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("raw-sample").selected_text(p.sample.label()).show_ui(ui, |ui| {
                    for s in RawSample::ALL {
                        ui.selectable_value(&mut p.sample, s, s.label());
                    }
                });
                ui.add_enabled_ui(p.sample != RawSample::U8, |ui| {
                    ui.selectable_value(&mut p.big_endian, false, "Little endian");
                    ui.selectable_value(&mut p.big_endian, true, "Big endian");
                });
            });
            ui.end_row();
            ui.label("Size");
            ui.horizontal(|ui| {
                crate::ui::grab::drag_value(ui, num(&mut p.width, 1..=65535).suffix(" px"));
                ui.label("×");
                crate::ui::grab::drag_value(ui, num(&mut p.height, 1..=65535).suffix(" px"));
            });
            ui.end_row();
            ui.label("Header bytes");
            crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut p.offset).range(0..=usize::MAX).update_while_editing(false))
                .on_hover_text("Bytes to skip at the start of the file");
            ui.end_row();
            ui.label("Row padding");
            let r = ui.add_enabled(
                !p.planar,
                egui::DragValue::new(&mut p.row_padding).range(0..=65535).update_while_editing(false).suffix(" bytes"),
            );
            crate::ui::grab::mark(ui, &r);
            r.on_hover_text("Extra bytes at the end of every row");
            ui.end_row();
            ui.label("Storage");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut p.planar, false, "Interleaved (RGBRGB…)");
                ui.selectable_value(&mut p.planar, true, "Planar (RR…GG…BB…)");
            });
            ui.end_row();
            ui.label("");
            ui.checkbox(&mut p.flip_vertical, "Rows are bottom to top");
            ui.end_row();
        });
    }

    pub fn show(&mut self, ctx: &Context) -> Option<RawImport> {
        if !self.open {
            return None;
        }
        let mut open = self.open;
        let mut result = None;
        egui::Window::new("Import raw data")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.content_rect().center())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("File");
                    let r = ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(300.0));
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.load();
                    }
                    if ui.button("Browse…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_file() {
                            self.path = p.display().to_string();
                            self.load();
                        }
                    }
                    if ui.button("Load").clicked() {
                        self.load();
                    }
                });
                if let Some(e) = &self.load_error {
                    ui.colored_label(Color32::from_rgb(255, 120, 110), e);
                }
                ui.separator();
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| self.form(ui));
                    ui.add_space(8.0);
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(256.0, 256.0), egui::Sense::hover());
                    paint_checker_rect(ui.painter(), rect, 8.0);
                    if let Some(t) = &self.preview {
                        let s = t.size_vec2();
                        let fit = (256.0 / s.x).min(256.0 / s.y);
                        let r = egui::Rect::from_center_size(rect.center(), s * fit);
                        ui.painter().image(t.id(), r, egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(1., 1.)), Color32::WHITE);
                    }
                });
                if let Some((_, bytes)) = &self.bytes {
                    let need = self.params.expected_len();
                    let have = bytes.len();
                    let msg = if need == have {
                        RichText::new(format!("{have} bytes: exact fit")).color(Color32::from_rgb(110, 225, 110))
                    } else if need < have {
                        RichText::new(format!("{have} bytes; these settings use {need} ({} left over)", have - need))
                            .color(Color32::from_rgb(240, 200, 80))
                    } else {
                        RichText::new(format!("{have} bytes; these settings need {need}")).color(Color32::from_rgb(255, 120, 110))
                    };
                    ui.label(msg);
                    let guesses = guess_sizes(have, &self.params);
                    if !guesses.is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("Sizes that fit exactly:").small());
                            for (w, h) in guesses {
                                if ui.small_button(format!("{w}×{h}")).clicked() {
                                    self.params.width = w;
                                    self.params.height = h;
                                }
                            }
                        });
                    }
                    if let Some(e) = &self.decode_error {
                        ui.colored_label(Color32::from_rgb(255, 120, 110), e);
                    }
                }
                ui.separator();
                let ready = self.bytes.is_some() && self.decode_error.is_none();
                ui.horizontal(|ui| {
                    for (label, as_layer) in [("Open as new image", false), ("Add as layer", true)] {
                        if ui.add_enabled(ready, egui::Button::new(label)).clicked() {
                            if let Some((path, bytes)) = &self.bytes {
                                match decode(bytes, &self.params) {
                                    Ok(px) => {
                                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                                        result = Some(RawImport { pixels: px, name, as_layer });
                                    }
                                    Err(e) => self.decode_error = Some(e),
                                }
                            }
                        }
                    }
                });
            });
        self.update_preview(ctx);
        self.open = open && result.is_none();
        result
    }
}
