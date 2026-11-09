//! Modal-ish dialog windows.

use std::path::PathBuf;

use eframe::egui::{self, Context, RichText, Ui};
use imged_core::ops::convolve::{EdgeMode, Kernel};
use imged_core::ops::filters::Filter;
use imged_core::{io, Color, Command, Depth, Document};

use crate::actions::{Action, KeyResolver};

/// A dialog window centered on the screen.
fn dialog_window<'a>(ctx: &Context, title: &'a str) -> egui::Window<'a> {
    egui::Window::new(title)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
}

fn depth_ui(ui: &mut Ui, depth: &mut Depth, allowed: &[Depth]) {
    ui.horizontal(|ui| {
        ui.label("Depth");
        for d in Depth::ALL {
            ui.add_enabled_ui(allowed.contains(&d), |ui| {
                ui.selectable_value(depth, d, d.label());
            });
        }
    });
}

// ----- New image -----------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NewFill {
    Transparent,
    White,
    Secondary,
}

pub struct NewImageDialog {
    pub open: bool,
    w: u32,
    h: u32,
    depth: Depth,
    fill: NewFill,
}

impl Default for NewImageDialog {
    fn default() -> Self {
        NewImageDialog { open: false, w: 1024, h: 768, depth: Depth::U8, fill: NewFill::White }
    }
}

impl NewImageDialog {
    /// Returns the new document when confirmed.
    pub fn show(&mut self, ctx: &Context, secondary: Color) -> Option<Document> {
        let mut result = None;
        let mut open = self.open;
        dialog_window(ctx, "New image").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.w).range(1..=32768).suffix(" px").update_while_editing(false));
                ui.label("×");
                crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.h).range(1..=32768).suffix(" px").update_while_editing(false));
            });
            depth_ui(ui, &mut self.depth, &Depth::ALL);
            ui.horizontal(|ui| {
                ui.label("Background");
                ui.selectable_value(&mut self.fill, NewFill::White, "White");
                ui.selectable_value(&mut self.fill, NewFill::Transparent, "Transparent");
                ui.selectable_value(&mut self.fill, NewFill::Secondary, "Secondary color");
            });
            if ui.button("Create").clicked() {
                let fill = match self.fill {
                    NewFill::Transparent => None,
                    NewFill::White => Some(Color::WHITE),
                    NewFill::Secondary => Some(secondary),
                };
                result = Some(Document::new(self.w, self.h, self.depth, fill));
            }
        });
        self.open = open && result.is_none();
        result
    }
}

// ----- Canvas size -----------------------------------------------------------

#[derive(Default)]
pub struct CanvasDialog {
    pub open: bool,
    w: u32,
    h: u32,
    /// 0..=2 in each axis: left/center/right, top/middle/bottom.
    anchor: (u8, u8),
}

impl CanvasDialog {
    pub fn open_for(&mut self, doc: &Document) {
        self.w = doc.width;
        self.h = doc.height;
        self.anchor = (1, 1);
        self.open = true;
    }

    pub fn show(&mut self, ctx: &Context, doc: &Document, actions: &mut Vec<Action>) {
        let mut open = self.open;
        let mut done = false;
        dialog_window(ctx, "Canvas size").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.w).range(1..=65536).suffix(" px").update_while_editing(false));
                ui.label("×");
                crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.h).range(1..=65536).suffix(" px").update_while_editing(false));
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Presets");
                for pct in [50u32, 200] {
                    if ui.small_button(format!("{pct}%")).on_hover_text("Of the current canvas size").clicked() {
                        self.w = (doc.width * pct / 100).max(1);
                        self.h = (doc.height * pct / 100).max(1);
                    }
                }
                for (w, h) in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (1080, 1080), (800, 600)] {
                    if ui.small_button(format!("{w}×{h}")).clicked() {
                        (self.w, self.h) = (w, h);
                    }
                }
                if ui.small_button("Swap").on_hover_text("Swap width and height").clicked() {
                    std::mem::swap(&mut self.w, &mut self.h);
                }
            });
            ui.label("Anchor");
            egui::Grid::new("anchor").spacing([2.0, 2.0]).show(ui, |ui| {
                for y in 0..3u8 {
                    for x in 0..3u8 {
                        let sel = self.anchor == (x, y);
                        if ui.add(egui::Button::new(if sel { "•" } else { " " }).min_size(egui::vec2(22.0, 22.0)).selected(sel)).clicked() {
                            self.anchor = (x, y);
                        }
                    }
                    ui.end_row();
                }
            });
            ui.label(RichText::new("Layers are not scaled or cropped.").small().weak());
            if ui.button("Apply").clicked() {
                let dx = (self.w as i32 - doc.width as i32) * self.anchor.0 as i32 / 2;
                let dy = (self.h as i32 - doc.height as i32) * self.anchor.1 as i32 / 2;
                actions.push(Action::Cmd(Command::CanvasSize { width: self.w, height: self.h, anchor: (dx, dy) }));
                done = true;
            }
        });
        self.open = open && !done;
    }
}

// ----- Convolution --------------------------------------------------------------

pub struct ConvolutionDialog {
    pub open: bool,
    kernel: Kernel,
    /// The kernel as editable text (rows of comma-separated numbers).
    text: String,
    text_error: Option<String>,
    auto_divisor: bool,
    preview: bool,
    previewing: bool,
    dirty: bool,
}

impl Default for ConvolutionDialog {
    fn default() -> Self {
        let kernel = Kernel::from_weights(3, vec![1., 2., 1., 2., 4., 2., 1., 2., 1.]);
        ConvolutionDialog {
            open: false,
            text: kernel_to_text(&kernel),
            text_error: None,
            kernel,
            auto_divisor: true,
            preview: true,
            previewing: false,
            dirty: true,
        }
    }
}

impl ConvolutionDialog {
    pub fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "convolution.auto_divisor", &mut self.auto_divisor);
        p.bool(load, "convolution.include_alpha", &mut self.kernel.include_alpha);
        p.bool(load, "convolution.preview", &mut self.preview);
    }

    pub fn cancel_preview(&mut self, doc: &mut Document) {
        if self.previewing {
            doc.clear_preview();
            self.previewing = false;
        }
        self.dirty = true;
    }

    pub fn show(&mut self, ctx: &Context, doc: &mut Document, scope: imged_core::ops::transform::Scope, actions: &mut Vec<Action>) {
        if !self.open {
            self.cancel_preview(doc);
            return;
        }
        let mut open = self.open;
        let mut apply = false;
        let mut changed = false;
        dialog_window(ctx, "Convolution").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                egui::ComboBox::from_label("Preset").selected_text("Choose…").show_ui(ui, |ui| {
                    for (name, k) in Kernel::presets() {
                        if ui.selectable_label(false, name).clicked() {
                            self.kernel = k;
                            self.auto_divisor = (self.kernel.divisor - self.kernel.auto_divisor()).abs() < 1e-6;
                            changed = true;
                        }
                    }
                });
                ui.label("Size");
                for s in [3usize, 5, 7, 9] {
                    if ui.selectable_label(self.kernel.size == s, format!("{s}×{s}")).clicked() && self.kernel.size != s {
                        self.kernel = self.kernel.resized(s);
                        changed = true;
                    }
                }
            });
            let te = ui.add(egui::TextEdit::multiline(&mut self.text)
                    .font(egui::TextStyle::Monospace)
                    .desired_rows(9)
                    .desired_width(380.0)
                    .hint_text("1, 2, 1\n2, 4, 2\n1, 2, 1"));
            if te.changed() {
                match parse_kernel(&self.text) {
                    Ok((n, w)) => {
                        self.kernel = Kernel { size: n, weights: w, ..self.kernel.clone() };
                        self.text_error = None;
                        changed = true;
                    }
                    Err(e) => self.text_error = Some(e),
                }
            }
            if te.lost_focus() && self.text_error.is_none() {
                self.text = kernel_to_text(&self.kernel);
            }
            match &self.text_error {
                Some(e) => {
                    ui.colored_label(egui::Color32::from_rgb(255, 120, 110), e);
                }
                None => {
                    ui.label(
                        RichText::new("Rows on lines, numbers separated by commas or spaces. Fractions like 1/16 work. Paste with Ctrl+V.")
                            .small()
                            .weak(),
                    );
                }
            }

            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut self.auto_divisor, "Auto divisor").changed();
                ui.add_enabled_ui(!self.auto_divisor, |ui| {
                    changed |= crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.kernel.divisor).speed(0.1).prefix("divisor ").update_while_editing(false)).changed();
                });
                changed |= crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.kernel.bias).speed(0.01).prefix("bias ").update_while_editing(false)).changed();
            });
            ui.horizontal(|ui| {
                changed |= ui
                    .checkbox(&mut self.kernel.include_alpha, "Include alpha")
                    .on_hover_text("Also filter transparency (blurs edges of transparent areas)")
                    .changed();
                ui.label("Edges");
                for (m, l) in [(EdgeMode::Clamp, "Extend"), (EdgeMode::Transparent, "Transparent"), (EdgeMode::Wrap, "Wrap")] {
                    changed |= ui.selectable_value(&mut self.kernel.edge, m, l).changed();
                }
            });
            changed |= ui.checkbox(&mut self.preview, "Preview").changed();
            ui.horizontal(|ui| {
                if ui.button("Apply").clicked() {
                    apply = true;
                }
                if ui.button("Identity").on_hover_text("Reset the kernel").clicked() {
                    self.kernel = Kernel::identity(self.kernel.size);
                    changed = true;
                }
                if ui.button("Copy").on_hover_text("Copy the kernel as text").clicked() {
                    ui.ctx().copy_text(kernel_to_text(&self.kernel));
                }
            });
        });
        // Presets, size buttons and Identity rewrite the text.
        if changed && self.text_error.is_none() && parse_kernel(&self.text).ok() != Some((self.kernel.size, self.kernel.weights.clone())) {
            self.text = kernel_to_text(&self.kernel);
        }
        if self.auto_divisor {
            self.kernel.divisor = self.kernel.auto_divisor();
        }
        if changed {
            self.dirty = true;
        }
        if apply {
            self.cancel_preview(doc);
            actions.push(Action::Filter(Filter::Convolve(self.kernel.clone())));
            open = false;
        } else if self.preview && self.dirty && open {
            self.dirty = false;
            let f = Filter::Convolve(self.kernel.clone());
            imged_core::command::preview_filter(doc, &f, scope);
            self.previewing = true;
        } else if !self.preview && self.previewing {
            self.cancel_preview(doc);
        }
        self.open = open;
        if !open {
            self.cancel_preview(doc);
        }
    }
}

/// Formats a number compactly: integers without decimals, otherwise up to 4 decimals.
fn fmt_num(v: f32) -> String {
    if v.fract() == 0.0 && v.abs() < 1e7 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// The kernel as rows of comma-separated numbers, right-aligned in columns.
pub fn kernel_to_text(k: &Kernel) -> String {
    let cells: Vec<String> = k.weights.iter().map(|v| fmt_num(*v)).collect();
    let width = cells.iter().map(|s| s.chars().count()).max().unwrap_or(1);
    (0..k.size)
        .map(|y| {
            (0..k.size)
                .map(|x| format!("{:>width$}", cells[y * k.size + x]))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parses rows of numbers (separated by commas, semicolons, tabs or spaces;
/// rows by new lines). Returns the side length and the weights.
pub fn parse_kernel(text: &str) -> Result<(usize, Vec<f32>), String> {
    let parse = |t: &str| -> Result<f32, String> {
        let t = t.trim();
        if let Some((a, b)) = t.split_once('/') {
            let (a, b): (f32, f32) =
                (a.trim().parse().map_err(|_| format!("'{t}' isn't a number"))?, b.trim().parse().map_err(|_| format!("'{t}' isn't a number"))?);
            if b == 0.0 {
                return Err(format!("'{t}' divides by zero"));
            }
            return Ok(a / b);
        }
        t.parse::<f32>().map_err(|_| format!("'{t}' isn't a number"))
    };
    let rows: Vec<Vec<f32>> = text
        .lines()
        .map(|l| l.trim().trim_end_matches([',', ';']))
        .filter(|l| !l.is_empty())
        .map(|l| l.split([',', ';', ' ', '\t']).filter(|s| !s.trim().is_empty()).map(parse).collect::<Result<Vec<_>, _>>())
        .collect::<Result<_, _>>()?;
    let n = rows.len();
    if n == 0 {
        return Err("The kernel is empty".into());
    }
    if let Some((i, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != n) {
        return Err(format!("Row {} has {} numbers; a {n}-row kernel needs {n} per row", i + 1, r.len()));
    }
    if n % 2 == 0 {
        return Err(format!("{n}×{n} has no center pixel; use an odd size (3×3, 5×5, …)"));
    }
    Ok((n, rows.into_iter().flatten().collect()))
}

// ----- Export ---------------------------------------------------------------------

pub struct ExportDialog {
    pub open: bool,
    pub path: PathBuf,
    path_text: String,
    pub depth: Depth,
    pub background: Color,
}

impl Default for ExportDialog {
    fn default() -> Self {
        ExportDialog { open: false, path: PathBuf::new(), path_text: String::new(), depth: Depth::U8, background: Color::WHITE }
    }
}

pub struct ExportRequest {
    pub path: PathBuf,
    pub depth: Depth,
    pub background: Color,
}

impl ExportDialog {
    pub fn open_for(&mut self, path: PathBuf, doc: &Document) {
        self.path_text = path.display().to_string();
        self.path = path;
        self.depth = doc.max_depth();
        self.open = true;
    }

    pub fn show(&mut self, ctx: &Context) -> Option<ExportRequest> {
        let mut open = self.open;
        let mut req = None;
        dialog_window(ctx, "Export").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("File");
                if ui.add(egui::TextEdit::singleline(&mut self.path_text).desired_width(300.0)).changed() {
                    self.path = PathBuf::from(&self.path_text);
                }
                if ui.button("Browse…").clicked() {
                    let mut d = rfd::FileDialog::new();
                    if let Some(dir) = self.path.parent().filter(|p| p.is_dir()) {
                        d = d.set_directory(dir);
                    }
                    if let Some(name) = self.path.file_name() {
                        d = d.set_file_name(name.to_string_lossy());
                    }
                    for ext in io::save_extensions() {
                        d = d.add_filter(ext.to_uppercase(), &[ext]);
                    }
                    if let Some(p) = d.save_file() {
                        self.path_text = p.display().to_string();
                        self.path = p;
                    }
                }
            });
            ui.label(RichText::new(format!("Formats: {}", io::save_extensions().join(", "))).small().weak());
            let ext = io::extension(&self.path);
            let allowed = io::save_depths(&ext);
            if allowed.is_empty() {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("Unknown format. Use one of: {}", io::save_extensions().join(", ")));
                return;
            }
            if !allowed.contains(&self.depth) {
                self.depth = *allowed.iter().find(|d| **d >= self.depth).unwrap_or(allowed.last().unwrap());
            }
            depth_ui(ui, &mut self.depth, allowed);
            if io::lacks_alpha(&ext) {
                ui.horizontal(|ui| {
                    ui.label("This format has no transparency; background:");
                    let mut c = super::widgets::to_color32(self.background);
                    if ui.color_edit_button_srgba(&mut c).changed() {
                        let [r, g, b, _] = c.to_srgba_unmultiplied();
                        self.background = Color::from_rgba8(r, g, b, 255);
                    }
                });
            }
            if ui.button("Save").clicked() {
                req = Some(ExportRequest { path: self.path.clone(), depth: self.depth, background: self.background });
            }
        });
        self.open = open && req.is_none();
        req
    }
}

// ----- Save project ----------------------------------------------------------------

#[derive(Default)]
pub struct SaveProjectDialog {
    pub open: bool,
    path: String,
}

impl SaveProjectDialog {
    pub fn open_for(&mut self, suggested: PathBuf) {
        // Try the system dialog first; fall back to typing a path.
        let mut d = rfd::FileDialog::new().add_filter("Photoshop / GIMP project (PSD)", &["psd"]).set_title("Save project");
        if let Some(dir) = suggested.parent().filter(|p| p.is_dir()) {
            d = d.set_directory(dir);
        }
        if let Some(n) = suggested.file_name() {
            d = d.set_file_name(n.to_string_lossy());
        }
        self.path = suggested.display().to_string();
        self.open = true;
        if let Some(p) = d.save_file() {
            self.path = p.display().to_string();
        }
    }

    /// Returns the chosen path (with a .psd extension) when confirmed.
    pub fn show(&mut self, ctx: &Context) -> Option<PathBuf> {
        let mut open = self.open;
        let mut res = None;
        dialog_window(ctx, "Save project").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("File");
                let te = ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(340.0));
                if te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    res = Some(PathBuf::from(self.path.trim()));
                }
            });
            ui.label(
                RichText::new("Saved as PSD with all layers, names, positions, visibility, opacity, groups and adjustment layers. Photoshop, GIMP and Krita open it.")
                    .small()
                    .weak(),
            );
            if ui.button("Save").clicked() {
                res = Some(PathBuf::from(self.path.trim()));
            }
        });
        let res = res.filter(|p| !p.as_os_str().is_empty()).map(|p| {
            if io::is_project(&p) { p } else { p.with_extension("psd") }
        });
        self.open = open && res.is_none();
        res
    }
}

// ----- Open from path (fallback when no file dialog is available) -----------

#[derive(Default)]
pub struct OpenPathDialog {
    pub open: bool,
    pub as_layer: bool,
    path: String,
}

impl OpenPathDialog {
    pub fn show(&mut self, ctx: &Context) -> Option<(PathBuf, bool)> {
        let mut open = self.open;
        let mut res = None;
        let title = if self.as_layer { "Open as layer" } else { "Open" };
        dialog_window(ctx, title).open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.label("No file dialog is available; type a path:");
            let te = ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(360.0));
            if ui.button("Open").clicked() || (te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                res = Some((PathBuf::from(self.path.trim()), self.as_layer));
            }
        });
        self.open = open && res.is_none();
        res
    }
}

// ----- Key help -----------------------------------------------------------------

pub fn keys_window(ctx: &Context, open: &mut bool, keys: &dyn KeyResolver) {
    use crate::tools::ToolKind;
    dialog_window(ctx, "Keyboard shortcuts").open(open).collapsible(false).show(ctx, |ui| {
        let row = |ui: &mut Ui, a: Action, label: &str| {
            if let Some(s) = keys.shortcut_for(&a) {
                ui.label(label);
                ui.monospace(s);
                ui.end_row();
            }
        };
        egui::Grid::new("keys").striped(true).show(ui, |ui| {
            for t in ToolKind::ALL {
                row(ui, Action::SelectTool(t), t.name());
            }
            row(ui, Action::SwapColors, "Swap colors");
            row(ui, Action::ResetColors, "Black / white");
            row(ui, Action::ToggleSnapping, "Toggle snapping");
            row(ui, Action::BrushBigger, "Bigger brush / text");
            row(ui, Action::BrushSmaller, "Smaller brush / text");
            row(ui, Action::Cmd(Command::Undo), "Undo");
            row(ui, Action::Cmd(Command::Redo), "Redo");
            row(ui, Action::ZoomIn, "Zoom in");
            row(ui, Action::ZoomOut, "Zoom out");
            row(ui, Action::ZoomFit, "Fit to window");
            row(ui, Action::ZoomActual, "100%");
            row(ui, Action::Open, "Open");
            row(ui, Action::Export, "Export");
            row(ui, Action::Cmd(Command::NewLayer), "New layer");
            row(ui, Action::Cmd(Command::DuplicateLayer), "Duplicate layer");
            row(ui, Action::Cmd(Command::MergeDown), "Merge down");
            row(ui, Action::Cmd(Command::Group), "Group ticked layers");
            row(ui, Action::Cmd(Command::Ungroup), "Ungroup");
            row(ui, Action::Filter(Filter::Invert), "Invert");
            row(ui, Action::Filter(Filter::AutoLevels), "Auto levels");
        });
        ui.label(RichText::new("Mouse: wheel zooms, middle-drag or Space+drag pans.").small().weak());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_text_roundtrip() {
        let k = Kernel::from_weights(3, vec![1., -2., 1., 2., 40., 2., 0.5, 2., 1.]);
        let t = kernel_to_text(&k);
        assert_eq!(t, "  1,  -2,   1\n  2,  40,   2\n0.5,   2,   1");
        assert_eq!(parse_kernel(&t).unwrap(), (3, k.weights.clone()));
        assert_eq!(parse_kernel("1 2 1\n2\t4 2;\n1,2,1").unwrap().1, vec![1., 2., 1., 2., 4., 2., 1., 2., 1.]);
        assert_eq!(parse_kernel("1/16, 1/8\n1, 1").unwrap_err(), "2×2 has no center pixel; use an odd size (3×3, 5×5, …)");
        assert!(parse_kernel("1, 2\n1").is_err());
        assert!((parse_kernel("1/16").unwrap().1[0] - 0.0625).abs() < 1e-6);
    }
}
