//! Layer / Image → "Edit with ImageMagick…": the active layer and the ticked
//! ones (or the flattened image) are written to a temporary folder, the
//! command line the user edits is run, and its output replaces the layers
//! (or the image). Operations that make several images (e.g. -fft) give one
//! layer each. The temporary files are deleted afterwards.
//!
//! The program is configurable (remembered in the settings file), so
//! GraphicsMagick (`gm convert`) or ImageMagick 6 (`convert`) work too.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use eframe::egui::{self, Color32, Context, RichText};
use imged_core::composite::flatten;
use imged_core::command::NewLayer;
use imged_core::{Command, Depth, Document, LayerId, Pixels};

/// The first of `magick`, `convert` (ImageMagick 6) and `gm convert` found
/// in PATH (`magick` if none is).
pub fn default_command() -> String {
    let found = |name: &str| {
        std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|d| d.join(name).is_file()))
    };
    if found("magick") {
        "magick".into()
    } else if found("convert") {
        "convert".into()
    } else if found("gm") {
        "gm convert".into()
    } else {
        "magick".into()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Layer,
    Image,
}

struct Running {
    rx: Receiver<Result<Vec<Pixels>, String>>,
    target: Target,
    /// The layers given to the program (top first), with names and offsets.
    inputs: Vec<(LayerId, String, (i32, i32))>,
}

#[derive(Default)]
pub struct MagickDialog {
    pub open: bool,
    target: Option<Target>,
    dir: Option<PathBuf>,
    /// The whole command line, as typed.
    line: String,
    /// The input files and the output file as currently written in `line`.
    paths: (String, String),
    error: Option<String>,
    running: Option<Running>,
}

fn quote(p: &Path) -> String {
    shell_words::quote(&p.display().to_string()).into_owned()
}

/// File extension for the given depth: 8/16-bit as PNG, float as TIFF.
fn ext(depth: Depth) -> &'static str {
    match depth {
        Depth::F32 => "tif",
        _ => "png",
    }
}

/// Input file names: `in.png`, or `in-0.png`, `in-1.png`… for several.
fn input_names(depth: Depth, n: usize) -> Vec<String> {
    let e = ext(depth);
    if n <= 1 {
        vec![format!("in.{e}")]
    } else {
        (0..n).map(|i| format!("in-{i}.{e}")).collect()
    }
}

/// The layers given to the program for `t`: the active layer and the
/// ticked ones, top first (none for the whole image).
fn input_layers(doc: &Document, t: Target) -> Vec<LayerId> {
    match t {
        Target::Image => Vec::new(),
        Target::Layer => {
            let mut ids = doc.channel_merge_set();
            ids.sort_by_key(|id| std::cmp::Reverse(doc.index_of(*id).unwrap_or(0)));
            ids
        }
    }
}

fn depth_for(doc: &Document, t: Target) -> Depth {
    match t {
        Target::Layer => input_layers(doc, t)
            .iter()
            .filter_map(|id| doc.layer(*id))
            .map(|l| l.depth())
            .max_by_key(|d| d.bytes_per_channel())
            .unwrap_or(Depth::U8),
        Target::Image => doc.max_depth(),
    }
}

/// The (inputs, output) parts of the command line for `t`.
fn path_args(doc: &Document, dir: &Path, t: Target) -> (String, String) {
    let depth = depth_for(doc, t);
    let n = input_layers(doc, t).len();
    let inputs: Vec<String> = input_names(depth, n).iter().map(|f| quote(&dir.join(f))).collect();
    let out = quote(&dir.join(format!("out.{}", ext(depth))));
    // A multi-page TIFF would be one file: +adjoin writes a file per image.
    let out = if depth == Depth::F32 { format!("+adjoin {out}") } else { out };
    (inputs.join(" "), out)
}

/// The program's output files: `out.ext`, or `out-0.ext`, `out-1.ext`…
/// (ImageMagick numbers them when an operation like -fft makes several).
fn output_files(dir: &Path, e: &str) -> Vec<PathBuf> {
    let mut found: Vec<(i64, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let stem = name.strip_suffix(&format!(".{e}"))?;
            let n = if stem == "out" { -1 } else { stem.strip_prefix("out-")?.parse::<i64>().ok()? };
            Some((n, entry.path()))
        })
        .collect();
    found.sort();
    found.into_iter().map(|(_, p)| p).collect()
}

impl MagickDialog {
    /// Opens the window for `target`, with the command line pre-filled.
    pub fn start(&mut self, doc: &Document, command: &str, target: Target) {
        if self.running.is_some() {
            self.open = true;
            return;
        }
        self.cleanup();
        let dir = std::env::temp_dir().join(format!("imged-magick-{:08x}", random()));
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.error = Some(format!("Couldn't create {}: {e}", dir.display()));
        }
        let (inp, out) = path_args(doc, &dir, target);
        self.line = format!("{command} {inp} {out}");
        self.paths = (inp, out);
        self.dir = Some(dir);
        self.target = Some(target);
        self.error = None;
        self.open = true;
    }

    fn set_target(&mut self, doc: &Document, t: Target) {
        self.target = Some(t);
        let Some(dir) = &self.dir else { return };
        let (inp, out) = path_args(doc, dir, t);
        self.line = self.line.replace(&self.paths.0, &inp).replace(&self.paths.1, &out);
        self.paths = (inp, out);
    }

    /// Keeps the input files in the command line in step with the layers
    /// ticked in the layer panel while the window is open.
    fn refresh_inputs(&mut self, doc: &Document) {
        let (Some(dir), Some(t)) = (&self.dir, self.target) else { return };
        if self.running.is_some() {
            return;
        }
        let args = path_args(doc, dir, t);
        if args != self.paths {
            self.line = self.line.replace(&self.paths.0, &args.0).replace(&self.paths.1, &args.1);
            self.paths = args;
        }
    }

    /// Deletes the temporary folder.
    fn cleanup(&mut self) {
        if let Some(d) = self.dir.take() {
            std::fs::remove_dir_all(d).ok();
        }
    }

    fn run(&mut self, doc: &Document) {
        let (Some(dir), Some(target)) = (self.dir.clone(), self.target) else { return };
        let argv = match shell_words::split(&self.line) {
            Ok(a) if !a.is_empty() => a,
            Ok(_) => return self.error = Some("The command line is empty".into()),
            Err(e) => return self.error = Some(format!("Couldn't parse the command line: {e}")),
        };
        let depth = depth_for(doc, target);
        let ids = input_layers(doc, target);
        let images: Vec<_> = match target {
            Target::Layer => ids.iter().filter_map(|id| doc.layer(*id)).map(|l| l.pixels.to_f32()).collect(),
            Target::Image => vec![flatten(doc)],
        };
        for (name, img) in input_names(depth, images.len()).iter().zip(&images) {
            let path = dir.join(name);
            if let Err(e) = imged_core::io::save(&path, img, depth, [1.0; 4]) {
                return self.error = Some(format!("Couldn't write {}: {e}", path.display()));
            }
        }
        let e = ext(depth);
        for old in output_files(&dir, e) {
            std::fs::remove_file(old).ok();
        }
        let inputs = ids.iter().filter_map(|id| doc.layer(*id)).map(|l| (l.id, l.name.clone(), l.offset)).collect();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let res = std::process::Command::new(&argv[0])
                .args(&argv[1..])
                .output()
                .map_err(|e| format!("Couldn't run '{}': {e}", argv[0]))
                .and_then(|o| {
                    if o.status.success() {
                        Ok(())
                    } else {
                        let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                        Err(if err.is_empty() { format!("'{}' failed ({})", argv[0], o.status) } else { err })
                    }
                })
                .and_then(|_| {
                    let files = output_files(&dir, e);
                    if files.is_empty() {
                        return Err(format!(
                            "The program wrote no out.{e} in {}\n(Is the output file name in the command?)",
                            dir.display()
                        ));
                    }
                    files
                        .iter()
                        .map(|f| {
                            imged_core::io::load(f)
                                .map(|l| l.pixels)
                                .map_err(|err| format!("Couldn't read the result {}: {err}", f.display()))
                        })
                        .collect()
                });
            tx.send(res).ok();
        });
        self.error = None;
        self.running = Some(Running { rx, target, inputs });
    }

    /// The command that applies the program's output images.
    fn result_command(r: &Running, outputs: Vec<Pixels>, doc: &Document) -> Command {
        let label = "ImageMagick".to_string();
        let n = outputs.len();
        match r.target {
            Target::Layer => {
                let one_to_one = n == r.inputs.len();
                let (_, top_name, top_off) = r.inputs.first().cloned().unwrap_or((0, "Layer".into(), (0, 0)));
                let results = outputs
                    .into_iter()
                    .enumerate()
                    .map(|(i, pixels)| {
                        let (name, offset) = if one_to_one {
                            (r.inputs[i].1.clone(), r.inputs[i].2)
                        } else if n == 1 {
                            (top_name.clone(), top_off)
                        } else {
                            (format!("{top_name} ({i})"), top_off)
                        };
                        NewLayer { pixels, offset, name }
                    })
                    .collect();
                Command::ReplaceLayers { ids: r.inputs.iter().map(|i| i.0).collect(), results, label }
            }
            Target::Image => {
                let base = doc.layers.last().map_or("Image".to_string(), |l| l.name.clone());
                let results = outputs
                    .into_iter()
                    .enumerate()
                    .map(|(i, pixels)| {
                        let name = if n == 1 { base.clone() } else { format!("{base} ({i})") };
                        NewLayer { pixels, offset: (0, 0), name }
                    })
                    .collect();
                Command::ReplaceImage { results, label }
            }
        }
    }

    /// Shows the window. Returns the command that applies the result, once
    /// the program has finished.
    pub fn show(&mut self, ctx: &Context, doc: &Document, command: &mut String) -> Option<Command> {
        let mut result = None;
        if let Some(r) = &self.running {
            match r.rx.try_recv() {
                Ok(Ok(outputs)) => {
                    result = Some(Self::result_command(r, outputs, doc));
                    self.running = None;
                    self.cleanup();
                    self.open = false;
                }
                Ok(Err(e)) => {
                    self.error = Some(e);
                    self.running = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
                Err(_) => self.running = None,
            }
        }
        if !self.open {
            return result;
        }
        self.refresh_inputs(doc);
        let mut open = true;
        let mut run = false;
        egui::Window::new("Edit with ImageMagick")
            .open(&mut open)
            .collapsible(false)
            .default_width(560.0)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.content_rect().center())
            .show(ctx, |ui| {
                let cur = self.target.unwrap_or(Target::Layer);
                let layers = input_layers(doc, Target::Layer);
                ui.horizontal(|ui| {
                    ui.label("Apply to");
                    let layer_label =
                        if layers.len() > 1 { format!("Layers ({})", layers.len()) } else { "Layer".to_string() };
                    for (t, label, tip) in [
                        (
                            Target::Layer,
                            layer_label.as_str(),
                            "The active layer and the ticked ones. One output per input replaces each layer; \
                             any other number of outputs replaces them with new layers.",
                        ),
                        (Target::Image, "Image", "The flattened image; the output replaces all layers"),
                    ] {
                        if ui.selectable_label(cur == t, label).on_hover_text(tip).clicked() && cur != t {
                            self.set_target(doc, t);
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Program");
                    let before = command.clone();
                    let r = ui.add(egui::TextEdit::singleline(command).desired_width(160.0).font(egui::TextStyle::Monospace));
                    r.on_hover_text("e.g. magick, convert (ImageMagick 6), gm convert (GraphicsMagick) or a full path. Remembered.");
                    if *command != before && self.line.starts_with(before.as_str()) {
                        self.line = format!("{command}{}", &self.line[before.len()..]);
                    }
                });
                ui.label(
                    RichText::new(
                        "Add the operations between the input and output files, e.g.  -blur 0x3  or  -resize 50%. \
                         Operations that make several images (like -fft) give one layer each.",
                    )
                    .small()
                    .weak(),
                );
                if cur == Target::Layer && layers.len() > 1 {
                    let names = input_names(depth_for(doc, cur), layers.len());
                    let list: Vec<String> = names
                        .iter()
                        .zip(&layers)
                        .filter_map(|(f, id)| Some(format!("{f} = {}", doc.layer(*id)?.name)))
                        .collect();
                    ui.label(RichText::new(list.join(",  ")).small().weak());
                }
                ui.add(
                    egui::TextEdit::multiline(&mut self.line)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace),
                );
                if let Some(e) = &self.error {
                    egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        ui.colored_label(Color32::from_rgb(255, 120, 110), e);
                    });
                }
                ui.horizontal(|ui| {
                    if self.running.is_some() {
                        ui.spinner();
                        ui.label("Running…");
                    } else if ui.button("Run").on_hover_text("Run the command and replace the layers/image with its output").clicked() {
                        run = true;
                    }
                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
            });
        if run {
            self.run(doc);
        }
        if !open || !self.open {
            self.open = false;
            if self.running.is_none() {
                self.cleanup();
            }
        }
        result
    }
}

fn random() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
    h.finish() as u32
}
