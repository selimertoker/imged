//! The application: layout, input routing and action dispatch.

use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui::{self, Color32, CursorIcon, Key, PointerButton, RichText, Sense, Stroke, StrokeKind, TextureHandle, Ui};
use imged_core::command::{ExecContext, Outcome};
use imged_core::composite::{composite_document, flatten};
use imged_core::ops::edges::{snap_point, Snapped};
use imged_core::ops::paint::PaintMode;
use imged_core::ops::text::FontLibrary;
use imged_core::ops::transform::Scope;
use imged_core::{io, Color, Command, Depth, Document, Px, Rect as IRect};

use crate::actions::{Action, Chord, Dialog, KeyResolver, Keymap};
use crate::renderer::{checker_texture, paint_checker, CanvasRenderer, CpuTiledRenderer};
use crate::state::{ColorState, DisplaySettings, Settings, Status};
use crate::tools::{create_tools, Button, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::ui::colors::ColorPanel;
use crate::project::AutoSaver;
use crate::ui::dialogs::{
    keys_window, CanvasDialog, ConvolutionDialog, ExportDialog, NewImageDialog, OpenPathDialog, SaveProjectDialog,
};
use crate::ui::raw::RawDialog;
use crate::ui::filters::FiltersPanel;
use crate::ui::layers::LayersPanel;
use crate::ui::widgets::{icon_button, section_header, Icon};
use crate::view::ViewXform;

#[derive(Default)]
struct PointerState {
    pressed: Option<Button>,
    last_pos: Option<egui::Pos2>,
    snapped: Option<Snapped>,
}

#[derive(Default)]
struct Dialogs {
    new_image: NewImageDialog,
    canvas: CanvasDialog,
    convolution: ConvolutionDialog,
    export: ExportDialog,
    open_path: OpenPathDialog,
    save_project: SaveProjectDialog,
    raw: RawDialog,
    magick: crate::magick::MagickDialog,
    keys: bool,
    /// A document waiting for "discard changes?" confirmation.
    replace: Option<Document>,
}

pub struct ImgedApp {
    /// Number fields grab the mouse while dragged.
    grab: crate::ui::grab::CursorGrab,
    doc: Document,
    view: ViewXform,
    needs_fit: bool,
    /// Keep re-fitting when the canvas area changes size until the user zooms or pans.
    auto_fit: bool,
    last_canvas_rect: egui::Rect,
    /// Document size at the last fit (refit on change while auto-fitting).
    last_doc_size: (u32, u32),
    renderer: Box<dyn CanvasRenderer>,
    /// The wgpu device, when eframe runs on wgpu (for the GPU renderer).
    gpu_state: Option<eframe::egui_wgpu::RenderState>,
    /// View → "GPU rendering" (remembered).
    use_gpu: bool,
    /// View → "GPU filters" (remembered): convolution and resizing on the GPU.
    gpu_filters: bool,
    /// Whether `renderer` is the GPU one.
    renderer_is_gpu: bool,
    checker: Option<TextureHandle>,
    tools: Vec<Box<dyn Tool>>,
    current: ToolKind,
    colors: ColorState,
    settings: Settings,
    display: DisplaySettings,
    fonts: Option<FontLibrary>,
    status: Status,
    keys: Box<dyn KeyResolver>,
    color_panel: ColorPanel,
    layers_panel: LayersPanel,
    filters_panel: FiltersPanel,
    /// Cached outline of the selection for marching ants.
    ants: crate::ui::ants::Ants,
    /// A paste event arrived for the current Ctrl+V press.
    paste_seen: bool,
    /// The document is a new canvas (see [`Self::is_blank`]).
    blank: bool,
    paste_watch: PasteWatch,
    dialogs: Dialogs,
    /// Color under the cursor while the color picker is active (shown, not picked).
    hover_color: Option<Color>,
    pending: Vec<Action>,
    pointer: PointerState,
    cursor_info: Option<(i32, i32, Px)>,
    title: String,
    /// The PSD this document was opened from / saved to.
    project_path: Option<PathBuf>,
    /// `content_token` at the last save/open (None: never saved).
    saved_token: Option<u64>,
    autosave: AutoSaver,
    /// What the settings file holds now.
    prefs: crate::persist::Prefs,
    prefs_checked: Option<std::time::Instant>,
}

impl ImgedApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        // Small print (tool help, hints, layer details) one notch bigger.
        cc.egui_ctx.all_styles_mut(|style| {
            if let Some(f) = style.text_styles.get_mut(&egui::TextStyle::Small) {
                f.size = 11.5;
            }
        });
        let mut app = ImgedApp {
            grab: Default::default(),
            doc: Document::new(800, 600, Depth::U8, Some(Color::WHITE)),
            view: ViewXform::default(),
            needs_fit: true,
            auto_fit: true,
            last_canvas_rect: egui::Rect::NOTHING,
            last_doc_size: (0, 0),
            renderer: Box::new(CpuTiledRenderer::default()),
            gpu_state: cc.wgpu_render_state.clone().inspect(crate::gpu_filters::register),
            use_gpu: std::env::var("IMGED_RENDERER").map_or(true, |v| !v.eq_ignore_ascii_case("cpu")),
            renderer_is_gpu: false,
            gpu_filters: true,
            checker: None,
            tools: create_tools(),
            current: ToolKind::Brush,
            colors: ColorState::default(),
            settings: Settings::default(),
            display: DisplaySettings::default(),
            fonts: None,
            status: Status::default(),
            keys: Box::new(Keymap::default()),
            color_panel: ColorPanel::default(),
            layers_panel: LayersPanel::default(),
            filters_panel: FiltersPanel::default(),
            ants: Default::default(),
            paste_seen: false,
            blank: true,
            paste_watch: PasteWatch::default(),
            dialogs: Dialogs::default(),
            hover_color: None,
            pending: Vec::new(),
            pointer: PointerState::default(),
            cursor_info: None,
            title: String::new(),
            project_path: None,
            saved_token: None,
            autosave: AutoSaver::default(),
            prefs: crate::persist::Prefs::default(),
            prefs_checked: None,
        };
        let mut p = crate::persist::load();
        app.sync_prefs(&mut p, true);
        app.prefs = app.collect_prefs();
        if !files.is_empty() {
            app.open_paths(&files, false);
        }
        app
    }

    // ----- remembered settings ----------------------------------------------

    /// Loads (`load`) or stores every remembered option.
    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        let s = &mut self.settings;
        for (key, v) in [
            ("anti_aliasing", &mut s.antialias),
            ("snap_to_edges", &mut s.snapping),
            ("grow_layers", &mut s.grow_layers),
            ("keep_ratio", &mut s.keep_ratio),
            ("move_after_select", &mut s.move_after_select),
            ("select_to_layer", &mut s.select_to_layer),
            ("select_add", &mut s.select_add),
            ("new_layer_on_color", &mut s.new_layer_on_color),
            ("new_layer_on_tool", &mut s.new_layer_on_tool),
            ("checkerboard", &mut self.display.checkerboard),
            ("gpu_rendering", &mut self.use_gpu),
            ("gpu_filters", &mut self.gpu_filters),
        ] {
            p.bool(load, key, v);
        }
        p.string(load, "magick_command", &mut s.magick_command);
        self.dialogs.convolution.sync_prefs(p, load);
        for t in &mut self.tools {
            t.sync_prefs(p, load);
        }
        if load {
            if let Some(pal) = &p.palette {
                let colors: Vec<Color> = pal.iter().filter_map(|h| Color::from_hex(h)).collect();
                if !colors.is_empty() || pal.is_empty() {
                    self.colors.palette = colors;
                }
            }
        } else {
            p.palette = Some(self.colors.palette.iter().map(|c| c.to_hex()).collect());
        }
    }

    /// Switches between the GPU and CPU canvas renderers (View → GPU rendering).
    fn update_renderer(&mut self) {
        imged_core::accel::set_enabled(self.gpu_filters);
        let want = self.use_gpu && self.gpu_state.is_some();
        if want == self.renderer_is_gpu {
            return;
        }
        self.renderer = match (&self.gpu_state, want) {
            (Some(rs), true) => Box::new(crate::renderer_gpu::GpuRenderer::new(rs)),
            _ => Box::new(CpuTiledRenderer::default()),
        };
        self.renderer_is_gpu = want;
        self.doc.mark_all_dirty();
    }

    fn collect_prefs(&mut self) -> crate::persist::Prefs {
        let mut p = crate::persist::Prefs::default();
        self.sync_prefs(&mut p, false);
        p
    }

    /// Writes the settings file when something changed (checked twice a second).
    fn save_prefs_if_changed(&mut self) {
        if self.prefs_checked.is_some_and(|t| t.elapsed() < Duration::from_millis(500)) {
            return;
        }
        self.prefs_checked = Some(std::time::Instant::now());
        let p = self.collect_prefs();
        if p != self.prefs {
            if let Err(e) = crate::persist::save(&p) {
                self.status.error(format!("Couldn't save settings: {e}"));
            }
            self.prefs = p;
        }
    }

    // ----- helpers ----------------------------------------------------------

    fn tool_index(&self) -> usize {
        ToolKind::ALL.iter().position(|k| *k == self.current).unwrap()
    }

    /// Runs `f` with the current tool and a context borrowing the rest of the app.
    fn with_tool<R>(&mut self, f: impl FnOnce(&mut dyn Tool, &mut ToolCtx) -> R) -> R {
        let idx = self.tool_index();
        let mut ctx = ToolCtx {
            doc: &mut self.doc,
            colors: &mut self.colors,
            settings: &mut self.settings,
            fonts: &mut self.fonts,
            status: &mut self.status,
            zoom: self.view.zoom,
            actions: &mut self.pending,
        };
        f(self.tools[idx].as_mut(), &mut ctx)
    }

    /// Removes previews owned by panels/dialogs (they would conflict with a new edit).
    fn cancel_panel_previews(&mut self) {
        if self.filters_panel.is_previewing() {
            self.filters_panel.cancel(&mut self.doc);
        }
        self.dialogs.convolution.cancel_preview(&mut self.doc);
    }

    /// Whether the document has changes that aren't in a saved project.
    fn is_modified(&self) -> bool {
        match self.saved_token {
            Some(t) => t != self.doc.content_token(),
            None => !self.is_untouched(),
        }
    }

    fn doc_name(&self) -> String {
        self.project_path
            .as_ref()
            .or(self.doc.path.as_ref())
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".into())
    }

    fn save_project(&mut self, path: PathBuf) {
        self.cancel_panel_previews();
        self.with_tool(|t, c| t.commit(c));
        match imged_core::io::psd::save(&self.doc, &path) {
            Ok(depth) => {
                self.status.info(format!("Saved project {} ({})", path.display(), depth.label()));
                self.project_path = Some(path);
                self.saved_token = Some(self.doc.content_token());
            }
            Err(e) => self.status.error(format!("Couldn't save the project: {e}")),
        }
    }

    fn save_project_as(&mut self) {
        let suggested = self.project_path.clone().unwrap_or_else(|| {
            let base = self.doc.path.clone().unwrap_or_else(|| {
                std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join("untitled")
            });
            base.with_extension("psd")
        });
        self.dialogs.save_project.open_for(suggested);
    }

    /// Clipboard → new layer (or a new image if nothing has been done yet).
    fn paste_image(&mut self) {
        match crate::clipboard::paste_image() {
            Ok(crate::clipboard::Pasted::Image(px, fmt)) => {
                let (w, h) = (px.width(), px.height());
                if self.is_blank() {
                    let mut doc = Document::from_pixels(px, "Pasted");
                    doc.clear_history();
                    self.replace_document(doc);
                } else {
                    // Top-left of the visible canvas area.
                    let id = self.doc.add_layer_from("Pasted", px, (0, 0));
                    self.doc.set_active(id);
                }
                self.status.info(format!("Pasted a {w}×{h} image ({fmt})"));
            }
            Ok(crate::clipboard::Pasted::Files(files)) => {
                let as_layer = !self.is_blank();
                self.open_paths(&files, as_layer);
            }
            Err(e) => self.status.error(format!("Couldn't paste: {e}")),
        }
    }

    /// The pointer when it is over the canvas, otherwise the canvas middle.
    fn zoom_anchor(&self, ctx: &egui::Context) -> egui::Pos2 {
        let area = self.last_canvas_rect;
        ctx.pointer_hover_pos().filter(|p| area.contains(*p)).unwrap_or(area.center())
    }

    /// Nothing done to the document since it was created or opened.
    fn is_untouched(&self) -> bool {
        self.doc.undo_label().is_none() && self.doc.redo_label().is_none() && self.doc.layers.len() == 1 && self.project_path.is_none()
    }

    /// An untouched new canvas (not an opened or pasted image): pasting or
    /// dropping files replaces it instead of adding layers.
    fn is_blank(&self) -> bool {
        self.blank && self.is_untouched()
    }

    fn set_document(&mut self, doc: Document) {
        self.with_tool(|t, ctx| t.cancel(ctx));
        self.autosave.reset();
        let token = doc.content_token();
        self.project_path = doc.path.clone().filter(|p| io::is_project(p));
        self.saved_token = self.project_path.as_ref().map(|_| token);
        self.filters_panel = FiltersPanel::default();
        self.dialogs.convolution.open = false;
        self.display.exposure = 0.0;
        self.display.srgb_encode = doc.linear_hint;
        self.doc = doc;
        self.blank = false;
        self.layers_panel = LayersPanel::default();
        self.needs_fit = true;
    }

    /// Replaces the document, asking first if the current one has edits.
    fn replace_document(&mut self, doc: Document) {
        if self.is_untouched() {
            self.set_document(doc);
        } else {
            self.dialogs.replace = Some(doc);
        }
    }

    fn load_named(&mut self, path: &Path) -> Option<(String, io::Loaded)> {
        match io::load(path) {
            Ok(l) => {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "Image".into());
                Some((name, l))
            }
            Err(e) => {
                self.status.error(format!("Couldn't open {}: {e}", path.display()));
                None
            }
        }
    }

    /// Opens images. As layers: each is added to the current image. Otherwise
    /// the first becomes a new image and the rest are added to it as layers
    /// (with the canvas grown to fit them all).
    pub fn open_paths(&mut self, paths: &[PathBuf], as_layer: bool) {
        // Projects (PSD): one opens as a document; with others, its layers are added.
        let (projects, images): (Vec<PathBuf>, Vec<PathBuf>) = paths.iter().cloned().partition(|p| io::is_project(p));
        if !projects.is_empty() {
            let mut projects = projects.into_iter();
            if !as_layer && images.is_empty() && paths.len() == 1 {
                let p = projects.next().unwrap();
                match imged_core::io::psd::load(&p) {
                    Ok(mut doc) => {
                        doc.path = Some(p.clone());
                        let n = doc.layers.len();
                        let depth = doc.max_depth();
                        self.replace_document(doc);
                        self.status.info(format!("Opened project {} ({n} layers, {})", p.display(), depth.label()));
                    }
                    Err(e) => self.status.error(format!("Couldn't open {}: {e}", p.display())),
                }
                return;
            }
            for p in projects {
                match imged_core::io::psd::load(&p) {
                    Ok(src) => {
                        for l in src.layers {
                            let id = self.doc.add_layer_from(&l.name, (*l.pixels).clone(), l.offset);
                            if let Some(nl) = self.doc.layer_mut(id) {
                                nl.visible = l.visible;
                                nl.opacity = l.opacity;
                            }
                        }
                    }
                    Err(e) => self.status.error(format!("Couldn't open {}: {e}", p.display())),
                }
            }
            if images.is_empty() {
                return;
            }
            return self.open_paths(&images, true);
        }
        let loaded: Vec<(String, io::Loaded)> = paths.iter().filter_map(|p| self.load_named(p)).collect();
        if loaded.is_empty() {
            return;
        }
        let n = loaded.len();
        if as_layer {
            for (name, l) in loaded {
                self.doc.add_layer_from(&name, l.pixels, (0, 0));
            }
            self.status.info(if n == 1 { "Added 1 layer".to_string() } else { format!("Added {n} layers") });
            return;
        }
        let first_path = paths.first().cloned();
        let linear = loaded[0].1.linear;
        let (name, depth, hdr) = (loaded[0].0.clone(), loaded[0].1.pixels.depth(), loaded[0].1.pixels.has_hdr_values());
        let mut doc = if n == 1 {
            let (name, l) = loaded.into_iter().next().unwrap();
            Document::from_pixels(l.pixels, &name)
        } else {
            // Several images: each becomes a layer above a transparent
            // background, so all of them can be tiled.
            let deepest = loaded.iter().map(|(_, l)| l.pixels.depth()).max().unwrap_or(Depth::U8);
            let w = loaded.iter().map(|(_, l)| l.pixels.width()).max().unwrap_or(1);
            let h = loaded.iter().map(|(_, l)| l.pixels.height()).max().unwrap_or(1);
            let mut doc = Document::new(w, h, deepest, None);
            for (name, l) in loaded {
                doc.add_layer_from(&name, l.pixels, (0, 0));
            }
            doc.fit_canvas_to_layers();
            doc
        };
        doc.clear_history();
        doc.path = first_path;
        doc.linear_hint = linear;
        self.replace_document(doc);
        self.status.info(if n == 1 {
            format!("Opened {name} ({}{})", depth.label(), if hdr { ", contains HDR values" } else { "" })
        } else {
            format!("Opened {n} images as layers. The Tile button arranges them side by side.")
        });
    }

    fn open_dialog(&mut self, as_layer: bool) {
        let exts = io::open_extensions();
        let mut d = rfd::FileDialog::new().add_filter("Images", &exts).add_filter("All files", &["*"]);
        if let Some(dir) = self.doc.path.as_ref().and_then(|p| p.parent()) {
            d = d.set_directory(dir);
        }
        d = d.set_title(if as_layer { "Open as layers" } else { "Open (select several to get layers)" });
        if let Some(paths) = d.pick_files() {
            self.open_paths(&paths, as_layer);
        }
    }

    /// Copies the flattened canvas to the clipboard as a PNG image.
    fn copy_image(&mut self) {
        self.cancel_panel_previews();
        let img = flatten(&self.doc).convert::<u8>();
        match crate::clipboard::set_image(img.width as usize, img.height as usize, img.data) {
            Ok(()) => self.status.info(format!("Copied {}×{} image to the clipboard", self.doc.width, self.doc.height)),
            Err(e) => self.status.error(format!("Couldn't copy to the clipboard: {e}")),
        }
    }

    fn export(&mut self, path: &Path, depth: Depth, background: Color) {
        self.cancel_panel_previews();
        let img = flatten(&self.doc);
        match io::save(path, &img, depth, background.px()) {
            Ok(d) => {
                self.status.info(format!("Exported {} ({})", path.display(), d.label()));
                self.doc.path.get_or_insert_with(|| path.to_path_buf());
            }
            Err(e) => self.status.error(format!("Export failed: {e}")),
        }
    }

    fn exec(&mut self, cmd: Command) {
        let undo_redo = matches!(cmd, Command::Undo | Command::Redo);
        let label = if undo_redo {
            if matches!(cmd, Command::Undo) { self.doc.undo_label() } else { self.doc.redo_label() }
        } else {
            None
        };
        let mut ectx = ExecContext { fonts: self.fonts.as_mut(), grow_layers: self.settings.grow_layers };
        match cmd.execute(&mut self.doc, &mut ectx) {
            Ok(Outcome::Unchanged) if undo_redo => {
                self.status.info(if matches!(cmd, Command::Undo) { "Nothing to undo" } else { "Nothing to redo" })
            }
            Ok(_) if undo_redo => {
                let verb = if matches!(cmd, Command::Undo) { "Undid" } else { "Redid" };
                self.status.info(format!("{verb}: {}", label.unwrap_or_default()));
            }
            Ok(_) => {}
            Err(e) => self.status.error(e),
        }
    }

    fn dispatch(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::Cmd(cmd) => {
                let keeps_tool = matches!(cmd, Command::SetActive(_) | Command::ToggleVisible(_));
                if !keeps_tool {
                    self.cancel_panel_previews();
                    self.with_tool(|t, c| t.cancel(c));
                }
                self.exec(cmd);
            }
            Action::SelectTool(k) => {
                if k != self.current {
                    self.with_tool(|t, c| t.deactivate(c));
                    self.current = k;
                }
            }
            Action::SwapColors => self.colors.swap(),
            Action::ResetColors => {
                self.colors.primary = Color::BLACK;
                self.colors.secondary = Color::WHITE;
            }
            Action::ZoomIn | Action::ZoomOut => {
                let anchor = self.zoom_anchor(ctx);
                let z = self.view.step(matches!(action, Action::ZoomIn));
                self.view.zoom_around(anchor, z);
                self.auto_fit = false;
            }
            Action::ZoomFit => self.needs_fit = true,
            Action::ZoomActual => {
                let anchor = self.zoom_anchor(ctx);
                self.view.zoom_around(anchor, 1.0);
                self.auto_fit = false;
            }
            Action::NewImage => self.dialogs.new_image.open = true,
            Action::Open => self.open_dialog(false),
            Action::OpenAsLayer => self.open_dialog(true),
            Action::Export => {
                let path = self.doc.path.clone().unwrap_or_else(|| {
                    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join("untitled.png")
                });
                self.dialogs.export.open_for(path, &self.doc);
            }
            Action::CopyImage => self.copy_image(),
            Action::Magick(image) => {
                let t = if image { crate::magick::Target::Image } else { crate::magick::Target::Layer };
                self.dialogs.magick.start(&self.doc, &self.settings.magick_command, t);
            }
            Action::ToggleFilterLayer => {
                let id = self.doc.active;
                let on = !self.doc.active_layer().filter_on;
                self.dispatch(Action::Cmd(Command::SetFilterOn { id, on }), ctx);
            }
            Action::PasteImage => self.paste_image(),
            Action::SaveProject => match self.project_path.clone() {
                Some(p) => self.save_project(p),
                None => self.save_project_as(),
            },
            Action::SaveProjectAs => self.save_project_as(),
            Action::ImportRaw => self.dialogs.raw.start(),
            Action::Filter(f) => {
                self.cancel_panel_previews();
                self.with_tool(|t, c| t.cancel(c));
                self.exec(Command::Filter(f, self.settings.transform_scope));
            }
            Action::ShowDialog(d) => match d {
                Dialog::CanvasSize => self.dialogs.canvas.open_for(&self.doc),
                Dialog::Convolution => self.dialogs.convolution.open = true,
                Dialog::Keys => self.dialogs.keys = true,
            },
            Action::ToggleSnapping => {
                self.settings.snapping = !self.settings.snapping;
                self.status.info(if self.settings.snapping { "Snapping on" } else { "Snapping off" });
            }
            Action::BrushBigger | Action::BrushSmaller => {
                let i = self.tool_index();
                self.tools[i].adjust_size(matches!(action, Action::BrushBigger));
            }
            Action::Cancel => {
                self.cancel_panel_previews();
                self.with_tool(|t, c| t.cancel(c));
            }
            Action::Commit => self.with_tool(|t, c| t.commit(c)),
            Action::Nudge(dx, dy) => {
                if self.current == ToolKind::Move {
                    self.exec(Command::NudgeLayerBy { dx, dy });
                }
            }
            Action::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    // ----- input --------------------------------------------------------------

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let captures = self.tools[self.tool_index()].captures_keys();
        let (events, now) = ctx.input(|i| (i.events.clone(), i.time));
        for e in events {
            if matches!(e, egui::Event::Paste(_) | egui::Event::PasteImage(_)) {
                // Ctrl+V outside a text box: paste an image (any format) as a layer.
                self.pending.push(Action::PasteImage);
                self.paste_seen = true;
                continue;
            }
            // egui only emits a paste event for text or PNG clipboards; for
            // other image types (JPEG, BMP, …) only the key release arrives.
            // Ctrl may be released before V, so remember when it was let go.
            if let egui::Event::ModifiersChanged(m) = e {
                if self.paste_watch.command && !m.command {
                    self.paste_watch.released_at = Some(now);
                }
                self.paste_watch.command = m.command;
            }
            if let egui::Event::Key { key: Key::V, pressed: false, modifiers, .. } = e {
                let recent = self.paste_watch.released_at.is_some_and(|t| now - t < 1.0);
                if (modifiers.command || recent) && !std::mem::take(&mut self.paste_seen) {
                    self.pending.push(Action::PasteImage);
                }
                self.paste_watch.released_at = None;
                continue;
            }
            if let egui::Event::Key { pressed: true, .. } = e {
                // A V press reached us, so this was not a swallowed Ctrl+V.
                self.paste_watch.released_at = None;
            }
            if let egui::Event::Key { key, pressed: true, modifiers, .. } = e {
                let chord = Chord::from_event(key, modifiers);
                // While a tool holds uncommitted state, plain letter keys
                // shouldn't switch tools out from under it.
                if captures && !chord.ctrl && !matches!(key, Key::Escape | Key::Enter) {
                    continue;
                }
                let acts = self.keys.resolve(chord);
                self.pending.extend(acts);
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let files: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
        if !files.is_empty() {
            // An untouched document is replaced; otherwise files become layers.
            let as_layer = !self.is_blank();
            self.open_paths(&files, as_layer);
        }
    }

    // ----- panels ---------------------------------------------------------------

    fn menu_bar(&mut self, ui: &mut Ui) {
        let keys = &*self.keys;
        let item = |ui: &mut Ui, label: &str, action: Action, pending: &mut Vec<Action>| {
            let sc = keys.shortcut_for(&action).unwrap_or_default();
            if ui.add(egui::Button::new(label).shortcut_text(sc)).clicked() {
                pending.push(action);
            }
        };
        let p = &mut self.pending;
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                item(ui, "New…", Action::NewImage, p);
                item(ui, "Open…", Action::Open, p);
                item(ui, "Open as layer…", Action::OpenAsLayer, p);
                if ui.button("Open from path…").clicked() {
                    self.dialogs.open_path.as_layer = false;
                    self.dialogs.open_path.open = true;
                }
                if ui.button("Open as layer from path…").clicked() {
                    self.dialogs.open_path.as_layer = true;
                    self.dialogs.open_path.open = true;
                }
                item(ui, "Import raw…", Action::ImportRaw, p);
                ui.separator();
                item(ui, "Save project", Action::SaveProject, p);
                item(ui, "Save project as…", Action::SaveProjectAs, p);
                item(ui, "Export…", Action::Export, p);
                ui.separator();
                item(ui, "Paste image from clipboard", Action::PasteImage, p);
                item(ui, "Copy image to clipboard", Action::CopyImage, p);
                ui.separator();
                let autosave = crate::project::autosave_dir();
                ui.label(RichText::new(format!("Autosaves every minute to {}", autosave.display())).small().weak())
                    .on_hover_text(self.autosave.last_path.as_ref().map_or("Nothing autosaved yet".to_string(), |p| format!("Last: {}", p.display())));
                ui.separator();
                item(ui, "Quit", Action::Quit, p);
            });
            ui.menu_button("Edit", |ui| {
                let undo = self.doc.undo_label().map_or("Undo".to_string(), |l| format!("Undo {l}"));
                let redo = self.doc.redo_label().map_or("Redo".to_string(), |l| format!("Redo {l}"));
                item(ui, &undo, Action::Cmd(Command::Undo), p);
                item(ui, &redo, Action::Cmd(Command::Redo), p);
                ui.separator();
                item(ui, "Select all", Action::Cmd(Command::SelectAll), p);
                item(ui, "Deselect", Action::Cmd(Command::SelectNone), p);
                item(ui, "Invert selection", Action::Cmd(Command::InvertSelection), p);
                item(ui, "Cut selection to new layer", Action::Cmd(Command::SelectionToLayer { copy: false }), p);
                item(ui, "Copy selection to new layer", Action::Cmd(Command::SelectionToLayer { copy: true }), p);
                ui.separator();
                item(ui, "Swap colors", Action::SwapColors, p);
                item(ui, "Toggle snapping", Action::ToggleSnapping, p);
            });
            ui.menu_button("Image", |ui| {
                item(ui, "Canvas size…", Action::ShowDialog(Dialog::CanvasSize), p);
                item(ui, "Fit canvas to layers", Action::Cmd(Command::FitCanvasToLayers), p);
                item(ui, "Crop to selection", Action::Cmd(Command::CropToSelection), p);
                item(ui, "Edit with ImageMagick…", Action::Magick(true), p);
                ui.separator();
                for (label, cmd) in [
                    ("Flip image horizontally", Command::Flip { horizontal: true, scope: Scope::Image }),
                    ("Flip image vertically", Command::Flip { horizontal: false, scope: Scope::Image }),
                    ("Rotate image 90° clockwise", Command::RotateQuarter { turns_cw: 1, scope: Scope::Image }),
                    ("Rotate image 90° counter-clockwise", Command::RotateQuarter { turns_cw: 3, scope: Scope::Image }),
                    ("Rotate image 180°", Command::RotateQuarter { turns_cw: 2, scope: Scope::Image }),
                ] {
                    item(ui, label, Action::Cmd(cmd), p);
                }
                ui.separator();
                item(ui, "Flatten", Action::Cmd(Command::Flatten), p);
                ui.menu_button("Depth for new layers", |ui| {
                    for d in Depth::ALL {
                        ui.radio_value(&mut self.doc.default_depth, d, d.label());
                    }
                });
            });
            ui.menu_button("Layer", |ui| {
                item(ui, "New layer", Action::Cmd(Command::NewLayer), p);
                item(ui, "Edit with ImageMagick…", Action::Magick(false), p);
                item(ui, "Toggle filter layer", Action::ToggleFilterLayer, p);
                item(ui, "Duplicate", Action::Cmd(Command::DuplicateLayer), p);
                item(ui, "Delete", Action::Cmd(Command::DeleteLayer), p);
                ui.separator();
                item(ui, "Bring forward", Action::Cmd(Command::MoveLayer { up: true }), p);
                item(ui, "Send backward", Action::Cmd(Command::MoveLayer { up: false }), p);
                ui.separator();
                item(ui, "Merge down", Action::Cmd(Command::MergeDown), p);
                item(ui, "Merge ticked layers", Action::Cmd(Command::MergeSelected), p);
                item(ui, "Group ticked layers", Action::Cmd(Command::Group), p);
                item(ui, "Ungroup", Action::Cmd(Command::Ungroup), p);
                ui.separator();
                item(ui, "Tile layers (no overlaps, square-ish)", Action::Cmd(Command::TileLayers { gap: 0 }), p);
                item(ui, "Fit canvas to layers", Action::Cmd(Command::FitCanvasToLayers), p);
                ui.separator();
                ui.menu_button("Convert depth", |ui| {
                    for d in Depth::ALL {
                        item(ui, d.label(), Action::Cmd(Command::ConvertDepth(d)), p);
                    }
                });
                item(ui, "Fit layer to canvas", Action::Cmd(Command::LayerToCanvasSize), p);
                item(ui, "Trim transparent edges", Action::Cmd(Command::CropLayerToContent), p);
            });
            ui.menu_button("Filters", |ui| {
                use imged_core::ops::filters::Filter;
                item(ui, "Invert", Action::Filter(Filter::Invert), p);
                item(ui, "Auto levels", Action::Filter(Filter::AutoLevels), p);
                item(ui, "Desaturate", Action::Filter(Filter::Saturation(0.0)), p);
                item(ui, "Convolution…", Action::ShowDialog(Dialog::Convolution), p);
            });
            ui.menu_button("View", |ui| {
                item(ui, "Zoom in", Action::ZoomIn, p);
                item(ui, "Zoom out", Action::ZoomOut, p);
                item(ui, "Fit to window", Action::ZoomFit, p);
                item(ui, "Actual size", Action::ZoomActual, p);
                ui.separator();
                ui.checkbox(&mut self.display.checkerboard, "Checkerboard behind transparency");
                ui.checkbox(&mut self.display.srgb_encode, "Linear-to-sRGB display")
                    .on_hover_text("Turn on for scene-linear data (EXR, PFM, Radiance HDR)");
                ui.separator();
                let tip = format!(
                    "Composite the layers on the graphics card (off: on the processor)\nNow: {}",
                    self.renderer.name()
                );
                ui.add_enabled(self.gpu_state.is_some(), egui::Checkbox::new(&mut self.use_gpu, "GPU rendering"))
                    .on_hover_text(tip)
                    .on_disabled_hover_text("Not available: the window isn't drawn with wgpu");
                let accel = imged_core::accel::available();
                let tip = format!(
                    "Run convolution and resizing on the graphics card (off: on the processor)\n{}",
                    accel.as_deref().unwrap_or("")
                );
                if ui
                    .add_enabled(accel.is_some(), egui::Checkbox::new(&mut self.gpu_filters, "GPU filters"))
                    .on_hover_text(tip)
                    .on_disabled_hover_text("Not available: the graphics card can't run compute shaders")
                    .changed()
                {
                    imged_core::accel::set_enabled(self.gpu_filters);
                }
            });
            ui.menu_button("Help", |ui| {
                item(ui, "Keyboard shortcuts", Action::ShowDialog(Dialog::Keys), p);
            });
        });
    }

    fn left_panel(&mut self, ui: &mut Ui) {
        // File actions pinned to the bottom-left corner.
        egui::Panel::bottom("file-buttons").show(ui, |ui| {
            ui.add_space(4.0);
            let keys = &*self.keys;
            let tip = |a: &Action, text: &str| match keys.shortcut_for(a) {
                Some(s) => format!("{text} ({s})"),
                None => text.to_string(),
            };
            let half = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
            let size = egui::vec2(half, 24.0);
            let rows: [[(&str, Action, &str); 2]; 3] = [
                [
                    ("Open…", Action::Open, "Open images or a PSD project; select several images to open them as layers"),
                    ("Open as layer…", Action::OpenAsLayer, "Add images to this one as new layers"),
                ],
                [
                    ("Save project", Action::SaveProject, "Save all layers as a PSD project (Photoshop, GIMP, Krita)"),
                    ("Export…", Action::Export, "Save the flattened image to a file"),
                ],
                [
                    ("Paste image", Action::PasteImage, "Add the image on the clipboard (any format) as a new layer"),
                    ("Copy as PNG", Action::CopyImage, "Copy the whole canvas to the clipboard as a PNG image"),
                ],
            ];
            for row in rows {
                ui.horizontal(|ui| {
                    for (label, action, help) in row {
                        let t = tip(&action, help);
                        if ui.add_sized(size, egui::Button::new(label)).on_hover_text(t).clicked() {
                            self.pending.push(action);
                        }
                    }
                });
            }
            ui.add_space(2.0);
        });

        egui::ScrollArea::vertical().id_salt("left-scroll").show(ui, |ui| {
            section_header(ui, "TOOLS");
            let keys = &*self.keys;
            egui::Grid::new("tool-grid").spacing([4.0, 4.0]).show(ui, |ui| {
                for (i, k) in ToolKind::ALL.iter().enumerate() {
                    let sc = keys.shortcut_for(&Action::SelectTool(*k)).unwrap_or_default();
                    let tip = format!("{} ({sc})\n{}", k.name(), k.help());
                    if icon_button(ui, Icon::Tool(*k), *k == self.current, &tip).clicked() {
                        self.pending.push(Action::SelectTool(*k));
                    }
                    if i % 4 == 3 {
                        ui.end_row();
                    }
                }
            });
            ui.separator();
            section_header(ui, &self.current.name().to_uppercase());
            ui.label(RichText::new(self.current.help()).small().weak());
            self.with_tool(|t, c| t.options_ui(ui, c));
            self.common_tool_options(ui);

            ui.separator();
            section_header(ui, "TRANSFORM & FILTERS");
            ui.horizontal(|ui| {
                ui.label("Apply to");
                ui.selectable_value(&mut self.settings.transform_scope, Scope::Layer, "Layer")
                    .on_hover_text("The active layer (or group)");
                ui.selectable_value(&mut self.settings.transform_scope, Scope::Image, "Image")
                    .on_hover_text("Every layer and the canvas");
                let tip = if self.doc.selection.is_some() {
                    "Only the selected part of the active layer"
                } else {
                    "Only the selected part of the active layer (nothing is selected now, so this acts on the layer)"
                };
                ui.selectable_value(&mut self.settings.transform_scope, Scope::Selection, "Selection").on_hover_text(tip);
            });
            let scope = self.settings.transform_scope;
            ui.horizontal(|ui| {
                for (icon, tip, cmd) in [
                    (Icon::FlipH, "Mirror horizontally", Command::Flip { horizontal: true, scope }),
                    (Icon::FlipV, "Mirror vertically", Command::Flip { horizontal: false, scope }),
                    (Icon::RotateCcw, "Rotate 90° counter-clockwise", Command::RotateQuarter { turns_cw: 3, scope }),
                    (Icon::RotateCw, "Rotate 90° clockwise", Command::RotateQuarter { turns_cw: 1, scope }),
                ] {
                    if icon_button(ui, icon, false, tip).clicked() {
                        self.pending.push(Action::Cmd(cmd));
                    }
                }
            });

            ui.add_space(2.0);
            self.filters_panel.ui(ui, &mut self.doc, self.settings.transform_scope, &mut self.pending);
        });
    }

    /// Options several tools share, shown under the current tool's own.
    fn common_tool_options(&mut self, ui: &mut Ui) {
        use ToolKind::*;
        let k = self.current;
        let s = &mut self.settings;
        if matches!(k, Brush | Pen | Shape | Bucket | Text) {
            ui.checkbox(&mut s.grow_layers, "Grow layer when painting outside it")
                .on_hover_text("Off: painting is clipped to the layer's area");
        }
        if matches!(k, Brush | Pen | Shape | Bucket | Text) {
            ui.checkbox(&mut s.new_layer_on_color, "New layer when the color changes")
                .on_hover_text("Drawing with a different color than last time starts a new layer");
            ui.checkbox(&mut s.new_layer_on_tool, "New layer when the tool changes")
                .on_hover_text("Drawing with a different tool than last time starts a new layer");
        }
        if matches!(k, Brush | Shape | Bucket | Text) {
            ui.horizontal(|ui| {
                ui.label("Paint");
                egui::ComboBox::from_id_salt("paint-mode").selected_text(s.paint_mode.label()).show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.paint_mode, PaintMode::Replace, PaintMode::Replace.label())
                        .on_hover_text("Pixels take the paint color including its alpha; transparent paint erases");
                    ui.selectable_value(&mut s.paint_mode, PaintMode::Over, PaintMode::Over.label())
                        .on_hover_text("Paint is blended over existing pixels");
                });
                ui.checkbox(&mut s.antialias, "Anti-aliasing")
                    .on_hover_text("Smooth edges for painting, shapes and text (off: hard, pixel-exact edges)");
            });
        }
        if self.tools[self.tool_index()].snap_option() {
            let s = &mut self.settings;
            let snap_tip = "Snap to edges in the image, and to the middle of the canvas and of the selection (S)";
            ui.checkbox(&mut s.snapping, "Snap to edges").on_hover_text(snap_tip);
            if s.snapping {
                let mut t = s.snap_threshold * 100.0;
                if ui
                    .add(egui::Slider::new(&mut t, 1.0..=100.0).update_while_editing(false).suffix("%").text("Edge strength"))
                    .on_hover_text("Minimum color difference that counts as an edge")
                    .changed()
                {
                    s.snap_threshold = t / 100.0;
                }
            }
        }
    }

    fn right_panel(&mut self, ui: &mut Ui) {
        // History and zoom, pinned to the bottom of the right panel.
        egui::Panel::bottom("history-zoom").show(ui, |ui| {
            ui.add_space(4.0);
            let keys = &*self.keys;
            for (cmd, label, next) in [
                (Command::Undo, "Undo", self.doc.undo_label()),
                (Command::Redo, "Redo", self.doc.redo_label()),
            ] {
                ui.horizontal(|ui| {
                    let sc = keys.shortcut_for(&Action::Cmd(cmd.clone())).unwrap_or_default();
                    let b = ui
                        .add_enabled(next.is_some(), egui::Button::new(label).min_size(egui::vec2(52.0, 0.0)))
                        .on_hover_text(format!("{label} ({sc})"));
                    if b.clicked() {
                        self.pending.push(Action::Cmd(cmd));
                    }
                    let text = next.unwrap_or_else(|| "nothing".into());
                    ui.add(egui::Label::new(RichText::new(text).weak()).truncate());
                });
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Zoom");
                // The buttons zoom around the middle of the canvas area.
                let mid = self.last_canvas_rect.center();
                if ui.button(" - ").on_hover_text("Zoom out (-)").clicked() {
                    let z = self.view.step(false);
                    self.view.zoom_around(mid, z);
                    self.auto_fit = false;
                }
                let mut pct = self.view.zoom * 100.0;
                let r = crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut pct)
                        .range(2.0..=6400.0)
                        .speed(self.view.zoom.max(0.05))
                        .max_decimals(1)
                        .suffix(" %")
                        .update_while_editing(false),
                );
                if r.changed() {
                    self.view.zoom_around(mid, pct / 100.0);
                    self.auto_fit = false;
                }
                if ui.button(" + ").on_hover_text("Zoom in (=)").clicked() {
                    let z = self.view.step(true);
                    self.view.zoom_around(mid, z);
                    self.auto_fit = false;
                }
                if ui.small_button("Fit").on_hover_text("Fit to window (0)").clicked() {
                    self.pending.push(Action::ZoomFit);
                }
                if ui.small_button("1:1").on_hover_text("100% (1)").clicked() {
                    self.pending.push(Action::ZoomActual);
                }
            });
            ui.add_space(2.0);
        });
        section_header(ui, "COLOR");
        self.color_panel.ui(ui, &mut self.colors, self.hover_color);
        ui.separator();
        section_header(ui, "LAYERS");
        self.layers_panel.ui(ui, &mut self.doc, &mut self.pending);
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if let Some((msg, err, t)) = &self.status.message {
                let age = t.elapsed().as_secs_f32();
                if age < 8.0 {
                    let c = if *err { Color32::from_rgb(255, 120, 110) } else { ui.visuals().text_color() };
                    ui.colored_label(c, msg);
                    ui.ctx().request_repaint_after(Duration::from_millis(500));
                    ui.separator();
                }
            }
            if let Some((x, y, p)) = self.cursor_info {
                ui.monospace(format!("{x}, {y}"));
                self.pixel_readout(ui, p);
            }
            if let Some(s) = self.tools[self.tool_index()].status() {
                ui.separator();
                ui.monospace(s);
            }
            if self.settings.snapping {
                ui.label(RichText::new("SNAP").small().color(crate::ui::widgets::ACCENT));
            }
            if let Some(l) = self.pointer.snapped.and_then(|s| s.guide_label()) {
                ui.label(RichText::new(format!("snapped to {l}")).small().color(Color32::from_rgb(80, 220, 255)));
            }
            if let Some(m) = self.keys.mode_label() {
                ui.monospace(m);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.monospace(format!("{:.0}%", self.view.zoom * 100.0));
                ui.separator();
                ui.monospace(format!(
                    "{}×{} · {} layer{} · {:.1} MB",
                    self.doc.width,
                    self.doc.height,
                    self.doc.layers.len(),
                    if self.doc.layers.len() == 1 { "" } else { "s" },
                    self.doc.pixel_bytes() as f64 / 1_048_576.0
                ));
                ui.separator();
                ui.checkbox(&mut self.display.srgb_encode, "sRGB")
                    .on_hover_text("Display linear data with sRGB encoding (view only)");
                crate::ui::grab::drag_value(ui, egui::DragValue::new(&mut self.display.exposure).range(-16.0..=16.0).speed(0.05).prefix("EV ").update_while_editing(false))
                    .on_hover_text("Display exposure in stops (view only; drag, double-click to type)");
                if self.display.exposure != 0.0 && ui.small_button("0").on_hover_text("Reset exposure").clicked() {
                    self.display.exposure = 0.0;
                }
            });
        });
    }

    /// Pixel values at the image's depth (the deepest visible layer), each
    /// channel in its own color, followed by the maximum value.
    fn pixel_readout(&self, ui: &mut Ui, p: Px) {
        let depth = self.doc.layers.iter().filter(|l| l.visible).map(|l| l.depth()).max().unwrap_or(Depth::U8);
        let fmt = |v: f32| match depth {
            Depth::U8 => format!("{}", (v.clamp(0.0, 1.0) * 255.0).round() as u32),
            Depth::U16 => format!("{}", (v.clamp(0.0, 1.0) * 65535.0).round() as u32),
            Depth::F32 => format!("{v:.4}"),
        };
        let max = match depth {
            Depth::U8 => "255",
            Depth::U16 => "65535",
            Depth::F32 => "1.0",
        };
        let colors = [
            Color32::from_rgb(255, 105, 105),
            Color32::from_rgb(110, 225, 110),
            Color32::from_rgb(110, 160, 255),
            Color32::WHITE,
        ];
        ui.spacing_mut().item_spacing.x = 6.0;
        let r = ui
            .horizontal(|ui| {
                for (v, c) in p.iter().zip(colors) {
                    ui.label(RichText::new(fmt(*v)).monospace().color(c));
                }
                ui.label(RichText::new(format!("/ {max}")).monospace().weak());
            })
            .response;
        r.on_hover_text(format!(
            "Red, green, blue, alpha of the visible image under the cursor ({}; previews included)",
            depth.label()
        ));
    }

    // ----- canvas ----------------------------------------------------------------

    fn canvas(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let doc_size = (self.doc.width, self.doc.height);
        if rect != self.last_canvas_rect || doc_size != self.last_doc_size {
            self.last_canvas_rect = rect;
            self.last_doc_size = doc_size;
            if self.auto_fit {
                self.needs_fit = true;
            }
        }
        if self.needs_fit {
            self.view.fit(rect, (self.doc.width, self.doc.height));
            self.needs_fit = false;
            self.auto_fit = true;
        }

        let (hover, pressed_p, pressed_s, released_p, released_s, mods, scroll, zoom_delta, space, delta) = ui.input(|i| {
            (
                i.pointer.hover_pos(),
                i.pointer.button_pressed(PointerButton::Primary),
                i.pointer.button_pressed(PointerButton::Secondary),
                i.pointer.button_released(PointerButton::Primary),
                i.pointer.button_released(PointerButton::Secondary),
                i.modifiers,
                i.smooth_scroll_delta,
                i.zoom_delta(),
                i.key_down(Key::Space),
                i.pointer.delta(),
            )
        });

        // Zoom (wheel, pinch, Ctrl+wheel) around the pointer.
        if resp.hovered() {
            if let Some(p) = hover {
                if zoom_delta != 1.0 {
                    self.view.zoom_around(p, self.view.zoom * zoom_delta);
                    self.auto_fit = false;
                } else if scroll.y != 0.0 {
                    let f = (1.0015f32).powf(scroll.y);
                    self.view.zoom_around(p, self.view.zoom * f);
                    self.auto_fit = false;
                }
            }
        }
        // Pan with middle drag or Space+drag.
        let panning = resp.dragged_by(PointerButton::Middle) || (space && resp.dragged_by(PointerButton::Primary));
        if panning {
            self.view.origin += delta;
            self.auto_fit = false;
        }

        // Tool input.
        let snaps = self.settings.snapping && self.tools[self.tool_index()].uses_snapping();
        if !snaps || !resp.hovered() {
            self.pointer.snapped = None;
        }
        let to_event = |app: &mut Self, kind: PointerKind, pos: egui::Pos2, button: Button| -> PointerEvent {
            let raw = app.view.to_image(pos);
            let mut p = raw;
            app.pointer.snapped = None;
            if snaps {
                let s = snap_point(&app.doc, raw, app.settings.snap_params(app.view.zoom));
                p = s.point;
                if s.x || s.y {
                    app.pointer.snapped = Some(s);
                }
            }
            PointerEvent { kind, pos: p, raw, button, shift: mods.shift, ctrl: mods.command, alt: mods.alt }
        };
        let mut events = Vec::new();
        if let Some(pos) = hover {
            let moved = self.pointer.last_pos != Some(pos);
            let mut just_pressed = false;
            if self.pointer.pressed.is_none() && !space && resp.hovered() && (pressed_p || pressed_s) {
                let b = if pressed_p { Button::Primary } else { Button::Secondary };
                self.pointer.pressed = Some(b);
                events.push((PointerKind::Press, pos, b));
                just_pressed = true;
            }
            if let Some(b) = self.pointer.pressed {
                let released = match b {
                    Button::Primary => released_p,
                    Button::Secondary => released_s,
                };
                if moved && !just_pressed {
                    events.push((PointerKind::Drag, pos, b));
                }
                // A quick click can press and release within one frame.
                if released {
                    self.pointer.pressed = None;
                    events.push((PointerKind::Release, pos, b));
                }
            } else if moved && resp.hovered() {
                events.push((PointerKind::Hover, pos, Button::Primary));
            }
            self.pointer.last_pos = Some(pos);
        } else if let Some(b) = self.pointer.pressed {
            // Released outside the window.
            if released_p || released_s {
                self.pointer.pressed = None;
                let pos = self.pointer.last_pos.unwrap_or_default();
                events.push((PointerKind::Release, pos, b));
            }
        }
        for (kind, pos, b) in events {
            if kind == PointerKind::Press {
                self.cancel_panel_previews();
            }
            let ev = to_event(self, kind, pos, b);
            if std::env::var_os("IMGED_DEBUG_INPUT").is_some() {
                eprintln!("{:?} {:?} {:?}", ev.kind, ev.pos, ev.button);
            }
            self.with_tool(|t, c| t.pointer(&ev, c));
        }

        // Cursor readout.
        self.cursor_info = hover.filter(|_| resp.hovered()).and_then(|p| {
            let ip = self.view.to_image(p);
            let (x, y) = (ip.x.floor() as i32, ip.y.floor() as i32);
            self.doc
                .canvas_rect()
                .contains(x, y)
                .then(|| (x, y, composite_document(&self.doc, IRect::new(x, y, 1, 1)).get(0, 0)))
        });

        // Color picker: show the color under the cursor in the color panel.
        let hover_color = match (self.current, hover.filter(|_| resp.hovered())) {
            (ToolKind::Eyedropper, Some(p)) => self.tools[self.tool_index()].hover_color(&self.doc, self.view.to_image(p)),
            _ => None,
        };
        if hover_color != self.hover_color {
            self.hover_color = hover_color;
            ui.ctx().request_repaint();
        }

        // Render.
        let ctx = ui.ctx().clone();
        self.renderer.sync(&ctx, &mut self.doc, &self.display);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(38));
        let canvas_screen = self.view.rect_to_screen(self.doc.canvas_rect());
        let shadow = canvas_screen.translate(egui::vec2(3.0, 3.0));
        painter.rect_filled(shadow, 0.0, Color32::from_black_alpha(90));
        if self.display.checkerboard {
            let checker = self.checker.get_or_insert_with(|| checker_texture(&ctx));
            paint_checker(&painter, checker, canvas_screen.intersect(rect), 8.0);
        }
        self.renderer.paint(&painter, &self.view);
        painter.rect_stroke(canvas_screen, 0.0, Stroke::new(1.0, Color32::from_gray(20)), StrokeKind::Outside);

        // Active layer bounds when it differs from the canvas.
        let lb = self.doc.targets_bounds();
        if lb != self.doc.canvas_rect() && !lb.is_empty() && self.current != ToolKind::Move {
            let r = self.view.rect_to_screen(lb);
            painter.extend(egui::Shape::dashed_line(
                &[r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()],
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 200, 0, 140)),
                4.0,
                4.0,
            ));
        }

        if let Some(sel) = &self.doc.selection {
            let t = ui.input(|i| i.time);
            self.ants.paint(&painter, &self.view, sel, t);
            ui.ctx().request_repaint_after(Duration::from_millis(120));
        }
        self.tools[self.tool_index()].overlay(&painter, &self.view, &self.doc);

        // Snap guides.
        if let Some(s) = self.pointer.snapped {
            // Edges in magenta, the canvas/selection middle in cyan (dashed).
            let edge = Stroke::new(1.0, Color32::from_rgb(255, 0, 200));
            let mid = Stroke::new(1.0, Color32::from_rgb(80, 220, 255));
            let p = self.view.to_screen(s.point);
            let line = |a: egui::Pos2, b: egui::Pos2, guide: bool| {
                if guide {
                    painter.extend(egui::Shape::dashed_line(&[a, b], mid, 6.0, 4.0));
                } else {
                    painter.line_segment([a, b], edge);
                }
            };
            if s.x {
                line(egui::pos2(p.x, rect.top()), egui::pos2(p.x, rect.bottom()), s.x_guide.is_some());
            }
            if s.y {
                line(egui::pos2(rect.left(), p.y), egui::pos2(rect.right(), p.y), s.y_guide.is_some());
            }
        }

        if resp.hovered() {
            let icon = if panning || space {
                CursorIcon::Grabbing
            } else {
                self.tools[self.tool_index()].cursor()
            };
            ctx.set_cursor_icon(icon);
        }
    }

    fn dialogs(&mut self, ctx: &egui::Context) {
        if let Some(cmd) = self.dialogs.magick.show(ctx, &self.doc, &mut self.settings.magick_command) {
            self.pending.push(Action::Cmd(cmd));
        }
        if let Some(doc) = self.dialogs.new_image.show(ctx, self.colors.secondary) {
            let direct = self.is_untouched();
            self.replace_document(doc);
            self.blank = direct;
        }
        self.dialogs.canvas.show(ctx, &self.doc, &mut self.pending);
        self.dialogs.convolution.show(ctx, &mut self.doc, self.settings.transform_scope, &mut self.pending);
        if let Some(req) = self.dialogs.export.show(ctx) {
            self.export(&req.path, req.depth, req.background);
        }
        if let Some(p) = self.dialogs.save_project.show(ctx) {
            self.save_project(p);
        }
        if let Some(r) = self.dialogs.raw.show(ctx) {
            if r.as_layer {
                let id = self.doc.add_layer_from(&r.name, r.pixels, (0, 0));
                self.doc.set_active(id);
            } else {
                let mut doc = Document::from_pixels(r.pixels, &r.name);
                doc.clear_history();
                self.replace_document(doc);
            }
        }
        if let Some((p, as_layer)) = self.dialogs.open_path.show(ctx) {
            self.open_paths(&[p], as_layer);
        }
        let mut keys_open = self.dialogs.keys;
        keys_window(ctx, &mut keys_open, &*self.keys);
        self.dialogs.keys = keys_open;

        if self.dialogs.replace.is_some() {
            let mut decision = None;
            egui::Window::new("Discard changes?").pivot(egui::Align2::CENTER_CENTER).default_pos(ctx.content_rect().center()).collapsible(false).resizable(false).show(ctx, |ui| {
                ui.label("The current image has changes that will be lost.");
                ui.horizontal(|ui| {
                    if ui.button("Discard and continue").clicked() {
                        decision = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        decision = Some(false);
                    }
                });
            });
            match decision {
                Some(true) => {
                    let d = self.dialogs.replace.take().unwrap();
                    self.set_document(d);
                }
                Some(false) => self.dialogs.replace = None,
                None => {}
            }
        }
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        let name = self.doc_name();
        let t = format!("{}{name} — imged", if self.is_modified() { "*" } else { "" });
        if t != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(t.clone()));
            self.title = t;
        }
        let n = self.doc_name();
        self.autosave.tick(&self.doc, &n);
        if let Some(e) = self.autosave.last_error.take() {
            self.status.error(format!("Autosave failed: {e}"));
        }
        // Wake up for the next autosave check even when idle.
        ctx.request_repaint_after(crate::project::AUTOSAVE_EVERY);
    }
}

impl eframe::App for ImgedApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_keys(&ctx);
        self.handle_dropped_files(&ctx);
        self.update_renderer();

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("tools").resizable(false).exact_size(276.0).show(ui, |ui| self.left_panel(ui));
        egui::Panel::right("side").resizable(false).exact_size(300.0).show(ui, |ui| self.right_panel(ui));
        egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| self.canvas(ui));
        self.dialogs(&ctx);

        let actions = std::mem::take(&mut self.pending);
        if !actions.is_empty() {
            for a in actions {
                self.dispatch(a, &ctx);
            }
            ctx.request_repaint();
        }
        self.update_title(&ctx);
        self.save_prefs_if_changed();
        self.grab.end_frame(&ctx);
    }

    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        self.grab.filter_input(ctx, raw_input);
    }
}

/// Tracks Ctrl for detecting Ctrl+V when egui swallowed it without an event.
#[derive(Default)]
struct PasteWatch {
    command: bool,
    released_at: Option<f64>,
}
