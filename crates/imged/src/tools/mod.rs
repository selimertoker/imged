//! Interactive tools.
//!
//! Each tool receives pointer events in image coordinates (already snapped
//! to edges when snapping is on and the tool opts in), shows its options in
//! the left panel and draws overlays on the canvas. Tools change the
//! document through core [`Command`]s or, for live strokes, through
//! [`LivePaint`] previews that are committed as one undo step.

mod brush;
mod bucket;
mod eyedropper;
mod live;
mod move_tool;
mod pen;
mod region_preview;
mod select;
mod shape;
mod text;
mod resize;
mod rotate;
mod warp;

use eframe::egui::{self, CursorIcon, Key, Painter, Ui};
use imged_core::command::{ExecContext, Outcome};
use imged_core::ops::text::FontLibrary;
use imged_core::{Command, Document, Vec2};

pub use live::LivePaint;

use crate::actions::Action;
use crate::state::{ColorState, Settings, Status};
use crate::view::ViewXform;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolKind {
    Move,
    Brush,
    Pen,
    Eyedropper,
    Shape,
    Bucket,
    Text,
    BoxSelect,
    MagicWand,
    OutlineSelect,
    Resize,
    Rotate,
    Tilt,
    Shear,
}

impl ToolKind {
    pub const ALL: [ToolKind; 14] = [
        ToolKind::Move,
        ToolKind::Brush,
        ToolKind::Pen,
        ToolKind::Eyedropper,
        ToolKind::Shape,
        ToolKind::Bucket,
        ToolKind::Text,
        ToolKind::BoxSelect,
        ToolKind::MagicWand,
        ToolKind::OutlineSelect,
        ToolKind::Resize,
        ToolKind::Rotate,
        ToolKind::Tilt,
        ToolKind::Shear,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ToolKind::Move => "Move layer",
            ToolKind::Brush => "Brush",
            ToolKind::Pen => "Pen",
            ToolKind::Eyedropper => "Color picker",
            ToolKind::Shape => "Shapes",
            ToolKind::Bucket => "Paint bucket",
            ToolKind::Text => "Text",
            ToolKind::BoxSelect => "Box select",
            ToolKind::MagicWand => "Magic wand",
            ToolKind::OutlineSelect => "Outline select",
            ToolKind::Resize => "Resize",
            ToolKind::Rotate => "Rotate",
            ToolKind::Tilt => "Tilt (perspective)",
            ToolKind::Shear => "Shear",
        }
    }

    pub fn key(self) -> Key {
        match self {
            ToolKind::Move => Key::V,
            ToolKind::Brush => Key::B,
            ToolKind::Pen => Key::N,
            ToolKind::Eyedropper => Key::I,
            ToolKind::Shape => Key::U,
            ToolKind::Bucket => Key::G,
            ToolKind::Text => Key::T,
            ToolKind::BoxSelect => Key::M,
            ToolKind::MagicWand => Key::W,
            ToolKind::OutlineSelect => Key::O,
            ToolKind::Resize => Key::E,
            ToolKind::Rotate => Key::R,
            ToolKind::Tilt => Key::P,
            ToolKind::Shear => Key::H,
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            ToolKind::Move => {
                "Drag to move the active layer (or group). With snapping on, its edges snap to visible edges. Arrow keys nudge."
            }
            ToolKind::Brush => "Left: primary color, right: secondary. Shift+click draws a straight line.",
            ToolKind::Pen => {
                "Square, hard-edged pixels: sets them exactly to the color (alpha included). Shift+click draws a straight line."
            }
            ToolKind::Eyedropper => "Hover to preview, left click picks the primary color, right click the secondary.",
            ToolKind::Shape => "Drag to draw. Shift constrains to squares, circles and 45° angles.",
            ToolKind::Bucket => "Fills the clicked area. Left: primary, right: secondary.",
            ToolKind::Text => "Click to place text, drag to move it. Enter or Apply to commit.",
            ToolKind::BoxSelect => "Drag a box to select it.",
            ToolKind::MagicWand => "Click a color region to select it.",
            ToolKind::OutlineSelect => {
                "Click inside a picture surrounded by a solid-color outline or background to select the inside."
            }
            ToolKind::Resize => {
                "Drag a corner to resize, drag inside to move, or type a size. Untick \"Move all corners\" to warp freely. Enter applies."
            }
            ToolKind::Rotate => "Drag on the canvas or set the angle; the anchor is the point it turns around. Enter applies.",
            ToolKind::Tilt => "Drag an edge handle to tilt in 3D around the opposite edge.",
            ToolKind::Shear => "Drag an edge handle along the edge to slant.",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Primary,
    Secondary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    Press,
    Drag,
    Release,
    Hover,
}

#[derive(Clone, Copy, Debug)]
pub struct PointerEvent {
    pub kind: PointerKind,
    /// Image coordinates, snapped if the tool uses snapping.
    pub pos: Vec2,
    /// Image coordinates before snapping.
    pub raw: Vec2,
    pub button: Button,
    pub shift: bool,
    // Not used by the current tools; available for future modifiers.
    #[allow(dead_code)]
    pub ctrl: bool,
    #[allow(dead_code)]
    pub alt: bool,
}

/// Everything a tool may touch.
pub struct ToolCtx<'a> {
    pub doc: &'a mut Document,
    pub colors: &'a mut ColorState,
    pub settings: &'a mut Settings,
    pub fonts: &'a mut Option<FontLibrary>,
    pub status: &'a mut Status,
    pub zoom: f32,
    pub actions: &'a mut Vec<Action>,
}

impl ToolCtx<'_> {
    /// Executes a command, reporting errors in the status bar.
    pub fn exec(&mut self, cmd: Command) -> Option<Outcome> {
        let mut ectx = ExecContext { fonts: self.fonts.as_mut(), grow_layers: self.settings.grow_layers };
        match cmd.execute(self.doc, &mut ectx) {
            Ok(o) => Some(o),
            Err(e) => {
                self.status.error(e);
                None
            }
        }
    }

    pub fn color_for(&self, b: Button) -> imged_core::Color {
        match b {
            Button::Primary => self.colors.primary,
            Button::Secondary => self.colors.secondary,
        }
    }

    /// The Transform section's "Apply to" choice.
    pub fn scope(&self) -> imged_core::ops::transform::Scope {
        self.settings.transform_scope
    }

    /// Called when a drawing tool starts drawing with button `b`: starts a
    /// new layer first when "New layer when the color/tool changes" asks for
    /// it (the very first drawing counts as a change).
    /// Whether drawing with `tool` and button `b` now would start a new
    /// layer (the "new layer when the color / tool changes" options).
    pub fn would_start_layer(&self, tool: ToolKind, b: Button) -> bool {
        let c = self.color_for(b);
        let col = [c.r.to_bits(), c.g.to_bits(), c.b.to_bits(), c.a.to_bits()];
        let s = &*self.settings;
        match s.last_draw {
            _ if !s.new_layer_on_color && !s.new_layer_on_tool => false,
            None => true,
            Some((t, last)) => (s.new_layer_on_tool && t != tool) || (s.new_layer_on_color && last != col),
        }
    }

    /// Starts a new layer if [`Self::would_start_layer`]; returns whether it did.
    pub fn begin_drawing(&mut self, tool: ToolKind, b: Button) -> bool {
        let new = self.would_start_layer(tool, b);
        if new {
            self.exec(Command::NewLayer);
        }
        let c = self.color_for(b);
        self.settings.last_draw = Some((tool, [c.r.to_bits(), c.g.to_bits(), c.b.to_bits(), c.a.to_bits()]));
        new
    }

    pub fn fonts(&mut self) -> &mut FontLibrary {
        self.fonts.get_or_insert_with(FontLibrary::new)
    }
}

pub trait Tool {
    #[allow(dead_code)]
    fn kind(&self) -> ToolKind;
    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx);
    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx);
    fn overlay(&self, _painter: &Painter, _view: &ViewXform, _doc: &Document) {}
    /// Abandon any in-progress interaction and remove previews.
    fn cancel(&mut self, _ctx: &mut ToolCtx) {}
    /// Commit pending work (Enter).
    fn commit(&mut self, _ctx: &mut ToolCtx) {}
    /// Called when switching to another tool. Defaults to cancelling.
    fn deactivate(&mut self, ctx: &mut ToolCtx) {
        self.cancel(ctx);
    }
    /// Whether pointer positions should snap to image edges when snapping is on.
    fn uses_snapping(&self) -> bool {
        false
    }
    /// Whether the tool options show the snapping checkbox.
    fn snap_option(&self) -> bool {
        self.uses_snapping()
    }
    fn cursor(&self) -> CursorIcon {
        CursorIcon::Crosshair
    }
    /// Color to show in the color panel while hovering (without picking).
    fn hover_color(&self, _doc: &Document, _pos: Vec2) -> Option<imged_core::Color> {
        None
    }
    /// Extra text for the status bar (e.g. the size of the shape being drawn).
    fn status(&self) -> Option<String> {
        None
    }

    /// Saves (`load` = false) or restores the tool's checkbox states.
    fn sync_prefs(&mut self, _p: &mut crate::persist::Prefs, _load: bool) {}
    /// `[` / `]` keys.
    fn adjust_size(&mut self, _bigger: bool) {}
    /// True while the tool has uncommitted state (e.g. text being edited);
    /// global shortcuts that would conflict are suppressed.
    fn captures_keys(&self) -> bool {
        false
    }
}

pub fn create_tools() -> Vec<Box<dyn Tool>> {
    ToolKind::ALL
        .iter()
        .map(|k| -> Box<dyn Tool> {
            match k {
                ToolKind::Move => Box::new(move_tool::MoveTool::default()),
                ToolKind::Brush => Box::new(brush::BrushTool::default()),
                ToolKind::Pen => Box::new(pen::PenTool::default()),
                ToolKind::Eyedropper => Box::new(eyedropper::EyedropperTool::default()),
                ToolKind::Shape => Box::new(shape::ShapeTool::default()),
                ToolKind::Bucket => Box::new(bucket::BucketTool::default()),
                ToolKind::Text => Box::new(text::TextTool::default()),
                ToolKind::BoxSelect => Box::new(select::BoxSelectTool::default()),
                ToolKind::MagicWand => Box::new(select::MagicWandTool::default()),
                ToolKind::OutlineSelect => Box::new(select::OutlineSelectTool::default()),
                ToolKind::Resize => Box::new(resize::ResizeTool::default()),
                ToolKind::Rotate => Box::new(rotate::RotateTool::default()),
                ToolKind::Tilt => Box::new(warp::WarpTool::new(warp::WarpMode::Tilt)),
                ToolKind::Shear => Box::new(warp::WarpTool::new(warp::WarpMode::Shear)),
            }
        })
        .collect()
}

/// How selection tools use their result (shared settings).
pub(crate) fn select_mode(ctx: &ToolCtx) -> imged_core::command::SelectMode {
    imged_core::command::SelectMode { to_layer: ctx.settings.select_to_layer, add: ctx.settings.select_add }
}

/// Shared "selection result" handling: report and optionally switch to Move.
pub(crate) fn after_selection(ctx: &mut ToolCtx, outcome: Option<Outcome>) {
    match outcome {
        Some(Outcome::NewLayer(_)) => {
            ctx.status.info(if ctx.settings.select_add {
                "Selection added to the layer"
            } else {
                "Selection moved to a new layer"
            });
            if ctx.settings.move_after_select {
                ctx.actions.push(Action::SelectTool(ToolKind::Move));
            }
        }
        Some(_) => {
            if let Some(b) = ctx.doc.selection.as_ref().and_then(|m| m.bounds()) {
                ctx.status.info(format!("Selected {} × {} px", b.w, b.h));
            }
        }
        None => {}
    }
}

/// The options shared by every selection tool.
pub(crate) fn selection_options(ui: &mut Ui, ctx: &mut ToolCtx) {
    ui.checkbox(&mut ctx.settings.select_to_layer, "Move to new layer")
        .on_hover_text("On: the selected pixels move to a new layer.\nOff: the area stays selected (marching ants) for filters and transforms.");
    let add_tip = if ctx.settings.select_to_layer {
        "Each new selection joins the layer made by the previous one"
    } else {
        "Each new selection is added to the current one"
    };
    ui.checkbox(&mut ctx.settings.select_add, "Add to previous selection").on_hover_text(add_tip);
    if ctx.settings.select_to_layer {
        ui.checkbox(&mut ctx.settings.move_after_select, "Switch to Move after selecting");
    }
    if ctx.doc.selection.is_some() {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for (label, tip, cmd) in [
                ("Deselect", "Ctrl+Shift+A", Command::SelectNone),
                ("Invert", "Select everything else (Ctrl+Shift+I)", Command::InvertSelection),
                ("Cut to layer", "Move the selected pixels to a new layer", Command::SelectionToLayer { copy: false }),
                ("Copy to layer", "Copy the selected pixels to a new layer", Command::SelectionToLayer { copy: true }),
                ("Crop", "Crop the canvas to the selection", Command::CropToSelection),
            ] {
                if ui.small_button(label).on_hover_text(tip).clicked() {
                    ctx.actions.push(Action::Cmd(cmd));
                }
            }
        });
    }
}

/// Percent slider for 0..1 tolerances.
pub(crate) fn tolerance_slider(ui: &mut Ui, value: &mut f32, label: &str) {
    let mut pct = *value * 100.0;
    if ui
        .add(egui::Slider::new(&mut pct, 0.0..=100.0).update_while_editing(false).suffix("%").text(label))
        .changed()
    {
        *value = pct / 100.0;
    }
}

/// Dashed rectangle outline in screen space (marching-ants style).
pub(crate) fn dashed_rect(painter: &Painter, r: egui::Rect) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    painter.add(egui::Shape::line(pts.to_vec(), egui::Stroke::new(1.0, egui::Color32::BLACK)));
    painter.extend(egui::Shape::dashed_line(&pts, egui::Stroke::new(1.0, egui::Color32::WHITE), 4.0, 4.0));
}
