//! Editor state shared between panels and tools (not part of the document).

use std::collections::VecDeque;
use std::time::Instant;

use imged_core::ops::edges::SnapParams;
use imged_core::ops::paint::PaintMode;
use imged_core::ops::transform::Scope;
use imged_core::Color;

/// How many recently used colors are remembered.
pub const RECENT_COLORS: usize = 33;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSlot {
    Primary,
    Secondary,
}

pub struct ColorState {
    pub primary: Color,
    pub secondary: Color,
    /// Most recent first.
    pub recent: VecDeque<Color>,
    /// The user's palette (any length).
    pub palette: Vec<Color>,
    /// Which slot the color wheel / hex field edits.
    pub editing: ColorSlot,
}

impl Default for ColorState {
    fn default() -> Self {
        ColorState {
            primary: Color::BLACK,
            secondary: Color::WHITE,
            recent: VecDeque::new(),
            palette: default_palette(),
            editing: ColorSlot::Primary,
        }
    }
}

pub fn default_palette() -> Vec<Color> {
    vec![
        Color::from_rgba8(0xE5, 0x39, 0x35, 255),
        Color::from_rgba8(0xFB, 0x8C, 0x00, 255),
        Color::from_rgba8(0xFD, 0xD8, 0x35, 255),
        Color::from_rgba8(0x43, 0xA0, 0x47, 255),
        Color::from_rgba8(0x1E, 0x88, 0xE5, 255),
        Color::from_rgba8(0x8E, 0x24, 0xAA, 255),
        Color::BLACK,
        Color::from_rgba8(0x80, 0x80, 0x80, 255),
        Color::WHITE,
        Color::TRANSPARENT,
    ]
}

impl ColorState {
    pub fn get(&self, slot: ColorSlot) -> Color {
        match slot {
            ColorSlot::Primary => self.primary,
            ColorSlot::Secondary => self.secondary,
        }
    }

    pub fn set(&mut self, slot: ColorSlot, c: Color) {
        match slot {
            ColorSlot::Primary => self.primary = c,
            ColorSlot::Secondary => self.secondary = c,
        }
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.primary, &mut self.secondary);
    }

    /// Records that `c` was used to paint (moves it to the front of the recents).
    pub fn mark_used(&mut self, c: Color) {
        self.recent.retain(|r| r.to_rgba8() != c.to_rgba8() || r.is_hdr() != c.is_hdr());
        self.recent.push_front(c);
        self.recent.truncate(RECENT_COLORS);
    }
}

pub struct Settings {
    pub paint_mode: PaintMode,
    pub antialias: bool,
    pub snapping: bool,
    /// Snap radius in *screen* pixels (converted using the zoom).
    pub snap_radius_screen: f32,
    pub snap_threshold: f32,
    /// Scope of flip / rotate / resize buttons.
    pub transform_scope: Scope,
    /// Switch to the Move tool after a selection creates a layer.
    pub move_after_select: bool,
    /// Selection tools move the selected pixels to a new layer (else they
    /// keep a selection).
    pub select_to_layer: bool,
    /// Selection tools add to the previous selection.
    pub select_add: bool,
    /// Drawing tools start a new layer when the color / the tool differs
    /// from the last drawing.
    pub new_layer_on_color: bool,
    pub new_layer_on_tool: bool,
    /// The ImageMagick (or GraphicsMagick) command, e.g. "magick" or "gm convert".
    pub magick_command: String,
    /// The tool and color of the last drawing (for the two options above).
    pub last_draw: Option<(crate::tools::ToolKind, [u32; 4])>,
    /// Painting outside the active layer grows the layer (brush, pen, text,
    /// shapes, bucket). Off: paint is clipped to the layer.
    pub grow_layers: bool,
    /// Keep the width/height ratio when resizing (Resize tool).
    pub keep_ratio: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            paint_mode: PaintMode::Replace,
            antialias: true,
            snapping: false,
            snap_radius_screen: 10.0,
            snap_threshold: 0.05,
            transform_scope: Scope::Layer,
            move_after_select: true,
            select_to_layer: true,
            select_add: false,
            new_layer_on_color: false,
            new_layer_on_tool: false,
            last_draw: None,
            magick_command: crate::magick::default_command(),
            grow_layers: true,
            keep_ratio: true,
        }
    }
}

impl Settings {
    pub fn snap_params(&self, zoom: f32) -> SnapParams {
        SnapParams { radius: (self.snap_radius_screen / zoom.max(1e-3)).max(1.0), threshold: self.snap_threshold }
    }
}

/// How pixel values are turned into screen colors (view only, never saved).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplaySettings {
    /// Exposure in stops (multiplies values by 2^exposure).
    pub exposure: f32,
    /// Encode linear values to sRGB for display (for EXR / PFM / HDR data).
    pub srgb_encode: bool,
    pub checkerboard: bool,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        DisplaySettings { exposure: 0.0, srgb_encode: false, checkerboard: true }
    }
}

/// A short message in the status bar.
#[derive(Default)]
pub struct Status {
    pub message: Option<(String, bool, Instant)>,
}

impl Status {
    pub fn info(&mut self, msg: impl Into<String>) {
        self.message = Some((msg.into(), false, Instant::now()));
    }

    pub fn error(&mut self, msg: impl Into<String>) {
        self.message = Some((msg.into(), true, Instant::now()));
    }
}
