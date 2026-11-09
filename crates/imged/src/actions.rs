//! Actions and key bindings.
//!
//! Everything the user can trigger (menu items, buttons, shortcuts) is an
//! [`Action`]. Document changes are wrapped [`Command`]s from the core, so
//! the same code path serves the GUI now and a CLI later.
//!
//! Keys are resolved by a [`KeyResolver`]. The default [`Keymap`] maps
//! single chords to actions. A vim-style modal resolver can implement the
//! same trait later: it receives each key event, keeps its own pending state
//! (counts, operators, modes) and emits zero or more actions.

use std::collections::HashMap;

use eframe::egui::{Key, Modifiers};
use imged_core::ops::filters::Filter;
use imged_core::Command;

use crate::tools::ToolKind;

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// A document command (undoable through the core history).
    Cmd(Command),
    /// A color filter, applied to the layer or the whole image depending on
    /// the "Apply to" choice at the time it runs.
    Filter(Filter),
    SelectTool(ToolKind),
    SwapColors,
    ResetColors,
    ZoomIn,
    ZoomOut,
    ZoomFit,
    ZoomActual,
    NewImage,
    Open,
    OpenAsLayer,
    Export,
    /// Copy the flattened canvas to the clipboard as a PNG image.
    CopyImage,
    /// Edit with ImageMagick: the active layer (false) or the image (true).
    Magick(bool),
    /// Turn the active layer into a filter layer or back.
    ToggleFilterLayer,
    /// Paste an image (any format) from the clipboard as a new layer.
    PasteImage,
    /// Save the layered project (PSD); asks for a file the first time.
    SaveProject,
    SaveProjectAs,
    /// File → Import raw…
    ImportRaw,
    ShowDialog(Dialog),
    ToggleSnapping,
    BrushBigger,
    BrushSmaller,
    /// Escape: cancel the current tool interaction.
    Cancel,
    /// Enter: commit the current tool interaction (text, tilt, …).
    Commit,
    /// Nudge the active layer with the keyboard.
    Nudge(i32, i32),
    Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialog {
    CanvasSize,
    Convolution,
    Keys,
}

/// A key plus modifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chord {
    pub key: Key,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Chord {
    pub const fn new(key: Key) -> Self {
        Chord { key, ctrl: false, shift: false, alt: false }
    }
    pub const fn ctrl(key: Key) -> Self {
        Chord { key, ctrl: true, shift: false, alt: false }
    }
    pub const fn ctrl_shift(key: Key) -> Self {
        Chord { key, ctrl: true, shift: true, alt: false }
    }
    pub const fn shift(key: Key) -> Self {
        Chord { key, ctrl: false, shift: true, alt: false }
    }

    pub fn from_event(key: Key, m: Modifiers) -> Self {
        Chord { key, ctrl: m.command || m.ctrl, shift: m.shift, alt: m.alt }
    }

    pub fn label(&self) -> String {
        let mut s = String::new();
        if self.ctrl {
            s.push_str("Ctrl+");
        }
        if self.alt {
            s.push_str("Alt+");
        }
        if self.shift {
            s.push_str("Shift+");
        }
        s.push_str(self.key.name());
        s
    }
}

/// Turns key presses into actions.
pub trait KeyResolver {
    fn resolve(&mut self, chord: Chord) -> Vec<Action>;
    /// Shortcut text for menus/tooltips, if the action has one.
    fn shortcut_for(&self, action: &Action) -> Option<String>;
    /// Optional mode indicator for the status bar (e.g. "-- NORMAL --").
    fn mode_label(&self) -> Option<String> {
        None
    }
}

/// The default one-chord-per-action key map.
pub struct Keymap {
    pub bindings: HashMap<Chord, Action>,
}

impl Default for Keymap {
    fn default() -> Self {
        use Key::*;
        let mut b = HashMap::new();
        let mut bind = |c: Chord, a: Action| {
            b.insert(c, a);
        };
        for t in ToolKind::ALL {
            bind(Chord::new(t.key()), Action::SelectTool(t));
        }
        bind(Chord::new(X), Action::SwapColors);
        bind(Chord::new(D), Action::ResetColors);
        bind(Chord::new(S), Action::ToggleSnapping);
        bind(Chord::new(Plus), Action::ZoomIn);
        bind(Chord::new(Equals), Action::ZoomIn);
        bind(Chord::new(Minus), Action::ZoomOut);
        bind(Chord::new(Num0), Action::ZoomFit);
        bind(Chord::new(Num1), Action::ZoomActual);
        bind(Chord::new(CloseBracket), Action::BrushBigger);
        bind(Chord::new(OpenBracket), Action::BrushSmaller);
        bind(Chord::new(Escape), Action::Cancel);
        bind(Chord::new(Enter), Action::Commit);
        bind(Chord::new(ArrowLeft), Action::Nudge(-1, 0));
        bind(Chord::new(ArrowRight), Action::Nudge(1, 0));
        bind(Chord::new(ArrowUp), Action::Nudge(0, -1));
        bind(Chord::new(ArrowDown), Action::Nudge(0, 1));
        bind(Chord::shift(ArrowLeft), Action::Nudge(-10, 0));
        bind(Chord::shift(ArrowRight), Action::Nudge(10, 0));
        bind(Chord::shift(ArrowUp), Action::Nudge(0, -10));
        bind(Chord::shift(ArrowDown), Action::Nudge(0, 10));
        bind(Chord::ctrl(Z), Action::Cmd(Command::Undo));
        bind(Chord::ctrl_shift(Z), Action::Cmd(Command::Redo));
        bind(Chord::ctrl(Y), Action::Cmd(Command::Redo));
        bind(Chord::ctrl(N), Action::NewImage);
        bind(Chord::ctrl(O), Action::Open);
        bind(Chord::ctrl_shift(O), Action::OpenAsLayer);
        bind(Chord::ctrl(E), Action::Export);
        bind(Chord::ctrl(S), Action::SaveProject);
        bind(Chord::ctrl_shift(S), Action::SaveProjectAs);
        bind(Chord::ctrl(V), Action::PasteImage);
        bind(Chord::ctrl_shift(C), Action::CopyImage);
        bind(Chord::ctrl(Q), Action::Quit);
        bind(Chord::ctrl(I), Action::Filter(Filter::Invert));
        bind(Chord::ctrl_shift(L), Action::Filter(Filter::AutoLevels));
        bind(Chord::ctrl_shift(N), Action::Cmd(Command::NewLayer));
        bind(Chord::ctrl(J), Action::Cmd(Command::DuplicateLayer));
        bind(Chord::new(Delete), Action::Cmd(Command::DeleteLayer));
        bind(Chord::ctrl(CloseBracket), Action::Cmd(Command::MoveLayer { up: true }));
        bind(Chord::ctrl(OpenBracket), Action::Cmd(Command::MoveLayer { up: false }));
        bind(Chord::ctrl(M), Action::Cmd(Command::MergeDown));
        bind(Chord::ctrl(G), Action::Cmd(Command::Group));
        bind(Chord::ctrl_shift(G), Action::Cmd(Command::Ungroup));
        bind(Chord::ctrl(A), Action::Cmd(Command::SelectAll));
        bind(Chord::ctrl_shift(A), Action::Cmd(Command::SelectNone));
        bind(Chord::ctrl_shift(I), Action::Cmd(Command::InvertSelection));
        bind(Chord::ctrl(K), Action::ShowDialog(Dialog::Convolution));
        bind(Chord::new(F1), Action::ShowDialog(Dialog::Keys));
        Keymap { bindings: b }
    }
}

impl KeyResolver for Keymap {
    fn resolve(&mut self, chord: Chord) -> Vec<Action> {
        self.bindings.get(&chord).cloned().into_iter().collect()
    }

    fn shortcut_for(&self, action: &Action) -> Option<String> {
        let mut found: Vec<String> =
            self.bindings.iter().filter(|(_, a)| *a == action).map(|(c, _)| c.label()).collect();
        found.sort_by_key(|s| s.len());
        found.into_iter().next()
    }
}
