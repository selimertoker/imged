//! Layers and their private undo histories.
//!
//! Pixel edits are recorded on the layer they touch, as region patches
//! (before/after crops of the changed rectangle) or full replacements when
//! the layer's size or position changes. Every entry carries a global
//! sequence number from the owning document so that a single "undo" can pick
//! the most recent change across the document and the active layers.

use std::sync::Arc;

use crate::geom::Rect;
use crate::ops::filters::LayerFilter;
use crate::pixel::Depth;
use crate::pixels::Pixels;

pub type LayerId = u64;
pub type GroupId = u64;

/// Maximum number of undo entries kept per layer.
pub const LAYER_HISTORY_LIMIT: usize = 64;

#[derive(Clone, Debug)]
pub enum PatchKind {
    /// A rectangle (layer-local coords) changed; size and offset did not.
    Region { x: i32, y: i32, before: Pixels, after: Pixels },
    /// The whole layer was replaced (size, depth or offset may differ).
    Full {
        before: (Arc<Pixels>, (i32, i32)),
        after: (Arc<Pixels>, (i32, i32)),
    },
}

#[derive(Clone, Debug)]
pub struct Patch {
    pub seq: u64,
    pub label: String,
    pub kind: PatchKind,
}

#[derive(Clone, Debug, Default)]
pub struct LayerHistory {
    pub undo: Vec<Arc<Patch>>,
    pub redo: Vec<Arc<Patch>>,
}

impl LayerHistory {
    pub fn top_undo_seq(&self) -> Option<u64> {
        self.undo.last().map(|p| p.seq)
    }
    pub fn top_redo_seq(&self) -> Option<u64> {
        self.redo.last().map(|p| p.seq)
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    /// Shared so that document snapshots and previews are cheap to clone;
    /// mutate through [`Layer::pixels_mut`] (copy-on-write).
    pub pixels: Arc<Pixels>,
    /// Position of the layer's top-left corner on the canvas.
    pub offset: (i32, i32),
    pub visible: bool,
    pub opacity: f32,
    /// Layers sharing a group id are "grouped": they act as one layer
    /// but keep their own pixels and can be separated again.
    pub group: Option<GroupId>,
    /// A filter layer: its pixels are not drawn but mask where `filter`
    /// changes everything below it (an empty layer = everywhere).
    pub filter_on: bool,
    /// The filter layer's settings (kept while `filter_on` is off).
    pub filter: LayerFilter,
    pub history: LayerHistory,
}

impl Layer {
    pub fn new(id: LayerId, name: impl Into<String>, pixels: Pixels, offset: (i32, i32)) -> Self {
        Layer {
            id,
            name: name.into(),
            pixels: Arc::new(pixels),
            offset,
            visible: true,
            opacity: 1.0,
            group: None,
            filter_on: false,
            filter: LayerFilter::default(),
            history: LayerHistory::default(),
        }
    }

    pub fn is_filter(&self) -> bool {
        self.filter_on
    }

    pub fn depth(&self) -> Depth {
        self.pixels.depth()
    }

    pub fn width(&self) -> u32 {
        self.pixels.width()
    }

    pub fn height(&self) -> u32 {
        self.pixels.height()
    }

    /// The layer's area in canvas coordinates.
    pub fn bounds(&self) -> Rect {
        Rect::new(self.offset.0, self.offset.1, self.width(), self.height())
    }

    pub fn pixels_mut(&mut self) -> &mut Pixels {
        Arc::make_mut(&mut self.pixels)
    }

    fn push(&mut self, patch: Patch) {
        self.history.undo.push(Arc::new(patch));
        self.history.redo.clear();
        if self.history.undo.len() > LAYER_HISTORY_LIMIT {
            let excess = self.history.undo.len() - LAYER_HISTORY_LIMIT;
            self.history.undo.drain(..excess);
        }
    }

    /// Runs `edit` on the pixels and records the change to `rect`
    /// (layer-local; clipped to the layer). `edit` must not touch pixels
    /// outside `rect` and must not resize the buffer.
    pub fn edit_region(&mut self, seq: u64, label: &str, rect: Rect, edit: impl FnOnce(&mut Pixels)) {
        let rect = rect.intersect(&self.pixels.rect());
        if rect.is_empty() {
            return;
        }
        let before = self.pixels.crop(rect);
        edit(self.pixels_mut());
        self.record_region(seq, label, rect, before);
    }

    /// Records an edit that has already been applied. `before` must be the
    /// crop of `rect` taken before the edit.
    pub fn record_region(&mut self, seq: u64, label: &str, rect: Rect, before: Pixels) {
        let after = self.pixels.crop(rect);
        if after == before {
            return;
        }
        self.push(Patch {
            seq,
            label: label.to_string(),
            kind: PatchKind::Region { x: rect.x, y: rect.y, before, after },
        });
    }

    /// Replaces the whole layer (pixels and offset) and records it.
    pub fn replace(&mut self, seq: u64, label: &str, pixels: Pixels, offset: (i32, i32)) {
        let before = (self.pixels.clone(), self.offset);
        self.pixels = Arc::new(pixels);
        self.offset = offset;
        let after = (self.pixels.clone(), self.offset);
        self.push(Patch {
            seq,
            label: label.to_string(),
            kind: PatchKind::Full { before, after },
        });
    }

    /// Moves the layer and records it (cheap: pixels are shared).
    pub fn move_to(&mut self, seq: u64, offset: (i32, i32)) {
        if offset == self.offset {
            return;
        }
        let before = (self.pixels.clone(), self.offset);
        self.offset = offset;
        let after = (self.pixels.clone(), self.offset);
        self.push(Patch {
            seq,
            label: "Move".into(),
            kind: PatchKind::Full { before, after },
        });
    }

    /// Moves the layer and folds the move into the newest history entry when
    /// that is a move labelled `label` with sequence number `seq` (so a run
    /// of arrow-key nudges is one undo step). Returns false, changing
    /// nothing, when there is no such entry.
    pub fn extend_move(&mut self, seq: u64, label: &str, offset: (i32, i32)) -> bool {
        let Some(top) = self.history.undo.last() else { return false };
        if top.seq != seq || top.label != label {
            return false;
        }
        let PatchKind::Full { before, after } = &top.kind else { return false };
        if !Arc::ptr_eq(&after.0, &self.pixels) || after.1 != self.offset {
            return false;
        }
        let patch = Patch {
            seq,
            label: label.to_string(),
            kind: PatchKind::Full { before: before.clone(), after: (self.pixels.clone(), offset) },
        };
        *self.history.undo.last_mut().unwrap() = Arc::new(patch);
        self.history.redo.clear();
        self.offset = offset;
        true
    }

    /// Moves the layer and records it with a custom label.
    pub fn move_labelled(&mut self, seq: u64, label: &str, offset: (i32, i32)) {
        if offset == self.offset {
            return;
        }
        let before = (self.pixels.clone(), self.offset);
        self.offset = offset;
        let after = (self.pixels.clone(), self.offset);
        self.push(Patch { seq, label: label.to_string(), kind: PatchKind::Full { before, after } });
    }

    /// Undoes the newest entry. Returns the canvas rect that changed.
    pub fn undo(&mut self) -> Option<Rect> {
        let patch = self.history.undo.pop()?;
        let dirty = self.apply(&patch, true);
        self.history.redo.push(patch);
        Some(dirty)
    }

    pub fn redo(&mut self) -> Option<Rect> {
        let patch = self.history.redo.pop()?;
        let dirty = self.apply(&patch, false);
        self.history.undo.push(patch);
        Some(dirty)
    }

    fn apply(&mut self, patch: &Patch, backwards: bool) -> Rect {
        match &patch.kind {
            PatchKind::Region { x, y, before, after } => {
                let src = if backwards { before } else { after };
                self.pixels_mut().paste(src, *x, *y);
                Rect::new(*x, *y, src.width(), src.height()).translate(self.offset.0, self.offset.1)
            }
            PatchKind::Full { before, after } => {
                let old = self.bounds();
                let (px, off) = if backwards { before } else { after };
                self.pixels = px.clone();
                self.offset = *off;
                old.union(&self.bounds())
            }
        }
    }
}
