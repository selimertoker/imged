//! The document: canvas size, layer stack, groups and history.
//!
//! History is split in two:
//! * each layer records its own pixel edits ([`crate::layer::LayerHistory`]);
//! * the document records structural changes (adding, deleting, reordering,
//!   merging layers, canvas-wide transforms) as swappable snapshots of the
//!   layer list. Snapshots are cheap because pixels are shared via `Arc`.
//!
//! Every entry gets a sequence number from one counter. [`Document::undo`]
//! undoes whichever is newest: the last structural change or the last edit
//! of the active layer (or every member of its group).

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use crate::buffer::Buffer;
use crate::color::Color;
use crate::composite::{composite_views, LayerView};
use crate::geom::Rect;
use crate::layer::{GroupId, Layer, LayerId};
use crate::mask::Mask;
use crate::ops::filters::LayerFilter;
use crate::pixel::{Depth, Px};
use crate::pixels::Pixels;

/// Maximum number of structural undo entries.
pub const DOC_HISTORY_LIMIT: usize = 64;

/// A temporary stand-in for a layer's pixels, shown while a tool or filter
/// is being adjusted. Never part of history.
#[derive(Clone, Debug)]
pub struct Preview {
    pub pixels: Arc<Pixels>,
    pub offset: (i32, i32),
}

#[derive(Clone, Debug)]
struct DocState {
    width: u32,
    height: u32,
    layers: Vec<Layer>,
    active: LayerId,
    selected: BTreeSet<LayerId>,
    selection: Option<Arc<Mask>>,
}

#[derive(Clone, Debug)]
struct DocEntry {
    seq: u64,
    label: String,
    state: DocState,
}

#[derive(Debug)]
pub struct Document {
    pub width: u32,
    pub height: u32,
    /// Bottom to top.
    pub layers: Vec<Layer>,
    pub active: LayerId,
    /// Extra layers ticked in the layer panel for multi-layer operations.
    pub selected: BTreeSet<LayerId>,
    /// The selected area (canvas coordinates, coverage 0..1), if any.
    pub selection: Option<Arc<Mask>>,
    /// The layer the last "selection to layer" created, so further selections
    /// in "add" mode can join it.
    pub extract_layer: Option<LayerId>,
    /// Depth for newly created layers.
    pub default_depth: Depth,
    /// Hint that pixel values are scene-linear (e.g. loaded from EXR/PFM),
    /// so viewers should encode them to sRGB for display.
    pub linear_hint: bool,
    pub path: Option<PathBuf>,
    pub preview: HashMap<LayerId, Preview>,
    seq: u64,
    next_id: u64,
    undo: Vec<DocEntry>,
    redo: Vec<DocEntry>,
    revision: u64,
    dirty: Option<Rect>,
    /// Like `dirty`, but not clipped to the canvas (for renderers that keep
    /// whole layers, e.g. as GPU textures).
    dirty_unclipped: Option<Rect>,
    layer_counter: u32,
}

impl Document {
    /// New document with a single background layer (`fill: None` = transparent).
    pub fn new(width: u32, height: u32, depth: Depth, fill: Option<Color>) -> Self {
        let pixels = match fill {
            Some(c) => Pixels::filled(depth, width, height, c.px()),
            None => Pixels::new(depth, width, height),
        };
        Document::from_pixels(pixels, "Background")
    }

    /// Document sized to `pixels`, which becomes its only layer.
    pub fn from_pixels(pixels: Pixels, name: &str) -> Self {
        let (w, h, depth) = (pixels.width(), pixels.height(), pixels.depth());
        let mut doc = Document {
            width: w,
            height: h,
            layers: Vec::new(),
            active: 0,
            selected: BTreeSet::new(),
            selection: None,
            extract_layer: None,
            default_depth: depth,
            linear_hint: false,
            path: None,
            preview: HashMap::new(),
            seq: 0,
            next_id: 1,
            undo: Vec::new(),
            redo: Vec::new(),
            revision: 0,
            dirty: Some(Rect::from_size(w, h)),
            dirty_unclipped: Some(Rect::from_size(w, h)),
            layer_counter: 0,
        };
        let id = doc.alloc_id();
        doc.layers.push(Layer::new(id, name, pixels, (0, 0)));
        doc.active = id;
        doc
    }

    // ----- bookkeeping ---------------------------------------------------

    pub fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    pub fn alloc_id(&mut self) -> LayerId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn next_layer_name(&mut self) -> String {
        self.layer_counter += 1;
        format!("Layer {}", self.layer_counter)
    }

    /// Increases on every change; use it to invalidate caches.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn canvas_rect(&self) -> Rect {
        Rect::from_size(self.width, self.height)
    }

    pub fn mark_dirty(&mut self, rect: Rect) {
        self.revision += 1;
        if !rect.is_empty() {
            self.dirty_unclipped = Some(self.dirty_unclipped.map_or(rect, |d| d.union(&rect)));
        }
        let r = rect.intersect(&self.canvas_rect());
        if r.is_empty() {
            return;
        }
        self.dirty = Some(match self.dirty {
            Some(d) => d.union(&r),
            None => r,
        });
    }

    pub fn mark_all_dirty(&mut self) {
        self.mark_dirty(self.canvas_rect());
    }

    /// Returns and clears the region that needs re-rendering.
    pub fn take_dirty(&mut self) -> Option<Rect> {
        self.dirty.take()
    }

    /// Returns and clears the changed region, not clipped to the canvas
    /// (layer pixels outside the canvas count too).
    pub fn take_dirty_unclipped(&mut self) -> Option<Rect> {
        self.dirty_unclipped.take()
    }

    // ----- layer lookup --------------------------------------------------

    pub fn index_of(&self, id: LayerId) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub fn active_index(&self) -> usize {
        self.index_of(self.active).unwrap_or(self.layers.len().saturating_sub(1))
    }

    pub fn active_layer(&self) -> &Layer {
        &self.layers[self.active_index()]
    }

    pub fn active_layer_mut(&mut self) -> &mut Layer {
        let i = self.active_index();
        &mut self.layers[i]
    }

    pub fn set_active(&mut self, id: LayerId) {
        if self.index_of(id).is_some() {
            self.active = id;
        }
    }

    /// Index range of the block (single layer or group) containing `index`.
    pub fn block_at(&self, index: usize) -> std::ops::Range<usize> {
        let Some(g) = self.layers.get(index).and_then(|l| l.group) else {
            return index..index + 1;
        };
        let mut a = index;
        while a > 0 && self.layers[a - 1].group == Some(g) {
            a -= 1;
        }
        let mut b = index + 1;
        while b < self.layers.len() && self.layers[b].group == Some(g) {
            b += 1;
        }
        a..b
    }

    pub fn active_block(&self) -> std::ops::Range<usize> {
        self.block_at(self.active_index())
    }

    /// Layers affected by operations on the active layer: the active layer,
    /// or all members of its group.
    pub fn targets(&self) -> Vec<LayerId> {
        self.layers[self.active_block()].iter().map(|l| l.id).collect()
    }

    /// Moves the active layer/group by (dx, dy) for an arrow key. A run of
    /// nudges of the same layers with nothing else in between is one undo
    /// step.
    pub fn nudge_targets(&mut self, dx: i32, dy: i32) -> bool {
        const LABEL: &str = "Move with keys";
        if dx == 0 && dy == 0 {
            return false;
        }
        let ids = self.targets();
        let mut dirty = Rect::default();
        // Merge when the newest step anywhere is a nudge of exactly these layers.
        let merge = match self.next_undo() {
            Some((seq, false, label)) if label == LABEL => {
                let touched: Vec<LayerId> =
                    self.layers.iter().filter(|l| l.history.top_undo_seq() == Some(seq)).map(|l| l.id).collect();
                let mut a = touched.clone();
                let mut b = ids.clone();
                a.sort_unstable();
                b.sort_unstable();
                (a == b).then_some(seq)
            }
            _ => None,
        };
        if let Some(seq) = merge {
            let ok = ids.iter().all(|id| {
                let l = self.layer(*id).unwrap();
                l.history.undo.last().is_some_and(|p| p.seq == seq)
            });
            if ok {
                let mut all = true;
                for id in &ids {
                    let l = self.layer_mut(*id).unwrap();
                    dirty = dirty.union(&l.bounds());
                    let to = (l.offset.0 + dx, l.offset.1 + dy);
                    all &= l.extend_move(seq, LABEL, to);
                    dirty = dirty.union(&l.bounds());
                }
                if all {
                    self.redo.clear();
                    self.mark_dirty(dirty);
                    return true;
                }
            }
        }
        let seq = self.next_seq();
        for id in ids {
            let l = self.layer_mut(id).unwrap();
            dirty = dirty.union(&l.bounds());
            l.move_labelled(seq, LABEL, (l.offset.0 + dx, l.offset.1 + dy));
            dirty = dirty.union(&l.bounds());
        }
        self.mark_dirty(dirty);
        true
    }

    pub fn group_members(&self, g: GroupId) -> Vec<LayerId> {
        self.layers.iter().filter(|l| l.group == Some(g)).map(|l| l.id).collect()
    }

    /// Topmost visible layer with a non-transparent pixel at canvas (x, y).
    pub fn layer_at(&self, x: i32, y: i32) -> Option<LayerId> {
        self.layers
            .iter()
            .rev()
            .filter(|l| l.visible)
            .find(|l| l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1)[3] > 0.0)
            .map(|l| l.id)
    }

    /// Topmost visible layer with non-transparent pixels under `mask` (canvas coords).
    pub fn layer_under_mask(&self, mask: &Mask) -> Option<LayerId> {
        let b = mask.bounds()?;
        self.layers.iter().rev().filter(|l| l.visible).find_map(|l| {
            let r = b.intersect(&l.bounds());
            for y in r.y..r.bottom() {
                for x in r.x..r.right() {
                    if mask.get(x, y) > 0.0 && l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1)[3] > 0.0 {
                        return Some(l.id);
                    }
                }
            }
            None
        })
    }

    /// Bounds (canvas coords) of the targets.
    pub fn targets_bounds(&self) -> Rect {
        self.targets()
            .iter()
            .filter_map(|id| self.layer(*id))
            .fold(Rect::default(), |acc, l| acc.union(&l.bounds()))
    }

    /// Composite of the active block only (as if merged), over `rect`.
    pub fn composite_targets(&self, rect: Rect) -> Buffer<f32> {
        let ids = self.targets();
        let views: Vec<LayerView> = self
            .layers
            .iter()
            .filter(|l| ids.contains(&l.id))
            .map(|l| LayerView { opacity: 1.0, ..LayerView::of(l) })
            .collect();
        composite_views(&views, rect, None)
    }

    // ----- previews ------------------------------------------------------

    pub fn set_preview(&mut self, id: LayerId, pixels: Arc<Pixels>, offset: (i32, i32)) {
        let mut dirty = Rect::new(offset.0, offset.1, pixels.width(), pixels.height());
        if let Some(old) = self.preview.get(&id) {
            dirty = dirty.union(&Rect::new(old.offset.0, old.offset.1, old.pixels.width(), old.pixels.height()));
        }
        if let Some(l) = self.layer(id) {
            dirty = dirty.union(&l.bounds());
        }
        self.preview.insert(id, Preview { pixels, offset });
        self.mark_dirty(dirty);
    }

    /// Like [`Document::set_preview`] but only marks `dirty` (canvas coords)
    /// for re-rendering. Use when the preview differs from the previous one
    /// only inside `dirty` (live brush strokes).
    pub fn set_preview_partial(&mut self, id: LayerId, pixels: Arc<Pixels>, offset: (i32, i32), dirty: Rect) {
        self.preview.insert(id, Preview { pixels, offset });
        self.mark_dirty(dirty);
    }

    /// Takes a preview out for in-place mutation (put it back with `set_preview`).
    pub fn take_preview(&mut self, id: LayerId) -> Option<Preview> {
        self.preview.remove(&id)
    }

    pub fn clear_preview(&mut self) {
        if self.preview.is_empty() {
            return;
        }
        let mut dirty = Rect::default();
        for (id, p) in self.preview.drain() {
            dirty = dirty.union(&Rect::new(p.offset.0, p.offset.1, p.pixels.width(), p.pixels.height()));
            if let Some(l) = self.layers.iter().find(|l| l.id == id) {
                dirty = dirty.union(&l.bounds());
            }
        }
        self.mark_dirty(dirty);
    }

    // ----- history -------------------------------------------------------

    fn snapshot(&self) -> DocState {
        DocState {
            width: self.width,
            height: self.height,
            layers: self.layers.clone(),
            active: self.active,
            selected: self.selected.clone(),
            selection: self.selection.clone(),
        }
    }

    fn restore(&mut self, s: DocState) -> DocState {
        let current = self.snapshot();
        self.width = s.width;
        self.height = s.height;
        self.layers = s.layers;
        self.active = s.active;
        self.selected = s.selected;
        self.selection = s.selection;
        self.preview.clear();
        self.mark_all_dirty();
        current
    }

    /// Runs a structural change and records it as one undoable step.
    /// `f` returns `false` to signal that nothing changed (no entry is kept).
    pub fn structural(&mut self, label: &str, f: impl FnOnce(&mut Document) -> bool) -> bool {
        let before = self.snapshot();
        let seq = self.next_seq();
        if !f(self) {
            return false;
        }
        self.undo.push(DocEntry { seq, label: label.into(), state: before });
        self.redo.clear();
        if self.undo.len() > DOC_HISTORY_LIMIT {
            self.undo.remove(0);
        }
        self.preview.clear();
        self.mark_all_dirty();
        true
    }

    /// Records the same per-layer edit on every target with one sequence
    /// number, so that undo treats it as one step.
    /// `f` receives each layer and returns the new (pixels, offset), or
    /// `None` to leave the layer untouched.
    pub fn edit_targets(
        &mut self,
        label: &str,
        f: impl Fn(&Layer) -> Option<(Pixels, (i32, i32))>,
    ) -> bool {
        let ids = self.targets();
        self.edit_layers(&ids, label, f)
    }

    /// Like [`Document::edit_targets`] for an explicit set of layers.
    pub fn edit_layers(
        &mut self,
        ids: &[LayerId],
        label: &str,
        f: impl Fn(&Layer) -> Option<(Pixels, (i32, i32))>,
    ) -> bool {
        let seq = self.next_seq();
        let mut dirty = Rect::default();
        let mut changed = false;
        for &id in ids {
            let Some(layer) = self.layer(id) else { continue };
            if let Some((px, off)) = f(layer) {
                let layer = self.layer_mut(id).unwrap();
                dirty = dirty.union(&layer.bounds());
                layer.replace(seq, label, px, off);
                dirty = dirty.union(&layer.bounds());
                changed = true;
            }
        }
        self.preview.clear();
        self.mark_dirty(dirty);
        changed
    }

    /// Label of the step [`Document::undo`] would undo.
    pub fn undo_label(&self) -> Option<String> {
        self.next_undo().map(|(_, _, l)| l)
    }

    pub fn redo_label(&self) -> Option<String> {
        self.next_redo().map(|(_, _, l)| l)
    }

    /// (seq, is_structural, label) of the newest undoable step, across the
    /// document and every layer: history is one global timeline.
    fn next_undo(&self) -> Option<(u64, bool, String)> {
        let doc = self.undo.last().map(|e| (e.seq, true, e.label.clone()));
        let lay = self
            .layers
            .iter()
            .filter_map(|l| l.history.undo.last().map(|p| (p.seq, false, p.label.clone())))
            .max_by_key(|(s, _, _)| *s);
        [doc, lay].into_iter().flatten().max_by_key(|(s, _, _)| *s)
    }

    /// The oldest undone step, if it is still redoable. A step undone before
    /// newer edits were made is no longer redoable (its sequence number is
    /// older than the newest step still in the undo history).
    fn next_redo(&self) -> Option<(u64, bool, String)> {
        let doc = self.redo.last().map(|e| (e.seq, true, e.label.clone()));
        let lay = self
            .layers
            .iter()
            .filter_map(|l| l.history.redo.last().map(|p| (p.seq, false, p.label.clone())))
            .min_by_key(|(s, _, _)| *s);
        let next = [doc, lay].into_iter().flatten().min_by_key(|(s, _, _)| *s)?;
        let newest_undo = self.next_undo().map_or(0, |(s, _, _)| s);
        (next.0 > newest_undo).then_some(next)
    }

    /// Undoes the newest step anywhere in the document.
    pub fn undo(&mut self) -> bool {
        match self.next_undo() {
            Some((_, true, _)) => self.undo_structural(),
            Some((seq, false, _)) => {
                let mut dirty = Rect::default();
                for l in &mut self.layers {
                    if l.history.top_undo_seq() == Some(seq) {
                        if let Some(r) = l.undo() {
                            dirty = dirty.union(&r);
                        }
                    }
                }
                self.preview.clear();
                self.mark_dirty(dirty);
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.next_redo() {
            Some((_, true, _)) => self.redo_structural(),
            Some((seq, false, _)) => {
                let mut dirty = Rect::default();
                for l in &mut self.layers {
                    if l.history.top_redo_seq() == Some(seq) {
                        if let Some(r) = l.redo() {
                            dirty = dirty.union(&r);
                        }
                    }
                }
                self.preview.clear();
                self.mark_dirty(dirty);
                true
            }
            None => false,
        }
    }

    /// Forgets all undo/redo history (document and layers), e.g. after
    /// assembling a freshly opened document.
    pub fn clear_history(&mut self) {
        self.undo.clear();
        self.redo.clear();
        for l in &mut self.layers {
            l.history = Default::default();
        }
    }

    fn undo_structural(&mut self) -> bool {
        let Some(e) = self.undo.pop() else { return false };
        let current = self.restore(e.state);
        self.redo.push(DocEntry { seq: e.seq, label: e.label, state: current });
        true
    }

    fn redo_structural(&mut self) -> bool {
        let Some(e) = self.redo.pop() else { return false };
        let current = self.restore(e.state);
        self.undo.push(DocEntry { seq: e.seq, label: e.label, state: current });
        true
    }

    // ----- layer management ---------------------------------------------

    /// Inserts `layer` above the active block and makes it active.
    fn insert_above_active(&mut self, layer: Layer) {
        let at = if self.layers.is_empty() { 0 } else { self.active_block().end };
        self.active = layer.id;
        self.layers.insert(at, layer);
    }

    /// Adds an empty canvas-sized layer above the active block.
    pub fn add_layer(&mut self) -> LayerId {
        let id = self.alloc_id();
        let name = self.next_layer_name();
        let px = Pixels::new(self.default_depth, self.width, self.height);
        self.structural("New layer", |d| {
            d.insert_above_active(Layer::new(id, name, px, (0, 0)));
            true
        });
        id
    }

    /// Adds `pixels` as a new layer at `offset` above the active block.
    pub fn add_layer_from(&mut self, name: &str, pixels: Pixels, offset: (i32, i32)) -> LayerId {
        let id = self.alloc_id();
        let name = name.to_string();
        self.structural("Add layer", |d| {
            d.insert_above_active(Layer::new(id, name, pixels, offset));
            true
        });
        id
    }

    /// Turns a layer into a filter layer (or back) as one undo step.
    pub fn set_filter_on(&mut self, id: LayerId, on: bool) -> bool {
        if self.layer(id).is_none_or(|l| l.filter_on == on) {
            return false;
        }
        let label = if on { "Make filter layer" } else { "Make pixel layer" };
        let ok = self.structural(label, |d| {
            d.layer_mut(id).unwrap().filter_on = on;
            true
        });
        self.mark_all_dirty();
        ok
    }

    /// Changes a filter layer's settings as one undo step.
    pub fn set_layer_filter(&mut self, id: LayerId, filter: LayerFilter) -> bool {
        if self.layer(id).is_none_or(|l| l.filter == filter) {
            return false;
        }
        let ok = self.structural("Change filter layer", |d| {
            d.layer_mut(id).unwrap().filter = filter;
            true
        });
        self.mark_all_dirty();
        ok
    }

    /// Changes a filter layer's settings without recording history (for
    /// live dragging; finish with [`Document::set_layer_filter`]).
    pub fn set_layer_filter_live(&mut self, id: LayerId, filter: LayerFilter) {
        if let Some(l) = self.layer_mut(id) {
            if l.filter != filter {
                l.filter = filter;
                self.mark_all_dirty();
            }
        }
    }

    /// Luma range of what is visible below layer `id` (for Auto levels on a
    /// filter layer).
    pub fn luma_range_below(&self, id: LayerId) -> Option<(f32, f32)> {
        let i = self.index_of(id)?;
        let views: Vec<LayerView> = self.layers[..i].iter().filter(|l| l.visible).map(LayerView::of).collect();
        let img = composite_views(&views, self.canvas_rect(), None);
        let px = Pixels::F32(img);
        crate::ops::filters::luma_range_pixels([&px])
    }

    pub fn duplicate_active(&mut self) -> bool {
        let block = self.active_block();
        let new_ids: Vec<LayerId> = block.clone().map(|_| self.alloc_id()).collect();
        let new_group = self.layers[block.start].group.map(|_| self.alloc_id());
        self.structural("Duplicate layer", |d| {
            let mut copies: Vec<Layer> = d.layers[block.clone()].to_vec();
            for (l, id) in copies.iter_mut().zip(&new_ids) {
                l.id = *id;
                l.name = format!("{} copy", l.name);
                l.group = new_group;
                l.history = Default::default();
            }
            let at = block.end;
            d.active = *new_ids.last().unwrap();
            d.layers.splice(at..at, copies);
            true
        })
    }

    /// Deletes the active block. The last layer can't be deleted.
    pub fn delete_active(&mut self) -> bool {
        let block = self.active_block();
        if block.len() >= self.layers.len() {
            return false;
        }
        self.structural("Delete layer", |d| {
            let removed: Vec<LayerId> = d.layers[block.clone()].iter().map(|l| l.id).collect();
            d.layers.drain(block.clone());
            for id in removed {
                d.selected.remove(&id);
            }
            let i = block.start.saturating_sub(1).min(d.layers.len() - 1);
            d.active = d.layers[i].id;
            true
        })
    }

    /// Moves the active block one block up (`up = true`) or down.
    pub fn move_active(&mut self, up: bool) -> bool {
        let block = self.active_block();
        let (neighbor, label) = if up {
            if block.end >= self.layers.len() {
                return false;
            }
            (self.block_at(block.end), "Bring forward")
        } else {
            if block.start == 0 {
                return false;
            }
            (self.block_at(block.start - 1), "Send backward")
        };
        self.structural(label, |d| {
            let lo = block.start.min(neighbor.start);
            let hi = block.end.max(neighbor.end);
            // New order, bottom to top.
            let (lower, upper) = if up { (neighbor, block) } else { (block, neighbor) };
            let mut new: Vec<Layer> = d.layers[lower].to_vec();
            new.extend_from_slice(&d.layers[upper]);
            d.layers.splice(lo..hi, new);
            true
        })
    }

    /// Toggles visibility of a layer (its whole group). Not undoable.
    pub fn toggle_visible(&mut self, id: LayerId) {
        let Some(i) = self.index_of(id) else { return };
        let vis = !self.layers[i].visible;
        let block = self.block_at(i);
        let mut dirty = Rect::default();
        for l in &mut self.layers[block] {
            l.visible = vis;
            dirty = dirty.union(&l.bounds());
        }
        self.mark_layers_dirty(dirty);
    }

    /// Marks `dirty` for re-rendering, or the whole canvas when the active
    /// document contains a filter layer (it affects everything below).
    fn mark_layers_dirty(&mut self, dirty: Rect) {
        if self.layers.iter().any(|l| l.is_filter()) {
            self.mark_all_dirty();
        } else {
            self.mark_dirty(dirty);
        }
    }

    pub fn set_opacity(&mut self, id: LayerId, opacity: f32) {
        let Some(i) = self.index_of(id) else { return };
        let block = self.block_at(i);
        let mut dirty = Rect::default();
        for l in &mut self.layers[block] {
            l.opacity = opacity.clamp(0.0, 1.0);
            dirty = dirty.union(&l.bounds());
        }
        self.mark_layers_dirty(dirty);
    }

    pub fn rename(&mut self, id: LayerId, name: String) {
        if let Some(l) = self.layer_mut(id) {
            l.name = name;
        }
    }

    /// Composites `layers` into one new layer covering their union bounds.
    fn merge_layers(&mut self, layers: &[Layer], name: String) -> Layer {
        let bounds = layers.iter().fold(Rect::default(), |a, l| a.union(&l.bounds()));
        let depth = layers.iter().filter(|l| !l.is_filter()).map(|l| l.depth()).max().unwrap_or(self.default_depth);
        let views: Vec<LayerView> = layers
            .iter()
            .filter(|l| l.visible)
            .map(LayerView::of)
            .collect();
        let merged = composite_views(&views, bounds, None);
        let id = self.alloc_id();
        Layer::new(id, name, Pixels::from_f32(&merged, depth), (bounds.x, bounds.y))
    }

    /// Merges the active block into the block below it.
    pub fn merge_down(&mut self) -> bool {
        let block = self.active_block();
        if block.start == 0 {
            return false;
        }
        let below = self.block_at(block.start - 1);
        let range = below.start..block.end;
        let parts: Vec<Layer> = self.layers[range.clone()].to_vec();
        let name = self.layers[below.start].name.clone();
        let merged = self.merge_layers(&parts, name);
        self.structural("Merge down", |d| {
            d.active = merged.id;
            d.layers.splice(range, [merged]);
            true
        })
    }

    /// Merges the active layer and all ticked layers into one (placed at the topmost).
    pub fn merge_selected(&mut self) -> bool {
        let mut ids = self.selected.clone();
        for id in self.targets() {
            ids.insert(id);
        }
        // Include whole groups of every selected layer.
        let mut all = BTreeSet::new();
        for id in &ids {
            if let Some(i) = self.index_of(*id) {
                for l in &self.layers[self.block_at(i)] {
                    all.insert(l.id);
                }
            }
        }
        if all.len() < 2 {
            return false;
        }
        let parts: Vec<Layer> = self.layers.iter().filter(|l| all.contains(&l.id)).cloned().collect();
        let top = self.layers.iter().rposition(|l| all.contains(&l.id)).unwrap();
        let name = self.layers[top].name.clone();
        let merged = self.merge_layers(&parts, name);
        self.structural("Merge layers", |d| {
            let top_id = d.layers[top].id;
            let mut out = Vec::with_capacity(d.layers.len());
            for l in d.layers.drain(..) {
                if l.id == top_id {
                    out.push(merged.clone());
                } else if !all.contains(&l.id) {
                    out.push(l);
                }
            }
            d.layers = out;
            d.active = merged.id;
            d.selected.clear();
            true
        })
    }

    /// Groups the active layer and the ticked layers: they are moved
    /// next to each other (at the position of the topmost) and tagged with a
    /// shared group id.
    pub fn group_selected(&mut self) -> bool {
        let mut ids = self.selected.clone();
        ids.insert(self.active);
        let mut all = BTreeSet::new();
        for id in &ids {
            if let Some(i) = self.index_of(*id) {
                for l in &self.layers[self.block_at(i)] {
                    all.insert(l.id);
                }
            }
        }
        if all.len() < 2 {
            return false;
        }
        let gid = self.alloc_id();
        self.structural("Group layers", |d| {
            let top = d.layers.iter().rposition(|l| all.contains(&l.id)).unwrap();
            let top_id = d.layers[top].id;
            let members: Vec<Layer> = d
                .layers
                .iter()
                .filter(|l| all.contains(&l.id))
                .cloned()
                .map(|mut l| {
                    l.group = Some(gid);
                    l
                })
                .collect();
            let mut out = Vec::with_capacity(d.layers.len());
            for l in d.layers.drain(..) {
                if l.id == top_id {
                    out.extend(members.iter().cloned());
                } else if !all.contains(&l.id) {
                    out.push(l);
                }
            }
            d.layers = out;
            d.selected.clear();
            true
        })
    }

    /// Separates a group back into independent layers.
    pub fn ungroup(&mut self, group: GroupId) -> bool {
        if self.group_members(group).is_empty() {
            return false;
        }
        self.structural("Ungroup layers", |d| {
            for l in &mut d.layers {
                if l.group == Some(group) {
                    l.group = None;
                }
            }
            true
        })
    }

    /// Flattens everything into one layer.
    pub fn flatten(&mut self) -> bool {
        if self.layers.len() < 2 {
            return false;
        }
        let parts = self.layers.clone();
        let mut merged = self.merge_layers(&parts, "Background".into());
        // Clip to the canvas like most editors do.
        let canvas = self.canvas_rect();
        let px = merged.pixels.crop(canvas.translate(-merged.offset.0, -merged.offset.1));
        merged.pixels = Arc::new(px);
        merged.offset = (0, 0);
        self.structural("Flatten", |d| {
            d.active = merged.id;
            d.layers = vec![merged];
            d.selected.clear();
            true
        })
    }

    /// Arranges the layers (groups move as one) so none overlap,
    /// in rows forming a roughly square block at the canvas origin, then fits
    /// the canvas to it. A bottom layer that is a single solid color (or
    /// empty) counts as the background: it isn't tiled but refilled to the
    /// new canvas size. Any other bottom layer is tiled like the rest.
    pub fn tile_layers(&mut self, gap: u32) -> bool {
        if self.layers.is_empty() {
            return false;
        }
        // With ticked layers, only those (and the active one) are tiled.
        let picked: Option<BTreeSet<LayerId>> = (!self.selected.is_empty()).then(|| self.channel_merge_set().into_iter().collect());
        let first = self.block_at(0);
        let bg_fill = if picked.is_some() {
            None
        } else {
            (first.len() == 1).then(|| uniform_color(&self.layers[0].pixels)).flatten()
        };
        let mut blocks = Vec::new();
        let mut i = if bg_fill.is_some() { first.end } else { 0 };
        while i < self.layers.len() {
            let b = self.block_at(i);
            i = b.end;
            if picked.as_ref().is_none_or(|p| self.layers[b.clone()].iter().any(|l| p.contains(&l.id))) {
                blocks.push(b);
            }
        }
        if blocks.is_empty() {
            return false;
        }
        let bounds: Vec<Rect> = blocks
            .iter()
            .map(|b| self.layers[b.clone()].iter().fold(Rect::default(), |a, l| a.union(&l.bounds())))
            .collect();
        let sizes: Vec<(u32, u32)> = bounds.iter().map(|r| (r.w.max(1), r.h.max(1))).collect();
        let (pos, (w, h)) = crate::ops::layout::pack_square(&sizes, gap);
        if w == 0 || h == 0 {
            return false;
        }
        // A subset is tiled from its current top-left corner, and the canvas
        // only grows (if needed) to show it.
        let origin = match picked {
            Some(_) => bounds.iter().fold(None, |a: Option<Rect>, r| Some(a.map_or(*r, |a| a.union(r)))).map_or((0, 0), |r| (r.x, r.y)),
            None => (0, 0),
        };
        let subset = picked.is_some();
        self.structural("Tile layers", |d| {
            for ((block, r), (x, y)) in blocks.iter().zip(&bounds).zip(&pos) {
                let (dx, dy) = (x + origin.0 - r.x, y + origin.1 - r.y);
                for l in &mut d.layers[block.clone()] {
                    l.offset = (l.offset.0 + dx, l.offset.1 + dy);
                }
            }
            if subset {
                let need = Rect::new(origin.0, origin.1, w, h).union(&d.canvas_rect());
                if need != d.canvas_rect() {
                    d.resize_canvas_raw(need);
                }
                return true;
            }
            d.width = w;
            d.height = h;
            if let Some(c) = bg_fill {
                let l = &mut d.layers[0];
                l.pixels = Arc::new(Pixels::filled(l.depth(), w, h, c));
                l.offset = (0, 0);
            }
            true
        })
    }

    /// Puts the layers in the given order (bottom to top). `order` must
    /// contain every layer exactly once and keep groups together.
    pub fn set_order(&mut self, order: &[LayerId]) -> bool {
        let current: Vec<LayerId> = self.layers.iter().map(|l| l.id).collect();
        if order == current.as_slice() || order.len() != current.len() {
            return false;
        }
        let mut sorted_a = order.to_vec();
        let mut sorted_b = current.clone();
        sorted_a.sort_unstable();
        sorted_b.sort_unstable();
        if sorted_a != sorted_b {
            return false;
        }
        let layers: Vec<Layer> = order.iter().map(|id| self.layer(*id).unwrap().clone()).collect();
        // Groups must stay contiguous.
        let mut seen_done = BTreeSet::new();
        let mut last: Option<GroupId> = None;
        for l in &layers {
            if l.group != last {
                if let Some(g) = l.group {
                    if !seen_done.insert(g) {
                        return false;
                    }
                }
                last = l.group;
            }
        }
        self.structural("Reorder layers", |d| {
            d.layers = layers;
            true
        })
    }

    /// Replaces the active layer with four grayscale layers holding its red,
    /// green, blue and alpha channels (top to bottom: R, G, B, A).
    pub fn split_channels(&mut self) -> bool {
        let i = self.active_index();
        let src = self.layers[i].clone();
        let names = ["R", "G", "B", "A"];
        let ids: Vec<LayerId> = (0..4).map(|_| self.alloc_id()).collect();
        let mut new_layers = Vec::new();
        for c in (0..4).rev() {
            let mut px = (*src.pixels).clone();
            px.map_pixels(|p| [p[c], p[c], p[c], 1.0]);
            let mut l = Layer::new(ids[c], format!("{} {}", src.name, names[c]), px, src.offset);
            l.visible = src.visible;
            new_layers.push(l);
        }
        self.structural("Split channels", |d| {
            d.layers.splice(i..i + 1, new_layers);
            d.active = ids[0];
            d.selected.clear();
            true
        })
    }

    /// Combines 3 (RGB) or 4 (RGBA) layers into one, using each layer's
    /// brightness as one channel. Layers named "… R", "… G", "… B", "… A"
    /// (as made by [`Document::split_channels`]) go to their channel; otherwise
    /// the order from top to bottom is R, G, B, A. Areas outside a layer, or
    /// transparent, count as 0.
    pub fn merge_channels(&mut self, ids: &[LayerId]) -> bool {
        if !(3..=4).contains(&ids.len()) {
            return false;
        }
        // Top to bottom.
        let mut layers: Vec<&Layer> = self.layers.iter().rev().filter(|l| ids.contains(&l.id)).collect();
        if layers.len() != ids.len() {
            return false;
        }
        let tag = |l: &Layer| match l.name.rsplit(' ').next() {
            Some("R") => Some(0),
            Some("G") => Some(1),
            Some("B") => Some(2),
            Some("A") => Some(3),
            _ => None,
        };
        let tags: Vec<Option<usize>> = layers.iter().map(|l| tag(l)).collect();
        let mut by_tag = tags.iter().flatten().copied().collect::<Vec<_>>();
        by_tag.sort_unstable();
        by_tag.dedup();
        if by_tag.len() == layers.len() && by_tag.iter().all(|t| *t < layers.len()) {
            layers.sort_by_key(|l| tag(l));
        }
        let bounds = layers.iter().fold(Rect::default(), |a, l| a.union(&l.bounds()));
        let depth = layers.iter().map(|l| l.depth()).max().unwrap_or(self.default_depth);
        let out = Buffer::<f32>::from_fn(bounds.w, bounds.h, |x, y| {
            let (cx, cy) = (bounds.x + x as i32, bounds.y + y as i32);
            let mut o = [0.0, 0.0, 0.0, 1.0];
            for (c, l) in layers.iter().enumerate() {
                let p = l.pixels.get_or_clear(cx - l.offset.0, cy - l.offset.1);
                o[c] = crate::pixel::luma(p) * p[3];
            }
            o
        });
        let top = layers.iter().map(|l| self.index_of(l.id).unwrap()).max().unwrap();
        let first = &layers[0].name;
        let name = match first.rsplit_once(' ') {
            Some((base, _)) if tags[0].is_some() => base.to_string(),
            _ => "Merged channels".to_string(),
        };
        let id = self.alloc_id();
        let merged = Layer::new(id, name, Pixels::from_f32(&out, depth), (bounds.x, bounds.y));
        let ids: Vec<LayerId> = ids.to_vec();
        self.structural("Merge channels", |d| {
            let top_id = d.layers[top].id;
            let mut out = Vec::with_capacity(d.layers.len());
            for l in d.layers.drain(..) {
                if l.id == top_id {
                    out.push(merged.clone());
                } else if !ids.contains(&l.id) {
                    out.push(l);
                }
            }
            d.layers = out;
            d.active = id;
            d.selected.clear();
            true
        })
    }

    /// The layers a channel merge would use: the active layer and the ticked ones.
    pub fn channel_merge_set(&self) -> Vec<LayerId> {
        let mut s: Vec<LayerId> = self.selected.iter().copied().filter(|id| self.layer(*id).is_some()).collect();
        if !s.contains(&self.active) {
            s.push(self.active);
        }
        s
    }

    /// Resizes the canvas to exactly contain every layer.
    pub fn fit_canvas_to_layers(&mut self) -> bool {
        let u = self.layers.iter().fold(Rect::default(), |a, l| a.union(&l.bounds()));
        if u.is_empty() || u == self.canvas_rect() {
            return false;
        }
        self.structural("Fit canvas to layers", |d| {
            d.width = u.w;
            d.height = u.h;
            for l in &mut d.layers {
                l.offset = (l.offset.0 - u.x, l.offset.1 - u.y);
            }
            true
        })
    }

    /// Converts the target layers to another storage depth (undoable per layer).
    pub fn convert_depth(&mut self, depth: Depth) -> bool {
        self.edit_targets(&format!("Convert to {}", depth.short()), |l| {
            (l.depth() != depth).then(|| (l.pixels.convert(depth), l.offset))
        })
    }

    /// Grows (or shrinks) the active layers to exactly cover the canvas.
    pub fn layer_to_canvas_size(&mut self) -> bool {
        let canvas = self.canvas_rect();
        self.edit_targets("Layer to canvas size", |l| {
            if l.bounds() == canvas {
                return None;
            }
            let mut px = Pixels::new(l.depth(), canvas.w, canvas.h);
            px.paste(&l.pixels, l.offset.0, l.offset.1);
            Some((px, (0, 0)))
        })
    }

    /// Shrinks the active layers to their non-transparent content.
    pub fn crop_layer_to_content(&mut self) -> bool {
        self.edit_targets("Crop layer to content", |l| {
            let b = l.pixels.opaque_bounds()?;
            if b == l.pixels.rect() {
                return None;
            }
            Some((l.pixels.crop(b), (l.offset.0 + b.x, l.offset.1 + b.y)))
        })
    }

    // ----- selection -----------------------------------------------------

    /// Replaces the selection (`None` deselects) as one undo step.
    pub fn set_selection(&mut self, label: &str, mask: Option<Mask>) -> bool {
        let mask = mask.and_then(|m| {
            // Store it cropped to its covered area.
            let b = m.bounds()?.intersect(&self.canvas_rect());
            (!b.is_empty()).then(|| crop_mask(&m, b))
        });
        if mask.is_none() && self.selection.is_none() {
            return false;
        }
        let mask = mask.map(Arc::new);
        self.structural(label, |d| {
            d.selection = mask;
            true
        })
    }

    /// The selection combined with `mask` (union, keeping the higher coverage).
    pub fn selection_plus(&self, mask: &Mask) -> Mask {
        match &self.selection {
            None => mask.clone(),
            Some(old) => {
                let r = old.rect.union(&mask.rect);
                let mut m = Mask::new(r);
                for y in r.y..r.bottom() {
                    for x in r.x..r.right() {
                        m.set(x, y, old.get(x, y).max(mask.get(x, y)));
                    }
                }
                m
            }
        }
    }

    /// Moves (or copies, if `copy`) the pixels under `mask` (canvas coords)
    /// from the active layer/group into a new layer above it, or into the
    /// existing layer `into`. Returns the receiving layer's id, or `None` if
    /// the mask selects nothing.
    pub fn selection_to_layer(&mut self, mask: &Mask, copy: bool, into: Option<LayerId>) -> Option<LayerId> {
        let into = into.filter(|id| self.layer(*id).is_some_and(|l| !l.is_filter()));
        let sources: Vec<LayerId> =
            self.targets().into_iter().filter(|id| Some(*id) != into && !self.layer(*id).unwrap().is_filter()).collect();
        let src_bounds = sources
            .iter()
            .filter_map(|id| self.layer(*id))
            .fold(Rect::default(), |a, l| a.union(&l.bounds()));
        let bbox = mask.bounds()?.intersect(&src_bounds);
        if bbox.is_empty() {
            return None;
        }
        let layers: Vec<Layer> = sources.iter().filter_map(|id| self.layer(*id).cloned()).collect();
        let depth = layers.iter().map(|l| l.depth()).max().unwrap_or(self.default_depth);
        let single = layers.len() == 1;

        // Extract: composite of the sources, weighted by the mask.
        let mut out = Buffer::<f32>::new(bbox.w, bbox.h);
        for l in &layers {
            if !l.visible && !single {
                continue;
            }
            for y in bbox.y..bbox.bottom() {
                for x in bbox.x..bbox.right() {
                    let m = mask.get(x, y);
                    if m <= 0.0 {
                        continue;
                    }
                    let mut p: Px = l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1);
                    p[3] *= m * if single { 1.0 } else { l.opacity };
                    let (ox, oy) = ((x - bbox.x) as u32, (y - bbox.y) as u32);
                    let d = out.get(ox, oy);
                    out.set(ox, oy, crate::pixel::over(p, d));
                }
            }
        }
        out.opaque_bounds()?;

        let new_id = into.unwrap_or_else(|| self.alloc_id());
        let new_layer = match into {
            // Join the existing layer: grow it and lay the new pixels over it.
            Some(id) => {
                let l = self.layer(id).unwrap();
                let area = l.bounds().union(&bbox);
                let extracted = Pixels::F32(out);
                let views = [LayerView { opacity: 1.0, ..LayerView::of(l) }, LayerView::pixels(&extracted, (bbox.x, bbox.y), 1.0)];
                let merged = composite_views(&views, area, None);
                let mut nl = l.clone();
                nl.pixels = Arc::new(Pixels::from_f32(&merged, l.depth().max(depth)));
                nl.offset = (area.x, area.y);
                nl
            }
            None => {
                let mut nl = Layer::new(new_id, "Selection", Pixels::from_f32(&out, depth), (bbox.x, bbox.y));
                if single {
                    nl.opacity = layers[0].opacity;
                }
                nl
            }
        };
        let label = match (copy, into.is_some()) {
            (_, true) => "Add selection to layer",
            (true, false) => "Copy selection to layer",
            (false, false) => "Cut selection to layer",
        };
        self.structural(label, |d| {
            if !copy {
                for id in &sources {
                    let l = d.layer_mut(*id).unwrap();
                    let off = l.offset;
                    let local = bbox.translate(-off.0, -off.1).intersect(&l.pixels.rect());
                    if local.is_empty() {
                        continue;
                    }
                    let px = l.pixels_mut();
                    for y in local.y..local.bottom() {
                        for x in local.x..local.right() {
                            let m = mask.get(x + off.0, y + off.1);
                            if m > 0.0 {
                                let mut p = px.get(x as u32, y as u32);
                                p[3] *= 1.0 - m;
                                px.set(x as u32, y as u32, p);
                            }
                        }
                    }
                }
            }
            match into {
                Some(id) => {
                    let i = d.index_of(id).unwrap();
                    d.layers[i] = new_layer;
                    d.active = id;
                }
                None => d.insert_above_active(new_layer),
            }
            true
        });
        self.extract_layer = Some(new_id);
        Some(new_id)
    }

    /// Sets the canvas size. `anchor` is the position of the old canvas's
    /// top-left corner inside the new one. Layers keep their pixels.
    pub fn resize_canvas(&mut self, width: u32, height: u32, anchor: (i32, i32)) -> bool {
        if width == 0 || height == 0 || (width == self.width && height == self.height && anchor == (0, 0)) {
            return false;
        }
        self.structural("Canvas size", |d| {
            d.width = width;
            d.height = height;
            for l in &mut d.layers {
                l.offset.0 += anchor.0;
                l.offset.1 += anchor.1;
            }
            true
        })
    }

    /// Sets the canvas to `r` (current canvas coords) without recording history.
    pub fn resize_canvas_raw(&mut self, r: Rect) -> bool {
        if r.is_empty() || r == self.canvas_rect() {
            return false;
        }
        self.width = r.w;
        self.height = r.h;
        for l in &mut self.layers {
            l.offset = (l.offset.0 - r.x, l.offset.1 - r.y);
        }
        true
    }

    /// Whether layer `id` has non-transparent pixels under `mask`.
    pub fn layer_under_mask_only(&self, id: LayerId, mask: &Mask) -> bool {
        let (Some(l), Some(b)) = (self.layer(id), mask.bounds()) else { return false };
        let r = b.intersect(&l.bounds());
        (r.y..r.bottom()).any(|y| {
            (r.x..r.right())
                .any(|x| mask.get(x, y) > 0.0 && l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1)[3] > 0.0)
        })
    }

    /// Crops the canvas to `rect` (canvas coords).
    pub fn crop_canvas(&mut self, rect: Rect) -> bool {
        let r = rect.intersect(&self.canvas_rect());
        if r.is_empty() {
            return false;
        }
        self.resize_canvas(r.w, r.h, (-r.x, -r.y))
    }

    /// Runs `f` over every layer (pixels, offset) inside one structural step.
    /// Used for canvas-wide transforms.
    pub fn transform_all(
        &mut self,
        label: &str,
        new_size: (u32, u32),
        f: impl Fn(&Layer) -> (Pixels, (i32, i32)),
    ) -> bool {
        let results: Vec<(Pixels, (i32, i32))> = self.layers.iter().map(&f).collect();
        self.structural(label, |d| {
            for (l, (px, off)) in d.layers.iter_mut().zip(results) {
                l.pixels = Arc::new(px);
                l.offset = off;
            }
            d.width = new_size.0;
            d.height = new_size.1;
            true
        })
    }

    /// Changes whenever the saved content changes (edits, undo/redo, layer
    /// order, names, visibility, opacity, canvas size), but not for previews.
    /// Compare tokens to see if a document has unsaved changes.
    pub fn content_token(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.width, self.height).hash(&mut h);
        self.undo.len().hash(&mut h);
        self.undo.last().map(|e| e.seq).hash(&mut h);
        for l in &self.layers {
            (l.id, &l.name, l.visible, l.opacity.to_bits(), l.offset, l.group).hash(&mut h);
            (l.filter_on, format!("{:?}", l.filter)).hash(&mut h);
            (l.history.undo.len(), l.history.undo.last().map(|p| p.seq)).hash(&mut h);
            (Arc::as_ptr(&l.pixels) as usize).hash(&mut h);
        }
        h.finish()
    }

    /// Total memory used by layer pixels (excluding history).
    pub fn pixel_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.pixels.byte_size()).sum()
    }

    /// The deepest layer depth in the document.
    pub fn max_depth(&self) -> Depth {
        self.layers.iter().filter(|l| !l.is_filter()).map(|l| l.depth()).max().unwrap_or(self.default_depth)
    }
}

/// The part of `m` inside `r`.
fn crop_mask(m: &Mask, r: Rect) -> Mask {
    let mut out = Mask::new(r);
    for y in r.y..r.bottom() {
        for x in r.x..r.right() {
            out.set(x, y, m.get(x, y));
        }
    }
    out
}

/// The single color of a layer whose pixels are all identical.
fn uniform_color(px: &Pixels) -> Option<Px> {
    let same = crate::dispatch!(px, b => {
        let first = &b.data[..4.min(b.data.len())];
        b.data.chunks_exact(4).all(|c| c == first)
    });
    (same && px.width() > 0 && px.height() > 0).then(|| px.get(0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> Color {
        Color::rgba(1.0, 0.0, 0.0, 1.0)
    }

    #[test]
    fn smart_undo_prefers_newest() {
        let mut d = Document::new(8, 8, Depth::U8, Some(Color::WHITE));
        let bg = d.active;
        let seq = d.next_seq();
        d.active_layer_mut().edit_region(seq, "paint", Rect::new(0, 0, 2, 2), |p| p.set(0, 0, red().px()));
        let l2 = d.add_layer();
        assert_eq!(d.active, l2);
        // Newest is the structural add; undo removes the layer.
        assert!(d.undo());
        assert_eq!(d.layers.len(), 1);
        assert_eq!(d.active, bg);
        // Next undo reverts the paint.
        assert!(d.undo());
        assert_eq!(d.layers[0].pixels.get(0, 0), Color::WHITE.px());
        // Redo both.
        assert!(d.redo());
        assert_eq!(d.layers[0].pixels.get(0, 0), red().px());
        assert!(d.redo());
        assert_eq!(d.layers.len(), 2);
    }

    #[test]
    fn global_history_across_layers() {
        let mut d = Document::new(4, 4, Depth::U8, Some(Color::WHITE));
        let a = d.active;
        let b = d.add_layer();
        let s = d.next_seq();
        d.layer_mut(a).unwrap().edit_region(s, "a", Rect::new(0, 0, 1, 1), |p| p.set(0, 0, red().px()));
        let s = d.next_seq();
        d.layer_mut(b).unwrap().edit_region(s, "b", Rect::new(1, 1, 1, 1), |p| p.set(1, 1, red().px()));
        d.set_active(a);
        // The newest step is undone first, whichever layer is active.
        assert_eq!(d.undo_label().as_deref(), Some("b"));
        d.undo();
        assert_eq!(d.layer(b).unwrap().pixels.get(1, 1)[3], 0.0);
        assert_eq!(d.undo_label().as_deref(), Some("a"));
        d.undo();
        assert_eq!(d.layer(a).unwrap().pixels.get(0, 0), Color::WHITE.px());
        assert_eq!(d.redo_label().as_deref(), Some("a"));
        d.redo();
        // A new edit drops the remaining redo ("b").
        let s = d.next_seq();
        d.layer_mut(a).unwrap().edit_region(s, "c", Rect::new(2, 2, 1, 1), |p| p.set(2, 2, red().px()));
        assert_eq!(d.redo_label(), None);
        assert!(!d.redo());
    }

    #[test]
    fn selection_cut() {
        let mut d = Document::new(4, 4, Depth::U8, Some(Color::WHITE));
        let src = d.active;
        let mask = Mask::full(Rect::new(1, 1, 2, 2));
        let id = d.selection_to_layer(&mask, false, None).unwrap();
        let l = d.layer(id).unwrap();
        assert_eq!(l.offset, (1, 1));
        assert_eq!(l.width(), 2);
        assert_eq!(d.layer(src).unwrap().pixels.get(1, 1)[3], 0.0);
        assert_eq!(d.layer(src).unwrap().pixels.get(0, 0)[3], 1.0);
        d.undo();
        assert_eq!(d.layers.len(), 1);
        assert_eq!(d.layer(src).unwrap().pixels.get(1, 1)[3], 1.0);
    }

    #[test]
    fn group_and_move() {
        let mut d = Document::new(4, 4, Depth::U8, None);
        let a = d.active;
        let b = d.add_layer();
        let c = d.add_layer();
        d.set_active(c);
        d.selected.insert(a);
        assert!(d.group_selected());
        // a and c are now adjacent and grouped at the top; b is at the bottom.
        let order: Vec<_> = d.layers.iter().map(|l| l.id).collect();
        assert_eq!(order, vec![b, a, c]);
        assert_eq!(d.targets().len(), 2);
        assert!(d.move_active(false));
        let order: Vec<_> = d.layers.iter().map(|l| l.id).collect();
        assert_eq!(order, vec![a, c, b]);
        let g = d.layer(a).unwrap().group.unwrap();
        assert!(d.ungroup(g));
        assert_eq!(d.targets(), vec![c]);
    }

    #[test]
    fn nudges_merge_into_one_step() {
        let mut doc = Document::new(10, 10, Depth::U8, None);
        doc.add_layer();
        assert!(doc.nudge_targets(1, 0));
        assert!(doc.nudge_targets(1, 0));
        assert!(doc.nudge_targets(0, 2));
        assert_eq!(doc.active_layer().offset, (2, 2));
        assert!(doc.undo());
        assert_eq!(doc.active_layer().offset, (0, 0));
        assert!(doc.redo());
        assert_eq!(doc.active_layer().offset, (2, 2));
        // Something else in between starts a new step.
        doc.set_active(doc.layers[0].id);
        doc.nudge_targets(1, 1);
        doc.set_active(doc.layers[1].id);
        doc.nudge_targets(1, 1);
        assert!(doc.undo());
        assert_eq!(doc.active_layer().offset, (2, 2));
        assert_eq!(doc.layers[0].offset, (1, 1));
    }

    #[test]
    fn tile_only_ticked() {
        let mut d = Document::new(300, 300, Depth::U8, Some(Color::WHITE));
        let mut ids = Vec::new();
        for i in 0..4 {
            ids.push(d.add_layer_from("img", Pixels::filled(Depth::U8, 100, 60, [1.0, 0.0, 0.0, 1.0]), (20 + i, 30)));
        }
        d.selected.insert(ids[0]);
        d.selected.insert(ids[1]);
        d.set_active(ids[1]);
        assert!(d.tile_layers(0));
        // The two ticked layers are packed from their top-left; the others stay.
        let o: Vec<_> = ids.iter().map(|id| d.layer(*id).unwrap().offset).collect();
        assert_eq!(o[2], (22, 30));
        assert_eq!(o[3], (23, 30));
        let mut ab = vec![o[0], o[1]];
        ab.sort();
        assert!(ab == vec![(20, 30), (20, 90)] || ab == vec![(20, 30), (120, 30)], "{ab:?}");
        assert_eq!((d.width, d.height), (300, 300));
        assert_eq!(d.layers[0].bounds(), Rect::new(0, 0, 300, 300));
    }

    #[test]
    fn tile_and_fit() {
        let mut d = Document::new(50, 50, Depth::U8, Some(Color::WHITE));
        for _ in 0..4 {
            d.add_layer_from("img", Pixels::filled(Depth::U8, 100, 60, [1.0, 0.0, 0.0, 1.0]), (5, 5));
        }
        assert!(d.tile_layers(0));
        assert_eq!((d.width, d.height), (200, 120));
        let mut offs: Vec<_> = d.layers[1..].iter().map(|l| l.offset).collect();
        offs.sort();
        assert_eq!(offs, vec![(0, 0), (0, 60), (100, 0), (100, 60)]);
        // Solid background follows the canvas.
        assert_eq!(d.layers[0].bounds(), Rect::new(0, 0, 200, 120));
        assert!(d.undo());
        assert_eq!((d.width, d.height), (50, 50));

        // A bottom layer with an actual image is tiled too, never left outside.
        let mut d = Document::from_pixels(Pixels::filled(Depth::U8, 100, 60, [0.0, 0.0, 1.0, 1.0]), "img0");
        d.layers[0].pixels = Arc::new({
            let mut p = (*d.layers[0].pixels).clone();
            p.set(0, 0, [1.0; 4]);
            p
        });
        for _ in 0..3 {
            d.add_layer_from("img", Pixels::filled(Depth::U8, 100, 60, [1.0, 0.0, 0.0, 1.0]), (5, 5));
        }
        assert!(d.tile_layers(0));
        assert_eq!((d.width, d.height), (200, 120));
        let mut offs: Vec<_> = d.layers.iter().map(|l| l.offset).collect();
        offs.sort();
        assert_eq!(offs, vec![(0, 0), (0, 60), (100, 0), (100, 60)]);

        let mut d = Document::new(10, 10, Depth::U8, None);
        d.add_layer_from("off", Pixels::new(Depth::U8, 4, 4), (-3, 12));
        assert!(d.fit_canvas_to_layers());
        assert_eq!((d.width, d.height), (13, 16));
        assert_eq!(d.layers[0].offset, (3, 0));
        assert_eq!(d.layers[1].offset, (0, 12));
    }

    #[test]
    fn split_and_merge_channels() {
        let mut d = Document::new(2, 1, Depth::U8, None);
        d.layers[0].pixels = Arc::new(Pixels::filled(Depth::U8, 2, 1, [1.0, 0.0, 0.2, 0.6]));
        d.layers[0].name = "pic".into();
        assert!(d.split_channels());
        let names: Vec<_> = d.layers.iter().rev().map(|l| l.name.clone()).collect();
        assert_eq!(names, ["pic R", "pic G", "pic B", "pic A"]);
        assert_eq!(d.layers[3].pixels.get(0, 0), [1.0, 1.0, 1.0, 1.0]);
        // Shuffle the order: names decide the channels.
        let order: Vec<_> = d.layers.iter().map(|l| l.id).rev().collect();
        assert!(d.set_order(&order));
        let all: Vec<_> = d.layers.iter().map(|l| l.id).collect();
        assert!(d.merge_channels(&all));
        assert_eq!(d.layers.len(), 1);
        assert_eq!(d.layers[0].name, "pic");
        let p = d.layers[0].pixels.get(0, 0);
        assert!((p[0] - 1.0).abs() < 0.01 && p[1] < 0.01 && (p[2] - 0.2).abs() < 0.01 && (p[3] - 0.6).abs() < 0.01, "{p:?}");
    }

    #[test]
    fn merge_mixed_depth() {
        let mut d = Document::new(2, 2, Depth::U8, Some(Color::WHITE));
        let id = d.add_layer_from("hdr", Pixels::filled(Depth::F32, 1, 1, [2.0, 0.0, 0.0, 1.0]), (1, 1));
        d.set_active(id);
        assert!(d.merge_down());
        assert_eq!(d.layers.len(), 1);
        assert_eq!(d.layers[0].depth(), Depth::F32);
        assert_eq!(d.layers[0].pixels.get(1, 1), [2.0, 0.0, 0.0, 1.0]);
    }
}
