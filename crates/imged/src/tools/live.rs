//! Live painting through document previews.

use std::sync::Arc;

use imged_core::layer::LayerId;
use imged_core::ops::paint::{paint_mask, PaintMode};
use imged_core::{Document, Mask, Pixels, Px, Rect, Vec2};

/// Paints onto a copy of the active layer that the document shows as a
/// preview; [`LivePaint::commit`] makes it real as one undo step.
///
/// The working copy covers the layer *and* the canvas, so strokes outside a
/// small layer (e.g. one created from a selection) are kept: on commit the
/// layer grows just enough to hold them.
pub struct LivePaint {
    pub id: LayerId,
    /// The layer's pixels when painting started.
    orig: Arc<Pixels>,
    orig_offset: (i32, i32),
    /// Working area (possibly larger than the layer).
    base: Arc<Pixels>,
    pub offset: (i32, i32),
    /// Everything painted so far (base-local).
    touched: Rect,
    /// Region painted by the latest update (restored on the next one).
    last: Rect,
}

impl LivePaint {
    /// `grow`: allow painting outside the layer (it grows on commit).
    pub fn begin(doc: &mut Document, grow: bool) -> Self {
        doc.clear_preview();
        let l = doc.active_layer();
        let lb = l.bounds();
        let area = if grow { lb.union(&doc.canvas_rect()) } else { lb };
        let base = if area == lb {
            l.pixels.clone()
        } else {
            let mut px = Pixels::new(l.depth(), area.w, area.h);
            px.paste(&l.pixels, lb.x - area.x, lb.y - area.y);
            Arc::new(px)
        };
        LivePaint {
            id: l.id,
            orig: l.pixels.clone(),
            orig_offset: l.offset,
            base,
            offset: (area.x, area.y),
            touched: Rect::default(),
            last: Rect::default(),
        }
    }

    /// The working area in base-local coordinates.
    pub fn layer_rect(&self) -> Rect {
        self.base.rect()
    }

    /// Canvas → base-local coordinates.
    pub fn local(&self, p: Vec2) -> Vec2 {
        Vec2::new(p.x - self.offset.0 as f32, p.y - self.offset.1 as f32)
    }

    /// Repaints `region` (base-local): base pixels plus `parts` (masks in
    /// base-local coordinates and their colors). The previously painted
    /// region is restored first, so shapes can be redrawn as they change.
    pub fn update(&mut self, doc: &mut Document, parts: &[(&Mask, Px)], region: Rect, mode: PaintMode) {
        let region = region.union(&self.last).intersect(&self.base.rect());
        self.last = region;
        if region.is_empty() {
            return;
        }
        let mut arc = match doc.take_preview(self.id) {
            Some(p) => p.pixels,
            None => Arc::new((*self.base).clone()),
        };
        let px = Arc::make_mut(&mut arc);
        px.paste(&self.base.crop(region), region.x, region.y);
        for (m, c) in parts {
            paint_mask(px, None, m, region, *c, mode);
        }
        self.touched = self.touched.union(&region);
        let dirty = region.translate(self.offset.0, self.offset.1);
        // The first update also covers the old layer area (the preview replaces it).
        let dirty = if Arc::ptr_eq(&self.orig, &self.base) { dirty } else { dirty.union(&self.orig_bounds()) };
        doc.set_preview_partial(self.id, arc, self.offset, dirty);
    }

    fn orig_bounds(&self) -> Rect {
        Rect::new(self.orig_offset.0, self.orig_offset.1, self.orig.width(), self.orig.height())
    }

    /// Makes the preview permanent. Returns false if nothing was painted.
    pub fn commit(self, doc: &mut Document, label: &str) -> bool {
        let Some(p) = doc.take_preview(self.id) else { return false };
        if self.touched.is_empty() {
            doc.mark_dirty(self.orig_bounds());
            return false;
        }
        let seq = doc.next_seq();
        let Some(layer) = doc.layer_mut(self.id) else { return false };
        if !Arc::ptr_eq(&layer.pixels, &self.orig) {
            // The layer changed underneath us (e.g. undo mid-stroke); drop the stroke.
            doc.mark_all_dirty();
            return false;
        }
        // Where the original layer sits inside the working area.
        let orig_in_base = self.orig_bounds().translate(-self.offset.0, -self.offset.1);
        let touched_canvas = self.touched.translate(self.offset.0, self.offset.1);
        if orig_in_base.union(&self.touched) == orig_in_base {
            // Stayed inside the layer: record just the touched region.
            let local = self.touched.translate(-orig_in_base.x, -orig_in_base.y);
            let before = self.orig.crop(local);
            layer.pixels = if Arc::ptr_eq(&self.orig, &self.base) {
                p.pixels
            } else {
                Arc::new(p.pixels.crop(orig_in_base))
            };
            layer.record_region(seq, label, local, before);
        } else {
            // Grow the layer to cover the paint.
            let keep = orig_in_base.union(&self.touched);
            let px = p.pixels.crop(keep);
            layer.replace(seq, label, px, (self.offset.0 + keep.x, self.offset.1 + keep.y));
        }
        doc.mark_dirty(touched_canvas.union(&self.orig_bounds()));
        true
    }

    pub fn cancel(self, doc: &mut Document) {
        if doc.take_preview(self.id).is_some() {
            doc.mark_dirty(self.touched.union(&self.last).translate(self.offset.0, self.offset.1).union(&self.orig_bounds()));
        }
    }

    /// Removes the preview (when a command will repaint the same thing).
    pub fn discard(self, doc: &mut Document) {
        self.cancel(doc);
    }
}
