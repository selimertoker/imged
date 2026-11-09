//! CPU layer compositing.
//!
//! This is the reference implementation; a GPU compositor can replace it
//! behind the same inputs (a list of [`LayerView`]s and a target rectangle).

use rayon::prelude::*;

use crate::buffer::Buffer;
use crate::dispatch;
use crate::document::Document;
use crate::geom::Rect;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use crate::layer::Layer;
use crate::ops::filters::LayerFilter;
use crate::pixel::{over, Px};
use crate::pixels::Pixels;

/// A layer as seen by the compositor (possibly a preview instead of the real pixels).
#[derive(Clone, Copy)]
pub struct LayerView<'a> {
    pub pixels: &'a Pixels,
    pub offset: (i32, i32),
    pub opacity: f32,
    /// Filter layers change everything composited below them, where their
    /// pixels are opaque (everywhere when `FilterView::everywhere`).
    pub filter: Option<FilterView>,
}

#[derive(Clone, Copy)]
pub struct FilterView {
    pub filter: LayerFilter,
    /// The layer has no opaque pixel, so the filter applies everywhere.
    pub everywhere: bool,
}

impl FilterView {
    pub fn of(l: &Layer, pixels: &Arc<Pixels>) -> Option<FilterView> {
        l.filter_on.then(|| FilterView { filter: l.filter, everywhere: is_blank(pixels) })
    }
}

/// Whether a buffer has no opaque pixel (cached per buffer).
pub fn is_blank(px: &Arc<Pixels>) -> bool {
    type Cache = Mutex<HashMap<usize, (Weak<Pixels>, bool)>>;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = Arc::as_ptr(px) as usize;
    if let Some((w, v)) = cache.lock().unwrap().get(&key) {
        if w.upgrade().is_some_and(|a| Arc::ptr_eq(&a, px)) {
            return *v;
        }
    }
    let blank = px.opaque_bounds().is_none();
    let mut c = cache.lock().unwrap();
    if c.len() > 256 {
        c.retain(|_, (w, _)| w.strong_count() > 0);
    }
    c.insert(key, (Arc::downgrade(px), blank));
    blank
}

impl<'a> LayerView<'a> {
    /// The layer as it is (its own pixels, offset, opacity and filter).
    pub fn of(l: &'a Layer) -> Self {
        LayerView { pixels: &l.pixels, offset: l.offset, opacity: l.opacity, filter: FilterView::of(l, &l.pixels) }
    }

    /// A pixel layer view.
    pub fn pixels(pixels: &'a Pixels, offset: (i32, i32), opacity: f32) -> Self {
        LayerView { pixels, offset, opacity, filter: None }
    }

    pub fn bounds(&self) -> Rect {
        Rect::new(self.offset.0, self.offset.1, self.pixels.width(), self.pixels.height())
    }
}

/// Composites `views` (bottom to top) into an f32 buffer covering `rect`
/// (canvas coordinates), optionally over a solid background.
pub fn composite_views(views: &[LayerView], rect: Rect, background: Option<Px>) -> Buffer<f32> {
    let mut out = match background {
        Some(bg) => Buffer::<f32>::filled(rect.w, rect.h, bg),
        None => Buffer::<f32>::new(rect.w, rect.h),
    };
    if rect.is_empty() {
        return out;
    }
    let row_len = rect.w as usize * 4;
    out.data.par_chunks_mut(row_len).enumerate().for_each(|(ry, row)| {
        let cy = rect.y + ry as i32;
        for v in views {
            if v.opacity <= 0.0 {
                continue;
            }
            if let Some(fv) = v.filter {
                // Applies to everything below, where the layer's pixels are
                // opaque, blended by the layer's opacity.
                let k = v.opacity.min(1.0);
                let lb = v.bounds();
                for (i, px) in row.chunks_exact_mut(4).enumerate() {
                    let x = rect.x + i as i32;
                    let m = if fv.everywhere {
                        1.0
                    } else if lb.contains(x, cy) {
                        v.pixels.get((x - v.offset.0) as u32, (cy - v.offset.1) as u32)[3]
                    } else {
                        0.0
                    };
                    let k = k * m;
                    if k <= 0.0 {
                        continue;
                    }
                    let d = [px[0], px[1], px[2], px[3]];
                    let a = fv.filter.apply_px(d);
                    for c in 0..4 {
                        px[c] = d[c] + (a[c] - d[c]) * k;
                    }
                }
                continue;
            }
            let lb = v.bounds();
            if cy < lb.y || cy >= lb.bottom() {
                continue;
            }
            let x0 = rect.x.max(lb.x);
            let x1 = rect.right().min(lb.right());
            if x1 <= x0 {
                continue;
            }
            let ly = (cy - v.offset.1) as u32;
            dispatch!(v.pixels, b => {
                for cx in x0..x1 {
                    let mut s = b.get((cx - v.offset.0) as u32, ly);
                    if s[3] <= 0.0 {
                        continue;
                    }
                    s[3] *= v.opacity;
                    let i = (cx - rect.x) as usize * 4;
                    let d = [row[i], row[i + 1], row[i + 2], row[i + 3]];
                    let o = over(s, d);
                    row[i..i + 4].copy_from_slice(&o);
                }
            });
        }
    });
    out
}

/// Views of the document's visible layers, honoring live previews.
pub fn document_views(doc: &Document) -> Vec<LayerView<'_>> {
    doc.layers
        .iter()
        .filter(|l| l.visible)
        .map(|l| match doc.preview.get(&l.id) {
            Some(p) => LayerView { pixels: &p.pixels, offset: p.offset, opacity: l.opacity, filter: FilterView::of(l, &p.pixels) },
            None => LayerView::of(l),
        })
        .collect()
}

/// Composites the visible document into `rect` (canvas coordinates).
pub fn composite_document(doc: &Document, rect: Rect) -> Buffer<f32> {
    composite_views(&document_views(doc), rect, None)
}

/// Composites the whole canvas.
pub fn flatten(doc: &Document) -> Buffer<f32> {
    composite_document(doc, Rect::from_size(doc.width, doc.height))
}

/// Average color of the visible document over `rect` (canvas coordinates,
/// clipped to the canvas), averaged in premultiplied space so transparent
/// pixels don't pull the color toward their hidden RGB.
pub fn average_color(doc: &Document, rect: Rect) -> Option<Px> {
    let r = rect.intersect(&doc.canvas_rect());
    if r.is_empty() {
        return None;
    }
    let img = composite_document(doc, r);
    let mut acc = [0.0f64; 4];
    for p in img.data.chunks_exact(4) {
        let a = p[3] as f64;
        acc[0] += p[0] as f64 * a;
        acc[1] += p[1] as f64 * a;
        acc[2] += p[2] as f64 * a;
        acc[3] += a;
    }
    let n = r.area() as f64;
    let a = acc[3] / n;
    Some(if acc[3] <= 1e-12 {
        [0.0; 4]
    } else {
        [(acc[0] / acc[3]) as f32, (acc[1] / acc[3]) as f32, (acc[2] / acc[3]) as f32, a as f32]
    })
}
