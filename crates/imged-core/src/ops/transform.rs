//! Geometric transforms: flips, rotations, resampling, perspective tilt and shear.
//!
//! Layer-level functions take a buffer and return a new buffer plus the
//! offset change. Document-level functions in this module apply them to the
//! active layers or the whole image.

use rayon::prelude::*;

use crate::buffer::Buffer;
use crate::document::Document;
use crate::geom::{Rect, Vec2};
use crate::map_pixels;
use crate::mask::Mask;
use crate::pixel::{premultiply, unpremultiply, Channel};
use crate::pixels::Pixels;
use crate::layer::LayerId;
use std::collections::HashMap;

/// What a transform or filter applies to: the active layer (or group),
/// every layer and the canvas, or the selected part of the active layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Layer,
    Image,
    /// Acts like `Layer` when nothing is selected.
    Selection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeFilter {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl ResizeFilter {
    pub const ALL: [ResizeFilter; 4] =
        [ResizeFilter::Nearest, ResizeFilter::Bilinear, ResizeFilter::Bicubic, ResizeFilter::Lanczos3];

    pub fn label(self) -> &'static str {
        match self {
            ResizeFilter::Nearest => "Nearest (pixel art)",
            ResizeFilter::Bilinear => "Bilinear",
            ResizeFilter::Bicubic => "Bicubic",
            ResizeFilter::Lanczos3 => "Lanczos",
        }
    }

    fn support(self) -> f32 {
        match self {
            ResizeFilter::Nearest => 0.5,
            ResizeFilter::Bilinear => 1.0,
            ResizeFilter::Bicubic => 2.0,
            ResizeFilter::Lanczos3 => 3.0,
        }
    }

    fn weight(self, x: f32) -> f32 {
        let x = x.abs();
        match self {
            ResizeFilter::Nearest => {
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            ResizeFilter::Bilinear => (1.0 - x).max(0.0),
            ResizeFilter::Bicubic => {
                // Catmull-Rom (a = -0.5)
                let a = -0.5;
                if x < 1.0 {
                    (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
                } else if x < 2.0 {
                    a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
                } else {
                    0.0
                }
            }
            ResizeFilter::Lanczos3 => {
                if x < 1e-6 {
                    1.0
                } else if x < 3.0 {
                    let px = std::f32::consts::PI * x;
                    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
                } else {
                    0.0
                }
            }
        }
    }
}

// ----- simple layer transforms ------------------------------------------

pub fn flip_h<T: Channel>(b: &Buffer<T>) -> Buffer<T> {
    let w = b.width;
    Buffer::from_fn(b.width, b.height, |x, y| b.get(w - 1 - x, y))
}

pub fn flip_v<T: Channel>(b: &Buffer<T>) -> Buffer<T> {
    let h = b.height;
    Buffer::from_fn(b.width, b.height, |x, y| b.get(x, h - 1 - y))
}

pub fn rotate90<T: Channel>(b: &Buffer<T>, clockwise: bool) -> Buffer<T> {
    let (w, h) = (b.width, b.height);
    if clockwise {
        Buffer::from_fn(h, w, |x, y| b.get(y, h - 1 - x))
    } else {
        Buffer::from_fn(h, w, |x, y| b.get(w - 1 - y, x))
    }
}

pub fn rotate180<T: Channel>(b: &Buffer<T>) -> Buffer<T> {
    let (w, h) = (b.width, b.height);
    Buffer::from_fn(w, h, |x, y| b.get(w - 1 - x, h - 1 - y))
}

/// Builds `out_rect` (in the same coordinate frame as `inverse`'s output)
/// by mapping each output pixel center back into `src` coordinates.
pub fn warp_inverse<T: Channel>(
    src: &Buffer<T>,
    out_rect: Rect,
    smooth: bool,
    inverse: impl Fn(f32, f32) -> Option<(f32, f32)> + Sync,
) -> Buffer<T> {
    Buffer::from_fn(out_rect.w, out_rect.h, |x, y| {
        let (cx, cy) = (out_rect.x as f32 + x as f32 + 0.5, out_rect.y as f32 + y as f32 + 0.5);
        match inverse(cx, cy) {
            Some((sx, sy)) if smooth => src.sample_bilinear(sx, sy),
            Some((sx, sy)) => src.sample_nearest(sx, sy),
            None => [0.0; 4],
        }
    })
}

/// Rotates by `degrees` (counter-clockwise on screen) around the buffer's
/// center. Returns the expanded buffer and the shift of its top-left corner.
pub fn rotate<T: Channel>(b: &Buffer<T>, degrees: f32, smooth: bool) -> (Buffer<T>, (i32, i32)) {
    let (w, h) = (b.width as f32, b.height as f32);
    let c = Vec2::new(w / 2.0, h / 2.0);
    // Screen y points down, so a visually counter-clockwise rotation is -angle.
    let a = -degrees.to_radians();
    let corners = [Vec2::new(0.0, 0.0), Vec2::new(w, 0.0), Vec2::new(w, h), Vec2::new(0.0, h)]
        .map(|p| c + (p - c).rotate(a));
    let out = bounds_of(&corners);
    let warped = warp_inverse(b, out, smooth, |x, y| {
        let p = c + (Vec2::new(x, y) - c).rotate(-a);
        Some((p.x, p.y))
    });
    (warped, (out.x, out.y))
}

fn bounds_of(points: &[Vec2]) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in points {
        x0 = x0.min(p.x);
        y0 = y0.min(p.y);
        x1 = x1.max(p.x);
        y1 = y1.max(p.y);
    }
    // Snap away from tiny float noise so axis-aligned results stay exact.
    let r = |v: f32| (v * 1000.0).round() / 1000.0;
    Rect::covering(r(x0), r(y0), r(x1), r(y1))
}

/// Resamples to `nw` x `nh` with a separable filter in premultiplied space.
pub fn resize<T: Channel>(b: &Buffer<T>, nw: u32, nh: u32, filter: ResizeFilter) -> Buffer<T> {
    let (nw, nh) = (nw.max(1), nh.max(1));
    if b.width == 0 || b.height == 0 {
        return Buffer::new(nw, nh);
    }
    if filter == ResizeFilter::Nearest {
        let (sx, sy) = (b.width as f32 / nw as f32, b.height as f32 / nh as f32);
        return Buffer::from_fn(nw, nh, |x, y| {
            let px = (((x as f32 + 0.5) * sx) as u32).min(b.width - 1);
            let py = (((y as f32 + 0.5) * sy) as u32).min(b.height - 1);
            b.get(px, py)
        });
    }
    if let Some(a) = crate::accel::get() {
        let pack = |ws: Vec<(usize, Vec<f32>)>| {
            let mut aw = crate::accel::AxisWeights { starts: Vec::new(), offsets: vec![0], weights: Vec::new() };
            for (lo, w) in ws {
                aw.starts.push(lo as u32);
                aw.weights.extend(w);
                aw.offsets.push(aw.weights.len() as u32);
            }
            aw
        };
        let horiz = pack(axis_weights(b.width as usize, nw as usize, filter));
        let vert = pack(axis_weights(b.height as usize, nh as usize, filter));
        let out = a
            .resize(b.width, b.height, crate::accel::data_of(&b.data), &horiz, &vert)
            .and_then(crate::accel::out_as::<T>);
        if let Some(data) = out.filter(|d| d.len() == nw as usize * nh as usize * 4) {
            return Buffer { width: nw, height: nh, data };
        }
    }
    resize_cpu(b, nw, nh, filter)
}

/// [`resize`] on the CPU (the reference for accelerators).
pub fn resize_cpu<T: Channel>(b: &Buffer<T>, nw: u32, nh: u32, filter: ResizeFilter) -> Buffer<T> {
    let (nw, nh) = (nw.max(1), nh.max(1));
    if b.width == 0 || b.height == 0 {
        return Buffer::new(nw, nh);
    }
    let src: Vec<[f32; 4]> = b
        .data
        .par_chunks(4)
        .map(|d| premultiply([d[0].to_f32(), d[1].to_f32(), d[2].to_f32(), d[3].to_f32()]))
        .collect();
    let horiz = resample_axis(&src, b.width as usize, b.height as usize, nw as usize, true, filter);
    let both = resample_axis(&horiz, nw as usize, b.height as usize, nh as usize, false, filter);
    let mut out = Buffer::<T>::new(nw, nh);
    out.data.par_chunks_mut(4).zip(both.par_iter()).for_each(|(d, p)| {
        let mut p = *p;
        p[3] = p[3].clamp(0.0, 1.0);
        let u = unpremultiply(p);
        for c in 0..4 {
            d[c] = T::from_f32(u[c]);
        }
    });
    out
}

/// Weights for each output coordinate: (first source index, weights).
pub fn axis_weights(src_len: usize, dst_len: usize, filter: ResizeFilter) -> Vec<(usize, Vec<f32>)> {
    let scale = src_len as f32 / dst_len as f32;
    let fscale = scale.max(1.0); // widen the filter when downscaling
    let support = filter.support() * fscale;
    (0..dst_len)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let lo = ((center - support).floor() as isize).max(0) as usize;
            let hi = ((center + support).ceil() as usize).min(src_len);
            let mut w: Vec<f32> = (lo..hi)
                .map(|j| filter.weight((j as f32 + 0.5 - center) / fscale))
                .collect();
            let s: f32 = w.iter().sum();
            if s.abs() > 1e-9 {
                w.iter_mut().for_each(|v| *v /= s);
            }
            (lo, w)
        })
        .collect()
}

fn resample_axis(
    src: &[[f32; 4]],
    w: usize,
    h: usize,
    new_len: usize,
    horizontal: bool,
    filter: ResizeFilter,
) -> Vec<[f32; 4]> {
    if horizontal {
        let weights = axis_weights(w, new_len, filter);
        let mut out = vec![[0.0f32; 4]; new_len * h];
        out.par_chunks_mut(new_len).enumerate().for_each(|(y, row)| {
            for (x, (lo, ws)) in weights.iter().enumerate() {
                let mut acc = [0.0f32; 4];
                for (k, wt) in ws.iter().enumerate() {
                    let p = src[y * w + lo + k];
                    for c in 0..4 {
                        acc[c] += p[c] * wt;
                    }
                }
                row[x] = acc;
            }
        });
        out
    } else {
        let weights = axis_weights(h, new_len, filter);
        let mut out = vec![[0.0f32; 4]; w * new_len];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let (lo, ws) = &weights[y];
            for (x, cell) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 4];
                for (k, wt) in ws.iter().enumerate() {
                    let p = src[(lo + k) * w + x];
                    for c in 0..4 {
                        acc[c] += p[c] * wt;
                    }
                }
                *cell = acc;
            }
        });
        out
    }
}

// ----- homography --------------------------------------------------------

/// 3x3 projective transform, row-major.
#[derive(Clone, Copy, Debug)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    /// Transform mapping `from[i]` to `to[i]` for four point pairs.
    pub fn from_points(from: [Vec2; 4], to: [Vec2; 4]) -> Option<Homography> {
        // Solve A h = b for h = (h0..h7), h8 = 1.
        let mut a = [[0.0f64; 9]; 8];
        for i in 0..4 {
            let (x, y) = (from[i].x as f64, from[i].y as f64);
            let (u, v) = (to[i].x as f64, to[i].y as f64);
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        // Gaussian elimination with partial pivoting.
        for col in 0..8 {
            let piv = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
            if a[piv][col].abs() < 1e-12 {
                return None;
            }
            a.swap(col, piv);
            for row in 0..8 {
                if row != col {
                    let f = a[row][col] / a[col][col];
                    for k in col..9 {
                        a[row][k] -= f * a[col][k];
                    }
                }
            }
        }
        let mut h = [0.0f64; 9];
        for i in 0..8 {
            h[i] = a[i][8] / a[i][i];
        }
        h[8] = 1.0;
        Some(Homography(h))
    }

    pub fn apply(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let h = &self.0;
        let (x, y) = (x as f64, y as f64);
        let w = h[6] * x + h[7] * y + h[8];
        if w.abs() < 1e-12 {
            return None;
        }
        Some((((h[0] * x + h[1] * y + h[2]) / w) as f32, ((h[3] * x + h[4] * y + h[5]) / w) as f32))
    }
}

/// Warps the whole buffer so its corners (TL, TR, BR, BL) land on `quad`
/// (in the buffer's own coordinate frame). Returns the new buffer and the
/// offset of its top-left corner relative to the old one.
pub fn warp_quad<T: Channel>(b: &Buffer<T>, quad: [Vec2; 4]) -> Option<(Buffer<T>, (i32, i32))> {
    let (w, h) = (b.width as f32, b.height as f32);
    let src = [Vec2::new(0.0, 0.0), Vec2::new(w, 0.0), Vec2::new(w, h), Vec2::new(0.0, h)];
    let inv = Homography::from_points(quad, src)?;
    let out = bounds_of(&quad);
    if out.is_empty() || out.area() > 400_000_000 {
        return None;
    }
    let warped = warp_inverse(b, out, true, |x, y| {
        let (sx, sy) = inv.apply(x, y)?;
        // Reject the mirrored solution behind the projection center.
        (sx > -1.0 && sy > -1.0 && sx < w + 1.0 && sy < h + 1.0).then_some((sx, sy))
    });
    Some((warped, (out.x, out.y)))
}

/// An edge of a rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    Top,
    Right,
    Bottom,
    Left,
}

impl Edge {
    pub const ALL: [Edge; 4] = [Edge::Top, Edge::Right, Edge::Bottom, Edge::Left];

    pub fn label(self) -> &'static str {
        match self {
            Edge::Top => "Top",
            Edge::Right => "Right",
            Edge::Bottom => "Bottom",
            Edge::Left => "Left",
        }
    }

    /// Indices of this edge's two corners in TL, TR, BR, BL order.
    pub fn corners(self) -> (usize, usize) {
        match self {
            Edge::Top => (0, 1),
            Edge::Right => (1, 2),
            Edge::Bottom => (2, 3),
            Edge::Left => (3, 0),
        }
    }

    pub fn opposite(self) -> Edge {
        match self {
            Edge::Top => Edge::Bottom,
            Edge::Bottom => Edge::Top,
            Edge::Left => Edge::Right,
            Edge::Right => Edge::Left,
        }
    }

    /// Unit vector pointing from the opposite edge toward this one.
    pub fn outward(self) -> Vec2 {
        match self {
            Edge::Top => Vec2::new(0.0, -1.0),
            Edge::Bottom => Vec2::new(0.0, 1.0),
            Edge::Left => Vec2::new(-1.0, 0.0),
            Edge::Right => Vec2::new(1.0, 0.0),
        }
    }

    /// Unit vector along the edge.
    pub fn along(self) -> Vec2 {
        self.outward().perp()
    }
}

pub fn rect_corners(w: f32, h: f32) -> [Vec2; 4] {
    [Vec2::new(0.0, 0.0), Vec2::new(w, 0.0), Vec2::new(w, h), Vec2::new(0.0, h)]
}

/// Corners after tilting a `w` x `h` image in 3D: it hinges on the edge
/// opposite `edge`, and `edge` swings away from the viewer by `degrees`
/// (negative swings it toward the viewer). `focal` is the camera distance in
/// pixels; smaller means stronger perspective.
pub fn tilt_quad(w: f32, h: f32, edge: Edge, degrees: f32, focal: f32) -> [Vec2; 4] {
    let corners = rect_corners(w, h);
    let center = Vec2::new(w / 2.0, h / 2.0);
    let n = edge.outward();
    let hinge = corners[edge.opposite().corners().0];
    let theta = degrees.to_radians();
    let f = focal.max(1.0);
    corners.map(|p| {
        let d = (p - hinge).dot(n); // distance from the hinge line
        let along = p - n * d; // projection onto the hinge line
        let rotated = along + n * (d * theta.cos());
        let z = d * theta.sin();
        let s = f / (f + z).max(f * 0.05);
        center + (rotated - center) * s
    })
}

/// Finds the tilt angle whose projected `edge` midpoint lands on `target`
/// (buffer coords), for interactive dragging. Dragging the edge inward
/// tilts it away from the viewer; dragging it outward tilts it toward them.
pub fn tilt_angle_for(w: f32, h: f32, edge: Edge, focal: f32, target: Vec2) -> f32 {
    let (a, b) = edge.corners();
    let n = edge.outward();
    // Signed position of the projected edge along its outward normal, relative to the target.
    let offset = |deg: f32| {
        let q = tilt_quad(w, h, edge, deg, focal);
        ((q[a] + q[b]) * 0.5 - target).dot(n)
    };
    let start = offset(0.0);
    if start.abs() < 1e-3 {
        return 0.0;
    }
    // Inward drag (edge is outside the target) → positive angles.
    let dir: f32 = if start > 0.0 { 1.0 } else { -1.0 };
    let step: f32 = 0.25 * dir;
    let mut prev = (0.0f32, start);
    let mut best = prev;
    let mut deg = step;
    while deg.abs() <= 80.0 {
        let o = offset(deg);
        if o.abs() < best.1.abs() {
            best = (deg, o);
        }
        if o.signum() != prev.1.signum() {
            // Bracketed the first crossing: bisect it.
            let (mut lo, mut hi) = (prev.0, deg);
            for _ in 0..30 {
                let mid = (lo + hi) / 2.0;
                if offset(mid).signum() == prev.1.signum() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return (lo + hi) / 2.0;
        }
        prev = (deg, o);
        deg += step;
    }
    best.0
}

/// Corners after sliding `edge` along itself by `amount` pixels while the
/// opposite edge stays fixed (a parallelogram).
pub fn shear_quad(w: f32, h: f32, edge: Edge, amount: f32) -> [Vec2; 4] {
    let mut q = rect_corners(w, h);
    let (a, b) = edge.corners();
    let d = edge.along() * amount;
    q[a] = q[a] + d;
    q[b] = q[b] + d;
    q
}

pub fn warp_pixels(px: &Pixels, quad: [Vec2; 4]) -> Option<(Pixels, (i32, i32))> {
    match px {
        Pixels::U8(b) => warp_quad(b, quad).map(|(o, d)| (Pixels::U8(o), d)),
        Pixels::U16(b) => warp_quad(b, quad).map(|(o, d)| (Pixels::U16(o), d)),
        Pixels::F32(b) => warp_quad(b, quad).map(|(o, d)| (Pixels::F32(o), d)),
    }
}

// ----- document-level ----------------------------------------------------

/// The scope actually used: `Selection` without a selection acts like `Layer`.
pub fn effective_scope(doc: &Document, scope: Scope) -> Scope {
    if scope == Scope::Selection && doc.selection.is_none() {
        Scope::Layer
    } else {
        scope
    }
}

/// Layers a transform with this scope changes (filter layers too: their
/// pixels are the mask).
pub fn scope_layers(doc: &Document, scope: Scope) -> Vec<LayerId> {
    let ids = match effective_scope(doc, scope) {
        Scope::Layer | Scope::Selection => doc.targets(),
        Scope::Image => doc.layers.iter().map(|l| l.id).collect(),
    };
    ids.into_iter().filter(|id| doc.layer(*id).is_some()).collect()
}

/// The rectangle a transform with this scope is relative to (what its
/// corners, center and anchors refer to): the active layer/group's bounds,
/// the canvas, or the selection's bounds.
pub fn scope_pivot(doc: &Document, scope: Scope) -> Rect {
    match effective_scope(doc, scope) {
        Scope::Layer => scope_layers(doc, Scope::Layer)
            .iter()
            .filter_map(|id| doc.layer(*id))
            .fold(Rect::default(), |a, l| a.union(&l.bounds())),
        Scope::Image => doc.canvas_rect(),
        Scope::Selection => doc.selection.as_ref().and_then(|m| m.bounds()).unwrap_or_default(),
    }
}

/// The result of a transform, computed without touching the document, so it
/// can be shown as a preview and applied later.
pub struct Plan {
    pub label: &'static str,
    /// New pixels and offset per changed layer (current canvas coordinates).
    pub layers: Vec<(LayerId, Pixels, (i32, i32))>,
    /// New canvas rectangle in current canvas coordinates, if it changes.
    pub canvas: Option<Rect>,
    /// The transformed selection (selection scope).
    pub selection: Option<Mask>,
}

impl Plan {
    /// Shows the plan as a live preview.
    pub fn preview(&self, doc: &mut Document) {
        doc.clear_preview();
        for (id, px, off) in &self.layers {
            doc.set_preview(*id, std::sync::Arc::new(px.clone()), *off);
        }
    }

    /// Applies the plan as one undo step.
    pub fn apply(self, doc: &mut Document) -> bool {
        if self.layers.is_empty() && self.canvas.is_none() && self.selection.is_none() {
            return false;
        }
        doc.clear_preview();
        if self.canvas.is_none() && self.selection.is_none() {
            let ids: Vec<LayerId> = self.layers.iter().map(|r| r.0).collect();
            let results = self.layers;
            return doc.edit_layers(&ids, self.label, |l| {
                results.iter().find(|r| r.0 == l.id).map(|(_, px, off)| (px.clone(), *off))
            });
        }
        let mut results: HashMap<LayerId, (Pixels, (i32, i32))> =
            self.layers.into_iter().map(|(id, px, off)| (id, (px, off))).collect();
        let canvas = self.canvas;
        let selection = self.selection;
        doc.structural(self.label, |d| {
            for l in &mut d.layers {
                if let Some((px, off)) = results.remove(&l.id) {
                    l.pixels = std::sync::Arc::new(px);
                    l.offset = off;
                }
            }
            if let Some(m) = selection {
                d.selection = Some(std::sync::Arc::new(m));
            }
            if let Some(c) = canvas {
                for l in &mut d.layers {
                    l.offset = (l.offset.0 - c.x, l.offset.1 - c.y);
                }
                d.width = c.w;
                d.height = c.h;
            }
            true
        })
    }
}

/// Crops a layer to its painted pixels (keeps it as is when fully transparent).
pub fn trimmed(px: Pixels, off: (i32, i32)) -> (Pixels, (i32, i32)) {
    match px.opaque_bounds() {
        Some(b) if b != px.rect() => (px.crop(b), (off.0 + b.x, off.1 + b.y)),
        _ => (px, off),
    }
}

/// `px` (at `off`) with every pixel's alpha multiplied by `f(mask value)`.
fn masked(px: &Pixels, off: (i32, i32), mask: &Mask, f: impl Fn(f32) -> f32) -> Pixels {
    map_pixels!(px, b => mask_buffer(b, off, mask, &f))
}

fn mask_buffer<T: Channel>(b: &Buffer<T>, off: (i32, i32), mask: &Mask, f: &impl Fn(f32) -> f32) -> Buffer<T> {
    let mut out = b.clone();
    for y in 0..b.height {
        for x in 0..b.width {
            let k = f(mask.get(x as i32 + off.0, y as i32 + off.1));
            if k < 1.0 {
                let i = (y as usize * b.width as usize + x as usize) * 4 + 3;
                out.data[i] = T::from_f32(out.data[i].to_f32() * k.max(0.0));
            }
        }
    }
    out
}

/// A selection mask as a pixel buffer (coverage in alpha), for transforming
/// it like a layer.
fn mask_pixels(m: &Mask) -> (Pixels, (i32, i32)) {
    let b = Buffer::<f32>::from_fn(m.rect.w, m.rect.h, |x, y| {
        [1.0, 1.0, 1.0, m.get(m.rect.x + x as i32, m.rect.y + y as i32)]
    });
    (Pixels::F32(b), (m.rect.x, m.rect.y))
}

fn pixels_mask(px: &Pixels, off: (i32, i32)) -> Mask {
    let r = Rect::new(off.0, off.1, px.width(), px.height());
    let mut m = Mask::new(r);
    for y in 0..r.h {
        for x in 0..r.w {
            let a = px.get(x, y)[3];
            if a > 0.0 {
                m.set(r.x + x as i32, r.y + y as i32, a.min(1.0));
            }
        }
    }
    m
}

/// Plans a per-piece transform. `f` maps (pixels, offset, pivot) to new
/// pixels and offset, in canvas coordinates.
/// * Layer scope: each pixel layer of the active layer/group is transformed
///   relative to their combined bounds.
/// * Image scope: every pixel layer, relative to the canvas (which stays).
/// * Selection scope: only the selected part of the active layer/group is
///   transformed (relative to the selection's bounds) and laid over the
///   rest; the selection is transformed with it.
fn plan_pieces(
    doc: &Document,
    scope: Scope,
    label: &'static str,
    trim: bool,
    f: impl Fn(&Pixels, (i32, i32), Rect) -> Option<(Pixels, (i32, i32))>,
) -> Option<Plan> {
    let scope = effective_scope(doc, scope);
    let pivot = scope_pivot(doc, scope);
    if pivot.is_empty() {
        return None;
    }
    let ids = scope_layers(doc, scope);
    let mut layers = Vec::new();
    let mut selection = None;
    match scope {
        Scope::Layer | Scope::Image => {
            for id in ids {
                let l = doc.layer(id)?;
                let (px, off) = f(&l.pixels, l.offset, pivot)?;
                let (px, off) = if trim { trimmed(px, off) } else { (px, off) };
                layers.push((id, px, off));
            }
        }
        Scope::Selection => {
            let mask = doc.selection.clone()?;
            for id in ids {
                let l = doc.layer(id)?;
                let inside = masked(&l.pixels, l.offset, &mask, |m| m);
                let Some(ib) = inside.opaque_bounds() else { continue };
                let piece = inside.crop(ib);
                let poff = (l.offset.0 + ib.x, l.offset.1 + ib.y);
                let (tp, toff) = f(&piece, poff, pivot)?;
                let rest = masked(&l.pixels, l.offset, &mask, |m| 1.0 - m);
                let views = [
                    crate::composite::LayerView::pixels(&rest, l.offset, 1.0),
                    crate::composite::LayerView::pixels(&tp, toff, 1.0),
                ];
                let area = l.bounds().union(&Rect::new(toff.0, toff.1, tp.width(), tp.height()));
                let merged = Pixels::from_f32(&crate::composite::composite_views(&views, area, None), l.depth());
                let (px, off) = trimmed(merged, (area.x, area.y));
                layers.push((id, px, off));
            }
            let (mp, moff) = mask_pixels(&mask);
            if let Some((tp, toff)) = f(&mp, moff, pivot) {
                selection = Some(pixels_mask(&tp, toff));
            }
        }
    }
    Some(Plan { label, layers, canvas: None, selection })
}

pub fn flip(doc: &mut Document, horizontal: bool, scope: Scope) -> bool {
    let label = if horizontal { "Flip horizontal" } else { "Flip vertical" };
    let f = |p: &Pixels| if horizontal { map_pixels!(p, b => flip_h(b)) } else { map_pixels!(p, b => flip_v(b)) };
    match effective_scope(doc, scope) {
        Scope::Image => {
            let (w, h) = (doc.width as i32, doc.height as i32);
            doc.transform_all(label, (doc.width, doc.height), |l| {
                let b = l.bounds();
                let off = if horizontal { (w - b.right(), b.y) } else { (b.x, h - b.bottom()) };
                (f(&l.pixels), off)
            })
        }
        s => plan_pieces(doc, s, label, false, |px, off, pv| {
            let (w, h) = (px.width() as i32, px.height() as i32);
            let o = if horizontal { (2 * pv.x + pv.w as i32 - off.0 - w, off.1) } else { (off.0, 2 * pv.y + pv.h as i32 - off.1 - h) };
            Some((f(px), o))
        })
        .is_some_and(|p| p.apply(doc)),
    }
}

/// Rotates by a multiple of 90° (`quarter_turns_cw` in 1..=3).
pub fn rotate_quarter(doc: &mut Document, quarter_turns_cw: u32, scope: Scope) -> bool {
    let q = quarter_turns_cw % 4;
    if q == 0 {
        return false;
    }
    let label = match q {
        1 => "Rotate 90° clockwise",
        2 => "Rotate 180°",
        _ => "Rotate 90° counter-clockwise",
    };
    let f = move |p: &Pixels| match q {
        1 => map_pixels!(p, b => rotate90(b, true)),
        2 => map_pixels!(p, b => rotate180(b)),
        _ => map_pixels!(p, b => rotate90(b, false)),
    };
    match effective_scope(doc, scope) {
        Scope::Image => {
            let (w, h) = (doc.width as i32, doc.height as i32);
            let size = if q == 2 { (doc.width, doc.height) } else { (doc.height, doc.width) };
            doc.transform_all(label, size, |l| {
                let b = l.bounds();
                let off = match q {
                    1 => (h - b.bottom(), b.x),
                    2 => (w - b.right(), h - b.bottom()),
                    _ => (b.y, w - b.right()),
                };
                (f(&l.pixels), off)
            })
        }
        s => plan_pieces(doc, s, label, false, |px, off, pv| {
            // Around the pivot's center, in doubled coordinates.
            let (cx2, cy2) = (2 * pv.x + pv.w as i32, 2 * pv.y + pv.h as i32);
            let (x0, y0) = off;
            let (x1, y1) = (x0 + px.width() as i32, y0 + px.height() as i32);
            let o = match q {
                1 => ((cx2 + cy2).div_euclid(2) - y1, x0 + (cy2 - cx2).div_euclid(2)),
                2 => (cx2 - x1, cy2 - y1),
                _ => (y0 + (cx2 - cy2).div_euclid(2), (cx2 + cy2).div_euclid(2) - x1),
            };
            Some((f(px), o))
        })
        .is_some_and(|p| p.apply(doc)),
    }
}

/// Rotation by `degrees` (counter-clockwise) around `center` (canvas
/// coords; `None` = the middle of the scope's pivot rectangle). For the
/// whole image the canvas grows to fit (the center then only shifts the
/// result, so it is ignored). Rotated layers are trimmed to their painted
/// pixels.
pub fn plan_rotate(doc: &Document, degrees: f32, smooth: bool, scope: Scope, center: Option<Vec2>) -> Option<Plan> {
    if degrees.rem_euclid(360.0).abs() < 1e-4 {
        return None;
    }
    let a = -degrees.to_radians();
    let rotate_piece = |px: &Pixels, off: (i32, i32), c: Vec2| {
        let (ox, oy) = (off.0 as f32, off.1 as f32);
        let rot = |p: Vec2| c + (p - c).rotate(a);
        let out = bounds_of(&rect_corners(px.width() as f32, px.height() as f32).map(|p| rot(p + Vec2::new(ox, oy))));
        let inv = |x: f32, y: f32| {
            let p = c + (Vec2::new(x, y) - c).rotate(-a);
            Some((p.x - ox, p.y - oy))
        };
        let px = map_pixels!(px, b => warp_inverse(b, out, smooth, inv));
        trimmed(px, (out.x, out.y))
    };
    if effective_scope(doc, scope) == Scope::Image {
        let r = doc.canvas_rect();
        let c = Vec2::new(r.w as f32 / 2.0, r.h as f32 / 2.0);
        let canvas = bounds_of(&rect_corners(r.w as f32, r.h as f32).map(|p| c + (p - c).rotate(a)));
        let layers = scope_layers(doc, scope)
            .iter()
            .filter_map(|id| doc.layer(*id))
            .map(|l| {
                let (px, off) = rotate_piece(&l.pixels, l.offset, c);
                (l.id, px, off)
            })
            .collect();
        return Some(Plan { label: "Rotate image", layers, canvas: Some(canvas), selection: None });
    }
    plan_pieces(doc, scope, "Rotate", true, |px, off, pv| Some(rotate_piece(px, off, center.unwrap_or_else(|| rect_center(pv)))))
}

/// Scales a piece so the rectangle `from` maps onto `to`.
fn scale_piece(px: &Pixels, off: (i32, i32), from: Rect, to: Rect, filter: ResizeFilter) -> (Pixels, (i32, i32)) {
    let sx = to.w as f64 / from.w.max(1) as f64;
    let sy = to.h as f64 / from.h.max(1) as f64;
    let b = Rect::new(off.0, off.1, px.width(), px.height());
    let x0 = to.x + ((b.x - from.x) as f64 * sx).round() as i32;
    let y0 = to.y + ((b.y - from.y) as f64 * sy).round() as i32;
    let x1 = to.x + ((b.right() - from.x) as f64 * sx).round() as i32;
    let y1 = to.y + ((b.bottom() - from.y) as f64 * sy).round() as i32;
    let (w, h) = ((x1 - x0).max(1) as u32, (y1 - y0).max(1) as u32);
    (map_pixels!(px, p => resize(p, w, h, filter)), (x0, y0))
}

/// Resize to `nw` × `nh`: the active layer/group or the selection (keeping
/// its top-left), or the whole image with its canvas.
pub fn plan_resize(doc: &Document, nw: u32, nh: u32, filter: ResizeFilter, scope: Scope) -> Option<Plan> {
    if nw == 0 || nh == 0 {
        return None;
    }
    let from = scope_pivot(doc, scope);
    if from.is_empty() || (from.w, from.h) == (nw, nh) {
        return None;
    }
    plan_resize_to(doc, Rect::new(from.x, from.y, nw, nh), filter, scope)
}

/// Scales so the scope's pivot rectangle becomes `to` (canvas coords). For
/// the whole image, the canvas is resized to `to`'s size.
pub fn plan_resize_to(doc: &Document, to: Rect, filter: ResizeFilter, scope: Scope) -> Option<Plan> {
    let from = scope_pivot(doc, scope);
    if from.is_empty() || to.is_empty() || from == to {
        return None;
    }
    if effective_scope(doc, scope) == Scope::Image {
        let to = Rect::new(0, 0, to.w, to.h);
        if (from.w, from.h) == (to.w, to.h) {
            return None;
        }
        let layers = scope_layers(doc, scope)
            .iter()
            .filter_map(|id| doc.layer(*id))
            .map(|l| {
                let (px, off) = scale_piece(&l.pixels, l.offset, from, to, filter);
                (l.id, px, off)
            })
            .collect();
        return Some(Plan { label: "Resize image", layers, canvas: Some(to), selection: None });
    }
    plan_pieces(doc, scope, "Resize", false, |px, off, pv| Some(scale_piece(px, off, pv, to, filter)))
}

/// Warps the scope's pivot rectangle onto `quad` (canvas coords, corners
/// TL, TR, BR, BL); results are trimmed.
pub fn plan_warp(doc: &Document, label: &'static str, quad: [Vec2; 4], scope: Scope) -> Option<Plan> {
    let pivot = scope_pivot(doc, scope);
    if pivot.is_empty() {
        return None;
    }
    let src = rect_corners(pivot.w as f32, pivot.h as f32).map(|p| p + Vec2::new(pivot.x as f32, pivot.y as f32));
    let fwd = Homography::from_points(src, quad)?;
    plan_pieces(doc, scope, label, true, |px, off, _| {
        // Each piece goes through the same canvas-space homography.
        let (lx, ly) = (off.0 as f32, off.1 as f32);
        let lq = rect_corners(px.width() as f32, px.height() as f32).map(|p| {
            let (x, y) = fwd.apply(p.x + lx, p.y + ly).unwrap_or((p.x + lx, p.y + ly));
            Vec2::new(x - lx, y - ly)
        });
        let (out, d) = warp_pixels(px, lq)?;
        Some((out, (off.0 + d.0, off.1 + d.1)))
    })
}

pub fn rotate_free(doc: &mut Document, degrees: f32, smooth: bool, scope: Scope, center: Option<Vec2>) -> bool {
    plan_rotate(doc, degrees, smooth, scope, center).is_some_and(|p| p.apply(doc))
}

/// The middle of a rectangle.
pub fn rect_center(r: Rect) -> Vec2 {
    Vec2::new(r.x as f32 + r.w as f32 / 2.0, r.y as f32 + r.h as f32 / 2.0)
}

pub fn resize_doc(doc: &mut Document, nw: u32, nh: u32, filter: ResizeFilter, scope: Scope) -> bool {
    plan_resize(doc, nw, nh, filter, scope).is_some_and(|p| p.apply(doc))
}

/// Applies a quad warp (tilt/shear/free transform).
pub fn warp_targets(doc: &mut Document, label: &'static str, quad: [Vec2; 4], scope: Scope) -> bool {
    plan_warp(doc, label, quad, scope).is_some_and(|p| p.apply(doc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(w: u32, h: u32) -> Buffer<f32> {
        Buffer::from_fn(w, h, |x, y| [x as f32, y as f32, 0.0, 1.0])
    }

    #[test]
    fn rot90_roundtrip() {
        let b = numbered(3, 2);
        let r = rotate90(&b, true);
        assert_eq!((r.width, r.height), (2, 3));
        // Top-left after CW rotation is the old bottom-left.
        assert_eq!(r.get(0, 0), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(rotate90(&r, false), b);
    }

    #[test]
    fn resize_identity_and_scale() {
        let b = Buffer::<u8>::filled(4, 4, [0.5, 0.25, 1.0, 1.0]);
        for f in ResizeFilter::ALL {
            let r = resize(&b, 8, 2, f);
            assert_eq!((r.width, r.height), (8, 2));
            let p = r.get(3, 1);
            assert!((p[0] - 0.5).abs() < 0.01, "{f:?} {p:?}");
        }
    }

    #[test]
    fn homography_identity() {
        let q = rect_corners(10.0, 5.0);
        let h = Homography::from_points(q, q).unwrap();
        let (x, y) = h.apply(3.0, 4.0).unwrap();
        assert!((x - 3.0).abs() < 1e-4 && (y - 4.0).abs() < 1e-4);
    }

    #[test]
    fn tilt_shrinks_edge() {
        let q = tilt_quad(100.0, 100.0, Edge::Top, 30.0, 300.0);
        // Hinge (bottom) corners stay put… approximately (projection center is the middle).
        let top_w = q[1].x - q[0].x;
        let bot_w = q[2].x - q[3].x;
        assert!(top_w < bot_w);
        let a = tilt_angle_for(100.0, 100.0, Edge::Top, 300.0, (q[0] + q[1]) * 0.5);
        assert!((a - 30.0).abs() < 0.1, "{a}");
        // Dragging outward tilts toward the viewer (negative angle, wider edge).
        let a = tilt_angle_for(100.0, 100.0, Edge::Top, 300.0, Vec2::new(50.0, -10.0));
        assert!(a < 0.0, "{a}");
        let q = tilt_quad(100.0, 100.0, Edge::Top, a, 300.0);
        assert!(q[1].x - q[0].x > 100.0);
    }

    #[test]
    fn rotate_trims_and_previews() {
        let mut d = Document::new(40, 40, crate::pixel::Depth::U8, None);
        let id = d.add_layer_from("sq", Pixels::filled(crate::pixel::Depth::U8, 10, 10, [1.0; 4]), (10, 10));
        d.set_active(id);
        // 90° via the free path: no transparent margins remain.
        let plan = plan_rotate(&d, 90.0, false, Scope::Layer, None).unwrap();
        plan.preview(&mut d);
        assert!(d.preview.contains_key(&id));
        assert!(plan_rotate(&d, 90.0, false, Scope::Layer, None).unwrap().apply(&mut d));
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(10, 10, 10, 10));
        // 45°: trimmed to the diamond's bounds, centered on the old center.
        assert!(rotate_free(&mut d, 45.0, true, Scope::Layer, None));
        let b = d.layer(id).unwrap().bounds();
        assert!(b.w <= 16 && b.w >= 13, "{b:?}");
        assert!(((b.x as f32 + b.w as f32 / 2.0) - 15.0).abs() <= 1.0, "{b:?}");
    }

    #[test]
    fn resize_group_scales_bounds() {
        let mut d = Document::new(40, 40, crate::pixel::Depth::U8, None);
        let a = d.add_layer_from("a", Pixels::filled(crate::pixel::Depth::U8, 10, 10, [1.0; 4]), (0, 0));
        let b = d.add_layer_from("b", Pixels::filled(crate::pixel::Depth::U8, 10, 10, [1.0; 4]), (10, 0));
        d.set_active(b);
        d.selected.insert(a);
        assert!(d.group_selected());
        assert!(resize_doc(&mut d, 40, 20, ResizeFilter::Nearest, Scope::Layer));
        assert_eq!(d.layer(a).unwrap().bounds(), Rect::new(0, 0, 20, 20));
        assert_eq!(d.layer(b).unwrap().bounds(), Rect::new(20, 0, 20, 20));
    }

    #[test]
    fn image_scope_resize_and_layer_flip() {
        let mut d = Document::new(40, 20, crate::pixel::Depth::U8, None);
        let id = d.add_layer_from("a", Pixels::filled(crate::pixel::Depth::U8, 10, 10, [1.0; 4]), (0, 0));
        assert!(resize_doc(&mut d, 20, 10, ResizeFilter::Nearest, Scope::Image));
        assert_eq!((d.width, d.height), (20, 10));
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(0, 0, 5, 5));
        // Flipping a layer keeps it in place; an anchor rotation moves it.
        d.set_active(id);
        assert!(flip(&mut d, true, Scope::Layer));
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(0, 0, 5, 5));
        assert!(rotate_free(&mut d, -90.0, false, Scope::Layer, Some(Vec2::new(0.0, 0.0))));
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(-5, 0, 5, 5));
    }

    #[test]
    fn doc_rotate_image_quarter() {
        let mut d = Document::new(4, 2, crate::pixel::Depth::U8, Some(crate::color::Color::WHITE));
        assert!(rotate_quarter(&mut d, 1, Scope::Image));
        assert_eq!((d.width, d.height), (2, 4));
        assert_eq!(d.layers[0].bounds(), Rect::new(0, 0, 2, 4));
    }
}
