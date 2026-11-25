//! Stitching two photos of a flat surface (a whiteboard, or a document
//! projected onto one) into one image.
//!
//! The user clicks the same letter on both layers. [`align`] snaps that pair
//! of clicks together (searching rotation and scale), grows a set of matched
//! feature points outward from it, fits a homography and a smooth residual
//! field (lens distortion, boards that are not perfectly flat). [`merge`]
//! then refines the alignment densely in the overlap, warps the moving layer
//! onto the fixed one, and composites them: where the photos disagree (text
//! written between the two shots) the layer that has ink there wins, or the
//! upper layer if asked.
//!
//! Coordinates: unless noted, points are layer-local "pixel center"
//! coordinates, where the center of pixel (i, j) is (i, j) (OpenCV's
//! convention). Canvas coordinates in the GUI have centers at +0.5.
//!
//! The image analysis uses the system OpenCV (`stitch` cargo feature). The
//! geometry, the warping and the final compositing are plain Rust, generic
//! over the channel type like the rest of the crate.

#![cfg_attr(not(feature = "stitch"), allow(dead_code))]

use std::sync::Arc;

use rayon::prelude::*;

use crate::buffer::Buffer;
use crate::document::Document;
use crate::geom::Rect;
use crate::layer::{Layer, LayerId};
use crate::pixel::{Channel, Px};
use crate::pixels::Pixels;

#[cfg(feature = "stitch")]
#[path = "stitch_cv.rs"]
mod cv;

/// What to align.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StitchMode {
    /// Writing on the board: the whole photo is used.
    #[default]
    Board,
    /// A document projected onto the board and scrolled between the photos:
    /// only the projected area (found around the clicked point) is aligned
    /// and kept, so the board and wall, which did not move with the
    /// document, are left out.
    Projection,
}

impl StitchMode {
    pub fn label(self) -> &'static str {
        match self {
            StitchMode::Board => "Board",
            StitchMode::Projection => "Projected document",
        }
    }
}

/// Whether this build can stitch (it needs OpenCV).
pub const AVAILABLE: bool = cfg!(feature = "stitch");

/// A 3×3 projective transform, row-major, mapping (x, y, 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    pub const IDENTITY: Homography = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    /// Maps a point; `None` when it lands at or behind the horizon.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.0;
        let w = m[6] * x + m[7] * y + m[8];
        if w <= 1e-9 {
            return None;
        }
        Some(((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w))
    }

    pub fn mul(&self, o: &Homography) -> Homography {
        let (a, b) = (&self.0, &o.0);
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
            }
        }
        Homography(r)
    }

    pub fn inverse(&self) -> Option<Homography> {
        let m = &self.0;
        let c00 = m[4] * m[8] - m[5] * m[7];
        let c01 = m[5] * m[6] - m[3] * m[8];
        let c02 = m[3] * m[7] - m[4] * m[6];
        let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        let inv = [
            c00,
            m[2] * m[7] - m[1] * m[8],
            m[1] * m[5] - m[2] * m[4],
            c01,
            m[0] * m[8] - m[2] * m[6],
            m[2] * m[3] - m[0] * m[5],
            c02,
            m[1] * m[6] - m[0] * m[7],
            m[0] * m[4] - m[1] * m[3],
        ];
        let mut r = [0.0; 9];
        for i in 0..9 {
            r[i] = inv[i] / det;
        }
        // Keep w positive for points in front of the camera.
        if r[8] < 0.0 {
            r.iter_mut().for_each(|v| *v = -*v);
        }
        Some(Homography(r))
    }

    /// Diagonal scaling `diag(sx, sy, 1)`.
    pub fn scale(sx: f64, sy: f64) -> Homography {
        Homography([sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0])
    }
}

/// A smooth 2D displacement field sampled on a regular grid. Node (i, j)
/// sits at `origin + (i, j) * cell`; samples are bilinear and clamp to the
/// edge nodes (the fields decay to zero well inside their grids).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Field {
    pub origin: [f32; 2],
    pub cell: f32,
    pub w: usize,
    pub h: usize,
    pub d: Vec<[f32; 2]>,
}

impl Field {
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> [f32; 2] {
        if self.w == 0 || self.h == 0 {
            return [0.0, 0.0];
        }
        let fx = ((x - self.origin[0]) / self.cell).clamp(0.0, (self.w - 1) as f32);
        let fy = ((y - self.origin[1]) / self.cell).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let g = |x: usize, y: usize| self.d[y * self.w + x];
        let (a, b, c, d) = (g(x0, y0), g(x1, y0), g(x0, y1), g(x1, y1));
        let mut o = [0.0; 2];
        for k in 0..2 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            o[k] = top + (bot - top) * ty;
        }
        o
    }

    pub fn max_abs(&self) -> f32 {
        self.d.iter().map(|v| v[0].abs().max(v[1].abs())).fold(0.0, f32::max)
    }
}

/// Result of [`align`]: how the moving layer maps onto the fixed one.
#[derive(Clone, Debug, PartialEq)]
pub struct Alignment {
    /// Moving-layer pixel → fixed-layer pixel (layer-local, before `field`).
    pub h: Homography,
    /// Residual correction in fixed-layer coordinates:
    /// `fixed ≈ h(moving) + field(fixed)`.
    pub field: Field,
    /// Matched points used for the final fit, as (fixed, moving) pairs.
    pub matches: Vec<([f32; 2], [f32; 2])>,
    /// Rotation (degrees) and scale found at the clicked point.
    pub angle: f32,
    pub scale: f32,
    /// How well the clicked letters matched (normalized correlation, 0..1).
    pub seed_score: f32,
    /// Median distance (fixed-layer pixels) between matched points after the fit.
    pub residual: f32,
    /// The clicked point on the fixed layer after snapping.
    pub seed_fixed: [f32; 2],
    pub mode: StitchMode,
    /// In projection mode: the projected area on each layer (corners in order).
    pub quad_fixed: Option<[[f32; 2]; 4]>,
    pub quad_moving: Option<[[f32; 2]; 4]>,
}

impl Alignment {
    /// Maps a moving-layer point to the fixed layer (forward, approximate field).
    pub fn forward(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let (fx, fy) = self.h.apply(x as f64, y as f64)?;
        let d = self.field.sample(fx as f32, fy as f32);
        Some([fx as f32 + d[0], fy as f32 + d[1]])
    }
}

/// Options for [`merge`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergeOptions {
    /// In regions where the photos disagree, take the upper layer instead of
    /// the one with ink there.
    pub prefer_upper: bool,
    /// Whether the fixed layer is above the moving one in the layer stack.
    pub fixed_on_top: bool,
    /// Also return the moving layer warped into place (to keep it as a layer).
    pub want_warped: bool,
}

/// Result of [`merge`]. Rectangles are in fixed-layer coordinates.
#[derive(Clone, Debug)]
pub struct StitchOutput {
    pub merged: Pixels,
    pub merged_rect: Rect,
    pub warped: Option<(Pixels, Rect)>,
    /// Number of regions where the photos disagreed.
    pub conflicts: usize,
}

/// Progress callback: (fraction 0..1, what is happening).
pub type Progress<'a> = &'a (dyn Fn(f32, &str) + Sync);

/// Aligns `moving` onto `fixed` from one pair of clicked points (layer-local,
/// pixel-center coordinates).
pub fn align(
    fixed: &Pixels,
    moving: &Pixels,
    seed_fixed: [f32; 2],
    seed_moving: [f32; 2],
    mode: StitchMode,
    progress: Progress,
) -> Result<Alignment, String> {
    #[cfg(feature = "stitch")]
    {
        cv::align(fixed, moving, seed_fixed, seed_moving, mode, progress)
    }
    #[cfg(not(feature = "stitch"))]
    {
        let _ = (fixed, moving, seed_fixed, seed_moving, mode, progress);
        Err(UNAVAILABLE.into())
    }
}

/// Warps `moving` onto `fixed` with `al` and composites the two.
pub fn merge(fixed: &Pixels, moving: &Pixels, al: &Alignment, opt: MergeOptions, progress: Progress) -> Result<StitchOutput, String> {
    #[cfg(feature = "stitch")]
    {
        cv::merge(fixed, moving, al, opt, progress)
    }
    #[cfg(not(feature = "stitch"))]
    {
        let _ = (fixed, moving, al, opt, progress);
        Err(UNAVAILABLE.into())
    }
}

#[cfg(not(feature = "stitch"))]
const UNAVAILABLE: &str = "Stitching needs OpenCV: rebuild imged with `--features stitch`";

// ----- mapping and compositing (no OpenCV) --------------------------------

/// Fixed-layer point → moving-layer point, with the residual field and the
/// dense overlap correction.
#[derive(Clone, Debug)]
pub(crate) struct Mapping {
    pub hinv: Homography,
    pub field: Field,
    /// Dense correction in the overlap (`None` before it is computed).
    pub flow: Option<Field>,
    pub quad_fixed: Option<[[f32; 2]; 4]>,
    pub quad_moving: Option<[[f32; 2]; 4]>,
}

impl Mapping {
    pub fn new(al: &Alignment) -> Result<Mapping, String> {
        Ok(Mapping {
            hinv: al.h.inverse().ok_or("The alignment is degenerate")?,
            field: al.field.clone(),
            flow: None,
            quad_fixed: al.quad_fixed,
            quad_moving: al.quad_moving,
        })
    }

    #[inline]
    pub fn to_moving(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let (mut x, mut y) = (x, y);
        if let Some(f) = &self.flow {
            let d = f.sample(x, y);
            x += d[0];
            y += d[1];
        }
        let d = self.field.sample(x, y);
        let (mx, my) = self.hinv.apply((x - d[0]) as f64, (y - d[1]) as f64)?;
        if let Some(q) = &self.quad_moving {
            if !in_quad(q, mx as f32, my as f32) {
                return None;
            }
        }
        Some([mx as f32, my as f32])
    }

    /// Whether a fixed-layer pixel takes part (inside the projected area).
    #[inline]
    pub fn fixed_used(&self, x: f32, y: f32) -> bool {
        self.quad_fixed.as_ref().is_none_or(|q| in_quad(q, x, y))
    }
}

/// Point in a convex quadrilateral (either winding).
pub fn in_quad(q: &[[f32; 2]; 4], x: f32, y: f32) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let c = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        if c != 0.0 {
            if sign == 0.0 {
                sign = c.signum();
            } else if c.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Samples the moving layer at a fixed-layer pixel (straight alpha).
#[inline]
pub(crate) fn sample_moving<T: Channel>(b: &Buffer<T>, map: &Mapping, x: f32, y: f32) -> Px {
    match map.to_moving(x, y) {
        Some([mx, my]) => b.sample_bilinear(mx + 0.5, my + 0.5),
        None => [0.0; 4],
    }
}

/// Straight-alpha "a over b".
#[inline]
pub(crate) fn over(a: Px, b: Px) -> Px {
    let ao = a[3] + b[3] * (1.0 - a[3]);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let mut o = [0.0; 4];
    for c in 0..3 {
        o[c] = (a[c] * a[3] + b[c] * b[3] * (1.0 - a[3])) / ao;
    }
    o[3] = ao;
    o
}

/// The blended overlap computed by the analysis, in fixed-layer coordinates.
pub(crate) struct OverlapBlend {
    pub rect: Rect,
    pub rgb: Vec<[f32; 3]>,
    /// Where both layers are valid and `rgb` applies.
    pub used: Vec<bool>,
}

/// Renders the merged image over `rect` (fixed-layer coordinates).
pub(crate) fn render_merged(
    fixed: &Pixels,
    moving: &Pixels,
    map: &Mapping,
    blend: &OverlapBlend,
    rect: Rect,
    depth: crate::pixel::Depth,
) -> Pixels {
    fn go<T: Channel, F: Channel, M: Channel>(
        fixed: &Buffer<F>,
        moving: &Buffer<M>,
        map: &Mapping,
        blend: &OverlapBlend,
        rect: Rect,
    ) -> Buffer<T> {
        Buffer::<T>::from_fn(rect.w, rect.h, |i, j| {
            let (x, y) = (rect.x + i as i32, rect.y + j as i32);
            if blend.rect.contains(x, y) {
                let k = (y - blend.rect.y) as usize * blend.rect.w as usize + (x - blend.rect.x) as usize;
                if blend.used[k] {
                    let c = blend.rgb[k];
                    return [c[0], c[1], c[2], 1.0];
                }
            }
            let a = if fixed.in_bounds(x, y) && map.fixed_used(x as f32, y as f32) {
                fixed.get(x as u32, y as u32)
            } else {
                [0.0; 4]
            };
            if a[3] >= 1.0 {
                return a;
            }
            over(a, sample_moving(moving, map, x as f32, y as f32))
        })
    }
    use crate::pixel::Depth;
    macro_rules! with_out {
        ($t:ty) => {
            crate::dispatch!(fixed, f => crate::dispatch!(moving, m => Pixels::from_buffer(go::<$t, _, _>(f, m, map, blend, rect))))
        };
    }
    match depth {
        Depth::U8 => with_out!(u8),
        Depth::U16 => with_out!(u16),
        Depth::F32 => with_out!(f32),
    }
}

/// Renders the moving layer warped into place over `rect` (fixed-layer coordinates).
pub(crate) fn render_warped(moving: &Pixels, map: &Mapping, rect: Rect) -> Pixels {
    crate::map_pixels!(moving, m => Buffer::from_fn(rect.w, rect.h, |i, j| {
        sample_moving(m, map, (rect.x + i as i32) as f32, (rect.y + j as i32) as f32)
    }))
}

/// Reads the fixed layer's pixels inside `r` as RGB + validity (alpha ≥ 0.99
/// and inside the projected area).
pub(crate) fn read_fixed(fixed: &Pixels, map: &Mapping, r: Rect) -> (Vec<[f32; 3]>, Vec<bool>) {
    crate::dispatch!(fixed, b => read_rows(r, |x, y| {
        if b.in_bounds(x, y) && map.fixed_used(x as f32, y as f32) { b.get(x as u32, y as u32) } else { [0.0; 4] }
    }))
}

/// The moving layer warped into `r` as RGB + validity.
pub(crate) fn read_warped(moving: &Pixels, map: &Mapping, r: Rect) -> (Vec<[f32; 3]>, Vec<bool>) {
    crate::dispatch!(moving, b => read_rows(r, |x, y| sample_moving(b, map, x as f32, y as f32)))
}

fn read_rows(r: Rect, f: impl Fn(i32, i32) -> Px + Sync) -> (Vec<[f32; 3]>, Vec<bool>) {
    let n = r.w as usize * r.h as usize;
    let mut rgb = vec![[0.0f32; 3]; n];
    let mut valid = vec![false; n];
    if n == 0 {
        return (rgb, valid);
    }
    rgb.par_chunks_mut(r.w as usize).zip(valid.par_chunks_mut(r.w as usize)).enumerate().for_each(|(j, (row, vrow))| {
        for i in 0..r.w as usize {
            let p = f(r.x + i as i32, r.y + j as i32);
            if p[3] >= 0.99 {
                row[i] = [p[0], p[1], p[2]];
                vrow[i] = true;
            }
        }
    });
    (rgb, valid)
}

// ----- applying the result to the document ----------------------------------

/// The result of a stitch, ready to go into the document as one undo step.
#[derive(Clone, Debug, PartialEq)]
pub struct StitchApply {
    pub fixed: LayerId,
    pub moving: LayerId,
    pub merged: Arc<Pixels>,
    /// Canvas position of the merged layer.
    pub merged_offset: (i32, i32),
    /// The moving layer warped into place (canvas position), kept hidden
    /// with the fixed layer when `Some`; otherwise both sources are removed.
    pub warped: Option<(Arc<Pixels>, (i32, i32))>,
}

/// Adds the stitched layer above the two sources, hides (or removes) them and
/// grows the canvas to fit. One undo step.
pub fn apply(doc: &mut Document, a: &StitchApply) -> bool {
    let (Some(fi), Some(mi)) = (doc.index_of(a.fixed), doc.index_of(a.moving)) else {
        return false;
    };
    let id = doc.alloc_id();
    let top = fi.max(mi);
    doc.structural("Stitch", |d| {
        let mut l = Layer::new(id, "Stitched", (*a.merged).clone(), a.merged_offset);
        l.group = d.layers[top].group;
        d.layers.insert(top + 1, l);
        match &a.warped {
            Some((px, off)) => {
                if let Some(m) = d.layer_mut(a.moving) {
                    m.pixels = px.clone();
                    m.offset = *off;
                    m.object = None;
                    m.visible = false;
                }
                if let Some(f) = d.layer_mut(a.fixed) {
                    f.visible = false;
                }
            }
            None => d.layers.retain(|l| l.id != a.fixed && l.id != a.moving),
        }
        d.active = id;
        d.selected.retain(|s| *s != a.fixed && *s != a.moving);
        // Fit the canvas to the visible layers that have something in them
        // (the stitched one included), so nothing visible is cut off and an
        // empty background left over from tiling the photos doesn't count.
        let u = d
            .layers
            .iter()
            .filter(|l| l.visible && (l.id == id || has_content(&l.pixels)))
            .fold(Rect::default(), |u, l| u.union(&l.bounds()));
        if u != d.canvas_rect() {
            d.width = u.w;
            d.height = u.h;
            for l in &mut d.layers {
                l.offset = (l.offset.0 - u.x, l.offset.1 - u.y);
            }
            if let Some(sel) = d.selection.take() {
                d.selection = Some(Arc::new((*sel).clone().translated(-u.x, -u.y)));
            }
        }
        true
    })
}

fn has_content(p: &Pixels) -> bool {
    crate::dispatch!(p, b => b.data.par_chunks(4 * 4096).any(|c| c.chunks(4).any(|px| px[3].to_f32() > 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_inverse_round_trip() {
        let h = Homography([0.9, 0.05, 30.0, -0.1, 1.1, -12.0, 1e-5, -2e-5, 1.0]);
        let hi = h.inverse().unwrap();
        let (x, y) = h.apply(123.0, 456.0).unwrap();
        let (bx, by) = hi.apply(x, y).unwrap();
        assert!((bx - 123.0).abs() < 1e-6 && (by - 456.0).abs() < 1e-6);
        let id = h.mul(&hi);
        for (i, v) in id.0.iter().enumerate() {
            let e = if i % 4 == 0 { 1.0 } else { 0.0 };
            assert!((v / id.0[8] - e).abs() < 1e-9);
        }
    }

    #[test]
    fn field_sampling_is_bilinear_and_clamped() {
        let f = Field { origin: [0.0, 0.0], cell: 10.0, w: 2, h: 2, d: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]] };
        assert_eq!(f.sample(5.0, 5.0), [1.0, 2.0]);
        assert_eq!(f.sample(-50.0, 100.0), [0.0, 4.0]);
    }

    #[test]
    fn quad_test() {
        let q = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(in_quad(&q, 5.0, 5.0));
        assert!(!in_quad(&q, 11.0, 5.0));
        let r = [q[3], q[2], q[1], q[0]];
        assert!(in_quad(&r, 1.0, 9.0));
    }

    #[test]
    fn apply_adds_layer_and_grows_canvas() {
        let mut doc = Document::from_pixels(Pixels::new(crate::Depth::U8, 10, 10), "a");
        let a = doc.layers[0].id;
        let b = doc.add_layer_from("b", Pixels::new(crate::Depth::U8, 10, 10), (0, 0));
        let ap = StitchApply {
            fixed: a,
            moving: b,
            merged: Arc::new(Pixels::filled(crate::Depth::U8, 25, 10, [1.0; 4])),
            merged_offset: (-5, 0),
            warped: Some((Arc::new(Pixels::new(crate::Depth::U8, 12, 10)), (8, 0))),
        };
        assert!(apply(&mut doc, &ap));
        assert_eq!((doc.width, doc.height), (25, 10));
        assert_eq!(doc.layers.len(), 3);
        let top = doc.layers.last().unwrap();
        assert_eq!(top.name, "Stitched");
        assert_eq!(top.offset, (0, 0));
        assert!(!doc.layer(a).unwrap().visible && !doc.layer(b).unwrap().visible);
        assert_eq!(doc.layer(b).unwrap().offset, (13, 0));
        doc.undo();
        assert_eq!(doc.layers.len(), 2);
        assert_eq!(doc.width, 10);
    }
    /// A fake whiteboard: rows of random handwriting-like strokes.
    #[cfg(feature = "stitch")]
    fn board(w: u32, h: u32) -> Buffer<u8> {
        let mut b = Buffer::<u8>::filled(w, h, [0.93, 0.93, 0.9, 1.0]);
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let mut y = 40.0;
        while y < h as f32 - 40.0 {
            let mut x = 30.0;
            while x < w as f32 - 60.0 {
                // A "letter": a few connected segments.
                let (mut px, mut py) = (x + rnd() * 6.0, y + rnd() * 20.0);
                for _ in 0..3 + (rnd() * 3.0) as usize {
                    let (nx, ny) = (x + rnd() * 22.0, y + rnd() * 28.0);
                    let n = ((nx - px).hypot(ny - py) * 2.0) as usize + 1;
                    for k in 0..=n {
                        let t = k as f32 / n as f32;
                        let (cx, cy) = (px + (nx - px) * t, py + (ny - py) * t);
                        for dy in -2..=2 {
                            for dx in -2..=2 {
                                let (ix, iy) = (cx as i32 + dx, cy as i32 + dy);
                                if (dx * dx + dy * dy) <= 3 && b.in_bounds(ix, iy) {
                                    b.set(ix as u32, iy as u32, [0.1, 0.1, 0.15, 1.0]);
                                }
                            }
                        }
                    }
                    (px, py) = (nx, ny);
                }
                x += 30.0 + rnd() * 25.0;
            }
            y += 55.0 + rnd() * 15.0;
        }
        b
    }

    #[cfg(feature = "stitch")]
    #[test]
    fn aligns_rotated_scaled_photo_and_keeps_new_ink() {
        let src = board(1700, 1000);
        // Fixed photo: the left part. Moving photo: the right part, turned
        // 12° and seen 10% larger, with extra writing in the overlap.
        let fixed = Pixels::U8(src.crop(Rect::new(0, 0, 1000, 1000)));
        let g = Homography::scale(1.0, 1.0);
        let (a, s) = (12f64.to_radians(), 1.0 / 1.1);
        // Moving pixel -> source pixel.
        let to_src = Homography([s * a.cos(), -s * a.sin(), 650.0, s * a.sin(), s * a.cos(), 60.0, 0.0, 0.0, 1.0]).mul(&g);
        let mut mv = Buffer::<u8>::from_fn(1000, 1000, |x, y| {
            let (sx, sy) = to_src.apply(x as f64, y as f64).unwrap();
            src.sample_bilinear(sx as f32 + 0.5, sy as f32 + 0.5)
        });
        // New ink only in the moving photo, centered at source (850, 520).
        let from_src = to_src.inverse().unwrap();
        let (nx, ny) = from_src.apply(850.0, 520.0).unwrap();
        // An "X" written in between the two photos.
        for t in -25..=25 {
            for w in -2..=2 {
                for (dx, dy) in [(t, t + w), (t, -t + w)] {
                    mv.set((nx as i32 + dx) as u32, (ny as i32 + dy) as u32, [0.05, 0.05, 0.05, 1.0]);
                }
            }
        }
        let moving = Pixels::U8(mv);
        // Click the same spot (a little off on the moving photo).
        let (bx, by) = from_src.apply(800.0, 300.0).unwrap();
        let none = |_: f32, _: &str| {};
        let al = align(&fixed, &moving, [800.0, 300.0], [bx as f32 + 4.0, by as f32 - 3.0], StitchMode::Board, &none).unwrap();
        assert!((al.angle - 12.0).abs() < 2.0, "angle {}", al.angle);
        for (mx, my) in [(100.0, 100.0), (300.0, 700.0), (50.0, 900.0)] {
            let (sx, sy) = to_src.apply(mx, my).unwrap();
            let p = al.forward(mx as f32, my as f32).unwrap();
            assert!((p[0] - sx as f32).hypot(p[1] - sy as f32) < 1.5, "({mx},{my}) -> {p:?}, want ({sx},{sy})");
        }
        let at = |out: &StitchOutput| {
            let (x, y) = (850 - out.merged_rect.x, 520 - out.merged_rect.y);
            out.merged.get(x as u32, y as u32)[0]
        };
        let opt = MergeOptions { prefer_upper: false, fixed_on_top: true, want_warped: true };
        let ink = merge(&fixed, &moving, &al, opt, &none).unwrap();
        assert!(ink.conflicts >= 1);
        assert!(at(&ink) < 0.3, "new ink kept: {}", at(&ink));
        assert!(ink.merged_rect.w > 1500 && ink.warped.is_some());
        let upper = merge(&fixed, &moving, &al, MergeOptions { prefer_upper: true, ..opt }, &none).unwrap();
        assert!(at(&upper) > 0.7, "upper (fixed) layer wins: {}", at(&upper));
    }
    /// Two photos of a document projected onto a board, scrolled by 300
    /// document pixels between them, the camera turned 3°. Marks on the board
    /// around the projection don't move with the document and must be ignored.
    #[cfg(feature = "stitch")]
    #[test]
    fn projection_mode_follows_the_document() {
        let doc = board(1000, 1400);
        let marks = board(1200, 900);
        let (ox, oy, sc) = (150.0f64, 100.0f64, 0.9f64);
        let scene = |sx: f64, sy: f64, scroll: f64| -> Px {
            if (ox..ox + 900.0).contains(&sx) && (oy..oy + 700.0).contains(&sy) {
                doc.sample_bilinear(((sx - ox) / sc) as f32 + 0.5, ((sy - oy) / sc + scroll) as f32 + 0.5)
            } else {
                let m = marks.sample_bilinear(sx as f32 + 0.5, sy as f32 + 0.5);
                [m[0] * 0.45, m[1] * 0.45, m[2] * 0.45, 1.0]
            }
        };
        let a = 3f64.to_radians();
        // Camera of the second photo: scene -> photo.
        let cam = Homography([a.cos(), -a.sin(), 25.0, a.sin(), a.cos(), -10.0, 0.0, 0.0, 1.0]);
        let cam_inv = cam.inverse().unwrap();
        let fixed = Pixels::U8(Buffer::from_fn(1200, 900, |x, y| scene(x as f64, y as f64, 0.0)));
        let moving = Pixels::U8(Buffer::from_fn(1200, 900, |x, y| {
            let (sx, sy) = cam_inv.apply(x as f64, y as f64).unwrap();
            scene(sx, sy, 300.0)
        }));
        // Document point (500, 450): at scene y 505 in the first photo, 235 in the second.
        let (mx, my) = cam.apply(600.0, 235.0).unwrap();
        let none = |_: f32, _: &str| {};
        let al = align(&fixed, &moving, [600.0, 505.0], [mx as f32, my as f32], StitchMode::Projection, &none).unwrap();
        assert!(al.quad_fixed.is_some() && al.quad_moving.is_some());
        for (sx, sy) in [(300.0, 200.0), (800.0, 300.0), (500.0, 400.0)] {
            let (px, py) = cam.apply(sx, sy).unwrap();
            let p = al.forward(px as f32, py as f32).unwrap();
            let want = (sx as f32, sy as f32 + 270.0);
            assert!((p[0] - want.0).hypot(p[1] - want.1) < 2.0, "({sx},{sy}) -> {p:?}, want {want:?}");
        }
        let out = merge(&fixed, &moving, &al, MergeOptions { prefer_upper: false, fixed_on_top: true, want_warped: false }, &none).unwrap();
        // Only the projected document is kept: about 900 wide, 700 + 270 tall.
        let r = out.merged_rect;
        assert!((880..=960).contains(&r.w) && (940..=1020).contains(&r.h), "{r:?}");
    }
}
