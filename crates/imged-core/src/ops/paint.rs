//! Painting: brush dabs, shapes, and compositing a coverage mask with a color.
//!
//! Tools build a [`Mask`] (coverage per pixel, layer-local coordinates) and
//! then composite it with [`paint_mask`]. Keeping coverage separate from
//! color makes strokes uniform (overlapping dabs don't stack up) and lets
//! "Replace" mode paint transparent colors.

use crate::buffer::Buffer;
use crate::dispatch;
use crate::geom::{Rect, Vec2};
use crate::mask::Mask;
use crate::pixel::{lerp_premul, over, Channel, Px};
use crate::pixels::Pixels;

/// How painted color combines with existing pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintMode {
    /// Pixel becomes the paint color (alpha included), blended only by
    /// coverage. Painting a transparent color erases.
    Replace,
    /// Standard alpha blending: paint is layered on top.
    Over,
}

impl PaintMode {
    pub fn label(self) -> &'static str {
        match self {
            PaintMode::Replace => "Replace",
            PaintMode::Over => "Blend over",
        }
    }
}

#[inline(always)]
fn coverage(d: f32, aa: bool) -> f32 {
    if aa {
        (0.5 - d).clamp(0.0, 1.0)
    } else if d <= 0.0 {
        1.0
    } else {
        0.0
    }
}

// ----- signed distance functions (negative inside) ----------------------

fn sd_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = if ba.dot(ba) < 1e-9 { 0.0 } else { (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0) };
    (pa - ba * h).len()
}

fn sd_box(p: Vec2, center: Vec2, half: Vec2) -> f32 {
    let d = Vec2::new((p.x - center.x).abs() - half.x, (p.y - center.y).abs() - half.y);
    let outside = Vec2::new(d.x.max(0.0), d.y.max(0.0)).len();
    outside + d.x.max(d.y).min(0.0)
}

fn sd_ellipse(p: Vec2, center: Vec2, r: Vec2) -> f32 {
    let r = Vec2::new(r.x.max(0.01), r.y.max(0.01));
    let q = p - center;
    let k0 = Vec2::new(q.x / r.x, q.y / r.y).len();
    let k1 = Vec2::new(q.x / (r.x * r.x), q.y / (r.y * r.y)).len();
    if k1 < 1e-9 {
        return -r.x.min(r.y);
    }
    k0 * (k0 - 1.0) / k1
}

fn sd_polygon(p: Vec2, v: &[Vec2]) -> f32 {
    let mut d = (p - v[0]).dot(p - v[0]);
    let mut s = 1.0;
    let n = v.len();
    let mut j = n - 1;
    for i in 0..n {
        let e = v[j] - v[i];
        let w = p - v[i];
        let t = (w.dot(e) / e.dot(e).max(1e-9)).clamp(0.0, 1.0);
        let b = w - e * t;
        d = d.min(b.dot(b));
        let c1 = p.y >= v[i].y;
        let c2 = p.y < v[j].y;
        let c3 = e.x * w.y > e.y * w.x;
        if (c1 && c2 && c3) || (!c1 && !c2 && !c3) {
            s = -s;
        }
        j = i;
    }
    s * d.sqrt()
}

/// Fills `mask` within `bounds` using a signed distance function.
fn raster_sdf(mask: &mut Mask, bounds: Rect, aa: bool, sdf: impl Fn(Vec2) -> f32) {
    let r = bounds.intersect(&mask.rect);
    for y in r.y..r.bottom() {
        for x in r.x..r.right() {
            let c = coverage(sdf(Vec2::new(x as f32 + 0.5, y as f32 + 0.5)), aa);
            if c > 0.0 {
                mask.max_at(x, y, c);
            }
        }
    }
}

// ----- brush --------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brush {
    /// Diameter in pixels.
    pub size: f32,
    /// 1.0 = hard edge, 0.0 = fully soft.
    pub hardness: f32,
    pub antialias: bool,
}

impl Brush {
    pub fn radius(&self) -> f32 {
        (self.size * 0.5).max(0.5)
    }

    /// Rectangle a dab at `p` can touch.
    pub fn dab_bounds(&self, p: Vec2) -> Rect {
        let r = self.radius() + 1.0;
        Rect::covering(p.x - r, p.y - r, p.x + r, p.y + r)
    }

    /// Stamps one dab into `mask`.
    pub fn dab(&self, mask: &mut Mask, p: Vec2) {
        let r = self.radius();
        let inner = r * self.hardness.clamp(0.0, 1.0);
        let soft = self.hardness < 0.99;
        let aa = self.antialias;
        let bounds = self.dab_bounds(p).intersect(&mask.rect);
        for y in bounds.y..bounds.bottom() {
            for x in bounds.x..bounds.right() {
                let d = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - p).len();
                let c = if soft {
                    if d >= r {
                        0.0
                    } else if d <= inner {
                        1.0
                    } else {
                        let t = (d - inner) / (r - inner);
                        1.0 - t * t * (3.0 - 2.0 * t)
                    }
                } else {
                    coverage(d - r, aa)
                };
                if c > 0.0 {
                    mask.max_at(x, y, c);
                }
            }
        }
    }

    /// Stamps dabs along a segment. Returns the touched rectangle.
    pub fn segment(&self, mask: &mut Mask, a: Vec2, b: Vec2) -> Rect {
        let spacing = (self.radius() * 0.25).max(0.35);
        let len = (b - a).len();
        let steps = (len / spacing).ceil().max(1.0) as usize;
        let mut touched = self.dab_bounds(a).union(&self.dab_bounds(b));
        for i in 0..=steps {
            let p = a + (b - a) * (i as f32 / steps as f32);
            self.dab(mask, p);
        }
        touched = touched.intersect(&mask.rect);
        touched
    }
}

// ----- pen ------------------------------------------------------------------

/// A square, hard, pixel-exact tip: every covered pixel gets the color as is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pen {
    /// Side length in pixels.
    pub size: u32,
}

impl Pen {
    /// The square of pixels the pen covers at `p`.
    pub fn cell(&self, p: Vec2) -> Rect {
        let n = self.size.max(1) as i32;
        let x0 = p.x.floor() as i32 - (n - 1) / 2;
        let y0 = p.y.floor() as i32 - (n - 1) / 2;
        Rect::new(x0, y0, n as u32, n as u32)
    }

    /// Stamps one square. Returns the covered rectangle.
    pub fn dab(&self, mask: &mut Mask, p: Vec2) -> Rect {
        let r = self.cell(p);
        let c = r.intersect(&mask.rect);
        for y in c.y..c.bottom() {
            for x in c.x..c.right() {
                mask.set(x, y, 1.0);
            }
        }
        r
    }

    /// Stamps squares along the pixel line (Bresenham) from `a` to `b`.
    pub fn segment(&self, mask: &mut Mask, a: Vec2, b: Vec2) -> Rect {
        let (mut x0, mut y0) = (a.x.floor() as i32, a.y.floor() as i32);
        let (x1, y1) = (b.x.floor() as i32, b.y.floor() as i32);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut touched = Rect::default();
        loop {
            touched = touched.union(&self.dab(mask, Vec2::new(x0 as f32 + 0.5, y0 as f32 + 0.5)));
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
        touched
    }
}

// ----- shapes -------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    Line,
    Rectangle,
    Ellipse,
    Arrow,
}

impl ShapeKind {
    pub const ALL: [ShapeKind; 4] = [ShapeKind::Line, ShapeKind::Rectangle, ShapeKind::Ellipse, ShapeKind::Arrow];
    pub fn label(self) -> &'static str {
        match self {
            ShapeKind::Line => "Line",
            ShapeKind::Rectangle => "Box",
            ShapeKind::Ellipse => "Circle / ellipse",
            ShapeKind::Arrow => "Arrow",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeFill {
    /// Outline in the primary color.
    Outline,
    /// Filled with the primary color.
    Fill,
    /// Outline in primary, filled with secondary.
    OutlineAndFill,
}

impl ShapeFill {
    pub const ALL: [ShapeFill; 3] = [ShapeFill::Outline, ShapeFill::Fill, ShapeFill::OutlineAndFill];
    pub fn label(self) -> &'static str {
        match self {
            ShapeFill::Outline => "Outline",
            ShapeFill::Fill => "Filled",
            ShapeFill::OutlineAndFill => "Outline + fill (secondary)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub kind: ShapeKind,
    /// Start / end points (for boxes and ellipses: opposite corners).
    pub a: Vec2,
    pub b: Vec2,
    pub width: f32,
    pub fill: ShapeFill,
    pub antialias: bool,
    /// Arrow head length as a multiple of the line width (plus a minimum).
    pub head_scale: f32,
    /// Rounded box corners, round line caps. Off = sharp corners and flat caps.
    pub rounded: bool,
    /// Corner radius for rounded boxes, in pixels.
    pub corner_radius: f32,
    /// Snap the geometry to the pixel grid so edges cover whole pixels
    /// instead of being smeared across two.
    pub pixel_aligned: bool,
}

/// Rounded box; `r` is clamped to fit.
fn sd_round_box(p: Vec2, center: Vec2, half: Vec2, r: f32) -> f32 {
    let r = r.min(half.x).min(half.y).max(0.0);
    sd_box(p, center, Vec2::new(half.x - r, half.y - r)) - r
}

/// Thick line from `a` to `b`: round caps, or flat caps extended by `extend`.
fn sd_line(p: Vec2, a: Vec2, b: Vec2, hw: f32, round: bool, extend: f32) -> f32 {
    if round {
        return sd_segment(p, a, b) - hw;
    }
    let d = b - a;
    let len = d.len();
    let dir = if len < 1e-6 { Vec2::new(1.0, 0.0) } else { d * (1.0 / len) };
    let q = p - a;
    let local = Vec2::new(q.dot(dir), q.dot(dir.perp()));
    sd_box(local, Vec2::new(len * 0.5, 0.0), Vec2::new(len * 0.5 + extend, hw))
}

impl Shape {
    /// A shape with default styling (sharp, unaligned, antialiased outline).
    pub fn new(kind: ShapeKind, a: Vec2, b: Vec2) -> Self {
        Shape {
            kind,
            a,
            b,
            width: 1.0,
            fill: ShapeFill::Outline,
            antialias: true,
            head_scale: 4.0,
            rounded: false,
            corner_radius: 8.0,
            pixel_aligned: false,
        }
    }

    /// The box the shape occupies, as (min, max) corners. When pixel-aligned,
    /// it covers every pixel cell between the two points, inclusive.
    pub fn frame(&self) -> (Vec2, Vec2) {
        let (a, b) = (self.a, self.b);
        if self.pixel_aligned {
            // A point on an exact grid line (e.g. snapped to an edge) is a
            // boundary; any other point means "include the pixel under it".
            let hi = |m: f32, lo: f32| if m.fract() == 0.0 && m > lo { m } else { m.floor() + 1.0 };
            let lo = Vec2::new(a.x.min(b.x).floor(), a.y.min(b.y).floor());
            (lo, Vec2::new(hi(a.x.max(b.x), lo.x), hi(a.y.max(b.y), lo.y)))
        } else {
            (Vec2::new(a.x.min(b.x), a.y.min(b.y)), Vec2::new(a.x.max(b.x), a.y.max(b.y)))
        }
    }

    /// Width and height in pixels, as shown to the user.
    pub fn size(&self) -> (f32, f32) {
        let (lo, hi) = self.frame();
        (hi.x - lo.x, hi.y - lo.y)
    }

    fn stroke_width(&self) -> f32 {
        if self.pixel_aligned {
            self.width.round().max(1.0)
        } else {
            self.width.max(0.5)
        }
    }

    /// Line endpoints: pixel centers when aligned.
    fn ends(&self) -> (Vec2, Vec2) {
        if self.pixel_aligned {
            let c = |p: Vec2| Vec2::new(p.x.floor() + 0.5, p.y.floor() + 0.5);
            (c(self.a), c(self.b))
        } else {
            (self.a, self.b)
        }
    }

    /// Rectangle the shape can touch.
    pub fn bounds(&self) -> Rect {
        let pad = self.width * (self.head_scale + 2.0) + 2.0;
        Rect::covering(
            self.a.x.min(self.b.x) - pad,
            self.a.y.min(self.b.y) - pad,
            self.a.x.max(self.b.x) + pad,
            self.a.y.max(self.b.y) + pad,
        )
    }

    /// Builds the (outline, fill) coverage masks inside `area`.
    /// Either may be `None` depending on the fill style. Box and ellipse
    /// outlines are drawn inside the shape's frame.
    pub fn masks(&self, area: Rect) -> (Option<Mask>, Option<Mask>) {
        let area = self.bounds().intersect(&area);
        let w = self.stroke_width();
        let hw = w * 0.5;
        let aa = self.antialias;
        let (lo, hi) = self.frame();
        let center = (lo + hi) * 0.5;
        let half = (hi - lo) * 0.5;
        let want_outline = self.fill != ShapeFill::Fill;
        let want_fill = self.fill != ShapeFill::Outline;
        let ext = if self.pixel_aligned { 0.5 } else { 0.0 };
        let mut outline = None;
        let mut fill = None;
        match self.kind {
            ShapeKind::Line => {
                let (a, b) = self.ends();
                let mut m = Mask::new(area);
                raster_sdf(&mut m, area, aa, |p| sd_line(p, a, b, hw, self.rounded, ext));
                outline = Some(m);
            }
            ShapeKind::Arrow => {
                let (a, b) = self.ends();
                let mut m = Mask::new(area);
                let dir = (b - a).normalized();
                let head_len = (w * self.head_scale).max(8.0).min((b - a).len());
                let head_half = head_len * 0.6;
                let base = b - dir * head_len;
                let tri = [b, base + dir.perp() * head_half, base - dir.perp() * head_half];
                // Stop the shaft inside the head so the tip stays sharp.
                let shaft_end = b - dir * (head_len * 0.5);
                raster_sdf(&mut m, area, aa, |p| {
                    sd_line(p, a, shaft_end, hw, self.rounded, ext).min(sd_polygon(p, &tri))
                });
                outline = Some(m);
            }
            ShapeKind::Rectangle => {
                let r = if self.rounded { self.corner_radius.max(0.0) } else { 0.0 };
                let r = if self.pixel_aligned { r.round() } else { r };
                if want_fill {
                    let mut m = Mask::new(area);
                    raster_sdf(&mut m, area, aa, |p| sd_round_box(p, center, half, r));
                    fill = Some(m);
                }
                if want_outline {
                    let inner = Vec2::new(half.x - w, half.y - w);
                    let inner_r = (r - w).max(0.0);
                    let mut m = Mask::new(area);
                    raster_sdf(&mut m, area, aa, |p| {
                        let outer = sd_round_box(p, center, half, r);
                        if inner.x <= 0.0 || inner.y <= 0.0 {
                            outer
                        } else {
                            outer.max(-sd_round_box(p, center, inner, inner_r))
                        }
                    });
                    outline = Some(m);
                }
            }
            ShapeKind::Ellipse => {
                if want_fill {
                    let mut m = Mask::new(area);
                    raster_sdf(&mut m, area, aa, |p| sd_ellipse(p, center, half));
                    fill = Some(m);
                }
                if want_outline {
                    let inner = Vec2::new(half.x - w, half.y - w);
                    let mut m = Mask::new(area);
                    raster_sdf(&mut m, area, aa, |p| {
                        let outer = sd_ellipse(p, center, half);
                        if inner.x <= 0.0 || inner.y <= 0.0 {
                            outer
                        } else {
                            outer.max(-sd_ellipse(p, center, inner))
                        }
                    });
                    outline = Some(m);
                }
            }
        }
        (outline, fill)
    }
}

// ----- compositing ---------------------------------------------------------

#[inline(always)]
pub fn blend(base: Px, color: Px, cov: f32, mode: PaintMode) -> Px {
    match mode {
        PaintMode::Replace => lerp_premul(base, color, cov),
        PaintMode::Over => {
            let mut c = color;
            c[3] *= cov;
            over(c, base)
        }
    }
}

/// Composites `color` through `mask` (layer-local coords) into `dst`.
/// When `base` is given, pixels are computed from it instead of from `dst`
/// (used for live strokes so repeated updates don't accumulate).
pub fn paint_mask_buffer<T: Channel>(
    dst: &mut Buffer<T>,
    base: Option<&Buffer<T>>,
    mask: &Mask,
    region: Rect,
    color: Px,
    mode: PaintMode,
) {
    let r = region.intersect(&mask.rect).intersect(&dst.rect());
    for y in r.y..r.bottom() {
        for x in r.x..r.right() {
            let cov = mask.get(x, y);
            let b = match base {
                Some(b) => b.get(x as u32, y as u32),
                None => dst.get(x as u32, y as u32),
            };
            let out = if cov > 0.0 { blend(b, color, cov, mode) } else { b };
            if base.is_some() || cov > 0.0 {
                dst.set(x as u32, y as u32, out);
            }
        }
    }
}

/// Runtime-depth version of [`paint_mask_buffer`]. `base` must have the same depth as `dst`.
pub fn paint_mask(dst: &mut Pixels, base: Option<&Pixels>, mask: &Mask, region: Rect, color: Px, mode: PaintMode) {
    match (dst, base) {
        (Pixels::U8(d), Some(Pixels::U8(b))) => paint_mask_buffer(d, Some(b), mask, region, color, mode),
        (Pixels::U16(d), Some(Pixels::U16(b))) => paint_mask_buffer(d, Some(b), mask, region, color, mode),
        (Pixels::F32(d), Some(Pixels::F32(b))) => paint_mask_buffer(d, Some(b), mask, region, color, mode),
        (d, _) => dispatch!(d, d => paint_mask_buffer(d, None, mask, region, color, mode)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_paints_transparent() {
        let mut px = Pixels::filled(crate::pixel::Depth::U8, 4, 4, [1.0, 1.0, 1.0, 1.0]);
        let mask = Mask::full(Rect::new(1, 1, 2, 2));
        paint_mask(&mut px, None, &mask, mask.rect, [0.0, 0.0, 0.0, 0.0], PaintMode::Replace);
        assert_eq!(px.get(1, 1)[3], 0.0);
        assert_eq!(px.get(0, 0)[3], 1.0);
    }

    #[test]
    fn over_ignores_transparent() {
        let mut px = Pixels::filled(crate::pixel::Depth::U8, 2, 2, [1.0, 1.0, 1.0, 1.0]);
        let mask = Mask::full(Rect::new(0, 0, 2, 2));
        paint_mask(&mut px, None, &mask, mask.rect, [0.0, 0.0, 0.0, 0.0], PaintMode::Over);
        assert_eq!(px.get(1, 1), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn hard_rect_outline() {
        // Unaligned box from 1.0 to 9.0: the 1px outline lies inside, on cells 1 and 8.
        let s = Shape { antialias: false, ..Shape::new(ShapeKind::Rectangle, Vec2::new(1.0, 1.0), Vec2::new(9.0, 9.0)) };
        let (o, f) = s.masks(Rect::from_size(12, 12));
        let o = o.unwrap();
        assert!(f.is_none());
        assert_eq!(o.get(1, 1), 1.0);
        assert_eq!(o.get(8, 5), 1.0);
        assert_eq!(o.get(5, 5), 0.0);
        assert_eq!(o.get(9, 5), 0.0);
        // Sharp corner: the corner pixel is fully covered even with antialiasing.
        let s = Shape { width: 3.0, ..Shape::new(ShapeKind::Rectangle, Vec2::new(1.0, 1.0), Vec2::new(9.0, 9.0)) };
        assert_eq!(s.masks(Rect::from_size(12, 12)).0.unwrap().get(1, 1), 1.0);
    }

    #[test]
    fn pixel_aligned_box_is_crisp() {
        // Dragging from the middle of pixel (2,3) to the middle of pixel (7,6).
        let s = Shape {
            pixel_aligned: true,
            ..Shape::new(ShapeKind::Rectangle, Vec2::new(2.4, 3.6), Vec2::new(7.3, 6.2))
        };
        assert_eq!(s.size(), (6.0, 4.0));
        let o = s.masks(Rect::from_size(12, 12)).0.unwrap();
        for (x, y, v) in [(2, 3, 1.0), (7, 3, 1.0), (7, 6, 1.0), (4, 3, 1.0), (4, 4, 0.0), (8, 3, 0.0), (1, 3, 0.0)] {
            assert_eq!(o.get(x, y), v, "({x},{y})");
        }
        // Aligned horizontal line covers whole pixels, endpoints included.
        let l = Shape { pixel_aligned: true, ..Shape::new(ShapeKind::Line, Vec2::new(2.2, 5.7), Vec2::new(6.9, 5.1)) };
        let m = l.masks(Rect::from_size(12, 12)).0.unwrap();
        for x in 2..=6 {
            assert_eq!(m.get(x, 5), 1.0, "x={x}");
        }
        assert_eq!(m.get(1, 5), 0.0);
        assert_eq!(m.get(7, 5), 0.0);
        assert_eq!(m.get(4, 4), 0.0);
        // Snapped (integer) corners are boundaries: exactly 10x5.
        let s = Shape { pixel_aligned: true, ..Shape::new(ShapeKind::Rectangle, Vec2::new(10.0, 5.0), Vec2::new(20.0, 10.0)) };
        assert_eq!(s.size(), (10.0, 5.0));
    }

    #[test]
    fn rounded_box_has_round_corners() {
        let s = Shape {
            rounded: true,
            corner_radius: 4.0,
            fill: ShapeFill::Fill,
            ..Shape::new(ShapeKind::Rectangle, Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0))
        };
        let f = s.masks(Rect::from_size(10, 10)).1.unwrap();
        assert_eq!(f.get(0, 0), 0.0);
        assert_eq!(f.get(5, 0), 1.0);
    }

    #[test]
    fn pen_is_square_and_hard() {
        let pen = Pen { size: 3 };
        let mut m = Mask::new(Rect::from_size(10, 10));
        pen.dab(&mut m, Vec2::new(4.9, 4.1));
        assert_eq!(m.count(), 9);
        assert_eq!(m.bounds(), Some(Rect::new(3, 3, 3, 3)));
        let mut m = Mask::new(Rect::from_size(10, 10));
        Pen { size: 1 }.segment(&mut m, Vec2::new(0.5, 0.5), Vec2::new(5.5, 2.5));
        assert_eq!(m.count(), 6);
        assert!(m.data.iter().all(|v| *v == 0.0 || *v == 1.0));
    }

    #[test]
    fn brush_dab_single_pixel() {
        let b = Brush { size: 1.0, hardness: 1.0, antialias: false };
        let mut m = Mask::new(Rect::from_size(5, 5));
        b.dab(&mut m, Vec2::new(2.5, 2.5));
        assert_eq!(m.count(), 1);
    }
}
