//! Edge snapping: move a point to nearby color boundaries in the image.
//!
//! Boundaries are measured *between* pixels: a vertical boundary at integer
//! x = bx separates columns bx-1 and bx. Snapping to boundaries (not pixel
//! centers) makes a snapped box enclose an object exactly.
//! x and y snap independently, to vertical and horizontal edges respectively.

use crate::composite::{composite_document, composite_views, LayerView};
use crate::layer::LayerId;
use crate::document::Document;
use crate::geom::{Rect, Vec2};
use crate::pixel::max_diff;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapParams {
    /// Search distance in image pixels.
    pub radius: f32,
    /// Minimum color difference (0..1) across a boundary to count as an edge.
    pub threshold: f32,
}

impl Default for SnapParams {
    fn default() -> Self {
        SnapParams { radius: 8.0, threshold: 0.05 }
    }
}

/// Which coordinates were snapped, and to which guide (if not an edge).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Snapped {
    pub point: Vec2,
    pub x: bool,
    pub y: bool,
    /// e.g. "1/2 canvas width" when x snapped to a guide.
    pub x_guide: Option<&'static str>,
    pub y_guide: Option<&'static str>,
}

impl Snapped {
    /// A short description of the guides snapped to ("canvas center", …).
    pub fn guide_label(&self) -> Option<String> {
        match (self.x_guide, self.y_guide) {
            (Some("1/2 canvas width"), Some("1/2 canvas height")) => Some("canvas center".into()),
            (Some("selection center"), Some("selection center")) => Some("selection center".into()),
            (Some(a), Some(b)) => Some(format!("{a} · {b}")),
            (Some(a), None) | (None, Some(a)) => Some(a.into()),
            (None, None) => None,
        }
    }
}

/// Snaps to the middle of the canvas and of the selection when closer than
/// what `out` already snapped to (within `radius`).
fn snap_guides(doc: &Document, p: Vec2, radius: f32, mut out: Snapped) -> Snapped {
    let mut gx = vec![(doc.width as f32 / 2.0, "1/2 canvas width")];
    let mut gy = vec![(doc.height as f32 / 2.0, "1/2 canvas height")];
    if let Some(b) = doc.selection.as_ref().and_then(|m| m.bounds()) {
        gx.push((b.x as f32 + b.w as f32 / 2.0, "selection center"));
        gy.push((b.y as f32 + b.h as f32 / 2.0, "selection center"));
    }
    let pick = |gs: &[(f32, &'static str)], v: f32, cur: Option<f32>| {
        gs.iter()
            .filter(|(g, _)| (g - v).abs() <= radius && cur.is_none_or(|c| (g - v).abs() <= (c - v).abs()))
            .min_by(|a, b| (a.0 - v).abs().total_cmp(&(b.0 - v).abs()))
            .copied()
    };
    if let Some((g, label)) = pick(&gx, p.x, out.x.then_some(out.point.x)) {
        out.point.x = g;
        out.x = true;
        out.x_guide = Some(label);
    }
    if let Some((g, label)) = pick(&gy, p.y, out.y.then_some(out.point.y)) {
        out.point.y = g;
        out.y = true;
        out.y_guide = Some(label);
    }
    out
}

/// Snaps `p` (canvas coords) to the nearest image edges within the radius,
/// or to the middle of the canvas or of the selection when one is closer.
/// Canvas borders always count as edges.
pub fn snap_point(doc: &Document, p: Vec2, params: SnapParams) -> Snapped {
    let out = snap_edges(doc, p, params);
    snap_guides(doc, p, params.radius, out)
}

fn snap_edges(doc: &Document, p: Vec2, params: SnapParams) -> Snapped {
    let r = params.radius.max(1.0).ceil() as i32;
    let (px, py) = (p.x.round() as i32, p.y.round() as i32);
    let window = Rect::from_corners(px - r - 1, py - r - 1, px + r + 2, py + r + 2);
    let area = window.intersect(&doc.canvas_rect());
    let mut out = Snapped { point: p, ..Default::default() };
    let best = |cands: &mut dyn Iterator<Item = i32>, v: f32| -> Option<i32> {
        cands
            .filter(|c| (*c as f32 - v).abs() <= params.radius)
            .min_by(|a, b| (*a as f32 - v).abs().total_cmp(&(*b as f32 - v).abs()))
    };
    // Canvas borders.
    let border_x = [0, doc.width as i32];
    let border_y = [0, doc.height as i32];

    if area.is_empty() {
        if let Some(bx) = best(&mut border_x.into_iter(), p.x) {
            out.point.x = bx as f32;
            out.x = true;
        }
        if let Some(by) = best(&mut border_y.into_iter(), p.y) {
            out.point.y = by as f32;
            out.y = true;
        }
        return out;
    }
    let img = composite_document(doc, area);
    let get = |x: i32, y: i32| img.get_or_clear(x - area.x, y - area.y);
    // Only consider rows/columns close to the pointer across the edge direction.
    let band = r;

    let vertical_edges = (area.x + 1..area.right()).filter(|&bx| {
        (py - band..=py + band)
            .filter(|y| *y >= area.y && *y < area.bottom())
            .any(|y| max_diff(get(bx - 1, y), get(bx, y)) >= params.threshold)
    });
    let mut xs = vertical_edges.chain(border_x);
    if let Some(bx) = best(&mut xs, p.x) {
        out.point.x = bx as f32;
        out.x = true;
    }

    let horizontal_edges = (area.y + 1..area.bottom()).filter(|&by| {
        (px - band..=px + band)
            .filter(|x| *x >= area.x && *x < area.right())
            .any(|x| max_diff(get(x, by - 1), get(x, by)) >= params.threshold)
    });
    let mut ys = horizontal_edges.chain(border_y);
    if let Some(by) = best(&mut ys, p.y) {
        out.point.y = by as f32;
        out.y = true;
    }
    out
}

/// Snaps a moving layer's edges to what is visible under and around it.
///
/// Built once when a drag starts: it composites every visible layer except
/// the moving ones and records where colors change (so only edges that can
/// actually be seen count, not edges of buried layers), plus the bounds of
/// other layers wherever those bounds are visible, plus the canvas border.
pub struct MoveSnapper {
    canvas: Rect,
    /// For each boundary column `x` (between pixels x-1 and x), the sorted
    /// rows where the color changes across it.
    cols: Vec<Vec<u32>>,
    /// For each boundary row `y`, the sorted columns where the color changes.
    rows: Vec<Vec<u32>>,
    /// Visible layer edges and canvas borders.
    layer_x: Vec<i32>,
    layer_y: Vec<i32>,
}

/// Result of [`MoveSnapper::snap`]: the correction to apply and where the
/// matched edges are (for drawing guides).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RectSnap {
    pub dx: i32,
    pub dy: i32,
    pub guide_x: Option<i32>,
    pub guide_y: Option<i32>,
}

fn any_in(sorted: &[u32], lo: i32, hi: i32) -> bool {
    if hi < lo || hi < 0 {
        return false;
    }
    let lo = lo.max(0) as u32;
    let i = sorted.partition_point(|v| *v < lo);
    i < sorted.len() && sorted[i] as i32 <= hi
}

impl MoveSnapper {
    pub fn new(doc: &Document, moving: &[LayerId], threshold: f32) -> Self {
        let canvas = doc.canvas_rect();
        let others: Vec<&crate::layer::Layer> =
            doc.layers.iter().filter(|l| l.visible && !moving.contains(&l.id)).collect();
        let views: Vec<LayerView> =
            others.iter().map(|l| LayerView::of(l)).collect();
        let img = composite_views(&views, canvas, None).convert::<u8>();
        let (w, h) = (canvas.w as usize, canvas.h as usize);
        let t = (threshold.clamp(0.0, 1.0) * 255.0).round().max(1.0) as i32;
        let px = |x: usize, y: usize| {
            let i = (y * w + x) * 4;
            &img.data[i..i + 4]
        };
        let differs = |a: &[u8], b: &[u8]| {
            if a[3] == 0 && b[3] == 0 {
                return false;
            }
            (0..4).any(|c| (a[c] as i32 - b[c] as i32).abs() >= t)
        };
        let mut cols = vec![Vec::new(); w + 1];
        let mut rows = vec![Vec::new(); h + 1];
        for y in 0..h {
            for x in 0..w {
                let p = px(x, y);
                if x > 0 && differs(px(x - 1, y), p) {
                    cols[x].push(y as u32);
                }
                if y > 0 && differs(px(x, y - 1), p) {
                    rows[y].push(x as u32);
                }
            }
        }

        // A layer edge counts where that layer is the topmost painted one just inside it.
        let topmost = |x: i32, y: i32| {
            others
                .iter()
                .rev()
                .find(|l| l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1)[3] > 0.0)
                .map(|l| l.id)
        };
        let mut layer_x = vec![0, canvas.w as i32];
        let mut layer_y = vec![0, canvas.h as i32];
        for l in &others {
            let b = l.bounds();
            if b.is_empty() {
                continue;
            }
            let visible_along = |fixed: i32, vertical: bool| {
                let (lo, hi) = if vertical { (b.y, b.bottom()) } else { (b.x, b.right()) };
                let n = 32.min(hi - lo).max(1);
                (0..n).any(|k| {
                    let t = if n <= 1 { lo } else { lo + (hi - lo - 1) * k / (n - 1) };
                    let (x, y) = if vertical { (fixed, t) } else { (t, fixed) };
                    topmost(x, y) == Some(l.id)
                })
            };
            if visible_along(b.x, true) {
                layer_x.push(b.x);
            }
            if visible_along(b.right() - 1, true) {
                layer_x.push(b.right());
            }
            if visible_along(b.y, false) {
                layer_y.push(b.y);
            }
            if visible_along(b.bottom() - 1, false) {
                layer_y.push(b.bottom());
            }
        }
        MoveSnapper { canvas, cols, rows, layer_x, layer_y }
    }

    /// Finds the smallest shift (within `radius`) that puts an edge of
    /// `rect` (canvas coords) on a visible edge, per axis.
    pub fn snap(&self, rect: Rect, radius: f32) -> RectSnap {
        let r = radius.max(1.0).round() as i32;
        let mut out = RectSnap::default();
        // Image content edges only count next to the moving rect (rows it spans
        // for vertical edges, and vice versa); layer edges count anywhere.
        let (y0, y1) = (rect.y - r - self.canvas.y, rect.bottom() + r - self.canvas.y);
        let (x0, x1) = (rect.x - r - self.canvas.x, rect.right() + r - self.canvas.x);
        let mut best_x: Option<(i32, i32)> = None; // (shift, edge)
        let mut best_y: Option<(i32, i32)> = None;
        let consider = |best: &mut Option<(i32, i32)>, edge: i32, at: i32| {
            let d = at - edge;
            if d.abs() <= r && best.is_none_or(|(bd, _)| d.abs() < bd.abs()) {
                *best = Some((d, at));
            }
        };
        for edge in [rect.x, rect.right()] {
            for bx in edge - r..=edge + r {
                let ci = bx - self.canvas.x;
                if ci > 0 && (ci as usize) < self.cols.len() && any_in(&self.cols[ci as usize], y0, y1) {
                    consider(&mut best_x, edge, bx);
                }
            }
            for pos in &self.layer_x {
                consider(&mut best_x, edge, *pos);
            }
        }
        for edge in [rect.y, rect.bottom()] {
            for by in edge - r..=edge + r {
                let ri = by - self.canvas.y;
                if ri > 0 && (ri as usize) < self.rows.len() && any_in(&self.rows[ri as usize], x0, x1) {
                    consider(&mut best_y, edge, by);
                }
            }
            for pos in &self.layer_y {
                consider(&mut best_y, edge, *pos);
            }
        }
        if let Some((d, at)) = best_x {
            out.dx = d;
            out.guide_x = Some(at);
        }
        if let Some((d, at)) = best_y {
            out.dy = d;
            out.guide_y = Some(at);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::pixel::Depth;
    use crate::pixels::Pixels;

    #[test]
    fn move_snaps_to_neighbor_and_ignores_buried() {
        let mut d = Document::new(200, 200, Depth::U8, Some(Color::WHITE));
        let a = d.add_layer_from("a", Pixels::filled(Depth::U8, 50, 50, [1.0, 0.0, 0.0, 1.0]), (20, 20));
        // `buried` is completely covered by `cover`.
        let _buried = d.add_layer_from("b", Pixels::filled(Depth::U8, 30, 30, [0.0, 1.0, 0.0, 1.0]), (120, 120));
        let _cover = d.add_layer_from("c", Pixels::filled(Depth::U8, 60, 60, [0.0, 0.0, 1.0, 1.0]), (110, 110));
        let m = d.add_layer_from("m", Pixels::filled(Depth::U8, 20, 20, [0.0; 4]), (0, 0));
        let _ = a;
        let s = MoveSnapper::new(&d, &[m], 0.05);
        // Left edge 3px right of a's right edge (x=70) → snaps back to 70.
        let r = s.snap(Rect::new(73, 30, 20, 20), 5.0);
        assert_eq!((r.dx, r.guide_x), (-3, Some(70)));
        // Near the buried layer's edge (x=120) but inside the cover: no snap there;
        // the cover's left edge (110) is too far.
        let r = s.snap(Rect::new(123, 140, 10, 10), 4.0);
        assert_eq!(r.guide_x, None);
        // Top aligned with a's top (y=20) from far away horizontally.
        let r = s.snap(Rect::new(150, 22, 10, 10), 4.0);
        assert_eq!((r.dy, r.guide_y), (-2, Some(20)));
    }

    #[test]
    fn snaps_to_square() {
        let mut d = Document::new(40, 40, Depth::U8, Some(Color::WHITE));
        d.add_layer_from("sq", Pixels::filled(Depth::U8, 10, 10, [0.0, 0.0, 0.0, 1.0]), (15, 15));
        let s = snap_point(&d, Vec2::new(13.2, 17.0), SnapParams::default());
        assert!(s.x);
        assert_eq!(s.point.x, 15.0);
        let s = snap_point(&d, Vec2::new(20.0, 26.6), SnapParams::default());
        assert!(s.y);
        assert_eq!(s.point.y, 25.0);
    }
}
