//! Outline detection: find the solid-color border around the clicked point
//! and select everything inside it.
//!
//! Typical uses: a picture with a 1px frame inside a screenshot, a photo
//! pasted on a flat background, a UI panel with a border.
//!
//! Algorithm:
//! 1. Cast 8 rays from the click point and record the colors along each.
//! 2. A color that appears on *every* ray (and differs from the clicked
//!    color) is a candidate outline. Candidates are ranked by how far the
//!    farthest ray had to travel to reach them (innermost first).
//! 3. Flood-fill from the click point, treating the outline color as a wall.
//!    If the fill escapes to the image border, try the next candidate.
//! 4. Fill holes: anything enclosed by the region joins it, so details
//!    inside the picture that happen to share the outline color are kept.

use std::collections::HashMap;

use crate::geom::{Rect, Vec2};
use crate::mask::Mask;
use crate::ops::fill::flood;
use crate::pixel::{max_diff, Px};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineParams {
    /// Color tolerance for recognizing the outline color (0..1).
    pub tolerance: f32,
    /// Add the outline ring itself to the selection.
    pub include_outline: bool,
    /// Select the bounding rectangle of the region instead of its exact shape.
    pub as_rectangle: bool,
    /// Instead of an outline, look for the solid background around the
    /// click and select the smallest rectangle that has only that color
    /// outside it (e.g. the bounding box of text on a plain background).
    pub content_box: bool,
}

impl Default for OutlineParams {
    fn default() -> Self {
        OutlineParams { tolerance: 0.08, include_outline: false, as_rectangle: false, content_box: false }
    }
}

#[derive(Debug)]
pub struct OutlineResult {
    pub mask: Mask,
    pub outline_color: Px,
}

fn quantize(p: Px) -> u32 {
    if p[3] <= 0.0 {
        return u32::MAX; // all transparent pixels are one "color"
    }
    let q = |v: f32| ((v.clamp(0.0, 1.0) * 31.0).round() as u32) & 31;
    q(p[0]) | q(p[1]) << 5 | q(p[2]) << 10 | q(p[3]) << 15
}

pub fn detect_outline(
    domain: Rect,
    seed: Vec2,
    params: OutlineParams,
    get: impl Fn(i32, i32) -> Px + Sync,
) -> Result<OutlineResult, String> {
    let (sx, sy) = (seed.x.floor() as i32, seed.y.floor() as i32);
    if !domain.contains(sx, sy) {
        return Err("Click inside the image".into());
    }
    let seed_color = get(sx, sy);
    let tol = params.tolerance.max(0.0) + 1e-6;
    let first_hits = cast_rays(domain, sx, sy, &get);
    if params.content_box {
        return content_box(domain, (sx, sy), seed_color, tol, &first_hits, &get);
    }

    // 2. Candidates present on all rays, not similar to the seed color.
    let mut candidates: Vec<(usize, Px)> = first_hits[0]
        .iter()
        .filter(|(_, (_, p, _))| max_diff(*p, seed_color) > tol)
        .filter_map(|(key, (_, p, _))| {
            let far = first_hits.iter().map(|m| m.get(key).map(|h| h.0)).collect::<Option<Vec<_>>>()?;
            Some((far.into_iter().max().unwrap_or(0), *p))
        })
        .collect();
    candidates.sort_by_key(|c| c.0);
    if candidates.is_empty() {
        return Err("No solid-color outline surrounds this point".into());
    }

    // 3. Flood inside each candidate's walls until one is closed.
    let (w, h) = (domain.w as usize, domain.h as usize);
    let idx = |x: i32, y: i32| (y - domain.y) as usize * w + (x - domain.x) as usize;
    for (_, color) in candidates.into_iter().take(8) {
        // Which pixels are the wall color, computed once (in parallel).
        let wall = classify(domain, &get, |p| max_diff(p, color) <= tol);
        let is_wall = |x: i32, y: i32| wall[idx(x, y)];
        let mut region = vec![false; w * h];
        let mut b = Bounds::default();
        flood(domain, (sx, sy), false, |x, y| !is_wall(x, y), |x, y| {
            region[idx(x, y)] = true;
            b.add(x, y);
        });
        let Some(bb) = b.rect() else { continue };
        if bb.x == domain.x || bb.y == domain.y || bb.right() == domain.right() || bb.bottom() == domain.bottom() {
            continue; // escaped to the border
        }

        // 4. Fill holes: flood the outside from the border of the region's
        // bounds plus a pixel; anything there not reached is enclosed.
        let win = Rect::new(bb.x - 1, bb.y - 1, bb.w + 2, bb.h + 2);
        let mut outside = vec![false; win.area() as usize];
        let widx = |x: i32, y: i32| (y - win.y) as usize * win.w as usize + (x - win.x) as usize;
        flood(win, (win.x, win.y), false, |x, y| !region[idx(x, y)], |x, y| outside[widx(x, y)] = true);
        let mut sel = vec![false; w * h];
        for y in bb.y..bb.bottom() {
            for x in bb.x..bb.right() {
                sel[idx(x, y)] = !outside[widx(x, y)];
            }
        }
        let mut sb = bb;
        if params.include_outline {
            // The ring: wall pixels touching the selection, and every wall
            // pixel connected to them (diagonals too).
            let mut seeds = Vec::new();
            for y in win.y..win.bottom() {
                for x in win.x..win.right() {
                    if !sel[idx(x, y)]
                        && is_wall(x, y)
                        && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                            .iter()
                            .any(|(dx, dy)| win.contains(x + dx, y + dy) && sel[idx(x + dx, y + dy)])
                    {
                        seeds.push((x, y));
                    }
                }
            }
            let mut ring = vec![false; w * h];
            let mut rb = Bounds::default();
            for s in seeds {
                if !ring[idx(s.0, s.1)] {
                    let mut found = Vec::new();
                    flood(domain, s, true, is_wall, |x, y| found.push((x, y)));
                    for (x, y) in found {
                        ring[idx(x, y)] = true;
                        rb.add(x, y);
                    }
                }
            }
            if let Some(r) = rb.rect() {
                for y in r.y..r.bottom() {
                    for x in r.x..r.right() {
                        sel[idx(x, y)] |= ring[idx(x, y)];
                    }
                }
                sb = sb.union(&r);
            }
        }
        let mask = if params.as_rectangle {
            Mask::full(sb)
        } else {
            let mut m = Mask::new(sb);
            for y in sb.y..sb.bottom() {
                for x in sb.x..sb.right() {
                    if sel[idx(x, y)] {
                        m.set(x, y, 1.0);
                    }
                }
            }
            m
        };
        return Ok(OutlineResult { mask, outline_color: color });
    }
    Err("The outline isn't closed (the region reaches the image border)".into())
}

/// `test` for every pixel of `domain`, row by row in parallel.
fn classify(domain: Rect, get: &(impl Fn(i32, i32) -> Px + Sync), test: impl Fn(Px) -> bool + Sync) -> Vec<bool> {
    use rayon::prelude::*;
    let w = domain.w as usize;
    let mut out = vec![false; w * domain.h as usize];
    out.par_chunks_mut(w.max(1)).enumerate().for_each(|(j, row)| {
        let y = domain.y + j as i32;
        for (i, v) in row.iter_mut().enumerate() {
            *v = test(get(domain.x + i as i32, y));
        }
    });
    out
}

/// Bounding box of visited pixels.
#[derive(Default)]
struct Bounds(Option<(i32, i32, i32, i32)>);

impl Bounds {
    fn add(&mut self, x: i32, y: i32) {
        self.0 = Some(match self.0 {
            None => (x, y, x, y),
            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
        });
    }
    fn rect(&self) -> Option<Rect> {
        self.0.map(|(x0, y0, x1, y1)| Rect::new(x0, y0, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32))
    }
}

const DIRS: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Step 1, rays from the seed in 8 directions: for each, the first step
/// (and position) at which each quantized color appears.
type RayHits = Vec<HashMap<u32, (usize, Px, (i32, i32))>>;

fn cast_rays(domain: Rect, sx: i32, sy: i32, get: &impl Fn(i32, i32) -> Px) -> RayHits {
    let mut first_hits = Vec::with_capacity(8);
    for (dx, dy) in DIRS {
        let mut map = HashMap::new();
        let (mut x, mut y, mut step) = (sx, sy, 0usize);
        while domain.contains(x, y) {
            let p = get(x, y);
            map.entry(quantize(p)).or_insert((step, p, (x, y)));
            x += dx;
            y += dy;
            step += 1;
        }
        first_hits.push(map);
    }
    first_hits
}

/// Content box: the background is the clicked color, or a color that
/// surrounds the click on every side (when the click is on the content);
/// of these, the one covering the largest connected area wins. The result
/// is the smallest rectangle inside that area's extent with only the
/// background color (within `tol`) outside it.
fn content_box(
    domain: Rect,
    seed: (i32, i32),
    seed_color: Px,
    tol: f32,
    hits: &RayHits,
    get: &impl Fn(i32, i32) -> Px,
) -> Result<OutlineResult, String> {
    let mut options = vec![(0usize, seed_color, seed)];
    let mut around: Vec<(usize, Px, (i32, i32))> = hits[0]
        .iter()
        .filter(|(_, (_, p, _))| max_diff(*p, seed_color) > tol)
        .filter_map(|(key, (_, p, at))| {
            let far = hits.iter().map(|m| m.get(key).map(|h| h.0)).collect::<Option<Vec<_>>>()?;
            Some((far.into_iter().max().unwrap_or(0), *p, *at))
        })
        .collect();
    around.sort_by_key(|c| c.0);
    options.extend(around.into_iter().take(3));

    let mut best: Option<(usize, Px, Rect)> = None;
    for (_, color, at) in options {
        let same = |x: i32, y: i32| max_diff(get(x, y), color) <= tol;
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        let n = flood(domain, at, false, same, |x, y| {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        });
        if n > 0 && best.is_none_or(|b| n > b.0) {
            best = Some((n, color, Rect::new(x0, y0, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32)));
        }
    }
    let (_, color, area) = best.ok_or("Nothing found here")?;
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for y in area.y..area.bottom() {
        for x in area.x..area.right() {
            if max_diff(get(x, y), color) > tol {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 {
        return Err("There is only the solid color here".into());
    }
    let r = Rect::new(x0, y0, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32);
    Ok(OutlineResult { mask: Mask::full(r), outline_color: color })
}

/// Adds wall pixels connected to the region's boundary.
#[cfg(test)]
mod tests {
    use super::*;

    /// 12x12 white image with a black frame at [2..10) and noise inside.
    fn framed(x: i32, y: i32) -> Px {
        let on_frame = (2..10).contains(&x) && (2..10).contains(&y) && (x == 2 || x == 9 || y == 2 || y == 9);
        if on_frame {
            [0.0, 0.0, 0.0, 1.0]
        } else if (3..9).contains(&x) && (3..9).contains(&y) {
            // "photo": a gradient that includes a black pixel
            if x == 5 && y == 5 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [x as f32 / 12.0, y as f32 / 12.0, 0.5, 1.0]
            }
        } else {
            [1.0, 1.0, 1.0, 1.0]
        }
    }

    #[test]
    fn finds_framed_picture() {
        let r = detect_outline(Rect::from_size(12, 12), Vec2::new(4.5, 4.5), OutlineParams::default(), framed).unwrap();
        assert_eq!(r.outline_color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(r.mask.bounds(), Some(Rect::new(3, 3, 6, 6)));
        // The black pixel inside the picture is kept.
        assert_eq!(r.mask.get(5, 5), 1.0);
        assert_eq!(r.mask.count(), 36);
    }

    #[test]
    fn content_box_of_text() {
        // Gray background, "text" pixels at (3,4), (6,5) and (8,7), all inside a black frame.
        let img = |x: i32, y: i32| -> Px {
            if x == 1 || y == 1 || x == 10 || y == 10 {
                [0.0, 0.0, 0.0, 1.0]
            } else if [(3, 4), (6, 5), (8, 7)].contains(&(x, y)) {
                [0.9, 0.1, 0.1, 1.0]
            } else {
                [0.5, 0.5, 0.52, 1.0]
            }
        };
        let p = OutlineParams { content_box: true, ..Default::default() };
        let want = Some(Rect::new(3, 4, 6, 4));
        // Clicking the background or the content gives the same box.
        for at in [Vec2::new(2.5, 8.5), Vec2::new(6.5, 5.5)] {
            let r = detect_outline(Rect::from_size(12, 12), at, p, img).unwrap();
            assert_eq!(r.mask.bounds(), want);
        }
    }

    #[test]
    fn include_outline_ring() {
        let p = OutlineParams { include_outline: true, ..Default::default() };
        let r = detect_outline(Rect::from_size(12, 12), Vec2::new(4.5, 4.5), p, framed).unwrap();
        assert_eq!(r.mask.bounds(), Some(Rect::new(2, 2, 8, 8)));
        assert_eq!(r.mask.count(), 64);
    }
}
