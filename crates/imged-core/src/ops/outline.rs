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
    get: impl Fn(i32, i32) -> Px,
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
    for (_, color) in candidates.into_iter().take(8) {
        let wall = |x: i32, y: i32| max_diff(get(x, y), color) <= tol;
        let mut region = Mask::new(domain);
        let mut escaped = false;
        flood(domain, (sx, sy), false, |x, y| !wall(x, y), |x, y| {
            region.set(x, y, 1.0);
            if x == domain.x || y == domain.y || x == domain.right() - 1 || y == domain.bottom() - 1 {
                escaped = true;
            }
        });
        if escaped || region.count() == 0 {
            continue;
        }

        // 4. Fill holes: flood the outside from the border; anything not reached
        // and not already in the region is enclosed.
        let mut outside = Mask::new(domain);
        let border: Vec<(i32, i32)> = (domain.x..domain.right())
            .flat_map(|x| [(x, domain.y), (x, domain.bottom() - 1)])
            .chain((domain.y..domain.bottom()).flat_map(|y| [(domain.x, y), (domain.right() - 1, y)]))
            .collect();
        for (bx, by) in border {
            if outside.get(bx, by) > 0.0 || region.get(bx, by) > 0.0 {
                continue;
            }
            flood(
                domain,
                (bx, by),
                false,
                |x, y| region.get(x, y) <= 0.0,
                |x, y| outside.set(x, y, 1.0),
            );
        }
        let mut sel = Mask::new(domain);
        for y in domain.y..domain.bottom() {
            for x in domain.x..domain.right() {
                if outside.get(x, y) <= 0.0 {
                    sel.set(x, y, 1.0);
                }
            }
        }
        if params.include_outline {
            // The ring is reachable from outside, so add wall pixels touching the region.
            let snapshot = sel.clone();
            flood_ring(domain, &snapshot, &mut sel, &wall);
        }
        if params.as_rectangle {
            if let Some(b) = sel.bounds() {
                sel = Mask::full(b);
            }
        }
        return Ok(OutlineResult { mask: sel, outline_color: color });
    }
    Err("The outline isn't closed (the region reaches the image border)".into())
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
fn flood_ring(domain: Rect, region: &Mask, sel: &mut Mask, wall: &impl Fn(i32, i32) -> bool) {
    let mut seeds = Vec::new();
    for y in domain.y..domain.bottom() {
        for x in domain.x..domain.right() {
            if region.get(x, y) > 0.0 {
                continue;
            }
            if wall(x, y)
                && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|(dx, dy)| region.get(x + dx, y + dy) > 0.0)
            {
                seeds.push((x, y));
            }
        }
    }
    for s in seeds {
        if sel.get(s.0, s.1) > 0.0 {
            continue;
        }
        flood(domain, s, true, wall, |x, y| sel.set(x, y, 1.0));
    }
}

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
