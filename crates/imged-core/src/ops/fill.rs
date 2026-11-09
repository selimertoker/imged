//! Flood fill (paint bucket) and magic-wand selection masks.

use crate::geom::{Rect, Vec2};
use crate::mask::Mask;
use crate::pixel::{max_diff, Px};

/// Parameters for color-region detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionParams {
    /// Max per-channel difference (0..1) from the seed color, alpha included.
    pub tolerance: f32,
    /// Only pixels connected to the seed (otherwise every matching pixel).
    pub contiguous: bool,
    /// Also connect through diagonal neighbors.
    pub diagonal: bool,
}

impl Default for RegionParams {
    fn default() -> Self {
        RegionParams { tolerance: 0.1, contiguous: true, diagonal: false }
    }
}

/// Mask of pixels in `domain` similar to the pixel at `seed`.
/// `get` reads a pixel in the same coordinates as `domain`.
pub fn region_mask(domain: Rect, seed: Vec2, params: RegionParams, get: impl Fn(i32, i32) -> Px) -> Option<Mask> {
    let (sx, sy) = (seed.x.floor() as i32, seed.y.floor() as i32);
    if !domain.contains(sx, sy) {
        return None;
    }
    let target = get(sx, sy);
    let tol = params.tolerance.max(0.0) + 1e-6;
    let matches = |x: i32, y: i32| max_diff(get(x, y), target) <= tol;
    let mut mask = Mask::new(domain);
    if !params.contiguous {
        for y in domain.y..domain.bottom() {
            for x in domain.x..domain.right() {
                if matches(x, y) {
                    mask.set(x, y, 1.0);
                }
            }
        }
        return Some(mask);
    }
    flood(domain, (sx, sy), params.diagonal, matches, |x, y| mask.set(x, y, 1.0));
    Some(mask)
}

/// Generic 4/8-connected flood fill. Calls `visit` once for every reachable
/// pixel for which `passable` is true, starting at `seed`.
pub fn flood(
    domain: Rect,
    seed: (i32, i32),
    diagonal: bool,
    passable: impl Fn(i32, i32) -> bool,
    mut visit: impl FnMut(i32, i32),
) -> usize {
    if !domain.contains(seed.0, seed.1) || !passable(seed.0, seed.1) {
        return 0;
    }
    let w = domain.w as usize;
    let mut seen = vec![false; domain.area() as usize];
    let idx = |x: i32, y: i32| (y - domain.y) as usize * w + (x - domain.x) as usize;
    let mut stack = vec![seed];
    seen[idx(seed.0, seed.1)] = true;
    let mut count = 0;
    let n4 = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let n8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
    let neighbors: &[(i32, i32)] = if diagonal { &n8 } else { &n4 };
    while let Some((x, y)) = stack.pop() {
        visit(x, y);
        count += 1;
        for (dx, dy) in neighbors {
            let (nx, ny) = (x + dx, y + dy);
            if domain.contains(nx, ny) {
                let i = idx(nx, ny);
                if !seen[i] && passable(nx, ny) {
                    seen[i] = true;
                    stack.push((nx, ny));
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_stops_at_wall() {
        // 5x1: white white black white white
        let px = |x: i32, _y: i32| if x == 2 { [0.0, 0.0, 0.0, 1.0] } else { [1.0; 4] };
        let m = region_mask(Rect::from_size(5, 1), Vec2::new(0.5, 0.5), RegionParams::default(), px).unwrap();
        assert_eq!(m.count(), 2);
        let m = region_mask(
            Rect::from_size(5, 1),
            Vec2::new(0.5, 0.5),
            RegionParams { contiguous: false, ..Default::default() },
            px,
        )
        .unwrap();
        assert_eq!(m.count(), 4);
    }
}
