//! Packing rectangles into rows that form a roughly square block
//! (like ImageMagick's `+append` / `-append` combined into a grid).

/// Places items of the given sizes in rows (left to right, rows top to
/// bottom, each row top-aligned) without overlap, choosing the row width so
/// the result is as close to a square as possible. Returns the top-left of
/// each item and the total size.
pub fn pack_square(sizes: &[(u32, u32)], gap: u32) -> (Vec<(i32, i32)>, (u32, u32)) {
    if sizes.is_empty() {
        return (Vec::new(), (0, 0));
    }
    let original: Vec<usize> = (0..sizes.len()).collect();
    let mut by_height = original.clone();
    by_height.sort_by_key(|&i| std::cmp::Reverse(sizes[i].1));

    let area: f64 = sizes.iter().map(|(w, h)| (*w as f64 + gap as f64) * (*h as f64 + gap as f64)).sum();
    let widest = sizes.iter().map(|s| s.0).max().unwrap_or(0) as u64;
    let total = sizes.iter().map(|s| s.0 as u64 + gap as u64).sum::<u64>();
    let mut widths: Vec<u64> = (10..=300).step_by(5).map(|f| (area.sqrt() * f as f64 / 100.0) as u64).collect();
    widths.push(widest);
    widths.push(total);
    // Every prefix of the original order as a row width (exact row breaks).
    let mut acc = 0u64;
    for (w, _) in sizes {
        acc += *w as u64 + gap as u64;
        widths.push(acc);
    }

    let best_for = |order: &[usize]| {
        widths
            .iter()
            .filter(|w| **w >= widest)
            .map(|w| shelf(order, sizes, *w, gap))
            .min_by_key(|(_, (w, h))| (*w.max(h), *w as u64 * *h as u64))
            .unwrap()
    };
    let a = best_for(&original);
    let b = best_for(&by_height);
    let score = |r: &(Vec<(i32, i32)>, (u32, u32))| r.1 .0.max(r.1 .1) as f64;
    // Keep the layer order unless sorting packs noticeably tighter.
    if score(&a) <= score(&b) * 1.08 {
        a
    } else {
        b
    }
}

fn shelf(order: &[usize], sizes: &[(u32, u32)], max_w: u64, gap: u32) -> (Vec<(i32, i32)>, (u32, u32)) {
    let mut pos = vec![(0, 0); sizes.len()];
    let (mut x, mut y, mut row_h, mut total_w) = (0u64, 0u64, 0u64, 0u64);
    for &i in order {
        let (w, h) = (sizes[i].0 as u64, sizes[i].1 as u64);
        if x > 0 && x + w > max_w {
            y += row_h + gap as u64;
            x = 0;
            row_h = 0;
        }
        pos[i] = (x as i32, y as i32);
        total_w = total_w.max(x + w);
        x += w + gap as u64;
        row_h = row_h.max(h);
    }
    (pos, (total_w as u32, (y + row_h) as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlaps(a: (i32, i32, u32, u32), b: (i32, i32, u32, u32)) -> bool {
        a.0 < b.0 + b.2 as i32 && b.0 < a.0 + a.2 as i32 && a.1 < b.1 + b.3 as i32 && b.1 < a.1 + a.3 as i32
    }

    #[test]
    fn nine_squares_make_a_grid() {
        let sizes = vec![(100, 100); 9];
        let (pos, size) = pack_square(&sizes, 0);
        assert_eq!(size, (300, 300));
        for i in 0..9 {
            for j in i + 1..9 {
                let a = (pos[i].0, pos[i].1, 100, 100);
                let b = (pos[j].0, pos[j].1, 100, 100);
                assert!(!overlaps(a, b));
            }
        }
    }

    #[test]
    fn mixed_sizes_no_overlap_and_squarish() {
        let sizes = vec![(400, 300), (200, 500), (300, 300), (640, 480), (100, 80), (250, 250)];
        let (pos, (w, h)) = pack_square(&sizes, 4);
        for i in 0..sizes.len() {
            for j in i + 1..sizes.len() {
                let a = (pos[i].0, pos[i].1, sizes[i].0, sizes[i].1);
                let b = (pos[j].0, pos[j].1, sizes[j].0, sizes[j].1);
                assert!(!overlaps(a, b), "{i} {j}");
            }
        }
        let ratio = w.max(h) as f32 / w.min(h) as f32;
        assert!(ratio < 1.6, "{w}x{h}");
    }
}
