//! Robust model fitting and polygon helpers for [`super`], in plain Rust.
//!
//! These used to come from OpenCV, but OpenCV 5 moved them to other modules
//! (calib3d → geometry, imgproc → geometry) and changed their argument
//! order, so they are done here to build against every OpenCV version.

use super::Homography;

/// Small deterministic random generator (xorshift64*), so results repeat.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n.max(1)
    }
}

fn sample(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(k);
    while v.len() < k {
        let i = rng.below(n);
        if !v.contains(&i) {
            v.push(i);
        }
    }
    v
}

// ----- linear algebra --------------------------------------------------------

/// Eigenvector of the smallest eigenvalue of a symmetric 9×9 matrix (Jacobi).
fn smallest_eigenvector(mut a: [[f64; 9]; 9]) -> [f64; 9] {
    let mut v = [[0.0; 9]; 9];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..60 {
        let off: f64 = (0..9).flat_map(|i| (0..9).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..9 {
            for q in p + 1..9 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..9 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..9 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let k = (0..9).min_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap_or(8);
    let mut e = [0.0; 9];
    for (i, row) in v.iter().enumerate() {
        e[i] = row[k];
    }
    e
}

/// Similarity that moves the points' centroid to 0 and their mean distance to √2.
fn normalizer(p: &[[f64; 2]]) -> Homography {
    let n = p.len() as f64;
    let (cx, cy) = (p.iter().map(|q| q[0]).sum::<f64>() / n, p.iter().map(|q| q[1]).sum::<f64>() / n);
    let d = p.iter().map(|q| (q[0] - cx).hypot(q[1] - cy)).sum::<f64>() / n;
    let s = if d > 1e-12 { std::f64::consts::SQRT_2 / d } else { 1.0 };
    Homography([s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0])
}

/// Normalized direct linear transform: the homography taking `src` to `dst`
/// (least squares for more than 4 points).
fn dlt(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Homography> {
    if src.len() < 4 {
        return None;
    }
    let (ts, td) = (normalizer(src), normalizer(dst));
    let mut ata = [[0.0f64; 9]; 9];
    for (s, d) in src.iter().zip(dst) {
        let (x, y) = ts.apply(s[0], s[1])?;
        let (u, v) = td.apply(d[0], d[1])?;
        let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u];
        let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v];
        for i in 0..9 {
            for j in 0..9 {
                ata[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
            }
        }
    }
    let h = Homography(smallest_eigenvector(ata));
    let mut h = td.inverse()?.mul(&h).mul(&ts);
    if h.0[8].abs() > 1e-12 {
        let s = h.0[8];
        h.0.iter_mut().for_each(|v| *v /= s);
    }
    // Points in front of the camera need w > 0 (the overall sign is free).
    let n = src.len() as f64;
    let (cx, cy) = (src.iter().map(|q| q[0]).sum::<f64>() / n, src.iter().map(|q| q[1]).sum::<f64>() / n);
    if h.0[6] * cx + h.0[7] * cy + h.0[8] < 0.0 {
        h.0.iter_mut().for_each(|v| *v = -*v);
    }
    h.0.iter().all(|v| v.is_finite()).then_some(h)
}

/// Least-squares similarity (rotation, uniform scale, translation).
fn similarity(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Homography> {
    let n = src.len() as f64;
    if src.len() < 2 {
        return None;
    }
    let c = |p: &[[f64; 2]]| [p.iter().map(|q| q[0]).sum::<f64>() / n, p.iter().map(|q| q[1]).sum::<f64>() / n];
    let (cs, cd) = (c(src), c(dst));
    let (mut sa, mut sb, mut ss) = (0.0, 0.0, 0.0);
    for (s, d) in src.iter().zip(dst) {
        let (x, y) = (s[0] - cs[0], s[1] - cs[1]);
        let (u, v) = (d[0] - cd[0], d[1] - cd[1]);
        sa += x * u + y * v;
        sb += x * v - y * u;
        ss += x * x + y * y;
    }
    if ss < 1e-12 {
        return None;
    }
    let (a, b) = (sa / ss, sb / ss);
    Some(Homography([a, -b, cd[0] - (a * cs[0] - b * cs[1]), b, a, cd[1] - (b * cs[0] + a * cs[1]), 0.0, 0.0, 1.0]))
}

fn collinear(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let area = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs();
    let scale = (b[0] - a[0]).hypot(b[1] - a[1]).max((c[0] - a[0]).hypot(c[1] - a[1])).max(1e-9);
    area < 1e-3 * scale * scale
}

/// Which model [`ransac`] fits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Model {
    Homography,
    Similarity,
}

/// Fits `model` to the pairs robustly; `thr` is the inlier distance in
/// destination pixels. Returns the model and which pairs are inliers.
pub(crate) fn ransac(src: &[[f32; 2]], dst: &[[f32; 2]], model: Model, thr: f64) -> Option<(Homography, Vec<bool>)> {
    let src: Vec<[f64; 2]> = src.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let dst: Vec<[f64; 2]> = dst.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let n = src.len();
    let k = if model == Model::Homography { 4 } else { 2 };
    if n < k {
        return None;
    }
    let fit = |idx: &[usize]| -> Option<Homography> {
        let s: Vec<[f64; 2]> = idx.iter().map(|&i| src[i]).collect();
        let d: Vec<[f64; 2]> = idx.iter().map(|&i| dst[i]).collect();
        match model {
            Model::Homography => dlt(&s, &d),
            Model::Similarity => similarity(&s, &d),
        }
    };
    let thr2 = thr * thr;
    let inliers = |h: &Homography| -> Vec<bool> {
        src.iter()
            .zip(&dst)
            .map(|(s, d)| h.apply(s[0], s[1]).is_some_and(|(x, y)| (x - d[0]).powi(2) + (y - d[1]).powi(2) <= thr2))
            .collect()
    };
    let count = |m: &[bool]| m.iter().filter(|v| **v).count();

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let (mut best, mut best_n) = (None::<(Homography, Vec<bool>)>, 0usize);
    let mut iters = 3000usize;
    let mut it = 0;
    while it < iters {
        it += 1;
        let idx = sample(&mut rng, n, k);
        if k == 4 {
            let deg = |p: &Vec<[f64; 2]>| {
                (0..4).any(|a| (a + 1..4).any(|b| (b + 1..4).any(|c| collinear(p[idx[a]], p[idx[b]], p[idx[c]]))))
            };
            if deg(&src) || deg(&dst) {
                continue;
            }
        }
        let Some(h) = fit(&idx) else { continue };
        let m = inliers(&h);
        let c = count(&m);
        if c > best_n {
            best_n = c;
            best = Some((h, m));
            // Adaptive stop: enough samples to hit an all-inlier set with 99.5%.
            let w = (c as f64 / n as f64).powi(k as i32);
            if w > 1e-9 {
                iters = iters.min(((1.0 - 0.995f64).ln() / (1.0 - w).max(1e-12).ln()).ceil().max(50.0) as usize);
            }
        }
    }
    let (mut h, mut m) = best?;
    // Refit on the inliers a few times.
    for _ in 0..4 {
        let idx: Vec<usize> = (0..n).filter(|&i| m[i]).collect();
        let Some(h2) = fit(&idx) else { break };
        let m2 = inliers(&h2);
        if count(&m2) < count(&m) {
            break;
        }
        (h, m) = (h2, m2);
    }
    Some((h, m))
}

// ----- polygons --------------------------------------------------------------

fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Convex hull (Andrew's monotone chain), counter-clockwise in a y-up frame.
pub(crate) fn convex_hull(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut p = pts.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut h: Vec<[f64; 2]> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = h.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in iter {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

/// Douglas–Peucker on an open chain; keeps the first and last points.
fn simplify(chain: &[[f64; 2]], eps: f64, out: &mut Vec<[f64; 2]>) {
    if chain.len() < 3 {
        out.extend_from_slice(&chain[..chain.len().saturating_sub(1)]);
        return;
    }
    let (a, b) = (chain[0], chain[chain.len() - 1]);
    let (k, d) = (1..chain.len() - 1).map(|i| (i, seg_dist(chain[i], a, b))).max_by(|x, y| x.1.total_cmp(&y.1)).unwrap();
    if d > eps {
        simplify(&chain[..=k], eps, out);
        simplify(&chain[k..], eps, out);
    } else {
        out.push(a);
    }
}

/// Simplifies a closed polygon (like OpenCV's approxPolyDP with closed=true).
fn approx_closed(poly: &[[f64; 2]], eps: f64) -> Vec<[f64; 2]> {
    let n = poly.len();
    if n < 4 {
        return poly.to_vec();
    }
    // Split at the two points farthest apart.
    let (mut i0, mut i1, mut best) = (0, 0, -1.0);
    for i in 0..n {
        for j in i + 1..n {
            let d = (poly[i][0] - poly[j][0]).hypot(poly[i][1] - poly[j][1]);
            if d > best {
                (i0, i1, best) = (i, j, d);
            }
        }
    }
    let first: Vec<[f64; 2]> = poly[i0..=i1].to_vec();
    let second: Vec<[f64; 2]> = poly[i1..].iter().chain(poly[..=i0].iter()).copied().collect();
    let mut out = Vec::new();
    simplify(&first, eps, &mut out);
    simplify(&second, eps, &mut out);
    out
}

/// Smallest-area rectangle around a convex polygon (corners in order).
fn min_area_rect(hull: &[[f64; 2]]) -> [[f64; 2]; 4] {
    let mut best = (f64::MAX, [[0.0; 2]; 4]);
    for i in 0..hull.len() {
        let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
        let l = (b[0] - a[0]).hypot(b[1] - a[1]);
        if l < 1e-12 {
            continue;
        }
        let (ux, uy) = ((b[0] - a[0]) / l, (b[1] - a[1]) / l);
        let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in hull {
            let (u, v) = (p[0] * ux + p[1] * uy, -p[0] * uy + p[1] * ux);
            (u0, u1, v0, v1) = (u0.min(u), u1.max(u), v0.min(v), v1.max(v));
        }
        let area = (u1 - u0) * (v1 - v0);
        if area < best.0 {
            let at = |u: f64, v: f64| [u * ux - v * uy, u * uy + v * ux];
            best = (area, [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)]);
        }
    }
    best.1
}

/// The quadrilateral that best outlines a blob given by its contour points.
pub(crate) fn quad_of(contour: &[[f64; 2]]) -> Option<[[f32; 2]; 4]> {
    let hull = convex_hull(contour);
    if hull.len() < 3 {
        return None;
    }
    let len: f64 = (0..hull.len()).map(|i| {
        let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
        (b[0] - a[0]).hypot(b[1] - a[1])
    }).sum();
    let mut eps = 0.01 * len;
    for _ in 0..12 {
        let ap = approx_closed(&hull, eps);
        if ap.len() == 4 {
            return Some([ap[0], ap[1], ap[2], ap[3]].map(|p| [p[0] as f32, p[1] as f32]));
        }
        if ap.len() < 4 {
            break;
        }
        eps *= 1.4;
    }
    Some(min_area_rect(&hull).map(|p| [p[0] as f32, p[1] as f32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ransac_recovers_homography_with_outliers() {
        let h = Homography([0.8, 0.1, 40.0, -0.05, 1.1, 12.0, 2e-4, -1e-4, 1.0]);
        let mut rng = Rng(7);
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..300 {
            let p = [rng.below(1000) as f64, rng.below(800) as f64];
            let (x, y) = h.apply(p[0], p[1]).unwrap();
            let q = if i % 3 == 0 { [rng.below(1000) as f64, rng.below(800) as f64] } else { [x + 0.3, y - 0.2] };
            src.push([p[0] as f32, p[1] as f32]);
            dst.push([q[0] as f32, q[1] as f32]);
        }
        let (e, m) = ransac(&src, &dst, Model::Homography, 2.0).unwrap();
        assert!(m.iter().filter(|v| **v).count() >= 195);
        for p in [[100.0, 100.0], [900.0, 700.0], [500.0, 50.0]] {
            let (a, b) = (h.apply(p[0], p[1]).unwrap(), e.apply(p[0], p[1]).unwrap());
            assert!((a.0 - b.0).hypot(a.1 - b.1) < 1.0, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn ransac_similarity() {
        let a = 0.3f64;
        let h = Homography([1.2 * a.cos(), -1.2 * a.sin(), 5.0, 1.2 * a.sin(), 1.2 * a.cos(), -7.0, 0.0, 0.0, 1.0]);
        let src: Vec<[f32; 2]> = (0..20).map(|i| [(i * 37 % 200) as f32, (i * 53 % 150) as f32]).collect();
        let mut dst: Vec<[f32; 2]> = src.iter().map(|p| { let (x, y) = h.apply(p[0] as f64, p[1] as f64).unwrap(); [x as f32, y as f32] }).collect();
        dst[3] = [0.0, 0.0];
        let (e, m) = ransac(&src, &dst, Model::Similarity, 1.0).unwrap();
        assert!(!m[3] && m.iter().filter(|v| **v).count() == 19);
        let (x, y) = e.apply(50.0, 60.0).unwrap();
        let (x2, y2) = h.apply(50.0, 60.0).unwrap();
        assert!((x - x2).hypot(y - y2) < 1e-6);
    }

    #[test]
    fn quad_of_a_tilted_quadrilateral() {
        let q = [[10.0, 20.0], [300.0, 40.0], [280.0, 250.0], [30.0, 220.0]];
        // Dense contour along the edges.
        let mut c = Vec::new();
        for i in 0..4 {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            for k in 0..50 {
                let t = k as f64 / 50.0;
                c.push([(a[0] + (b[0] - a[0]) * t).round(), (a[1] + (b[1] - a[1]) * t).round()]);
            }
        }
        let r = quad_of(&c).unwrap();
        for p in q {
            assert!(r.iter().any(|s| (s[0] as f64 - p[0]).hypot(s[1] as f64 - p[1]) < 2.0), "{p:?} not in {r:?}");
        }
    }
}
