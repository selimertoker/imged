//! The OpenCV half of [`super`]: feature matching, the seed search, dense
//! flow and the morphology used to find conflicting regions.
//!
//! It builds against OpenCV 4.x and 5.x. OpenCV 5 moved functions between
//! modules (calib3d and parts of imgproc became `geometry`, features2d became
//! `features`) and added arguments to others, so only functions that are the
//! same in both are called here: `*_def` variants with required arguments
//! only, SIFT from whichever module has it, and the model fitting and polygon
//! code in plain Rust (`stitch_fit.rs`).

use opencv::core::{self, KeyPoint, Mat, Point, Size, Vec2f, Vector};
use opencv::prelude::*;
use opencv::{imgproc, video};
use rayon::prelude::*;

use super::fit::{self, Model};
use super::*;
use crate::geom::Rect;

// OpenCV constants whose module differs between versions (imgproc in 4.x,
// geometry in 5.x); the values are the same in all of them.
const DIST_L2: i32 = 2;
const DIST_MASK_5: i32 = 5;

type R<T> = Result<T, String>;

/// Reports progress; stops with [`CANCELLED`] when asked to.
fn step(progress: Progress, f: f32, what: &str) -> R<()> {
    if progress(f, what) {
        Ok(())
    } else {
        Err(CANCELLED.into())
    }
}

trait Cv<T> {
    fn cv(self) -> R<T>;
}
impl<T> Cv<T> for opencv::Result<T> {
    fn cv(self) -> R<T> {
        self.map_err(|e| format!("OpenCV: {e}"))
    }
}

/// Long side of the images used for matching (full resolution is only used
/// for the final warp and the overlap analysis).
const WORK: f32 = 2400.0;

// ----- small single-channel images ----------------------------------------

#[derive(Clone)]
struct Img<T> {
    w: usize,
    h: usize,
    d: Vec<T>,
}

type Gray = Img<u8>;
type FImg = Img<f32>;

impl<T: core::DataType + Copy + Default> Img<T> {
    fn new(w: usize, h: usize) -> Self {
        Img { w, h, d: vec![T::default(); w * h] }
    }

    fn mat(&self) -> R<Mat> {
        Mat::new_rows_cols_with_data(self.h as i32, self.w as i32, &self.d).cv()?.try_clone().cv()
    }

    fn from_mat(m: &Mat) -> R<Self> {
        let m = if m.is_continuous() { m.clone() } else { m.try_clone().cv()? };
        Ok(Img { w: m.cols() as usize, h: m.rows() as usize, d: m.data_typed::<T>().cv()?.to_vec() })
    }

    #[inline]
    fn at(&self, x: usize, y: usize) -> T {
        self.d[y * self.w + x]
    }
}

fn kernel(d: i32) -> R<Mat> {
    let d = d.max(1) | 1;
    imgproc::get_structuring_element_def(imgproc::MORPH_ELLIPSE, Size::new(d, d)).cv()
}

fn resize<T: core::DataType + Copy + Default>(g: &Img<T>, w: usize, h: usize, interp: i32) -> R<Img<T>> {
    if (w, h) == (g.w, g.h) {
        return Ok(g.clone());
    }
    let mut out = Mat::default();
    imgproc::resize(&g.mat()?, &mut out, Size::new(w.max(1) as i32, h.max(1) as i32), 0.0, 0.0, interp).cv()?;
    Img::from_mat(&out)
}

fn scaled<T: core::DataType + Copy + Default>(g: &Img<T>, s: f32) -> R<Img<T>> {
    let (w, h) = (((g.w as f32 * s).round() as usize).max(1), ((g.h as f32 * s).round() as usize).max(1));
    resize(g, w, h, if s < 1.0 { imgproc::INTER_AREA } else { imgproc::INTER_LINEAR })
}

fn morph<T: core::DataType + Copy + Default>(g: &Img<T>, op: i32, d: i32) -> R<Img<T>> {
    let mut out = Mat::default();
    imgproc::morphology_ex_def(&g.mat()?, &mut out, op, &kernel(d)?).cv()?;
    Img::from_mat(&out)
}

fn blur(g: &FImg, sigma: f32) -> R<FImg> {
    let mut out = Mat::default();
    imgproc::gaussian_blur_def(&g.mat()?, &mut out, Size::new(0, 0), sigma as f64).cv()?;
    Img::from_mat(&out)
}

/// Luminance with transparency composited on white (an empty board).
fn gray_of(p: &Pixels) -> Gray {
    let (w, h) = (p.width() as usize, p.height() as usize);
    let mut d = vec![0u8; w * h];
    crate::dispatch!(p, b => d.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let q = b.get(x as u32, y as u32);
            let l = 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2];
            *v = ((l * q[3] + 1.0 - q[3]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }));
    Img { w, h, d }
}

/// Divides out uneven lighting: the board becomes white, ink stays dark.
fn flatten_light(g: &Gray) -> R<Gray> {
    let k = ((g.w.max(g.h) / 120).max(9)) as i32 | 1;
    let small = scaled(g, 0.25)?;
    let kk = (k / 4).max(3) | 1;
    let closed = morph(&small, imgproc::MORPH_CLOSE, kk)?;
    let f = FImg { w: closed.w, h: closed.h, d: closed.d.iter().map(|&v| v as f32).collect() };
    let bg = resize(&blur(&f, kk as f32 / 2.0)?, g.w, g.h, imgproc::INTER_LINEAR)?;
    Ok(Img { w: g.w, h: g.h, d: g.d.iter().zip(&bg.d).map(|(&v, &b)| ((v as f32 / b.max(1.0)).min(1.0) * 255.0) as u8).collect() })
}

// ----- seed search -----------------------------------------------------------

fn sim(angle_deg: f64, scale: f64, src: [f64; 2], dst: [f64; 2]) -> [f64; 6] {
    let a = angle_deg.to_radians();
    let (c, s) = (scale * a.cos(), scale * a.sin());
    [c, -s, dst[0] - (c * src[0] - s * src[1]), s, c, dst[1] - (s * src[0] + c * src[1])]
}

struct Snap {
    score: f32,
    angle: f64,
    scale: f64,
    /// The fixed-image point matching the moving seed.
    at: [f64; 2],
}

/// Finds the rotation and scale at the clicked letters: B's patch around its
/// seed is rotated and scaled and searched for in a window around A's seed.
fn snap(a: &Gray, b: &Gray, pa: [f64; 2], pb: [f64; 2]) -> R<Snap> {
    let f = 0.5;
    let (a, b) = (scaled(a, f as f32)?, scaled(b, f as f32)?);
    let (pa, pb) = ([pa[0] * f, pa[1] * f], [pb[0] * f, pb[1] * f]);
    let big = a.w.max(a.h) as f64;
    let r = (big * 0.035) as i32;
    let s = (big * 0.05) as i32;
    let t = 2 * r + 1;
    // Both images are inverted (ink bright, board 0), so the zero border that
    // warp_affine_def pads with is empty board; correlation doesn't change.
    // Window around A's seed, padded with empty board.
    let (cx, cy) = (pa[0].round() as i32, pa[1].round() as i32);
    let ws = 2 * (s + r) + 1;
    let mut win = Gray::new(ws as usize, ws as usize);
    for y in 0..ws {
        for x in 0..ws {
            let (sx, sy) = (cx - s - r + x, cy - s - r + y);
            let inside = sx >= 0 && sy >= 0 && (sx as usize) < a.w && (sy as usize) < a.h;
            win.d[(y * ws + x) as usize] = if inside { 255 - a.at(sx as usize, sy as usize) } else { 0 };
        }
    }
    let win = win.mat()?;
    let bm = Gray { w: b.w, h: b.h, d: b.d.iter().map(|&v| 255 - v).collect() }.mat()?;
    let score = |ang: f64, sc: f64| -> R<(f32, f64, f64, Point)> {
        let m = sim(ang, sc, pb, [r as f64, r as f64]);
        let mm = Mat::from_slice_2d(&[[m[0], m[1], m[2]], [m[3], m[4], m[5]]]).cv()?;
        let mut patch = Mat::default();
        imgproc::warp_affine_def(&bm, &mut patch, &mm, Size::new(t, t)).cv()?;
        let mut res = Mat::default();
        imgproc::match_template_def(&win, &patch, &mut res, imgproc::TM_CCOEFF_NORMED).cv()?;
        let (mut mx, mut loc) = (0.0, Point::default());
        core::min_max_loc(&res, None, Some(&mut mx), None, Some(&mut loc), &core::no_array()).cv()?;
        Ok((if mx.is_finite() { mx as f32 } else { -1.0 }, ang, sc, loc))
    };
    let scales: Vec<f64> = (0..11).map(|i| (0.6f64.ln() + (1.7f64.ln() - 0.6f64.ln()) * i as f64 / 10.0).exp()).collect();
    let search = |angles: &[f64]| -> R<(f32, f64, f64, Point)> {
        let mut best = (-2.0, 0.0, 1.0, Point::default());
        for &an in angles {
            for &sc in &scales {
                let c = score(an, sc)?;
                if c.0 > best.0 {
                    best = c;
                }
            }
        }
        Ok(best)
    };
    // Most photos are only a few degrees apart: try those first.
    let mut best = search(&[-8.0, -4.0, 0.0, 4.0, 8.0])?;
    if best.0 < 0.5 {
        let wide: Vec<f64> = (-8..=8).map(|i| i as f64 * 4.0).collect();
        let w = search(&wide)?;
        if w.0 > best.0 {
            best = w;
        }
    }
    for (da, ds) in [(2.0, 1.04f64), (0.5, 1.01)] {
        let (_, ang, sc, _) = best;
        for i in -2..=2 {
            for j in -2..=2 {
                let c = score(ang + i as f64 * da, sc * ds.powi(j))?;
                if c.0 > best.0 {
                    best = c;
                }
            }
        }
    }
    let (sc_, ang, scale, loc) = best;
    let at = [(cx - s + loc.x) as f64 / f, (cy - s + loc.y) as f64 / f];
    Ok(Snap { score: sc_, angle: ang, scale, at })
}

// ----- features and guided matching ----------------------------------------

struct Features {
    pts: Vec<[f32; 2]>,
    desc: Vec<f32>,
}

/// SIFT keypoints and descriptors. SIFT is in `features2d` in OpenCV 4 and
/// in `features` in OpenCV 5.
fn sift(img: &Mat, mask: &Mat) -> R<(Vector<KeyPoint>, Mat)> {
    let mut kps = Vector::<KeyPoint>::new();
    let mut desc = Mat::default();
    opencv::opencv_has_module_features2d! {
        {
            let mut s = opencv::features2d::SIFT::create_def().cv()?;
            s.detect_and_compute(img, mask, &mut kps, &mut desc, false).cv()?;
        } else {
            opencv::opencv_has_module_features! {
                {
                    let mut s = opencv::features::SIFT::create_def().cv()?;
                    s.detect_and_compute(img, mask, &mut kps, &mut desc, false).cv()?;
                } else {
                    return Err("This OpenCV has no SIFT (features2d / features module)".into());
                }
            }
        }
    }
    Ok((kps, desc))
}

fn features(g: &Gray, mask: Option<&Gray>) -> R<Features> {
    let img = g.mat()?;
    let mask = match mask {
        Some(m) => m.mat()?,
        None => Mat::default(),
    };
    let (kps, desc) = sift(&img, &mask)?;
    let pts = kps.iter().map(|k| [k.pt().x, k.pt().y]).collect::<Vec<_>>();
    let desc = if pts.is_empty() { Vec::new() } else { FImg::from_mat(&desc)?.d };
    Ok(Features { pts, desc })
}

/// Uniform grid over points for radius queries.
struct PointGrid {
    cell: f32,
    w: usize,
    h: usize,
    cells: Vec<Vec<u32>>,
}

impl PointGrid {
    fn new(pts: &[[f32; 2]], w: usize, h: usize, cell: f32) -> Self {
        let (gw, gh) = ((w as f32 / cell) as usize + 1, (h as f32 / cell) as usize + 1);
        let mut cells = vec![Vec::new(); gw * gh];
        for (i, p) in pts.iter().enumerate() {
            let (x, y) = ((p[0] / cell).clamp(0.0, (gw - 1) as f32) as usize, (p[1] / cell).clamp(0.0, (gh - 1) as f32) as usize);
            cells[y * gw + x].push(i as u32);
        }
        PointGrid { cell, w: gw, h: gh, cells }
    }

    fn near(&self, pts: &[[f32; 2]], p: [f32; 2], r: f32, out: &mut Vec<u32>) {
        out.clear();
        let x0 = ((p[0] - r) / self.cell).floor().max(0.0) as usize;
        let y0 = ((p[1] - r) / self.cell).floor().max(0.0) as usize;
        let x1 = (((p[0] + r) / self.cell).floor() as isize).clamp(-1, self.w as isize - 1);
        let y1 = (((p[1] + r) / self.cell).floor() as isize).clamp(-1, self.h as isize - 1);
        for y in y0 as isize..=y1 {
            for x in x0 as isize..=x1 {
                for &i in &self.cells[y as usize * self.w + x as usize] {
                    let q = pts[i as usize];
                    if (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) <= r * r {
                        out.push(i);
                    }
                }
            }
        }
    }
}

fn desc_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// For each moving point in `idx`, the fixed point near where `h` predicts
/// it with the closest descriptor (ratio test against the other candidates).
fn guided(h: &Homography, fa: &Features, fb: &Features, grid: &PointGrid, idx: &[usize], tol: f32) -> Vec<(usize, usize)> {
    idx.par_iter()
        .filter_map(|&i| {
            let p = fb.pts[i];
            let (x, y) = h.apply(p[0] as f64, p[1] as f64)?;
            let mut cand = Vec::new();
            grid.near(&fa.pts, [x as f32, y as f32], tol, &mut cand);
            let db = &fb.desc[i * 128..i * 128 + 128];
            let (mut best, mut second, mut bj) = (f32::MAX, f32::MAX, 0usize);
            for &j in &cand {
                let d = desc_dist(&fa.desc[j as usize * 128..j as usize * 128 + 128], db);
                if d < best {
                    second = best;
                    best = d;
                    bj = j as usize;
                } else if d < second {
                    second = d;
                }
            }
            // Ratio 0.85 on distances = 0.7225 on squared distances.
            (best < f32::MAX && best <= 0.7225 * second).then_some((i, bj))
        })
        .collect()
}

/// Robust fit: a homography with enough points, a similarity with few.
fn fit(src: &[[f32; 2]], dst: &[[f32; 2]], thr: f64) -> R<Option<(Homography, Vec<bool>)>> {
    Ok(if src.len() >= 12 {
        fit::ransac(src, dst, Model::Homography, thr)
    } else if src.len() >= 3 {
        fit::ransac(src, dst, Model::Similarity, thr)
    } else {
        None
    })
}

// ----- projected area ------------------------------------------------------

/// The bright projected area around `seed` on a raw (not light-flattened)
/// working image, as a quadrilateral.
fn find_projection(g: &Gray, seed: [f64; 2]) -> R<Option<[[f32; 2]; 4]>> {
    let big = g.w.max(g.h) as f32;
    let fl = FImg { w: g.w, h: g.h, d: g.d.iter().map(|&v| v as f32).collect() };
    let b = blur(&fl, big / 300.0)?;
    let b8 = Gray { w: g.w, h: g.h, d: b.d.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect() };
    let mut bin = Mat::default();
    imgproc::threshold(&b8.mat()?, &mut bin, 0.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU).cv()?;
    let bin = morph(&morph(&Gray::from_mat(&bin)?, imgproc::MORPH_OPEN, (big / 150.0) as i32)?, imgproc::MORPH_CLOSE, (big / 100.0) as i32)?;
    let (mut labels, mut stats, mut cents) = (Mat::default(), Mat::default(), Mat::default());
    imgproc::connected_components_with_stats_def(&bin.mat()?, &mut labels, &mut stats, &mut cents).cv()?;
    let lab = Img::<i32>::from_mat(&labels)?;
    let (sx, sy) = (seed[0].clamp(0.0, (g.w - 1) as f64) as usize, seed[1].clamp(0.0, (g.h - 1) as f64) as usize);
    let l = lab.at(sx, sy);
    if l == 0 {
        return Ok(None);
    }
    let comp = Gray { w: g.w, h: g.h, d: lab.d.iter().map(|&v| if v == l { 255 } else { 0 }).collect() };
    let area = comp.d.iter().filter(|&&v| v != 0).count() as f32 / (g.w * g.h) as f32;
    if !(0.03..0.97).contains(&area) {
        return Ok(None);
    }
    // Boundary pixels of the component, outlined by a quadrilateral.
    let inside = |x: usize, y: usize| comp.d[y * g.w + x] != 0;
    let mut edge = Vec::new();
    for y in 0..g.h {
        for x in 0..g.w {
            if inside(x, y) && (x == 0 || y == 0 || x + 1 == g.w || y + 1 == g.h || !inside(x - 1, y) || !inside(x + 1, y) || !inside(x, y - 1) || !inside(x, y + 1)) {
                edge.push([x as f64, y as f64]);
            }
        }
    }
    Ok(fit::quad_of(&edge))
}

fn shrink_quad(q: &[[f32; 2]; 4], f: f32) -> [[f32; 2]; 4] {
    let c = [q.iter().map(|p| p[0]).sum::<f32>() / 4.0, q.iter().map(|p| p[1]).sum::<f32>() / 4.0];
    q.map(|p| [c[0] + (p[0] - c[0]) * f, c[1] + (p[1] - c[1]) * f])
}

fn quad_mask(w: usize, h: usize, q: &[[f32; 2]; 4]) -> R<Gray> {
    let mut d = vec![0u8; w * h];
    d.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            if in_quad(q, x as f32, y as f32) {
                *v = 255;
            }
        }
    });
    Ok(Img { w, h, d })
}

// ----- align ---------------------------------------------------------------

pub(super) fn align(
    fixed: &Pixels,
    moving: &Pixels,
    seed_fixed: [f32; 2],
    seed_moving: [f32; 2],
    mode: StitchMode,
    progress: Progress,
) -> R<Alignment> {
    step(progress, 0.02, "Preparing the images")?;
    let (ga_full, gb_full) = rayon::join(|| gray_of(fixed), || gray_of(moving));
    let sa = (WORK / ga_full.w.max(ga_full.h) as f32).min(1.0);
    let sb = (WORK / gb_full.w.max(gb_full.h) as f32).min(1.0);
    let (ra, rb) = (scaled(&ga_full, sa)?, scaled(&gb_full, sb)?);
    drop((ga_full, gb_full));
    let (ga, gb) = (flatten_light(&ra)?, flatten_light(&rb)?);
    let pa0 = [seed_fixed[0] as f64 * sa as f64, seed_fixed[1] as f64 * sa as f64];
    let pb0 = [seed_moving[0] as f64 * sb as f64, seed_moving[1] as f64 * sb as f64];

    // Projected document: find the projected area on both photos.
    let (mut quad_a, mut quad_b) = (None, None);
    if mode == StitchMode::Projection {
        step(progress, 0.08, "Finding the projected area")?;
        quad_a = find_projection(&ra, pa0)?;
        quad_b = find_projection(&rb, pb0)?;
        if quad_a.is_none() || quad_b.is_none() {
            return Err("Could not find the projected area around the clicked point. Try \"Board\" mode, or click inside the brightest (projected) part.".into());
        }
    }

    step(progress, 0.12, "Matching the clicked letters")?;
    let sn = snap(&ga, &gb, pa0, pb0)?;
    if sn.score < 0.3 {
        return Err(format!(
            "The clicked spots don't look alike (match {:.0}%). Click the same letter or symbol on both layers.",
            sn.score.max(0.0) * 100.0
        ));
    }
    let s = sim(sn.angle, sn.scale, pb0, sn.at);
    let mut h = Homography([s[0], s[1], s[2], s[3], s[4], s[5], 0.0, 0.0, 1.0]);

    step(progress, 0.2, "Finding features")?;
    // In projection mode, leave out the edges of the projected area: they
    // stay put while the document moves.
    let mask_a = quad_a.map(|q| quad_mask(ga.w, ga.h, &shrink_quad(&q, 0.94))).transpose()?;
    let mask_b = quad_b.map(|q| quad_mask(gb.w, gb.h, &shrink_quad(&q, 0.94))).transpose()?;
    let (fa, fb) = rayon::join(|| features(&ga, mask_a.as_ref()), || features(&gb, mask_b.as_ref()));
    let (fa, fb) = (fa?, fb?);
    if fa.pts.len() < 8 || fb.pts.len() < 8 {
        return Err("Too few details to match on these layers".into());
    }
    let grid = PointGrid::new(&fa.pts, ga.w, ga.h, 24.0);

    step(progress, 0.55, "Matching outward from the clicked letters")?;
    let diag = (ga.w as f32).hypot(ga.h as f32);
    let (mut r, mut tol) = (diag * 0.06, diag * 0.02);
    let mut fitted = false;
    loop {
        let idx: Vec<usize> = (0..fb.pts.len())
            .filter(|&i| (fb.pts[i][0] as f64 - pb0[0]).hypot(fb.pts[i][1] as f64 - pb0[1]) < r as f64)
            .collect();
        let pairs = guided(&h, &fa, &fb, &grid, &idx, tol);
        let src: Vec<[f32; 2]> = pairs.iter().map(|p| fb.pts[p.0]).collect();
        let dst: Vec<[f32; 2]> = pairs.iter().map(|p| fa.pts[p.1]).collect();
        if let Some((hn, inl)) = fit(&src, &dst, (tol / 4.0).max(2.0) as f64)? {
            if inl.iter().filter(|v| **v).count() >= 6 {
                h = hn;
                fitted = true;
            }
        }
        if r > diag * 1.5 {
            break;
        }
        r *= 1.6;
        tol = (tol * 0.6).max(6.0);
    }
    if !fitted {
        return Err("Could not find matching letters around the clicked point".into());
    }

    step(progress, 0.75, "Fitting the residual field")?;
    // Final matches over the whole image, kept where they agree with their neighbors.
    let all: Vec<usize> = (0..fb.pts.len()).collect();
    let pairs = guided(&h, &fa, &fb, &grid, &all, 6.0);
    let res: Vec<[f32; 2]> = pairs
        .iter()
        .map(|&(i, j)| {
            let p = h.apply(fb.pts[i][0] as f64, fb.pts[i][1] as f64).unwrap_or((f64::NAN, f64::NAN));
            [fa.pts[j][0] - p.0 as f32, fa.pts[j][1] - p.1 as f32]
        })
        .collect();
    let dst: Vec<[f32; 2]> = pairs.iter().map(|p| fa.pts[p.1]).collect();
    let consistent: Vec<bool> = (0..pairs.len())
        .into_par_iter()
        .map(|k| {
            if !res[k][0].is_finite() {
                return false;
            }
            // Median residual of the 15 nearest other matches.
            let mut near: Vec<(f32, usize)> = (0..pairs.len())
                .filter(|&m| m != k && res[m][0].is_finite())
                .map(|m| ((dst[m][0] - dst[k][0]).powi(2) + (dst[m][1] - dst[k][1]).powi(2), m))
                .collect();
            let n = near.len().min(15);
            if n < 4 {
                return false;
            }
            near.select_nth_unstable_by(n - 1, |a, b| a.0.total_cmp(&b.0));
            let mut xs: Vec<f32> = near[..n].iter().map(|v| res[v.1][0]).collect();
            let mut ys: Vec<f32> = near[..n].iter().map(|v| res[v.1][1]).collect();
            xs.sort_by(f32::total_cmp);
            ys.sort_by(f32::total_cmp);
            (res[k][0] - xs[n / 2]).hypot(res[k][1] - ys[n / 2]) < 1.5
        })
        .collect();
    let good: Vec<usize> = (0..pairs.len()).filter(|&k| consistent[k]).collect();
    if good.len() < 8 {
        return Err(format!("Only {} matching points were found; the layers may not overlap enough", good.len()));
    }

    // Smooth field by normalized convolution on a padded grid.
    let cellw = 8.0f32;
    let sigma = 60.0 / cellw;
    let pad = (3.0 * sigma).ceil() as usize;
    let (gw, gh) = ((ga.w as f32 / cellw) as usize + 2 + 2 * pad, (ga.h as f32 / cellw) as usize + 2 + 2 * pad);
    let (mut nx, mut ny, mut den) = (FImg::new(gw, gh), FImg::new(gw, gh), FImg::new(gw, gh));
    for &k in &good {
        let (x, y) = ((dst[k][0] / cellw).round() as usize + pad, (dst[k][1] / cellw).round() as usize + pad);
        let i = y.min(gh - 1) * gw + x.min(gw - 1);
        nx.d[i] += res[k][0];
        ny.d[i] += res[k][1];
        den.d[i] += 1.0;
    }
    let (nx, ny, den) = (blur(&nx, sigma)?, blur(&ny, sigma)?, blur(&den, sigma)?);
    let prior = 0.02 * den.d.iter().fold(0.0f32, |a, &b| a.max(b));
    let inv = 1.0 / sa;
    let field = Field {
        origin: [-(pad as f32) * cellw * inv, -(pad as f32) * cellw * inv],
        cell: cellw * inv,
        w: gw,
        h: gh,
        d: (0..gw * gh).map(|i| [nx.d[i] / (den.d[i] + prior) * inv, ny.d[i] / (den.d[i] + prior) * inv]).collect(),
    };

    // Full-resolution homography: fixed = S_a^-1 · h · S_b.
    let hf = Homography::scale(inv as f64, inv as f64).mul(&h).mul(&Homography::scale(sb as f64, sb as f64));
    check_plausible(&hf, moving.width(), moving.height())?;
    let matches: Vec<([f32; 2], [f32; 2])> =
        good.iter().map(|&k| ([dst[k][0] * inv, dst[k][1] * inv], [fb.pts[pairs[k].0][0] / sb, fb.pts[pairs[k].0][1] / sb])).collect();
    let al = Alignment {
        h: hf,
        field,
        matches,
        angle: sn.angle as f32,
        scale: sn.scale as f32,
        seed_score: sn.score,
        residual: 0.0,
        seed_fixed: [sn.at[0] as f32 * inv, sn.at[1] as f32 * inv],
        mode,
        quad_fixed: quad_a.map(|q| q.map(|p| [p[0] * inv, p[1] * inv])),
        quad_moving: quad_b.map(|q| q.map(|p| [p[0] / sb, p[1] / sb])),
    };
    let mut errs: Vec<f32> = al
        .matches
        .iter()
        .filter_map(|(f, m)| al.forward(m[0], m[1]).map(|p| (p[0] - f[0]).hypot(p[1] - f[1])))
        .collect();
    errs.sort_by(f32::total_cmp);
    let residual = errs.get(errs.len() / 2).copied().unwrap_or(0.0);
    step(progress, 1.0, "Aligned")?;
    Ok(Alignment { residual, ..al })
}

/// Rejects fits that put part of the moving photo behind the horizon or
/// stretch it extremely (a bad fit from clustered matches extrapolates like
/// that, and the warped image would be huge and meaningless).
fn check_plausible(h: &Homography, w: u32, ht: u32) -> R<()> {
    let m = &h.0;
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for j in 0..=8 {
        for i in 0..=8 {
            let (x, y) = (w as f64 * i as f64 / 8.0, ht as f64 * j as f64 / 8.0);
            let wv = m[6] * x + m[7] * y + m[8];
            lo = lo.min(wv);
            hi = hi.max(wv);
        }
    }
    if lo <= 0.0 || lo / hi < 0.15 {
        return Err("The alignment came out implausibly stretched. Click a letter nearer the middle of the overlap, or check that it is the same letter on both photos.".into());
    }
    Ok(())
}

// ----- merge ---------------------------------------------------------------

/// Background (paper/board color) of an RGB crop, estimated at 1/4 size by
/// removing strokes with a morphological closing.
struct Background {
    w: usize,
    h: usize,
    planes: [Vec<f32>; 3],
}

impl Background {
    fn new(rgb: &[[f32; 3]], valid: &[bool], w: usize, h: usize, k: i32) -> R<Background> {
        let (sw, sh) = (w.div_ceil(4), h.div_ceil(4));
        let mut planes: [Vec<f32>; 3] = Default::default();
        let kk = (k / 4).max(3) | 1;
        for (c, plane) in planes.iter_mut().enumerate() {
            let mut small = FImg::new(sw, sh);
            small.d.par_chunks_mut(sw).enumerate().for_each(|(sy, row)| {
                for (sx, v) in row.iter_mut().enumerate() {
                    let (mut s, mut n) = (0.0, 0.0);
                    for y in sy * 4..(sy * 4 + 4).min(h) {
                        for x in sx * 4..(sx * 4 + 4).min(w) {
                            let i = y * w + x;
                            if valid[i] {
                                s += rgb[i][c];
                                n += 1.0;
                            }
                        }
                    }
                    *v = if n > 0.0 { s / n } else { f32::NAN };
                }
            });
            fill_holes(&mut small);
            let closed = morph(&small, imgproc::MORPH_CLOSE, kk)?;
            *plane = blur(&closed, kk as f32 / 2.0)?.d;
        }
        Ok(Background { w: sw, h: sh, planes })
    }

    #[inline]
    fn sample(&self, x: usize, y: usize) -> [f32; 3] {
        let fx = ((x as f32 - 1.5) / 4.0).clamp(0.0, (self.w - 1) as f32);
        let fy = ((y as f32 - 1.5) / 4.0).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let mut o = [0.0; 3];
        for (c, p) in self.planes.iter().enumerate() {
            let g = |x: usize, y: usize| p[y * self.w + x];
            let top = g(x0, y0) + (g(x1, y0) - g(x0, y0)) * tx;
            let bot = g(x0, y1) + (g(x1, y1) - g(x0, y1)) * tx;
            o[c] = (top + (bot - top) * ty).max(1e-3);
        }
        o
    }
}

/// Replaces NaN cells (no valid pixels) by spreading their valid
/// neighbors outward, so invalid areas don't darken the background near them.
fn fill_holes(g: &mut FImg) {
    let (w, h) = (g.w, g.h);
    for _ in 0..64 {
        let src = g.d.clone();
        let mut left = false;
        for y in 0..h {
            for x in 0..w {
                if !src[y * w + x].is_nan() {
                    continue;
                }
                let (mut s, mut n) = (0.0, 0.0);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let v = src[ny as usize * w + nx as usize];
                        if !v.is_nan() {
                            s += v;
                            n += 1.0;
                        }
                    }
                }
                if n > 0.0 {
                    g.d[y * w + x] = s / n;
                } else {
                    left = true;
                }
            }
        }
        if !left {
            return;
        }
    }
    let valid: Vec<f32> = g.d.iter().copied().filter(|v| !v.is_nan()).collect();
    let mean = if valid.is_empty() { 0.0 } else { valid.iter().sum::<f32>() / valid.len() as f32 };
    g.d.iter_mut().filter(|v| v.is_nan()).for_each(|v| *v = mean);
}

/// Darkest channel of the color divided by the background, 0..255 (255 =
/// empty board or not valid).
fn reflectance_min(rgb: &[[f32; 3]], valid: &[bool], bg: &Background, w: usize, h: usize) -> Gray {
    let mut d = vec![255u8; w * h];
    d.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let i = y * w + x;
            if valid[i] {
                let b = bg.sample(x, y);
                let m = (0..3).map(|c| (rgb[i][c] / b[c]).clamp(0.0, 1.0)).fold(1.0f32, f32::min);
                *v = (m * 255.0) as u8;
            }
        }
    });
    Img { w, h, d }
}

/// Dense residual motion between the fixed crop and the warped moving crop,
/// smoothed and trusted only near ink, as a field in fixed-layer coordinates.
fn dense_flow(a: &Gray, b: &Gray, both: &[bool], rect: Rect) -> R<Field> {
    let (hw, hh) = (a.w.div_ceil(2), a.h.div_ceil(2));
    if hw < 16 || hh < 16 {
        return Ok(Field::default()); // too small to refine; no correction
    }
    let (a2, b2) = (resize(a, hw, hh, imgproc::INTER_AREA)?, resize(b, hw, hh, imgproc::INTER_AREA)?);
    let mut dis = video::DISOpticalFlow::create(video::DISOpticalFlow_PRESET_MEDIUM).cv()?;
    let mut flow = Mat::default();
    dis.calc(&a2.mat()?, &b2.mat()?, &mut flow).cv()?;
    let flow = if flow.is_continuous() { flow } else { flow.try_clone().cv()? };
    let f = flow.data_typed::<Vec2f>().cv()?;
    // Confidence: ink in the fixed crop, inside the overlap, small motions only.
    let soft = blur(&FImg { w: hw, h: hh, d: a2.d.iter().map(|&v| 255.0 - v as f32).collect() }, 2.0)?;
    let limit = 0.01 * (hw as f32).hypot(hh as f32);
    let sigma = 20.0f32;
    let pad = (3.0 * sigma).ceil() as usize;
    let (pw, ph) = (hw + 2 * pad, hh + 2 * pad);
    let (mut nx, mut ny, mut den) = (FImg::new(pw, ph), FImg::new(pw, ph), FImg::new(pw, ph));
    for y in 0..hh {
        for x in 0..hw {
            let i = y * hw + x;
            let (fx, fy) = (f[i][0], f[i][1]);
            let inside = both[(2 * y).min(a.h - 1) * a.w + (2 * x).min(a.w - 1)];
            let c = (soft.d[i] / 255.0 - 0.08).max(0.0);
            if !inside || c <= 0.0 || !fx.is_finite() || !fy.is_finite() || fx.hypot(fy) > limit {
                continue;
            }
            let j = (y + pad) * pw + x + pad;
            nx.d[j] = fx * c;
            ny.d[j] = fy * c;
            den.d[j] = c;
        }
    }
    let (nx, ny, den) = (blur(&nx, sigma)?, blur(&ny, sigma)?, blur(&den, sigma)?);
    let prior = 0.01 * den.d.iter().fold(0.0f32, |a, &b| a.max(b)) + 1e-6;
    Ok(Field {
        origin: [rect.x as f32 + 0.5 - 2.0 * pad as f32, rect.y as f32 + 0.5 - 2.0 * pad as f32],
        cell: 2.0,
        w: pw,
        h: ph,
        d: (0..pw * ph).map(|i| [2.0 * nx.d[i] / (den.d[i] + prior), 2.0 * ny.d[i] / (den.d[i] + prior)]).collect(),
    })
}

fn bbox(pts: impl Iterator<Item = [f32; 2]>) -> Option<Rect> {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in pts {
        if p[0].is_finite() && p[1].is_finite() {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
    }
    (x0 <= x1).then(|| Rect::covering(x0, y0, x1 + 1.0, y1 + 1.0))
}

/// Points along the edges of a quad (or a w×h rectangle).
fn outline(q: [[f32; 2]; 4]) -> Vec<[f32; 2]> {
    let mut v = Vec::new();
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        for k in 0..64 {
            let t = k as f32 / 64.0;
            v.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    v
}

fn rect_quad(w: u32, h: u32) -> [[f32; 2]; 4] {
    let (w, h) = (w as f32 - 1.0, h as f32 - 1.0);
    [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
}

pub(super) fn merge(fixed: &Pixels, moving: &Pixels, al: &Alignment, opt: MergeOptions, progress: Progress) -> R<StitchOutput> {
    let mut map = Mapping::new(al)?;
    let (fw, fh) = (fixed.width(), fixed.height());
    let fixed_area = match al.quad_fixed {
        Some(q) => bbox(q.into_iter()).unwrap_or_default().intersect(&Rect::from_size(fw, fh)),
        None => Rect::from_size(fw, fh),
    };
    let mq = al.quad_moving.unwrap_or_else(|| rect_quad(moving.width(), moving.height()));
    let big = fw.max(fh) as i32;
    let limit = Rect::from_corners(-3 * big / 2, -3 * big / 2, fw as i32 + 3 * big / 2, fh as i32 + 3 * big / 2);
    let edge: Vec<[f32; 2]> = outline(mq)
        .into_iter()
        .map(|p| al.forward(p[0], p[1]).map(|q| q.map(|v| v.clamp(-1e7, 1e7))))
        .collect::<Option<_>>()
        .ok_or("The alignment puts part of the photo behind the horizon; try aligning again from another letter")?;
    let warped_area = bbox(edge.into_iter()).ok_or("The aligned layer lands outside the image")?.intersect(&limit);
    let canvas = fixed_area.union(&warped_area);
    let ov = fixed_area.intersect(&warped_area);
    if ov.is_empty() {
        return Err("The layers don't overlap after aligning".into());
    }
    // One pixel of real context around the overlap for the distance transforms.
    let o = Rect::new(ov.x - 1, ov.y - 1, ov.w + 2, ov.h + 2);
    let (w, h) = (o.w as usize, o.h as usize);
    let k = ((fw.max(fh) / 120).max(15)) as i32 | 1;

    step(progress, 0.05, "Warping")?;
    let ((argb, ava), (brgb0, bva0)) = rayon::join(|| read_fixed(fixed, &map, o), || read_warped(moving, &map, o));
    step(progress, 0.25, "Lighting")?;
    let bga = Background::new(&argb, &ava, w, h, k)?;
    let rmin_a = reflectance_min(&argb, &ava, &bga, w, h);
    {
        let bgb0 = Background::new(&brgb0, &bva0, w, h, k)?;
        let rmin_b0 = reflectance_min(&brgb0, &bva0, &bgb0, w, h);
        let both: Vec<bool> = ava.iter().zip(&bva0).map(|(a, b)| *a && *b).collect();
        step(progress, 0.35, "Refining the alignment")?;
        map.flow = Some(dense_flow(&rmin_a, &rmin_b0, &both, o)?);
    }
    drop((brgb0, bva0));
    step(progress, 0.5, "Warping")?;
    let (brgb, bva) = read_warped(moving, &map, o);
    let bgb = Background::new(&brgb, &bva, w, h, k)?;
    let rmin_b = reflectance_min(&brgb, &bva, &bgb, w, h);
    let both: Vec<bool> = ava.iter().zip(&bva).map(|(a, b)| *a && *b).collect();

    // ---- where the photos disagree
    step(progress, 0.65, "Comparing the photos")?;
    let (strong, weak) = ((0.70 * 255.0) as u8, (0.88 * 255.0) as u8);
    let board: Vec<bool> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            if !both[i] {
                return false;
            }
            let (x, y) = (i % w, i / w);
            let (a, b) = (bga.sample(x, y), bgb.sample(x, y));
            ((a[0] + a[1] + a[2]) / (b[0] + b[1] + b[2])).ln().abs() < 0.2
        })
        .collect();
    let mask = |f: &dyn Fn(usize) -> bool| Gray { w, h, d: (0..w * h).map(|i| if f(i) { 255 } else { 0 }).collect() };
    let weak_a = morph(&mask(&|i| rmin_a.d[i] < weak), imgproc::MORPH_DILATE, 7)?;
    let weak_b = morph(&mask(&|i| rmin_b.d[i] < weak), imgproc::MORPH_DILATE, 7)?;
    let only_a = morph(&mask(&|i| board[i] && rmin_a.d[i] < strong && weak_b.d[i] == 0), imgproc::MORPH_OPEN, 3)?;
    let only_b = morph(&mask(&|i| board[i] && rmin_b.d[i] < strong && weak_a.d[i] == 0), imgproc::MORPH_OPEN, 3)?;
    drop((weak_a, weak_b, board));
    let grown = morph(&mask(&|i| only_a.d[i] != 0 || only_b.d[i] != 0), imgproc::MORPH_DILATE, k)?;
    let (mut labels, mut stats, mut cents) = (Mat::default(), Mat::default(), Mat::default());
    let n = imgproc::connected_components_with_stats_def(&grown.mat()?, &mut labels, &mut stats, &mut cents).cv()? as usize;
    drop(grown);
    let lab = Img::<i32>::from_mat(&labels)?;
    drop(labels);
    let (mut sum_a, mut sum_b) = (vec![0usize; n], vec![0usize; n]);
    for i in 0..w * h {
        let l = lab.d[i] as usize;
        sum_a[l] += (only_a.d[i] != 0) as usize;
        sum_b[l] += (only_b.d[i] != 0) as usize;
    }
    let min_px = (k * k / 8) as usize;
    let keep: Vec<bool> = (0..n).map(|l| l > 0 && sum_a[l].max(sum_b[l]) >= min_px).collect();
    let take_a: Vec<bool> = (0..n).map(|l| if opt.prefer_upper { opt.fixed_on_top } else { sum_a[l] >= sum_b[l] }).collect();
    let conflicts = keep.iter().filter(|v| **v).count();
    let creg = FImg { w, h, d: lab.d.iter().map(|&l| keep[l as usize] as u8 as f32).collect() };
    let sel = FImg { w, h, d: lab.d.iter().map(|&l| (keep[l as usize] && take_a[l as usize]) as u8 as f32).collect() };
    drop((lab, only_a, only_b));
    let (cs, ss) = (blur(&creg, 3.0)?, blur(&sel, 3.0)?);
    drop((creg, sel));

    // ---- feathering across the overlap
    step(progress, 0.75, "Blending")?;
    let dist = |v: &[bool]| -> R<FImg> {
        let g = Gray { w, h, d: v.iter().map(|&b| b as u8 * 255).collect() };
        let mut out = Mat::default();
        imgproc::distance_transform_def(&g.mat()?, &mut out, DIST_L2, DIST_MASK_5).cv()?;
        FImg::from_mat(&out)
    };
    let (da, db) = (dist(&ava)?, dist(&bva)?);
    let mut rgb = argb;
    let used: Vec<bool> = both;
    rgb.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, px) in row.iter_mut().enumerate() {
            let i = y * w + x;
            if !used[i] {
                continue;
            }
            let wa = da.d[i] / (da.d[i] + db.d[i]).max(1e-6);
            let c = cs.d[i];
            let wr = ss.d[i] + (1.0 - c) * wa;
            let (ba, bb) = (bga.sample(x, y), bgb.sample(x, y));
            for ch in 0..3 {
                let bg = wa * ba[ch] + (1.0 - wa) * bb[ch];
                let ra = (px[ch] / ba[ch]).clamp(0.0, 1.0);
                let rb = (brgb[i][ch] / bb[ch]).clamp(0.0, 1.0);
                px[ch] = bg * (wr * ra + (1.0 - wr) * rb);
            }
        }
    });
    drop(brgb);
    let blend = OverlapBlend { rect: o, rgb, used };

    step(progress, 0.85, "Rendering")?;
    let depth = fixed.depth().max(moving.depth());
    let merged = render_merged(fixed, moving, &map, &blend, canvas, depth);
    drop(blend);
    let warped = opt.want_warped.then(|| {
        let r = warped_area.intersect(&canvas);
        (render_warped(moving, &map, r), r)
    });
    step(progress, 1.0, "Done")?;
    Ok(StitchOutput { merged, merged_rect: canvas, warped, conflicts })
}
