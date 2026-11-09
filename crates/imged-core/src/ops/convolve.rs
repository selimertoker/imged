//! Convolution with a user-defined kernel.

use rayon::prelude::*;

use crate::buffer::Buffer;
use crate::pixel::{premultiply, unpremultiply, Channel};

/// How pixels outside the image are treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMode {
    /// Repeat the nearest edge pixel.
    Clamp,
    /// Treat outside as transparent.
    Transparent,
    /// Wrap around (tileable images).
    Wrap,
}

/// A square convolution kernel. `out = sum(w * in) / divisor + bias`.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// Odd side length.
    pub size: usize,
    /// Row-major weights, `size * size` entries.
    pub weights: Vec<f32>,
    pub divisor: f32,
    pub bias: f32,
    /// Also convolve the alpha channel. When false, alpha is kept and color
    /// is convolved straight (not premultiplied).
    pub include_alpha: bool,
    pub edge: EdgeMode,
}

impl Kernel {
    pub fn identity(size: usize) -> Self {
        let size = size | 1;
        let mut weights = vec![0.0; size * size];
        weights[size * size / 2] = 1.0;
        Kernel { size, weights, divisor: 1.0, bias: 0.0, include_alpha: true, edge: EdgeMode::Clamp }
    }

    pub fn from_weights(size: usize, weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), size * size);
        let mut k = Kernel { size, weights, ..Kernel::identity(size) };
        k.divisor = k.auto_divisor();
        k
    }

    /// Sum of weights, or 1 if the sum is (near) zero — keeps brightness constant.
    pub fn auto_divisor(&self) -> f32 {
        let s: f32 = self.weights.iter().sum();
        if s.abs() < 1e-6 {
            1.0
        } else {
            s
        }
    }

    /// Resizes to a new odd size, keeping centered weights.
    pub fn resized(&self, size: usize) -> Kernel {
        let size = size.max(1) | 1;
        let mut out = vec![0.0; size * size];
        let (oc, nc) = ((self.size / 2) as isize, (size / 2) as isize);
        for y in 0..size as isize {
            for x in 0..size as isize {
                let (sx, sy) = (x - nc + oc, y - nc + oc);
                if sx >= 0 && sy >= 0 && (sx as usize) < self.size && (sy as usize) < self.size {
                    out[y as usize * size + x as usize] = self.weights[sy as usize * self.size + sx as usize];
                }
            }
        }
        Kernel { size, weights: out, ..self.clone() }
    }

    pub fn presets() -> Vec<(&'static str, Kernel)> {
        let gauss5 = {
            let r = [1.0, 4.0, 6.0, 4.0, 1.0];
            let mut w = Vec::with_capacity(25);
            for a in r {
                for b in r {
                    w.push(a * b);
                }
            }
            Kernel::from_weights(5, w)
        };
        let mut edge = Kernel::from_weights(3, vec![-1., -1., -1., -1., 8., -1., -1., -1., -1.]);
        edge.include_alpha = false;
        let mut emboss = Kernel::from_weights(3, vec![-2., -1., 0., -1., 1., 1., 0., 1., 2.]);
        emboss.include_alpha = false;
        let mut sobel_x = Kernel::from_weights(3, vec![-1., 0., 1., -2., 0., 2., -1., 0., 1.]);
        sobel_x.include_alpha = false;
        sobel_x.bias = 0.5;
        let mut sobel_y = Kernel::from_weights(3, vec![-1., -2., -1., 0., 0., 0., 1., 2., 1.]);
        sobel_y.include_alpha = false;
        sobel_y.bias = 0.5;
        vec![
            ("Identity", Kernel::identity(3)),
            ("Box blur 3×3", Kernel::from_weights(3, vec![1.0; 9])),
            ("Box blur 5×5", Kernel::from_weights(5, vec![1.0; 25])),
            ("Gaussian 3×3", Kernel::from_weights(3, vec![1., 2., 1., 2., 4., 2., 1., 2., 1.])),
            ("Gaussian 5×5", gauss5),
            ("Sharpen", Kernel::from_weights(3, vec![0., -1., 0., -1., 5., -1., 0., -1., 0.])),
            ("Strong sharpen", Kernel::from_weights(3, vec![-1., -1., -1., -1., 9., -1., -1., -1., -1.])),
            ("Edge detect", edge),
            ("Emboss", emboss),
            ("Sobel X", sobel_x),
            ("Sobel Y", sobel_y),
            (
                "Motion blur",
                Kernel::from_weights(5, (0..25).map(|i| if i % 6 == 0 { 1.0 } else { 0.0 }).collect()),
            ),
        ]
    }
}

pub fn convolve<T: Channel>(src: &Buffer<T>, k: &Kernel) -> Buffer<T> {
    let (w, h) = (src.width as i32, src.height as i32);
    if w == 0 || h == 0 || k.size == 0 {
        return src.clone();
    }
    // Work in f32; premultiplied when alpha is convolved so transparent
    // pixels don't bleed their hidden color.
    let f: Vec<[f32; 4]> = src
        .data
        .par_chunks(4)
        .map(|d| {
            let p = [d[0].to_f32(), d[1].to_f32(), d[2].to_f32(), d[3].to_f32()];
            if k.include_alpha {
                premultiply(p)
            } else {
                p
            }
        })
        .collect();
    let r = (k.size / 2) as i32;
    let inv = 1.0 / if k.divisor.abs() < 1e-9 { 1.0 } else { k.divisor };
    let taps: Vec<(i32, i32, f32)> = (0..k.size * k.size)
        .filter(|i| k.weights[*i] != 0.0)
        .map(|i| ((i % k.size) as i32 - r, (i / k.size) as i32 - r, k.weights[i]))
        .collect();
    let mut out = Buffer::<T>::new(src.width, src.height);
    out.data.par_chunks_mut(w as usize * 4).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for &(dx, dy, wt) in &taps {
                let (mut sx, mut sy) = (x + dx, y + dy);
                let p = match k.edge {
                    EdgeMode::Clamp => {
                        sx = sx.clamp(0, w - 1);
                        sy = sy.clamp(0, h - 1);
                        f[(sy * w + sx) as usize]
                    }
                    EdgeMode::Wrap => {
                        sx = sx.rem_euclid(w);
                        sy = sy.rem_euclid(h);
                        f[(sy * w + sx) as usize]
                    }
                    EdgeMode::Transparent => {
                        if sx < 0 || sy < 0 || sx >= w || sy >= h {
                            continue;
                        }
                        f[(sy * w + sx) as usize]
                    }
                };
                for c in 0..4 {
                    acc[c] += p[c] * wt;
                }
            }
            let orig = f[(y * w + x) as usize];
            let mut o = [0.0f32; 4];
            if k.include_alpha {
                o[3] = (acc[3] * inv + k.bias).clamp(0.0, 1.0);
                // Bias is applied to the straight color, so add it after unpremultiplying.
                let u = unpremultiply([acc[0] * inv, acc[1] * inv, acc[2] * inv, o[3]]);
                o[0] = u[0] + k.bias;
                o[1] = u[1] + k.bias;
                o[2] = u[2] + k.bias;
            } else {
                for c in 0..3 {
                    o[c] = acc[c] * inv + k.bias;
                }
                o[3] = orig[3];
            }
            let i = x as usize * 4;
            for c in 0..4 {
                row[i + c] = T::from_f32(o[c]);
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_noop() {
        let mut b = Buffer::<u8>::new(3, 3);
        b.set(1, 1, [1.0, 0.5, 0.0, 1.0]);
        let o = convolve(&b, &Kernel::identity(3));
        assert_eq!(o, b);
    }

    #[test]
    fn blur_spreads() {
        let mut b = Buffer::<f32>::filled(3, 3, [0.0, 0.0, 0.0, 1.0]);
        b.set(1, 1, [9.0, 9.0, 9.0, 1.0]);
        let o = convolve(&b, &Kernel::from_weights(3, vec![1.0; 9]));
        assert!((o.get(0, 0)[0] - 1.0).abs() < 1e-5);
        assert!((o.get(1, 1)[0] - 1.0).abs() < 1e-5);
    }
}
