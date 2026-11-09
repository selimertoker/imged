//! Color filters: invert, gamma, brightness, hue, auto levels, and the
//! [`Filter`] enum that bundles them with convolution for commands/previews.

use crate::buffer::Buffer;
use crate::color::{hsv_to_rgb, rgb_to_hsv};
use crate::map_pixels;
use crate::ops::convolve::{convolve, Kernel};
use crate::pixel::{luma, Channel, Px};
use crate::pixels::Pixels;

/// A per-layer color operation. Every variant is pure: `Pixels -> Pixels`.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// Inverts RGB. Alpha is kept. For HDR data, values are mirrored around 0.5.
    Invert,
    /// `out = in^(1/gamma)`; gamma > 1 brightens mid-tones.
    Gamma(f32),
    /// Adds an offset to RGB (-1..1 is the useful range).
    Brightness(f32),
    /// Multiplies RGB around mid-grey (contrast, 1.0 = unchanged).
    Contrast(f32),
    /// Rotates hue by the given number of degrees.
    Hue(f32),
    /// Scales saturation (1.0 = unchanged, 0 = greyscale).
    Saturation(f32),
    /// Rescales brightness so the darkest pixel becomes 0 and the brightest 1.
    AutoLevels,
    /// Maps luma `lo` → 0 and `hi` → 1 (what AutoLevels computes; used to
    /// apply one range to several layers).
    Levels { lo: f32, hi: f32 },
    /// Combined tone adjustment (one undo step for the adjustments panel).
    Adjust(Adjustments),
    /// Turns transparent where the color is within `tolerance` of `color`.
    ColorToAlpha { color: Px, tolerance: f32 },
    Convolve(Kernel),
    /// Exchanges two channels (0 = R, 1 = G, 2 = B, 3 = A).
    SwapChannels(usize, usize),
}

impl Filter {
    pub fn label(&self) -> &'static str {
        match self {
            Filter::Invert => "Invert",
            Filter::Gamma(_) => "Gamma",
            Filter::Brightness(_) => "Brightness",
            Filter::Contrast(_) => "Contrast",
            Filter::Hue(_) => "Hue",
            Filter::Saturation(_) => "Saturation",
            Filter::AutoLevels | Filter::Levels { .. } => "Auto levels",
            Filter::Adjust(_) => "Adjust colors",
            Filter::ColorToAlpha { .. } => "Color to alpha",
            Filter::Convolve(_) => "Convolution",
            Filter::SwapChannels(..) => "Swap channels",
        }
    }

    /// Applies the filter, returning a new buffer of the same depth and size.
    pub fn apply(&self, px: &Pixels) -> Pixels {
        map_pixels!(px, b => self.apply_buffer(b))
    }

    pub fn apply_buffer<T: Channel>(&self, b: &Buffer<T>) -> Buffer<T> {
        let mut out = b.clone();
        match self {
            Filter::Invert => out.map_pixels(invert_px),
            Filter::Gamma(g) => {
                let inv = 1.0 / g.max(1e-4);
                out.map_pixels(|p| [pow_signed(p[0], inv), pow_signed(p[1], inv), pow_signed(p[2], inv), p[3]])
            }
            Filter::Brightness(v) => out.map_pixels(|p| [p[0] + v, p[1] + v, p[2] + v, p[3]]),
            Filter::Contrast(c) => out.map_pixels(|p| {
                [(p[0] - 0.5) * c + 0.5, (p[1] - 0.5) * c + 0.5, (p[2] - 0.5) * c + 0.5, p[3]]
            }),
            Filter::Hue(deg) => out.map_pixels(|p| {
                let (h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
                let (r, g, b) = hsv_to_rgb(h + deg, s, v);
                [r, g, b, p[3]]
            }),
            Filter::Saturation(f) => out.map_pixels(|p| {
                let l = luma(p);
                [l + (p[0] - l) * f, l + (p[1] - l) * f, l + (p[2] - l) * f, p[3]]
            }),
            Filter::AutoLevels => {
                if let Some((lo, hi)) = luma_range(b) {
                    return Filter::Levels { lo, hi }.apply_buffer(b);
                }
            }
            Filter::Levels { lo, hi } => {
                let (lo, hi) = (*lo, *hi);
                let scale = if hi - lo > 1e-9 { 1.0 / (hi - lo) } else { 1.0 };
                out.map_pixels(|p| [(p[0] - lo) * scale, (p[1] - lo) * scale, (p[2] - lo) * scale, p[3]]);
            }
            Filter::Adjust(a) => {
                let a = *a;
                out.map_pixels(move |p| a.apply_px(p))
            }
            Filter::ColorToAlpha { color, tolerance } => {
                let (c, t) = (*color, *tolerance);
                out.map_pixels(|p| {
                    let d = (p[0] - c[0]).abs().max((p[1] - c[1]).abs()).max((p[2] - c[2]).abs());
                    if d <= t {
                        [p[0], p[1], p[2], 0.0]
                    } else {
                        p
                    }
                })
            }
            Filter::Convolve(k) => return convolve(b, k),
            Filter::SwapChannels(a, c) => {
                let (a, c) = (*a, *c);
                if a < 4 && c < 4 && a != c {
                    out.data.chunks_exact_mut(4).for_each(|p| p.swap(a, c));
                }
            }
        }
        out
    }
}

/// Exposure / brightness / contrast / gamma / hue / saturation / alpha in one pass.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Adjustments {
    /// In stops: RGB is multiplied by 2^exposure (0 = unchanged).
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub hue: f32,
    pub saturation: f32,
    /// Added to alpha (-1..1): 1 makes everything opaque, -1 fully
    /// transparent, 0 leaves it unchanged.
    pub alpha: f32,
}

impl Default for Adjustments {
    fn default() -> Self {
        Adjustments { exposure: 0.0, brightness: 0.0, contrast: 1.0, gamma: 1.0, hue: 0.0, saturation: 1.0, alpha: 0.0 }
    }
}

impl Adjustments {
    pub fn is_identity(&self) -> bool {
        *self == Adjustments::default()
    }

    #[inline]
    pub fn apply_px(&self, p: Px) -> Px {
        let mut c = [p[0], p[1], p[2]];
        if self.exposure != 0.0 {
            let k = 2f32.powf(self.exposure);
            c = c.map(|v| v * k);
        }
        if self.hue != 0.0 || self.saturation != 1.0 {
            let (h, s, v) = rgb_to_hsv(c[0], c[1], c[2]);
            let (r, g, b) = hsv_to_rgb(h + self.hue, (s * self.saturation).max(0.0), v);
            c = [r, g, b];
        }
        let inv_g = 1.0 / self.gamma.max(1e-4);
        for v in &mut c {
            if self.gamma != 1.0 {
                *v = pow_signed(*v, inv_g);
            }
            *v = (*v - 0.5) * self.contrast + 0.5 + self.brightness;
        }
        [c[0], c[1], c[2], (p[3] + self.alpha).clamp(0.0, 1.0)]
    }
}

/// The settings of a filter layer: the color filters panel's sliders and
/// toggles, applied to everything below the layer.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LayerFilter {
    pub adjust: Adjustments,
    pub invert: bool,
    /// Luma range mapped to 0..1 (from Auto levels).
    pub levels: Option<(f32, f32)>,
    /// Two channels exchanged (0 = R … 3 = A).
    pub swap: Option<(usize, usize)>,
}

impl LayerFilter {
    pub fn is_identity(&self) -> bool {
        *self == LayerFilter::default()
    }

    #[inline]
    pub fn apply_px(&self, p: Px) -> Px {
        let mut p = p;
        if let Some((lo, hi)) = self.levels {
            let scale = if hi - lo > 1e-9 { 1.0 / (hi - lo) } else { 1.0 };
            for c in &mut p[..3] {
                *c = (*c - lo) * scale;
            }
        }
        if !self.adjust.is_identity() {
            p = self.adjust.apply_px(p);
        }
        if self.invert {
            p = invert_px(p);
        }
        if let Some((a, b)) = self.swap {
            if a < 4 && b < 4 {
                p.swap(a, b);
            }
        }
        p
    }

    /// One-line description for the layer list.
    pub fn summary(&self) -> String {
        let a = &self.adjust;
        let mut parts = Vec::new();
        if a.exposure != 0.0 {
            parts.push(format!("exposure {:+.1}", a.exposure));
        }
        if a.brightness != 0.0 {
            parts.push("brightness".to_string());
        }
        if a.contrast != 1.0 {
            parts.push("contrast".into());
        }
        if a.gamma != 1.0 {
            parts.push("gamma".into());
        }
        if a.hue != 0.0 {
            parts.push("hue".into());
        }
        if a.saturation != 1.0 {
            parts.push("saturation".into());
        }
        if a.alpha != 0.0 {
            parts.push("alpha".into());
        }
        if self.invert {
            parts.push("invert".into());
        }
        if self.levels.is_some() {
            parts.push("levels".into());
        }
        if self.swap.is_some() {
            parts.push("swap".into());
        }
        if parts.is_empty() {
            "no change yet".into()
        } else {
            parts.join(", ")
        }
    }
}

#[inline]
fn invert_px(p: Px) -> Px {
    [1.0 - p[0], 1.0 - p[1], 1.0 - p[2], p[3]]
}

#[inline]
fn pow_signed(v: f32, e: f32) -> f32 {
    if v >= 0.0 {
        v.powf(e)
    } else {
        -(-v).powf(e)
    }
}

/// Minimum and maximum luma over several layers together.
pub fn luma_range_pixels<'a>(layers: impl IntoIterator<Item = &'a Pixels>) -> Option<(f32, f32)> {
    layers
        .into_iter()
        .filter_map(|p| crate::dispatch!(p, b => luma_range(b)))
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

/// Minimum and maximum luma over pixels that aren't fully transparent.
pub fn luma_range<T: Channel>(b: &Buffer<T>) -> Option<(f32, f32)> {
    use rayon::prelude::*;
    b.data
        .par_chunks(4)
        .filter_map(|d| {
            let p = [d[0].to_f32(), d[1].to_f32(), d[2].to_f32(), d[3].to_f32()];
            (p[3] > 0.0).then(|| {
                let l = luma(p);
                (l, l)
            })
        })
        .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::Depth;

    #[test]
    fn auto_levels_stretches() {
        let mut b = Buffer::<f32>::new(2, 1);
        b.set(0, 0, [0.2, 0.2, 0.2, 1.0]);
        b.set(1, 0, [0.6, 0.6, 0.6, 1.0]);
        let o = Filter::AutoLevels.apply_buffer(&b);
        assert!(o.get(0, 0)[0].abs() < 1e-6);
        assert!((o.get(1, 0)[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn swap_channels() {
        let p = Pixels::filled(Depth::U8, 1, 1, [1.0, 0.0, 0.2, 0.6]);
        let o = Filter::SwapChannels(0, 3).apply(&p);
        let v = o.get(0, 0);
        assert!((v[0] - 0.6).abs() < 0.01 && v[3] == 1.0 && v[1] == 0.0);
    }

    #[test]
    fn invert_u16() {
        let p = Pixels::filled(Depth::U16, 1, 1, [0.0, 1.0, 0.25, 0.5]);
        let o = Filter::Invert.apply(&p);
        let v = o.get(0, 0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 0.0);
        assert!((v[3] - 0.5).abs() < 1e-4);
    }
}
