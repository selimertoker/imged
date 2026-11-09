//! Channel types and color depths.
//!
//! Every pixel algorithm in the crate is written once, generic over [`Channel`].
//! A channel knows how to convert itself to and from `f32`, where `0.0..=1.0`
//! is the displayable range. Integer channels clamp on conversion; `f32`
//! channels keep out-of-range (HDR) values untouched.

use std::fmt;

/// Storage depth of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Depth {
    U8,
    U16,
    F32,
}

impl Depth {
    pub const ALL: [Depth; 3] = [Depth::U8, Depth::U16, Depth::F32];

    pub fn label(self) -> &'static str {
        match self {
            Depth::U8 => "8-bit",
            Depth::U16 => "16-bit",
            Depth::F32 => "32-bit float",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Depth::U8 => "u8",
            Depth::U16 => "u16",
            Depth::F32 => "f32",
        }
    }

    pub fn bytes_per_channel(self) -> usize {
        match self {
            Depth::U8 => 1,
            Depth::U16 => 2,
            Depth::F32 => 4,
        }
    }

    pub fn parse(s: &str) -> Option<Depth> {
        match s.to_ascii_lowercase().as_str() {
            "8" | "u8" | "8bit" | "8-bit" => Some(Depth::U8),
            "16" | "u16" | "16bit" | "16-bit" => Some(Depth::U16),
            "32" | "f32" | "float" | "32f" | "32-bit float" => Some(Depth::F32),
            _ => None,
        }
    }
}

impl fmt::Display for Depth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A single color channel value.
pub trait Channel:
    Copy + Clone + Default + PartialEq + Send + Sync + fmt::Debug + 'static
{
    const DEPTH: Depth;
    /// The value representing full intensity / full opacity.
    const ONE: Self;
    fn to_f32(self) -> f32;
    /// Converts from `f32`. Integer channels round and clamp to `0..=1`;
    /// float channels store the value as-is (NaN becomes 0).
    fn from_f32(v: f32) -> Self;
    /// Wraps a buffer of this channel type into the runtime-depth [`Pixels`](crate::pixels::Pixels).
    fn wrap(b: crate::buffer::Buffer<Self>) -> crate::pixels::Pixels;
}

impl Channel for u8 {
    const DEPTH: Depth = Depth::U8;
    fn wrap(b: crate::buffer::Buffer<Self>) -> crate::pixels::Pixels {
        crate::pixels::Pixels::U8(b)
    }
    const ONE: Self = u8::MAX;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 * (1.0 / 255.0)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }
}

impl Channel for u16 {
    const DEPTH: Depth = Depth::U16;
    fn wrap(b: crate::buffer::Buffer<Self>) -> crate::pixels::Pixels {
        crate::pixels::Pixels::U16(b)
    }
    const ONE: Self = u16::MAX;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 * (1.0 / 65535.0)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
    }
}

impl Channel for f32 {
    const DEPTH: Depth = Depth::F32;
    fn wrap(b: crate::buffer::Buffer<Self>) -> crate::pixels::Pixels {
        crate::pixels::Pixels::F32(b)
    }
    const ONE: Self = 1.0;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        if v.is_nan() {
            0.0
        } else {
            v
        }
    }
}

/// An RGBA pixel in `f32`, straight (non-premultiplied) alpha.
pub type Px = [f32; 4];

#[inline(always)]
pub fn premultiply(p: Px) -> Px {
    [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]]
}

#[inline(always)]
pub fn unpremultiply(p: Px) -> Px {
    if p[3] <= 1e-8 {
        [0.0, 0.0, 0.0, 0.0]
    } else {
        let inv = 1.0 / p[3];
        [p[0] * inv, p[1] * inv, p[2] * inv, p[3]]
    }
}

/// Porter-Duff "source over destination" for straight-alpha pixels.
#[inline(always)]
pub fn over(src: Px, dst: Px) -> Px {
    let sa = src[3];
    if sa >= 1.0 {
        return src;
    }
    if sa <= 0.0 {
        return dst;
    }
    let da = dst[3] * (1.0 - sa);
    let a = sa + da;
    if a <= 1e-8 {
        return [0.0; 4];
    }
    let inv = 1.0 / a;
    [
        (src[0] * sa + dst[0] * da) * inv,
        (src[1] * sa + dst[1] * da) * inv,
        (src[2] * sa + dst[2] * da) * inv,
        a,
    ]
}

/// Linear interpolation in premultiplied space (no color fringes at alpha edges).
#[inline(always)]
pub fn lerp_premul(a: Px, b: Px, t: f32) -> Px {
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let pa = premultiply(a);
    let pb = premultiply(b);
    unpremultiply([
        pa[0] + (pb[0] - pa[0]) * t,
        pa[1] + (pb[1] - pa[1]) * t,
        pa[2] + (pb[2] - pa[2]) * t,
        pa[3] + (pb[3] - pa[3]) * t,
    ])
}

/// Rec. 709 luma of the color channels (alpha ignored).
#[inline(always)]
pub fn luma(p: Px) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// Largest per-channel absolute difference, alpha included.
#[inline(always)]
pub fn max_diff(a: Px, b: Px) -> f32 {
    // Fully transparent pixels compare equal regardless of their hidden color.
    if a[3] <= 0.0 && b[3] <= 0.0 {
        return 0.0;
    }
    (a[0] - b[0])
        .abs()
        .max((a[1] - b[1]).abs())
        .max((a[2] - b[2]).abs())
        .max((a[3] - b[3]).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_u8() {
        for v in 0..=255u8 {
            assert_eq!(u8::from_f32(v.to_f32()), v);
        }
    }

    #[test]
    fn f32_keeps_hdr() {
        assert_eq!(f32::from_f32(12.5), 12.5);
        assert_eq!(u16::from_f32(12.5), u16::MAX);
    }

    #[test]
    fn over_opaque() {
        assert_eq!(over([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 1.0]);
        let r = over([1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]);
        assert!((r[0] - 0.5).abs() < 1e-6 && (r[2] - 0.5).abs() < 1e-6 && r[3] == 1.0);
    }
}
