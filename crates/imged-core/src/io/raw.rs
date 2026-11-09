//! Decoding headerless ("raw") pixel data: you say what the bytes are.

use crate::buffer::Buffer;
use crate::color::{hsv_to_rgb, Color};
use crate::pixel::Px;
use crate::pixels::Pixels;

/// Channel order / color model of the samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawLayout {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
    Argb,
    /// Hue, saturation, lightness (hue scaled to the full sample range).
    Hsl,
    Hsla,
    /// Hue, saturation, value.
    Hsv,
    Hsva,
}

impl RawLayout {
    pub const ALL: [RawLayout; 11] = [
        RawLayout::Gray,
        RawLayout::GrayAlpha,
        RawLayout::Rgb,
        RawLayout::Rgba,
        RawLayout::Bgr,
        RawLayout::Bgra,
        RawLayout::Argb,
        RawLayout::Hsl,
        RawLayout::Hsla,
        RawLayout::Hsv,
        RawLayout::Hsva,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RawLayout::Gray => "Grayscale",
            RawLayout::GrayAlpha => "Grayscale + alpha",
            RawLayout::Rgb => "RGB",
            RawLayout::Rgba => "RGBA",
            RawLayout::Bgr => "BGR",
            RawLayout::Bgra => "BGRA",
            RawLayout::Argb => "ARGB",
            RawLayout::Hsl => "HSL",
            RawLayout::Hsla => "HSLA",
            RawLayout::Hsv => "HSV",
            RawLayout::Hsva => "HSVA",
        }
    }

    pub fn channels(self) -> usize {
        match self {
            RawLayout::Gray => 1,
            RawLayout::GrayAlpha => 2,
            RawLayout::Rgb | RawLayout::Bgr | RawLayout::Hsl | RawLayout::Hsv => 3,
            _ => 4,
        }
    }

    /// Converts one pixel's normalized samples to straight RGBA.
    fn to_rgba(self, s: &[f32]) -> Px {
        match self {
            RawLayout::Gray => [s[0], s[0], s[0], 1.0],
            RawLayout::GrayAlpha => [s[0], s[0], s[0], s[1]],
            RawLayout::Rgb => [s[0], s[1], s[2], 1.0],
            RawLayout::Rgba => [s[0], s[1], s[2], s[3]],
            RawLayout::Bgr => [s[2], s[1], s[0], 1.0],
            RawLayout::Bgra => [s[2], s[1], s[0], s[3]],
            RawLayout::Argb => [s[1], s[2], s[3], s[0]],
            RawLayout::Hsl | RawLayout::Hsla => {
                let c = hsl_to_rgb(s[0] * 360.0, s[1], s[2]);
                [c.0, c.1, c.2, if self == RawLayout::Hsla { s[3] } else { 1.0 }]
            }
            RawLayout::Hsv | RawLayout::Hsva => {
                let c = hsv_to_rgb(s[0] * 360.0, s[1], s[2]);
                [c.0, c.1, c.2, if self == RawLayout::Hsva { s[3] } else { 1.0 }]
            }
        }
    }
}

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = Color::from_hsl(h, s, l, 1.0);
    (c.r, c.g, c.b)
}

/// Sample type of each channel value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawSample {
    U8,
    U16,
    U32,
    F32,
    F64,
}

impl RawSample {
    pub const ALL: [RawSample; 5] = [RawSample::U8, RawSample::U16, RawSample::U32, RawSample::F32, RawSample::F64];

    pub fn label(self) -> &'static str {
        match self {
            RawSample::U8 => "8-bit int",
            RawSample::U16 => "16-bit int",
            RawSample::U32 => "32-bit int",
            RawSample::F32 => "32-bit float",
            RawSample::F64 => "64-bit float",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            RawSample::U8 => 1,
            RawSample::U16 => 2,
            RawSample::U32 | RawSample::F32 => 4,
            RawSample::F64 => 8,
        }
    }

    /// Reads one sample normalized so integers map 0..=max → 0..=1.
    fn read(self, b: &[u8], big_endian: bool) -> f32 {
        macro_rules! num {
            ($t:ty, $n:expr) => {{
                let mut a = [0u8; $n];
                a.copy_from_slice(&b[..$n]);
                if big_endian {
                    <$t>::from_be_bytes(a)
                } else {
                    <$t>::from_le_bytes(a)
                }
            }};
        }
        match self {
            RawSample::U8 => b[0] as f32 / 255.0,
            RawSample::U16 => num!(u16, 2) as f32 / 65535.0,
            RawSample::U32 => (num!(u32, 4) as f64 / u32::MAX as f64) as f32,
            RawSample::F32 => num!(f32, 4),
            RawSample::F64 => num!(f64, 8) as f32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawParams {
    pub width: u32,
    pub height: u32,
    pub layout: RawLayout,
    pub sample: RawSample,
    pub big_endian: bool,
    /// Channels stored one after another (RRR…GGG…) instead of interleaved.
    pub planar: bool,
    /// Bytes to skip at the start of the file.
    pub offset: usize,
    /// Extra bytes at the end of each row (interleaved only).
    pub row_padding: usize,
    /// Rows are stored bottom to top.
    pub flip_vertical: bool,
}

impl Default for RawParams {
    fn default() -> Self {
        RawParams {
            width: 256,
            height: 256,
            layout: RawLayout::Rgb,
            sample: RawSample::U8,
            big_endian: false,
            planar: false,
            offset: 0,
            row_padding: 0,
            flip_vertical: false,
        }
    }
}

impl RawParams {
    pub fn bytes_per_pixel(&self) -> usize {
        self.layout.channels() * self.sample.bytes()
    }

    /// Bytes the image needs, header offset included.
    pub fn expected_len(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        let body = if self.planar {
            w * h * self.bytes_per_pixel()
        } else {
            (w * self.bytes_per_pixel() + self.row_padding) * h
        };
        self.offset + body
    }
}

/// Decodes `bytes` as described by `p`. 8 and 16-bit integer data keep their
/// depth; everything else becomes 32-bit float. A file that is too short is
/// an error; extra trailing bytes are ignored.
pub fn decode(bytes: &[u8], p: &RawParams) -> Result<Pixels, String> {
    if p.width == 0 || p.height == 0 {
        return Err("Width and height must be at least 1".into());
    }
    let need = p.expected_len();
    if bytes.len() < need {
        return Err(format!("The file has {} bytes but these settings need {need}", bytes.len()));
    }
    let (w, h) = (p.width as usize, p.height as usize);
    let ch = p.layout.channels();
    let sb = p.sample.bytes();
    let row_len = w * ch * sb + p.row_padding;
    let plane = w * h * sb;
    let data = &bytes[p.offset..];
    let img = Buffer::<f32>::from_fn(p.width, p.height, |x, y| {
        let (x, y) = (x as usize, y as usize);
        let sy = if p.flip_vertical { h - 1 - y } else { y };
        let mut s = [0.0f32; 4];
        for (c, v) in s.iter_mut().enumerate().take(ch) {
            let at = if p.planar { c * plane + (sy * w + x) * sb } else { sy * row_len + (x * ch + c) * sb };
            *v = p.sample.read(&data[at..at + sb], p.big_endian);
        }
        p.layout.to_rgba(&s[..ch])
    });
    Ok(match p.sample {
        RawSample::U8 => Pixels::U8(img.convert()),
        RawSample::U16 => Pixels::U16(img.convert()),
        _ => Pixels::F32(img),
    })
}

/// Plausible (width, height) pairs for a file of `len` bytes, most
/// square-like first (exact fits only).
pub fn guess_sizes(len: usize, p: &RawParams) -> Vec<(u32, u32)> {
    let bpp = p.bytes_per_pixel();
    if bpp == 0 || len <= p.offset {
        return Vec::new();
    }
    let body = len - p.offset;
    if body % bpp != 0 {
        return Vec::new();
    }
    let n = body / bpp;
    let mut out: Vec<(u32, u32)> = (1..=((n as f64).sqrt() as usize))
        .filter(|w| n % w == 0)
        .flat_map(|w| [(w, n / w), (n / w, w)])
        .filter(|(w, h)| *w >= 8 && *h >= 8)
        .map(|(w, h)| (w as u32, h as u32))
        .collect();
    out.sort_by(|a, b| {
        let r = |s: &(u32, u32)| (s.0.max(s.1) as f64 / s.0.min(s.1) as f64, s.0 < s.1);
        r(a).partial_cmp(&r(b)).unwrap()
    });
    out.dedup();
    out.truncate(12);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_and_bgr16_be() {
        let bytes = [255, 0, 0, 0, 255, 0];
        let p = RawParams { width: 2, height: 1, ..Default::default() };
        let px = decode(&bytes, &p).unwrap();
        assert_eq!(px.get(0, 0), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(px.get(1, 0), [0.0, 1.0, 0.0, 1.0]);
        // BGR, u16 big endian: blue = 0xFFFF.
        let bytes = [0xFF, 0xFF, 0, 0, 0, 0];
        let p = RawParams { width: 1, height: 1, layout: RawLayout::Bgr, sample: RawSample::U16, big_endian: true, ..Default::default() };
        assert_eq!(decode(&bytes, &p).unwrap().get(0, 0), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn planar_float_hsl_and_errors() {
        // Gray f32 planar with header, flipped.
        let mut bytes = vec![9u8; 4];
        for v in [0.25f32, 2.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let p = RawParams {
            width: 1,
            height: 2,
            layout: RawLayout::Gray,
            sample: RawSample::F32,
            offset: 4,
            flip_vertical: true,
            planar: true,
            ..Default::default()
        };
        let px = decode(&bytes, &p).unwrap();
        assert_eq!(px.get(0, 0)[0], 2.0);
        assert_eq!(px.get(0, 1)[0], 0.25);
        // HSL: h=0, s=1, l=0.5 is pure red.
        let p = RawParams { width: 1, height: 1, layout: RawLayout::Hsl, ..Default::default() };
        let c = decode(&[0, 255, 128], &p).unwrap().get(0, 0);
        assert!(c[0] > 0.99 && c[1] < 0.01 && c[2] < 0.01, "{c:?}");
        assert!(decode(&[0, 0], &p).is_err());
        let g = guess_sizes(64 * 32 * 3, &RawParams::default());
        assert_eq!(g[0], (64, 32));
    }
}
