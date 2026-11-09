//! Colors: RGBA in `f32` (straight alpha), HSV conversion and hex codes.
//!
//! Values are stored in the same space as pixel data (no implicit
//! linearization). RGB may exceed 1.0 for HDR colors.

use crate::pixel::Px;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const BLACK: Color = Color::rgba(0.0, 0.0, 0.0, 1.0);
    pub const WHITE: Color = Color::rgba(1.0, 1.0, 1.0, 1.0);
    pub const TRANSPARENT: Color = Color::rgba(0.0, 0.0, 0.0, 0.0);

    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Color { r, g, b, a }
    }

    pub fn from_px(p: Px) -> Self {
        Color::rgba(p[0], p[1], p[2], p[3])
    }

    pub fn px(self) -> Px {
        [self.r, self.g, self.b, self.a]
    }

    pub fn from_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color::rgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a as f32 / 255.0)
    }

    pub fn to_rgba8(self) -> [u8; 4] {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        [q(self.r), q(self.g), q(self.b), q(self.a)]
    }

    pub fn is_hdr(self) -> bool {
        [self.r, self.g, self.b].iter().any(|v| *v > 1.0 || *v < 0.0)
    }

    /// `#RRGGBB` or `#RRGGBBAA` (alpha only when not fully opaque).
    pub fn to_hex(self) -> String {
        let [r, g, b, a] = self.to_rgba8();
        if a == 255 {
            format!("#{r:02X}{g:02X}{b:02X}")
        } else {
            format!("#{r:02X}{g:02X}{b:02X}{a:02X}")
        }
    }

    /// Parses `RGB`, `RGBA`, `RRGGBB` or `RRGGBBAA`, with or without `#`.
    pub fn from_hex(s: &str) -> Option<Color> {
        let s = s.trim().trim_start_matches('#');
        if !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let nib = |i: usize| u8::from_str_radix(&s[i..i + 1], 16).ok().map(|v| v * 17);
        let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
        match s.len() {
            3 => Some(Color::from_rgba8(nib(0)?, nib(1)?, nib(2)?, 255)),
            4 => Some(Color::from_rgba8(nib(0)?, nib(1)?, nib(2)?, nib(3)?)),
            6 => Some(Color::from_rgba8(byte(0)?, byte(2)?, byte(4)?, 255)),
            8 => Some(Color::from_rgba8(byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
            _ => None,
        }
    }

    /// Hue in degrees `[0,360)`, saturation and value. Value may exceed 1 for HDR.
    pub fn to_hsv(self) -> (f32, f32, f32) {
        rgb_to_hsv(self.r, self.g, self.b)
    }

    /// Hue in degrees, saturation and lightness (HSL).
/// Oklab (L, a, b) of the color, treating its channels as sRGB-encoded
    /// (as CSS does). Values outside 0..1 are extended linearly/mirrored.
    pub fn to_oklab(self) -> (f32, f32, f32) {
        let (r, g, b) = (srgb_to_linear(self.r), srgb_to_linear(self.g), srgb_to_linear(self.b));
        let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
        let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
        let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
        let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
        (
            0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
            1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
            0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        )
    }

    pub fn from_oklab(lightness: f32, a: f32, b: f32, alpha: f32) -> Color {
        let l = lightness + 0.396_337_78 * a + 0.215_803_76 * b;
        let m = lightness - 0.105_561_346 * a - 0.063_854_17 * b;
        let s = lightness - 0.089_484_18 * a - 1.291_485_5 * b;
        let (l, m, s) = (l * l * l, m * m * m, s * s * s);
        let r = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
        let g = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
        let bb = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
        Color::rgba(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(bb), alpha)
    }

    pub fn to_hsl(self) -> (f32, f32, f32) {
        let (h, _, _) = rgb_to_hsv(self.r, self.g, self.b);
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let l = (max + min) / 2.0;
        let d = max - min;
        let s = if d <= 1e-9 { 0.0 } else { d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-9) };
        (h, s, l)
    }

    pub fn from_hsl(h: f32, s: f32, l: f32, a: f32) -> Color {
        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let v = l + c / 2.0;
        let sv = if v <= 1e-9 { 0.0 } else { c / v };
        Color::from_hsv(h, sv, v, a)
    }

    pub fn from_hsv(h: f32, s: f32, v: f32, a: f32) -> Color {
        let (r, g, b) = hsv_to_rgb(h, s, v);
        Color::rgba(r, g, b, a)
    }
}

pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d <= 1e-9 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    let s = if max <= 1e-9 { 0.0 } else { d / max };
    (h, s, max)
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (r + m, g + m, b + m)
}

fn srgb_to_linear(v: f32) -> f32 {
    let a = v.abs();
    let l = if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
    l.copysign(v)
}

fn linear_to_srgb(v: f32) -> f32 {
    let a = v.abs();
    let s = if a <= 0.003_130_8 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
    s.copysign(v)
}

#[cfg(test)]
mod tests {
    #[test]
    fn oklab_known_values() {
        let (l, a, b) = super::Color::rgba(1.0, 0.0, 0.0, 1.0).to_oklab();
        assert!((l - 0.627_955).abs() < 1e-3 && (a - 0.224_863).abs() < 1e-3 && (b - 0.125_846).abs() < 1e-3, "{l} {a} {b}");
        let (l, a, b) = super::Color::WHITE.to_oklab();
        assert!((l - 1.0).abs() < 1e-3 && a.abs() < 1e-3 && b.abs() < 1e-3);
        let c = super::Color::rgba(0.2, 0.5, 0.9, 0.7);
        let (l, a, b) = c.to_oklab();
        let d = super::Color::from_oklab(l, a, b, 0.7);
        assert!((c.r - d.r).abs() < 1e-4 && (c.g - d.g).abs() < 1e-4 && (c.b - d.b).abs() < 1e-4 && d.a == 0.7);
    }

    use super::*;

    #[test]
    fn hex() {
        assert_eq!(Color::from_hex("#ff0000").unwrap(), Color::rgba(1.0, 0.0, 0.0, 1.0));
        assert_eq!(Color::from_hex("0f08").unwrap().to_hex(), "#00FF0088");
        assert!(Color::from_hex("zz").is_none());
        assert_eq!(Color::from_hex("#12345678").unwrap().to_hex(), "#12345678");
    }

    #[test]
    fn hsl_roundtrip() {
        let c = Color::rgba(0.2, 0.6, 0.9, 1.0);
        let (h, s, l) = c.to_hsl();
        assert!((l - 0.55).abs() < 1e-5);
        let d = Color::from_hsl(h, s, l, 1.0);
        assert!((c.r - d.r).abs() < 1e-5 && (c.g - d.g).abs() < 1e-5 && (c.b - d.b).abs() < 1e-5, "{d:?}");
        assert_eq!(Color::from_hsl(0.0, 1.0, 0.5, 1.0).to_rgba8(), [255, 0, 0, 255]);
    }

    #[test]
    fn hsv_roundtrip() {
        let c = Color::rgba(0.2, 0.6, 0.9, 1.0);
        let (h, s, v) = c.to_hsv();
        let d = Color::from_hsv(h, s, v, 1.0);
        assert!((c.r - d.r).abs() < 1e-5 && (c.g - d.g).abs() < 1e-5 && (c.b - d.b).abs() < 1e-5);
    }
}
