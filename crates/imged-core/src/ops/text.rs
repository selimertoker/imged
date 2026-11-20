//! Text rendering into coverage masks, using system fonts.

use std::collections::HashMap;

use ab_glyph::{Font, FontVec, PxScale, ScaleFont};

use crate::geom::{Rect, Vec2};
use crate::mask::Mask;

/// What to draw and where.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TextSpec {
    pub text: String,
    pub family: String,
    pub size: f32,
    pub bold: bool,
    pub italic: bool,
    /// Degrees, counter-clockwise, around `pos`.
    pub rotation: f32,
    /// Top-left of the first line (before rotation), canvas coords.
    pub pos: Vec2,
    pub antialias: bool,
    /// Multiplier for the font's line height.
    pub line_spacing: f32,
}

/// Lazily loaded system font database with a cache of parsed fonts.
pub struct FontLibrary {
    db: fontdb::Database,
    cache: HashMap<(String, bool, bool), Option<FontVec>>,
    families: Vec<String>,
    main: Vec<String>,
}

impl Default for FontLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl FontLibrary {
    /// Scans system fonts (can take a moment on systems with many fonts).
    pub fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        let mut families: Vec<String> = db
            .faces()
            .filter_map(|f| f.families.first().map(|(n, _)| n.clone()))
            .collect();
        families.sort_by_key(|s| s.to_lowercase());
        families.dedup();
        let main = main_families(&families);
        FontLibrary { db, cache: HashMap::new(), families, main }
    }

    pub fn families(&self) -> &[String] {
        &self.families
    }

    /// Families without the many per-script / per-language variants that
    /// large collections (Noto, Source Han, …) install; see [`main_families`].
    pub fn main_families(&self) -> &[String] {
        &self.main
    }

    /// A sensible default family present on this system.
    pub fn default_family(&self) -> String {
        for pref in ["DejaVu Sans", "Noto Sans", "Liberation Sans", "Cantarell", "Ubuntu", "FreeSans"] {
            if self.families.iter().any(|f| f == pref) {
                return pref.to_string();
            }
        }
        self.families.first().cloned().unwrap_or_default()
    }

    fn font(&mut self, family: &str, bold: bool, italic: bool) -> Option<&FontVec> {
        let key = (family.to_string(), bold, italic);
        if !self.cache.contains_key(&key) {
            let families = [fontdb::Family::Name(family), fontdb::Family::SansSerif];
            let query = fontdb::Query {
                families: &families,
                weight: if bold { fontdb::Weight::BOLD } else { fontdb::Weight::NORMAL },
                style: if italic { fontdb::Style::Italic } else { fontdb::Style::Normal },
                stretch: fontdb::Stretch::Normal,
            };
            let font = self.db.query(&query).and_then(|id| {
                self.db
                    .with_face_data(id, |data, index| FontVec::try_from_vec_and_index(data.to_vec(), index).ok())
                    .flatten()
            });
            self.cache.insert(key.clone(), font);
        }
        self.cache.get(&key).and_then(|f| f.as_ref())
    }

    /// Rasterizes `spec` into a canvas-space mask. Returns the mask and the
    /// four corners of the (rotated) text box for overlays.
    pub fn render(&mut self, spec: &TextSpec) -> Option<(Mask, [Vec2; 4])> {
        let font = self.font(&spec.family, spec.bold, spec.italic)?;
        let scale = PxScale::from(spec.size.max(1.0));
        let sf = font.as_scaled(scale);
        let line_h = (sf.ascent() - sf.descent() + sf.line_gap()) * spec.line_spacing.max(0.1);

        // Lay out glyphs relative to (0, 0) = top-left of the first line.
        let mut glyphs = Vec::new();
        let mut width: f32 = 0.0;
        let lines: Vec<&str> = spec.text.split('\n').collect();
        for (li, line) in lines.iter().enumerate() {
            let baseline = sf.ascent() + li as f32 * line_h;
            let mut x = 0.0f32;
            let mut prev = None;
            for ch in line.chars() {
                let id = font.glyph_id(ch);
                if let Some(p) = prev {
                    x += sf.kern(p, id);
                }
                glyphs.push(id.with_scale_and_position(scale, ab_glyph::point(x, baseline)));
                x += sf.h_advance(id);
                prev = Some(id);
            }
            width = width.max(x);
        }
        let height = line_h * (lines.len() as f32 - 1.0) + sf.ascent() - sf.descent();
        let pad = 2.0;
        let local = Rect::covering(-pad, -pad, width + pad, height + pad);
        let mut m = Mask::new(local);
        for g in glyphs {
            if let Some(og) = font.outline_glyph(g) {
                let b = og.px_bounds();
                og.draw(|x, y, c| {
                    let (px, py) = (b.min.x as i32 + x as i32, b.min.y as i32 + y as i32);
                    let c = if spec.antialias { c } else if c >= 0.5 { 1.0 } else { 0.0 };
                    if c > 0.0 {
                        m.max_at(px, py, c.min(1.0));
                    }
                });
            }
        }

        let a = -spec.rotation.to_radians();
        let box_corners = [
            Vec2::new(0.0, 0.0),
            Vec2::new(width, 0.0),
            Vec2::new(width, height),
            Vec2::new(0.0, height),
        ]
        .map(|p| spec.pos + p.rotate(a));

        if spec.rotation.rem_euclid(360.0).abs() < 1e-3 {
            let (dx, dy) = (spec.pos.x.round() as i32, spec.pos.y.round() as i32);
            return Some((m.translated(dx, dy), box_corners));
        }

        // Rotate around the anchor by inverse-mapping each output pixel.
        let corners = [
            Vec2::new(local.x as f32, local.y as f32),
            Vec2::new(local.right() as f32, local.y as f32),
            Vec2::new(local.right() as f32, local.bottom() as f32),
            Vec2::new(local.x as f32, local.bottom() as f32),
        ]
        .map(|p| spec.pos + p.rotate(a));
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for c in corners {
            x0 = x0.min(c.x);
            y0 = y0.min(c.y);
            x1 = x1.max(c.x);
            y1 = y1.max(c.y);
        }
        let out_rect = Rect::covering(x0, y0, x1, y1);
        let mut out = Mask::new(out_rect);
        for y in out_rect.y..out_rect.bottom() {
            for x in out_rect.x..out_rect.right() {
                let p = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - spec.pos).rotate(-a);
                let v = sample_mask(&m, p.x, p.y);
                let v = if spec.antialias { v } else if v >= 0.5 { 1.0 } else { 0.0 };
                if v > 0.0 {
                    out.set(x, y, v);
                }
            }
        }
        Some((out, box_corners))
    }
}

/// Drops families that are variants of another family, detected by name:
/// * "Base Word…" is a variant of "Base" when "Base" is itself a family and
///   has at least 4 such variants (e.g. "Noto Sans Thai" → "Noto Sans");
/// * among 8+ families sharing a first word with no such base family
///   (e.g. "Noto Kufi Arabic", "Noto Color Emoji"), those of 3+ words.
pub fn main_families(all: &[String]) -> Vec<String> {
    use std::collections::{HashMap, HashSet};
    let set: HashSet<&str> = all.iter().map(|s| s.as_str()).collect();
    let words = |s: &str| s.split_whitespace().map(str::to_string).collect::<Vec<_>>();
    // Longest proper word-prefix that is also a family.
    let base_of = |f: &str| -> Option<String> {
        let w = words(f);
        (1..w.len()).rev().map(|n| w[..n].join(" ")).find(|p| set.contains(p.as_str()))
    };
    let mut children: HashMap<String, usize> = HashMap::new();
    for f in all {
        if let Some(b) = base_of(f) {
            *children.entry(b).or_default() += 1;
        }
    }
    let mut first_word: HashMap<String, usize> = HashMap::new();
    for f in all {
        if let Some(w) = f.split_whitespace().next() {
            *first_word.entry(w.to_string()).or_default() += 1;
        }
    }
    all.iter()
        .filter(|f| {
            if let Some(b) = base_of(f) {
                if children.get(&b).copied().unwrap_or(0) >= 4 {
                    return false;
                }
            }
            let w = words(f);
            let crowded = w.first().is_some_and(|fw| first_word.get(fw).copied().unwrap_or(0) >= 8);
            let is_base = children.get(f.as_str()).copied().unwrap_or(0) > 0;
            !(crowded && !is_base && w.len() >= 3 && base_of(f).is_none())
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noto_variants_are_hidden() {
        let all: Vec<String> = [
            "DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif", "Noto Sans", "Noto Sans Thai", "Noto Sans Arabic",
            "Noto Sans CJK JP", "Noto Sans Hebrew", "Noto Sans Mono", "Noto Serif", "Noto Serif Thai",
            "Noto Kufi Arabic", "Noto Naskh Arabic", "Noto Color Emoji", "Noto Music", "Liberation Sans",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let main = main_families(&all);
        assert_eq!(
            main,
            ["DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif", "Noto Sans", "Noto Serif", "Noto Serif Thai", "Noto Music", "Liberation Sans"]
        );
    }
}

fn sample_mask(m: &Mask, x: f32, y: f32) -> f32 {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as i32, y0 as i32);
    let top = m.get(x0, y0) * (1.0 - tx) + m.get(x0 + 1, y0) * tx;
    let bot = m.get(x0, y0 + 1) * (1.0 - tx) + m.get(x0 + 1, y0 + 1) * tx;
    top * (1.0 - ty) + bot * ty
}
