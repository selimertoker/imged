//! Editable objects: shapes and text that stay editable on their own layer.
//!
//! An object layer is an ordinary layer whose pixels are a rendering of its
//! [`LayerObject`], so compositing, export and the renderers don't need to
//! know about objects. Editing the object renders it again. Pixel edits
//! (filters, transforms, painting) turn the layer into plain pixels, and
//! undoing them brings the object back (see [`crate::layer::Patch`]).
//! Moving the layer keeps the object: its geometry is stored as it was when
//! the layer was at `origin`.

use crate::color::Color;
use crate::geom::{Rect, Vec2};
use crate::mask::Mask;
use crate::ops::paint::{Shape, ShapeFill};
use crate::ops::text::{FontLibrary, TextSpec};
use crate::pixel::{over, Depth};
use crate::pixels::Pixels;

/// What an object layer shows. Geometry is in canvas coordinates.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ObjectKind {
    Shape { shape: Shape, color: Color, fill_color: Color },
    Text { spec: TextSpec, color: Color },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerObject {
    pub kind: ObjectKind,
    /// The layer's offset when `kind`'s geometry was last placed.
    pub origin: (i32, i32),
}

impl ObjectKind {
    /// The same object moved by (dx, dy).
    pub fn translated(&self, dx: f32, dy: f32) -> ObjectKind {
        let d = Vec2::new(dx, dy);
        match self {
            ObjectKind::Shape { shape, color, fill_color } => {
                let mut s = *shape;
                s.a = s.a + d;
                s.b = s.b + d;
                ObjectKind::Shape { shape: s, color: *color, fill_color: *fill_color }
            }
            ObjectKind::Text { spec, color } => {
                let mut s = spec.clone();
                s.pos = s.pos + d;
                ObjectKind::Text { spec: s, color: *color }
            }
        }
    }

    /// A name for the layer and the history ("Rectangle", "Text: Hello").
    pub fn label(&self) -> String {
        match self {
            ObjectKind::Shape { shape, .. } => shape.kind.label().to_string(),
            ObjectKind::Text { spec, .. } => {
                let first: String = spec.text.lines().next().unwrap_or("").chars().take(24).collect();
                format!("Text: {first}")
            }
        }
    }

    pub fn is_text(&self) -> bool {
        matches!(self, ObjectKind::Text { .. })
    }

    /// Renders the object into a new buffer of `depth`. Returns the pixels
    /// and where they go on the canvas.
    pub fn render(&self, depth: Depth, fonts: Option<&mut FontLibrary>) -> Result<(Pixels, (i32, i32)), String> {
        let parts: Vec<(Mask, Color)> = match self {
            ObjectKind::Shape { shape, color, fill_color } => {
                let (outline, fill) = shape.masks(shape.bounds());
                let fill_c = if shape.fill == ShapeFill::Fill { *color } else { *fill_color };
                fill.map(|m| (m, fill_c)).into_iter().chain(outline.map(|m| (m, *color))).collect()
            }
            ObjectKind::Text { spec, color } => {
                if spec.text.trim().is_empty() {
                    Vec::new()
                } else {
                    let fonts = fonts.ok_or("No font library available")?;
                    let (mask, _) = fonts.render(spec).ok_or_else(|| format!("Font '{}' not found", spec.family))?;
                    vec![(mask, *color)]
                }
            }
        };
        Ok(paint_parts(&parts, depth))
    }
}

impl LayerObject {
    /// The object as it is now, for a layer at `offset`.
    pub fn at(&self, offset: (i32, i32)) -> ObjectKind {
        if offset == self.origin {
            self.kind.clone()
        } else {
            self.kind.translated((offset.0 - self.origin.0) as f32, (offset.1 - self.origin.1) as f32)
        }
    }
}

/// Coverage masks painted in order (each over the previous) onto a
/// transparent buffer that just fits them.
fn paint_parts(parts: &[(Mask, Color)], depth: Depth) -> (Pixels, (i32, i32)) {
    let area = parts.iter().map(|(m, _)| m.rect).fold(Rect::default(), |a, r| a.union(&r));
    if area.is_empty() {
        return (Pixels::new(depth, 1, 1), (0, 0));
    }
    let mut px = Pixels::new(depth, area.w, area.h);
    for y in 0..area.h {
        for x in 0..area.w {
            let (cx, cy) = (area.x + x as i32, area.y + y as i32);
            let mut p = [0.0f32; 4];
            for (m, c) in parts {
                let cov = m.get(cx, cy);
                if cov > 0.0 {
                    let mut s = c.px();
                    s[3] *= cov.min(1.0);
                    p = over(s, p);
                }
            }
            if p[3] > 0.0 {
                px.set(x, y, p);
            }
        }
    }
    crate::ops::transform::trimmed(px, (area.x, area.y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::paint::ShapeKind;

    pub fn rect_shape(a: Vec2, b: Vec2) -> Shape {
        Shape {
            kind: ShapeKind::Rectangle,
            a,
            b,
            width: 2.0,
            fill: ShapeFill::OutlineAndFill,
            antialias: false,
            head_scale: 4.0,
            rounded: false,
            corner_radius: 0.0,
            pixel_aligned: true,
        }
    }

    #[test]
    fn renders_shape_where_it_is() {
        let o = ObjectKind::Shape {
            shape: rect_shape(Vec2::new(10.0, 20.0), Vec2::new(30.0, 25.0)),
            color: Color::BLACK,
            fill_color: Color::WHITE,
        };
        let (px, off) = o.render(Depth::U8, None).unwrap();
        assert_eq!((off, px.width(), px.height()), ((10, 20), 20, 5));
        assert_eq!(px.get(0, 0), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(px.get(10, 2), [1.0, 1.0, 1.0, 1.0]);
        let moved = LayerObject { kind: o, origin: (10, 20) }.at((15, 20));
        assert_eq!(moved.render(Depth::U8, None).unwrap().1, (15, 20));
    }
}
