//! Coverage masks used for painting and selections.

use crate::geom::Rect;

/// Per-pixel coverage in `0..=1` over `rect`. The coordinate space depends on
/// the use: layer-local for painting, canvas coordinates for selections.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub rect: Rect,
    pub data: Vec<f32>,
}

impl Mask {
    pub fn new(rect: Rect) -> Self {
        Mask { rect, data: vec![0.0; rect.area() as usize] }
    }

    /// A fully covered rectangle.
    pub fn full(rect: Rect) -> Self {
        Mask { rect, data: vec![1.0; rect.area() as usize] }
    }

    #[inline(always)]
    fn idx(&self, x: i32, y: i32) -> Option<usize> {
        if self.rect.contains(x, y) {
            Some((y - self.rect.y) as usize * self.rect.w as usize + (x - self.rect.x) as usize)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get(&self, x: i32, y: i32) -> f32 {
        self.idx(x, y).map_or(0.0, |i| self.data[i])
    }

    #[inline(always)]
    pub fn set(&mut self, x: i32, y: i32, v: f32) {
        if let Some(i) = self.idx(x, y) {
            self.data[i] = v;
        }
    }

    /// Keeps the larger of the existing and the new coverage.
    #[inline(always)]
    pub fn max_at(&mut self, x: i32, y: i32, v: f32) {
        if let Some(i) = self.idx(x, y) {
            if v > self.data[i] {
                self.data[i] = v;
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|v| *v <= 0.0)
    }

    /// Tight bounds of non-zero coverage.
    pub fn bounds(&self) -> Option<Rect> {
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        let w = self.rect.w as usize;
        for (i, v) in self.data.iter().enumerate() {
            if *v > 0.0 {
                let x = (i % w) as i32 + self.rect.x;
                let y = (i / w) as i32 + self.rect.y;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
        (x0 != i32::MAX).then(|| Rect::from_corners(x0, y0, x1, y1))
    }

    pub fn translated(mut self, dx: i32, dy: i32) -> Self {
        self.rect = self.rect.translate(dx, dy);
        self
    }

    /// Number of covered pixels (coverage > 0).
    pub fn count(&self) -> usize {
        self.data.iter().filter(|v| **v > 0.0).count()
    }
}
