//! `Buffer<T>`: a tightly packed RGBA image with channels of type `T`.

use rayon::prelude::*;

use crate::geom::Rect;
use crate::pixel::{premultiply, unpremultiply, Channel, Px};

/// RGBA pixel buffer, row-major, 4 channels per pixel, straight alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer<T: Channel> {
    pub width: u32,
    pub height: u32,
    pub data: Vec<T>,
}

impl<T: Channel> Buffer<T> {
    /// Fully transparent buffer.
    pub fn new(width: u32, height: u32) -> Self {
        Buffer {
            width,
            height,
            data: vec![T::default(); width as usize * height as usize * 4],
        }
    }

    pub fn filled(width: u32, height: u32, color: Px) -> Self {
        let px = [
            T::from_f32(color[0]),
            T::from_f32(color[1]),
            T::from_f32(color[2]),
            T::from_f32(color[3]),
        ];
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..(width as usize * height as usize) {
            data.extend_from_slice(&px);
        }
        Buffer { width, height, data }
    }

    pub fn from_raw(width: u32, height: u32, data: Vec<T>) -> Option<Self> {
        (data.len() == width as usize * height as usize * 4).then_some(Buffer { width, height, data })
    }

    pub fn rect(&self) -> Rect {
        Rect::from_size(self.width, self.height)
    }

    #[inline(always)]
    pub fn index(&self, x: u32, y: u32) -> usize {
        (y as usize * self.width as usize + x as usize) * 4
    }

    #[inline(always)]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }

    #[inline(always)]
    pub fn get(&self, x: u32, y: u32) -> Px {
        let i = self.index(x, y);
        let d = &self.data[i..i + 4];
        [d[0].to_f32(), d[1].to_f32(), d[2].to_f32(), d[3].to_f32()]
    }

    /// Out-of-bounds reads return transparent black.
    #[inline(always)]
    pub fn get_or_clear(&self, x: i32, y: i32) -> Px {
        if self.in_bounds(x, y) {
            self.get(x as u32, y as u32)
        } else {
            [0.0; 4]
        }
    }

    #[inline(always)]
    pub fn set(&mut self, x: u32, y: u32, p: Px) {
        let i = self.index(x, y);
        let d = &mut self.data[i..i + 4];
        d[0] = T::from_f32(p[0]);
        d[1] = T::from_f32(p[1]);
        d[2] = T::from_f32(p[2]);
        d[3] = T::from_f32(p[3]);
    }

    /// Bilinear sample at continuous coordinates (pixel centers at +0.5),
    /// interpolating in premultiplied space. Outside the buffer is transparent.
    pub fn sample_bilinear(&self, x: f32, y: f32) -> Px {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);
        let p00 = premultiply(self.get_or_clear(x0, y0));
        let p10 = premultiply(self.get_or_clear(x0 + 1, y0));
        let p01 = premultiply(self.get_or_clear(x0, y0 + 1));
        let p11 = premultiply(self.get_or_clear(x0 + 1, y0 + 1));
        let mut out = [0.0f32; 4];
        for c in 0..4 {
            let top = p00[c] + (p10[c] - p00[c]) * tx;
            let bot = p01[c] + (p11[c] - p01[c]) * tx;
            out[c] = top + (bot - top) * ty;
        }
        unpremultiply(out)
    }

    pub fn sample_nearest(&self, x: f32, y: f32) -> Px {
        self.get_or_clear(x.floor() as i32, y.floor() as i32)
    }

    /// Copies out a sub-rectangle. Parts outside the buffer are transparent.
    pub fn crop(&self, r: Rect) -> Buffer<T> {
        let mut out = Buffer::new(r.w, r.h);
        let src = self.rect().intersect(&r);
        if src.is_empty() {
            return out;
        }
        let row_len = src.w as usize * 4;
        for y in src.y..src.bottom() {
            let si = self.index(src.x as u32, y as u32);
            let di = out.index((src.x - r.x) as u32, (y - r.y) as u32);
            out.data[di..di + row_len].copy_from_slice(&self.data[si..si + row_len]);
        }
        out
    }

    /// Copies `src` into `self` at (x, y), replacing pixels (no blending).
    pub fn paste(&mut self, src: &Buffer<T>, x: i32, y: i32) {
        let dst = self.rect().intersect(&Rect::new(x, y, src.width, src.height));
        if dst.is_empty() {
            return;
        }
        let row_len = dst.w as usize * 4;
        for yy in dst.y..dst.bottom() {
            let si = src.index((dst.x - x) as u32, (yy - y) as u32);
            let di = self.index(dst.x as u32, yy as u32);
            self.data[di..di + row_len].copy_from_slice(&src.data[si..si + row_len]);
        }
    }

    pub fn convert<U: Channel>(&self) -> Buffer<U> {
        Buffer {
            width: self.width,
            height: self.height,
            data: self.data.par_iter().map(|&v| U::from_f32(v.to_f32())).collect(),
        }
    }

    pub fn to_f32(&self) -> Buffer<f32> {
        self.convert()
    }

    /// Applies `f` to every pixel in parallel.
    pub fn map_pixels(&mut self, f: impl Fn(Px) -> Px + Sync) {
        self.data.par_chunks_mut(4).for_each(|d| {
            let p = [d[0].to_f32(), d[1].to_f32(), d[2].to_f32(), d[3].to_f32()];
            let o = f(p);
            for c in 0..4 {
                d[c] = T::from_f32(o[c]);
            }
        });
    }

    /// Builds a new buffer of the given size by evaluating `f(x, y)` per pixel in parallel.
    pub fn from_fn(width: u32, height: u32, f: impl Fn(u32, u32) -> Px + Sync) -> Self {
        let mut out = Buffer::<T>::new(width, height);
        if width == 0 || height == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(width as usize * 4)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let p = f(x, y as u32);
                    let i = x as usize * 4;
                    for c in 0..4 {
                        row[i + c] = T::from_f32(p[c]);
                    }
                }
            });
        out
    }

    /// Tight bounding box of pixels with alpha > 0, or `None` if fully transparent.
    pub fn opaque_bounds(&self) -> Option<Rect> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.data[self.index(x, y) + 3].to_f32() > 0.0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        (x0 != u32::MAX).then(|| Rect::from_corners(x0 as i32, y0 as i32, x1 as i32, y1 as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_and_paste() {
        let mut b = Buffer::<u8>::filled(4, 4, [1.0, 0.0, 0.0, 1.0]);
        let c = b.crop(Rect::new(-1, -1, 3, 3));
        assert_eq!(c.get(0, 0), [0.0; 4]);
        assert_eq!(c.get(1, 1), [1.0, 0.0, 0.0, 1.0]);
        let g = Buffer::<u8>::filled(2, 2, [0.0, 1.0, 0.0, 1.0]);
        b.paste(&g, 3, 3);
        assert_eq!(b.get(3, 3), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(b.get(2, 2), [1.0, 0.0, 0.0, 1.0]);
    }
}
