//! `Pixels`: a buffer whose channel depth is chosen at runtime.
//!
//! Layers of different depths live in the same `Vec`, so they can't share one
//! generic type. `Pixels` wraps the three concrete buffers, and the
//! [`dispatch!`](crate::dispatch) / [`map_pixels!`](crate::map_pixels) macros
//! forward to code written generically over [`Channel`].

use crate::buffer::Buffer;
use crate::geom::Rect;
use crate::pixel::{Channel, Depth, Px};

#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    U8(Buffer<u8>),
    U16(Buffer<u16>),
    F32(Buffer<f32>),
}

/// Runs `$body` with `$b` bound to the inner `Buffer<T>` (by whatever
/// reference mode `$pix` has). The body must have the same type in every arm.
///
/// ```ignore
/// let w = dispatch!(&pixels, b => b.width);
/// dispatch!(&mut pixels, b => b.map_pixels(|p| p));
/// ```
#[macro_export]
macro_rules! dispatch {
    ($pix:expr, $b:ident => $body:expr) => {
        match $pix {
            $crate::pixels::Pixels::U8($b) => $body,
            $crate::pixels::Pixels::U16($b) => $body,
            $crate::pixels::Pixels::F32($b) => $body,
        }
    };
}

/// Like [`dispatch!`] but `$body` returns a `Buffer<T>` of the same depth,
/// which is wrapped back into `Pixels`.
#[macro_export]
macro_rules! map_pixels {
    ($pix:expr, $b:ident => $body:expr) => {
        match $pix {
            $crate::pixels::Pixels::U8($b) => $crate::pixels::Pixels::U8($body),
            $crate::pixels::Pixels::U16($b) => $crate::pixels::Pixels::U16($body),
            $crate::pixels::Pixels::F32($b) => $crate::pixels::Pixels::F32($body),
        }
    };
}

impl Pixels {
    pub fn new(depth: Depth, w: u32, h: u32) -> Self {
        match depth {
            Depth::U8 => Pixels::U8(Buffer::new(w, h)),
            Depth::U16 => Pixels::U16(Buffer::new(w, h)),
            Depth::F32 => Pixels::F32(Buffer::new(w, h)),
        }
    }

    pub fn filled(depth: Depth, w: u32, h: u32, color: Px) -> Self {
        match depth {
            Depth::U8 => Pixels::U8(Buffer::filled(w, h, color)),
            Depth::U16 => Pixels::U16(Buffer::filled(w, h, color)),
            Depth::F32 => Pixels::F32(Buffer::filled(w, h, color)),
        }
    }

    pub fn from_buffer<T: Channel>(b: Buffer<T>) -> Self {
        T::wrap(b)
    }

    /// Wraps an f32 buffer, converting to `depth`.
    pub fn from_f32(b: &Buffer<f32>, depth: Depth) -> Self {
        match depth {
            Depth::U8 => Pixels::U8(b.convert()),
            Depth::U16 => Pixels::U16(b.convert()),
            Depth::F32 => Pixels::F32(b.clone()),
        }
    }

    pub fn depth(&self) -> Depth {
        match self {
            Pixels::U8(_) => Depth::U8,
            Pixels::U16(_) => Depth::U16,
            Pixels::F32(_) => Depth::F32,
        }
    }

    pub fn width(&self) -> u32 {
        dispatch!(self, b => b.width)
    }

    pub fn height(&self) -> u32 {
        dispatch!(self, b => b.height)
    }

    pub fn rect(&self) -> Rect {
        Rect::from_size(self.width(), self.height())
    }

    pub fn get(&self, x: u32, y: u32) -> Px {
        dispatch!(self, b => b.get(x, y))
    }

    pub fn get_or_clear(&self, x: i32, y: i32) -> Px {
        dispatch!(self, b => b.get_or_clear(x, y))
    }

    pub fn set(&mut self, x: u32, y: u32, p: Px) {
        dispatch!(self, b => b.set(x, y, p))
    }

    pub fn crop(&self, r: Rect) -> Pixels {
        map_pixels!(self, b => b.crop(r))
    }

    /// Pastes `src` (converted to this depth if needed) at (x, y).
    pub fn paste(&mut self, src: &Pixels, x: i32, y: i32) {
        let src = if src.depth() == self.depth() {
            std::borrow::Cow::Borrowed(src)
        } else {
            std::borrow::Cow::Owned(src.convert(self.depth()))
        };
        match (self, src.as_ref()) {
            (Pixels::U8(d), Pixels::U8(s)) => d.paste(s, x, y),
            (Pixels::U16(d), Pixels::U16(s)) => d.paste(s, x, y),
            (Pixels::F32(d), Pixels::F32(s)) => d.paste(s, x, y),
            _ => unreachable!("depths were unified above"),
        }
    }

    pub fn convert(&self, depth: Depth) -> Pixels {
        if depth == self.depth() {
            return self.clone();
        }
        match depth {
            Depth::U8 => Pixels::U8(dispatch!(self, b => b.convert())),
            Depth::U16 => Pixels::U16(dispatch!(self, b => b.convert())),
            Depth::F32 => Pixels::F32(dispatch!(self, b => b.convert())),
        }
    }

    pub fn to_f32(&self) -> Buffer<f32> {
        match self {
            Pixels::F32(b) => b.clone(),
            other => dispatch!(other, b => b.to_f32()),
        }
    }

    pub fn map_pixels(&mut self, f: impl Fn(Px) -> Px + Sync) {
        dispatch!(self, b => b.map_pixels(&f))
    }

    pub fn byte_size(&self) -> usize {
        self.width() as usize * self.height() as usize * 4 * self.depth().bytes_per_channel()
    }

    pub fn opaque_bounds(&self) -> Option<Rect> {
        dispatch!(self, b => b.opaque_bounds())
    }

    /// True if any value lies outside `0..=1` (only possible for f32).
    pub fn has_hdr_values(&self) -> bool {
        match self {
            Pixels::F32(b) => b.data.iter().any(|v| *v > 1.0 || *v < 0.0),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_depth_paste() {
        let mut a = Pixels::new(Depth::U8, 4, 4);
        let b = Pixels::filled(Depth::F32, 2, 2, [0.5, 0.5, 0.5, 1.0]);
        a.paste(&b, 1, 1);
        let p = a.get(1, 1);
        assert!((p[0] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(a.depth(), Depth::U8);
    }
}
