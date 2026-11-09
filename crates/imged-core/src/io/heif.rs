//! HEIC / HEIF through the system libheif (`heif` feature).

use std::path::Path;

use libheif_rs::{Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, Image, LibHeif, RgbChroma};

use crate::buffer::Buffer;
use crate::pixel::Depth;
use crate::pixels::Pixels;

fn err(e: impl std::fmt::Display) -> String {
    format!("HEIF: {e}")
}

/// Decodes the primary image at 8 bits, or 16 bits for 10/12-bit files.
pub fn decode(bytes: &[u8]) -> Result<Pixels, String> {
    let lib = LibHeif::new();
    let ctx = HeifContext::read_from_bytes(bytes).map_err(err)?;
    let handle = ctx.primary_image_handle().map_err(err)?;
    let bits = handle.luma_bits_per_pixel();
    let high = bits > 8;
    let chroma = if high { RgbChroma::HdrRgbaLe } else { RgbChroma::Rgba };
    let img = lib.decode(&handle, ColorSpace::Rgb(chroma), None).map_err(err)?;
    let planes = img.planes();
    let p = planes.interleaved.ok_or("HEIF: no interleaved plane")?;
    let (w, h, stride) = (p.width, p.height, p.stride);
    if high {
        let max = ((1u32 << bits.min(16)) - 1) as f32;
        let mut b = Buffer::<u16>::new(w, h);
        for y in 0..h as usize {
            let row = &p.data[y * stride..];
            for x in 0..w as usize {
                for c in 0..4 {
                    let i = (x * 4 + c) * 2;
                    let v = u16::from_le_bytes([row[i], row[i + 1]]) as f32 / max;
                    b.data[(y * w as usize + x) * 4 + c] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
        }
        Ok(Pixels::U16(b))
    } else {
        let mut b = Buffer::<u8>::new(w, h);
        for y in 0..h as usize {
            let src = &p.data[y * stride..y * stride + w as usize * 4];
            b.data[y * w as usize * 4..(y + 1) * w as usize * 4].copy_from_slice(src);
        }
        Ok(Pixels::U8(b))
    }
}

pub fn load(path: &Path) -> Result<Pixels, String> {
    decode(&std::fs::read(path).map_err(|e| e.to_string())?)
}

/// Saves 8-bit RGBA as HEIC (HEVC; needs libheif's x265 encoder plugin).
pub fn save(path: &Path, img: &Buffer<u8>, quality: u8) -> Result<(), String> {
    let (w, h) = (img.width, img.height);
    let mut image = Image::new(w, h, ColorSpace::Rgb(RgbChroma::Rgba)).map_err(err)?;
    image.create_plane(Channel::Interleaved, w, h, 8).map_err(err)?;
    {
        let planes = image.planes_mut();
        let p = planes.interleaved.ok_or("HEIF: no interleaved plane")?;
        let stride = p.stride;
        for y in 0..h as usize {
            let src = &img.data[y * w as usize * 4..(y + 1) * w as usize * 4];
            p.data[y * stride..y * stride + src.len()].copy_from_slice(src);
        }
    }
    let lib = LibHeif::new();
    let mut ctx = HeifContext::new().map_err(err)?;
    let mut enc = lib
        .encoder_for_format(CompressionFormat::Hevc)
        .map_err(|e| format!("HEIC saving needs libheif's HEVC encoder (x265 plugin): {e}"))?;
    enc.set_quality(EncoderQuality::Lossy(quality)).map_err(err)?;
    ctx.encode_image(&image, &mut enc, None).map_err(err)?;
    let s = path.to_str().ok_or("HEIF: the path is not valid UTF-8")?;
    ctx.write_to_file(s).map_err(err)
}

/// Depths HEIC can be saved at here.
pub const DEPTHS: &[Depth] = &[Depth::U8];
