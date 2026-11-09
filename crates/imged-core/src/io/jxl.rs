//! JPEG XL: decoding with jxl-oxide and lossless encoding with zune-jpegxl
//! (both pure Rust; `jxl` feature).

use std::path::Path;

use jxl_oxide::JxlImage;
use zune_core::bit_depth::BitDepth;
use zune_core::colorspace::ColorSpace;
use zune_core::options::EncoderOptions;
use zune_jpegxl::JxlSimpleEncoder;

use crate::buffer::Buffer;
use crate::pixel::Depth;
use crate::pixels::Pixels;

/// Decodes the first frame. Float images open as f32, >8-bit as u16.
pub fn decode(bytes: &[u8]) -> Result<Pixels, String> {
    let image = JxlImage::builder().read(std::io::Cursor::new(bytes)).map_err(|e| format!("JPEG XL: {e}"))?;
    let bd = &image.image_header().metadata.bit_depth;
    let (float, bits) = (matches!(bd, jxl_oxide::image::BitDepth::FloatSample { .. }), bd.bits_per_sample());
    let render = image.render_frame(0).map_err(|e| format!("JPEG XL: {e}"))?;
    let fb = render.image_all_channels();
    let (w, h, ch) = (fb.width() as u32, fb.height() as u32, fb.channels());
    let src = fb.buf();
    let mut b = Buffer::<f32>::new(w, h);
    for i in 0..(w * h) as usize {
        let s = &src[i * ch..i * ch + ch];
        let p = match ch {
            1 => [s[0], s[0], s[0], 1.0],
            2 => [s[0], s[0], s[0], s[1]],
            3 => [s[0], s[1], s[2], 1.0],
            _ => [s[0], s[1], s[2], s[3]],
        };
        b.data[i * 4..i * 4 + 4].copy_from_slice(&p);
    }
    Ok(if float {
        Pixels::F32(b)
    } else if bits > 8 {
        Pixels::U16(b.convert())
    } else {
        Pixels::U8(b.convert())
    })
}

pub fn load(path: &Path) -> Result<Pixels, String> {
    decode(&std::fs::read(path).map_err(|e| e.to_string())?)
}

/// Saves losslessly: 8- or 16-bit, RGB when fully opaque, else RGBA.
/// (The encoder marks alpha as 8-bit even in 16-bit files, so images with
/// transparency are saved at 8 bits.)
pub fn save(path: &Path, img: &Buffer<f32>, depth: Depth) -> Result<Depth, String> {
    if img.width < 2 || img.height < 2 {
        return Err("JPEG XL saving needs an image of at least 2×2 pixels".into());
    }
    let opaque = img.data.chunks_exact(4).all(|p| p[3] >= 1.0);
    let depth = if depth == Depth::U8 || !opaque { Depth::U8 } else { Depth::U16 };
    let keep = |c: usize| opaque && c % 4 == 3;
    let (data, bd): (Vec<u8>, _) = match depth {
        Depth::U8 => {
            let b = img.convert::<u8>().data;
            (b.into_iter().enumerate().filter(|(i, _)| !keep(*i)).map(|(_, v)| v).collect(), BitDepth::Eight)
        }
        _ => {
            let b = img.convert::<u16>().data;
            let v = b.into_iter().enumerate().filter(|(i, _)| !keep(*i)).flat_map(|(_, v)| v.to_ne_bytes()).collect();
            (v, BitDepth::Sixteen)
        }
    };
    let cs = if opaque { ColorSpace::RGB } else { ColorSpace::RGBA };
    let opts = EncoderOptions::new(img.width as usize, img.height as usize, cs, bd);
    let mut out = Vec::new();
    JxlSimpleEncoder::new(&data, opts).encode(&mut out).map_err(|e| format!("JPEG XL: {e:?}"))?;
    std::fs::write(path, out).map_err(|e| e.to_string())?;
    Ok(depth)
}
