//! Loading and saving images.
//!
//! Every format the `image` crate supports goes through it; PFM (portable
//! float map) is handled here because `image` doesn't support it. Images load
//! at their native depth: 8-bit → u8, 16-bit → u16, float formats → f32.
//!
//! AVIF *decoding* needs the system `dav1d` library, so it is behind the
//! `avif-decode` cargo feature. AVIF encoding is pure Rust and always on.
//! HEIC/HEIF goes through the system libheif (`heif` feature, default on);
//! JPEG XL through pure-Rust crates (`jxl` feature, default on; saving is
//! lossless).

#[cfg(feature = "heif")]
pub mod heif;
#[cfg(feature = "jxl")]
pub mod jxl;
pub mod psd;
pub mod raw;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use image::codecs::pnm::{PnmEncoder, PnmSubtype, SampleEncoding};
use image::{DynamicImage, ImageBuffer, ImageFormat};

use crate::buffer::Buffer;
use crate::pixel::{Depth, Px};
use crate::pixels::Pixels;

/// A decoded image.
pub struct Loaded {
    pub pixels: Pixels,
    /// The data is scene-linear (EXR, Radiance HDR, PFM).
    pub linear: bool,
}

pub fn extension(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

fn can_read(f: ImageFormat) -> bool {
    f.reading_enabled() && (f != ImageFormat::Avif || cfg!(feature = "avif-decode"))
}

/// File extensions that can be opened (everything the `image` crate reads, plus PFM).
pub fn open_extensions() -> Vec<&'static str> {
    let mut v: Vec<&'static str> =
        ImageFormat::all().filter(|f| can_read(*f)).flat_map(|f| f.extensions_str().iter().copied()).collect();
    v.push("pfm");
    v.extend(["psd", "psb"]);
    if cfg!(feature = "heif") {
        v.extend(["heic", "heif", "hif"]);
    }
    if cfg!(feature = "jxl") {
        v.push("jxl");
    }
    v
}

fn is_heif(ext: &str) -> bool {
    cfg!(feature = "heif") && matches!(ext, "heic" | "heif" | "hif")
}

/// Layered project files (opened as documents, not single images).
pub fn is_project(path: &Path) -> bool {
    matches!(extension(path).as_str(), "psd" | "psb")
}

/// File extensions that can be saved (everything the `image` crate writes, plus PFM).
pub fn save_extensions() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = ImageFormat::all()
        .filter(|f| f.writing_enabled())
        .flat_map(|f| f.extensions_str().iter().copied())
        .collect();
    v.push("pfm");
    if cfg!(feature = "heif") {
        v.extend(["heic", "heif"]);
    }
    if cfg!(feature = "jxl") {
        v.push("jxl");
    }
    v
}

/// Depths a file type can store (empty = can't save this type).
pub fn save_depths(ext: &str) -> &'static [Depth] {
    match ext {
        "png" | "avif" | "pam" => &[Depth::U8, Depth::U16],
        "jxl" if cfg!(feature = "jxl") => &[Depth::U8, Depth::U16],
        "heic" | "heif" if cfg!(feature = "heif") => &[Depth::U8],
        "tif" | "tiff" => &[Depth::U8, Depth::U16, Depth::F32],
        "ff" => &[Depth::U16],
        "exr" | "hdr" | "pfm" => &[Depth::F32],
        "jpg" | "jpeg" | "bmp" | "webp" | "qoi" | "tga" | "ico" | "gif" | "ppm" | "pgm" | "pbm" | "pnm" => {
            &[Depth::U8]
        }
        _ => &[],
    }
}

/// Formats that can't store alpha (alpha is composited over `background`).
pub fn lacks_alpha(ext: &str) -> bool {
    matches!(ext, "jpg" | "jpeg" | "hdr" | "pfm" | "ppm" | "pgm" | "pbm" | "pnm")
}

pub fn load(path: &Path) -> Result<Loaded, String> {
    let ext = extension(path);
    if ext == "pfm" {
        let f = File::open(path).map_err(|e| e.to_string())?;
        return Ok(Loaded { pixels: Pixels::F32(read_pfm(&mut BufReader::new(f))?), linear: true });
    }
    // Formats the `image` crate doesn't know, also detected by content.
    let head = {
        let mut b = [0u8; 16];
        let n = File::open(path).and_then(|mut f| std::io::Read::read(&mut f, &mut b)).unwrap_or(0);
        b[..n].to_vec()
    };
    #[cfg(feature = "jxl")]
    if ext == "jxl" || is_jxl(&head) {
        return Ok(Loaded { pixels: jxl::load(path)?, linear: false });
    }
    #[cfg(feature = "heif")]
    if is_heif(&ext) || is_heif_bytes(&head) {
        return Ok(Loaded { pixels: heif::load(path)?, linear: false });
    }
    let _ = &head;
    let reader = image::ImageReader::open(path)
        .map_err(|e| e.to_string())?
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    if reader.format() == Some(ImageFormat::Avif) && !cfg!(feature = "avif-decode") {
        return Err("AVIF decoding needs imged built with the `avif-decode` feature (requires libdav1d)".into());
    }
    let mut reader = reader;
    reader.no_limits();
    let img = reader.decode().map_err(|e| e.to_string())?;
    let linear = matches!(ext.as_str(), "exr" | "hdr");
    Ok(Loaded { pixels: from_dynamic(img), linear })
}

/// JPEG XL signature (bare codestream or container).
pub fn is_jxl(b: &[u8]) -> bool {
    b.starts_with(&[0xFF, 0x0A]) || b.starts_with(&[0, 0, 0, 0x0C, b'J', b'X', b'L', b' '])
}

/// HEIF/HEIC "ftyp" box with a HEIF brand (AVIF is left to `image`).
pub fn is_heif_bytes(b: &[u8]) -> bool {
    b.len() >= 12
        && &b[4..8] == b"ftyp"
        && matches!(&b[8..12], b"heic" | b"heix" | b"hevc" | b"hevx" | b"heim" | b"heis" | b"mif1" | b"msf1")
}

/// Decodes an image from memory (e.g. the clipboard).
pub fn load_bytes(bytes: &[u8]) -> Result<Pixels, String> {
    #[cfg(feature = "jxl")]
    if is_jxl(bytes) {
        return jxl::decode(bytes);
    }
    #[cfg(feature = "heif")]
    if is_heif_bytes(bytes) {
        return heif::decode(bytes);
    }
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    Ok(from_dynamic(img))
}

pub fn from_dynamic(img: DynamicImage) -> Pixels {
    use image::ColorType::*;
    let (w, h) = (img.width(), img.height());
    match img.color() {
        Rgb32F | Rgba32F => Pixels::F32(Buffer::from_raw(w, h, img.into_rgba32f().into_raw()).unwrap()),
        L16 | La16 | Rgb16 | Rgba16 => Pixels::U16(Buffer::from_raw(w, h, img.into_rgba16().into_raw()).unwrap()),
        _ => Pixels::U8(Buffer::from_raw(w, h, img.into_rgba8().into_raw()).unwrap()),
    }
}

/// Picks the depth to write: `depth` if the type supports it, otherwise
/// the nearest supported one (preferring more precision).
pub fn effective_depth(ext: &str, depth: Depth) -> Option<Depth> {
    let depths = save_depths(ext);
    if depths.is_empty() {
        return None;
    }
    Some(if depths.contains(&depth) {
        depth
    } else {
        *depths.iter().find(|d| **d >= depth).unwrap_or(depths.last().unwrap())
    })
}

/// Saves a flattened image. `depth` is adjusted to what the format supports;
/// formats without alpha are composited over `background`.
/// Returns the depth that was actually written.
pub fn save(path: &Path, img: &Buffer<f32>, depth: Depth, background: Px) -> Result<Depth, String> {
    let ext = extension(path);
    let depth = effective_depth(&ext, depth).ok_or_else(|| format!("Unsupported file type '.{ext}'"))?;
    let mut img = img.clone();
    if lacks_alpha(&ext) {
        img.map_pixels(|p| crate::pixel::over(p, [background[0], background[1], background[2], 1.0]));
    }
    if ext == "pfm" {
        let f = File::create(path).map_err(|e| e.to_string())?;
        write_pfm(&mut BufWriter::new(f), &img).map_err(|e| e.to_string())?;
        return Ok(depth);
    }
    #[cfg(feature = "jxl")]
    if ext == "jxl" {
        return jxl::save(path, &img, depth);
    }
    #[cfg(feature = "heif")]
    if is_heif(&ext) {
        heif::save(path, &img.convert::<u8>(), 90)?;
        return Ok(Depth::U8);
    }
    let dynimg = to_dynamic(&img, depth);

    // PNM variants need an explicit subtype; the generic path always writes PAM-style headers.
    let pnm = match ext.as_str() {
        "ppm" | "pnm" => Some((PnmSubtype::Pixmap(SampleEncoding::Binary), DynamicImage::ImageRgb8(dynimg.to_rgb8()))),
        "pgm" => Some((PnmSubtype::Graymap(SampleEncoding::Binary), DynamicImage::ImageLuma8(dynimg.to_luma8()))),
        "pbm" => {
            // PBM samples are 0 = black, 1 = white in image's encoder.
            let mut l = dynimg.to_luma8();
            l.pixels_mut().for_each(|p| p.0[0] = u8::from(p.0[0] >= 128));
            Some((PnmSubtype::Bitmap(SampleEncoding::Binary), DynamicImage::ImageLuma8(l)))
        }
        _ => None,
    };
    if let Some((subtype, img)) = pnm {
        let f = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
        img.write_with_encoder(PnmEncoder::new(f).with_subtype(subtype)).map_err(|e| e.to_string())?;
        return Ok(depth);
    }

    let dynimg = if lacks_alpha(&ext) || ext == "hdr" {
        match depth {
            Depth::U8 => DynamicImage::ImageRgb8(dynimg.into_rgb8()),
            Depth::U16 => DynamicImage::ImageRgb16(dynimg.into_rgb16()),
            Depth::F32 => DynamicImage::ImageRgb32F(dynimg.into_rgb32f()),
        }
    } else {
        dynimg
    };
    let format = ImageFormat::from_extension(&ext).ok_or_else(|| format!("Unsupported file type '.{ext}'"))?;
    dynimg.save_with_format(path, format).map_err(|e| e.to_string())?;
    Ok(depth)
}

pub fn to_dynamic(img: &Buffer<f32>, depth: Depth) -> DynamicImage {
    let (w, h) = (img.width, img.height);
    match depth {
        Depth::U8 => DynamicImage::ImageRgba8(ImageBuffer::from_raw(w, h, img.convert::<u8>().data).unwrap()),
        Depth::U16 => DynamicImage::ImageRgba16(ImageBuffer::from_raw(w, h, img.convert::<u16>().data).unwrap()),
        Depth::F32 => DynamicImage::ImageRgba32F(ImageBuffer::from_raw(w, h, img.data.clone()).unwrap()),
    }
}

// ----- PFM ----------------------------------------------------------------

fn read_token(r: &mut impl BufRead) -> Result<String, String> {
    let mut tok = String::new();
    let mut byte = [0u8; 1];
    loop {
        if r.read(&mut byte).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let c = byte[0] as char;
        if c == '#' && tok.is_empty() {
            let mut skip = String::new();
            r.read_line(&mut skip).map_err(|e| e.to_string())?;
            continue;
        }
        if c.is_ascii_whitespace() {
            if tok.is_empty() {
                continue;
            }
            break;
        }
        tok.push(c);
    }
    if tok.is_empty() {
        Err("Unexpected end of PFM header".into())
    } else {
        Ok(tok)
    }
}

/// Reads a PFM (`PF` = RGB, `Pf` = greyscale). Rows are stored bottom-to-top;
/// a negative scale means little-endian.
pub fn read_pfm(r: &mut impl BufRead) -> Result<Buffer<f32>, String> {
    let magic = read_token(r)?;
    let channels = match magic.as_str() {
        "PF" => 3,
        "Pf" => 1,
        "PF4" => 4,
        _ => return Err("Not a PFM file".into()),
    };
    let w: u32 = read_token(r)?.parse().map_err(|_| "Bad PFM width")?;
    let h: u32 = read_token(r)?.parse().map_err(|_| "Bad PFM height")?;
    let scale: f32 = read_token(r)?.parse().map_err(|_| "Bad PFM scale")?;
    let little = scale < 0.0;
    let n = w as usize * h as usize * channels;
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw).map_err(|e| format!("PFM data truncated: {e}"))?;
    let vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| {
            let b = [c[0], c[1], c[2], c[3]];
            if little {
                f32::from_le_bytes(b)
            } else {
                f32::from_be_bytes(b)
            }
        })
        .collect();
    let mut out = Buffer::<f32>::new(w, h);
    for y in 0..h {
        let src_row = (h - 1 - y) as usize; // bottom-to-top
        for x in 0..w {
            let i = (src_row * w as usize + x as usize) * channels;
            let p = match channels {
                1 => [vals[i], vals[i], vals[i], 1.0],
                3 => [vals[i], vals[i + 1], vals[i + 2], 1.0],
                _ => [vals[i], vals[i + 1], vals[i + 2], vals[i + 3]],
            };
            out.set(x, y, p);
        }
    }
    Ok(out)
}

/// Writes an RGB little-endian PFM (alpha is dropped).
pub fn write_pfm(w: &mut impl Write, img: &Buffer<f32>) -> std::io::Result<()> {
    write!(w, "PF\n{} {}\n-1.0\n", img.width, img.height)?;
    for y in (0..img.height).rev() {
        for x in 0..img.width {
            let p = img.get(x, y);
            for v in &p[..3] {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfm_roundtrip() {
        let mut b = Buffer::<f32>::filled(3, 2, [0.0, 0.0, 0.0, 1.0]);
        b.set(0, 0, [10.5, -1.0, 0.25, 1.0]);
        b.set(2, 1, [0.0, 3.0, 7.0, 1.0]);
        let mut data = Vec::new();
        write_pfm(&mut data, &b).unwrap();
        let r = read_pfm(&mut std::io::Cursor::new(data)).unwrap();
        assert_eq!(r, b);
    }

    #[test]
    fn save_and_load_png16() {
        let dir = std::env::temp_dir().join(format!("imged-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.png");
        let b = Buffer::<f32>::filled(2, 2, [0.5, 0.25, 0.125, 0.75]);
        assert_eq!(save(&p, &b, Depth::U16, [1.0; 4]).unwrap(), Depth::U16);
        let l = load(&p).unwrap();
        assert_eq!(l.pixels.depth(), Depth::U16);
        let px = l.pixels.get(1, 1);
        assert!((px[0] - 0.5).abs() < 1e-4 && (px[3] - 0.75).abs() < 1e-4);
        let p = dir.join("t.exr");
        assert_eq!(save(&p, &Buffer::filled(2, 2, [4.0, 0.0, 0.0, 1.0]), Depth::U8, [1.0; 4]).unwrap(), Depth::F32);
        let l = load(&p).unwrap();
        assert_eq!(l.pixels.depth(), Depth::F32);
        assert_eq!(l.pixels.get(0, 0)[0], 4.0);
        // Every writable format round-trips (pixel values loosely, for lossy ones).
        let src = Buffer::<f32>::from_fn(8, 6, |x, y| {
            let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            [v, v, v, 1.0]
        });
        for ext in save_extensions() {
            if ext == "jpeg" || ext == "tif" || ext == "pnm" || ext == "heif" {
                continue; // aliases
            }
            let p = dir.join(format!("t.{ext}"));
            let d = save(&p, &src, Depth::U8, [1.0; 4]).unwrap_or_else(|e| panic!("{ext}: {e}"));
            if ext == "avif" && !cfg!(feature = "avif-decode") {
                continue;
            }
            let l = load(&p).unwrap_or_else(|e| panic!("{ext} (saved as {d:?}): {e}"));
            assert_eq!((l.pixels.width(), l.pixels.height()), (8, 6), "{ext}");
            if !matches!(ext, "jpg" | "webp" | "avif" | "heic") {
                assert!((l.pixels.get(0, 0)[0] - 1.0).abs() < 0.02, "{ext}: {:?}", l.pixels.get(0, 0));
                assert!(l.pixels.get(1, 0)[0] < 0.02, "{ext}: {:?}", l.pixels.get(1, 0));
            }
        }
        #[cfg(feature = "jxl")]
        {
            // JPEG XL is lossless: 16-bit when opaque, 8-bit with transparency.
            let g = Buffer::<f32>::from_fn(5, 4, |x, y| [x as f32 / 4.0, y as f32 / 3.0, 0.123, 1.0]);
            let p = dir.join("g.jxl");
            assert_eq!(save(&p, &g, Depth::U16, [1.0; 4]).unwrap(), Depth::U16);
            let l = load(&p).unwrap();
            assert_eq!(l.pixels.depth(), Depth::U16);
            let q = l.pixels.get(3, 2);
            assert!((q[0] - 0.75).abs() < 1e-4 && (q[2] - 0.123).abs() < 1e-4 && q[3] == 1.0, "{q:?}");
            let g = Buffer::<f32>::from_fn(5, 4, |x, _| [x as f32 / 4.0, 0.0, 0.0, 0.5]);
            assert_eq!(save(&p, &g, Depth::U16, [1.0; 4]).unwrap(), Depth::U8);
            let q = load(&p).unwrap().pixels.get(4, 0);
            assert!((q[0] - 1.0).abs() < 1e-3 && (q[3] - 0.5).abs() < 0.01, "{q:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
