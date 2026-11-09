//! Copying images to and from the system clipboard.
//!
//! Text needs nothing here: with eframe's `clipboard` feature, egui handles
//! Ctrl+C / Ctrl+X / Ctrl+V (CLIPBOARD) and, on X11/Wayland, the PRIMARY
//! selection (selecting copies, middle-click pastes) in every text box.
//!
//! One `arboard::Clipboard` is kept alive for the whole run: on Linux the
//! owner of a clipboard selection has to keep serving it.

use std::sync::{Mutex, OnceLock};

static CLIPBOARD: OnceLock<Mutex<Option<arboard::Clipboard>>> = OnceLock::new();

fn with<R>(f: impl FnOnce(&mut arboard::Clipboard) -> Result<R, arboard::Error>) -> Result<R, String> {
    let cell = CLIPBOARD.get_or_init(|| Mutex::new(arboard::Clipboard::new().ok()));
    let mut guard = cell.lock().map_err(|_| "clipboard lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = arboard::Clipboard::new().ok();
    }
    let cb = guard.as_mut().ok_or("no clipboard available")?;
    f(cb).map_err(|e| e.to_string())
}

/// Copies an RGBA8 image to the CLIPBOARD (offered to other apps as PNG).
pub fn set_image(width: usize, height: usize, rgba: Vec<u8>) -> Result<(), String> {
    with(|cb| cb.set_image(arboard::ImageData { width, height, bytes: rgba.into() }))
}

/// What an image paste produced.
pub enum Pasted {
    /// Decoded image data and the clipboard type it came from.
    Image(imged_core::Pixels, String),
    /// Files copied in a file manager.
    Files(Vec<std::path::PathBuf>),
}

/// Ranks clipboard types: lossless image types first, then any other image
/// type the `image` crate may decode.
fn image_rank(fmt: &str) -> Option<u32> {
    let f = fmt.to_ascii_lowercase();
    let order = [
        "image/png", "png", "image/tiff", "image/x-exr", "image/webp", "image/bmp", "image/x-bmp", "image/x-ms-bmp",
        "image/qoi", "image/gif", "image/avif", "image/x-portable-anymap", "image/x-tga", "image/jpeg", "image/jpg",
    ];
    if let Some(i) = order.iter().position(|o| *o == f) {
        return Some(i as u32);
    }
    (f.starts_with("image/") && !f.contains("svg")).then_some(100)
}

/// Reads an image from the clipboard in whatever format the source app
/// offers (PNG, JPEG, BMP, TIFF, WebP, …), or the files a file manager copied.
pub fn paste_image() -> Result<Pasted, String> {
    use clipboard_rs::Clipboard as _;
    let cb = clipboard_rs::ClipboardContext::new().map_err(|e| e.to_string())?;
    let formats = cb.available_formats().map_err(|e| e.to_string())?;
    let mut images: Vec<(u32, &String)> = formats.iter().filter_map(|f| Some((image_rank(f)?, f))).collect();
    images.sort();
    let mut last_err = None;
    for (_, fmt) in images {
        match cb.get_buffer(fmt) {
            Ok(bytes) if !bytes.is_empty() => match imged_core::io::load_bytes(&bytes) {
                Ok(px) => return Ok(Pasted::Image(px, fmt.clone())),
                Err(e) => last_err = Some(format!("{fmt}: {e}")),
            },
            Ok(_) => {}
            Err(e) => last_err = Some(format!("{fmt}: {e}")),
        }
    }
    // Files copied in a file manager, or a path copied as text.
    let mut files: Vec<std::path::PathBuf> = cb
        .get_files()
        .unwrap_or_default()
        .into_iter()
        .map(|f| std::path::PathBuf::from(f.strip_prefix("file://").map(percent_decode).unwrap_or(f)))
        .filter(|p| p.is_file())
        .collect();
    if files.is_empty() {
        if let Ok(t) = cb.get_text() {
            let p = std::path::PathBuf::from(t.trim().strip_prefix("file://").map(percent_decode).unwrap_or_else(|| t.trim().to_string()));
            if p.is_file() {
                files.push(p);
            }
        }
    }
    if !files.is_empty() {
        return Ok(Pasted::Files(files));
    }
    Err(last_err.unwrap_or_else(|| {
        if formats.is_empty() {
            "The clipboard is empty".to_string()
        } else {
            "The clipboard has no image".to_string()
        }
    }))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
