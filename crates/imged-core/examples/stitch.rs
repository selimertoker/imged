//! Stitches two photos from the command line (the same code as the Stitch tool).
//!
//! ```sh
//! cargo run --release -p imged-core --features stitch --example stitch -- \
//!     fixed.jpg moving.jpg FX FY MX MY out.png [--prefer-upper] [--projection]
//! ```
//! (FX, FY) and (MX, MY) are the same letter on each photo, in pixels.

use std::path::Path;
use std::time::Instant;

use imged_core::composite::flatten;
use imged_core::ops::stitch::{self, MergeOptions, StitchMode};
use imged_core::{io, Depth, Document};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags: Vec<&String> = args.iter().filter(|a| a.starts_with("--")).collect();
    let pos: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() != 7 {
        return Err("usage: stitch fixed moving FX FY MX MY out.png [--prefer-upper] [--projection]".into());
    }
    let num = |s: &str| s.parse::<f32>().map_err(|e| e.to_string());
    let t = Instant::now();
    let fixed = io::load(Path::new(pos[0]))?.pixels;
    let moving = io::load(Path::new(pos[1]))?.pixels;
    println!("loaded {}x{} and {}x{} in {:.1}s", fixed.width(), fixed.height(), moving.width(), moving.height(), t.elapsed().as_secs_f32());
    let mode = if flags.iter().any(|f| *f == "--projection") { StitchMode::Projection } else { StitchMode::Board };
    let progress = |f: f32, s: &str| {
        println!("  [{:3.0}%] {s} ({:.1}s)", f * 100.0, t.elapsed().as_secs_f32());
        true
    };
    let al = stitch::align(&fixed, &moving, [num(pos[2])?, num(pos[3])?], [num(pos[4])?, num(pos[5])?], mode, &progress)?;
    println!(
        "aligned: angle {:.1}°, scale {:.3}, seed match {:.0}%, {} matches, median error {:.2} px, field up to {:.1} px",
        al.angle,
        al.scale,
        al.seed_score * 100.0,
        al.matches.len(),
        al.residual,
        al.field.max_abs()
    );
    let opt = MergeOptions { prefer_upper: flags.iter().any(|f| *f == "--prefer-upper"), fixed_on_top: true, want_warped: false };
    let out = stitch::merge(&fixed, &moving, &al, opt, &progress)?;
    println!("merged {}x{}, {} conflicting regions, {:.1}s", out.merged_rect.w, out.merged_rect.h, out.conflicts, t.elapsed().as_secs_f32());
    let depth = out.merged.depth().min(Depth::U16);
    let doc = Document::from_pixels(out.merged, "stitched");
    io::save(Path::new(pos[6]), &flatten(&doc), depth, [0.0; 4])?;
    println!("saved in {:.1}s", t.elapsed().as_secs_f32());
    Ok(())
}
