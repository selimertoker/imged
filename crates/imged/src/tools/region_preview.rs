//! Hover preview for the paint bucket, magic wand and outline select: the
//! area a click would cover is tinted and outlined under the pointer.
//!
//! The pixels the tool looks at are captured once per document revision;
//! the area itself is computed on a worker thread (one job at a time, the
//! newest request waiting), so hovering never blocks the UI.

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use eframe::egui::{self, Color32, ColorImage, Painter, TextureHandle, TextureOptions};
use imged_core::command::{region_layer, RegionKind, RegionSource};
use imged_core::{Document, LayerId, Rect, Vec2};

use super::{select_mode, ToolCtx};
use crate::ui::ants::{boundary, MAX_RUNS};
use crate::view::ViewXform;

/// Longest side of the tint texture (larger areas are sampled down).
const TINT_MAX: u32 = 768;

#[derive(Clone, Copy, PartialEq)]
struct Key {
    px: (i32, i32),
    rev: u64,
    kind: RegionKind,
    sample_merged: bool,
    grow: bool,
    layer: LayerId,
    color: Color32,
}

#[derive(Clone, Copy, PartialEq)]
struct SourceKey {
    rev: u64,
    fill: bool,
    sample_merged: bool,
    grow: bool,
    layer: LayerId,
}

struct Job {
    key: Key,
    source: Arc<RegionSource>,
}

struct Done {
    key: Key,
    area: Option<(Rect, Vec<(i32, i32, i32, i32)>, ColorImage)>,
}

struct Shown {
    key: Key,
    bounds: Rect,
    runs: Vec<(i32, i32, i32, i32)>,
    image: Option<ColorImage>,
    tex: Option<TextureHandle>,
}

#[derive(Default)]
struct Worker {
    rx: Option<Receiver<Done>>,
    pending: Option<Job>,
    shown: Option<Shown>,
    egui: Option<egui::Context>,
}

#[derive(Default)]
pub struct RegionPreview {
    /// Last hovered canvas position.
    pos: Option<Vec2>,
    requested: Option<Key>,
    source: Option<(SourceKey, Arc<RegionSource>)>,
    worker: RefCell<Worker>,
}

fn compute(job: &Job) -> Done {
    let at = Vec2::new(job.key.px.0 as f32 + 0.5, job.key.px.1 as f32 + 0.5);
    let area = job.source.region(job.key.kind, at).and_then(|m| {
        let b = m.bounds()?;
        let runs = boundary(&m);
        let step = b.w.max(b.h).div_ceil(TINT_MAX).max(1);
        let (w, h) = (b.w.div_ceil(step), b.h.div_ceil(step));
        let c = job.key.color;
        let mut img = ColorImage::filled([w as usize, h as usize], Color32::TRANSPARENT);
        for j in 0..h {
            for i in 0..w {
                let x = b.x + (i * step + step / 2).min(b.w - 1) as i32;
                let y = b.y + (j * step + step / 2).min(b.h - 1) as i32;
                if m.get(x, y) >= 0.5 {
                    img.pixels[(j * w + i) as usize] = c;
                }
            }
        }
        Some((b, runs, img))
    });
    Done { key: job.key, area }
}

impl Worker {
    fn start(&mut self, job: Job) {
        let (tx, rx) = channel();
        let ctx = self.egui.clone();
        std::thread::spawn(move || {
            tx.send(compute(&job)).ok();
            if let Some(c) = ctx {
                c.request_repaint();
            }
        });
        self.rx = Some(rx);
    }

    fn poll(&mut self) {
        let Some(rx) = &self.rx else { return };
        match rx.try_recv() {
            Ok(done) => {
                self.rx = None;
                self.shown = done.area.map(|(bounds, runs, image)| Shown {
                    key: done.key,
                    bounds,
                    runs,
                    image: Some(image),
                    tex: None,
                });
                if let Some(next) = self.pending.take() {
                    self.start(next);
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(_) => self.rx = None,
        }
    }
}

impl RegionPreview {
    /// Remembers the hovered position (`None` hides the preview).
    pub fn hover(&mut self, pos: Option<Vec2>) {
        self.pos = pos;
    }

    /// Requests the area for the current position and settings when any of
    /// them changed. `fill`: the bucket (else a selection tool).
    pub fn update(&mut self, ctx: &mut ToolCtx, kind: RegionKind, fill: bool, sample_merged: bool, color: Color32) {
        let Some(pos) = self.pos else { return };
        let mode = select_mode(ctx);
        let doc = &*ctx.doc;
        let px = (pos.x.floor() as i32, pos.y.floor() as i32);
        let layer = if fill || sample_merged { doc.active } else { region_layer(doc, pos, mode) };
        let rev = doc.revision();
        let grow = fill && ctx.settings.grow_layers;
        let key = Key { px, rev, kind, sample_merged, grow, layer, color };
        if self.requested == Some(key) {
            return;
        }
        self.requested = Some(key);
        let skey = SourceKey { rev, fill, sample_merged, grow, layer };
        if self.source.as_ref().is_none_or(|(k, _)| *k != skey) {
            self.source = RegionSource::capture(ctx.doc, fill, sample_merged, grow, layer).map(|s| (skey, Arc::new(s)));
        }
        let Some((_, source)) = &self.source else { return };
        let job = Job { key, source: source.clone() };
        let mut w = self.worker.borrow_mut();
        w.poll();
        if w.rx.is_some() {
            w.pending = Some(job);
        } else {
            w.start(job);
        }
    }

    /// Draws the latest computed area, unless the document changed since,
    /// the pointer is off the canvas or a button is held.
    pub fn paint(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        let mut w = self.worker.borrow_mut();
        if w.egui.is_none() {
            w.egui = Some(painter.ctx().clone());
        }
        w.poll();
        let over = painter.ctx().pointer_hover_pos().is_some_and(|p| painter.clip_rect().contains(p));
        let busy = painter.ctx().input(|i| i.pointer.any_down());
        if self.pos.is_none() || !over || busy {
            return;
        }
        let Some(shown) = &mut w.shown else { return };
        if shown.key.rev != doc.revision() {
            return;
        }
        if let Some(img) = shown.image.take() {
            shown.tex = Some(painter.ctx().load_texture("region-preview", img, TextureOptions::NEAREST));
        }
        let r = view.rect_to_screen(shown.bounds);
        if let Some(t) = &shown.tex {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(t.id(), r, uv, Color32::WHITE);
        }
        let black = egui::Stroke::new(1.0, Color32::BLACK);
        let white = egui::Stroke::new(1.0, Color32::WHITE);
        let clip = painter.clip_rect();
        let draw = |a: egui::Pos2, b: egui::Pos2| {
            if clip.intersects(egui::Rect::from_two_pos(a, b).expand(1.0)) {
                painter.line_segment([a, b], black);
                painter.extend(egui::Shape::dashed_line(&[a, b], white, 3.0, 3.0));
            }
        };
        if shown.runs.len() > MAX_RUNS {
            let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
            for p in pts.windows(2) {
                draw(p[0], p[1]);
            }
        } else {
            let to = |x: i32, y: i32| view.to_screen(Vec2::new(x as f32, y as f32));
            for &(x0, y0, x1, y1) in &shown.runs {
                draw(to(x0, y0), to(x1, y1));
            }
        }
    }
}

/// Tint for the selection tools' preview.
pub const SELECT_TINT: Color32 = Color32::from_rgba_premultiplied(0, 50, 100, 100);

/// Tint for the bucket: the paint color, half transparent.
pub fn paint_tint(c: imged_core::Color) -> Color32 {
    let [r, g, b, a] = c.to_rgba8();
    Color32::from_rgba_unmultiplied(r, g, b, ((a as f32) * 0.55) as u8 + 20)
}
