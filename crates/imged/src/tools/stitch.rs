//! Stitch tool: click the same letter on two layers (photos of a whiteboard
//! or of a projected document); the second layer is aligned onto the first
//! and the two are merged into a new layer.
//!
//! Aligning and merging run on a background thread (a few seconds each for
//! phone photos); the tool polls them while its options are drawn.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use eframe::egui::{self, Align2, Color32, FontId, Painter, RichText, Stroke, Ui};
use imged_core::layer::LayerId;
use imged_core::ops::stitch::{self, Alignment, MergeOptions, StitchApply, StitchMode, StitchOutput};
use imged_core::{Command, Document, Pixels, Vec2};

use super::{Button, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

/// A clicked point on a layer.
#[derive(Clone, Copy, Debug)]
struct Pick {
    layer: LayerId,
    /// Canvas position at the time of the click.
    canvas: Vec2,
    /// Layer-local, pixel-center coordinates.
    local: [f32; 2],
}

enum JobResult {
    Aligned(Result<Alignment, String>),
    Merged(Result<StitchOutput, String>),
}

struct Job {
    rx: Receiver<JobResult>,
    progress: Arc<Mutex<(f32, String)>>,
    /// The two layers' ids (fixed, moving) the job works on.
    layers: (LayerId, LayerId),
}

struct Aligned {
    fixed: LayerId,
    moving: LayerId,
    al: Arc<Alignment>,
}

pub struct StitchTool {
    picks: Vec<Pick>,
    mode: StitchMode,
    prefer_upper: bool,
    keep_sources: bool,
    /// Merge as soon as the layers are aligned.
    auto_apply: bool,
    job: Option<Job>,
    aligned: Option<Aligned>,
    /// The layer the pointer is over (for the status bar).
    hover: Option<String>,
}

impl Default for StitchTool {
    fn default() -> Self {
        StitchTool {
            picks: Vec::new(),
            mode: StitchMode::Board,
            prefer_upper: false,
            keep_sources: true,
            auto_apply: false,
            job: None,
            aligned: None,
            hover: None,
        }
    }
}

/// The topmost visible pixel layer with something at `p` (canvas coordinates).
fn layer_at(doc: &Document, p: Vec2) -> Option<LayerId> {
    let (x, y) = (p.x.floor() as i32, p.y.floor() as i32);
    doc.layers
        .iter()
        .rev()
        .filter(|l| l.visible && !l.is_filter() && l.bounds().contains(x, y))
        .find(|l| l.pixels.get((x - l.offset.0) as u32, (y - l.offset.1) as u32)[3] > 0.0)
        .map(|l| l.id)
}

impl StitchTool {
    fn reset(&mut self) {
        self.picks.clear();
        self.job = None;
        self.aligned = None;
    }

    fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn pixels(doc: &Document, id: LayerId) -> Option<Arc<Pixels>> {
        doc.layer(id).map(|l| l.pixels.clone())
    }

    fn spawn(&mut self, layers: (LayerId, LayerId), label: &str, work: impl FnOnce(&(dyn Fn(f32, &str) + Sync)) -> JobResult + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        let progress = Arc::new(Mutex::new((0.0, label.to_string())));
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let report = move |f: f32, s: &str| {
                if let Ok(mut g) = p2.lock() {
                    *g = (f, s.to_string());
                }
            };
            let _ = tx.send(work(&report));
        });
        self.job = Some(Job { rx, progress, layers });
    }

    fn start_align(&mut self, ctx: &mut ToolCtx) {
        let [a, b] = [self.picks[0], self.picks[1]];
        let (Some(fixed), Some(moving)) = (Self::pixels(ctx.doc, a.layer), Self::pixels(ctx.doc, b.layer)) else {
            self.reset();
            return;
        };
        self.aligned = None;
        let mode = self.mode;
        self.spawn((a.layer, b.layer), "Aligning", move |p| {
            JobResult::Aligned(stitch::align(&fixed, &moving, a.local, b.local, mode, p))
        });
    }

    fn start_merge(&mut self, ctx: &mut ToolCtx) {
        let Some(al) = &self.aligned else { return };
        let (fid, mid) = (al.fixed, al.moving);
        let (Some(fixed), Some(moving)) = (Self::pixels(ctx.doc, fid), Self::pixels(ctx.doc, mid)) else {
            ctx.status.error("A layer to stitch is gone");
            self.reset();
            return;
        };
        let (fi, mi) = (ctx.doc.index_of(fid).unwrap_or(0), ctx.doc.index_of(mid).unwrap_or(0));
        let opt = MergeOptions { prefer_upper: self.prefer_upper, fixed_on_top: fi > mi, want_warped: self.keep_sources };
        let al = al.al.clone();
        self.spawn((fid, mid), "Merging", move |p| JobResult::Merged(stitch::merge(&fixed, &moving, &al, opt, p)));
    }

    /// Checks the background job; returns true while it is still running.
    fn poll(&mut self, ctx: &mut ToolCtx) -> bool {
        let Some(job) = &self.job else { return false };
        let msg = match job.rx.try_recv() {
            Ok(m) => m,
            Err(mpsc::TryRecvError::Empty) => return true,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.job = None;
                ctx.status.error("Stitching stopped unexpectedly");
                return false;
            }
        };
        let (fid, mid) = job.layers;
        self.job = None;
        match msg {
            JobResult::Aligned(Ok(al)) => {
                ctx.status.info(format!(
                    "Aligned with {} matching points (median error {:.1} px, turned {:.1}°, scaled {:.2}×)",
                    al.matches.len(),
                    al.residual,
                    al.angle,
                    al.scale
                ));
                self.aligned = Some(Aligned { fixed: fid, moving: mid, al: Arc::new(al) });
                if self.auto_apply {
                    self.start_merge(ctx);
                }
            }
            JobResult::Aligned(Err(e)) | JobResult::Merged(Err(e)) => ctx.status.error(e),
            JobResult::Merged(Ok(out)) => {
                let Some(off) = ctx.doc.layer(fid).map(|l| l.offset) else {
                    ctx.status.error("The fixed layer is gone");
                    return false;
                };
                let at = |r: imged_core::Rect| (off.0 + r.x, off.1 + r.y);
                let apply = StitchApply {
                    fixed: fid,
                    moving: mid,
                    merged_offset: at(out.merged_rect),
                    merged: Arc::new(out.merged),
                    warped: out.warped.map(|(px, r)| (Arc::new(px), at(r))),
                };
                if ctx.exec(Command::Stitch(apply)).is_some() {
                    ctx.status.info(match out.conflicts {
                        0 => "Stitched".to_string(),
                        1 => "Stitched; 1 region differed between the photos".to_string(),
                        n => format!("Stitched; {n} regions differed between the photos"),
                    });
                    self.picks.clear();
                    self.aligned = None;
                }
            }
        }
        false
    }

    /// Drops picks whose layer no longer exists (undo, deletion).
    fn validate(&mut self, doc: &Document) {
        let before = self.picks.len();
        self.picks.retain(|p| doc.layer(p.layer).is_some());
        if self.picks.len() != before {
            self.aligned = None;
        }
        if let Some(a) = &self.aligned {
            if doc.layer(a.fixed).is_none() || doc.layer(a.moving).is_none() {
                self.aligned = None;
            }
        }
    }
}

impl Tool for StitchTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Stitch
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        self.validate(ctx.doc);
        if self.poll(ctx) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        if !stitch::AVAILABLE {
            ui.label(RichText::new("This build has no OpenCV: rebuild with `--features stitch` to stitch.").color(Color32::YELLOW));
        }
        let busy = self.busy();
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                ui.label("Align");
                let mut changed = ui.selectable_value(&mut self.mode, StitchMode::Board, "Board").on_hover_text("Writing on a whiteboard").changed();
                changed |= ui
                    .selectable_value(&mut self.mode, StitchMode::Projection, "Projection")
                    .on_hover_text("A document projected onto the board and scrolled between the photos: only the projected area is aligned and kept (click inside it)")
                    .changed();
                if changed {
                    self.aligned = None;
                }
            });
            ui.checkbox(&mut self.prefer_upper, "Prefer upper layer where they differ").on_hover_text(
                "Where the photos disagree (something was written or erased in between), take the layer that is higher in the layer list.\nOff: take the layer that has ink there.",
            );
            ui.checkbox(&mut self.keep_sources, "Keep the photos as hidden layers")
                .on_hover_text("The second photo is kept warped into place, for touching up the result by hand");
            ui.checkbox(&mut self.auto_apply, "Stitch right after aligning").on_hover_text("Off: check the matched points first, then press Stitch (Enter)");
        });

        let name = |id: LayerId| ctx.doc.layer(id).map(|l| l.name.clone()).unwrap_or_default();
        match self.picks.len() {
            0 => ui.label("1. Click a letter or symbol on the layer that stays in place."),
            1 => ui.label(format!("2. Click the same letter on the other layer (1: {}).", name(self.picks[0].layer))),
            _ => ui.label(format!("1: {}\n2: {} (aligned onto 1)", name(self.picks[0].layer), name(self.picks[1].layer))),
        };
        if let Some(job) = &self.job {
            let (f, s) = job.progress.lock().map(|g| g.clone()).unwrap_or((0.0, String::new()));
            ui.add(egui::ProgressBar::new(f).text(s).show_percentage());
        }
        if let Some(a) = &self.aligned {
            let al = &a.al;
            ui.label(
                RichText::new(format!(
                    "{} matching points · median error {:.1} px · turned {:.1}° · scaled {:.2}×",
                    al.matches.len(),
                    al.residual,
                    al.angle,
                    al.scale
                ))
                .small(),
            );
        }
        ui.horizontal(|ui| {
            if ui.add_enabled(!busy && self.aligned.is_some(), egui::Button::new("Stitch")).on_hover_text("Merge the two layers (Enter)").clicked() {
                self.start_merge(ctx);
            }
            if ui.add_enabled(!busy && self.picks.len() == 2, egui::Button::new("Align again")).clicked() {
                self.start_align(ctx);
            }
            if ui.add_enabled(!self.picks.is_empty() || busy, egui::Button::new("Reset")).on_hover_text("Esc").clicked() {
                self.reset();
            }
        });
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        let name_at = |p| layer_at(ctx.doc, p).and_then(|id| ctx.doc.layer(id)).map(|l| l.name.clone());
        match ev.kind {
            PointerKind::Hover => {
                self.hover = name_at(ev.raw);
                return;
            }
            PointerKind::Press => {}
            _ => return,
        }
        if self.busy() {
            return;
        }
        if ev.button == Button::Secondary {
            self.picks.pop();
            self.aligned = None;
            return;
        }
        let Some(id) = layer_at(ctx.doc, ev.raw) else {
            ctx.status.error("Nothing to stitch here: click on a photo");
            return;
        };
        if self.picks.len() >= 2 {
            self.reset();
        }
        if self.picks.first().is_some_and(|p| p.layer == id) {
            ctx.status.error("That is the same layer: click the same letter on the other photo (hide or move layers to reach it)");
            return;
        }
        let l = ctx.doc.layer(id).unwrap();
        let local = [ev.raw.x - l.offset.0 as f32 - 0.5, ev.raw.y - l.offset.1 as f32 - 0.5];
        self.picks.push(Pick { layer: id, canvas: ev.raw, local });
        if self.picks.len() == 2 {
            if stitch::AVAILABLE {
                self.start_align(ctx);
            } else {
                ctx.status.error("This build has no OpenCV: rebuild with --features stitch");
            }
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        let dot = |p: Vec2, c: Color32, r: f32| painter.circle_filled(view.to_screen(p), r, c);
        if let Some(a) = &self.aligned {
            let (Some(fl), Some(ml)) = (doc.layer(a.fixed), doc.layer(a.moving)) else { return };
            let fo = Vec2::new(fl.offset.0 as f32 + 0.5, fl.offset.1 as f32 + 0.5);
            let mo = Vec2::new(ml.offset.0 as f32 + 0.5, ml.offset.1 as f32 + 0.5);
            let step = (a.al.matches.len() / 400).max(1);
            for (f, m) in a.al.matches.iter().step_by(step) {
                dot(Vec2::new(fo.x + f[0], fo.y + f[1]), Color32::from_rgb(40, 200, 90), 2.5);
                if ml.visible {
                    dot(Vec2::new(mo.x + m[0], mo.y + m[1]), Color32::from_rgb(40, 160, 255), 2.5);
                }
            }
            for q in [a.al.quad_fixed.map(|q| (q, fo)), a.al.quad_moving.map(|q| (q, mo))].into_iter().flatten() {
                let (q, o) = q;
                let mut pts: Vec<_> = q.iter().map(|p| view.to_screen(Vec2::new(o.x + p[0], o.y + p[1]))).collect();
                pts.push(pts[0]);
                painter.add(egui::Shape::line(pts, Stroke::new(1.5, Color32::from_rgb(255, 200, 0))));
            }
        }
        for (i, p) in self.picks.iter().enumerate() {
            let s = view.to_screen(p.canvas);
            painter.circle_stroke(s, 9.0, Stroke::new(3.0, Color32::BLACK));
            painter.circle_stroke(s, 9.0, Stroke::new(1.5, Color32::from_rgb(255, 80, 60)));
            let t = egui::pos2(s.x + 12.0, s.y - 12.0);
            painter.text(t + egui::vec2(1.0, 1.0), Align2::LEFT_BOTTOM, (i + 1).to_string(), FontId::proportional(15.0), Color32::BLACK);
            painter.text(t, Align2::LEFT_BOTTOM, (i + 1).to_string(), FontId::proportional(15.0), Color32::from_rgb(255, 80, 60));
        }
    }

    fn cancel(&mut self, _ctx: &mut ToolCtx) {
        self.reset();
    }

    fn commit(&mut self, ctx: &mut ToolCtx) {
        if !self.busy() && self.aligned.is_some() {
            self.start_merge(ctx);
        }
    }

    /// Keep the clicks and any running job when switching tools (to hide or
    /// move a layer in between); Esc or Reset clears them.
    fn deactivate(&mut self, _ctx: &mut ToolCtx) {}

    fn status(&self) -> Option<String> {
        if let Some(job) = &self.job {
            let (f, s) = job.progress.lock().map(|g| g.clone()).unwrap_or_default();
            return Some(format!("{s}… {:.0}%", f * 100.0));
        }
        self.hover.as_ref().map(|n| format!("Layer: {n}"))
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "stitch.prefer_upper", &mut self.prefer_upper);
        p.bool(load, "stitch.keep_sources", &mut self.keep_sources);
        p.bool(load, "stitch.auto_apply", &mut self.auto_apply);
        let mut projection = self.mode == StitchMode::Projection;
        p.bool(load, "stitch.projection", &mut projection);
        if load {
            self.mode = if projection { StitchMode::Projection } else { StitchMode::Board };
        }
    }
}
