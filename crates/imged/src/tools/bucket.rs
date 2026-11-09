use eframe::egui::{Painter, Ui};
use imged_core::command::RegionKind;
use imged_core::ops::fill::RegionParams;
use imged_core::{Command, Document};

use super::region_preview::{paint_tint, RegionPreview};
use super::{tolerance_slider, Button, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

#[derive(Default)]
pub struct BucketTool {
    params: RegionParams,
    sample_merged: bool,
    preview: RegionPreview,
}

impl BucketTool {
    /// A fill that starts a new (empty) layer detects the area from the
    /// visible image, as filling the empty layer would cover everything.
    fn samples_merged(&self, ctx: &ToolCtx, b: Button) -> bool {
        self.sample_merged || ctx.would_start_layer(ToolKind::Bucket, b)
    }

    fn update_preview(&mut self, ctx: &mut ToolCtx) {
        let merged = self.samples_merged(ctx, Button::Primary);
        let tint = paint_tint(ctx.colors.primary);
        self.preview.update(ctx, RegionKind::Fill(self.params), true, merged, tint);
    }
}

impl Tool for BucketTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Bucket
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        tolerance_slider(ui, &mut self.params.tolerance, "Tolerance");
        ui.checkbox(&mut self.params.contiguous, "Contiguous")
            .on_hover_text("Off: fill every matching pixel in the layer");
        ui.checkbox(&mut self.params.diagonal, "Connect diagonally");
        ui.checkbox(&mut self.sample_merged, "Sample all layers").on_hover_text(
            "Detect the area from the visible image instead of the active layer (always, when the fill starts a new layer)",
        );
        self.update_preview(ctx);
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Hover => {
                self.preview.hover(Some(ev.pos));
                self.update_preview(ctx);
                return;
            }
            PointerKind::Press => {}
            _ => return,
        }
        self.preview.hover(Some(ev.pos));
        let color = ctx.color_for(ev.button);
        let sample_merged = self.samples_merged(ctx, ev.button);
        ctx.begin_drawing(ToolKind::Bucket, ev.button);
        let cmd = Command::Fill { at: ev.pos, color, params: self.params, sample_merged, mode: ctx.settings.paint_mode };
        if let Some(imged_core::command::Outcome::Changed) = ctx.exec(cmd) {
            ctx.colors.mark_used(color);
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        self.preview.paint(painter, view, doc);
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "bucket.contiguous", &mut self.params.contiguous);
        p.bool(load, "bucket.diagonal", &mut self.params.diagonal);
        p.bool(load, "bucket.sample_all", &mut self.sample_merged);
    }
}
