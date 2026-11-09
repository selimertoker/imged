//! Selection tools. By default the selected pixels move to a new layer;
//! with "Move to new layer" off, the area stays selected (marching ants) so
//! filters and transforms can use it.

use eframe::egui::{self, Painter, Ui};
use imged_core::ops::fill::RegionParams;
use imged_core::ops::outline::OutlineParams;
use imged_core::{Command, Document, Rect, Vec2};

use super::region_preview::{RegionPreview, SELECT_TINT};
use imged_core::command::RegionKind;
use super::{after_selection, dashed_rect, select_mode, selection_options, tolerance_slider, PointerEvent, PointerKind, Tool, ToolCtx, ToolKind};
use crate::view::ViewXform;

#[derive(Default)]
pub struct BoxSelectTool {
    drag: Option<(Vec2, Vec2)>,
}

fn box_rect(a: Vec2, b: Vec2) -> Rect {
    Rect::from_corners(a.x.round() as i32, a.y.round() as i32, b.x.round() as i32, b.y.round() as i32)
}

impl Tool for BoxSelectTool {
    fn kind(&self) -> ToolKind {
        ToolKind::BoxSelect
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        selection_options(ui, ctx);
        if let Some((a, b)) = self.drag {
            let r = box_rect(a, b);
            ui.label(format!("{} × {} at {}, {}", r.w, r.h, r.x, r.y));
        }
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        match ev.kind {
            PointerKind::Press => self.drag = Some((ev.pos, ev.pos)),
            PointerKind::Drag => {
                if let Some((a, _)) = self.drag {
                    let mut b = ev.pos;
                    if ev.shift {
                        let d = b - a;
                        let s = d.x.abs().max(d.y.abs());
                        b = Vec2::new(a.x + s * d.x.signum(), a.y + s * d.y.signum());
                    }
                    self.drag = Some((a, b));
                }
            }
            PointerKind::Release => {
                let Some((a, b)) = self.drag.take() else { return };
                let r = box_rect(a, b);
                if r.is_empty() {
                    // A click without dragging deselects (when keeping selections).
                    if !ctx.settings.select_to_layer && !ctx.settings.select_add && ctx.doc.selection.is_some() {
                        ctx.exec(Command::SelectNone);
                    }
                    return;
                }
                let out = ctx.exec(Command::SelectRect { rect: r, mode: select_mode(ctx) });
                after_selection(ctx, out);
            }
            PointerKind::Hover => {}
        }
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, _doc: &Document) {
        if let Some((a, b)) = self.drag {
            let r = box_rect(a, b);
            dashed_rect(painter, view.rect_to_screen(r));
        }
    }

    fn cancel(&mut self, _ctx: &mut ToolCtx) {
        self.drag = None;
    }

    fn uses_snapping(&self) -> bool {
        true
    }

    fn status(&self) -> Option<String> {
        let (a, b) = self.drag?;
        let r = box_rect(a, b);
        Some(format!("{} × {} px", r.w, r.h))
    }
}

#[derive(Default)]
pub struct MagicWandTool {
    params: RegionParams,
    sample_merged: bool,
    preview: RegionPreview,
}

impl Tool for MagicWandTool {
    fn kind(&self) -> ToolKind {
        ToolKind::MagicWand
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        tolerance_slider(ui, &mut self.params.tolerance, "Tolerance");
        ui.checkbox(&mut self.params.contiguous, "Contiguous");
        ui.checkbox(&mut self.params.diagonal, "Connect diagonally");
        ui.checkbox(&mut self.sample_merged, "Sample all layers");
        selection_options(ui, ctx);
        self.preview.update(ctx, RegionKind::Wand(self.params), false, self.sample_merged, SELECT_TINT);
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        self.preview.hover(Some(ev.pos));
        if ev.kind == PointerKind::Hover {
            self.preview.update(ctx, RegionKind::Wand(self.params), false, self.sample_merged, SELECT_TINT);
        }
        if ev.kind != PointerKind::Press {
            return;
        }
        let out = ctx.exec(Command::MagicWand {
            at: ev.pos,
            params: self.params,
            sample_merged: self.sample_merged,
            mode: select_mode(ctx),
        });
        after_selection(ctx, out);
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        self.preview.paint(painter, view, doc);
    }

    fn cursor(&self) -> egui::CursorIcon {
        egui::CursorIcon::PointingHand
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "wand.contiguous", &mut self.params.contiguous);
        p.bool(load, "wand.diagonal", &mut self.params.diagonal);
        p.bool(load, "wand.sample_all", &mut self.sample_merged);
    }
}

#[derive(Default)]
pub struct OutlineSelectTool {
    params: OutlineParams,
    sample_merged: bool,
    preview: RegionPreview,
}

impl Tool for OutlineSelectTool {
    fn kind(&self) -> ToolKind {
        ToolKind::OutlineSelect
    }

    fn options_ui(&mut self, ui: &mut Ui, ctx: &mut ToolCtx) {
        tolerance_slider(ui, &mut self.params.tolerance, "Outline tolerance");
        ui.checkbox(&mut self.params.include_outline, "Include the outline");
        ui.add_enabled(!self.params.content_box, egui::Checkbox::new(&mut self.params.as_rectangle, "Select bounding box"))
            .on_hover_text("Select the rectangle around the detected area instead of its exact shape");
        ui.checkbox(&mut self.params.content_box, "Content on a solid background").on_hover_text(
            "Instead of an outline, find the solid color around the click and select the smallest rectangle with only that color \
             outside it, e.g. the bounding box of text on a plain background. Colors within the outline tolerance count as the solid color.",
        );
        ui.checkbox(&mut self.sample_merged, "Sample all layers");
        selection_options(ui, ctx);
        self.preview.update(ctx, RegionKind::Outline(self.params), false, self.sample_merged, SELECT_TINT);
    }

    fn pointer(&mut self, ev: &PointerEvent, ctx: &mut ToolCtx) {
        self.preview.hover(Some(ev.pos));
        if ev.kind == PointerKind::Hover {
            self.preview.update(ctx, RegionKind::Outline(self.params), false, self.sample_merged, SELECT_TINT);
        }
        if ev.kind != PointerKind::Press {
            return;
        }
        let out = ctx.exec(Command::SelectOutline {
            at: ev.pos,
            params: self.params,
            sample_merged: self.sample_merged,
            mode: select_mode(ctx),
        });
        after_selection(ctx, out);
    }

    fn overlay(&self, painter: &Painter, view: &ViewXform, doc: &Document) {
        self.preview.paint(painter, view, doc);
    }

    fn cursor(&self) -> egui::CursorIcon {
        egui::CursorIcon::PointingHand
    }

    fn sync_prefs(&mut self, p: &mut crate::persist::Prefs, load: bool) {
        p.bool(load, "outline.include_outline", &mut self.params.include_outline);
        p.bool(load, "outline.bounding_box", &mut self.params.as_rectangle);
        p.bool(load, "outline.content_box", &mut self.params.content_box);
        p.bool(load, "outline.sample_all", &mut self.sample_merged);
    }
}
