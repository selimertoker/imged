//! Layer panel: list with visibility, multi-select, groups (shown
//! as a single entry), thumbnails, per-layer history and properties.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, RichText, Sense, Stroke, TextureHandle, TextureOptions, Ui, Vec2};
use imged_core::composite::{composite_views, LayerView};
use imged_core::layer::{GroupId, LayerId};
use imged_core::{Command, Depth, Document, Pixels, Rect};

use super::widgets::{paint_checker_rect, paint_icon, Icon, ACCENT};
use crate::actions::Action;

const THUMB: f32 = 34.0;
/// Name color of groups.
pub const GROUP_BLUE: Color32 = Color32::from_rgb(0x5A, 0xA9, 0xFF);
/// Color of the filter-layer icon when on.
pub const FILTER_ON: Color32 = Color32::from_rgb(0xF0, 0xA8, 0x48);

#[derive(Default)]
pub struct LayersPanel {
    /// Keyed by layer id, or by group id for groups.
    thumbs: HashMap<u64, TextureHandle>,
    thumbs_rev: Option<u64>,
    thumbs_time: Option<Instant>,
    /// Layer being renamed and the edited name.
    renaming: Option<(LayerId, String)>,
    rename_needs_focus: bool,
    /// Entry index (top to bottom) being dragged to reorder.
    dragging: Option<usize>,
}

/// One entry in the list: a single layer or a whole group.
struct Entry {
    /// Layer whose name/properties represent the entry (top member of a group).
    id: LayerId,
    group: Option<(GroupId, usize)>,
    visible: bool,
    name: String,
    depth: Depth,
    bounds: Rect,
    opacity: f32,
    contains_active: bool,
    ticked: bool,
    /// Layer ids of the entry, bottom to top.
    members: Vec<LayerId>,
    /// A filter layer (single layers only), and what its filter does.
    filter: bool,
    summary: String,
}

fn thumbnail(px: &Pixels) -> ColorImage {
    let (w, h) = (px.width().max(1), px.height().max(1));
    let scale = (48.0 / w.max(h) as f32).min(1.0);
    let (tw, th) = (((w as f32 * scale).round() as usize).max(1), ((h as f32 * scale).round() as usize).max(1));
    let mut out = Vec::with_capacity(tw * th);
    for y in 0..th {
        for x in 0..tw {
            let sx = ((x as f32 + 0.5) / scale) as u32;
            let sy = ((y as f32 + 0.5) / scale) as u32;
            let p = px.get(sx.min(w - 1), sy.min(h - 1));
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            out.push(Color32::from_rgba_unmultiplied(q(p[0]), q(p[1]), q(p[2]), q(p[3])));
        }
    }
    ColorImage::new([tw, th], out)
}

fn entries(doc: &Document) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut i = doc.layers.len();
    while i > 0 {
        i -= 1;
        let block = doc.block_at(i);
        let members = &doc.layers[block.clone()];
        let top = &doc.layers[block.end - 1];
        let group = top.group.filter(|_| block.len() > 1).map(|g| (g, block.len()));
        out.push(Entry {
            id: top.id,
            group,
            visible: members.iter().any(|l| l.visible),
            name: top.name.clone(),
            depth: members.iter().map(|l| l.depth()).max().unwrap_or(top.depth()),
            bounds: members.iter().fold(Rect::default(), |a, l| a.union(&l.bounds())),
            opacity: top.opacity,
            contains_active: members.iter().any(|l| l.id == doc.active),
            ticked: members.iter().any(|l| doc.selected.contains(&l.id)),
            members: members.iter().map(|l| l.id).collect(),
            filter: group.is_none() && top.is_filter(),
            summary: top.filter.summary(),
        });
        i = block.start;
    }
    out
}

impl LayersPanel {
    fn refresh_thumbs(&mut self, ctx: &egui::Context, doc: &Document) {
        let rev = doc.revision();
        let due = self.thumbs_time.is_none_or(|t| t.elapsed() > Duration::from_millis(300));
        if self.thumbs_rev == Some(rev) || !due {
            if self.thumbs_rev != Some(rev) {
                ctx.request_repaint_after(Duration::from_millis(300));
            }
            return;
        }
        self.thumbs_rev = Some(rev);
        self.thumbs_time = Some(Instant::now());
        let mut keep: Vec<u64> = Vec::new();
        let set = |thumbs: &mut HashMap<u64, TextureHandle>, key: u64, img: ColorImage| match thumbs.get_mut(&key) {
            Some(t) => t.set(img, TextureOptions::LINEAR),
            None => {
                let t = ctx.load_texture(format!("thumb-{key}"), img, TextureOptions::LINEAR);
                thumbs.insert(key, t);
            }
        };
        let mut i = 0;
        while i < doc.layers.len() {
            let block = doc.block_at(i);
            i = block.end;
            let members = &doc.layers[block.clone()];
            match members[0].group.filter(|_| members.len() > 1) {
                Some(g) => {
                    // Composite of the group, as it looks merged.
                    let views: Vec<LayerView> = members
                        .iter()
                        .map(|l| match doc.preview.get(&l.id) {
                            Some(p) => LayerView { pixels: &p.pixels, offset: p.offset, opacity: 1.0, filter: None },
                            None => LayerView { opacity: 1.0, filter: None, ..LayerView::of(l) },
                        })
                        .collect();
                    let r = views.iter().fold(Rect::default(), |a, v| a.union(&v.bounds()));
                    let img = Pixels::F32(composite_views(&views, r, None));
                    set(&mut self.thumbs, g, thumbnail(&img));
                    keep.push(g);
                }
                None => {
                    let l = &members[0];
                    let px = doc.preview.get(&l.id).map(|p| &*p.pixels).unwrap_or(&l.pixels);
                    set(&mut self.thumbs, l.id, thumbnail(px));
                    keep.push(l.id);
                }
            }
        }
        self.thumbs.retain(|k, _| keep.contains(k));
    }

    pub fn ui(&mut self, ui: &mut Ui, doc: &mut Document, actions: &mut Vec<Action>) {
        self.refresh_thumbs(ui.ctx(), doc);
        let cmd = |actions: &mut Vec<Action>, c: Command| actions.push(Action::Cmd(c));

        // Toolbar.
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            if ui.small_button("New").on_hover_text("New empty layer (Ctrl+Shift+N)").clicked() {
                cmd(actions, Command::NewLayer);
            }
            if ui.small_button("Duplicate").on_hover_text("Ctrl+J").clicked() {
                cmd(actions, Command::DuplicateLayer);
            }
            if ui.small_button("Delete").on_hover_text("Delete").clicked() {
                cmd(actions, Command::DeleteLayer);
            }
            // Split one layer into channels, or merge 3-4 layers as channels.
            let set = doc.channel_merge_set();
            if (3..=4).contains(&set.len()) {
                let label = if set.len() == 3 { "Merge as RGB" } else { "Merge as RGBA" };
                if ui
                    .small_button(label)
                    .on_hover_text(
                        "Use the active and ticked layers as the color channels of one layer.\n\
                         Layers named \"… R\", \"… G\", \"… B\", \"… A\" go to their channel; otherwise top to bottom = R, G, B, A.",
                    )
                    .clicked()
                {
                    cmd(actions, Command::MergeChannels(set));
                }
            } else if ui
                .small_button("Split channels")
                .on_hover_text("Replace the active layer with 4 grayscale layers: its R, G, B and A channels.\nTick 3 or 4 layers to merge them back.")
                .clicked()
            {
                cmd(actions, Command::SplitChannels);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            if ui.small_button("Merge down").on_hover_text("Merge with the layer below (Ctrl+M)").clicked() {
                cmd(actions, Command::MergeDown);
            }
            let ticked = !doc.selected.is_empty();
            if ui
                .add_enabled(ticked, egui::Button::new("Merge ticked").small())
                .on_hover_text("Merge the active layer and all ticked layers into one")
                .clicked()
            {
                cmd(actions, Command::MergeSelected);
            }
            if ui
                .add_enabled(ticked, egui::Button::new("Group").small())
                .on_hover_text("Group the active and ticked layers so they act as one layer.\nThey can be separated again. (Ctrl+G)")
                .clicked()
            {
                cmd(actions, Command::Group);
            }
            if ui
                .add_enabled(doc.layers.len() > 1, egui::Button::new("Tile").small())
                .on_hover_text(
                    "Arrange the layers side by side and on top of each other (no overlaps, roughly square)\n\
                     and fit the canvas to them. A plain single-color bottom layer stays as the background.\n\
                     With layers ticked, only those (and the active one) are tiled.",
                )
                .clicked()
            {
                cmd(actions, Command::TileLayers { gap: 0 });
            }
        });
        ui.separator();

        // List, top first. Groups are a single entry.
        let mut set_active = None;
        let mut toggle_vis = None;
        let mut toggle_tick = None;
        let mut toggle_filter = None;
        let list = entries(doc);
        let mut rects = Vec::with_capacity(list.len());
        let scroll = egui::ScrollArea::vertical().id_salt("layers-list").max_height(ui.available_height() - 140.0).show(ui, |ui| {
            for (i, e) in list.iter().enumerate() {
                let r = self.row(ui, doc, e, &mut set_active, &mut toggle_vis, &mut toggle_tick, &mut toggle_filter);
                if r.1 {
                    self.dragging = Some(i);
                }
                rects.push(r.0);
            }
        });
        // Drag a layer up or down to reorder.
        if let Some(from) = self.dragging {
            let (pos, released) = ui.input(|i| (i.pointer.interact_pos(), i.pointer.any_released()));
            let slot = pos.map(|p| rects.iter().take_while(|r| p.y > r.center().y).count());
            if let Some(slot) = slot {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                let y = if slot < rects.len() { rects[slot].top() } else { rects.last().map_or(0.0, |r| r.bottom()) };
                let clip = scroll.inner_rect;
                if y >= clip.top() - 2.0 && y <= clip.bottom() + 2.0 {
                    ui.painter().hline(clip.x_range(), y, Stroke::new(2.0, ACCENT));
                }
            }
            if released {
                self.dragging = None;
                if let Some(slot) = slot {
                    // Entries are top to bottom; slot is the gap before entry `slot`.
                    let to = if slot > from { slot - 1 } else { slot };
                    if to != from {
                        let mut blocks: Vec<&Entry> = list.iter().collect();
                        let moved = blocks.remove(from);
                        blocks.insert(to, moved);
                        let order: Vec<LayerId> =
                            blocks.iter().rev().flat_map(|e| e.members.iter().copied()).collect();
                        cmd(actions, Command::Reorder(order));
                    }
                }
            }
        }
        if let Some(id) = set_active {
            doc.set_active(id);
        }
        if let Some(id) = toggle_vis {
            cmd(actions, Command::ToggleVisible(id));
        }
        if let Some(id) = toggle_filter {
            let on = !doc.layer(id).is_some_and(|l| l.filter_on);
            cmd(actions, Command::SetFilterOn { id, on });
        }
        if let Some(id) = toggle_tick {
            // Ticking a group entry ticks all of its members.
            let members = match doc.layer(id).and_then(|l| l.group) {
                Some(g) => doc.group_members(g),
                None => vec![id],
            };
            let on = !members.iter().any(|m| doc.selected.contains(m));
            for m in members {
                if on {
                    doc.selected.insert(m);
                } else {
                    doc.selected.remove(&m);
                }
            }
        }

        ui.separator();
        self.properties(ui, doc, actions);
    }

    /// Draws one entry. Returns its rectangle and whether a drag started on it.
    #[allow(clippy::too_many_arguments)]
    fn row(
        &mut self,
        ui: &mut Ui,
        doc: &mut Document,
        e: &Entry,
        set_active: &mut Option<LayerId>,
        toggle_vis: &mut Option<LayerId>,
        toggle_tick: &mut Option<LayerId>,
        toggle_filter: &mut Option<LayerId>,
    ) -> (egui::Rect, bool) {
        let mut drag_started = false;
        let id = e.id;
        let frame = egui::Frame::new()
            .fill(if e.contains_active { ACCENT.gamma_multiply(0.25) } else { Color32::TRANSPARENT })
            .corner_radius(4.0)
            .inner_margin(2.0);
        let resp = frame.show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                let mut ticked = e.ticked;
                if ui.checkbox(&mut ticked, "").on_hover_text("Tick for merge / group").changed() {
                    *toggle_tick = Some(id);
                }
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    if eye(ui, e.visible).clicked() {
                        *toggle_vis = Some(id);
                    }
                    if e.group.is_none() && filter_toggle(ui, e.filter).clicked() {
                        *toggle_filter = Some(id);
                    }
                });
                let (r, thumb_resp) = ui.allocate_exact_size(Vec2::splat(THUMB), Sense::click_and_drag());
                if thumb_resp.clicked() {
                    *set_active = Some(id);
                }
                if thumb_resp.drag_started() {
                    drag_started = true;
                }
                thumb_resp.on_hover_cursor(egui::CursorIcon::Grab);
                paint_checker_rect(ui.painter(), r, 5.0);
                let key = e.group.map_or(id, |(g, _)| g);
                if let Some(t) = self.thumbs.get(&key) {
                    let s = t.size_vec2();
                    let fit = (THUMB / s.x).min(THUMB / s.y);
                    let tr = egui::Rect::from_center_size(r.center(), s * fit);
                    ui.painter().image(t.id(), tr, egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(1., 1.)), Color32::WHITE);
                }
                let border = if e.group.is_some() { Stroke::new(1.5, GROUP_BLUE) } else { Stroke::new(1.0, Color32::from_gray(80)) };
                ui.painter().rect_stroke(r, 0.0, border, egui::StrokeKind::Outside);
                ui.vertical(|ui| {
                    if self.renaming.as_ref().is_some_and(|(rid, _)| *rid == id) {
                        self.rename_box(ui, doc, id);
                        return;
                    }
                    let mut name = RichText::new(&e.name);
                    if e.group.is_some() {
                        name = name.color(GROUP_BLUE).strong();
                    }
                    if !e.visible {
                        name = name.weak();
                    }
                    let lbl = ui
                        .add(egui::Label::new(name).sense(Sense::click_and_drag()).truncate().selectable(false))
                        .on_hover_text("Click to select, double-click to rename, drag to move up or down");
                    if lbl.drag_started() {
                        drag_started = true;
                    }
                    if lbl.double_clicked() {
                        self.renaming = Some((id, e.name.clone()));
                        self.rename_needs_focus = true;
                    } else if lbl.clicked() {
                        *set_active = Some(id);
                    }
                    let detail = match e.group {
                        Some((_, n)) => format!("group of {n} · {}×{}", e.bounds.w, e.bounds.h),
                        None if e.filter => format!(
                            "filter: {}{}",
                            e.summary,
                            if e.opacity < 1.0 { format!(" · {:.0}%", e.opacity * 100.0) } else { String::new() }
                        ),
                        None => format!(
                            "{} · {}×{}{}",
                            e.depth.short(),
                            e.bounds.w,
                            e.bounds.h,
                            if e.opacity < 1.0 { format!(" · {:.0}%", e.opacity * 100.0) } else { String::new() }
                        ),
                    };
                    ui.add(egui::Label::new(RichText::new(detail).small().weak()).truncate().selectable(false));
                });
            });
        });
        (resp.response.rect, drag_started)
    }

    /// The inline rename box: Enter or clicking elsewhere keeps the name, Esc cancels.
    fn rename_box(&mut self, ui: &mut Ui, doc: &mut Document, id: LayerId) {
        let Some((_, name)) = &mut self.renaming else { return };
        let te = ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
        if self.rename_needs_focus {
            te.request_focus();
            self.rename_needs_focus = false;
            return;
        }
        let cancel = ui.input(|i| i.key_pressed(egui::Key::Escape));
        if cancel {
            self.renaming = None;
            te.surrender_focus();
        } else if te.lost_focus() || !te.has_focus() {
            let (_, n) = self.renaming.take().unwrap();
            if !n.trim().is_empty() {
                doc.rename(id, n.trim().to_string());
            }
        }
    }

    fn properties(&mut self, ui: &mut Ui, doc: &mut Document, actions: &mut Vec<Action>) {
        let id = doc.active;
        let Some(l) = doc.layer(id) else { return };
        let group = l.group.map(|g| doc.group_members(g)).filter(|m| m.len() > 1);
        let (name, opacity, depth, offset, size) = (l.name.clone(), l.opacity, l.depth(), l.offset, (l.width(), l.height()));

        match &group {
            Some(m) => {
                let b = doc.targets_bounds();
                ui.horizontal(|ui| {
                    let top = doc.layer(*m.last().unwrap()).map(|l| l.name.clone()).unwrap_or_default();
                    ui.label(RichText::new(top).strong().color(GROUP_BLUE));
                    ui.label(RichText::new(format!("({} layers)", m.len())).small().weak());
                    if ui.small_button("Ungroup").on_hover_text("Separate the grouped layers again (Ctrl+Shift+G)").clicked() {
                        actions.push(Action::Cmd(Command::Ungroup));
                    }
                });
                ui.label(RichText::new(format!("{}×{} at {}, {}", b.w, b.h, b.x, b.y)).small().weak());
            }
            None => {
                ui.label(RichText::new(name).strong());
            }
        }
        let mut o = opacity * 100.0;
        if ui.add(egui::Slider::new(&mut o, 0.0..=100.0).update_while_editing(false).suffix("%").text("Opacity")).changed() {
            doc.set_opacity(id, o / 100.0);
        }
        ui.horizontal(|ui| {
            ui.label("Depth");
            let mut d = depth;
            egui::ComboBox::from_id_salt("layer-depth").selected_text(d.label()).show_ui(ui, |ui| {
                for opt in Depth::ALL {
                    ui.selectable_value(&mut d, opt, opt.label());
                }
            });
            if d != depth {
                actions.push(Action::Cmd(Command::ConvertDepth(d)));
            }
        });
        if group.is_none() {
            ui.label(RichText::new(format!("{}×{} at {}, {}", size.0, size.1, offset.0, offset.1)).small().weak());
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button("Fit to canvas")
                .on_hover_text("Resize the layer's area to the canvas (no scaling; pixels outside are cut)")
                .clicked()
            {
                actions.push(Action::Cmd(Command::LayerToCanvasSize));
            }
            if ui
                .small_button("Fit the canvas")
                .on_hover_text("Resize the canvas so every layer fits inside it")
                .clicked()
            {
                actions.push(Action::Cmd(Command::FitCanvasToLayers));
            }
            if ui.small_button("Trim").on_hover_text("Crop the layer to its non-transparent pixels").clicked() {
                actions.push(Action::Cmd(Command::CropLayerToContent));
            }
            canvas_size_button(ui, actions);
        });
    }

}

fn canvas_size_button(ui: &mut Ui, actions: &mut Vec<Action>) {
    if ui.small_button("Canvas size").on_hover_text("Change the canvas size without scaling (presets, anchor)").clicked() {
        actions.push(Action::ShowDialog(crate::actions::Dialog::CanvasSize));
    }
}

/// The filter-layer toggle (a funnel), under the eye.
fn filter_toggle(ui: &mut Ui, on: bool) -> egui::Response {
    let (r, resp) = ui.allocate_exact_size(Vec2::new(18.0, 14.0), Sense::click());
    let c = if on {
        FILTER_ON
    } else if resp.hovered() {
        Color32::from_gray(170)
    } else {
        Color32::from_gray(75)
    };
    paint_icon(ui.painter(), egui::Rect::from_center_size(r.center(), Vec2::new(12.0, 12.0)), Icon::Funnel, c);
    resp.on_hover_text(if on {
        "Filter layer: its color filters change the layers below, where its pixels are opaque \
         (everywhere if it is empty). Click to make it a normal layer."
    } else {
        "Make this a filter layer: color filters applied to it then change the layers below it, \
         and its pixels only mark where (empty = everywhere)"
    })
}

fn eye(ui: &mut Ui, visible: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
    let c = if visible { ui.visuals().strong_text_color() } else { ui.visuals().weak_text_color() };
    let p = ui.painter();
    let center = rect.center();
    let pts: Vec<egui::Pos2> = (0..=16)
        .map(|k| {
            let t = k as f32 / 16.0 * std::f32::consts::PI;
            center + Vec2::new(-t.cos() * 7.0, -t.sin() * 4.5)
        })
        .chain((0..=16).map(|k| {
            let t = k as f32 / 16.0 * std::f32::consts::PI;
            center + Vec2::new(t.cos() * 7.0, t.sin() * 4.5)
        }))
        .collect();
    p.add(egui::Shape::closed_line(pts, Stroke::new(1.2, c)));
    if visible {
        p.circle_filled(center, 2.4, c);
    } else {
        p.line_segment([center + Vec2::new(-7.0, 5.0), center + Vec2::new(7.0, -5.0)], Stroke::new(1.2, c));
    }
    resp.on_hover_text(if visible { "Hide" } else { "Show" })
}
