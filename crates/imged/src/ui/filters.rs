//! Color filters (in the "Transform & filters" section), with live preview.
//!
//! On a normal layer the sliders preview a change that Apply bakes into the
//! pixels. On a filter layer they show and edit the layer's own settings
//! live (no Apply): its filters change the layers below it.

use eframe::egui::{self, RichText, Ui};
use imged_core::command::preview_filter;
use imged_core::layer::LayerId;
use imged_core::ops::filters::{Adjustments, Filter, LayerFilter};
use imged_core::ops::transform::Scope;
use imged_core::{Command, Document};

use crate::actions::{Action, Dialog};

pub struct FiltersPanel {
    adj: Adjustments,
    previewing: bool,
    /// Channels for the swap operation (0 = R … 3 = A).
    swap: (usize, usize),
    scope: Scope,
    /// A filter layer being changed live, with its settings before.
    edit: Option<(LayerId, LayerFilter)>,
}

impl Default for FiltersPanel {
    fn default() -> Self {
        FiltersPanel { adj: Adjustments::default(), previewing: false, swap: (0, 1), scope: Scope::Layer, edit: None }
    }
}

const CHANNELS: [&str; 4] = ["R", "G", "B", "A"];

/// The tone sliders. Returns (changed, a slider is held).
fn sliders(ui: &mut Ui, a: &mut Adjustments) -> (bool, bool) {
    let mut changed = false;
    let mut held = false;
    let mut track = |r: egui::Response| {
        changed |= r.changed();
        held |= r.dragged() || r.is_pointer_button_down_on();
        r
    };
    track(ui.add(egui::Slider::new(&mut a.exposure, -5.0..=5.0).update_while_editing(false).suffix(" EV").text("Exposure")))
        .on_hover_text("Multiplies the colors by 2^EV (+1 doubles the light)");
    track(ui.add(egui::Slider::new(&mut a.brightness, -1.0..=1.0).update_while_editing(false).text("Brightness")))
        .on_hover_text("Adds to every color channel: +1 turns everything white, -1 black");
    track(ui.add(egui::Slider::new(&mut a.contrast, 0.0..=3.0).update_while_editing(false).text("Contrast")))
        .on_hover_text("Spreads the colors away from mid-gray (above 1) or pulls them toward it (below 1)");
    track(ui.add(egui::Slider::new(&mut a.gamma, 0.1..=5.0).update_while_editing(false).logarithmic(true).text("Gamma")))
        .on_hover_text("Bends the midtones: above 1 brightens shadows and midtones, below 1 darkens them; black and white stay");
    track(ui.add(egui::Slider::new(&mut a.hue, -180.0..=180.0).update_while_editing(false).suffix("°").text("Hue")))
        .on_hover_text("Turns every color around the color wheel by this angle");
    track(ui.add(egui::Slider::new(&mut a.saturation, 0.0..=3.0).update_while_editing(false).text("Saturation")))
        .on_hover_text("Color intensity: 0 makes gray, 1 leaves it unchanged, above 1 makes colors more vivid");
    let mut alpha = a.alpha * 100.0;
    let r = track(
        ui.add(egui::Slider::new(&mut alpha, -100.0..=100.0).update_while_editing(false).max_decimals(0).suffix("%").text("Alpha")),
    )
    .on_hover_text("Added to the opacity of every pixel: +100% makes everything opaque (removes transparency), -100% fully transparent");
    if r.changed() {
        a.alpha = alpha / 100.0;
    }
    (changed, held)
}

fn channel_box(ui: &mut Ui, id: &str, v: &mut usize) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(id).width(44.0).selected_text(CHANNELS[*v]).show_ui(ui, |ui| {
        for (i, name) in CHANNELS.iter().enumerate() {
            changed |= ui.selectable_value(v, i, *name).changed();
        }
    });
    changed
}

impl FiltersPanel {
    pub fn is_previewing(&self) -> bool {
        self.previewing
    }

    /// Drops the preview and resets the sliders.
    pub fn cancel(&mut self, doc: &mut Document) {
        if self.previewing {
            doc.clear_preview();
        }
        self.previewing = false;
        self.adj = Adjustments::default();
    }

    fn preview(&mut self, doc: &mut Document, scope: Scope) {
        if self.adj.is_identity() {
            if self.previewing {
                doc.clear_preview();
                self.previewing = false;
            }
            return;
        }
        preview_filter(doc, &Filter::Adjust(self.adj), scope);
        self.previewing = true;
    }

    /// Ends a live filter-layer edit: restores the settings from before and
    /// records the change as one undo step.
    fn finish_edit(&mut self, doc: &mut Document, actions: &mut Vec<Action>) {
        if let Some((id, before)) = self.edit.take() {
            if let Some(now) = doc.layer(id).map(|l| l.filter) {
                doc.set_layer_filter_live(id, before);
                if now != before {
                    actions.push(Action::Cmd(Command::SetLayerFilter { id, filter: now }));
                }
            }
        }
    }

    /// `scope`: the "Apply to" choice (layer, image or selection).
    pub fn ui(&mut self, ui: &mut Ui, doc: &mut Document, scope: Scope, actions: &mut Vec<Action>) {
        if doc.active_layer().is_filter() {
            if self.previewing {
                self.cancel(doc);
            }
            self.filter_layer_ui(ui, doc, actions);
            return;
        }
        self.finish_edit(doc, actions);
        if scope != self.scope {
            self.scope = scope;
            if self.previewing {
                self.preview(doc, scope);
            }
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Invert").on_hover_text("Invert colors (Ctrl+I)").clicked() {
                actions.push(Action::Filter(Filter::Invert));
            }
            if ui
                .button("Auto levels")
                .on_hover_text("Stretch brightness so the darkest pixel becomes 0% and the brightest 100% (Ctrl+Shift+L)")
                .clicked()
            {
                actions.push(Action::Filter(Filter::AutoLevels));
            }
            if ui.button("Convolution…").on_hover_text("Custom kernel filter (Ctrl+K)").clicked() {
                actions.push(Action::ShowDialog(Dialog::Convolution));
            }
        });
        let (changed, _) = sliders(ui, &mut self.adj);
        if changed {
            self.preview(doc, scope);
        }
        ui.horizontal(|ui| {
            let active = !self.adj.is_identity();
            if ui.add_enabled(active, egui::Button::new("Apply")).clicked() {
                doc.clear_preview();
                actions.push(Action::Filter(Filter::Adjust(self.adj)));
                self.previewing = false;
                self.adj = Adjustments::default();
            }
            if ui.add_enabled(active, egui::Button::new("Reset")).clicked() {
                self.cancel(doc);
            }
        });

        // Channel swap: [R▾] Switch [G▾]
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            channel_box(ui, "swap-a", &mut self.swap.0);
            let (a, b) = self.swap;
            let tip = format!("Exchange the {} and {} channels", CHANNELS[a], CHANNELS[b]);
            if ui.add_enabled(a != b, egui::Button::new("Switch")).on_hover_text(tip).clicked() {
                actions.push(Action::Filter(Filter::SwapChannels(a, b)));
            }
            channel_box(ui, "swap-b", &mut self.swap.1);
        });
    }

    /// The active layer is a filter layer: edit its settings live.
    fn filter_layer_ui(&mut self, ui: &mut Ui, doc: &mut Document, actions: &mut Vec<Action>) {
        let id = doc.active;
        if self.edit.is_some_and(|(eid, _)| eid != id) {
            self.finish_edit(doc, actions);
        }
        let l = doc.active_layer();
        let before = l.filter;
        let mut f = l.filter;
        let everywhere = imged_core::composite::is_blank(&l.pixels);
        ui.label(
            RichText::new(format!(
                "Filter layer \"{}\": these settings change the layers below it{}.",
                l.name,
                if everywhere { "" } else { ", where its pixels are opaque" }
            ))
            .small()
            .color(crate::ui::layers::FILTER_ON),
        );
        // Toggles commit at once; sliders stay live while dragged.
        let mut commit_now = false;
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(f.invert, "Invert").on_hover_text("Invert the colors below").clicked() {
                f.invert = !f.invert;
                commit_now = true;
            }
            let tip = "Stretch the brightness of what is below so its darkest pixel becomes 0% and the brightest 100%";
            if ui.selectable_label(f.levels.is_some(), "Auto levels").on_hover_text(tip).clicked() {
                f.levels = if f.levels.is_some() { None } else { doc.luma_range_below(id) };
                commit_now = true;
            }
            ui.add_enabled(false, egui::Button::new("Convolution…"))
                .on_disabled_hover_text("Convolution can't be used on a filter layer (it needs neighboring pixels)");
        });
        let (changed, held) = sliders(ui, &mut f.adjust);
        ui.horizontal(|ui| {
            let (mut a, mut b) = f.swap.unwrap_or(self.swap);
            let mut swap_changed = channel_box(ui, "fswap-a", &mut a);
            let on = f.swap.is_some();
            let tip = format!("Exchange the {} and {} channels of what is below", CHANNELS[a], CHANNELS[b]);
            if ui.add_enabled(a != b, egui::Button::selectable(on, "Switch")).on_hover_text(tip).clicked() {
                f.swap = if on { None } else { Some((a, b)) };
                commit_now = true;
            }
            swap_changed |= channel_box(ui, "fswap-b", &mut b);
            self.swap = (a, b);
            if swap_changed && f.swap.is_some() {
                f.swap = (a != b).then_some((a, b));
                commit_now = true;
            }
        });
        if ui.add_enabled(!f.is_identity(), egui::Button::new("Reset")).on_hover_text("Remove every change").clicked() {
            f = LayerFilter::default();
            commit_now = true;
        }
        if f != before {
            if self.edit.is_none() {
                self.edit = Some((id, before));
            }
            doc.set_layer_filter_live(id, f);
        }
        if commit_now || (changed && !held) || (!held && self.edit.is_some()) {
            self.finish_edit(doc, actions);
        }
    }
}
