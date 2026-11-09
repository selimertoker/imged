//! Commands: the single entry point for changing a document.
//!
//! GUI buttons, tool releases and keyboard shortcuts all build a [`Command`]
//! and call [`Command::execute`]. A future CLI (`imged -c "invert; resize 50%"`)
//! or vim-style command line only needs to parse text into these values, and
//! a GPU backend can intercept the variants it accelerates.

use crate::color::Color;
use crate::composite::composite_document;
use crate::document::Document;
use crate::geom::{Rect, Vec2};
use crate::layer::LayerId;
use crate::mask::Mask;
use crate::ops::fill::{region_mask, RegionParams};
use crate::ops::filters::{Filter, LayerFilter};
use crate::ops::outline::{detect_outline, OutlineParams};
use crate::ops::paint::{paint_mask, Brush, PaintMode, Pen, Shape};
use crate::ops::text::{FontLibrary, TextSpec};
use crate::ops::transform::{self, Edge, ResizeFilter, Scope};
use crate::pixel::Depth;
use crate::pixels::Pixels;
use crate::buffer::Buffer;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    // --- color / pixel filters
    /// A color filter on the active layer/group, every layer, or the selection.
    Filter(Filter, Scope),

    // --- geometry. Each acts on its scope's pivot rectangle: the active
    // layer/group's bounds, the canvas, or the selection's bounds.
    Flip { horizontal: bool, scope: Scope },
    RotateQuarter { turns_cw: u32, scope: Scope },
    /// `center` is the point to turn around (canvas coords; `None` = the
    /// middle of the pivot rectangle).
    Rotate { degrees: f32, smooth: bool, scope: Scope, center: Option<Vec2> },
    Resize { width: u32, height: u32, filter: ResizeFilter, scope: Scope },
    /// `anchor` = where the old canvas's top-left lands in the new canvas.
    CanvasSize { width: u32, height: u32, anchor: (i32, i32) },
    CropCanvas(Rect),
    /// Perspective tilt, hinged on the edge opposite `edge`.
    Tilt { edge: Edge, degrees: f32, focal: f32, scope: Scope },
    /// Slide `edge` along itself by `amount` pixels (parallelogram).
    Shear { edge: Edge, amount: f32, scope: Scope },
    /// Free transform: the pivot rectangle is warped onto `quad` (canvas
    /// coords; corners top-left, top-right, bottom-right, bottom-left).
    Warp { quad: [Vec2; 4], scope: Scope },
    /// Scale so the pivot rectangle becomes `rect`.
    ResizeTo { rect: Rect, filter: ResizeFilter, scope: Scope },

    // --- painting on the active layer (canvas coordinates)
    Stroke { points: Vec<Vec2>, brush: Brush, color: Color, mode: PaintMode },
    /// Square, hard pen: sets the covered pixels to `color` (alpha included).
    Pen { points: Vec<Vec2>, pen: Pen, color: Color },
    Shape { shape: Shape, color: Color, fill_color: Color, mode: PaintMode },
    Fill { at: Vec2, color: Color, params: RegionParams, sample_merged: bool, mode: PaintMode },
    Text { spec: TextSpec, color: Color, mode: PaintMode },

    // --- selections
    SelectRect { rect: Rect, mode: SelectMode },
    SelectMask { mask: Mask, mode: SelectMode },
    MagicWand { at: Vec2, params: RegionParams, sample_merged: bool, mode: SelectMode },
    SelectOutline { at: Vec2, params: OutlineParams, sample_merged: bool, mode: SelectMode },
    SelectAll,
    SelectNone,
    InvertSelection,
    /// Moves (or copies) the selected pixels of the active layer/group to a new layer.
    SelectionToLayer { copy: bool },
    /// Crops the canvas to the selection's bounds.
    CropToSelection,

    // --- layers
    NewLayer,
    DeleteLayer,
    DuplicateLayer,
    MoveLayer { up: bool },
    MoveLayerBy { dx: i32, dy: i32 },
    /// Like `MoveLayerBy`, but consecutive nudges are one undo step.
    NudgeLayerBy { dx: i32, dy: i32 },
    SetActive(LayerId),
    ToggleVisible(LayerId),
    MergeDown,
    MergeSelected,
    /// Groups the active and ticked layers (they then act as one layer).
    Group,
    Ungroup,
    /// Turns a layer into a filter layer (its pixels then mask where its
    /// filter changes the layers below), or back.
    SetFilterOn { id: LayerId, on: bool },
    /// Changes a filter layer's settings.
    SetLayerFilter { id: LayerId, filter: LayerFilter },
    /// Replaces layers with an external program's results: one result per
    /// layer replaces each layer's pixels in place; any other number replaces
    /// the layers with new layers (the first on top) where the topmost was.
    ReplaceLayers { ids: Vec<LayerId>, results: Vec<NewLayer>, label: String },
    /// Replaces the whole image with these layers (the first on top); the
    /// canvas takes the largest size.
    ReplaceImage { results: Vec<NewLayer>, label: String },
    Flatten,
    /// Split the active layer into R, G, B and A grayscale layers.
    SplitChannels,
    /// Combine 3 (RGB) or 4 (RGBA) layers into one.
    MergeChannels(Vec<LayerId>),
    /// New layer order, bottom to top.
    Reorder(Vec<LayerId>),
    /// Arranges all layers above the background so they don't overlap, in
    /// rows that make a roughly square image; the canvas is fitted to them.
    TileLayers { gap: u32 },
    /// Resizes the canvas to the union of all layers.
    FitCanvasToLayers,
    ConvertDepth(Depth),
    LayerToCanvasSize,
    CropLayerToContent,

    // --- history
    Undo,
    Redo,
}

/// What a selection tool does with the area it found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectMode {
    /// Move the selected pixels to a new layer (instead of keeping a selection).
    pub to_layer: bool,
    /// Add to the existing selection; with `to_layer`, the pixels join the
    /// layer made by the previous selection.
    pub add: bool,
}

impl Default for SelectMode {
    fn default() -> Self {
        SelectMode { to_layer: true, add: false }
    }
}

/// Services some commands need beyond the document.
pub struct ExecContext<'a> {
    pub fonts: Option<&'a mut FontLibrary>,
    /// Painting outside the active layer grows it (up to the canvas). When
    /// false, paint is clipped to the layer.
    pub grow_layers: bool,
}

impl Default for ExecContext<'_> {
    fn default() -> Self {
        ExecContext { fonts: None, grow_layers: true }
    }
}

/// What a successful command did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Changed,
    Unchanged,
    /// A new layer was created (selection commands).
    NewLayer(LayerId),
}

impl Command {
    pub fn label(&self) -> String {
        match self {
            Command::Filter(f, _) => f.label().into(),
            Command::Flip { horizontal: true, .. } => "Flip horizontal".into(),
            Command::Flip { .. } => "Flip vertical".into(),
            Command::RotateQuarter { .. } | Command::Rotate { .. } => "Rotate".into(),
            Command::Resize { .. } => "Resize".into(),
            Command::CanvasSize { .. } => "Canvas size".into(),
            Command::CropCanvas(_) => "Crop".into(),
            Command::Tilt { .. } => "Tilt".into(),
            Command::Shear { .. } => "Shear".into(),
            Command::Warp { .. } => "Transform".into(),
            Command::ResizeTo { .. } => "Resize layer".into(),
            Command::SplitChannels => "Split channels".into(),
            Command::MergeChannels(_) => "Merge channels".into(),
            Command::Reorder(_) => "Reorder layers".into(),
            Command::Stroke { .. } => "Brush".into(),
            Command::Pen { .. } => "Pen".into(),
            Command::TileLayers { .. } => "Tile layers".into(),
            Command::FitCanvasToLayers => "Fit canvas to layers".into(),
            Command::Shape { shape, .. } => shape.kind.label().into(),
            Command::Fill { .. } => "Fill".into(),
            Command::Text { .. } => "Text".into(),
            Command::SelectRect { .. } | Command::SelectMask { .. } => "Selection".into(),
            Command::SelectAll => "Select all".into(),
            Command::SelectNone => "Deselect".into(),
            Command::InvertSelection => "Invert selection".into(),
            Command::SelectionToLayer { copy: true } => "Copy selection to layer".into(),
            Command::SelectionToLayer { copy: false } => "Cut selection to layer".into(),
            Command::CropToSelection => "Crop to selection".into(),
            Command::Group => "Group layers".into(),
            Command::Ungroup => "Ungroup layers".into(),
            Command::SetFilterOn { on: true, .. } => "Make filter layer".into(),
            Command::SetFilterOn { on: false, .. } => "Make pixel layer".into(),
            Command::SetLayerFilter { .. } => "Change filter layer".into(),
            Command::ReplaceLayers { label, .. } | Command::ReplaceImage { label, .. } => label.clone(),
            Command::MagicWand { .. } => "Magic wand".into(),
            Command::SelectOutline { .. } => "Outline select".into(),
            other => format!("{other:?}"),
        }
    }

    pub fn execute(&self, doc: &mut Document, ctx: &mut ExecContext) -> Result<Outcome, String> {
        let changed = |b: bool| Ok(if b { Outcome::Changed } else { Outcome::Unchanged });
        match self {
            Command::Filter(f, scope) if transform::effective_scope(doc, *scope) != Scope::Image && doc.active_layer().is_filter() => {
                // On a filter layer, filters change its settings instead of pixels.
                let id = doc.active;
                let mut lf = doc.active_layer().filter;
                match f {
                    Filter::Invert => lf.invert = !lf.invert,
                    Filter::Adjust(a) => lf.adjust = *a,
                    Filter::AutoLevels => lf.levels = doc.luma_range_below(id),
                    Filter::Levels { lo, hi } => lf.levels = Some((*lo, *hi)),
                    Filter::SwapChannels(a, b) => lf.swap = Some((*a, *b)),
                    Filter::Gamma(g) => lf.adjust.gamma = *g,
                    Filter::Brightness(b) => lf.adjust.brightness = *b,
                    Filter::Contrast(c) => lf.adjust.contrast = *c,
                    Filter::Hue(h) => lf.adjust.hue = *h,
                    Filter::Saturation(v) => lf.adjust.saturation = *v,
                    Filter::Convolve(_) | Filter::ColorToAlpha { .. } => {
                        return Err(format!("{} can't be used on a filter layer", f.label()));
                    }
                }
                changed(doc.set_layer_filter(id, lf))
            }
            Command::Filter(f, scope) => {
                let ids = filter_layers(doc, *scope);
                let f = resolve_filter(doc, &ids, f);
                let scope = transform::effective_scope(doc, *scope);
                let sel = doc.selection.clone();
                changed(doc.edit_layers(&ids, f.label(), |l| Some((filtered(l, &f, scope, sel.as_deref()), l.offset))))
            }
            Command::Flip { horizontal, scope } => changed(transform::flip(doc, *horizontal, *scope)),
            Command::RotateQuarter { turns_cw, scope } => changed(transform::rotate_quarter(doc, *turns_cw, *scope)),
            Command::Rotate { degrees, smooth, scope, center } => {
                changed(transform::rotate_free(doc, *degrees, *smooth, *scope, *center))
            }
            Command::Resize { width, height, filter, scope } => {
                changed(transform::resize_doc(doc, *width, *height, *filter, *scope))
            }
            Command::CanvasSize { width, height, anchor } => changed(doc.resize_canvas(*width, *height, *anchor)),
            Command::CropCanvas(r) => changed(doc.crop_canvas(*r)),
            Command::Tilt { edge, degrees, focal, scope } => {
                let b = transform::scope_pivot(doc, *scope);
                let q = transform::tilt_quad(b.w as f32, b.h as f32, *edge, *degrees, *focal)
                    .map(|p| p + Vec2::new(b.x as f32, b.y as f32));
                changed(transform::warp_targets(doc, "Tilt", q, *scope))
            }
            Command::Shear { edge, amount, scope } => {
                let b = transform::scope_pivot(doc, *scope);
                let q = transform::shear_quad(b.w as f32, b.h as f32, *edge, *amount)
                    .map(|p| p + Vec2::new(b.x as f32, b.y as f32));
                changed(transform::warp_targets(doc, "Shear", q, *scope))
            }
            Command::Stroke { points, brush, color, mode } => {
                if points.is_empty() {
                    return Ok(Outcome::Unchanged);
                }
                let bounds = points.iter().fold(Rect::default(), |a, p| a.union(&brush.dab_bounds(*p)));
                let mut mask = Mask::new(bounds.intersect(&paint_area(doc, ctx.grow_layers)));
                brush.dab(&mut mask, points[0]);
                for w in points.windows(2) {
                    brush.segment(&mut mask, w[0], w[1]);
                }
                changed(paint_on_active(doc, "Brush", vec![(mask, color.px())], *mode, ctx.grow_layers))
            }
            Command::Pen { points, pen, color } => {
                if points.is_empty() {
                    return Ok(Outcome::Unchanged);
                }
                let bounds = points.iter().fold(Rect::default(), |a, p| a.union(&pen.cell(*p)));
                let mut mask = Mask::new(bounds.intersect(&paint_area(doc, ctx.grow_layers)));
                pen.dab(&mut mask, points[0]);
                for w in points.windows(2) {
                    pen.segment(&mut mask, w[0], w[1]);
                }
                changed(paint_on_active(doc, "Pen", vec![(mask, color.px())], PaintMode::Replace, ctx.grow_layers))
            }
            Command::Shape { shape, color, fill_color, mode } => {
                let (outline, fill) = shape.masks(paint_area(doc, ctx.grow_layers));
                let fill_px = if shape.fill == crate::ops::paint::ShapeFill::Fill { color.px() } else { fill_color.px() };
                let mut parts = Vec::new();
                if let Some(m) = fill {
                    parts.push((m, fill_px));
                }
                if let Some(m) = outline {
                    parts.push((m, color.px()));
                }
                changed(paint_on_active(doc, shape.kind.label(), parts, *mode, ctx.grow_layers))
            }
            Command::Fill { at, color, params, sample_merged, mode } => {
                let mask = if *sample_merged {
                    let img = composite_document(doc, doc.canvas_rect());
                    region_mask(doc.canvas_rect(), *at, *params, |x, y| img.get(x as u32, y as u32))
                } else {
                    // Outside the layer counts as transparent, so filling
                    // there grows the layer.
                    let layer = doc.active_layer();
                    let (px, off) = (layer.pixels.clone(), layer.offset);
                    region_mask(paint_area(doc, ctx.grow_layers), *at, *params, |x, y| px.get_or_clear(x - off.0, y - off.1))
                };
                let Some(mask) = mask else { return Ok(Outcome::Unchanged) };
                changed(paint_on_active(doc, "Fill", vec![(mask, color.px())], *mode, ctx.grow_layers))
            }
            Command::Text { spec, color, mode } => {
                let fonts = ctx.fonts.as_deref_mut().ok_or("No font library available")?;
                let (mask, _) = fonts.render(spec).ok_or_else(|| format!("Font '{}' not found", spec.family))?;
                changed(paint_on_active(doc, "Text", vec![(mask, color.px())], *mode, ctx.grow_layers))
            }
            Command::SelectRect { rect, mode } => select(doc, &Mask::full(*rect), *mode),
            Command::SelectMask { mask, mode } => select(doc, mask, *mode),
            Command::MagicWand { at, params, sample_merged, mode } => {
                if !sample_merged {
                    activate_layer_at(doc, *at, *mode);
                }
                let mask = selection_source(doc, *sample_merged, |domain, get| region_mask(domain, *at, *params, get))
                    .ok_or("Click inside the layer")?;
                select(doc, &mask, *mode)
            }
            Command::SelectOutline { at, params, sample_merged, mode } => {
                if !sample_merged {
                    activate_layer_at(doc, *at, *mode);
                }
                let res = selection_source(doc, *sample_merged, |domain, get| {
                    Some(detect_outline(domain, *at, *params, get))
                })
                .ok_or("Click inside the layer")??;
                select(doc, &res.mask, *mode)
            }
            Command::SelectAll => changed(doc.set_selection("Select all", Some(Mask::full(doc.canvas_rect())))),
            Command::SelectNone => changed(doc.set_selection("Deselect", None)),
            Command::InvertSelection => {
                let r = doc.canvas_rect();
                let mut m = Mask::full(r);
                if let Some(s) = &doc.selection {
                    for y in r.y..r.bottom() {
                        for x in r.x..r.right() {
                            m.set(x, y, 1.0 - s.get(x, y));
                        }
                    }
                }
                changed(doc.set_selection("Invert selection", Some(m)))
            }
            Command::SelectionToLayer { copy } => {
                let mask = doc.selection.clone().ok_or("Nothing is selected")?;
                let out = doc.selection_to_layer(&mask, *copy, None).ok_or("The selected area of the layer is empty")?;
                doc.set_selection("Deselect", None);
                Ok(Outcome::NewLayer(out))
            }
            Command::CropToSelection => {
                let b = doc.selection.as_ref().and_then(|m| m.bounds()).ok_or("Nothing is selected")?;
                changed(doc.structural("Crop to selection", |d| {
                    let ok = d.resize_canvas_raw(b);
                    d.selection = d.selection.take().map(|m| std::sync::Arc::new((*m).clone().translated(-b.x, -b.y)));
                    ok
                }))
            }
            Command::NewLayer => Ok(Outcome::NewLayer(doc.add_layer())),
            Command::DeleteLayer => changed(doc.delete_active()),
            Command::DuplicateLayer => changed(doc.duplicate_active()),
            Command::MoveLayer { up } => changed(doc.move_active(*up)),
            Command::NudgeLayerBy { dx, dy } => changed(doc.nudge_targets(*dx, *dy)),
            Command::MoveLayerBy { dx, dy } => {
                if *dx == 0 && *dy == 0 {
                    return Ok(Outcome::Unchanged);
                }
                let seq = doc.next_seq();
                let mut dirty = Rect::default();
                for id in doc.targets() {
                    let l = doc.layer_mut(id).unwrap();
                    dirty = dirty.union(&l.bounds());
                    l.move_to(seq, (l.offset.0 + dx, l.offset.1 + dy));
                    dirty = dirty.union(&l.bounds());
                }
                doc.mark_dirty(dirty);
                Ok(Outcome::Changed)
            }
            Command::SetActive(id) => {
                doc.set_active(*id);
                Ok(Outcome::Unchanged)
            }
            Command::ToggleVisible(id) => {
                doc.toggle_visible(*id);
                Ok(Outcome::Changed)
            }
            Command::MergeDown => changed(doc.merge_down()),
            Command::MergeSelected => changed(doc.merge_selected()),
            Command::Group => changed(doc.group_selected()),
            Command::Ungroup => match doc.active_layer().group {
                Some(g) => changed(doc.ungroup(g)),
                None => Ok(Outcome::Unchanged),
            },
            Command::SetFilterOn { id, on } => changed(doc.set_filter_on(*id, *on)),
            Command::SetLayerFilter { id, filter } => changed(doc.set_layer_filter(*id, *filter)),
            Command::ReplaceLayers { ids, results, label } => {
                let ids: Vec<LayerId> = ids.iter().copied().filter(|id| doc.layer(*id).is_some()).collect();
                if ids.is_empty() || results.is_empty() {
                    return Ok(Outcome::Unchanged);
                }
                if ids.len() == results.len() {
                    return changed(doc.structural(label, |d| {
                        for (id, r) in ids.iter().zip(results) {
                            let l = d.layer_mut(*id).unwrap();
                            l.pixels = Arc::new(r.pixels.clone());
                            l.offset = r.offset;
                        }
                        true
                    }));
                }
                let top = ids.iter().filter_map(|id| doc.index_of(*id)).max().unwrap();
                let group = doc.layers[top].group;
                let new_ids: Vec<LayerId> = results.iter().map(|_| doc.alloc_id()).collect();
                changed(doc.structural(label, |d| {
                    // Removing the layers below `top` shifts it down.
                    let below = ids.iter().filter_map(|id| d.index_of(*id)).filter(|i| *i < top).count();
                    d.layers.retain(|l| !ids.contains(&l.id));
                    let at = top - below;
                    // Each insert at `at` pushes the previous ones up, so the first ends on top.
                    for (id, r) in new_ids.iter().zip(results) {
                        let mut l = crate::layer::Layer::new(*id, r.name.clone(), r.pixels.clone(), r.offset);
                        l.group = group;
                        d.layers.insert(at, l);
                    }
                    d.active = new_ids[0];
                    d.selected.retain(|id| !ids.contains(id));
                    true
                }))
            }
            Command::ReplaceImage { results, label } => {
                if results.is_empty() {
                    return Ok(Outcome::Unchanged);
                }
                let ids: Vec<LayerId> = results.iter().map(|_| doc.alloc_id()).collect();
                let w = results.iter().map(|r| r.pixels.width()).max().unwrap_or(1);
                let h = results.iter().map(|r| r.pixels.height()).max().unwrap_or(1);
                changed(doc.structural(label, |d| {
                    d.width = w;
                    d.height = h;
                    d.layers = ids
                        .iter()
                        .zip(results)
                        .rev()
                        .map(|(id, r)| crate::layer::Layer::new(*id, r.name.clone(), r.pixels.clone(), (0, 0)))
                        .collect();
                    d.active = ids[0];
                    d.selected.clear();
                    d.selection = None;
                    true
                }))
            }
            Command::Flatten => changed(doc.flatten()),
            Command::SplitChannels => changed(doc.split_channels()),
            Command::MergeChannels(ids) => changed(doc.merge_channels(ids)),
            Command::Reorder(order) => changed(doc.set_order(order)),
            Command::Warp { quad, scope } => changed(transform::warp_targets(doc, "Warp", *quad, *scope)),
            Command::ResizeTo { rect, filter, scope } => {
                changed(transform::plan_resize_to(doc, *rect, *filter, *scope).is_some_and(|p| p.apply(doc)))
            }
            Command::TileLayers { gap } => changed(doc.tile_layers(*gap)),
            Command::FitCanvasToLayers => changed(doc.fit_canvas_to_layers()),
            Command::ConvertDepth(d) => changed(doc.convert_depth(*d)),
            Command::LayerToCanvasSize => changed(doc.layer_to_canvas_size()),
            Command::CropLayerToContent => changed(doc.crop_layer_to_content()),
            Command::Undo => changed(doc.undo()),
            Command::Redo => changed(doc.redo()),
        }
    }
}

/// Applies a selection tool's result: either keeps it as the selection
/// (replacing or adding to it), or moves the pixels under `mask` from the
/// active layer to a new layer (or, when adding, into the layer the previous
/// selection made). If the active layer has nothing there, the topmost
/// visible layer that does is used.
fn select(doc: &mut Document, mask: &Mask, mode: SelectMode) -> Result<Outcome, String> {
    if !mode.to_layer {
        let m = if mode.add { doc.selection_plus(mask) } else { mask.clone() };
        let label = if mode.add && doc.selection.is_some() { "Add to selection" } else { "Select" };
        return changed_outcome(doc.set_selection(label, Some(m)));
    }
    let into = if mode.add { doc.extract_layer.filter(|id| doc.layer(*id).is_some()) } else { None };
    if let Some(id) = doc.selection_to_layer(mask, false, into) {
        return Ok(Outcome::NewLayer(id));
    }
    let under = doc
        .layers
        .iter()
        .rev()
        .filter(|l| l.visible && Some(l.id) != into && !l.is_filter())
        .map(|l| l.id)
        .find(|id| doc.layer_under_mask_only(*id, mask));
    match under {
        Some(id) if !doc.targets().contains(&id) => {
            doc.set_active(id);
            doc.selection_to_layer(mask, false, into).map(Outcome::NewLayer)
        }
        _ => None,
    }
    .ok_or_else(|| "Nothing selected (the area is empty or transparent)".into())
}

fn changed_outcome(b: bool) -> Result<Outcome, String> {
    Ok(if b { Outcome::Changed } else { Outcome::Unchanged })
}

/// Makes the layer under the click active when the active layer/group has
/// nothing there, so selections "pick what you clicked". When adding to the
/// previous selection's layer, that layer is never the source.
fn activate_layer_at(doc: &mut Document, at: Vec2, mode: SelectMode) {
    let id = region_layer(doc, at, mode);
    doc.set_active(id);
}

/// The layer a magic wand / outline click at `at` works on: the active
/// layer/group when it has a pixel there, else the topmost visible layer
/// that does (else the active layer).
pub fn region_layer(doc: &Document, at: Vec2, mode: SelectMode) -> LayerId {
    let (x, y) = (at.x.floor() as i32, at.y.floor() as i32);
    let skip = if mode.to_layer && mode.add { doc.extract_layer } else { None };
    let has_pixel = doc.targets().iter().filter(|id| Some(**id) != skip).filter_map(|id| doc.layer(*id)).any(|l| {
        l.bounds().contains(x, y) && l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1)[3] > 0.0
    });
    if has_pixel {
        return doc.active;
    }
    doc.layers
        .iter()
        .rev()
        .filter(|l| l.visible && Some(l.id) != skip && !l.is_filter())
        .find(|l| l.pixels.get_or_clear(x - l.offset.0, y - l.offset.1)[3] > 0.0)
        .map_or(doc.active, |l| l.id)
}

/// A layer made from an external program's output.
#[derive(Clone, Debug, PartialEq)]
pub struct NewLayer {
    pub pixels: Pixels,
    pub offset: (i32, i32),
    pub name: String,
}

/// What the bucket, magic wand or outline select would cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegionKind {
    Fill(RegionParams),
    Wand(RegionParams),
    Outline(OutlineParams),
}

enum SourceImage {
    Composite(Buffer<f32>),
    Layer(Arc<Pixels>, (i32, i32)),
}

/// The pixels a region tool looks at, captured so its area can be
/// previewed (on another thread) without changing the document.
pub struct RegionSource {
    pub domain: Rect,
    img: SourceImage,
}

impl RegionSource {
    /// Captures what a click would look at. `fill`: the bucket (else the
    /// selection tools); `layer`: the layer a selection click works on
    /// (from [`region_layer`]).
    pub fn capture(doc: &mut Document, fill: bool, sample_merged: bool, grow: bool, layer: LayerId) -> Option<RegionSource> {
        if sample_merged {
            let domain = doc.canvas_rect();
            return Some(RegionSource { domain, img: SourceImage::Composite(composite_document(doc, domain)) });
        }
        if fill {
            let l = doc.active_layer();
            let (px, off) = (l.pixels.clone(), l.offset);
            return Some(RegionSource { domain: paint_area(doc, grow), img: SourceImage::Layer(px, off) });
        }
        let before = doc.active;
        doc.active = layer;
        let domain = doc.targets_bounds();
        let img = (!domain.is_empty()).then(|| doc.composite_targets(domain));
        doc.active = before;
        // The composite starts at the domain's corner: store it as a layer.
        let img = img?;
        let px = Arc::new(Pixels::F32(img));
        Some(RegionSource { domain, img: SourceImage::Layer(px, (domain.x, domain.y)) })
    }

    pub fn get(&self, x: i32, y: i32) -> crate::pixel::Px {
        match &self.img {
            SourceImage::Composite(b) => b.get_or_clear(x - self.domain.x, y - self.domain.y),
            SourceImage::Layer(px, off) => px.get_or_clear(x - off.0, y - off.1),
        }
    }

    /// The area a click at `at` covers (before any "add to selection").
    pub fn region(&self, kind: RegionKind, at: Vec2) -> Option<Mask> {
        let get = |x, y| self.get(x, y);
        match kind {
            RegionKind::Fill(p) | RegionKind::Wand(p) => region_mask(self.domain, at, p, get),
            RegionKind::Outline(p) => detect_outline(self.domain, at, p, get).ok().map(|r| r.mask),
        }
    }
}

/// Shows filter `f` as a live preview on the layers it would change.
pub fn preview_filter(doc: &mut Document, f: &Filter, scope: Scope) {
    let ids = filter_layers(doc, scope);
    let f = resolve_filter(doc, &ids, f);
    let eff = transform::effective_scope(doc, scope);
    let sel = doc.selection.clone();
    doc.clear_preview();
    for id in ids {
        let Some(l) = doc.layer(id) else { continue };
        let px = std::sync::Arc::new(filtered(l, &f, eff, sel.as_deref()));
        let off = l.offset;
        doc.set_preview(id, px, off);
    }
}

/// `l` with filter `f` applied; with selection scope only inside the
/// selection (blended by its coverage).
pub fn filtered(l: &crate::layer::Layer, f: &Filter, scope: Scope, selection: Option<&Mask>) -> Pixels {
    let out = f.apply(&l.pixels);
    match (scope, selection) {
        (Scope::Selection, Some(m)) => {
            let orig = l.pixels.to_f32();
            let new = out.to_f32();
            let mut b = orig.clone();
            for y in 0..b.height {
                for x in 0..b.width {
                    let k = m.get(x as i32 + l.offset.0, y as i32 + l.offset.1);
                    if k <= 0.0 {
                        continue;
                    }
                    let (o, n) = (orig.get(x, y), new.get(x, y));
                    b.set(x, y, std::array::from_fn(|c| o[c] + (n[c] - o[c]) * k));
                }
            }
            Pixels::from_f32(&b, l.depth())
        }
        _ => out,
    }
}

/// Runs a region detector over either the merged image or the active
/// layer/group, in canvas coordinates.
fn selection_source<R>(
    doc: &Document,
    sample_merged: bool,
    f: impl FnOnce(Rect, &dyn Fn(i32, i32) -> crate::pixel::Px) -> Option<R>,
) -> Option<R> {
    let domain = if sample_merged { doc.canvas_rect() } else { doc.targets_bounds() };
    if domain.is_empty() {
        return None;
    }
    let img = if sample_merged { composite_document(doc, domain) } else { doc.composite_targets(domain) };
    let get = |x: i32, y: i32| img.get_or_clear(x - domain.x, y - domain.y);
    f(domain, &get)
}

/// Pixel layers a filter with this scope changes: the active layer/group
/// (also for the selection scope), or all. Filter layers are skipped (their
/// pixels are only a mask).
pub fn filter_layers(doc: &Document, scope: Scope) -> Vec<LayerId> {
    transform::scope_layers(doc, scope).into_iter().filter(|id| !doc.layer(*id).unwrap().is_filter()).collect()
}

/// Turns filters that look at the whole set of layers (auto levels) into a
/// fixed per-pixel filter, so every layer gets the same mapping.
pub fn resolve_filter(doc: &Document, ids: &[LayerId], f: &Filter) -> Filter {
    match f {
        Filter::AutoLevels => {
            let px = ids.iter().filter_map(|id| doc.layer(*id)).map(|l| &*l.pixels);
            match crate::ops::filters::luma_range_pixels(px) {
                Some((lo, hi)) => Filter::Levels { lo, hi },
                None => f.clone(),
            }
        }
        _ => f.clone(),
    }
}

/// Area painting tools may cover: the active layer plus the canvas.
pub fn paint_area(doc: &Document, grow: bool) -> Rect {
    let lb = doc.active_layer().bounds();
    if grow {
        lb.union(&doc.canvas_rect())
    } else {
        lb
    }
}

/// Composites masks (canvas coordinates) onto the active layer as one undo
/// step. If paint lands outside the layer, the layer grows to include it.
pub fn paint_on_active(
    doc: &mut Document,
    label: &str,
    parts: Vec<(Mask, crate::pixel::Px)>,
    mode: PaintMode,
    grow: bool,
) -> bool {
    let limit = paint_area(doc, grow);
    let touched =
        parts.iter().filter_map(|(m, _)| m.bounds()).fold(Rect::default(), |a, b| a.union(&b)).intersect(&limit);
    if touched.is_empty() {
        return false;
    }
    let seq = doc.next_seq();
    let layer = doc.active_layer_mut();
    let lb = layer.bounds();
    if lb.union(&touched) == lb {
        let local = touched.translate(-lb.x, -lb.y);
        layer.edit_region(seq, label, local, |px| {
            for (m, color) in parts {
                paint_mask(px, None, &m.translated(-lb.x, -lb.y), local, color, mode);
            }
        });
    } else {
        let area = lb.union(&touched);
        let mut px = Pixels::new(layer.depth(), area.w, area.h);
        px.paste(&layer.pixels, lb.x - area.x, lb.y - area.y);
        let local = touched.translate(-area.x, -area.y);
        for (m, color) in parts {
            paint_mask(&mut px, None, &m.translated(-area.x, -area.y), local, color, mode);
        }
        layer.replace(seq, label, px, (area.x, area.y));
    }
    doc.mark_dirty(touched);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::paint::{ShapeFill, ShapeKind};

    #[test]
    fn shape_and_undo() {
        let mut d = Document::new(20, 20, Depth::U8, Some(Color::WHITE));
        let cmd = Command::Shape {
            shape: Shape {
                width: 2.0,
                fill: ShapeFill::OutlineAndFill,
                antialias: false,
                ..Shape::new(ShapeKind::Rectangle, Vec2::new(2.0, 2.0), Vec2::new(10.0, 10.0))
            },
            color: Color::BLACK,
            fill_color: Color::rgba(1.0, 0.0, 0.0, 1.0),
            mode: PaintMode::Replace,
        };
        assert_eq!(cmd.execute(&mut d, &mut ExecContext::default()), Ok(Outcome::Changed));
        assert_eq!(d.layers[0].pixels.get(2, 2), Color::BLACK.px());
        assert_eq!(d.layers[0].pixels.get(6, 6), [1.0, 0.0, 0.0, 1.0]);
        Command::Undo.execute(&mut d, &mut ExecContext::default()).unwrap();
        assert_eq!(d.layers[0].pixels.get(6, 6), Color::WHITE.px());
    }

    #[test]
    fn painting_outside_grows_layer() {
        let mut d = Document::new(20, 20, Depth::U8, None);
        let id = d.add_layer_from("small", Pixels::filled(Depth::U8, 4, 4, [1.0; 4]), (2, 2));
        d.set_active(id);
        let cmd = Command::Stroke {
            points: vec![Vec2::new(15.5, 15.5)],
            brush: Brush { size: 1.0, hardness: 1.0, antialias: false },
            color: Color::BLACK,
            mode: PaintMode::Replace,
        };
        cmd.execute(&mut d, &mut ExecContext::default()).unwrap();
        let l = d.layer(id).unwrap();
        assert_eq!(l.bounds(), Rect::new(2, 2, 14, 14));
        assert_eq!(l.pixels.get(13, 13), Color::BLACK.px());
        assert_eq!(l.pixels.get(0, 0), [1.0; 4]);
        d.undo();
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(2, 2, 4, 4));
        // With growing off, paint outside the layer is dropped.
        let mut ctx = ExecContext { grow_layers: false, ..Default::default() };
        assert_eq!(cmd.execute(&mut d, &mut ctx), Ok(Outcome::Unchanged));
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(2, 2, 4, 4));
    }

    #[test]
    fn bucket_fill_transparent() {
        let mut d = Document::new(5, 5, Depth::F32, Some(Color::WHITE));
        let cmd = Command::Fill {
            at: Vec2::new(1.0, 1.0),
            color: Color::TRANSPARENT,
            params: RegionParams::default(),
            sample_merged: false,
            mode: PaintMode::Replace,
        };
        cmd.execute(&mut d, &mut ExecContext::default()).unwrap();
        assert_eq!(d.layers[0].pixels.get(4, 4)[3], 0.0);
    }

    #[test]
    fn wand_cuts_to_layer() {
        let mut d = Document::new(6, 6, Depth::U8, Some(Color::WHITE));
        Command::Shape {
            shape: Shape {
                fill: ShapeFill::Fill,
                antialias: false,
                ..Shape::new(ShapeKind::Rectangle, Vec2::new(1.0, 1.0), Vec2::new(4.0, 4.0))
            },
            color: Color::BLACK,
            fill_color: Color::BLACK,
            mode: PaintMode::Replace,
        }
        .execute(&mut d, &mut ExecContext::default())
        .unwrap();
        let out = Command::MagicWand {
            at: Vec2::new(2.0, 2.0),
            params: RegionParams::default(),
            sample_merged: false,
            mode: SelectMode::default(),
        }
        .execute(&mut d, &mut ExecContext::default())
        .unwrap();
        let Outcome::NewLayer(id) = out else { panic!() };
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(1, 1, 3, 3));
        assert_eq!(d.layers[0].pixels.get(2, 2)[3], 0.0);
    }

    #[test]
    fn selections_keep_add_and_join_layers() {
        let mut d = Document::new(10, 10, Depth::U8, Some(Color::WHITE));
        let keep = SelectMode { to_layer: false, add: false };
        let add = SelectMode { to_layer: false, add: true };
        let ctx = &mut ExecContext::default();
        Command::SelectRect { rect: Rect::new(0, 0, 2, 2), mode: keep }.execute(&mut d, ctx).unwrap();
        Command::SelectRect { rect: Rect::new(5, 5, 2, 2), mode: add }.execute(&mut d, ctx).unwrap();
        let sel = d.selection.clone().unwrap();
        assert_eq!(sel.count(), 8);
        assert_eq!(d.layers.len(), 1);
        // A filter with selection scope only changes the selected pixels.
        Command::Filter(Filter::Invert, Scope::Selection).execute(&mut d, ctx).unwrap();
        assert_eq!(d.layers[0].pixels.get(0, 0)[0], 0.0);
        assert_eq!(d.layers[0].pixels.get(3, 3)[0], 1.0);
        // Undo restores the pixels; undo again removes the added area.
        d.undo();
        d.undo();
        assert_eq!(d.selection.as_ref().unwrap().count(), 4);
        // Moving to a layer, then adding: both areas end up in one layer.
        let to = SelectMode { to_layer: true, add: false };
        let to_add = SelectMode { to_layer: true, add: true };
        Command::SelectRect { rect: Rect::new(0, 0, 2, 2), mode: to }.execute(&mut d, ctx).unwrap();
        Command::SelectRect { rect: Rect::new(6, 6, 2, 2), mode: to_add }.execute(&mut d, ctx).unwrap();
        assert_eq!(d.layers.len(), 2);
        assert_eq!(d.layers[1].bounds(), Rect::new(0, 0, 8, 8));
        assert_eq!(d.layers[0].pixels.get(7, 7)[3], 0.0);
    }

    #[test]
    fn rotate_selection_only() {
        let mut d = Document::new(10, 10, Depth::U8, None);
        d.layers[0].pixels = std::sync::Arc::new(Pixels::filled(Depth::U8, 10, 10, [1.0, 0.0, 0.0, 1.0]));
        let ctx = &mut ExecContext::default();
        // A 4×2 selection at the top-left, rotated 90° clockwise around its top-left corner.
        Command::SelectRect { rect: Rect::new(0, 0, 4, 2), mode: SelectMode { to_layer: false, add: false } }
            .execute(&mut d, ctx)
            .unwrap();
        Command::RotateQuarter { turns_cw: 1, scope: Scope::Selection }.execute(&mut d, ctx).unwrap();
        let sel = d.selection.clone().unwrap();
        assert_eq!(sel.bounds().unwrap(), Rect::new(1, -1, 2, 4).intersect(&Rect::new(-10, -10, 30, 30)));
        // Pixels outside the selection stay where they were.
        assert_eq!(d.layers[0].pixels.get_or_clear(9 - d.layers[0].offset.0, 9 - d.layers[0].offset.1)[3], 1.0);
        // The rest of the image is still opaque, and the canvas didn't change.
        assert_eq!((d.width, d.height), (10, 10));
    }

    #[test]
    fn filter_layers_mask_and_settings() {
        let mut d = Document::new(4, 1, Depth::U8, Some(Color::WHITE));
        let ctx = &mut ExecContext::default();
        Command::NewLayer.execute(&mut d, ctx).unwrap();
        let id = d.active;
        Command::SetFilterOn { id, on: true }.execute(&mut d, ctx).unwrap();
        // Invert on a filter layer changes its settings; an empty layer applies everywhere.
        Command::Filter(Filter::Invert, Scope::Layer).execute(&mut d, ctx).unwrap();
        assert!(d.layer(id).unwrap().filter.invert);
        let img = crate::composite::flatten(&d);
        assert_eq!(img.get(0, 0), [0.0, 0.0, 0.0, 1.0]);
        // Painting on it makes the painted area the mask (its colors aren't drawn).
        let cmd = Command::Stroke {
            points: vec![Vec2::new(0.5, 0.5)],
            brush: Brush { size: 1.0, hardness: 1.0, antialias: false },
            color: Color::rgba(1.0, 0.0, 0.0, 1.0),
            mode: PaintMode::Replace,
        };
        cmd.execute(&mut d, ctx).unwrap();
        let img = crate::composite::flatten(&d);
        assert_eq!(img.get(0, 0), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(img.get(3, 0), [1.0, 1.0, 1.0, 1.0]);
        // Alpha offset +100% makes everything opaque.
        let mut lf = d.layer(id).unwrap().filter;
        lf.invert = false;
        lf.adjust.alpha = 1.0;
        assert!(d.set_layer_filter(id, lf));
        d.undo();
        assert!(d.layer(id).unwrap().filter.invert);
        // Turning it back into a pixel layer shows the red pixel.
        Command::SetFilterOn { id, on: false }.execute(&mut d, ctx).unwrap();
        assert_eq!(crate::composite::flatten(&d).get(0, 0), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn replace_image_and_layer() {
        let mut d = Document::new(4, 4, Depth::U8, Some(Color::WHITE));
        let ctx = &mut ExecContext::default();
        let id = d.active;
        let px = Pixels::filled(Depth::U16, 2, 3, [0.0, 1.0, 0.0, 1.0]);
        let one = |name: &str, offset| NewLayer { pixels: px.clone(), offset, name: name.into() };
        let label = "Magick".to_string();
        Command::ReplaceLayers { ids: vec![id], results: vec![one("x", (1, 1))], label: label.clone() }
            .execute(&mut d, ctx)
            .unwrap();
        assert_eq!(d.layer(id).unwrap().bounds(), Rect::new(1, 1, 2, 3));
        // Two layers in, three out: they replace both where the top one was.
        let a = d.add_layer();
        let b = d.add_layer();
        d.set_active(a);
        d.selected.insert(id);
        let results = vec![one("r0", (0, 0)), one("r1", (0, 0)), one("r2", (0, 0))];
        Command::ReplaceLayers { ids: vec![a, id], results, label: label.clone() }.execute(&mut d, ctx).unwrap();
        let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["r2", "r1", "r0", d.layer(b).unwrap().name.as_str()]);
        assert_eq!(d.active_layer().name, "r0");
        d.undo();
        assert_eq!(d.layers.len(), 3);
        let results = vec![one("p", (0, 0)), NewLayer { pixels: Pixels::new(Depth::U8, 5, 1), offset: (0, 0), name: "q".into() }];
        Command::ReplaceImage { results, label }.execute(&mut d, ctx).unwrap();
        assert_eq!((d.width, d.height, d.layers.len()), (5, 3, 2));
        assert_eq!(d.layers[1].name, "p");
        d.undo();
        assert_eq!((d.width, d.height), (4, 4));
    }

}
