//! Project files: layered documents saved as Photoshop PSD, through the
//! `ag-psd` crate. Photoshop, GIMP, Krita and Affinity open these.
//!
//! What is kept: every layer's pixels, name, position (layers may hang off
//! the canvas), visibility and opacity; groups become PSD layer
//! groups (and PSD groups open as groups).
//!
//! Precision: a PSD has one bit depth for the whole file, so the document is
//! saved at its deepest layer's depth (8, 16 or 32-bit float). `ag-psd` keeps
//! pixels as 8-bit RGBA internally, so for 16 and 32-bit files the channel
//! data is written and read here directly through the crate's raw-channel
//! model (big-endian samples; ZIP with prediction for 16-bit, as Photoshop
//! writes, and plain ZIP for 32-bit),
//! which keeps full precision. Filter layers are saved as hidden pixel layers
//! (their pixels are the mask) with their settings in the file's XMP
//! metadata. Photoshop adjustment layers from other programs open as filter
//! layers where the kinds match (brightness/contrast, hue/saturation, levels,
//! exposure, invert). Layer masks, blend modes, effects and other adjustment
//! kinds of foreign files are not applied.

use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use ag_psd::psd::{
    AdjustmentLayer, ChannelId, ColorMode, Compression, ImageResources, Layer as PLayer, LayerAdditionalInfo, LayerRawData,
    LayerRawDataChannel, PixelData, Psd, ReadOptions, WriteOptions,
};

use crate::buffer::Buffer;
use crate::document::Document;
use crate::layer::{GroupId, Layer};
use crate::ops::filters::LayerFilter;
use crate::pixel::Depth;
use crate::pixels::Pixels;

/// A cheap, thread-safe copy of what a project file stores (pixels are
/// shared), so saving can run in the background.
#[derive(Clone)]
pub struct Snapshot {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<SnapLayer>,
}

#[derive(Clone)]
pub struct SnapLayer {
    pub name: String,
    pub pixels: Arc<Pixels>,
    pub offset: (i32, i32),
    pub visible: bool,
    pub opacity: f32,
    pub group: Option<GroupId>,
    /// Set for filter layers.
    pub filter: Option<LayerFilter>,
}

impl Snapshot {
    pub fn of(doc: &Document) -> Self {
        Snapshot {
            width: doc.width,
            height: doc.height,
            layers: doc
                .layers
                .iter()
                .map(|l| SnapLayer {
                    name: l.name.clone(),
                    pixels: l.pixels.clone(),
                    offset: l.offset,
                    visible: l.visible,
                    opacity: l.opacity,
                    group: l.group,
                    filter: l.filter_on.then_some(l.filter),
                })
                .collect(),
        }
    }

    pub fn depth(&self) -> Depth {
        self.layers.iter().filter(|l| l.filter.is_none()).map(|l| l.pixels.depth()).max().unwrap_or(Depth::U8)
    }
}

// ----- channel coding (16 / 32-bit) -----------------------------------------

fn zip(data: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(data).expect("in-memory write");
    e.finish().expect("in-memory write")
}

fn unzip(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(data).read_to_end(&mut out).ok()?;
    Some(out)
}

/// Photoshop's "ZIP with prediction" filter, in place, for one channel.
fn predict(data: &mut [u8], w: usize, bytes: usize, encode: bool) {
    let row_len = w * bytes;
    for row in data.chunks_exact_mut(row_len) {
        match bytes {
            1 => delta_u8(row, encode),
            2 => {
                let mut v: Vec<u16> = row.chunks_exact(2).map(|b| u16::from_be_bytes([b[0], b[1]])).collect();
                if encode {
                    for i in (1..v.len()).rev() {
                        v[i] = v[i].wrapping_sub(v[i - 1]);
                    }
                } else {
                    for i in 1..v.len() {
                        v[i] = v[i].wrapping_add(v[i - 1]);
                    }
                }
                for (d, s) in row.chunks_exact_mut(2).zip(v) {
                    d.copy_from_slice(&s.to_be_bytes());
                }
            }
            _ => {
                // 32-bit: bytes are split into planes (all MSBs first), then delta-coded.
                if encode {
                    let mut planes = vec![0u8; row_len];
                    for x in 0..w {
                        for k in 0..4 {
                            planes[k * w + x] = row[x * 4 + k];
                        }
                    }
                    delta_u8(&mut planes, true);
                    row.copy_from_slice(&planes);
                } else {
                    delta_u8(row, false);
                    let planes = row.to_vec();
                    for x in 0..w {
                        for k in 0..4 {
                            row[x * 4 + k] = planes[k * w + x];
                        }
                    }
                }
            }
        }
    }
}

fn delta_u8(row: &mut [u8], encode: bool) {
    if encode {
        for i in (1..row.len()).rev() {
            row[i] = row[i].wrapping_sub(row[i - 1]);
        }
    } else {
        for i in 1..row.len() {
            row[i] = row[i].wrapping_add(row[i - 1]);
        }
    }
}

/// PackBits rows, preceded by a table of row byte counts.
fn unpack_rle(data: &[u8], w: usize, h: usize, bytes: usize, large: bool) -> Option<Vec<u8>> {
    let entry = if large { 4 } else { 2 };
    let table = data.get(..h * entry)?;
    let mut pos = h * entry;
    let row_len = w * bytes;
    let mut out = Vec::with_capacity(row_len * h);
    for r in 0..h {
        let n = if large {
            u32::from_be_bytes(table[r * 4..r * 4 + 4].try_into().ok()?) as usize
        } else {
            u16::from_be_bytes([table[r * 2], table[r * 2 + 1]]) as usize
        };
        let src = data.get(pos..pos + n)?;
        pos += n;
        let start = out.len();
        let mut i = 0;
        while i < src.len() && out.len() - start < row_len {
            let c = src[i] as i8;
            i += 1;
            if c >= 0 {
                let k = c as usize + 1;
                out.extend_from_slice(src.get(i..i + k)?);
                i += k;
            } else if c != -128 {
                let v = *src.get(i)?;
                i += 1;
                out.extend(std::iter::repeat_n(v, 1 - c as isize as usize));
            }
        }
        out.resize(start + row_len, 0);
    }
    Some(out)
}

/// Decodes one channel to big-endian sample bytes.
fn channel_bytes(c: &LayerRawDataChannel, w: usize, h: usize, bytes: usize, large: bool) -> Option<Vec<u8>> {
    let data = c.data.as_deref().unwrap_or(&[]);
    let want = w * h * bytes;
    let mut out = match c.compression {
        Compression::RawData => data.to_vec(),
        Compression::RleCompressed => unpack_rle(data, w, h, bytes, large)?,
        Compression::ZipWithoutPrediction => unzip(data)?,
        Compression::ZipWithPrediction => {
            let mut d = unzip(data)?;
            d.resize(want, 0);
            predict(&mut d, w, bytes, false);
            d
        }
    };
    out.resize(want, 0);
    Some(out)
}

fn sample(b: &[u8], bytes: usize) -> f32 {
    match bytes {
        1 => b[0] as f32 / 255.0,
        2 => u16::from_be_bytes([b[0], b[1]]) as f32 / 65535.0,
        _ => f32::from_be_bytes([b[0], b[1], b[2], b[3]]),
    }
}

/// Decodes a layer's RGB(A) channels at full precision.
fn decode_raw(raw: &LayerRawData, w: usize, h: usize) -> Option<Pixels> {
    if raw.color_mode != ColorMode::Rgb {
        return None;
    }
    let bytes = match raw.bits_per_channel as u32 {
        8 => 1,
        16 => 2,
        32 => 4,
        _ => return None,
    };
    let find = |id: ChannelId| raw.channels.iter().find(|c| c.id == id);
    let mut planes = Vec::new();
    for id in [ChannelId::Color0, ChannelId::Color1, ChannelId::Color2] {
        planes.push(channel_bytes(find(id)?, w, h, bytes, raw.large)?);
    }
    let alpha = match find(ChannelId::Transparency) {
        Some(c) => Some(channel_bytes(c, w, h, bytes, raw.large)?),
        None => None,
    };
    let img = Buffer::<f32>::from_fn(w as u32, h as u32, |x, y| {
        let i = (y as usize * w + x as usize) * bytes;
        let a = alpha.as_ref().map_or(1.0, |p| sample(&p[i..], bytes));
        [sample(&planes[0][i..], bytes), sample(&planes[1][i..], bytes), sample(&planes[2][i..], bytes), a]
    });
    Some(match bytes {
        1 => Pixels::U8(img.convert()),
        2 => Pixels::U16(img.convert()),
        _ => Pixels::F32(img),
    })
}

/// Encodes a layer as big-endian channels at 16 or 32 bits.
fn encode_raw(px: &Pixels, depth: Depth) -> LayerRawData {
    let (w, h) = (px.width() as usize, px.height() as usize);
    let bytes = depth.bytes_per_channel();
    let f = px.to_f32();
    let mut channels = Vec::new();
    for (id, c) in [(ChannelId::Transparency, 3), (ChannelId::Color0, 0), (ChannelId::Color1, 1), (ChannelId::Color2, 2)] {
        let mut d = Vec::with_capacity(w * h * bytes);
        for p in f.data.chunks_exact(4) {
            let v = p[c];
            match depth {
                Depth::U16 => d.extend_from_slice(&u16::from_f32(v).to_be_bytes()),
                _ => d.extend_from_slice(&v.to_be_bytes()),
            }
        }
        // 16-bit: ZIP with prediction (what Photoshop writes). 32-bit: plain
        // ZIP, which more readers support.
        let compression = if depth == Depth::U16 {
            predict(&mut d, w, bytes, true);
            Compression::ZipWithPrediction
        } else {
            Compression::ZipWithoutPrediction
        };
        channels.push(LayerRawDataChannel { id, compression, data: Some(zip(&d)) });
    }
    LayerRawData { color_mode: ColorMode::Rgb, bits_per_channel: (bytes * 8) as f64, channels, large: false }
}

use crate::pixel::Channel as _;

fn rgba8(px: &Pixels) -> PixelData {
    let b = px.to_f32().convert::<u8>();
    PixelData { width: b.width, height: b.height, data: b.data }
}

// ----- save --------------------------------------------------------------------

/// A Photoshop adjustment layer as filter-layer settings (approximately:
/// the models differ), for files from other programs.
fn from_psd_adjustment(a: &AdjustmentLayer) -> Option<LayerFilter> {
    let mut f = LayerFilter::default();
    match a {
        AdjustmentLayer::Brightness(b) => {
            f.adjust.brightness = (b.brightness.unwrap_or(0.0) / 255.0) as f32;
            f.adjust.contrast = (1.0 + b.contrast.unwrap_or(0.0) / 100.0) as f32;
        }
        AdjustmentLayer::HueSaturation(h) => {
            let m = h.master.clone().unwrap_or_default();
            f.adjust.hue = m.hue as f32;
            f.adjust.saturation = (1.0 + m.saturation / 100.0) as f32;
            f.adjust.brightness = (m.lightness / 200.0) as f32;
        }
        AdjustmentLayer::Levels(l) => {
            let c = l.rgb.clone()?;
            f.levels = Some(((c.shadow_input / 255.0) as f32, (c.highlight_input / 255.0) as f32));
            if c.midtone_input > 0.0 {
                f.adjust.gamma = c.midtone_input as f32;
            }
        }
        AdjustmentLayer::Exposure(e) => {
            f.adjust.exposure = e.exposure.unwrap_or(0.0) as f32;
            f.adjust.brightness = e.offset.unwrap_or(0.0) as f32;
            f.adjust.gamma = e.gamma.filter(|g| *g > 0.0).unwrap_or(1.0) as f32;
        }
        AdjustmentLayer::Invert(_) => f.invert = true,
        _ => return None,
    }
    Some(f)
}

/// imged's own data in the PSD's XMP metadata: which layers are filter
/// layers and their settings. Other programs see those layers as hidden
/// pixel layers (their pixels are the mask).
#[derive(serde::Serialize, serde::Deserialize)]
struct XmpFilterLayer {
    /// Index in the bottom-to-top layer list.
    index: usize,
    name: String,
    visible: bool,
    filter: LayerFilter,
}

const XMP_OPEN: &str = "<imged:filterLayers>";
const XMP_CLOSE: &str = "</imged:filterLayers>";

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
}

fn filter_xmp(entries: &[XmpFilterLayer]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let json = serde_json::to_string(entries).ok()?;
    Some(format!(
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\
<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\" \
xmlns:imged=\"urn:imged:1.0\">{XMP_OPEN}{}{XMP_CLOSE}</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>",
        xml_escape(&json)
    ))
}

fn parse_filter_xmp(xmp: &str) -> Vec<XmpFilterLayer> {
    let Some(a) = xmp.find(XMP_OPEN) else { return Vec::new() };
    let rest = &xmp[a + XMP_OPEN.len()..];
    let Some(b) = rest.find(XMP_CLOSE) else { return Vec::new() };
    serde_json::from_str(&xml_unescape(&rest[..b])).unwrap_or_default()
}

fn to_player(l: &SnapLayer, depth: Depth) -> PLayer {
    let (w, h) = (l.pixels.width() as i32, l.pixels.height() as i32);
    let mut out = PLayer {
        additional_info: LayerAdditionalInfo { name: Some(l.name.clone()), ..Default::default() },
        left: Some(l.offset.0 as f64),
        top: Some(l.offset.1 as f64),
        right: Some((l.offset.0 + w) as f64),
        bottom: Some((l.offset.1 + h) as f64),
        opacity: Some(l.opacity as f64),
        // Filter layers' pixels are a mask: other programs shouldn't draw them.
        hidden: Some(!l.visible || l.filter.is_some()),
        ..Default::default()
    };
    if depth == Depth::U8 {
        out.image_data = Some(rgba8(&l.pixels));
    } else {
        out.raw_data = Some(encode_raw(&l.pixels, depth));
    }
    out
}

/// Encodes a snapshot as PSD bytes. Returns the bytes and the depth used.
pub fn encode(snap: &Snapshot) -> Result<(Vec<u8>, Depth), String> {
    if snap.width > 30000 || snap.height > 30000 {
        return Err("PSD files are limited to 30000×30000 pixels".into());
    }
    let depth = snap.depth();
    let mut children: Vec<PLayer> = Vec::new();
    let mut i = 0;
    while i < snap.layers.len() {
        let l = &snap.layers[i];
        match l.group {
            Some(g) => {
                let mut j = i;
                while j < snap.layers.len() && snap.layers[j].group == Some(g) {
                    j += 1;
                }
                let members = &snap.layers[i..j];
                let top = &members[members.len() - 1];
                children.push(PLayer {
                    additional_info: LayerAdditionalInfo { name: Some(top.name.clone()), ..Default::default() },
                    hidden: Some(members.iter().all(|m| !m.visible)),
                    opacity: Some(1.0),
                    opened: Some(true),
                    children: Some(members.iter().map(|m| to_player(m, depth)).collect()),
                    ..Default::default()
                });
                i = j;
            }
            None => {
                children.push(to_player(l, depth));
                i += 1;
            }
        }
    }
    // Flattened preview for apps that only read the composite.
    let doc_like = crate::composite::composite_views(
        &snap
            .layers
            .iter()
            .filter(|l| l.visible)
            .map(|l| crate::composite::LayerView {
                pixels: &l.pixels,
                offset: l.offset,
                opacity: l.opacity,
                filter: l.filter.map(|filter| crate::composite::FilterView {
                    filter,
                    everywhere: crate::composite::is_blank(&l.pixels),
                }),
            })
            .collect::<Vec<_>>(),
        crate::geom::Rect::from_size(snap.width, snap.height),
        None,
    );
    let composite = rgba8(&Pixels::F32(doc_like));
    let psd = Psd {
        width: snap.width as f64,
        height: snap.height as f64,
        color_mode: Some(ColorMode::Rgb),
        bits_per_channel: Some((depth.bytes_per_channel() * 8) as f64),
        image_data: Some(composite),
        children: Some(children),
        image_resources: filter_xmp(
            &snap
                .layers
                .iter()
                .enumerate()
                .filter_map(|(index, l)| {
                    Some(XmpFilterLayer { index, name: l.name.clone(), visible: l.visible, filter: l.filter? })
                })
                .collect::<Vec<_>>(),
        )
        .map(|x| ImageResources { xmp_metadata: Some(x), ..Default::default() }),
        ..Default::default()
    };
    let bytes = catch_unwind(AssertUnwindSafe(|| {
        ag_psd::write_psd(&psd, &WriteOptions { compress: Some(true), ..Default::default() })
    }))
    .map_err(|e| {
        let msg = e.downcast_ref::<String>().cloned().or(e.downcast_ref::<&str>().map(|s| s.to_string()));
        format!("Couldn't write the PSD: {}", msg.unwrap_or_else(|| "internal error".into()))
    })?;
    Ok((bytes, depth))
}

/// Saves the document as a PSD file. Returns the depth written.
pub fn save(doc: &Document, path: &Path) -> Result<Depth, String> {
    save_snapshot(&Snapshot::of(doc), path)
}

/// Saves a snapshot, writing to a temporary file first so a crash or full
/// disk never leaves a half-written project behind.
pub fn save_snapshot(snap: &Snapshot, path: &Path) -> Result<Depth, String> {
    let (bytes, depth) = encode(snap)?;
    let tmp = path.with_extension("psd.partial");
    std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    Ok(depth)
}

// ----- load ---------------------------------------------------------------------

struct Loaded {
    name: String,
    pixels: Pixels,
    offset: (i32, i32),
    visible: bool,
    opacity: f32,
    group: Option<usize>,
    filter: Option<LayerFilter>,
}

fn layer_pixels(l: &PLayer) -> Result<(Pixels, (i32, i32)), String> {
    let (left, top) = (l.left.unwrap_or(0.0) as i32, l.top.unwrap_or(0.0) as i32);
    let (right, bottom) = (l.right.unwrap_or(0.0) as i32, l.bottom.unwrap_or(0.0) as i32);
    let (w, h) = ((right - left).max(0) as usize, (bottom - top).max(0) as usize);
    if w == 0 || h == 0 {
        return Ok((Pixels::new(Depth::U8, 1, 1), (left, top)));
    }
    if let Some(px) = l.raw_data.as_ref().and_then(|r| decode_raw(r, w, h)) {
        return Ok((px, (left, top)));
    }
    // Other color modes / codings: let ag-psd decode (8-bit).
    let pd = ag_psd::get_layer_image_data(l).map_err(|e| format!("{e:?}"))?;
    let pd = pd.ok_or("A layer has no pixel data")?;
    let b = Buffer::<u8>::from_raw(pd.width, pd.height, pd.data).ok_or("Bad layer pixel data")?;
    Ok((Pixels::U8(b), (left, top)))
}

fn collect(children: &[PLayer], group: Option<usize>, hidden: bool, groups: &mut usize, out: &mut Vec<Loaded>) -> Result<(), String> {
    for c in children {
        let name = c.additional_info.name.clone().unwrap_or_else(|| "Layer".into());
        let is_hidden = hidden || c.hidden == Some(true);
        match &c.children {
            Some(kids) => {
                // Top-level groups become groups; nested ones join their parent.
                let g = group.or_else(|| {
                    *groups += 1;
                    Some(*groups)
                });
                collect(kids, g, is_hidden, groups, out)?;
            }
            None => {
                let filter = c.additional_info.adjustment.as_ref().and_then(from_psd_adjustment);
                let (pixels, offset) = match filter {
                    Some(_) => (Pixels::new(Depth::U8, 1, 1), (0, 0)),
                    None => layer_pixels(c)?,
                };
                out.push(Loaded {
                    name,
                    pixels,
                    offset,
                    visible: !is_hidden,
                    opacity: c.opacity.unwrap_or(1.0) as f32,
                    group,
                    filter,
                });
            }
        }
    }
    Ok(())
}

/// Opens a PSD (or PSB) as a layered document.
pub fn load(path: &Path) -> Result<Document, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    load_bytes(&bytes)
}

pub fn load_bytes(bytes: &[u8]) -> Result<Document, String> {
    let opts = ReadOptions {
        use_raw_data: Some(true),
        skip_thumbnail: Some(true),
        skip_linked_files_data: Some(true),
        total_memory_limit: None,
        ..Default::default()
    };
    let psd = catch_unwind(AssertUnwindSafe(|| ag_psd::read_psd(bytes, &opts)))
        .map_err(|_| "Couldn't read the PSD (internal error)".to_string())?
        .map_err(|e| format!("Couldn't read the PSD: {e:?}"))?;
    let (w, h) = (psd.width as u32, psd.height as u32);
    let mut loaded = Vec::new();
    let mut groups = 0;
    if let Some(children) = &psd.children {
        collect(children, None, false, &mut groups, &mut loaded)?;
    }
    if loaded.is_empty() {
        // A flat file: use the composite image.
        let pd = ag_psd::get_composite_image_data(&psd)
            .map_err(|e| format!("{e:?}"))?
            .ok_or("The PSD has no layers and no image")?;
        let b = Buffer::<u8>::from_raw(pd.width, pd.height, pd.data).ok_or("Bad PSD image data")?;
        loaded.push(Loaded {
            name: "Background".into(),
            pixels: Pixels::U8(b),
            offset: (0, 0),
            visible: true,
            opacity: 1.0,
            group: None,
            filter: None,
        });
    }
    let mut doc = Document::new(w.max(1), h.max(1), loaded[0].pixels.depth(), None);
    let group_ids: Vec<GroupId> = (0..=groups).map(|_| doc.alloc_id()).collect();
    let mut layers = Vec::new();
    for l in loaded {
        let id = doc.alloc_id();
        let mut layer = Layer::new(id, l.name, l.pixels, l.offset);
        layer.visible = l.visible;
        layer.opacity = l.opacity.clamp(0.0, 1.0);
        layer.group = l.group.map(|g| group_ids[g]);
        if let Some(f) = l.filter {
            layer.filter_on = true;
            layer.filter = f;
        }
        layers.push(layer);
    }
    // imged's filter layers (stored as hidden pixel layers).
    let xmp = psd.image_resources.as_ref().and_then(|r| r.xmp_metadata.as_deref()).unwrap_or("");
    for e in parse_filter_xmp(xmp) {
        if let Some(l) = layers.get_mut(e.index).filter(|l| l.name == e.name) {
            l.filter_on = true;
            l.filter = e.filter;
            l.visible = e.visible;
        }
    }
    doc.default_depth = layers.iter().filter(|l| !l.is_filter()).map(|l| l.depth()).max().unwrap_or(Depth::U8);
    doc.active = layers.last().unwrap().id;
    doc.layers = layers;
    doc.clear_history();
    doc.mark_all_dirty();
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;

    fn roundtrip(depth: Depth) {
        let mut d = Document::new(30, 20, depth, Some(Color::WHITE));
        let mut px = Pixels::new(depth, 10, 5);
        px.set(1, 2, [0.1234, 0.5, 0.75, 0.5]);
        if depth == Depth::F32 {
            px.set(3, 3, [4.5, 0.0, 1.0, 1.0]);
        }
        let a = d.add_layer_from("a", px, (-3, 4));
        let b = d.add_layer_from("b", Pixels::filled(depth, 4, 4, [0.0, 1.0, 0.0, 1.0]), (20, 10));
        d.layer_mut(b).unwrap().opacity = 0.5;
        d.layer_mut(a).unwrap().visible = false;
        d.set_active(b);
        d.selected.insert(a);
        assert!(d.group_selected());

        let (bytes, written) = encode(&Snapshot::of(&d)).unwrap();
        assert_eq!(written, depth);
        let e = load_bytes(&bytes).unwrap();
        assert_eq!((e.width, e.height), (30, 20));
        assert_eq!(e.layers.len(), 3);
        let names: Vec<_> = e.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["Background", "a", "b"]);
        let la = &e.layers[1];
        assert_eq!(la.offset, (-3, 4));
        assert_eq!(la.depth(), depth);
        assert!(!la.visible);
        assert!((e.layers[2].opacity - 0.5).abs() < 0.01);
        assert!(la.group.is_some() && la.group == e.layers[2].group);
        assert!(e.layers[0].group.is_none());
        let p = la.pixels.get(1, 2);
        let tol = if depth == Depth::U8 { 0.003 } else { 1e-4 };
        assert!((p[0] - 0.1234).abs() < tol && (p[3] - 0.5).abs() < tol, "{depth:?} {p:?}");
        if depth == Depth::F32 {
            assert_eq!(la.pixels.get(3, 3)[0], 4.5);
        }
    }

    #[test]
    fn filter_layers_roundtrip() {
        let mut d = Document::new(8, 8, Depth::U16, Some(Color::rgba(0.2, 0.4, 0.6, 1.0)));
        let id = d.add_layer_from("mask", Pixels::filled(Depth::U16, 4, 4, [1.0; 4]), (2, 2));
        let mut f = LayerFilter::default();
        f.invert = true;
        f.adjust.exposure = 0.5;
        f.adjust.hue = 30.0;
        d.layer_mut(id).unwrap().filter = f;
        d.layer_mut(id).unwrap().filter_on = true;
        let before = crate::composite::flatten(&d);
        let (bytes, _) = encode(&Snapshot::of(&d)).unwrap();
        let e = load_bytes(&bytes).unwrap();
        let l = &e.layers[1];
        assert!(l.filter_on && l.visible);
        assert_eq!(l.filter, f);
        let after = crate::composite::flatten(&e);
        for (x, y) in [(0, 0), (3, 3)] {
            let p = (before.get(x, y), after.get(x, y));
            assert!((0..4).all(|i| (p.0[i] - p.1[i]).abs() < 1e-3), "{p:?}");
        }
        assert!((before.get(3, 3)[0] - before.get(0, 0)[0]).abs() > 0.1);
    }

    #[test]
    fn roundtrip_all_depths() {
        for d in Depth::ALL {
            roundtrip(d);
        }
    }

    #[test]
    fn prediction_inverts() {
        for bytes in [1usize, 2, 4] {
            let orig: Vec<u8> = (0..(5 * bytes * 3) as u32).map(|i| (i * 37 % 251) as u8).collect();
            let mut d = orig.clone();
            predict(&mut d, 5, bytes, true);
            assert_ne!(d, orig);
            predict(&mut d, 5, bytes, false);
            assert_eq!(d, orig);
        }
    }
}

#[cfg(test)]
mod export_samples {
    use super::*;

    /// Writes sample files for checking with other programs:
    /// `IMGED_PSD_OUT=dir cargo test -p imged-core write_samples -- --ignored`
    #[test]
    #[ignore]
    fn write_samples() {
        let dir = std::env::var("IMGED_PSD_OUT").unwrap();
        for depth in Depth::ALL {
            let mut d = Document::new(64, 48, depth, Some(crate::color::Color::WHITE));
            let a = d.add_layer_from("red", Pixels::filled(depth, 20, 20, [1.0, 0.0, 0.0, 1.0]), (5, 5));
            let b = d.add_layer_from("green half", Pixels::filled(depth, 20, 20, [0.0, 1.0, 0.0, 1.0]), (30, 20));
            d.layer_mut(b).unwrap().opacity = 0.5;
            d.add_layer_from("hidden", Pixels::filled(depth, 8, 8, [0.0, 0.0, 1.0, 1.0]), (50, 2));
            d.layers.last_mut().unwrap().visible = false;
            d.set_active(b);
            d.selected.insert(a);
            d.group_selected();
            save(&d, &Path::new(&dir).join(format!("sample-{}.psd", depth.short()))).unwrap();
        }
    }
}
