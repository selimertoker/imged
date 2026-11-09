//! GPU canvas renderer (wgpu): layers live in GPU textures and are
//! composited in a shader every time the document changes.
//!
//! - Each visible layer (or its live preview) is a texture, kept between
//!   frames and keyed by its pixel buffer; edits upload only the changed
//!   region. 8-bit layers are `Rgba8Unorm`, 16-bit and float layers
//!   `Rgba32Float` (exact values, HDR included).
//! - Layers are drawn bottom to top into a premultiplied `Rgba16Float`
//!   canvas with "over" blending. A filter layer reads that canvas and
//!   writes the filtered result into a second one (ping-pong), so filter
//!   layers behave exactly like on the CPU.
//! - A display pass applies the view exposure / sRGB encoding into an 8-bit
//!   texture with mipmaps (smooth when zoomed out, crisp pixels when zoomed
//!   in), which egui draws directly.
//!
//! Exports, filters and tools keep using the CPU compositor
//! (`imged_core::composite`), which stays the reference. When the canvas or
//! a layer is larger than the GPU allows, the CPU renderer is used instead.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use eframe::egui::{self, Color32, Painter, Rect, TextureId};
use eframe::egui_wgpu::{self, wgpu};
use imged_core::{Document, Pixels, Rect as IRect};

use crate::renderer::{CanvasRenderer, CpuTiledRenderer};
use crate::state::DisplaySettings;
use crate::view::ViewXform;

const ACC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Shader parameters for one pass (matches `Params` in gpu.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rect: [f32; 4],
    canvas: [f32; 4],
    adj0: [f32; 4],
    adj1: [f32; 4],
    extra: [f32; 4],
}

struct LayerTex {
    pixels: Weak<Pixels>,
    tex: wgpu::Texture,
    bind: wgpu::BindGroup,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    used: bool,
}

/// The two canvas-sized accumulation textures.
struct Acc {
    size: (u32, u32),
    views: [wgpu::TextureView; 2],
    binds: [wgpu::BindGroup; 2],
}

/// The displayed 8-bit texture and its mip levels.
struct Out {
    size: (u32, u32),
    #[cfg_attr(not(test), allow(dead_code))]
    tex: wgpu::Texture,
    /// One view per mip level (render targets).
    levels: Vec<wgpu::TextureView>,
    /// Each level as a texture source for the next one.
    mip_binds: Vec<wgpu::BindGroup>,
    full_view: wgpu::TextureView,
}

struct Pipes {
    layer: wgpu::RenderPipeline,
    filter: wgpu::RenderPipeline,
    display: wgpu::RenderPipeline,
    mip: wgpu::RenderPipeline,
    params_layout: wgpu::BindGroupLayout,
    tex_layout: wgpu::BindGroupLayout,
    mip_layout: wgpu::BindGroupLayout,
    linear: wgpu::Sampler,
}

enum Op {
    Layer { key: usize, slot: u32 },
    Filter { key: usize, slot: u32 },
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    egui: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    max_dim: u32,
    pipes: Pipes,
    layers: HashMap<usize, LayerTex>,
    acc: Option<Acc>,
    out: Option<Out>,
    tex_id: Option<TextureId>,
    params: wgpu::Buffer,
    params_slots: u32,
    params_bind: wgpu::BindGroup,
    align: u32,
    display: DisplaySettings,
    size: (u32, u32),
    /// Used for documents too large for the GPU.
    cpu: Option<CpuTiledRenderer>,
    /// Everything must be uploaded and drawn again (e.g. after the CPU fallback).
    force: bool,
    /// The GPU adapter, for the View menu.
    pub adapter: String,
}

fn tex_layout_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn params_buffer(device: &wgpu::Device, slots: u32, align: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imged params"),
        size: (slots * align) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn params_bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, buf: &wgpu::Buffer) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("imged params"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
            }),
        }],
    })
}

impl GpuRenderer {
    pub fn new(rs: &egui_wgpu::RenderState) -> Self {
        let device = rs.device.clone();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imged canvas"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imged params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            }],
        });
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imged texture"),
            entries: &[tex_layout_entry(0, false)],
        });
        let mip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imged mip"),
            entries: &[
                tex_layout_entry(0, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline = |label: &str,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        vs: &str,
                        fs: &str,
                        format: wgpu::TextureFormat,
                        blend: Option<wgpu::BlendState>| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let over = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let p = Some(&params_layout);
        let t = Some(&tex_layout);
        let pipes = Pipes {
            layer: pipeline("imged layer", &[p, t], "vs_quad", "fs_layer", ACC_FORMAT, Some(over)),
            filter: pipeline("imged filter", &[p, t, t], "vs_full", "fs_filter", ACC_FORMAT, None),
            display: pipeline("imged display", &[p, None, t], "vs_full", "fs_display", OUT_FORMAT, None),
            mip: pipeline("imged mip", &[None, None, None, Some(&mip_layout)], "vs_full", "fs_mip", OUT_FORMAT, None),
            linear: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("imged mip"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            params_layout,
            tex_layout,
            mip_layout,
        };
        let limits = device.limits();
        let align = limits.min_uniform_buffer_offset_alignment.max(std::mem::size_of::<Params>() as u32);
        let params_slots = 64;
        let params = params_buffer(&device, params_slots, align);
        let params_bind = params_bind(&device, &pipes.params_layout, &params);
        let info = rs.adapter.get_info();
        GpuRenderer {
            adapter: format!("{} ({:?})", info.name, info.backend),
            max_dim: limits.max_texture_dimension_2d,
            device,
            queue: rs.queue.clone(),
            egui: rs.renderer.clone(),
            pipes,
            layers: HashMap::new(),
            acc: None,
            out: None,
            tex_id: None,
            params,
            params_slots,
            params_bind,
            align,
            display: DisplaySettings::default(),
            size: (0, 0),
            cpu: None,
            force: true,
        }
    }

    /// Whether the document fits the GPU's texture size limit.
    fn fits(&self, doc: &Document) -> bool {
        let ok = |w: u32, h: u32| w <= self.max_dim && h <= self.max_dim;
        ok(doc.width, doc.height)
            && doc.layers.iter().filter(|l| l.visible).all(|l| {
                let px = doc.preview.get(&l.id).map_or(&l.pixels, |p| &p.pixels);
                ok(px.width(), px.height())
            })
    }

    /// A texture to draw into (`target`) or only to read from.
    fn texture(&self, label: &str, size: (u32, u32), format: wgpu::TextureFormat, mips: u32, target: bool) -> wgpu::Texture {
        let usage = if target {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        };
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn tex_bind(&self, view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.tex_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }],
        })
    }

    /// (Re)creates the canvas-sized textures.
    fn ensure_targets(&mut self, size: (u32, u32)) {
        if self.acc.as_ref().is_none_or(|a| a.size != size) {
            let make = |s: &Self| s.texture("imged canvas", size, ACC_FORMAT, 1, true).create_view(&Default::default());
            let views = [make(self), make(self)];
            let binds = [self.tex_bind(&views[0]), self.tex_bind(&views[1])];
            self.acc = Some(Acc { size, views, binds });
        }
        if self.out.as_ref().is_none_or(|o| o.size != size) {
            let mips = 32 - size.0.max(size.1).max(1).leading_zeros();
            let tex = self.texture("imged display", size, OUT_FORMAT, mips, true);
            let level = |l: u32| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: l,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            };
            let levels: Vec<_> = (0..mips).map(level).collect();
            let mip_binds = levels
                .iter()
                .map(|v| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipes.mip_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.pipes.linear) },
                        ],
                    })
                })
                .collect();
            let full_view = tex.create_view(&Default::default());
            let sampler = wgpu::SamplerDescriptor {
                label: Some("imged canvas"),
                // Crisp pixels when zoomed in, smooth (mipmapped) when zoomed out.
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            };
            let mut egui = self.egui.write();
            match self.tex_id {
                Some(id) => egui.update_egui_texture_from_wgpu_texture_with_sampler_options(&self.device, &full_view, sampler, id),
                None => {
                    self.tex_id = Some(egui.register_native_texture_with_sampler_options(&self.device, &full_view, sampler))
                }
            }
            drop(egui);
            self.out = Some(Out { size, tex, levels, mip_binds, full_view });
        }
    }

    /// Uploads `rect` (layer coordinates) of `px` into `tex`.
    fn write(&self, tex: &wgpu::Texture, px: &Pixels, r: IRect) {
        let w = px.width() as usize;
        let origin = wgpu::Origin3d { x: r.x as u32, y: r.y as u32, z: 0 };
        let size = wgpu::Extent3d { width: r.w, height: r.h, depth_or_array_layers: 1 };
        let dst = wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin, aspect: wgpu::TextureAspect::All };
        let start = (r.y as usize * w + r.x as usize) * 4;
        match px {
            Pixels::U8(b) => {
                let layout = wgpu::TexelCopyBufferLayout { offset: start as u64, bytes_per_row: Some(w as u32 * 4), rows_per_image: None };
                self.queue.write_texture(dst, &b.data, layout, size);
            }
            Pixels::F32(b) => {
                let layout =
                    wgpu::TexelCopyBufferLayout { offset: start as u64 * 4, bytes_per_row: Some(w as u32 * 16), rows_per_image: None };
                self.queue.write_texture(dst, bytemuck::cast_slice(&b.data), layout, size);
            }
            Pixels::U16(b) => {
                // Converted to float, row by row, for the changed rectangle only.
                let mut data = Vec::with_capacity(r.w as usize * r.h as usize * 4);
                for y in r.y..r.bottom() {
                    let row = (y as usize * w + r.x as usize) * 4;
                    data.extend(b.data[row..row + r.w as usize * 4].iter().map(|v| *v as f32 / 65535.0));
                }
                let layout = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(r.w * 16), rows_per_image: None };
                self.queue.write_texture(dst, bytemuck::cast_slice(&data), layout, size);
            }
        }
    }

    /// Makes sure `px` (placed at `offset`) is on the GPU and up to date
    /// with the changed region `dirty`. Returns its key.
    fn upload(&mut self, px: &Arc<Pixels>, offset: (i32, i32), dirty: Option<IRect>) -> usize {
        let key = Arc::as_ptr(px) as usize;
        let size = (px.width(), px.height());
        let format = match **px {
            Pixels::U8(_) => wgpu::TextureFormat::Rgba8Unorm,
            _ => wgpu::TextureFormat::Rgba32Float,
        };
        let same = self.layers.get(&key).is_some_and(|t| {
            t.size == size && t.format == format && t.pixels.upgrade().is_some_and(|p| Arc::ptr_eq(&p, px))
        });
        let whole = IRect::from_size(size.0, size.1);
        if same && !self.force {
            let t = self.layers.get_mut(&key).unwrap();
            t.used = true;
            let changed = dirty.map(|d| IRect::new(d.x - offset.0, d.y - offset.1, d.w, d.h).intersect(&whole));
            if let Some(r) = changed.filter(|r| !r.is_empty()) {
                let tex = self.layers[&key].tex.clone();
                self.write(&tex, px, r);
            }
            return key;
        }
        let tex = self.texture("imged layer", size, format, 1, false);
        self.write(&tex, px, whole);
        let bind = self.tex_bind(&tex.create_view(&Default::default()));
        self.layers.insert(key, LayerTex { pixels: Arc::downgrade(px), tex, bind, size, format, used: true });
        key
    }

    fn render(&mut self, doc: &Document, display: &DisplaySettings, dirty: Option<IRect>) {
        let size = (doc.width.max(1), doc.height.max(1));
        self.ensure_targets(size);
        for t in self.layers.values_mut() {
            t.used = false;
        }

        // Parameters for every pass, then the passes.
        let canvas = [size.0 as f32, size.1 as f32];
        let mut params: Vec<Params> = Vec::new();
        let mut ops = Vec::new();
        for l in doc.layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
            let (px, offset) = match doc.preview.get(&l.id) {
                Some(p) => (&p.pixels, p.offset),
                None => (&l.pixels, l.offset),
            };
            let key = self.upload(px, offset, dirty);
            let rect = [offset.0 as f32, offset.1 as f32, px.width() as f32, px.height() as f32];
            let slot = params.len() as u32;
            if l.filter_on {
                let f = l.filter;
                if f.is_identity() {
                    continue;
                }
                let everywhere = imged_core::composite::is_blank(px);
                let a = f.adjust;
                let mut flags = 0u32;
                if !a.is_identity() {
                    flags |= 1;
                }
                if f.invert {
                    flags |= 2;
                }
                let (lo, hi) = f.levels.unwrap_or((0.0, 1.0));
                if f.levels.is_some() {
                    flags |= 4;
                }
                let (sa, sb) = f.swap.filter(|(a, b)| *a < 4 && *b < 4).unwrap_or((0, 0));
                if f.swap.is_some() && sa != sb {
                    flags |= 8;
                }
                params.push(Params {
                    rect,
                    canvas: [canvas[0], canvas[1], l.opacity, if everywhere { 1.0 } else { 2.0 }],
                    adj0: [a.exposure, a.brightness, a.contrast, a.gamma],
                    adj1: [a.hue, a.saturation, a.alpha, flags as f32],
                    extra: [lo, hi, sa as f32, sb as f32],
                });
                ops.push(Op::Filter { key, slot });
            } else {
                params.push(Params { rect, canvas: [canvas[0], canvas[1], l.opacity, 0.0], ..Default::default() });
                ops.push(Op::Layer { key, slot });
            }
        }
        let display_slot = params.len() as u32;
        let gain = 2f32.powf(display.exposure);
        params.push(Params {
            adj0: [gain, if display.srgb_encode { 1.0 } else { 0.0 }, 0.0, 0.0],
            ..Default::default()
        });
        self.layers.retain(|_, t| t.used);
        self.force = false;

        if params.len() as u32 > self.params_slots {
            self.params_slots = (params.len() as u32).next_power_of_two();
            self.params = params_buffer(&self.device, self.params_slots, self.align);
            self.params_bind = params_bind(&self.device, &self.pipes.params_layout, &self.params);
        }
        let mut bytes = vec![0u8; params.len() * self.align as usize];
        for (i, p) in params.iter().enumerate() {
            let at = i * self.align as usize;
            bytes[at..at + std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        self.queue.write_buffer(&self.params, 0, &bytes);

        let acc = self.acc.as_ref().unwrap();
        let out = self.out.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("imged canvas") });
        let pass = |enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, clear: bool| {
            let load = if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load };
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime()
        };
        let off = |slot: u32| slot * self.align;
        let mut cur = 0usize;
        let mut rp = Some(pass(&mut enc, &acc.views[0], true));
        for op in &ops {
            match *op {
                Op::Layer { key, slot } => {
                    if rp.is_none() {
                        rp = Some(pass(&mut enc, &acc.views[cur], false));
                    }
                    let r = rp.as_mut().unwrap();
                    r.set_pipeline(&self.pipes.layer);
                    r.set_bind_group(0, &self.params_bind, &[off(slot)]);
                    r.set_bind_group(1, &self.layers[&key].bind, &[]);
                    r.draw(0..6, 0..1);
                }
                Op::Filter { key, slot } => {
                    drop(rp.take());
                    let mut r = pass(&mut enc, &acc.views[1 - cur], true);
                    r.set_pipeline(&self.pipes.filter);
                    r.set_bind_group(0, &self.params_bind, &[off(slot)]);
                    r.set_bind_group(1, &self.layers[&key].bind, &[]);
                    r.set_bind_group(2, &acc.binds[cur], &[]);
                    r.draw(0..3, 0..1);
                    drop(r);
                    cur = 1 - cur;
                }
            }
        }
        drop(rp);
        {
            let mut r = pass(&mut enc, &out.levels[0], true);
            r.set_pipeline(&self.pipes.display);
            r.set_bind_group(0, &self.params_bind, &[off(display_slot)]);
            r.set_bind_group(2, &acc.binds[cur], &[]);
            r.draw(0..3, 0..1);
        }
        for l in 1..out.levels.len() {
            let mut r = pass(&mut enc, &out.levels[l], true);
            r.set_pipeline(&self.pipes.mip);
            r.set_bind_group(3, &out.mip_binds[l - 1], &[]);
            r.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        let _ = &out.full_view;
    }
}

impl CanvasRenderer for GpuRenderer {
    fn sync(&mut self, ctx: &egui::Context, doc: &mut Document, display: &DisplaySettings) {
        if !self.fits(doc) {
            doc.take_dirty_unclipped();
            self.cpu.get_or_insert_with(|| {
                // A fresh renderer draws everything.
                doc.mark_all_dirty();
                CpuTiledRenderer::default()
            });
            self.cpu.as_mut().unwrap().sync(ctx, doc, display);
            self.force = true;
            return;
        }
        if self.cpu.take().is_some() {
            self.force = true;
        }
        doc.take_dirty();
        let dirty = doc.take_dirty_unclipped();
        let size = (doc.width, doc.height);
        if dirty.is_none() && !self.force && *display == self.display && size == self.size {
            return;
        }
        self.display = *display;
        self.size = size;
        self.render(doc, display, dirty);
    }

    fn paint(&self, painter: &Painter, view: &ViewXform) {
        if let Some(cpu) = &self.cpu {
            return cpu.paint(painter, view);
        }
        let Some(id) = self.tex_id else { return };
        let r = view.rect_to_screen(IRect::from_size(self.size.0, self.size.1));
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        painter.image(id, r, uv, Color32::WHITE);
    }

    fn name(&self) -> String {
        if self.cpu.is_some() {
            format!("CPU (too large for the GPU: max {} px)", self.max_dim)
        } else {
            format!("GPU: {}", self.adapter)
        }
    }
}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        if let Some(id) = self.tex_id.take() {
            self.egui.write().free_texture(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imged_core::ops::filters::{Adjustments, LayerFilter};
    use imged_core::{Buffer, Depth};

    /// A headless wgpu device (any adapter, software ones included).
    fn render_state() -> Option<egui_wgpu::RenderState> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
        let renderer = egui_wgpu::Renderer::new(&device, OUT_FORMAT, Default::default());
        Some(egui_wgpu::RenderState {
            adapter,
            available_adapters: Vec::new(),
            instance,
            device,
            queue,
            target_format: OUT_FORMAT,
            renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        })
    }

    fn read_output(r: &GpuRenderer) -> Vec<u8> {
        let out = r.out.as_ref().unwrap();
        let (w, h) = out.size;
        let row = (w * 4).div_ceil(256) * 256;
        let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = r.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            out.tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        r.queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        r.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let data = buf.slice(..).get_mapped_range().unwrap();
        (0..h).flat_map(|y| data[(y * row) as usize..(y * row + w * 4) as usize].to_vec()).collect()
    }

    fn buffer<T: imged_core::Channel>(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Buffer<T> {
        let mut b = Buffer::<T>::new(w, h);
        for y in 0..h {
            for x in 0..w {
                b.set(x, y, f(x, y));
            }
        }
        b
    }

    /// Every kind of layer: 8/16-bit and float (HDR), offsets, opacity,
    /// a masked filter layer and one that applies everywhere.
    fn test_doc() -> Document {
        let (w, h) = (64, 48);
        let bg = buffer::<u8>(w, h, |x, y| [x as f32 / 63.0, y as f32 / 47.0, 0.4, if x < 6 { 0.0 } else { 1.0 }]);
        let mut doc = Document::from_pixels(Pixels::U8(bg), "bg");
        let a = buffer::<u16>(30, 20, |x, y| [0.9, x as f32 / 29.0, 0.2, y as f32 / 19.0]);
        let id = doc.add_layer_from("u16", Pixels::U16(a), (10, 5));
        doc.layer_mut(id).unwrap().opacity = 0.6;
        let f = buffer::<f32>(40, 20, |x, _| [x as f32 / 26.0, 0.3, 1.2, 0.8]);
        doc.add_layer_from("f32", Pixels::F32(f), (-5, 20));
        let mask = buffer::<u8>(24, 24, |x, y| {
            let d = ((x as f32 - 12.0).powi(2) + (y as f32 - 12.0).powi(2)).sqrt();
            [0.0, 0.0, 0.0, (1.0 - d / 12.0).clamp(0.0, 1.0)]
        });
        let m = doc.add_layer_from("mask", Pixels::U8(mask), (30, 10));
        doc.set_filter_on(m, true);
        let adjust = Adjustments { hue: 40.0, saturation: 1.3, ..Default::default() };
        doc.set_layer_filter(m, LayerFilter { adjust, invert: true, ..Default::default() });
        let top = buffer::<u8>(10, 10, |x, _| [0.1, 0.8, 0.9, x as f32 / 9.0]);
        doc.add_layer_from("top", Pixels::U8(top), (50, 35));
        let all = doc.add_layer_from("all", Pixels::new(Depth::U8, 1, 1), (0, 0));
        doc.set_filter_on(all, true);
        let adjust =
            Adjustments { exposure: 0.5, brightness: -0.05, contrast: 1.2, gamma: 1.4, alpha: -0.1, ..Default::default() };
        doc.set_layer_filter(all, LayerFilter { adjust, levels: Some((0.1, 0.9)), swap: Some((0, 2)), invert: false });
        doc.layer_mut(all).unwrap().opacity = 0.7;
        doc
    }

    #[test]
    fn gpu_matches_cpu() {
        let Some(rs) = render_state() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let mut doc = test_doc();
        let mut r = GpuRenderer::new(&rs);
        let ctx = egui::Context::default();
        for display in [
            DisplaySettings::default(),
            DisplaySettings { exposure: 0.3, srgb_encode: true, checkerboard: true },
        ] {
            r.sync(&ctx, &mut doc, &display);
            let gpu = read_output(&r);
            let cpu = CpuTiledRenderer::render_tile(&doc, doc.canvas_rect(), &display);
            let cpu: Vec<u8> = cpu.pixels.iter().flat_map(|c| c.to_array()).collect();
            let worst = gpu.iter().zip(&cpu).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            eprintln!("{}: max difference {worst}/255", r.adapter);
            assert!(worst <= 3, "GPU and CPU differ by up to {worst}/255");
        }
        // An edit inside a layer is uploaded (only the changed part).
        let id = doc.layers[1].id;
        let seq = doc.next_seq();
        doc.layer_mut(id).unwrap().edit_region(seq, "t", imged_core::Rect::new(0, 0, 8, 8), |p| {
            for y in 0..8 {
                for x in 0..8 {
                    p.set(x, y, [0.0, 0.0, 1.0, 1.0]);
                }
            }
        });
        doc.mark_dirty(imged_core::Rect::new(10, 5, 8, 8));
        let display = DisplaySettings::default();
        r.sync(&ctx, &mut doc, &display);
        let gpu = read_output(&r);
        let cpu = CpuTiledRenderer::render_tile(&doc, doc.canvas_rect(), &display);
        let cpu: Vec<u8> = cpu.pixels.iter().flat_map(|c| c.to_array()).collect();
        let worst = gpu.iter().zip(&cpu).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(worst <= 3, "after an edit, GPU and CPU differ by up to {worst}/255");
    }
}
