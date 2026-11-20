//! GPU accelerator for imged-core (see `imged_core::accel`): convolution
//! and resizing as compute shaders (gpu_filters.wgsl), on the window's wgpu
//! device.
//!
//! The pixels are sent in their own depth (8-bit images as 4 bytes per
//! pixel) and come back the same way, so the copies stay as small as the
//! image itself. Images larger than the device's buffer limits are left to
//! the CPU.

use std::sync::Mutex;

use eframe::egui_wgpu::{self, wgpu};
use imged_core::accel::{Accelerator, AxisWeights, ImageData, ImageOut};
use imged_core::ops::convolve::{EdgeMode, Kernel};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    w: u32,
    h: u32,
    depth: u32,
    ntaps: u32,
    edge: u32,
    alpha: u32,
    nw: u32,
    nh: u32,
    inv: f32,
    bias: f32,
    v_idx: u32,
    v_wts: u32,
}

pub struct GpuFilters {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    convolve: wgpu::ComputePipeline,
    resize_rows: wgpu::ComputePipeline,
    resize_cols: wgpu::ComputePipeline,
    limits: wgpu::Limits,
    adapter: String,
    /// One operation at a time (they share the queue and wait for it).
    lock: Mutex<()>,
}

/// Registers the GPU accelerator when the device can run compute shaders.
pub fn register(rs: &egui_wgpu::RenderState) {
    if let Some(g) = GpuFilters::new(&rs.adapter, &rs.device, &rs.queue) {
        imged_core::accel::register(Box::new(g));
    }
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// u32 words per pixel for a depth (0 u8, 1 u16, 2 f32).
fn words(depth: u32) -> u64 {
    [1, 2, 4][depth as usize]
}

fn upload_words<'a>(src: &ImageData<'a>) -> (u32, &'a [u8]) {
    match src {
        ImageData::U8(d) => (0, bytemuck::cast_slice(d)),
        ImageData::U16(d) => (1, bytemuck::cast_slice(d)),
        ImageData::F32(d) => (2, bytemuck::cast_slice(d)),
    }
}

fn output(depth: u32, bytes: &[u8]) -> ImageOut {
    match depth {
        0 => ImageOut::U8(bytes.to_vec()),
        1 => ImageOut::U16(bytemuck::pod_collect_to_vec(bytes)),
        _ => ImageOut::F32(bytemuck::pod_collect_to_vec(bytes)),
    }
}

impl GpuFilters {
    pub fn new(adapter: &wgpu::Adapter, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let limits = device.limits();
        let caps = adapter.get_downlevel_capabilities();
        if !caps.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            || limits.max_storage_buffers_per_shader_stage < 5
            || limits.max_compute_invocations_per_workgroup < 256
        {
            return None;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("imged filters"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_filters.wgsl").into()),
        });
        use wgpu::BufferBindingType as B;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("imged filters"),
            entries: &[
                entry(0, B::Uniform),
                entry(1, B::Storage { read_only: true }),
                entry(2, B::Storage { read_only: false }),
                entry(3, B::Storage { read_only: false }),
                entry(4, B::Storage { read_only: true }),
                entry(5, B::Storage { read_only: true }),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("imged filters"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = |e: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let info = adapter.get_info();
        Some(GpuFilters {
            convolve: pipe("convolve"),
            resize_rows: pipe("resize_rows"),
            resize_cols: pipe("resize_cols"),
            layout,
            limits,
            adapter: format!("{} ({:?})", info.name, info.backend),
            device: device.clone(),
            queue: queue.clone(),
            lock: Mutex::new(()),
        })
    }

    /// Whether a buffer of `bytes` fits the device.
    fn fits(&self, bytes: u64) -> bool {
        bytes <= self.limits.max_storage_buffer_binding_size && bytes <= self.limits.max_buffer_size
    }

    fn buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: size.max(16).next_multiple_of(4), usage, mapped_at_creation: false })
    }

    fn init_buffer(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let b = self.buffer(data.len() as u64, usage | wgpu::BufferUsages::COPY_DST);
        if !data.is_empty() {
            if data.len().is_multiple_of(4) {
                self.queue.write_buffer(&b, 0, data);
            } else {
                let mut padded = data.to_vec();
                padded.resize(data.len().next_multiple_of(4), 0);
                self.queue.write_buffer(&b, 0, &padded);
            }
        }
        b
    }

    /// Runs `passes` (pipeline, workgroups) over the bindings and reads back
    /// `out_bytes` of the output.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        u: Uniforms,
        src: &[u8],
        out_bytes: u64,
        mid_bytes: u64,
        idx: &[u32],
        wts: &[f32],
        passes: &[(&wgpu::ComputePipeline, [u32; 2])],
    ) -> Option<Vec<u8>> {
        use wgpu::BufferUsages as B;
        if !self.fits(src.len() as u64) || !self.fits(out_bytes) || !self.fits(mid_bytes) {
            return None;
        }
        let max = self.limits.max_compute_workgroups_per_dimension;
        if passes.iter().any(|(_, [x, y])| *x > max || *y > max) {
            return None;
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let ubuf = self.init_buffer(bytemuck::bytes_of(&u), B::UNIFORM);
        let sbuf = self.init_buffer(src, B::STORAGE);
        let dbuf = self.buffer(out_bytes, B::STORAGE | B::COPY_SRC);
        let mbuf = self.buffer(mid_bytes, B::STORAGE);
        let ibuf = self.init_buffer(bytemuck::cast_slice(idx), B::STORAGE);
        let wbuf = self.init_buffer(bytemuck::cast_slice(wts), B::STORAGE);
        let read = self.buffer(out_bytes, B::MAP_READ | B::COPY_DST);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ibuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wbuf.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_bind_group(0, &bind, &[]);
            for (pipe, [x, y]) in passes {
                p.set_pipeline(pipe);
                p.dispatch_workgroups((*x).max(1), (*y).max(1), 1);
            }
        }
        let n = out_bytes.next_multiple_of(4);
        enc.copy_buffer_to_buffer(&dbuf, 0, &read, 0, n);
        self.queue.submit([enc.finish()]);
        let slice = read.slice(..n);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok()?;
        let data = slice.get_mapped_range().ok()?[..out_bytes as usize].to_vec();
        read.unmap();
        Some(data)
    }
}

impl Accelerator for GpuFilters {
    fn name(&self) -> String {
        format!("GPU: {}", self.adapter)
    }

    fn convolve(&self, w: u32, h: u32, src: ImageData, k: &Kernel) -> Option<ImageOut> {
        let (depth, bytes) = upload_words(&src);
        let r = (k.size / 2) as i32;
        let taps: Vec<f32> = (0..k.size * k.size)
            .filter(|i| k.weights[*i] != 0.0)
            .flat_map(|i| [((i % k.size) as i32 - r) as f32, ((i / k.size) as i32 - r) as f32, k.weights[i]])
            .collect();
        let u = Uniforms {
            w,
            h,
            depth,
            ntaps: (taps.len() / 3) as u32,
            edge: match k.edge {
                EdgeMode::Clamp => 0,
                EdgeMode::Transparent => 1,
                EdgeMode::Wrap => 2,
            },
            alpha: k.include_alpha as u32,
            inv: 1.0 / if k.divisor.abs() < 1e-9 { 1.0 } else { k.divisor },
            bias: k.bias,
            ..Default::default()
        };
        let out = w as u64 * h as u64 * words(depth) * 4;
        let data = self.run(u, bytes, out, 16, &[0], &taps, &[(&self.convolve, [w.div_ceil(16), h.div_ceil(16)])])?;
        Some(output(depth, &data))
    }

    fn resize(&self, w: u32, h: u32, src: ImageData, horiz: &AxisWeights, vert: &AxisWeights) -> Option<ImageOut> {
        let (depth, bytes) = upload_words(&src);
        let (nw, nh) = (horiz.starts.len() as u32, vert.starts.len() as u32);
        let mut idx = Vec::with_capacity((nw + nh) as usize * 2 + 2);
        idx.extend(&horiz.starts);
        idx.extend(&horiz.offsets);
        let v_idx = idx.len() as u32;
        idx.extend(&vert.starts);
        idx.extend(&vert.offsets);
        let mut wts = horiz.weights.clone();
        let v_wts = wts.len() as u32;
        wts.extend(&vert.weights);
        let u = Uniforms { w, h, depth, nw, nh, v_idx, v_wts, ..Default::default() };
        let out = nw as u64 * nh as u64 * words(depth) * 4;
        let mid = nw as u64 * h as u64 * 16;
        let passes = [
            (&self.resize_rows, [nw.div_ceil(16), h.div_ceil(16)]),
            (&self.resize_cols, [nw.div_ceil(16), nh.div_ceil(16)]),
        ];
        let data = self.run(u, bytes, out, mid, &idx, &wts, &passes)?;
        Some(output(depth, &data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imged_core::ops::convolve::convolve_cpu;
    use imged_core::ops::transform::{resize_cpu, ResizeFilter};
    use imged_core::{Buffer, Channel};

    fn gpu() -> Option<GpuFilters> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let desc = wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;
        GpuFilters::new(&adapter, &device, &queue)
    }

    fn image<T: Channel>(w: u32, h: u32) -> Buffer<T> {
        Buffer::from_fn(w, h, |x, y| {
            let a = if (x / 5 + y / 3) % 4 == 0 { 0.0 } else { ((x * 7 + y * 3) % 11) as f32 / 10.0 };
            [(x % 13) as f32 / 12.0, (y % 7) as f32 / 6.0, ((x ^ y) % 5) as f32 / 4.0, a]
        })
    }

    fn max_diff<T: Channel>(a: &Buffer<T>, b: &Buffer<T>) -> f32 {
        assert_eq!((a.width, a.height), (b.width, b.height));
        a.data.iter().zip(&b.data).map(|(x, y)| (x.to_f32() - y.to_f32()).abs()).fold(0.0, f32::max)
    }

    fn check<T: Channel>(g: &GpuFilters, tol: f32) {
        let src = image::<T>(53, 37);
        for (name, mut k) in Kernel::presets(5) {
            for edge in [EdgeMode::Clamp, EdgeMode::Transparent, EdgeMode::Wrap] {
                k.edge = edge;
                let cpu = convolve_cpu(&src, &k);
                let out = g.convolve(53, 37, imged_core::accel::data_of(&src.data), &k).unwrap();
                let gpu = Buffer { width: 53, height: 37, data: imged_core::accel::out_as::<T>(out).unwrap() };
                let d = max_diff(&cpu, &gpu);
                assert!(d <= tol, "{name} {edge:?} {:?}: differs by {d}", T::DEPTH);
            }
        }
        for (nw, nh) in [(20, 70), (106, 12), (53, 37)] {
            for f in [ResizeFilter::Bilinear, ResizeFilter::Bicubic, ResizeFilter::Lanczos3] {
                let cpu = resize_cpu(&src, nw, nh, f);
                let horiz = pack(imged_core::ops::transform::axis_weights(53, nw as usize, f));
                let vert = pack(imged_core::ops::transform::axis_weights(37, nh as usize, f));
                let out = g.resize(53, 37, imged_core::accel::data_of(&src.data), &horiz, &vert).unwrap();
                let gpu = Buffer { width: nw, height: nh, data: imged_core::accel::out_as::<T>(out).unwrap() };
                let d = max_diff(&cpu, &gpu);
                assert!(d <= tol, "resize {nw}x{nh} {f:?} {:?}: differs by {d}", T::DEPTH);
            }
        }
    }

    fn pack(ws: Vec<(usize, Vec<f32>)>) -> AxisWeights {
        let mut aw = AxisWeights { starts: Vec::new(), offsets: vec![0], weights: Vec::new() };
        for (lo, w) in ws {
            aw.starts.push(lo as u32);
            aw.weights.extend(w);
            aw.offsets.push(aw.weights.len() as u32);
        }
        aw
    }

    /// Timing on 4000x3000 (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn bench_filters() {
        let Some(g) = gpu() else { return };
        let src = image::<u8>(4000, 3000);
        let k = Kernel::presets(9).into_iter().find(|(n, _)| n.starts_with("Gaussian")).unwrap().1;
        let t = std::time::Instant::now();
        g.convolve(4000, 3000, imged_core::accel::data_of(&src.data), &k).unwrap();
        eprintln!("9x9 convolution: GPU {:?}", t.elapsed());
        let t = std::time::Instant::now();
        convolve_cpu(&src, &k);
        eprintln!("9x9 convolution: CPU {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let f = ResizeFilter::Lanczos3;
        let (horiz, vert) = (pack(imged_core::ops::transform::axis_weights(4000, 1500, f)), pack(imged_core::ops::transform::axis_weights(3000, 1100, f)));
        g.resize(4000, 3000, imged_core::accel::data_of(&src.data), &horiz, &vert).unwrap();
        eprintln!("Lanczos resize: GPU {:?}", t.elapsed());
        let t = std::time::Instant::now();
        resize_cpu(&src, 1500, 1100, f);
        eprintln!("Lanczos resize: CPU {:?}", t.elapsed());
    }

    #[test]
    fn gpu_filters_match_cpu() {
        let Some(g) = gpu() else {
            eprintln!("no compute-capable adapter; skipped");
            return;
        };
        // One 8-bit level (rounding at a .5 boundary); float rounding otherwise.
        check::<u8>(&g, 1.0 / 255.0 + 1e-6);
        check::<u16>(&g, 2.0 / 65535.0);
        check::<f32>(&g, 1e-4);
    }
}
