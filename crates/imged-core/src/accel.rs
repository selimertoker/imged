//! Hooks for running heavy operations elsewhere (the GUI registers a GPU
//! implementation). The core stays free of GPU code: an operation asks
//! [`get`] for an accelerator and falls back to its own CPU code when there
//! is none, or when the accelerator declines (returns `None`, e.g. because
//! the image is too large for the device).
//!
//! Accelerated today: convolution and resizing, the operations doing enough
//! work per pixel to be worth sending the pixels to the GPU and back.
//! Per-pixel color filters, flips and warps are cheaper on the CPU (rayon)
//! than the round trip.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use crate::ops::convolve::Kernel;
use crate::pixel::Channel;

/// Pixels handed to an accelerator: interleaved RGBA, straight alpha.
pub enum ImageData<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

/// An accelerator's result, in the depth it was given.
pub enum ImageOut {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

/// One axis of a separable resize: for each output pixel, the first source
/// pixel and its weights (`starts[i]..`, `weights[offsets[i]..offsets[i + 1]]`).
pub struct AxisWeights {
    pub starts: Vec<u32>,
    pub offsets: Vec<u32>,
    pub weights: Vec<f32>,
}

pub trait Accelerator: Send + Sync {
    /// What runs the operations (for the UI).
    fn name(&self) -> String;
    /// `convolve` (see [`crate::ops::convolve::convolve`]) of a `w`×`h` image.
    fn convolve(&self, w: u32, h: u32, src: ImageData, k: &Kernel) -> Option<ImageOut>;
    /// Separable resize in premultiplied space (see
    /// [`crate::ops::transform::resize`]): rows with `horiz`, then columns
    /// with `vert`, giving an image `horiz.starts.len()` × `vert.starts.len()`.
    fn resize(&self, w: u32, h: u32, src: ImageData, horiz: &AxisWeights, vert: &AxisWeights) -> Option<ImageOut>;
}

static ACCEL: OnceLock<Box<dyn Accelerator>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(true);

/// Registers the accelerator (once; later calls are ignored).
pub fn register(a: Box<dyn Accelerator>) {
    ACCEL.set(a).ok();
}

/// Turns the accelerator on or off (it stays registered).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// The registered accelerator, when enabled.
pub fn get() -> Option<&'static dyn Accelerator> {
    ACCEL.get().filter(|_| ENABLED.load(Ordering::Relaxed)).map(|b| b.as_ref())
}

/// Whether an accelerator is registered (enabled or not).
pub fn available() -> Option<String> {
    ACCEL.get().map(|a| a.name())
}

/// Views a channel buffer as [`ImageData`]. (Takes the `Vec` itself: that
/// is the type the downcast can see.)
#[allow(clippy::ptr_arg)]
pub fn data_of<T: Channel>(data: &Vec<T>) -> ImageData<'_> {
    let any = data as &dyn Any;
    if let Some(d) = any.downcast_ref::<Vec<u8>>() {
        ImageData::U8(d)
    } else if let Some(d) = any.downcast_ref::<Vec<u16>>() {
        ImageData::U16(d)
    } else {
        ImageData::F32(any.downcast_ref::<Vec<f32>>().expect("u8, u16 or f32 channels"))
    }
}

/// Takes an accelerator result back as channel values (`None` if the depth
/// doesn't match).
pub fn out_as<T: Channel>(out: ImageOut) -> Option<Vec<T>> {
    let b: Box<dyn Any> = match out {
        ImageOut::U8(v) => Box::new(v),
        ImageOut::U16(v) => Box::new(v),
        ImageOut::F32(v) => Box::new(v),
    };
    b.downcast::<Vec<T>>().ok().map(|v| *v)
}
