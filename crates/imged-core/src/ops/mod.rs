//! Image operations. Each submodule is generic over the channel type where
//! it touches pixels, so every layer depth gets the same behavior.

pub mod convolve;
pub mod edges;
pub mod fill;
pub mod filters;
pub mod layout;
pub mod outline;
pub mod paint;
pub mod stitch;
pub mod text;
pub mod transform;
