//! imged-core: the headless part of imged.
//!
//! Everything that edits pixels lives here, with no GUI dependency, so the
//! same code can back the GUI, a future CLI and future GPU paths:
//!
//! * [`pixel`], [`buffer`], [`pixels`] — storage generic over the channel type
//!   (`u8`, `u16`, `f32`), with [`Pixels`] picking the depth at runtime.
//! * [`layer`], [`document`] — layer stack, groups, per-layer and
//!   structural undo history.
//! * [`ops`] — image operations (filters, convolution, transforms, painting,
//!   fills, selections, edge snapping, text).
//! * [`command`] — a serializable-ish [`Command`] enum: the single entry
//!   point that GUI buttons, keybindings and (later) the CLI all go through.
//! * [`io`] — loading and saving, including PFM.

pub mod accel;
pub mod buffer;
pub mod color;
pub mod command;
pub mod composite;
pub mod document;
pub mod geom;
pub mod io;
pub mod layer;
pub mod mask;
pub mod object;
pub mod ops;
pub mod pixel;
pub mod pixels;

pub use buffer::Buffer;
pub use color::Color;
pub use command::Command;
pub use document::Document;
pub use geom::{Rect, Vec2};
pub use layer::{Layer, LayerId};
pub use mask::Mask;
pub use pixel::{Channel, Depth, Px};
pub use pixels::Pixels;
