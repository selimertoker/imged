//! Number fields grab the mouse while dragged: the cursor disappears and
//! stays inside the window however far the mouse moves, until the button is
//! released; then it reappears where the drag started.
//!
//! How: number fields are added through [`drag_value`], which marks the
//! field being dragged. While it is, the window grabs the pointer (locked on
//! Wayland; confined and warped back to the start every frame on X11, which
//! can't lock), and [`CursorGrab::filter_input`] replaces the real pointer
//! positions with a virtual position moved by the raw mouse motion, so the
//! value keeps changing even when the real pointer can't move any further.

use eframe::egui::{self, Context, CursorIcon, Event, Id, Pos2, RawInput, Response, Ui, ViewportCommand};

fn marker() -> Id {
    Id::unique("imged-grab-drag-value")
}

/// Adds a number field that grabs the mouse while it is dragged.
pub fn drag_value(ui: &mut Ui, w: egui::DragValue<'_>) -> Response {
    let r = ui.add(w);
    mark(ui, &r);
    r
}

/// Makes an already added number field grab the mouse while dragged.
pub fn mark(ui: &Ui, r: &Response) {
    if r.dragged() {
        ui.ctx().data_mut(|d| d.insert_temp(marker(), r.id));
    }
}

#[derive(Default)]
pub struct CursorGrab {
    active: bool,
    /// Where the drag started (points); the cursor comes back here.
    start: Pos2,
    /// The pointer position egui sees while grabbing.
    virt: Pos2,
    /// The button was released during this frame's input.
    released: bool,
}

fn wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
}

impl CursorGrab {
    /// Called from `raw_input_hook`, before egui sees the input.
    pub fn filter_input(&mut self, ctx: &Context, raw: &mut RawInput) {
        if !self.active {
            return;
        }
        let ppp = ctx.pixels_per_point().max(0.1);
        let mut out = Vec::with_capacity(raw.events.len());
        for ev in raw.events.drain(..) {
            match ev {
                Event::MouseMoved(d) => {
                    self.virt += d / ppp;
                    out.push(Event::PointerMoved(self.virt));
                }
                Event::PointerMoved(_) | Event::PointerGone => {}
                Event::PointerButton { button, pressed, modifiers, .. } => {
                    out.push(Event::PointerButton { pos: self.virt, button, pressed, modifiers });
                    if !pressed && button == egui::PointerButton::Primary {
                        self.released = true;
                        // Back to where the drag started.
                        out.push(Event::PointerMoved(self.start));
                    }
                }
                other => out.push(other),
            }
        }
        raw.events = out;
    }

    /// Called at the end of each frame: starts or ends the grab.
    pub fn end_frame(&mut self, ctx: &Context) {
        let marked: Option<Id> = ctx.data(|d| d.get_temp(marker()));
        let dragging = marked.is_some() && ctx.dragged_id() == marked;
        let down = ctx.input(|i| i.pointer.primary_down());
        if !self.active && dragging && down && !self.released {
            let Some(p) = ctx.input(|i| i.pointer.interact_pos()) else { return };
            self.active = true;
            self.start = p;
            self.virt = p;
            let mode = if wayland() { egui::viewport::CursorGrab::Locked } else { egui::viewport::CursorGrab::Confined };
            ctx.send_viewport_cmd(ViewportCommand::CursorGrab(mode));
        }
        if self.active && (self.released || !dragging || !down) {
            self.active = false;
            ctx.send_viewport_cmd(ViewportCommand::CursorGrab(egui::viewport::CursorGrab::None));
            ctx.send_viewport_cmd(ViewportCommand::CursorPosition(self.start));
            ctx.data_mut(|d| d.remove::<Id>(marker()));
        }
        self.released = false;
        if self.active {
            ctx.set_cursor_icon(CursorIcon::None);
            if !wayland() {
                // X11 can't lock the pointer: keep it from reaching the window's edge.
                ctx.send_viewport_cmd(ViewportCommand::CursorPosition(self.start));
            }
        }
        if !dragging && marked.is_some() && !self.active {
            ctx.data_mut(|d| d.remove::<Id>(marker()));
        }
    }
}
