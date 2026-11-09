//! "Marching ants": the animated outline of the selection.

use std::sync::Arc;

use eframe::egui::{self, Color32, Painter, Pos2, Stroke};
use imged_core::Mask;

use crate::view::ViewXform;

/// More boundary runs than this are drawn as the selection's bounding box.
pub const MAX_RUNS: usize = 40_000;

#[derive(Default)]
pub struct Ants {
    /// The mask the runs were computed for (by pointer identity).
    key: Option<usize>,
    /// Boundary runs in canvas coordinates: (x0, y0, x1, y1), horizontal or vertical.
    runs: Vec<(i32, i32, i32, i32)>,
}

fn inside(m: &Mask, x: i32, y: i32) -> bool {
    m.get(x, y) >= 0.5
}

/// The edges between selected and unselected pixels, merged into runs.
pub fn boundary(m: &Mask) -> Vec<(i32, i32, i32, i32)> {
    let r = m.rect;
    let mut runs = Vec::new();
    // Horizontal edges: between rows y-1 and y.
    for y in r.y..=r.bottom() {
        let mut start = None;
        for x in r.x..=r.right() {
            let edge = x < r.right() && inside(m, x, y - 1) != inside(m, x, y);
            match (edge, start) {
                (true, None) => start = Some(x),
                (false, Some(s)) => {
                    runs.push((s, y, x, y));
                    start = None;
                }
                _ => {}
            }
        }
    }
    // Vertical edges: between columns x-1 and x.
    for x in r.x..=r.right() {
        let mut start = None;
        for y in r.y..=r.bottom() {
            let edge = y < r.bottom() && inside(m, x - 1, y) != inside(m, x, y);
            match (edge, start) {
                (true, None) => start = Some(y),
                (false, Some(s)) => {
                    runs.push((x, s, x, y));
                    start = None;
                }
                _ => {}
            }
        }
        if runs.len() > MAX_RUNS {
            break;
        }
    }
    runs
}

impl Ants {
    pub fn paint(&mut self, painter: &Painter, view: &ViewXform, sel: &Arc<Mask>, time: f64) {
        let key = Arc::as_ptr(sel) as usize;
        if self.key != Some(key) {
            self.key = Some(key);
            self.runs = boundary(sel);
        }
        let clip = painter.clip_rect();
        let offset = (time * 16.0) as f32 % 8.0;
        let black = Stroke::new(1.0, Color32::BLACK);
        let white = Stroke::new(1.0, Color32::WHITE);
        let draw = |a: Pos2, b: Pos2| {
            let seg = egui::Rect::from_two_pos(a, b).expand(1.0);
            if !clip.intersects(seg) {
                return;
            }
            painter.line_segment([a, b], black);
            painter.extend(egui::Shape::dashed_line_with_offset(&[a, b], white, &[4.0], &[4.0], offset));
        };
        if self.runs.len() > MAX_RUNS {
            if let Some(b) = sel.bounds() {
                let r = view.rect_to_screen(b);
                let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
                for w in pts.windows(2) {
                    draw(w[0], w[1]);
                }
            }
            return;
        }
        let to = |x: i32, y: i32| view.to_screen(imged_core::Vec2::new(x as f32, y as f32));
        for &(x0, y0, x1, y1) in &self.runs {
            draw(to(x0, y0), to(x1, y1));
        }
    }
}
