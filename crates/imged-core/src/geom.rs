//! Small geometry helpers: integer rectangles and float points.

/// Integer rectangle. `x`/`y` may be negative (e.g. layers hanging off the canvas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Rect { x, y, w, h }
    }

    pub fn from_size(w: u32, h: u32) -> Self {
        Rect::new(0, 0, w, h)
    }

    /// Builds a rectangle from two corner points (inclusive-exclusive, any order).
    pub fn from_corners(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        let (ax, bx) = (x0.min(x1), x0.max(x1));
        let (ay, by) = (y0.min(y1), y0.max(y1));
        Rect::new(ax, ay, (bx - ax) as u32, (by - ay) as u32)
    }

    /// Smallest integer rectangle covering the float box `[x0,x1] x [y0,y1]`.
    pub fn covering(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        let ax = x0.min(x1).floor() as i32;
        let ay = y0.min(y1).floor() as i32;
        let bx = x0.max(x1).ceil() as i32;
        let by = y0.max(y1).ceil() as i32;
        Rect::from_corners(ax, ay, bx, by)
    }

    pub fn right(&self) -> i32 {
        self.x + self.w as i32
    }

    pub fn bottom(&self) -> i32 {
        self.y + self.h as i32
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }

    pub fn intersect(&self, o: &Rect) -> Rect {
        let x0 = self.x.max(o.x);
        let y0 = self.y.max(o.y);
        let x1 = self.right().min(o.right());
        let y1 = self.bottom().min(o.bottom());
        if x1 <= x0 || y1 <= y0 {
            Rect::new(x0, y0, 0, 0)
        } else {
            Rect::from_corners(x0, y0, x1, y1)
        }
    }

    pub fn union(&self, o: &Rect) -> Rect {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        Rect::from_corners(
            self.x.min(o.x),
            self.y.min(o.y),
            self.right().max(o.right()),
            self.bottom().max(o.bottom()),
        )
    }

    pub fn translate(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(self.x + dx, self.y + dy, self.w, self.h)
    }

    pub fn expand(&self, by: i32) -> Rect {
        Rect::from_corners(self.x - by, self.y - by, self.right() + by, self.bottom() + by)
    }
}

/// 2D float point / vector in image pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    pub fn normalized(self) -> Vec2 {
        let l = self.len();
        if l < 1e-9 {
            Vec2::ZERO
        } else {
            Vec2::new(self.x / l, self.y / l)
        }
    }

    pub fn perp(self) -> Vec2 {
        Vec2::new(-self.y, self.x)
    }

    pub fn rotate(self, radians: f32) -> Vec2 {
        let (s, c) = radians.sin_cos();
        Vec2::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}
impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}
impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f32) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}
impl std::ops::Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}
