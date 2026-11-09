// GPU compositing for the canvas (see renderer_gpu.rs).
//
// Layers are composited into a premultiplied Rgba16Float canvas with
// "over" blending, which gives the same result as the CPU compositor's
// straight-alpha `over`. Filter layers read the result so far and write the
// filtered version into the other of two canvas textures. The display pass
// applies the view exposure / sRGB encoding and writes premultiplied 8-bit
// values, exactly like the CPU renderer's Color32 tiles.

struct Params {
    // x, y, w, h of the layer (canvas pixels)
    rect: vec4<f32>,
    // canvas width, height, layer opacity, mode (0 pixels, 1 filter everywhere, 2 filter masked by the layer's alpha)
    canvas: vec4<f32>,
    // exposure, brightness, contrast, gamma  (display pass: gain, sRGB encode)
    adj0: vec4<f32>,
    // hue, saturation, alpha offset, flags (1 adjust, 2 invert, 4 levels, 8 swap)
    adj1: vec4<f32>,
    // levels low, high, swap channel a, swap channel b
    extra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(1) @binding(0) var layer_tex: texture_2d<f32>;
@group(2) @binding(0) var acc_tex: texture_2d<f32>;
@group(3) @binding(0) var mip_src: texture_2d<f32>;
@group(3) @binding(1) var mip_sampler: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
};

// A quad over the layer's rectangle.
@vertex
fn vs_quad(@builtin(vertex_index) i: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let px = P.rect.xy + corners[i] * P.rect.zw;
    var o: VOut;
    o.pos = vec4<f32>(px.x / P.canvas.x * 2.0 - 1.0, 1.0 - px.y / P.canvas.y * 2.0, 0.0, 1.0);
    return o;
}

// A triangle covering the whole target.
@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VOut;
    o.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return o;
}

fn inside(p: vec2<i32>, dims: vec2<i32>) -> bool {
    return p.x >= 0 && p.y >= 0 && p.x < dims.x && p.y < dims.y;
}

@fragment
fn fs_layer(v: VOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(v.pos.xy - P.rect.xy));
    if !inside(p, vec2<i32>(textureDimensions(layer_tex))) {
        discard;
    }
    let s = textureLoad(layer_tex, p, 0);
    let a = clamp(s.a * P.canvas.z, 0.0, 1.0);
    if a <= 0.0 {
        discard;
    }
    return vec4<f32>(s.rgb * a, a);
}

fn unpremultiply(c: vec4<f32>) -> vec4<f32> {
    if c.a <= 1e-8 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

fn rem_euclid(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    var h = 0.0;
    if d > 1e-9 {
        if mx == c.r {
            h = 60.0 * rem_euclid((c.g - c.b) / d, 6.0);
        } else if mx == c.g {
            h = 60.0 * ((c.b - c.r) / d + 2.0);
        } else {
            h = 60.0 * ((c.r - c.g) / d + 4.0);
        }
    }
    var s = 0.0;
    if mx > 1e-9 {
        s = d / mx;
    }
    return vec3<f32>(h, s, mx);
}

fn hsv_to_rgb(hue: f32, s: f32, v: f32) -> vec3<f32> {
    let h = rem_euclid(hue, 360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - abs((h % 2.0) - 1.0));
    var rgb = vec3<f32>(c, 0.0, x);
    switch i32(h) {
        case 0: { rgb = vec3<f32>(c, x, 0.0); }
        case 1: { rgb = vec3<f32>(x, c, 0.0); }
        case 2: { rgb = vec3<f32>(0.0, c, x); }
        case 3: { rgb = vec3<f32>(0.0, x, c); }
        case 4: { rgb = vec3<f32>(x, 0.0, c); }
        default: {}
    }
    return rgb + vec3<f32>(v - c);
}

fn pow_signed(v: f32, e: f32) -> f32 {
    if v >= 0.0 {
        return pow(v, e);
    }
    return -pow(-v, e);
}

// LayerFilter::apply_px: levels, adjustments, invert, channel swap.
fn apply_filter(p_in: vec4<f32>) -> vec4<f32> {
    var p = p_in;
    let flags = u32(P.adj1.w);
    if (flags & 4u) != 0u {
        let lo = P.extra.x;
        let hi = P.extra.y;
        var scale = 1.0;
        if hi - lo > 1e-9 {
            scale = 1.0 / (hi - lo);
        }
        p = vec4<f32>((p.rgb - vec3<f32>(lo)) * scale, p.a);
    }
    if (flags & 1u) != 0u {
        var c = p.rgb;
        if P.adj0.x != 0.0 {
            c = c * exp2(P.adj0.x);
        }
        if P.adj1.x != 0.0 || P.adj1.y != 1.0 {
            let hsv = rgb_to_hsv(c);
            c = hsv_to_rgb(hsv.x + P.adj1.x, max(hsv.y * P.adj1.y, 0.0), hsv.z);
        }
        let inv_g = 1.0 / max(P.adj0.w, 1e-4);
        if P.adj0.w != 1.0 {
            c = vec3<f32>(pow_signed(c.r, inv_g), pow_signed(c.g, inv_g), pow_signed(c.b, inv_g));
        }
        c = (c - vec3<f32>(0.5)) * P.adj0.z + vec3<f32>(0.5 + P.adj0.y);
        p = vec4<f32>(c, clamp(p.a + P.adj1.z, 0.0, 1.0));
    }
    if (flags & 2u) != 0u {
        p = vec4<f32>(vec3<f32>(1.0) - p.rgb, p.a);
    }
    if (flags & 8u) != 0u {
        let a = i32(P.extra.z);
        let b = i32(P.extra.w);
        let va = p[a];
        p[a] = p[b];
        p[b] = va;
    }
    return p;
}

@fragment
fn fs_filter(v: VOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(v.pos.xy));
    let d = unpremultiply(textureLoad(acc_tex, p, 0));
    var m = 1.0;
    if P.canvas.w > 1.5 {
        let lp = p - vec2<i32>(P.rect.xy);
        if inside(lp, vec2<i32>(textureDimensions(layer_tex))) {
            m = textureLoad(layer_tex, lp, 0).a;
        } else {
            m = 0.0;
        }
    }
    let k = min(P.canvas.z, 1.0) * m;
    var r = d;
    if k > 0.0 {
        r = d + (apply_filter(d) - d) * k;
    }
    return vec4<f32>(r.rgb * r.a, r.a);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// To 8-bit premultiplied, like Color32::from_rgba_unmultiplied of the CPU renderer.
@fragment
fn fs_display(v: VOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(v.pos.xy));
    let c = unpremultiply(textureLoad(acc_tex, p, 0));
    var rgb = clamp(c.rgb * P.adj0.x, vec3<f32>(0.0), vec3<f32>(1.0));
    if P.adj0.y > 0.5 {
        rgb = linear_to_srgb(rgb);
    }
    let a = clamp(c.a, 0.0, 1.0);
    return vec4<f32>(rgb * a, a);
}

// Mipmap level from the level above: the average of each 2x2 block.
@fragment
fn fs_mip(v: VOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(mip_src));
    let uv = (floor(v.pos.xy) * 2.0 + vec2<f32>(1.0)) / dims;
    return textureSampleLevel(mip_src, mip_sampler, uv, 0.0);
}
