// GPU versions of imged_core::ops::convolve::convolve_cpu and
// imged_core::ops::transform::resize_cpu (see gpu_filters.rs). The math and
// its order follow the CPU code, so results match it to float rounding.
//
// Pixels travel as u32 words in their own depth: 8-bit RGBA = 1 word,
// 16-bit = 2 words, float = 4 words (bit patterns).

struct Uniforms {
    w: u32,
    h: u32,
    depth: u32,     // 0 u8, 1 u16, 2 f32
    ntaps: u32,
    edge: u32,      // 0 clamp, 1 transparent, 2 wrap
    alpha: u32,     // convolve alpha too (premultiplied)
    nw: u32,
    nh: u32,
    inv: f32,
    bias: f32,
    v_idx: u32,     // where the vertical axis starts in idx
    v_wts: u32,     // where the vertical weights start in wts
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> mid: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> idx: array<u32>;
@group(0) @binding(5) var<storage, read> wts: array<f32>;

// Channel::to_f32 (multiplying by the reciprocal, like the CPU).
fn load(i: u32) -> vec4<f32> {
    switch U.depth {
        case 0u: {
            let v = src[i];
            return vec4<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), f32(v >> 24u)) * (1.0 / 255.0);
        }
        case 1u: {
            let a = src[2u * i];
            let b = src[2u * i + 1u];
            return vec4<f32>(f32(a & 65535u), f32(a >> 16u), f32(b & 65535u), f32(b >> 16u)) * (1.0 / 65535.0);
        }
        default: {
            return vec4<f32>(bitcast<f32>(src[4u * i]), bitcast<f32>(src[4u * i + 1u]), bitcast<f32>(src[4u * i + 2u]), bitcast<f32>(src[4u * i + 3u]));
        }
    }
}

// Channel::from_f32: integers round and clamp, floats keep the value (NaN -> 0).
fn store(i: u32, p: vec4<f32>) {
    switch U.depth {
        case 0u: {
            let q = vec4<u32>(clamp(p, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
            dst[i] = q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
        }
        case 1u: {
            let q = vec4<u32>(clamp(p, vec4<f32>(0.0), vec4<f32>(1.0)) * 65535.0 + 0.5);
            dst[2u * i] = q.x | (q.y << 16u);
            dst[2u * i + 1u] = q.z | (q.w << 16u);
        }
        default: {
            let f = select(p, vec4<f32>(0.0), p != p);
            dst[4u * i] = bitcast<u32>(f.x);
            dst[4u * i + 1u] = bitcast<u32>(f.y);
            dst[4u * i + 2u] = bitcast<u32>(f.z);
            dst[4u * i + 3u] = bitcast<u32>(f.w);
        }
    }
}

fn premultiply(p: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(p.rgb * p.a, p.a);
}

fn unpremultiply(p: vec4<f32>) -> vec4<f32> {
    if p.a <= 1e-8 {
        return vec4<f32>(0.0);
    }
    let inv = 1.0 / p.a;
    return vec4<f32>(p.rgb * inv, p.a);
}

fn load_conv(x: i32, y: i32) -> vec4<f32> {
    let p = load(u32(y) * U.w + u32(x));
    if U.alpha != 0u {
        return premultiply(p);
    }
    return p;
}

@compute @workgroup_size(16, 16)
fn convolve(@builtin(global_invocation_id) g: vec3<u32>) {
    if g.x >= U.w || g.y >= U.h {
        return;
    }
    let x = i32(g.x);
    let y = i32(g.y);
    let w = i32(U.w);
    let h = i32(U.h);
    var acc = vec4<f32>(0.0);
    for (var t = 0u; t < U.ntaps; t++) {
        var sx = x + i32(wts[3u * t]);
        var sy = y + i32(wts[3u * t + 1u]);
        let wt = wts[3u * t + 2u];
        if U.edge == 0u {
            sx = clamp(sx, 0, w - 1);
            sy = clamp(sy, 0, h - 1);
        } else if U.edge == 1u {
            if sx < 0 || sy < 0 || sx >= w || sy >= h {
                continue;
            }
        } else {
            sx = sx - w * i32(floor(f32(sx) / f32(w)));
            sy = sy - h * i32(floor(f32(sy) / f32(h)));
        }
        acc += load_conv(sx, sy) * wt;
    }
    let orig = load(g.y * U.w + g.x);
    var o: vec4<f32>;
    if U.alpha != 0u {
        let a = clamp(acc.a * U.inv + U.bias, 0.0, 1.0);
        let u = unpremultiply(vec4<f32>(acc.rgb * U.inv, a));
        o = vec4<f32>(u.rgb + vec3<f32>(U.bias), a);
    } else {
        o = vec4<f32>(acc.rgb * U.inv + vec3<f32>(U.bias), orig.a);
    }
    store(g.y * U.w + g.x, o);
}

// Resize, pass 1: rows (w -> nw) into `mid`, premultiplied.
@compute @workgroup_size(16, 16)
fn resize_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    if g.x >= U.nw || g.y >= U.h {
        return;
    }
    let lo = idx[g.x];
    let k0 = idx[U.nw + g.x];
    let k1 = idx[U.nw + g.x + 1u];
    var acc = vec4<f32>(0.0);
    for (var k = k0; k < k1; k++) {
        acc += premultiply(load(g.y * U.w + lo + (k - k0))) * wts[k];
    }
    mid[g.y * U.nw + g.x] = acc;
}

// Resize, pass 2: columns (h -> nh) from `mid`, back to straight alpha.
@compute @workgroup_size(16, 16)
fn resize_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    if g.x >= U.nw || g.y >= U.nh {
        return;
    }
    let lo = idx[U.v_idx + g.y];
    let k0 = idx[U.v_idx + U.nh + g.y];
    let k1 = idx[U.v_idx + U.nh + g.y + 1u];
    var acc = vec4<f32>(0.0);
    for (var k = k0; k < k1; k++) {
        acc += mid[(lo + (k - k0)) * U.nw + g.x] * wts[U.v_wts + k];
    }
    acc.a = clamp(acc.a, 0.0, 1.0);
    store(g.y * U.nw + g.x, unpremultiply(acc));
}
