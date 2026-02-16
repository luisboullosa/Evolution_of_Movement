struct Params {
    seed: u32,
    time: f32,
    spatial_scale: f32,
    time_scale: f32,
    time_amplitude: f32,
    resource_scale: f32,
    count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> coords: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> base_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_values: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn hash31(p: vec3<f32>) -> f32 {
    let h = sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719)));
    return fract(h * 43758.5453);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (vec3<f32>(3.0) - 2.0 * f);

    let s = f32(params.seed) * 0.001;

    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0) + vec3<f32>(s, s * 0.37, s * 0.73));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0) + vec3<f32>(s, s * 0.37, s * 0.73));

    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let y0 = mix(x00, x10, u.y);

    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    let y1 = mix(x01, x11, u.y);

    return mix(y0, y1, u.z);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let c = coords[idx];
    let p = vec3<f32>(
        c.x * max(params.spatial_scale, 1e-6),
        c.y * max(params.spatial_scale, 1e-6),
        params.time * max(params.time_scale, 1e-6),
    );

    let temporal = noise3(p);
    let centered = temporal - 0.5;
    let value = (base_values[idx] + centered * params.time_amplitude) * params.resource_scale;
    out_values[idx] = max(value, 0.0);
}
