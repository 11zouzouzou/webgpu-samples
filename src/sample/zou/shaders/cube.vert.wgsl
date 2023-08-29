struct Uniforms {
    modelViewProjectionMatrix: mat4x4<f32>,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) fragUV: vec2<f32>,
    @location(1) fragPosition: vec4<f32>,
}

@vertex
fn main(
    @location(0) position: vec4<f32>,
    @location(1) uv: vec2<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.Position = uniforms.modelViewProjectionMatrix * position;
    out.fragUV = uv;
    out.fragPosition =  0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
    return out;
}