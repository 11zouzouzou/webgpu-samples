struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) cell: vec2f,
  @location(1) vUv: vec2f,
};

@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(3) var defaultSampler: sampler;
@group(0) @binding(4) var defaultTexture: texture_2d<f32>;

@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
   let color = textureSample(defaultTexture, defaultSampler, input.vUv);
    return color;
    // let c = input.cell / grid;
    // return vec4<f32>(c, 1.0 - c.x, 1.0);
}
