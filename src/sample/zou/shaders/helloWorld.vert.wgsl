 
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) cell: vec2f,
  @location(1) vUv: vec2f,
};

@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> cellState: array<u32>;

@vertex
fn main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let i = f32(instance); // Save the instance_index as a float
  // Compute the cell coordinate from the instance_index
    let cell = vec2<f32>(i % grid.x, floor(i / grid.x));
    let state = f32(cellState[instance]);
    let cellOffset = cell / grid * 2.0; // Updated
    let gridPos = (position * state + 1.0) / grid - 1.0 + cellOffset;
    var output: VertexOutput;
    output.position = vec4<f32>(gridPos, 0.0, 1.0);
    output.cell = cell;
    output.vUv = uv;
    return output;
}
