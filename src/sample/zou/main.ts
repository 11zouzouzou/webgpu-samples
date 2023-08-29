import { SampleInit, makeSample } from '../../components/SampleLayout';
import HelloWorldVertWGSL from './shaders/helloWorld.vert.wgsl';
import HelloWorldFragWGSL from './shaders/helloWorld.frag.wgsl';
import HelloWorldComputeWGSL from './shaders/helloWorld.compute.wgsl';
import CubeFragWGSL from './shaders/cube.frag.wgsl';
import CubeVertWGSL from './shaders/cube.vert.wgsl';
import { mat4, vec3 } from 'wgpu-matrix';
import { cubeVertexArray, cubeVertexCount } from '../../meshes/cube';
interface InitWebGPUResult {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  size: { width: number; height: number };
}
const initWebGPU = async (
  canvas: HTMLCanvasElement
): Promise<InitWebGPUResult> => {
  // 判断当前设备是否支持WebGPU
  if (!navigator.gpu) throw new Error('Not Support WebGPU');
  // 请求Adapter对象，GPU在浏览器中的抽象代理
  const adapter = await navigator.gpu.requestAdapter({
    /* 
    high-performance 高性能电源管理
    low-power 节能电源管理模式 */
    powerPreference: 'high-performance',
  });

  const device = await adapter.requestDevice(); //请求GPU设备

  const context = canvas.getContext('webgpu') as GPUCanvasContext; //获取webgpu上下文

  const format = navigator.gpu.getPreferredCanvasFormat(); //获取浏览器默认的颜色格式

  const devicePixelRatio = window.devicePixelRatio || 1; //设备分辨率

  //canvas 实际size set
  const size = {
    width: canvas.clientWidth * devicePixelRatio,
    height: canvas.clientHeight * devicePixelRatio,
  };
  canvas.width = size.width;
  canvas.height = size.height;

  //配置WebGPU//https://www.w3.org/TR/webgpu/#dom-gpucanvascontext-configure
  context.configure({
    device,
    format,
    // Alpha合成模式，opaque为不透明,premultiplied为前乘
    alphaMode: 'premultiplied',
  });

  return { device, context, format, size };
};

// 用于从类型数组中创建GPUBuffer的辅助函数
const createBuffer = (
  device: GPUDevice,
  arr: Float32Array | Uint16Array | Uint32Array,
  usage: number
): GPUBuffer => {
  //   Align to 4 bytes (thanks @chrimsonite)
  //
  const desc = {
    size: (arr.byteLength + 3) & ~3,
    usage: usage,
    mappedAtCreation: true,
  };
  const buffer = device.createBuffer(desc);

  //   const writeArray =
  //     arr instanceof Uint16Array
  //       ? new Uint16Array(buffer.getMappedRange())
  //       : new Float32Array(buffer.getMappedRange());
  //   writeArray.set(arr);
  buffer.unmap();
  return buffer;
};

const initRenderPipeline = async (
  device: GPUDevice,
  format: GPUTextureFormat,
  vertexBufferLayout: GPUVertexBufferLayout,
  pipelineLayout?: GPUPipelineLayout
): Promise<GPURenderPipeline> => {
  const descriptor: GPURenderPipelineDescriptor = {
    // 顶点着色器
    vertex: {
      // 着色程序
      module: device.createShaderModule({
        code: HelloWorldVertWGSL,
      }),
      // 主函数
      entryPoint: 'main',
      buffers: [vertexBufferLayout],
    },
    // 片元着色器
    fragment: {
      // 着色程序
      module: device.createShaderModule({
        code: HelloWorldFragWGSL,
      }),
      // 主函数
      entryPoint: 'main',
      // 渲染目标
      targets: [
        {
          // 颜色格式
          format: format,
        },
      ],
    },
    // 初始配置
    primitive: {
      //绘制独立三角形
      topology: 'triangle-list',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
    // 渲染管线的布局
    layout: pipelineLayout ? pipelineLayout : 'auto',
  };
  // 返回异步管线
  return await device.createRenderPipelineAsync(descriptor);
};

const initCubeRenderPipeline = async (
  device: GPUDevice,
  format: GPUTextureFormat,
  vertexBufferLayout: GPUVertexBufferLayout,
  pipelineLayout?: GPUPipelineLayout
): Promise<GPURenderPipeline> => {
  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout ? pipelineLayout : 'auto',
    vertex: {
      module: device.createShaderModule({
        code: CubeVertWGSL,
      }),
      entryPoint: 'main',
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: device.createShaderModule({
        code: CubeFragWGSL,
      }),
      entryPoint: 'main',
      targets: [
        {
          format: format,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    },
  });
  return pipeline;
};

const initComputePipeline = async (
  device: GPUDevice,
  pipelineLayout?: GPUPipelineLayout
): Promise<GPUComputePipeline> => {
  // Create the compute shader that will process the game of life simulation.
  const simulationShaderModule = device.createShaderModule({
    label: 'Life simulation shader',
    code: HelloWorldComputeWGSL.replace(
      /REPLACE_CPU_WORKGROUP_SIZE/g,
      WORKGROUP_SIZE + ''
    ),
  });
  // Create a compute pipeline that updates the game state.
  return await device.createComputePipelineAsync({
    label: 'Simulation pipeline',
    layout: pipelineLayout ? pipelineLayout : 'auto',
    compute: {
      module: simulationShaderModule,
      entryPoint: 'computeMain',
    },
  });
};

const draw = async (
  device: GPUDevice,
  context: GPUCanvasContext,
  computePipeline: GPUComputePipeline,
  pipeline: GPURenderPipeline,
  computeBindGroup: GPUBindGroup,
  bindGroup: GPUBindGroup,
  vertexBuffer: GPUBuffer,
  indexCount: number,
  instanceCount: number,
  depthTexture: GPUTexture,
  renderTarget?: GPUTexture
) => {
  //GPU纹理视图
  const view = context.getCurrentTexture().createView();
  // 渲染通道配置数据
  const renderPassDescriptor: GPURenderPassDescriptor = {
    // 颜色附件
    colorAttachments: [
      {
        view: renderTarget ? renderTarget.createView() : view,
        // resolveTarget: view,//多重采样
        // 绘图前是否清空view，建议清空clear
        loadOp: 'clear', // clear/load
        // 清理画布的颜色
        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
        //绘制完成后，是否保留颜色信息
        storeOp: 'store', // store/discard
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),

      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  };

  // 创建指令编码器
  const commandEncoder = device.createCommandEncoder();

  // Start a compute pass
  const computePass = commandEncoder.beginComputePass();

  computePass.setPipeline(computePipeline),
    computePass.setBindGroup(0, computeBindGroup);
  const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
  computePass.end();

  // 建立渲染通道，类似图层
  const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

  // 传入渲染管线
  passEncoder.setPipeline(pipeline);
  //设置顶点缓存
  passEncoder.setVertexBuffer(0, vertexBuffer);
  //设置uniform 参数
  passEncoder.setBindGroup(0, bindGroup);
  // 绘图顶点
  passEncoder.draw(indexCount, instanceCount);
  // 结束编码
  passEncoder.end();

  // 结束指令编写,并返回GPU指令缓冲区
  const gpuCommandBuffer = commandEncoder.finish();

  // 向GPU提交绘图指令，所有指令将在提交后执行
  device.queue.submit([gpuCommandBuffer]);
};
let g_cellStateStorage: GPUBuffer[] = [];
let g_bindGroup: GPUBindGroup[] = [];
let g_renderPipline: GPURenderPipeline | undefined;
const updateStorage = (device: GPUDevice) => {
  //Create storage buffer
  // Create an array representing the active state of each cell.
  const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
  // Create two storage buffers to hold the cell state.
  g_cellStateStorage = [
    createBuffer(
      device,
      cellStateArray,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    ),
    createBuffer(
      device,
      cellStateArray,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    ),
  ];

  // Mark every third cell of the first grid as active.
  for (let i = 0; i < cellStateArray.length; i += 3) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
  }
  device.queue.writeBuffer(g_cellStateStorage[0], 0, cellStateArray);

  // Mark every other cell of the second grid as active.
  // for (let i = 0; i < cellStateArray.length; ++i) {
  //   cellStateArray[i] = i % 2;
  // }
  // device.queue.writeBuffer(g_cellStateStorage[1], 0, cellStateArray);
};
const updateBindGrounp = (
  device: GPUDevice,
  renderPipeline: GPURenderPipeline,
  bindGroupLayout?: GPUBindGroupLayout
) => {
  g_bindGroup = [
    device.createBindGroup({
      label: 'Cell renderer bind group A',
      layout: bindGroupLayout
        ? bindGroupLayout
        : renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: g_uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: g_cellStateStorage[0] },
        },
        {
          binding: 2,
          resource: { buffer: g_cellStateStorage[1] },
        },
        {
          binding: 3,
          resource: g_defaultSampler,
        },
        {
          binding: 4,
          resource: g_defaultTexture.createView(),
        },
      ],
    }),
    device.createBindGroup({
      label: 'Cell renderer bind group B',
      layout: bindGroupLayout
        ? bindGroupLayout
        : renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: g_uniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: g_cellStateStorage[1] },
        },
        {
          binding: 2,
          resource: { buffer: g_cellStateStorage[0] },
        },
        {
          binding: 3,
          resource: g_defaultSampler,
        },
        {
          binding: 4,
          resource: g_defaultTexture.createView(),
        },
      ],
    }),
  ];
};

const updateCubeBindGrounp = (
  device: GPUDevice,
  renderPipeline: GPURenderPipeline,
  bindGroupLayout?: GPUBindGroupLayout
) => {
  g_cubeBindGroup = device.createBindGroup({
    label: 'Cell renderer bind group Cube',
    layout: bindGroupLayout
      ? bindGroupLayout
      : renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: g_cubeUniformBuffer },
      },
      {
        binding: 1,
        resource: g_cubeSampler,
      },
      {
        binding: 2,
        resource: g_cubeTexture.createView(),
      },
    ],
  });
};

let GRID_SIZE = 32;
const WORKGROUP_SIZE = 8;
let g_uniformBuffer: GPUBuffer | undefined;
let g_device: GPUDevice | undefined;
let g_defaultTexture: GPUTexture | undefined;
let g_defaultSampler: GPUSampler | undefined;

let g_cubeUniformBuffer: GPUBuffer | undefined;
let g_cubeSampler: GPUSampler | undefined;
let g_cubeBindGroup: GPUBindGroup | undefined;
let g_cubeTexture: GPUTexture | undefined;

const init: SampleInit = async ({ canvas, pageState, gui, stats }) => {
  const webgpuResult = await initWebGPU(canvas);
  g_device = webgpuResult.device;
  if (!pageState.active) return;
  {
    const HWfolder = gui.addFolder('pre');
    HWfolder.open();
    HWfolder.add({ gridSize: GRID_SIZE }, 'gridSize', 1, 48, 1).onChange(
      (v) => {
        GRID_SIZE = v;
        g_device.queue.writeBuffer(
          g_uniformBuffer,
          0,
          new Float32Array([GRID_SIZE, GRID_SIZE])
        );
        updateStorage(g_device);
        updateBindGrounp(g_device, g_renderPipline); //buffer change
      }
    );
    stats.showPanel(0);
  }

  // Create a buffer with the vertices for a single cell.
  const vertices = new Float32Array([
    //   X,    Y
    -0.8,
    -0.8,
    0.0,
    0.0, // Triangle 1 uv
    0.8,
    -0.8,
    1.0,
    0.0,
    0.8,
    0.8,
    1.0,
    1.0,
    -0.8,
    -0.8,
    0.0,
    0.0, // Triangle 2 uv
    0.8,
    0.8,
    1.0,
    1.0,
    -0.8,
    0.8,
    0.0,
    1.0,
  ]);

  const vertexBuffer = createBuffer(
    webgpuResult.device,
    vertices,
    GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  );
  webgpuResult.device.queue.writeBuffer(vertexBuffer, 0, vertices);

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 16,
    attributes: [
      {
        format: 'float32x2',
        offset: 0,
        shaderLocation: 0, // Position. Matches @location(0) in the @vertex shader.
      },
      {
        // uv
        shaderLocation: 1,
        offset: 8,
        format: 'float32x2',
      },
    ],
  };

  //Create uniform buffer
  const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
  const uniformBuffer = createBuffer(
    webgpuResult.device,
    uniformArray,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  );
  g_uniformBuffer = uniformBuffer;

  webgpuResult.device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

  updateStorage(webgpuResult.device);

  //Create texture load

  const response = await fetch(
    new URL('../../../assets/img/Di-3d.png', import.meta.url).toString()
  );
  const imageBitmap = await createImageBitmap(await response.blob());
  const defaultTexture = webgpuResult.device.createTexture({
    size: [imageBitmap.width, imageBitmap.height, 1],
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  webgpuResult.device.queue.copyExternalImageToTexture(
    { source: imageBitmap, flipY: true },
    { texture: defaultTexture },
    [imageBitmap.width, imageBitmap.height]
  );

  g_defaultTexture = defaultTexture;
  //Create sampler
  const defaultSampler = webgpuResult.device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });
  g_defaultSampler = defaultSampler;

  // Create the bind group layout and pipeline layout.
  const bindGroupLayout = webgpuResult.device.createBindGroupLayout({
    label: 'Cell Bind Group Layout',
    entries: [
      {
        binding: 0,
        visibility:
          GPUShaderStage.VERTEX |
          GPUShaderStage.FRAGMENT |
          GPUShaderStage.COMPUTE,
        buffer: {}, // Grid uniform buffer
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }, // Cell state input buffer
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }, // Cell state output buffer
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
      {
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          viewDimension: '2d',
        },
      },
    ],
  });

  const pipelineLayout = webgpuResult.device.createPipelineLayout({
    label: 'Cell Pipeline Layout',
    bindGroupLayouts: [bindGroupLayout],
  });

  const renderPipelineResult = await initRenderPipeline(
    webgpuResult.device,
    webgpuResult.format,
    vertexBufferLayout,
    pipelineLayout
  );
  g_renderPipline = renderPipelineResult;
  const computePipelineResult = await initComputePipeline(
    webgpuResult.device,
    pipelineLayout
  );

  // Create a vertex buffer from the cube data.
  const verticesCubeBuffer = webgpuResult.device.createBuffer({
    size: cubeVertexArray.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(verticesCubeBuffer.getMappedRange()).set(cubeVertexArray);
  verticesCubeBuffer.unmap();

  const cubeBindGroupLayout = webgpuResult.device.createBindGroupLayout({
    label: 'Cube Bind Group Layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          viewDimension: '2d',
        },
      },
    ],
  });
  const cubePipelineLayout = webgpuResult.device.createPipelineLayout({
    label: 'Cube Pipeline Layout',
    bindGroupLayouts: [cubeBindGroupLayout],
  });

  const cubeVertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 4 * 10,
    attributes: [
      {
        //pos
        shaderLocation: 0,
        offset: 0,
        format: 'float32x4',
      },
      {
        //uv
        shaderLocation: 1,
        offset: 4 * 8,
        format: 'float32x2',
      },
    ],
  };

  const cubeRenderPipeline = await initCubeRenderPipeline(
    webgpuResult.device,
    webgpuResult.format,
    cubeVertexBufferLayout,
    cubePipelineLayout
  );

  const depthTexture = webgpuResult.device.createTexture({
    size: [webgpuResult.size.width, webgpuResult.size.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const aspect = webgpuResult.size.width / webgpuResult.size.height;
  const projectionMatrix = mat4.perspective(
    (2 * Math.PI) / 5,
    aspect,
    1,
    100.0
  );
  const modelViewProjectionMatrix = mat4.create();

  const getTransformationMatrix = () => {
    const viewMatrix = mat4.identity();
    mat4.translate(viewMatrix, vec3.fromValues(0, 0, -4), viewMatrix);
    const now = Date.now() / 1000;
    mat4.rotate(
      viewMatrix,
      vec3.fromValues(Math.sin(now), Math.cos(now), 0),
      1,
      viewMatrix
    );

    mat4.multiply(projectionMatrix, viewMatrix, modelViewProjectionMatrix);

    return modelViewProjectionMatrix as Float32Array;
  };

  const transformuUniformBuffer = webgpuResult.device.createBuffer({
    size: 4 * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  g_cubeUniformBuffer = transformuUniformBuffer;

  const cubeTexture = webgpuResult.device.createTexture({
    size: [webgpuResult.size.width, webgpuResult.size.height],
    // size: [imageBitmap.width, imageBitmap.height],
    format: webgpuResult.format,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  g_cubeTexture = cubeTexture;
  const cubeSampler = webgpuResult.device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });
  g_cubeSampler = cubeSampler;

  // Create a bind group to pass the grid uniforms into the pipeline
  updateBindGrounp(webgpuResult.device, renderPipelineResult, bindGroupLayout);
  updateCubeBindGrounp(
    webgpuResult.device,
    cubeRenderPipeline,
    cubeBindGroupLayout
  );
  //cube render pass

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // Assigned later

        clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),

      depthClearValue: 1.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    },
  };

  let step = 0;
  const frame = () => {
    // Sample is no longer the active page.
    if (!pageState.active) return;
    //test
    stats.begin();

    //draw
    const computeBindGroup = g_bindGroup[step % 64 < 32 ? 0 : 1];
    step++; // Increment the step count

    draw(
      webgpuResult.device,
      webgpuResult.context,
      computePipelineResult,
      renderPipelineResult,
      computeBindGroup,
      g_bindGroup[step % 64 < 32 ? 0 : 1],
      vertexBuffer,
      vertices.length / 4,
      GRID_SIZE * GRID_SIZE,
      depthTexture,
      cubeTexture
    );

    //draw cube
    const transformationMatrix = getTransformationMatrix();
    webgpuResult.device.queue.writeBuffer(
      transformuUniformBuffer,
      0,
      transformationMatrix.buffer,
      transformationMatrix.byteOffset,
      transformationMatrix.byteLength
    );
    const swapChainTexture = webgpuResult.context.getCurrentTexture();
    // prettier-ignore
    renderPassDescriptor.colorAttachments[0].view = swapChainTexture.createView();

    const commandEncoder = webgpuResult.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(cubeRenderPipeline);
    passEncoder.setBindGroup(0, g_cubeBindGroup);
    passEncoder.setVertexBuffer(0, verticesCubeBuffer);
    passEncoder.draw(cubeVertexCount, 1, 0, 0);
    passEncoder.end();

    webgpuResult.device.queue.submit([commandEncoder.finish()]);

    //test
    stats.end();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
};

const HelloWorld: () => JSX.Element = () => {
  return makeSample({
    name: 'zou hello world',
    description: 'Shows rendering pass',
    gui: true,
    stats: true,
    init,
    sources: [
      {
        name: __filename.substring(__dirname.length + 1),
        contents: __SOURCE__,
      },
      {
        name: './shaders/helloWorld.vert.wgsl',
        contents: HelloWorldVertWGSL,
        editable: true,
      },
      {
        name: './shaders/helloWorld.frag.wgsl',
        contents: HelloWorldFragWGSL,
        editable: true,
      },
    ],
    filename: __filename,
  });
};

export default HelloWorld;
