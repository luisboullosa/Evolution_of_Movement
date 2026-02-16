use crate::hex_grid::HexCoord;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    seed: u32,
    time: f32,
    spatial_scale: f32,
    time_scale: f32,
    time_amplitude: f32,
    resource_scale: f32,
    count: u32,
    _pad: u32,
}

pub struct GpuResourceUpdater {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    param_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    cell_count: u32,
}

impl GpuResourceUpdater {
    /// Create a reusable compute pipeline that updates per-cell resource values on the GPU.
    ///
    /// The updater keeps device/pipeline/buffers alive across frames so each frame only
    /// uploads a tiny uniform block (`GpuParams`) and dispatches a compute pass.
    pub fn new(coords: &[HexCoord], base: &[f64]) -> Result<Self, String> {
        if coords.is_empty() || base.is_empty() || coords.len() != base.len() {
            return Err("invalid GPU resource initialization input".to_string());
        }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no GPU adapter available".to_string())?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("resource-updater-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|e| format!("failed to create GPU device: {}", e))?;

        // Store axial coordinates and base resource values as static GPU buffers.
        let coords_data: Vec<[f32; 2]> = coords
            .iter()
            .map(|c| [c.q as f32, c.r as f32])
            .collect();
        let base_data: Vec<f32> = base.iter().map(|v| *v as f32).collect();
        let cell_count = coords_data.len() as u32;

        let coords_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resource-coords"),
            contents: bytemuck::cast_slice(&coords_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let base_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resource-base"),
            contents: bytemuck::cast_slice(&base_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resource-params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resource-output"),
            size: (cell_count as u64) * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resource-readback"),
            size: (cell_count as u64) * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The WGSL compute shader performs a time-varying 2D spatial noise sample.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resource-update-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resource_update.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("resource-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resource-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resource-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resource-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coords_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: base_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            param_buffer,
            out_buffer,
            readback_buffer,
            cell_count,
        })
    }

    pub fn compute(
        &mut self,
        seed: u32,
        time: f32,
        spatial_scale: f32,
        time_scale: f32,
        time_amplitude: f32,
        resource_scale: f32,
    ) -> Option<Vec<f64>> {
        if self.cell_count == 0 {
            return Some(Vec::new());
        }

        // Per-frame dynamic parameters (time, scaling, seed) are uploaded via uniform buffer.
        let params = GpuParams {
            seed,
            time,
            spatial_scale,
            time_scale,
            time_amplitude,
            resource_scale,
            count: self.cell_count,
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.param_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resource-encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resource-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // Workgroup size is 64 in WGSL; round up to cover all cells.
            let workgroups = (self.cell_count + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let byte_len = (self.cell_count as u64) * std::mem::size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(&self.out_buffer, 0, &self.readback_buffer, 0, byte_len);

        self.queue.submit(Some(encoder.finish()));

        // Synchronous readback: simple and robust for desktop interactive use.
        // (Can be replaced with async staging if needed for higher throughput.)
        let buffer_slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        match rx.recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let values_f32 = bytemuck::cast_slice::<u8, f32>(&data);
                let out: Vec<f64> = values_f32
                    .iter()
                    .take(self.cell_count as usize)
                    .map(|v| *v as f64)
                    .collect();
                drop(data);
                self.readback_buffer.unmap();
                Some(out)
            }
            _ => {
                self.readback_buffer.unmap();
                None
            }
        }
    }
}
