use std::iter::once;

use capuronii::compile;
use naga::valid::Validator;
use parse::{
    ast_parser::parse_expression_list_entry, latex_parser, name_resolver::resolve_names,
    type_checker::type_check,
};
use typed_index_collections::ti_vec;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BufferDescriptor, BufferUsages, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelFlags, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    wgt::CommandEncoderDescriptor,
};

#[test]
fn test_main() {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    if !adapter
        .get_downlevel_capabilities()
        .flags
        .contains(DownlevelFlags::COMPUTE_SHADERS)
    {
        eprintln!("wgpu Adapter does not support compute shaders! skipping test.");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();

    let heap_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("heap buffer"),
        size: 20000 * 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let constant_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("input buffer"),
        size: 10000 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("output buffer"),
        size: 10000 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let upload_staging = device.create_buffer(&BufferDescriptor {
        label: Some("upload staging buffer"),
        size: 10000 * 4,
        usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    });
    let download_staging = device.create_buffer(&BufferDescriptor {
        label: Some("download staging buffer"),
        size: 10000 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let eval_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("eval bind group"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let eval_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Eval bind group"),
        layout: &eval_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: constant_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: heap_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&eval_bind_group_layout],
        push_constant_ranges: &[],
    });

    let test_case = |src: &str| -> Vec<f32> {
        let tex = latex_parser::parse_latex(src).unwrap();
        let expr = parse_expression_list_entry(&tex).unwrap();
        let (a, b) = resolve_names(&ti_vec![expr]);
        let (checked, map) = type_check(&a);
        let expr = &checked.first().unwrap().value;
        let module = compile(expr);
        dbg!(&checked);

        let mut v = Validator::new(Default::default(), Default::default());
        let info = v.validate(&module).unwrap();
        println!(
            "Compiled MSL: {}",
            naga::back::msl::write_string(&module, &info, &Default::default(), &Default::default(),)
                .unwrap()
                .0
        );
        let s = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &s,
            entry_point: Some("capuronii_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let mut c = enc.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        c.set_pipeline(&pipeline);
        c.set_bind_group(0, Some(&eval_bind_group), &[]);
        c.dispatch_workgroups(1, 1, 1);
        drop(c);

        enc.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &download_staging,
            0,
            output_buffer.size(),
        );

        let cb = enc.finish();
        queue.submit(once(cb));
        let slice = download_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let view = slice.get_mapped_range();

        let r = view
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect();
        drop(view);
        download_staging.unmap();
        r
    };
    dbg!(&test_case("[1,2,3][1]")[0]);
    dbg!(&test_case("[1,2,3,4][[1,2,3]>2][1]")[0]);
    dbg!(&test_case("[1,2,3,4][ [ ([1,2][[1,2]> 1])[1] , 2 , 3] > 2 ][1]")[0]);
    dbg!(
        &test_case(
            r"[ (a, \{ b > 2: [1,2,3,4], a > 2: [4,3,2,1], 1 \}[2] ) \operatorname{for} a = [1,2,3,4], b = [1,3,5] ][2]"
        )[0..2]
    );
}
