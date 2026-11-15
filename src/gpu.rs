//! The Vulkan compute tier: a logical device, a compute queue, device-local
//! weight buffers and six SPIR-V pipelines. No SDK, no ash, no build script -
//! the same `vulkan-1.dll` the probe loads, and the `.spv` blobs are checked in
//! next to their GLSL sources (`tools/shaders.sh` rebuilds them).
//!
//! Two things make this worth having on a MoE decode step. The lm_head is a
//! 248320 x 2048 bf16 matrix: one GiB of weights read once per token, which is
//! 58 ms of CPU memory traffic and a few ms of device traffic. And a layer's
//! routed experts are independent gemvs, so they can be recorded into one
//! command buffer, submitted, and left running while the CPU works the experts
//! that live in host RAM - `VkBatch::submit` then `VkBatch::wait` is the overlap.
//!
//! Vendor-neutral by construction: NVIDIA, AMD and Intel all take the same
//! code path, because nothing here names a vendor extension.

use crate::config::Result;
use crate::quant::Q4;
use crate::vk::{Device, Vulkan};
use std::ffi::c_void;
use std::os::raw::c_char;

const ST_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const ST_DEVICE_CREATE_INFO: u32 = 3;
const ST_SUBMIT_INFO: u32 = 4;
const ST_MEMORY_ALLOCATE_INFO: u32 = 5;
const ST_FENCE_CREATE_INFO: u32 = 8;
const ST_BUFFER_CREATE_INFO: u32 = 12;
const ST_SHADER_MODULE_CREATE_INFO: u32 = 16;
const ST_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const ST_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const ST_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const ST_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const ST_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const ST_WRITE_DESCRIPTOR_SET: u32 = 35;
const ST_COMMAND_POOL_CREATE_INFO: u32 = 39;
const ST_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const ST_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const ST_MEMORY_BARRIER: u32 = 46;

const DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const BUFFER_USAGE_TRANSFER_SRC: u32 = 1;
const BUFFER_USAGE_TRANSFER_DST: u32 = 2;
const BUFFER_USAGE_STORAGE_BUFFER: u32 = 0x20;
const MEMORY_PROPERTY_DEVICE_LOCAL: u32 = 1;
const MEMORY_PROPERTY_HOST_VISIBLE: u32 = 2;
const MEMORY_PROPERTY_HOST_COHERENT: u32 = 4;
const QUEUE_COMPUTE: u32 = 2;
const QUEUE_GRAPHICS: u32 = 1;
const SHADER_STAGE_COMPUTE: u32 = 0x20;
const PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
const PIPELINE_STAGE_COMPUTE_SHADER: u32 = 0x800;
const PIPELINE_STAGE_TRANSFER: u32 = 0x1000;
const ACCESS_SHADER_READ: u32 = 0x20;
const ACCESS_SHADER_WRITE: u32 = 0x40;
const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
const ACCESS_HOST_READ: u32 = 0x2000;
const PIPELINE_STAGE_HOST: u32 = 0x4000;
const COMMAND_POOL_RESET_COMMAND_BUFFER: u32 = 2;
const WHOLE_SIZE: u64 = u64::MAX;

/// The spec only guarantees 65535 workgroups per dispatch dimension, and the
/// lm_head has 248320 rows, so a tall gemv is recorded as several dispatches
/// with `row_base` walking up the matrix.
const MAX_GROUPS: u32 = 65535;
/// Every pipeline shares one layout: four storage buffers and a 32-byte push
/// range. The shaders use 16-24 bytes of it, which is legal and means one
/// descriptor set layout and one pipeline layout for the whole tier.
const PUSH_BYTES: u32 = 32;
pub(crate) const SCRATCH_FLOATS: usize = 1 << 21;
const STAGING_BYTES: usize = 64 << 20;
const MAX_SETS: u32 = 8192;

type Handle = *mut c_void;
type Gpa = unsafe extern "system" fn(Handle, *const c_char) -> Handle;

#[repr(C)]
struct QueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    family: u32,
    count: u32,
    priorities: *const f32,
}

#[repr(C)]
struct DeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_info_count: u32,
    queue_infos: *const QueueCreateInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
    features: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QueueFamily {
    flags: u32,
    count: u32,
    timestamp_bits: u32,
    granularity: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MemType {
    flags: u32,
    heap: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MemHeap {
    size: u64,
    flags: u32,
    _pad: u32,
}

#[repr(C)]
struct MemProps {
    type_count: u32,
    types: [MemType; 32],
    heap_count: u32,
    heaps: [MemHeap; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MemRequirements {
    size: u64,
    alignment: u64,
    type_bits: u32,
    _pad: u32,
}

#[repr(C)]
struct MemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    size: u64,
    type_index: u32,
}

#[repr(C)]
struct BufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing: u32,
    family_count: u32,
    families: *const u32,
}

#[repr(C)]
struct ShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    code: *const u32,
}

#[repr(C)]
struct DescriptorSetLayoutBinding {
    binding: u32,
    kind: u32,
    count: u32,
    stages: u32,
    immutable: *const c_void,
}

#[repr(C)]
struct DescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    bindings: *const DescriptorSetLayoutBinding,
}

#[repr(C)]
struct PushConstantRange {
    stages: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct PipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    set_layouts: *const Handle,
    push_count: u32,
    push_ranges: *const PushConstantRange,
}

#[repr(C)]
struct PipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: Handle,
    name: *const c_char,
    specialization: *const c_void,
}

#[repr(C)]
struct ComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: PipelineShaderStageCreateInfo,
    layout: Handle,
    base_handle: Handle,
    base_index: i32,
}

#[repr(C)]
struct DescriptorPoolSize {
    kind: u32,
    count: u32,
}

#[repr(C)]
struct DescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    size_count: u32,
    sizes: *const DescriptorPoolSize,
}

#[repr(C)]
struct DescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    pool: Handle,
    count: u32,
    layouts: *const Handle,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DescriptorBufferInfo {
    buffer: Handle,
    offset: u64,
    range: u64,
}

#[repr(C)]
struct WriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    set: Handle,
    binding: u32,
    element: u32,
    count: u32,
    kind: u32,
    images: *const c_void,
    buffers: *const DescriptorBufferInfo,
    texels: *const c_void,
}

#[repr(C)]
struct CommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    family: u32,
}

#[repr(C)]
struct CommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    pool: Handle,
    level: u32,
    count: u32,
}

#[repr(C)]
struct CommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    inheritance: *const c_void,
}

#[repr(C)]
struct MemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access: u32,
    dst_access: u32,
}

#[repr(C)]
struct BufferCopy {
    src: u64,
    dst: u64,
    size: u64,
}

#[repr(C)]
struct SubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_count: u32,
    waits: *const Handle,
    wait_stages: *const u32,
    cb_count: u32,
    cbs: *const Handle,
    signal_count: u32,
    signals: *const Handle,
}

#[repr(C)]
struct FenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

macro_rules! decl {
    ($($n:ident = $sig:ty),* $(,)?) => {
        $(type $n = $sig;)*
    };
}

decl! {
    CreateBuffer = unsafe extern "system" fn(Handle, *const BufferCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyBuffer = unsafe extern "system" fn(Handle, Handle, *const c_void),
    BufferMemReq = unsafe extern "system" fn(Handle, Handle, *mut MemRequirements),
    AllocateMemory = unsafe extern "system" fn(Handle, *const MemoryAllocateInfo, *const c_void, *mut Handle) -> i32,
    FreeMemory = unsafe extern "system" fn(Handle, Handle, *const c_void),
    BindBufferMemory = unsafe extern "system" fn(Handle, Handle, Handle, u64) -> i32,
    MapMemory = unsafe extern "system" fn(Handle, Handle, u64, u64, u32, *mut *mut c_void) -> i32,
    UnmapMemory = unsafe extern "system" fn(Handle, Handle),
    CreateShaderModule = unsafe extern "system" fn(Handle, *const ShaderModuleCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyShaderModule = unsafe extern "system" fn(Handle, Handle, *const c_void),
    CreateSetLayout = unsafe extern "system" fn(Handle, *const DescriptorSetLayoutCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroySetLayout = unsafe extern "system" fn(Handle, Handle, *const c_void),
    CreatePipelineLayout = unsafe extern "system" fn(Handle, *const PipelineLayoutCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyPipelineLayout = unsafe extern "system" fn(Handle, Handle, *const c_void),
    CreateComputePipelines = unsafe extern "system" fn(Handle, Handle, u32, *const ComputePipelineCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyPipeline = unsafe extern "system" fn(Handle, Handle, *const c_void),
    CreateDescriptorPool = unsafe extern "system" fn(Handle, *const DescriptorPoolCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyDescriptorPool = unsafe extern "system" fn(Handle, Handle, *const c_void),
    ResetDescriptorPool = unsafe extern "system" fn(Handle, Handle, u32) -> i32,
    AllocateSets = unsafe extern "system" fn(Handle, *const DescriptorSetAllocateInfo, *mut Handle) -> i32,
    UpdateSets = unsafe extern "system" fn(Handle, u32, *const WriteDescriptorSet, u32, *const c_void),
    CreateCommandPool = unsafe extern "system" fn(Handle, *const CommandPoolCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyCommandPool = unsafe extern "system" fn(Handle, Handle, *const c_void),
    AllocateCommandBuffers = unsafe extern "system" fn(Handle, *const CommandBufferAllocateInfo, *mut Handle) -> i32,
    BeginCommandBuffer = unsafe extern "system" fn(Handle, *const CommandBufferBeginInfo) -> i32,
    EndCommandBuffer = unsafe extern "system" fn(Handle) -> i32,
    ResetCommandBuffer = unsafe extern "system" fn(Handle, u32) -> i32,
    CmdBindPipeline = unsafe extern "system" fn(Handle, u32, Handle),
    CmdBindSets = unsafe extern "system" fn(Handle, u32, Handle, u32, u32, *const Handle, u32, *const u32),
    CmdPushConstants = unsafe extern "system" fn(Handle, Handle, u32, u32, u32, *const c_void),
    CmdDispatch = unsafe extern "system" fn(Handle, u32, u32, u32),
    CmdBarrier = unsafe extern "system" fn(Handle, u32, u32, u32, u32, *const MemoryBarrier, u32, *const c_void, u32, *const c_void),
    CmdCopyBuffer = unsafe extern "system" fn(Handle, Handle, Handle, u32, *const BufferCopy),
    QueueSubmit = unsafe extern "system" fn(Handle, u32, *const SubmitInfo, Handle) -> i32,
    CreateFence = unsafe extern "system" fn(Handle, *const FenceCreateInfo, *const c_void, *mut Handle) -> i32,
    DestroyFence = unsafe extern "system" fn(Handle, Handle, *const c_void),
    ResetFences = unsafe extern "system" fn(Handle, u32, *const Handle) -> i32,
    WaitForFences = unsafe extern "system" fn(Handle, u32, *const Handle, u32, u64) -> i32,
    GetFenceStatus = unsafe extern "system" fn(Handle, Handle) -> i32,
    DeviceWaitIdle = unsafe extern "system" fn(Handle) -> i32,
}

struct Api {
    create_buffer: CreateBuffer,
    destroy_buffer: DestroyBuffer,
    buffer_mem_req: BufferMemReq,
    allocate_memory: AllocateMemory,
    free_memory: FreeMemory,
    bind_buffer_memory: BindBufferMemory,
    map_memory: MapMemory,
    unmap_memory: UnmapMemory,
    create_shader_module: CreateShaderModule,
    destroy_shader_module: DestroyShaderModule,
    create_set_layout: CreateSetLayout,
    destroy_set_layout: DestroySetLayout,
    create_pipeline_layout: CreatePipelineLayout,
    destroy_pipeline_layout: DestroyPipelineLayout,
    create_compute_pipelines: CreateComputePipelines,
    destroy_pipeline: DestroyPipeline,
    create_descriptor_pool: CreateDescriptorPool,
    destroy_descriptor_pool: DestroyDescriptorPool,
    reset_descriptor_pool: ResetDescriptorPool,
    allocate_sets: AllocateSets,
    update_sets: UpdateSets,
    create_command_pool: CreateCommandPool,
    destroy_command_pool: DestroyCommandPool,
    allocate_command_buffers: AllocateCommandBuffers,
    begin_command_buffer: BeginCommandBuffer,
    end_command_buffer: EndCommandBuffer,
    reset_command_buffer: ResetCommandBuffer,
    cmd_bind_pipeline: CmdBindPipeline,
    cmd_bind_sets: CmdBindSets,
    cmd_push_constants: CmdPushConstants,
    cmd_dispatch: CmdDispatch,
    cmd_barrier: CmdBarrier,
    cmd_copy_buffer: CmdCopyBuffer,
    queue_submit: QueueSubmit,
    create_fence: CreateFence,
    destroy_fence: DestroyFence,
    reset_fences: ResetFences,
    wait_for_fences: WaitForFences,
    get_fence_status: GetFenceStatus,
    device_wait_idle: DeviceWaitIdle,
}

fn cstr(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}

unsafe fn raw(gpa: Gpa, h: Handle, name: &str) -> Handle {
    let n = cstr(name);
    gpa(h, n.as_ptr() as *const c_char)
}

macro_rules! api {
    ($gpa:expr, $dev:expr, $($f:ident = $n:literal : $t:ty),* $(,)?) => {{
        $(
            let p = raw($gpa, $dev, $n);
            if p.is_null() {
                return Err(format!("this Vulkan device does not export {}", $n));
            }
            let $f: $t = std::mem::transmute(p);
        )*
        Api { $($f),* }
    }};
}

/// A buffer and the memory bound to it. `host` is null for device-local weight
/// buffers, which the CPU never touches after the upload copy.
struct Buf {
    buffer: Handle,
    mem: Handle,
    size: u64,
    host: *mut u8,
    /// Which half format the bytes are in. The dtype belongs to the weight
    /// buffer, so `gemv_half` reads it here instead of every dispatch site
    /// having to thread a flag through the layer structs.
    f16: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    GemvBf16 = 0,
    GemvQ4 = 1,
    RmsNorm = 2,
    SiluMul = 3,
    Axpy = 4,
    GemvF16 = 5,
    GdnConv = 6,
    GdnRec = 7,
    MoeReduce = 8,
    GemvBf16Batch = 9,
    GemvF16Batch = 10,
    RmsNormBatch = 11,
    SiluMulBatch = 12,
    AxpyBatch = 13,
}

const SPV: [&[u8]; 14] = [
    include_bytes!("shaders/gemv_bf16.spv"),
    include_bytes!("shaders/gemv_q4.spv"),
    include_bytes!("shaders/rmsnorm.spv"),
    include_bytes!("shaders/silu_mul.spv"),
    include_bytes!("shaders/axpy.spv"),
    include_bytes!("shaders/gemv_f16.spv"),
    include_bytes!("shaders/gdn_conv.spv"),
    include_bytes!("shaders/gdn_rec.spv"),
    include_bytes!("shaders/moe_reduce.spv"),
    include_bytes!("shaders/gemv_bf16_batch.spv"),
    include_bytes!("shaders/gemv_f16_batch.spv"),
    include_bytes!("shaders/rmsnorm_batch.spv"),
    include_bytes!("shaders/silu_mul_batch.spv"),
    include_bytes!("shaders/axpy_batch.spv"),
];

pub struct VkGpu {
    vk: Vulkan,
    api: Api,
    device: Handle,
    queue: Handle,
    pub info: Device,
    mem_types: Vec<(u32, u32)>,
    bufs: Vec<Buf>,
    set_layout: Handle,
    pipe_layout: Handle,
    pipes: [Handle; SPV.len()],
    pool: Handle,
    cmd_pool: Handle,
    cb: Handle,
    fence: Handle,
    scratch: usize,
    dummy: usize,
    staging: usize,
    scratch_floats: usize,
    pub submitted: u64,
    pub dispatches: u64,
    /// Wall time spent blocked on the fence, and how many times. A decode layer
    /// is several short batches, so this is where submit/wait overhead shows up
    /// separately from the dispatches themselves.
    pub sync_ms: f64,
    pub syncs: u64,
}

// The handles are only ever used from the thread that owns the VkGpu; the CPU
// workers that run alongside a submitted batch never touch it.
unsafe impl Send for VkGpu {}

impl VkGpu {
    /// Open the ranked device `index` (the same index `fennec probe` prints).
    pub fn open(index: usize) -> Result<VkGpu> {
        let mut vk = Vulkan::load();
        if !vk.error.is_empty() {
            return Err(format!("no Vulkan loader: {}", vk.error));
        }
        let devs = vk.devices();
        if devs.is_empty() {
            return Err(if vk.error.is_empty() {
                "no Vulkan device".into()
            } else {
                vk.error.clone()
            });
        }
        let info = devs
            .get(index)
            .ok_or_else(|| format!("device {index} of {} not present", devs.len()))?
            .clone();
        let phys = vk.phys(index).ok_or("physical device handle lost")?;
        let inst = vk.handle();
        unsafe {
            let gpa: Gpa = {
                let p = raw(
                    std::mem::transmute::<Handle, Gpa>(
                        vk.proc(std::ptr::null_mut(), "vkGetInstanceProcAddr")
                            .ok_or("vkGetInstanceProcAddr missing")?,
                    ),
                    inst,
                    "vkGetDeviceProcAddr",
                );
                if p.is_null() {
                    return Err("vkGetDeviceProcAddr missing".into());
                }
                std::mem::transmute(p)
            };
            type GetFamilies = unsafe extern "system" fn(Handle, *mut u32, *mut QueueFamily);
            let get_families: GetFamilies = std::mem::transmute(
                vk.proc(inst, "vkGetPhysicalDeviceQueueFamilyProperties")
                    .ok_or("vkGetPhysicalDeviceQueueFamilyProperties missing")?,
            );
            let mut nf = 0u32;
            get_families(phys, &mut nf, std::ptr::null_mut());
            let mut fams = vec![QueueFamily::default(); nf as usize];
            get_families(phys, &mut nf, fams.as_mut_ptr());
            // A compute-only family keeps graphics work off the queue; fall back
            // to the universal family, which every device has.
            let family = fams
                .iter()
                .position(|f| f.flags & QUEUE_COMPUTE != 0 && f.flags & QUEUE_GRAPHICS == 0)
                .or_else(|| fams.iter().position(|f| f.flags & QUEUE_COMPUTE != 0))
                .ok_or("no compute queue family")? as u32;

            let priority = 1.0f32;
            let qi = QueueCreateInfo {
                s_type: ST_DEVICE_QUEUE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                family,
                count: 1,
                priorities: &priority,
            };
            // A portability driver (MoltenVK on macOS) advertises this extension
            // and the spec requires enabling it when supported. Everywhere else
            // the list stays empty and this is the same zero-extension device.
            let subset = b"VK_KHR_portability_subset\0";
            let exts: Vec<*const c_char> =
                if info.extensions.iter().any(|e| e == "VK_KHR_portability_subset") {
                    vec![subset.as_ptr() as *const c_char]
                } else {
                    Vec::new()
                };
            let di = DeviceCreateInfo {
                s_type: ST_DEVICE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                queue_info_count: 1,
                queue_infos: &qi,
                layer_count: 0,
                layers: std::ptr::null(),
                ext_count: exts.len() as u32,
                exts: if exts.is_empty() { std::ptr::null() } else { exts.as_ptr() },
                features: std::ptr::null(),
            };
            type CreateDevice =
                unsafe extern "system" fn(Handle, *const DeviceCreateInfo, *const c_void, *mut Handle) -> i32;
            let create_device: CreateDevice = std::mem::transmute(
                vk.proc(inst, "vkCreateDevice").ok_or("vkCreateDevice missing")?,
            );
            let mut device: Handle = std::ptr::null_mut();
            let r = create_device(phys, &di, std::ptr::null(), &mut device);
            if r != 0 {
                return Err(format!("vkCreateDevice on {}: {r}", info.name));
            }
            type GetQueue = unsafe extern "system" fn(Handle, u32, u32, *mut Handle);
            let get_queue: GetQueue =
                std::mem::transmute(raw(gpa, device, "vkGetDeviceQueue"));
            let mut queue: Handle = std::ptr::null_mut();
            get_queue(device, family, 0, &mut queue);

            let api = api!(gpa, device,
                create_buffer = "vkCreateBuffer": CreateBuffer,
                destroy_buffer = "vkDestroyBuffer": DestroyBuffer,
                buffer_mem_req = "vkGetBufferMemoryRequirements": BufferMemReq,
                allocate_memory = "vkAllocateMemory": AllocateMemory,
                free_memory = "vkFreeMemory": FreeMemory,
                bind_buffer_memory = "vkBindBufferMemory": BindBufferMemory,
                map_memory = "vkMapMemory": MapMemory,
                unmap_memory = "vkUnmapMemory": UnmapMemory,
                create_shader_module = "vkCreateShaderModule": CreateShaderModule,
                destroy_shader_module = "vkDestroyShaderModule": DestroyShaderModule,
                create_set_layout = "vkCreateDescriptorSetLayout": CreateSetLayout,
                destroy_set_layout = "vkDestroyDescriptorSetLayout": DestroySetLayout,
                create_pipeline_layout = "vkCreatePipelineLayout": CreatePipelineLayout,
                destroy_pipeline_layout = "vkDestroyPipelineLayout": DestroyPipelineLayout,
                create_compute_pipelines = "vkCreateComputePipelines": CreateComputePipelines,
                destroy_pipeline = "vkDestroyPipeline": DestroyPipeline,
                create_descriptor_pool = "vkCreateDescriptorPool": CreateDescriptorPool,
                destroy_descriptor_pool = "vkDestroyDescriptorPool": DestroyDescriptorPool,
                reset_descriptor_pool = "vkResetDescriptorPool": ResetDescriptorPool,
                allocate_sets = "vkAllocateDescriptorSets": AllocateSets,
                update_sets = "vkUpdateDescriptorSets": UpdateSets,
                create_command_pool = "vkCreateCommandPool": CreateCommandPool,
                destroy_command_pool = "vkDestroyCommandPool": DestroyCommandPool,
                allocate_command_buffers = "vkAllocateCommandBuffers": AllocateCommandBuffers,
                begin_command_buffer = "vkBeginCommandBuffer": BeginCommandBuffer,
                end_command_buffer = "vkEndCommandBuffer": EndCommandBuffer,
                reset_command_buffer = "vkResetCommandBuffer": ResetCommandBuffer,
                cmd_bind_pipeline = "vkCmdBindPipeline": CmdBindPipeline,
                cmd_bind_sets = "vkCmdBindDescriptorSets": CmdBindSets,
                cmd_push_constants = "vkCmdPushConstants": CmdPushConstants,
                cmd_dispatch = "vkCmdDispatch": CmdDispatch,
                cmd_barrier = "vkCmdPipelineBarrier": CmdBarrier,
                cmd_copy_buffer = "vkCmdCopyBuffer": CmdCopyBuffer,
                queue_submit = "vkQueueSubmit": QueueSubmit,
                create_fence = "vkCreateFence": CreateFence,
                destroy_fence = "vkDestroyFence": DestroyFence,
                reset_fences = "vkResetFences": ResetFences,
                wait_for_fences = "vkWaitForFences": WaitForFences,
                get_fence_status = "vkGetFenceStatus": GetFenceStatus,
                device_wait_idle = "vkDeviceWaitIdle": DeviceWaitIdle,
            );

            type GetMemProps = unsafe extern "system" fn(Handle, *mut MemProps);
            let get_mem: GetMemProps = std::mem::transmute(
                vk.proc(inst, "vkGetPhysicalDeviceMemoryProperties")
                    .ok_or("vkGetPhysicalDeviceMemoryProperties missing")?,
            );
            let mut mp = MemProps {
                type_count: 0,
                types: [MemType::default(); 32],
                heap_count: 0,
                heaps: [MemHeap::default(); 16],
            };
            get_mem(phys, &mut mp);
            let mem_types = mp.types[..mp.type_count.min(32) as usize]
                .iter()
                .map(|t| (t.flags, t.heap))
                .collect();

            let bindings: Vec<DescriptorSetLayoutBinding> = (0..5)
                .map(|b| DescriptorSetLayoutBinding {
                    binding: b,
                    kind: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    count: 1,
                    stages: SHADER_STAGE_COMPUTE,
                    immutable: std::ptr::null(),
                })
                .collect();
            let slci = DescriptorSetLayoutCreateInfo {
                s_type: ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                binding_count: bindings.len() as u32,
                bindings: bindings.as_ptr(),
            };
            let mut set_layout: Handle = std::ptr::null_mut();
            if (api.create_set_layout)(device, &slci, std::ptr::null(), &mut set_layout) != 0 {
                return Err("vkCreateDescriptorSetLayout failed".into());
            }
            let push = PushConstantRange { stages: SHADER_STAGE_COMPUTE, offset: 0, size: PUSH_BYTES };
            let plci = PipelineLayoutCreateInfo {
                s_type: ST_PIPELINE_LAYOUT_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                set_layout_count: 1,
                set_layouts: &set_layout,
                push_count: 1,
                push_ranges: &push,
            };
            let mut pipe_layout: Handle = std::ptr::null_mut();
            if (api.create_pipeline_layout)(device, &plci, std::ptr::null(), &mut pipe_layout) != 0 {
                return Err("vkCreatePipelineLayout failed".into());
            }

            let entry = cstr("main");
            let mut pipes = [std::ptr::null_mut(); SPV.len()];
            for (i, spv) in SPV.iter().enumerate() {
                // include_bytes! is byte-aligned; the driver reads words.
                let words: Vec<u32> = spv
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let smci = ShaderModuleCreateInfo {
                    s_type: ST_SHADER_MODULE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: 0,
                    code_size: words.len() * 4,
                    code: words.as_ptr(),
                };
                let mut module: Handle = std::ptr::null_mut();
                if (api.create_shader_module)(device, &smci, std::ptr::null(), &mut module) != 0 {
                    return Err(format!("shader {i}: vkCreateShaderModule failed"));
                }
                let cpci = ComputePipelineCreateInfo {
                    s_type: ST_COMPUTE_PIPELINE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: 0,
                    stage: PipelineShaderStageCreateInfo {
                        s_type: ST_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        p_next: std::ptr::null(),
                        flags: 0,
                        stage: SHADER_STAGE_COMPUTE,
                        module,
                        name: entry.as_ptr() as *const c_char,
                        specialization: std::ptr::null(),
                    },
                    layout: pipe_layout,
                    base_handle: std::ptr::null_mut(),
                    base_index: 0,
                };
                let r = (api.create_compute_pipelines)(
                    device,
                    std::ptr::null_mut(),
                    1,
                    &cpci,
                    std::ptr::null(),
                    &mut pipes[i],
                );
                (api.destroy_shader_module)(device, module, std::ptr::null());
                if r != 0 {
                    return Err(format!("shader {i}: vkCreateComputePipelines {r}"));
                }
            }

            let size = DescriptorPoolSize { kind: DESCRIPTOR_TYPE_STORAGE_BUFFER, count: MAX_SETS * 4 };
            let dpci = DescriptorPoolCreateInfo {
                s_type: ST_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                max_sets: MAX_SETS,
                size_count: 1,
                sizes: &size,
            };
            let mut pool: Handle = std::ptr::null_mut();
            if (api.create_descriptor_pool)(device, &dpci, std::ptr::null(), &mut pool) != 0 {
                return Err("vkCreateDescriptorPool failed".into());
            }
            let cpci = CommandPoolCreateInfo {
                s_type: ST_COMMAND_POOL_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: COMMAND_POOL_RESET_COMMAND_BUFFER,
                family,
            };
            let mut cmd_pool: Handle = std::ptr::null_mut();
            if (api.create_command_pool)(device, &cpci, std::ptr::null(), &mut cmd_pool) != 0 {
                return Err("vkCreateCommandPool failed".into());
            }
            let cbai = CommandBufferAllocateInfo {
                s_type: ST_COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                pool: cmd_pool,
                level: 0,
                count: 1,
            };
            let mut cb: Handle = std::ptr::null_mut();
            if (api.allocate_command_buffers)(device, &cbai, &mut cb) != 0 {
                return Err("vkAllocateCommandBuffers failed".into());
            }
            let fci = FenceCreateInfo { s_type: ST_FENCE_CREATE_INFO, p_next: std::ptr::null(), flags: 0 };
            let mut fence: Handle = std::ptr::null_mut();
            if (api.create_fence)(device, &fci, std::ptr::null(), &mut fence) != 0 {
                return Err("vkCreateFence failed".into());
            }

            let mut gpu = VkGpu {
                vk,
                api,
                device,
                queue,
                info,
                mem_types,
                bufs: Vec::new(),
                set_layout,
                pipe_layout,
                pipes,
                pool,
                cmd_pool,
                cb,
                fence,
                scratch: usize::MAX,
                dummy: usize::MAX,
                staging: usize::MAX,
                scratch_floats: 0,
                submitted: 0,
                dispatches: 0,
                sync_ms: 0.0,
                syncs: 0,
            };
            gpu.dummy = gpu.alloc_host(16)?;
            gpu.reserve(SCRATCH_FLOATS)?;
            Ok(gpu)
        }
    }

    pub fn name(&self) -> &str {
        &self.info.name
    }

    fn mem_type(&self, bits: u32, want: u32, avoid: u32) -> Option<u32> {
        for i in 0..self.mem_types.len() {
            let f = self.mem_types[i].0;
            if bits & (1 << i) != 0 && f & want == want && (avoid == 0 || f & avoid == 0) {
                return Some(i as u32);
            }
        }
        // An iGPU's only device-local type is also host-visible, so the strict
        // preference above finds nothing there and this is the path it takes.
        if avoid != 0 {
            for i in 0..self.mem_types.len() {
                let f = self.mem_types[i].0;
                if bits & (1 << i) != 0 && f & want == want {
                    return Some(i as u32);
                }
            }
        }
        None
    }

    fn new_buf(&mut self, size: u64, usage: u32, want: u32, avoid: u32) -> Result<usize> {
        let size = size.max(16);
        unsafe {
            let bci = BufferCreateInfo {
                s_type: ST_BUFFER_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                size,
                usage,
                sharing: 0,
                family_count: 0,
                families: std::ptr::null(),
            };
            let mut buffer: Handle = std::ptr::null_mut();
            if (self.api.create_buffer)(self.device, &bci, std::ptr::null(), &mut buffer) != 0 {
                return Err(format!("vkCreateBuffer({size} bytes) failed"));
            }
            let mut req = MemRequirements::default();
            (self.api.buffer_mem_req)(self.device, buffer, &mut req);
            let ti = match self.mem_type(req.type_bits, want, avoid) {
                Some(t) => t,
                None => {
                    (self.api.destroy_buffer)(self.device, buffer, std::ptr::null());
                    return Err(format!("no memory type for {size} bytes (want {want:#x})"));
                }
            };
            let mai = MemoryAllocateInfo {
                s_type: ST_MEMORY_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                size: req.size,
                type_index: ti,
            };
            let mut mem: Handle = std::ptr::null_mut();
            if (self.api.allocate_memory)(self.device, &mai, std::ptr::null(), &mut mem) != 0 {
                (self.api.destroy_buffer)(self.device, buffer, std::ptr::null());
                return Err(format!("vkAllocateMemory({} bytes) failed: out of device memory", req.size));
            }
            if (self.api.bind_buffer_memory)(self.device, buffer, mem, 0) != 0 {
                (self.api.free_memory)(self.device, mem, std::ptr::null());
                (self.api.destroy_buffer)(self.device, buffer, std::ptr::null());
                return Err("vkBindBufferMemory failed".into());
            }
            let mut host: *mut u8 = std::ptr::null_mut();
            if want & MEMORY_PROPERTY_HOST_VISIBLE != 0 {
                let mut p: *mut c_void = std::ptr::null_mut();
                if (self.api.map_memory)(self.device, mem, 0, req.size, 0, &mut p) != 0 {
                    (self.api.free_memory)(self.device, mem, std::ptr::null());
                    (self.api.destroy_buffer)(self.device, buffer, std::ptr::null());
                    return Err("vkMapMemory failed".into());
                }
                host = p as *mut u8;
            }
            self.bufs.push(Buf { buffer, mem, size: req.size, host, f16: false });
            Ok(self.bufs.len() - 1)
        }
    }

    fn alloc_host(&mut self, size: usize) -> Result<usize> {
        self.new_buf(
            size as u64,
            BUFFER_USAGE_STORAGE_BUFFER | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
            MEMORY_PROPERTY_HOST_VISIBLE | MEMORY_PROPERTY_HOST_COHERENT,
            0,
        )
    }

    fn alloc_device(&mut self, size: usize) -> Result<usize> {
        self.new_buf(
            size as u64,
            BUFFER_USAGE_STORAGE_BUFFER | BUFFER_USAGE_TRANSFER_DST | BUFFER_USAGE_TRANSFER_SRC,
            MEMORY_PROPERTY_DEVICE_LOCAL,
            MEMORY_PROPERTY_HOST_VISIBLE,
        )
    }

    /// Grow the host-visible activation arena so `floats` elements fit. One
    /// buffer holds every intermediate, so an op can read what the previous op
    /// wrote without a round trip through the CPU.
    pub fn reserve(&mut self, floats: usize) -> Result<()> {
        if floats <= self.scratch_floats {
            return Ok(());
        }
        let s = self.alloc_host(floats * 4)?;
        if self.scratch != usize::MAX {
            self.free(self.scratch);
        }
        self.scratch = s;
        self.scratch_floats = floats;
        Ok(())
    }

    pub fn arena(&self) -> usize {
        self.scratch_floats
    }

    fn free(&mut self, id: usize) {
        if id >= self.bufs.len() || self.bufs[id].buffer.is_null() {
            return;
        }
        unsafe {
            let b = &self.bufs[id];
            if !b.host.is_null() {
                (self.api.unmap_memory)(self.device, b.mem);
            }
            (self.api.destroy_buffer)(self.device, b.buffer, std::ptr::null());
            (self.api.free_memory)(self.device, b.mem, std::ptr::null());
        }
        self.bufs[id].buffer = std::ptr::null_mut();
        self.bufs[id].host = std::ptr::null_mut();
    }

    fn begin(&mut self) -> Result<()> {
        unsafe {
            (self.api.reset_command_buffer)(self.cb, 0);
            let bi = CommandBufferBeginInfo {
                s_type: ST_COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                inheritance: std::ptr::null(),
            };
            if (self.api.begin_command_buffer)(self.cb, &bi) != 0 {
                return Err("vkBeginCommandBuffer failed".into());
            }
        }
        Ok(())
    }

    fn barrier(&mut self, src: u32, dst: u32, sa: u32, da: u32) {
        let b = MemoryBarrier { s_type: ST_MEMORY_BARRIER, p_next: std::ptr::null(), src_access: sa, dst_access: da };
        unsafe {
            (self.api.cmd_barrier)(self.cb, src, dst, 0, 1, &b, 0, std::ptr::null(), 0, std::ptr::null());
        }
    }

    fn flush(&mut self) -> Result<()> {
        unsafe {
            if (self.api.end_command_buffer)(self.cb) != 0 {
                return Err("vkEndCommandBuffer failed".into());
            }
            let si = SubmitInfo {
                s_type: ST_SUBMIT_INFO,
                p_next: std::ptr::null(),
                wait_count: 0,
                waits: std::ptr::null(),
                wait_stages: std::ptr::null(),
                cb_count: 1,
                cbs: &self.cb,
                signal_count: 0,
                signals: std::ptr::null(),
            };
            (self.api.reset_fences)(self.device, 1, &self.fence);
            if (self.api.queue_submit)(self.queue, 1, &si, self.fence) != 0 {
                return Err("vkQueueSubmit failed".into());
            }
            self.submitted += 1;
            self.wait()
        }
    }

    /// Block until the last submitted batch is finished and its writes are
    /// visible to the host. The counterpart to `VkBatch::submit` for a caller that
    /// does CPU work in between.
    pub fn wait(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let r = self.wait_fence();
        self.sync_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.syncs += 1;
        r
    }

    fn wait_fence(&mut self) -> Result<()> {
        unsafe {
            // A decode layer is three short batches, and vkWaitForFences parks
            // the thread in the kernel until an interrupt wakes it - that wake
            // measured longer than the dispatch itself. Poll first for a few
            // hundred microseconds, which covers a dense projection, and only
            // then block, so a long upload still sleeps instead of spinning.
            let t = std::time::Instant::now();
            loop {
                let r = (self.api.get_fence_status)(self.device, self.fence);
                if r == 0 {
                    return Ok(());
                }
                if r != 1 {
                    return Err(format!("vkGetFenceStatus: {r} (device lost)"));
                }
                if t.elapsed().as_micros() >= 400 {
                    break;
                }
                std::hint::spin_loop();
            }
            // 30 s: a cold lm_head upload on a shared iGPU can be slow, but a
            // hang should surface as an error rather than a frozen process.
            let r = (self.api.wait_for_fences)(self.device, 1, &self.fence, 1, 30_000_000_000);
            if r != 0 {
                return Err(format!("vkWaitForFences: {r} (device lost or timeout)"));
            }
        }
        Ok(())
    }

    /// Copy bytes into a device-local buffer, in bounded chunks so a 1 GiB
    /// lm_head never needs a 1 GiB staging allocation.
    ///
    /// Each chunk is submitted and waited on before the next host write: the
    /// staging buffer is one block of memory, and recording sixteen copies
    /// against it before submitting would have the device read whatever the
    /// host wrote last, sixteen times.
    pub fn upload(&mut self, bytes: &[u8]) -> Result<usize> {
        self.upload_half(bytes, false)
    }

    /// The same copy, tagged with the half format the bytes are in. f16 and
    /// bf16 are both two bytes per weight but decode differently, and the tag
    /// is what lets one `gemv_half` call site serve either checkpoint.
    pub fn upload_half(&mut self, bytes: &[u8], f16: bool) -> Result<usize> {
        let dst = self.alloc_half(bytes.len(), f16)?;
        self.write_buf(dst, 0, bytes)?;
        Ok(dst)
    }

    /// FQ4 weights as two device buffers: nibbles and f32 scales.
    pub fn upload_q4(&mut self, q: &Q4) -> Result<(usize, usize)> {
        if q.cols % 8 != 0 || q.group % 8 != 0 {
            return Err(format!("the q4 shader needs cols and group divisible by 8, got {} / {}", q.cols, q.group));
        }
        if q.signed {
            return Err("the q4 shader reads offset-binary nibbles; a borrowed pack must be flipped first".into());
        }
        // The shader indexes scales by cols/group with floor division, so a
        // short final group would read the next row's scale. The CPU kernel
        // handles partial groups; refuse to diverge from it silently.
        if q.cols % q.group != 0 {
            return Err(format!("the q4 shader needs cols divisible by group, got {} / {}", q.cols, q.group));
        }
        let w = self.upload(&q.data)?;
        let mut sb = Vec::with_capacity(q.scale.len() * 4);
        for s in &q.scale {
            sb.extend_from_slice(&s.to_le_bytes());
        }
        let s = self.upload(&sb)?;
        Ok((w, s))
    }

    /// An empty device buffer tagged with the half format its bytes will be in.
    /// Filling it with `write_buf` lets several tensors share one allocation, so
    /// projections that consume the same x become a single dispatch.
    pub fn alloc_half(&mut self, bytes: usize, f16: bool) -> Result<usize> {
        let dst = self.alloc_device(bytes)?;
        self.bufs[dst].f16 = f16;
        Ok(dst)
    }

    /// Overwrite part of an existing device buffer in place. `upload` always
    /// allocates, which is wrong for persistent recurrent state: the kernels
    /// hold the buffer id across steps, so the bytes have to land in the same
    /// buffer.
    pub fn write_buf(&mut self, id: usize, at: usize, bytes: &[u8]) -> Result<()> {
        if at + bytes.len() > self.bufs[id].size as usize {
            return Err(format!("write_buf: {} bytes at {at} into a {}-byte buffer", bytes.len(), self.bufs[id].size));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if self.staging == usize::MAX {
            self.staging = self.alloc_host(STAGING_BYTES)?;
        }
        let cap = self.bufs[self.staging].size as usize;
        let mut done = 0usize;
        while done < bytes.len() {
            let n = cap.min(bytes.len() - done);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(done), self.bufs[self.staging].host, n);
            }
            self.begin()?;
            unsafe {
                let cp = BufferCopy { src: 0, dst: (at + done) as u64, size: n as u64 };
                (self.api.cmd_copy_buffer)(self.cb, self.bufs[self.staging].buffer, self.bufs[id].buffer, 1, &cp);
            }
            self.barrier(PIPELINE_STAGE_TRANSFER, PIPELINE_STAGE_COMPUTE_SHADER, ACCESS_TRANSFER_WRITE, ACCESS_SHADER_READ | ACCESS_SHADER_WRITE);
            self.flush()?;
            done += n;
        }
        Ok(())
    }

    /// The reverse leg: device buffer to host, staged in bounded chunks.
    pub fn read_buf(&mut self, id: usize, out: &mut [u8]) -> Result<()> {
        if out.len() > self.bufs[id].size as usize {
            return Err(format!("read_buf: {} bytes out of a {}-byte buffer", out.len(), self.bufs[id].size));
        }
        if out.is_empty() {
            return Ok(());
        }
        if self.staging == usize::MAX {
            self.staging = self.alloc_host(STAGING_BYTES)?;
        }
        let cap = self.bufs[self.staging].size as usize;
        let mut off = 0usize;
        while off < out.len() {
            let n = cap.min(out.len() - off);
            self.begin()?;
            unsafe {
                let cp = BufferCopy { src: off as u64, dst: 0, size: n as u64 };
                (self.api.cmd_copy_buffer)(self.cb, self.bufs[id].buffer, self.bufs[self.staging].buffer, 1, &cp);
            }
            self.barrier(PIPELINE_STAGE_TRANSFER, PIPELINE_STAGE_HOST, ACCESS_TRANSFER_WRITE, ACCESS_HOST_READ);
            self.flush()?;
            unsafe {
                std::ptr::copy_nonoverlapping(self.bufs[self.staging].host, out.as_mut_ptr().add(off), n);
            }
            off += n;
        }
        Ok(())
    }

    pub fn write(&mut self, off: usize, v: &[f32]) {
        assert!(off + v.len() <= self.scratch_floats, "activation arena too small");
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), (self.bufs[self.scratch].host as *mut f32).add(off), v.len());
        }
    }

    pub fn read(&self, off: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        self.read_into(off, &mut out);
        out
    }

    pub fn read_into(&self, off: usize, out: &mut [f32]) {
        assert!(off + out.len() <= self.scratch_floats, "activation arena too small");
        unsafe {
            std::ptr::copy_nonoverlapping((self.bufs[self.scratch].host as *const f32).add(off), out.as_mut_ptr(), out.len());
        }
    }

    /// Start a command buffer. Every op records one descriptor set, one push and
    /// one dispatch (several for a tall gemv), separated by barriers.
    pub fn batch(&mut self) -> Result<VkBatch<'_>> {
        unsafe {
            (self.api.reset_descriptor_pool)(self.device, self.pool, 0);
        }
        self.begin()?;
        Ok(VkBatch { gpu: self, ops: 0, live: true })
    }

    fn set(&mut self, bufs: [usize; 4]) -> Result<Handle> {
        unsafe {
            let dsai = DescriptorSetAllocateInfo {
                s_type: ST_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                pool: self.pool,
                count: 1,
                layouts: &self.set_layout,
            };
            let mut set: Handle = std::ptr::null_mut();
            if (self.api.allocate_sets)(self.device, &dsai, &mut set) != 0 {
                return Err("descriptor pool exhausted".into());
            }
            let mut infos: Vec<DescriptorBufferInfo> = bufs
                .iter()
                .map(|&b| DescriptorBufferInfo {
                    buffer: self.bufs[b].buffer,
                    offset: 0,
                    range: WHOLE_SIZE,
                })
                .collect();
            // Binding 4 is the activation buffer (bufs[1]) again, viewed as
            // vec4 by the half-precision gemv so it loads x four floats at a
            // time like the CUDA kernel. Every other shader ignores it.
            infos.push(DescriptorBufferInfo {
                buffer: self.bufs[bufs[1]].buffer,
                offset: 0,
                range: WHOLE_SIZE,
            });
            let writes: Vec<WriteDescriptorSet> = (0..5)
                .map(|i| WriteDescriptorSet {
                    s_type: ST_WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    set,
                    binding: i as u32,
                    element: 0,
                    count: 1,
                    kind: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    images: std::ptr::null(),
                    buffers: &infos[i],
                    texels: std::ptr::null(),
                })
                .collect();
            (self.api.update_sets)(self.device, 5, writes.as_ptr(), 0, std::ptr::null());
            Ok(set)
        }
    }

    fn record(&mut self, op: Op, bufs: [usize; 4], push: &[u32], groups: u32, rows: u32, tall: bool) -> Result<()> {
        self.record_inner(op, bufs, push, groups, rows, tall, true)
    }

    /// Like `record` but without the trailing pipeline barrier. Independent
    /// dispatches (one expert's gemv vs another's) can then overlap on the
    /// device; the caller inserts a barrier at the next real dependency.
    fn record_nb(&mut self, op: Op, bufs: [usize; 4], push: &[u32], groups: u32, rows: u32, tall: bool) -> Result<()> {
        self.record_inner(op, bufs, push, groups, rows, tall, false)
    }

    fn record_inner(&mut self, op: Op, bufs: [usize; 4], push: &[u32], groups: u32, rows: u32, tall: bool, barrier: bool) -> Result<()> {
        let set = self.set(bufs)?;
        let mut pc = [0u32; 8];
        pc[..push.len()].copy_from_slice(push);
        unsafe {
            (self.api.cmd_bind_pipeline)(self.cb, PIPELINE_BIND_POINT_COMPUTE, self.pipes[op as usize]);
            (self.api.cmd_bind_sets)(self.cb, PIPELINE_BIND_POINT_COMPUTE, self.pipe_layout, 0, 1, &set, 0, std::ptr::null());
            if !tall {
                (self.api.cmd_push_constants)(self.cb, self.pipe_layout, SHADER_STAGE_COMPUTE, 0, PUSH_BYTES, pc.as_ptr() as *const c_void);
                (self.api.cmd_dispatch)(self.cb, groups, 1, 1);
                self.dispatches += 1;
            } else {
                let mut base = 0u32;
                while base < rows {
                    let n = MAX_GROUPS.min(rows - base);
                    pc[push.len()] = base;
                    (self.api.cmd_push_constants)(self.cb, self.pipe_layout, SHADER_STAGE_COMPUTE, 0, PUSH_BYTES, pc.as_ptr() as *const c_void);
                    (self.api.cmd_dispatch)(self.cb, n, 1, 1);
                    self.dispatches += 1;
                    base += n;
                }
            }
            if barrier {
                self.barrier(PIPELINE_STAGE_COMPUTE_SHADER, PIPELINE_STAGE_COMPUTE_SHADER, ACCESS_SHADER_WRITE, ACCESS_SHADER_READ);
            }
        }
        Ok(())
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        unsafe {
            if (self.api.device_wait_idle)(self.device) != 0 {
                return Err("vkDeviceWaitIdle failed".into());
            }
        }
        Ok(())
    }

    pub fn bytes_held(&self) -> u64 {
        self.bufs.iter().map(|b| if b.buffer.is_null() { 0 } else { b.size }).sum()
    }

    /// Zero the telemetry. Load uploads every expert in chunks and each chunk
    /// is a submit plus a wait, so counters read straight after load describe
    /// the load, not the decode. Call this once the model is resident.
    pub fn reset_counters(&mut self) {
        self.submitted = 0;
        self.dispatches = 0;
        self.sync_ms = 0.0;
        self.syncs = 0;
    }
}

/// A recorded command buffer. `submit` hands it to the queue and returns while
/// the device is still working; `wait` joins it. The gap between the two is
/// where the CPU does its share of the same token.
pub struct VkBatch<'a> {
    gpu: &'a mut VkGpu,
    pub ops: usize,
    live: bool,
}

impl<'a> VkBatch<'a> {
    /// y = W x for a two-byte-per-weight matrix, in whichever half format the
    /// buffer was uploaded as. The shaders are identical apart from the unpack.
    pub fn gemv_half(&mut self, w: usize, rows: usize, cols: usize, x_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        assert!(cols % 2 == 0, "half pairs need an even column count");
        let s = self.gpu.scratch;
        let op = if self.gpu.bufs[w].f16 { Op::GemvF16 } else { Op::GemvBf16 };
        // The shader takes four rows per workgroup, and row_base counts
        // workgroups, so both the group count and the tall-split bound shrink
        // by four. Binding 3 is the weight buffer again, read as uvec4.
        let groups = (rows as u32 + 3) / 4;
        self.gpu.record(op, [w, s, s, w], &[rows as u32, cols as u32, x_off as u32, y_off as u32], groups, groups, true)?;
        self.ops += 1;
        Ok(self)
    }

    pub fn gemv_q4(&mut self, w: usize, sc: usize, rows: usize, cols: usize, group: usize, x_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        self.gpu.record(
            Op::GemvQ4,
            [w, sc, s, s],
            &[rows as u32, cols as u32, group as u32, x_off as u32, y_off as u32],
            rows as u32,
            rows as u32,
            true,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// Expert gemvs recorded without a barrier after each one: every expert in
    /// a phase writes its own arena slot, so they are independent and the device
    /// can run them concurrently. A `barrier` separates the phases.
    pub fn gemv_q4_nb(&mut self, w: usize, sc: usize, rows: usize, cols: usize, group: usize, x_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        self.gpu.record_nb(
            Op::GemvQ4,
            [w, sc, s, s],
            &[rows as u32, cols as u32, group as u32, x_off as u32, y_off as u32],
            rows as u32,
            rows as u32,
            true,
        )?;
        self.ops += 1;
        Ok(self)
    }

    pub fn gemv_half_nb(&mut self, w: usize, rows: usize, cols: usize, x_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        assert!(cols % 2 == 0, "half pairs need an even column count");
        let s = self.gpu.scratch;
        let op = if self.gpu.bufs[w].f16 { Op::GemvF16 } else { Op::GemvBf16 };
        let groups = (rows as u32 + 3) / 4;
        self.gpu.record_nb(op, [w, s, s, w], &[rows as u32, cols as u32, x_off as u32, y_off as u32], groups, groups, true)?;
        self.ops += 1;
        Ok(self)
    }

    pub fn silu_mul_nb(&mut self, n: usize, a_off: usize, b_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record_nb(Op::SiluMul, [s, s, s, d], &[n as u32, a_off as u32, b_off as u32, y_off as u32], n.div_ceil(256) as u32, 1, false)?;
        self.ops += 1;
        Ok(self)
    }

    pub fn barrier(&mut self) -> Result<&mut VkBatch<'a>> {
        self.gpu.barrier(PIPELINE_STAGE_COMPUTE_SHADER, PIPELINE_STAGE_COMPUTE_SHADER, ACCESS_SHADER_WRITE, ACCESS_SHADER_READ);
        Ok(self)
    }

    /// One weighted sum of every expert slot into the accumulator, in pick
    /// order, replacing the per-expert axpy chain and its barriers.
    pub fn moe_reduce(&mut self, n: usize, h: usize, stride: usize, base: usize, out_off: usize, acc_off: usize, w_off: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::MoeReduce,
            [s, s, s, d],
            &[n as u32, h as u32, stride as u32, base as u32, out_off as u32, acc_off as u32, w_off as u32],
            h.div_ceil(256) as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    pub fn rmsnorm(&mut self, n: usize, x_off: usize, w_off: usize, y_off: usize, eps: f32) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::RmsNorm,
            [s, s, s, d],
            &[n as u32, x_off as u32, w_off as u32, y_off as u32, eps.to_bits()],
            1,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    pub fn silu_mul(&mut self, n: usize, a_off: usize, b_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::SiluMul,
            [s, s, s, d],
            &[n as u32, a_off as u32, b_off as u32, y_off as u32],
            n.div_ceil(256) as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    pub fn axpy(&mut self, n: usize, x_off: usize, y_off: usize, w: f32, set: bool) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::Axpy,
            [s, s, d, d],
            &[n as u32, x_off as u32, y_off as u32, w.to_bits(), set as u32],
            n.div_ceil(256) as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    // ── Batched variants: process n_tokens vectors per dispatch ──

    /// Batched y[t] = W x[t] for bf16/f16 weights. One dispatch amortises the
    /// weight read across all n_tokens tokens — the prefill batching win.
    pub fn gemv_half_batch(&mut self, w: usize, rows: usize, cols: usize, n_tokens: usize, x_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        assert!(cols % 2 == 0, "half pairs need an even column count");
        let s = self.gpu.scratch;
        let op = if self.gpu.bufs[w].f16 { Op::GemvF16Batch } else { Op::GemvBf16Batch };
        let groups = (rows as u32 + 3) / 4;
        self.gpu.record(
            op,
            [w, s, s, w],
            &[
                rows as u32,
                cols as u32,
                n_tokens as u32,
                cols as u32,     // x_stride
                rows as u32,     // y_stride
                x_off as u32,
                y_off as u32,
                0,               // row_base (filled by tall-split)
            ],
            groups,
            groups,
            true,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// Batched RMSNorm: one workgroup per token, n floats each.
    pub fn rmsnorm_batch(&mut self, n: usize, n_tokens: usize, x_off: usize, w_off: usize, y_off: usize, eps: f32) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::RmsNormBatch,
            [s, s, s, d],
            &[n as u32, n_tokens as u32, x_off as u32, w_off as u32, y_off as u32, eps.to_bits(), 0, 0],
            n_tokens as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// Batched silu_mul: n elements per token, n_tokens tokens.
    pub fn silu_mul_batch(&mut self, n: usize, n_tokens: usize, a_off: usize, b_off: usize, y_off: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        let total = n * n_tokens;
        self.gpu.record(
            Op::SiluMulBatch,
            [s, s, s, d],
            &[n as u32, n_tokens as u32, a_off as u32, b_off as u32, y_off as u32, 0, 0, 0],
            total.div_ceil(256) as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// Batched axpy: n elements per token, n_tokens tokens.
    pub fn axpy_batch(&mut self, n: usize, n_tokens: usize, x_off: usize, y_off: usize, w: f32, set: bool) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        let total = n * n_tokens;
        self.gpu.record(
            Op::AxpyBatch,
            [s, s, d, d],
            &[n as u32, n_tokens as u32, x_off as u32, y_off as u32, w.to_bits(), set as u32, 0, 0],
            total.div_ceil(256) as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// Depthwise conv + SiLU in place over the qkv projection. `state` is the
    /// persistent rolling window, one thread per channel so it needs no barrier.
    pub fn gdn_conv(&mut self, w: usize, state: usize, base: usize, cd: usize, kernel: usize, hist: usize) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::GdnConv,
            [s, w, state, d],
            &[base as u32, cd as u32, kernel as u32, hist as u32],
            cd.div_ceil(256) as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// The gated delta net, one workgroup per value head. `w` packs
    /// a_log[nv] | dt_bias[nv] | norm_w[dv]; `state` is the head-major recurrence.
    pub fn gdn_rec(&mut self, w: usize, state: usize, base: usize, out_off: usize, kd: usize, nv: usize, dk: usize, dv: usize, rep: usize, eps: f32) -> Result<&mut VkBatch<'a>> {
        let s = self.gpu.scratch;
        let d = self.gpu.dummy;
        self.gpu.record(
            Op::GdnRec,
            [s, w, state, d],
            &[base as u32, out_off as u32, kd as u32, nv as u32, dk as u32, dv as u32, rep as u32, eps.to_bits()],
            nv as u32,
            1,
            false,
        )?;
        self.ops += 1;
        Ok(self)
    }

    /// Hand the batch to the queue and return immediately.
    pub fn submit(&mut self) -> Result<()> {
        unsafe {
            if (self.gpu.api.end_command_buffer)(self.gpu.cb) != 0 {
                return Err("vkEndCommandBuffer failed".into());
            }
            let si = SubmitInfo {
                s_type: ST_SUBMIT_INFO,
                p_next: std::ptr::null(),
                wait_count: 0,
                waits: std::ptr::null(),
                wait_stages: std::ptr::null(),
                cb_count: 1,
                cbs: &self.gpu.cb,
                signal_count: 0,
                signals: std::ptr::null(),
            };
            (self.gpu.api.reset_fences)(self.gpu.device, 1, &self.gpu.fence);
            if (self.gpu.api.queue_submit)(self.gpu.queue, 1, &si, self.gpu.fence) != 0 {
                return Err("vkQueueSubmit failed".into());
            }
            self.gpu.submitted += 1;
        }
        // The command buffer is out of the recording state now, so Drop must not
        // end it a second time. This is what lets a caller submit, do CPU work,
        // and only then wait.
        self.live = false;
        Ok(())
    }

    /// Block until the submitted batch is finished and its writes are visible.
    pub fn wait(&mut self) -> Result<()> {
        self.live = false;
        self.gpu.wait()
    }

    pub fn run(&mut self) -> Result<()> {
        self.submit()?;
        self.wait()
    }
}

impl Drop for VkBatch<'_> {
    fn drop(&mut self) {
        // A batch that was recorded but never submitted would leave the command
        // buffer in the recording state and the next begin() would fail.
        if self.live {
            unsafe {
                let _ = (self.gpu.api.end_command_buffer)(self.gpu.cb);
            }
        }
    }
}

impl Drop for VkGpu {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.api.device_wait_idle)(self.device);
            for i in 0..self.bufs.len() {
                self.free(i);
            }
            (self.api.destroy_fence)(self.device, self.fence, std::ptr::null());
            (self.api.destroy_command_pool)(self.device, self.cmd_pool, std::ptr::null());
            (self.api.destroy_descriptor_pool)(self.device, self.pool, std::ptr::null());
            for p in self.pipes {
                (self.api.destroy_pipeline)(self.device, p, std::ptr::null());
            }
            (self.api.destroy_pipeline_layout)(self.device, self.pipe_layout, std::ptr::null());
            (self.api.destroy_set_layout)(self.device, self.set_layout, std::ptr::null());
            type DestroyDevice = unsafe extern "system" fn(Handle, *const c_void);
            if let Some(p) = self.vk.proc(self.vk.handle(), "vkDestroyDevice") {
                let f: DestroyDevice = std::mem::transmute(p);
                f(self.device, std::ptr::null());
            }
        }
    }
}

/// The device tier, whichever API the vendor policy picked. Both variants
/// expose the same arena, the same float offsets and the same six kernels, so
/// every call site in `model.rs` is backend-agnostic. Metal is not a third
/// variant: it is Vulkan through MoltenVK, and `Backend::is_vulkan` says so.
pub enum Gpu {
    Vk(VkGpu),
    Cuda(crate::cudagpu::CudaGpu),
}

impl Gpu {
    /// Open the ranked device `index` on Vulkan. Kept as the default entry point
    /// so a caller that does not care about the vendor policy still works.
    pub fn open(index: usize) -> Result<Gpu> {
        Ok(Gpu::Vk(VkGpu::open(index)?))
    }

    pub fn open_backend(index: usize, backend: crate::backend::Backend) -> Result<Gpu> {
        match backend {
            crate::backend::Backend::Cuda => Ok(Gpu::Cuda(crate::cudagpu::CudaGpu::open(index)?)),
            _ => Ok(Gpu::Vk(VkGpu::open(index)?)),
        }
    }

    pub fn info(&self) -> &Device {
        match self {
            Gpu::Vk(g) => &g.info,
            Gpu::Cuda(g) => &g.info,
        }
    }

    /// What actually opened, read off the device rather than the request: a
    /// portability driver is Metal even though it runs the Vulkan code path.
    pub fn backend(&self) -> crate::backend::Backend {
        match self {
            Gpu::Vk(g) if g.info.portability() => crate::backend::Backend::Metal,
            Gpu::Vk(_) => crate::backend::Backend::Vulkan,
            Gpu::Cuda(_) => crate::backend::Backend::Cuda,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Gpu::Vk(g) => g.name(),
            Gpu::Cuda(g) => g.name(),
        }
    }

    pub fn reserve(&mut self, floats: usize) -> Result<()> {
        match self {
            Gpu::Vk(g) => g.reserve(floats),
            Gpu::Cuda(g) => g.reserve(floats),
        }
    }

    pub fn arena(&self) -> usize {
        match self {
            Gpu::Vk(g) => g.arena(),
            Gpu::Cuda(g) => g.arena(),
        }
    }

    pub fn upload(&mut self, bytes: &[u8]) -> Result<usize> {
        match self {
            Gpu::Vk(g) => g.upload(bytes),
            Gpu::Cuda(g) => g.upload(bytes),
        }
    }

    pub fn upload_half(&mut self, bytes: &[u8], f16: bool) -> Result<usize> {
        match self {
            Gpu::Vk(g) => g.upload_half(bytes, f16),
            Gpu::Cuda(g) => g.upload_half(bytes, f16),
        }
    }

    pub fn upload_q4(&mut self, q: &Q4) -> Result<(usize, usize)> {
        match self {
            Gpu::Vk(g) => g.upload_q4(q),
            Gpu::Cuda(g) => g.upload_q4(q),
        }
    }

    pub fn alloc_half(&mut self, bytes: usize, f16: bool) -> Result<usize> {
        match self {
            Gpu::Vk(g) => g.alloc_half(bytes, f16),
            Gpu::Cuda(g) => g.alloc_half(bytes, f16),
        }
    }

    pub fn write_buf(&mut self, id: usize, at: usize, bytes: &[u8]) -> Result<()> {
        match self {
            Gpu::Vk(g) => g.write_buf(id, at, bytes),
            Gpu::Cuda(g) => g.write_buf(id, at, bytes),
        }
    }

    pub fn read_buf(&mut self, id: usize, out: &mut [u8]) -> Result<()> {
        match self {
            Gpu::Vk(g) => g.read_buf(id, out),
            Gpu::Cuda(g) => g.read_buf(id, out),
        }
    }

    pub fn write(&mut self, off: usize, v: &[f32]) {
        match self {
            Gpu::Vk(g) => g.write(off, v),
            Gpu::Cuda(g) => g.write(off, v),
        }
    }

    pub fn read(&self, off: usize, n: usize) -> Vec<f32> {
        match self {
            Gpu::Vk(g) => g.read(off, n),
            Gpu::Cuda(g) => g.read(off, n),
        }
    }

    pub fn read_into(&self, off: usize, out: &mut [f32]) {
        match self {
            Gpu::Vk(g) => g.read_into(off, out),
            Gpu::Cuda(g) => g.read_into(off, out),
        }
    }

    /// Whether `read_into` also waits for the submitted work. CUDA's DtoH copy
    /// blocks until it completes; Vulkan's arena is host-visible, so its read is
    /// a plain copy and the caller must wait first. A caller that reads straight
    /// after a batch can skip its own wait only when this is true.
    pub fn read_syncs(&self) -> bool {
        matches!(self, Gpu::Cuda(_))
    }

    pub fn wait(&mut self) -> Result<()> {
        match self {
            Gpu::Vk(g) => g.wait(),
            Gpu::Cuda(g) => g.wait(),
        }
    }

    pub fn wait_idle(&mut self) -> Result<()> {
        match self {
            Gpu::Vk(g) => g.wait_idle(),
            Gpu::Cuda(g) => g.wait_idle(),
        }
    }

    pub fn bytes_held(&self) -> u64 {
        match self {
            Gpu::Vk(g) => g.bytes_held(),
            Gpu::Cuda(g) => g.bytes_held(),
        }
    }

    pub fn reset_counters(&mut self) {
        match self {
            Gpu::Vk(g) => g.reset_counters(),
            Gpu::Cuda(g) => g.reset_counters(),
        }
    }

    pub fn submitted(&self) -> u64 {
        match self {
            Gpu::Vk(g) => g.submitted,
            Gpu::Cuda(g) => g.submitted,
        }
    }

    pub fn dispatches(&self) -> u64 {
        match self {
            Gpu::Vk(g) => g.dispatches,
            Gpu::Cuda(g) => g.dispatches,
        }
    }

    pub fn sync_ms(&self) -> f64 {
        match self {
            Gpu::Vk(g) => g.sync_ms,
            Gpu::Cuda(g) => g.sync_ms,
        }
    }

    pub fn syncs(&self) -> u64 {
        match self {
            Gpu::Vk(g) => g.syncs,
            Gpu::Cuda(g) => g.syncs,
        }
    }

    pub fn batch(&mut self) -> Result<Batch<'_>> {
        match self {
            Gpu::Vk(g) => Ok(Batch::Vk(g.batch()?)),
            Gpu::Cuda(g) => Ok(Batch::Cuda(g.batch()?)),
        }
    }
}

/// The chainable ops return `&mut Self` so a caller can record a whole layer in
/// one expression. Each arm re-wraps its own batch type, which is why these are
/// written out instead of forwarded: `&mut VkBatch` is not `&mut Batch`.
pub enum Batch<'a> {
    Vk(VkBatch<'a>),
    Cuda(crate::cudagpu::CudaBatch<'a>),
}

impl<'a> Batch<'a> {
    pub fn gemv_half(&mut self, w: usize, rows: usize, cols: usize, x_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gemv_half(w, rows, cols, x_off, y_off)?; }
            Batch::Cuda(b) => { b.gemv_half(w, rows, cols, x_off, y_off)?; }
        };
        Ok(self)
    }

    pub fn gemv_q4(&mut self, w: usize, sc: usize, rows: usize, cols: usize, group: usize, x_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gemv_q4(w, sc, rows, cols, group, x_off, y_off)?; }
            Batch::Cuda(b) => { b.gemv_q4(w, sc, rows, cols, group, x_off, y_off)?; }
        };
        Ok(self)
    }

    /// A projection recorded without a trailing pipeline barrier, for kernels
    /// that write disjoint slices and read the same input. On the Vulkan tier
    /// they then fill the device together instead of serializing on a
    /// COMPUTE->COMPUTE barrier each; the caller places one `barrier` at the
    /// next real dependency. CUDA launches into one in-order stream, which
    /// already orders them, so its arm is the plain recording call.
    pub fn gemv_half_indep(&mut self, w: usize, rows: usize, cols: usize, x_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gemv_half_nb(w, rows, cols, x_off, y_off)?; }
            Batch::Cuda(b) => { b.gemv_half(w, rows, cols, x_off, y_off)?; }
        };
        Ok(self)
    }

    /// Same, for an FQ4 weight block.
    pub fn gemv_q4_indep(&mut self, w: usize, sc: usize, rows: usize, cols: usize, group: usize, x_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gemv_q4_nb(w, sc, rows, cols, group, x_off, y_off)?; }
            Batch::Cuda(b) => { b.gemv_q4(w, sc, rows, cols, group, x_off, y_off)?; }
        };
        Ok(self)
    }

    /// Close a phase of independent kernels. A no-op on CUDA, where the stream
    /// already orders every launch behind the previous one's writes.
    pub fn barrier(&mut self) -> Result<&mut Batch<'a>> {
        if let Batch::Vk(b) = self {
            b.barrier()?;
        }
        Ok(self)
    }

    pub fn rmsnorm(&mut self, n: usize, x_off: usize, w_off: usize, y_off: usize, eps: f32) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.rmsnorm(n, x_off, w_off, y_off, eps)?; }
            Batch::Cuda(b) => { b.rmsnorm(n, x_off, w_off, y_off, eps)?; }
        };
        Ok(self)
    }

    pub fn silu_mul(&mut self, n: usize, a_off: usize, b_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.silu_mul(n, a_off, b_off, y_off)?; }
            Batch::Cuda(b) => { b.silu_mul(n, a_off, b_off, y_off)?; }
        };
        Ok(self)
    }

    pub fn axpy(&mut self, n: usize, x_off: usize, y_off: usize, w: f32, set: bool) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.axpy(n, x_off, y_off, w, set)?; }
            Batch::Cuda(b) => { b.axpy(n, x_off, y_off, w, set)?; }
        };
        Ok(self)
    }

    // ── Batched variants ──

    pub fn gemv_half_batch(&mut self, w: usize, rows: usize, cols: usize, n_tokens: usize, x_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gemv_half_batch(w, rows, cols, n_tokens, x_off, y_off)?; }
            Batch::Cuda(b) => {
                // CUDA falls back to per-token dispatches; same weights, different offsets.
                for t in 0..n_tokens {
                    b.gemv_half(w, rows, cols, x_off + t * cols, y_off + t * rows)?;
                }
            }
        };
        Ok(self)
    }

    pub fn rmsnorm_batch(&mut self, n: usize, n_tokens: usize, x_off: usize, w_off: usize, y_off: usize, eps: f32) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.rmsnorm_batch(n, n_tokens, x_off, w_off, y_off, eps)?; }
            Batch::Cuda(b) => {
                for t in 0..n_tokens {
                    b.rmsnorm(n, x_off + t * n, w_off, y_off + t * n, eps)?;
                }
            }
        };
        Ok(self)
    }

    pub fn silu_mul_batch(&mut self, n: usize, n_tokens: usize, a_off: usize, b_off: usize, y_off: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.silu_mul_batch(n, n_tokens, a_off, b_off, y_off)?; }
            Batch::Cuda(b) => {
                // CUDA: one launch covering all tokens' elements
                b.silu_mul(n * n_tokens, a_off, b_off, y_off)?;
            }
        };
        Ok(self)
    }

    pub fn axpy_batch(&mut self, n: usize, n_tokens: usize, x_off: usize, y_off: usize, w: f32, set: bool) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.axpy_batch(n, n_tokens, x_off, y_off, w, set)?; }
            Batch::Cuda(b) => {
                b.axpy(n * n_tokens, x_off, y_off, w, set)?;
            }
        };
        Ok(self)
    }

    pub fn gdn_conv(&mut self, w: usize, state: usize, base: usize, cd: usize, kernel: usize, hist: usize) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gdn_conv(w, state, base, cd, kernel, hist)?; }
            Batch::Cuda(b) => { b.gdn_conv(w, state, base, cd, kernel, hist)?; }
        };
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gdn_rec(&mut self, w: usize, state: usize, base: usize, out_off: usize, kd: usize, nv: usize, dk: usize, dv: usize, rep: usize, eps: f32) -> Result<&mut Batch<'a>> {
        match self {
            Batch::Vk(b) => { b.gdn_rec(w, state, base, out_off, kd, nv, dk, dv, rep, eps)?; }
            Batch::Cuda(b) => { b.gdn_rec(w, state, base, out_off, kd, nv, dk, dv, rep, eps)?; }
        };
        Ok(self)
    }

    pub fn submit(&mut self) -> Result<()> {
        match self {
            Batch::Vk(b) => b.submit(),
            Batch::Cuda(b) => b.submit(),
        }
    }

    pub fn wait(&mut self) -> Result<()> {
        match self {
            Batch::Vk(b) => b.wait(),
            Batch::Cuda(b) => b.wait(),
        }
    }

    pub fn run(&mut self) -> Result<()> {
        match self {
            Batch::Vk(b) => b.run(),
            Batch::Cuda(b) => b.run(),
        }
    }

    pub fn ops(&self) -> usize {
        match self {
            Batch::Vk(b) => b.ops,
            Batch::Cuda(b) => b.ops,
        }
    }
}