//! CUDA through the driver library that ships with the NVIDIA driver: no
//! toolkit, no nvcc, no build script. `nvcuda.dll` / `libcuda.so.1` is loaded
//! with `libloading` exactly like the Vulkan loader, and the kernels are PTX
//! checked in beside the SPIR-V and JIT'd by the driver at `cuModuleLoadData`.
//!
//! This is the backend `backend::Backend::for_vendor` picks for NVIDIA. AMD
//! stays on Vulkan and Apple goes through MoltenVK to Metal; nothing here is
//! reachable on either, and `load()` fails cleanly when the driver library is
//! absent so the caller can fall back.
//!
//! TensorRT is deliberately not used: it compiles a whole graph into an engine
//! and owns the memory, which cannot express hot experts in VRAM with the rest
//! in a host pool and the CPU working its share while batches are in flight.

use libloading::Library;
use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int, c_uint};

pub type Result<T> = std::result::Result<T, String>;

type Device = c_int;
/// CUdeviceptr is a 64-bit integer, not a pointer, on every platform.
pub type DevPtr = u64;
pub type Handle = *mut c_void;

thread_local! {
    /// The context this thread last made current, so `bind` can skip the driver
    /// call when it would be a no-op. Per thread because CUDA's current context
    /// is per thread.
    static BOUND: std::cell::Cell<Handle> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

const CU_CTX_SCHED_SPIN: c_uint = 0x01;
const CU_CTX_MAP_HOST: c_uint = 0x08;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;

// CUDA graph capture flags
const CU_STREAM_CAPTURE_MODE_RELAXED: c_uint = 2;

/// The driver entry points this backend uses. Resolved once at load, so a
/// launch is a direct call through a function pointer, same as the Vulkan api.
struct Api {
    init: unsafe extern "system" fn(c_uint) -> c_int,
    device_get_count: unsafe extern "system" fn(*mut c_int) -> c_int,
    device_get: unsafe extern "system" fn(*mut Device, c_int) -> c_int,
    device_get_name: unsafe extern "system" fn(*mut c_char, c_int, Device) -> c_int,
    device_total_mem: unsafe extern "system" fn(*mut usize, Device) -> c_int,
    device_get_attribute: unsafe extern "system" fn(*mut c_int, c_int, Device) -> c_int,
    driver_get_version: unsafe extern "system" fn(*mut c_int) -> c_int,
    ctx_create: unsafe extern "system" fn(*mut Handle, c_uint, Device) -> c_int,
    ctx_destroy: unsafe extern "system" fn(Handle) -> c_int,
    ctx_set_current: unsafe extern "system" fn(Handle) -> c_int,
    ctx_synchronize: unsafe extern "system" fn() -> c_int,
    module_load_data_ex: unsafe extern "system" fn(
        *mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void,
    ) -> c_int,
    module_unload: unsafe extern "system" fn(Handle) -> c_int,
    module_get_function: unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> c_int,
    mem_alloc: unsafe extern "system" fn(*mut DevPtr, usize) -> c_int,
    mem_free: unsafe extern "system" fn(DevPtr) -> c_int,
    mem_get_info: unsafe extern "system" fn(*mut usize, *mut usize) -> c_int,
    memcpy_htod: unsafe extern "system" fn(DevPtr, *const c_void, usize) -> c_int,
    memcpy_dtoh: unsafe extern "system" fn(*mut c_void, DevPtr, usize) -> c_int,
    launch_kernel: unsafe extern "system" fn(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle,
        *mut *mut c_void, *mut *mut c_void,
    ) -> c_int,
    get_error_string: unsafe extern "system" fn(c_int, *mut *const c_char) -> c_int,
    // Stream and graph APIs for command buffer reuse
    stream_create: unsafe extern "system" fn(*mut Handle) -> c_int,
    stream_destroy: unsafe extern "system" fn(Handle) -> c_int,
    stream_synchronize: unsafe extern "system" fn(Handle) -> c_int,
    stream_begin_capture: unsafe extern "system" fn(Handle, c_uint) -> c_int,
    stream_end_capture: unsafe extern "system" fn(Handle, *mut Handle) -> c_int,
    graph_instantiate: unsafe extern "system" fn(*mut Handle, Handle, *mut c_void, *mut c_char, usize) -> c_int,
    graph_launch: unsafe extern "system" fn(Handle, Handle) -> c_int,
    graph_destroy: unsafe extern "system" fn(Handle) -> c_int,
    graph_exec_destroy: unsafe extern "system" fn(Handle) -> c_int,
}

macro_rules! resolve {
    ($lib:expr, $name:literal, $ty:ty) => {
        *$lib.get::<$ty>(concat!($name, "\0").as_bytes())
            .map_err(|e| format!("{}: {e}", $name))?
    };
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub index: usize,
    pub name: String,
    pub total_gib: f64,
    pub free_gib: f64,
    pub compute: (i32, i32),
}

pub struct Cuda {
    /// Held, never read: dropping it unloads the driver library while the
    /// function pointers in `api` are still in use.
    #[allow(dead_code)]
    lib: Library,
    api: Api,
    pub driver_version: i32,
    ctx: Handle,
    module: Handle,
    pub funcs: Vec<Handle>,
    pub devices: Vec<CudaDevice>,
    pub current: usize,
}

impl Cuda {
    /// Load the driver library and initialise it. Fails with the loader's own
    /// error when there is no NVIDIA driver, which is the signal to fall back.
    pub fn load() -> Result<Cuda> {
        let mut err = String::new();
        let mut lib = None;
        for cand in candidate_names() {
            match unsafe { Library::new(cand) } {
                Ok(l) => {
                    lib = Some(l);
                    break;
                }
                Err(e) => err = format!("{cand}: {e}"),
            }
        }
        let lib = lib.ok_or_else(|| format!("no CUDA driver library: {err}"))?;
        // SAFETY: every symbol is a documented driver entry point with the
        // signature declared in Api; the library outlives every call through it.
        let api = unsafe {
            Api {
                init: resolve!(lib, "cuInit", unsafe extern "system" fn(c_uint) -> c_int),
                device_get_count: resolve!(lib, "cuDeviceGetCount", unsafe extern "system" fn(*mut c_int) -> c_int),
                device_get: resolve!(lib, "cuDeviceGet", unsafe extern "system" fn(*mut Device, c_int) -> c_int),
                device_get_name: resolve!(lib, "cuDeviceGetName", unsafe extern "system" fn(*mut c_char, c_int, Device) -> c_int),
                device_total_mem: resolve!(lib, "cuDeviceTotalMem_v2", unsafe extern "system" fn(*mut usize, Device) -> c_int),
                device_get_attribute: resolve!(lib, "cuDeviceGetAttribute", unsafe extern "system" fn(*mut c_int, c_int, Device) -> c_int),
                driver_get_version: resolve!(lib, "cuDriverGetVersion", unsafe extern "system" fn(*mut c_int) -> c_int),
                ctx_create: resolve!(lib, "cuCtxCreate_v2", unsafe extern "system" fn(*mut Handle, c_uint, Device) -> c_int),
                ctx_destroy: resolve!(lib, "cuCtxDestroy_v2", unsafe extern "system" fn(Handle) -> c_int),
                ctx_set_current: resolve!(lib, "cuCtxSetCurrent", unsafe extern "system" fn(Handle) -> c_int),
                ctx_synchronize: resolve!(lib, "cuCtxSynchronize", unsafe extern "system" fn() -> c_int),
                module_load_data_ex: resolve!(lib, "cuModuleLoadDataEx", unsafe extern "system" fn(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> c_int),
                module_unload: resolve!(lib, "cuModuleUnload", unsafe extern "system" fn(Handle) -> c_int),
                module_get_function: resolve!(lib, "cuModuleGetFunction", unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> c_int),
                mem_alloc: resolve!(lib, "cuMemAlloc_v2", unsafe extern "system" fn(*mut DevPtr, usize) -> c_int),
                mem_free: resolve!(lib, "cuMemFree_v2", unsafe extern "system" fn(DevPtr) -> c_int),
                mem_get_info: resolve!(lib, "cuMemGetInfo_v2", unsafe extern "system" fn(*mut usize, *mut usize) -> c_int),
                memcpy_htod: resolve!(lib, "cuMemcpyHtoD_v2", unsafe extern "system" fn(DevPtr, *const c_void, usize) -> c_int),
                memcpy_dtoh: resolve!(lib, "cuMemcpyDtoH_v2", unsafe extern "system" fn(*mut c_void, DevPtr, usize) -> c_int),
                launch_kernel: resolve!(lib, "cuLaunchKernel", unsafe extern "system" fn(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void) -> c_int),
                get_error_string: resolve!(lib, "cuGetErrorString", unsafe extern "system" fn(c_int, *mut *const c_char) -> c_int),
                stream_create: resolve!(lib, "cuStreamCreate", unsafe extern "system" fn(*mut Handle) -> c_int),
                stream_destroy: resolve!(lib, "cuStreamDestroy_v2", unsafe extern "system" fn(Handle) -> c_int),
                stream_synchronize: resolve!(lib, "cuStreamSynchronize", unsafe extern "system" fn(Handle) -> c_int),
                stream_begin_capture: resolve!(lib, "cuStreamBeginCapture_v2", unsafe extern "system" fn(Handle, c_uint) -> c_int),
                stream_end_capture: resolve!(lib, "cuStreamEndCapture", unsafe extern "system" fn(Handle, *mut Handle) -> c_int),
                graph_instantiate: resolve!(lib, "cuGraphInstantiate_v2", unsafe extern "system" fn(*mut Handle, Handle, *mut c_void, *mut c_char, usize) -> c_int),
                graph_launch: resolve!(lib, "cuGraphLaunch", unsafe extern "system" fn(Handle, Handle) -> c_int),
                graph_destroy: resolve!(lib, "cuGraphDestroy", unsafe extern "system" fn(Handle) -> c_int),
                graph_exec_destroy: resolve!(lib, "cuGraphExecDestroy", unsafe extern "system" fn(Handle) -> c_int),
            }
        };
        let mut cuda = Cuda {
            lib,
            api,
            driver_version: 0,
            ctx: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
            funcs: Vec::new(),
            devices: Vec::new(),
            current: 0,
        };
        cuda.check(unsafe { (cuda.api.init)(0) }, "cuInit")?;
        unsafe { (cuda.api.driver_get_version)(&mut cuda.driver_version) };
        cuda.devices = cuda.enumerate()?;
        Ok(cuda)
    }

    /// A CUDA context is current per thread. `serve` loads the model on one
    /// thread and answers each request on another, so every entry point has to
    /// rebind or the first call there fails with "invalid device context".
    ///
    /// The rebind is cached per thread: a decode token issues over a thousand
    /// launches, uploads and reads, and each one was paying a cross-DLL driver
    /// call to re-set a context that was already current. Nothing else in this
    /// process changes the current context, so the cached handle stays valid
    /// until a different `Cu` binds on the same thread.
    fn bind(&self) {
        if self.ctx.is_null() || BOUND.with(|b| b.get()) == self.ctx {
            return;
        }
        if unsafe { (self.api.ctx_set_current)(self.ctx) } == 0 {
            BOUND.with(|b| b.set(self.ctx));
        }
    }

    fn check(&self, code: c_int, what: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        let detail = unsafe {
            if (self.api.get_error_string)(code, &mut p) == 0 && !p.is_null() {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            } else {
                format!("CUresult {code}")
            }
        };
        Err(format!("{what}: {detail}"))
    }

    fn enumerate(&self) -> Result<Vec<CudaDevice>> {
        let mut n: c_int = 0;
        self.check(unsafe { (self.api.device_get_count)(&mut n) }, "cuDeviceGetCount")?;
        let mut out = Vec::new();
        for i in 0..n {
            let mut dev: Device = 0;
            self.check(unsafe { (self.api.device_get)(&mut dev, i) }, "cuDeviceGet")?;
            let mut buf = [0u8; 256];
            self.check(
                unsafe { (self.api.device_get_name)(buf.as_mut_ptr() as *mut c_char, 256, dev) },
                "cuDeviceGetName",
            )?;
            let name = buf.iter().copied().take_while(|&b| b != 0).collect::<Vec<u8>>();
            let mut bytes = 0usize;
            self.check(unsafe { (self.api.device_total_mem)(&mut bytes, dev) }, "cuDeviceTotalMem")?;
            let (mut major, mut minor) = (0i32, 0i32);
            unsafe {
                (self.api.device_get_attribute)(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                (self.api.device_get_attribute)(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
            }
            // Query free memory on this device's context
            let free_bytes = self.query_free_mem()?;
            out.push(CudaDevice {
                index: i as usize,
                name: String::from_utf8_lossy(&name).into_owned(),
                total_gib: bytes as f64 / 2f64.powi(30),
                free_gib: free_bytes as f64 / 2f64.powi(30),
                compute: (major, minor),
            });
        }
        Ok(out)
    }

    /// Query free memory on the current context. Returns bytes available for allocation.
    pub fn query_free_mem(&self) -> Result<usize> {
        if self.ctx.is_null() {
            return Ok(0);
        }
        let mut free: usize = 0;
        let mut total: usize = 0;
        self.check(
            unsafe { (self.api.mem_get_info)(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok(free)
    }

    /// Create a context on `index` and JIT the checked-in PTX. SCHED_SPIN makes
    /// `cuCtxSynchronize` poll rather than park the thread, which is the same
    /// trade the Vulkan fence takes: a decode batch is shorter than a wake.
    pub fn open(&mut self, index: usize, ptx: &[&str]) -> Result<()> {
        if index >= self.devices.len() {
            return Err(format!("CUDA device {index} of {} not present", self.devices.len()));
        }
        let mut ordinal: Device = 0;
        self.check(unsafe { (self.api.device_get)(&mut ordinal, index as c_int) }, "cuDeviceGet")?;
        let mut ctx: Handle = std::ptr::null_mut();
        self.check(
            unsafe { (self.api.ctx_create)(&mut ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, ordinal) },
            "cuCtxCreate",
        )?;
        self.ctx = ctx;
        self.current = index;
        // One module holding every kernel, so a launch is a function pointer
        // out of `funcs` indexed by the same Op enum the Vulkan path uses.
        let joined = ptx.join("\n");
        let mut image = joined.into_bytes();
        image.push(0);
        let mut module: Handle = std::ptr::null_mut();
        // The Ex form with log buffers: a bare cuModuleLoadData reports only
        // "a PTX JIT compilation failed", which does not name the line. The
        // driver's own error log does, and this PTX is hand-written.
        let (mut err_log, mut info_log) = (vec![0u8; 8192], vec![0u8; 8192]);
        let mut opts = [
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut vals = [
            err_log.as_mut_ptr() as *mut c_void,
            err_log.len() as *mut c_void,
            info_log.as_mut_ptr() as *mut c_void,
            info_log.len() as *mut c_void,
        ];
        let code = unsafe {
            (self.api.module_load_data_ex)(
                &mut module,
                image.as_ptr() as *const c_void,
                opts.len() as c_uint,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            )
        };
        if code != 0 {
            let log = |b: &[u8]| {
                let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
                String::from_utf8_lossy(&b[..end]).trim().to_string()
            };
            let (e, i) = (log(&err_log), log(&info_log));
            let mut msg = self.check(code, "cuModuleLoadDataEx (PTX JIT)").unwrap_err();
            if !e.is_empty() {
                msg.push_str("\n--- PTX JIT error log ---\n");
                msg.push_str(&e);
            }
            if !i.is_empty() {
                msg.push_str("\n--- PTX JIT info log ---\n");
                msg.push_str(&i);
            }
            return Err(msg);
        }
        self.module = module;
        Ok(())
    }

    pub fn function(&mut self, name: &str) -> Result<Handle> {
        self.bind();
        let mut c = name.as_bytes().to_vec();
        c.push(0);
        let mut f: Handle = std::ptr::null_mut();
        self.check(
            unsafe { (self.api.module_get_function)(&mut f, self.module, c.as_ptr() as *const c_char) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(f)
    }

    pub fn alloc(&mut self, bytes: usize) -> Result<DevPtr> {
        self.bind();
        let mut p: DevPtr = 0;
        self.check(unsafe { (self.api.mem_alloc)(&mut p, bytes.max(16)) }, "cuMemAlloc")?;
        Ok(p)
    }

    pub fn free(&mut self, p: DevPtr) {
        self.bind();
        if p != 0 {
            unsafe { (self.api.mem_free)(p) };
        }
    }

    pub fn write(&self, dst: DevPtr, src: &[u8]) -> Result<()> {
        self.bind();
        self.check(
            unsafe { (self.api.memcpy_htod)(dst, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    pub fn read(&self, dst: &mut [u8], src: DevPtr) -> Result<()> {
        self.bind();
        self.check(
            unsafe { (self.api.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, src, dst.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Launch with a parameter block of pointers to the arguments, which is the
    /// driver API's calling convention. `params` must stay alive for the call.
    pub fn launch(
        &mut self,
        f: Handle,
        grid: u32,
        block: u32,
        shared: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.bind();
        self.check(
            unsafe {
                (self.api.launch_kernel)(
                    f, grid, 1, 1, block, 1, 1, shared, std::ptr::null_mut(),
                    params.as_mut_ptr(), std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    pub fn sync(&mut self) -> Result<()> {
        self.bind();
        self.check(unsafe { (self.api.ctx_synchronize)() }, "cuCtxSynchronize")
    }

    /// Create a CUDA stream for async operations and graph capture.
    pub fn create_stream(&mut self) -> Result<Handle> {
        self.bind();
        let mut stream: Handle = std::ptr::null_mut();
        self.check(
            unsafe { (self.api.stream_create)(&mut stream) },
            "cuStreamCreate",
        )?;
        Ok(stream)
    }

    /// Destroy a stream created with create_stream.
    pub fn destroy_stream(&mut self, stream: Handle) {
        if stream.is_null() {
            return;
        }
        self.bind();
        unsafe { (self.api.stream_destroy)(stream) };
    }

    /// Begin capturing kernel launches on a stream. Returns when capture is active;
    /// subsequent launches on this stream are recorded, not executed.
    pub fn begin_capture(&mut self, stream: Handle) -> Result<()> {
        self.bind();
        self.check(
            unsafe { (self.api.stream_begin_capture)(stream, CU_STREAM_CAPTURE_MODE_RELAXED) },
            "cuStreamBeginCapture",
        )
    }

    /// End capture and return the recorded graph. The stream reverts to normal mode.
    pub fn end_capture(&mut self, stream: Handle) -> Result<Handle> {
        self.bind();
        let mut graph: Handle = std::ptr::null_mut();
        self.check(
            unsafe { (self.api.stream_end_capture)(stream, &mut graph) },
            "cuStreamEndCapture",
        )?;
        Ok(graph)
    }

    /// Instantiate a graph into an executable that can be launched.
    pub fn instantiate_graph(&mut self, graph: Handle) -> Result<Handle> {
        self.bind();
        let mut exec: Handle = std::ptr::null_mut();
        let mut log_buf = vec![0u8; 1024];
        self.check(
            unsafe {
                (self.api.graph_instantiate)(
                    &mut exec,
                    graph,
                    std::ptr::null_mut(),
                    log_buf.as_mut_ptr() as *mut c_char,
                    log_buf.len(),
                )
            },
            "cuGraphInstantiate",
        )?;
        Ok(exec)
    }

    /// Launch an instantiated graph on a stream.
    pub fn launch_graph(&mut self, exec: Handle, stream: Handle) -> Result<()> {
        self.bind();
        self.check(
            unsafe { (self.api.graph_launch)(exec, stream) },
            "cuGraphLaunch",
        )
    }

    /// Destroy a graph created with end_capture.
    pub fn destroy_graph(&mut self, graph: Handle) {
        if graph.is_null() {
            return;
        }
        self.bind();
        unsafe { (self.api.graph_destroy)(graph) };
    }

    /// Destroy an instantiated graph executable.
    pub fn destroy_graph_exec(&mut self, exec: Handle) {
        if exec.is_null() {
            return;
        }
        self.bind();
        unsafe { (self.api.graph_exec_destroy)(exec) };
    }

    /// Synchronize a specific stream (wait for all work on it to complete).
    pub fn sync_stream(&mut self, stream: Handle) -> Result<()> {
        self.bind();
        self.check(
            unsafe { (self.api.stream_synchronize)(stream) },
            "cuStreamSynchronize",
        )
    }

    /// Launch a kernel on a specific stream instead of the default.
    pub fn launch_on_stream(
        &mut self,
        f: Handle,
        stream: Handle,
        grid: u32,
        block: u32,
        shared: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.bind();
        self.check(
            unsafe {
                (self.api.launch_kernel)(
                    f, grid, 1, 1, block, 1, 1, shared, stream,
                    params.as_mut_ptr(), std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                let _ = (self.api.ctx_set_current)(self.ctx);
                let _ = (self.api.ctx_synchronize)();
            }
            if !self.module.is_null() {
                (self.api.module_unload)(self.module);
            }
            if !self.ctx.is_null() {
                (self.api.ctx_destroy)(self.ctx);
            }
        }
    }
}

/// The driver library names, in try order. It ships with the display driver,
/// not with a toolkit, so this works on a machine that has never installed CUDA.
#[cfg(target_os = "windows")]
fn candidate_names() -> &'static [&'static str] {
    &["nvcuda.dll"]
}
#[cfg(not(target_os = "windows"))]
fn candidate_names() -> &'static [&'static str] {
    &["libcuda.so.1", "libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"]
}

/// The driver version as CUDA reports it: 12040 -> "12.4".
pub fn version_string(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_version_decodes_as_major_minor() {
        assert_eq!(version_string(12040), "12.4");
        assert_eq!(version_string(13000), "13.0");
        assert_eq!(version_string(11080), "11.8");
    }

    #[test]
    fn the_driver_library_is_the_one_that_ships_with_the_display_driver() {
        let names = candidate_names();
        assert!(!names.is_empty());
        #[cfg(target_os = "windows")]
        assert_eq!(names[0], "nvcuda.dll");
        #[cfg(not(target_os = "windows"))]
        assert_eq!(names[0], "libcuda.so.1");
        // No toolkit library: the whole point is that nvcc is not required.
        assert!(!names.iter().any(|n| n.contains("cudart")));
    }

    #[test]
    fn a_missing_driver_is_an_error_not_a_panic() {
        // On a machine with no NVIDIA driver this is the fallback signal. On one
        // with a driver it must still enumerate at least one device.
        match Cuda::load() {
            Ok(c) => assert!(!c.devices.is_empty(), "a loaded driver reported no devices"),
            Err(e) => assert!(e.contains("CUDA"), "{e}"),
        }
    }
}