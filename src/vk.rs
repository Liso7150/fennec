//! Vulkan device discovery, vendor-neutral: no NVIDIA or AMD headers, no SDK,
//! no build script. The loader is `vulkan-1.dll` / `libvulkan.so.1` /
//! `libMoltenVK.dylib`, and every device this instance can see is reported with
//! its vendor id, class and heap sizes, which is what the tier scheduler needs
//! and what colibri's `COLI_VK_DEV` selector does with the same list.
//!
//! macOS has no system Vulkan. MoltenVK is the ICD that translates these same
//! Vulkan calls and SPIR-V shaders to Metal, so it is a first-class loader
//! candidate here, not a fallback: the compute tier runs on Apple GPUs through
//! it with no shader changes. Two portability hooks are required for that and
//! are no-ops everywhere else - the instance flag that lets a portability driver
//! be enumerated at all, and the device extension it advertises.

use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::os::raw::c_char;
use std::path::Path;

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_API_VERSION_1_0: u32 = (1 << 22) | 0;
/// VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR. Without it the loader hides
/// MoltenVK and `vkEnumeratePhysicalDevices` reports zero devices on a Mac.
const VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR: u32 = 0x1;
const PORTABILITY_ENUMERATION_EXT: &str = "VK_KHR_portability_enumeration";
/// Advertised by a portability driver; the spec requires enabling it at
/// vkCreateDevice when the physical device supports it.
const PORTABILITY_SUBSET_EXT: &str = "VK_KHR_portability_subset";
/// VkExtensionProperties: extensionName[256] then specVersion.
const VK_MAX_EXTENSION_NAME: usize = 256;

#[repr(C)]
struct AppInfo {
    s_type: u32,
    p_next: *const c_void,
    p_app_name: *const c_char,
    app_version: u32,
    p_engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct InstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const AppInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MemHeap {
    size: u64,
    flags: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MemType {
    property_flags: u32,
    heap_index: u32,
}

/// VkPhysicalDeviceMemoryProperties, in the order the spec declares the members:
/// memoryTypeCount, memoryTypes[], memoryHeapCount, memoryHeaps[]. Declaring the
/// counts heap-first puts every heap size inside the type array and reports
/// system RAM as VRAM.
#[repr(C)]
struct MemProps {
    type_count: u32,
    types: [MemType; VK_MAX_MEMORY_TYPES],
    heap_count: u32,
    heaps: [MemHeap; VK_MAX_MEMORY_HEAPS],
}

const VK_MAX_MEMORY_HEAPS: usize = 16;
const VK_MAX_MEMORY_TYPES: usize = 32;
/// VkMemoryPropertyFlagBits, read from memoryTypes[].propertyFlags
/// (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT is 0x1; capacity here comes from the heaps.)
const MTYPE_HOST_VISIBLE: u32 = 0x2;
/// VkMemoryHeapFlagBits, read from memoryHeaps[].flags
/// (0x1, not the 0x2 that VkDeviceQueueCreateFlags uses)
const MHEAP_DEVICE_LOCAL: u32 = 0x1;

#[derive(Debug, Clone)]
pub struct Device {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub class: u32,
    /// decoded "1.4" from VkPhysicalDeviceProperties.apiVersion
    pub api: String,
    /// vendor-decoded driver branch, best effort
    pub driver: String,
    pub device_local_gib: f64,
    pub dedicated_gib: f64,
    pub host_visible_gib: f64,
    /// memoryHeaps[] as reported: (size, flags, reachable by a host-visible type)
    pub heaps: Vec<(u64, u32, bool)>,
    /// Device extensions as reported. `VK_KHR_portability_subset` here means a
    /// translation layer (MoltenVK -> Metal), and the spec requires enabling it.
    pub extensions: Vec<String>,
    pub vendor: &'static str,
    pub rank: u8,
}

impl Device {
    /// True when Vulkan is being translated to Metal underneath.
    pub fn portability(&self) -> bool {
        self.extensions.iter().any(|e| e == PORTABILITY_SUBSET_EXT)
    }

    /// What the tier planner can actually spend. A discrete GPU has dedicated
    /// VRAM; Apple Silicon and iGPUs have one unified heap that is both
    /// device-local and host-visible, so `dedicated_gib` is 0 there and sizing
    /// the device tier from it would place nothing on the GPU at all.
    pub fn usable_gib(&self) -> f64 {
        if self.dedicated_gib > 0.0 {
            self.dedicated_gib
        } else {
            self.device_local_gib
        }
    }

    /// The backend this device should run on: CUDA for NVIDIA, Metal for Apple,
    /// Vulkan for everyone else. AMD stays on Vulkan deliberately - the same
    /// vendor-neutral path, no HIP toolchain.
    pub fn preferred_backend(&self) -> crate::backend::Backend {
        crate::backend::Backend::for_vendor(self.vendor_id, self.portability())
    }
}

pub fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x1002 => "AMD",
        0x8086 => "Intel",
        0x106B => "Apple",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x1414 => "Microsoft",
        0x1AF4 => "Red Hat / virtio",
        _ => "other",
    }
}

/// Vulkan's VkPhysicalDeviceType, ranked for "who should do the math":
/// DISCRETE 2 > INTEGRATED 1 > VIRTUAL 3 > OTHER 0 > CPU 4.
fn rank_of(class: u32, vendor_id: u32) -> u8 {
    let _ = vendor_id;
    match class {
        2 => 5,
        1 => 4,
        3 => 3,
        0 => 2,
        _ => 1,
    }
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        self.class <= 3 && self.class != 4
    }
}

pub struct Vulkan {
    lib: Option<Library>,
    instance: *mut c_void,
    devices: Vec<*mut c_void>,
    loaded: bool,
    pub error: String,
}

unsafe fn get_inst_fn(lib: &Library, inst: *mut c_void, name: &str) -> Option<*mut c_void> {
    type GPA = unsafe extern "system" fn(*mut c_void, *const c_char) -> *mut c_void;
    let gpa: Symbol<GPA> = unsafe { lib.get(b"vkGetInstanceProcAddr") }.ok()?;
    let mut buf = Vec::with_capacity(name.len() + 1);
    buf.extend_from_slice(name.as_bytes());
    buf.push(0);
    // vkGetInstanceProcAddr answers null for a name the loader does not know
    let p = unsafe { gpa(inst, buf.as_ptr() as *const c_char) };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

/// The loader names this platform ships, in try order. macOS has no system
/// Vulkan, so MoltenVK - the ICD that translates these calls to Metal - leads.
#[cfg(target_os = "windows")]
const LOADER_NAMES: [&str; 2] = ["vulkan-1.dll", "vulkan.dll"];
#[cfg(target_os = "macos")]
const LOADER_NAMES: [&str; 3] = ["libMoltenVK.dylib", "libvulkan.1.dylib", "libvulkan.dylib"];
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const LOADER_NAMES: [&str; 2] = ["libvulkan.so.1", "libvulkan.so"];

/// Every path worth trying, explicit override first. `FENNEC_VULKAN_LIB` names a
/// specific loader; the macOS directories cover Homebrew (both prefixes), a
/// LunarG SDK under ~/VulkanSDK, and a dylib bundled beside the executable.
pub fn loader_candidates(
    env: &dyn Fn(&str) -> Option<String>,
    exe_dir: Option<&str>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(p) = env("FENNEC_VULKAN_LIB") {
        let p = p.trim();
        if !p.is_empty() {
            out.push(p.to_string());
        }
    }
    out.extend(LOADER_NAMES.iter().map(|s| s.to_string()));
    if cfg!(target_os = "macos") {
        for dir in macos_loader_dirs(env, exe_dir) {
            for name in ["libMoltenVK.dylib", "libvulkan.1.dylib"] {
                out.push(format!("{dir}/{name}"));
            }
        }
    }
    out
}

fn macos_loader_dirs(env: &dyn Fn(&str) -> Option<String>, exe_dir: Option<&str>) -> Vec<String> {
    let mut dirs = vec!["/opt/homebrew/lib".to_string(), "/usr/local/lib".to_string()];
    if let Some(d) = exe_dir {
        dirs.push(d.to_string());
        dirs.push(format!("{d}/../lib"));
        dirs.push(format!("{d}/../Frameworks"));
    }
    if let Some(home) = env("HOME") {
        let sdk = Path::new(&home).join("VulkanSDK");
        if let Ok(entries) = std::fs::read_dir(&sdk) {
            let mut versions: Vec<String> = entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            versions.sort_by(|a, b| b.cmp(a));
            for v in versions {
                dirs.push(sdk.join(v).join("macOS").join("lib").to_string_lossy().into_owned());
            }
        }
    }
    dirs
}

/// VkExtensionProperties is a 256-byte name followed by a u32 spec version.
fn extension_names(raw: &[u8]) -> Vec<String> {
    raw.chunks_exact(VK_MAX_EXTENSION_NAME + 4)
        .map(|c| {
            let end = c[..VK_MAX_EXTENSION_NAME]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(VK_MAX_EXTENSION_NAME);
            String::from_utf8_lossy(&c[..end]).into_owned()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

unsafe fn instance_extensions(lib: &Library) -> Vec<String> {
    type Enum = unsafe extern "system" fn(*const c_char, *mut u32, *mut c_void) -> i32;
    let Ok(sym) = (unsafe { lib.get::<Enum>(b"vkEnumerateInstanceExtensionProperties") }) else {
        return Vec::new();
    };
    let mut n = 0u32;
    if unsafe { sym(std::ptr::null(), &mut n, std::ptr::null_mut()) } != 0 || n == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u8; n as usize * (VK_MAX_EXTENSION_NAME + 4)];
    if unsafe { sym(std::ptr::null(), &mut n, buf.as_mut_ptr() as *mut c_void) } != 0 {
        return Vec::new();
    }
    extension_names(&buf)
}

impl Vulkan {
    pub fn load() -> Vulkan {
        let mut err = String::new();
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()));
        for cand in loader_candidates(&|k| std::env::var(k).ok(), exe_dir.as_deref()) {
            match unsafe { Library::new(&cand) } {
                Ok(lib) => return Vulkan {
                    lib: Some(lib),
                    instance: std::ptr::null_mut(),
                    devices: Vec::new(),
                    loaded: true,
                    error: String::new(),
                },
                Err(e) => err = format!("{cand}: {e}"),
            }
        }
        Vulkan {
            lib: None,
            instance: std::ptr::null_mut(),
            devices: Vec::new(),
            loaded: false,
            error: err,
        }
    }

    fn entry(&self) -> Option<*mut c_void> {
        self.gpa(std::ptr::null_mut(), "vkCreateInstance")
    }

    fn gpa(&self, inst: *mut c_void, name: &str) -> Option<*mut c_void> {
        let lib = self.lib.as_ref()?;
        unsafe { get_inst_fn(lib, inst, name) }
    }

    /// Create an instance and enumerate every device. The only extension ever
    /// enabled is portability enumeration, and only when the loader advertises
    /// it: without that flag the loader hides MoltenVK and a Mac reports no
    /// devices at all. Everywhere else this is the same zero-extension instance.
    pub fn devices(&mut self) -> Vec<Device> {
        if !self.loaded {
            return Vec::new();
        }
        unsafe {
            type CreateInstance =
                unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> i32;
            let fp: CreateInstance = match self.entry() {
                Some(p) => std::mem::transmute(p),
                None => return Vec::new(),
            };
            let app = "Fennec\0";
            let eng = "fennec\0";
            let info = AppInfo {
                s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
                p_next: std::ptr::null(),
                p_app_name: app.as_ptr() as *const c_char,
                app_version: 1,
                p_engine_name: eng.as_ptr() as *const c_char,
                engine_version: 1,
                api_version: VK_API_VERSION_1_0,
            };
            let portability = self
                .lib
                .as_ref()
                .map(|lib| instance_extensions(lib))
                .unwrap_or_default()
                .iter()
                .any(|e| e == PORTABILITY_ENUMERATION_EXT);
            let ext_name = b"VK_KHR_portability_enumeration\0";
            let exts: [*const c_char; 1] = [ext_name.as_ptr() as *const c_char];
            let ci = InstanceCreateInfo {
                s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: if portability { VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR } else { 0 },
                p_application_info: &info,
                enabled_layer_count: 0,
                pp_enabled_layer_names: std::ptr::null(),
                enabled_extension_count: u32::from(portability),
                pp_enabled_extension_names: if portability { exts.as_ptr() } else { std::ptr::null() },
            };
            let mut inst: *mut c_void = std::ptr::null_mut();
            if fp(&ci, std::ptr::null(), &mut inst) != 0 || inst.is_null() {
                self.error = "vkCreateInstance failed".into();
                return Vec::new();
            }
            self.instance = inst;
            type Enum = unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32;
            let fp: Enum = match self.gpa(inst, "vkEnumeratePhysicalDevices") {
                Some(p) => std::mem::transmute(p),
                None => return Vec::new(),
            };
            let mut n = 0u32;
            if fp(inst, &mut n, std::ptr::null_mut()) != 0 || n == 0 {
                self.error = "no physical devices".into();
                return Vec::new();
            }
            let mut list = vec![std::ptr::null_mut::<c_void>(); n as usize];
            if fp(inst, &mut n, list.as_mut_ptr()) != 0 {
                self.error = "vkEnumeratePhysicalDevices failed".into();
                return Vec::new();
            }
            self.devices = list.clone();
            let mut out = Vec::new();
            for (i, d) in list.iter().enumerate() {
                let mut buf = [0u8; 4096];
                type GetProps = unsafe extern "system" fn(*mut c_void, *mut c_void);
                let fp: GetProps = match self.gpa(inst, "vkGetPhysicalDeviceProperties") {
                    Some(p) => std::mem::transmute(p),
                    None => break,
                };
                fp(*d, buf.as_mut_ptr() as *mut c_void);
                let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
                let api = u32_at(0);
                let driver = u32_at(4);
                let vendor = u32_at(8);
                let device_id = u32_at(12);
                let class = u32_at(16);
                let name: Vec<u8> = buf[20..276].iter().copied().take_while(|&b| b != 0).collect();
                let name = String::from_utf8_lossy(&name).into_owned();
                let extensions = self.device_extensions(inst, *d);
                let mut mp = MemProps {
                    heap_count: 0,
                    type_count: 0,
                    heaps: [MemHeap::default(); VK_MAX_MEMORY_HEAPS],
                    types: [MemType::default(); VK_MAX_MEMORY_TYPES],
                };
                let fp: GetProps =
                    match self.gpa(inst, "vkGetPhysicalDeviceMemoryProperties") {
                        Some(p) => std::mem::transmute(p),
                        None => break,
                    };
                fp(*d, &mut mp as *mut MemProps as *mut c_void);
                let nh = mp.heap_count.min(VK_MAX_MEMORY_HEAPS as u32) as usize;
                let nt = mp.type_count.min(VK_MAX_MEMORY_TYPES as u32) as usize;
                let mut local = 0u64;
                let mut visible = 0u64;
                for k in 0..nh {
                    let h = mp.heaps[k];
                    if h.flags & MHEAP_DEVICE_LOCAL != 0 {
                        local = local.max(h.size);
                    }
                }
                for t in 0..nt {
                    if mp.types[t].property_flags & MTYPE_HOST_VISIBLE == 0 {
                        continue;
                    }
                    let k = mp.types[t].heap_index as usize;
                    if k < nh {
                        visible = visible.max(mp.heaps[k].size);
                    }
                }
                // An iGPU's one heap is device-local *and* the same block of RAM
                // the host sees, so "dedicated" means a device-local heap strictly
                // smaller than that shared heap - real VRAM. WDDM exposes the VRAM
                // as host-visible too (BAR window), which is why size, not flags,
                // is what separates it from system RAM.
                let host_heaps: Vec<usize> = (0..nt)
                    .filter(|&t| mp.types[t].property_flags & MTYPE_HOST_VISIBLE != 0)
                    .map(|t| mp.types[t].heap_index as usize)
                    .collect();
                let dedicated = (0..nh)
                    .filter(|k| mp.heaps[*k].flags & MHEAP_DEVICE_LOCAL != 0 && mp.heaps[*k].size < visible)
                    .map(|k| mp.heaps[k].size)
                    .max()
                    .unwrap_or(0);
                let heaps: Vec<(u64, u32, bool)> = (0..nh)
                    .map(|k| (mp.heaps[k].size, mp.heaps[k].flags, host_heaps.contains(&k)))
                    .collect();
                out.push(Device {
                    index: i,
                    name,
                    vendor_id: vendor,
                    device_id,
                    class,
                    api: version_string(api),
                    driver: driver_string(vendor, driver),
                    device_local_gib: local as f64 / 2f64.powi(30),
                    dedicated_gib: dedicated as f64 / 2f64.powi(30),
                    host_visible_gib: visible as f64 / 2f64.powi(30),
                    heaps,
                    extensions,
                    vendor: vendor_name(vendor),
                    rank: rank_of(class, vendor),
                });
            }
            out.sort_by(|a, b| b.rank.cmp(&a.rank).then_with(|| b.device_local_gib.total_cmp(&a.device_local_gib)));
            // The reported index is the ranked one, so the handle list has to be
            // reordered with it: `Device.index` is what a caller passes back to
            // select a device, and enumeration order is not rank order.
            self.devices = out.iter().map(|d| list[d.index]).collect();
            for (i, d) in out.iter_mut().enumerate() {
                d.index = i;
            }
            out
        }
    }

    /// NVIDIA's R510+ packing: major<<22 | minor<<14 | subminor<<6 | build.
    /// Checked on this box against `nvidia-smi`: 0x9b438000 -> 621.14.0.
    pub fn nvidia_driver(d: u32) -> (u32, u32, u32) {
        ((d >> 22) & 0x3ff, (d >> 14) & 0xff, (d >> 6) & 0xff)
    }

    /// The live instance, for the compute tier. Null until `devices()` has run.
    pub(crate) fn handle(&self) -> *mut c_void {
        self.instance
    }

    /// Physical device by *ranked* index, the same one `Device.index` reports.
    pub(crate) fn phys(&self, index: usize) -> Option<*mut c_void> {
        self.devices.get(index).copied()
    }

    pub(crate) fn proc(&self, inst: *mut c_void, name: &str) -> Option<*mut c_void> {
        self.gpa(inst, name)
    }

    /// A portability driver (MoltenVK) advertises VK_KHR_portability_subset.
    /// That is the honest signal that these Vulkan calls become Metal, since
    /// the reported device name is the real GPU, not the translation layer.
    unsafe fn device_extensions(&self, inst: *mut c_void, phys: *mut c_void) -> Vec<String> {
        type Enum =
            unsafe extern "system" fn(*mut c_void, *const c_char, *mut u32, *mut c_void) -> i32;
        let Some(p) = self.gpa(inst, "vkEnumerateDeviceExtensionProperties") else {
            return Vec::new();
        };
        let fp: Enum = unsafe { std::mem::transmute(p) };
        let mut n = 0u32;
        if unsafe { fp(phys, std::ptr::null(), &mut n, std::ptr::null_mut()) } != 0 || n == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; n as usize * (VK_MAX_EXTENSION_NAME + 4)];
        if unsafe { fp(phys, std::ptr::null(), &mut n, buf.as_mut_ptr() as *mut c_void) } != 0 {
            return Vec::new();
        }
        extension_names(&buf)
    }
}


pub fn class_name(c: u32) -> &'static str {
    match c {
        0 => "other",
        1 => "integrated GPU",
        2 => "discrete GPU",
        3 => "virtual GPU",
        4 => "CPU",
        _ => "unknown",
    }
}

/// VK_API_VERSION_MAJOR / MINOR / PATCH, as bitfields, not decimals.
pub fn version_string(v: u32) -> String {
    format!("{}.{}.{}", (v >> 22) & 0x3ff, (v >> 12) & 0x3ff, v & 0xfff)
}

pub fn driver_string(vendor_id: u32, d: u32) -> String {
    match vendor_id {
        0x10DE => {
            let (a, b, c) = Vulkan::nvidia_driver(d);
            format!("{a}.{b}.{c}")
        }
        // Intel and AMD each publish their own packing and neither matches the
        // Windows driver version, so show the raw value rather than a wrong sum
        _ => format!("0x{d:08x} (raw, vendor packing)"),
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        if self.instance.is_null() || !self.loaded {
            return;
        }
        unsafe {
            type Destroy = unsafe extern "system" fn(*mut c_void, *const c_void);
            if let Some(p) = self.gpa(std::ptr::null_mut(), "vkDestroyInstance") {
                let fp: Destroy = std::mem::transmute(p);
                fp(self.instance, std::ptr::null());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(dedicated: f64, local: f64, extensions: &[&str], vendor_id: u32) -> Device {
        Device {
            index: 0,
            name: "test".into(),
            vendor_id,
            device_id: 0,
            class: 0,
            api: "1.3".into(),
            driver: "0".into(),
            device_local_gib: local,
            dedicated_gib: dedicated,
            host_visible_gib: local,
            heaps: Vec::new(),
            extensions: extensions.iter().map(|s| s.to_string()).collect(),
            vendor: "test",
            rank: 0,
        }
    }

    #[test]
    fn the_env_override_wins_and_blank_ones_are_dropped() {
        let got = loader_candidates(&|k| (k == "FENNEC_VULKAN_LIB").then(|| "  /opt/libvk.so  ".into()), None);
        assert_eq!(got[0], "/opt/libvk.so");
        let blank = loader_candidates(&|_| Some("   ".into()), None);
        assert!(!blank.iter().any(|p| p.trim().is_empty()));
        assert!(blank.contains(&LOADER_NAMES[0].to_string()));
    }

    #[test]
    fn every_platform_offers_its_own_loader_names() {
        let got = loader_candidates(&|_| None, None);
        for name in LOADER_NAMES {
            assert!(got.contains(&name.to_string()), "{name} missing from {got:?}");
        }
    }

    #[test]
    fn macos_looks_where_brew_and_the_sdk_put_moltenvk() {
        let got = loader_candidates(&|_| None, Some("/Applications/Fennec.app/Contents/MacOS"));
        if cfg!(target_os = "macos") {
            assert!(got.iter().any(|p| p == "/opt/homebrew/lib/libMoltenVK.dylib"));
            assert!(got.iter().any(|p| p.ends_with("../Frameworks/libMoltenVK.dylib")));
        } else {
            assert!(!got.iter().any(|p| p.contains("MoltenVK")));
        }
    }

    #[test]
    fn extension_names_reads_the_256_byte_records() {
        let mut raw = Vec::new();
        for name in ["VK_KHR_portability_subset", "VK_KHR_swapchain"] {
            let mut rec = [0u8; VK_MAX_EXTENSION_NAME + 4];
            rec[..name.len()].copy_from_slice(name.as_bytes());
            rec[VK_MAX_EXTENSION_NAME..].copy_from_slice(&7u32.to_le_bytes());
            raw.extend_from_slice(&rec);
        }
        assert_eq!(extension_names(&raw), ["VK_KHR_portability_subset", "VK_KHR_swapchain"]);
        assert!(extension_names(&raw[..VK_MAX_EXTENSION_NAME + 3]).is_empty());
    }

    #[test]
    fn apple_unified_memory_has_no_dedicated_heap_but_is_still_usable() {
        // One heap that is both DEVICE_LOCAL and HOST_VISIBLE, so the
        // `heap.size < visible` test for dedicated memory yields zero.
        let apple = dev(0.0, 16.0, &["VK_KHR_portability_subset"], 0x106B);
        assert_eq!(apple.dedicated_gib, 0.0);
        assert_eq!(apple.usable_gib(), 16.0);
        assert!(apple.portability());
        assert_eq!(apple.preferred_backend(), crate::backend::Backend::Metal);
    }

    #[test]
    fn a_discrete_gpu_reports_its_dedicated_heap_and_vulkan_or_cuda() {
        let nv = dev(7.77, 15.76, &[], 0x10DE);
        assert_eq!(nv.usable_gib(), 7.77);
        assert!(!nv.portability());
        assert_eq!(nv.preferred_backend(), crate::backend::Backend::Cuda);
        let amd = dev(16.0, 31.5, &[], 0x1002);
        assert_eq!(amd.preferred_backend(), crate::backend::Backend::Vulkan);
    }
}
