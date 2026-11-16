//! Windows device discovery beyond Vulkan: DXGI for GPUs, DXCore for the NPU.
//!
//! Vulkan does not enumerate an NPU. On Windows the Intel AI Boost (and any
//! other `ComputeAccelerator`) is exposed through **DXCore**, not DXGI, as a
//! non-graphics `GENERIC_ML` adapter - the rule Microsoft's own
//! `Samples/DirectMLNpuInference` uses. This module finds it and reports
//! whether a D3D12 device can actually be created on it, because finding the
//! adapter and being allowed to compute on it are two different things.
//!
//! GUIDs and vtable order are transcribed from `dxcore_interface.h` in
//! microsoft/DirectX-Headers. No SDK, no import library: everything is loaded
//! at runtime, same as the Vulkan path.

#[cfg(windows)]
use crate::vk::vendor_name;

#[derive(Clone, Debug)]
pub struct Adapter {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub vendor: &'static str,
    pub dedicated_gib: f64,
    pub shared_gib: f64,
    pub software: bool,
    pub graphics: bool,
    pub generic_ml: bool,
    pub core_compute: bool,
}

impl Adapter {
    /// Microsoft's NPU rule: not a graphics adapter, but ML or core compute.
    pub fn is_npu(&self) -> bool {
        !self.graphics && (self.generic_ml || self.core_compute)
    }
}

#[derive(Clone, Debug)]
pub struct Npu {
    pub adapter: Adapter,
    /// `D3D12CreateDevice` HRESULT at feature level 1_0_GENERIC. `None` when
    /// d3d12.dll could not be loaded at all.
    pub d3d12_hr: Option<i32>,
}

impl Npu {
    pub fn usable(&self) -> bool {
        self.d3d12_hr == Some(0)
    }

    /// The failure in words, so `probe` never prints a bare hex code.
    pub fn verdict(&self) -> String {
        match self.d3d12_hr {
            Some(0) => "D3D12 device created: DirectML compute is available".into(),
            Some(hr) => format!(
                "D3D12CreateDevice refused ({hr:#010x} {}): the NPU is detected but this \
                 driver/runtime pair will not give it a compute device",
                hresult_name(hr)
            ),
            None => "d3d12.dll did not load, so no compute device was attempted".into(),
        }
    }
}

pub fn hresult_name(hr: i32) -> &'static str {
    match hr as u32 {
        0 => "S_OK",
        0x887a0004 => "DXGI_ERROR_UNSUPPORTED",
        0x887a0001 => "DXGI_ERROR_INVALID_CALL",
        0x887a0002 => "DXGI_ERROR_NOT_FOUND",
        0x887a0005 => "DXGI_ERROR_DEVICE_REMOVED",
        0x80070057 => "E_INVALIDARG",
        0x80004002 => "E_NOINTERFACE",
        0x8007000e => "E_OUTOFMEMORY",
        _ => "unknown",
    }
}

#[derive(Debug, Default)]
pub struct Report {
    /// DXGI adapters: the GPUs, plus the software renderer.
    pub adapters: Vec<Adapter>,
    /// DXCore non-graphics ML/compute adapters.
    pub npus: Vec<Npu>,
    pub directml_loadable: bool,
    /// Non-fatal: why one of the two enumerations came back empty.
    pub notes: Vec<String>,
}

impl Report {
    pub fn amd(&self) -> Vec<&Adapter> {
        self.adapters.iter().filter(|a| a.vendor_id == 0x1002).collect()
    }
    pub fn nvidia(&self) -> Vec<&Adapter> {
        self.adapters.iter().filter(|a| a.vendor_id == 0x10DE).collect()
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use libloading::Library;
    use std::ffi::c_void;

    type Handle = *mut c_void;
    type Hresult = i32;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Guid {
        a: u32,
        b: u16,
        c: u16,
        d: [u8; 8],
    }

    const fn g(a: u32, b: u16, c: u16, d: [u8; 8]) -> Guid {
        Guid { a, b, c, d }
    }

    const IID_IDXGIFACTORY1: Guid =
        g(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]);
    const IID_FACTORY: Guid =
        g(0x78ee5945, 0xc36e, 0x4b13, [0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06]);
    const IID_LIST: Guid =
        g(0x526c7776, 0x40e9, 0x459b, [0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28]);
    const IID_ADAPTER: Guid =
        g(0xf0db4c7f, 0xfe5a, 0x42a2, [0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e]);
    const IID_ID3D12DEVICE: Guid =
        g(0x189819f1, 0x1db6, 0x4b57, [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7]);

    const ATTR_GRAPHICS: Guid =
        g(0x0c9ece4d, 0x2f6e, 0x4f01, [0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1]);
    const ATTR_CORE_COMPUTE: Guid =
        g(0x248e2800, 0xa793, 0x4724, [0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90]);
    const ATTR_GENERIC_ML: Guid =
        g(0xb71b0d41, 0x1088, 0x422f, [0xa2, 0x7c, 0x02, 0x50, 0xb7, 0xd3, 0xa9, 0x88]);

    // DXGI_ADAPTER_DESC1: WCHAR[128], four UINTs, three SIZE_Ts, LUID, flags.
    #[repr(C)]
    struct AdapterDesc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video: usize,
        dedicated_system: usize,
        shared_system: usize,
        luid_low: u32,
        luid_high: i32,
        flags: u32,
    }

    type CreateDxgiFactory1 = unsafe extern "system" fn(*const Guid, *mut Handle) -> Hresult;
    type EnumAdapters1 = unsafe extern "system" fn(Handle, u32, *mut Handle) -> Hresult;
    type GetDesc1 = unsafe extern "system" fn(Handle, *mut AdapterDesc1) -> Hresult;
    type Release = unsafe extern "system" fn(Handle) -> u32;
    type CreateAdapterFactory = unsafe extern "system" fn(*const Guid, *mut Handle) -> Hresult;
    type CreateAdapterList =
        unsafe extern "system" fn(Handle, u32, *const Guid, *const Guid, *mut Handle) -> Hresult;
    type GetAdapter = unsafe extern "system" fn(Handle, u32, *const Guid, *mut Handle) -> Hresult;
    type GetAdapterCount = unsafe extern "system" fn(Handle) -> u32;
    type IsAttributeSupported = unsafe extern "system" fn(Handle, *const Guid) -> u8;
    type GetPropertySize = unsafe extern "system" fn(Handle, u32, *mut usize) -> Hresult;
    type GetProperty = unsafe extern "system" fn(Handle, u32, usize, *mut c_void) -> Hresult;
    type D3D12CreateDevice =
        unsafe extern "system" fn(Handle, u32, *const Guid, *mut Handle) -> Hresult;

    /// COM calls are vtable slots, not exports. Every interface here derives
    /// IUnknown (3 slots); IDXCore* derive it directly, IDXGIFactory also
    /// carries IDXGIObject's four before its own five.
    unsafe fn vt(obj: Handle, i: usize) -> *mut c_void {
        let v = unsafe { *(obj as *const *const *mut c_void) };
        unsafe { *v.add(i) }
    }

    unsafe fn release(obj: Handle) {
        if !obj.is_null() {
            let r: Release = unsafe { std::mem::transmute(vt(obj, 2)) };
            unsafe { r(obj) };
        }
    }

    unsafe fn sym<T>(lib: &Library, name: &[u8]) -> Option<T> {
        let s = unsafe { lib.get::<*mut c_void>(name) }.ok()?;
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&*s) })
    }

    fn gib(b: usize) -> f64 {
        b as f64 / 2f64.powi(30)
    }

    unsafe fn dxgi(out: &mut Report) {
        let Ok(lib) = (unsafe { Library::new("dxgi.dll") }) else {
            out.notes.push("dxgi.dll did not load".into());
            return;
        };
        let Some(create) = (unsafe { sym::<CreateDxgiFactory1>(&lib, b"CreateDXGIFactory1\0") }) else {
            out.notes.push("CreateDXGIFactory1 missing from dxgi.dll".into());
            return;
        };
        let mut factory: Handle = std::ptr::null_mut();
        if unsafe { create(&IID_IDXGIFACTORY1, &mut factory) } != 0 || factory.is_null() {
            out.notes.push("CreateDXGIFactory1 failed".into());
            return;
        }
        // IUnknown(3) + IDXGIObject(4) + IDXGIFactory(5) -> EnumAdapters1 at 12.
        let enum_adapters: EnumAdapters1 = unsafe { std::mem::transmute(vt(factory, 12)) };
        for i in 0..32u32 {
            let mut a: Handle = std::ptr::null_mut();
            if unsafe { enum_adapters(factory, i, &mut a) } != 0 || a.is_null() {
                break;
            }
            // IDXGIAdapter1::GetDesc1 is slot 10.
            let get_desc: GetDesc1 = unsafe { std::mem::transmute(vt(a, 10)) };
            let mut d: AdapterDesc1 = unsafe { std::mem::zeroed() };
            if unsafe { get_desc(a, &mut d) } == 0 {
                let end = d.description.iter().position(|&c| c == 0).unwrap_or(128);
                out.adapters.push(Adapter {
                    name: String::from_utf16_lossy(&d.description[..end]),
                    vendor_id: d.vendor_id,
                    device_id: d.device_id,
                    vendor: vendor_name(d.vendor_id),
                    dedicated_gib: gib(d.dedicated_video),
                    shared_gib: gib(d.shared_system),
                    software: d.flags & 2 != 0,
                    graphics: true,
                    generic_ml: false,
                    core_compute: false,
                });
            }
            unsafe { release(a) };
        }
        unsafe { release(factory) };
    }

    unsafe fn driver_description(adapter: Handle) -> String {
        // DXCoreAdapterProperty::DriverDescription = 2, a null-terminated char[].
        let size_of: GetPropertySize = unsafe { std::mem::transmute(vt(adapter, 7)) };
        let get: GetProperty = unsafe { std::mem::transmute(vt(adapter, 6)) };
        let mut n = 0usize;
        if unsafe { size_of(adapter, 2, &mut n) } != 0 || n == 0 || n > 4096 {
            return "<no description>".into();
        }
        let mut buf = vec![0u8; n];
        if unsafe { get(adapter, 2, n, buf.as_mut_ptr() as *mut c_void) } != 0 {
            return "<unreadable>".into();
        }
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }

    /// DXCore has no DXGI descriptor: ids come from HardwareIDParts (property
    /// 14, five u32) and the memory sizes are separate u64 properties.
    unsafe fn hardware(adapter: Handle) -> (u32, u32, f64, f64) {
        let size_of: GetPropertySize = unsafe { std::mem::transmute(vt(adapter, 7)) };
        let get: GetProperty = unsafe { std::mem::transmute(vt(adapter, 6)) };
        let mut ids = [0u32; 5];
        let mut n = 0usize;
        if unsafe { size_of(adapter, 14, &mut n) } == 0 && n == std::mem::size_of_val(&ids) {
            let _ = unsafe { get(adapter, 14, n, ids.as_mut_ptr() as *mut c_void) };
        }
        let mem = |prop: u32| -> f64 {
            let mut v = 0u64;
            let mut n = 0usize;
            if unsafe { size_of(adapter, prop, &mut n) } == 0
                && n == 8
                && unsafe { get(adapter, prop, 8, &mut v as *mut u64 as *mut c_void) } == 0
            {
                gib(v as usize)
            } else {
                0.0
            }
        };
        (ids[0], ids[1], mem(7), mem(9))
    }

    unsafe fn dxcore(out: &mut Report, d3d12: Option<&Library>) {
        let Ok(lib) = (unsafe { Library::new("dxcore.dll") }) else {
            out.notes.push("dxcore.dll did not load: no NPU enumeration".into());
            return;
        };
        let Some(create) = (unsafe { sym::<CreateAdapterFactory>(&lib, b"DXCoreCreateAdapterFactory\0") }) else {
            out.notes.push("DXCoreCreateAdapterFactory missing".into());
            return;
        };
        let mut factory: Handle = std::ptr::null_mut();
        if unsafe { create(&IID_FACTORY, &mut factory) } != 0 || factory.is_null() {
            out.notes.push("DXCoreCreateAdapterFactory failed".into());
            return;
        }
        let make_list: CreateAdapterList = unsafe { std::mem::transmute(vt(factory, 3)) };
        let create_device =
            d3d12.and_then(|l| unsafe { sym::<D3D12CreateDevice>(l, b"D3D12CreateDevice\0") });
        // GENERIC_ML is the attribute an NPU carries; CORE_COMPUTE is the
        // fallback the sample tries when no ML adapter exists.
        for attr in [ATTR_GENERIC_ML, ATTR_CORE_COMPUTE] {
            let mut list: Handle = std::ptr::null_mut();
            if unsafe { make_list(factory, 1, &attr, &IID_LIST, &mut list) } != 0 || list.is_null() {
                continue;
            }
            let count: GetAdapterCount = unsafe { std::mem::transmute(vt(list, 4)) };
            let get_adapter: GetAdapter = unsafe { std::mem::transmute(vt(list, 3)) };
            for i in 0..unsafe { count(list) } {
                let mut a: Handle = std::ptr::null_mut();
                if unsafe { get_adapter(list, i, &IID_ADAPTER, &mut a) } != 0 || a.is_null() {
                    continue;
                }
                let is_attr: IsAttributeSupported = unsafe { std::mem::transmute(vt(a, 4)) };
                let (vendor_id, device_id, dedicated_gib, shared_gib) = unsafe { hardware(a) };
                let ad = Adapter {
                    name: unsafe { driver_description(a) },
                    vendor_id,
                    device_id,
                    vendor: if vendor_id == 0 { "unknown" } else { vendor_name(vendor_id) },
                    dedicated_gib,
                    shared_gib,
                    software: false,
                    graphics: unsafe { is_attr(a, &ATTR_GRAPHICS) } != 0,
                    generic_ml: unsafe { is_attr(a, &ATTR_GENERIC_ML) } != 0,
                    core_compute: unsafe { is_attr(a, &ATTR_CORE_COMPUTE) } != 0,
                };
                if ad.is_npu() && !out.npus.iter().any(|n| n.adapter.name == ad.name) {
                    // D3D_FEATURE_LEVEL_1_0_GENERIC. 1_0_CORE is the fallback.
                    let hr = create_device.map(|cd| {
                        let mut dev: Handle = std::ptr::null_mut();
                        let hr = unsafe { cd(a, 0x100, &IID_ID3D12DEVICE, &mut dev) };
                        if hr == 0 {
                            unsafe { release(dev) };
                        }
                        hr
                    });
                    out.npus.push(Npu { adapter: ad, d3d12_hr: hr });
                }
                unsafe { release(a) };
            }
            unsafe { release(list) };
        }
        unsafe { release(factory) };
    }

    pub fn probe() -> Report {
        let mut out = Report::default();
        unsafe {
            let d3d12 = Library::new("d3d12.dll").ok();
            out.directml_loadable = Library::new("DirectML.dll").is_ok();
            dxgi(&mut out);
            dxcore(&mut out, d3d12.as_ref());
        }
        out
    }
}

#[cfg(windows)]
pub fn probe() -> Report {
    imp::probe()
}

#[cfg(not(windows))]
pub fn probe() -> Report {
    Report {
        notes: vec!["DXGI and DXCore are Windows APIs; this build enumerates Vulkan only".into()],
        ..Report::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ad(graphics: bool, ml: bool, compute: bool) -> Adapter {
        Adapter {
            name: "x".into(),
            vendor_id: 0x8086,
            device_id: 0,
            vendor: "Intel",
            dedicated_gib: 0.0,
            shared_gib: 0.0,
            software: false,
            graphics,
            generic_ml: ml,
            core_compute: compute,
        }
    }

    #[test]
    fn the_npu_rule_excludes_every_gpu() {
        // A GPU reports GENERIC_ML too, so the attribute alone is not the test:
        // it is the absence of graphics that makes it an NPU.
        assert!(!ad(true, true, true).is_npu());
        assert!(ad(false, true, false).is_npu());
        assert!(ad(false, false, true).is_npu());
        assert!(!ad(false, false, false).is_npu());
        assert!(!ad(true, false, false).is_npu());
    }

    #[test]
    fn a_refused_device_is_not_reported_as_usable() {
        let n = Npu { adapter: ad(false, true, false), d3d12_hr: Some(0x887a0004u32 as i32) };
        assert!(!n.usable());
        assert!(n.verdict().contains("DXGI_ERROR_UNSUPPORTED"));
        assert!(n.verdict().contains("detected"));
        let ok = Npu { adapter: ad(false, true, false), d3d12_hr: Some(0) };
        assert!(ok.usable());
        let missing = Npu { adapter: ad(false, true, false), d3d12_hr: None };
        assert!(!missing.usable());
        assert!(missing.verdict().contains("d3d12.dll"));
    }

    #[test]
    fn vendor_filters_pick_the_right_adapters() {
        let mut r = Report::default();
        let mut nv = ad(true, true, true);
        nv.vendor_id = 0x10DE;
        let mut amd = ad(true, true, true);
        amd.vendor_id = 0x1002;
        r.adapters.push(nv);
        r.adapters.push(amd);
        assert_eq!(r.nvidia().len(), 1);
        assert_eq!(r.amd().len(), 1);
        assert_eq!(r.amd()[0].vendor_id, 0x1002);
    }

    #[test]
    fn amd_and_nvidia_are_named_not_bucketed_as_other() {
        // Both vendors take the same Vulkan code path; nothing here may depend
        // on a vendor extension, so the ids have to resolve to real names.
        assert_eq!(crate::vk::vendor_name(0x1002), "AMD");
        assert_eq!(crate::vk::vendor_name(0x10DE), "NVIDIA");
        assert_eq!(crate::vk::vendor_name(0x8086), "Intel");
    }
}