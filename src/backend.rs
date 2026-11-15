//! Which compute backend a device gets. The default is the vendor, not a
//! benchmark: NVIDIA runs CUDA, Apple runs Metal, everyone else runs Vulkan.
//!
//! AMD stays on Vulkan deliberately. The Vulkan path names no vendor extension,
//! so the same SPIR-V and the same descriptor layout already work on RDNA; a
//! HIP backend would add a toolchain and a second kernel source to maintain for
//! hardware this project has never measured.
//!
//! Metal is reached through MoltenVK, the ICD that translates Vulkan calls and
//! SPIR-V to Metal. `Metal` and `Vulkan` therefore share one code path and differ
//! only in the loader and the portability hooks, which `vk.rs` detects on its
//! own. `Cuda` is a genuinely different API: the driver library that ships with
//! the NVIDIA driver, with PTX checked in beside the SPIR-V.
//!
//! TensorRT is not an option here. It compiles a whole ONNX graph into an engine
//! and owns the memory, which cannot express the tiered residency this engine is
//! built around - hot experts in VRAM, the rest in a host pool, the CPU working
//! its share while device batches are in flight.
//!
//! `Ort` is the portability wrapper for the devices Fennec has no hand-written
//! kernel for: ONNX Runtime reaches DirectML (AMD/Intel GPU), OpenVINO (Intel
//! NPU) and CoreML (Apple) through one API. It accelerates the *dense* F16
//! projections as ONNX MatMul graphs; the int4 routed experts stay on Fennec's
//! own host kernels, because there is no ONNX export of that MoE and ORT's
//! session allocator cannot express tiered residency. NVIDIA keeps the
//! hand-written Driver-API/PTX path - it needs no runtime and was measured level
//! with ORT's CUDA EP, so there is nothing to gain by routing it through ORT.

/// An explicit choice, from `--backend` or the environment.
pub const BACKENDS: [&str; 5] = ["cuda", "vulkan", "metal", "cpu", "ort"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Vulkan,
    Metal,
    Cpu,
    Ort,
}

impl Backend {
    /// The vendor policy. `portability` is the device advertising
    /// `VK_KHR_portability_subset`, which is how a Mac is recognised even when
    /// the GPU reports a vendor id we do not name.
    pub fn for_vendor(vendor_id: u32, portability: bool) -> Backend {
        match vendor_id {
            0x10DE => Backend::Cuda,
            0x106B => Backend::Metal,
            _ if portability => Backend::Metal,
            _ => Backend::Vulkan,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
            Backend::Ort => "ort",
        }
    }

    /// What the user sees in `probe` and the startup line.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA (NVIDIA driver API, PTX)",
            Backend::Vulkan => "Vulkan (SPIR-V)",
            Backend::Metal => "Metal (MoltenVK, SPIR-V translated)",
            Backend::Cpu => "CPU only",
            Backend::Ort => "ONNX Runtime (DirectML / OpenVINO / CoreML)",
        }
    }

    pub fn parse(text: &str) -> Option<Backend> {
        match text.trim().to_ascii_lowercase().as_str() {
            "cuda" | "nvidia" | "nv" => Some(Backend::Cuda),
            "vulkan" | "vk" => Some(Backend::Vulkan),
            "metal" | "moltenvk" | "apple" | "mps" => Some(Backend::Metal),
            "cpu" | "host" | "none" => Some(Backend::Cpu),
            "ort" | "onnx" | "onnxruntime" | "directml" | "dml" | "openvino" | "coreml" => {
                Some(Backend::Ort)
            }
            _ => None,
        }
    }

    /// True when this backend is the Vulkan API, whichever name it goes by.
    pub fn is_vulkan(self) -> bool {
        matches!(self, Backend::Vulkan | Backend::Metal)
    }
}

/// The environment the suite already speaks. `FENNEC_BACKEND` is the explicit
/// knob; the per-backend flags `FENNEC_CUDA` / `FENNEC_METAL` / `FENNEC_VULKAN`
/// switch it on too.
pub fn from_env(env: &dyn Fn(&str) -> Option<String>) -> Option<Backend> {
    if let Some(text) = env("FENNEC_BACKEND") {
        if let Some(b) = Backend::parse(&text) {
            return Some(b);
        }
    }
    for (key, backend) in [
        ("FENNEC_CUDA", Backend::Cuda),
        ("FENNEC_METAL", Backend::Metal),
        ("FENNEC_VULKAN", Backend::Vulkan),
    ] {
        if env(key).is_some_and(|v| flag(&v)) {
            return Some(backend);
        }
    }
    None
}

fn flag(value: &str) -> bool {
    !matches!(value.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off")
}

/// Explicit request beats the environment, which beats the vendor policy.
/// An unparseable request is an error rather than a silent fallback, so a typo
/// in `--backend` cannot quietly put the model on the wrong device.
pub fn resolve(
    requested: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
    vendor_id: u32,
    portability: bool,
) -> Result<Backend, String> {
    if let Some(text) = requested {
        return Backend::parse(text)
            .ok_or_else(|| format!("unknown backend {text:?}; expected one of {}", BACKENDS.join(", ")));
    }
    Ok(from_env(env).unwrap_or_else(|| Backend::for_vendor(vendor_id, portability)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn the_vendor_policy_is_cuda_nvidia_metal_apple_vulkan_everyone_else() {
        assert_eq!(Backend::for_vendor(0x10DE, false), Backend::Cuda);
        assert_eq!(Backend::for_vendor(0x1002, false), Backend::Vulkan);
        assert_eq!(Backend::for_vendor(0x8086, false), Backend::Vulkan);
        assert_eq!(Backend::for_vendor(0x106B, false), Backend::Metal);
        assert_eq!(Backend::for_vendor(0x13B5, false), Backend::Vulkan);
        // A portability driver is Metal even under a vendor id we do not name.
        assert_eq!(Backend::for_vendor(0x1002, true), Backend::Metal);
        assert_eq!(Backend::for_vendor(0, true), Backend::Metal);
    }

    #[test]
    fn an_explicit_request_beats_the_environment_and_the_vendor() {
        let e = env(&[("FENNEC_BACKEND", "vulkan")]);
        assert_eq!(resolve(Some("cpu"), &e, 0x10DE, false).unwrap(), Backend::Cpu);
        assert_eq!(resolve(None, &e, 0x10DE, false).unwrap(), Backend::Vulkan);
        assert_eq!(resolve(None, &env(&[]), 0x10DE, false).unwrap(), Backend::Cuda);
    }

    #[test]
    fn a_typo_is_an_error_not_a_silent_fallback() {
        let err = resolve(Some("cudaa"), &env(&[]), 0x10DE, false).unwrap_err();
        assert!(err.contains("cudaa"), "{err}");
        assert!(err.contains("cuda, vulkan, metal, cpu, ort"), "{err}");
    }

    #[test]
    fn the_backend_flags_switch_the_device_tier_on() {
        for (flag, want) in [
            ("FENNEC_CUDA", Backend::Cuda),
            ("FENNEC_METAL", Backend::Metal),
            ("FENNEC_VULKAN", Backend::Vulkan),
        ] {
            assert_eq!(from_env(&env(&[(flag, "1")])), Some(want));
            for off in ["0", "false", "no", "off", ""] {
                assert_eq!(from_env(&env(&[(flag, off)])), None, "{flag}={off:?}");
            }
        }
        // The first flag in the list wins, so the order is the precedence.
        let e = env(&[("FENNEC_CUDA", "1"), ("FENNEC_VULKAN", "1")]);
        assert_eq!(from_env(&e), Some(Backend::Cuda));
        // FENNEC_BACKEND wins over a flag that disagrees.
        let e = env(&[("FENNEC_BACKEND", "metal"), ("FENNEC_CUDA", "1")]);
        assert_eq!(from_env(&e), Some(Backend::Metal));
    }

    #[test]
    fn parsing_accepts_the_names_people_actually_type() {
        for (text, want) in [
            ("CUDA", Backend::Cuda),
            (" cuda ", Backend::Cuda),
            ("nvidia", Backend::Cuda),
            ("vk", Backend::Vulkan),
            ("MoltenVK", Backend::Metal),
            ("apple", Backend::Metal),
            ("CPU", Backend::Cpu),
            ("ort", Backend::Ort),
            ("ONNX", Backend::Ort),
            ("directml", Backend::Ort),
            ("openvino", Backend::Ort),
            ("coreml", Backend::Ort),
        ] {
            assert_eq!(Backend::parse(text), Some(want), "{text:?}");
        }
        assert_eq!(Backend::parse("tensorrt"), None);
        assert_eq!(Backend::parse(""), None);
    }

    #[test]
    fn ort_is_the_portability_wrapper_and_is_not_the_vulkan_path() {
        assert_eq!(Backend::Ort.name(), "ort");
        assert!(!Backend::Ort.is_vulkan());
        // An explicit ort request beats the vendor policy, even on NVIDIA.
        assert_eq!(resolve(Some("ort"), &env(&[]), 0x10DE, false).unwrap(), Backend::Ort);
        assert_eq!(resolve(Some("directml"), &env(&[]), 0x1002, false).unwrap(), Backend::Ort);
        // The vendor default is unchanged: NVIDIA keeps the hand-written CUDA path.
        assert_eq!(resolve(None, &env(&[]), 0x10DE, false).unwrap(), Backend::Cuda);
    }

    #[test]
    fn metal_and_vulkan_share_the_vulkan_code_path() {
        assert!(Backend::Metal.is_vulkan());
        assert!(Backend::Vulkan.is_vulkan());
        assert!(!Backend::Cuda.is_vulkan());
        assert!(!Backend::Cpu.is_vulkan());
    }
}