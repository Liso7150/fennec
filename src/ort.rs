//! ONNX Runtime as the portability wrapper for the devices Fennec has no
//! hand-written kernel for: DirectML (AMD/Intel GPU), OpenVINO (Intel NPU) and
//! CoreML (Apple). NVIDIA keeps the hand-written Driver-API/PTX path in
//! `cudagpu` - it needs no runtime and measured level with ORT's CUDA EP.
//!
//! ORT runs ONNX graphs and owns its session allocator, so it accelerates the
//! *dense* F16 projections (lm_head, the gated-delta-net and attention
//! projections, the shared expert) as MatMul graphs. The int4 routed experts
//! stay on Fennec's own host kernels: there is no ONNX export of that MoE and
//! ORT cannot express tiered residency. This module is therefore a dense
//! accelerator, not a replacement for the arena-based `Gpu` tier.
//!
//! The runtime is loaded at run time from a signed `onnxruntime.dll`
//! (`ORT_DYLIB_PATH`, or beside the executable) through the `ort` crate's
//! `load-dynamic` feature - no import library, no MSVC, no toolkit at build
//! time, the same contract as the Vulkan and CUDA paths. The crate is an
//! optional dependency behind the `ort` feature so the default offline build
//! is unchanged.
//!
//! Two layers live here. The graph builder and the execution-provider choice
//! are pure and always compiled, so they are unit-tested without the runtime.
//! The session wrapper is behind `#[cfg(feature = "ort")]` because it needs the
//! `ort` crate, which Smart App Control blocks from building on this box until
//! the engine is signed.

use crate::vk::Device;

/// Which ONNX Runtime execution provider a device gets. The wrapper names no
/// vendor extension: one API reaches every accelerator the runtime supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtEp {
    /// NVIDIA GPU through CUDA (cuDNN required).
    Cuda,
    /// AMD / Intel GPU through DirectX 12 (Windows 10 1903+).
    DirectML,
    /// Intel AI Boost NPU through the OpenVINO plugin (never goes via D3D12).
    OpenVINO,
    /// Apple Silicon through CoreML (macOS only).
    CoreML,
    /// The CPU EP, the fallback when no accelerator EP registers.
    Cpu,
}

impl OrtEp {
    pub fn name(self) -> &'static str {
        match self {
            OrtEp::Cuda => "CUDA",
            OrtEp::DirectML => "DirectML",
            OrtEp::OpenVINO => "OpenVINO",
            OrtEp::CoreML => "CoreML",
            OrtEp::Cpu => "CPU",
        }
    }

    /// Pick the provider from the device. The NPU is the Intel AI Boost adapter
    /// that DXCore reports as a non-graphics `GENERIC_ML` device (see `dx`);
    /// Vulkan does not enumerate it, so it is recognised by name here. Apple is
    /// the portability subset or the Apple vendor id; NVIDIA is CUDA; every
    /// other GPU on Windows is DirectML, and elsewhere the CPU EP is the fallback.
    pub fn for_device(dev: &Device) -> OrtEp {
        if dev.vendor_id == 0x8086 && dev.name.contains("AI Boost") {
            OrtEp::OpenVINO
        } else if dev.portability() || dev.vendor_id == 0x106B {
            OrtEp::CoreML
        } else if dev.vendor_id == 0x10DE {
            OrtEp::Cuda
        } else if cfg!(windows) {
            OrtEp::DirectML
        } else {
            OrtEp::Cpu
        }
    }

    /// The provider when no device could be enumerated (no Vulkan loader). The
    /// platform's own accelerator API is the honest default; the CPU EP is the
    /// last resort and is what the runtime falls back to per-node anyway.
    pub fn fallback() -> OrtEp {
        if cfg!(target_os = "macos") {
            OrtEp::CoreML
        } else if cfg!(windows) {
            OrtEp::DirectML
        } else {
            OrtEp::Cpu
        }
    }
}

// --- minimal ONNX protobuf writer -----------------------------------------
//
// ONNX is protobuf; a single-MatMul graph is small enough to emit by hand, the
// same way the benchmark lanes build their models. Field numbers are from the
// ONNX IR spec (onnx.ai/onnx/repo-docs/IR.html). Weights are graph initializers,
// so only the activation vector is a run-time input.

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        out.push(b | if v != 0 { 0x80 } else { 0 });
        if v == 0 {
            break;
        }
    }
}

fn tag(field: u32, wire: u8, out: &mut Vec<u8>) {
    varint(((field as u64) << 3) | wire as u64, out);
}

fn len_delim(field: u32, body: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(body.len() as u64, out);
    out.extend_from_slice(body);
}

fn string_field(field: u32, s: &str, out: &mut Vec<u8>) {
    len_delim(field, s.as_bytes(), out);
}

fn int_field(field: u32, v: u64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(v, out);
}

/// A tensor type with a fixed shape, as a ValueInfoProto / TypeProto body.
fn tensor_type(elem: u32, dims: &[i64], out: &mut Vec<u8>) {
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        int_field(1, d as u64, &mut dim); // Dimension.dim_value
        len_delim(1, &dim, &mut shape); // TensorShapeProto.dim
    }
    let mut tensor = Vec::new();
    int_field(1, elem as u64, &mut tensor); // Tensor.elem_type
    len_delim(2, &shape, &mut tensor); // Tensor.shape
    let mut ty = Vec::new();
    len_delim(1, &tensor, &mut ty); // TypeProto.tensor_type
    out.extend_from_slice(&ty);
}

fn value_info(name: &str, elem: u32, dims: &[i64], out: &mut Vec<u8>) {
    let mut vi = Vec::new();
    string_field(1, name, &mut vi); // ValueInfoProto.name
    let mut ty = Vec::new();
    tensor_type(elem, dims, &mut ty);
    len_delim(2, &ty, &mut vi); // ValueInfoProto.type
    out.extend_from_slice(&vi);
}

pub const F32: u32 = 1;

/// Build a single-MatMul ONNX graph computing `y = W . x`:
/// `W [rows, cols]` is an F32 initializer, `x [cols, 1]` the input, `y [rows, 1]`
/// the output. The engine's weights are widened from F16 once, here, when the
/// session is built - not per token. F32 is used because the CPU EP has no F16
/// MatMul kernel (it folds F16 to F32 anyway) and F32 is correct on every EP.
pub fn matmul_graph(rows: usize, cols: usize, w_f32: &[f32]) -> Vec<u8> {
    assert_eq!(w_f32.len(), rows * cols, "weight shape mismatch");

    // TensorProto initializer "W": dims, data_type=F32, name, raw_data.
    let mut w = Vec::new();
    tag(1, 0, &mut w);
    varint(rows as u64, &mut w); // dims[0]
    tag(1, 0, &mut w);
    varint(cols as u64, &mut w); // dims[1]
    int_field(2, F32 as u64, &mut w); // data_type
    string_field(8, "W", &mut w); // name
    let raw: Vec<u8> = w_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    len_delim(9, &raw, &mut w); // raw_data

    // NodeProto MatMul(W, x) -> y.
    let mut node = Vec::new();
    string_field(1, "W", &mut node); // input[0]
    string_field(1, "x", &mut node); // input[1]
    string_field(2, "y", &mut node); // output[0]
    string_field(3, "matmul", &mut node); // name
    string_field(4, "MatMul", &mut node); // op_type

    // GraphProto.
    let mut graph = Vec::new();
    len_delim(1, &node, &mut graph); // node
    string_field(2, "fennec_dense", &mut graph); // name
    len_delim(5, &w, &mut graph); // initializer
    let mut xin = Vec::new();
    value_info("x", F32, &[cols as i64, 1], &mut xin);
    len_delim(11, &xin, &mut graph); // input
    let mut yout = Vec::new();
    value_info("y", F32, &[rows as i64, 1], &mut yout);
    len_delim(12, &yout, &mut graph); // output

    // OperatorSetIdProto: default domain, opset 17.
    let mut opset = Vec::new();
    string_field(1, "", &mut opset);
    int_field(2, 17, &mut opset);

    // ModelProto.
    let mut model = Vec::new();
    int_field(1, 8, &mut model); // ir_version
    len_delim(8, &opset, &mut model); // opset_import
    len_delim(7, &graph, &mut model); // graph
    model
}

/// Widen the checkpoint's F16 dense weights to the F32 the graph initializer
/// carries. `f16` selects half vs bfloat16, matching `kernel::gemv_half`.
pub fn widen_half(bytes: &[u8], f16: bool) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            if f16 {
                half_to_f32(bits)
            } else {
                f32::from_bits((bits as u32) << 16)
            }
        })
        .collect()
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // subnormal: normalise
                let mut e = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

// --- the session wrapper (needs the ort crate) ----------------------------

#[cfg(feature = "ort")]
mod session {
    use super::{matmul_graph, widen_half, OrtEp};
    use crate::config::Result;
    use crate::vk::Device;
    use ort::session::Session;
    use ort::value::TensorRef;
    use std::path::Path;
    use std::sync::Mutex;

    /// The loaded runtime. One environment per process (ORT allows no more);
    /// sessions are built from it. `init_from` dlopens the signed DLL - no
    /// link-time dependency. `Send + Sync` because `Environment` is, so one
    /// `Arc<Ort>` can back every projection and every serve request thread.
    pub struct Ort {
        env: ort::environment::Environment,
        pub ep: OrtEp,
    }

    impl Ort {
        /// Load `onnxruntime.dll` from `path` (or `ORT_DYLIB_PATH` / beside the
        /// exe when `None`) and choose the provider for `dev`, or the platform
        /// fallback when no device could be enumerated.
        pub fn open(path: Option<&Path>, dev: Option<&Device>) -> Result<Ort> {
            let builder = match path {
                Some(p) => ort::init_from(p).map_err(|e| format!("ort init: {e}"))?,
                None => ort::init(),
            };
            let env = builder.build().map_err(|e| format!("ort env: {e}"))?;
            let ep = dev.map(OrtEp::for_device).unwrap_or_else(OrtEp::fallback);
            Ok(Ort { env, ep })
        }

        fn dispatch(&self) -> ort::ep::ExecutionProviderDispatch {
            let ep = match self.ep {
                OrtEp::Cuda => ort::ep::CUDA::default().build(),
                OrtEp::DirectML => ort::ep::DirectML::default().build(),
                OrtEp::OpenVINO => ort::ep::OpenVINO::default().build(),
                OrtEp::CoreML => ort::ep::CoreML::default().build(),
                OrtEp::Cpu => ort::ep::CPU::default().build(),
            };
            // A registration failure must raise, not silently fall back to CPU
            // and look accelerated while running at CPU speed.
            ep.error_on_failure()
        }

        /// Build a dense projection from already-widened F32 weights: a MatMul
        /// session whose initializer is this projection's own weights. One
        /// session per projection; the weight is resident, only `x` is fed in.
        pub fn dense_f32(&self, w: &[f32], rows: usize, cols: usize) -> Result<OrtDense> {
            let graph = matmul_graph(rows, cols, w);
            let session = Session::builder(&self.env)
                .map_err(|e| format!("ort builder: {e}"))?
                .with_execution_providers([self.dispatch()])
                .map_err(|e| format!("ort ep: {e}"))?
                .commit_from_memory(&graph)
                .map_err(|e| format!("ort session: {e}"))?;
            Ok(OrtDense { session: Mutex::new(session), rows, cols })
        }

        /// The same, widening the checkpoint's F16/BF16 weights once here rather
        /// than per token. `f16` selects half vs bfloat16, as `kernel::gemv_half`.
        pub fn dense(&self, w: &[u8], rows: usize, cols: usize, f16: bool) -> Result<OrtDense> {
            self.dense_f32(&widen_half(w, f16), rows, cols)
        }
    }

    /// One dense projection parked in an ORT session. `gemv` mirrors
    /// `Dense::gemv`'s `&self` shape: ORT's `run` needs `&mut Session`, so the
    /// session sits behind a mutex and the projection stays `Send + Sync` and
    /// shareable across the decode and serve threads exactly like the CPU
    /// weights it replaces.
    pub struct OrtDense {
        session: Mutex<Session>,
        rows: usize,
        cols: usize,
    }

    impl OrtDense {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn cols(&self) -> usize {
            self.cols
        }

        pub fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
            if x.len() != self.cols || out.len() != self.rows {
                return Err(format!(
                    "ort gemv shape: x {} cols {}, out {} rows {}",
                    x.len(), self.cols, out.len(), self.rows
                ));
            }
            let tensor = TensorRef::from_array_view(([self.cols, 1usize], x))
                .map_err(|e| format!("ort input: {e}"))?;
            let mut session = self.session.lock().map_err(|_| "ort session poisoned".to_string())?;
            let outputs = session.run(ort::inputs!["x" => tensor]).map_err(|e| format!("ort run: {e}"))?;
            let (_shape, y) = outputs["y"].try_extract_tensor::<f32>().map_err(|e| format!("ort output: {e}"))?;
            out.copy_from_slice(&y[..self.rows]);
            Ok(())
        }
    }
}

#[cfg(feature = "ort")]
pub use session::{Ort, OrtDense};

#[cfg(test)]
mod tests {
    use super::*;

    // Read back a length-delimited protobuf field, returning (field, wire, body).
    fn fields(buf: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < buf.len() {
            let (key, n) = read_varint(&buf[i..]);
            i += n;
            let field = (key >> 3) as u32;
            let wire = (key & 7) as u8;
            match wire {
                0 => {
                    let (_, n) = read_varint(&buf[i..]);
                    i += n;
                    out.push((field, wire, Vec::new()));
                }
                2 => {
                    let (len, n) = read_varint(&buf[i..]);
                    i += n;
                    out.push((field, wire, buf[i..i + len as usize].to_vec()));
                    i += len as usize;
                }
                _ => break,
            }
        }
        out
    }

    fn read_varint(buf: &[u8]) -> (u64, usize) {
        let mut v = 0u64;
        let mut s = 0;
        for (i, &b) in buf.iter().enumerate() {
            v |= ((b & 0x7f) as u64) << s;
            if b & 0x80 == 0 {
                return (v, i + 1);
            }
            s += 7;
        }
        (v, buf.len())
    }

    #[test]
    fn the_graph_is_a_single_matmul_with_the_weight_as_initializer() {
        let w = vec![0.5f32; 6];
        let g = matmul_graph(2, 3, &w);
        let model = fields(&g);
        // ModelProto.graph is field 7.
        let graph = &model.iter().find(|(f, _, _)| *f == 7).unwrap().2;
        let gf = fields(graph);
        let nodes: Vec<_> = gf.iter().filter(|(f, _, _)| *f == 1).collect();
        assert_eq!(nodes.len(), 1, "exactly one node");
        let node = fields(&nodes[0].2);
        let op = node.iter().find(|(f, _, _)| *f == 4).unwrap();
        assert_eq!(op.2, b"MatMul");
        let inputs: Vec<_> = node.iter().filter(|(f, _, _)| *f == 1).map(|(_, _, b)| b.clone()).collect();
        assert_eq!(inputs, vec![b"W".to_vec(), b"x".to_vec()]);
        // initializer is field 5, named W, F32, raw_data 6 floats.
        let init = &gf.iter().find(|(f, _, _)| *f == 5).unwrap().2;
        let it = fields(init);
        let name = it.iter().find(|(f, _, _)| *f == 8).unwrap();
        assert_eq!(name.2, b"W");
        let raw = &it.iter().find(|(f, _, _)| *f == 9).unwrap().2;
        assert_eq!(raw.len(), 6 * 4);
    }

    #[test]
    fn widening_f16_and_bf16_matches_the_kernel() {
        // 1.0 in f16 is 0x3c00; in bf16 is 0x3f80.
        assert_eq!(widen_half(&[0x00, 0x3c], true), vec![1.0f32]);
        assert_eq!(widen_half(&[0x80, 0x3f], false), vec![1.0f32]);
        // -2.0 f16 = 0xc000.
        assert_eq!(widen_half(&[0x00, 0xc0], true), vec![-2.0f32]);
    }

    fn dev(vendor_id: u32, vendor: &'static str, name: &str, portability: bool) -> Device {
        Device {
            index: 0,
            name: name.into(),
            vendor_id,
            device_id: 0,
            class: 2,
            api: String::new(),
            driver: String::new(),
            device_local_gib: 8.0,
            free_gib: 0.0,
            dedicated_gib: 8.0,
            host_visible_gib: 0.0,
            heaps: vec![],
            extensions: if portability { vec!["VK_KHR_portability_subset".into()] } else { vec![] },
            vendor,
            rank: 0,
        }
    }

    #[test]
    fn the_provider_follows_the_device() {
        // DirectML is a Windows API: a non-Apple GPU gets it only on Windows and
        // falls back to the CPU EP elsewhere. The AI Boost (OpenVINO) and Apple
        // (CoreML) picks are platform-independent — they precede the Windows
        // branch in for_device.
        let gpu_ep = if cfg!(windows) { OrtEp::DirectML } else { OrtEp::Cpu };
        assert_eq!(OrtEp::for_device(&dev(0x1002, "AMD", "Radeon", false)), gpu_ep);
        assert_eq!(OrtEp::for_device(&dev(0x8086, "Intel", "Arc", false)), gpu_ep);
        assert_eq!(OrtEp::for_device(&dev(0x10DE, "NVIDIA", "RTX 4060", false)), OrtEp::Cuda);
        assert_eq!(OrtEp::for_device(&dev(0x8086, "Intel", "Intel(R) AI Boost", false)), OrtEp::OpenVINO);
        assert_eq!(OrtEp::for_device(&dev(0x106B, "Apple", "M3", true)), OrtEp::CoreML);
        assert_eq!(OrtEp::for_device(&dev(0, "Apple", "M3", true)), OrtEp::CoreML);
    }
}