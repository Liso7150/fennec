//! safetensors reader: mmap the shards, walk the header, hand back f32 rows.
//!
//! Standard HF naming is the input format (71.9 GB of bf16 in 26 shards for this
//! model); no GGUF conversion and no hub download is needed to start.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::config::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    U8,
    I8,
    I32,
    I64,
    U32,
}

impl DType {
    pub fn parse(s: &str) -> Result<DType> {
        Ok(match s {
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            "F64" => DType::F64,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U32" => DType::U32,
            other => return Err(format!("unsupported dtype {other}")),
        })
    }

    pub fn size(self) -> usize {
        match self {
            DType::BF16 | DType::F16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::U8 | DType::I8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Info {
    pub file: usize,
    pub off: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl Info {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn bytes(&self) -> usize {
        self.elems() * self.dtype.size()
    }
    /// Rows x cols for a 2-D matrix; for a 3-D fused expert stack, (E * rows, cols)
    /// is not what a caller wants, so 3-D reports the leading dim as "tensors".
    pub fn as_matrix(&self) -> Result<(usize, usize)> {
        match self.shape.len() {
            2 => Ok((self.shape[0], self.shape[1])),
            _ => Err(format!("tensor is {}-D", self.shape.len())),
        }
    }
}

pub struct Shards {
    pub dir: PathBuf,
    infos: HashMap<String, Info>,
    order: Vec<String>,
    files: Vec<Option<Arc<Mmap>>>,
    pub total_bytes: u64,
    /// Already-quantised experts, when a matching FQ4 sidecar is attached.
    fq4: Option<crate::quant::Sidecar>,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    f16_to_f32_rust(bits)
}

#[inline]
fn f16_to_f32_rust(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // Subnormal: the value is man * 2^-24. Shifting man up until its
            // implicit leading 1 lands at bit 10 turns it into (1 + frac/1024)
            // times 2^(-14 - shifts), so the f32 exponent is 127 - 14 - shifts.
            let mut shifts = 0i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                shifts += 1;
            }
            (sign << 31) | ((127 - 14 - shifts) as u32) << 23 | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (man << 13)
    } else {
        (sign << 31) | (((exp + 127 - 15) as u32) << 23) | (man << 13)
    };
    f32::from_bits(out)
}

pub fn bits_to_f32(dtype: DType, raw: u64) -> f32 {
    match dtype {
        DType::BF16 => bf16_to_f32(raw as u16),
        DType::F16 => f16_to_f32(raw as u16),
        DType::F32 => f32::from_bits(raw as u32),
        DType::F64 => f64::from_bits(raw) as f32,
        DType::U8 => (raw & 0xff) as u8 as f32,
        DType::I8 => ((raw & 0xff) as u8) as i8 as f32,
        DType::I32 => (raw as u32) as i32 as f32,
        DType::I64 => (raw as u64) as i64 as f32,
        DType::U32 => (raw as u32) as f32,
    }
}

/// One element at a byte offset, not an element index: a tensor's offset inside
/// its shard is not guaranteed to be a multiple of its element size.
fn read_number(bytes: &[u8], dtype: DType, i: usize) -> f32 {
    let raw = match dtype.size() {
        1 => bytes[i] as u64,
        2 => u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u64,
        4 => u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as u64,
        _ => u64::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
            bytes[i + 4],
            bytes[i + 5],
            bytes[i + 6],
            bytes[i + 7],
        ]),
    };
    bits_to_f32(dtype, raw)
}

impl Shards {
    pub fn open(dir: impl AsRef<Path>) -> Result<Shards> {
        let dir = dir.as_ref().to_path_buf();
        let index = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = Vec::new();
        let mut map: HashMap<String, String> = HashMap::new();
        if index.is_file() {
            let raw: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&index).map_err(|e| e.to_string())?)
                    .map_err(|e| format!("index: {e}"))?;
            if let Some(o) = raw.get("weight_map").and_then(|v| v.as_object()) {
                for (k, v) in o {
                    map.insert(k.clone(), v.as_str().unwrap_or("").to_string());
                }
            }
        }
        let mut present: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let p = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                present.push(p);
            }
        }
        present.sort();
        for p in &present {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if map.values().any(|v| *v == name) || map.is_empty() {
                files.push(p.clone());
            }
        }
        if files.is_empty() {
            return Err(format!("no safetensors shards in {}", dir.display()));
        }
        let mut infos = HashMap::new();
        let mut order = Vec::new();
        let mut maps = Vec::new();
        let mut total = 0u64;
        for (idx, path) in files.iter().enumerate() {
            let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("{} mmap: {e}", path.display()))?;
            // A shard is a file on disk: a short or malformed header must be an
            // error naming the file, not a panic inside the loader.
            if mmap.len() < 8 {
                return Err(format!("{}: shorter than a safetensors header", path.display()));
            }
            let hl = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let end = 8usize.checked_add(hl).filter(|e| *e <= mmap.len())
                .ok_or_else(|| format!("{}: header length {hl} past the {}-byte file", path.display(), mmap.len()))?;
            let head: serde_json::Value =
                serde_json::from_slice(&mmap[8..end]).map_err(|e| format!("{}: {e}", path.display()))?;
            let obj = head.as_object().ok_or("bad safetensors header")?;
            let base = end;
            for (k, v) in obj {
                if k == "__metadata__" {
                    continue;
                }
                let off = v
                    .get("data_offsets")
                    .and_then(|d| d.as_array())
                    .filter(|a| a.len() == 2)
                    .ok_or_else(|| format!("{k}: no data_offsets"))?;
                let (o0, o1) = (
                    off[0].as_u64().ok_or_else(|| format!("{k}: bad data_offsets"))? as usize,
                    off[1].as_u64().ok_or_else(|| format!("{k}: bad data_offsets"))? as usize,
                );
                if o1 < o0 || base.checked_add(o1).map_or(true, |e| e > mmap.len()) {
                    return Err(format!("{}: {k} claims bytes {o0}..{o1} past the {}-byte file", path.display(), mmap.len()));
                }
                let dtype = DType::parse(v.get("dtype").and_then(|d| d.as_str()).unwrap_or(""))?;
                let dims = v
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| format!("{k}: no shape"))?;
                let mut shape = Vec::with_capacity(dims.len());
                let mut elems = 1usize;
                for dim in dims {
                    let dim = dim
                        .as_u64()
                        .and_then(|d| usize::try_from(d).ok())
                        .ok_or_else(|| format!("{k}: invalid shape dimension"))?;
                    elems = elems
                        .checked_mul(dim)
                        .ok_or_else(|| format!("{k}: shape element count overflows usize"))?;
                    shape.push(dim);
                }
                let bytes = elems
                    .checked_mul(dtype.size())
                    .ok_or_else(|| format!("{k}: tensor byte count overflows usize"))?;
                if bytes != o1 - o0 {
                    return Err(format!(
                        "{}: {k} shape {dtype:?} {:?} needs {bytes} bytes, but data_offsets span {}",
                        path.display(),
                        shape,
                        o1 - o0
                    ));
                }
                // a tensor only belongs to this dir if the index says so, or there is no index
                infos.insert(
                    k.clone(),
                    Info {
                        file: idx,
                        off: base + o0,
                        dtype,
                        shape,
                    },
                );
                order.push(k.clone());
                total += (o1 - o0) as u64;
            }
            maps.push(Some(Arc::new(mmap)));
        }
        order.sort();
        // The multimodal export nests the text tower under `model.language_model.`;
        // the text-only int4 export puts it at `model.`. Register the other
        // spelling as an alias so one loader reads both, without touching the
        // inventory that `order` and `total_bytes` report.
        let aliases: Vec<(String, Info)> = order
            .iter()
            .filter_map(|n| {
                let alt = if let Some(rest) = n.strip_prefix("model.language_model.") {
                    format!("model.{rest}")
                } else if let Some(rest) = n.strip_prefix("model.") {
                    format!("model.language_model.{rest}")
                } else {
                    return None;
                };
                infos.get(n).map(|i| (alt, i.clone()))
            })
            .collect();
        for (alt, info) in aliases {
            infos.entry(alt).or_insert(info);
        }
        Ok(Shards {
            dir,
            infos,
            order,
            files: maps,
            total_bytes: total,
            fq4: None,
        })
    }

    /// Attach an FQ4 expert cache. The sidecar records the checkpoint it was
    /// quantised from; attaching one built from a different model would silently
    /// serve the wrong weights, so a mismatch is an error rather than a fallback.
    pub fn attach_fq4(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let sc = crate::quant::Sidecar::open(path)?;
        sc.check_source(self.total_bytes, self.order.len())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.fq4 = Some(sc);
        Ok(())
    }

    pub fn fq4(&self) -> Option<&crate::quant::Sidecar> {
        self.fq4.as_ref()
    }

    pub fn data(&self, file: usize) -> &[u8] {
        self.files[file].as_ref().expect("mmap dropped")
    }

    /// The shard mapping behind a tensor plus the tensor's absolute byte offset
    /// in it. This is what lets a weight stay a clean file page instead of being
    /// copied into private memory the OS has to find room for.
    pub fn mapped(&self, name: &str) -> Option<(Arc<Mmap>, usize)> {
        let i = self.info(name)?;
        Some((Arc::clone(self.files[i.file].as_ref()?), i.off))
    }

    /// How many shard files are mmap'd.
    pub fn n_shards(&self) -> usize {
        self.files.len()
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn info(&self, name: &str) -> Option<&Info> {
        self.infos.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.infos.contains_key(name)
    }

    /// Copy one contiguous byte range out verbatim. Pre-quantised containers
    /// store nibbles, which must not go through the f32 conversion path.
    pub fn slice_bytes(&self, name: &str, start: usize, end: usize) -> Result<Vec<u8>> {
        let i = self.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if start > end || end > i.bytes() {
            return Err(format!("{name}: byte slice {start}..{end} past {}", i.bytes()));
        }
        Ok(self.data(i.file)[i.off + start..i.off + end].to_vec())
    }

    /// Copy one contiguous element range out as f32.
    ///
    /// The bf16 and f32 paths convert in word blocks instead of calling a
    /// `match` per element: the lm_head is 508M elements and read once per
    /// decoded token, so the per-element branch is the difference between a
    /// memory-bound token and a scalar-bound one.
    pub fn slice_f32(&self, name: &str, start: usize, end: usize) -> Result<Vec<f32>> {
        let i = self.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let bytes = self.data(i.file);
        if end > i.elems() {
            return Err(format!("{name}: slice {start}..{end} past {}", i.elems()));
        }
        if start > end {
            return Err(format!("{name}: slice {start}..{end} is backwards"));
        }
        let sz = i.dtype.size();
        let n = end - start;
        let mut out = Vec::with_capacity(n);
        match i.dtype {
            DType::BF16 => {
                for c in bytes[i.off + start * sz..i.off + end * sz].chunks_exact(2) {
                    let b = u16::from_le_bytes([c[0], c[1]]);
                    out.push(f32::from_bits((b as u32) << 16));
                }
            }
            DType::F32 => {
                for c in bytes[i.off + start * sz..i.off + end * sz].chunks_exact(4) {
                    out.push(f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                }
            }
            other => {
                for k in 0..n {
                    out.push(read_number(bytes, other, i.off + (start + k) * sz));
                }
            }
        }
        Ok(out)
    }

    /// Materialise a whole tensor as f32 (row-major, as stored).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let n = self.info(name).ok_or_else(|| format!("missing tensor {name}"))?.elems();
        self.slice_f32(name, 0, n)
    }

    /// Row `r` of a 2-D tensor (or of the `r`-th flattened row of a 3-D one).
    pub fn row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let i = self.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let cols = *i.shape.last().ok_or_else(|| format!("{name}: no last dim"))?;
        self.slice_f32(name, row * cols, (row + 1) * cols)
    }

    pub fn rows(&self, name: &str) -> Result<usize> {
        let i = self.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let cols = *i.shape.last().ok_or_else(|| format!("{name}: no last dim"))?;
        Ok(i.elems() / cols)
    }

    /// Flat index of one element inside a fused 3-D stack, e.g. experts[e, r, c].
    pub fn slice3_f32(&self, name: &str, idx: usize, start: usize, end: usize) -> Result<Vec<f32>> {
        let i = self.info(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if i.shape.len() != 3 {
            return Err(format!("{name}: not 3-D"));
        }
        let rc = i.shape[1] * i.shape[2];
        self.slice_f32(name, idx * rc + start, idx * rc + end)
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use std::path::Path;

    /// One shard holding (name, dtype, shape, payload) in the order given.
    pub fn shard(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut head = serde_json::Map::new();
        let mut off = 0usize;
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            head.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}),
            );
            off += bytes.len();
            payload.extend_from_slice(bytes);
        }
        let json = serde_json::Value::Object(head).to_string();
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(json.as_bytes());
        out.extend_from_slice(&payload);
        std::fs::write(path, out).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::shard;
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fennec-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn both_tensor_prefixes_name_the_same_weight() {
        let d = dir("alias");
        // The text-only export's spelling; the loader must also answer to the
        // multimodal wrapper's, because that is what the bf16 export writes.
        shard(&d.join("model-00000.safetensors"), &[
            ("model.layers.0.mlp.gate.weight", "F32", vec![2, 2], vec![0u8; 16]),
        ]);
        let sh = Shards::open(&d).unwrap();
        let short = sh.info("model.layers.0.mlp.gate.weight").expect("short name");
        let long = sh.info("model.language_model.layers.0.mlp.gate.weight").expect("aliased name");
        assert_eq!((short.file, short.off, &short.shape), (long.file, long.off, &long.shape));
        assert!(sh.contains("model.language_model.layers.0.mlp.gate.weight"));
        // The alias is a lookup convenience, not a second tensor: the inventory
        // the sidecar source check compares must still count each weight once.
        assert_eq!(sh.names().len(), 1);
        assert_eq!(sh.total_bytes, 16);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_multimodal_spelling_aliases_back_too() {
        let d = dir("alias2");
        shard(&d.join("model-00000.safetensors"), &[
            ("model.language_model.embed_tokens.weight", "F32", vec![4], vec![0u8; 16]),
        ]);
        let sh = Shards::open(&d).unwrap();
        assert!(sh.info("model.embed_tokens.weight").is_some());
        assert_eq!(sh.names().len(), 1);
        std::fs::remove_dir_all(&d).ok();
    }

    /// Truncating f32 -> f16 for values that are exactly representable, so the
    /// fixture and the assertion share one number.
    fn f16_bits(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
    }

    /// An offset read has to land on the element it was asked for. The whole-tensor
    /// read everything else uses starts at 0, where a dropped `start` is invisible;
    /// an embedding lookup is a row read, and getting row 0 for every token is
    /// fluent-looking nonsense rather than an error.
    #[test]
    fn an_offset_slice_reads_the_elements_it_asked_for() {
        let d = dir("offset");
        let (rows, cols) = (4usize, 4usize);
        let want: Vec<f32> = (0..rows * cols).map(|k| (k + 1) as f32).collect();
        let h: Vec<u8> = want.iter().flat_map(|v| f16_bits(*v).to_le_bytes()).collect();
        let i8: Vec<u8> = want.iter().map(|v| *v as u8).collect();
        // A one-byte tensor first, so the F16 payload starts at an odd byte
        // offset: an element index computed by dividing that offset would be
        // wrong even with `start` added back.
        shard(&d.join("model-00000.safetensors"), &[
            ("pad", "U8", vec![1], vec![0u8]),
            ("h", "F16", vec![rows, cols], h),
            ("b", "I8", vec![rows, cols], i8),
        ]);
        let sh = Shards::open(&d).unwrap();
        for name in ["h", "b"] {
            for r in 0..rows {
                assert_eq!(sh.row_f32(name, r).unwrap(), want[r * cols..(r + 1) * cols], "{name} row {r}");
            }
            assert_eq!(sh.slice_f32(name, 5, 11).unwrap(), want[5..11], "{name} mid range");
            assert_eq!(sh.tensor_f32(name).unwrap(), want, "{name} whole");
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_malformed_shard_is_an_error_not_a_panic() {
        let d = dir("bad");
        let p = d.join("model-00000.safetensors");

        std::fs::write(&p, b"short").unwrap();
        assert!(Shards::open(&d).is_err(), "a file shorter than the header");

        // A header length that runs past the end of the file.
        let mut out = 1_000_000u64.to_le_bytes().to_vec();
        out.extend_from_slice(b"{}");
        std::fs::write(&p, out).unwrap();
        assert!(Shards::open(&d).is_err(), "header length past EOF");

        // A tensor whose data_offsets point past the payload.
        let json = br#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,4096]}}"#;
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(json);
        out.extend_from_slice(&[0u8; 16]);
        std::fs::write(&p, out).unwrap();
        assert!(Shards::open(&d).is_err(), "payload past EOF");

        let cases: &[(&[u8], &str)] = &[
            (
                br#"{"a":{"dtype":"F32","shape":[5],"data_offsets":[0,16]}}"#,
                "needs 20 bytes",
            ),
            (
                br#"{"a":{"dtype":"F32","shape":["4"],"data_offsets":[0,16]}}"#,
                "invalid shape dimension",
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,16]}}"#,
                "overflows usize",
            ),
        ];
        for (json, expected) in cases {
            let mut out = (json.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(json);
            out.extend_from_slice(&[0u8; 16]);
            std::fs::write(&p, out).unwrap();
            let error = Shards::open(&d).err().expect("malformed shape must fail");
            assert!(error.contains(expected), "{error}");
        }

        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_byte_slice_copies_nibbles_verbatim() {
        let d = dir("bytes");
        shard(&d.join("model-00000.safetensors"), &[
            ("q", "U8", vec![6], vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab]),
        ]);
        let sh = Shards::open(&d).unwrap();
        assert_eq!(sh.slice_bytes("q", 1, 4).unwrap(), vec![0x23, 0x45, 0x67]);
        assert!(sh.slice_bytes("q", 0, 7).is_err());
        assert!(sh.slice_bytes("q", 4, 2).is_err());
        assert!(sh.slice_bytes("missing", 0, 1).is_err());
        std::fs::remove_dir_all(&d).ok();
    }

    /// Every f16 subnormal, against the definition man * 2^-24. The renormalising
    /// loop used to land the exponent eight too high, which made all 1023 of
    /// them 256x too large — small enough to look like noise in a logits diff.
    #[test]
    fn f16_subnormals_are_not_scaled_by_256() {
        for man in 1u16..1024 {
            let want = (man as f32) * 5.960464477539063e-8;
            for bits in [man, man | 0x8000] {
                let got = f16_to_f32(bits);
                let expect = if bits & 0x8000 != 0 { -want } else { want };
                assert_eq!(got, expect, "f16 {bits:#06x}");
            }
        }
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x0400), 6.103515625e-5);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
    }
}
