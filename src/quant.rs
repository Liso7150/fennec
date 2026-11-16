//! FQ4: grouped int4 weights, the format Fennec trains on.
//!
//! Layout is inherited from rabbit's `.qs` sidecars because it is the one thing
//! those two engines got right for CPU MoE: one f32 scale per `group` columns of
//! one row, low nibble = even column, value stored +8. Group 32 means the AVX2
//! kernel consumes exactly one scale per 32 weights, so no bookkeeping sits in the
//! inner loop.
//!
//! ```text
//! row r, group g:  w[r][g*32 .. g*32+32] = (nibble - 8) * scale[r * ngroups + g]
//! ```

use crate::config::Result;
use memmap2::Mmap;
use std::ops::Deref;
use std::sync::Arc;

/// Where a pack's nibbles live.
///
/// `Map` is the whole point of the host tier on a machine that cannot hold the
/// model twice. An owned 13 GiB expert pool is anonymous memory: Windows has to
/// find room for all of it, and when it cannot it trims the working set and
/// compresses the rest, so a "RAM-resident" pick costs a page fault. Measured on
/// a 31.5 GiB box: 25.3 GiB committed, 10.0 GiB resident, and the host-expert
/// stage swinging 48 -> 85 ms/token between runs on identical work. Mapped
/// nibbles are clean file pages instead: the OS drops them under pressure
/// without writing anything, and reads them back from NVMe.
#[derive(Debug, Clone)]
pub enum Store {
    Own(Vec<u8>),
    /// Mapping, byte offset, byte length.
    Map(Arc<Mmap>, usize, usize),
}

impl Store {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Store::Own(v) => v,
            Store::Map(m, off, len) => &m[*off..*off + *len],
        }
    }

    /// Allocator capacity, which for a mapping is just its length.
    pub fn held(&self) -> usize {
        match self {
            Store::Own(v) => v.capacity(),
            Store::Map(_, _, len) => *len,
        }
    }

    fn own_mut(&mut self) -> &mut Vec<u8> {
        match self {
            Store::Own(v) => v,
            Store::Map(..) => panic!("cannot grow a mapped pack"),
        }
    }
}

impl Deref for Store {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.bytes()
    }
}

impl PartialEq for Store {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}

impl PartialEq<[u8]> for Store {
    fn eq(&self, other: &[u8]) -> bool {
        self.bytes() == other
    }
}

/// Opt-in: prefer the planar-activation f32 dot over the int8 `maddubs` dot for
/// host experts. The f32 path showed ~1.3x in a microbenchmark; default off keeps
/// the int8 kernel that measured 92->38 ms/token. A/B with --planar-dot.
static PLANAR_DOT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_planar_dot(v: bool) { PLANAR_DOT.store(v, std::sync::atomic::Ordering::Relaxed); }
pub fn planar_dot() -> bool { PLANAR_DOT.load(std::sync::atomic::Ordering::Relaxed) }

#[derive(Debug, Clone)]
pub struct Q4 {
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub data: Store,
    pub scale: Vec<f32>,
    /// Nibbles are two's complement rather than offset by 8. The pre-quantised
    /// container stores them that way; reading it in place saves the copy and
    /// the XOR pass, and costs the kernel one vector XOR per 32 columns.
    pub signed: bool,
}

/// Int2 grouped weights: 4 values per byte, one f32 scale per `group` columns.
/// Layout: row r, group g: w[r][g*group .. g*group+group] = (2-bit - 2) * scale[r * ngroups + g]
/// 2-bit values are 0..3, offset by 2 to get -2..1 range.
#[derive(Debug, Clone)]
pub struct Q2 {
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub data: Store,
    pub scale: Vec<f32>,
}

pub const MAGIC: &[u8; 8] = b"FNN1Q4\0\0";
pub const MAGIC_Q2: &[u8; 8] = b"FNN1Q2\0\0";

/// round-to-nearest, ties-to-even: what torch/numpy `round` does, so a Rust
/// quantizer and a Python reference produce the same nibbles.
#[inline]
pub fn round_half_even(v: f32) -> f32 {
    let f = v.floor();
    let d = v - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f * 0.5).fract() == 0.0 {
        f
    } else {
        f + 1.0
    }
}

/// One row block of the packer: `src` holds `rows` consecutive rows of width
/// `cols`, `data` and `scale` are the matching output slices.
fn quant_rows(src: &[f32], rows: usize, cols: usize, group: usize, data: &mut [u8], scale: &mut [f32]) {
    const QMAX: f32 = 7.0;
    let ng = cols.div_ceil(group);
    let half = cols / 2;
    for r in 0..rows {
        let src = &src[r * cols..(r + 1) * cols];
        for g in 0..ng {
            let c0 = g * group;
            let c1 = (c0 + group).min(cols);
            let mut amax = 0f32;
            for &v in &src[c0..c1] {
                amax = amax.max(v.abs());
            }
            let sc = (amax / QMAX).max(1e-8);
            scale[r * ng + g] = sc;
            for (i, &v) in src[c0..c1].iter().enumerate() {
                // divide, do not multiply by a precomputed reciprocal: `v * (1/sc)`
                // is one rounding away from `v / sc`, and that ulp is enough to
                // flip a level at an exact .5 tie, which is the whole point of
                // agreeing with the reference quantiser bit for bit.
                let q = round_half_even((v / sc).clamp(-8.0, QMAX)) + 8.0;
                let nib = q as u8;
                let pos = r * half + (c0 + i) / 2;
                if (c0 + i) & 1 == 0 {
                    data[pos] = (data[pos] & 0xf0) | nib;
                } else {
                    data[pos] = (data[pos] & 0x0f) | (nib << 4);
                }
            }
        }
    }
}

/// Int2 quantization: 4 values per byte, 2-bit unsigned (0..3), offset by 2 for signed range (-2..1).
fn quant_rows_q2(src: &[f32], rows: usize, cols: usize, group: usize, data: &mut [u8], scale: &mut [f32]) {
    const QMAX: f32 = 1.0;  // 2-bit max value after offset
    let ng = cols.div_ceil(group);
    let quarter = cols / 4;  // 4 values per byte
    for r in 0..rows {
        let src = &src[r * cols..(r + 1) * cols];
        for g in 0..ng {
            let c0 = g * group;
            let c1 = (c0 + group).min(cols);
            let mut amax = 0f32;
            for &v in &src[c0..c1] {
                amax = amax.max(v.abs());
            }
            let sc = (amax / QMAX).max(1e-8);
            scale[r * ng + g] = sc;
            for (i, &v) in src[c0..c1].iter().enumerate() {
                // Quantize to 0..3 range (2-bit unsigned)
                let q = round_half_even((v / sc).clamp(-2.0, QMAX)) + 2.0;
                let bits = (q as u8) & 0x03;
                let byte_pos = r * quarter + (c0 + i) / 4;
                let bit_offset = ((c0 + i) % 4) * 2;
                data[byte_pos] = (data[byte_pos] & !(0x03 << bit_offset)) | (bits << bit_offset);
            }
        }
    }
}

/// An activation permuted into the order the pairs-layout kernel reads, plus
/// its -8x bias. Built once per token and reused across every row of a matrix.
///
/// The pack stores element 2j in byte j's low nibble and 2j+1 in its high one,
/// so widening a byte yields the even and odd elements in separate vectors.
/// Untangling them back into column order costs eight shuffles per 32 weights,
/// all on the one port that issues them. Permuting x instead pays that once per
/// token rather than once per row, and the inner loop keeps only the two
/// widening shifts it cannot avoid.
pub struct DotPrep {
    x: Vec<f32>,
    xp: Vec<f32>,
    nx: Vec<f32>,
    q: Option<I8Prep>,
}

/// The same activations quantised to int8, in the column order the nibble split
/// produces, plus the per-lane offset correction.
///
/// A 32-byte weight load covers 64 columns: its low nibbles are the even ones and
/// its high nibbles the odd ones. `xe`/`xo` hold the activations in those two
/// orders, so one `maddubs` pair consumes 64 weights without a shuffle. Lane i of
/// the resulting integer dot covers columns 8i..8i+7, which is what `corr` is
/// summed over.
///
/// Built only for the pre-quantised container. Its 4-bit weights already carry
/// more error than an int8 activation adds, and the golden band for that export
/// is 2.5e-1 against this kernel's ~4e-3. An FQ4 copy quantised from a bf16
/// export is checked against a bit-exact FQ4 reference at 1e-6, so it keeps the
/// f32 dot.
struct I8Prep {
    xe: Vec<i8>,
    xo: Vec<i8>,
    corr: Vec<i32>,
    /// amax/127 per block: the integer dot is against `x * 127/amax`, so it has
    /// to be scaled back before the weight scale is applied.
    d: Vec<f32>,
}

impl I8Prep {
    fn empty(cols: usize) -> I8Prep {
        let blocks = cols / 64;
        I8Prep {
            xe: vec![0i8; blocks * 32],
            xo: vec![0i8; blocks * 32],
            corr: vec![0i32; blocks * 8],
            d: vec![0f32; blocks],
        }
    }

    fn fill(&mut self, x: &[f32]) {
        for (b, chunk) in x.chunks_exact(64).enumerate() {
            let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
            // 127 rather than 128: the grid stays symmetric, so the offset
            // correction below is exact rather than off by one at the extreme.
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            self.d[b] = amax / 127.0;
            let mut q = [0i8; 64];
            for c in 0..64 {
                q[c] = round_half_even(chunk[c] * inv).clamp(-127.0, 127.0) as i8;
            }
            for j in 0..32 {
                self.xe[b * 32 + j] = q[2 * j];
                self.xo[b * 32 + j] = q[2 * j + 1];
            }
            for i in 0..8 {
                let s: i32 = q[i * 8..i * 8 + 8].iter().map(|&v| v as i32).sum();
                // The nibbles are offset-binary, so each lane owes 8 * its octet.
                self.corr[b * 8 + i] = 8 * s;
            }
        }
    }
}

impl DotPrep {
    pub fn new(cols: usize) -> DotPrep {
        DotPrep::with_int(cols, false)
    }

    /// A prep that also builds the int8 form on every `fill`, for packs whose
    /// weights are already 4-bit.
    pub fn with_int(cols: usize, int: bool) -> DotPrep {
        let usable = int && cols % 64 == 0;
        DotPrep {
            x: vec![0f32; cols],
            xp: vec![0f32; cols],
            nx: vec![0f32; cols],
            q: if usable { Some(I8Prep::empty(cols)) } else { None },
        }
    }

    pub fn fill(&mut self, x: &[f32]) {
        let cols = x.len();
        assert_eq!(cols, self.x.len(), "prep was sized for a different width");
        self.x.copy_from_slice(x);
        let full = cols & !31;
        for c in (0..full).step_by(32) {
            for j in 0..8 {
                self.xp[c + j] = x[c + 2 * j];
                self.xp[c + 8 + j] = x[c + 2 * j + 1];
                self.xp[c + 16 + j] = x[c + 16 + 2 * j];
                self.xp[c + 24 + j] = x[c + 17 + 2 * j];
            }
        }
        self.xp[full..].copy_from_slice(&x[full..]);
        for (n, &v) in self.nx.iter_mut().zip(&self.xp) {
            *n = -8.0 * v;
        }
        if let Some(q) = &mut self.q {
            q.fill(x);
        }
    }
}

impl Q4 {
    /// A prep sized for this pack's rows, filled with `x`.
    pub fn prepare(&self, x: &[f32]) -> DotPrep {
        let mut p = DotPrep::with_int(self.cols, self.signed && self.group % 64 == 0);
        p.fill(x);
        p
    }

    pub fn ngroups(&self) -> usize {
        self.cols.div_ceil(self.group)
    }

    /// Whether the integer dot applies: the container's own 4-bit pack, with a
    /// group and width the 64-column kernel walks exactly.
    pub fn wants_int(&self) -> bool {
        self.signed && self.group % 64 == 0 && self.cols % 64 == 0
    }

    pub fn bytes_data(&self) -> usize {
        self.rows * self.cols.div_ceil(2)
    }

    pub fn bytes_scale(&self) -> usize {
        self.rows * self.ngroups() * 4
    }

    /// Every byte of the sidecar per matrix element, including the nibble header.
    pub fn bytes_total(&self) -> usize {
        self.bytes_data() + self.bytes_scale()
    }

    /// What the allocator holds rather than what the matrix uses. `push` grows by
    /// doubling, so a layer's resident set can commit well above its own size.
    pub fn bytes_held(&self) -> usize {
        self.data.held() + self.scale.capacity() * 4
    }

    /// True when the nibbles are borrowed from a mapping rather than owned.
    pub fn is_mapped(&self) -> bool {
        matches!(self.data, Store::Map(..))
    }

    /// Grouped int4 over real weights. Rows are split across threads, so every
    /// worker owns a disjoint byte range and the result is bit-identical to the
    /// serial version.
    pub fn quantize(src: &[f32], rows: usize, cols: usize, group: usize, threads: usize) -> Q4 {
        assert_eq!(src.len(), rows * cols);
        assert_eq!(cols % 2, 0, "int4 packing needs an even column count");
        let ng = cols.div_ceil(group);
        let half = cols / 2;
        let mut data = vec![0u8; rows * half];
        let mut scale = vec![0f32; rows * ng];
        let nthreads = threads.max(1).min(rows);
        let per = rows.div_ceil(nthreads);
        std::thread::scope(|s| {
            let mut drest = data.as_mut_slice();
            let mut srest = scale.as_mut_slice();
            let mut r0 = 0usize;
            while r0 < rows {
                let r1 = (r0 + per).min(rows);
                let (dlo, dhi) = drest.split_at_mut((r1 - r0) * half);
                let (slo, shi) = srest.split_at_mut((r1 - r0) * ng);
                drest = dhi;
                srest = shi;
                let block = &src[r0 * cols..r1 * cols];
                s.spawn(move || quant_rows(block, r1 - r0, cols, group, dlo, slo));
                r0 = r1;
            }
        });
        Q4 {
            rows,
            cols,
            group,
            data: Store::Own(data),
            scale,
            signed: false,
        }
    }

    /// A pack with no rows, ready to be filled by `push`. Used when the rows
    /// that belong in a matrix are not contiguous, e.g. one layer's resident
    /// experts: they are ranked by usage, so they arrive as scattered blocks.
    pub fn empty(cols: usize, group: usize) -> Q4 {
        Q4 {
            rows: 0,
            cols,
            group,
            data: Store::Own(Vec::new()),
            scale: Vec::new(),
            signed: false,
        }
    }

    /// A pack sized for exactly `rows` rows, so `push` never reallocates.
    ///
    /// Doubling growth is not free at this scale: a 40-layer resident set built
    /// with `empty` committed 16.88 GiB of allocator capacity for 13.08 GiB of
    /// matrix, and the 3.8 GiB difference is what pushed the process past
    /// physical RAM and turned every host pick into a page fault.
    pub fn with_capacity(rows: usize, cols: usize, group: usize) -> Q4 {
        Q4 {
            rows: 0,
            cols,
            group,
            data: Store::Own(Vec::with_capacity(rows * cols.div_ceil(2))),
            scale: Vec::with_capacity(rows * cols.div_ceil(group)),
            signed: false,
        }
    }

    /// A contiguous row block as its own pack. The device tier uploads one
    /// expert at a time out of a gathered matrix, and a slot is a row block.
    pub fn slice_rows(&self, r0: usize, r1: usize) -> Q4 {
        let ng = self.ngroups();
        let half = self.cols / 2;
        Q4 {
            rows: r1 - r0,
            cols: self.cols,
            group: self.group,
            data: Store::Own(self.data[r0 * half..r1 * half].to_vec()),
            scale: self.scale[r0 * ng..r1 * ng].to_vec(),
            signed: self.signed,
        }
    }

    /// Concatenate another pack of the same width. `rows` is the only bookkeeping
    /// that has to move, so a gathered matrix costs one memcpy per block.
    pub fn push(&mut self, other: &Q4) {
        assert_eq!(other.cols, self.cols, "row blocks must be the same width");
        assert_eq!(other.group, self.group, "row blocks must share a group size");
        assert_eq!(other.signed, other.signed, "row blocks must share a nibble encoding");
        self.data.own_mut().extend_from_slice(&other.data);
        self.scale.extend_from_slice(&other.scale);
        self.rows += other.rows;
    }

    /// Shrink allocator capacity to match the actual size. Call after loading
    /// to release memory wasted by Vec's doubling growth strategy.
    pub fn shrink_to_fit(&mut self) {
        self.data.own_mut().shrink_to_fit();
        self.scale.shrink_to_fit();
    }

    #[inline]
    fn nib(&self, r: usize, c: usize) -> f32 {
        let byte = self.data[r * (self.cols / 2) + c / 2];
        let n = if c & 1 == 0 { byte & 0x0f } else { byte >> 4 };
        let n = if self.signed { n ^ 8 } else { n };
        n as f32 - 8.0
    }

    pub fn dequant_row(&self, r: usize) -> Vec<f32> {
        let ng = self.ngroups();
        let mut out = vec![0f32; self.cols];
        for g in 0..ng {
            let sc = self.scale[r * ng + g];
            let c1 = ((g + 1) * self.group).min(self.cols);
            for c in g * self.group..c1 {
                out[c] = self.nib(r, c) * sc;
            }
        }
        out
    }

    /// Convert this Q4 matrix to Q2 by dequantizing to f32 then requantizing.
    /// Used when --host-int2 is set to reduce RAM at the cost of precision.
    /// Streams row-by-row to avoid allocating a temporary f32 matrix.
    pub fn to_q2(&self, _threads: usize) -> Q2 {
        let mut q2 = Q2::with_capacity(self.rows, self.cols, self.group);
        
        // Process sequentially row by row to avoid large temporary allocation.
        // The Q2 push() method handles quantization internally.
        for r in 0..self.rows {
            let row_f32 = self.dequant_row(r);
            q2.push(&row_f32);
        }
        q2
    }

    /// y = dot(row r, x)
    pub fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        self.dot_prepped(r, &self.prepare(x))
    }

    /// The same dot against an activation prepared once for a whole matrix.
    /// `dot_row` is this with a throwaway prep, which is the right trade for one
    /// row and the wrong one for the two thousand a host expert reads.
    pub fn dot_prepped(&self, r: usize, p: &DotPrep) -> f32 {
        debug_assert_eq!(p.x.len(), self.cols);
        #[cfg(target_arch = "x86_64")]
        if let Some(ip) = &p.q {
            if !planar_dot()
                && self.wants_int()
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
            {
                return unsafe { avx2::dot_prepped_int(self, r, ip) };
            }
        }
        // The kernel walks 32 columns at a time, so any group that is a whole
        // number of those chunks vectorises. The int4 export ships group 64;
        // gating on == 32 dropped it to the scalar nibble loop, which measured
        // 0.82 GB/s against the vector path's 2.56 GB/s on the same weights.
        #[cfg(target_arch = "x86_64")]
        if self.group % 32 == 0
            && self.cols % 32 == 0
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe { avx2::dot_prepped(self, r, p) };
        }
        let ng = self.ngroups();
        let mut y = 0f32;
        for g in 0..ng {
            let sc = self.scale[r * ng + g];
            let c1 = ((g + 1) * self.group).min(self.cols);
            let mut acc = 0f32;
            for c in g * self.group..c1 {
                acc += p.x[c] * self.nib(r, c);
            }
            y += acc * sc;
        }
        y
    }

    /// The error the int4 round trip actually introduces on real weights.
    pub fn rel_error(&self, src: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for r in 0..self.rows {
            let d = self.dequant_row(r);
            for c in 0..self.cols {
                let a = src[r * self.cols + c] as f64;
                let e = d[c] as f64 - a;
                num += e * e;
                den += a * a;
            }
        }
        (num / den.max(1e-30)).sqrt() as f32
    }

    pub fn to_bytes(&self, name: &str) -> Vec<u8> {
        let mut payload: Vec<u8> = Vec::with_capacity(self.bytes_total());
        payload.extend_from_slice(&self.data);
        let scale_off = payload.len();
        for b in &self.scale {
            payload.extend_from_slice(&b.to_le_bytes());
        }
        let header = serde_json::to_vec(&serde_json::json!({
            "items": { name: {
                "shape": [self.rows, self.cols],
                "group": self.group,
                "data": [0usize, self.data.len()],
                "scale": [scale_off, scale_off + self.scale.len() * 4],
            } }
        }))
        .expect("header is plain integers");
        let mut file = Vec::with_capacity(16 + header.len() + payload.len());
        file.extend_from_slice(MAGIC);
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        file
    }

    pub fn write(&self, name: &str, path: &str) -> Result<()> {
        let file = self.to_bytes(name);
        std::fs::write(path, &file).map_err(|e| format!("{path}: {e}"))?;
        Ok(())
    }

    /// Read one matrix out of an FNN1 sidecar (or a directory of them) by name.
    pub fn read(path: &str, name: &str) -> Result<Q4> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Q4::from_bytes(&bytes, name)
    }

    pub fn from_bytes(bytes: &[u8], name: &str) -> Result<Q4> {
        if &bytes[..8.min(bytes.len())] != &MAGIC[..] {
            return Err("not an FNN1Q4 sidecar".into());
        }
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let head: serde_json::Value = serde_json::from_slice(&bytes[16..16 + hl])
            .map_err(|e| format!("sidecar header: {e}"))?;
        let it = head
            .get("items")
            .and_then(|i| i.get(name))
            .ok_or_else(|| format!("sidecar has no {name}"))?;
        let shape = it["shape"].as_array().unwrap();
        let (rows, cols) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
        let group = it["group"].as_u64().unwrap() as usize;
        let base = 16 + hl;
        let (d0, d1) = arr2(&it["data"]);
        let (s0, s1) = arr2(&it["scale"]);
        let data = bytes[base + d0..base + d1].to_vec();
        let mut scale = Vec::with_capacity((s1 - s0) / 4);
        for c in (base + s0..base + s1).step_by(4) {
            scale.push(f32::from_le_bytes(bytes[c..c + 4].try_into().unwrap()));
        }
        Ok(Q4 {
            rows,
            cols,
            group,
            data: Store::Own(data),
            scale,
            signed: false,
        })
    }

    /// Adopt a rabbit `.qs` pair: the tensor itself is U8 nibbles, the sibling
    /// `{name}.qs` is the f32 scale row-major. Same packing, so the kernels work
    /// unchanged.
    pub fn from_qs(packed: &[u8], qs: &[f32], rows: usize, cols: usize, group: usize) -> Q4 {
        Q4 {
            rows,
            cols,
            group,
            data: Store::Own(packed.to_vec()),
            scale: qs.to_vec(),
            signed: false,
        }
    }

    /// One matrix borrowed straight out of a shard mapping: `w_off`/`w_len`
    /// locate the nibbles in the file, `qs` are its scales, already read out.
    ///
    /// Only the scales are copied. They are a ninth of the pack, and keeping them
    /// owned means the kernel's `scale.as_ptr()` stays a plain aligned f32 load
    /// regardless of where the tensor happens to sit inside its shard.
    pub fn mapped(
        map: Arc<Mmap>,
        w_off: usize,
        rows: usize,
        cols: usize,
        group: usize,
        qs: Vec<f32>,
        signed: bool,
    ) -> Q4 {
        let len = rows * cols.div_ceil(2);
        assert!(w_off + len <= map.len(), "mapped pack runs past its shard");
        assert_eq!(qs.len(), rows * cols.div_ceil(group));
        Q4 {
            rows,
            cols,
            group,
            data: Store::Map(map, w_off, len),
            scale: qs,
            signed,
        }
    }
}

impl Q2 {
    pub fn ngroups(&self) -> usize {
        self.cols.div_ceil(self.group)
    }

    pub fn bytes_data(&self) -> usize {
        self.rows * self.cols.div_ceil(4)  // 4 values per byte
    }

    pub fn bytes_scale(&self) -> usize {
        self.rows * self.ngroups() * 4
    }

    pub fn bytes_total(&self) -> usize {
        self.bytes_data() + self.bytes_scale()
    }

    pub fn bytes_held(&self) -> usize {
        self.data.held() + self.scale.capacity() * 4
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.data, Store::Map(..))
    }

    /// Grouped int2 quantization over real weights.
    pub fn quantize(src: &[f32], rows: usize, cols: usize, group: usize, threads: usize) -> Q2 {
        assert_eq!(src.len(), rows * cols);
        assert_eq!(cols % 4, 0, "int2 packing needs a column count divisible by 4");
        let ng = cols.div_ceil(group);
        let quarter = cols / 4;
        let mut data = vec![0u8; rows * quarter];
        let mut scale = vec![0f32; rows * ng];
        let nthreads = threads.max(1).min(rows);
        let per = rows.div_ceil(nthreads);
        std::thread::scope(|s| {
            let mut drest = data.as_mut_slice();
            let mut srest = scale.as_mut_slice();
            let mut r0 = 0usize;
            while r0 < rows {
                let r1 = (r0 + per).min(rows);
                let (dlo, dhi) = drest.split_at_mut((r1 - r0) * quarter);
                let (slo, shi) = srest.split_at_mut((r1 - r0) * ng);
                drest = dhi;
                srest = shi;
                let block = &src[r0 * cols..r1 * cols];
                s.spawn(move || quant_rows_q2(block, r1 - r0, cols, group, dlo, slo));
                r0 = r1;
            }
        });
        Q2 {
            rows,
            cols,
            group,
            data: Store::Own(data),
            scale,
        }
    }

    pub fn empty(cols: usize, group: usize) -> Q2 {
        Q2 {
            rows: 0,
            cols,
            group,
            data: Store::Own(Vec::new()),
            scale: Vec::new(),
        }
    }

    /// Shrink allocator capacity to match the actual size. Call after loading
    /// to release memory wasted by Vec's doubling growth strategy.
    pub fn shrink_to_fit(&mut self) {
        self.data.own_mut().shrink_to_fit();
        self.scale.shrink_to_fit();
    }

    pub fn with_capacity(rows: usize, cols: usize, group: usize) -> Q2 {
        let ng = cols.div_ceil(group);
        let quarter = cols / 4;
        Q2 {
            rows: 0,
            cols,
            group,
            data: Store::Own(Vec::with_capacity(rows * quarter)),
            scale: Vec::with_capacity(rows * ng),
        }
    }

    pub fn push(&mut self, src: &[f32]) {
        let ng = self.ngroups();
        let quarter = self.cols / 4;
        let d = self.data.own_mut();
        let old_len = d.len();
        d.resize(old_len + quarter, 0);
        self.scale.resize(self.rows * ng + ng, 0.0);
        quant_rows_q2(src, 1, self.cols, self.group, &mut d[old_len..], &mut self.scale[self.rows * ng..]);
        self.rows += 1;
    }

    /// Extract the 2-bit value at row r, column c (0..3 range, offset by 2 for signed -2..1).
    #[inline]
    fn bits(&self, r: usize, c: usize) -> f32 {
        let quarter = self.cols / 4;
        let byte_pos = r * quarter + c / 4;
        let bit_offset = (c % 4) * 2;
        let byte = self.data.bytes()[byte_pos];
        let val = ((byte >> bit_offset) & 0x03) as f32;
        val - 2.0  // Convert from 0..3 to -2..1
    }

    pub fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        let ng = self.ngroups();
        let mut y = 0f32;
        for g in 0..ng {
            let sc = self.scale[r * ng + g];
            let c1 = ((g + 1) * self.group).min(self.cols);
            let mut acc = 0f32;
            for c in g * self.group..c1 {
                acc += x[c] * self.bits(r, c);
            }
            y += acc * sc;
        }
        y
    }

    pub fn rel_error(&self, src: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for r in 0..self.rows {
            for c in 0..self.cols {
                let orig = src[r * self.cols + c] as f64;
                let ng = self.ngroups();
                let g = c / self.group;
                let sc = self.scale[r * ng + g] as f64;
                let recon = self.bits(r, c) as f64 * sc;
                let diff = orig - recon;
                num += diff * diff;
                den += orig * orig;
            }
        }
        (num / den.max(1e-30)).sqrt() as f32
    }
}

fn arr2(v: &serde_json::Value) -> (usize, usize) {
    let a = v.as_array().unwrap();
    (a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize)
}

/// A sidecar is a file on disk, so its header is untrusted: a malformed pair
/// must be a `None`, not a panic inside the loader.
fn pair(v: &serde_json::Value) -> Option<(usize, usize)> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    Some((a[0].as_u64()? as usize, a[1].as_u64()? as usize))
}

/// One matrix inside a sidecar: its shape, and where its two payloads sit
/// relative to the end of the header.
#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    data: (usize, usize),
    scale: (usize, usize),
}

/// An FNN1 sidecar held as a memory map.
///
/// `Q4::read` slurps the whole file to return one matrix. That is fine for a
/// single tensor and ruinous for a 20 GiB expert cache, so this copies out only
/// the rows asked for: a cold start then reads int4 instead of re-quantising
/// bf16, and the resident set costs a memcpy per expert rather than a
/// read-convert-pack pass over the source shard.
pub struct Sidecar {
    map: Mmap,
    base: usize,
    items: serde_json::Value,
}

impl Sidecar {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Sidecar> {
        let path = path.as_ref();
        let show = || path.display().to_string();
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", show()))?;
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("{} mmap: {e}", show()))?;
        if map.len() < 16 || &map[..8] != &MAGIC[..] {
            return Err(format!("{}: not an FNN1Q4 sidecar", show()));
        }
        let hl = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
        let end = 16usize.checked_add(hl).ok_or_else(|| format!("{}: header length overflow", show()))?;
        if end > map.len() {
            return Err(format!("{}: header runs past the file", show()));
        }
        let items: serde_json::Value =
            serde_json::from_slice(&map[16..end]).map_err(|e| format!("{} header: {e}", show()))?;
        if items.get("items").and_then(|i| i.as_object()).is_none() {
            return Err(format!("{}: header has no items object", show()));
        }
        Ok(Sidecar { map, base: end, items })
    }

    pub fn item(&self, name: &str) -> Option<Item> {
        let it = self.items.get("items")?.get(name)?;
        let shape = it["shape"].as_array()?;
        if shape.len() != 2 {
            return None;
        }
        let rows = shape[0].as_u64()? as usize;
        let cols = shape[1].as_u64()? as usize;
        let group = it["group"].as_u64()? as usize;
        if group == 0 {
            return None;
        }
        Some(Item { rows, cols, group, data: pair(&it["data"])?, scale: pair(&it["scale"])? })
    }

    pub fn names(&self) -> Vec<String> {
        self.items["items"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default()
    }

    /// Verify the sidecar was quantised from this checkpoint. The fingerprint is
    /// the shard byte total and tensor count, which is what `quant` records.
    pub fn check_source(&self, total_bytes: u64, tensors: usize) -> Result<()> {
        let src = self
            .items
            .get("source")
            .ok_or("sidecar records no source checkpoint; rebuild it with `fennec quant`")?;
        let b = src["total_bytes"].as_u64().ok_or("sidecar source has no total_bytes")?;
        let n = src["tensors"].as_u64().ok_or("sidecar source has no tensors")?;
        if b != total_bytes || n as usize != tensors {
            return Err(format!(
                "sidecar was built from a different checkpoint ({b} bytes / {n} tensors, \
                 this one is {total_bytes} / {tensors})"
            ));
        }
        Ok(())
    }

    /// Rows `[r0, r1)` of one matrix, copied out of the map.
    pub fn rows(&self, it: &Item, r0: usize, r1: usize) -> Result<Q4> {
        if r0 > r1 || r1 > it.rows {
            return Err(format!("row range {r0}..{r1} outside {} rows", it.rows));
        }
        let half = it.cols.div_ceil(2);
        let ng = it.cols.div_ceil(it.group);
        let d = self.payload(it.data)?;
        let s = self.payload(it.scale)?;
        if d.len() != it.rows * half || s.len() != it.rows * ng * 4 {
            return Err("sidecar payload does not match its declared shape".into());
        }
        let mut scale = Vec::with_capacity((r1 - r0) * ng);
        for c in (r0 * ng * 4..r1 * ng * 4).step_by(4) {
            scale.push(f32::from_le_bytes(s[c..c + 4].try_into().unwrap()));
        }
        Ok(Q4 {
            rows: r1 - r0,
            cols: it.cols,
            group: it.group,
            data: Store::Own(d[r0 * half..r1 * half].to_vec()),
            scale,
            signed: false,
        })
    }

    fn payload(&self, span: (usize, usize)) -> Result<&[u8]> {
        let (a, b) = span;
        let lo = self.base.checked_add(a).ok_or("sidecar offset overflow")?;
        let hi = self.base.checked_add(b).ok_or("sidecar offset overflow")?;
        if a > b || hi > self.map.len() {
            return Err("sidecar payload runs past the file".into());
        }
        Ok(&self.map[lo..hi])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_hits_the_int4_noise_bound() {
        // Symmetric int4 with scale = amax/7 puts one quantisation step at amax/7,
        // so uniform data in [-amax, amax] has RMS noise (amax/7)/sqrt(12) against a
        // signal RMS of amax/sqrt(3): exactly 1/14 = 7.1% relative error. Anything
        // much above that means the packer, not the format, is wrong.
        let src: Vec<f32> = (0..256 * 64).map(|i| ((i % 97) as f32 - 48.0) * 0.013).collect();
        let q = Q4::quantize(&src, 256, 64, 32, 4);
        let e = q.rel_error(&src);
        assert!(e < 0.075, "relative error {e}");
        assert!(e > 0.03, "relative error {e}: int4 cannot be this exact");
    }

    #[test]
    fn threads_do_not_change_the_bytes() {
        let src: Vec<f32> = (0..128 * 32).map(|i| (i as f32).sin() * 0.7).collect();
        let a = Q4::quantize(&src, 128, 32, 32, 1);
        let b = Q4::quantize(&src, 128, 32, 32, 8);
        assert_eq!(a.data, b.data);
        assert_eq!(a.scale, b.scale);
    }

    #[test]
    fn sidecar_round_trip() {
        let src: Vec<f32> = (0..64 * 32).map(|i| (i as f32).cos() * 0.4).collect();
        let q = Q4::quantize(&src, 64, 32, 32, 2);
        let mut buf = Vec::new();
        buf.extend_from_slice(&q.to_bytes("w"));
        let r = Q4::from_bytes(&buf, "w").unwrap();
        assert_eq!(r.data, q.data);
        assert_eq!(r.scale, q.scale);
    }

    #[test]
    fn rabbit_qs_pair_loads_unchanged() {
        // rabbit packs the low nibble first with a +8 bias and one f32 scale per
        // group: feeding that pair in must reproduce the same dot products.
        let src: Vec<f32> = (0..4 * 32).map(|i| (i as f32 - 64.0) * 0.02).collect();
        let q = Q4::quantize(&src, 4, 32, 32, 1);
        let r = Q4::from_qs(&q.data, &q.scale, 4, 32, 32);
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        for row in 0..4 {
            assert_eq!(q.dot_row(row, &x), r.dot_row(row, &x));
        }
    }

    #[test]
    fn every_group_width_vectorises_to_the_same_dot_product() {
        // The shipped int4 export uses group 64, the FQ4 cache group 32. The
        // kernel walks 32 columns at a time, so any whole number of chunks per
        // group must agree with dequant_row, which is group-generic scalar code
        // and therefore an independent oracle. cols=96 leaves a short final
        // group at 64 and 128, which is what the chunk bound is for.
        for &(rows, cols) in &[(4usize, 96usize), (3, 256), (2, 512)] {
            let src: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.017)
                .collect();
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.31).sin() * 0.9).collect();
            for group in [32usize, 64, 128] {
                let q = Q4::quantize(&src, rows, cols, group, 2);
                for r in 0..rows {
                    let w = q.dequant_row(r);
                    let want: f64 =
                        w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
                    let got = q.dot_row(r, &x) as f64;
                    let err = (got - want).abs();
                    assert!(
                        err <= 2e-5 * want.abs() + 1e-5,
                        "rows={rows} cols={cols} group={group} row={r}: got {got}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_borrowed_pack_dots_exactly_like_the_owned_one() {
        // The host tier reads the checkpoint's two's-complement nibbles in place
        // instead of copying 12 GiB into private memory. `byte ^ 0x88` is the whole
        // difference between the two encodings, so the borrowed pack must reproduce
        // the owned pack bit for bit - on the scalar path and on the AVX2 path, where
        // the flip is one xor per vector rather than a pass over the matrix.
        for &(rows, cols, group) in &[(4usize, 40usize, 32usize), (3, 256, 64), (2, 512, 32)] {
            let src: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.017)
                .collect();
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.31).sin() * 0.9).collect();
            let owned = Q4::quantize(&src, rows, cols, group, 2);

            // A non-zero offset, because a real tensor never starts at byte 0.
            let mut file = vec![0xa5u8; 8];
            file.extend(owned.data.iter().map(|b| b ^ 0x88));
            let p = tmp("borrow");
            std::fs::write(&p, &file).unwrap();
            let map = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&p).unwrap()) }.unwrap());
            let borrowed = Q4::mapped(map, 8, rows, cols, group, owned.scale.clone(), true);

            assert!(borrowed.is_mapped() && !owned.is_mapped());
            assert_eq!(borrowed.bytes_data(), owned.bytes_data());
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(borrowed.nib(r, c), owned.nib(r, c), "rows={rows} cols={cols} group={group} nib {r},{c}");
                }
                assert_eq!(borrowed.dequant_row(r), owned.dequant_row(r));
                // Like for like: the container pack also has the int8 kernel,
                // which is a different activation grid, so the bit-for-bit
                // claim is about the f32 kernel and needs a prep without it.
                let mut plain = DotPrep::new(cols);
                plain.fill(&x);
                assert_eq!(borrowed.dot_prepped(r, &plain), owned.dot_row(r, &x), "rows={rows} cols={cols} group={group} row={r}");
            }
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn the_prepped_kernel_matches_an_f64_dot_at_the_shipped_shape() {
        // 2048 columns at group 64 is the checkpoint's expert row. The permuted
        // activation and the fma bias are only right if every product lands where
        // the scalar oracle puts it, so the band left is f32 ordering, not format
        // loss - and both nibble encodings have to agree with it independently.
        let (rows, cols, group) = (6usize, 2048usize, 64usize);
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.017)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.31).sin() * 0.9).collect();
        let owned = Q4::quantize(&src, rows, cols, group, 2);

        let mut file = vec![0xa5u8; 8];
        file.extend(owned.data.iter().map(|b| b ^ 0x88));
        let p = tmp("shape");
        std::fs::write(&p, &file).unwrap();
        let map = Arc::new(unsafe { Mmap::map(&std::fs::File::open(&p).unwrap()) }.unwrap());
        let borrowed = Q4::mapped(map, 8, rows, cols, group, owned.scale.clone(), true);

        let mut worst = 0f64;
        for q in [&owned, &borrowed] {
            let prep = q.prepare(&x);
            for r in 0..rows {
                let w = q.dequant_row(r);
                let want: f64 = w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
                // Band against the magnitude of the terms, not of their sum: a
                // row whose dot cancels has a large relative error under any
                // f32 ordering, and a mis-paired permutation would be off by
                // the whole term sum, which this still separates by 1000x.
                let terms: f64 =
                    w.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                // The container pack dots against int8 activations, so its band
                // is the quantisation grid, not f32 ordering: each activation is
                // within amax/254 of the truth, which over a row whose |x| averages
                // half its peak is under 1% of the term sum. A mis-paired
                // permutation is off by the whole term sum, so this still
                // separates the two by 100x. The FQ4 copy keeps the f32 band.
                let band = if q.wants_int() { 1e-2 } else { 1e-6 };
                for got in [q.dot_prepped(r, &prep), q.dot_row(r, &x)] {
                    let err = (got as f64 - want).abs();
                    assert!(
                        err <= band * terms,
                        "signed={} row {r}: abs {err:.3e} over terms {terms:.3e}",
                        q.signed
                    );
                    worst = worst.max(err / terms);
                }
            }
        }
        println!("  worst error {worst:.3e} of the term sum, both encodings, {rows}x{cols} g{group}");
        let _ = std::fs::remove_file(&p);
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("fennec-q4-{}-{tag}-{n}.fq4", std::process::id()))
    }

    /// The same splice the CLI writer does, small enough to hold in RAM here.
    fn write_sc(path: &std::path::Path, items: &[(&str, &Q4)], source: (u64, usize)) {
        let mut payload = Vec::new();
        let mut table = serde_json::Map::new();
        for (name, q) in items {
            let one = q.to_bytes(name);
            let hl = u64::from_le_bytes(one[8..16].try_into().unwrap()) as usize;
            let head: serde_json::Value = serde_json::from_slice(&one[16..16 + hl]).unwrap();
            let mut item = head["items"][*name].as_object().unwrap().clone();
            let off = payload.len();
            for k in ["data", "scale"] {
                let a = item[k].as_array().unwrap();
                let (a0, a1) = (a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize);
                item[k] = serde_json::json!([off + a0, off + a1]);
            }
            payload.extend_from_slice(&one[16 + hl..]);
            table.insert(name.to_string(), serde_json::Value::Object(item));
        }
        let json = serde_json::to_vec(&serde_json::json!({
            "items": table, "source": { "total_bytes": source.0, "tensors": source.1 }
        }))
        .unwrap();
        let mut f = Vec::new();
        f.extend_from_slice(MAGIC);
        f.extend_from_slice(&(json.len() as u64).to_le_bytes());
        f.extend_from_slice(&json);
        f.extend_from_slice(&payload);
        std::fs::write(path, f).unwrap();
    }

    #[test]
    fn a_mapped_row_range_is_the_matrix_quantising_would_have_produced() {
        // This is the whole point of the cache: a cold start must get the exact
        // bytes the quantiser produces, not an approximation of them.
        let src: Vec<f32> = (0..64 * 32).map(|i| (i as f32).sin() * 0.6).collect();
        let q = Q4::quantize(&src, 64, 32, 32, 4);
        let p = tmp("rows");
        write_sc(&p, &[("blk.0.ffn.gate_up_exps", &q)], (4096, 7));
        let sc = Sidecar::open(&p).unwrap();
        sc.check_source(4096, 7).unwrap();
        let it = sc.item("blk.0.ffn.gate_up_exps").unwrap();
        assert_eq!((it.rows, it.cols, it.group), (64, 32, 32));
        for (r0, r1) in [(0, 64), (8, 24), (63, 64), (0, 1)] {
            let got = sc.rows(&it, r0, r1).unwrap();
            let want = q.slice_rows(r0, r1);
            assert_eq!(got.data, want.data, "rows {r0}..{r1} data");
            assert_eq!(got.scale, want.scale, "rows {r0}..{r1} scale");
            assert_eq!(got.rows, r1 - r0);
        }
        assert!(sc.rows(&it, 0, 65).is_err());
        assert!(sc.rows(&it, 9, 4).is_err());
        assert!(sc.item("blk.9.ffn.gate_up_exps").is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_cache_from_another_checkpoint_is_refused() {
        let src: Vec<f32> = (0..32 * 32).map(|i| (i as f32) * 0.01).collect();
        let q = Q4::quantize(&src, 32, 32, 32, 1);
        let p = tmp("source");
        write_sc(&p, &[("m", &q)], (4096, 7));
        let sc = Sidecar::open(&p).unwrap();
        assert!(sc.check_source(4096, 7).is_ok());
        assert!(sc.check_source(4097, 7).is_err());
        assert!(sc.check_source(4096, 8).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_malformed_sidecar_is_an_error_not_a_panic() {
        let p = tmp("bad");
        std::fs::write(&p, b"not a sidecar at all").unwrap();
        assert!(Sidecar::open(&p).is_err());
        // A header that claims to be longer than the file.
        let mut f = Vec::new();
        f.extend_from_slice(MAGIC);
        f.extend_from_slice(&1_000_000u64.to_le_bytes());
        std::fs::write(&p, &f).unwrap();
        assert!(Sidecar::open(&p).is_err());
        // Valid magic and length, but the payload offsets run past the file.
        let json = serde_json::to_vec(&serde_json::json!({
            "items": { "m": { "shape": [4, 32], "group": 32,
                              "data": [0, 64], "scale": [64, 128] } },
            "source": { "total_bytes": 1, "tensors": 1 }
        }))
        .unwrap();
        let mut f = Vec::new();
        f.extend_from_slice(MAGIC);
        f.extend_from_slice(&(json.len() as u64).to_le_bytes());
        f.extend_from_slice(&json);
        std::fs::write(&p, &f).unwrap();
        let sc = Sidecar::open(&p).unwrap();
        let it = sc.item("m").unwrap();
        assert!(sc.rows(&it, 0, 4).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_integer_dot_tracks_the_f32_dot_on_a_container_pack() {
        // Flipping every byte of a quantize() pack turns its offset-binary
        // nibbles into the container's two's complement, so the two dots are
        // reading the same weights and only the activation grid differs.
        let (rows, cols, group) = (5usize, 256usize, 64usize);
        let src: Vec<f32> =
            (0..rows * cols).map(|k| ((k * 37) % 211) as f32 / 105.0 - 1.0).collect();
        let mut q = Q4::quantize(&src, rows, cols, group, 2);
        for b in q.data.own_mut() {
            *b ^= 0x88;
        }
        q.signed = true;
        assert!(q.wants_int(), "group 64 and width 256 must take the integer path");

        let x: Vec<f32> = (0..cols).map(|c| (c as f32 * 0.21).sin() * 0.9).collect();
        let p = q.prepare(&x);
        assert!(p.q.is_some(), "a container pack must build the int8 prep");
        let mut worst = 0f64;
        for r in 0..rows {
            let w = q.dequant_row(r);
            let want: f64 = w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
            let terms: f64 = w.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
            let got = q.dot_prepped(r, &p) as f64;
            let err = (got - want).abs();
            // int8 over a 64-column block puts each activation within 1/254 of
            // the block's amax, so the dot lands well inside 1% of its terms.
            assert!(err <= 0.01 * terms, "row {r}: abs {err:.3e} over terms {terms:.3e}");
            worst = worst.max(err / terms);
        }
        println!("  int8 vs f64 worst relative error {worst:.3e}");
    }

    #[test]
    fn an_fq4_copy_keeps_the_f32_dot() {
        // The bf16 export's FQ4 copy is checked against a bit-exact reference at
        // 1e-6, so it must not silently pick up the integer activation grid.
        let src: Vec<f32> = (0..4 * 128).map(|k| (k as f32 * 0.017).cos()).collect();
        let q = Q4::quantize(&src, 4, 128, 64, 2);
        assert!(!q.signed);
        assert!(!q.wants_int());
        let x: Vec<f32> = (0..128).map(|c| (c as f32 * 0.11).sin()).collect();
        assert!(q.prepare(&x).q.is_none());
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{DotPrep, I8Prep, Q4};
    use std::arch::x86_64::*;

    /// One chunk == 32 weights == 16 packed bytes, read as two 8-byte halves.
    /// A group is `group / 32` chunks sharing one scale, hoisted out of the loop.
    ///
    /// Widening a byte gives its low nibble in one vector and its high nibble in
    /// another, i.e. the even and odd columns already separated. `p` carries x
    /// in that same order, so nothing has to be unpacked or permuted here: the
    /// only shuffle-class ops left are the two `vpmovzxbd` and the two shifts
    /// that extract the nibbles. `fma(u, x, -8x)` rounds once to the same product
    /// `(u - 8) * x` would, so the bias costs no accuracy and no extra op.
    ///
    /// # Safety
    /// Requires AVX2 and FMA, `p` prepared for `q.cols` rows of width `q.cols`,
    /// row `r` inside the matrix, `q.group % 32 == 0` and `q.cols % 32 == 0`.
    /// The integer dot: 64 columns per iteration against int8 activations.
    ///
    /// # Safety
    /// Requires AVX2 and FMA, `p` built for `q.cols`, row `r` inside the matrix,
    /// and `q.wants_int()`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_prepped_int(q: &Q4, r: usize, p: &I8Prep) -> f32 {
        if q.signed {
            unsafe { dot_int::<true>(q, r, p) }
        } else {
            unsafe { dot_int::<false>(q, r, p) }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn block_i32<const SIGNED: bool>(
        row_bytes: *const u8,
        p: &I8Prep,
        b: usize,
        m4: __m256i,
        flip: __m256i,
        one: __m256i,
    ) -> __m256i {
        let mut v = unsafe { _mm256_loadu_si256(row_bytes.add(b * 32) as *const __m256i) };
        if SIGNED {
            // One XOR on the packed byte flips bit 3 of both nibbles, so the
            // split below lands on offset-binary without a second pass.
            v = _mm256_xor_si256(v, flip);
        }
        let lo = _mm256_and_si256(v, m4);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4);
        let xe = unsafe { _mm256_loadu_si256(p.xe.as_ptr().add(b * 32) as *const __m256i) };
        let xo = unsafe { _mm256_loadu_si256(p.xo.as_ptr().add(b * 32) as *const __m256i) };
        // maddubs takes the nibbles unsigned and the activations signed.
        // 15 * 127 * 2 = 3810 fits the i16 pair sum with room to spare.
        let s = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xe), one),
            _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xo), one),
        );
        _mm256_sub_epi32(s, unsafe { _mm256_loadu_si256(p.corr.as_ptr().add(b * 8) as *const __m256i) })
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_int<const SIGNED: bool>(q: &Q4, r: usize, p: &I8Prep) -> f32 {
        let blocks = q.cols / 64;
        let per = q.group / 64;
        let ng = q.ngroups();
        let row_bytes = unsafe { q.data.as_ptr().add(r * (q.cols / 2)) };
        let row_scale = unsafe { q.scale.as_ptr().add(r * ng) };
        let m4 = _mm256_set1_epi8(15);
        let flip = _mm256_set1_epi8(0x88u8 as i8);
        let one = _mm256_set1_epi16(1);
        // Four independent accumulators break the serial FMA chain: a single
        // accumulator makes every block wait the ~4-cycle FMA latency of the
        // last, so a 2048-column row (32 blocks) stalls ~128 cycles per row on
        // the accumulate alone. Four chains let the two FMA ports stay busy and
        // the maddubs of the next block overlap the accumulate of this one. The
        // sum is the same set of products, only the addition order changes,
        // which stays well inside the routed-expert golden band.
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let scale = |b: usize| _mm256_set1_ps(unsafe { *row_scale.add(b / per) } * unsafe { *p.d.as_ptr().add(b) });
        let mut b = 0;
        while b + 4 <= blocks {
            let s0 = unsafe { block_i32::<SIGNED>(row_bytes, p, b, m4, flip, one) };
            let s1 = unsafe { block_i32::<SIGNED>(row_bytes, p, b + 1, m4, flip, one) };
            let s2 = unsafe { block_i32::<SIGNED>(row_bytes, p, b + 2, m4, flip, one) };
            let s3 = unsafe { block_i32::<SIGNED>(row_bytes, p, b + 3, m4, flip, one) };
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), scale(b), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), scale(b + 1), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s2), scale(b + 2), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s3), scale(b + 3), a3);
            b += 4;
        }
        while b < blocks {
            let s = unsafe { block_i32::<SIGNED>(row_bytes, p, b, m4, flip, one) };
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), scale(b), a0);
            b += 1;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps::<1>(h, h)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_prepped(q: &Q4, r: usize, p: &DotPrep) -> f32 {
        if q.signed {
            unsafe { dot_chunks::<true>(q, r, p) }
        } else {
            unsafe { dot_chunks::<false>(q, r, p) }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dot_chunks<const SIGNED: bool>(q: &Q4, r: usize, p: &DotPrep) -> f32 {
        let chunks = q.cols / 32;
        let per = q.group / 32;
        let ng = q.ngroups();
        let row_bytes = unsafe { q.data.as_ptr().add(r * (q.cols / 2)) };
        let row_scale = unsafe { q.scale.as_ptr().add(r * ng) };
        let mask = _mm256_set1_epi32(0x0f);
        let flip = _mm256_set1_epi32(8);
        let mut acc_all = _mm256_setzero_ps();
        for gi in 0..ng {
            let sc = _mm256_set1_ps(unsafe { *row_scale.add(gi) });
            // The last group can be short when cols is not a multiple of group.
            let g1 = ((gi + 1) * per).min(chunks);
            for g in gi * per..g1 {
                let b = unsafe { row_bytes.add(g * 16) };
                let xp = unsafe { p.xp.as_ptr().add(g * 32) };
                let nx = unsafe { p.nx.as_ptr().add(g * 32) };
                // Eight bytes each, not sixteen: the second load would run past
                // the row on the final chunk of the final row.
                let v0 = _mm256_cvtepu8_epi32(unsafe { _mm_loadl_epi64(b as *const __m128i) });
                let v1 = _mm256_cvtepu8_epi32(unsafe { _mm_loadl_epi64(b.add(8) as *const __m128i) });
                let (mut l0, mut h0) = (_mm256_and_si256(v0, mask), _mm256_srli_epi32::<4>(v0));
                let (mut l1, mut h1) = (_mm256_and_si256(v1, mask), _mm256_srli_epi32::<4>(v1));
                if SIGNED {
                    // two's complement nibble -> the offset-binary the rest of
                    // the kernel expects, without touching the mapped bytes
                    l0 = _mm256_xor_si256(l0, flip);
                    h0 = _mm256_xor_si256(h0, flip);
                    l1 = _mm256_xor_si256(l1, flip);
                    h1 = _mm256_xor_si256(h1, flip);
                }
                let t0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(l0),
                    unsafe { _mm256_loadu_ps(xp) },
                    unsafe { _mm256_loadu_ps(nx) },
                );
                let t1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(h0),
                    unsafe { _mm256_loadu_ps(xp.add(8)) },
                    unsafe { _mm256_loadu_ps(nx.add(8)) },
                );
                let t2 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(l1),
                    unsafe { _mm256_loadu_ps(xp.add(16)) },
                    unsafe { _mm256_loadu_ps(nx.add(16)) },
                );
                let t3 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(h1),
                    unsafe { _mm256_loadu_ps(xp.add(24)) },
                    unsafe { _mm256_loadu_ps(nx.add(24)) },
                );
                let s = _mm256_add_ps(_mm256_add_ps(t0, t1), _mm256_add_ps(t2, t3));
                acc_all = _mm256_fmadd_ps(s, sc, acc_all);
            }
        }
        let s = _mm_add_ps(
            _mm256_castps256_ps128(acc_all),
            _mm256_extractf128_ps::<1>(acc_all),
        );
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
    }
}
