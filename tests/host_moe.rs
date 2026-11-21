//! The host (RAM) tier of the MoE, measured instead of assumed.
//!
//! On the served CUDA decode `moe host experts` costs 17.5-20.7 ms/token - a
//! third to two fifths of the step, the largest single stage in the engine. Two
//! questions about it have been answered from arithmetic and never from a
//! measurement, and #464 is the third attempt at the second one:
//!
//! * *Which kernel runs?* The int8-activation `maddubs` dot (`avx2::dot_prepped_int`,
//!     shipped by #81) and the float planar nibble dot (`avx2::dot_prepped`, #98)
//!     both exist, and `Q4::dot_prepped` chooses between them on `Q4::wants_int()`.
//!     Section 1 prints what the container actually asks for instead of what a
//!     commit message implies; section 3 times both shapes over the same bytes.
//! * *How fast can this box be?* `Bandwidth::host_gbs` is not that number. It is
//!     `host_bytes / host_ns` taken across the very section whose speed is in
//!     question (`Model::host_traffic` src/model.rs:3976, `host_ns` src/model.rs:1228),
//!     which is the same span as the `moe host experts` stage - so it reports the
//!     achieved rate and can only ever agree with the stage it is meant to price.
//!     r23 showed the second reason not to trust it: the same shape read 4.2 GB/s
//!     cold and 15.6 warm 90 s apart, with the streamed window moving with it, so
//!     it is a lane span other tiers share rather than a cost of a byte. #473 took
//!     it out of `Bandwidth::apply`, which leaves `Plan::build` pricing with the
//!     profile constant and this ceiling as the only independent number. Section 2
//!     is what measures the ceiling; sections 1 and 3 say how far the kernel is
//!     from it.
//!
//! The ceiling is measured twice on purpose. Since #419 the host tier's weights
//! are borrowed from the shard mapping - they are file pages, not bytes the
//! process owns - so the same read runs over a private allocation and over a
//! shard. If the mapping is the slower of the two, that is the tier's real
//! ceiling, no kernel shape reaches the private-page number, and the lever is
//! placement rather than code.
//!
//! # Reading the output
//!
//! * The pool is a `OnceLock`, so one process measures one worker count: run the
//!     probe once per width with `PROBE_THREADS`. Six is the width at which this
//!     box pins one worker per performance core; eight and above stay unpinned and
//!     let the scheduler place workers on efficiency cores, which crack every
//!     256-bit op in half. `gemv_i8_throughput_at_the_real_projection_shapes` in
//!     `tests/dense_i8.rs` has the same rule for the same reason.
//! * Absolute GB/s across sessions is box state, not code. Quote the ratios:
//!     stage against ceiling, int8 against float, scattered against sequential.
//! * `#\[ignore]`d, prints, and asserts nothing about speed. The only assertion is
//!     that both kernels land within the int8 band of a dequantised reference,
//!     which is what makes the timing rows a comparison of shapes rather than of
//!     two different functions.

use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use fennec::config::TextConfig;
use fennec::kernel::{self, SharedPtr};
use fennec::model;
use fennec::quant::{self, DotPrep, Q4};
use fennec::safetensors::Shards;

const DEFAULT: &str = r"D:\fennec-models\qwen36-35b";
const MIB: f64 = (1u64 << 20) as f64;

/// GB/s from a byte count and the microseconds it took.
fn gbs(bytes: usize, us: f64) -> f64 {
    bytes as f64 / (us * 1e3)
}

fn dir() -> String {
    env::var("FENNEC_HOST_MODEL").unwrap_or_else(|_| DEFAULT.to_string())
}

/// The mapping, the config and the bound prefix, or `None` with a printed
/// reason: the container is 20 GiB and lives on one machine, and a lane that
/// fails on the other two platforms tests nothing.
fn ckpt() -> Option<(Arc<Shards>, Arc<TextConfig>)> {
    let d = dir();
    if !std::path::Path::new(&d).is_dir() {
        println!("  {d} is not on this machine: skipped");
        return None;
    }
    let mut cfg = TextConfig::from_file(&format!("{d}/config.json"))
        .inspect_err(|e| println!("  config unreadable: {e}"))
        .ok()?;
    let sh = match Shards::open(&d) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            println!("  checkpoint unreadable: {e}");
            return None;
        }
    };
    model::bind_prefix(&sh, &mut cfg);
    Some((sh, Arc::new(cfg)))
}

/// `n` expert ids strided through the layer rather than adjacent: the same set on
/// every run, and one that spreads over the shard instead of clustering in the
/// first block, which is the gather order a router actually produces.
fn spread(experts: usize, n: usize) -> Vec<usize> {
    (0..n).map(|k| (k * 97 + 13) % experts).collect()
}

/// Read every byte of `buf` in `workers` disjoint runs, eight u64 accumulators
/// deep, one `par_index` per lap.
///
/// The shape matters: a byte-at-a-time sum tops out near one byte per cycle and
/// would report the ALU, not the memory system. Eight loads plus eight adds per
/// 64 bytes is under three cycles per vector of work on a performance core, so at
/// any clock this box runs at it asks for more bytes than the channels can hand
/// over - which is the point of the row. `black_box` per rep keeps LLVM from
/// noticing that nothing writes the buffer and hoisting the loads out of the
/// repeat loop, which would turn the sustained phase back into a multiply.
///
/// A lap is timed first and then repeated until three seconds have passed: one
/// short burst measures a ramp and a page fault, and #470 learned that on the
/// other side of the same process. Returns the sustained bytes/s.
fn read_ceiling(buf: &[u8], workers: usize) -> f64 {
    let acc = AtomicU64::new(0);
    let p = SharedPtr::new(&acc as *const AtomicU64 as *mut AtomicU64);
    let bytes = buf.len();
    let lap = |reps: u32| {
        let data = SharedPtr::new(buf.as_ptr() as *mut u8);
        kernel::par_index(bytes, workers, |start, len| {
            // Safety: par_index blocks until every worker returns, the runs it
            // hands out are disjoint and read-only, and the buffer outlives the
            // call. `read_unaligned` because a shard's data starts at a header
            // length, not at an eight-byte boundary.
            let src = unsafe { std::slice::from_raw_parts(data.at(start), len) };
            let words = len / 64;
            let mut w = [0u64; 8];
            for _ in 0..reps {
                let base = std::hint::black_box(src).as_ptr();
                for c in 0..words {
                    for k in 0..8 {
                        let o = c * 64 + k * 8;
                        w[k] = w[k].wrapping_add(unsafe { (base.add(o) as *const u64).read_unaligned() });
                    }
                }
                for &b in &src[words * 64..] {
                    w[0] = w[0].wrapping_add(b as u64);
                }
            }
            let sum = w.into_iter().fold(0u64, |a, v| a.wrapping_add(v));
            unsafe { (*p.at(0)).fetch_add(sum, Ordering::Relaxed) };
        });
    };
    lap(1);
    let t = std::time::Instant::now();
    lap(1);
    let one = t.elapsed().as_secs_f64().max(1e-4);
    let reps = (1.0 / one).max(1.0) as u32;
    let t = std::time::Instant::now();
    let mut laps = 0u32;
    while t.elapsed().as_secs_f64() < 3.0 {
        lap(reps);
        laps += 1;
    }
    let secs = t.elapsed().as_secs_f64();
    let moved = bytes as f64 * laps as f64 * reps as f64;
    println!(
        "    {:>6.0} MiB x {laps} laps x {reps} reps in {secs:.2} s = {:5.1} GB/s (read, checksum {:02x})",
        bytes as f64 / MIB,
        moved / secs / 1e9,
        acc.load(Ordering::Relaxed) & 0xff,
    );
    moved / secs
}

/// One host-expert phase pair, split across the pool exactly the way
/// `MoeLayer::cpu_picks` splits it: `picks x i` gate/up rows, then `picks x h`
/// down rows, two dispatches and two barriers. Repeats until 1.2 s have passed so
/// a single dispatch's barrier is not the measurement, and returns the microseconds
/// per call plus the weight bytes one call reads.
fn timed_picks(picks: &[(Q4, Q4)], x: &[f32], workers: usize) -> (f64, usize) {
    let h = x.len();
    let np = picks.len();
    let i = picks[0].1.cols;
    let int = picks[0].0.wants_int();
    let mut px = DotPrep::with_int(h, int);
    px.fill(x);
    let mut hh = vec![0f32; np * i];
    let mut parts = vec![vec![0f32; h]; np];
    let hhp = SharedPtr::new(hh.as_mut_ptr());
    let pp = SharedPtr::new(parts.as_mut_ptr());
    let once = || {
        kernel::par_index(np * i, workers, |s, l| {
            for g in s..s + l {
                let p = g / i;
                let j = g - p * i;
                let (gu, _) = &picks[p];
                let a = gu.dot_prepped(j, &px);
                let b = gu.dot_prepped(i + j, &px);
                // silu(a) * b, without the transcendental: this is a bytes probe.
                unsafe { *hhp.at(p * i + j) = a * b };
            }
        });
        kernel::par_index(np * h, workers, |s, l| {
            // One prep per worker, refilled whenever the run crosses into the
            // next expert, as in cpu_picks. A run that starts mid-expert must
            // refill at once or it dots a half-filled activation.
            let mut ph = DotPrep::with_int(i, int);
            let mut cur = usize::MAX;
            for g in s..s + l {
                let p = g / h;
                let j = g - p * h;
                if p != cur {
                    let src = unsafe { std::slice::from_raw_parts(hhp.at(p * i), i) };
                    ph.fill(src);
                    cur = p;
                }
                let v = picks[p].1.dot_prepped(j, &ph);
                unsafe { (&mut *pp.at(p))[j] += v };
            }
        });
    };
    once();
    let t = std::time::Instant::now();
    once();
    let one = t.elapsed().as_secs_f64().max(1e-4);
    let reps = (1.2 / one).max(1.0) as usize;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        once();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let mut sink = 0f64;
    for part in &parts {
        for &v in part {
            sink += v as f64;
        }
    }
    std::hint::black_box(sink);
    let bytes = picks
        .iter()
        .map(|(gu, dn)| gu.bytes_total() + dn.bytes_total())
        .sum();
    (us, bytes)
}

/// Error of each branch against a dequantised fp32 reference, over the first
/// `rows` rows, as a fraction of the row's term sum (the same normalisation
/// `quant.rs`'s own kernel test uses: a row that cancels has a huge relative
/// error under any ordering). Without this the two timing rows could be two
/// different functions rather than two shapes of one function.
fn agree(q: &Q4, x: &[f32], rows: usize) -> (f64, f64) {
    let mut int_worst = 0f64;
    let mut flt_worst = 0f64;
    let mut p = DotPrep::with_int(x.len(), true);
    p.fill(x);
    for r in 0..rows {
        let w = q.dequant_row(r);
        let want: f64 = w.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let terms: f64 = w.iter().zip(x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        quant::set_planar_dot(false);
        let a = q.dot_prepped(r, &p) as f64;
        quant::set_planar_dot(true);
        let b = q.dot_prepped(r, &p) as f64;
        quant::set_planar_dot(false);
        int_worst = int_worst.max((a - want).abs() / terms.max(1e-9));
        flt_worst = flt_worst.max((b - want).abs() / terms.max(1e-9));
    }
    (int_worst, flt_worst)
}

#[test]
#[ignore = "throughput probe: needs a quiet box, the 35B container, and ~1 GiB of touched pages"]
fn host_expert_tier_shapes() {
    let Some((sh, cfg)) = ckpt() else { return };
    let workers: usize = env::var("PROBE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    kernel::set_threads(workers);

    let (h, i) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let (nl, ne) = (cfg.num_hidden_layers, cfg.num_experts);
    println!("  host expert shapes on {}", dir());
    println!("  h={h} i={i} experts={ne} layers={nl} top_k={} physical cores {}", cfg.num_experts_per_tok, kernel::physical_cores());
    println!("  pool: {workers} workers (one width per process; re-run with PROBE_THREADS)");

    let group = match model::packed_group(&sh, &cfg.prefix, h, i) {
        Some(g) => g,
        None => {
            println!("  not a packed int4 container (packed_group refused it): nothing to measure");
            return;
        }
    };

    // ---- 1. what the container asks the dot for -------------------------------
    let ids = spread(ne, cfg.num_experts_per_tok);
    let packs = model::map_experts(&sh, &cfg, 0, &ids, group)
        .expect("map experts")
        .expect("a packed container");
    let (gu, dn) = &packs[0];
    let pick = gu.bytes_total() + dn.bytes_total();
    println!("  pack: gate_up {}x{h} g{group}, down {}x{i} g{group}", gu.rows, dn.rows);
    println!(
        "  wants_int: gate_up {} down {} (signed {}, group {group} % 64 == {}, cols {h}/{i})",
        gu.wants_int(),
        dn.wants_int(),
        gu.signed,
        group % 64,
    );
    println!(
        "  served branch: {}",
        if gu.wants_int() && !quant::planar_dot() {
            "avx2::dot_prepped_int, int8 activations through maddubs"
        } else {
            "avx2::dot_prepped, the float planar nibble dot"
        }
    );
    println!(
        "  one pick reads {} nibble + {} scale = {} bytes ({:.2} MiB); {} picks x {nl} layers \
         = {} MiB/token if every routed pick is a host pick",
        gu.bytes_data() + dn.bytes_data(),
        pick - (gu.bytes_data() + dn.bytes_data()),
        pick,
        pick as f64 / MIB,
        cfg.num_experts_per_tok,
        cfg.num_experts_per_tok * nl * pick / (1 << 20),
    );

    // ---- 2. the ceiling, private pages and file pages -------------------------
    println!("  read ceilings, {workers}-worker pool, 3 s sustained:");
    let owned: Vec<u8> = vec![0xA5u8; 512 << 20];
    println!("  a private allocation:");
    let priv_bw = read_ceiling(&owned, workers);
    drop(owned);
    let (big, blen) = (0..sh.n_shards())
        .map(|f| (f, sh.data(f).len()))
        .max_by_key(|&(_, n)| n)
        .expect("a shard");
    println!("  shard {big} ({} MiB), still file pages:", blen / (1 << 20));
    let file_bw = read_ceiling(sh.data(big), workers);
    let floor = std::time::Instant::now();
    for _ in 0..500 {
        kernel::par_index(workers, workers, |_, _| {});
    }
    let dispatch = floor.elapsed().as_secs_f64() * 1e6 / 500.0;
    let ceil = priv_bw.min(file_bw);
    println!(
        "  file pages reach {:.0}% of private pages; an empty {workers}-task dispatch \
         costs {dispatch:.1} us, so the two per layer are {:.2} ms/token over {nl} layers",
        100.0 * file_bw / priv_bw,
        2.0 * dispatch * nl as f64 / 1e3,
    );

    // ---- 3. the two kernels, over the same bytes ------------------------------
    let x: Vec<f32> = (0..h).map(|k| ((k as f32 * 0.37).sin()) * 0.8).collect();
    let (e_int, e_flt) = agree(gu, &x, 16);
    println!(
        "  gate_up, 16 rows against a dequantised reference: int8 {e_int:.2e}, \
         float {e_flt:.2e} of the term sum"
    );
    // The bands quant.rs holds its own kernel test to: the int8 branch is priced
    // against the activation grid, the float branch against f32 ordering.
    assert!(
        e_int < 1e-2,
        "the int8 host dot is off a dequantised row by {e_int:.3} of its term sum, so \
         the rows below are not the served kernel"
    );
    assert!(
        e_flt < 1e-5,
        "the float host dot is off a dequantised row by {e_flt:.3} of its term sum, so \
         the comparison below is not one function in two shapes"
    );
    let time_shape = |label: &str, int: bool| -> (f64, usize) {
        quant::set_planar_dot(!int);
        let r = timed_picks(&packs, &x, workers);
        quant::set_planar_dot(false);
        println!(
            "    {label:<34} {:8.1} us for {} picks   {:5.1} GB/s",
            r.0,
            packs.len(),
            gbs(r.1, r.0),
        );
        r
    };
    println!("  {} host picks, row-split across the pool, as served:", packs.len());
    let (us_int, bytes) = time_shape("int8 activations (the served path)", true);
    let (us_flt, _) = time_shape("float planar nibble dot (--planar-dot)", false);
    println!(
        "  int8 is {:.2}x the float shape; at {:.1} GB/s it runs at {:.0}% of the {:.1} GB/s ceiling",
        us_flt / us_int,
        gbs(bytes, us_int),
        100.0 * gbs(bytes, us_int) / (ceil / 1e9),
        ceil / 1e9,
    );

    // ---- 4. does the per-token gather defeat the prefetcher? ------------------
    // Same layers, same pick count, same bytes: strided expert ids against
    // adjacent ones. If a decode's random gather costs more than a pager's
    // dream, the loss is address order and no kernel can buy it back.
    let layers: Vec<usize> = (0..4).map(|k| k * (nl / 4)).collect();
    let mut scattered = Vec::new();
    let mut adjacent = Vec::new();
    for &l in &layers {
        if let Some(ps) = model::map_experts(&sh, &cfg, l, &spread(ne, 8), group).ok().flatten() {
            scattered.extend(ps);
        }
        if let Some(ps) = model::map_experts(&sh, &cfg, l, &(0..8).collect::<Vec<_>>(), group)
            .ok()
            .flatten()
        {
            adjacent.extend(ps);
        }
    }
    if scattered.len() == adjacent.len() && !scattered.is_empty() {
        let (us_s, bytes_s) = timed_picks(&scattered, &x, workers);
        let (us_a, bytes_a) = timed_picks(&adjacent, &x, workers);
        println!(
            "  {} picks, strided ids:   {us_s:8.1} us, {:5.1} GB/s",
            scattered.len(),
            gbs(bytes_s, us_s),
        );
        println!(
            "  {} picks, adjacent ids:  {us_a:8.1} us, {:5.1} GB/s",
            adjacent.len(),
            gbs(bytes_a, us_a),
        );
        println!("  gather order costs {:+.1}%", 100.0 * (us_s / us_a - 1.0));
        println!(
            "  {nl} layers at the strided rate is {:.1} ms/token, if every routed pick \
             is a host pick.",
            nl as f64 * us_s / layers.len() as f64 / 1e3,
        );
    }
    println!("  compare with prof::MOE_CPU on a served decode (17.5-20.7 ms/token): a stage");
    println!("    that misses this projection by more than the dispatch floor is not picks alone.");
}
