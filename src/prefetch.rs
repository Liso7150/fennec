//! Background prefetcher for cold experts on SSD.
//!
//! While layer N computes, a worker reads layer N+1's hottest streamed experts
//! from disk into an LRU cache, so a router pick that has no tier to live in is
//! often already in RAM by the time it is asked for.
//!
//! The hint has to be *predictive*. Re-asking for the experts this layer just
//! missed is a guaranteed miss: the read cannot land before the same layer's
//! disk path runs, so the caller blocks on the pool and does the identical read
//! a second time. `sched::LayerPlan::streamed` arrives ranked hottest-first (it
//! is the tail of `rank_order`), so its prefix is the prediction and no separate
//! predictor is needed.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use crate::config::TextConfig;
use crate::quant::Q4;
use crate::safetensors::Shards;

/// How long a caller waits for a read the worker already owns before doing it
/// itself. Generous on purpose: the alternative is a duplicate read of the same
/// bytes, which costs the disk and delays the token that was waiting for it. A
/// streamed expert is at worst a bf16 shard read plus a quantise pass, which
/// measures single-digit milliseconds here.
pub(crate) const AHEAD_WAIT: Duration = Duration::from_millis(250);

/// Cache key: (layer, expert_id) uniquely identifies one expert's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub layer: usize,
    pub expert_id: usize,
}

/// A cached expert: gate_up and down projections, already quantised to int4.
#[derive(Clone)]
pub struct CachedExpert {
    pub gate_up: Q4,
    pub down: Q4,
}

impl CachedExpert {
    /// Byte footprint of the int4 data + scales, for cache budget accounting.
    pub fn byte_size(&self) -> usize {
        self.gate_up.data.len() + self.gate_up.scale.len() * 4
            + self.down.data.len() + self.down.scale.len() * 4
    }
}

/// A request to prefetch an expert from disk.
#[derive(Clone, Copy, Debug)]
pub struct PrefetchRequest {
    pub layer: usize,
    pub expert_id: usize,
}

/// LRU cache for prefetched experts. Values are held behind `Arc` so a hit costs
/// a reference count rather than a copy of a megabyte and a half of weights.
pub struct LruCache {
    data: HashMap<CacheKey, Arc<CachedExpert>>,
    order: Vec<CacheKey>,
    max_bytes: usize,
    current_bytes: usize,
}

impl LruCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            data: HashMap::new(),
            order: Vec::new(),
            max_bytes,
            current_bytes: 0,
        }
    }

    /// Whether this key is in the cache, without touching recency.
    pub fn holds(&self, key: &CacheKey) -> bool {
        self.data.contains_key(key)
    }

    pub fn insert(&mut self, key: CacheKey, expert: Arc<CachedExpert>) {
        let size = expert.byte_size();

        while self.current_bytes + size > self.max_bytes && !self.order.is_empty() {
            let evict_key = self.order.pop().unwrap();
            if let Some(evicted) = self.data.remove(&evict_key) {
                self.current_bytes -= evicted.byte_size();
            }
        }

        if let Some(old) = self.data.remove(&key) {
            self.current_bytes -= old.byte_size();
            self.order.retain(|k| *k != key);
        }

        self.order.insert(0, key);
        self.current_bytes += size;
        self.data.insert(key, expert);
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<Arc<CachedExpert>> {
        let expert = self.data.get(key).cloned()?;
        self.order.retain(|k| k != key);
        self.order.insert(0, *key);
        Some(expert)
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.data.contains_key(key)
    }

    pub fn entry_count(&self) -> usize {
        self.data.len()
    }

    pub fn byte_size(&self) -> usize {
        self.current_bytes
    }
}

/// The work list plus the set of keys a caller may be waiting on. One condvar
/// guards both: a dropped request has to wake anyone waiting for it, and a
/// completed one has to do the same, so they cannot live under separate locks.
#[derive(Default)]
struct Queue {
    reqs: VecDeque<PrefetchRequest>,
    pending: HashSet<CacheKey>,
    /// The request the worker is inside right now. A caller that needs a *queued*
    /// expert has to be able to tell "loading" from "seventh in line", or it pays
    /// the whole queue as latency.
    in_flight: Option<CacheKey>,
    stop: bool,
}

struct Inner {
    queue: Mutex<Queue>,
    ready: Condvar,
    cache: Mutex<LruCache>,
    shards: Arc<Shards>,
    config: Arc<TextConfig>,
    group: usize,
    threads: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    loaded: AtomicU64,
}

/// Background prefetcher that loads experts from disk into cache.
pub struct Prefetcher {
    inner: Arc<Inner>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Prefetcher {
    /// Start a new prefetcher with the given cache budget (bytes).
    pub fn start(
        cache_bytes: usize,
        shards: Arc<Shards>,
        config: Arc<TextConfig>,
        group: usize,
        threads: usize,
    ) -> Self {
        let inner = Arc::new(Inner {
            queue: Mutex::new(Queue::default()),
            ready: Condvar::new(),
            cache: Mutex::new(LruCache::new(cache_bytes)),
            shards,
            config,
            group,
            threads,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            loaded: AtomicU64::new(0),
        });
        let handle = {
            let inner = inner.clone();
            thread::spawn(move || Self::run(inner))
        };
        Self {
            inner,
            handle: Some(handle),
        }
    }

    fn run(inner: Arc<Inner>) {
        loop {
            let req = {
                let mut q = inner.queue.lock().unwrap();
                loop {
                    if q.stop {
                        return;
                    }
                    match q.reqs.pop_front() {
                        Some(r) => break r,
                        None => q = inner.ready.wait(q).unwrap(),
                    }
                }
            };
            let key = CacheKey {
                layer: req.layer,
                expert_id: req.expert_id,
            };
            // Announce the read before starting it, so a caller that needs this
            // expert while it is loading waits for it instead of duplicating it.
            inner.queue.lock().unwrap().in_flight = Some(key);
            let held = inner.cache.lock().unwrap().contains(&key);
            if !held {
                match load_single_expert(
                    &inner.shards,
                    &inner.config,
                    req.layer,
                    req.expert_id,
                    inner.group,
                    inner.threads,
                ) {
                    Ok(expert) => {
                        inner.cache.lock().unwrap().insert(key, Arc::new(expert));
                        inner.loaded.fetch_add(1, Relaxed);
                    }
                    Err(e) => {
                        eprintln!("[prefetch] layer {} expert {}: {e}", req.layer, req.expert_id);
                    }
                }
            }
            let mut q = inner.queue.lock().unwrap();
            q.in_flight = None;
            q.pending.remove(&key);
            drop(q);
            inner.ready.notify_all();
        }
    }

    /// Queue `ids` for `layer`, replacing whatever was held for an earlier layer.
    /// A request for a layer the token has already passed is not late, it is
    /// wrong: it is the one thing that lets the worker fall permanently behind,
    /// so the list is never more than one layer deep.
    pub fn prefetch_ahead(&self, layer: usize, ids: &[usize]) {
        if ids.is_empty() {
            return;
        }
        let mut q = self.inner.queue.lock().unwrap();
        let stale: Vec<PrefetchRequest> = q.reqs.drain(..).collect();
        for s in stale {
            q.pending.remove(&CacheKey {
                layer: s.layer,
                expert_id: s.expert_id,
            });
        }
        for &expert_id in ids {
            let key = CacheKey { layer, expert_id };
            if q.pending.insert(key) {
                q.reqs.push_back(PrefetchRequest { layer, expert_id });
            }
        }
        drop(q);
        self.inner.ready.notify_all();
    }

    /// An expert the worker has already delivered, counting the hit either way so
    /// the overlap the tier split exists for is readable from the log rather than
    /// assumed from the design.
    pub fn get_cached(&self, layer: usize, expert_id: usize) -> Option<Arc<CachedExpert>> {
        let found = {
            let mut c = self.inner.cache.lock().unwrap();
            c.get(&CacheKey { layer, expert_id })
        };
        match &found {
            Some(_) => self.inner.hits.fetch_add(1, Relaxed),
            None => self.inner.misses.fetch_add(1, Relaxed),
        };
        found
    }

    /// Whether this expert is already delivered or already being fetched, without
    /// touching the counters or the recency order. The pool asks before spending
    /// a second read on bytes this tier is already moving for it.
    pub fn covers(&self, layer: usize, expert_id: usize) -> bool {
        let key = CacheKey { layer, expert_id };
        self.inner.cache.lock().unwrap().holds(&key)
            || self.inner.queue.lock().unwrap().pending.contains(&key)
    }

    /// Wait out a read the worker already owns for this expert, so the caller
    /// never pays for the same disk traffic twice. A read that is queued but not
    /// started is pulled to the front first: making the token wait behind experts
    /// it does not need yet is the one way a prefetch thread can add latency
    /// instead of hiding it. `None` with nothing pending returns immediately;
    /// that is the ordinary path, not a timeout.
    pub fn wait_cached(
        &self,
        layer: usize,
        expert_id: usize,
        wait: Duration,
    ) -> Option<Arc<CachedExpert>> {
        let key = CacheKey { layer, expert_id };
        {
            let mut q = self.inner.queue.lock().unwrap();
            if !q.pending.contains(&key) {
                return None;
            }
            if q.in_flight != Some(key) {
                if let Some(at) = q.reqs.iter().position(|r| CacheKey { layer: r.layer, expert_id: r.expert_id } == key)
                {
                    if let Some(req) = q.reqs.remove(at) {
                        q.reqs.push_front(req);
                        self.inner.ready.notify_one();
                    }
                }
            }
            let (q, _) = self
                .inner
                .ready
                .wait_timeout_while(q, wait, |q| q.pending.contains(&key))
                .ok()?;
            drop(q);
        }
        let mut c = self.inner.cache.lock().unwrap();
        c.get(&key)
    }

    /// (cache hits, cache misses, experts loaded, MiB held).
    pub fn stats(&self) -> (u64, u64, u64, usize) {
        let held = self.inner.cache.lock().unwrap().byte_size();
        (
            self.inner.hits.load(Relaxed),
            self.inner.misses.load(Relaxed),
            self.inner.loaded.load(Relaxed),
            held / (1024 * 1024),
        )
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        {
            let mut q = self.inner.queue.lock().unwrap();
            q.stop = true;
            q.reqs.clear();
            q.pending.clear();
        }
        self.inner.ready.notify_all();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Load a single expert's gate_up and down projections from the shards.
/// Mirrors the loading logic in `gather_experts` but for one expert at a time.
fn load_single_expert(
    sh: &Shards,
    cfg: &TextConfig,
    layer: usize,
    expert_id: usize,
    group: usize,
    threads: usize,
) -> crate::config::Result<CachedExpert> {
    let (h, i) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let p = format!("{}.layers.{layer}.mlp.experts", cfg.prefix);

    // Fast path: FQ4 sidecar cache (memcpy, no quantisation)
    if let Some(sc) = sh.fq4() {
        if let (Some(gi), Some(di)) = (
            sc.item(&format!("blk.{layer}.ffn.gate_up_exps")),
            sc.item(&format!("blk.{layer}.ffn.down_exps")),
        ) {
            if gi.group == group && di.group == group && gi.cols == h && di.cols == i {
                let gate_up = sc.rows(&gi, expert_id * 2 * i, (expert_id + 1) * 2 * i)?;
                let down = sc.rows(&di, expert_id * h, (expert_id + 1) * h)?;
                return Ok(CachedExpert { gate_up, down });
            }
        }
    }

    // Packed expert format: one expert per shard read
    if sh.contains(&format!("{p}.0.merged_weight")) {
        let (gu, dn) = crate::model::packed_expert(sh, &cfg.prefix, layer, expert_id, h, i, group)?;
        return Ok(CachedExpert { gate_up: gu, down: dn });
    }

    // Slow path: read bf16 slices and quantise (handles fused, separate, and Edge0 formats)
    let (gu_slice, dn_slice) = crate::model::load_expert_pair(sh, &cfg.prefix, layer, expert_id, h, i)?;
    let gate_up = Q4::quantize(&gu_slice, 2 * i, h, group, threads);
    let down = Q4::quantize(&dn_slice, h, i, group, threads);
    Ok(CachedExpert { gate_up, down })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_expert(size: usize) -> Arc<CachedExpert> {
        Arc::new(CachedExpert {
            gate_up: Q4 {
                rows: size,
                cols: 64,
                group: 64,
                data: crate::quant::Store::Own(vec![0u8; size]),
                scale: vec![1.0; size],
                signed: false,
            },
            down: Q4 {
                rows: size,
                cols: 64,
                group: 64,
                data: crate::quant::Store::Own(vec![0u8; size]),
                scale: vec![1.0; size],
                signed: false,
            },
        })
    }

    #[test]
    fn test_lru_cache() {
        // mk_expert(n) costs 10*n bytes (data + scale for gate_up and down)
        let mut cache = LruCache::new(51_000);

        let key1 = CacheKey { layer: 0, expert_id: 1 };
        let key2 = CacheKey { layer: 0, expert_id: 2 };
        let key3 = CacheKey { layer: 0, expert_id: 3 };

        // 1000 bytes each
        cache.insert(key1, mk_expert(100));
        assert!(cache.contains(&key1));

        cache.insert(key2, mk_expert(100));
        assert!(cache.contains(&key2));

        // Access key1 to make it most recently used
        assert!(cache.get(&key1).is_some());

        // 49500 bytes: total would be 51500 > 51000, so key2 (LRU) is evicted
        cache.insert(key3, mk_expert(4950));
        assert!(cache.contains(&key3));
        assert!(cache.contains(&key1));
        assert!(!cache.contains(&key2));
    }

    #[test]
    fn test_cache_key_uniqueness() {
        let k1 = CacheKey { layer: 0, expert_id: 5 };
        let k2 = CacheKey { layer: 1, expert_id: 5 };
        assert_ne!(k1, k2);
    }

    /// A hit hands back the same allocation, not a copy of the weights.
    #[test]
    fn test_cache_hit_shares_the_expert() {
        let mut cache = LruCache::new(1 << 20);
        let key = CacheKey { layer: 3, expert_id: 7 };
        cache.insert(key, mk_expert(64));
        let a = cache.get(&key).unwrap();
        let b = cache.get(&key).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.byte_size(), 64 * 10);
    }
}
