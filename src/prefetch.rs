//! Background prefetcher for cold experts on SSD.
//!
//! While layer N computes, the prefetcher loads layer N+1's streamed experts
//! from disk into an LRU cache. This overlaps disk I/O with compute, reducing
//! the effective latency of the disk tier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Condvar};
use std::thread;

use crate::config::TextConfig;
use crate::quant::Q4;
use crate::safetensors::Shards;

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
#[derive(Clone, Debug)]
pub struct PrefetchRequest {
    pub expert_id: usize,
    pub layer: usize,
    pub priority: u32,
}

/// LRU cache for prefetched experts.
pub struct LruCache {
    data: HashMap<CacheKey, CachedExpert>,
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

    pub fn insert(&mut self, key: CacheKey, expert: CachedExpert) {
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
        self.data.insert(key, expert);
        self.current_bytes += size;
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&CachedExpert> {
        if self.data.contains_key(key) {
            self.order.retain(|k| k != key);
            self.order.insert(0, *key);
            self.data.get(key)
        } else {
            None
        }
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
    let gu_name = format!("{p}.gate_up_proj");
    let dn_name = format!("{p}.down_proj");

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

    // Slow path: read bf16 slices and quantise
    let gu_slice = sh.slice_f32(&gu_name, expert_id * 2 * i * h, (expert_id + 1) * 2 * i * h)?;
    let gate_up = Q4::quantize(&gu_slice, 2 * i, h, group, threads);
    let dn_slice = sh.slice_f32(&dn_name, expert_id * h * i, (expert_id + 1) * h * i)?;
    let down = Q4::quantize(&dn_slice, h, i, group, threads);
    Ok(CachedExpert { gate_up, down })
}

/// Background prefetcher that loads experts from disk into cache.
#[allow(dead_code)]
pub struct Prefetcher {
    queue: Arc<(Mutex<Vec<PrefetchRequest>>, Condvar)>,
    cache: Arc<Mutex<LruCache>>,
    shards: Arc<Shards>,
    config: Arc<TextConfig>,
    group: usize,
    threads: usize,
    _handle: Option<thread::JoinHandle<()>>,
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
        let queue = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let cache = Arc::new(Mutex::new(LruCache::new(cache_bytes)));

        let queue_clone = queue.clone();
        let cache_clone = cache.clone();
        let shards_clone = shards.clone();
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            Self::prefetch_loop(queue_clone, cache_clone, shards_clone, config_clone, group, threads);
        });

        Self {
            queue,
            cache,
            shards,
            config,
            group,
            threads,
            _handle: Some(handle),
        }
    }

    fn prefetch_loop(
        queue: Arc<(Mutex<Vec<PrefetchRequest>>, Condvar)>,
        cache: Arc<Mutex<LruCache>>,
        shards: Arc<Shards>,
        config: Arc<TextConfig>,
        group: usize,
        threads: usize,
    ) {
        loop {
            let (lock, cvar) = &*queue;
            let mut queue_guard = lock.lock().unwrap();

            while queue_guard.is_empty() {
                queue_guard = cvar.wait(queue_guard).unwrap();
            }

            queue_guard.sort_by(|a, b| b.priority.cmp(&a.priority));
            let req = queue_guard.remove(0);
            drop(queue_guard);

            let key = CacheKey {
                layer: req.layer,
                expert_id: req.expert_id,
            };

            {
                let c = cache.lock().unwrap();
                if c.contains(&key) {
                    continue;
                }
            }

            match load_single_expert(&shards, &config, req.layer, req.expert_id, group, threads) {
                Ok(expert) => {
                    let mut c = cache.lock().unwrap();
                    c.insert(key, expert);
                }
                Err(e) => {
                    eprintln!("[prefetch] failed to load expert {}/{}: {}", req.layer, req.expert_id, e);
                }
            }
        }
    }

    /// Queue a prefetch request for the next layer's streamed experts.
    pub fn prefetch_next_layer(&self, layer: usize, streamed_experts: &[usize]) {
        let (lock, cvar) = &*self.queue;
        let mut queue = lock.lock().unwrap();

        for &expert_id in streamed_experts {
            queue.push(PrefetchRequest {
                expert_id,
                layer,
                priority: 100 - layer as u32,
            });
        }

        cvar.notify_one();
    }

    /// Check if an expert is in the cache.
    pub fn is_cached(&self, layer: usize, expert_id: usize) -> bool {
        let key = CacheKey { layer, expert_id };
        let c = self.cache.lock().unwrap();
        c.contains(&key)
    }

    /// Get a cached expert (returns None if not present).
    pub fn get_cached(&self, layer: usize, expert_id: usize) -> Option<CachedExpert> {
        let key = CacheKey { layer, expert_id };
        let mut c = self.cache.lock().unwrap();
        c.get(&key).cloned()
    }

    /// Get cache statistics: (entry_count, byte_size).
    pub fn cache_stats(&self) -> (usize, usize) {
        let c = self.cache.lock().unwrap();
        (c.entry_count(), c.byte_size())
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.queue;
        let mut queue = lock.lock().unwrap();
        queue.clear();
        cvar.notify_one();
        drop(queue);

        if let Some(handle) = self._handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache() {
        // mk_expert(n) costs 10*n bytes (data + scale for gate_up and down)
        let mut cache = LruCache::new(51_000);

        let mk_expert = |size: usize| CachedExpert {
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
        };

        let key1 = CacheKey { layer: 0, expert_id: 1 };
        let key2 = CacheKey { layer: 0, expert_id: 2 };
        let key3 = CacheKey { layer: 0, expert_id: 3 };

        // 1000 bytes each
        cache.insert(key1, mk_expert(100));
        assert!(cache.contains(&key1));

        cache.insert(key2, mk_expert(100));
        assert!(cache.contains(&key2));

        // Access key1 to make it most recently used
        cache.get(&key1);

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
}
