//! Background prefetcher for cold experts on SSD.
//!
//! While layer N computes, the prefetcher loads layer N+1's streamed experts
//! from disk into an LRU cache. This overlaps disk I/O with compute, reducing
//! the effective latency of the disk tier.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Condvar};
use std::thread;

/// A request to prefetch an expert from disk.
#[derive(Clone, Debug)]
pub struct PrefetchRequest {
    pub expert_id: usize,
    pub layer: usize,
    pub priority: u32,  // higher = more urgent
}

/// LRU cache for prefetched experts.
pub struct LruCache {
    data: HashMap<usize, Vec<u8>>,
    order: Vec<usize>,  // front = most recently used
    max_size: usize,
    current_size: usize,
}

impl LruCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: HashMap::new(),
            order: Vec::new(),
            max_size,
            current_size: 0,
        }
    }

    pub fn insert(&mut self, id: usize, data: Vec<u8>) {
        let size = data.len();
        
        // Evict if over budget
        while self.current_size + size > self.max_size && !self.order.is_empty() {
            let evict_id = self.order.pop().unwrap();
            if let Some(evicted) = self.data.remove(&evict_id) {
                self.current_size -= evicted.len();
            }
        }
        
        // Remove from old position if updating
        if let Some(old) = self.data.remove(&id) {
            self.current_size -= old.len();
            self.order.retain(|&x| x != id);
        }
        
        // Insert at front (most recent)
        self.order.insert(0, id);
        self.data.insert(id, data);
        self.current_size += size;
    }

    pub fn get(&mut self, id: usize) -> Option<&Vec<u8>> {
        if let Some(data) = self.data.get(&id) {
            // Move to front (mark as recently used)
            self.order.retain(|&x| x != id);
            self.order.insert(0, id);
            Some(data)
        } else {
            None
        }
    }

    pub fn contains(&self, id: usize) -> bool {
        self.data.contains_key(&id)
    }

    pub fn size(&self) -> usize {
        self.current_size
    }
}

/// Background prefetcher that loads experts from disk into cache.
pub struct Prefetcher {
    queue: Arc<(Mutex<Vec<PrefetchRequest>>, Condvar)>,
    cache: Arc<Mutex<LruCache>>,
    _handle: Option<thread::JoinHandle<()>>,
}

impl Prefetcher {
    /// Start a new prefetcher with the given cache size (bytes).
    pub fn start(cache_size: usize) -> Self {
        let queue = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let cache = Arc::new(Mutex::new(LruCache::new(cache_size)));
        
        let queue_clone = queue.clone();
        let cache_clone = cache.clone();
        
        let handle = thread::spawn(move || {
            Self::prefetch_loop(queue_clone, cache_clone);
        });
        
        Self {
            queue,
            cache,
            _handle: Some(handle),
        }
    }

    fn prefetch_loop(
        queue: Arc<(Mutex<Vec<PrefetchRequest>>, Condvar)>,
        cache: Arc<Mutex<LruCache>>,
    ) {
        loop {
            let (lock, cvar) = &*queue;
            let mut queue_guard = lock.lock().unwrap();
            
            // Wait for a request
            while queue_guard.is_empty() {
                queue_guard = cvar.wait(queue_guard).unwrap();
            }
            
            // Pop highest priority request
            queue_guard.sort_by(|a, b| b.priority.cmp(&a.priority));
            let req = queue_guard.remove(0);
            drop(queue_guard);
            
            // Check if already cached
            {
                let c = cache.lock().unwrap();
                if c.contains(req.expert_id) {
                    continue;
                }
            }
            
            // TODO: Load expert from disk
            // For now, just insert a placeholder
            // In production, this would read from the shard file
            let dummy_data = vec![0u8; 1024];  // placeholder
            let mut c = cache.lock().unwrap();
            c.insert(req.expert_id, dummy_data);
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
                priority: 100 - layer as u32,  // earlier layers = higher priority
            });
        }
        
        cvar.notify_one();
    }

    /// Check if an expert is in the cache.
    pub fn is_cached(&self, expert_id: usize) -> bool {
        let c = self.cache.lock().unwrap();
        c.contains(expert_id)
    }

    /// Get cached expert data (blocks if not present).
    pub fn get_cached(&self, expert_id: usize) -> Option<Vec<u8>> {
        let mut c = self.cache.lock().unwrap();
        c.get(expert_id).cloned()
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let c = self.cache.lock().unwrap();
        (c.data.len(), c.current_size)
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Signal the thread to exit
        let (lock, cvar) = &*self.queue;
        let mut queue = lock.lock().unwrap();
        queue.clear();
        cvar.notify_one();
        drop(queue);
        
        // Wait for thread to finish
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
        let mut cache = LruCache::new(1000);
        
        cache.insert(1, vec![1, 2, 3]);
        assert!(cache.contains(1));
        assert_eq!(cache.size(), 3);
        
        cache.insert(2, vec![4, 5]);
        assert!(cache.contains(2));
        assert_eq!(cache.size(), 5);
        
        // Access 1 to make it recent
        cache.get(1);
        
        // Insert large item to trigger eviction
        cache.insert(3, vec![0; 996]);
        assert!(cache.contains(3));
        assert!(cache.contains(1));  // 1 was accessed recently
        assert!(!cache.contains(2));  // 2 was evicted (least recent)
    }

    #[test]
    fn test_prefetcher_basic() {
        let prefetcher = Prefetcher::start(10000);
        
        prefetcher.prefetch_next_layer(0, &[1, 2, 3]);
        thread::sleep(std::time::Duration::from_millis(100));
        
        // Check that experts were cached
        assert!(prefetcher.is_cached(1));
        assert!(prefetcher.is_cached(2));
        assert!(prefetcher.is_cached(3));
        
        let (count, size) = prefetcher.cache_stats();
        assert_eq!(count, 3);
        assert!(size > 0);
    }
}
