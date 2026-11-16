//! Cache clock plugin: tracks prompt cache TTL and provides keep-warm mechanism.
//!
//! Implements the "Cache Clock" mod: shows countdown until chat cache goes cold,
//! offers keep-warm ping to avoid full-price cache rebuilds.
//!
//! Cache TTL is typically 1 hour for most sessions, 5 minutes for some.

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cache TTL configuration.
#[derive(Clone, Debug)]
pub struct CacheTtlConfig {
    /// Default TTL in seconds (3600 = 1 hour).
    pub default_ttl_secs: u64,
    /// Short TTL in seconds (300 = 5 minutes).
    pub short_ttl_secs: u64,
    /// Keep-warm threshold: ping when this many seconds remain.
    pub keep_warm_threshold_secs: u64,
    /// Whether to automatically send keep-warm pings.
    pub auto_keep_warm: bool,
}

impl Default for CacheTtlConfig {
    fn default() -> Self {
        Self {
            default_ttl_secs: 3600,
            short_ttl_secs: 300,
            keep_warm_threshold_secs: 300,
            auto_keep_warm: true,
        }
    }
}

/// Cache state tracking.
struct CacheState {
    /// When the cache was last refreshed.
    last_refresh: Option<Instant>,
    /// Current TTL being used.
    current_ttl: Duration,
    /// Number of keep-warm pings sent.
    ping_count: u64,
}

/// Cache clock service: tracks cache TTL and provides keep-warm mechanism.
pub struct CacheClockService {
    config: RwLock<CacheTtlConfig>,
    state: RwLock<CacheState>,
    total_cache_hits: AtomicU64,
    total_cache_misses: AtomicU64,
}

impl CacheClockService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(CacheTtlConfig::default()),
            state: RwLock::new(CacheState {
                last_refresh: None,
                current_ttl: Duration::from_secs(3600),
                ping_count: 0,
            }),
            total_cache_hits: AtomicU64::new(0),
            total_cache_misses: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> CacheTtlConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: CacheTtlConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Mark the cache as refreshed (called after each successful request).
    pub fn mark_refreshed(&self) {
        let config = self.config();
        let mut state = self.state.write().unwrap();
        state.last_refresh = Some(Instant::now());
        state.current_ttl = Duration::from_secs(config.default_ttl_secs);
        self.total_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark a cache miss (cold start, full price).
    pub fn mark_miss(&self) {
        self.total_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Send a keep-warm ping (resets the cache timer).
    pub fn keep_warm_ping(&self) {
        let mut state = self.state.write().unwrap();
        state.last_refresh = Some(Instant::now());
        state.ping_count += 1;
    }

    /// Get seconds remaining until cache expires.
    pub fn seconds_remaining(&self) -> Option<u64> {
        let state = self.state.read().unwrap();
        state.last_refresh.map(|refreshed| {
            let elapsed = refreshed.elapsed().as_secs();
            let ttl = state.current_ttl.as_secs();
            ttl.saturating_sub(elapsed)
        })
    }

    /// Check if cache is about to expire (within keep-warm threshold).
    pub fn should_keep_warm(&self) -> bool {
        let config = self.config();
        if !config.auto_keep_warm {
            return false;
        }
        match self.seconds_remaining() {
            Some(remaining) => remaining <= config.keep_warm_threshold_secs,
            None => false,
        }
    }

    /// Check if cache has expired.
    pub fn is_expired(&self) -> bool {
        match self.seconds_remaining() {
            Some(0) => true,
            None => true,
            _ => false,
        }
    }

    /// Get cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.total_cache_hits.load(Ordering::Relaxed);
        let misses = self.total_cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get total cache hits.
    pub fn total_hits(&self) -> u64 {
        self.total_cache_hits.load(Ordering::Relaxed)
    }

    /// Get total cache misses.
    pub fn total_misses(&self) -> u64 {
        self.total_cache_misses.load(Ordering::Relaxed)
    }

    /// Get number of keep-warm pings sent.
    pub fn ping_count(&self) -> u64 {
        self.state.read().unwrap().ping_count
    }

    /// Reset all counters and state.
    pub fn reset(&self) {
        *self.state.write().unwrap() = CacheState {
            last_refresh: None,
            current_ttl: Duration::from_secs(3600),
            ping_count: 0,
        };
        self.total_cache_hits.store(0, Ordering::Relaxed);
        self.total_cache_misses.store(0, Ordering::Relaxed);
    }
}

impl Default for CacheClockService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for CacheClockService {
    fn name(&self) -> &'static str {
        "cache-clock"
    }
}

/// Cache clock plugin.
pub struct CacheClockPlugin {
    config: CacheTtlConfig,
}

impl CacheClockPlugin {
    pub fn new(config: CacheTtlConfig) -> Self {
        Self { config }
    }
}

impl Default for CacheClockPlugin {
    fn default() -> Self {
        Self::new(CacheTtlConfig::default())
    }
}

impl Plugin for CacheClockPlugin {
    fn name(&self) -> &'static str {
        "cache-clock"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = Arc::new(CacheClockService::new());
        service.set_config(self.config.clone());

        ctx.register_service("cache-clock", CacheClockServiceHandle(service.clone()))?;

        let svc = service.clone();
        ctx.on("request/complete", move |_event| {
            svc.mark_refreshed();
        })?;

        let svc2 = service.clone();
        ctx.on("request/cache_miss", move |_event| {
            svc2.mark_miss();
        })?;

        ctx.effect(|| {})?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("cache-clock/unloading", ()));
        Ok(())
    }
}

struct CacheClockServiceHandle(Arc<CacheClockService>);

impl CacheClockServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &CacheClockService {
        &self.0
    }
}

impl Service for CacheClockServiceHandle {
    fn name(&self) -> &'static str {
        "cache-clock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_clock_service() {
        let svc = CacheClockService::new();
        assert_eq!(svc.total_hits(), 0);
        assert_eq!(svc.total_misses(), 0);
        assert_eq!(svc.seconds_remaining(), None);
        assert!(svc.is_expired());
    }

    #[test]
    fn test_mark_refreshed() {
        let svc = CacheClockService::new();
        svc.mark_refreshed();
        assert_eq!(svc.total_hits(), 1);
        assert!(svc.seconds_remaining().is_some());
        assert!(!svc.is_expired());
    }

    #[test]
    fn test_hit_rate() {
        let svc = CacheClockService::new();
        assert_eq!(svc.hit_rate(), 0.0);

        svc.mark_refreshed();
        svc.mark_refreshed();
        svc.mark_miss();
        assert!((svc.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_keep_warm_ping() {
        let svc = CacheClockService::new();
        assert_eq!(svc.ping_count(), 0);
        svc.keep_warm_ping();
        assert_eq!(svc.ping_count(), 1);
    }

    #[test]
    fn test_reset() {
        let svc = CacheClockService::new();
        svc.mark_refreshed();
        svc.mark_miss();
        svc.keep_warm_ping();
        svc.reset();
        assert_eq!(svc.total_hits(), 0);
        assert_eq!(svc.total_misses(), 0);
        assert_eq!(svc.ping_count(), 0);
    }
}
