//! Memory bandwidth profiling and optimization for bandwidth-bound workloads.
//!
//! Modern LLM inference is often memory-bandwidth-bound rather than compute-bound,
//! especially for large models on consumer hardware. This module measures actual
//! memory bandwidth, identifies bottlenecks, and provides optimization hints.
//!
//! Key insight: speed depends more on memory bandwidth than FLOPS.
//! A $800 RTX 5060 Ti can match a $6,600 RTX 5090 for 763B models at 2-bit
//! because both are bandwidth-bound, not compute-bound.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct BandwidthProfile {
    pub read_gbs: f64,
    pub write_gbs: f64,
    pub total_gbs: f64,
    pub bytes_transferred: u64,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct BandwidthReport {
    pub profiles: Vec<(String, BandwidthProfile)>,
    pub bottleneck: Option<String>,
    pub is_bandwidth_bound: bool,
    pub optimization_hints: Vec<String>,
}

pub struct BandwidthProfiler {
    profiles: Vec<(String, BandwidthProfile)>,
}

impl BandwidthProfiler {
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Measure memory bandwidth for an operation
    pub fn measure<F, T>(&mut self, name: &str, bytes: u64, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        let total_gbs = if duration.as_secs_f64() > 0.0 {
            (bytes as f64) / duration.as_secs_f64() / 1e9
        } else {
            0.0
        };

        let profile = BandwidthProfile {
            read_gbs: total_gbs * 0.7, // Estimate 70% read, 30% write
            write_gbs: total_gbs * 0.3,
            total_gbs,
            bytes_transferred: bytes,
            duration,
        };

        self.profiles.push((name.to_string(), profile));
        result
    }

    /// Measure read-only bandwidth
    pub fn measure_read<F, T>(&mut self, name: &str, bytes: u64, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        let read_gbs = if duration.as_secs_f64() > 0.0 {
            (bytes as f64) / duration.as_secs_f64() / 1e9
        } else {
            0.0
        };

        let profile = BandwidthProfile {
            read_gbs,
            write_gbs: 0.0,
            total_gbs: read_gbs,
            bytes_transferred: bytes,
            duration,
        };

        self.profiles.push((name.to_string(), profile));
        result
    }

    /// Generate a report analyzing bandwidth bottlenecks
    pub fn report(&self) -> BandwidthReport {
        if self.profiles.is_empty() {
            return BandwidthReport {
                profiles: Vec::new(),
                bottleneck: None,
                is_bandwidth_bound: false,
                optimization_hints: Vec::new(),
            };
        }

        // Find the slowest operation (potential bottleneck)
        let slowest = self.profiles
            .iter()
            .max_by(|a, b| a.1.duration.cmp(&b.1.duration));

        let bottleneck = slowest.map(|(name, _)| name.clone());

        // Check if workload is bandwidth-bound
        // Heuristic: if average bandwidth < 50 GB/s, likely bandwidth-bound on consumer hardware
        let avg_bandwidth = self.profiles.iter()
            .map(|(_, p)| p.total_gbs)
            .sum::<f64>() / self.profiles.len() as f64;

        let is_bandwidth_bound = avg_bandwidth < 50.0;

        // Generate optimization hints
        let mut hints = Vec::new();

        if is_bandwidth_bound {
            hints.push("Workload is memory-bandwidth-bound".to_string());
            hints.push("Consider quantization (int8/int4) to reduce memory traffic".to_string());
            hints.push("Use batched operations to amortize weight reads".to_string());
            hints.push("Optimize data layout for cache locality".to_string());
            
            // Check for specific patterns
            for (name, profile) in &self.profiles {
                if profile.total_gbs < 20.0 && profile.bytes_transferred > 1_000_000 {
                    hints.push(format!(
                        "{}: Very low bandwidth ({:.2} GB/s) - consider data layout optimization",
                        name, profile.total_gbs
                    ));
                }
            }
        } else {
            hints.push("Workload is compute-bound or well-optimized for memory bandwidth".to_string());
        }

        BandwidthReport {
            profiles: self.profiles.clone(),
            bottleneck,
            is_bandwidth_bound,
            optimization_hints: hints,
        }
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        self.profiles.clear();
    }
}

impl Default for BandwidthProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate memory bandwidth for a matrix operation
pub fn estimate_matrix_bandwidth(rows: usize, cols: usize, element_size: usize) -> u64 {
    (rows * cols * element_size) as u64
}

/// Estimate memory bandwidth for a GEMV operation
pub fn estimate_gemv_bandwidth(rows: usize, cols: usize, element_size: usize) -> u64 {
    // Read matrix + read vector + write result
    let matrix_bytes = rows * cols * element_size;
    let vector_bytes = cols * element_size;
    let result_bytes = rows * element_size;
    (matrix_bytes + vector_bytes + result_bytes) as u64
}

/// Check if an operation is likely bandwidth-bound
pub fn is_bandwidth_bound(bytes: u64, duration: Duration, peak_bandwidth_gbs: f64) -> bool {
    let actual_bandwidth = (bytes as f64) / duration.as_secs_f64() / 1e9;
    // If actual bandwidth is < 50% of peak, likely bandwidth-bound
    actual_bandwidth < peak_bandwidth_gbs * 0.5
}

/// Suggest optimal quantization based on bandwidth analysis
pub fn suggest_quantization(current_bandwidth_gbs: f64, target_bandwidth_gbs: f64) -> Option<&'static str> {
    if current_bandwidth_gbs >= target_bandwidth_gbs {
        None // Already fast enough
    } else if current_bandwidth_gbs >= target_bandwidth_gbs * 0.5 {
        Some("int8") // 2x reduction
    } else if current_bandwidth_gbs >= target_bandwidth_gbs * 0.25 {
        Some("int4") // 4x reduction
    } else {
        Some("int2") // 8x reduction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_measures_operation() {
        let mut profiler = BandwidthProfiler::new();
        let data = vec![0u8; 1_000_000]; // 1 MB
        
        profiler.measure_read("test", data.len() as u64, || {
            let _sum: u64 = data.iter().map(|&x| x as u64).sum();
        });

        let report = profiler.report();
        assert_eq!(report.profiles.len(), 1);
        assert_eq!(report.profiles[0].0, "test");
    }

    #[test]
    fn estimate_gemv_bandwidth_correct() {
        let bytes = estimate_gemv_bandwidth(1024, 2048, 2);
        // Matrix: 1024 * 2048 * 2 = 4 MB
        // Vector: 2048 * 2 = 4 KB
        // Result: 1024 * 2 = 2 KB
        // Total: ~4 MB
        assert!(bytes > 4_000_000);
    }

    #[test]
    fn suggest_quantization_levels() {
        assert_eq!(suggest_quantization(100.0, 50.0), None);
        assert_eq!(suggest_quantization(40.0, 50.0), Some("int8"));
        assert_eq!(suggest_quantization(20.0, 50.0), Some("int4"));
        assert_eq!(suggest_quantization(10.0, 50.0), Some("int2"));
    }
}
