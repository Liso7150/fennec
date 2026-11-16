//! Input router plugin: intercepts large inputs and routes through cheap model for summarization.
//!
//! Implements the "Input Router" mod: big files and connector results are read by
//! a cheaper model first; the main model gets short notes + path to full content.
//!
//! Expected savings: ~74k tokens → 5.9k tokens (92% reduction on large files).

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::{Arc, RwLock};

/// Input router configuration.
#[derive(Clone, Debug)]
pub struct InputRouterConfig {
    /// Line threshold: files with fewer lines pass through unmodified.
    pub edit_escape_lines: usize,
    /// Size threshold: inputs smaller than this (bytes) pass through.
    pub size_threshold: usize,
    /// Coverage threshold: fraction of input that must be covered by summary.
    pub coverage_threshold: f32,
    /// Maximum summary length in characters.
    pub max_summary_chars: usize,
    /// Whether to include file path in summary.
    pub include_path: bool,
    /// File patterns to bypass (e.g., config files, instructions).
    pub bypass_patterns: Vec<String>,
}

impl Default for InputRouterConfig {
    fn default() -> Self {
        Self {
            edit_escape_lines: 100,
            size_threshold: 4096,
            coverage_threshold: 0.5,
            max_summary_chars: 2000,
            include_path: true,
            bypass_patterns: vec![
                ".env".to_string(),
                "config".to_string(),
                "instructions".to_string(),
                "readme".to_string(),
            ],
        }
    }
}

/// Summary of a large input.
#[derive(Clone, Debug)]
pub struct InputSummary {
    /// Original file path or source identifier.
    pub source: String,
    /// Summary text (cheap model output).
    pub summary: String,
    /// Number of lines in original input.
    pub original_lines: usize,
    /// Number of characters in original input.
    pub original_chars: usize,
    /// Estimated token savings (original - summary).
    pub estimated_savings: usize,
    /// Whether the full content is still accessible via path.
    pub full_content_available: bool,
}

/// Input router service.
pub struct InputRouterService {
    config: RwLock<InputRouterConfig>,
    total_routed: RwLock<u64>,
    total_bypassed: RwLock<u64>,
    total_savings: RwLock<u64>,
    recent_summaries: RwLock<Vec<InputSummary>>,
}

impl InputRouterService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(InputRouterConfig::default()),
            total_routed: RwLock::new(0),
            total_bypassed: RwLock::new(0),
            total_savings: RwLock::new(0),
            recent_summaries: RwLock::new(Vec::new()),
        }
    }

    pub fn config(&self) -> InputRouterConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: InputRouterConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Check if an input should be bypassed (passed through unmodified).
    pub fn should_bypass(&self, source: &str, content: &str) -> bool {
        let config = self.config();

        let line_count = content.lines().count();
        if line_count < config.edit_escape_lines {
            return true;
        }

        if content.len() < config.size_threshold {
            return true;
        }

        let source_lower = source.to_lowercase();
        for pattern in &config.bypass_patterns {
            if source_lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        false
    }

    /// Create a summary of large input content.
    /// In production, this would call a cheap model; here we use heuristic extraction.
    pub fn summarize(&self, source: &str, content: &str) -> InputSummary {
        let config = self.config();
        let lines: Vec<&str> = content.lines().collect();
        let original_lines = lines.len();
        let original_chars = content.len();

        let mut summary_parts = Vec::new();

        if config.include_path {
            summary_parts.push(format!("Source: {}", source));
        }

        summary_parts.push(format!("File: {} lines, {} bytes", original_lines, original_chars));

        let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
        let sample_count = (non_empty.len() as f32 * config.coverage_threshold).min(20.0) as usize;

        if sample_count > 0 && sample_count < non_empty.len() {
            summary_parts.push("\nKey excerpts:".to_string());
            let step = non_empty.len() / sample_count.max(1);
            for i in (0..non_empty.len()).step_by(step.max(1)).take(sample_count) {
                let line = non_empty[i].trim();
                let truncated = if line.len() > 100 {
                    format!("{}...", &line[..100])
                } else {
                    line.to_string()
                };
                summary_parts.push(format!("  L{}: {}", i + 1, truncated));
            }
        }

        let summary = summary_parts.join("\n");
        let summary = if summary.len() > config.max_summary_chars {
            summary[..config.max_summary_chars].to_string()
        } else {
            summary
        };

        let estimated_savings = original_chars.saturating_sub(summary.len());

        *self.total_routed.write().unwrap() += 1;
        *self.total_savings.write().unwrap() += estimated_savings as u64;

        let input_summary = InputSummary {
            source: source.to_string(),
            summary: summary.clone(),
            original_lines,
            original_chars,
            estimated_savings,
            full_content_available: true,
        };

        self.recent_summaries.write().unwrap().push(input_summary.clone());

        input_summary
    }

    /// Route an input: either bypass or summarize.
    pub fn route(&self, source: &str, content: &str) -> RouteResult {
        if self.should_bypass(source, content) {
            *self.total_bypassed.write().unwrap() += 1;
            RouteResult::Bypassed(content.to_string())
        } else {
            let summary = self.summarize(source, content);
            RouteResult::Summarized(summary)
        }
    }

    /// Get routing statistics.
    pub fn stats(&self) -> RoutingStats {
        RoutingStats {
            total_routed: *self.total_routed.read().unwrap(),
            total_bypassed: *self.total_bypassed.read().unwrap(),
            total_savings_chars: *self.total_savings.read().unwrap(),
        }
    }

    /// Get recent summaries.
    pub fn recent_summaries(&self, limit: usize) -> Vec<InputSummary> {
        let summaries = self.recent_summaries.read().unwrap();
        summaries.iter().rev().take(limit).cloned().collect()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.total_routed.write().unwrap() = 0;
        *self.total_bypassed.write().unwrap() = 0;
        *self.total_savings.write().unwrap() = 0;
        self.recent_summaries.write().unwrap().clear();
    }
}

impl Default for InputRouterService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for InputRouterService {
    fn name(&self) -> &'static str {
        "input-router"
    }
}

/// Result of routing an input.
#[derive(Clone, Debug)]
pub enum RouteResult {
    /// Input was small enough to bypass summarization.
    Bypassed(String),
    /// Input was summarized.
    Summarized(InputSummary),
}

/// Routing statistics.
#[derive(Clone, Debug)]
pub struct RoutingStats {
    pub total_routed: u64,
    pub total_bypassed: u64,
    pub total_savings_chars: u64,
}

/// Input router plugin.
pub struct InputRouterPlugin {
    config: InputRouterConfig,
}

impl InputRouterPlugin {
    pub fn new(config: InputRouterConfig) -> Self {
        Self { config }
    }
}

impl Default for InputRouterPlugin {
    fn default() -> Self {
        Self::new(InputRouterConfig::default())
    }
}

impl Plugin for InputRouterPlugin {
    fn name(&self) -> &'static str {
        "input-router"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = Arc::new(InputRouterService::new());
        service.set_config(self.config.clone());

        ctx.register_service("input-router", InputRouterServiceHandle(service.clone()))?;

        let svc = service.clone();
        ctx.on_intercept("file/read", move |event| {
            if let Some(payload) = event.payload::<FileReadPayload>() {
                let path = payload.path.clone();
                let content = payload.content.clone();
                let result = svc.route(&path, &content);
                match result {
                    RouteResult::Bypassed(_) => event,
                    RouteResult::Summarized(summary) => {
                        let savings = summary.estimated_savings;
                        event
                            .with_payload(summary.summary.clone())
                            .with_metadata("summarized", "true")
                            .with_metadata("original_source", &path)
                            .with_metadata("savings", format!("{}", savings))
                    }
                }
            } else {
                event
            }
        })?;

        ctx.effect(|| {})?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("input-router/unloading", ()));
        Ok(())
    }
}

/// Payload for file read events.
#[derive(Clone, Debug)]
pub struct FileReadPayload {
    pub path: String,
    pub content: String,
}

struct InputRouterServiceHandle(Arc<InputRouterService>);

impl InputRouterServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &InputRouterService {
        &self.0
    }
}

impl Service for InputRouterServiceHandle {
    fn name(&self) -> &'static str {
        "input-router"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_bypass_small_file() {
        let svc = InputRouterService::new();
        let content = "line 1\nline 2\nline 3";
        assert!(svc.should_bypass("test.txt", content));
    }

    fn large_content() -> String {
        (0..200).map(|i| format!("line {} with enough padding to exceed the size threshold easily\n", i)).collect()
    }

    #[test]
    fn test_should_not_bypass_large_file() {
        let svc = InputRouterService::new();
        let content = large_content();
        assert!(!svc.should_bypass("large.rs", &content));
    }

    #[test]
    fn test_should_bypass_config_file() {
        let svc = InputRouterService::new();
        let content = large_content();
        assert!(svc.should_bypass(".env", &content));
    }

    #[test]
    fn test_summarize() {
        let svc = InputRouterService::new();
        let content = large_content();
        let summary = svc.summarize("test.txt", &content);
        assert!(summary.summary.len() < content.len());
        assert_eq!(summary.original_lines, 200);
        assert!(summary.estimated_savings > 0);
    }

    #[test]
    fn test_route_bypass() {
        let svc = InputRouterService::new();
        let result = svc.route("small.txt", "hello");
        assert!(matches!(result, RouteResult::Bypassed(_)));
    }

    #[test]
    fn test_route_summarize() {
        let svc = InputRouterService::new();
        let content = large_content();
        let result = svc.route("large.txt", &content);
        assert!(matches!(result, RouteResult::Summarized(_)));
    }

    #[test]
    fn test_stats() {
        let svc = InputRouterService::new();
        svc.route("small.txt", "hello");
        let content = large_content();
        svc.route("large.txt", &content);

        let stats = svc.stats();
        assert_eq!(stats.total_bypassed, 1);
        assert_eq!(stats.total_routed, 1);
        assert!(stats.total_savings_chars > 0);
    }
}
