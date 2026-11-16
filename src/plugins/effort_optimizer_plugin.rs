//! Effort optimizer plugin: classifies prompt difficulty and suggests effort level.
//!
//! Implements the "Effort Optimizer" mod: avoids over-effort on simple tasks by
//! classifying prompts into effort levels (low/medium/high/xhigh/max).
//!
//! Effort levels map to different generation parameters:
//! - Low: fast, minimal reasoning, lower quality threshold
//! - Medium: balanced speed/quality
//! - High: thorough reasoning, higher quality
//! - XHigh: extensive analysis, multiple passes
//! - Max: maximum effort, no shortcuts

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::{Arc, RwLock};

/// Effort level classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl EffortLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            EffortLevel::Low => "low",
            EffortLevel::Medium => "medium",
            EffortLevel::High => "high",
            EffortLevel::XHigh => "xhigh",
            EffortLevel::Max => "max",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "low" => Some(EffortLevel::Low),
            "medium" | "med" => Some(EffortLevel::Medium),
            "high" => Some(EffortLevel::High),
            "xhigh" | "x-high" => Some(EffortLevel::XHigh),
            "max" => Some(EffortLevel::Max),
            _ => None,
        }
    }

    /// Suggested max_tokens for this effort level.
    pub fn suggested_max_tokens(&self) -> u32 {
        match self {
            EffortLevel::Low => 512,
            EffortLevel::Medium => 2048,
            EffortLevel::High => 4096,
            EffortLevel::XHigh => 8192,
            EffortLevel::Max => 16384,
        }
    }

    /// Suggested temperature for this effort level.
    pub fn suggested_temperature(&self) -> f32 {
        match self {
            EffortLevel::Low => 0.3,
            EffortLevel::Medium => 0.5,
            EffortLevel::High => 0.7,
            EffortLevel::XHigh => 0.7,
            EffortLevel::Max => 0.8,
        }
    }
}

impl std::fmt::Display for EffortLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Effort classification result.
#[derive(Clone, Debug)]
pub struct EffortClassification {
    pub level: EffortLevel,
    pub confidence: f32,
    pub reason: String,
}

/// Effort optimizer configuration.
#[derive(Clone, Debug)]
pub struct EffortOptimizerConfig {
    /// Whether to auto-apply effort level or just suggest.
    pub auto_apply: bool,
    /// Default effort level when classification is uncertain.
    pub default_level: EffortLevel,
    /// Confidence threshold below which we use the default level.
    pub confidence_threshold: f32,
}

impl Default for EffortOptimizerConfig {
    fn default() -> Self {
        Self {
            auto_apply: true,
            default_level: EffortLevel::Medium,
            confidence_threshold: 0.6,
        }
    }
}

/// Effort optimizer service.
pub struct EffortOptimizerService {
    config: RwLock<EffortOptimizerConfig>,
    classification_count: RwLock<u64>,
    last_classifications: RwLock<Vec<(String, EffortClassification)>>,
}

impl EffortOptimizerService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(EffortOptimizerConfig::default()),
            classification_count: RwLock::new(0),
            last_classifications: RwLock::new(Vec::new()),
        }
    }

    pub fn config(&self) -> EffortOptimizerConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: EffortOptimizerConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Classify a prompt and determine the appropriate effort level.
    pub fn classify(&self, prompt: &str) -> EffortClassification {
        let config = self.config();
        let prompt_lower = prompt.to_lowercase();

        let mut score: f32 = 0.0;
        let mut reasons = Vec::new();

        let low_keywords = [
            "summarize", "list", "format", "convert", "translate",
            "simple", "quick", "basic", "find", "search", "lookup",
            "check", "verify", "count", "extract",
        ];
        for kw in &low_keywords {
            if prompt_lower.contains(kw) {
                score -= 0.3;
                reasons.push(format!("contains '{}'", kw));
                break;
            }
        }

        let high_keywords = [
            "analyze", "design", "architect", "optimize", "debug",
            "complex", "difficult", "research", "investigate",
            "implement", "build", "create", "develop", "refactor",
        ];
        for kw in &high_keywords {
            if prompt_lower.contains(kw) {
                score += 0.3;
                reasons.push(format!("contains '{}'", kw));
                break;
            }
        }

        let xhigh_keywords = [
            "comprehensive", "thorough", "exhaustive", "detailed analysis",
            "multi-step", "complex reasoning", "prove", "formal verification",
        ];
        for kw in &xhigh_keywords {
            if prompt_lower.contains(kw) {
                score += 0.5;
                reasons.push(format!("contains '{}'", kw));
                break;
            }
        }

        if prompt.len() > 500 {
            score += 0.2;
            reasons.push("long prompt (>500 chars)".to_string());
        }
        if prompt.len() > 2000 {
            score += 0.3;
            reasons.push("very long prompt (>2000 chars)".to_string());
        }

        let code_indicators = ["```", "fn ", "func ", "def ", "class ", "struct ", "impl "];
        let has_code = code_indicators.iter().any(|&ind| prompt.contains(ind));
        if has_code {
            score += 0.2;
            reasons.push("contains code".to_string());
        }

        let confidence = if reasons.is_empty() {
            0.5
        } else {
            (0.5 + reasons.len() as f32 * 0.1).min(1.0)
        };

        let level = if confidence < config.confidence_threshold {
            config.default_level.clone()
        } else if score < -0.2 {
            EffortLevel::Low
        } else if score < 0.2 {
            EffortLevel::Medium
        } else if score < 0.5 {
            EffortLevel::High
        } else if score < 0.8 {
            EffortLevel::XHigh
        } else {
            EffortLevel::Max
        };

        let reason = if reasons.is_empty() {
            "default classification".to_string()
        } else {
            reasons.join(", ")
        };

        let classification = EffortClassification {
            level,
            confidence,
            reason,
        };

        *self.classification_count.write().unwrap() += 1;
        self.last_classifications.write().unwrap().push((
            prompt.chars().take(100).collect(),
            classification.clone(),
        ));

        classification
    }

    /// Get the effective effort level for a prompt (considers auto_apply setting).
    pub fn effective_level(&self, prompt: &str) -> EffortLevel {
        let config = self.config();
        let classification = self.classify(prompt);

        if config.auto_apply {
            classification.level
        } else {
            config.default_level
        }
    }

    /// Get classification statistics.
    pub fn stats(&self) -> (u64, Vec<(String, EffortClassification)>) {
        let count = *self.classification_count.read().unwrap();
        let recent = self.last_classifications.read().unwrap().clone();
        (count, recent)
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.classification_count.write().unwrap() = 0;
        self.last_classifications.write().unwrap().clear();
    }
}

impl Default for EffortOptimizerService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for EffortOptimizerService {
    fn name(&self) -> &'static str {
        "effort-optimizer"
    }
}

/// Effort optimizer plugin.
pub struct EffortOptimizerPlugin {
    config: EffortOptimizerConfig,
}

impl EffortOptimizerPlugin {
    pub fn new(config: EffortOptimizerConfig) -> Self {
        Self { config }
    }
}

impl Default for EffortOptimizerPlugin {
    fn default() -> Self {
        Self::new(EffortOptimizerConfig::default() )
    }
}

impl Plugin for EffortOptimizerPlugin {
    fn name(&self) -> &'static str {
        "effort-optimizer"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = Arc::new(EffortOptimizerService::new());
        service.set_config(self.config.clone());

        ctx.register_service("effort-optimizer", EffortOptimizerServiceHandle(service.clone()))?;

        let svc = service.clone();
        ctx.on_intercept("prompt/submit", move |event| {
            if let Some(prompt) = event.payload::<String>() {
                let classification = svc.classify(prompt);
                event
                    .with_metadata("effort_level", classification.level.to_string())
                    .with_metadata("effort_confidence", format!("{:.2}", classification.confidence))
                    .with_metadata("effort_reason", classification.reason)
            } else {
                event
            }
        })?;

        ctx.effect(|| {})?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("effort-optimizer/unloading", ()));
        Ok(())
    }
}

struct EffortOptimizerServiceHandle(Arc <EffortOptimizerService>);

impl EffortOptimizerServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &EffortOptimizerService {
        &self.0
    }
}

impl Service for EffortOptimizerServiceHandle {
    fn name(&self) -> &'static str {
        "effort-optimizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effort_level_display() {
        assert_eq!(EffortLevel::Low.as_str(), "low");
        assert_eq!(EffortLevel::Medium.as_str(), "medium");
        assert_eq!(EffortLevel::High.as_str(), "high");
    }

    #[test]
    fn test_classify_simple_task() {
        let svc = EffortOptimizerService::new();
        let result = svc.classify("summarize this document");
        assert_eq!(result.level, EffortLevel::Low);
    }

    #[test]
    fn test_classify_complex_task() {
        let svc = EffortOptimizerService::new();
        let result = svc.classify("analyze and optimize this complex algorithm");
        assert!(result.level == EffortLevel::High || result.level == EffortLevel::XHigh);
    }

    #[test]
    fn test_classify_code_task() {
        let svc = EffortOptimizerService::new();
        let result = svc.classify("implement this function:\n```rust\nfn foo() {}\n```");
        // "implement" is a high keyword, code indicator adds more, so expect High or above
        assert!(result.level == EffortLevel::High || result.level == EffortLevel::XHigh);
    }

    #[test]
    fn test_effective_level() {
        let svc = EffortOptimizerService::new();
        svc.set_config(EffortOptimizerConfig {
            auto_apply: true,
            ..Default::default()
        });
        let level = svc.effective_level("quick lookup");
        assert_eq!(level, EffortLevel::Low);
    }

    #[test]
    fn test_stats() {
        let svc = EffortOptimizerService::new();
        svc.classify("task 1");
        svc.classify("task 2");
        let (count, recent) = svc.stats();
        assert_eq!(count, 2);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_suggested_params() {
        assert_eq!(EffortLevel::Low.suggested_max_tokens(), 512);
        assert_eq!(EffortLevel::Max.suggested_max_tokens(), 16384);
        assert!(EffortLevel::Low.suggested_temperature() < EffortLevel::Max.suggested_temperature());
    }
}
