//! Model router plugin: routes requests to different models based on task difficulty.
//!
//! Implements the "Model Router" and "Input Router" mods: routes subagents and
//! expensive operations to cheaper models based on task classification.
//!
//! Model tiers:
//! - Cheap (e.g., Haiku): summarization, classification, simple tasks
//! - Normal (e.g., Sonnet): standard tasks
//! - Expensive (e.g., Opus): complex reasoning, hard tasks

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::{Arc, RwLock};

/// Model tier classification.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelTier {
    Cheap,
    Normal,
    Expensive,
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelTier::Cheap => write!(f, "cheap"),
            ModelTier::Normal => write!(f, "normal"),
            ModelTier::Expensive => write!(f, "expensive"),
        }
    }
}

/// Task difficulty classification.
#[derive(Clone, Debug)]
pub struct TaskDifficulty {
    /// Difficulty score (0.0 = easy, 1.0 = hard).
    pub score: f32,
    /// Confidence in the classification (0.0 to 1.0).
    pub confidence: f32,
    /// Suggested tier based on score.
    pub suggested_tier: ModelTier,
    /// Reason for the classification.
    pub reason: String,
}

impl TaskDifficulty {
    /// Classify a task based on its score.
    pub fn from_score(score: f32, confidence: f32, reason: String) -> Self {
        let suggested_tier = if score < 0.3 {
            ModelTier::Cheap
        } else if score < 0.7 {
            ModelTier::Normal
        } else {
            ModelTier::Expensive
        };
        Self {
            score,
            confidence,
            suggested_tier,
            reason,
        }
    }
}

/// Model router configuration.
#[derive(Clone, Debug)]
pub struct ModelRouterConfig {
    /// Model name for cheap tier.
    pub cheap_model: String,
    /// Model name for normal tier.
    pub normal_model: String,
    /// Model name for expensive tier.
    pub expensive_model: String,
    /// Score threshold for cheap tier (< threshold).
    pub cheap_threshold: f32,
    /// Score threshold for normal tier (< threshold).
    pub normal_threshold: f32,
    /// Whether to auto-route or just suggest.
    pub auto_route: bool,
}

impl Default for ModelRouterConfig {
    fn default() -> Self {
        Self {
            cheap_model: "haiku".to_string(),
            normal_model: "sonnet".to_string(),
            expensive_model: "opus".to_string(),
            cheap_threshold: 0.35,
            normal_threshold: 0.7,
            auto_route: true,
        }
    }
}

/// Model router service: classifies tasks and routes to appropriate models.
pub struct ModelRouterService {
    config: RwLock<ModelRouterConfig>,
    routing_log: RwLock<Vec<RoutingDecision>>,
    stats: RwLock<RoutingStats>,
}

/// A routing decision that was made.
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    pub task_description: String,
    pub difficulty: TaskDifficulty,
    pub routed_to: ModelTier,
    pub timestamp: u64,
}

/// Routing statistics.
#[derive(Clone, Debug, Default)]
pub struct RoutingStats {
    pub total_routed: u64,
    pub cheap_count: u64,
    pub normal_count: u64,
    pub expensive_count: u64,
}

impl ModelRouterService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(ModelRouterConfig::default()),
            routing_log: RwLock::new(Vec::new()),
            stats: RwLock::new(RoutingStats::default()),
        }
    }

    pub fn config(&self) -> ModelRouterConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: ModelRouterConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Classify a task and determine the appropriate model tier.
    pub fn classify(&self, task_description: &str) -> TaskDifficulty {
        let config = self.config();

        let score = self.estimate_difficulty(task_description);
        let confidence = self.estimate_confidence(task_description, score);
        let reason = format!("score={:.2}, confidence={:.2}", score, confidence);

        let suggested_tier = if score < config.cheap_threshold {
            ModelTier::Cheap
        } else if score < config.normal_threshold {
            ModelTier::Normal
        } else {
            ModelTier::Expensive
        };

        TaskDifficulty {
            score,
            confidence,
            suggested_tier,
            reason,
        }
    }

    /// Route a task to the appropriate model.
    pub fn route(&self, task_description: &str) -> (String, TaskDifficulty) {
        let difficulty = self.classify(task_description);
        let config = self.config();

        let model = match &difficulty.suggested_tier {
            ModelTier::Cheap => config.cheap_model.clone(),
            ModelTier::Normal => config.normal_model.clone(),
            ModelTier::Expensive => config.expensive_model.clone(),
        };

        let decision = RoutingDecision {
            task_description: task_description.to_string(),
            difficulty: difficulty.clone(),
            routed_to: difficulty.suggested_tier.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.routing_log.write().unwrap().push(decision);

        let mut stats = self.stats.write().unwrap();
        stats.total_routed += 1;
        match difficulty.suggested_tier {
            ModelTier::Cheap => stats.cheap_count += 1,
            ModelTier::Normal => stats.normal_count += 1,
            ModelTier::Expensive => stats.expensive_count += 1,
        }

        (model, difficulty)
    }

    /// Get routing statistics.
    pub fn stats(&self) -> RoutingStats {
        self.stats.read().unwrap().clone()
    }

    /// Get recent routing decisions.
    pub fn recent_decisions(&self, limit: usize) -> Vec<RoutingDecision> {
        let log = self.routing_log.read().unwrap();
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Estimate task difficulty (0.0 = easy, 1.0 = hard).
    /// This is a simple heuristic; a real implementation would use a classifier model.
    fn estimate_difficulty(&self, task: &str) -> f32 {
        let task_lower = task.to_lowercase();

        let mut score: f32 = 0.5;

        let easy_keywords = [
            "summarize", "summary", "classify", "simple", "basic", "list",
            "extract", "format", "convert", "translate", "count",
        ];
        for keyword in &easy_keywords {
            if task_lower.contains(keyword) {
                score -= 0.2;
            }
        }

        let hard_keywords = [
            "analyze", "reason", "complex", "difficult", "research",
            "implement", "design", "architecture", "optimize", "debug",
        ];
        for keyword in &hard_keywords {
            if task_lower.contains(keyword) {
                score += 0.2;
            }
        }

        if task.len() > 1000 {
            score += 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Estimate confidence in the difficulty classification.
    fn estimate_confidence(&self, task: &str, _score: f32) -> f32 {
        let mut confidence: f32 = 0.7;

        if task.len() < 50 {
            confidence -= 0.2;
        }
        if task.len() > 500 {
            confidence += 0.1;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Clear routing log and stats.
    pub fn reset(&self) {
        self.routing_log.write().unwrap().clear();
        *self.stats.write().unwrap() = RoutingStats::default();
    }
}

impl Default for ModelRouterService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for ModelRouterService {
    fn name(&self) -> &'static str {
        "model-router"
    }
}

/// Model router plugin.
pub struct ModelRouterPlugin {
    config: ModelRouterConfig,
}

impl ModelRouterPlugin {
    pub fn new(config: ModelRouterConfig) -> Self {
        Self { config }
    }
}

impl Default for ModelRouterPlugin {
    fn default() -> Self {
        Self::new(ModelRouterConfig::default())
    }
}

impl Plugin for ModelRouterPlugin {
    fn name(&self) -> &'static str {
        "model-router"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = Arc::new(ModelRouterService::new());
        service.set_config(self.config.clone());

        ctx.register_service("model-router", ModelRouterServiceHandle(service.clone()))?;

        let svc = service.clone();
        ctx.on_intercept("request/route", move |event| {
            if let Some(task_desc) = event.payload::<String>() {
                let (model, difficulty) = svc.route(task_desc);
                event
                    .with_payload(model)
                    .with_metadata("difficulty_score", format!("{:.2}", difficulty.score))
                    .with_metadata("difficulty_tier", difficulty.suggested_tier.to_string())
            } else {
                event
            }
        })?;

        ctx.effect(|| {})?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("model-router/unloading", ()));
        Ok(())
    }
}

struct ModelRouterServiceHandle(Arc<ModelRouterService>);

impl ModelRouterServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &ModelRouterService {
        &self.0
    }
}

impl Service for ModelRouterServiceHandle {
    fn name(&self) -> &'static str {
        "model-router"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_router_service() {
        let svc = ModelRouterService::new();
        let stats = svc.stats();
        assert_eq!(stats.total_routed, 0);
    }

    #[test]
    fn test_classify_easy_task() {
        let svc = ModelRouterService::new();
        let difficulty = svc.classify("summarize this document");
        assert_eq!(difficulty.suggested_tier, ModelTier::Cheap);
    }

    #[test]
    fn test_classify_hard_task() {
        let svc = ModelRouterService::new();
        let difficulty = svc.classify("analyze and optimize the complex architecture");
        assert!(difficulty.score > 0.5);
    }

    #[test]
    fn test_route_task() {
        let svc = ModelRouterService::new();
        let (model, difficulty) = svc.route("summarize this text");
        assert_eq!(model, "haiku");
        assert_eq!(difficulty.suggested_tier, ModelTier::Cheap);

        let stats = svc.stats();
        assert_eq!(stats.total_routed, 1);
        assert_eq!(stats.cheap_count, 1);
    }

    #[test]
    fn test_recent_decisions() {
        let svc = ModelRouterService::new();
        svc.route("task 1");
        svc.route("task 2");
        svc.route("task 3");

        let decisions = svc.recent_decisions(2);
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].task_description, "task 3");
        assert_eq!(decisions[1].task_description, "task 2");
    }

    #[test]
    fn test_reset() {
        let svc = ModelRouterService::new();
        svc.route("task 1");
        svc.route("task 2");
        svc.reset();

        let stats = svc.stats();
        assert_eq!(stats.total_routed, 0);
        assert!(svc.recent_decisions(10).is_empty());
    }
}
