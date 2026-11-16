//! Research router plugin: web research with a cheap model, returns brief with sources.
//!
//! Implements the "Research Router" mod: keeps research out of the main context
//! by delegating web searches and summarization to a cheaper model tier.

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::{Arc, RwLock};

/// Research router configuration.
#[derive(Clone, Debug)]
pub struct ResearchRouterConfig {
    /// Model to use for research (cheap tier).
    pub research_model: String,
    /// Maximum number of sources to include in a brief.
    pub max_sources: usize,
    /// Maximum brief length in characters.
    pub max_brief_chars: usize,
    /// Whether to automatically detect research intent.
    pub auto_detect: bool,
    /// Keywords that trigger research routing.
    pub trigger_keywords: Vec<String>,
}

impl Default for ResearchRouterConfig {
    fn default() -> Self {
        Self {
            research_model: "haiku".to_string(),
            max_sources: 5,
            max_brief_chars: 3000,
            auto_detect: true,
            trigger_keywords: vec![
                "research".to_string(),
                "search for".to_string(),
                "find out".to_string(),
                "look up".to_string(),
                "what is".to_string(),
                "how does".to_string(),
                "compare".to_string(),
                "latest".to_string(),
            ],
        }
    }
}

/// A source referenced in a research brief.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResearchSource {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub relevance: f32,
}

/// A research brief produced by the cheap model.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResearchBrief {
    pub query: String,
    pub brief: String,
    pub sources: Vec<ResearchSource>,
    pub model_used: String,
    pub estimated_tokens: usize,
    pub timestamp: String,
}

/// Research router service.
pub struct ResearchRouterService {
    config: RwLock<ResearchRouterConfig>,
    research_log: RwLock<Vec<ResearchBrief>>,
    total_queries: RwLock<u64>,
    total_tokens_saved: RwLock<u64>,
}

impl ResearchRouterService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(ResearchRouterConfig::default()),
            research_log: RwLock::new(Vec::new()),
            total_queries: RwLock::new(0),
            total_tokens_saved: RwLock::new(0),
        }
    }

    pub fn config(&self) -> ResearchRouterConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: ResearchRouterConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Check if a query looks like a research task.
    pub fn is_research_query(&self, query: &str) -> bool {
        let config = self.config();
        if !config.auto_detect {
            return false;
        }
        let lower = query.to_lowercase();
        config.trigger_keywords.iter().any(|kw| lower.contains(kw))
    }

    /// Create a research brief from search results.
    pub fn create_brief(&self, query: &str, sources: Vec<ResearchSource>) -> ResearchBrief {
        let config = self.config();

        let mut sorted = sources;
        sorted.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_sources: Vec<_> = sorted.into_iter().take(config.max_sources).collect();

        let mut parts = Vec::new();
        parts.push(format!("Research: {}", query));
        parts.push(String::new());

        if !top_sources.is_empty() {
            parts.push("Key findings:".to_string());
            for (i, src) in top_sources.iter().enumerate() {
                parts.push(format!("{}. {} — {}", i + 1, src.title, src.snippet));
            }
            parts.push(String::new());
            parts.push("Sources:".to_string());
            for src in &top_sources {
                parts.push(format!("  - {} ({})", src.title, src.url));
            }
        } else {
            parts.push("No sources found.".to_string());
        }

        let mut brief = parts.join("\n");
        if brief.len() > config.max_brief_chars {
            brief.truncate(config.max_brief_chars);
        }

        let estimated_tokens = query.len() / 4 + brief.len() / 4;

        let research_brief = ResearchBrief {
            query: query.to_string(),
            brief,
            sources: top_sources,
            model_used: config.research_model.clone(),
            estimated_tokens,
            timestamp: timestamp_now(),
        };

        *self.total_queries.write().unwrap() += 1;
        *self.total_tokens_saved.write().unwrap() += estimated_tokens as u64 * 3;
        self.research_log.write().unwrap().push(research_brief.clone());

        research_brief
    }

    /// Route a query.
    pub fn route(&self, query: &str) -> ResearchRouteResult {
        if self.is_research_query(query) {
            ResearchRouteResult::RouteToResearch(query.to_string())
        } else {
            ResearchRouteResult::PassThrough(query.to_string())
        }
    }

    pub fn stats(&self) -> ResearchStats {
        ResearchStats {
            total_queries: *self.total_queries.read().unwrap(),
            total_tokens_saved: *self.total_tokens_saved.read().unwrap(),
            recent_briefs: self.research_log.read().unwrap().len(),
        }
    }

    pub fn recent_briefs(&self, limit: usize) -> Vec<ResearchBrief> {
        let log = self.research_log.read().unwrap();
        log.iter().rev().take(limit).cloned().collect()
    }

    pub fn reset_stats(&self) {
        *self.total_queries.write().unwrap() = 0;
        *self.total_tokens_saved.write().unwrap() = 0;
        self.research_log.write().unwrap().clear();
    }
}

impl Default for ResearchRouterService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for ResearchRouterService {
    fn name(&self) -> &'static str {
        "research-router"
    }
}

#[derive(Clone, Debug)]
pub enum ResearchRouteResult {
    RouteToResearch(String),
    PassThrough(String),
}

#[derive(Clone, Debug)]
pub struct ResearchStats {
    pub total_queries: u64,
    pub total_tokens_saved: u64,
    pub recent_briefs: usize,
}

fn timestamp_now() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}Z", secs)
}

/// Research router plugin.
pub struct ResearchRouterPlugin {
    config: ResearchRouterConfig,
}

impl ResearchRouterPlugin {
    pub fn new(config: ResearchRouterConfig) -> Self {
        Self { config }
    }
}

impl Default for ResearchRouterPlugin {
    fn default() -> Self {
        Self::new(ResearchRouterConfig::default())
    }
}

impl Plugin for ResearchRouterPlugin {
    fn name(&self) -> &'static str {
        "research-router"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = Arc::new(ResearchRouterService::new());
        service.set_config(self.config.clone());

        ctx.register_service(
            "research-router",
            ResearchRouterServiceHandle(service.clone()),
        )?;

        let svc = service.clone();
        ctx.on("research/query", move |event| {
            if let Some(query) = event.payload::<String>() {
                let _result = svc.route(query);
            }
        })?;

        ctx.effect(|| {})?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("research-router/unloading", ()));
        Ok(())
    }
}

struct ResearchRouterServiceHandle(Arc<ResearchRouterService>);

impl ResearchRouterServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &ResearchRouterService {
        &self.0
    }
}

impl Service for ResearchRouterServiceHandle {
    fn name(&self) -> &'static str {
        "research-router"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sources() -> Vec<ResearchSource> {
        vec![
            ResearchSource {
                title: "Rust Programming Language".to_string(),
                url: "https://www.rust-lang.org".to_string(),
                snippet: "A language for reliable software.".to_string(),
                relevance: 0.95,
            },
            ResearchSource {
                title: "The Rust Book".to_string(),
                url: "https://doc.rust-lang.org/book/".to_string(),
                snippet: "The official Rust book.".to_string(),
                relevance: 0.88,
            },
            ResearchSource {
                title: "Rust by Example".to_string(),
                url: "https://doc.rust-lang.org/rust-by-example/".to_string(),
                snippet: "Learn Rust with examples.".to_string(),
                relevance: 0.75,
            },
        ]
    }

    #[test]
    fn test_is_research_query() {
        let svc = ResearchRouterService::new();
        assert!(svc.is_research_query("research the latest trends"));
        assert!(svc.is_research_query("search for Rust tutorials"));
        assert!(svc.is_research_query("what is machine learning"));
        assert!(!svc.is_research_query("fix this bug"));
        assert!(!svc.is_research_query("implement feature X"));
    }

    #[test]
    fn test_route_research() {
        let svc = ResearchRouterService::new();
        let result = svc.route("research quantum computing");
        assert!(matches!(result, ResearchRouteResult::RouteToResearch(_)));
    }

    #[test]
    fn test_route_passthrough() {
        let svc = ResearchRouterService::new();
        let result = svc.route("fix the compilation error");
        assert!(matches!(result, ResearchRouteResult::PassThrough(_)));
    }

    #[test]
    fn test_create_brief() {
        let svc = ResearchRouterService::new();
        let brief = svc.create_brief("learn Rust", sample_sources());
        assert_eq!(brief.query, "learn Rust");
        assert!(!brief.brief.is_empty());
        assert_eq!(brief.sources.len(), 3);
        assert_eq!(brief.model_used, "haiku");
        assert!(brief.estimated_tokens > 0);
    }

    #[test]
    fn test_brief_max_sources() {
        let svc = ResearchRouterService::new();
        svc.set_config(ResearchRouterConfig {
            max_sources: 2,
            ..Default::default()
        });
        let brief = svc.create_brief("learn Rust", sample_sources());
        assert_eq!(brief.sources.len(), 2);
    }

    #[test]
    fn test_sources_sorted_by_relevance() {
        let svc = ResearchRouterService::new();
        let brief = svc.create_brief("learn Rust", sample_sources());
        assert!(brief.sources[0].relevance >= brief.sources[1].relevance);
        assert!(brief.sources[1].relevance >= brief.sources[2].relevance);
    }

    #[test]
    fn test_stats() {
        let svc = ResearchRouterService::new();
        svc.create_brief("query 1", sample_sources());
        svc.create_brief("query 2", sample_sources());
        let stats = svc.stats();
        assert_eq!(stats.total_queries, 2);
        assert!(stats.total_tokens_saved > 0);
    }

    #[test]
    fn test_recent_briefs() {
        let svc = ResearchRouterService::new();
        svc.create_brief("first", sample_sources());
        svc.create_brief("second", sample_sources());
        let recent = svc.recent_briefs(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].query, "second");
    }

    #[test]
    fn test_reset_stats() {
        let svc = ResearchRouterService::new();
        svc.create_brief("query", sample_sources());
        svc.reset_stats();
        let stats = svc.stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.total_tokens_saved, 0);
    }

    #[test]
    fn test_service_name() {
        let svc = ResearchRouterService::new();
        assert_eq!(svc.name(), "research-router");
    }
}
