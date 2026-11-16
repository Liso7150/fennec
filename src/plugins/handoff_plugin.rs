//! Handoff plugin: context summarization and export at token threshold.
//!
//! Implements the "Handoff" mod: at 250k tokens or on command, creates a detailed
//! context summary for a fresh chat session. Includes secret redaction and
//! structured export format.

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::{Arc, RwLock};

/// Handoff configuration.
#[derive(Clone, Debug)]
pub struct HandoffConfig {
    /// Token threshold for automatic handoff (250k default).
    pub token_threshold: u64,
    /// Whether to auto-trigger handoff at threshold.
    pub auto_handoff: bool,
    /// Secret patterns to redact (API keys, tokens, etc.).
    pub secret_patterns: Vec<String>,
    /// Maximum summary length in characters.
    pub max_summary_chars: usize,
}

impl Default for HandoffConfig {
    fn default() -> Self {
        Self {
            token_threshold: 250_000,
            auto_handoff: true,
            secret_patterns: vec![
                "(?i)(api[_-]?key|token|secret|password|auth)\\s*[:=]\\s*['\"]?[\\w\\-\\.]+".to_string(),
                "ghp_[a-zA-Z0-9]{36}".to_string(),
                "sk-[a-zA-Z0-9]{32,}".to_string(),
                "Bearer\\s+[a-zA-Z0-9\\-\\.]+".to_string(),
            ],
            max_summary_chars: 10000,
        }
    }
}

/// Handoff summary document.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HandoffSummary {
    /// Timestamp of handoff creation.
    pub timestamp: String,
    /// Total token count at handoff.
    pub token_count: u64,
    /// Conversation summary.
    pub summary: String,
    /// Key decisions made during the session.
    pub key_decisions: Vec<String>,
    /// Open questions or unresolved issues.
    pub open_questions: Vec<String>,
    /// Files that were modified.
    pub files_modified: Vec<String>,
    /// Next steps to continue work.
    pub next_steps: Vec<String>,
    /// Context that should be preserved.
    pub important_context: Vec<String>,
}

impl HandoffSummary {
    pub fn new(token_count: u64) -> Self {
        Self {
            timestamp: chrono_like_now(),
            token_count,
            summary: String::new(),
            key_decisions: Vec::new(),
            open_questions: Vec::new(),
            files_modified: Vec::new(),
            next_steps: Vec::new(),
            important_context: Vec::new(),
        }
    }

    /// Export as JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export as markdown.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Session Handoff\n\n");
        md.push_str(&format!("**Timestamp**: {}\n", self.timestamp));
        md.push_str(&format!("**Token Count**: {}\n\n", self.token_count));

        md.push_str("## Summary\n\n");
        md.push_str(&self.summary);
        md.push_str("\n\n");

        if !self.key_decisions.is_empty() {
            md.push_str("## Key Decisions\n\n");
            for d in &self.key_decisions {
                md.push_str(&format!("- {}\n", d));
            }
            md.push('\n');
        }

        if !self.files_modified.is_empty() {
            md.push_str("## Files Modified\n\n");
            for f in &self.files_modified {
                md.push_str(&format!("- `{}`\n", f));
            }
            md.push('\n');
        }

        if !self.open_questions.is_empty() {
            md.push_str("## Open Questions\n\n");
            for q in &self.open_questions {
                md.push_str(&format!("- {}\n", q));
            }
            md.push('\n');
        }

        if !self.next_steps.is_empty() {
            md.push_str("## Next Steps\n\n");
            for (i, s) in self.next_steps.iter().enumerate() {
                md.push_str(&format!("{}. {}\n", i + 1, s));
            }
            md.push('\n');
        }

        if !self.important_context.is_empty() {
            md.push_str("## Important Context\n\n");
            for c in &self.important_context {
                md.push_str(&format!("- {}\n", c));
            }
        }

        md
    }
}

fn chrono_like_now() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}Z", secs)
}

/// Handoff service.
pub struct HandoffService {
    config: RwLock<HandoffConfig>,
    current_tokens: RwLock<u64>,
    handoff_count: RwLock<u64>,
    last_handoff: RwLock<Option<HandoffSummary>>,
}

impl HandoffService {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(HandoffConfig::default()),
            current_tokens: RwLock::new(0),
            handoff_count: RwLock::new(0),
            last_handoff: RwLock::new(None),
        }
    }

    pub fn config(&self) -> HandoffConfig {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, config: HandoffConfig) {
        *self.config.write().unwrap() = config;
    }

    /// Update the current token count.
    pub fn update_tokens(&self, count: u64) {
        *self.current_tokens.write().unwrap() = count;
    }

    /// Add tokens to the current count.
    pub fn add_tokens(&self, count: u64) {
        let mut tokens = self.current_tokens.write().unwrap();
        *tokens += count;
    }

    /// Get the current token count.
    pub fn current_tokens(&self) -> u64 {
        *self.current_tokens.read().unwrap()
    }

    /// Check if we should trigger a handoff.
    pub fn should_handoff(&self) -> bool {
        let config = self.config();
        if !config.auto_handoff {
            return false;
        }
        self.current_tokens() >= config.token_threshold
    }

    /// Redact secrets from text.
    pub fn redact_secrets(&self, text: &str) -> String {
        let config = self.config();
        let mut result = text.to_string();

        for pattern in &config.secret_patterns {
            if let Ok(re) = fancy_regex::Regex::new(pattern) {
                result = re.replace_all(&result, "[REDACTED]").to_string();
            }
        }

        result
    }

    /// Create a handoff summary from conversation context.
    /// In production, this would use a cheap model to summarize; here we extract key info.
    pub fn create_handoff(&self, conversation_context: &ConversationContext) -> HandoffSummary {
        let token_count = self.current_tokens();
        let mut summary = HandoffSummary::new(token_count);

        summary.summary = self.redact_secrets(&conversation_context.summary);
        summary.key_decisions = conversation_context
            .decisions
            .iter()
            .map(|d| self.redact_secrets(d))
            .collect();
        summary.open_questions = conversation_context
            .questions
            .iter()
            .map(|q| self.redact_secrets(q))
            .collect();
        summary.files_modified = conversation_context.files_modified.clone();
        summary.next_steps = conversation_context
            .next_steps
            .iter()
            .map(|s| self.redact_secrets(s))
            .collect();
        summary.important_context = conversation_context
            .important_context
            .iter()
            .map(|c| self.redact_secrets(c))
            .collect();

        if summary.summary.len() > self.config().max_summary_chars {
            summary.summary = summary.summary[..self.config().max_summary_chars].to_string();
        }

        *self.handoff_count.write().unwrap() += 1;
        *self.last_handoff.write().unwrap() = Some(summary.clone());

        self.update_tokens(0);

        summary
    }

    /// Get handoff statistics.
    pub fn stats(&self) -> HandoffStats {
        HandoffStats {
            current_tokens: self.current_tokens(),
            token_threshold: self.config().token_threshold,
            handoff_count: *self.handoff_count.read().unwrap(),
            should_handoff: self.should_handoff(),
        }
    }

    /// Get the last handoff summary.
    pub fn last_handoff(&self) -> Option<HandoffSummary> {
        self.last_handoff.read().unwrap().clone()
    }
}

impl Default for HandoffService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for HandoffService {
    fn name(&self) -> &'static str {
        "handoff"
    }
}

/// Context information needed to create a handoff.
#[derive(Clone, Debug, Default)]
pub struct ConversationContext {
    pub summary: String,
    pub decisions: Vec<String>,
    pub questions: Vec<String>,
    pub files_modified: Vec<String>,
    pub next_steps: Vec<String>,
    pub important_context: Vec<String>,
}

/// Handoff statistics.
#[derive(Clone, Debug)]
pub struct HandoffStats {
    pub current_tokens: u64,
    pub token_threshold: u64,
    pub handoff_count: u64,
    pub should_handoff: bool,
}

/// Handoff plugin.
pub struct HandoffPlugin {
    config: HandoffConfig,
}

impl HandoffPlugin {
    pub fn new(config: HandoffConfig) -> Self {
        Self { config }
    }
}

impl Default for HandoffPlugin {
    fn default() -> Self {
        Self::new(HandoffConfig::default())
    }
}

impl Plugin for HandoffPlugin {
    fn name(&self) -> &'static str {
        "handoff"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        let service = Arc::new(HandoffService::new());
        service.set_config(self.config.clone());

        ctx.register_service("handoff", HandoffServiceHandle(service.clone()))?;

        let svc = service.clone();
        ctx.on("token/count", move |event| {
            if let Some(count) = event.payload::<u64>() {
                svc.update_tokens(*count);
            }
        })?;

        let svc2 = service.clone();
        ctx.on("token/add", move |event| {
            if let Some(count) = event.payload::<u64>() {
                svc2.add_tokens(*count);
            }
        })?;

        ctx.effect(|| {})?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("handoff/unloading", ()));
        Ok(())
    }
}

struct HandoffServiceHandle(Arc<HandoffService>);

impl HandoffServiceHandle {
    #[allow(dead_code)]
    fn inner(&self) -> &HandoffService {
        &self.0
    }
}

impl Service for HandoffServiceHandle {
    fn name(&self) -> &'static str {
        "handoff"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handoff_service_new() {
        let svc = HandoffService::new();
        assert_eq!(svc.current_tokens(), 0);
        assert!(!svc.should_handoff());
    }

    #[test]
    fn test_update_tokens() {
        let svc = HandoffService::new();
        svc.update_tokens(1000);
        assert_eq!(svc.current_tokens(), 1000);
    }

    #[test]
    fn test_add_tokens() {
        let svc = HandoffService::new();
        svc.add_tokens(500);
        svc.add_tokens(300);
        assert_eq!(svc.current_tokens(), 800);
    }

    #[test]
    fn test_should_handoff() {
        let svc = HandoffService::new();
        svc.update_tokens(250_001);
        assert!(svc.should_handoff());
    }

    #[test]
    fn test_redact_secrets() {
        let svc = HandoffService::new();
        let text = "api_key: sk-abc123def456ghi789jkl012mno345pq";
        let redacted = svc.redact_secrets(text);
        assert!(!redacted.contains("sk-abc123"));
        assert!(redacted.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_github_token() {
        let svc = HandoffService::new();
        let text = "token: ghp-abcdefghijklmnopqrstuvwxyz0123456789";
        let redacted = svc.redact_secrets(text);
        assert!(!redacted.contains("ghp_"));
        assert!(redacted.contains("[REDACTED]"));
    }

    #[test]
    fn test_create_handoff() {
        let svc = HandoffService::new();
        svc.update_tokens(250_000);

        let context = ConversationContext {
            summary: "Working on feature X".to_string(),
            decisions: vec!["Used approach Y".to_string()],
            questions: vec!["How to handle Z?".to_string()],
            files_modified: vec!["src/main.rs".to_string()],
            next_steps: vec!["Implement Z".to_string()],
            important_context: vec!["User prefers approach A".to_string()],
        };

        let handoff = svc.create_handoff(&context);
        assert_eq!(handoff.token_count, 250_000);
        assert_eq!(handoff.summary, "Working on feature X");
        assert_eq!(handoff.key_decisions.len(), 1);
        assert_eq!(handoff.files_modified.len(), 1);
    }

    #[test]
    fn test_handoff_summary_json() {
        let summary = HandoffSummary::new(1000);
        let json = summary.to_json();
        assert!(json.contains("token_count"));
        assert!(json.contains("1000"));
    }

    #[test]
    fn test_handoff_summary_markdown() {
        let mut summary = HandoffSummary::new(1000);
        summary.summary = "Test summary".to_string();
        summary.files_modified = vec!["test.rs".to_string()];
        let md = summary.to_markdown();
        assert!(md.contains("Session Handoff"));
        assert!(md.contains("Test summary"));
        assert!(md.contains("Files Modified"));
    }

    #[test]
    fn test_stats() {
        let svc = HandoffService::new();
        svc.update_tokens(100_000);
        let stats = svc.stats();
        assert_eq!(stats.current_tokens, 100_000);
        assert_eq!(stats.token_threshold, 250_000);
        assert!(!stats.should_handoff);
    }

    #[test]
    fn test_token_reset_after_handoff() {
        let svc = HandoffService::new();
        svc.update_tokens(250_000);
        let context = ConversationContext::default();
        svc.create_handoff(&context);
        assert_eq!(svc.current_tokens(), 0);
    }
}
