//! Tokenizer plugin: wraps the BPE tokenizer.
//!
//! This plugin makes the tokenizer pluggable, allowing different tokenization
//! schemes to be loaded and used. Also provides token counting and session-level
//! token tracking for cost optimization mods (handoff, input router, etc.).

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use crate::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Tokenizer service: provides access to the tokenizer and token counting.
pub struct TokenizerService {
    tokenizer: RwLock<Option<Arc<Tokenizer>>>,
    vocab_path: RwLock<Option<PathBuf>>,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
}

impl TokenizerService {
    /// Create a new tokenizer service.
    pub fn new() -> Self {
        Self {
            tokenizer: RwLock::new(None),
            vocab_path: RwLock::new(None),
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
        }
    }

    /// Get the tokenizer.
    pub fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        self.tokenizer.read().unwrap().clone()
    }

    /// Get the vocab path.
    pub fn vocab_path(&self) -> Option<PathBuf> {
        self.vocab_path.read().unwrap().clone()
    }

    /// Set the tokenizer.
    pub fn set_tokenizer(&self, tokenizer: Arc<Tokenizer>, path: PathBuf) {
        *self.tokenizer.write().unwrap() = Some(tokenizer);
        *self.vocab_path.write().unwrap() = Some(path);
    }

    /// Check if a tokenizer is loaded.
    pub fn is_loaded(&self) -> bool {
        self.tokenizer.read().unwrap().is_some()
    }

    /// Count tokens in the given text. Returns 0 if no tokenizer is loaded.
    pub fn count_tokens(&self, text: &str) -> usize {
        match self.tokenizer() {
            Some(tok) => tok.encode(text).len(),
            None => 0,
        }
    }

    /// Record input tokens (prompt tokens from the user).
    pub fn add_input_tokens(&self, count: u64) {
        self.input_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Record output tokens (generated tokens from the model).
    pub fn add_output_tokens(&self, count: u64) {
        self.output_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Get total input tokens seen this session.
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens.load(Ordering::Relaxed)
    }

    /// Get total output tokens seen this session.
    pub fn total_output_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::Relaxed)
    }

    /// Get total tokens (input + output) this session.
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens() + self.total_output_tokens()
    }

    /// Reset session token counters.
    pub fn reset_counters(&self) {
        self.input_tokens.store(0, Ordering::Relaxed);
        self.output_tokens.store(0, Ordering::Relaxed);
    }
}

impl Service for TokenizerService {
    fn name(&self) -> &'static str {
        "tokenizer"
    }
}

impl Default for TokenizerService {
    fn default() -> Self {
        Self::new()
    }
}

/// Tokenizer plugin: manages tokenizer lifecycle.
pub struct TokenizerPlugin {
    vocab_path: PathBuf,
}

impl TokenizerPlugin {
    /// Create a new tokenizer plugin with the vocab path.
    pub fn new(vocab_path: PathBuf) -> Self {
        Self { vocab_path }
    }
}

impl Plugin for TokenizerPlugin {
    fn name(&self) -> &'static str {
        "tokenizer"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Register the tokenizer service
        let service = TokenizerService::new();
        ctx.register_service("tokenizer", service)?;

        // Subscribe to tokenizer load requests
        ctx.on("tokenizer/load", |event| {
            if let Some(requested_path) = event.payload::<PathBuf>() {
                println!("Tokenizer load request: {:?}", requested_path);
            }
        })?;

        // Emit tokenizer plugin initialized event
        let path = self.vocab_path.clone();
        ctx.effect(move || {
            println!("Tokenizer plugin initialized with path: {:?}", path);
        })?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Clean up tokenizer resources
        ctx.emit(Event::new("tokenizer/unloading", ()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_service() {
        let service = TokenizerService::new();
        assert!(!service.is_loaded());
        assert!(service.tokenizer().is_none());
        assert_eq!(service.count_tokens("hello"), 0);
    }

    #[test]
    fn test_token_counters() {
        let service = TokenizerService::new();
        assert_eq!(service.total_input_tokens(), 0);
        assert_eq!(service.total_output_tokens(), 0);
        assert_eq!(service.total_tokens(), 0);

        service.add_input_tokens(100);
        service.add_output_tokens(50);
        assert_eq!(service.total_input_tokens(), 100);
        assert_eq!(service.total_output_tokens(), 50);
        assert_eq!(service.total_tokens(), 150);

        service.add_input_tokens(250000);
        assert_eq!(service.total_tokens(), 250150);

        service.reset_counters();
        assert_eq!(service.total_tokens(), 0);
    }
}
