//! Tokenizer plugin: wraps the BPE tokenizer.
//!
//! This plugin makes the tokenizer pluggable, allowing different tokenization
//! schemes to be loaded and used.

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use crate::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Tokenizer service: provides access to the tokenizer.
pub struct TokenizerService {
    tokenizer: RwLock<Option<Arc<Tokenizer>>>,
    vocab_path: RwLock<Option<PathBuf>>,
}

impl TokenizerService {
    /// Create a new tokenizer service.
    pub fn new() -> Self {
        Self {
            tokenizer: RwLock::new(None),
            vocab_path: RwLock::new(None),
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
    }
}
