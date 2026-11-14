//! Model plugin: wraps the model loading and inference.
//!
//! This plugin makes the model loader pluggable, allowing different model
//! formats (safetensors, GGUF, etc.) to be loaded through a unified interface.

use crate::model::Model;
use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use crate::safetensors::Shards;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Model service: provides access to the loaded model.
pub struct ModelService {
    model: RwLock<Option<Arc<Model>>>,
    shards: RwLock<Option<Arc<Shards>>>,
    model_path: RwLock<Option<PathBuf>>,
}

impl ModelService {
    /// Create a new model service.
    pub fn new() -> Self {
        Self {
            model: RwLock::new(None),
            shards: RwLock::new(None),
            model_path: RwLock::new(None),
        }
    }

    /// Get the loaded model.
    pub fn model(&self) -> Option<Arc<Model>> {
        self.model.read().unwrap().clone()
    }

    /// Get the model shards.
    pub fn shards(&self) -> Option<Arc<Shards>> {
        self.shards.read().unwrap().clone()
    }

    /// Get the model path.
    pub fn model_path(&self) -> Option<PathBuf> {
        self.model_path.read().unwrap().clone()
    }

    /// Set the loaded model.
    pub fn set_model(&self, model: Arc<Model>, shards: Arc<Shards>, path: PathBuf) {
        *self.model.write().unwrap() = Some(model);
        *self.shards.write().unwrap() = Some(shards);
        *self.model_path.write().unwrap() = Some(path);
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.read().unwrap().is_some()
    }
}

impl Service for ModelService {
    fn name(&self) -> &'static str {
        "model"
    }
}

impl Default for ModelService {
    fn default() -> Self {
        Self::new()
    }
}

/// Model plugin: manages model loading lifecycle.
pub struct ModelPlugin {
    model_path: PathBuf,
}

impl ModelPlugin {
    /// Create a new model plugin with the model path.
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }
}

impl Plugin for ModelPlugin {
    fn name(&self) -> &'static str {
        "model"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn dependencies(&self) -> Vec<&'static str> {
        vec!["backend"] // Model plugin depends on backend plugin
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Register the model service
        let service = ModelService::new();
        ctx.register_service("model", service)?;

        // Subscribe to model load requests
        ctx.on("model/load", |event| {
            if let Some(requested_path) = event.payload::<PathBuf>() {
                println!("Model load request: {:?}", requested_path);
            }
        })?;

        // Emit model plugin initialized event
        let path = self.model_path.clone();
        ctx.effect(move || {
            println!("Model plugin initialized with path: {:?}", path);
        })?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Clean up model resources
        ctx.emit(Event::new("model/unloading", ()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_service() {
        let service = ModelService::new();
        assert!(!service.is_loaded());
        assert!(service.model().is_none());
    }
}
