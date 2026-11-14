//! Backend plugin: wraps the compute backend selection and initialization.
//!
//! This plugin makes the backend (CPU, CUDA, Vulkan, Metal, ORT) pluggable,
//! allowing runtime selection and hot-swapping of compute backends.

use crate::backend::Backend;
use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::sync::RwLock;

/// Backend service: provides access to the active compute backend.
pub struct BackendService {
    backend: RwLock<Backend>,
}

impl BackendService {
    /// Create a new backend service.
    pub fn new() -> Self {
        Self {
            backend: RwLock::new(Backend::Cpu),
        }
    }

    /// Get the active backend.
    pub fn backend(&self) -> Backend {
        *self.backend.read().unwrap()
    }

    /// Set the active backend.
    pub fn set_backend(&self, backend: Backend) {
        *self.backend.write().unwrap() = backend;
    }
}

impl Service for BackendService {
    fn name(&self) -> &'static str {
        "backend"
    }
}

impl Default for BackendService {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend plugin: manages compute backend lifecycle.
pub struct BackendPlugin {
    preferred: Backend,
}

impl BackendPlugin {
    /// Create a new backend plugin with the preferred backend.
    pub fn new(preferred: Backend) -> Self {
        Self { preferred }
    }
}

impl Plugin for BackendPlugin {
    fn name(&self) -> &'static str {
        "backend"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Register the backend service
        let service = BackendService::new();
        service.set_backend(self.preferred);
        ctx.register_service("backend", service)?;

        // Subscribe to backend selection events
        ctx.on("backend/select", |event| {
            if let Some(backend) = event.payload::<Backend>() {
                println!("Backend selection event: {:?}", backend);
            }
        })?;

        // Emit backend initialized event
        let preferred = self.preferred;
        ctx.effect(move || {
            println!("Backend plugin initialized with: {:?}", preferred);
        })?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        ctx.emit(Event::new("backend/unloading", ()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_service() {
        let service = BackendService::new();
        assert_eq!(service.backend(), Backend::Cpu);

        service.set_backend(Backend::Vulkan);
        assert_eq!(service.backend(), Backend::Vulkan);
    }
}
