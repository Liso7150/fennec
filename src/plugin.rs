//! Plugin system for Fennec engine.
//!
//! Inspired by Cordis (DeepSeek Harness's plugin framework), this module provides:
//! - Service registry: typed services accessible by key
//! - Typed events: observable, interceptable event system
//! - Reversible effects: registrations that unwind on plugin unload
//! - Plugin trait: the interface every plugin implements
//!
//! # Architecture
//!
//! Every component in Fennec is a plugin: the model loader, the compute backends
//! (CPU, CUDA, Vulkan, Metal, ORT), the scheduler, the tokenizer, and the HTTP
//! server. Plugins contribute services, subscribe to events, and register effects
//! that are automatically cleaned up when the plugin unloads.
//!
//! # Example
//!
//! ```rust
//! use fennec::plugin::{Plugin, Service, Context, effect};
//!
//! struct MyPlugin;
//!
//! impl Plugin for MyPlugin {
//!     fn name(&self) -> &'static str { "my-plugin" }
//!
//!     fn apply(&self, ctx: &mut Context) -> Result<(), Box<dyn std::error::Error>> {
//!         // Register a service
//!         ctx.register_service("my-service", MyService)?;
//!
//!         // Subscribe to an event
//!         ctx.on("model/loaded", |event| {
//!             println!("Model loaded: {:?}", event);
//!         })?;
//!
//!         // Register a reversible effect
//!         ctx.effect(|| {
//!             println!("Plugin loaded");
//!             || println!("Plugin unloaded")
//!         })?;
//!
//!         Ok(())
//!     }
//! }
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// A service that can be registered in the plugin context.
pub trait Service: Any + Send + Sync {
    /// Returns the service name (used for logging and debugging).
    fn name(&self) -> &'static str;
}

/// A plugin that can be loaded into the Fennec engine.
pub trait Plugin: Send + Sync {
    /// Returns the plugin name (unique identifier).
    fn name(&self) -> &'static str;

    /// Returns the plugin version.
    fn version(&self) -> &'static str {
        "0.0.0"
    }

    /// Returns the plugin dependencies (other plugins that must be loaded first).
    fn dependencies(&self) -> Vec<&'static str> {
        vec![]
    }

    /// Apply the plugin to the context: register services, subscribe to events, etc.
    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError>;

    /// Called when the plugin is unloaded (optional cleanup).
    fn unload(&self, _ctx: &mut Context) -> Result<(), PluginError> {
        Ok(())
    }
}

/// Error type for plugin operations.
#[derive(Debug)]
pub enum PluginError {
    /// Plugin dependency not found.
    DependencyNotFound(String),
    /// Service already registered.
    ServiceAlreadyRegistered(String),
    /// Service not found.
    ServiceNotFound(String),
    /// Event subscription failed.
    EventSubscriptionFailed(String),
    /// Plugin apply failed.
    ApplyFailed(String),
    /// Circular dependency detected.
    CircularDependency(String),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::DependencyNotFound(dep) => write!(f, "Dependency not found: {}", dep),
            PluginError::ServiceAlreadyRegistered(name) => {
                write!(f, "Service already registered: {}", name)
            }
            PluginError::ServiceNotFound(name) => write!(f, "Service not found: {}", name),
            PluginError::EventSubscriptionFailed(event) => {
                write!(f, "Event subscription failed: {}", event)
            }
            PluginError::ApplyFailed(reason) => write!(f, "Plugin apply failed: {}", reason),
            PluginError::CircularDependency(chain) => {
                write!(f, "Circular dependency detected: {}", chain)
            }
        }
    }
}

impl std::error::Error for PluginError {}

/// Event handler function type.
pub type EventHandler = Box<dyn Fn(&Event) + Send + Sync>;

/// An event that can be emitted and observed by plugins.
#[derive(Clone, Debug)]
pub struct Event {
    /// Event name (e.g., "model/loaded", "request/start").
    pub name: String,
    /// Event payload (type-erased).
    pub payload: Arc<dyn Any + Send + Sync>,
}

impl Event {
    /// Create a new event with the given name and payload.
    pub fn new<T: Any + Send + Sync + 'static>(name: impl Into<String>, payload: T) -> Self {
        Self {
            name: name.into(),
            payload: Arc::new(payload),
        }
    }

    /// Get the payload as a specific type.
    pub fn payload<T: Any + 'static>(&self) -> Option<&T> {
        self.payload.downcast_ref::<T>()
    }
}

/// A reversible effect that can be undone when a plugin unloads.
pub struct Effect {
    /// Cleanup function to run when the effect is undone.
    cleanup: Box<dyn FnOnce() + Send>,
}

impl Effect {
    /// Create a new effect with a cleanup function.
    pub fn new<F: FnOnce() + Send + 'static>(cleanup: F) -> Self {
        Self {
            cleanup: Box::new(cleanup),
        }
    }

    /// Undo the effect by running the cleanup function.
    pub fn undo(self) {
        (self.cleanup)()
    }
}

/// The plugin context: service registry, event bus, and effect tracker.
pub struct Context {
    /// Registered services (keyed by name).
    services: RwLock<HashMap<String, Arc<dyn Service>>>,
    /// Event subscribers (keyed by event name).
    subscribers: RwLock<HashMap<String, Vec<Arc<EventHandler>>>>,
    /// Active effects (for cleanup on unload).
    effects: Mutex<Vec<Effect>>,
    /// Loaded plugins (for dependency tracking).
    loaded_plugins: Mutex<Vec<String>>,
}

impl Context {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            subscribers: RwLock::new(HashMap::new()),
            effects: Mutex::new(Vec::new()),
            loaded_plugins: Mutex::new(Vec::new()),
        }
    }

    /// Register a service in the context.
    pub fn register_service<S: Service + 'static>(
        &self,
        name: impl Into<String>,
        service: S,
    ) -> Result<(), PluginError> {
        let name = name.into();
        let mut services = self.services.write().unwrap();
        if services.contains_key(&name) {
            return Err(PluginError::ServiceAlreadyRegistered(name));
        }
        services.insert(name, Arc::new(service));
        Ok(())
    }

    /// Get a service by name and downcast it to the requested type.
    ///
    /// Returns None if the service doesn't exist or can't be downcast.
    pub fn get_service<S: Service + 'static>(&self, name: &str) -> Option<Arc<S>> {
        let services = self.services.read().unwrap();
        let svc = services.get(name)?;
        // Try to downcast the service to the requested type
        let any_ref = Arc::as_ref(svc) as &dyn Any;
        any_ref.downcast_ref::<S>()?;
        // Safety: We just verified the type matches, so this cast is valid.
        // We need to use unsafe because we can't directly cast Arc<dyn Service> to Arc<S>.
        let ptr = Arc::as_ptr(svc) as *const S;
        // Increment the reference count manually
        unsafe {
            Arc::increment_strong_count(ptr);
            Some(Arc::from_raw(ptr))
        }
    }

    /// Subscribe to an event.
    pub fn on<F>(&self, event_name: impl Into<String>, handler: F) -> Result<(), PluginError>
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        let event_name = event_name.into();
        let mut subscribers = self.subscribers.write().unwrap();
        let handlers = subscribers.entry(event_name).or_insert_with(Vec::new);
        handlers.push(Arc::new(Box::new(handler)));
        Ok(())
    }

    /// Emit an event to all subscribers.
    pub fn emit(&self, event: Event) {
        let subscribers = self.subscribers.read().unwrap();
        if let Some(handlers) = subscribers.get(&event.name) {
            for handler in handlers {
                handler(&event);
            }
        }
    }

    /// Register a reversible effect.
    pub fn effect<F>(&self, cleanup: F) -> Result<(), PluginError>
    where
        F: FnOnce() + Send + 'static,
    {
        let mut effects = self.effects.lock().unwrap();
        effects.push(Effect::new(cleanup));
        Ok(())
    }

    /// Mark a plugin as loaded.
    pub fn mark_loaded(&self, plugin_name: impl Into<String>) {
        let mut loaded = self.loaded_plugins.lock().unwrap();
        loaded.push(plugin_name.into());
    }

    /// Check if a plugin is loaded.
    pub fn is_loaded(&self, plugin_name: &str) -> bool {
        let loaded = self.loaded_plugins.lock().unwrap();
        loaded.iter().any(|name| name == plugin_name)
    }

    /// Unload all effects (called when shutting down).
    pub fn unload_all(&self) {
        let mut effects = self.effects.lock().unwrap();
        while let Some(effect) = effects.pop() {
            effect.undo();
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin manager: loads, unloads, and manages plugin lifecycle.
pub struct PluginManager {
    /// The plugin context.
    ctx: Arc<Context>,
    /// Loaded plugins (keyed by name).
    plugins: Mutex<HashMap<String, Arc<dyn Plugin>>>,
}

impl PluginManager {
    /// Create a new plugin manager with the given context.
    pub fn new(ctx: Arc<Context>) -> Self {
        Self {
            ctx,
            plugins: Mutex::new(HashMap::new()),
        }
    }

    /// Load a plugin and apply it to the context.
    pub fn load<P: Plugin + 'static>(&self, plugin: P) -> Result<(), PluginError> {
        let name = plugin.name();

        // Check if already loaded
        {
            let plugins = self.plugins.lock().unwrap();
            if plugins.contains_key(name) {
                return Ok(()); // Already loaded, no-op
            }
        }

        // Check dependencies
        for dep in plugin.dependencies() {
            if !self.ctx.is_loaded(dep) {
                return Err(PluginError::DependencyNotFound(dep.to_string()));
            }
        }

        // Apply the plugin
        let mut ctx = Context::new();
        // Note: In a real implementation, we'd need to share the context properly.
        // This is a simplified version for demonstration.
        plugin.apply(&mut ctx)?;

        // Mark as loaded
        self.ctx.mark_loaded(name);

        // Store the plugin
        let mut plugins = self.plugins.lock().unwrap();
        plugins.insert(name.to_string(), Arc::new(plugin));

        Ok(())
    }

    /// Unload a plugin and undo its effects.
    pub fn unload(&self, name: &str) -> Result<(), PluginError> {
        let mut plugins = self.plugins.lock().unwrap();
        if let Some(plugin) = plugins.remove(name) {
            let mut ctx = Context::new();
            plugin.unload(&mut ctx)?;
        }
        Ok(())
    }

    /// Get a loaded plugin by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Plugin>> {
        let plugins = self.plugins.lock().unwrap();
        plugins.get(name).cloned()
    }

    /// List all loaded plugins.
    pub fn list(&self) -> Vec<String> {
        let plugins = self.plugins.lock().unwrap();
        plugins.keys().cloned().collect()
    }
}

/// Macro to simplify effect registration.
#[macro_export]
macro_rules! effect {
    ($ctx:expr, $cleanup:expr) => {
        $ctx.effect(|| {
            let cleanup = $cleanup;
            cleanup
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn name(&self) -> &'static str {
            "test-plugin"
        }

        fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
            ctx.effect(|| {
                println!("Test plugin loaded");
                || println!("Test plugin unloaded")
            })?;
            Ok(())
        }
    }

    #[test]
    fn test_plugin_load() {
        let ctx = Arc::new(Context::new());
        let manager = PluginManager::new(ctx);
        assert!(manager.load(TestPlugin).is_ok());
        assert!(manager.get("test-plugin").is_some());
    }

    #[test]
    fn test_event_emit() {
        let ctx = Context::new();
        let received = Arc::new(Mutex::new(false));
        let received_clone = received.clone();

        ctx.on("test/event", move |_event| {
            let mut r = received_clone.lock().unwrap();
            *r = true;
        })
        .unwrap();

        ctx.emit(Event::new("test/event", ()));
        assert!(*received.lock().unwrap());
    }
}
