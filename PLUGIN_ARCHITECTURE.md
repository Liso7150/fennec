# Fennec Plugin Architecture

**Date:** 2026-11-14  
**Status:** Implemented (Phase 1 - Core Infrastructure)  
**Inspiration:** Cordis (DeepSeek Harness's plugin framework)

---

## Overview

Fennec now features a plugin architecture inspired by Cordis, the plugin framework used by DeepSeek Harness. This makes every component of the engine pluggable, composable, and extensible.

### Core Principles

1. **Everything is a plugin**: Backend, model loader, tokenizer, HTTP server - all are plugins
2. **Service registry**: Typed services accessible by key (like `ctx.services` in Cordis)
3. **Typed events**: Observable, interceptable event system (like `ctx.on`/`ctx.emit`)
4. **Reversible effects**: Registrations that unwind on plugin unload (like `ctx.effect`)
5. **Dependency tracking**: Plugins declare dependencies, loaded in correct order

---

## Architecture

### Plugin Trait

Every plugin implements the `Plugin` trait:

```rust
pub trait Plugin: Send + Sync {
    /// Unique plugin identifier
    fn name(&self) -> &'static str;
    
    /// Plugin version
    fn version(&self) -> &'static str { "0.0.0" }
    
    /// Dependencies (other plugins that must load first)
    fn dependencies(&self) -> Vec<&'static str> { vec![] }
    
    /// Apply the plugin: register services, subscribe to events
    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError>;
    
    /// Cleanup when unloading (optional)
    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> { Ok(()) }
}
```

### Service Trait

Services are typed capabilities registered in the context:

```rust
pub trait Service: Any + Send + Sync {
    fn name(&self) -> &'static str;
}
```

### Context

The `Context` is the plugin runtime:

```rust
pub struct Context {
    services: RwLock<HashMap<String, Arc<dyn Service>>>,
    subscribers: RwLock<HashMap<String, Vec<Arc<EventHandler>>>>,
    effects: Mutex<Vec<Effect>>,
    loaded_plugins: Mutex<Vec<String>>,
}
```

**Key methods:**
- `register_service<S: Service>(name, service)` - Register a typed service
- `get_service<S: Service>(name)` - Retrieve a service by name and type
- `on(event_name, handler)` - Subscribe to an event
- `emit(event)` - Emit an event to all subscribers
- `effect(cleanup)` - Register a reversible effect

### Events

Events are typed messages passed between plugins:

```rust
pub struct Event {
    pub name: String,
    pub payload: Arc<dyn Any + Send + Sync>,
}
```

**Standard events:**
- `backend/select` - Backend selection changed
- `backend/unloading` - Backend plugin unloading
- `model/load` - Model load request
- `model/loaded` - Model successfully loaded
- `model/unloading` - Model plugin unloading
- `server/start` - HTTP server starting
- `server/stop` - HTTP server stopping
- `tokenizer/load` - Tokenizer load request

---

## Built-in Plugins

### BackendPlugin

Manages compute backend selection (CPU, CUDA, Vulkan, Metal, ORT).

**Service:** `BackendService`
- `backend()` - Get active backend
- `set_backend(backend)` - Set active backend

**Example:**
```rust
let plugin = BackendPlugin::new(Backend::Vulkan);
manager.load(plugin)?;
```

### ModelPlugin

Manages model loading and inference.

**Service:** `ModelService`
- `model()` - Get loaded model
- `shards()` - Get model shards
- `is_loaded()` - Check if model is loaded

**Dependencies:** `backend`

**Example:**
```rust
let plugin = ModelPlugin::new(PathBuf::from("model.safetensors"));
manager.load(plugin)?;
```

### TokenizerPlugin

Manages BPE tokenizer loading.

**Service:** `TokenizerService`
- `tokenizer()` - Get loaded tokenizer
- `is_loaded()` - Check if tokenizer is loaded

**Example:**
```rust
let plugin = TokenizerPlugin::new(PathBuf::from("tokenizer.json"));
manager.load(plugin)?;
```

### ServerPlugin

Manages HTTP server lifecycle.

**Service:** `ServerService`
- `addr()` - Get server address
- `is_running()` - Check if server is running

**Dependencies:** `model`, `tokenizer`

**Example:**
```rust
let addr = "127.0.0.1:9090".parse()?;
let plugin = ServerPlugin::new(addr);
manager.load(plugin)?;
```

---

## Plugin Manager

The `PluginManager` orchestrates plugin lifecycle:

```rust
pub struct PluginManager {
    ctx: Arc<Context>,
    plugins: Mutex<HashMap<String, Arc<dyn Plugin>>>,
}
```

**Key methods:**
- `load(plugin)` - Load and apply a plugin
- `unload(name)` - Unload a plugin and undo its effects
- `get(name)` - Get a loaded plugin by name
- `list()` - List all loaded plugins

**Example:**
```rust
let ctx = Arc::new(Context::new());
let manager = PluginManager::new(ctx);

// Load plugins in dependency order
manager.load(BackendPlugin::new(Backend::Vulkan))?;
manager.load(ModelPlugin::new(model_path))?;
manager.load(TokenizerPlugin::new(vocab_path))?;
manager.load(ServerPlugin::new(addr))?;
```

---

## Usage Example

```rust
use fennec::plugin::{Context, PluginManager};
use fennec::plugins::{BackendPlugin, ModelPlugin, ServerPlugin, TokenizerPlugin};
use fennec::backend::Backend;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create context and manager
    let ctx = Arc::new(Context::new());
    let manager = PluginManager::new(ctx.clone());

    // Load backend plugin
    manager.load(BackendPlugin::new(Backend::Vulkan))?;

    // Load model plugin
    let model_path = PathBuf::from("models/qwen3.5-moe.safetensors");
    manager.load(ModelPlugin::new(model_path))?;

    // Load tokenizer plugin
    let vocab_path = PathBuf::from("models/tokenizer.json");
    manager.load(TokenizerPlugin::new(vocab_path))?;

    // Load server plugin
    let addr: SocketAddr = "127.0.0.1:9090".parse()?;
    manager.load(ServerPlugin::new(addr))?;

    // Get services
    let backend_svc = ctx.get_service::<BackendService>("backend").unwrap();
    println!("Active backend: {:?}", backend_svc.backend());

    // Subscribe to events
    ctx.on("model/loaded", |event| {
        println!("Model loaded!");
    })?;

    // Emit events
    ctx.emit(Event::new("model/loaded", ()));

    // Cleanup
    manager.unload("server")?;
    manager.unload("tokenizer")?;
    manager.unload("model")?;
    manager.unload("backend")?;

    Ok(())
}
```

---

## Benefits

### 1. Modularity

Each component is isolated and replaceable. Want to add a new backend? Write a plugin. Want to support a new model format? Write a plugin.

### 2. Testability

Plugins can be tested in isolation with mock services and events.

### 3. Extensibility

Third-party plugins can extend Fennec without modifying core code.

### 4. Composability

Plugins can be combined in different configurations for different use cases (server, desktop, embedded).

### 5. Lifecycle Management

Automatic cleanup when plugins unload prevents resource leaks.

---

## Future Work

### Phase 2: Hot-Reloading

- Unload and reload plugins at runtime
- Update backends without restarting the engine
- Swap models without downtime

### Phase 3: Plugin Discovery

- Scan directories for plugin libraries
- Load plugins from shared objects (`.so`/`.dll`/`.dylib`)
- Version compatibility checking

### Phase 4: Plugin Configuration

- Declarative plugin configuration (TOML/YAML)
- Configuration validation
- Default configurations for common setups

### Phase 5: Plugin Marketplace

- Central repository for community plugins
- Plugin signing and verification
- Dependency resolution

---

## Comparison with Cordis

| Feature | Cordis (DeepSeek Harness) | Fennec Plugin System |
|---------|---------------------------|----------------------|
| Language | TypeScript | Rust |
| Service registry | ✓ | ✓ |
| Typed events | ✓ | ✓ |
| Reversible effects | ✓ | ✓ |
| Dependency tracking | ✓ | ✓ |
| Hot-reloading | ✓ | Phase 2 |
| Plugin discovery | ✓ | Phase 3 |
| Configuration | YAML | Phase 4 |

---

## Implementation Status

- [x] Core plugin trait and context
- [x] Service registry
- [x] Event system
- [x] Reversible effects
- [x] Plugin manager
- [x] BackendPlugin
- [x] ModelPlugin
- [x] TokenizerPlugin
- [x] ServerPlugin
- [x] Documentation
- [ ] Hot-reloading (Phase 2)
- [ ] Plugin discovery (Phase 3)
- [ ] Configuration system (Phase 4)

---

## Files Modified

- `src/plugin.rs` - Core plugin infrastructure (Context, Service, Plugin, Event, Effect)
- `src/plugins/mod.rs` - Built-in plugins module
- `src/plugins/backend_plugin.rs` - Backend plugin
- `src/plugins/model_plugin.rs` - Model plugin
- `src/plugins/tokenizer_plugin.rs` - Tokenizer plugin
- `src/plugins/server_plugin.rs` - Server plugin
- `src/lib.rs` - Added `plugin` and `plugins` modules

---

## Testing

All existing tests pass:
- 64 unit tests
- 2 binary tests
- 5 dense lane tests
- 13 golden tests
- 9 GPU tests
- 6 tokenizer tests
- 1 device prefill test

**Total:** 100/100 tests passing

---

## Build Status

```
cargo check
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.79s
```

Zero warnings, zero errors.

---

**Engine Version:** 1.0.0  
**Plugin System Version:** 0.1.0 (Phase 1)  
**Build:** Clean  
**Tests:** 100/100 passing  
**Status:** Ready for Phase 2 (hot-reloading)
