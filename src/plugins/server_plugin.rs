//! Server plugin: wraps the HTTP server and API endpoints.
//!
//! This plugin makes the HTTP server pluggable, allowing different API
//! implementations (OpenAI-compatible, custom, etc.) to be added.

use crate::plugin::{Context, Event, Plugin, PluginError, Service};
use std::net::SocketAddr;
use std::sync::RwLock;

/// Server service: provides access to the HTTP server state.
pub struct ServerService {
    addr: RwLock<Option<SocketAddr>>,
    running: RwLock<bool>,
}

impl ServerService {
    /// Create a new server service.
    pub fn new() -> Self {
        Self {
            addr: RwLock::new(None),
            running: RwLock::new(false),
        }
    }

    /// Get the server address.
    pub fn addr(&self) -> Option<SocketAddr> {
        *self.addr.read().unwrap()
    }

    /// Set the server address.
    pub fn set_addr(&self, addr: SocketAddr) {
        *self.addr.write().unwrap() = Some(addr);
    }

    /// Check if the server is running.
    pub fn is_running(&self) -> bool {
        *self.running.read().unwrap()
    }

    /// Set the server running state.
    pub fn set_running(&self, running: bool) {
        *self.running.write().unwrap() = running;
    }
}

impl Service for ServerService {
    fn name(&self) -> &'static str {
        "server"
    }
}

impl Default for ServerService {
    fn default() -> Self {
        Self::new()
    }
}

/// Server plugin: manages HTTP server lifecycle.
pub struct ServerPlugin {
    addr: SocketAddr,
}

impl ServerPlugin {
    /// Create a new server plugin with the bind address.
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

impl Plugin for ServerPlugin {
    fn name(&self) -> &'static str {
        "server"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn dependencies(&self) -> Vec<&'static str> {
        vec!["model", "tokenizer"] // Server depends on model and tokenizer
    }

    fn apply(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Register the server service
        let service = ServerService::new();
        service.set_addr(self.addr);
        ctx.register_service("server", service)?;

        // Subscribe to server start/stop events
        ctx.on("server/start", |_event| {
            println!("Server start event received");
        })?;

        ctx.on("server/stop", |_event| {
            println!("Server stop event received");
        })?;

        // Emit server plugin initialized event
        let addr = self.addr;
        ctx.effect(move || {
            println!("Server plugin initialized, will bind to: {}", addr);
        })?;

        Ok(())
    }

    fn unload(&self, ctx: &mut Context) -> Result<(), PluginError> {
        // Clean up server resources
        ctx.emit(Event::new("server/unloading", ()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_server_service() {
        let service = ServerService::new();
        assert!(!service.is_running());
        assert!(service.addr().is_none());

        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9090);
        service.set_addr(addr);
        assert_eq!(service.addr(), Some(addr));
    }
}
