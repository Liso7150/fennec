//! Built-in plugins for Fennec engine.
//!
//! These plugins wrap the existing components (backend, model, server, etc.)
//! to make them pluggable and composable.

pub mod backend_plugin;
pub mod model_plugin;
pub mod server_plugin;
pub mod tokenizer_plugin;

pub use backend_plugin::BackendPlugin;
pub use model_plugin::ModelPlugin;
pub use server_plugin::ServerPlugin;
pub use tokenizer_plugin::TokenizerPlugin;
