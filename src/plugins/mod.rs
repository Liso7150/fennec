//! Built-in plugins for Fennec engine.
//!
//! These plugins wrap the existing components (backend, model, server, etc.)
//! to make them pluggable and composable.

pub mod backend_plugin;
pub mod cache_clock_plugin;
pub mod effort_optimizer_plugin;
pub mod handoff_plugin;
pub mod image_preprocess_plugin;
pub mod input_router_plugin;
pub mod model_plugin;
pub mod model_router_plugin;
pub mod research_router_plugin;
pub mod server_plugin;
pub mod tokenizer_plugin;

pub use backend_plugin::BackendPlugin;
pub use cache_clock_plugin::CacheClockPlugin;
pub use effort_optimizer_plugin::EffortOptimizerPlugin;
pub use handoff_plugin::HandoffPlugin;
pub use image_preprocess_plugin::ImagePreprocessPlugin;
pub use input_router_plugin::InputRouterPlugin;
pub use model_plugin::ModelPlugin;
pub use model_router_plugin::ModelRouterPlugin;
pub use research_router_plugin::ResearchRouterPlugin;
pub use server_plugin::ServerPlugin;
pub use tokenizer_plugin::TokenizerPlugin;
