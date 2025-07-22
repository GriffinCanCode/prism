//! Modular JavaScript Code Generation Backend
//!
//! This module implements a comprehensive JavaScript backend with proper separation
//! of concerns, following 2025 JavaScript best practices and Prism's design principles.
//!
//! ## Architecture
//!
//! The JavaScript backend is organized into focused modules:
//!
//! - [`core`] - Core JavaScript backend implementation and orchestration
//! - [`types`] - JavaScript type system with runtime validation and branded types
//! - [`semantic_preservation`] - Semantic type preservation with modern JavaScript patterns
//! - [`runtime_integration`] - Runtime integration with prism-runtime infrastructure
//! - [`validation`] - JavaScript-specific validation and linting integration
//! - [`optimization`] - JavaScript optimization with bundling and minification
//! - [`source_maps`] - Source map generation for debugging support
//! - [`esm_generation`] - Modern ESM-first code generation with tree-shaking
//! - [`performance`] - Performance optimizations and monitoring integration
//!
//! ## Design Principles
//!
//! 1. **Conceptual Cohesion**: Each module has a single, clear responsibility
//! 2. **2025 Best Practices**: Uses modern JavaScript features (ES2024+, async/await, modules)
//! 3. **Semantic Preservation**: Maintains business rules and domain knowledge in generated code
//! 4. **AI-First Metadata**: Rich metadata generation for AI comprehension
//! 5. **Runtime Safety**: Comprehensive runtime validation and type checking
//! 6. **Performance Focused**: Optimized for both development and production environments
//! 7. **Developer Experience**: Excellent debugging support with source maps and clear error messages

pub mod core;
pub mod types;
pub mod semantic_preservation;
pub mod runtime_integration;
pub mod validation;
pub mod optimization;
pub mod source_maps;
pub mod esm;
pub mod performance;

// Re-export the main backend for backward compatibility
pub use core::JavaScriptBackend;

// Re-export commonly used types
pub use types::{JavaScriptType, JavaScriptTypeConverter, JavaScriptFeatures, JavaScriptTarget};
pub use semantic_preservation::{SemanticTypePreserver, BusinessRuleGenerator};
pub use runtime_integration::{RuntimeIntegrator, CapabilityManager, EffectTracker};
pub use validation::{JavaScriptValidator, ValidationConfig};
pub use optimization::{JavaScriptOptimizer, OptimizationConfig};
pub use source_maps::{SourceMapGenerator, SourceMapConfig};
pub use esm::{ESMGenerator, ESMConfig};
pub use performance::{PerformanceMonitor, PerformanceConfig};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JavaScript backend configuration combining all module configurations
#[derive(Debug, Clone)]
pub struct JavaScriptBackendConfig {
    /// Target JavaScript version and features
    pub target: JavaScriptTarget,
    /// Enabled JavaScript features
    pub features: JavaScriptFeatures,
    /// Type system configuration
    pub type_config: types::TypeConfig,
    /// Semantic preservation configuration
    pub semantic_config: semantic_preservation::SemanticConfig,
    /// Runtime integration configuration
    pub runtime_config: runtime_integration::RuntimeConfig,
    /// Validation configuration
    pub validation_config: validation::ValidationConfig,
    /// Optimization configuration
    pub optimization_config: optimization::OptimizationConfig,
    /// Source map configuration
    pub source_map_config: source_maps::SourceMapConfig,
    /// ESM generation configuration
    pub esm_config: esm::ESMConfig,
    /// Performance configuration
    pub performance_config: performance::PerformanceConfig,
}

impl Default for JavaScriptBackendConfig {
    fn default() -> Self {
        Self {
            target: JavaScriptTarget::default(),
            features: JavaScriptFeatures::default(),
            type_config: types::TypeConfig::default(),
            semantic_config: semantic_preservation::SemanticConfig::default(),
            runtime_config: runtime_integration::RuntimeConfig::default(),
            validation_config: validation::ValidationConfig::default(),
            optimization_config: optimization::OptimizationConfig::default(),
            source_map_config: source_maps::SourceMapConfig::default(),
            esm_config: esm::ESMConfig::default(),
            performance_config: performance::PerformanceConfig::default(),
        }
    }
}

/// JavaScript compilation targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JavaScriptTarget {
    /// ES2020 for broader compatibility
    ES2020,
    /// ES2021 with modern features
    ES2021,
    /// ES2022 with class fields and top-level await
    ES2022,
    /// ES2023 with array methods and hashbang
    ES2023,
    /// ES2024 with latest features (default)
    ES2024,
    /// Node.js specific optimizations
    NodeJS,
    /// Browser-specific optimizations
    Browser,
    /// Deno-specific optimizations
    Deno,
}

impl Default for JavaScriptTarget {
    fn default() -> Self {
        Self::ES2024
    }
}

impl std::fmt::Display for JavaScriptTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ES2020 => write!(f, "ES2020"),
            Self::ES2021 => write!(f, "ES2021"),
            Self::ES2022 => write!(f, "ES2022"),
            Self::ES2023 => write!(f, "ES2023"),
            Self::ES2024 => write!(f, "ES2024"),
            Self::NodeJS => write!(f, "Node.js"),
            Self::Browser => write!(f, "Browser"),
            Self::Deno => write!(f, "Deno"),
        }
    }
}

/// JavaScript feature flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JavaScriptFeatures {
    /// Enable ES modules (import/export)
    pub esm: bool,
    /// Enable async/await syntax
    pub async_await: bool,
    /// Enable class syntax
    pub classes: bool,
    /// Enable destructuring
    pub destructuring: bool,
    /// Enable template literals
    pub template_literals: bool,
    /// Enable arrow functions
    pub arrow_functions: bool,
    /// Enable optional chaining
    pub optional_chaining: bool,
    /// Enable nullish coalescing
    pub nullish_coalescing: bool,
    /// Enable private class fields
    pub private_fields: bool,
    /// Enable top-level await
    pub top_level_await: bool,
    /// Enable BigInt support
    pub bigint: bool,
    /// Enable WeakRef and FinalizationRegistry
    pub weak_refs: bool,
    /// Enable Proxy for advanced metaprogramming
    pub proxy: bool,
    /// Enable Symbol for unique identifiers
    pub symbols: bool,
}

impl Default for JavaScriptFeatures {
    fn default() -> Self {
        Self {
            esm: true,
            async_await: true,
            classes: true,
            destructuring: true,
            template_literals: true,
            arrow_functions: true,
            optional_chaining: true,
            nullish_coalescing: true,
            private_fields: true,
            top_level_await: true,
            bigint: true,
            weak_refs: true,
            proxy: true,
            symbols: true,
        }
    }
}

/// JavaScript backend errors
#[derive(Debug, Clone)]
pub enum JavaScriptError {
    /// Code generation error
    CodeGeneration { message: String },
    /// Type conversion error
    TypeConversion { from: String, to: String, reason: String },
    /// Validation error
    Validation { issues: Vec<String> },
    /// Optimization error
    Optimization { message: String },
    /// Runtime integration error
    RuntimeIntegration { message: String },
    /// Source map generation error
    SourceMap { message: String },
    /// ESM generation error
    ESM { message: String },
    /// Performance error
    Performance { message: String },
}

impl std::fmt::Display for JavaScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeGeneration { message } => write!(f, "JavaScript code generation error: {}", message),
            Self::TypeConversion { from, to, reason } => write!(f, "Type conversion error from {} to {}: {}", from, to, reason),
            Self::Validation { issues } => write!(f, "JavaScript validation errors: {}", issues.join(", ")),
            Self::Optimization { message } => write!(f, "JavaScript optimization error: {}", message),
            Self::RuntimeIntegration { message } => write!(f, "Runtime integration error: {}", message),
            Self::SourceMap { message } => write!(f, "Source map generation error: {}", message),
            Self::ESM { message } => write!(f, "ESM generation error: {}", message),
            Self::Performance { message } => write!(f, "Performance optimization error: {}", message),
        }
    }
}

impl std::error::Error for JavaScriptError {}

/// Result type for JavaScript backend operations
pub type JavaScriptResult<T> = Result<T, JavaScriptError>; 