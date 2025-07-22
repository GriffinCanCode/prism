//! Modular TypeScript Code Generation Backend
//!
//! This module implements a comprehensive TypeScript backend with proper separation
//! of concerns, following 2025 TypeScript best practices and Prism's design principles.
//!
//! ## Architecture
//!
//! The TypeScript backend is organized into focused modules:
//!
//! - [`core`] - Core TypeScript backend implementation and orchestration
//! - [`types`] - TypeScript type system with 2025 features (branded types, template literals)
//! - [`semantic_preservation`] - Semantic type preservation with modern TypeScript patterns
//! - [`runtime_integration`] - Runtime integration with prism-runtime infrastructure
//! - [`validation`] - TypeScript-specific validation and linting integration
//! - [`optimization`] - TypeScript optimization with tree-shaking and bundling
//! - [`source_maps`] - Enhanced source map generation for debugging support
//! - [`esm_generation`] - Modern ESM-first code generation
//! - [`branded_types`] - Branded type system implementation
//! - [`template_literals`] - Template literal type generation
//!
//! ## Design Principles
//!
//! 1. **Conceptual Cohesion**: Each module has a single, clear responsibility
//! 2. **2025 Best Practices**: Uses modern TypeScript features like `satisfies`, template literals
//! 3. **Semantic Preservation**: Maintains business rules and domain knowledge in generated code
//! 4. **AI-First Metadata**: Rich metadata generation for AI comprehension
//! 5. **Zero-Cost Abstractions**: Semantic richness compiles away to efficient JavaScript
//! 6. **Runtime Integration**: Deep integration with prism-runtime for capability management
//! 7. **Developer Experience**: Excellent debugging support with source maps and type safety

pub mod core;
pub mod types;
pub mod semantic_preservation;
pub mod runtime_integration;
pub mod validation;
pub mod optimization;
pub mod source_maps;
pub mod esm_generation;
pub mod branded_types;
pub mod template_literals;

// Re-export the main backend for backward compatibility
pub use core::TypeScriptBackend;

// Re-export commonly used types
pub use types::{TypeScriptType, TypeScriptTypeConverter, TypeScriptFeatures, TypeScriptTarget};
pub use semantic_preservation::{SemanticTypePreserver, BusinessRuleGenerator};
pub use runtime_integration::{RuntimeIntegrator, CapabilityManager, EffectTracker};
pub use validation::{TypeScriptValidator, ValidationConfig, LintingIntegration};
pub use optimization::{TypeScriptOptimizer, OptimizationConfig, TreeShaking};
pub use source_maps::{SourceMapGenerator, SourceMapConfig, DebuggingSupport};
pub use esm_generation::{ESMGenerator, ESMConfig, ModuleSystem};
pub use branded_types::{BrandedTypeGenerator, BrandingConfig};
pub use template_literals::{TemplateLiteralGenerator, TemplateConfig};

/// TypeScript backend configuration combining all module configurations
#[derive(Debug, Clone)]
pub struct TypeScriptBackendConfig {
    /// Core backend configuration
    pub core_config: crate::backends::CodeGenConfig,
    /// TypeScript target version and features
    pub typescript_features: TypeScriptFeatures,
    /// TypeScript compilation target
    pub target: TypeScriptTarget,
    /// Semantic preservation configuration
    pub semantic_config: semantic_preservation::SemanticPreservationConfig,
    /// Runtime integration configuration
    pub runtime_config: runtime_integration::RuntimeIntegrationConfig,
    /// Validation configuration
    pub validation_config: validation::TypeScriptValidationConfig,
    /// Optimization configuration
    pub optimization_config: optimization::TypeScriptOptimizationConfig,
    /// Source map configuration
    pub source_map_config: source_maps::SourceMapConfig,
    /// ESM generation configuration
    pub esm_config: esm_generation::ESMConfig,
    /// Branded types configuration
    pub branding_config: branded_types::BrandingConfig,
    /// Template literal configuration
    pub template_config: template_literals::TemplateConfig,
}

impl Default for TypeScriptBackendConfig {
    fn default() -> Self {
        Self {
            core_config: crate::backends::CodeGenConfig::default(),
            typescript_features: TypeScriptFeatures::default(),
            target: TypeScriptTarget::default(),
            semantic_config: semantic_preservation::SemanticPreservationConfig::default(),
            runtime_config: runtime_integration::RuntimeIntegrationConfig::default(),
            validation_config: validation::TypeScriptValidationConfig::default(),
            optimization_config: optimization::TypeScriptOptimizationConfig::default(),
            source_map_config: source_maps::SourceMapConfig::default(),
            esm_config: esm_generation::ESMConfig::default(),
            branding_config: branded_types::BrandingConfig::default(),
            template_config: template_literals::TemplateConfig::default(),
        }
    }
}

/// Result type for TypeScript backend operations
pub type TypeScriptResult<T> = Result<T, TypeScriptError>;

/// TypeScript backend specific errors
#[derive(Debug, thiserror::Error)]
pub enum TypeScriptError {
    #[error("TypeScript type conversion error: {message}")]
    TypeConversion { message: String },
    
    #[error("TypeScript semantic preservation error: {message}")]
    SemanticPreservation { message: String },
    
    #[error("TypeScript runtime integration error: {message}")]
    RuntimeIntegration { message: String },
    
    #[error("TypeScript validation error: {message}")]
    Validation { message: String },
    
    #[error("TypeScript optimization error: {message}")]
    Optimization { message: String },
    
    #[error("TypeScript source map generation error: {message}")]
    SourceMap { message: String },
    
    #[error("TypeScript ESM generation error: {message}")]
    ESMGeneration { message: String },
    
    #[error("TypeScript branded type generation error: {message}")]
    BrandedType { message: String },
    
    #[error("TypeScript template literal generation error: {message}")]
    TemplateLiteral { message: String },
    
    #[error("General TypeScript backend error: {message}")]
    General { message: String },
}

impl From<TypeScriptError> for crate::CodeGenError {
    fn from(err: TypeScriptError) -> Self {
        crate::CodeGenError::CodeGenerationError {
            target: "TypeScript".to_string(),
            message: err.to_string(),
        }
    }
}

/// TypeScript backend feature configuration
#[derive(Debug, Clone)]
pub struct TypeScriptFeatures {
    /// Use TypeScript 5.x+ satisfies operator
    pub use_satisfies_operator: bool,
    /// Generate template literal types
    pub template_literal_types: bool,
    /// Generate branded types for semantic safety
    pub branded_types: bool,
    /// Use modern ESM imports/exports
    pub esm_modules: bool,
    /// Generate enhanced control flow analysis
    pub enhanced_control_flow: bool,
    /// Use const assertions for immutable types
    pub const_assertions: bool,
    /// Generate utility types for advanced type manipulation
    pub utility_types: bool,
    /// Generate discriminated unions for state management
    pub discriminated_unions: bool,
}

impl Default for TypeScriptFeatures {
    fn default() -> Self {
        Self {
            use_satisfies_operator: true,
            template_literal_types: true,
            branded_types: true,
            esm_modules: true,
            enhanced_control_flow: true,
            const_assertions: true,
            utility_types: true,
            discriminated_unions: true,
        }
    }
}

/// TypeScript compilation target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeScriptTarget {
    /// ES2022 target for modern environments
    ES2022,
    /// ES2023 target with latest features
    ES2023,
    /// ESNext for cutting-edge features
    ESNext,
    /// Node.js 18+ target
    Node18,
    /// Node.js 20+ target
    Node20,
    /// Browser target with wide compatibility
    Browser,
    /// Deno target with TypeScript native support
    Deno,
    /// Bun target with enhanced performance
    Bun,
}

impl Default for TypeScriptTarget {
    fn default() -> Self {
        Self::ES2023 // Default to ES2023 for 2025 best practices
    }
}

impl std::fmt::Display for TypeScriptTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ES2022 => write!(f, "ES2022"),
            Self::ES2023 => write!(f, "ES2023"),
            Self::ESNext => write!(f, "ESNext"),
            Self::Node18 => write!(f, "Node18"),
            Self::Node20 => write!(f, "Node20"),
            Self::Browser => write!(f, "Browser"),
            Self::Deno => write!(f, "Deno"),
            Self::Bun => write!(f, "Bun"),
        }
    }
} 