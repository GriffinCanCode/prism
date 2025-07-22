//! Modular Python Code Generation Backend
//!
//! This module implements a comprehensive Python backend with proper separation
//! of concerns, following 2025 Python best practices and Prism's design principles.
//!
//! ## Architecture
//!
//! The Python backend is organized into focused modules:
//!
//! - [`core`] - Core Python backend implementation and orchestration
//! - [`types`] - Python type system with 2025 features (type hints, generics, protocols)
//! - [`semantic_preservation`] - Semantic type preservation with modern Python patterns
//! - [`runtime_integration`] - Runtime integration with prism-runtime infrastructure
//! - [`validation`] - Python-specific validation and linting integration (mypy, ruff)
//! - [`optimization`] - Python optimization with performance hints and profiling
//! - [`ast_generation`] - Modern Python AST generation with semantic metadata
//! - [`dataclass_generation`] - Dataclass and Pydantic model generation for semantic types
//! - [`async_support`] - Async/await pattern generation for effects and capabilities
//! - [`packaging`] - Modern Python packaging with pyproject.toml generation
//!
//! ## Design Principles
//!
//! 1. **Conceptual Cohesion**: Each module has a single, clear responsibility
//! 2. **2025 Best Practices**: Uses modern Python features like structural pattern matching, type hints
//! 3. **Semantic Preservation**: Maintains business rules and domain knowledge in generated code
//! 4. **AI-First Metadata**: Rich metadata generation for AI comprehension
//! 5. **Zero-Cost Abstractions**: Semantic richness preserved through type hints and runtime validation
//! 6. **Runtime Integration**: Deep integration with prism-runtime for capability management
//! 7. **Developer Experience**: Excellent debugging support with comprehensive type hints

pub mod core;
pub mod types;
pub mod semantic_preservation;
pub mod runtime_integration;
pub mod validation;
pub mod optimization;
pub mod ast_generation;
pub mod dataclass_generation;
pub mod async_support;
pub mod packaging;

// Re-export the main backend for backward compatibility
pub use core::PythonBackend;

// Re-export commonly used types
pub use types::{PythonType, PythonTypeConverter, PythonFeatures, PythonTarget};
pub use semantic_preservation::{SemanticTypePreserver, BusinessRuleGenerator};
pub use runtime_integration::{RuntimeIntegrator, CapabilityManager, EffectTracker};
pub use validation::{PythonValidator, ValidationConfig, LintingIntegration};
pub use optimization::{PythonOptimizer, OptimizationConfig, PerformanceHints};
pub use ast_generation::{PythonASTGenerator, ASTConfig, ModuleGeneration};
pub use dataclass_generation::{DataclassGenerator, DataclassConfig};
pub use async_support::{AsyncPatternGenerator, AsyncConfig};
pub use packaging::{PackagingGenerator, PyProjectConfig};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Python backend configuration combining all module configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonBackendConfig {
    /// Core configuration
    pub core_config: crate::backends::CodeGenConfig,
    /// Python target version
    pub target: PythonTarget,
    /// Python language features to use
    pub python_features: PythonFeatures,
    /// Type system configuration
    pub type_config: types::PythonTypeConfig,
    /// Semantic preservation configuration
    pub semantic_config: semantic_preservation::SemanticPreservationConfig,
    /// Runtime integration configuration
    pub runtime_config: runtime_integration::RuntimeIntegrationConfig,
    /// Validation configuration
    pub validation_config: validation::ValidationConfig,
    /// Optimization configuration
    pub optimization_config: optimization::OptimizationConfig,
    /// AST generation configuration
    pub ast_config: ast_generation::ASTConfig,
    /// Dataclass generation configuration
    pub dataclass_config: dataclass_generation::DataclassConfig,
    /// Async support configuration
    pub async_config: async_support::AsyncConfig,
    /// Packaging configuration
    pub packaging_config: packaging::PyProjectConfig,
}

impl PythonBackendConfig {
    /// Create configuration from base CodeGenConfig
    pub fn from_codegen_config(config: &crate::backends::CodeGenConfig) -> Self {
        Self {
            core_config: config.clone(),
            target: PythonTarget::default(),
            python_features: PythonFeatures::default(),
            type_config: types::PythonTypeConfig::default(),
            semantic_config: semantic_preservation::SemanticPreservationConfig::default(),
            runtime_config: runtime_integration::RuntimeIntegrationConfig::default(),
            validation_config: validation::ValidationConfig::default(),
            optimization_config: optimization::OptimizationConfig::default(),
            ast_config: ast_generation::ASTConfig::default(),
            dataclass_config: dataclass_generation::DataclassConfig::default(),
            async_config: async_support::AsyncConfig::default(),
            packaging_config: packaging::PyProjectConfig::default(),
        }
    }
}

impl Default for PythonBackendConfig {
    fn default() -> Self {
        Self::from_codegen_config(&crate::backends::CodeGenConfig::default())
    }
}

/// Python backend error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum PythonError {
    #[error("Type conversion error: {message}")]
    TypeConversion { message: String },
    
    #[error("Code generation error: {message}")]
    CodeGeneration { message: String },
    
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    #[error("Runtime integration error: {message}")]
    RuntimeIntegration { message: String },
    
    #[error("AST generation error: {message}")]
    ASTGeneration { message: String },
    
    #[error("Semantic preservation error: {message}")]
    SemanticPreservation { message: String },
    
    #[error("Packaging error: {message}")]
    Packaging { message: String },
    
    #[error("Generic Python backend error: {message}")]
    Generic { message: String },
}

/// Python backend result type
pub type PythonResult<T> = Result<T, PythonError>;

/// Convert PythonError to CodeGenError
impl From<PythonError> for crate::CodeGenError {
    fn from(error: PythonError) -> Self {
        crate::CodeGenError::CodeGenerationError {
            target: "Python".to_string(),
            message: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_backend_config_creation() {
        let config = PythonBackendConfig::default();
        assert_eq!(config.target, PythonTarget::default());
        assert!(config.python_features.type_hints);
        assert!(config.python_features.dataclasses);
    }

    #[test]
    fn test_python_backend_config_from_codegen() {
        let codegen_config = crate::backends::CodeGenConfig {
            optimization_level: 3,
            debug_info: false,
            source_maps: false,
            target_options: HashMap::new(),
            ai_metadata_level: crate::backends::AIMetadataLevel::Comprehensive,
        };
        
        let python_config = PythonBackendConfig::from_codegen_config(&codegen_config);
        assert_eq!(python_config.core_config.optimization_level, 3);
        assert!(!python_config.core_config.debug_info);
        assert_eq!(
            python_config.core_config.ai_metadata_level, 
            crate::backends::AIMetadataLevel::Comprehensive
        );
    }

    #[test]
    fn test_python_error_conversion() {
        let python_error = PythonError::TypeConversion {
            message: "Failed to convert PIR type".to_string(),
        };
        
        let codegen_error: crate::CodeGenError = python_error.into();
        match codegen_error {
            crate::CodeGenError::CodeGenerationError { target, message } => {
                assert_eq!(target, "Python");
                assert!(message.contains("Type conversion error"));
            }
            _ => panic!("Expected CodeGenerationError"),
        }
    }
} 