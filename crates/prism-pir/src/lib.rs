//! Prism Intermediate Representation (PIR) - Semantic Bridge
//!
//! This crate implements PIR as the pure semantic bridge between compilation orchestration
//! and code generation, embodying Prism's core principle of Conceptual Cohesion by focusing
//! solely on intermediate representation concerns.
//!
//! ## Conceptual Purpose
//!
//! PIR serves a single, clear purpose: **the faithful preservation and transformation of 
//! Prism's semantic richness into a target-agnostic intermediate form**. This crate is the
//! authoritative contract between:
//!
//! - **Upstream**: Compilation orchestration (prism-compiler) 
//! - **Downstream**: Code generation systems (prism-codegen)
//!
//! ## Design Principles
//!
//! 1. **Semantic Fidelity**: Preserve all semantic information without loss
//! 2. **Target Agnosticism**: No assumptions about target platforms or runtime
//! 3. **Stable Contracts**: Versioned interfaces enabling independent evolution
//! 4. **AI Comprehensible**: Clear relationships and well-defined transformations
//! 5. **Business Context Preservation**: Maintain conceptual cohesion from smart modules
//! 6. **Effect Transparency**: Explicit representation of all computational effects
//!
//! ## Architecture
//!
//! The PIR crate is organized around five core conceptual modules:
//!
//! - [`types`] - Core PIR type system and semantic representation
//! - [`contracts`] - Producer/Consumer/Transformation interfaces
//! - [`validation`] - Semantic preservation and invariant checking
//! - [`transformations`] - PIR-to-PIR transformations with audit trails
//! - [`metadata`] - AI context and business capability preservation
//!
//! ## Usage
//!
//! ```rust
//! use prism_pir::{PrismIR, PIRBuilder, PIRValidator, SemanticPreservationCheck};
//!
//! // Producer: Compiler generates PIR from AST
//! let mut builder = PIRBuilder::new();
//! let pir = builder.build_from_ast(semantic_ast)?;
//!
//! // Validate semantic fidelity
//! let validator = PIRValidator::new();
//! validator.validate_semantic_preservation(&pir)?;
//!
//! // Consumer: Code generation consumes PIR
//! let artifacts = codegen_backend.generate_from_pir(&pir)?;
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Business logic modules organized by domain
pub mod semantic;
pub mod business;
pub mod ai_integration;
pub mod quality;
pub mod optimization;  // PIR optimization and normalization utilities
pub mod construction;  // NEW: Query-based construction subsystem

// Legacy modules (to be migrated or removed)
pub mod contracts;
pub mod validation;
pub mod effects;

// Re-export main types from new modular structure
pub use semantic::{
    PrismIR, PIRModule, PIRFunction, PIRSemanticType, PIRExpression,
    SemanticTypeRegistry, EffectGraph, CohesionMetrics, PIRMetadata,
};

pub use business::{
    BusinessContext, BusinessEntity, BusinessRelationship, BusinessConstraint,
};

pub use ai_integration::{
    AIMetadata, ModuleAIContext, FunctionAIContext, TypeAIContext,
    SemanticRelationships,
};

pub use quality::{
    PerformanceProfile, CPUUsageProfile, MemoryUsageProfile, IOProfile,
    NetworkProfile, ScalabilityProfile, TransformationHistory,
};

// Main PIR construction interface - use new construction subsystem
pub use construction::{
    PIRConstructionBuilder, ConstructionConfig, ConstructionResult,
    ASTToPIRQuery, PIRModuleQuery, PIRFunctionQuery, PIRTypeQuery,
    transformation::{ASTToPIRTransformer, SemanticContextExtractor, BusinessContextExtractorPlaceholder, 
                    AIMetadataExtractorPlaceholder, TransformationContext, TransformationDiagnostic},
    compiler_integration::{PIRConstructionQuery, PIRConstructionOrchestrator, PIRConstructionInput,
                          PIRCompilerIntegration, CompilerIntegrationConfig},
    semantic_preservation::{SemanticPreservationValidator, PreservationConfig, PreservationResult},
    business_extraction::{BusinessContextExtractor, BusinessExtractionConfig},
    effect_integration::{EffectSystemIntegrator, EffectIntegrationConfig, EffectIntegrationResult},
    ai_extraction::{AIMetadataExtractor as AIExtractor, AIExtractionConfig, AIExtractionResult},
};

// Backward compatibility aliases
pub type PIRBuilder = PIRConstructionBuilder;
pub type PIRBuilderConfig = ConstructionConfig;

// PIR optimization utilities
pub use optimization::{
    PIROptimizer, PIRNormalizer, TransformationUtils,
    OptimizerConfig, NormalizerConfig, OptimizationLevel,
};

// Legacy re-exports for compatibility (will be removed)
pub use contracts::{
    PIRProducer, PIRConsumer, PIRTransformation,
    TransformationResult,
};

pub use validation::{
    PIRValidator, SemanticPreservationCheck, ValidationResult,
    SemanticInvariant, PreservationRule,
};

pub use effects::{
    EffectCategory, EffectSystemBuilder, CapabilitySystem, ObjectCapability,
    SupplyChainPolicy, TrustLevel,
};

// Error types
use thiserror::Error;

/// PIR-specific error types
#[derive(Error, Debug)]
pub enum PIRError {
    /// Semantic preservation violation
    #[error("Semantic preservation violation: {message} at {location}")]
    SemanticViolation {
        /// Error message
        message: String,
        /// Location information
        location: String,
    },

    /// Invalid transformation
    #[error("Invalid PIR transformation: {operation} - {reason}")]
    InvalidTransformation {
        /// Transformation operation
        operation: String,
        /// Reason for failure
        reason: String,
    },

    /// Contract violation
    #[error("PIR contract violation: {contract} - {details}")]
    ContractViolation {
        /// Contract name
        contract: String,
        /// Violation details
        details: String,
    },

    /// Validation failure
    #[error("PIR validation failed: {check} - {message}")]
    ValidationFailure {
        /// Validation check name
        check: String,
        /// Failure message
        message: String,
    },

    /// Serialization error
    #[error("PIR serialization error: {source}")]
    SerializationError {
        /// Source error
        #[from]
        source: serde_json::Error,
    },

    /// Internal error
    #[error("Internal PIR error: {message}")]
    Internal {
        /// Error message
        message: String,
    },
}

/// Result type for PIR operations
pub type PIRResult<T> = Result<T, PIRError>;

/// PIR version information for compatibility tracking
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PIRVersion {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (additive changes)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl PIRVersion {
    /// Current PIR version
    pub const CURRENT: PIRVersion = PIRVersion {
        major: 1,
        minor: 0,
        patch: 0,
    };

    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &PIRVersion) -> bool {
        // Same major version required for compatibility
        self.major == other.major && self.minor >= other.minor
    }
}

impl std::fmt::Display for PIRVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// PIR configuration for customizing behavior
#[derive(Debug, Clone)]
pub struct PIRConfig {
    /// Enable semantic validation
    pub enable_validation: bool,
    /// Enable transformation auditing
    pub enable_audit_trail: bool,
    /// Enable AI metadata preservation
    pub enable_ai_metadata: bool,
    /// Enable performance profiling
    pub enable_performance_tracking: bool,
    /// Maximum transformation depth
    pub max_transformation_depth: usize,
    /// Validation strictness level
    pub validation_strictness: ValidationStrictness,
}

/// Validation strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStrictness {
    /// Permissive validation
    Permissive,
    /// Standard validation
    Standard,
    /// Strict validation
    Strict,
    /// Pedantic validation
    Pedantic,
}

impl Default for PIRConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_audit_trail: true,
            enable_ai_metadata: true,
            enable_performance_tracking: false,
            max_transformation_depth: 100,
            validation_strictness: ValidationStrictness::Standard,
        }
    }
}

/// PIR statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct PIRStatistics {
    /// Number of modules processed
    pub modules_processed: usize,
    /// Number of types processed
    pub types_processed: usize,
    /// Number of functions processed
    pub functions_processed: usize,
    /// Number of transformations applied
    pub transformations_applied: usize,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
    /// Memory usage peak
    pub peak_memory_usage: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pir_version_compatibility() {
        let v1_0_0 = PIRVersion { major: 1, minor: 0, patch: 0 };
        let v1_1_0 = PIRVersion { major: 1, minor: 1, patch: 0 };
        let v2_0_0 = PIRVersion { major: 2, minor: 0, patch: 0 };

        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
    }

    #[test]
    fn test_pir_config_defaults() {
        let config = PIRConfig::default();
        assert!(config.enable_validation);
        assert!(config.enable_audit_trail);
        assert!(config.enable_ai_metadata);
        assert_eq!(config.validation_strictness, ValidationStrictness::Standard);
    }
} 