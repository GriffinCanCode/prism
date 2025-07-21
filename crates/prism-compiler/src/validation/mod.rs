//! Validation Orchestration Subsystem
//!
//! This subsystem implements cross-target validation orchestration for the Prism compiler,
//! following proper Separation of Concerns by delegating to existing validators rather than
//! duplicating validation logic.
//!
//! ## Design Principles
//!
//! 1. **Pure Orchestration**: Coordinates existing validators, never duplicates logic
//! 2. **Delegation Pattern**: Uses existing validators from specialized crates
//! 3. **Result Aggregation**: Combines validation results from multiple sources
//! 4. **Clear Boundaries**: Compiler orchestrates, domain experts validate
//!
//! ## Architecture
//!
//! ```
//! validation/
//! ├── mod.rs              # Public API and orchestration
//! ├── orchestrator.rs     # Main validation orchestrator
//! ├── coordinator.rs      # Cross-target result coordination
//! ├── reporting.rs        # Validation report generation
//! └── integration.rs      # Integration with existing validators
//! ```
//!
//! ## Existing Validators Used
//!
//! - **`prism-pir::PIRValidator`**: PIR semantic preservation
//! - **`prism-semantic::SemanticValidator`**: Business rule validation
//! - **`prism-constraints::ConstraintEngine`**: Constraint validation
//! - **`prism-effects::EffectValidator`**: Effect system validation
//! - **`prism-codegen::CodeGenBackend::validate()`**: Target-specific validation
//!
//! ## Usage
//!
//! ```rust
//! use prism_compiler::validation::ValidationOrchestrator;
//!
//! let orchestrator = ValidationOrchestrator::new()?;
//! let report = orchestrator.validate_cross_target_consistency(pir, artifacts).await?;
//! ```

pub mod orchestrator;
pub mod coordinator;
pub mod reporting;
pub mod integration;

// Re-export main types for convenience
pub use orchestrator::{ValidationOrchestrator, ValidationOrchestratorConfig};
pub use coordinator::{ValidationCoordinator, ConsistencyAnalysis};
pub use reporting::{ValidationReport, ValidationSummary, ValidationRecommendation};
pub use integration::{ExistingValidatorRefs, ValidationDelegation, ValidatorIntegrationBuilder};

use crate::error::{CompilerError, CompilerResult};
use crate::context::CompilationTarget;
use prism_pir::PrismIR;
use prism_codegen::CodeArtifact;
use serde::{Serialize, Deserialize};

/// Main validation orchestration interface
pub trait ValidationOrchestration {
    /// Orchestrate cross-target validation using existing validators
    async fn validate_cross_target_consistency(
        &self,
        pir: &PrismIR,
        artifacts: &[(&CompilationTarget, &CodeArtifact)],
    ) -> CompilerResult<ValidationReport>;
    
    /// Get validation capabilities
    fn get_validation_capabilities(&self) -> ValidationCapabilities;
}

/// Validation orchestration capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCapabilities {
    /// Supported validation types
    pub supported_types: Vec<ValidationType>,
    /// Available target validators
    pub target_validators: Vec<CompilationTarget>,
    /// Integration status with existing validators
    pub integrations: ValidationIntegrationStatus,
}

/// Types of validation supported
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationType {
    /// PIR semantic preservation (delegated to prism-pir)
    SemanticPreservation,
    /// Business rule consistency (delegated to prism-semantic)
    BusinessRuleConsistency,
    /// Type constraint validation (delegated to prism-constraints)
    ConstraintValidation,
    /// Effect system validation (delegated to prism-effects)
    EffectValidation,
    /// Cross-target consistency (orchestrated)
    CrossTargetConsistency,
}

/// Integration status with existing validators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIntegrationStatus {
    /// PIR validator integration
    pub pir_validator: bool,
    /// Semantic validator integration
    pub semantic_validator: bool,
    /// Constraint engine integration
    pub constraint_engine: bool,
    /// Effect validator integration
    pub effect_validator: bool,
    /// Codegen backend validators integration
    pub codegen_validators: Vec<CompilationTarget>,
} 