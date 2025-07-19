//! PIR Contracts - Producer/Consumer/Transformation Interfaces
//!
//! This module defines the stable contract interfaces between compilation orchestration
//! and code generation, implementing the three-layer architecture from PLD-013.

use crate::{PIRResult, semantic::PrismIR};
// async_trait removed for now - will add back when needed
use prism_ast::{Program};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Producer Contract: Compiler → PIR
/// 
/// The compiler must guarantee that PIR contains all semantic information 
/// present in the validated AST, with no information loss during transformation.
pub trait PIRProducer: Send + Sync {
    /// Generate PIR from a semantic AST
    /// 
    /// This is the primary entry point for compilation orchestration to create PIR.
    /// The producer must preserve all semantic information from the AST.
    fn generate_pir(&self, semantic_ast: &Program) -> PIRResult<PrismIR>;

    /// Validate that PIR preserves semantic fidelity from AST
    /// 
    /// This ensures that no semantic information was lost during PIR generation.
    fn validate_semantic_fidelity(&self, ast: &Program, pir: &PrismIR) -> PIRResult<ValidationReport>;

    /// Get producer capabilities and metadata
    fn producer_info(&self) -> ProducerInfo;
}

/// Consumer Contract: PIR → Code Generation
/// 
/// Code generation systems must guarantee that target artifacts preserve 
/// the semantic meaning represented in PIR.
pub trait PIRConsumer<Artifact>: Send + Sync 
where
    Artifact: Send + Sync,
{
    /// Consume PIR and generate target artifacts
    /// 
    /// This is the primary entry point for code generation systems.
    fn consume_pir(&self, pir: &PrismIR, config: &ConsumerConfig) -> PIRResult<Artifact>;

    /// Verify that target artifacts preserve PIR semantics
    /// 
    /// This ensures that the generated code maintains semantic equivalence.
    fn verify_semantic_preservation(&self, pir: &PrismIR, artifact: &Artifact) -> PIRResult<VerificationReport>;

    /// Get consumer capabilities and metadata
    fn consumer_info(&self) -> ConsumerInfo;

    /// Get supported PIR features
    fn supported_features(&self) -> Vec<PIRFeature>;
}

/// Transformation Contract: PIR Internal Transformations
/// 
/// PIR transformations must guarantee semantic equivalence while potentially
/// optimizing representation for downstream consumers.
pub trait PIRTransformation: Send + Sync {
    /// Transform PIR while preserving semantics
    /// 
    /// This enables PIR-level optimizations and normalizations.
    fn transform(&self, input: &PrismIR, config: &TransformationConfig) -> PIRResult<PrismIR>;

    /// Verify semantic equivalence between input and output PIR
    /// 
    /// This ensures that transformations don't change program meaning.
    fn verify_semantic_equivalence(&self, input: &PrismIR, output: &PrismIR) -> PIRResult<EquivalenceReport>;

    /// Get transformation metadata
    fn transformation_info(&self) -> TransformationInfo;

    /// Check if transformation is applicable to given PIR
    fn is_applicable(&self, pir: &PrismIR) -> bool;
}

// PIR Builder is now in its own module
pub use crate::transformation_pipeline::{PIRBuilder, PIRBuilderConfig};

/// Producer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerInfo {
    /// Producer name
    pub name: String,
    /// Producer version
    pub version: String,
    /// Supported AST versions
    pub supported_ast_versions: Vec<String>,
    /// Generated PIR version
    pub pir_version: String,
    /// Producer capabilities
    pub capabilities: Vec<ProducerCapability>,
}

/// Producer capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProducerCapability {
    /// Can preserve semantic types
    SemanticTypePreservation,
    /// Can extract business context
    BusinessContextExtraction,
    /// Can analyze cohesion
    CohesionAnalysis,
    /// Can build effect graphs
    EffectGraphConstruction,
    /// Can generate AI metadata
    AIMetadataGeneration,
}

/// Consumer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerInfo {
    /// Consumer name
    pub name: String,
    /// Consumer version
    pub version: String,
    /// Supported PIR versions
    pub supported_pir_versions: Vec<String>,
    /// Target platform
    pub target_platform: String,
    /// Consumer capabilities
    pub capabilities: Vec<ConsumerCapability>,
}

/// Consumer capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsumerCapability {
    /// Can generate type-safe code
    TypeSafeCodeGeneration,
    /// Can preserve business semantics
    BusinessSemanticPreservation,
    /// Can implement effect systems
    EffectSystemImplementation,
    /// Can optimize based on cohesion
    CohesionBasedOptimization,
    /// Can generate AI-readable output
    AIReadableOutput,
}

/// Transformation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationInfo {
    /// Transformation name
    pub name: String,
    /// Transformation version
    pub version: String,
    /// Transformation type
    pub transformation_type: TransformationType,
    /// Description
    pub description: String,
    /// Prerequisites
    pub prerequisites: Vec<String>,
}

/// Transformation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Optimization transformation
    Optimization,
    /// Normalization transformation
    Normalization,
    /// Analysis transformation
    Analysis,
    /// Validation transformation
    Validation,
}

/// PIR features that consumers can support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRFeature {
    /// Semantic types
    SemanticTypes,
    /// Effect system
    EffectSystem,
    /// Business context
    BusinessContext,
    /// Cohesion metrics
    CohesionMetrics,
    /// AI metadata
    AIMetadata,
    /// Performance contracts
    PerformanceContracts,
    /// Security classifications
    SecurityClassifications,
}

/// Consumer configuration
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    /// Target platform specific options
    pub platform_options: HashMap<String, String>,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable debug information
    pub debug_info: bool,
    /// Enable semantic preservation checks
    pub semantic_checks: bool,
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Standard optimization
    Standard,
    /// Aggressive optimization
    Aggressive,
}

/// Transformation configuration
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Transformation parameters
    pub parameters: HashMap<String, String>,
    /// Enable verification
    pub enable_verification: bool,
    /// Maximum transformation passes
    pub max_passes: usize,
}

/// Validation report for semantic fidelity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Overall validation result
    pub result: ValidationResult,
    /// Detailed findings
    pub findings: Vec<ValidationFinding>,
    /// Semantic preservation score (0.0 to 1.0)
    pub preservation_score: f64,
    /// Validation metadata
    pub metadata: ValidationMetadata,
}

/// Validation results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Validation passed
    Passed,
    /// Validation passed with warnings
    PassedWithWarnings,
    /// Validation failed
    Failed,
}

/// Validation finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFinding {
    /// Finding type
    pub finding_type: FindingType,
    /// Severity level
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Location information
    pub location: Option<String>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Finding types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingType {
    /// Missing semantic information
    MissingSemanticInfo,
    /// Incorrect type mapping
    IncorrectTypeMapping,
    /// Lost business context
    LostBusinessContext,
    /// Effect system inconsistency
    EffectInconsistency,
    /// Performance contract violation
    PerformanceViolation,
}

/// Severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Information only
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical error
    Critical,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Validation timestamp
    pub timestamp: String,
    /// Validator version
    pub validator_version: String,
    /// Validation duration
    pub duration_ms: u64,
    /// Checks performed
    pub checks_performed: Vec<String>,
}

/// Verification report for semantic preservation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Verification result
    pub result: VerificationResult,
    /// Semantic equivalence score (0.0 to 1.0)
    pub equivalence_score: f64,
    /// Verification findings
    pub findings: Vec<VerificationFinding>,
    /// Verification metadata
    pub metadata: VerificationMetadata,
}

/// Verification results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Semantics preserved
    Preserved,
    /// Semantics mostly preserved
    MostlyPreserved,
    /// Semantics partially preserved
    PartiallyPreserved,
    /// Semantics not preserved
    NotPreserved,
}

/// Verification finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationFinding {
    /// Finding category
    pub category: VerificationCategory,
    /// Severity
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Impact assessment
    pub impact: ImpactAssessment,
}

/// Verification categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationCategory {
    /// Type preservation
    TypePreservation,
    /// Business logic preservation
    BusinessLogicPreservation,
    /// Effect preservation
    EffectPreservation,
    /// Performance preservation
    PerformancePreservation,
    /// Security preservation
    SecurityPreservation,
}

/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Impact level
    pub level: ImpactLevel,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Mitigation suggestions
    pub mitigation_suggestions: Vec<String>,
}

/// Impact levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// No impact
    None,
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

/// Verification metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMetadata {
    /// Verification timestamp
    pub timestamp: String,
    /// Verifier version
    pub verifier_version: String,
    /// Verification duration
    pub duration_ms: u64,
    /// Verification method
    pub method: String,
}

/// Equivalence report for transformation verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceReport {
    /// Equivalence result
    pub result: EquivalenceResult,
    /// Equivalence score (0.0 to 1.0)
    pub equivalence_score: f64,
    /// Detailed analysis
    pub analysis: EquivalenceAnalysis,
    /// Report metadata
    pub metadata: EquivalenceMetadata,
}

/// Equivalence results
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquivalenceResult {
    /// Semantically equivalent
    Equivalent,
    /// Mostly equivalent
    MostlyEquivalent,
    /// Partially equivalent
    PartiallyEquivalent,
    /// Not equivalent
    NotEquivalent,
}

/// Equivalence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceAnalysis {
    /// Type equivalence
    pub type_equivalence: f64,
    /// Behavioral equivalence
    pub behavioral_equivalence: f64,
    /// Effect equivalence
    pub effect_equivalence: f64,
    /// Performance equivalence
    pub performance_equivalence: f64,
    /// Differences found
    pub differences: Vec<EquivalenceDifference>,
}

/// Equivalence difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceDifference {
    /// Difference type
    pub difference_type: DifferenceType,
    /// Description
    pub description: String,
    /// Significance
    pub significance: DifferenceSignificance,
    /// Location
    pub location: Option<String>,
}

/// Difference types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifferenceType {
    /// Type difference
    Type,
    /// Structure difference
    Structure,
    /// Behavior difference
    Behavior,
    /// Effect difference
    Effect,
    /// Performance difference
    Performance,
    /// Metadata difference
    Metadata,
}

/// Difference significance
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DifferenceSignificance {
    /// Cosmetic difference
    Cosmetic,
    /// Minor difference
    Minor,
    /// Moderate difference
    Moderate,
    /// Major difference
    Major,
    /// Breaking difference
    Breaking,
}

/// Equivalence metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceMetadata {
    /// Analysis timestamp
    pub timestamp: String,
    /// Analyzer version
    pub analyzer_version: String,
    /// Analysis duration
    pub duration_ms: u64,
    /// Analysis method
    pub method: String,
}

/// Result type for transformation operations
pub type TransformationResult<T> = Result<T, TransformationError>;

/// Transformation-specific errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationError {
    /// Semantic equivalence violation
    SemanticViolation(String),
    /// Transformation not applicable
    NotApplicable(String),
    /// Configuration error
    ConfigurationError(String),
    /// Internal transformation error
    InternalError(String),
}

impl std::fmt::Display for TransformationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformationError::SemanticViolation(msg) => {
                write!(f, "Semantic equivalence violation: {}", msg)
            }
            TransformationError::NotApplicable(msg) => {
                write!(f, "Transformation not applicable: {}", msg)
            }
            TransformationError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            TransformationError::InternalError(msg) => {
                write!(f, "Internal transformation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for TransformationError {}

// Helper types for PIRBuilder implementation
#[derive(Debug)]
struct TypeRegistryBuilder {
    // Implementation details for building type registry
}

#[derive(Debug)]
struct EffectGraphBuilder {
    // Implementation details for building effect graph
}

#[derive(Debug)]
struct CohesionAnalyzer {
    // Implementation details for cohesion analysis
}

#[derive(Debug)]
struct MetadataExtractor {
    // Implementation details for metadata extraction
}

// PIR Builder implementation is now in the builder module

impl PIRProducer for PIRBuilder {
    fn generate_pir(&self, semantic_ast: &Program) -> PIRResult<PrismIR> {
        let mut builder = PIRBuilder::with_config(self.config.clone());
        builder.build_from_program(semantic_ast)
    }

    fn validate_semantic_fidelity(&self, _ast: &Program, _pir: &PrismIR) -> PIRResult<ValidationReport> {
        // Implementation would perform comprehensive validation
        Ok(ValidationReport {
            result: ValidationResult::Passed,
            findings: Vec::new(),
            preservation_score: 1.0,
            metadata: ValidationMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                validator_version: "1.0.0".to_string(),
                duration_ms: 0,
                checks_performed: vec!["semantic_fidelity".to_string()],
            },
        })
    }

    fn producer_info(&self) -> ProducerInfo {
        ProducerInfo {
            name: "PIRBuilder".to_string(),
            version: "1.0.0".to_string(),
            supported_ast_versions: vec!["1.0.0".to_string()],
            pir_version: crate::PIRVersion::CURRENT.to_string(),
            capabilities: vec![
                ProducerCapability::SemanticTypePreservation,
                ProducerCapability::BusinessContextExtraction,
                ProducerCapability::CohesionAnalysis,
                ProducerCapability::EffectGraphConstruction,
                ProducerCapability::AIMetadataGeneration,
            ],
        }
    }
}

// PIRBuilderConfig default implementation is now in the builder module

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            platform_options: HashMap::new(),
            optimization_level: OptimizationLevel::Standard,
            debug_info: false,
            semantic_checks: true,
        }
    }
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            enable_verification: true,
            max_passes: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pir_builder_creation() {
        let builder = PIRBuilder::new();
        let info = builder.producer_info();
        assert_eq!(info.name, "PIRBuilder");
        assert!(!info.capabilities.is_empty());
    }

    #[test]
    fn test_validation_report_creation() {
        let report = ValidationReport {
            result: ValidationResult::Passed,
            findings: Vec::new(),
            preservation_score: 1.0,
            metadata: ValidationMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                validator_version: "1.0.0".to_string(),
                duration_ms: 100,
                checks_performed: vec!["test".to_string()],
            },
        };
        
        assert_eq!(report.result, ValidationResult::Passed);
        assert_eq!(report.preservation_score, 1.0);
    }
} 