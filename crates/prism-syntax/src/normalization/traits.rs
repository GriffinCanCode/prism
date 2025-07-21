//! Common traits for style-specific normalizers.
//!
//! This module defines the unified interface that all style-specific normalizers
//! must implement, maintaining conceptual cohesion around "normalization interface
//! definition and style-agnostic coordination".

use crate::{
    detection::SyntaxStyle,
    normalization::{
        NormalizationContext, NormalizationError, NormalizationWarning, 
        ValidationLevel, WarningSeverity
    },
};
use prism_common::span::Span;
use std::collections::HashMap;

/// Common interface for all style-specific normalizers.
/// 
/// This trait ensures consistency across different syntax styles while
/// allowing for style-specific optimizations and handling. Each normalizer
/// focuses on a single responsibility: converting its specific syntax style
/// to the canonical representation.
/// 
/// # Conceptual Cohesion
/// 
/// This trait maintains conceptual cohesion by defining a clear contract
/// for "syntax-specific normalization with semantic preservation". All
/// implementations must preserve semantic meaning while generating
/// AI-comprehensible metadata.
pub trait StyleNormalizer {
    /// The input syntax type for this normalizer
    type Input;
    
    /// The intermediate representation used during normalization
    type Intermediate;
    
    /// Configuration type for this normalizer
    type Config: Default + Clone;
    
    /// Creates a new normalizer with default configuration.
    fn new() -> Self;
    
    /// Creates a new normalizer with custom configuration.
    fn with_config(config: Self::Config) -> Self;
    
    /// Returns the syntax style this normalizer handles.
    fn syntax_style(&self) -> SyntaxStyle;
    
    /// Normalizes input syntax to canonical form.
    /// 
    /// This method performs the core normalization work:
    /// 1. Converts syntax-specific structures to canonical representation
    /// 2. Preserves all semantic meaning during conversion
    /// 3. Generates AI-comprehensible metadata
    /// 4. Validates the normalized output
    /// 
    /// # Arguments
    /// 
    /// * `input` - The parsed syntax to normalize
    /// * `context` - Normalization context for tracking state and metrics
    /// 
    /// # Returns
    /// 
    /// The canonical representation or a normalization error.
    fn normalize(
        &self, 
        input: &Self::Input, 
        context: &mut NormalizationContext
    ) -> Result<Self::Intermediate, NormalizationError>;
    
    /// Validates the normalized output for correctness.
    /// 
    /// This method performs style-specific validation to ensure the
    /// normalization was successful and semantically correct.
    fn validate_normalized(
        &self, 
        normalized: &Self::Intermediate, 
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError>;
    
    /// Generates AI metadata for the normalized form.
    /// 
    /// This method creates structured metadata that AI systems can
    /// consume to understand the code's purpose, structure, and behavior.
    fn generate_ai_metadata(
        &self, 
        normalized: &Self::Intermediate, 
        context: &mut NormalizationContext
    ) -> Result<AIMetadata, NormalizationError>;
    
    /// Returns the capabilities and limitations of this normalizer.
    fn capabilities(&self) -> NormalizerCapabilities;
}

/// Configuration trait for normalizer configurations
pub trait NormalizerConfig: Default + Clone {
    /// Validates the configuration for correctness
    fn validate(&self) -> Result<(), ConfigurationError>;
    
    /// Merges this configuration with another, with the other taking precedence
    fn merge_with(&mut self, other: &Self);
}

/// Capabilities and limitations of a normalizer
#[derive(Debug, Clone)]
pub struct NormalizerCapabilities {
    /// Syntax constructs supported by this normalizer
    pub supported_constructs: Vec<String>,
    
    /// Syntax constructs not yet supported
    pub unsupported_constructs: Vec<String>,
    
    /// Whether this normalizer supports error recovery
    pub supports_error_recovery: bool,
    
    /// Whether this normalizer can generate AI metadata
    pub generates_ai_metadata: bool,
    
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
}

/// Performance characteristics of a normalizer
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// Average time complexity (Big O notation as string)
    pub time_complexity: String,
    
    /// Average space complexity (Big O notation as string)
    pub space_complexity: String,
    
    /// Whether the normalizer can process in parallel
    pub supports_parallel_processing: bool,
    
    /// Typical memory usage per node
    pub memory_per_node_bytes: usize,
}

/// AI metadata generated during normalization
#[derive(Debug, Clone)]
pub struct AIMetadata {
    /// Business context identified in the code
    pub business_context: Option<String>,
    
    /// Key domain concepts found
    pub domain_concepts: Vec<String>,
    
    /// Architectural patterns identified
    pub architectural_patterns: Vec<String>,
    
    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,
    
    /// Semantic relationships between elements
    pub semantic_relationships: Vec<SemanticRelationship>,
}

/// Complexity metrics for AI analysis
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: usize,
    
    /// Cognitive complexity
    pub cognitive_complexity: usize,
    
    /// Nesting depth
    pub nesting_depth: usize,
    
    /// Number of dependencies
    pub dependency_count: usize,
}

/// Semantic relationship between code elements
#[derive(Debug, Clone)]
pub struct SemanticRelationship {
    /// Source element
    pub source: String,
    
    /// Target element
    pub target: String,
    
    /// Type of relationship
    pub relationship_type: RelationshipType,
    
    /// Strength of the relationship (0.0 to 1.0)
    pub strength: f64,
}

/// Types of semantic relationships
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    /// One element depends on another
    Dependency,
    
    /// One element composes another
    Composition,
    
    /// One element inherits from another
    Inheritance,
    
    /// One element uses another
    Usage,
    
    /// Elements are associated
    Association,
    
    /// Elements implement the same interface
    Implementation,
}

/// Configuration error types
#[derive(Debug, thiserror::Error)]
pub enum ConfigurationError {
    #[error("Invalid configuration parameter: {parameter} = {value}")]
    InvalidParameter { parameter: String, value: String },
    
    #[error("Missing required configuration: {parameter}")]
    MissingRequired { parameter: String },
    
    #[error("Configuration conflict: {message}")]
    Conflict { message: String },
}

/// Common normalization utilities shared across all normalizers
pub struct NormalizationUtils;

impl NormalizationUtils {
    /// Creates a warning for unsupported constructs
    pub fn unsupported_construct_warning(
        construct: &str, 
        span: Option<Span>
    ) -> NormalizationWarning {
        NormalizationWarning {
            message: format!("Unsupported construct '{}' encountered", construct),
            span,
            severity: WarningSeverity::Warning,
            suggestion: Some(format!(
                "Consider using an alternative approach or filing an issue for '{}' support", 
                construct
            )),
        }
    }
    
    /// Creates a warning for semantic preservation issues
    pub fn semantic_preservation_warning(
        message: &str, 
        span: Option<Span>
    ) -> NormalizationWarning {
        NormalizationWarning {
            message: format!("Semantic preservation issue: {}", message),
            span,
            severity: WarningSeverity::Warning,
            suggestion: Some("Review the normalized output for semantic correctness".to_string()),
        }
    }
    
    /// Creates an AI metadata hint
    pub fn create_ai_hint(
        concept: &str, 
        description: &str, 
        confidence: f64
    ) -> AIHint {
        AIHint {
            concept: concept.to_string(),
            description: description.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
    
    /// Calculates basic complexity metrics for a code block
    pub fn calculate_complexity_metrics(
        statements: usize,
        conditions: usize,
        loops: usize,
        nesting_depth: usize
    ) -> ComplexityMetrics {
        // Basic cyclomatic complexity: conditions + loops + 1
        let cyclomatic_complexity = conditions + loops + 1;
        
        // Simplified cognitive complexity calculation
        let cognitive_complexity = conditions + loops + (nesting_depth * 2);
        
        ComplexityMetrics {
            cyclomatic_complexity,
            cognitive_complexity,
            nesting_depth,
            dependency_count: 0, // Will be calculated separately
        }
    }
}

/// AI hint for enhanced comprehension
#[derive(Debug, Clone)]
pub struct AIHint {
    /// The concept being described
    pub concept: String,
    
    /// Description of the concept
    pub description: String,
    
    /// Confidence in this hint (0.0 to 1.0)
    pub confidence: f64,
}

/// Trait for semantic validation during normalization
pub trait SemanticValidator {
    /// Validates semantic consistency of the normalized form
    fn validate_semantics(
        &self, 
        normalized: &dyn std::any::Any, 
        context: &NormalizationContext
    ) -> Result<Vec<NormalizationWarning>, NormalizationError>;
    
    /// Checks for common semantic issues
    fn check_common_issues(
        &self, 
        normalized: &dyn std::any::Any
    ) -> Vec<NormalizationWarning>;
}

/// Trait for AI metadata generation
pub trait AIMetadataGenerator {
    /// Generates comprehensive AI metadata
    fn generate_metadata(
        &self, 
        normalized: &dyn std::any::Any, 
        context: &NormalizationContext
    ) -> Result<AIMetadata, NormalizationError>;
    
    /// Extracts business concepts from the normalized form
    fn extract_business_concepts(
        &self, 
        normalized: &dyn std::any::Any
    ) -> Vec<String>;
    
    /// Identifies architectural patterns
    fn identify_patterns(
        &self, 
        normalized: &dyn std::any::Any
    ) -> Vec<String>;
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            cyclomatic_complexity: 1,
            cognitive_complexity: 0,
            nesting_depth: 0,
            dependency_count: 0,
        }
    }
}

impl Default for AIMetadata {
    fn default() -> Self {
        Self {
            business_context: None,
            domain_concepts: Vec::new(),
            architectural_patterns: Vec::new(),
            complexity_metrics: ComplexityMetrics::default(),
            semantic_relationships: Vec::new(),
        }
    }
}

impl Default for PerformanceCharacteristics {
    fn default() -> Self {
        Self {
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(n)".to_string(),
            supports_parallel_processing: false,
            memory_per_node_bytes: 256, // Reasonable default
        }
    }
}

impl Default for NormalizerCapabilities {
    fn default() -> Self {
        Self {
            supported_constructs: Vec::new(),
            unsupported_constructs: Vec::new(),
            supports_error_recovery: true,
            generates_ai_metadata: true,
            performance_characteristics: PerformanceCharacteristics::default(),
        }
    }
} 