//! AI Integration - Metadata and Context
//!
//! This module handles AI-comprehensible metadata and context information
//! that enables AI systems to understand and work with PIR effectively.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// AI metadata for PIR components
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Module-level context
    pub module_context: Option<ModuleAIContext>,
    /// Function-level contexts
    pub function_contexts: HashMap<String, FunctionAIContext>,
    /// Type-level contexts
    pub type_contexts: HashMap<String, TypeAIContext>,
    /// Semantic relationships between components
    pub relationships: SemanticRelationships,
    /// Business context information
    pub business_context: Option<crate::business::BusinessContext>,
}

/// Module-level AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleAIContext {
    /// Module purpose and responsibility
    pub purpose: String,
    /// Business capability this module represents
    pub capability: String,
    /// Key concepts and entities
    pub key_concepts: Vec<String>,
    /// Usage patterns and examples
    pub usage_patterns: Vec<UsagePattern>,
    /// Common integration points
    pub integration_points: Vec<IntegrationPoint>,
    /// AI-specific hints and guidance
    pub ai_hints: Vec<String>,
}

/// Function-level AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAIContext {
    /// Function's business purpose
    pub business_purpose: String,
    /// Algorithm description
    pub algorithm_description: Option<String>,
    /// Input-output relationships
    pub io_relationships: Vec<IORelationship>,
    /// Side effects and state changes
    pub side_effects: Vec<SideEffect>,
    /// Usage examples
    pub usage_examples: Vec<UsageExample>,
    /// Performance characteristics
    pub performance_notes: Vec<String>,
    /// Common mistakes to avoid
    pub common_mistakes: Vec<String>,
    /// AI-specific guidance
    pub ai_guidance: Vec<String>,
}

/// Type-level AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAIContext {
    /// Semantic meaning of the type
    pub semantic_meaning: String,
    /// Business domain this type belongs to
    pub business_domain: String,
    /// Invariants and constraints
    pub invariants: Vec<TypeInvariant>,
    /// Valid usage patterns
    pub valid_patterns: Vec<String>,
    /// Invalid usage patterns to avoid
    pub invalid_patterns: Vec<String>,
    /// Conversion and transformation rules
    pub conversion_rules: Vec<ConversionRule>,
    /// AI comprehension hints
    pub comprehension_hints: Vec<String>,
}

/// Usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Code example
    pub example: Option<String>,
    /// When to use this pattern
    pub use_cases: Vec<String>,
    /// When not to use this pattern
    pub anti_patterns: Vec<String>,
}

/// Integration point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPoint {
    /// Integration name
    pub name: String,
    /// External system or component
    pub external_system: String,
    /// Integration type
    pub integration_type: IntegrationType,
    /// Data flow direction
    pub data_flow: DataFlowDirection,
    /// Required protocols or formats
    pub protocols: Vec<String>,
}

/// Integration type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationType {
    /// API integration
    API,
    /// Database integration
    Database,
    /// Message queue integration
    MessageQueue,
    /// File system integration
    FileSystem,
    /// Network service integration
    NetworkService,
    /// Library integration
    Library,
}

/// Data flow direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFlowDirection {
    /// Data flows in
    Input,
    /// Data flows out
    Output,
    /// Bidirectional data flow
    Bidirectional,
}

/// Input-output relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IORelationship {
    /// Input parameter name
    pub input: String,
    /// Output component name
    pub output: String,
    /// Relationship type
    pub relationship_type: IORelationshipType,
    /// Transformation description
    pub transformation: Option<String>,
}

/// I/O relationship type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IORelationshipType {
    /// Direct mapping
    Direct,
    /// Transformed mapping
    Transformed,
    /// Aggregated mapping
    Aggregated,
    /// Filtered mapping
    Filtered,
    /// Computed mapping
    Computed,
}

/// Side effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffect {
    /// Effect type
    pub effect_type: SideEffectType,
    /// Description of the effect
    pub description: String,
    /// Affected resources
    pub affected_resources: Vec<String>,
    /// Conditions under which effect occurs
    pub conditions: Vec<String>,
}

/// Side effect type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SideEffectType {
    /// State modification
    StateModification,
    /// I/O operation
    IOOperation,
    /// Network communication
    NetworkCommunication,
    /// File system access
    FileSystemAccess,
    /// Database operation
    DatabaseOperation,
    /// Logging or monitoring
    Logging,
    /// Caching operation
    Caching,
}

/// Usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    /// Example name
    pub name: String,
    /// Example description
    pub description: String,
    /// Example code
    pub code: String,
    /// Expected outcome
    pub expected_outcome: String,
    /// Context or scenario
    pub context: Option<String>,
}

/// Type invariant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInvariant {
    /// Invariant name
    pub name: String,
    /// Invariant description
    pub description: String,
    /// Formal specification (if available)
    pub specification: Option<String>,
    /// Validation method
    pub validation: Option<String>,
}

/// Conversion rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionRule {
    /// Source type
    pub from_type: String,
    /// Target type
    pub to_type: String,
    /// Conversion method
    pub method: ConversionMethod,
    /// Conditions for conversion
    pub conditions: Vec<String>,
    /// Potential data loss
    pub data_loss: Option<String>,
}

/// Conversion method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversionMethod {
    /// Direct conversion
    Direct,
    /// Lossy conversion
    Lossy,
    /// Validated conversion
    Validated,
    /// Computed conversion
    Computed,
    /// Custom conversion
    Custom(String),
}

/// Semantic relationships
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticRelationships {
    /// Type relationships
    pub type_relationships: HashMap<String, Vec<TypeRelationship>>,
    /// Function relationships
    pub function_relationships: HashMap<String, Vec<FunctionRelationship>>,
    /// Module relationships
    pub module_relationships: HashMap<String, Vec<ModuleRelationship>>,
}

/// Type relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationship {
    /// Related type name
    pub related_type: String,
    /// Relationship kind
    pub relationship_kind: TypeRelationshipKind,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Relationship description
    pub description: Option<String>,
}

/// Type relationship kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeRelationshipKind {
    /// Inheritance relationship
    Inherits,
    /// Composition relationship
    Composes,
    /// Association relationship
    Associates,
    /// Dependency relationship
    Depends,
    /// Implementation relationship
    Implements,
    /// Similarity relationship
    Similar,
}

/// Function relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRelationship {
    /// Related function name
    pub related_function: String,
    /// Relationship kind
    pub relationship_kind: FunctionRelationshipKind,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Relationship description
    pub description: Option<String>,
}

/// Function relationship kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionRelationshipKind {
    /// Calls relationship
    Calls,
    /// Called by relationship
    CalledBy,
    /// Composes with relationship
    ComposesWith,
    /// Alternative to relationship
    AlternativeTo,
    /// Helper for relationship
    HelperFor,
    /// Overrides relationship
    Overrides,
}

/// Module relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRelationship {
    /// Related module name
    pub related_module: String,
    /// Relationship kind
    pub relationship_kind: ModuleRelationshipKind,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Relationship description
    pub description: Option<String>,
}

/// Module relationship kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleRelationshipKind {
    /// Imports relationship
    Imports,
    /// Exports to relationship
    ExportsTo,
    /// Collaborates with relationship
    CollaboratesWith,
    /// Extends relationship
    Extends,
    /// Aggregates relationship
    Aggregates,
    /// Similar purpose relationship
    SimilarPurpose,
}

impl AIMetadata {
    /// Set module context
    pub fn set_module_context(&mut self, context: ModuleAIContext) {
        self.module_context = Some(context);
    }

    /// Add function context
    pub fn add_function_context(&mut self, function_name: String, context: FunctionAIContext) {
        self.function_contexts.insert(function_name, context);
    }

    /// Add type context
    pub fn add_type_context(&mut self, type_name: String, context: TypeAIContext) {
        self.type_contexts.insert(type_name, context);
    }

    /// Set business context
    pub fn set_business_context(&mut self, context: crate::business::BusinessContext) {
        self.business_context = Some(context);
    }
} 