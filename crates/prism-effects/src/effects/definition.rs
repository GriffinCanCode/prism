//! Core effect definitions and hierarchies
//!
//! This module implements the hierarchical effect categories from PLD-003,
//! providing semantic-rich effect types that are AI-comprehensible and
//! enable fine-grained security control.

use prism_common::span::Span;
use prism_ast::{AstNode, Expr, Type, SecurityClassification};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Registry of all known effects in the system
#[derive(Debug, Default)]
pub struct EffectRegistry {
    /// Map of effect names to their definitions
    pub effects: HashMap<String, EffectDefinition>,
    /// Hierarchical organization of effects
    pub hierarchy: EffectHierarchy,
    /// Effect composition rules
    pub composition_rules: Vec<EffectCompositionRule>,
}

impl EffectRegistry {
    /// Create a new effect registry with built-in effects
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_effects();
        registry
    }

    /// Register a new effect definition
    pub fn register(&mut self, effect: EffectDefinition) -> Result<(), crate::EffectSystemError> {
        if self.effects.contains_key(&effect.name) {
            return Err(crate::EffectSystemError::EffectAlreadyRegistered {
                name: effect.name.clone(),
            });
        }
        
        self.hierarchy.add_effect(&effect);
        self.effects.insert(effect.name.clone(), effect);
        Ok(())
    }

    /// Get an effect definition by name
    pub fn get_effect(&self, name: &str) -> Option<&EffectDefinition> {
        self.effects.get(name)
    }

    /// Check if an effect is a subtype of another
    pub fn is_subeffect(&self, child: &str, parent: &str) -> bool {
        self.hierarchy.is_subeffect(child, parent)
    }

    /// Register all built-in effects from PLD-003
    fn register_builtin_effects(&mut self) {
        // IO Effects
        let io_effects = vec![
            EffectDefinition::new(
                "IO.FileSystem.Read".to_string(),
                "Read files from the file system".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Provides read access to file system resources")
             .with_security_implication("Can access sensitive file contents")
             .with_capability_requirement("FileSystem", vec!["Read".to_string()]),

            EffectDefinition::new(
                "IO.FileSystem.Write".to_string(),
                "Write files to the file system".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Provides write access to file system resources")
             .with_security_implication("Can modify or create files")
             .with_capability_requirement("FileSystem", vec!["Write".to_string()]),

            EffectDefinition::new(
                "IO.Network.Connect".to_string(),
                "Establish network connections".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Enables network communication")
             .with_security_implication("Can send data over network")
             .with_capability_requirement("Network", vec!["Connect".to_string()]),
        ];

        // Database Effects
        let database_effects = vec![
            EffectDefinition::new(
                "Database.Query".to_string(),
                "Execute database queries".to_string(),
                EffectCategory::Database,
            ).with_ai_context("Provides database query capabilities")
             .with_security_implication("Can access database contents")
             .with_capability_requirement("Database", vec!["Query".to_string()]),

            EffectDefinition::new(
                "Database.Transaction".to_string(),
                "Manage database transactions".to_string(),
                EffectCategory::Database,
            ).with_ai_context("Provides transactional database operations")
             .with_security_implication("Can modify database state")
             .with_capability_requirement("Database", vec!["Transaction".to_string()]),
        ];

        // Cryptography Effects
        let crypto_effects = vec![
            EffectDefinition::new(
                "Cryptography.KeyGeneration".to_string(),
                "Generate cryptographic keys".to_string(),
                EffectCategory::Security,
            ).with_ai_context("Generates cryptographic material")
             .with_security_implication("Creates sensitive cryptographic keys")
             .with_capability_requirement("Cryptography", vec!["KeyGeneration".to_string()]),

            EffectDefinition::new(
                "Cryptography.Encryption".to_string(),
                "Encrypt data using cryptographic algorithms".to_string(),
                EffectCategory::Security,
            ).with_ai_context("Provides data encryption capabilities")
             .with_security_implication("Processes potentially sensitive data")
             .with_capability_requirement("Cryptography", vec!["Encryption".to_string()]),
        ];

        // External AI Integration Effects (for server-based AI tools)
        let ai_integration_effects = vec![
            EffectDefinition::new(
                "ExternalAI.DataExport".to_string(),
                "Export data for external AI analysis".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Prepares and exports data for external AI systems to analyze")
             .with_security_implication("May expose sensitive data to external AI services")
             .with_capability_requirement("Network", vec!["Send".to_string()])
             .with_business_rule("Data must be sanitized before export"),

            EffectDefinition::new(
                "ExternalAI.MetadataGeneration".to_string(),
                "Generate AI-comprehensible metadata".to_string(),
                EffectCategory::Pure,
            ).with_ai_context("Creates structured metadata for external AI systems to understand code")
             .with_security_implication("Metadata may reveal code structure and business logic")
             .with_capability_requirement("FileSystem", vec!["Write".to_string()])
             .with_business_rule("Generated metadata must not include sensitive implementation details"),
        ];

        // Register all effects
        for effect in io_effects.into_iter()
            .chain(database_effects)
            .chain(crypto_effects)
            .chain(ai_integration_effects)
        {
            let _ = self.register(effect);
        }
    }
}

/// Hierarchical organization of effects
#[derive(Debug, Default)]
pub struct EffectHierarchy {
    /// Parent-child relationships between effects
    pub relationships: HashMap<String, Vec<String>>,
    /// Root effects (no parents)
    pub roots: HashSet<String>,
}

impl EffectHierarchy {
    /// Add an effect to the hierarchy
    pub fn add_effect(&mut self, effect: &EffectDefinition) {
        if let Some(parent) = &effect.parent_effect {
            self.relationships
                .entry(parent.clone())
                .or_default()
                .push(effect.name.clone());
        } else {
            self.roots.insert(effect.name.clone());
        }
    }

    /// Check if one effect is a subeffect of another
    pub fn is_subeffect(&self, child: &str, parent: &str) -> bool {
        if child == parent {
            return true;
        }

        if let Some(children) = self.relationships.get(parent) {
            for child_effect in children {
                if child_effect == child || self.is_subeffect(child, child_effect) {
                    return true;
                }
            }
        }

        false
    }
}

/// Definition of an effect type
#[derive(Debug, Clone)]
pub struct EffectDefinition {
    /// Unique name of the effect
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Effect category for classification
    pub category: EffectCategory,
    /// Parent effect in the hierarchy
    pub parent_effect: Option<String>,
    /// AI-comprehensible context
    pub ai_context: Option<String>,
    /// Security implications of this effect
    pub security_implications: Vec<String>,
    /// Business rules associated with this effect
    pub business_rules: Vec<String>,
    /// Capability requirements
    pub capability_requirements: HashMap<String, Vec<String>>,
    /// Effect parameters and constraints
    pub parameters: Vec<EffectParameter>,
    /// Examples of effect usage
    pub examples: Vec<String>,
    /// Common mistakes to avoid
    pub common_mistakes: Vec<String>,
}

impl EffectDefinition {
    /// Create a new effect definition
    pub fn new(name: String, description: String, category: EffectCategory) -> Self {
        Self {
            name,
            description,
            category,
            parent_effect: None,
            ai_context: None,
            security_implications: Vec::new(),
            business_rules: Vec::new(),
            capability_requirements: HashMap::new(),
            parameters: Vec::new(),
            examples: Vec::new(),
            common_mistakes: Vec::new(),
        }
    }

    /// Add AI context for better comprehension
    pub fn with_ai_context(mut self, context: impl Into<String>) -> Self {
        self.ai_context = Some(context.into());
        self
    }

    /// Add a security implication
    pub fn with_security_implication(mut self, implication: impl Into<String>) -> Self {
        self.security_implications.push(implication.into());
        self
    }

    /// Add a capability requirement
    pub fn with_capability_requirement(mut self, capability: impl Into<String>, permissions: Vec<String>) -> Self {
        self.capability_requirements.insert(capability.into(), permissions);
        self
    }

    /// Add a business rule
    pub fn with_business_rule(mut self, rule: impl Into<String>) -> Self {
        self.business_rules.push(rule.into());
        self
    }

    /// Add an effect parameter
    pub fn with_parameter(mut self, parameter: EffectParameter) -> Self {
        self.parameters.push(parameter);
        self
    }
}

/// Categories of effects as defined in PLD-003
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EffectCategory {
    /// Pure computation with no side effects
    Pure,
    /// Input/Output operations
    IO,
    /// Database operations
    Database,
    /// Network operations
    Network,
    /// Cryptographic operations
    Security,
    /// AI and machine learning operations
    AI,
    /// Memory management operations
    Memory,
    /// System-level operations
    System,
    /// Unsafe operations requiring special handling
    Unsafe,
    /// Custom effect category
    Custom(String),
}

impl fmt::Display for EffectCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pure => write!(f, "Pure"),
            Self::IO => write!(f, "IO"),
            Self::Database => write!(f, "Database"),
            Self::Network => write!(f, "Network"),
            Self::Security => write!(f, "Security"),
            Self::AI => write!(f, "AI"),
            Self::Memory => write!(f, "Memory"),
            Self::System => write!(f, "System"),
            Self::Unsafe => write!(f, "Unsafe"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Parameter for an effect
#[derive(Debug, Clone)]
pub struct EffectParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: AstNode<Type>,
    /// Parameter description
    pub description: String,
    /// Whether the parameter is required
    pub required: bool,
    /// Default value if any
    pub default_value: Option<AstNode<Expr>>,
    /// Constraints on the parameter
    pub constraints: Vec<ParameterConstraint>,
}

/// Constraint on an effect parameter
#[derive(Debug, Clone)]
pub struct ParameterConstraint {
    /// Constraint expression
    pub expression: AstNode<Expr>,
    /// Human-readable constraint description
    pub description: String,
    /// Severity of constraint violation
    pub severity: ConstraintSeverity,
}

/// Severity levels for constraint violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintSeverity {
    /// Error - must be satisfied
    Error,
    /// Warning - should be satisfied
    Warning,
    /// Info - informational constraint
    Info,
}

/// Concrete effect instance with runtime values
#[derive(Debug, Clone)]
pub struct Effect {
    /// The effect definition this instance is based on
    pub definition: String,
    /// Runtime parameters for this effect
    pub parameters: HashMap<String, AstNode<Expr>>,
    /// Source location where this effect occurs
    pub span: Span,
    /// Metadata for this specific effect instance
    pub metadata: EffectInstanceMetadata,
}

impl Effect {
    /// Create a new effect instance
    pub fn new(definition: String, span: Span) -> Self {
        Self {
            definition,
            parameters: HashMap::new(),
            span,
            metadata: EffectInstanceMetadata::default(),
        }
    }

    /// Add a parameter to this effect
    pub fn with_parameter(mut self, name: impl Into<String>, value: AstNode<Expr>) -> Self {
        self.parameters.insert(name.into(), value);
        self
    }

    /// Add metadata to this effect instance
    pub fn with_metadata(mut self, metadata: EffectInstanceMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Metadata for a specific effect instance
#[derive(Debug, Clone, Default)]
pub struct EffectInstanceMetadata {
    /// AI-readable context for this specific instance
    pub ai_context: Option<String>,
    /// Security classification of data processed
    pub security_classification: SecurityClassification,
    /// Whether this effect was inferred or explicit
    pub inferred: bool,
    /// Confidence level in effect inference (0.0 to 1.0)
    pub confidence: f64,
    /// Source of effect inference
    pub inference_source: Option<String>,
}

/// Rule for composing multiple effects
#[derive(Debug, Clone)]
pub struct EffectCompositionRule {
    /// Rule name for identification
    pub name: String,
    /// Description of what this rule does
    pub description: String,
    /// Input effects that trigger this rule
    pub input_effects: Vec<String>,
    /// Output effect produced by composition
    pub output_effect: String,
    /// Conditions under which this rule applies
    pub conditions: Vec<AstNode<Expr>>,
    /// AI-readable explanation of the composition
    pub ai_explanation: Option<String>,
}

impl EffectCompositionRule {
    /// Create a new composition rule
    pub fn new(
        name: String,
        description: String,
        input_effects: Vec<String>,
        output_effect: String,
    ) -> Self {
        Self {
            name,
            description,
            input_effects,
            output_effect,
            conditions: Vec::new(),
            ai_explanation: None,
        }
    }

    /// Add a condition to this rule
    pub fn with_condition(mut self, condition: AstNode<Expr>) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Add AI explanation for this rule
    pub fn with_ai_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.ai_explanation = Some(explanation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_registry_creation() {
        let registry = EffectRegistry::new();
        assert!(!registry.effects.is_empty(), "Registry should have built-in effects");
        assert!(registry.effects.contains_key("IO.FileSystem.Read"));
        assert!(registry.effects.contains_key("Database.Query"));
    }

    #[test]
    fn test_effect_definition_builder() {
        let effect = EffectDefinition::new(
            "Test.Effect".to_string(),
            "A test effect".to_string(),
            EffectCategory::Pure,
        )
        .with_ai_context("This is for testing")
        .with_security_implication("No security implications")
        .with_capability_requirement("Test", vec!["Read".to_string()]);

        assert_eq!(effect.name, "Test.Effect");
        assert_eq!(effect.category, EffectCategory::Pure);
        assert!(effect.ai_context.is_some());
        assert_eq!(effect.security_implications.len(), 1);
        assert_eq!(effect.capability_requirements.len(), 1);
    }

    #[test]
    fn test_effect_hierarchy() {
        let mut hierarchy = EffectHierarchy::default();
        let parent_effect = EffectDefinition::new(
            "Parent".to_string(),
            "Parent effect".to_string(),
            EffectCategory::IO,
        );
        let mut child_effect = EffectDefinition::new(
            "Child".to_string(),
            "Child effect".to_string(),
            EffectCategory::IO,
        );
        child_effect.parent_effect = Some("Parent".to_string());

        hierarchy.add_effect(&parent_effect);
        hierarchy.add_effect(&child_effect);

        assert!(hierarchy.is_subeffect("Child", "Parent"));
        assert!(hierarchy.is_subeffect("Parent", "Parent")); // Self-relationship
        assert!(!hierarchy.is_subeffect("Parent", "Child"));
    }
} 