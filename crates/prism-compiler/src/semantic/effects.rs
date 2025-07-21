//! Effect Signature Analysis
//!
//! This module implements effect signature analysis for the Prism compiler's
//! effect system integration. It focuses on analyzing and tracking effects
//! without duplicating effect system functionality.
//!
//! **Conceptual Responsibility**: Effect signature analysis
//! **What it does**: Analyze effect signatures, track effect usage, validate effect requirements
//! **What it doesn't do**: Manage effect system, enforce capabilities (delegates to prism-effects)

use prism_common::{NodeId, span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Effect signature for a function or operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSignature {
    /// Function or operation symbol
    pub symbol: Symbol,
    /// Effects produced by this operation
    pub effects: Vec<EffectInfo>,
    /// Required capabilities to perform this operation
    pub required_capabilities: Vec<String>,
    /// Effect constraints and conditions
    pub constraints: Vec<EffectConstraint>,
    /// Location where signature is defined
    pub location: Span,
}

/// Information about a specific effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectInfo {
    /// Effect name/identifier
    pub name: String,
    /// Effect category
    pub category: EffectCategory,
    /// Effect parameters if any
    pub parameters: Vec<EffectParameter>,
    /// Effect description for AI comprehension
    pub description: Option<String>,
}

/// Category of effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectCategory {
    /// I/O effects (file, network, etc.)
    IO,
    /// State mutation effects
    State,
    /// Memory allocation effects
    Memory,
    /// Exception/error effects
    Exception,
    /// Concurrency effects
    Concurrency,
    /// Security-sensitive effects
    Security,
    /// User-defined effects
    UserDefined(String),
}

/// Effect parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Parameter description
    pub description: Option<String>,
}

/// Effect constraint or condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint expression or condition
    pub condition: String,
    /// Error message if constraint is violated
    pub error_message: Option<String>,
}

/// Type of effect constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Precondition that must be true
    Precondition,
    /// Postcondition that will be true
    Postcondition,
    /// Capability requirement
    CapabilityRequired,
    /// Effect ordering constraint
    Ordering,
    /// Resource availability constraint
    Resource,
}

/// Effect analyzer for extracting effect signatures
#[derive(Debug)]
pub struct EffectAnalyzer {
    /// Configuration for effect analysis
    config: EffectAnalysisConfig,
}

/// Configuration for effect analysis
#[derive(Debug, Clone)]
pub struct EffectAnalysisConfig {
    /// Enable effect inference
    pub enable_inference: bool,
    /// Enable effect validation
    pub enable_validation: bool,
    /// Enable capability checking
    pub enable_capability_checking: bool,
}

impl Default for EffectAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_inference: true,
            enable_validation: true,
            enable_capability_checking: true,
        }
    }
}

impl EffectAnalyzer {
    /// Create a new effect analyzer
    pub fn new(config: EffectAnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze effect signatures for a set of symbols
    pub fn analyze_effects(&self, symbols: &[Symbol]) -> Vec<EffectSignature> {
        let mut signatures = Vec::new();

        for symbol in symbols {
            if let Some(signature) = self.extract_effect_signature(symbol) {
                signatures.push(signature);
            }
        }

        signatures
    }

    /// Extract effect signature for a specific symbol
    fn extract_effect_signature(&self, symbol: &Symbol) -> Option<EffectSignature> {
        // This is a placeholder - actual implementation would:
        // 1. Look up symbol information from symbol table
        // 2. Analyze function/operation for effect usage
        // 3. Extract effect signatures from annotations or infer them
        // 4. Build EffectSignature structure

        // For now, return None as this is a structural placeholder
        None
    }

    /// Validate effect signatures for consistency
    pub fn validate_signatures(&self, signatures: &[EffectSignature]) -> Vec<EffectValidationError> {
        let mut errors = Vec::new();

        if !self.config.enable_validation {
            return errors;
        }

        for signature in signatures {
            errors.extend(self.validate_single_signature(signature));
        }

        errors
    }

    /// Validate a single effect signature
    fn validate_single_signature(&self, signature: &EffectSignature) -> Vec<EffectValidationError> {
        let mut errors = Vec::new();

        // Validate effect consistency
        for effect in &signature.effects {
            if effect.name.is_empty() {
                errors.push(EffectValidationError {
                    error_type: EffectErrorType::EmptyEffectName,
                    message: "Effect name cannot be empty".to_string(),
                    location: signature.location,
                    suggested_fix: Some("Provide a meaningful effect name".to_string()),
                });
            }
        }

        // Validate capability requirements
        if self.config.enable_capability_checking {
            for capability in &signature.required_capabilities {
                if capability.is_empty() {
                    errors.push(EffectValidationError {
                        error_type: EffectErrorType::EmptyCapability,
                        message: "Capability name cannot be empty".to_string(),
                        location: signature.location,
                        suggested_fix: Some("Provide a valid capability name".to_string()),
                    });
                }
            }
        }

        errors
    }
}

/// Effect validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectValidationError {
    /// Type of validation error
    pub error_type: EffectErrorType,
    /// Error message
    pub message: String,
    /// Location where error occurred
    pub location: Span,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Type of effect validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectErrorType {
    /// Empty effect name
    EmptyEffectName,
    /// Empty capability name
    EmptyCapability,
    /// Missing required capability
    MissingCapability,
    /// Conflicting effects
    ConflictingEffects,
    /// Invalid effect constraint
    InvalidConstraint,
    /// Effect ordering violation
    OrderingViolation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_signature_creation() {
        let signature = EffectSignature {
            symbol: Symbol::intern("test_function"),
            effects: vec![EffectInfo {
                name: "IO".to_string(),
                category: EffectCategory::IO,
                parameters: Vec::new(),
                description: Some("File I/O operation".to_string()),
            }],
            required_capabilities: vec!["FileSystem".to_string()],
            constraints: Vec::new(),
            location: Span::dummy(),
        };

        assert_eq!(signature.effects.len(), 1);
        assert_eq!(signature.required_capabilities.len(), 1);
    }

    #[test]
    fn test_effect_analyzer_creation() {
        let config = EffectAnalysisConfig::default();
        let analyzer = EffectAnalyzer::new(config);

        assert!(analyzer.config.enable_inference);
        assert!(analyzer.config.enable_validation);
        assert!(analyzer.config.enable_capability_checking);
    }

    #[test]
    fn test_effect_validation() {
        let config = EffectAnalysisConfig::default();
        let analyzer = EffectAnalyzer::new(config);

        let signature = EffectSignature {
            symbol: Symbol::intern("test_function"),
            effects: vec![EffectInfo {
                name: "".to_string(), // Empty name should cause validation error
                category: EffectCategory::IO,
                parameters: Vec::new(),
                description: None,
            }],
            required_capabilities: Vec::new(),
            constraints: Vec::new(),
            location: Span::dummy(),
        };

        let errors = analyzer.validate_single_signature(&signature);
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].error_type, EffectErrorType::EmptyEffectName));
    }
} 