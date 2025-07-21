//! Semantic validation for documentation alignment with PLD-001 types.
//!
//! This module provides validation that ensures documentation is aligned
//! with semantic type information, business rules, and constraints as
//! specified in PLD-001: Semantic Type System.

use crate::{DocumentationResult, DocumentationError};
use crate::extraction::{DocumentationElement, ExtractedDocumentation};
use crate::validation::{ValidationViolation, ViolationType, ViolationSeverity, CheckerResult};
use prism_common::span::Span;
use prism_ast::{AstNode, Item, TypeDecl, FunctionDecl};
use std::collections::{HashMap, HashSet};

/// Semantic validation checker for PLD-001 integration
#[derive(Debug)]
pub struct SemanticValidator {
    /// Configuration for semantic validation
    config: SemanticValidationConfig,
    /// Cache of semantic type information
    semantic_cache: HashMap<String, SemanticTypeInfo>,
}

/// Configuration for semantic validation
#[derive(Debug, Clone)]
pub struct SemanticValidationConfig {
    /// Enable semantic type alignment checking
    pub check_type_alignment: bool,
    /// Enable business rule validation
    pub check_business_rules: bool,
    /// Enable constraint documentation validation
    pub check_constraint_docs: bool,
    /// Enable effect documentation validation
    pub check_effect_docs: bool,
    /// Strictness level for semantic validation
    pub strictness: SemanticStrictness,
}

/// Strictness levels for semantic validation
#[derive(Debug, Clone, Copy)]
pub enum SemanticStrictness {
    /// Lenient - warnings for misalignment
    Lenient,
    /// Standard - errors for major misalignment
    Standard,
    /// Strict - errors for any misalignment
    Strict,
}

/// Semantic type information for validation
#[derive(Debug, Clone)]
pub struct SemanticTypeInfo {
    /// Type name
    pub name: String,
    /// Business meaning
    pub business_meaning: Option<String>,
    /// Semantic constraints
    pub constraints: Vec<String>,
    /// Business rules
    pub business_rules: Vec<String>,
    /// Expected documentation elements
    pub expected_docs: Vec<String>,
    /// Effect signatures (for functions)
    pub effects: Vec<String>,
}

impl SemanticValidator {
    /// Create new semantic validator
    pub fn new(config: SemanticValidationConfig) -> Self {
        Self {
            config,
            semantic_cache: HashMap::new(),
        }
    }

    /// Create validator with default configuration
    pub fn default() -> Self {
        Self::new(SemanticValidationConfig::default())
    }

    /// Validate semantic alignment for a documentation element
    pub fn validate_semantic_alignment(
        &self,
        element: &DocumentationElement,
        ast_item: Option<&AstNode<Item>>,
    ) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Get semantic type information if available
        if let Some(semantic_info) = self.get_semantic_info(&element.name) {
            // Validate type documentation alignment
            if self.config.check_type_alignment {
                let alignment_result = self.check_type_alignment(element, &semantic_info)?;
                violations.extend(alignment_result.violations);
                warnings.extend(alignment_result.warnings);
                suggestions.extend(alignment_result.suggestions);
            }

            // Validate business rule documentation
            if self.config.check_business_rules {
                let business_result = self.check_business_rule_alignment(element, &semantic_info)?;
                violations.extend(business_result.violations);
                warnings.extend(business_result.warnings);
                suggestions.extend(business_result.suggestions);
            }

            // Validate constraint documentation
            if self.config.check_constraint_docs {
                let constraint_result = self.check_constraint_documentation(element, &semantic_info)?;
                violations.extend(constraint_result.violations);
                warnings.extend(constraint_result.warnings);
                suggestions.extend(constraint_result.suggestions);
            }
        }

        // Validate effect documentation for functions
        if matches!(element.element_type, crate::extraction::DocumentationElementType::Function) {
            if self.config.check_effect_docs {
                if let Some(ast_item) = ast_item {
                    let effect_result = self.check_effect_documentation(element, ast_item)?;
                    violations.extend(effect_result.violations);
                    warnings.extend(effect_result.warnings);
                    suggestions.extend(effect_result.suggestions);
                }
            }
        }

        Ok(CheckerResult {
            violations,
            warnings,
            suggestions,
        })
    }

    /// Check type documentation alignment with semantic types
    fn check_type_alignment(
        &self,
        element: &DocumentationElement,
        semantic_info: &SemanticTypeInfo,
    ) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check if documented purpose aligns with business meaning
        if let Some(business_meaning) = &semantic_info.business_meaning {
            let has_aligned_purpose = element.annotations.iter()
                .any(|a| a.name == "responsibility" && a.value.as_ref()
                    .map_or(false, |v| self.are_semantically_aligned(v, business_meaning)));

            if !has_aligned_purpose {
                let severity = match self.config.strictness {
                    SemanticStrictness::Lenient => ViolationSeverity::Warning,
                    SemanticStrictness::Standard => ViolationSeverity::Warning,
                    SemanticStrictness::Strict => ViolationSeverity::Error,
                };

                violations.push(ValidationViolation {
                    violation_type: ViolationType::InconsistentDocumentation,
                    severity,
                    message: "Documentation purpose may not align with semantic type meaning".to_string(),
                    location: element.location,
                    suggested_fix: Some(format!(
                        "Consider aligning documentation with business meaning: {}",
                        business_meaning
                    )),
                    rule_id: "SEMANTIC_TYPE_ALIGNMENT".to_string(),
                    context: [
                        ("semantic_meaning".to_string(), business_meaning.clone()),
                        ("element_type".to_string(), element.name.clone()),
                    ].into(),
                });
            }
        }

        // Check for missing expected documentation elements
        for expected_doc in &semantic_info.expected_docs {
            let has_expected = element.annotations.iter()
                .any(|a| a.name == expected_doc);

            if !has_expected {
                suggestions.push(format!(
                    "Consider adding @{} annotation as expected for this semantic type",
                    expected_doc
                ));
            }
        }

        Ok(CheckerResult {
            violations,
            warnings,
            suggestions,
        })
    }

    /// Check business rule alignment
    fn check_business_rule_alignment(
        &self,
        element: &DocumentationElement,
        semantic_info: &SemanticTypeInfo,
    ) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut suggestions = Vec::new();

        // Check if business rules are documented
        if !semantic_info.business_rules.is_empty() {
            let has_business_docs = element.annotations.iter()
                .any(|a| matches!(a.name.as_str(), "business" | "rule" | "constraint"));

            if !has_business_docs {
                suggestions.push(format!(
                    "Consider documenting business rules: {}",
                    semantic_info.business_rules.join(", ")
                ));
            }

            // Check if documented rules align with semantic rules
            for rule in &semantic_info.business_rules {
                let rule_documented = element.content.as_ref()
                    .map_or(false, |content| content.to_lowercase().contains(&rule.to_lowercase()))
                    || element.annotations.iter().any(|a| 
                        a.value.as_ref().map_or(false, |v| v.to_lowercase().contains(&rule.to_lowercase())));

                if !rule_documented {
                    suggestions.push(format!(
                        "Consider documenting business rule: {}",
                        rule
                    ));
                }
            }
        }

        Ok(CheckerResult {
            violations,
            warnings: Vec::new(),
            suggestions,
        })
    }

    /// Check constraint documentation
    fn check_constraint_documentation(
        &self,
        element: &DocumentationElement,
        semantic_info: &SemanticTypeInfo,
    ) -> DocumentationResult<CheckerResult> {
        let mut suggestions = Vec::new();

        // Check if constraints are documented
        if !semantic_info.constraints.is_empty() {
            let has_constraint_docs = element.annotations.iter()
                .any(|a| matches!(a.name.as_str(), "constraint" | "invariant" | "requires" | "ensures"));

            if !has_constraint_docs {
                suggestions.push("Consider documenting type constraints and invariants".to_string());
            }

            // Suggest specific constraint documentation
            for constraint in &semantic_info.constraints {
                suggestions.push(format!(
                    "Consider documenting constraint: {}",
                    constraint
                ));
            }
        }

        Ok(CheckerResult {
            violations: Vec::new(),
            warnings: Vec::new(),
            suggestions,
        })
    }

    /// Check effect documentation for functions
    fn check_effect_documentation(
        &self,
        element: &DocumentationElement,
        ast_item: &AstNode<Item>,
    ) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut suggestions = Vec::new();

        // Extract function information
        if let Item::Function(func_decl) = &ast_item.inner {
            // Check if function has effects that should be documented
            let has_effect_annotation = element.annotations.iter()
                .any(|a| a.name == "effects");

            // In a complete implementation, we'd analyze the function for actual effects
            // For now, suggest effect documentation for public functions
            if matches!(func_decl.visibility, prism_ast::Visibility::Public) && !has_effect_annotation {
                suggestions.push("Consider documenting function effects with @effects annotation".to_string());
            }

            // Check for side effect documentation
            let mentions_side_effects = element.content.as_ref()
                .map_or(false, |content| {
                    let content_lower = content.to_lowercase();
                    content_lower.contains("side effect") || 
                    content_lower.contains("mutates") ||
                    content_lower.contains("modifies")
                });

            if mentions_side_effects && !has_effect_annotation {
                suggestions.push("Function mentions side effects. Consider using @effects annotation".to_string());
            }

            // Check for capability requirements
            let mentions_capabilities = element.content.as_ref()
                .map_or(false, |content| {
                    let content_lower = content.to_lowercase();
                    content_lower.contains("requires") || 
                    content_lower.contains("permission") ||
                    content_lower.contains("capability")
                });

            if mentions_capabilities {
                let has_capability_docs = element.annotations.iter()
                    .any(|a| matches!(a.name.as_str(), "requires" | "capability"));

                if !has_capability_docs {
                    suggestions.push("Function mentions requirements. Consider using @requires annotation".to_string());
                }
            }
        }

        Ok(CheckerResult {
            violations,
            warnings: Vec::new(),
            suggestions,
        })
    }

    /// Get semantic type information for a given name
    fn get_semantic_info(&self, name: &str) -> Option<&SemanticTypeInfo> {
        self.semantic_cache.get(name)
    }

    /// Check if two strings are semantically aligned
    fn are_semantically_aligned(&self, doc_text: &str, semantic_meaning: &str) -> bool {
        // Simple semantic alignment check
        // In a complete implementation, this could use more sophisticated NLP
        let doc_words: HashSet<_> = doc_text.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let semantic_words: HashSet<_> = semantic_meaning.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let common_words = doc_words.intersection(&semantic_words).count();
        let min_words = doc_words.len().min(semantic_words.len());

        if min_words == 0 {
            return false;
        }

        // Consider aligned if they share at least 25% of words
        (common_words as f64 / min_words as f64) >= 0.25
    }

    /// Register semantic type information
    pub fn register_semantic_type(&mut self, info: SemanticTypeInfo) {
        self.semantic_cache.insert(info.name.clone(), info);
    }

    /// Load semantic type information from the semantic type system
    pub fn load_semantic_types(&mut self, _program: &prism_ast::Program) -> DocumentationResult<()> {
        // In a complete implementation, this would integrate with the semantic type system
        // to load actual semantic type information from PLD-001 types
        
        // For now, register some common semantic types as examples
        self.register_semantic_type(SemanticTypeInfo {
            name: "Money".to_string(),
            business_meaning: Some("Represents monetary value with currency safety".to_string()),
            constraints: vec![
                "Must have valid currency".to_string(),
                "Precision must be appropriate for currency".to_string(),
                "Cannot be negative for most operations".to_string(),
            ],
            business_rules: vec![
                "Currency conversions require exchange rates".to_string(),
                "Arithmetic operations must preserve precision".to_string(),
            ],
            expected_docs: vec!["param".to_string(), "returns".to_string(), "throws".to_string()],
            effects: vec!["Pure".to_string()],
        });

        self.register_semantic_type(SemanticTypeInfo {
            name: "AccountId".to_string(),
            business_meaning: Some("Unique identifier for user accounts".to_string()),
            constraints: vec![
                "Must be valid UUID format".to_string(),
                "Must exist in account database".to_string(),
            ],
            business_rules: vec![
                "Account IDs are immutable once created".to_string(),
                "Must pass validation checks".to_string(),
            ],
            expected_docs: vec!["param".to_string(), "validation".to_string()],
            effects: vec!["Pure".to_string()],
        });

        Ok(())
    }
}

impl Default for SemanticValidationConfig {
    fn default() -> Self {
        Self {
            check_type_alignment: true,
            check_business_rules: true,
            check_constraint_docs: true,
            check_effect_docs: true,
            strictness: SemanticStrictness::Standard,
        }
    }
}

impl Default for SemanticValidator {
    fn default() -> Self {
        Self::new(SemanticValidationConfig::default())
    }
} 