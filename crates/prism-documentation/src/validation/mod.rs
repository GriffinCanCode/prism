//! Documentation validation engine implementing PSG-003 standards.
//!
//! This module provides comprehensive validation of Prism documentation,
//! ensuring compliance with PSG-003: PrismDoc Standards. It validates
//! required annotations, documentation format, JSDoc compatibility,
//! and AI metadata completeness.

use crate::{DocumentationError, DocumentationResult};
use crate::extraction::ExtractedDocumentation;
use prism_common::span::Span;
use prism_ast::{Program, AstNode, Item, Stmt};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

pub mod rules;
pub mod checkers;
pub mod semantic;

#[cfg(test)]
mod tests;

pub use rules::*;
pub use checkers::*;
pub use semantic::*;

/// Core documentation validator implementing PSG-003 standards
#[derive(Debug)]
pub struct DocumentationValidator {
    /// Validation configuration
    config: ValidationConfig,
    /// Validation rules registry
    rules: ValidationRules,
    /// Custom validation checkers
    checkers: ValidationCheckers,
    /// Semantic validation for PLD-001 integration
    semantic_validator: SemanticValidator,
}

/// Configuration for documentation validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation strictness level
    pub strictness: ValidationStrictness,
    /// Enable JSDoc compatibility checking
    pub check_jsdoc_compatibility: bool,
    /// Enable AI context validation
    pub check_ai_context: bool,
    /// Require examples for public functions
    pub require_examples: bool,
    /// Require performance annotations for critical functions
    pub require_performance_annotations: bool,
    /// Custom validation rules
    pub custom_rules: Vec<String>,
    /// Excluded validation rules
    pub excluded_rules: HashSet<String>,
}

/// Validation strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStrictness {
    /// Lenient validation (warnings only)
    Lenient,
    /// Standard PSG-003 compliance
    Standard,
    /// Strict validation (all rules enforced)
    Strict,
    /// Pedantic validation (maximum strictness)
    Pedantic,
}

/// Validation result containing all findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall compliance status
    pub is_compliant: bool,
    /// Validation violations (errors)
    pub violations: Vec<ValidationViolation>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
    /// Validation statistics
    pub statistics: ValidationStatistics,
}

/// Individual validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Violation type
    pub violation_type: ViolationType,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Human-readable message
    pub message: String,
    /// Location in source code
    pub location: Span,
    /// Suggested fix (if available)
    pub suggested_fix: Option<String>,
    /// Rule that was violated
    pub rule_id: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of validation violations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Missing required annotation
    MissingRequiredAnnotation,
    /// Invalid annotation format
    InvalidAnnotationFormat,
    /// Inconsistent documentation
    InconsistentDocumentation,
    /// Missing JSDoc compatibility
    JSDocIncompatibility,
    /// Insufficient AI context
    InsufficientAIContext,
    /// Missing examples
    MissingExamples,
    /// Invalid documentation structure
    InvalidStructure,
    /// Content quality issue
    ContentQualityIssue,
}

/// Severity levels for violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Error level (must be fixed)
    Error,
    /// Warning level (should be fixed)
    Warning,
    /// Information level (could be improved)
    Info,
    /// Hint level (minor suggestion)
    Hint,
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    /// Total items validated
    pub total_items: usize,
    /// Items with complete documentation
    pub documented_items: usize,
    /// Items missing documentation
    pub undocumented_items: usize,
    /// PSG-003 compliance percentage
    pub compliance_percentage: f64,
    /// JSDoc compatibility percentage
    pub jsdoc_compatibility_percentage: f64,
    /// AI context completeness percentage
    pub ai_context_completeness_percentage: f64,
}

/// Validation error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    /// Rule evaluation failed
    #[error("Rule evaluation failed for '{rule_id}': {reason}")]
    RuleEvaluationFailed { rule_id: String, reason: String },
    
    /// Invalid configuration
    #[error("Invalid validation configuration: {reason}")]
    InvalidConfiguration { reason: String },
    
    /// Missing required data
    #[error("Missing required data for validation: {data_type}")]
    MissingRequiredData { data_type: String },
}

impl DocumentationValidator {
    /// Create a new validator with the given configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            rules: ValidationRules::new(&config),
            checkers: ValidationCheckers::new(),
            semantic_validator: SemanticValidator::default(),
            config,
        }
    }

    /// Create a validator with strict configuration
    pub fn strict() -> Self {
        Self::new(ValidationConfig::strict())
    }

    /// Create a validator with lenient configuration
    pub fn lenient() -> Self {
        Self::new(ValidationConfig::lenient())
    }

    /// Initialize semantic type information from program
    pub fn initialize_semantic_types(&mut self, program: &Program) -> DocumentationResult<()> {
        self.semantic_validator.load_semantic_types(program)
    }

    /// Validate documentation for a complete program
    pub fn validate_program(&self, program: &Program) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        let mut stats = ValidationStatistics::new();

        // Validate each top-level item
        for item in &program.items {
            let item_result = self.validate_item(item)?;
            violations.extend(item_result.violations);
            warnings.extend(item_result.warnings);
            suggestions.extend(item_result.suggestions);
            stats.merge(&item_result.statistics);
        }

        // Calculate overall compliance
        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        stats.calculate_percentages();

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings,
            suggestions,
            statistics: stats,
        })
    }

    /// Validate extracted documentation
    pub fn validate_extracted(&self, docs: &ExtractedDocumentation) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        let mut stats = ValidationStatistics::new();

        // Validate each documentation element
        for element in &docs.elements {
            let element_result = self.validate_documentation_element(element)?;
            violations.extend(element_result.violations);
            warnings.extend(element_result.warnings);
            suggestions.extend(element_result.suggestions);
            stats.merge(&element_result.statistics);
        }

        // Apply global validation rules
        let global_result = self.validate_global_rules(docs)?;
        violations.extend(global_result.violations);
        warnings.extend(global_result.warnings);
        suggestions.extend(global_result.suggestions);

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        stats.calculate_percentages();

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings,
            suggestions,
            statistics: stats,
        })
    }

    /// Validate a single AST item
    fn validate_item(&self, item: &AstNode<Item>) -> DocumentationResult<ValidationResult> {
        match &item.inner {
            Item::Module(module) => self.validate_module_item(item, module),
            Item::Function(function) => self.validate_function_item(item, function),
            Item::Type(type_def) => self.validate_type_item(item, type_def),
            Item::Const(const_def) => self.validate_const_item(item, const_def),
            Item::Variable(var_def) => self.validate_variable_item(item, var_def),
            _ => Ok(ValidationResult::empty()),
        }
    }

    /// Validate a module item
    fn validate_module_item(
        &self,
        item: &AstNode<Item>,
        module: &prism_ast::ModuleDecl,
    ) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check required module annotations
        let required_annotations = [
            ("responsibility", "Module responsibility declaration"),
            ("module", "Module name annotation"),
            ("description", "Module description"),
            ("author", "Module author information"),
        ];

        for (annotation_type, description) in &required_annotations {
            if !self.has_annotation(item, annotation_type) {
                violations.push(ValidationViolation {
                    violation_type: ViolationType::MissingRequiredAnnotation,
                    severity: ViolationSeverity::Error,
                    message: format!("Missing required @{} annotation for module", annotation_type),
                    location: item.span,
                    suggested_fix: Some(format!("Add @{} annotation with {}", annotation_type, description)),
                    rule_id: format!("PSG003_MODULE_REQUIRED_{}", annotation_type.to_uppercase()),
                    context: HashMap::new(),
                });
            }
        }

        // Validate module structure and sections
        if let Some(sections) = self.get_module_sections(module) {
            let section_result = self.validate_module_sections(sections)?;
            violations.extend(section_result.violations);
            warnings.extend(section_result.warnings);
            suggestions.extend(section_result.suggestions);
        }

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        let stats = ValidationStatistics::from_counts(1, if is_compliant { 1 } else { 0 });

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings,
            suggestions,
            statistics: stats,
        })
    }

    /// Validate a function item
    fn validate_function_item(
        &self,
        item: &AstNode<Item>,
        function: &prism_ast::FunctionDecl,
    ) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check if function is public
        let is_public = matches!(function.visibility, prism_ast::Visibility::Public);

        // Required annotations for all functions
        if !self.has_annotation(item, "responsibility") {
            violations.push(ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Missing required @responsibility annotation for function".to_string(),
                location: item.span,
                suggested_fix: Some("Add @responsibility annotation describing function purpose".to_string()),
                rule_id: "PSG003_FUNCTION_RESPONSIBILITY".to_string(),
                context: HashMap::new(),
            });
        }

        // Additional requirements for public functions
        if is_public {
            let public_requirements = [
                ("param", "Parameter documentation"),
                ("returns", "Return value documentation"),
                ("example", "Usage example"),
            ];

            for (annotation_type, description) in &public_requirements {
                if !self.has_annotation(item, annotation_type) {
                    let severity = if *annotation_type == "example" && !self.config.require_examples {
                        ViolationSeverity::Warning
                    } else {
                        ViolationSeverity::Error
                    };

                    violations.push(ValidationViolation {
                        violation_type: ViolationType::MissingRequiredAnnotation,
                        severity,
                        message: format!("Missing required @{} annotation for public function", annotation_type),
                        location: item.span,
                        suggested_fix: Some(format!("Add @{} annotation with {}", annotation_type, description)),
                        rule_id: format!("PSG003_PUBLIC_FUNCTION_{}", annotation_type.to_uppercase()),
                        context: HashMap::new(),
                    });
                }
            }

            // Validate parameter documentation completeness
            for param in &function.parameters {
                if !self.has_parameter_documentation(item, &param.name) {
                    violations.push(ValidationViolation {
                        violation_type: ViolationType::MissingRequiredAnnotation,
                        severity: ViolationSeverity::Error,
                        message: format!("Missing @param documentation for parameter '{}'", param.name),
                        location: item.span,
                        suggested_fix: Some(format!("Add @param {} description", param.name)),
                        rule_id: "PSG003_PARAM_DOCUMENTATION".to_string(),
                        context: [("parameter".to_string(), param.name.to_string())].into(),
                    });
                }
            }
        }

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        let stats = ValidationStatistics::from_counts(1, if is_compliant { 1 } else { 0 });

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings,
            suggestions,
            statistics: stats,
        })
    }

    /// Validate a type item
    fn validate_type_item(
        &self,
        item: &AstNode<Item>,
        type_def: &prism_ast::TypeDecl,
    ) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let is_public = matches!(type_def.visibility, prism_ast::Visibility::Public);

        if is_public {
            // Public types require documentation
            if !self.has_annotation(item, "responsibility") {
                violations.push(ValidationViolation {
                    violation_type: ViolationType::MissingRequiredAnnotation,
                    severity: ViolationSeverity::Error,
                    message: "Missing required @responsibility annotation for public type".to_string(),
                    location: item.span,
                    suggested_fix: Some("Add @responsibility annotation describing type purpose".to_string()),
                    rule_id: "PSG003_TYPE_RESPONSIBILITY".to_string(),
                    context: HashMap::new(),
                });
            }

            // Validate field documentation for structured types
            if let Some(fields) = self.get_type_fields(type_def) {
                for field in fields {
                    if !self.has_field_documentation(item, &field) {
                        violations.push(ValidationViolation {
                            violation_type: ViolationType::MissingRequiredAnnotation,
                            severity: ViolationSeverity::Error,
                            message: format!("Missing @field documentation for field '{}'", field),
                            location: item.span,
                            suggested_fix: Some(format!("Add @field {} description", field)),
                            rule_id: "PSG003_FIELD_DOCUMENTATION".to_string(),
                            context: [("field".to_string(), field)].into(),
                        });
                    }
                }
            }
        }

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        let stats = ValidationStatistics::from_counts(1, if is_compliant { 1 } else { 0 });

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings: Vec::new(),
            suggestions: Vec::new(),
            statistics: stats,
        })
    }

    /// Validate a const item
    fn validate_const_item(
        &self,
        item: &AstNode<Item>,
        const_def: &prism_ast::ConstDecl,
    ) -> DocumentationResult<ValidationResult> {
        // Constants generally don't require extensive documentation
        // unless they are public and part of the API
        let is_public = matches!(const_def.visibility, prism_ast::Visibility::Public);
        let violations = if is_public && !self.has_documentation(item) {
            vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Warning,
                message: "Public constant should have documentation".to_string(),
                location: item.span,
                suggested_fix: Some("Add documentation comment explaining the constant's purpose".to_string()),
                rule_id: "PSG003_PUBLIC_CONST_DOCUMENTATION".to_string(),
                context: HashMap::new(),
            }]
        } else {
            Vec::new()
        };

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        let stats = ValidationStatistics::from_counts(1, if is_compliant { 1 } else { 0 });

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings: Vec::new(),
            suggestions: Vec::new(),
            statistics: stats,
        })
    }

    /// Validate a variable item
    fn validate_variable_item(
        &self,
        _item: &AstNode<Item>,
        _var_def: &prism_ast::VariableDecl,
    ) -> DocumentationResult<ValidationResult> {
        // Variables typically don't require documentation unless they're global/public
        Ok(ValidationResult::empty())
    }

    /// Validate documentation element
    fn validate_documentation_element(
        &self,
        element: &crate::extraction::DocumentationElement,
    ) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Apply validation rules
        let rule_violations = self.rules.validate_element(element)?;
        violations.extend(rule_violations);

        // Apply specialized checkers
        
        // Content quality check
        let content_result = self.checkers.content_checker.check_element(element)?;
        violations.extend(content_result.violations);
        warnings.extend(content_result.warnings);
        suggestions.extend(content_result.suggestions);

        // JSDoc compatibility check (if enabled)
        if self.config.check_jsdoc_compatibility {
            let jsdoc_result = self.checkers.jsdoc_checker.check_element(element)?;
            violations.extend(jsdoc_result.violations);
            warnings.extend(jsdoc_result.warnings);
            suggestions.extend(jsdoc_result.suggestions);
        }

        // AI context check (if enabled)
        if self.config.check_ai_context {
            let ai_result = self.checkers.ai_context_checker.check_element(element)?;
            violations.extend(ai_result.violations);
            warnings.extend(ai_result.warnings);
            suggestions.extend(ai_result.suggestions);
        }

        // Business rule alignment check
        let business_result = self.checkers.business_rule_checker.check_element(element)?;
        violations.extend(business_result.violations);
        warnings.extend(business_result.warnings);
        suggestions.extend(business_result.suggestions);

        // Semantic validation (PLD-001 integration)
        let semantic_result = self.semantic_validator.validate_semantic_alignment(element, None)?;
        violations.extend(semantic_result.violations);
        warnings.extend(semantic_result.warnings);
        suggestions.extend(semantic_result.suggestions);

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        let stats = ValidationStatistics::from_counts(1, if is_compliant { 1 } else { 0 });

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings,
            suggestions,
            statistics: stats,
        })
    }

    /// Validate global documentation rules
    fn validate_global_rules(&self, docs: &ExtractedDocumentation) -> DocumentationResult<ValidationResult> {
        // Apply rules that span across multiple documentation elements
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Check for consistency across the documentation
        let consistency_result = self.checkers.consistency_checker.check_global_consistency(docs)?;
        violations.extend(consistency_result.violations);
        warnings.extend(consistency_result.warnings);
        suggestions.extend(consistency_result.suggestions);

        let is_compliant = violations.iter().all(|v| v.severity != ViolationSeverity::Error);
        let stats = ValidationStatistics::new();

        Ok(ValidationResult {
            is_compliant,
            violations,
            warnings,
            suggestions,
            statistics: stats,
        })
    }

    // Helper methods for AST analysis
    fn has_annotation(&self, item: &AstNode<Item>, annotation_type: &str) -> bool {
        // Check item attributes for the annotation
        item.metadata.attributes.iter().any(|attr| {
            self.extract_annotation_name(attr) == annotation_type
        })
    }

    fn has_documentation(&self, item: &AstNode<Item>) -> bool {
        // Check if item has documentation comment or any annotations
        item.metadata.doc_comment.is_some() || !item.metadata.attributes.is_empty()
    }

    fn has_parameter_documentation(&self, item: &AstNode<Item>, param_name: &str) -> bool {
        // Look for @param annotations that mention this parameter
        item.metadata.attributes.iter().any(|attr| {
            if self.extract_annotation_name(attr) == "param" {
                // Check if the annotation mentions this parameter name
                self.annotation_mentions_parameter(attr, param_name)
            } else {
                false
            }
        })
    }

    fn has_field_documentation(&self, item: &AstNode<Item>, field_name: &str) -> bool {
        // Look for @field annotations that mention this field
        item.metadata.attributes.iter().any(|attr| {
            if self.extract_annotation_name(attr) == "field" {
                // Check if the annotation mentions this field name
                self.annotation_mentions_field(attr, field_name)
            } else {
                false
            }
        })
    }

    fn get_module_sections(&self, module: &prism_ast::ModuleDecl) -> Option<Vec<String>> {
        // Extract section names from module body
        let mut sections = Vec::new();
        
        // In a complete implementation, we'd parse the module body
        // and extract section declarations like "section config", "section types", etc.
        // For now, return common expected sections
        sections.extend([
            "config".to_string(),
            "types".to_string(),
            "interface".to_string(),
            "internal".to_string(),
        ]);
        
        if sections.is_empty() {
            None
        } else {
            Some(sections)
        }
    }

    fn get_type_fields(&self, type_def: &prism_ast::TypeDecl) -> Option<Vec<String>> {
        // Extract field names from structured types
        match &type_def.definition {
            prism_ast::TypeDefinition::Struct(struct_def) => {
                let fields: Vec<String> = struct_def.fields.iter()
                    .map(|field| field.name.clone())
                    .collect();
                if fields.is_empty() { None } else { Some(fields) }
            },
            prism_ast::TypeDefinition::Record(record_def) => {
                let fields: Vec<String> = record_def.fields.iter()
                    .map(|field| field.name.clone())
                    .collect();
                if fields.is_empty() { None } else { Some(fields) }
            },
            _ => None,
        }
    }

    fn validate_module_sections(&self, sections: Vec<String>) -> DocumentationResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check for recommended section organization
        let recommended_sections = ["config", "types", "interface"];
        let has_recommended = recommended_sections.iter()
            .any(|rec| sections.iter().any(|sec| sec == rec));
            
        if !has_recommended {
            suggestions.push("Consider organizing module with standard sections: config, types, interface".to_string());
        }
        
        // Check for too many sections (potential over-organization)
        if sections.len() > 8 {
            suggestions.push("Module has many sections. Consider consolidating related functionality".to_string());
        }
        
        Ok(ValidationResult {
            is_compliant: true,
            violations,
            warnings: Vec::new(),
            suggestions,
            statistics: ValidationStatistics::new(),
        })
    }

    // Utility methods for annotation parsing
    fn extract_annotation_name(&self, attr: &prism_ast::Attribute) -> &str {
        &attr.name
    }

    fn annotation_mentions_parameter(&self, attr: &prism_ast::Attribute, param_name: &str) -> bool {
        // Check if the attribute value mentions the parameter name
        match &attr.value {
            Some(AttributeValue::String(s)) => s.contains(param_name),
            Some(AttributeValue::List(items)) => {
                items.iter().any(|item| match item {
                    AttributeValue::String(s) => s.contains(param_name),
                    _ => false,
                })
            },
            _ => false,
        }
    }

    fn annotation_mentions_field(&self, attr: &prism_ast::Attribute, field_name: &str) -> bool {
        // Check if the attribute value mentions the field name
        match &attr.value {
            Some(AttributeValue::String(s)) => s.contains(field_name),
            Some(AttributeValue::List(items)) => {
                items.iter().any(|item| match item {
                    AttributeValue::String(s) => s.contains(field_name),
                    _ => false,
                })
            },
            _ => false,
        }
    }
}

impl ValidationConfig {
    /// Create strict validation configuration
    pub fn strict() -> Self {
        Self {
            strictness: ValidationStrictness::Strict,
            check_jsdoc_compatibility: true,
            check_ai_context: true,
            require_examples: true,
            require_performance_annotations: true,
            custom_rules: Vec::new(),
            excluded_rules: HashSet::new(),
        }
    }

    /// Create lenient validation configuration
    pub fn lenient() -> Self {
        Self {
            strictness: ValidationStrictness::Lenient,
            check_jsdoc_compatibility: false,
            check_ai_context: false,
            require_examples: false,
            require_performance_annotations: false,
            custom_rules: Vec::new(),
            excluded_rules: HashSet::new(),
        }
    }

    /// Create standard PSG-003 compliant configuration
    pub fn standard() -> Self {
        Self {
            strictness: ValidationStrictness::Standard,
            check_jsdoc_compatibility: true,
            check_ai_context: true,
            require_examples: false,
            require_performance_annotations: false,
            custom_rules: Vec::new(),
            excluded_rules: HashSet::new(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::standard()
    }
}

impl ValidationResult {
    /// Create empty validation result
    pub fn empty() -> Self {
        Self {
            is_compliant: true,
            violations: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
            statistics: ValidationStatistics::new(),
        }
    }

    /// Merge another validation result into this one
    pub fn merge(&mut self, other: &ValidationResult) {
        self.is_compliant &= other.is_compliant;
        self.violations.extend(other.violations.clone());
        self.warnings.extend(other.warnings.clone());
        self.suggestions.extend(other.suggestions.clone());
        self.statistics.merge(&other.statistics);
    }
}

impl ValidationStatistics {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            total_items: 0,
            documented_items: 0,
            undocumented_items: 0,
            compliance_percentage: 0.0,
            jsdoc_compatibility_percentage: 0.0,
            ai_context_completeness_percentage: 0.0,
        }
    }

    /// Create statistics from counts
    pub fn from_counts(total: usize, documented: usize) -> Self {
        Self {
            total_items: total,
            documented_items: documented,
            undocumented_items: total - documented,
            compliance_percentage: if total > 0 { (documented as f64 / total as f64) * 100.0 } else { 0.0 },
            jsdoc_compatibility_percentage: 0.0,
            ai_context_completeness_percentage: 0.0,
        }
    }

    /// Merge another statistics into this one
    pub fn merge(&mut self, other: &ValidationStatistics) {
        self.total_items += other.total_items;
        self.documented_items += other.documented_items;
        self.undocumented_items += other.undocumented_items;
    }

    /// Calculate percentage values
    pub fn calculate_percentages(&mut self) {
        if self.total_items > 0 {
            self.compliance_percentage = (self.documented_items as f64 / self.total_items as f64) * 100.0;
        }
    }
}

impl Default for ValidationStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationViolation {
    /// Get the message for this violation
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Check if this violation is an error
    pub fn is_error(&self) -> bool {
        self.severity == ViolationSeverity::Error
    }

    /// Check if this violation is a warning
    pub fn is_warning(&self) -> bool {
        self.severity == ViolationSeverity::Warning
    }
} 