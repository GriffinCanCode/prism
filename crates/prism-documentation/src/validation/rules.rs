//! Validation rules implementing PSG-003 standards.
//!
//! This module defines the specific validation rules that enforce
//! PSG-003: PrismDoc Standards. Each rule is responsible for checking
//! a specific aspect of documentation compliance.

use crate::{DocumentationResult, DocumentationError};
use crate::extraction::{DocumentationElement, ExtractedDocumentation};
use crate::validation::{ValidationViolation, ViolationType, ViolationSeverity, ValidationConfig};
use prism_common::span::Span;
use prism_ast::{AstNode, Item, Attribute, AttributeValue};
use std::collections::{HashMap, HashSet};

/// Registry of all validation rules
#[derive(Debug)]
pub struct ValidationRules {
    /// Configuration for rule execution
    config: ValidationConfig,
    /// Individual rule implementations
    rules: HashMap<String, Box<dyn ValidationRule>>,
}

/// Trait for validation rules
pub trait ValidationRule: Send + Sync + std::fmt::Debug {
    /// Get the rule identifier
    fn rule_id(&self) -> &str;
    
    /// Get rule description
    fn description(&self) -> &str;
    
    /// Check if rule applies to the given element
    fn applies_to(&self, element: &DocumentationElement) -> bool;
    
    /// Validate the element against this rule
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>>;
}

impl ValidationRules {
    /// Create new validation rules registry
    pub fn new(config: &ValidationConfig) -> Self {
        let mut rules: HashMap<String, Box<dyn ValidationRule>> = HashMap::new();
        
        // Register core PSG-003 rules
        rules.insert("PSG003_MODULE_RESPONSIBILITY".to_string(), 
                    Box::new(ModuleResponsibilityRule::new()));
        rules.insert("PSG003_MODULE_DESCRIPTION".to_string(),
                    Box::new(ModuleDescriptionRule::new()));
        rules.insert("PSG003_MODULE_AUTHOR".to_string(),
                    Box::new(ModuleAuthorRule::new()));
        rules.insert("PSG003_FUNCTION_RESPONSIBILITY".to_string(),
                    Box::new(FunctionResponsibilityRule::new()));
        rules.insert("PSG003_PUBLIC_FUNCTION_PARAMS".to_string(),
                    Box::new(PublicFunctionParamsRule::new()));
        rules.insert("PSG003_PUBLIC_FUNCTION_RETURNS".to_string(),
                    Box::new(PublicFunctionReturnsRule::new()));
        rules.insert("PSG003_PUBLIC_FUNCTION_EXAMPLES".to_string(),
                    Box::new(PublicFunctionExamplesRule::new(config.require_examples)));
        rules.insert("PSG003_TYPE_RESPONSIBILITY".to_string(),
                    Box::new(TypeResponsibilityRule::new()));
        rules.insert("PSG003_TYPE_FIELD_DOCS".to_string(),
                    Box::new(TypeFieldDocumentationRule::new()));
        rules.insert("PSG003_RESPONSIBILITY_LENGTH".to_string(),
                    Box::new(ResponsibilityLengthRule::new()));
        rules.insert("PSG003_DESCRIPTION_LENGTH".to_string(),
                    Box::new(DescriptionLengthRule::new()));
        
        Self {
            config: config.clone(),
            rules,
        }
    }
    
    /// Get all applicable rules for an element
    pub fn get_applicable_rules(&self, element: &DocumentationElement) -> Vec<&Box<dyn ValidationRule>> {
        self.rules.values()
            .filter(|rule| {
                !self.config.excluded_rules.contains(rule.rule_id()) &&
                rule.applies_to(element)
            })
            .collect()
    }
    
    /// Validate element against all applicable rules
    pub fn validate_element(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();
        
        for rule in self.get_applicable_rules(element) {
            let rule_violations = rule.validate(element)?;
            violations.extend(rule_violations);
        }
        
        Ok(violations)
    }
}

/// Rule: Module must have @responsibility annotation
#[derive(Debug)]
pub struct ModuleResponsibilityRule;

impl ModuleResponsibilityRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for ModuleResponsibilityRule {
    fn rule_id(&self) -> &str {
        "PSG003_MODULE_RESPONSIBILITY"
    }
    
    fn description(&self) -> &str {
        "Modules must have @responsibility annotation declaring single responsibility"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Module)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_responsibility = element.annotations.iter()
            .any(|attr| attr.name == "responsibility");
            
        if !has_responsibility {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Missing required @responsibility annotation for module".to_string(),
                location: element.location,
                suggested_fix: Some("Add @responsibility annotation describing module's single responsibility".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Module must have @description annotation
#[derive(Debug)]
pub struct ModuleDescriptionRule;

impl ModuleDescriptionRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for ModuleDescriptionRule {
    fn rule_id(&self) -> &str {
        "PSG003_MODULE_DESCRIPTION"
    }
    
    fn description(&self) -> &str {
        "Modules must have @description annotation"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Module)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_description = element.annotations.iter()
            .any(|attr| attr.name == "description");
            
        if !has_description {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Missing required @description annotation for module".to_string(),
                location: element.location,
                suggested_fix: Some("Add @description annotation with brief functional description".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Module must have @author annotation
#[derive(Debug)]
pub struct ModuleAuthorRule;

impl ModuleAuthorRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for ModuleAuthorRule {
    fn rule_id(&self) -> &str {
        "PSG003_MODULE_AUTHOR"
    }
    
    fn description(&self) -> &str {
        "Modules must have @author annotation"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Module)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_author = element.annotations.iter()
            .any(|attr| attr.name == "author");
            
        if !has_author {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Missing required @author annotation for module".to_string(),
                location: element.location,
                suggested_fix: Some("Add @author annotation with team or individual responsible".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Functions must have @responsibility annotation
#[derive(Debug)]
pub struct FunctionResponsibilityRule;

impl FunctionResponsibilityRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for FunctionResponsibilityRule {
    fn rule_id(&self) -> &str {
        "PSG003_FUNCTION_RESPONSIBILITY"
    }
    
    fn description(&self) -> &str {
        "Functions must have @responsibility annotation"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Function)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_responsibility = element.annotations.iter()
            .any(|attr| attr.name == "responsibility");
            
        if !has_responsibility {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Missing required @responsibility annotation for function".to_string(),
                location: element.location,
                suggested_fix: Some("Add @responsibility annotation describing function's single responsibility".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Public functions must have @param documentation for all parameters
#[derive(Debug)]
pub struct PublicFunctionParamsRule;

impl PublicFunctionParamsRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for PublicFunctionParamsRule {
    fn rule_id(&self) -> &str {
        "PSG003_PUBLIC_FUNCTION_PARAMS"
    }
    
    fn description(&self) -> &str {
        "Public functions must document all parameters with @param annotations"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Function) &&
        matches!(element.visibility, crate::extraction::ElementVisibility::Public)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        // For now, just check if there are any @param annotations
        // TODO: In a complete implementation, we'd need to cross-reference with actual function parameters
        let has_param_docs = element.annotations.iter()
            .any(|attr| attr.name == "param");
            
        if !has_param_docs {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Public function missing @param documentation".to_string(),
                location: element.location,
                suggested_fix: Some("Add @param annotations for all function parameters".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Public functions must have @returns documentation
#[derive(Debug)]
pub struct PublicFunctionReturnsRule;

impl PublicFunctionReturnsRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for PublicFunctionReturnsRule {
    fn rule_id(&self) -> &str {
        "PSG003_PUBLIC_FUNCTION_RETURNS"
    }
    
    fn description(&self) -> &str {
        "Public functions must have @returns documentation"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Function) &&
        matches!(element.visibility, crate::extraction::ElementVisibility::Public)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_returns = element.annotations.iter()
            .any(|attr| attr.name == "returns");
            
        if !has_returns {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Public function missing @returns documentation".to_string(),
                location: element.location,
                suggested_fix: Some("Add @returns annotation describing return value".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Public functions should have @example documentation
#[derive(Debug)]
pub struct PublicFunctionExamplesRule {
    require_examples: bool,
}

impl PublicFunctionExamplesRule {
    pub fn new(require_examples: bool) -> Self {
        Self { require_examples }
    }
}

impl ValidationRule for PublicFunctionExamplesRule {
    fn rule_id(&self) -> &str {
        "PSG003_PUBLIC_FUNCTION_EXAMPLES"
    }
    
    fn description(&self) -> &str {
        "Public functions should have @example documentation"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Function) &&
        matches!(element.visibility, crate::extraction::ElementVisibility::Public)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_example = element.annotations.iter()
            .any(|attr| attr.name == "example");
            
        if !has_example {
            let severity = if self.require_examples {
                ViolationSeverity::Error
            } else {
                ViolationSeverity::Warning
            };
            
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingExamples,
                severity,
                message: "Public function missing @example documentation".to_string(),
                location: element.location,
                suggested_fix: Some("Add @example annotation with usage example".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Types must have @responsibility annotation
#[derive(Debug)]
pub struct TypeResponsibilityRule;

impl TypeResponsibilityRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for TypeResponsibilityRule {
    fn rule_id(&self) -> &str {
        "PSG003_TYPE_RESPONSIBILITY"
    }
    
    fn description(&self) -> &str {
        "Public types must have @responsibility annotation"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Type) &&
        matches!(element.visibility, crate::extraction::ElementVisibility::Public)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let has_responsibility = element.annotations.iter()
            .any(|attr| attr.name == "responsibility");
            
        if !has_responsibility {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Error,
                message: "Missing required @responsibility annotation for public type".to_string(),
                location: element.location,
                suggested_fix: Some("Add @responsibility annotation describing type's purpose".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: Types should have field documentation
#[derive(Debug)]
pub struct TypeFieldDocumentationRule;

impl TypeFieldDocumentationRule {
    pub fn new() -> Self {
        Self
    }
}

impl ValidationRule for TypeFieldDocumentationRule {
    fn rule_id(&self) -> &str {
        "PSG003_TYPE_FIELD_DOCS"
    }
    
    fn description(&self) -> &str {
        "Public types should document their fields"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        matches!(element.element_type, crate::extraction::DocumentationElementType::Type) &&
        matches!(element.visibility, crate::extraction::ElementVisibility::Public)
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        // Check if there are @field annotations
        let has_field_docs = element.annotations.iter()
            .any(|attr| attr.name == "field");
            
        if !has_field_docs {
            Ok(vec![ValidationViolation {
                violation_type: ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Warning,
                message: "Public type should document its fields with @field annotations".to_string(),
                location: element.location,
                suggested_fix: Some("Add @field annotations for each public field".to_string()),
                rule_id: self.rule_id().to_string(),
                context: HashMap::new(),
            }])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Rule: @responsibility annotations must be within character limit
#[derive(Debug)]
pub struct ResponsibilityLengthRule;

impl ResponsibilityLengthRule {
    pub fn new() -> Self {
        Self
    }
    
    const MAX_RESPONSIBILITY_LENGTH: usize = 80;
}

impl ValidationRule for ResponsibilityLengthRule {
    fn rule_id(&self) -> &str {
        "PSG003_RESPONSIBILITY_LENGTH"
    }
    
    fn description(&self) -> &str {
        "Responsibility annotations must be within 80 character limit"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        element.annotations.iter().any(|attr| attr.name == "responsibility")
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();
        
        for annotation in &element.annotations {
            if annotation.name == "responsibility" {
                if let Some(value) = &annotation.value {
                    if value.len() > Self::MAX_RESPONSIBILITY_LENGTH {
                        violations.push(ValidationViolation {
                            violation_type: ViolationType::InvalidAnnotationFormat,
                            severity: ViolationSeverity::Error,
                            message: format!(
                                "Responsibility annotation too long: {} characters (max {})",
                                value.len(),
                                Self::MAX_RESPONSIBILITY_LENGTH
                            ),
                            location: annotation.location,
                            suggested_fix: Some("Shorten responsibility statement to 80 characters or less".to_string()),
                            rule_id: self.rule_id().to_string(),
                            context: [("length".to_string(), value.len().to_string())].into(),
                        });
                    }
                }
            }
        }
        
        Ok(violations)
    }
}

/// Rule: @description annotations should be within reasonable length
#[derive(Debug)]
pub struct DescriptionLengthRule;

impl DescriptionLengthRule {
    pub fn new() -> Self {
        Self
    }
    
    const MAX_DESCRIPTION_LENGTH: usize = 120;
}

impl ValidationRule for DescriptionLengthRule {
    fn rule_id(&self) -> &str {
        "PSG003_DESCRIPTION_LENGTH"
    }
    
    fn description(&self) -> &str {
        "Description annotations should be within 120 character limit"
    }
    
    fn applies_to(&self, element: &DocumentationElement) -> bool {
        element.annotations.iter().any(|attr| attr.name == "description")
    }
    
    fn validate(&self, element: &DocumentationElement) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();
        
        for annotation in &element.annotations {
            if annotation.name == "description" {
                if let Some(value) = &annotation.value {
                    if value.len() > Self::MAX_DESCRIPTION_LENGTH {
                        violations.push(ValidationViolation {
                            violation_type: ViolationType::InvalidAnnotationFormat,
                            severity: ViolationSeverity::Warning,
                            message: format!(
                                "Description annotation quite long: {} characters (recommended max {})",
                                value.len(),
                                Self::MAX_DESCRIPTION_LENGTH
                            ),
                            location: annotation.location,
                            suggested_fix: Some("Consider shortening description or breaking into multiple lines".to_string()),
                            rule_id: self.rule_id().to_string(),
                            context: [("length".to_string(), value.len().to_string())].into(),
                        });
                    }
                }
            }
        }
        
        Ok(violations)
    }
} 