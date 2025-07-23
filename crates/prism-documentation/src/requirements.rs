//! PSG-003 requirement checking
//!
//! This module embodies the single concept of "PSG-003 Requirement Checking".
//! Following Prism's Conceptual Cohesion principle, this module is responsible
//! for ONE thing: enforcing PSG-003 required annotations and documentation standards.

use crate::{DocumentationError, DocumentationResult, ValidationViolation, ViolationSeverity};
use crate::extraction::{DocumentationElement, DocumentationElementType, ExtractedAnnotation};
use prism_ast::{Program, AstNode, Item, FunctionDecl, TypeDecl, ModuleDecl, Visibility};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Requirement checker for PSG-003 compliance
#[derive(Debug)]
pub struct RequirementChecker {
    /// Requirement rules configuration
    config: RequirementConfig,
    /// Required annotation registry
    required_annotations: RequiredAnnotationRegistry,
}

/// Configuration for requirement checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementConfig {
    /// Enforce module-level requirements
    pub enforce_module_requirements: bool,
    /// Enforce function-level requirements
    pub enforce_function_requirements: bool,
    /// Enforce type-level requirements
    pub enforce_type_requirements: bool,
    /// Enforce public visibility requirements
    pub enforce_public_requirements: bool,
    /// Custom requirement overrides
    pub custom_overrides: HashMap<String, RequirementOverride>,
    /// Strictness level
    pub strictness: RequirementStrictness,
}

/// Requirement strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementStrictness {
    /// Lenient - warnings only
    Lenient,
    /// Standard - PSG-003 compliance
    Standard,
    /// Strict - all requirements enforced
    Strict,
    /// Pedantic - maximum enforcement
    Pedantic,
}

/// Requirement override configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementOverride {
    /// Override type
    pub override_type: OverrideType,
    /// Elements to apply override to
    pub target_elements: Vec<String>,
    /// Override reason
    pub reason: String,
}

/// Types of requirement overrides
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverrideType {
    /// Disable requirement
    Disable,
    /// Make requirement optional
    Optional,
    /// Change severity level
    ChangeSeverity(ViolationSeverity),
}

/// Required annotation registry
#[derive(Debug)]
pub struct RequiredAnnotationRegistry {
    /// Module-level requirements
    module_requirements: Vec<RequiredAnnotationType>,
    /// Function-level requirements
    function_requirements: Vec<RequiredAnnotationType>,
    /// Type-level requirements
    type_requirements: Vec<RequiredAnnotationType>,
    /// Visibility-based requirements
    visibility_requirements: HashMap<Visibility, Vec<RequiredAnnotationType>>,
}

/// Types of required annotations according to PSG-003
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequiredAnnotationType {
    /// @responsibility annotation (required for modules, functions, types)
    Responsibility,
    /// @module annotation (required for modules)
    Module,
    /// @description annotation (required for public elements)
    Description,
    /// @author annotation (required for modules)
    Author,
    /// @param annotation (required for public functions with parameters)
    Param(String),
    /// @returns annotation (required for public functions with return values)
    Returns,
    /// @throws annotation (required for functions that can throw)
    Throws(String),
    /// @example annotation (recommended for public functions)
    Example,
    /// @effects annotation (required for functions with effects)
    Effects,
    /// @field annotation (required for public type fields)
    Field(String),
    /// @invariant annotation (recommended for types)
    Invariant,
    /// @aiContext annotation (recommended for AI integration)
    AIContext,
}

/// Annotation requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationRequirement {
    /// Annotation type
    pub annotation_type: RequiredAnnotationType,
    /// Requirement severity
    pub severity: ViolationSeverity,
    /// Whether requirement applies to private elements
    pub applies_to_private: bool,
    /// Custom validation rule
    pub custom_validation: Option<String>,
    /// Suggested content for missing annotation
    pub suggested_content: Option<String>,
}

impl RequirementChecker {
    /// Create a new requirement checker with default configuration
    pub fn new() -> Self {
        Self {
            config: RequirementConfig::default(),
            required_annotations: RequiredAnnotationRegistry::new(),
        }
    }

    /// Create a new requirement checker with custom configuration
    pub fn with_config(config: RequirementConfig) -> Self {
        Self {
            required_annotations: RequiredAnnotationRegistry::new(),
            config,
        }
    }

    /// Check requirements for a complete program
    pub fn check_program(&self, program: &Program) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for item in &program.items {
            let item_violations = self.check_item(item)?;
            violations.extend(item_violations);
        }

        Ok(violations)
    }

    /// Check requirements for a single item
    pub fn check_item(&self, item: &AstNode<Item>) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        match &item.kind {
            Item::Module(module_decl) => {
                let module_violations = self.check_module_requirements(item, module_decl)?;
                violations.extend(module_violations);
            }
            Item::Function(func_decl) => {
                let func_violations = self.check_function_requirements(item, func_decl)?;
                violations.extend(func_violations);
            }
            Item::Type(type_decl) => {
                let type_violations = self.check_type_requirements(item, type_decl)?;
                violations.extend(type_violations);
            }
            Item::Const(const_decl) => {
                let const_violations = self.check_const_requirements(item, const_decl)?;
                violations.extend(const_violations);
            }
            _ => {
                // Other item types have minimal requirements
            }
        }

        Ok(violations)
    }

    /// Check module requirements according to PSG-003
    fn check_module_requirements(&self, item: &AstNode<Item>, module_decl: &ModuleDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if !self.config.enforce_module_requirements {
            return Ok(violations);
        }

        // Extract existing annotations
        let existing_annotations = self.extract_annotation_names(&module_decl.attributes);

        // Check required annotations
        for requirement in &self.required_annotations.module_requirements {
            if !self.has_required_annotation(&existing_annotations, requirement) {
                let violation = self.create_missing_annotation_violation(
                    requirement,
                    &module_decl.name.to_string(),
                    item.span,
                    "module",
                );
                violations.push(violation);
            }
        }

        // Validate annotation content
        violations.extend(self.validate_module_annotation_content(module_decl)?);

        Ok(violations)
    }

    /// Check function requirements according to PSG-003
    fn check_function_requirements(&self, item: &AstNode<Item>, func_decl: &FunctionDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if !self.config.enforce_function_requirements {
            return Ok(violations);
        }

        // Extract existing annotations
        let existing_annotations = self.extract_annotation_names(&func_decl.attributes);

        // Check basic requirements
        for requirement in &self.required_annotations.function_requirements {
            if !self.has_required_annotation(&existing_annotations, requirement) {
                let violation = self.create_missing_annotation_violation(
                    requirement,
                    &func_decl.name.to_string(),
                    item.span,
                    "function",
                );
                violations.push(violation);
            }
        }

        // Check visibility-specific requirements
        if let Some(visibility_requirements) = self.required_annotations.visibility_requirements.get(&func_decl.visibility) {
            for requirement in visibility_requirements {
                if !self.has_required_annotation(&existing_annotations, requirement) {
                    let violation = self.create_missing_annotation_violation(
                        requirement,
                        &func_decl.name.to_string(),
                        item.span,
                        &format!("{:?} function", func_decl.visibility),
                    );
                    violations.push(violation);
                }
            }
        }

        // Check parameter documentation for public functions
        if func_decl.visibility == Visibility::Public {
            violations.extend(self.check_parameter_documentation(func_decl, item.span)?);
        }

        // Validate annotation content
        violations.extend(self.validate_function_annotation_content(func_decl)?);

        Ok(violations)
    }

    /// Check type requirements according to PSG-003
    fn check_type_requirements(&self, item: &AstNode<Item>, type_decl: &TypeDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if !self.config.enforce_type_requirements {
            return Ok(violations);
        }

        // Extract existing annotations
        let existing_annotations = self.extract_annotation_names(&type_decl.attributes);

        // Check required annotations
        for requirement in &self.required_annotations.type_requirements {
            if !self.has_required_annotation(&existing_annotations, requirement) {
                let violation = self.create_missing_annotation_violation(
                    requirement,
                    &type_decl.name.to_string(),
                    item.span,
                    "type",
                );
                violations.push(violation);
            }
        }

        // Check visibility-specific requirements
        if let Some(visibility_requirements) = self.required_annotations.visibility_requirements.get(&type_decl.visibility) {
            for requirement in visibility_requirements {
                if !self.has_required_annotation(&existing_annotations, requirement) {
                    let violation = self.create_missing_annotation_violation(
                        requirement,
                        &type_decl.name.to_string(),
                        item.span,
                        &format!("{:?} type", type_decl.visibility),
                    );
                    violations.push(violation);
                }
            }
        }

        // Validate annotation content
        violations.extend(self.validate_type_annotation_content(type_decl)?);

        Ok(violations)
    }

    /// Check constant requirements
    fn check_const_requirements(&self, item: &AstNode<Item>, const_decl: &prism_ast::ConstDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        // Constants have minimal requirements unless they're public
        if const_decl.visibility == Visibility::Public {
            let existing_annotations = self.extract_annotation_names(&const_decl.attributes);
            
            // Public constants should have descriptions
            if !existing_annotations.contains("description") && !existing_annotations.contains("responsibility") {
                violations.push(ValidationViolation {
                    violation_type: crate::validation::ViolationType::MissingRequiredAnnotation,
                    severity: ViolationSeverity::Warning,
                    message: format!("Public constant '{}' should have a description or responsibility annotation", const_decl.name),
                    location: item.span,
                    suggested_fix: Some(format!("Add @description or @responsibility annotation to explain the purpose of '{}'", const_decl.name)),
                    rule_id: "PSG003_MISSING_CONST_DESCRIPTION".to_string(),
                    context: {
                        let mut context = HashMap::new();
                        context.insert("element_type".to_string(), "constant".to_string());
                        context.insert("element_name".to_string(), const_decl.name.to_string());
                        context.insert("visibility".to_string(), format!("{:?}", const_decl.visibility));
                        context
                    },
                });
            }
        }

        Ok(violations)
    }

    /// Check parameter documentation for functions
    fn check_parameter_documentation(&self, func_decl: &FunctionDecl, span: Span) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        // For now, we'll use a simplified approach since the current AST might not have detailed parameter info
        // In a full implementation, this would check each parameter against @param annotations

        let existing_annotations = self.extract_annotation_names(&func_decl.attributes);
        
        // Check if function has parameters (simplified check)
        let has_param_annotations = existing_annotations.iter().any(|name| name == "param");
        
        // If this is a public function and we don't have param annotations, suggest adding them
        if func_decl.visibility == Visibility::Public && !has_param_annotations {
            violations.push(ValidationViolation {
                violation_type: crate::validation::ViolationType::MissingRequiredAnnotation,
                severity: ViolationSeverity::Warning,
                message: format!("Public function '{}' should document its parameters with @param annotations", func_decl.name),
                location: span,
                suggested_fix: Some("Add @param annotations for each function parameter".to_string()),
                rule_id: "PSG003_MISSING_PARAM_DOCS".to_string(),
                context: {
                    let mut context = HashMap::new();
                    context.insert("element_type".to_string(), "function".to_string());
                    context.insert("element_name".to_string(), func_decl.name.to_string());
                    context.insert("missing_annotation".to_string(), "param".to_string());
                    context
                },
            });
        }

        Ok(violations)
    }

    /// Validate module annotation content
    fn validate_module_annotation_content(&self, module_decl: &ModuleDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for attr in &module_decl.attributes {
            match attr.name.resolve().as_deref() {
                Some("responsibility") => {
                    violations.extend(self.validate_responsibility_content(attr, &module_decl.name.resolve().unwrap_or_default(), "module")?);
                }
                Some("description") => {
                    violations.extend(self.validate_description_content(attr, &module_decl.name.resolve().unwrap_or_default(), "module")?);
                }
                Some("author") => {
                    violations.extend(self.validate_author_content(attr, &module_decl.name.resolve().unwrap_or_default())?);
                }
                _ => {} // Other annotations don't need special validation
            }
        }

        Ok(violations)
    }

    /// Validate function annotation content
    fn validate_function_annotation_content(&self, func_decl: &FunctionDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for attr in &func_decl.attributes {
            match attr.name.resolve().as_deref() {
                Some("responsibility") => {
                    violations.extend(self.validate_responsibility_content(attr, &func_decl.name.resolve().unwrap_or_default(), "function")?);
                }
                Some("effects") => {
                    violations.extend(self.validate_effects_content(attr, &func_decl.name.resolve().unwrap_or_default())?);
                }
                _ => {} // Other annotations don't need special validation
            }
        }

        Ok(violations)
    }

    /// Validate type annotation content
    fn validate_type_annotation_content(&self, type_decl: &TypeDecl) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for attr in &type_decl.attributes {
            match attr.name.resolve().as_deref() {
                Some("responsibility") => {
                    violations.extend(self.validate_responsibility_content(attr, &type_decl.name.resolve().unwrap_or_default(), "type")?);
                }
                _ => {} // Other annotations don't need special validation
            }
        }

        Ok(violations)
    }

    /// Validate responsibility annotation content
    fn validate_responsibility_content(&self, attr: &prism_ast::Attribute, element_name: &str, element_type: &str) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if let Some(value) = self.get_first_string_argument(attr) {
            // Check length (PSG-003 specifies max 80 characters)
            if value.len() > 80 {
                violations.push(ValidationViolation {
                    violation_type: crate::validation::ViolationType::InvalidAnnotationFormat,
                    severity: ViolationSeverity::Error,
                    message: format!("Responsibility annotation for {} '{}' exceeds 80 character limit ({} characters)", element_type, element_name, value.len()),
                    location: Span::dummy(), // TODO: Get actual attribute location
                    suggested_fix: Some("Shorten the responsibility statement to 80 characters or less".to_string()),
                    rule_id: "PSG003_RESPONSIBILITY_TOO_LONG".to_string(),
                    context: {
                        let mut context = HashMap::new();
                        context.insert("element_type".to_string(), element_type.to_string());
                        context.insert("element_name".to_string(), element_name.to_string());
                        context.insert("current_length".to_string(), value.len().to_string());
                        context.insert("max_length".to_string(), "80".to_string());
                        context
                    },
                });
            }

            // Check for empty content
            if value.trim().is_empty() {
                violations.push(ValidationViolation {
                    violation_type: crate::validation::ViolationType::InvalidAnnotationFormat,
                    severity: ViolationSeverity::Error,
                    message: format!("Responsibility annotation for {} '{}' cannot be empty", element_type, element_name),
                    location: Span::dummy(),
                    suggested_fix: Some("Provide a clear, concise responsibility statement".to_string()),
                    rule_id: "PSG003_EMPTY_RESPONSIBILITY".to_string(),
                    context: {
                        let mut context = HashMap::new();
                        context.insert("element_type".to_string(), element_type.to_string());
                        context.insert("element_name".to_string(), element_name.to_string());
                        context
                    },
                });
            }
        }

        Ok(violations)
    }

    /// Validate description annotation content
    fn validate_description_content(&self, attr: &prism_ast::Attribute, element_name: &str, element_type: &str) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if let Some(value) = self.get_first_string_argument(attr) {
            // Check length (PSG-003 specifies max 100 characters for descriptions)
            if value.len() > 100 {
                violations.push(ValidationViolation {
                    violation_type: crate::validation::ViolationType::InvalidAnnotationFormat,
                    severity: ViolationSeverity::Warning,
                    message: format!("Description annotation for {} '{}' exceeds recommended 100 character limit ({} characters)", element_type, element_name, value.len()),
                    location: Span::dummy(),
                    suggested_fix: Some("Consider shortening the description or moving details to documentation comments".to_string()),
                    rule_id: "PSG003_DESCRIPTION_TOO_LONG".to_string(),
                    context: {
                        let mut context = HashMap::new();
                        context.insert("element_type".to_string(), element_type.to_string());
                        context.insert("element_name".to_string(), element_name.to_string());
                        context.insert("current_length".to_string(), value.len().to_string());
                        context.insert("recommended_length".to_string(), "100".to_string());
                        context
                    },
                });
            }
        }

        Ok(violations)
    }

    /// Validate author annotation content
    fn validate_author_content(&self, attr: &prism_ast::Attribute, element_name: &str) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if let Some(value) = self.get_first_string_argument(attr) {
            if value.trim().is_empty() {
                violations.push(ValidationViolation {
                    violation_type: crate::validation::ViolationType::InvalidAnnotationFormat,
                    severity: ViolationSeverity::Warning,
                    message: format!("Author annotation for module '{}' cannot be empty", element_name),
                    location: Span::dummy(),
                    suggested_fix: Some("Provide author name or team name".to_string()),
                    rule_id: "PSG003_EMPTY_AUTHOR".to_string(),
                    context: {
                        let mut context = HashMap::new();
                        context.insert("element_type".to_string(), "module".to_string());
                        context.insert("element_name".to_string(), element_name.to_string());
                        context
                    },
                });
            }
        }

        Ok(violations)
    }

    /// Validate effects annotation content
    fn validate_effects_content(&self, attr: &prism_ast::Attribute, element_name: &str) -> DocumentationResult<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        if let Some(value) = self.get_first_string_argument(attr) {
            // Effects should follow a specific format: [Effect.Type, Effect.Type]
            if !value.starts_with('[') || !value.ends_with(']') {
                violations.push(ValidationViolation {
                    violation_type: crate::validation::ViolationType::InvalidAnnotationFormat,
                    severity: ViolationSeverity::Warning,
                    message: format!("Effects annotation for function '{}' should be in format [Effect.Type, Effect.Type]", element_name),
                    location: Span::dummy(),
                    suggested_fix: Some("Format effects as a comma-separated list in square brackets".to_string()),
                    rule_id: "PSG003_INVALID_EFFECTS_FORMAT".to_string(),
                    context: {
                        let mut context = HashMap::new();
                        context.insert("element_type".to_string(), "function".to_string());
                        context.insert("element_name".to_string(), element_name.to_string());
                        context.insert("current_format".to_string(), value.clone());
                        context
                    },
                });
            }
        }

        Ok(violations)
    }

    /// Extract annotation names from attributes
    fn extract_annotation_names(&self, attributes: &[prism_ast::Attribute]) -> HashSet<String> {
        attributes.iter().map(|attr| attr.name.clone()).collect()
    }

    /// Check if required annotation is present
    fn has_required_annotation(&self, existing: &HashSet<String>, required: &RequiredAnnotationType) -> bool {
        match required {
            RequiredAnnotationType::Responsibility => existing.contains("responsibility"),
            RequiredAnnotationType::Module => existing.contains("module"),
            RequiredAnnotationType::Description => existing.contains("description"),
            RequiredAnnotationType::Author => existing.contains("author"),
            RequiredAnnotationType::Param(_) => existing.contains("param"),
            RequiredAnnotationType::Returns => existing.contains("returns") || existing.contains("return"),
            RequiredAnnotationType::Throws(_) => existing.contains("throws") || existing.contains("throw"),
            RequiredAnnotationType::Example => existing.contains("example"),
            RequiredAnnotationType::Effects => existing.contains("effects"),
            RequiredAnnotationType::Field(_) => existing.contains("field"),
            RequiredAnnotationType::Invariant => existing.contains("invariant"),
            RequiredAnnotationType::AIContext => existing.contains("aiContext") || existing.contains("ai"),
        }
    }

    /// Create a missing annotation violation
    fn create_missing_annotation_violation(
        &self,
        requirement: &RequiredAnnotationType,
        element_name: &str,
        location: Span,
        element_type: &str,
    ) -> ValidationViolation {
        let annotation_name = self.get_annotation_name(requirement);
        let message = format!("Missing required @{} annotation for {} '{}'", annotation_name, element_type, element_name);
        let suggested_fix = self.get_suggested_fix(requirement, element_name);
        let rule_id = format!("PSG003_MISSING_{}", annotation_name.to_uppercase());

        ValidationViolation {
            violation_type: crate::validation::ViolationType::MissingRequiredAnnotation,
            severity: self.get_requirement_severity(requirement),
            message,
            location,
            suggested_fix: Some(suggested_fix),
            rule_id,
            context: {
                let mut context = HashMap::new();
                context.insert("element_type".to_string(), element_type.to_string());
                context.insert("element_name".to_string(), element_name.to_string());
                context.insert("missing_annotation".to_string(), annotation_name);
                context
            },
        }
    }

    /// Get annotation name from requirement type
    fn get_annotation_name(&self, requirement: &RequiredAnnotationType) -> String {
        match requirement {
            RequiredAnnotationType::Responsibility => "responsibility".to_string(),
            RequiredAnnotationType::Module => "module".to_string(),
            RequiredAnnotationType::Description => "description".to_string(),
            RequiredAnnotationType::Author => "author".to_string(),
            RequiredAnnotationType::Param(_) => "param".to_string(),
            RequiredAnnotationType::Returns => "returns".to_string(),
            RequiredAnnotationType::Throws(_) => "throws".to_string(),
            RequiredAnnotationType::Example => "example".to_string(),
            RequiredAnnotationType::Effects => "effects".to_string(),
            RequiredAnnotationType::Field(_) => "field".to_string(),
            RequiredAnnotationType::Invariant => "invariant".to_string(),
            RequiredAnnotationType::AIContext => "aiContext".to_string(),
        }
    }

    /// Get suggested fix for missing annotation
    fn get_suggested_fix(&self, requirement: &RequiredAnnotationType, element_name: &str) -> String {
        match requirement {
            RequiredAnnotationType::Responsibility => {
                format!("Add @responsibility \"<single responsibility statement>\" to define what {} is responsible for", element_name)
            }
            RequiredAnnotationType::Module => {
                format!("Add @module \"{}\" to specify the module name", element_name)
            }
            RequiredAnnotationType::Description => {
                format!("Add @description \"<brief description>\" to describe what {} does", element_name)
            }
            RequiredAnnotationType::Author => {
                "Add @author \"<author name or team>\" to specify who owns this module".to_string()
            }
            RequiredAnnotationType::Param(_) => {
                "Add @param <name> \"<description>\" for each function parameter".to_string()
            }
            RequiredAnnotationType::Returns => {
                "Add @returns \"<description of return value>\" to document what the function returns".to_string()
            }
            RequiredAnnotationType::Throws(_) => {
                "Add @throws <ErrorType> \"<description>\" for each exception the function can throw".to_string()
            }
            RequiredAnnotationType::Example => {
                "Add @example with a code example showing how to use this function".to_string()
            }
            RequiredAnnotationType::Effects => {
                "Add @effects [Effect.Type] to document what effects this function performs".to_string()
            }
            RequiredAnnotationType::Field(_) => {
                "Add @field <name> \"<description>\" for each public field".to_string()
            }
            RequiredAnnotationType::Invariant => {
                "Add @invariant \"<invariant description>\" to document type invariants".to_string()
            }
            RequiredAnnotationType::AIContext => {
                "Add @aiContext \"<AI context>\" to provide AI-comprehensible information".to_string()
            }
        }
    }

    /// Get requirement severity based on type and configuration
    fn get_requirement_severity(&self, requirement: &RequiredAnnotationType) -> ViolationSeverity {
        match self.config.strictness {
            RequirementStrictness::Lenient => ViolationSeverity::Warning,
            RequirementStrictness::Standard => {
                match requirement {
                    RequiredAnnotationType::Responsibility => ViolationSeverity::Error,
                    RequiredAnnotationType::Module => ViolationSeverity::Error,
                    RequiredAnnotationType::Author => ViolationSeverity::Warning,
                    _ => ViolationSeverity::Warning,
                }
            }
            RequirementStrictness::Strict => ViolationSeverity::Error,
            RequirementStrictness::Pedantic => ViolationSeverity::Error,
        }
    }

    /// Get first string argument from attribute
    fn get_first_string_argument(&self, attr: &prism_ast::Attribute) -> Option<String> {
        attr.arguments.first().and_then(|arg| match arg {
            prism_ast::AttributeArgument::Literal(prism_ast::LiteralValue::String(s)) => Some(s.clone()),
            _ => None,
        })
    }
}

impl RequiredAnnotationRegistry {
    /// Create a new registry with default PSG-003 requirements
    pub fn new() -> Self {
        let mut registry = Self {
            module_requirements: Vec::new(),
            function_requirements: Vec::new(),
            type_requirements: Vec::new(),
            visibility_requirements: HashMap::new(),
        };

        // Set up default PSG-003 requirements
        registry.setup_default_requirements();
        registry
    }

    /// Set up default PSG-003 requirements
    fn setup_default_requirements(&mut self) {
        // Module requirements (PSG-003)
        self.module_requirements = vec![
            RequiredAnnotationType::Responsibility,
            RequiredAnnotationType::Module,
            RequiredAnnotationType::Description,
            RequiredAnnotationType::Author,
        ];

        // Function requirements (PSG-003)
        self.function_requirements = vec![
            RequiredAnnotationType::Responsibility,
        ];

        // Type requirements (PSG-003)
        self.type_requirements = vec![
            RequiredAnnotationType::Responsibility,
        ];

        // Public visibility requirements
        let mut public_requirements = vec![
            RequiredAnnotationType::Description,
            RequiredAnnotationType::Example,
        ];
        self.visibility_requirements.insert(Visibility::Public, public_requirements);
    }
}

impl Default for RequirementConfig {
    fn default() -> Self {
        Self {
            enforce_module_requirements: true,
            enforce_function_requirements: true,
            enforce_type_requirements: true,
            enforce_public_requirements: true,
            custom_overrides: HashMap::new(),
            strictness: RequirementStrictness::Standard,
        }
    }
}

impl Default for RequirementChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ast::{ModuleDecl, FunctionDecl, Attribute, AttributeArgument, LiteralValue};
    use prism_common::symbol::Symbol;

    #[test]
    fn test_module_requirement_checking() {
        let checker = RequirementChecker::new();
        
        // Create a module without required annotations
        let module_decl = ModuleDecl {
            name: Symbol::intern("TestModule"),
            attributes: vec![], // No attributes - should trigger violations
            sections: vec![],
            visibility: Visibility::Public,
        };

        let item = AstNode::new(
            Item::Module(module_decl),
            Span::dummy(),
            prism_common::NodeId::new(1),
        );

        let violations = checker.check_item(&item).unwrap();
        
        // Should have violations for missing required annotations
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.message.contains("@responsibility")));
        assert!(violations.iter().any(|v| v.message.contains("@module")));
        assert!(violations.iter().any(|v| v.message.contains("@description")));
        assert!(violations.iter().any(|v| v.message.contains("@author")));
    }

    #[test]
    fn test_function_requirement_checking() {
        let checker = RequirementChecker::new();
        
        // Create a public function without required annotations
        let func_decl = FunctionDecl {
            name: Symbol::intern("testFunction"),
            attributes: vec![], // No attributes - should trigger violations
            parameters: vec![],
            return_type: None,
            body: None,
            visibility: Visibility::Public,
        };

        let item = AstNode::new(
            Item::Function(func_decl),
            Span::dummy(),
            prism_common::NodeId::new(1),
        );

        let violations = checker.check_item(&item).unwrap();
        
        // Should have violations for missing required annotations
        assert!(!violations.is_empty());
        assert!(violations.iter().any(|v| v.message.contains("@responsibility")));
    }

    #[test]
    fn test_responsibility_content_validation() {
        let checker = RequirementChecker::new();
        
        // Create attribute with too-long responsibility
        let long_responsibility = "This is a very long responsibility statement that definitely exceeds the eighty character limit specified in PSG-003 standards";
        let attr = Attribute {
            name: "responsibility".to_string(),
            arguments: vec![AttributeArgument::Literal(LiteralValue::String(long_responsibility.to_string()))],
        };

        let violations = checker.validate_responsibility_content(&attr, "TestElement", "function").unwrap();
        
        assert!(!violations.is_empty());
        assert!(violations[0].message.contains("exceeds 80 character limit"));
        assert_eq!(violations[0].severity, ViolationSeverity::Error);
    }

    #[test]
    fn test_requirement_strictness_levels() {
        let mut config = RequirementConfig::default();
        
        // Test lenient mode
        config.strictness = RequirementStrictness::Lenient;
        let checker = RequirementChecker::with_config(config.clone());
        let severity = checker.get_requirement_severity(&RequiredAnnotationType::Responsibility);
        assert_eq!(severity, ViolationSeverity::Warning);
        
        // Test strict mode
        config.strictness = RequirementStrictness::Strict;
        let checker = RequirementChecker::with_config(config);
        let severity = checker.get_requirement_severity(&RequiredAnnotationType::Responsibility);
        assert_eq!(severity, ViolationSeverity::Error);
    }
} 