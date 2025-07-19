//! Main validation engine for syntax compliance.

use crate::normalization::CanonicalForm;
use thiserror::Error;

/// Main validator for syntax compliance
#[derive(Debug)]
pub struct Validator {
    /// Validation rules
    rules: Vec<ValidationRule>,
}

/// Result of validation process
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Overall validation score (0.0 to 1.0)
    pub overall_score: f64,
    
    /// Individual rule results
    pub rule_results: Vec<RuleResult>,
    
    /// Whether validation passed
    pub passed: bool,
    
    /// Validation errors
    pub errors: Vec<ValidationError>,
    
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Individual validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    
    /// Rule description
    pub description: String,
    
    /// Rule severity
    pub severity: RuleSeverity,
    
    /// Rule implementation
    pub rule_fn: fn(&CanonicalForm) -> RuleResult,
}

/// Result of applying a single rule
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// Rule name
    pub rule_name: String,
    
    /// Whether rule passed
    pub passed: bool,
    
    /// Score for this rule (0.0 to 1.0)
    pub score: f64,
    
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Severity of validation rules
#[derive(Debug, Clone, PartialEq)]
pub enum RuleSeverity {
    /// Error - must be fixed
    Error,
    
    /// Warning - should be fixed
    Warning,
    
    /// Info - nice to have
    Info,
}

/// Validation error
#[derive(Debug, Error, Clone)]
pub enum ValidationError {
    /// Rule validation failed
    #[error("Rule '{rule}' failed: {message}")]
    RuleFailed { rule: String, message: String },
    
    /// Missing required element
    #[error("Missing required element: {element}")]
    MissingRequired { element: String },
    
    /// Invalid structure
    #[error("Invalid structure: {description}")]
    InvalidStructure { description: String },
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    
    /// Source rule
    pub rule: String,
    
    /// Suggested fix
    pub suggestion: Option<String>,
}

impl Validator {
    /// Create a new validator with default rules
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
        }
    }
    
    /// Validate canonical form against all rules
    pub fn validate(&self, canonical: &CanonicalForm) -> Result<ValidationResult, ValidationError> {
        let mut rule_results = Vec::new();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Apply all validation rules
        for rule in &self.rules {
            let result = (rule.rule_fn)(canonical);
            
            if !result.passed {
                match rule.severity {
                    RuleSeverity::Error => {
                        errors.push(ValidationError::RuleFailed {
                            rule: rule.name.clone(),
                            message: result.error_message.unwrap_or_default(),
                        });
                    }
                    RuleSeverity::Warning => {
                        warnings.push(ValidationWarning {
                            message: result.error_message.unwrap_or_default(),
                            rule: rule.name.clone(),
                            suggestion: None,
                        });
                    }
                    RuleSeverity::Info => {
                        // Info level issues don't affect validation
                    }
                }
            }
            
            rule_results.push(result);
        }
        
        // Calculate overall score
        let total_score: f64 = rule_results.iter().map(|r| r.score).sum();
        let overall_score = if rule_results.is_empty() {
            1.0
        } else {
            total_score / rule_results.len() as f64
        };
        
        let passed = errors.is_empty();
        
        Ok(ValidationResult {
            overall_score,
            rule_results,
            passed,
            errors,
            warnings,
        })
    }
    
    /// Create default validation rules
    fn default_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "basic_structure".to_string(),
                description: "Basic structure validation".to_string(),
                severity: RuleSeverity::Error,
                rule_fn: Self::validate_basic_structure,
            },
            ValidationRule {
                name: "documentation_presence".to_string(),
                description: "Documentation presence check".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: Self::validate_documentation,
            },
        ]
    }
    
    /// Validate basic structure
    fn validate_basic_structure(_canonical: &CanonicalForm) -> RuleResult {
        // TODO: Implement actual structure validation
        RuleResult {
            rule_name: "basic_structure".to_string(),
            passed: true,
            score: 1.0,
            error_message: None,
        }
    }
    
    /// Validate documentation presence
    fn validate_documentation(_canonical: &CanonicalForm) -> RuleResult {
        // TODO: Implement actual documentation validation
        RuleResult {
            rule_name: "documentation_presence".to_string(),
            passed: true,
            score: 0.8,
            error_message: None,
        }
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
} 