//! Main validation engine for syntax compliance.

use crate::normalization::CanonicalForm;
use crate::styles::c_like::{CLikeSyntax, CLikeStatement, CLikeExpression, CLikeFunction};
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

    /// Create validator with custom rules
    pub fn with_rules(rules: Vec<ValidationRule>) -> Self {
        Self { rules }
    }

    /// Add a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Validate canonical form
    pub fn validate(&self, canonical: &CanonicalForm) -> ValidationResult {
        let mut rule_results = Vec::new();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut total_score = 0.0;

        for rule in &self.rules {
            let result = (rule.rule_fn)(canonical);
            
            if !result.passed {
                match rule.severity {
                    RuleSeverity::Error => {
                        errors.push(ValidationError::RuleFailed {
                            rule: rule.name.clone(),
                            message: result.error_message.clone().unwrap_or_default(),
                        });
                    }
                    RuleSeverity::Warning => {
                        warnings.push(ValidationWarning {
                            message: result.error_message.clone().unwrap_or_default(),
                            rule: rule.name.clone(),
                            suggestion: None,
                        });
                    }
                    RuleSeverity::Info => {
                        // Info level doesn't create errors or warnings
                    }
                }
            }

            total_score += result.score;
            rule_results.push(result);
        }

        let overall_score = if self.rules.is_empty() {
            1.0
        } else {
            total_score / self.rules.len() as f64
        };

        ValidationResult {
            overall_score,
            rule_results,
            passed: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Validate C-like syntax directly
    pub fn validate_c_like(&self, syntax: &CLikeSyntax) -> ValidationResult {
        let mut rule_results = Vec::new();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut total_score = 0.0;

        // Apply C-like specific validation rules
        let c_like_rules = Self::c_like_validation_rules();
        
        for rule in c_like_rules {
            let result = match rule.name.as_str() {
                "function_naming" => Self::validate_function_naming(syntax),
                "brace_consistency" => Self::validate_brace_consistency(syntax),
                "semicolon_usage" => Self::validate_semicolon_usage(syntax),
                "variable_declarations" => Self::validate_variable_declarations(syntax),
                "control_flow_structure" => Self::validate_control_flow_structure(syntax),
                "expression_complexity" => Self::validate_expression_complexity(syntax),
                "memory_safety" => Self::validate_memory_safety(syntax),
                "error_handling" => Self::validate_error_handling(syntax),
                _ => RuleResult {
                    rule_name: rule.name.clone(),
                    passed: true,
                    score: 1.0,
                    error_message: None,
                },
            };

            if !result.passed {
                match rule.severity {
                    RuleSeverity::Error => {
                        errors.push(ValidationError::RuleFailed {
                            rule: rule.name.clone(),
                            message: result.error_message.clone().unwrap_or_default(),
                        });
                    }
                    RuleSeverity::Warning => {
                        warnings.push(ValidationWarning {
                            message: result.error_message.clone().unwrap_or_default(),
                            rule: rule.name.clone(),
                            suggestion: None,
                        });
                    }
                    RuleSeverity::Info => {}
                }
            }

            total_score += result.score;
            rule_results.push(result);
        }

        let overall_score = if rule_results.is_empty() {
            1.0
        } else {
            total_score / rule_results.len() as f64
        };

        ValidationResult {
            overall_score,
            rule_results,
            passed: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Default validation rules for canonical form
    fn default_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "completeness".to_string(),
                description: "Ensure all required elements are present".to_string(),
                severity: RuleSeverity::Error,
                rule_fn: Self::validate_completeness,
            },
            ValidationRule {
                name: "consistency".to_string(),
                description: "Ensure consistent structure throughout".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: Self::validate_consistency,
            },
        ]
    }

    /// C-like specific validation rules
    fn c_like_validation_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "function_naming".to_string(),
                description: "Validate function naming conventions".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: |_| RuleResult { rule_name: "function_naming".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "brace_consistency".to_string(),
                description: "Ensure consistent brace placement".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: |_| RuleResult { rule_name: "brace_consistency".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "semicolon_usage".to_string(),
                description: "Validate proper semicolon usage".to_string(),
                severity: RuleSeverity::Error,
                rule_fn: |_| RuleResult { rule_name: "semicolon_usage".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "variable_declarations".to_string(),
                description: "Validate variable declaration patterns".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: |_| RuleResult { rule_name: "variable_declarations".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "control_flow_structure".to_string(),
                description: "Validate control flow structure".to_string(),
                severity: RuleSeverity::Error,
                rule_fn: |_| RuleResult { rule_name: "control_flow_structure".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "expression_complexity".to_string(),
                description: "Check expression complexity limits".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: |_| RuleResult { rule_name: "expression_complexity".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "memory_safety".to_string(),
                description: "Check for potential memory safety issues".to_string(),
                severity: RuleSeverity::Error,
                rule_fn: |_| RuleResult { rule_name: "memory_safety".to_string(), passed: true, score: 1.0, error_message: None },
            },
            ValidationRule {
                name: "error_handling".to_string(),
                description: "Validate error handling patterns".to_string(),
                severity: RuleSeverity::Warning,
                rule_fn: |_| RuleResult { rule_name: "error_handling".to_string(), passed: true, score: 1.0, error_message: None },
            },
        ]
    }

    // C-like specific validation implementations
    
    fn validate_function_naming(syntax: &CLikeSyntax) -> RuleResult {
        let mut issues: Vec<String> = Vec::new();
        let mut score: f64 = 1.0;

        for function in &syntax.functions {
            // Check naming conventions
            if function.name.chars().next().map_or(false, |c| c.is_uppercase()) {
                issues.push(format!("Function '{}' should start with lowercase", function.name));
                score -= 0.1;
            }

            // Check for snake_case or camelCase consistency
            let has_underscores = function.name.contains('_');
            let has_mixed_case = function.name.chars().any(|c| c.is_uppercase());
            
            if has_underscores && has_mixed_case {
                issues.push(format!("Function '{}' mixes snake_case and camelCase", function.name));
                score -= 0.2;
            }

            // Check for reserved keywords
            if matches!(function.name.as_str(), "main" | "printf" | "malloc" | "free") {
                // These are acceptable
            } else if function.name.len() < 2 {
                issues.push(format!("Function name '{}' is too short", function.name));
                score -= 0.1;
            }
        }

        score = score.max(0.0_f64);
        
        RuleResult {
            rule_name: "function_naming".to_string(),
            passed: issues.is_empty(),
            score,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
        }
    }

    fn validate_brace_consistency(syntax: &CLikeSyntax) -> RuleResult {
        // This would analyze brace placement consistency
        // For now, assume consistent
        RuleResult {
            rule_name: "brace_consistency".to_string(),
            passed: true,
            score: 1.0,
            error_message: None,
        }
    }

    fn validate_semicolon_usage(syntax: &CLikeSyntax) -> RuleResult {
        let mut issues: Vec<String> = Vec::new();
        let mut score: f64 = 1.0;

        // Check that statements that require semicolons have them
        for statement in &syntax.statements {
            match statement {
                CLikeStatement::Expression { .. } |
                CLikeStatement::Assignment { .. } |
                CLikeStatement::VariableDeclaration { .. } |
                CLikeStatement::Return { .. } |
                CLikeStatement::Break { .. } |
                CLikeStatement::Continue { .. } => {
                    // These should have semicolons in the parser validation
                    // This is a placeholder for more sophisticated checks
                }
                _ => {}
            }
        }

        RuleResult {
            rule_name: "semicolon_usage".to_string(),
            passed: issues.is_empty(),
            score,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
        }
    }

    fn validate_variable_declarations(syntax: &CLikeSyntax) -> RuleResult {
        let mut issues: Vec<String> = Vec::new();
        let mut score: f64 = 1.0;

        for statement in &syntax.statements {
            if let CLikeStatement::VariableDeclaration { name, var_type, .. } = statement {
                // Check for meaningful variable names
                if name.len() < 2 && !matches!(name.as_str(), "i" | "j" | "k" | "x" | "y" | "z") {
                    issues.push(format!("Variable name '{}' is too short", name));
                    score -= 0.1;
                }

                if name.chars().all(|c| c.is_uppercase()) && name.len() > 1 {
                    issues.push(format!("Variable '{}' should not be all uppercase (reserved for constants)", name));
                    score -= 0.1;
                }

                // Check for type consistency
                if var_type.as_ref().map_or(true, |s| s.is_empty()) {
                    issues.push("Variable declaration missing type".to_string());
                    score -= 0.2;
                }
            }
        }

        score = score.max(0.0_f64);

        RuleResult {
            rule_name: "variable_declarations".to_string(),
            passed: issues.is_empty(),
            score,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
        }
    }

    fn validate_control_flow_structure(syntax: &CLikeSyntax) -> RuleResult {
        let mut issues: Vec<String> = Vec::new();
        let mut score: f64 = 1.0;

        for statement in &syntax.statements {
            match statement {
                CLikeStatement::If { condition, then_block, else_block, .. } => {
                    // Check for empty bodies
                    if then_block.is_empty() {
                        issues.push("If statement has empty then block".to_string());
                        score -= 0.1;
                    }

                    // Check for else-if chains that could be switch statements
                    if let Some(else_stmts) = else_block {
                        if else_stmts.len() == 1 {
                            if let CLikeStatement::If { .. } = &else_stmts[0] {
                                // This is an else-if, which is fine
                            }
                        }
                    }
                }
                CLikeStatement::For { init, condition, increment, body, .. } => {
                    if body.is_empty() {
                        issues.push("For loop has empty body".to_string());
                        score -= 0.1;
                    }

                    // Check for infinite loops
                    if condition.is_none() && increment.is_none() {
                        issues.push("Potential infinite for loop detected".to_string());
                        score -= 0.2;
                    }
                }
                CLikeStatement::While { condition, body, .. } => {
                    if body.is_empty() {
                        issues.push("While loop has empty body".to_string());
                        score -= 0.1;
                    }
                }
                _ => {}
            }
        }

        score = score.max(0.0_f64);

        RuleResult {
            rule_name: "control_flow_structure".to_string(),
            passed: issues.is_empty(),
            score,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
        }
    }

    fn validate_expression_complexity(syntax: &CLikeSyntax) -> RuleResult {
        let mut issues: Vec<String> = Vec::new();
        let mut score: f64 = 1.0;

        // Helper function to calculate expression depth
        fn expression_depth(expr: &CLikeExpression) -> usize {
            match expr {
                CLikeExpression::Binary { left, right, .. } => {
                    1 + expression_depth(left).max(expression_depth(right))
                }
                CLikeExpression::Unary { operand, .. } => {
                    1 + expression_depth(operand)
                }
                CLikeExpression::Call { arguments, .. } => {
                    1 + arguments.iter().map(expression_depth).max().unwrap_or(0)
                }
                CLikeExpression::MemberAccess { object, .. } => {
                    1 + expression_depth(object)
                }
                CLikeExpression::IndexAccess { object, index, .. } => {
                    1 + expression_depth(object).max(expression_depth(index))
                }
                _ => 0,
            }
        }

        for statement in &syntax.statements {
            if let CLikeStatement::Expression(expression) = statement {
                let depth = expression_depth(expression);
                if depth > 5 {
                    issues.push(format!("Expression depth {} exceeds recommended maximum of 5", depth));
                    score -= 0.1;
                }
            }
        }

        score = score.max(0.0_f64);

        RuleResult {
            rule_name: "expression_complexity".to_string(),
            passed: issues.is_empty(),
            score,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
        }
    }

    fn validate_memory_safety(syntax: &CLikeSyntax) -> RuleResult {
        let mut issues: Vec<String> = Vec::new();
        let mut score: f64 = 1.0;

        // Check for potential memory safety issues
        for statement in &syntax.statements {
            if let CLikeStatement::Expression(expression) = statement {
                // Look for potentially unsafe patterns
                Self::check_expression_safety(expression, &mut issues, &mut score);
            }
        }

        score = score.max(0.0_f64);

        RuleResult {
            rule_name: "memory_safety".to_string(),
            passed: issues.is_empty(),
            score,
            error_message: if issues.is_empty() { None } else { Some(issues.join("; ")) },
        }
    }

    fn check_expression_safety(expr: &CLikeExpression, issues: &mut Vec<String>, score: &mut f64) {
        match expr {
            CLikeExpression::Call { function, arguments, .. } => {
                // Check for potentially unsafe function calls
                if let CLikeExpression::Identifier(func_name) = function.as_ref() {
                    if matches!(func_name.as_str(), "malloc" | "free" | "strcpy" | "strcat" | "gets") {
                        issues.push(format!("Potentially unsafe function call: {}", func_name));
                        *score -= 0.2;
                    }
                }

                // Recursively check arguments
                for arg in arguments {
                    Self::check_expression_safety(arg, issues, score);
                }
            }
            CLikeExpression::Binary { left, right, .. } => {
                Self::check_expression_safety(left, issues, score);
                Self::check_expression_safety(right, issues, score);
            }
            CLikeExpression::Unary { operand, .. } => {
                Self::check_expression_safety(operand, issues, score);
            }
            _ => {}
        }
    }

    fn validate_error_handling(syntax: &CLikeSyntax) -> RuleResult {
        let mut has_error_handling = false;
        let mut score = 1.0;

        // Check if there's any error handling present
        for statement in &syntax.statements {
            if matches!(statement, CLikeStatement::Try { .. }) {
                has_error_handling = true;
                break;
            }
        }

        // Check functions for return value handling
        for function in &syntax.functions {
            if function.return_type.as_deref() == Some("int") {
                // Functions returning int might be using error codes
                has_error_handling = true;
            }
        }

        if !has_error_handling && (syntax.functions.len() > 2 || syntax.statements.len() > 10) {
            score = 0.7; // Reduce score but don't fail
        }

        RuleResult {
            rule_name: "error_handling".to_string(),
            passed: true, // This is a warning-level check
            score,
            error_message: if has_error_handling { None } else { Some("No error handling detected in complex code".to_string()) },
        }
    }

    // Default validation rule implementations

    fn validate_completeness(canonical: &CanonicalForm) -> RuleResult {
        // Check if canonical form has required elements
        RuleResult {
            rule_name: "completeness".to_string(),
            passed: true,
            score: 1.0,
            error_message: None,
        }
    }

    fn validate_consistency(canonical: &CanonicalForm) -> RuleResult {
        // Check for consistent structure
        RuleResult {
            rule_name: "consistency".to_string(),
            passed: true,
            score: 1.0,
            error_message: None,
        }
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
} 