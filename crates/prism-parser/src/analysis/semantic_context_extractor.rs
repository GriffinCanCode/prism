//! Semantic Context Extraction
//!
//! This module embodies the single concept of "Semantic Context Extraction".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: extracting AI-comprehensible semantic context and metadata.
//!
//! **Conceptual Responsibility**: Extract semantic meaning and AI context
//! **What it does**: business context, AI hints, semantic relationships, conceptual metadata
//! **What it doesn't do**: parsing logic, token navigation, AST construction

use crate::core::{
    error::{ParseError, ParseResult},
};
use prism_ast::{AstNode, Item, Stmt, Expr, ExpressionStmt};
use prism_common::NodeId;
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use std::collections::HashMap;

/// Semantic constraint types for validation
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticConstraint {
    /// Minimum value constraint
    MinValue(f64),
    /// Maximum value constraint
    MaxValue(f64),
    /// Minimum length constraint
    MinLength(usize),
    /// Maximum length constraint
    MaxLength(usize),
    /// Pattern constraint
    Pattern(String),
    /// Format constraint
    Format(String),
    /// Currency constraint
    Currency(String),
    /// Non-negative constraint
    NonNegative,
    /// Immutable constraint
    Immutable,
    /// Validation required constraint
    Validated,
}

/// Semantic context extractor - extracts AI-comprehensible metadata
/// 
/// This struct embodies the single concept of extracting semantic meaning.
/// It analyzes parsed code to generate context that helps AI systems
/// understand business intent and conceptual relationships.
pub struct SemanticContextExtractor {
    /// Extracted semantic contexts by node ID
    contexts: HashMap<NodeId, SemanticContext>,
    /// Business domain mapping
    domain_classifier: DomainClassifier,
}

impl SemanticContextExtractor {
    /// Create a new semantic context extractor
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            domain_classifier: DomainClassifier::new(),
        }
    }

    /// Extract semantic context for a node (simplified version without coordinator)
    pub fn extract_context(&mut self, node_id: NodeId) -> ParseResult<SemanticContext> {
        if let Some(context) = self.contexts.get(&node_id) {
            return Ok(context.clone());
        }

        // For now, create a basic semantic context
        let mut context = SemanticContext::new();
        context.purpose = Some("Analyzed code construct".to_string());
        context.conceptual_role = Some("Language Element".to_string());
        context.add_ai_hint("Part of conceptually cohesive code organization");
        
        self.contexts.insert(node_id, context.clone());
        Ok(context)
    }

    /// Infer function purpose from name
    fn infer_function_purpose(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        
        if lower_name.starts_with("get_") || lower_name.starts_with("find_") || lower_name.starts_with("query_") {
            Some("Retrieve data".to_string())
        } else if lower_name.starts_with("set_") || lower_name.starts_with("update_") || lower_name.starts_with("modify_") {
            Some("Update data".to_string())
        } else if lower_name.starts_with("create_") || lower_name.starts_with("add_") || lower_name.starts_with("insert_") {
            Some("Create new data".to_string())
        } else if lower_name.starts_with("delete_") || lower_name.starts_with("remove_") {
            Some("Remove data".to_string())
        } else if lower_name.starts_with("validate_") || lower_name.starts_with("check_") {
            Some("Validate data or conditions".to_string())
        } else if lower_name.starts_with("calculate_") || lower_name.starts_with("compute_") {
            Some("Perform calculations".to_string())
        } else if lower_name.starts_with("send_") || lower_name.starts_with("notify_") {
            Some("Send notifications or messages".to_string())
        } else {
            Some(format!("Perform {} operation", name))
        }
    }

    /// Infer type purpose from name
    fn infer_type_purpose(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        
        if lower_name.ends_with("id") || lower_name.ends_with("identifier") {
            Some("Unique identifier".to_string())
        } else if lower_name.ends_with("config") || lower_name.ends_with("configuration") {
            Some("Configuration data".to_string())
        } else if lower_name.ends_with("request") || lower_name.ends_with("command") {
            Some("Request or command data".to_string())
        } else if lower_name.ends_with("response") || lower_name.ends_with("result") {
            Some("Response or result data".to_string())
        } else if lower_name.ends_with("event") {
            Some("Event data".to_string())
        } else if lower_name.ends_with("service") {
            Some("Service interface".to_string())
        } else if lower_name.ends_with("repository") || lower_name.ends_with("store") {
            Some("Data access interface".to_string())
        } else {
            Some(format!("Data structure for {}", name))
        }
    }

    /// Infer variable purpose from name
    fn infer_variable_purpose(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        
        if lower_name.starts_with("temp_") || lower_name.starts_with("tmp_") {
            Some("Temporary storage".to_string())
        } else if lower_name.starts_with("config_") {
            Some("Configuration value".to_string())
        } else if lower_name.starts_with("state_") {
            Some("State tracking".to_string())
        } else if lower_name.ends_with("_count") || lower_name.ends_with("_size") {
            Some("Quantity or measurement".to_string())
        } else if lower_name.ends_with("_flag") || lower_name.ends_with("_enabled") {
            Some("Boolean flag or switch".to_string())
        } else {
            Some(format!("Store {} data", name))
        }
    }

    /// Check if function is a query (side-effect free)
    fn is_query_function(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.starts_with("get_") || lower.starts_with("find_") || 
        lower.starts_with("query_") || lower.starts_with("search_") ||
        lower.starts_with("is_") || lower.starts_with("has_") ||
        lower.starts_with("can_") || lower.starts_with("should_")
    }

    /// Check if function is a command (may have side effects)
    fn is_command_function(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.starts_with("set_") || lower.starts_with("update_") ||
        lower.starts_with("create_") || lower.starts_with("delete_") ||
        lower.starts_with("send_") || lower.starts_with("process_") ||
        lower.starts_with("execute_") || lower.starts_with("run_")
    }

    /// Check if type is a value object
    fn is_value_object(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.ends_with("id") || lower.ends_with("value") ||
        lower.ends_with("amount") || lower.ends_with("price") ||
        lower.ends_with("email") || lower.ends_with("address")
    }

    /// Check if type is an entity
    fn is_entity(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower == "user" || lower == "account" || lower == "order" ||
        lower == "product" || lower == "customer" || lower == "invoice"
    }

    /// Check if type is a service
    fn is_service(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.ends_with("service") || lower.ends_with("manager") ||
        lower.ends_with("handler") || lower.ends_with("processor")
    }

    /// Check if variable is configuration
    fn is_configuration(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.starts_with("config_") || lower.ends_with("_config") ||
        lower.contains("setting") || lower.contains("option")
    }

    /// Check if variable is state
    fn is_state(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.starts_with("state_") || lower.ends_with("_state") ||
        lower.contains("current") || lower.contains("active")
    }

    /// Check if variable is temporary
    fn is_temporary(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.starts_with("temp_") || lower.starts_with("tmp_") ||
        lower.starts_with("_") || lower.len() <= 2
    }

    /// Check if function call is validation
    fn is_validation_call(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.starts_with("validate_") || lower.starts_with("verify_") ||
        lower.starts_with("check_") || lower.contains("valid")
    }

    /// Check if binary operation is business calculation
    fn is_business_calculation(&self, op: &prism_ast::BinaryExpr) -> bool {
        matches!(op.operator, prism_ast::BinaryOperator::Add | prism_ast::BinaryOperator::Subtract | 
                prism_ast::BinaryOperator::Multiply | prism_ast::BinaryOperator::Divide | prism_ast::BinaryOperator::Modulo)
    }

    /// Describe a semantic constraint
    fn describe_constraint(&self, constraint: &SemanticConstraint) -> String {
        match constraint {
            SemanticConstraint::MinValue(v) => format!("Minimum value: {}", v),
            SemanticConstraint::MaxValue(v) => format!("Maximum value: {}", v),
            SemanticConstraint::MinLength(l) => format!("Minimum length: {}", l),
            SemanticConstraint::MaxLength(l) => format!("Maximum length: {}", l),
            SemanticConstraint::Pattern(p) => format!("Pattern constraint: {}", p),
            SemanticConstraint::Format(f) => format!("Format constraint: {}", f),
            SemanticConstraint::Currency(c) => format!("Currency: {}", c),
            SemanticConstraint::NonNegative => "Must be non-negative".to_string(),
            SemanticConstraint::Immutable => "Immutable value".to_string(),
            SemanticConstraint::Validated => "Requires validation".to_string(),
        }
    }

    /// Get all extracted contexts
    pub fn get_all_contexts(&self) -> &HashMap<NodeId, SemanticContext> {
        &self.contexts
    }

    /// Generate AI comprehension summary
    pub fn generate_ai_summary(&self) -> AISummary {
        let mut summary = AISummary::new();
        
        for (node_id, context) in &self.contexts {
            if let Some(domain) = &context.domain {
                summary.add_domain(domain.clone());
            }
            
            if let Some(role) = &context.conceptual_role {
                summary.add_conceptual_role(role.clone());
            }
            
            for hint in &context.ai_hints {
                summary.add_insight(hint.clone());
            }
        }
        
        summary
    }
}

/// Domain classifier for business context
pub struct DomainClassifier {
    domain_patterns: HashMap<String, String>,
}

impl DomainClassifier {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Authentication & Security
        patterns.insert("auth".to_string(), "Authentication & Security".to_string());
        patterns.insert("login".to_string(), "Authentication & Security".to_string());
        patterns.insert("security".to_string(), "Authentication & Security".to_string());
        patterns.insert("permission".to_string(), "Authentication & Security".to_string());
        
        // User Management
        patterns.insert("user".to_string(), "User Management".to_string());
        patterns.insert("profile".to_string(), "User Management".to_string());
        patterns.insert("account".to_string(), "User Management".to_string());
        
        // Data Management
        patterns.insert("data".to_string(), "Data Management".to_string());
        patterns.insert("database".to_string(), "Data Management".to_string());
        patterns.insert("storage".to_string(), "Data Management".to_string());
        patterns.insert("repository".to_string(), "Data Management".to_string());
        
        // Business Logic
        patterns.insert("business".to_string(), "Business Logic".to_string());
        patterns.insert("rule".to_string(), "Business Logic".to_string());
        patterns.insert("process".to_string(), "Business Logic".to_string());
        patterns.insert("workflow".to_string(), "Business Logic".to_string());
        
        // Communication
        patterns.insert("message".to_string(), "Communication".to_string());
        patterns.insert("notification".to_string(), "Communication".to_string());
        patterns.insert("email".to_string(), "Communication".to_string());
        patterns.insert("api".to_string(), "Communication".to_string());
        
        Self {
            domain_patterns: patterns,
        }
    }

    pub fn classify_module(&self, name: &str) -> Option<String> {
        self.classify_by_patterns(name)
    }

    pub fn classify_function(&self, name: &str) -> Option<String> {
        self.classify_by_patterns(name)
    }

    pub fn classify_type(&self, name: &str) -> Option<String> {
        self.classify_by_patterns(name)
    }

    fn classify_by_patterns(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        
        for (pattern, domain) in &self.domain_patterns {
            if lower_name.contains(pattern) {
                return Some(domain.clone());
            }
        }
        
        None
    }
}

/// AI comprehension summary
#[derive(Debug, Clone)]
pub struct AISummary {
    pub domains: Vec<String>,
    pub conceptual_roles: Vec<String>,
    pub insights: Vec<String>,
}

impl AISummary {
    pub fn new() -> Self {
        Self {
            domains: Vec::new(),
            conceptual_roles: Vec::new(),
            insights: Vec::new(),
        }
    }

    pub fn add_domain(&mut self, domain: String) {
        if !self.domains.contains(&domain) {
            self.domains.push(domain);
        }
    }

    pub fn add_conceptual_role(&mut self, role: String) {
        if !self.conceptual_roles.contains(&role) {
            self.conceptual_roles.push(role);
        }
    }

    pub fn add_insight(&mut self, insight: String) {
        if !self.insights.contains(&insight) {
            self.insights.push(insight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test utilities would be defined here or in a test module

    #[test]
    fn test_function_purpose_inference() {
        let extractor = SemanticContextExtractor::new();
        
        assert_eq!(
            extractor.infer_function_purpose("get_user"),
            Some("Retrieve data".to_string())
        );
        
        assert_eq!(
            extractor.infer_function_purpose("validate_email"),
            Some("Validate data or conditions".to_string())
        );
        
        assert_eq!(
            extractor.infer_function_purpose("calculate_total"),
            Some("Perform calculations".to_string())
        );
    }

    #[test]
    fn test_domain_classification() {
        let classifier = DomainClassifier::new();
        
        assert_eq!(
            classifier.classify_module("UserAuth"),
            Some("Authentication & Security".to_string())
        );
        
        assert_eq!(
            classifier.classify_function("send_notification"),
            Some("Communication".to_string())
        );
        
        assert_eq!(
            classifier.classify_type("UserRepository"),
            Some("Data Management".to_string())
        );
    }

    #[test]
    fn test_pattern_recognition() {
        let extractor = SemanticContextExtractor::new();
        
        assert!(extractor.is_query_function("get_user"));
        assert!(extractor.is_command_function("update_profile"));
        assert!(extractor.is_value_object("UserId"));
        assert!(extractor.is_entity("User"));
        assert!(extractor.is_service("UserService"));
    }
} 