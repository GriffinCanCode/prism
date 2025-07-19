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
    token_stream_manager::TokenStreamManager,
    parsing_coordinator::ParsingCoordinator,
};
use prism_ast::{AstNode, NodeId, SemanticContext, AIHint, BusinessContext};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use std::collections::HashMap;

/// Semantic context extractor - extracts AI-comprehensible metadata
/// 
/// This struct embodies the single concept of extracting semantic meaning.
/// It analyzes parsed code to generate context that helps AI systems
/// understand business intent and conceptual relationships.
pub struct SemanticContextExtractor<'a> {
    /// Reference to the token stream manager (no ownership)
    token_stream: &'a TokenStreamManager,
    /// Reference to coordinator for accessing parsed nodes
    coordinator: &'a ParsingCoordinator,
    /// Extracted semantic contexts by node ID
    contexts: HashMap<NodeId, SemanticContext>,
    /// Business domain mapping
    domain_classifier: DomainClassifier,
}

impl<'a> SemanticContextExtractor<'a> {
    /// Create a new semantic context extractor
    pub fn new(
        token_stream: &'a TokenStreamManager,
        coordinator: &'a ParsingCoordinator,
    ) -> Self {
        Self {
            token_stream,
            coordinator,
            contexts: HashMap::new(),
            domain_classifier: DomainClassifier::new(),
        }
    }

    /// Extract semantic context for a node
    pub fn extract_context(&mut self, node_id: NodeId) -> ParseResult<SemanticContext> {
        if let Some(context) = self.contexts.get(&node_id) {
            return Ok(context.clone());
        }

        let context = self.analyze_node_semantics(node_id)?;
        self.contexts.insert(node_id, context.clone());
        Ok(context)
    }

    /// Analyze semantic meaning of a node
    fn analyze_node_semantics(&self, node_id: NodeId) -> ParseResult<SemanticContext> {
        // Get node information from coordinator
        let node_info = self.coordinator.get_node_info(node_id)?;
        
        let mut context = SemanticContext::new();
        
        match node_info.kind {
            // Module-level semantics
            NodeKind::Module(ref module) => {
                context.purpose = Some(format!("Define {} business capability", module.name));
                context.domain = self.domain_classifier.classify_module(&module.name);
                context.conceptual_role = Some("Module Boundary".to_string());
                
                context.add_ai_hint("Modules represent single business capabilities");
                context.add_ai_hint("Each module should have conceptual cohesion");
                
                if let Some(domain) = &context.domain {
                    context.add_business_context(BusinessContext::Domain(domain.clone()));
                }
            }
            
            // Function-level semantics
            NodeKind::Function(ref func) => {
                context.purpose = self.infer_function_purpose(&func.name);
                context.domain = self.domain_classifier.classify_function(&func.name);
                context.conceptual_role = Some("Operation".to_string());
                
                // Analyze function name patterns
                if self.is_query_function(&func.name) {
                    context.add_ai_hint("Query function - should be side-effect free");
                    context.add_business_context(BusinessContext::OperationType("Query".to_string()));
                } else if self.is_command_function(&func.name) {
                    context.add_ai_hint("Command function - may have side effects");
                    context.add_business_context(BusinessContext::OperationType("Command".to_string()));
                }
                
                // Analyze parameters for semantic meaning
                if let Some(params) = &func.parameters {
                    for param in params {
                        if let Some(param_context) = self.analyze_parameter_semantics(param) {
                            context.add_ai_hint(&format!("Parameter {} has semantic meaning: {}", 
                                param.name, param_context));
                        }
                    }
                }
            }
            
            // Type-level semantics
            NodeKind::Type(ref type_def) => {
                context.purpose = self.infer_type_purpose(&type_def.name);
                context.domain = self.domain_classifier.classify_type(&type_def.name);
                context.conceptual_role = Some("Data Model".to_string());
                
                // Analyze type name patterns
                if self.is_value_object(&type_def.name) {
                    context.add_ai_hint("Value object - immutable with equality semantics");
                    context.add_business_context(BusinessContext::DataPattern("Value Object".to_string()));
                } else if self.is_entity(&type_def.name) {
                    context.add_ai_hint("Entity - has identity and lifecycle");
                    context.add_business_context(BusinessContext::DataPattern("Entity".to_string()));
                } else if self.is_service(&type_def.name) {
                    context.add_ai_hint("Service - encapsulates business operations");
                    context.add_business_context(BusinessContext::DataPattern("Service".to_string()));
                }
                
                // Analyze constraints for business rules
                if let Some(constraints) = &type_def.constraints {
                    for constraint in constraints {
                        context.add_business_context(
                            BusinessContext::Constraint(self.describe_constraint(constraint))
                        );
                    }
                }
            }
            
            // Variable/binding semantics
            NodeKind::Variable(ref var) => {
                context.purpose = self.infer_variable_purpose(&var.name);
                context.conceptual_role = Some("Data Binding".to_string());
                
                // Analyze naming patterns
                if self.is_configuration(&var.name) {
                    context.add_ai_hint("Configuration value - affects system behavior");
                } else if self.is_state(&var.name) {
                    context.add_ai_hint("State variable - represents system state");
                } else if self.is_temporary(&var.name) {
                    context.add_ai_hint("Temporary value - used for computation");
                }
            }
            
            // Expression semantics
            NodeKind::Expression(ref expr) => {
                context.purpose = Some("Compute value or perform operation".to_string());
                context.conceptual_role = Some("Computation".to_string());
                
                // Analyze expression patterns
                match expr.kind {
                    ExprKind::Call(ref call) => {
                        context.add_ai_hint(&format!("Function call to {}", call.function_name));
                        if self.is_validation_call(&call.function_name) {
                            context.add_business_context(BusinessContext::OperationType("Validation".to_string()));
                        }
                    }
                    ExprKind::BinaryOp(ref op) => {
                        context.add_ai_hint(&format!("Binary operation: {}", op.operator));
                        if self.is_business_calculation(op) {
                            context.add_business_context(BusinessContext::OperationType("Calculation".to_string()));
                        }
                    }
                    _ => {}
                }
            }
            
            _ => {
                // Default semantic context
                context.purpose = Some("Code construct".to_string());
                context.conceptual_role = Some("Language Element".to_string());
            }
        }
        
        // Add span-based context
        context.source_location = Some(node_info.span);
        
        // Add conceptual cohesion metadata
        context.add_ai_hint("Part of conceptually cohesive code organization");
        
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
    fn is_business_calculation(&self, op: &BinaryOp) -> bool {
        matches!(op.operator.as_str(), "+" | "-" | "*" | "/" | "%")
    }

    /// Analyze parameter semantics
    fn analyze_parameter_semantics(&self, param: &Parameter) -> Option<String> {
        if let Some(type_name) = &param.type_annotation {
            if self.is_value_object(type_name) {
                Some("Value object parameter".to_string())
            } else if self.is_entity(type_name) {
                Some("Entity parameter".to_string())
            } else {
                None
            }
        } else {
            None
        }
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
            _ => "Custom constraint".to_string(),
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
        let extractor = create_test_extractor();
        
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
        let extractor = create_test_extractor();
        
        assert!(extractor.is_query_function("get_user"));
        assert!(extractor.is_command_function("update_profile"));
        assert!(extractor.is_value_object("UserId"));
        assert!(extractor.is_entity("User"));
        assert!(extractor.is_service("UserService"));
    }
} 