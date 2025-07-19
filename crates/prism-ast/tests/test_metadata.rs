//! Comprehensive tests for the metadata module

use prism_common::span::Position;
use prism_ast::{metadata::*, node::*};
use std::collections::HashMap;

#[test]
fn test_node_metadata_creation() {
    let metadata = NodeMetadata::default();
    
    assert!(metadata.ai_context.is_none());
    assert!(metadata.semantic_annotations.is_empty());
    assert!(metadata.business_rules.is_empty());
    assert!(metadata.performance_characteristics.is_empty());
    assert!(metadata.security_implications.is_empty());
    assert!(metadata.compliance_requirements.is_empty());
    assert!(!metadata.is_ai_generated);
    assert_eq!(metadata.semantic_importance, 0.0);
    assert!(!metadata.security_sensitive);
    assert!(metadata.documentation.is_none());
    assert!(metadata.examples.is_empty());
    assert!(metadata.common_mistakes.is_empty());
    assert!(metadata.related_concepts.is_empty());
    assert!(metadata.architectural_patterns.is_empty());
}

#[test]
fn test_ai_context_creation() {
    let context = AiContext::new();
    
    assert!(context.purpose.is_none());
    assert!(context.domain.is_none());
    assert!(context.description.is_none());
    assert!(context.capabilities.is_empty());
    assert!(context.side_effects.is_empty());
    assert!(context.preconditions.is_empty());
    assert!(context.postconditions.is_empty());
    assert!(context.invariants.is_empty());
    assert!(context.testing_recommendations.is_empty());
    assert!(context.refactoring_suggestions.is_empty());
}

#[test]
fn test_ai_context_builder_pattern() {
    let context = AiContext::new()
        .with_purpose("User authentication")
        .with_domain("Security")
        .with_description("Validates user credentials and establishes session")
        .with_capability("authenticate_user")
        .with_capability("create_session")
        .with_side_effect("Updates session store")
        .with_side_effect("Logs authentication attempt")
        .with_precondition("User credentials must be provided")
        .with_postcondition("Session is created on success")
        .with_invariant("Session state remains consistent")
        .with_testing_recommendation("Test with valid and invalid credentials")
        .with_refactoring_suggestion("Consider extracting session management");
    
    assert_eq!(context.purpose, Some("User authentication".to_string()));
    assert_eq!(context.domain, Some("Security".to_string()));
    assert_eq!(context.description, Some("Validates user credentials and establishes session".to_string()));
    assert_eq!(context.capabilities.len(), 2);
    assert_eq!(context.side_effects.len(), 2);
    assert_eq!(context.preconditions.len(), 1);
    assert_eq!(context.postconditions.len(), 1);
    assert_eq!(context.invariants.len(), 1);
    assert_eq!(context.testing_recommendations.len(), 1);
    assert_eq!(context.refactoring_suggestions.len(), 1);
}

#[test]
fn test_data_flow_info() {
    let mut data_flow = DataFlowInfo::default();
    data_flow.sources.push("user_input".to_string());
    data_flow.sources.push("database".to_string());
    data_flow.sinks.push("session_store".to_string());
    data_flow.transformations.push("hash_password".to_string());
    data_flow.validations.push("check_password_strength".to_string());
    data_flow.sensitivity_level = SensitivityLevel::Confidential;
    data_flow.encrypted = true;
    data_flow.retention_requirements = Some("30 days".to_string());
    
    assert_eq!(data_flow.sources.len(), 2);
    assert_eq!(data_flow.sinks.len(), 1);
    assert_eq!(data_flow.transformations.len(), 1);
    assert_eq!(data_flow.validations.len(), 1);
    assert_eq!(data_flow.sensitivity_level, SensitivityLevel::Confidential);
    assert!(data_flow.encrypted);
    assert_eq!(data_flow.retention_requirements, Some("30 days".to_string()));
}

#[test]
fn test_control_flow_info() {
    let mut control_flow = ControlFlowInfo::default();
    control_flow.can_branch = true;
    control_flow.can_loop = false;
    control_flow.can_throw = true;
    control_flow.can_return_early = true;
    control_flow.is_deterministic = false;
    control_flow.execution_dependencies.push("user_auth".to_string());
    
    assert!(control_flow.can_branch);
    assert!(!control_flow.can_loop);
    assert!(control_flow.can_throw);
    assert!(control_flow.can_return_early);
    assert!(!control_flow.is_deterministic);
    assert_eq!(control_flow.execution_dependencies.len(), 1);
}

#[test]
fn test_resource_usage() {
    let mut resource_usage = ResourceUsage::default();
    
    // Memory usage
    resource_usage.memory_usage.estimated_allocation = Some(1024);
    resource_usage.memory_usage.bounded = true;
    resource_usage.memory_usage.allocation_pattern = AllocationPattern::Heap;
    resource_usage.memory_usage.shared = false;
    
    // CPU usage
    resource_usage.cpu_usage.complexity = ComplexityClass::Linear;
    resource_usage.cpu_usage.bounded = true;
    resource_usage.cpu_usage.intensive = false;
    resource_usage.cpu_usage.parallelizable = true;
    
    // Network usage
    resource_usage.network_usage.makes_network_calls = true;
    resource_usage.network_usage.estimated_requests = Some(2);
    resource_usage.network_usage.protocols.push("HTTPS".to_string());
    resource_usage.network_usage.bounded = true;
    
    // Database usage
    resource_usage.database_usage.queries_database = true;
    resource_usage.database_usage.modifies_database = true;
    resource_usage.database_usage.tables_accessed.push("users".to_string());
    resource_usage.database_usage.uses_transactions = true;
    
    assert_eq!(resource_usage.memory_usage.estimated_allocation, Some(1024));
    assert_eq!(resource_usage.memory_usage.allocation_pattern, AllocationPattern::Heap);
    assert_eq!(resource_usage.cpu_usage.complexity, ComplexityClass::Linear);
    assert!(resource_usage.cpu_usage.parallelizable);
    assert!(resource_usage.network_usage.makes_network_calls);
    assert_eq!(resource_usage.network_usage.protocols.len(), 1);
    assert!(resource_usage.database_usage.queries_database);
    assert!(resource_usage.database_usage.uses_transactions);
}

#[test]
fn test_error_handling_info() {
    let mut error_handling = ErrorHandlingInfo::default();
    error_handling.error_types.push("AuthenticationError".to_string());
    error_handling.error_types.push("ValidationError".to_string());
    error_handling.recovery_strategies.push("Retry with backoff".to_string());
    error_handling.recoverable = true;
    error_handling.propagation_behavior = ErrorPropagation::Transform;
    
    assert_eq!(error_handling.error_types.len(), 2);
    assert_eq!(error_handling.recovery_strategies.len(), 1);
    assert!(error_handling.recoverable);
    assert_eq!(error_handling.propagation_behavior, ErrorPropagation::Transform);
}

#[test]
fn test_sensitivity_level_display() {
    assert_eq!(SensitivityLevel::Public.to_string(), "Public");
    assert_eq!(SensitivityLevel::Internal.to_string(), "Internal");
    assert_eq!(SensitivityLevel::Confidential.to_string(), "Confidential");
    assert_eq!(SensitivityLevel::Restricted.to_string(), "Restricted");
    assert_eq!(SensitivityLevel::TopSecret.to_string(), "Top Secret");
}

#[test]
fn test_sensitivity_level_default() {
    let default_level = SensitivityLevel::default();
    assert_eq!(default_level, SensitivityLevel::Public);
}

#[test]
fn test_allocation_pattern_default() {
    let default_pattern = AllocationPattern::default();
    assert_eq!(default_pattern, AllocationPattern::None);
}

#[test]
fn test_error_propagation_default() {
    let default_propagation = ErrorPropagation::default();
    assert_eq!(default_propagation, ErrorPropagation::Propagate);
}

#[test]
fn test_ai_context_security_sensitive_detection() {
    // Context with confidential data should be security sensitive
    let mut context = AiContext::new();
    context.data_flow.sensitivity_level = SensitivityLevel::Confidential;
    
    assert!(context.is_security_sensitive());
    
    // Context with security capability should be security sensitive
    let context_with_security = AiContext::new()
        .with_capability("security:authenticate");
    
    assert!(context_with_security.is_security_sensitive());
    
    // Context with security side effect should be security sensitive
    let context_with_security_effect = AiContext::new()
        .with_side_effect("security:log_access");
    
    assert!(context_with_security_effect.is_security_sensitive());
    
    // Public context without security aspects should not be security sensitive
    let public_context = AiContext::new()
        .with_purpose("Display user name");
    
    assert!(!public_context.is_security_sensitive());
}

#[test]
fn test_ai_context_performance_critical_detection() {
    // Context with intensive CPU usage should be performance critical
    let mut context = AiContext::new();
    context.resource_usage.cpu_usage.intensive = true;
    
    assert!(context.is_performance_critical());
    
    // Context with large memory allocation should be performance critical
    let mut context_with_memory = AiContext::new();
    context_with_memory.resource_usage.memory_usage.estimated_allocation = Some(10 * 1024 * 1024); // 10MB
    
    assert!(context_with_memory.is_performance_critical());
    
    // Context with network calls should be performance critical
    let mut context_with_network = AiContext::new();
    context_with_network.resource_usage.network_usage.makes_network_calls = true;
    
    assert!(context_with_network.is_performance_critical());
    
    // Simple context should not be performance critical
    let simple_context = AiContext::new()
        .with_purpose("Return constant value");
    
    assert!(!simple_context.is_performance_critical());
}

#[test]
fn test_ai_context_tags() {
    let context = AiContext::new()
        .with_domain("Authentication")
        .with_capability("security:validate");
    
    let tags = context.ai_tags();
    
    assert!(tags.contains(&"domain:Authentication".to_string()));
    assert!(tags.contains(&"security-sensitive".to_string()));
}

#[test]
fn test_ai_context_with_complex_scenario() {
    let context = AiContext::new()
        .with_purpose("Process user payment")
        .with_domain("Finance")
        .with_description("Handles credit card payment processing with fraud detection")
        .with_capability("payment:charge_card")
        .with_capability("fraud:detect_suspicious_activity")
        .with_side_effect("Updates payment records")
        .with_side_effect("Sends notification email")
        .with_precondition("Valid credit card information")
        .with_precondition("Sufficient account balance")
        .with_postcondition("Payment is processed or declined")
        .with_postcondition("Audit log is updated")
        .with_invariant("Account balance accuracy")
        .with_testing_recommendation("Test with various card types")
        .with_testing_recommendation("Test fraud detection scenarios")
        .with_refactoring_suggestion("Extract fraud detection to separate service");
    
    // Set up complex data flow
    let mut data_flow = DataFlowInfo::default();
    data_flow.sources.push("credit_card_form".to_string());
    data_flow.sources.push("user_account".to_string());
    data_flow.sinks.push("payment_gateway".to_string());
    data_flow.sinks.push("audit_log".to_string());
    data_flow.transformations.push("encrypt_card_data".to_string());
    data_flow.validations.push("validate_card_number".to_string());
    data_flow.sensitivity_level = SensitivityLevel::Restricted;
    data_flow.encrypted = true;
    
    // Set up control flow
    let mut control_flow = ControlFlowInfo::default();
    control_flow.can_branch = true;
    control_flow.can_throw = true;
    control_flow.can_return_early = true;
    control_flow.is_deterministic = false;
    
    // Set up resource usage
    let mut resource_usage = ResourceUsage::default();
    resource_usage.network_usage.makes_network_calls = true;
    resource_usage.network_usage.protocols.push("HTTPS".to_string());
    resource_usage.database_usage.queries_database = true;
    resource_usage.database_usage.modifies_database = true;
    resource_usage.database_usage.uses_transactions = true;
    
    // Set up error handling
    let mut error_handling = ErrorHandlingInfo::default();
    error_handling.error_types.push("PaymentDeclinedError".to_string());
    error_handling.error_types.push("FraudDetectedError".to_string());
    error_handling.error_types.push("NetworkError".to_string());
    error_handling.recovery_strategies.push("Retry payment".to_string());
    error_handling.recovery_strategies.push("Use backup gateway".to_string());
    error_handling.recoverable = true;
    error_handling.propagation_behavior = ErrorPropagation::Transform;
    
    let context = context
        .with_data_flow(data_flow)
        .with_control_flow(control_flow)
        .with_resource_usage(resource_usage)
        .with_error_handling(error_handling);
    
    // Verify all aspects
    assert_eq!(context.purpose, Some("Process user payment".to_string()));
    assert_eq!(context.domain, Some("Finance".to_string()));
    assert_eq!(context.capabilities.len(), 2);
    assert_eq!(context.side_effects.len(), 2);
    assert_eq!(context.preconditions.len(), 2);
    assert_eq!(context.postconditions.len(), 2);
    assert_eq!(context.invariants.len(), 1);
    assert_eq!(context.testing_recommendations.len(), 2);
    assert_eq!(context.refactoring_suggestions.len(), 1);
    
    assert!(context.is_security_sensitive());
    assert!(context.is_performance_critical());
    
    let tags = context.ai_tags();
    assert!(tags.contains(&"domain:Finance".to_string()));
    assert!(tags.contains(&"security-sensitive".to_string()));
    assert!(tags.contains(&"performance-critical".to_string()));
    assert!(tags.contains(&"control-flow:branching".to_string()));
    assert!(tags.contains(&"database:read".to_string()));
    assert!(tags.contains(&"database:write".to_string()));
}

#[test]
fn test_filesystem_usage() {
    let mut fs_usage = FilesystemUsage::default();
    fs_usage.reads_files = true;
    fs_usage.writes_files = true;
    fs_usage.paths_accessed.push("/tmp/cache".to_string());
    fs_usage.paths_accessed.push("/var/log/app.log".to_string());
    fs_usage.permissions_required.push("read".to_string());
    fs_usage.permissions_required.push("write".to_string());
    
    assert!(fs_usage.reads_files);
    assert!(fs_usage.writes_files);
    assert_eq!(fs_usage.paths_accessed.len(), 2);
    assert_eq!(fs_usage.permissions_required.len(), 2);
}

#[test]
fn test_memory_usage_patterns() {
    let mut memory_usage = MemoryUsage::default();
    memory_usage.estimated_allocation = Some(4096);
    memory_usage.bounded = true;
    memory_usage.allocation_pattern = AllocationPattern::Arena;
    memory_usage.shared = true;
    
    assert_eq!(memory_usage.estimated_allocation, Some(4096));
    assert!(memory_usage.bounded);
    assert_eq!(memory_usage.allocation_pattern, AllocationPattern::Arena);
    assert!(memory_usage.shared);
}

#[test]
fn test_cpu_usage_characteristics() {
    let cpu_usage = CpuUsage {
        complexity: ComplexityClass::Quadratic,
        bounded: false,
        intensive: true,
        parallelizable: false,
    };
    
    assert_eq!(cpu_usage.complexity, ComplexityClass::Quadratic);
    assert!(!cpu_usage.bounded);
    assert!(cpu_usage.intensive);
    assert!(!cpu_usage.parallelizable);
}

#[test]
fn test_node_metadata_builder() {
    let mut metadata = NodeMetadata::default();
    metadata.ai_context = Some(AiContext::new().with_purpose("Test"));
    metadata.semantic_annotations.push("important".to_string());
    metadata.business_rules.push("must validate input".to_string());
    metadata.performance_characteristics.push("O(1) access".to_string());
    metadata.security_implications.push("handles sensitive data".to_string());
    metadata.compliance_requirements.push("GDPR compliant".to_string());
    metadata.is_ai_generated = true;
    metadata.semantic_importance = 0.95;
    metadata.security_sensitive = true;
    metadata.documentation = Some("Core authentication function".to_string());
    metadata.examples.push("authenticate(user, password)".to_string());
    metadata.common_mistakes.push("Not handling null passwords".to_string());
    metadata.related_concepts.push("authorization".to_string());
    metadata.architectural_patterns.push("Strategy Pattern".to_string());
    
    assert!(metadata.ai_context.is_some());
    assert_eq!(metadata.semantic_annotations.len(), 1);
    assert_eq!(metadata.business_rules.len(), 1);
    assert_eq!(metadata.performance_characteristics.len(), 1);
    assert_eq!(metadata.security_implications.len(), 1);
    assert_eq!(metadata.compliance_requirements.len(), 1);
    assert!(metadata.is_ai_generated);
    assert_eq!(metadata.semantic_importance, 0.95);
    assert!(metadata.security_sensitive);
    assert!(metadata.documentation.is_some());
    assert_eq!(metadata.examples.len(), 1);
    assert_eq!(metadata.common_mistakes.len(), 1);
    assert_eq!(metadata.related_concepts.len(), 1);
    assert_eq!(metadata.architectural_patterns.len(), 1);
} 