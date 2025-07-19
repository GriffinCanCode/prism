//! Comprehensive tests for the AST node module

use prism_common::span::Position;
use prism_ast::{node::*, expr::*, metadata::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

#[test]
fn test_ast_node_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    
    let node = AstNode::new(expr, span, node_id);
    
    assert_eq!(node.span, span);
    assert_eq!(node.id, node_id);
    assert!(!node.has_ai_context());
    assert!(!node.has_semantic_annotations());
    assert!(!node.has_business_rules());
    assert!(!node.is_ai_generated());
    assert_eq!(node.semantic_importance(), 0.0);
    assert!(!node.is_security_sensitive());
}

#[test]
fn test_ast_node_with_metadata() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let mut metadata = NodeMetadata::default();
    metadata.is_ai_generated = true;
    metadata.semantic_importance = 0.8;
    metadata.security_sensitive = true;
    metadata.semantic_annotations.push("test annotation".to_string());
    metadata.business_rules.push("test rule".to_string());
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    
    let node = AstNode::with_metadata(expr, span, node_id, metadata);
    
    assert!(node.is_ai_generated());
    assert_eq!(node.semantic_importance(), 0.8);
    assert!(node.is_security_sensitive());
    assert!(node.has_semantic_annotations());
    assert!(node.has_business_rules());
    assert_eq!(node.semantic_annotations().len(), 1);
    assert_eq!(node.business_rules().len(), 1);
}

#[test]
fn test_ast_node_with_ai_context() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let ai_context = AiContext {
        purpose: Some("Calculate user age".to_string()),
        domain: Some("user_management".to_string()),
        description: Some("Computes the age of a user based on birth date".to_string()),
        capabilities: vec!["date_calculation".to_string()],
        side_effects: vec!["none".to_string()],
        preconditions: vec!["birth_date must be valid".to_string()],
        postconditions: vec!["returns positive integer".to_string()],
        invariants: vec!["age >= 0".to_string()],
        data_flow: DataFlowInfo {
            sources: vec!["user.birth_date".to_string()],
            sinks: vec!["age_display".to_string()],
            transformations: vec!["date_subtraction".to_string()],
            validations: vec!["date_format_check".to_string()],
            sensitivity_level: SensitivityLevel::Internal,
            encrypted: false,
            retention_requirements: Some("7 years".to_string()),
        },
        control_flow: ControlFlowInfo {
            can_branch: false,
            can_loop: false,
            can_throw: true,
            can_return_early: false,
            is_deterministic: true,
            execution_dependencies: vec![],
        },
        resource_usage: ResourceUsage {
            memory_usage: MemoryUsage {
                estimated_allocation: Some(64),
                bounded: true,
                allocation_pattern: AllocationPattern::Stack,
                shared: false,
            },
            cpu_usage: CpuUsage {
                complexity: ComplexityClass::Linear,
                bounded: true,
                intensive: false,
                parallelizable: false,
            },
            network_usage: NetworkUsage {
                makes_network_calls: false,
                estimated_requests: None,
                protocols: vec![],
                bounded: true,
            },
            filesystem_usage: FilesystemUsage {
                reads_files: false,
                writes_files: false,
                paths_accessed: vec![],
                permissions_required: vec![],
            },
            database_usage: DatabaseUsage {
                queries_database: false,
                modifies_database: false,
                tables_accessed: vec![],
                uses_transactions: false,
            },
        },
        error_handling: ErrorHandlingInfo {
            error_types: vec!["InvalidDateError".to_string()],
            recovery_strategies: vec!["default_age".to_string()],
            recoverable: true,
            propagation_behavior: ErrorPropagation::Propagate,
        },
        testing_recommendations: vec!["Test with edge dates".to_string()],
        refactoring_suggestions: vec!["Consider caching".to_string()],
    };
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(25),
    });
    
    let node = AstNode::new(expr, span, node_id).with_ai_context(ai_context);
    
    assert!(node.has_ai_context());
    let context = node.ai_context().unwrap();
    assert_eq!(context.purpose.as_ref().unwrap(), "Calculate user age");
    assert_eq!(context.domain.as_ref().unwrap(), "user_management");
    assert_eq!(context.capabilities.len(), 1);
    assert_eq!(context.data_flow.sources.len(), 1);
    assert_eq!(context.control_flow.is_deterministic, true);
    assert!(context.error_handling.recoverable);
}

#[test]
fn test_ast_node_semantic_annotations() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Variable(VariableExpr {
        name: Symbol::intern("user_id"),
    });
    
    let node = AstNode::new(expr, span, node_id)
        .with_semantic_annotation("Primary key identifier".to_string())
        .with_semantic_annotation("Used for database queries".to_string());
    
    assert!(node.has_semantic_annotations());
    assert_eq!(node.semantic_annotations().len(), 2);
    assert_eq!(node.semantic_annotations()[0], "Primary key identifier");
    assert_eq!(node.semantic_annotations()[1], "Used for database queries");
}

#[test]
fn test_ast_node_business_rules() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Float(100.0),
    });
    
    let node = AstNode::new(expr, span, node_id)
        .with_business_rule("Maximum withdrawal amount is $1000".to_string())
        .with_business_rule("Must validate account balance".to_string());
    
    assert!(node.has_business_rules());
    assert_eq!(node.business_rules().len(), 2);
    assert_eq!(node.business_rules()[0], "Maximum withdrawal amount is $1000");
    assert_eq!(node.business_rules()[1], "Must validate account balance");
}

#[test]
fn test_ast_node_map() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let original_expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    
    let original_node = AstNode::new(original_expr, span, node_id)
        .with_semantic_annotation("original".to_string());
    
    let mapped_node = original_node.map(|expr| {
        match expr {
            Expr::Literal(lit) => Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(84), // Double the value
            }),
            other => other,
        }
    });
    
    // Verify the mapping preserved metadata
    assert_eq!(mapped_node.span, span);
    assert_eq!(mapped_node.id, node_id);
    assert!(mapped_node.has_semantic_annotations());
    assert_eq!(mapped_node.semantic_annotations()[0], "original");
    
    // Verify the content was transformed
    if let Expr::Literal(lit) = &mapped_node.kind {
        if let LiteralValue::Integer(n) = &lit.value {
            assert_eq!(*n, 84);
        }
    }
}

#[test]
fn test_ast_node_map_ref() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::String("test".to_string()),
    });
    
    let node = AstNode::new(expr, span, node_id);
    let ref_node = node.map_ref(|expr| {
        match expr {
            Expr::Literal(lit) => &lit.value,
            _ => panic!("Expected literal"),
        }
    });
    
    assert_eq!(ref_node.span, span);
    assert_eq!(ref_node.id, node_id);
    
    if let LiteralValue::String(s) = ref_node.kind {
        assert_eq!(s, "test");
    }
}

#[test]
fn test_ast_node_as_ref_and_as_mut() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Boolean(true),
    });
    
    let mut node = AstNode::new(expr, span, node_id);
    
    // Test as_ref
    let expr_ref = node.as_ref();
    match expr_ref {
        Expr::Literal(lit) => {
            match &lit.value {
                LiteralValue::Boolean(b) => assert!(*b),
                _ => panic!("Expected boolean"),
            }
        }
        _ => panic!("Expected literal"),
    }
    
    // Test as_mut
    let expr_mut = node.as_mut();
    match expr_mut {
        Expr::Literal(lit) => {
            lit.value = LiteralValue::Boolean(false);
        }
        _ => panic!("Expected literal"),
    }
    
    // Verify the mutation
    match &node.kind {
        Expr::Literal(lit) => {
            match &lit.value {
                LiteralValue::Boolean(b) => assert!(!*b),
                _ => panic!("Expected boolean"),
            }
        }
        _ => panic!("Expected literal"),
    }
}

#[test]
fn test_ast_node_clone() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::String("cloneable".to_string()),
    });
    
    let original = AstNode::new(expr, span, node_id)
        .with_semantic_annotation("original annotation".to_string());
    
    let cloned = original.clone();
    
    // Verify structure is identical
    assert_eq!(cloned.span, original.span);
    assert_eq!(cloned.id, original.id);
    assert_eq!(cloned.has_semantic_annotations(), original.has_semantic_annotations());
    assert_eq!(cloned.semantic_annotations(), original.semantic_annotations());
    
    // Verify content is identical
    match (&original.kind, &cloned.kind) {
        (Expr::Literal(orig_lit), Expr::Literal(cloned_lit)) => {
            match (&orig_lit.value, &cloned_lit.value) {
                (LiteralValue::String(orig), LiteralValue::String(cloned)) => {
                    assert_eq!(orig, cloned);
                }
                _ => panic!("Expected string literals"),
            }
        }
        _ => panic!("Expected literal expressions"),
    }
}

#[test]
fn test_complexity_class_display() {
    assert_eq!(format!("{}", ComplexityClass::Constant), "O(1)");
    assert_eq!(format!("{}", ComplexityClass::Logarithmic), "O(log n)");
    assert_eq!(format!("{}", ComplexityClass::Linear), "O(n)");
    assert_eq!(format!("{}", ComplexityClass::Linearithmic), "O(n log n)");
    assert_eq!(format!("{}", ComplexityClass::Quadratic), "O(n²)");
    assert_eq!(format!("{}", ComplexityClass::Cubic), "O(n³)");
    assert_eq!(format!("{}", ComplexityClass::Exponential), "O(2ⁿ)");
    assert_eq!(format!("{}", ComplexityClass::Unknown), "O(?)");
}

#[test]
fn test_ast_node_ref_creation() {
    let source_id = SourceId::new(42);
    let node_ref = AstNodeRef::new(123, source_id);
    
    assert_eq!(node_ref.id(), 123);
    assert_eq!(node_ref.source_id(), source_id);
}

#[test]
fn test_ast_node_ref_equality() {
    let source_id = SourceId::new(1);
    let ref1 = AstNodeRef::new(10, source_id);
    let ref2 = AstNodeRef::new(10, source_id);
    let ref3 = AstNodeRef::new(20, source_id);
    
    assert_eq!(ref1, ref2);
    assert_ne!(ref1, ref3);
}

#[test]
fn test_ast_node_ref_hash() {
    use std::collections::HashSet;
    
    let source_id = SourceId::new(1);
    let ref1 = AstNodeRef::new(10, source_id);
    let ref2 = AstNodeRef::new(10, source_id);
    let ref3 = AstNodeRef::new(20, source_id);
    
    let mut set = HashSet::new();
    set.insert(ref1);
    set.insert(ref2); // Should not increase size (same as ref1)
    set.insert(ref3);
    
    assert_eq!(set.len(), 2);
}

#[cfg(feature = "serde")]
#[test]
fn test_ast_node_serialization() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::String("serializable".to_string()),
    });
    
    let node = AstNode::new(expr, span, node_id)
        .with_semantic_annotation("test annotation".to_string());
    
    // Test that the node can be serialized and deserialized
    let serialized = serde_json::to_string(&node).expect("Serialization failed");
    let deserialized: AstNode<Expr> = serde_json::from_str(&serialized).expect("Deserialization failed");
    
    assert_eq!(deserialized.span, node.span);
    assert_eq!(deserialized.id, node.id);
    assert_eq!(deserialized.has_semantic_annotations(), node.has_semantic_annotations());
}

#[test]
fn test_node_metadata_default() {
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
fn test_node_metadata_comprehensive() {
    let mut metadata = NodeMetadata::default();
    
    metadata.semantic_annotations.push("Primary business logic".to_string());
    metadata.business_rules.push("Must validate input".to_string());
    metadata.performance_characteristics.push("O(1) lookup".to_string());
    metadata.security_implications.push("Handles sensitive data".to_string());
    metadata.compliance_requirements.push("GDPR compliant".to_string());
    metadata.is_ai_generated = true;
    metadata.semantic_importance = 0.95;
    metadata.security_sensitive = true;
    metadata.documentation = Some("Core authentication function".to_string());
    metadata.examples.push("authenticate(user, password)".to_string());
    metadata.common_mistakes.push("Forgetting to hash password".to_string());
    metadata.related_concepts.push("authorization".to_string());
    metadata.architectural_patterns.push("Strategy Pattern".to_string());
    
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