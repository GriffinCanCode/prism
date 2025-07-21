//! Comprehensive tests for the type inference module

use prism_ast::{
    type_inference::{self, *},
    types::{self, *},
    expr::*,
    node::*,
    metadata::*,
    AstNode,
    Type,
    PrimitiveType,
    IntegerType,
    SecurityClassification,
    TypeConstraint,
    RangeConstraint,
    Effect,
    IOEffect,
    IOOperation,
    SemanticType,
    SemanticTypeMetadata,
    PatternConstraint,
};
use prism_common::{span::Span, symbol::Symbol, SourceId, Position, NodeId};

#[test]
fn test_type_environment_creation() {
    let env = TypeEnvironment::new();
    
    assert!(env.variables.is_empty());
    assert!(env.functions.is_empty());
    assert!(env.type_aliases.is_empty());
    assert!(env.scopes.is_empty());
}

#[test]
fn test_inferred_type_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let type_node = AstNode::new(
        Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
        span,
        NodeId::new(1),
    );
    
    let inferred_type = InferredType {
        type_node,
        confidence: 0.95,
        source: InferenceSource::Usage,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    assert_eq!(inferred_type.confidence, 0.95);
    match inferred_type.source {
        InferenceSource::Usage => {}, // Expected
        _ => panic!("Expected usage inference source"),
    }
    assert!(inferred_type.constraints.is_empty());
    assert!(inferred_type.ai_suggestions.is_empty());
}

#[test]
fn test_function_type_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        ),
        confidence: 1.0,
        source: InferenceSource::Explicit,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    let return_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(2),
        ),
        confidence: 0.9,
        source: InferenceSource::Context,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    let func_type = type_inference::FunctionType {
        parameters: vec![param_type],
        return_type,
        effects: vec![],
    };
    
    assert_eq!(func_type.parameters.len(), 1);
    assert_eq!(func_type.parameters[0].confidence, 1.0);
    assert_eq!(func_type.return_type.confidence, 0.9);
    assert!(func_type.effects.is_empty());
}

#[test]
fn test_inference_source_variants() {
    let explicit = InferenceSource::Explicit;
    let usage = InferenceSource::Usage;
    let context = InferenceSource::Context;
    let constraints = InferenceSource::Constraints;
    let ai_suggested = InferenceSource::AISuggested;
    let unified = InferenceSource::Unified(vec![
        InferenceSource::Usage,
        InferenceSource::Context,
    ]);
    
    // Test that all variants can be created
    match explicit {
        InferenceSource::Explicit => {}, // Expected
        _ => panic!("Expected explicit source"),
    }
    
    match unified {
        InferenceSource::Unified(sources) => {
            assert_eq!(sources.len(), 2);
        }
        _ => panic!("Expected unified source"),
    }
}

#[test]
fn test_ai_suggestion_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let suggestion = AISuggestion {
        suggested_type: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        ),
        confidence: 0.85,
        reasoning: "Variable name suggests string content".to_string(),
        evidence: vec![
            "Variable named 'message'".to_string(),
            "Used in string concatenation".to_string(),
        ],
    };
    
    assert_eq!(suggestion.confidence, 0.85);
    assert_eq!(suggestion.reasoning, "Variable name suggests string content");
    assert_eq!(suggestion.evidence.len(), 2);
    
    if let Type::Primitive(PrimitiveType::String) = &suggestion.suggested_type.kind {
        // Expected
    } else {
        panic!("Expected string type suggestion");
    }
}

#[test]
fn test_constraint_solver_creation() {
    let config = SolverConfig::default();
    let solver = ConstraintSolver::new(config);
    
    assert!(solver.constraints.is_empty());
    assert_eq!(solver.config.max_iterations, 100);
    assert_eq!(solver.config.convergence_threshold, 0.001);
    assert!(solver.config.enable_simplification);
}

#[test]
fn test_type_constraint_set() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let constraint = TypeConstraint::Range(RangeConstraint {
        min: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(0),
            }),
            span,
            NodeId::new(1),
        )),
        max: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(100),
            }),
            span,
            NodeId::new(2),
        )),
        inclusive: true,
    });
    
    let constraint_set = TypeConstraintSet {
        id: 1,
        constraints: vec![constraint],
        variables: {
            let mut vars = std::collections::HashSet::new();
            vars.insert(Symbol::intern("x"));
            vars
        },
        relationships: vec![],
    };
    
    assert_eq!(constraint_set.id, 1);
    assert_eq!(constraint_set.constraints.len(), 1);
    assert_eq!(constraint_set.variables.len(), 1);
    assert!(constraint_set.variables.contains(&Symbol::intern("x")));
    assert!(constraint_set.relationships.is_empty());
}

#[test]
fn test_constraint_relationship() {
    let relationship = ConstraintRelationship {
        relationship_type: RelationshipType::Implication,
        related_constraints: vec![1, 2, 3],
    };
    
    match relationship.relationship_type {
        RelationshipType::Implication => {}, // Expected
        _ => panic!("Expected implication relationship"),
    }
    
    assert_eq!(relationship.related_constraints.len(), 3);
    assert_eq!(relationship.related_constraints[0], 1);
}

#[test]
fn test_effect_tracker_creation() {
    let tracker = EffectTracker::new();
    
    assert!(tracker.current_effects.is_empty());
    assert!(tracker.effect_history.is_empty());
    assert!(tracker.composition_rules.is_empty());
}

#[test]
fn test_effect_context() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let io_effect = Effect::IO(IOEffect {
        operations: vec![IOOperation::Read],
        resources: vec![],
        constraints: vec![],
    });
    
    let context = EffectContext {
        id: 42,
        effects: vec![io_effect],
        span,
    };
    
    assert_eq!(context.id, 42);
    assert_eq!(context.effects.len(), 1);
    assert_eq!(context.span, span);
    
    match &context.effects[0] {
        Effect::IO(io) => {
            assert_eq!(io.operations.len(), 1);
            assert!(matches!(io.operations[0], IOOperation::Read));
        }
        _ => panic!("Expected IO effect"),
    }
}

#[test]
fn test_effect_composition_rule() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let input_effect1 = Effect::IO(IOEffect {
        operations: vec![IOOperation::Read],
        resources: vec![],
        constraints: vec![],
    });
    
    let input_effect2 = Effect::IO(IOEffect {
        operations: vec![IOOperation::Write],
        resources: vec![],
        constraints: vec![],
    });
    
    let output_effect = Effect::IO(IOEffect {
        operations: vec![IOOperation::Read, IOOperation::Write],
        resources: vec![],
        constraints: vec![],
    });
    
    let rule = type_inference::EffectCompositionRule {
        name: "Combine IO operations".to_string(),
        input_effects: vec![input_effect1, input_effect2],
        output_effect,
        conditions: vec![AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Boolean(true),
            }),
            span,
            NodeId::new(1),
        )],
    };
    
    assert_eq!(rule.name, "Combine IO operations");
    assert_eq!(rule.input_effects.len(), 2);
    assert_eq!(rule.conditions.len(), 1);
    
    match &rule.output_effect {
        Effect::IO(io) => {
            assert_eq!(io.operations.len(), 2);
        }
        _ => panic!("Expected IO output effect"),
    }
}

#[test]
fn test_ai_type_assistant_creation() {
    let config = AIConfig::default();
    let assistant = AITypeAssistant::new(config);
    
    assert!(assistant.type_patterns.is_empty());
    assert!(assistant.inference_history.is_empty());
    assert!(assistant.config.enable_pattern_matching);
    assert!(assistant.config.enable_learning);
    assert_eq!(assistant.config.max_suggestions, 5);
}

#[test]
fn test_type_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let pattern = TypePattern {
        name: "String variable pattern".to_string(),
        description: "Variables ending with '_name' are likely strings".to_string(),
        conditions: vec![
            PatternCondition {
                condition_type: ConditionType::VariableName,
                value: "*_name".to_string(),
            },
            PatternCondition {
                condition_type: ConditionType::UsageContext,
                value: "string_operation".to_string(),
            },
        ],
        suggested_type: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        ),
        confidence: 0.8,
    };
    
    assert_eq!(pattern.name, "String variable pattern");
    assert_eq!(pattern.conditions.len(), 2);
    assert_eq!(pattern.confidence, 0.8);
    
    match &pattern.conditions[0].condition_type {
        ConditionType::VariableName => {
            assert_eq!(pattern.conditions[0].value, "*_name");
        }
        _ => panic!("Expected variable name condition"),
    }
    
    match &pattern.conditions[1].condition_type {
        ConditionType::UsageContext => {
            assert_eq!(pattern.conditions[1].value, "string_operation");
        }
        _ => panic!("Expected usage context condition"),
    }
}

#[test]
fn test_pattern_condition_types() {
    let var_name_condition = PatternCondition {
        condition_type: ConditionType::VariableName,
        value: "user_*".to_string(),
    };
    
    let func_name_condition = PatternCondition {
        condition_type: ConditionType::FunctionName,
        value: "get_*".to_string(),
    };
    
    let usage_context_condition = PatternCondition {
        condition_type: ConditionType::UsageContext,
        value: "database_query".to_string(),
    };
    
    let constraint_condition = PatternCondition {
        condition_type: ConditionType::Constraint,
        value: "length > 0".to_string(),
    };
    
    let custom_condition = PatternCondition {
        condition_type: ConditionType::Custom("domain_specific".to_string()),
        value: "payment_amount".to_string(),
    };
    
    // Test all condition types can be created
    match var_name_condition.condition_type {
        ConditionType::VariableName => {}, // Expected
        _ => panic!("Expected variable name condition"),
    }
    
    match custom_condition.condition_type {
        ConditionType::Custom(ref name) => {
            assert_eq!(name, "domain_specific");
        }
        _ => panic!("Expected custom condition"),
    }
}

#[test]
fn test_inference_record() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let input_context = InferenceContext {
        expression: AstNode::new(
            Expr::Variable(VariableExpr {
                name: Symbol::intern("user_name"),
            }),
            span,
            NodeId::new(1),
        ),
        context: vec![],
        type_info: std::collections::HashMap::new(),
    };
    
    let inferred_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(2),
        ),
        confidence: 0.9,
        source: InferenceSource::AISuggested,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    let record = InferenceRecord {
        id: 123,
        input_context,
        inferred_type,
        success: true,
        feedback_score: Some(0.95),
    };
    
    assert_eq!(record.id, 123);
    assert!(record.success);
    assert_eq!(record.feedback_score, Some(0.95));
    
    if let Expr::Variable(var) = &record.input_context.expression.kind {
        assert_eq!(var.name, Symbol::intern("user_name"));
    }
    
    assert_eq!(record.inferred_type.confidence, 0.9);
}

#[test]
fn test_inference_config_creation() {
    let config = InferenceConfig::default();
    
    assert!(config.enable_ai_assistance);
    assert_eq!(config.max_inference_depth, 10);
    assert_eq!(config.min_confidence_threshold, 0.7);
    assert!(config.enable_constraint_propagation);
    assert!(config.enable_effect_inference);
    
    // Test custom config
    let custom_config = InferenceConfig {
        enable_ai_assistance: false,
        max_inference_depth: 5,
        min_confidence_threshold: 0.8,
        enable_constraint_propagation: false,
        enable_effect_inference: false,
    };
    
    assert!(!custom_config.enable_ai_assistance);
    assert_eq!(custom_config.max_inference_depth, 5);
    assert_eq!(custom_config.min_confidence_threshold, 0.8);
    assert!(!custom_config.enable_constraint_propagation);
    assert!(!custom_config.enable_effect_inference);
}

#[test]
fn test_solver_config_creation() {
    let config = SolverConfig::default();
    
    assert_eq!(config.max_iterations, 100);
    assert_eq!(config.convergence_threshold, 0.001);
    assert!(config.enable_simplification);
    
    // Test custom config
    let custom_config = SolverConfig {
        max_iterations: 50,
        convergence_threshold: 0.01,
        enable_simplification: false,
    };
    
    assert_eq!(custom_config.max_iterations, 50);
    assert_eq!(custom_config.convergence_threshold, 0.01);
    assert!(!custom_config.enable_simplification);
}

#[test]
fn test_ai_config_creation() {
    let config = AIConfig::default();
    
    assert!(config.enable_pattern_matching);
    assert!(config.enable_learning);
    assert_eq!(config.max_suggestions, 5);
    
    // Test custom config
    let custom_config = AIConfig {
        enable_pattern_matching: false,
        enable_learning: false,
        max_suggestions: 3,
    };
    
    assert!(!custom_config.enable_pattern_matching);
    assert!(!custom_config.enable_learning);
    assert_eq!(custom_config.max_suggestions, 3);
}

#[test]
fn test_type_inference_engine_creation() {
    let config = InferenceConfig::default();
    let engine = TypeInferenceEngine::new(config);
    
    // Check that all components are initialized
    assert!(engine.type_env.variables.is_empty());
    assert!(engine.constraint_solver.constraints.is_empty());
    assert!(engine.effect_tracker.current_effects.is_empty());
    assert!(engine.ai_assistant.type_patterns.is_empty());
    assert!(engine.config.enable_ai_assistance);
}

#[test]
fn test_inference_error_variants() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let cannot_infer = InferenceError::CannotInferType { span };
    let constraint_conflict = InferenceError::ConstraintConflict {
        message: "Conflicting constraints".to_string(),
    };
    let effect_failed = InferenceError::EffectInferenceFailed {
        message: "Effect inference failed".to_string(),
    };
    let circular_dependency = InferenceError::CircularDependency;
    let insufficient_info = InferenceError::InsufficientInformation;
    let ai_unavailable = InferenceError::AIUnavailable {
        reason: "AI service down".to_string(),
    };
    
    // Test error messages
    assert!(cannot_infer.to_string().contains("Cannot infer type"));
    assert!(constraint_conflict.to_string().contains("Conflicting constraints"));
    assert!(effect_failed.to_string().contains("Effect inference failed"));
    assert!(circular_dependency.to_string().contains("Circular"));
    assert!(insufficient_info.to_string().contains("Insufficient"));
    assert!(ai_unavailable.to_string().contains("AI service down"));
}

#[test]
fn test_relationship_type_variants() {
    let mutually_exclusive = RelationshipType::MutuallyExclusive;
    let implication = RelationshipType::Implication;
    let equivalence = RelationshipType::Equivalence;
    let conflict = RelationshipType::Conflict;
    
    // Test that all variants can be created and compared
    assert_eq!(mutually_exclusive, RelationshipType::MutuallyExclusive);
    assert_eq!(implication, RelationshipType::Implication);
    assert_eq!(equivalence, RelationshipType::Equivalence);
    assert_eq!(conflict, RelationshipType::Conflict);
    
    // Test inequality
    assert_ne!(mutually_exclusive, implication);
    assert_ne!(equivalence, conflict);
}

#[test]
fn test_complex_inference_scenario() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create a complex type inference scenario
    let mut type_env = TypeEnvironment::new();
    
    // Add some variables to the environment
    let string_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        ),
        confidence: 1.0,
        source: InferenceSource::Explicit,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    let int_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(2),
        ),
        confidence: 0.9,
        source: InferenceSource::Usage,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    type_env.variables.insert(Symbol::intern("name"), string_type);
    type_env.variables.insert(Symbol::intern("age"), int_type);
    
    // Create a function type
    let param_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(3),
        ),
        confidence: 1.0,
        source: InferenceSource::Explicit,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    let return_type = InferredType {
        type_node: AstNode::new(
            Type::Primitive(PrimitiveType::Boolean),
            span,
            NodeId::new(4),
        ),
        confidence: 0.8,
        source: InferenceSource::Context,
        constraints: vec![],
        ai_suggestions: vec![],
    };
    
    let func_type = type_inference::FunctionType {
        parameters: vec![param_type],
        return_type,
        effects: vec![],
    };
    
    type_env.functions.insert(Symbol::intern("validate"), func_type);
    
    // Verify the environment
    assert_eq!(type_env.variables.len(), 2);
    assert_eq!(type_env.functions.len(), 1);
    
    let name_type = type_env.variables.get(&Symbol::intern("name")).unwrap();
    assert_eq!(name_type.confidence, 1.0);
    
    let validate_func = type_env.functions.get(&Symbol::intern("validate")).unwrap();
    assert_eq!(validate_func.parameters.len(), 1);
    assert_eq!(validate_func.return_type.confidence, 0.8);
}

#[test]
fn test_ai_suggestion_with_evidence() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let suggestion = AISuggestion {
        suggested_type: AstNode::new(
            Type::Semantic(SemanticType {
                base_type: Box::new(AstNode::new(
                    Type::Primitive(PrimitiveType::String),
                    span,
                    NodeId::new(1),
                )),
                constraints: vec![
                    TypeConstraint::Pattern(PatternConstraint {
                        pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
                        flags: vec![],
                    }),
                ],
                metadata: SemanticTypeMetadata {
                    business_rules: vec!["Must be valid email format".to_string()],
                    examples: vec!["user@example.com".to_string()],
                    validation_rules: vec!["RFC 5322 compliant".to_string()],
                    ai_context: Some("Email address for user communication".to_string()),
                    security_classification: SecurityClassification::Internal,
                    compliance_requirements: vec!["GDPR".to_string()],
                },
            }),
            span,
            NodeId::new(2),
        ),
        confidence: 0.92,
        reasoning: "Variable name 'email' and usage patterns suggest email address type".to_string(),
        evidence: vec![
            "Variable named 'email'".to_string(),
            "Used with email validation function".to_string(),
            "Passed to sendEmail() function".to_string(),
            "Matches email regex pattern in code".to_string(),
        ],
    };
    
    assert_eq!(suggestion.confidence, 0.92);
    assert_eq!(suggestion.evidence.len(), 4);
    assert!(suggestion.reasoning.contains("email"));
    
    // Verify the suggested semantic type
    if let Type::Semantic(semantic) = &suggestion.suggested_type.kind {
        assert_eq!(semantic.constraints.len(), 1);
        assert_eq!(semantic.metadata.business_rules.len(), 1);
        assert_eq!(semantic.metadata.examples.len(), 1);
        assert_eq!(semantic.metadata.security_classification, SecurityClassification::Internal);
    } else {
        panic!("Expected semantic type suggestion");
    }
} 