//! Comprehensive tests for the types module

use prism_common::span::Position;
use prism_ast::{types::*, expr::*, node::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};
use std::collections::HashMap;

#[test]
fn test_primitive_types() {
    let bool_type = PrimitiveType::Boolean;
    let int_type = PrimitiveType::Integer(IntegerType::Signed(32));
    let uint_type = PrimitiveType::Integer(IntegerType::Unsigned(64));
    let natural_type = PrimitiveType::Integer(IntegerType::Natural);
    let bigint_type = PrimitiveType::Integer(IntegerType::BigInt);
    let float_type = PrimitiveType::Float(FloatType::F64);
    let decimal_type = PrimitiveType::Float(FloatType::Decimal);
    let string_type = PrimitiveType::String;
    let char_type = PrimitiveType::Char;
    let unit_type = PrimitiveType::Unit;
    let never_type = PrimitiveType::Never;
    
    // Test integer types
    match int_type {
        PrimitiveType::Integer(IntegerType::Signed(bits)) => assert_eq!(bits, 32),
        _ => panic!("Expected signed integer"),
    }
    
    match uint_type {
        PrimitiveType::Integer(IntegerType::Unsigned(bits)) => assert_eq!(bits, 64),
        _ => panic!("Expected unsigned integer"),
    }
    
    // Test float types
    match float_type {
        PrimitiveType::Float(FloatType::F64) => {}, // Expected
        _ => panic!("Expected F64 float"),
    }
    
    // Test other primitives
    assert!(matches!(bool_type, PrimitiveType::Boolean));
    assert!(matches!(string_type, PrimitiveType::String));
    assert!(matches!(char_type, PrimitiveType::Char));
    assert!(matches!(unit_type, PrimitiveType::Unit));
    assert!(matches!(never_type, PrimitiveType::Never));
}

#[test]
fn test_named_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let type_arg = AstNode::new(
        Type::Primitive(PrimitiveType::String),
        span,
        NodeId::new(1),
    );
    
    let named_type = NamedType {
        name: Symbol::intern("Vec"),
        type_arguments: vec![type_arg],
    };
    
    assert_eq!(named_type.name, Symbol::intern("Vec"));
    assert_eq!(named_type.type_arguments.len(), 1);
    
    if let Type::Primitive(PrimitiveType::String) = &named_type.type_arguments[0].kind {
        // Expected
    } else {
        panic!("Expected string type argument");
    }
}

#[test]
fn test_generic_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let type_param = TypeParameter {
        name: Symbol::intern("T"),
        bounds: vec![],
        default: Some(AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        )),
    };
    
    let base_type = Box::new(AstNode::new(
        Type::Named(NamedType {
            name: Symbol::intern("Container"),
            type_arguments: vec![],
        }),
        span,
        NodeId::new(2),
    ));
    
    let generic_type = GenericType {
        parameters: vec![type_param],
        base_type,
    };
    
    assert_eq!(generic_type.parameters.len(), 1);
    assert_eq!(generic_type.parameters[0].name, Symbol::intern("T"));
    assert!(generic_type.parameters[0].default.is_some());
}

#[test]
fn test_function_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param_type = AstNode::new(
        Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
        span,
        NodeId::new(1),
    );
    
    let return_type = Box::new(AstNode::new(
        Type::Primitive(PrimitiveType::String),
        span,
        NodeId::new(2),
    ));
    
    let io_effect = Effect::IO(IOEffect {
        operations: vec![IOOperation::Write],
        resources: vec![IOResource {
            resource_type: IOResourceType::StandardIO,
            identifier: "stdout".to_string(),
            access_pattern: AccessPattern::Write,
        }],
        constraints: vec![],
    });
    
    let func_type = FunctionType {
        parameters: vec![param_type],
        return_type,
        effects: vec![io_effect],
    };
    
    assert_eq!(func_type.parameters.len(), 1);
    assert_eq!(func_type.effects.len(), 1);
    
    match &func_type.effects[0] {
        Effect::IO(io) => {
            assert_eq!(io.operations.len(), 1);
            assert!(matches!(io.operations[0], IOOperation::Write));
        }
        _ => panic!("Expected IO effect"),
    }
}

#[test]
fn test_tuple_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let elements = vec![
        AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Type::Primitive(PrimitiveType::Boolean),
            span,
            NodeId::new(3),
        ),
    ];
    
    let tuple_type = TupleType { elements };
    
    assert_eq!(tuple_type.elements.len(), 3);
}

#[test]
fn test_array_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let element_type = Box::new(AstNode::new(
        Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
        span,
        NodeId::new(1),
    ));
    
    let size = Some(Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(10),
        }),
        span,
        NodeId::new(2),
    )));
    
    let array_type = ArrayType {
        element_type,
        size,
    };
    
    assert!(array_type.size.is_some());
    
    if let Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))) = &array_type.element_type.kind {
        // Expected
    } else {
        panic!("Expected i32 element type");
    }
}

#[test]
fn test_semantic_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let base_type = Box::new(AstNode::new(
        Type::Primitive(PrimitiveType::String),
        span,
        NodeId::new(1),
    ));
    
    let range_constraint = TypeConstraint::Range(RangeConstraint {
        min: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1),
            }),
            span,
            NodeId::new(2),
        )),
        max: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(100),
            }),
            span,
            NodeId::new(3),
        )),
        inclusive: true,
    });
    
    let pattern_constraint = TypeConstraint::Pattern(PatternConstraint {
        pattern: r"^[a-zA-Z0-9]+$".to_string(),
        flags: vec!["i".to_string()],
    });
    
    let business_rule_constraint = TypeConstraint::BusinessRule(BusinessRuleConstraint {
        description: "Username must be unique".to_string(),
        expression: AstNode::new(
            Expr::Call(CallExpr {
                callee: Box::new(AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("is_unique"),
                    }),
                    span,
                    NodeId::new(4),
                )),
                arguments: vec![],
                type_arguments: None,
                call_style: CallStyle::Function,
            }),
            span,
            NodeId::new(5),
        ),
        priority: 1,
    });
    
    let metadata = SemanticTypeMetadata {
        business_rules: vec!["Must be unique".to_string()],
        examples: vec!["john_doe".to_string(), "alice123".to_string()],
        validation_rules: vec!["Length between 1-100".to_string()],
        ai_context: Some("Username for user authentication".to_string()),
        security_classification: SecurityClassification::Internal,
        compliance_requirements: vec!["GDPR".to_string()],
    };
    
    let semantic_type = SemanticType {
        base_type,
        constraints: vec![range_constraint, pattern_constraint, business_rule_constraint],
        metadata,
    };
    
    assert_eq!(semantic_type.constraints.len(), 3);
    assert_eq!(semantic_type.metadata.business_rules.len(), 1);
    assert_eq!(semantic_type.metadata.examples.len(), 2);
    assert_eq!(semantic_type.metadata.security_classification, SecurityClassification::Internal);
    
    // Test constraint types
    match &semantic_type.constraints[0] {
        TypeConstraint::Range(range) => {
            assert!(range.inclusive);
            assert!(range.min.is_some());
            assert!(range.max.is_some());
        }
        _ => panic!("Expected range constraint"),
    }
    
    match &semantic_type.constraints[1] {
        TypeConstraint::Pattern(pattern) => {
            assert_eq!(pattern.pattern, r"^[a-zA-Z0-9]+$");
            assert_eq!(pattern.flags.len(), 1);
        }
        _ => panic!("Expected pattern constraint"),
    }
    
    match &semantic_type.constraints[2] {
        TypeConstraint::BusinessRule(rule) => {
            assert_eq!(rule.description, "Username must be unique");
            assert_eq!(rule.priority, 1);
        }
        _ => panic!("Expected business rule constraint"),
    }
}

#[test]
fn test_security_classification_default() {
    let default_classification = SecurityClassification::default();
    assert_eq!(default_classification, SecurityClassification::Public);
}

#[test]
fn test_dependent_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param = DependentParameter {
        name: Symbol::intern("n"),
        parameter_type: AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Natural)),
            span,
            NodeId::new(1),
        ),
        bounds: vec![ParameterBound::Lower(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(0),
            }),
            span,
            NodeId::new(2),
        ))],
        default_value: None,
        is_compile_time_constant: true,
        role: ParameterRole::Size,
    };
    
    let type_expression = AstNode::new(
        Expr::Call(CallExpr {
            callee: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("Array"),
                }),
                span,
                NodeId::new(3),
            )),
            arguments: vec![AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("n"),
                }),
                span,
                NodeId::new(4),
            )],
            type_arguments: None,
            call_style: CallStyle::Constructor,
        }),
        span,
        NodeId::new(5),
    );
    
    let constraint = ParameterConstraint {
        expression: AstNode::new(
            Expr::Binary(BinaryExpr {
                left: Box::new(AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("n"),
                    }),
                    span,
                    NodeId::new(6),
                )),
                operator: BinaryOperator::Less,
                right: Box::new(AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(1000),
                    }),
                    span,
                    NodeId::new(7),
                )),
            }),
            span,
            NodeId::new(8),
        ),
        description: "Array size must be reasonable".to_string(),
        severity: ConstraintSeverity::Error,
    };
    
    let validation_predicate = ValidationPredicate {
        expression: AstNode::new(
            Expr::Binary(BinaryExpr {
                left: Box::new(AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("n"),
                    }),
                    span,
                    NodeId::new(9),
                )),
                operator: BinaryOperator::GreaterEqual,
                right: Box::new(AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(0),
                    }),
                    span,
                    NodeId::new(10),
                )),
            }),
            span,
            NodeId::new(11),
        ),
        error_message: "Array size must be non-negative".to_string(),
        level: ValidationLevel::CompileTime,
    };
    
    let dependent_type = DependentType {
        parameters: vec![param],
        type_expression,
        parameter_constraints: vec![constraint],
        computation_rules: vec![],
        validation_predicates: vec![validation_predicate],
        refinement_conditions: vec![],
    };
    
    assert_eq!(dependent_type.parameters.len(), 1);
    assert_eq!(dependent_type.parameter_constraints.len(), 1);
    assert_eq!(dependent_type.validation_predicates.len(), 1);
    assert_eq!(dependent_type.parameters[0].name, Symbol::intern("n"));
    assert!(dependent_type.parameters[0].is_compile_time_constant);
    assert_eq!(dependent_type.parameters[0].role, ParameterRole::Size);
}

#[test]
fn test_effect_types() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // IO Effect
    let io_effect = Effect::IO(IOEffect {
        operations: vec![IOOperation::Read, IOOperation::Write],
        resources: vec![
            IOResource {
                resource_type: IOResourceType::File,
                identifier: "/tmp/data.txt".to_string(),
                access_pattern: AccessPattern::ReadWrite,
            },
        ],
        constraints: vec![IOConstraint {
            constraint_type: IOConstraintType::MaxFileSize,
            value: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(1024 * 1024), // 1MB
                }),
                span,
                NodeId::new(1),
            ),
        }],
    });
    
    // State Effect
    let state_effect = Effect::State(StateEffect {
        state_type: AstNode::new(
            Type::Named(NamedType {
                name: Symbol::intern("UserSession"),
                type_arguments: vec![],
            }),
            span,
            NodeId::new(2),
        ),
        access_pattern: AccessPattern::ReadWrite,
        constraints: vec![StateConstraint {
            constraint_type: StateConstraintType::ThreadSafe,
            value: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Boolean(true),
                }),
                span,
                NodeId::new(3),
            ),
        }],
    });
    
    // Database Effect
    let db_effect = Effect::Database(DatabaseEffect {
        operations: vec![DatabaseOperation::Read, DatabaseOperation::Write],
        transaction_required: true,
        constraints: vec![DatabaseConstraint {
            constraint_type: DatabaseConstraintType::QueryTimeout,
            value: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(30), // 30 seconds
                }),
                span,
                NodeId::new(4),
            ),
        }],
    });
    
    // Test IO Effect
    match &io_effect {
        Effect::IO(io) => {
            assert_eq!(io.operations.len(), 2);
            assert_eq!(io.resources.len(), 1);
            assert_eq!(io.constraints.len(), 1);
            assert!(matches!(io.operations[0], IOOperation::Read));
            assert!(matches!(io.operations[1], IOOperation::Write));
        }
        _ => panic!("Expected IO effect"),
    }
    
    // Test State Effect
    match &state_effect {
        Effect::State(state) => {
            assert!(matches!(state.access_pattern, AccessPattern::ReadWrite));
            assert_eq!(state.constraints.len(), 1);
        }
        _ => panic!("Expected state effect"),
    }
    
    // Test Database Effect
    match &db_effect {
        Effect::Database(db) => {
            assert_eq!(db.operations.len(), 2);
            assert!(db.transaction_required);
            assert_eq!(db.constraints.len(), 1);
        }
        _ => panic!("Expected database effect"),
    }
}

#[test]
fn test_union_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let members = vec![
        AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Type::Primitive(PrimitiveType::Boolean),
            span,
            NodeId::new(3),
        ),
    ];
    
    let discriminant = Some(UnionDiscriminant {
        field_name: Symbol::intern("type"),
        discriminant_type: Box::new(AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(4),
        )),
        tag_mapping: {
            let mut map = HashMap::new();
            map.insert("string".to_string(), 0);
            map.insert("number".to_string(), 1);
            map.insert("boolean".to_string(), 2);
            map
        },
    });
    
    let constraint = UnionConstraint::MaxMembers(3);
    
    let metadata = UnionMetadata {
        name: Some(Symbol::intern("JsonValue")),
        description: Some("A JSON value type".to_string()),
        semantics: UnionSemantics::Disjoint,
        ai_context: Some("Represents any valid JSON value".to_string()),
    };
    
    let union_type = UnionType {
        members,
        discriminant: Some(Box::new(discriminant.unwrap())),
        constraints: vec![constraint],
        common_operations: vec![],
        metadata,
    };
    
    assert_eq!(union_type.members.len(), 3);
    assert!(union_type.discriminant.is_some());
    assert_eq!(union_type.constraints.len(), 1);
    
    if let Some(disc) = &union_type.discriminant {
        assert_eq!(disc.field_name, Symbol::intern("type"));
        assert_eq!(disc.tag_mapping.len(), 3);
        assert_eq!(disc.tag_mapping.get("string"), Some(&0));
    }
}

#[test]
fn test_intersection_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let members = vec![
        AstNode::new(
            Type::Named(NamedType {
                name: Symbol::intern("Readable"),
                type_arguments: vec![],
            }),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Type::Named(NamedType {
                name: Symbol::intern("Writable"),
                type_arguments: vec![],
            }),
            span,
            NodeId::new(2),
        ),
    ];
    
    let constraint = IntersectionConstraint::Compatibility(vec![0, 1]);
    
    let metadata = IntersectionMetadata {
        name: Some(Symbol::intern("ReadWritable")),
        description: Some("A type that is both readable and writable".to_string()),
        semantics: IntersectionSemantics::Structural,
        ai_context: Some("Combines read and write capabilities".to_string()),
    };
    
    let intersection_type = IntersectionType {
        members,
        constraints: vec![constraint],
        merged_operations: vec![],
        metadata,
    };
    
    assert_eq!(intersection_type.members.len(), 2);
    assert_eq!(intersection_type.constraints.len(), 1);
    
    match &intersection_type.constraints[0] {
        IntersectionConstraint::Compatibility(indices) => {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0], 0);
            assert_eq!(indices[1], 1);
        }
        _ => panic!("Expected compatibility constraint"),
    }
}

#[test]
fn test_type_declarations() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Type alias
    let alias_decl = TypeDecl {
        name: Symbol::intern("UserId"),
        type_parameters: vec![],
        kind: TypeKind::Alias(AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(64))),
            span,
            NodeId::new(1),
        )),
        visibility: Visibility::Public,
    };
    
    // Enum type
    let enum_variant1 = EnumVariant {
        name: Symbol::intern("Success"),
        fields: vec![AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(2),
        )],
    };
    
    let enum_variant2 = EnumVariant {
        name: Symbol::intern("Error"),
        fields: vec![AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(3),
        )],
    };
    
    let enum_decl = TypeDecl {
        name: Symbol::intern("Result"),
        type_parameters: vec![],
        kind: TypeKind::Enum(EnumType {
            variants: vec![enum_variant1, enum_variant2],
        }),
        visibility: Visibility::Public,
    };
    
    // Struct type
    let struct_field1 = StructField {
        name: Symbol::intern("id"),
        field_type: AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(64))),
            span,
            NodeId::new(4),
        ),
        visibility: Visibility::Public,
    };
    
    let struct_field2 = StructField {
        name: Symbol::intern("name"),
        field_type: AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(5),
        ),
        visibility: Visibility::Public,
    };
    
    let struct_decl = TypeDecl {
        name: Symbol::intern("User"),
        type_parameters: vec![],
        kind: TypeKind::Struct(StructType {
            fields: vec![struct_field1, struct_field2],
        }),
        visibility: Visibility::Public,
    };
    
    // Test type alias
    assert_eq!(alias_decl.name, Symbol::intern("UserId"));
    match &alias_decl.kind {
        TypeKind::Alias(ty) => {
            if let Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(64))) = &ty.kind {
                // Expected
            } else {
                panic!("Expected u64 type");
            }
        }
        _ => panic!("Expected type alias"),
    }
    
    // Test enum
    assert_eq!(enum_decl.name, Symbol::intern("Result"));
    match &enum_decl.kind {
        TypeKind::Enum(enum_type) => {
            assert_eq!(enum_type.variants.len(), 2);
            assert_eq!(enum_type.variants[0].name, Symbol::intern("Success"));
            assert_eq!(enum_type.variants[1].name, Symbol::intern("Error"));
        }
        _ => panic!("Expected enum type"),
    }
    
    // Test struct
    assert_eq!(struct_decl.name, Symbol::intern("User"));
    match &struct_decl.kind {
        TypeKind::Struct(struct_type) => {
            assert_eq!(struct_type.fields.len(), 2);
            assert_eq!(struct_type.fields[0].name, Symbol::intern("id"));
            assert_eq!(struct_type.fields[1].name, Symbol::intern("name"));
        }
        _ => panic!("Expected struct type"),
    }
}

#[test]
fn test_complex_semantic_type() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create a complex semantic type for an email address
    let base_type = Box::new(AstNode::new(
        Type::Primitive(PrimitiveType::String),
        span,
        NodeId::new(1),
    ));
    
    let length_constraint = TypeConstraint::Length(LengthConstraint {
        min_length: Some(5),
        max_length: Some(254),
    });
    
    let pattern_constraint = TypeConstraint::Pattern(PatternConstraint {
        pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
        flags: vec![],
    });
    
    let format_constraint = TypeConstraint::Format(FormatConstraint {
        format: "email".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("allow_international".to_string(), "true".to_string());
            params
        },
    });
    
    let custom_constraint = TypeConstraint::Custom(CustomConstraint {
        name: "domain_whitelist".to_string(),
        expression: AstNode::new(
            Expr::Call(CallExpr {
                callee: Box::new(AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("is_allowed_domain"),
                    }),
                    span,
                    NodeId::new(2),
                )),
                arguments: vec![AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("email"),
                    }),
                    span,
                    NodeId::new(3),
                )],
                type_arguments: None,
                call_style: CallStyle::Function,
            }),
            span,
            NodeId::new(4),
        ),
    });
    
    let business_rule = TypeConstraint::BusinessRule(BusinessRuleConstraint {
        description: "Email must be verified before use".to_string(),
        expression: AstNode::new(
            Expr::Call(CallExpr {
                callee: Box::new(AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("is_verified"),
                    }),
                    span,
                    NodeId::new(5),
                )),
                arguments: vec![],
                type_arguments: None,
                call_style: CallStyle::Function,
            }),
            span,
            NodeId::new(6),
        ),
        priority: 2,
    });
    
    let metadata = SemanticTypeMetadata {
        business_rules: vec![
            "Must be unique per user".to_string(),
            "Must be verified".to_string(),
        ],
        examples: vec![
            "user@example.com".to_string(),
            "alice.smith@company.org".to_string(),
        ],
        validation_rules: vec![
            "RFC 5322 compliant".to_string(),
            "Domain must be whitelisted".to_string(),
        ],
        ai_context: Some("Email address for user communication and identification".to_string()),
        security_classification: SecurityClassification::Confidential,
        compliance_requirements: vec![
            "GDPR".to_string(),
            "CAN-SPAM".to_string(),
        ],
    };
    
    let email_type = SemanticType {
        base_type,
        constraints: vec![
            length_constraint,
            pattern_constraint,
            format_constraint,
            custom_constraint,
            business_rule,
        ],
        metadata,
    };
    
    assert_eq!(email_type.constraints.len(), 5);
    assert_eq!(email_type.metadata.business_rules.len(), 2);
    assert_eq!(email_type.metadata.examples.len(), 2);
    assert_eq!(email_type.metadata.validation_rules.len(), 2);
    assert_eq!(email_type.metadata.compliance_requirements.len(), 2);
    assert_eq!(email_type.metadata.security_classification, SecurityClassification::Confidential);
    
    // Verify each constraint type
    match &email_type.constraints[0] {
        TypeConstraint::Length(length) => {
            assert_eq!(length.min_length, Some(5));
            assert_eq!(length.max_length, Some(254));
        }
        _ => panic!("Expected length constraint"),
    }
    
    match &email_type.constraints[1] {
        TypeConstraint::Pattern(pattern) => {
            assert!(pattern.pattern.contains("@"));
        }
        _ => panic!("Expected pattern constraint"),
    }
    
    match &email_type.constraints[2] {
        TypeConstraint::Format(format) => {
            assert_eq!(format.format, "email");
            assert_eq!(format.parameters.len(), 1);
        }
        _ => panic!("Expected format constraint"),
    }
    
    match &email_type.constraints[3] {
        TypeConstraint::Custom(custom) => {
            assert_eq!(custom.name, "domain_whitelist");
        }
        _ => panic!("Expected custom constraint"),
    }
    
    match &email_type.constraints[4] {
        TypeConstraint::BusinessRule(rule) => {
            assert_eq!(rule.priority, 2);
            assert_eq!(rule.description, "Email must be verified before use");
        }
        _ => panic!("Expected business rule constraint"),
    }
}

#[test]
fn test_error_type() {
    let error_type = ErrorType {
        message: "Invalid type syntax".to_string(),
    };
    
    assert_eq!(error_type.message, "Invalid type syntax");
}

#[test]
fn test_visibility_variants() {
    assert_eq!(Visibility::default(), Visibility::Private);
    
    let public = Visibility::Public;
    let private = Visibility::Private;
    let internal = Visibility::Internal;
    
    assert_eq!(public, Visibility::Public);
    assert_eq!(private, Visibility::Private);
    assert_eq!(internal, Visibility::Internal);
}

#[test]
fn test_constraint_severity_and_validation_level() {
    let error_severity = ConstraintSeverity::Error;
    let warning_severity = ConstraintSeverity::Warning;
    let info_severity = ConstraintSeverity::Info;
    
    let compile_time_validation = ValidationLevel::CompileTime;
    let runtime_validation = ValidationLevel::Runtime;
    let both_validation = ValidationLevel::Both;
    
    assert_eq!(error_severity, ConstraintSeverity::Error);
    assert_eq!(warning_severity, ConstraintSeverity::Warning);
    assert_eq!(info_severity, ConstraintSeverity::Info);
    
    assert_eq!(compile_time_validation, ValidationLevel::CompileTime);
    assert_eq!(runtime_validation, ValidationLevel::Runtime);
    assert_eq!(both_validation, ValidationLevel::Both);
}

#[test]
fn test_parameter_roles() {
    let size_role = ParameterRole::Size;
    let index_role = ParameterRole::Index;
    let capacity_role = ParameterRole::Capacity;
    let precision_role = ParameterRole::Precision;
    let scale_role = ParameterRole::Scale;
    let custom_role = ParameterRole::Custom("timeout".to_string());
    
    assert_eq!(size_role, ParameterRole::Size);
    assert_eq!(index_role, ParameterRole::Index);
    assert_eq!(capacity_role, ParameterRole::Capacity);
    assert_eq!(precision_role, ParameterRole::Precision);
    assert_eq!(scale_role, ParameterRole::Scale);
    
    match custom_role {
        ParameterRole::Custom(name) => assert_eq!(name, "timeout"),
        _ => panic!("Expected custom role"),
    }
} 