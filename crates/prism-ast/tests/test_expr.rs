//! Comprehensive tests for the expression module

use prism_common::span::Position;
use prism_ast::{expr::*, node::*, types::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

#[test]
fn test_literal_expressions() {
    // Integer literal
    let int_lit = LiteralValue::Integer(42);
    assert_eq!(int_lit.to_string(), "42");
    
    // Float literal
    let float_lit = LiteralValue::Float(3.14);
    assert_eq!(float_lit.to_string(), "3.14");
    
    // String literal
    let string_lit = LiteralValue::String("hello".to_string());
    assert_eq!(string_lit.to_string(), "\"hello\"");
    
    // Boolean literal
    let bool_lit = LiteralValue::Boolean(true);
    assert_eq!(bool_lit.to_string(), "true");
    
    // Null literal
    let null_lit = LiteralValue::Null;
    assert_eq!(null_lit.to_string(), "null");
    
    // Money literal
    let money_lit = LiteralValue::Money {
        amount: 10.50,
        currency: "USD".to_string(),
    };
    assert_eq!(money_lit.to_string(), "10.5.USD");
    
    // Duration literal
    let duration_lit = LiteralValue::Duration {
        value: 5.0,
        unit: "minutes".to_string(),
    };
    assert_eq!(duration_lit.to_string(), "5.minutes");
    
    // Regex literal
    let regex_lit = LiteralValue::Regex(r"\d+".to_string());
    assert_eq!(regex_lit.to_string(), r#"r"\d+""#);
}

#[test]
fn test_binary_operators() {
    // Arithmetic operators
    assert_eq!(BinaryOperator::Add.to_string(), "+");
    assert_eq!(BinaryOperator::Subtract.to_string(), "-");
    assert_eq!(BinaryOperator::Multiply.to_string(), "*");
    assert_eq!(BinaryOperator::Divide.to_string(), "/");
    assert_eq!(BinaryOperator::Modulo.to_string(), "%");
    assert_eq!(BinaryOperator::Power.to_string(), "**");
    
    // Comparison operators
    assert_eq!(BinaryOperator::Equal.to_string(), "==");
    assert_eq!(BinaryOperator::NotEqual.to_string(), "!=");
    assert_eq!(BinaryOperator::Less.to_string(), "<");
    assert_eq!(BinaryOperator::LessEqual.to_string(), "<=");
    assert_eq!(BinaryOperator::Greater.to_string(), ">");
    assert_eq!(BinaryOperator::GreaterEqual.to_string(), ">=");
    
    // Logical operators
    assert_eq!(BinaryOperator::And.to_string(), "and");
    assert_eq!(BinaryOperator::Or.to_string(), "or");
    
    // Bitwise operators
    assert_eq!(BinaryOperator::BitAnd.to_string(), "&");
    assert_eq!(BinaryOperator::BitOr.to_string(), "|");
    assert_eq!(BinaryOperator::BitXor.to_string(), "^");
    assert_eq!(BinaryOperator::LeftShift.to_string(), "<<");
    assert_eq!(BinaryOperator::RightShift.to_string(), ">>");
    
    // Assignment operators
    assert_eq!(BinaryOperator::Assign.to_string(), "=");
    assert_eq!(BinaryOperator::AddAssign.to_string(), "+=");
    assert_eq!(BinaryOperator::SubtractAssign.to_string(), "-=");
    assert_eq!(BinaryOperator::MultiplyAssign.to_string(), "*=");
    assert_eq!(BinaryOperator::DivideAssign.to_string(), "/=");
    
    // Semantic operators (Prism-specific)
    assert_eq!(BinaryOperator::SemanticEqual.to_string(), "===");
    assert_eq!(BinaryOperator::TypeCompatible.to_string(), "~=");
    assert_eq!(BinaryOperator::ConceptualMatch.to_string(), "≈");
    
    // Range operators
    assert_eq!(BinaryOperator::Range.to_string(), "..");
    assert_eq!(BinaryOperator::RangeInclusive.to_string(), "..=");
}

#[test]
fn test_unary_operators() {
    assert_eq!(UnaryOperator::Not.to_string(), "not");
    assert_eq!(UnaryOperator::Negate.to_string(), "-");
    assert_eq!(UnaryOperator::BitNot.to_string(), "~");
    assert_eq!(UnaryOperator::Reference.to_string(), "&");
    assert_eq!(UnaryOperator::Dereference.to_string(), "*");
}

#[test]
fn test_binary_expression_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let left = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(10),
        }),
        span,
        NodeId::new(2),
    ));
    
    let right = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(20),
        }),
        span,
        NodeId::new(3),
    ));
    
    let binary_expr = BinaryExpr {
        left,
        operator: BinaryOperator::Add,
        right,
    };
    
    assert_eq!(binary_expr.operator, BinaryOperator::Add);
}

#[test]
fn test_unary_expression_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let operand = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        node_id,
    ));
    
    let unary_expr = UnaryExpr {
        operator: UnaryOperator::Negate,
        operand,
    };
    
    assert_eq!(unary_expr.operator, UnaryOperator::Negate);
}

#[test]
fn test_variable_expression() {
    let var_expr = VariableExpr {
        name: Symbol::intern("test_var"),
    };
    
    assert_eq!(var_expr.name, Symbol::intern("test_var"));
}

#[test]
fn test_call_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let callee = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("test_func"),
        }),
        span,
        node_id,
    ));
    
    let arg1 = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(2),
    );
    
    let call_expr = CallExpr {
        callee,
        arguments: vec![arg1],
        type_arguments: None,
        call_style: CallStyle::Function,
    };
    
    assert_eq!(call_expr.arguments.len(), 1);
    assert_eq!(call_expr.call_style, CallStyle::Function);
    assert!(call_expr.type_arguments.is_none());
}

#[test]
fn test_member_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let object = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("obj"),
        }),
        span,
        node_id,
    ));
    
    let member_expr = MemberExpr {
        object,
        member: Symbol::intern("field"),
        safe_navigation: false,
    };
    
    assert_eq!(member_expr.member, Symbol::intern("field"));
    assert!(!member_expr.safe_navigation);
}

#[test]
fn test_safe_navigation_member_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let object = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("obj"),
        }),
        span,
        node_id,
    ));
    
    let member_expr = MemberExpr {
        object,
        member: Symbol::intern("field"),
        safe_navigation: true,
    };
    
    assert!(member_expr.safe_navigation);
}

#[test]
fn test_index_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    let node_id = NodeId::new(1);
    
    let object = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("arr"),
        }),
        span,
        node_id,
    ));
    
    let index = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(0),
        }),
        span,
        NodeId::new(2),
    ));
    
    let index_expr = IndexExpr { object, index };
    
    // Verify the structure is correct
    if let Expr::Variable(var) = &index_expr.object.kind {
        assert_eq!(var.name, Symbol::intern("arr"));
    }
    
    if let Expr::Literal(lit) = &index_expr.index.kind {
        if let LiteralValue::Integer(n) = &lit.value {
            assert_eq!(*n, 0);
        }
    }
}

#[test]
fn test_array_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let elements = vec![
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1),
            }),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(2),
            }),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(3),
            }),
            span,
            NodeId::new(3),
        ),
    ];
    
    let array_expr = ArrayExpr { elements };
    
    assert_eq!(array_expr.elements.len(), 3);
}

#[test]
fn test_object_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let field1 = ObjectField {
        key: ObjectKey::Identifier(Symbol::intern("name")),
        value: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::String("test".to_string()),
            }),
            span,
            NodeId::new(1),
        ),
    };
    
    let field2 = ObjectField {
        key: ObjectKey::String("age".to_string()),
        value: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(25),
            }),
            span,
            NodeId::new(2),
        ),
    };
    
    let object_expr = ObjectExpr {
        fields: vec![field1, field2],
    };
    
    assert_eq!(object_expr.fields.len(), 2);
    
    // Test different key types
    match &object_expr.fields[0].key {
        ObjectKey::Identifier(sym) => assert_eq!(*sym, Symbol::intern("name")),
        _ => panic!("Expected identifier key"),
    }
    
    match &object_expr.fields[1].key {
        ObjectKey::String(s) => assert_eq!(s, "age"),
        _ => panic!("Expected string key"),
    }
}

#[test]
fn test_object_computed_key() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let computed_key = AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("key_var"),
        }),
        span,
        NodeId::new(1),
    );
    
    let field = ObjectField {
        key: ObjectKey::Computed(computed_key),
        value: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::String("value".to_string()),
            }),
            span,
            NodeId::new(2),
        ),
    };
    
    match &field.key {
        ObjectKey::Computed(expr) => {
            if let Expr::Variable(var) = &expr.kind {
                assert_eq!(var.name, Symbol::intern("key_var"));
            }
        }
        _ => panic!("Expected computed key"),
    }
}

#[test]
fn test_lambda_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param = Parameter {
        name: Symbol::intern("x"),
        type_annotation: None,
        default_value: None,
        is_mutable: false,
    };
    
    let body = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("x"),
        }),
        span,
        NodeId::new(1),
    ));
    
    let lambda = LambdaExpr {
        parameters: vec![param],
        return_type: None,
        body,
        is_async: false,
    };
    
    assert_eq!(lambda.parameters.len(), 1);
    assert!(!lambda.is_async);
    assert!(lambda.return_type.is_none());
    assert_eq!(lambda.parameters[0].name, Symbol::intern("x"));
    assert!(!lambda.parameters[0].is_mutable);
}

#[test]
fn test_async_lambda_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let body = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    ));
    
    let lambda = LambdaExpr {
        parameters: vec![],
        return_type: None,
        body,
        is_async: true,
    };
    
    assert!(lambda.is_async);
}

#[test]
fn test_if_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let condition = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Boolean(true),
        }),
        span,
        NodeId::new(1),
    ));
    
    let then_branch = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(2),
    ));
    
    let else_branch = Some(Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(0),
        }),
        span,
        NodeId::new(3),
    )));
    
    let if_expr = IfExpr {
        condition,
        then_branch,
        else_branch,
    };
    
    assert!(if_expr.else_branch.is_some());
}

#[test]
fn test_while_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let condition = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Boolean(true),
        }),
        span,
        NodeId::new(1),
    ));
    
    let body = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(2),
    ));
    
    let while_expr = WhileExpr { condition, body };
    
    if let Expr::Literal(lit) = &while_expr.condition.kind {
        if let LiteralValue::Boolean(b) = &lit.value {
            assert!(*b);
        }
    }
}

#[test]
fn test_for_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let iterable = Box::new(AstNode::new(
        Expr::Array(ArrayExpr { elements: vec![] }),
        span,
        NodeId::new(1),
    ));
    
    let body = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(2),
    ));
    
    let for_expr = ForExpr {
        variable: Symbol::intern("item"),
        iterable,
        body,
    };
    
    assert_eq!(for_expr.variable, Symbol::intern("item"));
}

#[test]
fn test_range_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let start = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(1),
        }),
        span,
        NodeId::new(1),
    ));
    
    let end = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(10),
        }),
        span,
        NodeId::new(2),
    ));
    
    let range_expr = RangeExpr {
        start,
        end,
        inclusive: true,
    };
    
    assert!(range_expr.inclusive);
}

#[test]
fn test_tuple_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let elements = vec![
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1),
            }),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::String("test".to_string()),
            }),
            span,
            NodeId::new(2),
        ),
    ];
    
    let tuple_expr = TupleExpr { elements };
    
    assert_eq!(tuple_expr.elements.len(), 2);
}

#[test]
fn test_await_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expression = Box::new(AstNode::new(
        Expr::Call(CallExpr {
            callee: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("async_func"),
                }),
                span,
                NodeId::new(1),
            )),
            arguments: vec![],
            type_arguments: None,
            call_style: CallStyle::Function,
        }),
        span,
        NodeId::new(2),
    ));
    
    let await_expr = AwaitExpr { expression };
    
    if let Expr::Call(call) = &await_expr.expression.kind {
        if let Expr::Variable(var) = &call.callee.kind {
            assert_eq!(var.name, Symbol::intern("async_func"));
        }
    }
}

#[test]
fn test_yield_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let value = Some(Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    )));
    
    let yield_expr = YieldExpr { value };
    
    assert!(yield_expr.value.is_some());
}

#[test]
fn test_type_assertion_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expression = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("value"),
        }),
        span,
        NodeId::new(1),
    ));
    
    let target_type = Box::new(AstNode::new(
        Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
        span,
        NodeId::new(2),
    ));
    
    let type_assertion = TypeAssertionExpr {
        expression,
        target_type,
    };
    
    if let Expr::Variable(var) = &type_assertion.expression.kind {
        assert_eq!(var.name, Symbol::intern("value"));
    }
}

#[test]
fn test_return_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let value = Some(Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    )));
    
    let return_expr = ReturnExpr { value };
    
    assert!(return_expr.value.is_some());
}

#[test]
fn test_break_and_continue_expressions() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let break_value = Some(Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(1),
        }),
        span,
        NodeId::new(1),
    )));
    
    let continue_value = Some(Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(2),
        }),
        span,
        NodeId::new(2),
    )));
    
    let break_expr = BreakExpr { value: break_value };
    let continue_expr = ContinueExpr { value: continue_value };
    
    assert!(break_expr.value.is_some());
    assert!(continue_expr.value.is_some());
}

#[test]
fn test_throw_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let exception = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::String("Error message".to_string()),
        }),
        span,
        NodeId::new(1),
    ));
    
    let throw_expr = ThrowExpr { exception };
    
    if let Expr::Literal(lit) = &throw_expr.exception.kind {
        if let LiteralValue::String(s) = &lit.value {
            assert_eq!(s, "Error message");
        }
    }
}

#[test]
fn test_error_expression() {
    let error_expr = ErrorExpr {
        message: "Parse error".to_string(),
    };
    
    assert_eq!(error_expr.message, "Parse error");
}

#[test]
fn test_expression_node_kind_traits() {
    // Test literal expression
    let literal = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    
    assert_eq!(literal.node_kind_name(), "Literal");
    assert_eq!(literal.semantic_domain(), Some("Data"));
    assert!(!literal.is_side_effectful());
    assert_eq!(literal.computational_complexity(), ComplexityClass::Constant);
    
    // Test side-effectful expression
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let call = Expr::Call(CallExpr {
        callee: Box::new(AstNode::new(
            Expr::Variable(VariableExpr {
                name: Symbol::intern("func"),
            }),
            span,
            NodeId::new(1),
        )),
        arguments: vec![],
        type_arguments: None,
        call_style: CallStyle::Function,
    });
    
    assert!(call.is_side_effectful());
    assert_eq!(call.computational_complexity(), ComplexityClass::Unknown);
    
    // Test assignment expression
    let left = Box::new(AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("x"),
        }),
        span,
        NodeId::new(1),
    ));
    
    let right = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(2),
    ));
    
    let assign = Expr::Binary(BinaryExpr {
        left,
        operator: BinaryOperator::Assign,
        right,
    });
    
    assert!(assign.is_side_effectful());
}

#[test]
fn test_expression_complexity_analysis() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Large array should be linear complexity
    let large_elements: Vec<_> = (0..200)
        .map(|i| {
            AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(i),
                }),
                span,
                NodeId::new(i as u32 + 1),
            )
        })
        .collect();
    
    let large_array = Expr::Array(ArrayExpr {
        elements: large_elements,
    });
    
    assert_eq!(large_array.computational_complexity(), ComplexityClass::Linear);
    
    // Small array should be constant complexity
    let small_array = Expr::Array(ArrayExpr {
        elements: vec![AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1),
            }),
            span,
            NodeId::new(1),
        )],
    });
    
    assert_eq!(small_array.computational_complexity(), ComplexityClass::Constant);
}

#[test]
fn test_call_style_variants() {
    assert_eq!(CallStyle::Function, CallStyle::Function);
    assert_eq!(CallStyle::Method, CallStyle::Method);
    assert_eq!(CallStyle::Constructor, CallStyle::Constructor);
    assert_eq!(CallStyle::Operator, CallStyle::Operator);
    assert_eq!(CallStyle::Capability, CallStyle::Capability);
}

#[test]
fn test_parameter_with_default_value() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let default_value = Some(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    ));
    
    let param = Parameter {
        name: Symbol::intern("x"),
        type_annotation: None,
        default_value,
        is_mutable: true,
    };
    
    assert!(param.default_value.is_some());
    assert!(param.is_mutable);
    
    if let Some(default) = &param.default_value {
        if let Expr::Literal(lit) = &default.kind {
            if let LiteralValue::Integer(n) = &lit.value {
                assert_eq!(*n, 42);
            }
        }
    }
} 