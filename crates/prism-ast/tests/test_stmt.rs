//! Comprehensive tests for the statement module

use prism_common::span::Position;
use prism_ast::{stmt::*, expr::*, types::*, pattern::*, node::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

#[test]
fn test_expression_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expression = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    );
    
    let expr_stmt = ExpressionStmt { expression };
    
    if let Expr::Literal(lit) = &expr_stmt.expression.kind {
        if let LiteralValue::Integer(n) = &lit.value {
            assert_eq!(*n, 42);
        }
    }
}

#[test]
fn test_variable_declaration() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let initializer = Some(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    ));
    
    let type_annotation = Some(AstNode::new(
        Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
        span,
        NodeId::new(2),
    ));
    
    let var_decl = VariableDecl {
        name: Symbol::intern("x"),
        type_annotation,
        initializer,
        is_mutable: true,
        visibility: Visibility::Private,
    };
    
    assert_eq!(var_decl.name, Symbol::intern("x"));
    assert!(var_decl.is_mutable);
    assert_eq!(var_decl.visibility, Visibility::Private);
    assert!(var_decl.type_annotation.is_some());
    assert!(var_decl.initializer.is_some());
}

#[test]
fn test_function_declaration() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param = stmt::Parameter {
        name: Symbol::intern("x"),
        type_annotation: Some(AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(1),
        )),
        default_value: None,
        is_mutable: false,
    };
    
    let return_type = Some(AstNode::new(
        Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
        span,
        NodeId::new(2),
    ));
    
    let body = Some(Box::new(AstNode::new(
        Stmt::Return(ReturnStmt {
            value: Some(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("x"),
                }),
                span,
                NodeId::new(3),
            )),
        }),
        span,
        NodeId::new(4),
    )));
    
    let func_decl = FunctionDecl {
        name: Symbol::intern("identity"),
        parameters: vec![param],
        return_type,
        body,
        visibility: Visibility::Public,
        attributes: vec![],
        contracts: None,
        is_async: false,
    };
    
    assert_eq!(func_decl.name, Symbol::intern("identity"));
    assert_eq!(func_decl.parameters.len(), 1);
    assert!(func_decl.return_type.is_some());
    assert!(func_decl.body.is_some());
    assert_eq!(func_decl.visibility, Visibility::Public);
    assert!(!func_decl.is_async);
    assert!(func_decl.contracts.is_none());
}

#[test]
fn test_async_function_declaration() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let func_decl = FunctionDecl {
        name: Symbol::intern("async_func"),
        parameters: vec![],
        return_type: None,
        body: None,
        visibility: Visibility::Public,
        attributes: vec![],
        contracts: None,
        is_async: true,
    };
    
    assert!(func_decl.is_async);
}

#[test]
fn test_function_with_contracts() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let requires = vec![AstNode::new(
        Expr::Binary(BinaryExpr {
            left: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("x"),
                }),
                span,
                NodeId::new(1),
            )),
            operator: BinaryOperator::Greater,
            right: Box::new(AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(0),
                }),
                span,
                NodeId::new(2),
            )),
        }),
        span,
        NodeId::new(3),
    )];
    
    let ensures = vec![AstNode::new(
        Expr::Binary(BinaryExpr {
            left: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("result"),
                }),
                span,
                NodeId::new(4),
            )),
            operator: BinaryOperator::GreaterEqual,
            right: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("x"),
                }),
                span,
                NodeId::new(5),
            )),
        }),
        span,
        NodeId::new(6),
    )];
    
    let contracts = Some(Contracts {
        requires,
        ensures,
        invariants: vec![],
    });
    
    let func_decl = FunctionDecl {
        name: Symbol::intern("increment"),
        parameters: vec![],
        return_type: None,
        body: None,
        visibility: Visibility::Public,
        attributes: vec![],
        contracts,
        is_async: false,
    };
    
    assert!(func_decl.contracts.is_some());
    
    if let Some(contracts) = &func_decl.contracts {
        assert_eq!(contracts.requires.len(), 1);
        assert_eq!(contracts.ensures.len(), 1);
        assert_eq!(contracts.invariants.len(), 0);
    }
}

#[test]
fn test_module_declaration() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let section = AstNode::new(
        SectionDecl {
            kind: SectionKind::Interface,
            items: vec![],
            visibility: Visibility::Public,
        },
        span,
        NodeId::new(1),
    );
    
    let module_decl = ModuleDecl {
        name: Symbol::intern("auth"),
        capability: Some("authentication".to_string()),
        description: Some("User authentication module".to_string()),
        dependencies: vec!["crypto".to_string(), "database".to_string()],
        stability: StabilityLevel::Stable,
        version: Some("1.0.0".to_string()),
        sections: vec![section],
        ai_context: Some("Handles user login and session management".to_string()),
        visibility: Visibility::Public,
    };
    
    assert_eq!(module_decl.name, Symbol::intern("auth"));
    assert_eq!(module_decl.capability, Some("authentication".to_string()));
    assert_eq!(module_decl.description, Some("User authentication module".to_string()));
    assert_eq!(module_decl.dependencies.len(), 2);
    assert_eq!(module_decl.stability, StabilityLevel::Stable);
    assert_eq!(module_decl.version, Some("1.0.0".to_string()));
    assert_eq!(module_decl.sections.len(), 1);
    assert!(module_decl.ai_context.is_some());
    assert_eq!(module_decl.visibility, Visibility::Public);
}

#[test]
fn test_stability_levels() {
    assert_eq!(StabilityLevel::default(), StabilityLevel::Experimental);
    
    let experimental = StabilityLevel::Experimental;
    let stable = StabilityLevel::Stable;
    let deprecated = StabilityLevel::Deprecated;
    
    assert_eq!(experimental, StabilityLevel::Experimental);
    assert_eq!(stable, StabilityLevel::Stable);
    assert_eq!(deprecated, StabilityLevel::Deprecated);
}

#[test]
fn test_section_declarations() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let config_section = SectionDecl {
        kind: SectionKind::Config,
        items: vec![],
        visibility: Visibility::Internal,
    };
    
    let types_section = SectionDecl {
        kind: SectionKind::Types,
        items: vec![],
        visibility: Visibility::Public,
    };
    
    let errors_section = SectionDecl {
        kind: SectionKind::Errors,
        items: vec![],
        visibility: Visibility::Public,
    };
    
    let custom_section = SectionDecl {
        kind: SectionKind::Custom("Utilities".to_string()),
        items: vec![],
        visibility: Visibility::Private,
    };
    
    assert_eq!(config_section.visibility, Visibility::Internal);
    assert_eq!(types_section.visibility, Visibility::Public);
    assert_eq!(errors_section.visibility, Visibility::Public);
    assert_eq!(custom_section.visibility, Visibility::Private);
    
    match &custom_section.kind {
        SectionKind::Custom(name) => assert_eq!(name, "Utilities"),
        _ => panic!("Expected custom section"),
    }
}

#[test]
fn test_import_declaration() {
    let import_item1 = ImportItem {
        name: Symbol::intern("authenticate"),
        alias: Some(Symbol::intern("auth")),
    };
    
    let import_item2 = ImportItem {
        name: Symbol::intern("hash_password"),
        alias: None,
    };
    
    let import_decl = ImportDecl {
        path: "auth/security".to_string(),
        items: ImportItems::Specific(vec![import_item1, import_item2]),
        alias: None,
    };
    
    assert_eq!(import_decl.path, "auth/security");
    
    match &import_decl.items {
        ImportItems::Specific(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].name, Symbol::intern("authenticate"));
            assert_eq!(items[0].alias, Some(Symbol::intern("auth")));
            assert_eq!(items[1].name, Symbol::intern("hash_password"));
            assert_eq!(items[1].alias, None);
        }
        _ => panic!("Expected specific imports"),
    }
}

#[test]
fn test_import_all() {
    let import_decl = ImportDecl {
        path: "std/collections".to_string(),
        items: ImportItems::All,
        alias: Some(Symbol::intern("collections")),
    };
    
    match &import_decl.items {
        ImportItems::All => {
            // Expected
        }
        _ => panic!("Expected import all"),
    }
    
    assert_eq!(import_decl.alias, Some(Symbol::intern("collections")));
}

#[test]
fn test_export_declaration() {
    let export_item1 = ExportItem {
        name: Symbol::intern("User"),
        alias: None,
    };
    
    let export_item2 = ExportItem {
        name: Symbol::intern("authenticate_user"),
        alias: Some(Symbol::intern("authenticate")),
    };
    
    let export_decl = ExportDecl {
        items: vec![export_item1, export_item2],
    };
    
    assert_eq!(export_decl.items.len(), 2);
    assert_eq!(export_decl.items[0].name, Symbol::intern("User"));
    assert_eq!(export_decl.items[0].alias, None);
    assert_eq!(export_decl.items[1].name, Symbol::intern("authenticate_user"));
    assert_eq!(export_decl.items[1].alias, Some(Symbol::intern("authenticate")));
}

#[test]
fn test_constant_declaration() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let const_decl = ConstDecl {
        name: Symbol::intern("MAX_USERS"),
        type_annotation: Some(AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(32))),
            span,
            NodeId::new(1),
        )),
        value: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1000),
            }),
            span,
            NodeId::new(2),
        ),
        visibility: Visibility::Public,
    };
    
    assert_eq!(const_decl.name, Symbol::intern("MAX_USERS"));
    assert!(const_decl.type_annotation.is_some());
    assert_eq!(const_decl.visibility, Visibility::Public);
    
    if let Expr::Literal(lit) = &const_decl.value.kind {
        if let LiteralValue::Integer(n) = &lit.value {
            assert_eq!(*n, 1000);
        }
    }
}

#[test]
fn test_if_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let condition = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Boolean(true),
        }),
        span,
        NodeId::new(1),
    );
    
    let then_branch = Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(1),
                }),
                span,
                NodeId::new(2),
            ),
        }),
        span,
        NodeId::new(3),
    ));
    
    let else_branch = Some(Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(0),
                }),
                span,
                NodeId::new(4),
            ),
        }),
        span,
        NodeId::new(5),
    )));
    
    let if_stmt = IfStmt {
        condition,
        then_branch,
        else_branch,
    };
    
    assert!(if_stmt.else_branch.is_some());
    
    if let Expr::Literal(lit) = &if_stmt.condition.kind {
        if let LiteralValue::Boolean(b) = &lit.value {
            assert!(*b);
        }
    }
}

#[test]
fn test_while_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let condition = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Boolean(true),
        }),
        span,
        NodeId::new(1),
    );
    
    let body = Box::new(AstNode::new(
        Stmt::Break(BreakStmt { value: None }),
        span,
        NodeId::new(2),
    ));
    
    let while_stmt = WhileStmt { condition, body };
    
    if let Expr::Literal(lit) = &while_stmt.condition.kind {
        if let LiteralValue::Boolean(b) = &lit.value {
            assert!(*b);
        }
    }
    
    if let Stmt::Break(break_stmt) = &while_stmt.body.kind {
        assert!(break_stmt.value.is_none());
    }
}

#[test]
fn test_for_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let iterable = AstNode::new(
        Expr::Array(ArrayExpr {
            elements: vec![
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
            ],
        }),
        span,
        NodeId::new(3),
    );
    
    let body = Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("item"),
                }),
                span,
                NodeId::new(4),
            ),
        }),
        span,
        NodeId::new(5),
    ));
    
    let for_stmt = ForStmt {
        variable: Symbol::intern("item"),
        iterable,
        body,
    };
    
    assert_eq!(for_stmt.variable, Symbol::intern("item"));
    
    if let Expr::Array(arr) = &for_stmt.iterable.kind {
        assert_eq!(arr.elements.len(), 2);
    }
}

#[test]
fn test_match_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expression = AstNode::new(
        Expr::Variable(VariableExpr {
            name: Symbol::intern("value"),
        }),
        span,
        NodeId::new(1),
    );
    
    let pattern1 = AstNode::new(
        Pattern::Literal(LiteralValue::Integer(1)),
        span,
        NodeId::new(2),
    );
    
    let body1 = Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::String("one".to_string()),
                }),
                span,
                NodeId::new(3),
            ),
        }),
        span,
        NodeId::new(4),
    ));
    
    let arm1 = stmt::MatchArm {
        pattern: pattern1,
        guard: None,
        body: body1,
    };
    
    let pattern2 = AstNode::new(
        Pattern::Wildcard,
        span,
        NodeId::new(5),
    );
    
    let body2 = Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::String("other".to_string()),
                }),
                span,
                NodeId::new(6),
            ),
        }),
        span,
        NodeId::new(7),
    ));
    
    let arm2 = stmt::MatchArm {
        pattern: pattern2,
        guard: None,
        body: body2,
    };
    
    let match_stmt = MatchStmt {
        expression,
        arms: vec![arm1, arm2],
    };
    
    assert_eq!(match_stmt.arms.len(), 2);
    
    if let Expr::Variable(var) = &match_stmt.expression.kind {
        assert_eq!(var.name, Symbol::intern("value"));
    }
    
    if let Pattern::Literal(LiteralValue::Integer(n)) = &match_stmt.arms[0].pattern.kind {
        assert_eq!(*n, 1);
    }
    
    if let Pattern::Wildcard = &match_stmt.arms[1].pattern.kind {
        // Expected
    } else {
        panic!("Expected wildcard pattern");
    }
}

#[test]
fn test_match_arm_with_guard() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let pattern = AstNode::new(
        Pattern::Identifier(Symbol::intern("x")),
        span,
        NodeId::new(1),
    );
    
    let guard = Some(AstNode::new(
        Expr::Binary(BinaryExpr {
            left: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("x"),
                }),
                span,
                NodeId::new(2),
            )),
            operator: BinaryOperator::Greater,
            right: Box::new(AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(0),
                }),
                span,
                NodeId::new(3),
            )),
        }),
        span,
        NodeId::new(4),
    ));
    
    let body = Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::String("positive".to_string()),
                }),
                span,
                NodeId::new(5),
            ),
        }),
        span,
        NodeId::new(6),
    ));
    
    let arm = stmt::MatchArm {
        pattern,
        guard,
        body,
    };
    
    assert!(arm.guard.is_some());
    
    if let Pattern::Identifier(sym) = &arm.pattern.kind {
        assert_eq!(sym, &Symbol::intern("x"));
    }
}

#[test]
fn test_return_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let return_with_value = ReturnStmt {
        value: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(42),
            }),
            span,
            NodeId::new(1),
        )),
    };
    
    let return_without_value = ReturnStmt { value: None };
    
    assert!(return_with_value.value.is_some());
    assert!(return_without_value.value.is_none());
    
    if let Some(value) = &return_with_value.value {
        if let Expr::Literal(lit) = &value.kind {
            if let LiteralValue::Integer(n) = &lit.value {
                assert_eq!(*n, 42);
            }
        }
    }
}

#[test]
fn test_break_and_continue_statements() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let break_stmt = BreakStmt {
        value: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1),
            }),
            span,
            NodeId::new(1),
        )),
    };
    
    let continue_stmt = ContinueStmt {
        value: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(2),
            }),
            span,
            NodeId::new(2),
        )),
    };
    
    assert!(break_stmt.value.is_some());
    assert!(continue_stmt.value.is_some());
}

#[test]
fn test_throw_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let throw_stmt = ThrowStmt {
        exception: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::String("Error occurred".to_string()),
            }),
            span,
            NodeId::new(1),
        ),
    };
    
    if let Expr::Literal(lit) = &throw_stmt.exception.kind {
        if let LiteralValue::String(s) = &lit.value {
            assert_eq!(s, "Error occurred");
        }
    }
}

#[test]
fn test_try_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let try_block = Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Call(CallExpr {
                    callee: Box::new(AstNode::new(
                        Expr::Variable(VariableExpr {
                            name: Symbol::intern("risky_function"),
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
            ),
        }),
        span,
        NodeId::new(3),
    ));
    
    let catch_clause = stmt::CatchClause {
        variable: Some(Symbol::intern("e")),
        error_type: None,
        body: Box::new(AstNode::new(
            Stmt::Expression(ExpressionStmt {
                expression: AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::String("Error handled".to_string()),
                    }),
                    span,
                    NodeId::new(4),
                ),
            }),
            span,
            NodeId::new(5),
        )),
    };
    
    let finally_block = Some(Box::new(AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::String("Cleanup".to_string()),
                }),
                span,
                NodeId::new(6),
            ),
        }),
        span,
        NodeId::new(7),
    )));
    
    let try_stmt = TryStmt {
        try_block,
        catch_clauses: vec![catch_clause],
        finally_block,
    };
    
    assert_eq!(try_stmt.catch_clauses.len(), 1);
    assert!(try_stmt.finally_block.is_some());
    assert_eq!(try_stmt.catch_clauses[0].variable, Some(Symbol::intern("e")));
}

#[test]
fn test_block_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let stmt1 = AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(1),
                }),
                span,
                NodeId::new(1),
            ),
        }),
        span,
        NodeId::new(2),
    );
    
    let stmt2 = AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(2),
                }),
                span,
                NodeId::new(3),
            ),
        }),
        span,
        NodeId::new(4),
    );
    
    let block_stmt = BlockStmt {
        statements: vec![stmt1, stmt2],
    };
    
    assert_eq!(block_stmt.statements.len(), 2);
}

#[test]
fn test_attributes() {
    let literal_arg = AttributeArgument::Literal(LiteralValue::String("test".to_string()));
    let named_arg = AttributeArgument::Named {
        name: "version".to_string(),
        value: LiteralValue::String("1.0".to_string()),
    };
    
    let attribute = Attribute {
        name: "deprecated".to_string(),
        arguments: vec![literal_arg, named_arg],
    };
    
    assert_eq!(attribute.name, "deprecated");
    assert_eq!(attribute.arguments.len(), 2);
    
    match &attribute.arguments[0] {
        AttributeArgument::Literal(LiteralValue::String(s)) => {
            assert_eq!(s, "test");
        }
        _ => panic!("Expected literal argument"),
    }
    
    match &attribute.arguments[1] {
        AttributeArgument::Named { name, value } => {
            assert_eq!(name, "version");
            if let LiteralValue::String(s) = value {
                assert_eq!(s, "1.0");
            }
        }
        _ => panic!("Expected named argument"),
    }
}

#[test]
fn test_error_statement() {
    let error_stmt = ErrorStmt {
        message: "Parse error: unexpected token".to_string(),
        context: "function declaration".to_string(),
    };
    
    assert_eq!(error_stmt.message, "Parse error: unexpected token");
    assert_eq!(error_stmt.context, "function declaration");
}

#[test]
fn test_visibility_default() {
    let default_visibility = Visibility::default();
    assert_eq!(default_visibility, Visibility::Private);
}

#[test]
fn test_parameter_with_defaults() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param = stmt::Parameter {
        name: Symbol::intern("count"),
        type_annotation: Some(AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(32))),
            span,
            NodeId::new(1),
        )),
        default_value: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(10),
            }),
            span,
            NodeId::new(2),
        )),
        is_mutable: false,
    };
    
    assert_eq!(param.name, Symbol::intern("count"));
    assert!(param.type_annotation.is_some());
    assert!(param.default_value.is_some());
    assert!(!param.is_mutable);
}

#[test]
fn test_complex_module_structure() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create a complex module with multiple sections
    let config_section = AstNode::new(
        SectionDecl {
            kind: SectionKind::Config,
            items: vec![AstNode::new(
                Stmt::Const(ConstDecl {
                    name: Symbol::intern("DEFAULT_TIMEOUT"),
                    type_annotation: None,
                    value: AstNode::new(
                        Expr::Literal(LiteralExpr {
                            value: LiteralValue::Integer(30),
                        }),
                        span,
                        NodeId::new(1),
                    ),
                    visibility: Visibility::Internal,
                }),
                span,
                NodeId::new(2),
            )],
            visibility: Visibility::Internal,
        },
        span,
        NodeId::new(3),
    );
    
    let interface_section = AstNode::new(
        SectionDecl {
            kind: SectionKind::Interface,
            items: vec![AstNode::new(
                Stmt::Function(FunctionDecl {
                    name: Symbol::intern("authenticate"),
                    parameters: vec![],
                    return_type: None,
                    body: None,
                    visibility: Visibility::Public,
                    attributes: vec![],
                    contracts: None,
                    is_async: false,
                }),
                span,
                NodeId::new(4),
            )],
            visibility: Visibility::Public,
        },
        span,
        NodeId::new(5),
    );
    
    let module_decl = ModuleDecl {
        name: Symbol::intern("authentication"),
        capability: Some("user_authentication".to_string()),
        description: Some("Comprehensive user authentication system".to_string()),
        dependencies: vec![
            "crypto".to_string(),
            "database".to_string(),
            "logging".to_string(),
        ],
        stability: StabilityLevel::Stable,
        version: Some("2.1.0".to_string()),
        sections: vec![config_section, interface_section],
        ai_context: Some("This module handles all aspects of user authentication including login, logout, session management, and security policies".to_string()),
        visibility: Visibility::Public,
    };
    
    assert_eq!(module_decl.name, Symbol::intern("authentication"));
    assert_eq!(module_decl.capability, Some("user_authentication".to_string()));
    assert_eq!(module_decl.dependencies.len(), 3);
    assert_eq!(module_decl.stability, StabilityLevel::Stable);
    assert_eq!(module_decl.sections.len(), 2);
    assert!(module_decl.ai_context.is_some());
    
    // Verify section contents
    if let SectionKind::Config = &module_decl.sections[0].kind.kind {
        assert_eq!(module_decl.sections[0].kind.items.len(), 1);
    }
    
    if let SectionKind::Interface = &module_decl.sections[1].kind.kind {
        assert_eq!(module_decl.sections[1].kind.items.len(), 1);
    }
} 