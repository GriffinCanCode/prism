//! Comprehensive tests for the lib module (main AST functionality)

use prism_common::span::Position;
use prism_ast::{*, stmt::Parameter};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};
use std::collections::HashMap;

#[test]
fn test_program_creation() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
    
    let items = vec![
        AstNode::new(
            Item::Function(FunctionDecl {
                name: Symbol::intern("main"),
                parameters: vec![],
                return_type: None,
                body: Some(Box::new(AstNode::new(
                    Stmt::Expression(ExpressionStmt {
                        expression: AstNode::new(
                            Expr::Literal(LiteralExpr {
                                value: LiteralValue::String("Hello, World!".to_string()),
                            }),
                            span,
                            NodeId::new(1),
                        ),
                    }),
                    span,
                    NodeId::new(2),
                ))),
                visibility: Visibility::Public,
                attributes: vec![],
                contracts: None,
                is_async: false,
            }),
            span,
            NodeId::new(3),
        ),
    ];
    
    let program = Program::new(items, source_id);
    
    assert_eq!(program.items.len(), 1);
    assert_eq!(program.source_id, source_id);
    assert!(program.metadata.primary_capability.is_none());
    assert!(program.metadata.capabilities.is_empty());
    assert!(program.metadata.dependencies.is_empty());
    assert!(program.metadata.security_implications.is_empty());
    assert!(program.metadata.performance_notes.is_empty());
    assert!(program.metadata.ai_insights.is_empty());
}

#[test]
fn test_program_with_metadata() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
    
    let metadata = ProgramMetadata {
        primary_capability: Some("web_server".to_string()),
        capabilities: vec![
            "http_handling".to_string(),
            "database_access".to_string(),
        ],
        dependencies: vec![
            "http".to_string(),
            "database".to_string(),
        ],
        security_implications: vec![
            "Handles user input".to_string(),
            "Database queries".to_string(),
        ],
        performance_notes: vec![
            "Uses connection pooling".to_string(),
            "Caches responses".to_string(),
        ],
        ai_insights: vec![
            "Well-structured HTTP server".to_string(),
            "Good separation of concerns".to_string(),
        ],
    };
    
    let program = Program::new(vec![], source_id).with_metadata(metadata);
    
    assert_eq!(program.metadata.primary_capability, Some("web_server".to_string()));
    assert_eq!(program.metadata.capabilities.len(), 2);
    assert_eq!(program.metadata.dependencies.len(), 2);
    assert_eq!(program.metadata.security_implications.len(), 2);
    assert_eq!(program.metadata.performance_notes.len(), 2);
    assert_eq!(program.metadata.ai_insights.len(), 2);
}

#[test]
fn test_item_variants() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Module item
    let module_item = Item::Module(ModuleDecl {
        name: Symbol::intern("auth"),
        capability: Some("authentication".to_string()),
        description: Some("Authentication module".to_string()),
        dependencies: vec!["crypto".to_string()],
        stability: StabilityLevel::Stable,
        version: Some("1.0.0".to_string()),
        sections: vec![],
        ai_context: Some("Handles user authentication".to_string()),
        visibility: Visibility::Public,
    });
    
    // Function item
    let function_item = Item::Function(FunctionDecl {
        name: Symbol::intern("authenticate"),
        parameters: vec![],
        return_type: None,
        body: None,
        visibility: Visibility::Public,
        attributes: vec![],
        contracts: None,
        is_async: false,
    });
    
    // Type item
    let type_item = Item::Type(TypeDecl {
        name: Symbol::intern("User"),
        type_parameters: vec![],
        kind: TypeKind::Struct(StructType {
            fields: vec![
                StructField {
                    name: Symbol::intern("id"),
                    field_type: AstNode::new(
                        Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(64))),
                        span,
                        NodeId::new(1),
                    ),
                    visibility: Visibility::Public,
                },
            ],
        }),
        visibility: Visibility::Public,
    });
    
    // Import item
    let import_item = Item::Import(ImportDecl {
        path: "std/collections".to_string(),
        items: ImportItems::All,
        alias: None,
    });
    
    // Export item
    let export_item = Item::Export(ExportDecl {
        items: vec![ExportItem {
            name: Symbol::intern("User"),
            alias: None,
        }],
    });
    
    // Constant item
    let const_item = Item::Const(ConstDecl {
        name: Symbol::intern("MAX_USERS"),
        type_annotation: None,
        value: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1000),
            }),
            span,
            NodeId::new(2),
        ),
        visibility: Visibility::Public,
    });
    
    // Variable item
    let variable_item = Item::Variable(VariableDecl {
        name: Symbol::intern("global_counter"),
        type_annotation: None,
        initializer: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(0),
            }),
            span,
            NodeId::new(3),
        )),
        is_mutable: true,
        visibility: Visibility::Private,
    });
    
    // Statement item
    let statement_item = Item::Statement(Stmt::Expression(ExpressionStmt {
        expression: AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::String("initialization".to_string()),
            }),
            span,
            NodeId::new(4),
        ),
    }));
    
    // Test all variants can be created
    match module_item {
        Item::Module(module) => {
            assert_eq!(module.name, Symbol::intern("auth"));
        }
        _ => panic!("Expected module item"),
    }
    
    match function_item {
        Item::Function(func) => {
            assert_eq!(func.name, Symbol::intern("authenticate"));
        }
        _ => panic!("Expected function item"),
    }
    
    match type_item {
        Item::Type(type_decl) => {
            assert_eq!(type_decl.name, Symbol::intern("User"));
        }
        _ => panic!("Expected type item"),
    }
    
    match import_item {
        Item::Import(import) => {
            assert_eq!(import.path, "std/collections");
        }
        _ => panic!("Expected import item"),
    }
    
    match export_item {
        Item::Export(export) => {
            assert_eq!(export.items.len(), 1);
        }
        _ => panic!("Expected export item"),
    }
    
    match const_item {
        Item::Const(const_decl) => {
            assert_eq!(const_decl.name, Symbol::intern("MAX_USERS"));
        }
        _ => panic!("Expected const item"),
    }
    
    match variable_item {
        Item::Variable(var_decl) => {
            assert_eq!(var_decl.name, Symbol::intern("global_counter"));
            assert!(var_decl.is_mutable);
        }
        _ => panic!("Expected variable item"),
    }
    
    match statement_item {
        Item::Statement(stmt) => {
            match stmt {
                Stmt::Expression(_) => {}, // Expected
                _ => panic!("Expected expression statement"),
            }
        }
        _ => panic!("Expected statement item"),
    }
}

#[test]
fn test_program_iterators() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
    
    let items = vec![
        // Module
        AstNode::new(
            Item::Module(ModuleDecl {
                name: Symbol::intern("auth"),
                capability: Some("authentication".to_string()),
                description: None,
                dependencies: vec![],
                stability: StabilityLevel::Stable,
                version: None,
                sections: vec![],
                ai_context: None,
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(1),
        ),
        // Function
        AstNode::new(
            Item::Function(FunctionDecl {
                name: Symbol::intern("login"),
                parameters: vec![],
                return_type: None,
                body: None,
                visibility: Visibility::Public,
                attributes: vec![],
                contracts: None,
                is_async: false,
            }),
            span,
            NodeId::new(2),
        ),
        // Another function
        AstNode::new(
            Item::Function(FunctionDecl {
                name: Symbol::intern("logout"),
                parameters: vec![],
                return_type: None,
                body: None,
                visibility: Visibility::Public,
                attributes: vec![],
                contracts: None,
                is_async: false,
            }),
            span,
            NodeId::new(3),
        ),
        // Type
        AstNode::new(
            Item::Type(TypeDecl {
                name: Symbol::intern("User"),
                type_parameters: vec![],
                kind: TypeKind::Struct(StructType {
                    fields: vec![],
                }),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(4),
        ),
        // Another type
        AstNode::new(
            Item::Type(TypeDecl {
                name: Symbol::intern("Session"),
                type_parameters: vec![],
                kind: TypeKind::Struct(StructType {
                    fields: vec![],
                }),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(5),
        ),
    ];
    
    let program = Program::new(items, source_id);
    
    // Test module iterator
    let modules: Vec<_> = program.modules().collect();
    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0].name, Symbol::intern("auth"));
    
    // Test function iterator
    let functions: Vec<_> = program.functions().collect();
    assert_eq!(functions.len(), 2);
    assert_eq!(functions[0].name, Symbol::intern("login"));
    assert_eq!(functions[1].name, Symbol::intern("logout"));
    
    // Test type iterator
    let types: Vec<_> = program.types().collect();
    assert_eq!(types.len(), 2);
    assert_eq!(types[0].name, Symbol::intern("User"));
    assert_eq!(types[1].name, Symbol::intern("Session"));
}

#[test]
fn test_program_ai_contexts() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
    
    // Create nodes with AI contexts
    let ai_context1 = AiContext::new()
        .with_purpose("User authentication")
        .with_domain("Security");
    
    let ai_context2 = AiContext::new()
        .with_purpose("Data validation")
        .with_domain("Data Processing");
    
    let items = vec![
        AstNode::new(
            Item::Function(FunctionDecl {
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
            NodeId::new(1),
        ).with_ai_context(ai_context1),
        AstNode::new(
            Item::Function(FunctionDecl {
                name: Symbol::intern("validate"),
                parameters: vec![],
                return_type: None,
                body: None,
                visibility: Visibility::Public,
                attributes: vec![],
                contracts: None,
                is_async: false,
            }),
            span,
            NodeId::new(2),
        ).with_ai_context(ai_context2),
        AstNode::new(
            Item::Function(FunctionDecl {
                name: Symbol::intern("utility"),
                parameters: vec![],
                return_type: None,
                body: None,
                visibility: Visibility::Public,
                attributes: vec![],
                contracts: None,
                is_async: false,
            }),
            span,
            NodeId::new(3),
        ), // No AI context
    ];
    
    let program = Program::new(items, source_id);
    
    // Test AI context collection
    let ai_contexts: Vec<_> = program.collect_ai_contexts();
    assert_eq!(ai_contexts.len(), 2);
    
    assert_eq!(ai_contexts[0].purpose, Some("User authentication".to_string()));
    assert_eq!(ai_contexts[0].domain, Some("Security".to_string()));
    
    assert_eq!(ai_contexts[1].purpose, Some("Data validation".to_string()));
    assert_eq!(ai_contexts[1].domain, Some("Data Processing".to_string()));
}

#[test]
fn test_program_semantic_type_summary() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
    
    // Create a program with various semantic types
    let items = vec![
        AstNode::new(
            Item::Type(TypeDecl {
                name: Symbol::intern("Email"),
                type_parameters: vec![],
                kind: TypeKind::Semantic(SemanticType {
                    base_type: Box::new(AstNode::new(
                        Type::Primitive(PrimitiveType::String),
                        span,
                        NodeId::new(1),
                    )),
                    constraints: vec![],
                    metadata: SemanticTypeMetadata {
                        business_rules: vec!["Must be valid email".to_string()],
                        examples: vec![],
                        validation_rules: vec![],
                        ai_context: Some("Email address".to_string()),
                        security_classification: SecurityClassification::Internal,
                        compliance_requirements: vec![],
                    },
                }),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Item::Type(TypeDecl {
                name: Symbol::intern("PhoneNumber"),
                type_parameters: vec![],
                kind: TypeKind::Semantic(SemanticType {
                    base_type: Box::new(AstNode::new(
                        Type::Primitive(PrimitiveType::String),
                        span,
                        NodeId::new(3),
                    )),
                    constraints: vec![],
                    metadata: SemanticTypeMetadata {
                        business_rules: vec!["Must be valid phone".to_string()],
                        examples: vec![],
                        validation_rules: vec![],
                        ai_context: Some("Phone number".to_string()),
                        security_classification: SecurityClassification::Confidential,
                        compliance_requirements: vec![],
                    },
                }),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(4),
        ),
        AstNode::new(
            Item::Type(TypeDecl {
                name: Symbol::intern("RegularType"),
                type_parameters: vec![],
                kind: TypeKind::Struct(StructType {
                    fields: vec![],
                }),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(5),
        ),
    ];
    
    let program = Program::new(items, source_id);
    
    // Test semantic type summary
    let summary = program.semantic_type_summary();
    
    // Should have semantic types grouped by security classification
    assert!(summary.contains_key("Internal"));
    assert!(summary.contains_key("Confidential"));
    assert_eq!(summary.get("Internal"), Some(&1));
    assert_eq!(summary.get("Confidential"), Some(&1));
    
    // Regular types shouldn't appear in semantic summary
    assert!(!summary.contains_key("Public"));
}

#[test]
fn test_complex_program_structure() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 1001, 1000), source_id);
    
    // Create a complex program with multiple types of items
    let items = vec![
        // Import statements
        AstNode::new(
            Item::Import(ImportDecl {
                path: "std/io".to_string(),
                items: ImportItems::Specific(vec![
                    ImportItem {
                        name: Symbol::intern("println"),
                        alias: None,
                    },
                ]),
                alias: None,
            }),
            span,
            NodeId::new(1),
        ),
        
        // Type definitions
        AstNode::new(
            Item::Type(TypeDecl {
                name: Symbol::intern("User"),
                type_parameters: vec![],
                kind: TypeKind::Struct(StructType {
                    fields: vec![
                        StructField {
                            name: Symbol::intern("id"),
                            field_type: AstNode::new(
                                Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(64))),
                                span,
                                NodeId::new(2),
                            ),
                            visibility: Visibility::Public,
                        },
                        StructField {
                            name: Symbol::intern("name"),
                            field_type: AstNode::new(
                                Type::Primitive(PrimitiveType::String),
                                span,
                                NodeId::new(3),
                            ),
                            visibility: Visibility::Public,
                        },
                    ],
                }),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(4),
        ),
        
        // Constants
        AstNode::new(
            Item::Const(ConstDecl {
                name: Symbol::intern("MAX_USERS"),
                type_annotation: Some(AstNode::new(
                    Type::Primitive(PrimitiveType::Integer(IntegerType::Unsigned(32))),
                    span,
                    NodeId::new(5),
                )),
                value: AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(1000),
                    }),
                    span,
                    NodeId::new(6),
                ),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(7),
        ),
        
        // Functions
        AstNode::new(
            Item::Function(FunctionDecl {
                name: Symbol::intern("create_user"),
                parameters: vec![
                    Parameter {
                        name: Symbol::intern("name"),
                        type_annotation: Some(AstNode::new(
                            Type::Primitive(PrimitiveType::String),
                            span,
                            NodeId::new(8),
                        )),
                        default_value: None,
                        is_mutable: false,
                    },
                ],
                return_type: Some(AstNode::new(
                    Type::Named(NamedType {
                        name: Symbol::intern("User"),
                        type_arguments: vec![],
                    }),
                    span,
                    NodeId::new(9),
                )),
                body: Some(Box::new(AstNode::new(
                    Stmt::Block(BlockStmt {
                        statements: vec![
                            AstNode::new(
                                Stmt::Return(ReturnStmt {
                                    value: Some(AstNode::new(
                                        Expr::Object(ObjectExpr {
                                            fields: vec![
                                                ObjectField {
                                                    key: ObjectKey::Identifier(Symbol::intern("id")),
                                                    value: AstNode::new(
                                                        Expr::Literal(LiteralExpr {
                                                            value: LiteralValue::Integer(1),
                                                        }),
                                                        span,
                                                        NodeId::new(10),
                                                    ),
                                                },
                                                ObjectField {
                                                    key: ObjectKey::Identifier(Symbol::intern("name")),
                                                    value: AstNode::new(
                                                        Expr::Variable(VariableExpr {
                                                            name: Symbol::intern("name"),
                                                        }),
                                                        span,
                                                        NodeId::new(11),
                                                    ),
                                                },
                                            ],
                                        }),
                                        span,
                                        NodeId::new(12),
                                    )),
                                }),
                                span,
                                NodeId::new(13),
                            ),
                        ],
                    }),
                    span,
                    NodeId::new(14),
                ))),
                visibility: Visibility::Public,
                attributes: vec![],
                contracts: None,
                is_async: false,
            }),
            span,
            NodeId::new(15),
        ),
        
        // Export statements
        AstNode::new(
            Item::Export(ExportDecl {
                items: vec![
                    ExportItem {
                        name: Symbol::intern("User"),
                        alias: None,
                    },
                    ExportItem {
                        name: Symbol::intern("create_user"),
                        alias: Some(Symbol::intern("createUser")),
                    },
                ],
            }),
            span,
            NodeId::new(16),
        ),
    ];
    
    let metadata = ProgramMetadata {
        primary_capability: Some("user_management".to_string()),
        capabilities: vec![
            "create_users".to_string(),
            "manage_user_data".to_string(),
        ],
        dependencies: vec![
            "std/io".to_string(),
        ],
        security_implications: vec![
            "Handles user personal data".to_string(),
        ],
        performance_notes: vec![
            "Linear time user creation".to_string(),
        ],
        ai_insights: vec![
            "Simple user management system".to_string(),
            "Could benefit from validation".to_string(),
        ],
    };
    
    let program = Program::new(items, source_id).with_metadata(metadata);
    
    // Verify program structure
    assert_eq!(program.items.len(), 5);
    assert_eq!(program.source_id, source_id);
    
    // Verify metadata
    assert_eq!(program.metadata.primary_capability, Some("user_management".to_string()));
    assert_eq!(program.metadata.capabilities.len(), 2);
    assert_eq!(program.metadata.dependencies.len(), 1);
    
    // Test iterators
    assert_eq!(program.modules().count(), 0);
    assert_eq!(program.functions().count(), 1);
    assert_eq!(program.types().count(), 1);
    
    // Verify specific items
    let user_type = program.types().next().unwrap();
    assert_eq!(user_type.name, Symbol::intern("User"));
    
    let create_user_func = program.functions().next().unwrap();
    assert_eq!(create_user_func.name, Symbol::intern("create_user"));
    assert_eq!(create_user_func.parameters.len(), 1);
    assert!(create_user_func.return_type.is_some());
    assert!(create_user_func.body.is_some());
}

#[test]
fn test_program_metadata_default() {
    let metadata = ProgramMetadata::default();
    
    assert!(metadata.primary_capability.is_none());
    assert!(metadata.capabilities.is_empty());
    assert!(metadata.dependencies.is_empty());
    assert!(metadata.security_implications.is_empty());
    assert!(metadata.performance_notes.is_empty());
    assert!(metadata.ai_insights.is_empty());
}

#[test]
fn test_program_cloning() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
    
    let original_program = Program::new(
        vec![AstNode::new(
            Item::Const(ConstDecl {
                name: Symbol::intern("TEST"),
                type_annotation: None,
                value: AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(42),
                    }),
                    span,
                    NodeId::new(1),
                ),
                visibility: Visibility::Public,
            }),
            span,
            NodeId::new(2),
        )],
        source_id,
    );
    
    let cloned_program = original_program.clone();
    
    assert_eq!(cloned_program.items.len(), original_program.items.len());
    assert_eq!(cloned_program.source_id, original_program.source_id);
    
    // Verify the cloned item
    match &cloned_program.items[0].kind {
        Item::Const(const_decl) => {
            assert_eq!(const_decl.name, Symbol::intern("TEST"));
        }
        _ => panic!("Expected const item"),
    }
}

#[test]
fn test_empty_program() {
    let source_id = SourceId::new(1);
    let program = Program::new(vec![], source_id);
    
    assert_eq!(program.items.len(), 0);
    assert_eq!(program.modules().count(), 0);
    assert_eq!(program.functions().count(), 0);
    assert_eq!(program.types().count(), 0);
    assert_eq!(program.collect_ai_contexts().len(), 0);
    
    let summary = program.semantic_type_summary();
    assert!(summary.is_empty());
} 