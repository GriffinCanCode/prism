//! Integration tests for the prism-ast crate
//! 
use prism_common::span::Position;
//! These tests verify that all AST components work together correctly,
//! testing real-world scenarios and complex interactions between modules.

use prism_ast::*;
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};
use std::collections::HashMap;

#[test]
fn test_complete_authentication_module() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 2001, 2000), source_id);
    let mut node_counter = 1;
    
    // Create a complete authentication module with all features
    let mut arena = arena::AstArena::new(source_id);
    
    // Email semantic type
    let email_type = AstNode::new(
        Type::Semantic(SemanticType {
            base_type: Box::new(AstNode::new(
                Type::Primitive(PrimitiveType::String),
                span,
                NodeId::new(node_counter),
            )),
            constraints: vec![
                TypeConstraint::Pattern(PatternConstraint {
                    pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
                    flags: vec![],
                }),
                TypeConstraint::Length(LengthConstraint {
                    min_length: Some(5),
                    max_length: Some(254),
                }),
                TypeConstraint::BusinessRule(BusinessRuleConstraint {
                    description: "Must be unique per user".to_string(),
                    expression: AstNode::new(
                        Expr::Call(CallExpr {
                            callee: Box::new(AstNode::new(
                                Expr::Variable(VariableExpr {
                                    name: Symbol::intern("is_unique_email"),
                                }),
                                span,
                                NodeId::new(node_counter + 1),
                            )),
                            arguments: vec![],
                            type_arguments: None,
                            call_style: CallStyle::Function,
                        }),
                        span,
                        NodeId::new(node_counter + 2),
                    ),
                    priority: 1,
                }),
            ],
            metadata: SemanticTypeMetadata {
                business_rules: vec![
                    "Must be verified before use".to_string(),
                    "Must be unique per user".to_string(),
                ],
                examples: vec![
                    "user@example.com".to_string(),
                    "alice.smith@company.org".to_string(),
                ],
                validation_rules: vec![
                    "RFC 5322 compliant".to_string(),
                    "Domain must be whitelisted".to_string(),
                ],
                ai_context: Some("Email address for user authentication and communication".to_string()),
                security_classification: SecurityClassification::Confidential,
                compliance_requirements: vec![
                    "GDPR".to_string(),
                    "CAN-SPAM".to_string(),
                ],
            },
        }),
        span,
        NodeId::new(node_counter + 3),
    );
    node_counter += 4;
    
    // User struct type
    let user_type = AstNode::new(
        Type::Named(NamedType {
            name: Symbol::intern("User"),
            type_arguments: vec![],
        }),
        span,
        NodeId::new(node_counter),
    );
    node_counter += 1;
    
    // Authentication function with contracts
    let auth_function = AstNode::new(
        Item::Function(FunctionDecl {
            name: Symbol::intern("authenticate"),
            parameters: vec![
                Parameter {
                    name: Symbol::intern("email"),
                    type_annotation: Some(email_type.clone()),
                    default_value: None,
                    is_mutable: false,
                },
                Parameter {
                    name: Symbol::intern("password"),
                    type_annotation: Some(AstNode::new(
                        Type::Semantic(SemanticType {
                            base_type: Box::new(AstNode::new(
                                Type::Primitive(PrimitiveType::String),
                                span,
                                NodeId::new(node_counter),
                            )),
                            constraints: vec![
                                TypeConstraint::Length(LengthConstraint {
                                    min_length: Some(8),
                                    max_length: Some(128),
                                }),
                            ],
                            metadata: SemanticTypeMetadata {
                                business_rules: vec!["Must be hashed before storage".to_string()],
                                examples: vec![],
                                validation_rules: vec!["Minimum 8 characters".to_string()],
                                ai_context: Some("User password for authentication".to_string()),
                                security_classification: SecurityClassification::Restricted,
                                compliance_requirements: vec!["Password policy compliance".to_string()],
                            },
                        }),
                        span,
                        NodeId::new(node_counter + 1),
                    )),
                    default_value: None,
                    is_mutable: false,
                },
            ],
            return_type: Some(AstNode::new(
                Type::Union(Box::new(UnionType {
                    members: vec![
                        user_type.clone(),
                        AstNode::new(
                            Type::Named(NamedType {
                                name: Symbol::intern("AuthError"),
                                type_arguments: vec![],
                            }),
                            span,
                            NodeId::new(node_counter + 2),
                        ),
                    ],
                    discriminant: None,
                    constraints: vec![],
                    common_operations: vec![],
                    metadata: UnionMetadata {
                        name: Some(Symbol::intern("AuthResult")),
                        description: Some("Result of authentication attempt".to_string()),
                        semantics: UnionSemantics::Disjoint,
                        ai_context: Some("Authentication can succeed with User or fail with AuthError".to_string()),
                    },
                })),
                span,
                NodeId::new(node_counter + 3),
            )),
            body: Some(Box::new(AstNode::new(
                Stmt::Block(BlockStmt {
                    statements: vec![
                        // Validate email
                        AstNode::new(
                            Stmt::If(IfStmt {
                                condition: AstNode::new(
                                    Expr::Unary(UnaryExpr {
                                        operator: UnaryOperator::Not,
                                        operand: Box::new(AstNode::new(
                                            Expr::Call(CallExpr {
                                                callee: Box::new(AstNode::new(
                                                    Expr::Variable(VariableExpr {
                                                        name: Symbol::intern("validate_email"),
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 4),
                                                )),
                                                arguments: vec![AstNode::new(
                                                    Expr::Variable(VariableExpr {
                                                        name: Symbol::intern("email"),
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 5),
                                                )],
                                                type_arguments: None,
                                                call_style: CallStyle::Function,
                                            }),
                                            span,
                                            NodeId::new(node_counter + 6),
                                        )),
                                    }),
                                    span,
                                    NodeId::new(node_counter + 7),
                                ),
                                then_branch: Box::new(AstNode::new(
                                    Stmt::Return(ReturnStmt {
                                        value: Some(AstNode::new(
                                            Expr::Call(CallExpr {
                                                callee: Box::new(AstNode::new(
                                                    Expr::Variable(VariableExpr {
                                                        name: Symbol::intern("AuthError"),
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 8),
                                                )),
                                                arguments: vec![AstNode::new(
                                                    Expr::Literal(LiteralExpr {
                                                        value: LiteralValue::String("Invalid email format".to_string()),
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 9),
                                                )],
                                                type_arguments: None,
                                                call_style: CallStyle::Constructor,
                                            }),
                                            span,
                                            NodeId::new(node_counter + 10),
                                        )),
                                    }),
                                    span,
                                    NodeId::new(node_counter + 11),
                                )),
                                else_branch: None,
                            }),
                            span,
                            NodeId::new(node_counter + 12),
                        ),
                        // Database lookup with effects
                        AstNode::new(
                            Stmt::Variable(VariableDecl {
                                name: Symbol::intern("user"),
                                type_annotation: Some(AstNode::new(
                                    Type::Effect(EffectType {
                                        base_type: Box::new(AstNode::new(
                                                                        Type::Union(Box::new(UnionType {
                                members: vec![
                                    user_type.clone(),
                                    AstNode::new(
                                        Type::Primitive(PrimitiveType::Unit),
                                        span,
                                        NodeId::new(node_counter + 13),
                                    ),
                                ],
                                discriminant: None,
                                constraints: vec![],
                                common_operations: vec![],
                                metadata: UnionMetadata::default(),
                            })),
                                            span,
                                            NodeId::new(node_counter + 14),
                                        )),
                                        effects: vec![
                                            Effect::Database(DatabaseEffect {
                                                operations: vec![DatabaseOperation::Read],
                                                transaction_required: false,
                                                constraints: vec![
                                                    DatabaseConstraint {
                                                        constraint_type: DatabaseConstraintType::QueryTimeout,
                                                        value: AstNode::new(
                                                            Expr::Literal(LiteralExpr {
                                                                value: LiteralValue::Integer(5000), // 5 seconds
                                                            }),
                                                            span,
                                                            NodeId::new(node_counter + 15),
                                                        ),
                                                    },
                                                ],
                                            }),
                                        ],
                                        composition_rules: vec![],
                                        capability_requirements: vec![
                                            CapabilityRequirement {
                                                name: Symbol::intern("database_read"),
                                                permissions: vec![
                                                    Permission {
                                                        name: Symbol::intern("read_users"),
                                                        scope: PermissionScope::Module(Symbol::intern("auth")),
                                                        level: PermissionLevel::Read,
                                                    },
                                                ],
                                                constraints: vec![],
                                            },
                                        ],
                                        metadata: EffectMetadata {
                                            name: Some(Symbol::intern("DatabaseRead")),
                                            description: Some("Database read operation".to_string()),
                                            category: EffectCategory::ResourceManagement,
                                            ai_context: Some("Reads user data from database".to_string()),
                                        },
                                    }),
                                    span,
                                    NodeId::new(node_counter + 16),
                                )),
                                initializer: Some(AstNode::new(
                                    Expr::Call(CallExpr {
                                        callee: Box::new(AstNode::new(
                                            Expr::Variable(VariableExpr {
                                                name: Symbol::intern("find_user_by_email"),
                                            }),
                                            span,
                                            NodeId::new(node_counter + 17),
                                        )),
                                        arguments: vec![AstNode::new(
                                            Expr::Variable(VariableExpr {
                                                name: Symbol::intern("email"),
                                            }),
                                            span,
                                            NodeId::new(node_counter + 18),
                                        )],
                                        type_arguments: None,
                                        call_style: CallStyle::Function,
                                    }),
                                    span,
                                    NodeId::new(node_counter + 19),
                                )),
                                is_mutable: false,
                                visibility: Visibility::Private,
                            }),
                            span,
                            NodeId::new(node_counter + 20),
                        ),
                        // Pattern matching on user result
                        AstNode::new(
                            Stmt::Match(MatchStmt {
                                expression: AstNode::new(
                                    Expr::Variable(VariableExpr {
                                        name: Symbol::intern("user"),
                                    }),
                                    span,
                                    NodeId::new(node_counter + 21),
                                ),
                                arms: vec![
                                    MatchArm {
                                        pattern: AstNode::new(
                                            Pattern::Identifier(Symbol::intern("found_user")),
                                            span,
                                            NodeId::new(node_counter + 22),
                                        ),
                                        guard: Some(AstNode::new(
                                            Expr::Call(CallExpr {
                                                callee: Box::new(AstNode::new(
                                                    Expr::Variable(VariableExpr {
                                                        name: Symbol::intern("verify_password"),
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 23),
                                                )),
                                                arguments: vec![
                                                    AstNode::new(
                                                        Expr::Variable(VariableExpr {
                                                            name: Symbol::intern("password"),
                                                        }),
                                                        span,
                                                        NodeId::new(node_counter + 24),
                                                    ),
                                                    AstNode::new(
                                                        Expr::Member(MemberExpr {
                                                            object: Box::new(AstNode::new(
                                                                Expr::Variable(VariableExpr {
                                                                    name: Symbol::intern("found_user"),
                                                                }),
                                                                span,
                                                                NodeId::new(node_counter + 25),
                                                            )),
                                                            member: Symbol::intern("password_hash"),
                                                            safe_navigation: false,
                                                        }),
                                                        span,
                                                        NodeId::new(node_counter + 26),
                                                    ),
                                                ],
                                                type_arguments: None,
                                                call_style: CallStyle::Function,
                                            }),
                                            span,
                                            NodeId::new(node_counter + 27),
                                        )),
                                        body: Box::new(AstNode::new(
                                            Stmt::Return(ReturnStmt {
                                                value: Some(AstNode::new(
                                                    Expr::Variable(VariableExpr {
                                                        name: Symbol::intern("found_user"),
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 28),
                                                )),
                                            }),
                                            span,
                                            NodeId::new(node_counter + 29),
                                        )),
                                    },
                                    MatchArm {
                                        pattern: AstNode::new(
                                            Pattern::Wildcard,
                                            span,
                                            NodeId::new(node_counter + 30),
                                        ),
                                        guard: None,
                                        body: Box::new(AstNode::new(
                                            Stmt::Return(ReturnStmt {
                                                value: Some(AstNode::new(
                                                    Expr::Call(CallExpr {
                                                        callee: Box::new(AstNode::new(
                                                            Expr::Variable(VariableExpr {
                                                                name: Symbol::intern("AuthError"),
                                                            }),
                                                            span,
                                                            NodeId::new(node_counter + 31),
                                                        )),
                                                        arguments: vec![AstNode::new(
                                                            Expr::Literal(LiteralExpr {
                                                                value: LiteralValue::String("Invalid credentials".to_string()),
                                                            }),
                                                            span,
                                                            NodeId::new(node_counter + 32),
                                                        )],
                                                        type_arguments: None,
                                                        call_style: CallStyle::Constructor,
                                                    }),
                                                    span,
                                                    NodeId::new(node_counter + 33),
                                                )),
                                            }),
                                            span,
                                            NodeId::new(node_counter + 34),
                                        )),
                                    },
                                ],
                            }),
                            span,
                            NodeId::new(node_counter + 35),
                        ),
                    ],
                }),
                span,
                NodeId::new(node_counter + 36),
            ))),
            visibility: Visibility::Public,
            attributes: vec![
                Attribute {
                    name: "security".to_string(),
                    arguments: vec![
                        AttributeArgument::Named {
                            name: "level".to_string(),
                            value: LiteralValue::String("high".to_string()),
                        },
                    ],
                },
            ],
            contracts: Some(Contracts {
                requires: vec![
                    AstNode::new(
                        Expr::Binary(BinaryExpr {
                            left: Box::new(AstNode::new(
                                Expr::Call(CallExpr {
                                    callee: Box::new(AstNode::new(
                                        Expr::Variable(VariableExpr {
                                            name: Symbol::intern("len"),
                                        }),
                                        span,
                                        NodeId::new(node_counter + 37),
                                    )),
                                    arguments: vec![AstNode::new(
                                        Expr::Variable(VariableExpr {
                                            name: Symbol::intern("email"),
                                        }),
                                        span,
                                        NodeId::new(node_counter + 38),
                                    )],
                                    type_arguments: None,
                                    call_style: CallStyle::Function,
                                }),
                                span,
                                NodeId::new(node_counter + 39),
                            )),
                            operator: BinaryOperator::Greater,
                            right: Box::new(AstNode::new(
                                Expr::Literal(LiteralExpr {
                                    value: LiteralValue::Integer(0),
                                }),
                                span,
                                NodeId::new(node_counter + 40),
                            )),
                        }),
                        span,
                        NodeId::new(node_counter + 41),
                    ),
                ],
                ensures: vec![
                    AstNode::new(
                        Expr::Binary(BinaryExpr {
                            left: Box::new(AstNode::new(
                                Expr::Call(CallExpr {
                                    callee: Box::new(AstNode::new(
                                        Expr::Variable(VariableExpr {
                                            name: Symbol::intern("is_valid_result"),
                                        }),
                                        span,
                                        NodeId::new(node_counter + 42),
                                    )),
                                    arguments: vec![AstNode::new(
                                        Expr::Variable(VariableExpr {
                                            name: Symbol::intern("result"),
                                        }),
                                        span,
                                        NodeId::new(node_counter + 43),
                                    )],
                                    type_arguments: None,
                                    call_style: CallStyle::Function,
                                }),
                                span,
                                NodeId::new(node_counter + 44),
                            )),
                            operator: BinaryOperator::Equal,
                            right: Box::new(AstNode::new(
                                Expr::Literal(LiteralExpr {
                                    value: LiteralValue::Boolean(true),
                                }),
                                span,
                                NodeId::new(node_counter + 45),
                            )),
                        }),
                        span,
                        NodeId::new(node_counter + 46),
                    ),
                ],
                invariants: vec![],
            }),
            is_async: false,
        }),
        span,
        NodeId::new(node_counter + 47),
    ).with_ai_context(
        AiContext::new()
            .with_purpose("Authenticate user with email and password")
            .with_domain("Security")
            .with_description("Validates user credentials and returns authenticated user or error")
            .with_capability("user_authentication")
            .with_capability("database_access")
            .with_side_effect("Database query for user lookup")
            .with_side_effect("Password verification")
            .with_precondition("Email must be valid format")
            .with_precondition("Password must meet length requirements")
            .with_postcondition("Returns User on successful authentication")
            .with_postcondition("Returns AuthError on failed authentication")
            .with_invariant("User data remains consistent")
            .with_testing_recommendation("Test with valid and invalid credentials")
            .with_testing_recommendation("Test with malformed email addresses")
            .with_refactoring_suggestion("Consider extracting password validation")
    );
    
    node_counter += 48;
    
    // Create the complete program
    let items = vec![auth_function];
    
    let metadata = ProgramMetadata {
        primary_capability: Some("user_authentication".to_string()),
        capabilities: vec![
            "email_validation".to_string(),
            "password_verification".to_string(),
            "database_access".to_string(),
            "security_enforcement".to_string(),
        ],
        dependencies: vec![
            "database".to_string(),
            "crypto".to_string(),
            "validation".to_string(),
        ],
        security_implications: vec![
            "Handles sensitive user credentials".to_string(),
            "Accesses user database".to_string(),
            "Performs cryptographic operations".to_string(),
        ],
        performance_notes: vec![
            "Database query per authentication".to_string(),
            "Password hashing verification".to_string(),
            "Email validation regex".to_string(),
        ],
        ai_insights: vec![
            "Well-structured authentication flow".to_string(),
            "Comprehensive error handling".to_string(),
            "Strong type safety with semantic types".to_string(),
            "Effect system tracks side effects".to_string(),
        ],
    };
    
    let program = Program::new(items, source_id).with_metadata(metadata);
    
    // Verify the complete program structure
    assert_eq!(program.items.len(), 1);
    assert_eq!(program.functions().count(), 1);
    assert_eq!(program.metadata.primary_capability, Some("user_authentication".to_string()));
    assert_eq!(program.metadata.capabilities.len(), 4);
    assert_eq!(program.metadata.dependencies.len(), 3);
    assert_eq!(program.metadata.security_implications.len(), 3);
    
    // Test AI context collection
    let ai_contexts: Vec<_> = program.collect_ai_contexts();
    assert_eq!(ai_contexts.len(), 1);
    assert_eq!(ai_contexts[0].purpose, Some("Authenticate user with email and password".to_string()));
    assert_eq!(ai_contexts[0].domain, Some("Security".to_string()));
    assert_eq!(ai_contexts[0].capabilities.len(), 2);
    assert_eq!(ai_contexts[0].side_effects.len(), 2);
    assert_eq!(ai_contexts[0].preconditions.len(), 2);
    assert_eq!(ai_contexts[0].postconditions.len(), 2);
    
    // Test visitor pattern with the complex AST
    let mut counter = NodeCounterVisitor::new();
    counter.visit_program(&program);
    
    // Should have counted all the nodes we created
    assert!(counter.total_count() > 40); // We created many nodes
    assert!(counter.expr_count > 20);
    assert!(counter.stmt_count > 5);
    assert!(counter.type_count > 5);
    assert!(counter.pattern_count > 0);
    
    println!("Integration test completed successfully!");
    println!("Total nodes: {}", counter.total_count());
    println!("Expressions: {}", counter.expr_count);
    println!("Statements: {}", counter.stmt_count);
    println!("Types: {}", counter.type_count);
    println!("Patterns: {}", counter.pattern_count);
}

// Helper visitor for counting nodes
struct NodeCounterVisitor {
    expr_count: usize,
    stmt_count: usize,
    type_count: usize,
    pattern_count: usize,
}

impl NodeCounterVisitor {
    fn new() -> Self {
        Self {
            expr_count: 0,
            stmt_count: 0,
            type_count: 0,
            pattern_count: 0,
        }
    }
    
    fn total_count(&self) -> usize {
        self.expr_count + self.stmt_count + self.type_count + self.pattern_count
    }
    
    fn visit_program(&mut self, program: &Program) {
        for item in &program.items {
            self.visit_item(&item.kind);
        }
    }
    
    fn visit_item(&mut self, item: &Item) {
        match item {
            Item::Function(func) => self.visit_function(func),
            Item::Type(type_decl) => self.visit_type_decl(type_decl),
            Item::Module(module) => self.visit_module(module),
            Item::Const(const_decl) => {
                self.visit_expr(&const_decl.value);
                if let Some(type_ann) = &const_decl.type_annotation {
                    self.visit_type(&type_ann.kind);
                }
            }
            Item::Variable(var_decl) => {
                if let Some(init) = &var_decl.initializer {
                    self.visit_expr(init);
                }
                if let Some(type_ann) = &var_decl.type_annotation {
                    self.visit_type(&type_ann.kind);
                }
            }
            Item::Statement(stmt) => self.visit_stmt(stmt),
            _ => {} // Handle other items as needed
        }
    }
    
    fn visit_function(&mut self, func: &FunctionDecl) {
        for param in &func.parameters {
            if let Some(default) = &param.default_value {
                self.visit_expr(default);
            }
            if let Some(type_ann) = &param.type_annotation {
                self.visit_type(&type_ann.kind);
            }
        }
        
        if let Some(return_type) = &func.return_type {
            self.visit_type(&return_type.kind);
        }
        
        if let Some(body) = &func.body {
            self.visit_stmt(&body.kind);
        }
        
        if let Some(contracts) = &func.contracts {
            for req in &contracts.requires {
                self.visit_expr(req);
            }
            for ens in &contracts.ensures {
                self.visit_expr(ens);
            }
            for inv in &contracts.invariants {
                self.visit_expr(inv);
            }
        }
    }
    
    fn visit_type_decl(&mut self, type_decl: &TypeDecl) {
        self.type_count += 1;
        match &type_decl.kind {
            TypeKind::Semantic(semantic) => {
                self.visit_type(&semantic.base_type.kind);
                for constraint in &semantic.constraints {
                    self.visit_type_constraint(constraint);
                }
            }
            TypeKind::Struct(struct_type) => {
                for field in &struct_type.fields {
                    self.visit_type(&field.field_type.kind);
                }
            }
            TypeKind::Enum(enum_type) => {
                for variant in &enum_type.variants {
                    for field in &variant.fields {
                        self.visit_type(&field.kind);
                    }
                }
            }
            TypeKind::Alias(aliased_type) => {
                self.visit_type(&aliased_type.kind);
            }
            _ => {}
        }
    }
    
    fn visit_module(&mut self, module: &ModuleDecl) {
        for section in &module.sections {
            for item in &section.kind.items {
                self.visit_stmt(&item.kind);
            }
        }
    }
    
    fn visit_expr(&mut self, expr: &AstNode<Expr>) {
        self.expr_count += 1;
        
        match &expr.kind {
            Expr::Binary(binary) => {
                self.visit_expr(&binary.left);
                self.visit_expr(&binary.right);
            }
            Expr::Unary(unary) => {
                self.visit_expr(&unary.operand);
            }
            Expr::Call(call) => {
                self.visit_expr(&call.callee);
                for arg in &call.arguments {
                    self.visit_expr(arg);
                }
            }
            Expr::Member(member) => {
                self.visit_expr(&member.object);
            }
            Expr::If(if_expr) => {
                self.visit_expr(&if_expr.condition);
                self.visit_expr(&if_expr.then_branch);
                if let Some(else_branch) = &if_expr.else_branch {
                    self.visit_expr(else_branch);
                }
            }
            Expr::Match(match_expr) => {
                self.visit_expr(&match_expr.scrutinee);
                for arm in &match_expr.arms {
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                }
            }
            Expr::Array(array) => {
                for element in &array.elements {
                    self.visit_expr(element);
                }
            }
            Expr::Object(object) => {
                for field in &object.fields {
                    self.visit_expr(&field.value);
                    if let ObjectKey::Computed(key_expr) = &field.key {
                        self.visit_expr(key_expr);
                    }
                }
            }
            Expr::TypeAssertion(assertion) => {
                self.visit_expr(&assertion.expression);
                self.visit_type(&assertion.target_type.kind);
            }
            _ => {} // Handle other expression types as needed
        }
    }
    
    fn visit_stmt(&mut self, stmt: &Stmt) {
        self.stmt_count += 1;
        
        match stmt {
            Stmt::Expression(expr_stmt) => {
                self.visit_expr(&expr_stmt.expression);
            }
            Stmt::Variable(var_decl) => {
                if let Some(init) = &var_decl.initializer {
                    self.visit_expr(init);
                }
                if let Some(type_ann) = &var_decl.type_annotation {
                    self.visit_type(&type_ann.kind);
                }
            }
            Stmt::If(if_stmt) => {
                self.visit_expr(&if_stmt.condition);
                self.visit_stmt(&if_stmt.then_branch.kind);
                if let Some(else_branch) = &if_stmt.else_branch {
                    self.visit_stmt(&else_branch.kind);
                }
            }
            Stmt::Match(match_stmt) => {
                self.visit_expr(&match_stmt.expression);
                for arm in &match_stmt.arms {
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_stmt(&arm.body.kind);
                }
            }
            Stmt::Block(block) => {
                for stmt in &block.statements {
                    self.visit_stmt(&stmt.kind);
                }
            }
            Stmt::Return(return_stmt) => {
                if let Some(value) = &return_stmt.value {
                    self.visit_expr(value);
                }
            }
            _ => {} // Handle other statement types as needed
        }
    }
    
    fn visit_type(&mut self, ty: &Type) {
        self.type_count += 1;
        
        match ty {
            Type::Named(named) => {
                for arg in &named.type_arguments {
                    self.visit_type(&arg.kind);
                }
            }
            Type::Function(func) => {
                for param in &func.parameters {
                    self.visit_type(&param.kind);
                }
                self.visit_type(&func.return_type.kind);
            }
            Type::Union(union) => {
                for member in &union.members {
                    self.visit_type(&member.kind);
                }
            }
            Type::Intersection(intersection) => {
                for member in &intersection.members {
                    self.visit_type(&member.kind);
                }
            }
            Type::Semantic(semantic) => {
                self.visit_type(&semantic.base_type.kind);
                for constraint in &semantic.constraints {
                    self.visit_type_constraint(constraint);
                }
            }
            Type::Effect(effect) => {
                self.visit_type(&effect.base_type.kind);
            }
            Type::Array(array) => {
                self.visit_type(&array.element_type.kind);
                if let Some(size) = &array.size {
                    self.visit_expr(size);
                }
            }
            Type::Tuple(tuple) => {
                for element in &tuple.elements {
                    self.visit_type(&element.kind);
                }
            }
            _ => {} // Handle other type variants
        }
    }
    
    fn visit_pattern(&mut self, pattern: &AstNode<Pattern>) {
        self.pattern_count += 1;
        
        match &pattern.kind {
            Pattern::Tuple(elements) => {
                for element in elements {
                    self.visit_pattern(element);
                }
            }
            Pattern::Array(elements) => {
                for element in elements {
                    self.visit_pattern(element);
                }
            }
            Pattern::Object(fields) => {
                for field in fields {
                    self.visit_pattern(&field.pattern);
                }
            }
            Pattern::Or(patterns) => {
                for pattern in patterns {
                    self.visit_pattern(pattern);
                }
            }
            _ => {} // Terminal patterns
        }
    }
    
    fn visit_type_constraint(&mut self, constraint: &TypeConstraint) {
        match constraint {
            TypeConstraint::Range(range) => {
                if let Some(min) = &range.min {
                    self.visit_expr(min);
                }
                if let Some(max) = &range.max {
                    self.visit_expr(max);
                }
            }
            TypeConstraint::Custom(custom) => {
                self.visit_expr(&custom.expression);
            }
            TypeConstraint::BusinessRule(rule) => {
                self.visit_expr(&rule.expression);
            }
            _ => {} // Other constraints don't contain expressions
        }
    }
}

#[test]
fn test_ast_serialization_roundtrip() {
    // Test that AST nodes can be serialized and deserialized correctly
    // This test requires the serde feature to be enabled
    
    #[cfg(feature = "serde")]
    {
        use serde_json;
        
        let source_id = SourceId::new(1);
        let span = Span::new(Position::new(1, 1, 0), Position::new(1, 101, 100), source_id);
        
        let original_expr = AstNode::new(
            Expr::Binary(BinaryExpr {
                left: Box::new(AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(10),
                    }),
                    span,
                    NodeId::new(1),
                )),
                operator: BinaryOperator::Add,
                right: Box::new(AstNode::new(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(20),
                    }),
                    span,
                    NodeId::new(2),
                )),
            }),
            span,
            NodeId::new(3),
        );
        
        // Serialize to JSON
        let serialized = serde_json::to_string(&original_expr).expect("Serialization failed");
        
        // Deserialize back
        let deserialized: AstNode<Expr> = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        // Verify the roundtrip
        assert_eq!(deserialized.id, original_expr.id);
        assert_eq!(deserialized.span, original_expr.span);
        
        match (&deserialized.kind, &original_expr.kind) {
            (Expr::Binary(deser_bin), Expr::Binary(orig_bin)) => {
                assert_eq!(deser_bin.operator, orig_bin.operator);
            }
            _ => panic!("Expression type mismatch after roundtrip"),
        }
        
        println!("Serialization roundtrip test passed!");
    }
    
    #[cfg(not(feature = "serde"))]
    {
        println!("Serialization test skipped (serde feature not enabled)");
    }
}

#[test] 
fn test_comprehensive_ast_construction() {
    // Test building a complex AST that uses all major components
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 501, 500), source_id);
    
    // Create a semantic email type
    let email_semantic_type = SemanticType {
        base_type: Box::new(AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(1),
        )),
        constraints: vec![
            TypeConstraint::Pattern(PatternConstraint {
                pattern: r"^[^\s@]+@[^\s@]+\.[^\s@]+$".to_string(),
                flags: vec![],
            }),
            TypeConstraint::Length(LengthConstraint {
                min_length: Some(5),
                max_length: Some(100),
            }),
        ],
        metadata: SemanticTypeMetadata {
            business_rules: vec!["Must be unique".to_string()],
            examples: vec!["user@example.com".to_string()],
            validation_rules: vec!["Valid email format".to_string()],
            ai_context: Some("User email address".to_string()),
            security_classification: SecurityClassification::Internal,
            compliance_requirements: vec!["GDPR".to_string()],
        },
    };
    
    // Create a dependent array type
    let dependent_array_type = DependentType {
        parameters: vec![
            DependentParameter {
                name: Symbol::intern("N"),
                parameter_type: AstNode::new(
                    Type::Primitive(PrimitiveType::Integer(IntegerType::Natural)),
                    span,
                    NodeId::new(2),
                ),
                bounds: vec![
                    ParameterBound::Lower(AstNode::new(
                        Expr::Literal(LiteralExpr {
                            value: LiteralValue::Integer(1),
                        }),
                        span,
                        NodeId::new(3),
                    )),
                ],
                default_value: None,
                is_compile_time_constant: true,
                role: ParameterRole::Size,
            },
        ],
        type_expression: AstNode::new(
            Expr::Call(CallExpr {
                callee: Box::new(AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("Array"),
                    }),
                    span,
                    NodeId::new(4),
                )),
                arguments: vec![AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("N"),
                    }),
                    span,
                    NodeId::new(5),
                )],
                type_arguments: None,
                call_style: CallStyle::Constructor,
            }),
            span,
            NodeId::new(6),
        ),
        parameter_constraints: vec![],
        computation_rules: vec![],
        validation_predicates: vec![
            ValidationPredicate {
                expression: AstNode::new(
                    Expr::Binary(BinaryExpr {
                        left: Box::new(AstNode::new(
                            Expr::Variable(VariableExpr {
                                name: Symbol::intern("N"),
                            }),
                            span,
                            NodeId::new(7),
                        )),
                        operator: BinaryOperator::Less,
                        right: Box::new(AstNode::new(
                            Expr::Literal(LiteralExpr {
                                value: LiteralValue::Integer(1000),
                            }),
                            span,
                            NodeId::new(8),
                        )),
                    }),
                    span,
                    NodeId::new(9),
                ),
                error_message: "Array size too large".to_string(),
                level: ValidationLevel::CompileTime,
            },
        ],
        refinement_conditions: vec![],
    };
    
    // Create an effect type with multiple effects
    let effect_type = EffectType {
        base_type: Box::new(AstNode::new(
            Type::Primitive(PrimitiveType::String),
            span,
            NodeId::new(10),
        )),
        effects: vec![
            Effect::IO(IOEffect {
                operations: vec![IOOperation::Read, IOOperation::Write],
                resources: vec![
                    IOResource {
                        resource_type: IOResourceType::File,
                        identifier: "/tmp/data.txt".to_string(),
                        access_pattern: AccessPattern::ReadWrite,
                    },
                ],
                constraints: vec![],
            }),
            Effect::Database(DatabaseEffect {
                operations: vec![DatabaseOperation::Read],
                transaction_required: false,
                constraints: vec![],
            }),
        ],
        composition_rules: vec![],
        capability_requirements: vec![
            CapabilityRequirement {
                name: Symbol::intern("file_access"),
                permissions: vec![
                    Permission {
                        name: Symbol::intern("read_write_temp"),
                        scope: PermissionScope::Global,
                        level: PermissionLevel::ReadWrite,
                    },
                ],
                constraints: vec![],
            },
        ],
        metadata: EffectMetadata {
            name: Some(Symbol::intern("FileAndDbAccess")),
            description: Some("File and database access".to_string()),
            category: EffectCategory::ResourceManagement,
            ai_context: Some("Manages file and database resources".to_string()),
        },
    };
    
    // Create a union type
    let union_type = UnionType {
        members: vec![
            AstNode::new(
                Type::Semantic(email_semantic_type),
                span,
                NodeId::new(11),
            ),
            AstNode::new(
                Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
                span,
                NodeId::new(12),
            ),
        ],
        discriminant: Some(Box::new(UnionDiscriminant {
            field_name: Symbol::intern("type"),
            discriminant_type: Box::new(AstNode::new(
                Type::Primitive(PrimitiveType::String),
                span,
                NodeId::new(13),
            )),
            tag_mapping: {
                let mut map = HashMap::new();
                map.insert("email".to_string(), 0);
                map.insert("id".to_string(), 1);
                map
            },
        })),
        constraints: vec![],
        common_operations: vec![],
        metadata: UnionMetadata {
            name: Some(Symbol::intern("EmailOrId")),
            description: Some("Either email or ID".to_string()),
            semantics: UnionSemantics::Disjoint,
            ai_context: Some("User identifier".to_string()),
        },
    };
    
    // Verify all types were created successfully
    assert_eq!(union_type.members.len(), 2);
    assert!(union_type.discriminant.is_some());
    
    if let Some(disc) = &union_type.discriminant {
        assert_eq!(disc.field_name, Symbol::intern("type"));
        assert_eq!(disc.tag_mapping.len(), 2);
    }
    
    assert_eq!(dependent_array_type.parameters.len(), 1);
    assert_eq!(dependent_array_type.validation_predicates.len(), 1);
    
    assert_eq!(effect_type.effects.len(), 2);
    assert_eq!(effect_type.capability_requirements.len(), 1);
    
    println!("Comprehensive AST construction test passed!");
} 