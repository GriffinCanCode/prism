//! Comprehensive tests for the visitor module

use prism_common::span::Position;
use prism_ast::{visitor::*, expr::*, stmt, types::*, pattern::*, node::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

// Test visitor that counts nodes
struct NodeCounter {
    expr_count: usize,
    stmt_count: usize,
    type_count: usize,
    pattern_count: usize,
}

impl NodeCounter {
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
}

impl AstVisitor for NodeCounter {
    type Result = ();
    
    fn visit_expr(&mut self, expr: &AstNode<Expr>) -> Self::Result {
        self.expr_count += 1;
        
        // Visit children based on expression type
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
            Expr::Index(index) => {
                self.visit_expr(&index.object);
                self.visit_expr(&index.index);
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
            Expr::Lambda(lambda) => {
                self.visit_expr(&lambda.body);
                for param in &lambda.parameters {
                    if let Some(default) = &param.default_value {
                        self.visit_expr(default);
                    }
                    if let Some(type_ann) = &param.type_annotation {
                        self.visit_type(type_ann);
                    }
                }
            }
            Expr::If(if_expr) => {
                self.visit_expr(&if_expr.condition);
                self.visit_expr(&if_expr.then_branch);
                if let Some(else_branch) = &if_expr.else_branch {
                    self.visit_expr(else_branch);
                }
            }
            Expr::While(while_expr) => {
                self.visit_expr(&while_expr.condition);
                self.visit_expr(&while_expr.body);
            }
            Expr::For(for_expr) => {
                self.visit_expr(&for_expr.iterable);
                self.visit_expr(&for_expr.body);
            }
            Expr::Range(range) => {
                self.visit_expr(&range.start);
                self.visit_expr(&range.end);
            }
            Expr::Tuple(tuple) => {
                for element in &tuple.elements {
                    self.visit_expr(element);
                }
            }
            Expr::Block(block) => {
                for stmt in &block.statements {
                    self.visit_stmt(stmt);
                }
                if let Some(final_expr) = &block.final_expr {
                    self.visit_expr(final_expr);
                }
            }
            Expr::Return(ret) => {
                if let Some(value) = &ret.value {
                    self.visit_expr(value);
                }
            }
            Expr::Break(brk) => {
                if let Some(value) = &brk.value {
                    self.visit_expr(value);
                }
            }
            Expr::Continue(cont) => {
                if let Some(value) = &cont.value {
                    self.visit_expr(value);
                }
            }
            Expr::Throw(throw) => {
                self.visit_expr(&throw.exception);
            }
            Expr::Await(await_expr) => {
                self.visit_expr(&await_expr.expression);
            }
            Expr::Yield(yield_expr) => {
                if let Some(value) = &yield_expr.value {
                    self.visit_expr(value);
                }
            }
            Expr::TypeAssertion(assertion) => {
                self.visit_expr(&assertion.expression);
                self.visit_type(&assertion.target_type);
            }
            // Terminal expressions don't need recursion
            Expr::Literal(_) | Expr::Variable(_) | Expr::Error(_) => {}
            // Match expressions need pattern visiting
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
            // Try expressions need special handling
            Expr::Try(try_expr) => {
                self.visit_expr(&try_expr.try_block);
                for catch in &try_expr.catch_clauses {
                    if let Some(pattern) = &catch.pattern {
                        self.visit_pattern(pattern);
                    }
                    self.visit_expr(&catch.body);
                }
                if let Some(finally) = &try_expr.finally_block {
                    self.visit_expr(finally);
                }
            }
        }
    }
    
    fn visit_stmt(&mut self, stmt: &AstNode<Stmt>) -> Self::Result {
        self.stmt_count += 1;
        
        match &stmt.kind {
            Stmt::Expression(expr_stmt) => {
                self.visit_expr(&expr_stmt.expression);
            }
            Stmt::Variable(var_decl) => {
                if let Some(init) = &var_decl.initializer {
                    self.visit_expr(init);
                }
                if let Some(type_ann) = &var_decl.type_annotation {
                    self.visit_type(type_ann);
                }
            }
            Stmt::Function(func_decl) => {
                for param in &func_decl.parameters {
                    if let Some(default) = &param.default_value {
                        self.visit_expr(default);
                    }
                    if let Some(type_ann) = &param.type_annotation {
                        self.visit_type(type_ann);
                    }
                }
                if let Some(return_type) = &func_decl.return_type {
                    self.visit_type(return_type);
                }
                if let Some(body) = &func_decl.body {
                    self.visit_stmt(body);
                }
            }
            Stmt::If(if_stmt) => {
                self.visit_expr(&if_stmt.condition);
                self.visit_stmt(&if_stmt.then_branch);
                if let Some(else_branch) = &if_stmt.else_branch {
                    self.visit_stmt(else_branch);
                }
            }
            Stmt::While(while_stmt) => {
                self.visit_expr(&while_stmt.condition);
                self.visit_stmt(&while_stmt.body);
            }
            Stmt::For(for_stmt) => {
                self.visit_expr(&for_stmt.iterable);
                self.visit_stmt(&for_stmt.body);
            }
            Stmt::Match(match_stmt) => {
                self.visit_expr(&match_stmt.expression);
                for arm in &match_stmt.arms {
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_stmt(&arm.body);
                }
            }
            Stmt::Return(return_stmt) => {
                if let Some(value) = &return_stmt.value {
                    self.visit_expr(value);
                }
            }
            Stmt::Break(break_stmt) => {
                if let Some(value) = &break_stmt.value {
                    self.visit_expr(value);
                }
            }
            Stmt::Continue(continue_stmt) => {
                if let Some(value) = &continue_stmt.value {
                    self.visit_expr(value);
                }
            }
            Stmt::Throw(throw_stmt) => {
                self.visit_expr(&throw_stmt.exception);
            }
            Stmt::Try(try_stmt) => {
                self.visit_stmt(&try_stmt.try_block);
                for catch in &try_stmt.catch_clauses {
                    if let Some(error_type) = &catch.error_type {
                        self.visit_type(error_type);
                    }
                    self.visit_stmt(&catch.body);
                }
                if let Some(finally) = &try_stmt.finally_block {
                    self.visit_stmt(finally);
                }
            }
            Stmt::Block(block_stmt) => {
                for stmt in &block_stmt.statements {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::Const(const_decl) => {
                self.visit_expr(&const_decl.value);
                if let Some(type_ann) = &const_decl.type_annotation {
                    self.visit_type(type_ann);
                }
            }
            Stmt::Module(module_decl) => {
                for section in &module_decl.sections {
                    for item in &section.kind.items {
                        self.visit_stmt(item);
                    }
                }
            }
            _ => {} // Other statements don't need special handling
        }
    }
    
    fn visit_type(&mut self, ty: &AstNode<Type>) -> Self::Result {
        self.type_count += 1;
        
        match &ty.kind {
            Type::Named(named) => {
                for type_arg in &named.type_arguments {
                    self.visit_type(type_arg);
                }
            }
            Type::Generic(generic) => {
                self.visit_type(&generic.base_type);
                for param in &generic.parameters {
                    for bound in &param.bounds {
                        self.visit_type(bound);
                    }
                    if let Some(default) = &param.default {
                        self.visit_type(default);
                    }
                }
            }
            Type::Function(func) => {
                for param in &func.parameters {
                    self.visit_type(param);
                }
                self.visit_type(&func.return_type);
            }
            Type::Tuple(tuple) => {
                for element in &tuple.elements {
                    self.visit_type(element);
                }
            }
            Type::Array(array) => {
                self.visit_type(&array.element_type);
                if let Some(size) = &array.size {
                    self.visit_expr(size);
                }
            }
            Type::Union(union) => {
                for member in &union.members {
                    self.visit_type(member);
                }
            }
            Type::Intersection(intersection) => {
                for member in &intersection.members {
                    self.visit_type(member);
                }
            }
            Type::Semantic(semantic) => {
                self.visit_type(&semantic.base_type);
            }
            Type::Dependent(dependent) => {
                for param in &dependent.parameters {
                    self.visit_type(&param.parameter_type);
                    if let Some(default) = &param.default_value {
                        self.visit_expr(default);
                    }
                }
                self.visit_expr(&dependent.type_expression);
            }
            Type::Effect(effect) => {
                self.visit_type(&effect.base_type);
            }
            _ => {} // Primitive and error types don't need recursion
        }
    }
    
    fn visit_pattern(&mut self, pattern: &AstNode<Pattern>) -> Self::Result {
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
            _ => {} // Terminal patterns don't need recursion
        }
    }
}

#[test]
fn test_default_visitor() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expr = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    );
    
    let mut visitor = DefaultVisitor;
    visitor.visit_expr(&expr);
    
    // DefaultVisitor should not panic and return unit
}

#[test]
fn test_node_counter_simple_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expr = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        }),
        span,
        NodeId::new(1),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_expr(&expr);
    
    assert_eq!(counter.expr_count, 1);
    assert_eq!(counter.stmt_count, 0);
    assert_eq!(counter.type_count, 0);
    assert_eq!(counter.pattern_count, 0);
    assert_eq!(counter.total_count(), 1);
}

#[test]
fn test_node_counter_binary_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let left = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(10),
        }),
        span,
        NodeId::new(1),
    ));
    
    let right = Box::new(AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(20),
        }),
        span,
        NodeId::new(2),
    ));
    
    let binary_expr = AstNode::new(
        Expr::Binary(BinaryExpr {
            left,
            operator: BinaryOperator::Add,
            right,
        }),
        span,
        NodeId::new(3),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_expr(&binary_expr);
    
    // Should count: binary expr + left literal + right literal = 3
    assert_eq!(counter.expr_count, 3);
    assert_eq!(counter.total_count(), 3);
}

#[test]
fn test_node_counter_complex_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create expression: func(a + b, [1, 2, 3])
    let call_expr = AstNode::new(
        Expr::Call(CallExpr {
            callee: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("func"),
                }),
                span,
                NodeId::new(1),
            )),
            arguments: vec![
                AstNode::new(
                    Expr::Binary(BinaryExpr {
                        left: Box::new(AstNode::new(
                            Expr::Variable(VariableExpr {
                                name: Symbol::intern("a"),
                            }),
                            span,
                            NodeId::new(2),
                        )),
                        operator: BinaryOperator::Add,
                        right: Box::new(AstNode::new(
                            Expr::Variable(VariableExpr {
                                name: Symbol::intern("b"),
                            }),
                            span,
                            NodeId::new(3),
                        )),
                    }),
                    span,
                    NodeId::new(4),
                ),
                AstNode::new(
                    Expr::Array(ArrayExpr {
                        elements: vec![
                            AstNode::new(
                                Expr::Literal(LiteralExpr {
                                    value: LiteralValue::Integer(1),
                                }),
                                span,
                                NodeId::new(5),
                            ),
                            AstNode::new(
                                Expr::Literal(LiteralExpr {
                                    value: LiteralValue::Integer(2),
                                }),
                                span,
                                NodeId::new(6),
                            ),
                            AstNode::new(
                                Expr::Literal(LiteralExpr {
                                    value: LiteralValue::Integer(3),
                                }),
                                span,
                                NodeId::new(7),
                            ),
                        ],
                    }),
                    span,
                    NodeId::new(8),
                ),
            ],
            type_arguments: None,
            call_style: CallStyle::Function,
        }),
        span,
        NodeId::new(9),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_expr(&call_expr);
    
    // Should count:
    // 1. Call expression
    // 2. Variable "func"
    // 3. Binary expression (a + b)
    // 4. Variable "a"
    // 5. Variable "b"
    // 6. Array expression
    // 7. Literal 1
    // 8. Literal 2
    // 9. Literal 3
    // Total: 9 expressions
    assert_eq!(counter.expr_count, 9);
    assert_eq!(counter.total_count(), 9);
}

#[test]
fn test_node_counter_statement() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let stmt = AstNode::new(
        Stmt::Expression(ExpressionStmt {
            expression: AstNode::new(
                Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(42),
                }),
                span,
                NodeId::new(1),
            ),
        }),
        span,
        NodeId::new(2),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_stmt(&stmt);
    
    // Should count: 1 statement + 1 expression = 2 total
    assert_eq!(counter.stmt_count, 1);
    assert_eq!(counter.expr_count, 1);
    assert_eq!(counter.total_count(), 2);
}

#[test]
fn test_node_counter_function_declaration() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let param = stmt::Parameter {
        name: Symbol::intern("x"),
        type_annotation: Some(AstNode::new(
            Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
            span,
            NodeId::new(1),
        )),
        default_value: Some(AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(0),
            }),
            span,
            NodeId::new(2),
        )),
        is_mutable: false,
    };
    
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
    
    let func_decl = AstNode::new(
        Stmt::Function(FunctionDecl {
            name: Symbol::intern("test_func"),
            parameters: vec![param],
            return_type: Some(AstNode::new(
                Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
                span,
                NodeId::new(5),
            )),
            body,
            visibility: Visibility::Public,
            attributes: vec![],
            contracts: None,
            is_async: false,
        }),
        span,
        NodeId::new(6),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_stmt(&func_decl);
    
    // Should count:
    // Statements: function decl + return stmt = 2
    // Expressions: default value + return value = 2
    // Types: param type + return type = 2
    // Total: 6
    assert_eq!(counter.stmt_count, 2);
    assert_eq!(counter.expr_count, 2);
    assert_eq!(counter.type_count, 2);
    assert_eq!(counter.total_count(), 6);
}

#[test]
fn test_node_counter_pattern() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let pattern = AstNode::new(
        Pattern::Tuple(vec![
            AstNode::new(
                Pattern::Identifier(Symbol::intern("x")),
                span,
                NodeId::new(1),
            ),
            AstNode::new(
                Pattern::Literal(LiteralValue::Integer(42)),
                span,
                NodeId::new(2),
            ),
            AstNode::new(
                Pattern::Wildcard,
                span,
                NodeId::new(3),
            ),
        ]),
        span,
        NodeId::new(4),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_pattern(&pattern);
    
    // Should count: tuple pattern + identifier + literal + wildcard = 4
    assert_eq!(counter.pattern_count, 4);
    assert_eq!(counter.total_count(), 4);
}

#[test]
fn test_node_counter_match_expression() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let match_expr = AstNode::new(
        Expr::Match(MatchExpr {
            scrutinee: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("value"),
                }),
                span,
                NodeId::new(1),
            )),
            arms: vec![
                expr::MatchArm {
                    pattern: AstNode::new(
                        Pattern::Literal(LiteralValue::Integer(1)),
                        span,
                        NodeId::new(2),
                    ),
                    guard: Some(AstNode::new(
                        Expr::Literal(LiteralExpr {
                            value: LiteralValue::Boolean(true),
                        }),
                        span,
                        NodeId::new(3),
                    )),
                    body: AstNode::new(
                        Expr::Literal(LiteralExpr {
                            value: LiteralValue::String("one".to_string()),
                        }),
                        span,
                        NodeId::new(4),
                    ),
                },
                expr::MatchArm {
                    pattern: AstNode::new(
                        Pattern::Wildcard,
                        span,
                        NodeId::new(5),
                    ),
                    guard: None,
                    body: AstNode::new(
                        Expr::Literal(LiteralExpr {
                            value: LiteralValue::String("other".to_string()),
                        }),
                        span,
                        NodeId::new(6),
                    ),
                },
            ],
        }),
        span,
        NodeId::new(7),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_expr(&match_expr);
    
    // Should count:
    // Expressions: match + scrutinee + guard + body1 + body2 = 5
    // Patterns: literal pattern + wildcard = 2
    // Total: 7
    assert_eq!(counter.expr_count, 5);
    assert_eq!(counter.pattern_count, 2);
    assert_eq!(counter.total_count(), 7);
}

// Test mutable visitor
struct ExpressionRewriter;

impl AstVisitorMut for ExpressionRewriter {
    type Result = ();
    
    fn visit_expr_mut(&mut self, expr: &mut AstNode<Expr>) -> Self::Result {
        // Replace all integer literals with 0
        match &mut expr.kind {
            Expr::Literal(lit) => {
                if let LiteralValue::Integer(_) = &lit.value {
                    lit.value = LiteralValue::Integer(0);
                }
            }
            Expr::Binary(binary) => {
                self.visit_expr_mut(&mut binary.left);
                self.visit_expr_mut(&mut binary.right);
            }
            _ => {} // Handle other cases as needed
        }
    }
    
    fn visit_stmt_mut(&mut self, _stmt: &mut AstNode<Stmt>) -> Self::Result {
        // Not implemented for this test
    }
    
    fn visit_type_mut(&mut self, _ty: &mut AstNode<Type>) -> Self::Result {
        // Not implemented for this test
    }
    
    fn visit_pattern_mut(&mut self, _pattern: &mut AstNode<Pattern>) -> Self::Result {
        // Not implemented for this test
    }
}

#[test]
fn test_mutable_visitor() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let mut expr = AstNode::new(
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
    
    let mut rewriter = ExpressionRewriter;
    rewriter.visit_expr_mut(&mut expr);
    
    // Both integer literals should now be 0
    if let Expr::Binary(binary) = &expr.kind {
        if let Expr::Literal(left_lit) = &binary.left.kind {
            if let LiteralValue::Integer(n) = &left_lit.value {
                assert_eq!(*n, 0);
            }
        }
        
        if let Expr::Literal(right_lit) = &binary.right.kind {
            if let LiteralValue::Integer(n) = &right_lit.value {
                assert_eq!(*n, 0);
            }
        }
    }
}

// Test collector
struct VariableCollector {
    variables: Vec<Symbol>,
}

impl VariableCollector {
    fn new() -> Self {
        Self {
            variables: Vec::new(),
        }
    }
}

impl AstCollector<Symbol> for VariableCollector {
    fn collect_expr(&mut self, expr: &AstNode<Expr>) -> Vec<Symbol> {
        let mut vars = Vec::new();
        
        match &expr.kind {
            Expr::Variable(var) => {
                vars.push(var.name);
                self.variables.push(var.name);
            }
            Expr::Binary(binary) => {
                vars.extend(self.collect_expr(&binary.left));
                vars.extend(self.collect_expr(&binary.right));
            }
            Expr::Call(call) => {
                vars.extend(self.collect_expr(&call.callee));
                for arg in &call.arguments {
                    vars.extend(self.collect_expr(arg));
                }
            }
            _ => {} // Handle other cases as needed
        }
        
        vars
    }
    
    fn collect_stmt(&mut self, _stmt: &AstNode<Stmt>) -> Vec<Symbol> {
        // Not implemented for this test
        Vec::new()
    }
    
    fn collect_type(&mut self, _ty: &AstNode<Type>) -> Vec<Symbol> {
        // Not implemented for this test
        Vec::new()
    }
    
    fn collect_pattern(&mut self, _pattern: &AstNode<Pattern>) -> Vec<Symbol> {
        // Not implemented for this test
        Vec::new()
    }
}

#[test]
fn test_collector() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create expression: func(x, y + z)
    let expr = AstNode::new(
        Expr::Call(CallExpr {
            callee: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("func"),
                }),
                span,
                NodeId::new(1),
            )),
            arguments: vec![
                AstNode::new(
                    Expr::Variable(VariableExpr {
                        name: Symbol::intern("x"),
                    }),
                    span,
                    NodeId::new(2),
                ),
                AstNode::new(
                    Expr::Binary(BinaryExpr {
                        left: Box::new(AstNode::new(
                            Expr::Variable(VariableExpr {
                                name: Symbol::intern("y"),
                            }),
                            span,
                            NodeId::new(3),
                        )),
                        operator: BinaryOperator::Add,
                        right: Box::new(AstNode::new(
                            Expr::Variable(VariableExpr {
                                name: Symbol::intern("z"),
                            }),
                            span,
                            NodeId::new(4),
                        )),
                    }),
                    span,
                    NodeId::new(5),
                ),
            ],
            type_arguments: None,
            call_style: CallStyle::Function,
        }),
        span,
        NodeId::new(6),
    );
    
    let mut collector = VariableCollector::new();
    let collected = collector.collect_expr(&expr);
    
    // Should collect: func, x, y, z
    assert_eq!(collected.len(), 4);
    assert!(collected.contains(&Symbol::intern("func")));
    assert!(collected.contains(&Symbol::intern("x")));
    assert!(collected.contains(&Symbol::intern("y")));
    assert!(collected.contains(&Symbol::intern("z")));
    
    // Should also be stored in collector
    assert_eq!(collector.variables.len(), 4);
}

#[test]
fn test_visitor_with_nested_structures() {
    let source_id = SourceId::new(1);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create a complex nested structure
    let complex_expr = AstNode::new(
        Expr::Lambda(LambdaExpr {
            parameters: vec![stmt::Parameter {
                name: Symbol::intern("x"),
                type_annotation: Some(AstNode::new(
                    Type::Primitive(PrimitiveType::Integer(IntegerType::Signed(32))),
                    span,
                    NodeId::new(1),
                )),
                default_value: None,
                is_mutable: false,
            }],
            return_type: None,
            body: Box::new(AstNode::new(
                Expr::If(IfExpr {
                    condition: Box::new(AstNode::new(
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
                    )),
                    then_branch: Box::new(AstNode::new(
                        Expr::Variable(VariableExpr {
                            name: Symbol::intern("x"),
                        }),
                        span,
                        NodeId::new(5),
                    )),
                    else_branch: Some(Box::new(AstNode::new(
                        Expr::Literal(LiteralExpr {
                            value: LiteralValue::Integer(0),
                        }),
                        span,
                        NodeId::new(6),
                    ))),
                }),
                span,
                NodeId::new(7),
            )),
            is_async: false,
        }),
        span,
        NodeId::new(8),
    );
    
    let mut counter = NodeCounter::new();
    counter.visit_expr(&complex_expr);
    
    // Should count all nested nodes
    assert!(counter.expr_count > 5); // Lambda, if, binary, variables, literals
    assert!(counter.type_count > 0); // Parameter type
    assert!(counter.total_count() > 6);
} 