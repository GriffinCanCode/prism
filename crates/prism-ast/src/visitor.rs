//! Visitor pattern for AST traversal
//!
//! This module provides visitor patterns for traversing and transforming AST nodes.

use crate::{AstNode, Expr, Pattern, Stmt, Type};

/// Visitor trait for AST traversal
pub trait AstVisitor {
    /// Result type for visitor methods
    type Result;

    /// Visit an expression
    fn visit_expr(&mut self, expr: &AstNode<Expr>) -> Self::Result;

    /// Visit a statement
    fn visit_stmt(&mut self, stmt: &AstNode<Stmt>) -> Self::Result;

    /// Visit a type
    fn visit_type(&mut self, ty: &AstNode<Type>) -> Self::Result;

    /// Visit a pattern
    fn visit_pattern(&mut self, pattern: &AstNode<Pattern>) -> Self::Result;
}

/// Mutable visitor trait for AST transformation
pub trait AstVisitorMut {
    /// Result type for visitor methods
    type Result;

    /// Visit an expression mutably
    fn visit_expr_mut(&mut self, expr: &mut AstNode<Expr>) -> Self::Result;

    /// Visit a statement mutably
    fn visit_stmt_mut(&mut self, stmt: &mut AstNode<Stmt>) -> Self::Result;

    /// Visit a type mutably
    fn visit_type_mut(&mut self, ty: &mut AstNode<Type>) -> Self::Result;

    /// Visit a pattern mutably
    fn visit_pattern_mut(&mut self, pattern: &mut AstNode<Pattern>) -> Self::Result;
}

/// Default visitor implementation that walks the AST
pub struct DefaultVisitor;

impl AstVisitor for DefaultVisitor {
    type Result = ();

    fn visit_expr(&mut self, expr: &AstNode<Expr>) -> Self::Result {
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
                }
            }
            Expr::Lambda(lambda) => {
                self.visit_expr(&lambda.body);
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
            Expr::Try(try_expr) => {
                self.visit_expr(&try_expr.try_block);
                for clause in &try_expr.catch_clauses {
                    if let Some(pattern) = &clause.pattern {
                        self.visit_pattern(pattern);
                    }
                    self.visit_expr(&clause.body);
                }
                if let Some(finally_block) = &try_expr.finally_block {
                    self.visit_expr(finally_block);
                }
            }
            Expr::TypeAssertion(assertion) => {
                self.visit_expr(&assertion.expression);
                self.visit_type(&assertion.target_type);
            }
            Expr::Await(await_expr) => {
                self.visit_expr(&await_expr.expression);
            }
            Expr::Yield(yield_expr) => {
                if let Some(value) = &yield_expr.value {
                    self.visit_expr(value);
                }
            }
            Expr::Actor(actor_expr) => {
                self.visit_expr(&actor_expr.actor_impl);
                for cap in &actor_expr.capabilities {
                    self.visit_expr(cap);
                }
                if let Some(config) = &actor_expr.config {
                    self.visit_expr(config);
                }
            }
            Expr::Spawn(spawn_expr) => {
                self.visit_expr(&spawn_expr.expression);
                for cap in &spawn_expr.capabilities {
                    self.visit_expr(cap);
                }
                if let Some(priority) = &spawn_expr.priority {
                    self.visit_expr(priority);
                }
            }
            Expr::Channel(channel_expr) => {
                if let Some(channel) = &channel_expr.channel {
                    self.visit_expr(channel);
                }
                if let Some(value) = &channel_expr.value {
                    self.visit_expr(value);
                }
                if let Some(buffer_size) = &channel_expr.buffer_size {
                    self.visit_expr(buffer_size);
                }
            }
            Expr::Select(select_expr) => {
                for arm in &select_expr.arms {
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                }
                if let Some(default_arm) = &select_expr.default_arm {
                    if let Some(guard) = &default_arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&default_arm.body);
                }
            }
            Expr::Range(range_expr) => {
                self.visit_expr(&range_expr.start);
                self.visit_expr(&range_expr.end);
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
            Expr::Return(return_expr) => {
                if let Some(value) = &return_expr.value {
                    self.visit_expr(value);
                }
            }
            Expr::Break(break_expr) => {
                if let Some(value) = &break_expr.value {
                    self.visit_expr(value);
                }
            }
            Expr::Continue(continue_expr) => {
                if let Some(value) = &continue_expr.value {
                    self.visit_expr(value);
                }
            }
            Expr::Throw(throw_expr) => {
                self.visit_expr(&throw_expr.exception);
            }
            // Base cases
            Expr::Literal(_) | Expr::Variable(_) => {},
            Expr::Error(_) => {},
            // Python-specific expressions - placeholders for now
            Expr::FormattedString(_) => {},
            Expr::ListComprehension(_) => {},
            Expr::SetComprehension(_) => {},
            Expr::DictComprehension(_) => {},
            Expr::GeneratorExpression(_) => {},
            Expr::NamedExpression(_) => {},
            Expr::StarredExpression(_) => {},
            // Concurrency expressions
            // Expr::Actor(actor_expr) => {
            //     self.visit_expr(&actor_expr.actor_impl);
            //     for cap in &actor_expr.capabilities {
            //         self.visit_expr(cap);
            //     }
            //     if let Some(config) = &actor_expr.config {
            //         self.visit_expr(config);
            //     }
            // },
            // Expr::Spawn(spawn_expr) => {
            //     self.visit_expr(&spawn_expr.expression);
            //     for cap in &spawn_expr.capabilities {
            //         self.visit_expr(cap);
            //     }
            //     if let Some(priority) = &spawn_expr.priority {
            //         self.visit_expr(priority);
            //     }
            // },
            // Expr::Channel(channel_expr) => {
            //     if let Some(channel) = &channel_expr.channel {
            //         self.visit_expr(channel);
            //     }
            //     if let Some(value) = &channel_expr.value {
            //         self.visit_expr(value);
            //     }
            //     if let Some(buffer_size) = &channel_expr.buffer_size {
            //         self.visit_expr(buffer_size);
            //     }
            // },
            // Expr::Select(select_expr) => {
            //     for arm in &select_expr.arms {
            //         match &arm.pattern {
            //             crate::ChannelPattern::Receive { channel, .. } => {
            //                 self.visit_expr(channel);
            //             },
            //             crate::ChannelPattern::Send { channel, value } => {
            //                 self.visit_expr(channel);
            //                 self.visit_expr(value);
            //             },
            //         }
            //         if let Some(guard) = &arm.guard {
            //             self.visit_expr(guard);
            //         }
            //         self.visit_expr(&arm.body);
            //     }
            //     if let Some(default_arm) = &select_expr.default_arm {
            //         match &default_arm.pattern {
            //             crate::ChannelPattern::Receive { channel, .. } => {
            //                 self.visit_expr(channel);
            //             },
            //             crate::ChannelPattern::Send { channel, value } => {
            //                 self.visit_expr(channel);
            //                 self.visit_expr(value);
            //             },
            //         }
            //         if let Some(guard) = &default_arm.guard {
            //             self.visit_expr(guard);
            //         }
            //         self.visit_expr(&default_arm.body);
            //     }
            // },
        }
    }

    fn visit_stmt(&mut self, stmt: &AstNode<Stmt>) -> Self::Result {
        match &stmt.kind {
            Stmt::Expression(expr_stmt) => {
                self.visit_expr(&expr_stmt.expression);
            }
            Stmt::Variable(var_decl) => {
                if let Some(initializer) = &var_decl.initializer {
                    self.visit_expr(initializer);
                }
                if let Some(type_annotation) = &var_decl.type_annotation {
                    self.visit_type(type_annotation);
                }
            }
            Stmt::Function(func_decl) => {
                if let Some(return_type) = &func_decl.return_type {
                    self.visit_type(return_type);
                }
                if let Some(body) = &func_decl.body {
                    self.visit_stmt(body);
                }
            }
            Stmt::Type(type_decl) => {
                // Type declarations would be visited here
            }
            Stmt::Module(module_decl) => {
                for section in &module_decl.sections {
                    for item in &section.kind.items {
                        self.visit_stmt(item);
                    }
                }
            }
            Stmt::Section(section_decl) => {
                for item in &section_decl.items {
                    self.visit_stmt(item);
                }
            }
            Stmt::Actor(actor_decl) => {
                for field in &actor_decl.state_fields {
                    if let Some(type_annotation) = &field.type_annotation {
                        self.visit_type(type_annotation);
                    }
                    if let Some(default_value) = &field.default_value {
                        self.visit_expr(default_value);
                    }
                }
                if let Some(message_type) = &actor_decl.message_type {
                    self.visit_type(message_type);
                }
                for handler in &actor_decl.message_handlers {
                    self.visit_stmt(&handler.body);
                    for effect in &handler.effects {
                        self.visit_expr(effect);
                    }
                }
                for capability in &actor_decl.capabilities {
                    self.visit_expr(capability);
                }
                for effect in &actor_decl.effects {
                    self.visit_expr(effect);
                }
                for hook in &actor_decl.lifecycle_hooks {
                    self.visit_stmt(&hook.body);
                }
            }
            Stmt::Import(_) | Stmt::Export(_) | Stmt::Const(_) => {
                // These would be handled specifically
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
                for catch_clause in &try_stmt.catch_clauses {
                    if let Some(error_type) = &catch_clause.error_type {
                        self.visit_type(error_type);
                    }
                    self.visit_stmt(&catch_clause.body);
                }
                if let Some(finally_block) = &try_stmt.finally_block {
                    self.visit_stmt(finally_block);
                }
            }
            Stmt::Block(block_stmt) => {
                for stmt in &block_stmt.statements {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::Error(_) => {},
            Stmt::Actor(actor_decl) => {
                // Visit message handlers
                for handler in &actor_decl.message_handlers {
                    self.visit_stmt(&handler.body);
                }
                // Visit capabilities
                for cap in &actor_decl.capabilities {
                    self.visit_expr(cap);
                }
                // Visit effects
                for effect in &actor_decl.effects {
                    self.visit_expr(effect);
                }
                // Visit lifecycle hooks
                for hook in &actor_decl.lifecycle_hooks {
                    self.visit_stmt(&hook.body);
                }
            },
        }
    }

    fn visit_type(&mut self, _ty: &AstNode<Type>) -> Self::Result {
        // Type visiting would be implemented here
    }

    fn visit_pattern(&mut self, pattern: &AstNode<Pattern>) -> Self::Result {
        match &pattern.kind {
            Pattern::Tuple(patterns) => {
                for pattern in patterns {
                    self.visit_pattern(pattern);
                }
            }
            Pattern::Array(patterns) => {
                for pattern in patterns {
                    self.visit_pattern(pattern);
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
            // Base cases
            Pattern::Wildcard | Pattern::Identifier(_) | Pattern::Literal(_) | Pattern::Rest(_) => {}
        }
    }
}

/// Trait for collecting information during AST traversal
pub trait AstCollector<T> {
    /// Collect information from an expression
    fn collect_expr(&mut self, expr: &AstNode<Expr>) -> Vec<T>;

    /// Collect information from a statement
    fn collect_stmt(&mut self, stmt: &AstNode<Stmt>) -> Vec<T>;

    /// Collect information from a type
    fn collect_type(&mut self, ty: &AstNode<Type>) -> Vec<T>;

    /// Collect information from a pattern
    fn collect_pattern(&mut self, pattern: &AstNode<Pattern>) -> Vec<T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiteralExpr, LiteralValue, VariableExpr};
    use prism_common::{span::Span, symbol::Symbol, NodeId};

    #[test]
    fn test_default_visitor() {
        let expr = AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(42),
            }),
            Span::dummy(),
            NodeId::new(1),
        );

        let mut visitor = DefaultVisitor;
        visitor.visit_expr(&expr);
        // Should not panic
    }

    #[test]
    fn test_visitor_with_binary_expr() {
        let left = AstNode::new(
            Expr::Variable(VariableExpr {
                name: Symbol::intern("x"),
            }),
            Span::dummy(),
            NodeId::new(1),
        );

        let right = AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(42),
            }),
            Span::dummy(),
            NodeId::new(2),
        );

        let binary = AstNode::new(
            Expr::Binary(crate::BinaryExpr {
                left: Box::new(left),
                operator: crate::BinaryOperator::Add,
                right: Box::new(right),
            }),
            Span::dummy(),
            NodeId::new(3),
        );

        let mut visitor = DefaultVisitor;
        visitor.visit_expr(&binary);
        // Should not panic and should visit both operands
    }
} 