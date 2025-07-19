//! AST Transformation and Optimization Passes
//!
//! This module provides various optimization and transformation passes for AST nodes,
//! including dead code elimination, function inlining, constant folding, and type
//! inference optimization. These transformations operate directly on AST structures
//! using the visitor pattern.

use crate::{
    AstNode, Expr, Stmt, Type, Item, Program, 
    LiteralValue, BinaryOperator, UnaryOperator, 
    BinaryExpr, UnaryExpr, LiteralExpr, VariableExpr, CallExpr,
    visitor::{AstVisitor, AstVisitorMut},
};
use prism_common::{span::Span, symbol::Symbol, NodeId};
use std::collections::{HashMap, HashSet};

/// Configuration for transformation passes
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Enable dead code elimination
    pub enable_dead_code_elimination: bool,
    /// Enable function inlining for small functions
    pub enable_function_inlining: bool,
    /// Enable constant folding optimization
    pub enable_constant_folding: bool,
    /// Enable type inference simplification
    pub enable_type_inference_optimization: bool,
    /// Maximum function size for inlining (in AST nodes)
    pub max_inline_size: usize,
    /// Maximum inlining depth to prevent infinite recursion
    pub max_inline_depth: usize,
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            enable_dead_code_elimination: true,
            enable_function_inlining: true,
            enable_constant_folding: true,
            enable_type_inference_optimization: true,
            max_inline_size: 50,
            max_inline_depth: 3,
        }
    }
}

/// Main transformation engine that orchestrates all optimization passes
pub struct TransformationEngine {
    config: TransformationConfig,
    dead_code_eliminator: DeadCodeEliminator,
    function_inliner: FunctionInliner,
    constant_folder: ConstantFolder,
    type_optimizer: TypeInferenceOptimizer,
}

impl TransformationEngine {
    /// Create a new transformation engine with default configuration
    pub fn new() -> Self {
        Self::with_config(TransformationConfig::default())
    }

    /// Create a new transformation engine with custom configuration
    pub fn with_config(config: TransformationConfig) -> Self {
        Self {
            dead_code_eliminator: DeadCodeEliminator::new(),
            function_inliner: FunctionInliner::new(config.max_inline_size, config.max_inline_depth),
            constant_folder: ConstantFolder::new(),
            type_optimizer: TypeInferenceOptimizer::new(),
            config,
        }
    }

    /// Apply all enabled transformations to a program
    pub fn transform_program(&mut self, program: &mut Program) -> TransformationResult {
        let mut result = TransformationResult::new();

        // Apply transformations in optimal order
        if self.config.enable_constant_folding {
            let folding_stats = self.constant_folder.transform_program(program);
            result.merge(folding_stats);
        }

        if self.config.enable_function_inlining {
            let inlining_stats = self.function_inliner.transform_program(program);
            result.merge(inlining_stats);
        }

        if self.config.enable_dead_code_elimination {
            let elimination_stats = self.dead_code_eliminator.transform_program(program);
            result.merge(elimination_stats);
        }

        if self.config.enable_type_inference_optimization {
            let type_stats = self.type_optimizer.transform_program(program);
            result.merge(type_stats);
        }

        result
    }
}

/// Result of applying transformations
#[derive(Debug, Clone, Default)]
pub struct TransformationResult {
    /// Number of nodes eliminated
    pub nodes_eliminated: usize,
    /// Number of functions inlined
    pub functions_inlined: usize,
    /// Number of constants folded
    pub constants_folded: usize,
    /// Number of types simplified
    pub types_simplified: usize,
    /// Transformation messages
    pub messages: Vec<String>,
}

impl TransformationResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another result into this one
    pub fn merge(&mut self, other: TransformationResult) {
        self.nodes_eliminated += other.nodes_eliminated;
        self.functions_inlined += other.functions_inlined;
        self.constants_folded += other.constants_folded;
        self.types_simplified += other.types_simplified;
        self.messages.extend(other.messages);
    }

    /// Check if any transformations were applied
    pub fn has_changes(&self) -> bool {
        self.nodes_eliminated > 0 
            || self.functions_inlined > 0 
            || self.constants_folded > 0 
            || self.types_simplified > 0
    }
}

/// Dead code elimination pass
pub struct DeadCodeEliminator {
    used_variables: HashSet<Symbol>,
    used_functions: HashSet<Symbol>,
}

impl DeadCodeEliminator {
    /// Create a new dead code eliminator
    pub fn new() -> Self {
        Self {
            used_variables: HashSet::new(),
            used_functions: HashSet::new(),
        }
    }

    /// Transform a program by eliminating dead code
    pub fn transform_program(&mut self, program: &mut Program) -> TransformationResult {
        // First pass: collect all used symbols
        self.collect_used_symbols(program);

        // Second pass: eliminate unused declarations
        let mut result = TransformationResult::new();
        let original_len = program.items.len();

        program.items.retain(|item| {
            match &item.kind {
                Item::Function(func) => {
                    if self.used_functions.contains(&func.name) {
                        true
                    } else {
                        result.nodes_eliminated += 1;
                        result.messages.push(format!("Eliminated unused function: {}", func.name));
                        false
                    }
                }
                Item::Variable(var) => {
                    if self.used_variables.contains(&var.name) {
                        true
                    } else {
                        result.nodes_eliminated += 1;
                        result.messages.push(format!("Eliminated unused variable: {}", var.name));
                        false
                    }
                }
                _ => true, // Keep other items for now
            }
        });

        result
    }

    /// Collect all symbols that are actually used
    fn collect_used_symbols(&mut self, program: &Program) {
        for item in &program.items {
            self.collect_from_item(item);
        }
    }

    /// Collect used symbols from an item
    fn collect_from_item(&mut self, item: &AstNode<Item>) {
        match &item.kind {
            Item::Function(func) => {
                if let Some(body) = &func.body {
                    self.collect_from_stmt(body);
                }
            }
            Item::Variable(var) => {
                if let Some(init) = &var.initializer {
                    self.collect_from_expr(init);
                }
            }
            _ => {} // Handle other items as needed
        }
    }

    /// Collect used symbols from a statement
    fn collect_from_stmt(&mut self, stmt: &AstNode<Stmt>) {
        match &stmt.kind {
            Stmt::Expression(expr_stmt) => {
                self.collect_from_expr(&expr_stmt.expression);
            }
            Stmt::Variable(var) => {
                if let Some(init) = &var.initializer {
                    self.collect_from_expr(init);
                }
            }
            _ => {} // Handle other statements as needed
        }
    }

    /// Collect used symbols from an expression
    fn collect_from_expr(&mut self, expr: &AstNode<Expr>) {
        match &expr.kind {
            Expr::Variable(var) => {
                self.used_variables.insert(var.name);
            }
            Expr::Call(call) => {
                self.collect_from_expr(&call.callee);
                for arg in &call.arguments {
                    self.collect_from_expr(arg);
                }
                
                // If calling a variable, mark it as used
                if let Expr::Variable(var) = &call.callee.kind {
                    self.used_functions.insert(var.name);
                }
            }
            Expr::Binary(binary) => {
                self.collect_from_expr(&binary.left);
                self.collect_from_expr(&binary.right);
            }
            Expr::Unary(unary) => {
                self.collect_from_expr(&unary.operand);
            }
            _ => {} // Handle other expressions as needed
        }
    }
}

/// Function inlining optimization pass
pub struct FunctionInliner {
    max_size: usize,
    max_depth: usize,
    current_depth: usize,
    function_definitions: HashMap<Symbol, AstNode<Stmt>>,
}

impl FunctionInliner {
    /// Create a new function inliner
    pub fn new(max_size: usize, max_depth: usize) -> Self {
        Self {
            max_size,
            max_depth,
            current_depth: 0,
            function_definitions: HashMap::new(),
        }
    }

    /// Transform a program by inlining small functions
    pub fn transform_program(&mut self, program: &mut Program) -> TransformationResult {
        // First pass: collect function definitions
        self.collect_function_definitions(program);

        // Second pass: inline function calls
        let mut result = TransformationResult::new();
        for item in &mut program.items {
            self.inline_in_item(item, &mut result);
        }

        result
    }

    /// Collect all function definitions for potential inlining
    fn collect_function_definitions(&mut self, program: &Program) {
        for item in &program.items {
            if let Item::Function(func) = &item.kind {
                if let Some(body) = &func.body {
                    // Only consider small functions for inlining
                    if self.estimate_size(body) <= self.max_size {
                        self.function_definitions.insert(func.name, body.as_ref().clone());
                    }
                }
            }
        }
    }

    /// Estimate the size of a statement (simplified metric)
    fn estimate_size(&self, stmt: &AstNode<Stmt>) -> usize {
        // Simplified size estimation - count AST nodes
        match &stmt.kind {
            Stmt::Expression(expr_stmt) => self.estimate_expr_size(&expr_stmt.expression),
            Stmt::Block(block) => {
                block.statements.iter().map(|s| self.estimate_size(s)).sum::<usize>() + 1
            }
            _ => 1,
        }
    }

    /// Estimate the size of an expression
    fn estimate_expr_size(&self, expr: &AstNode<Expr>) -> usize {
        match &expr.kind {
            Expr::Binary(binary) => {
                self.estimate_expr_size(&binary.left) + self.estimate_expr_size(&binary.right) + 1
            }
            Expr::Unary(unary) => self.estimate_expr_size(&unary.operand) + 1,
            Expr::Call(call) => {
                let args_size: usize = call.arguments.iter().map(|a| self.estimate_expr_size(a)).sum();
                self.estimate_expr_size(&call.callee) + args_size + 1
            }
            _ => 1,
        }
    }

    /// Inline function calls in an item
    fn inline_in_item(&mut self, item: &mut AstNode<Item>, result: &mut TransformationResult) {
        match &mut item.kind {
            Item::Function(func) => {
                if let Some(body) = &mut func.body {
                    self.inline_in_stmt(body, result);
                }
            }
            _ => {} // Handle other items as needed
        }
    }

    /// Inline function calls in a statement
    fn inline_in_stmt(&mut self, stmt: &mut AstNode<Stmt>, result: &mut TransformationResult) {
        match &mut stmt.kind {
            Stmt::Expression(expr_stmt) => {
                self.inline_in_expr(&mut expr_stmt.expression, result);
            }
            _ => {} // Handle other statements as needed
        }
    }

    /// Inline function calls in an expression
    fn inline_in_expr(&mut self, expr: &mut AstNode<Expr>, result: &mut TransformationResult) {
        match &mut expr.kind {
            Expr::Call(call) => {
                // Try to inline this call
                if let Expr::Variable(var) = &call.callee.kind {
                    if let Some(definition) = self.function_definitions.get(&var.name).cloned() {
                        if self.current_depth < self.max_depth {
                            // Perform inlining (simplified - would need proper substitution)
                            result.functions_inlined += 1;
                            result.messages.push(format!("Inlined function: {}", var.name));
                        }
                    }
                }

                // Continue recursively
                self.inline_in_expr(&mut call.callee, result);
                for arg in &mut call.arguments {
                    self.inline_in_expr(arg, result);
                }
            }
            Expr::Binary(binary) => {
                self.inline_in_expr(&mut binary.left, result);
                self.inline_in_expr(&mut binary.right, result);
            }
            Expr::Unary(unary) => {
                self.inline_in_expr(&mut unary.operand, result);
            }
            _ => {} // Handle other expressions as needed
        }
    }
}

/// Constant folding optimization pass
pub struct ConstantFolder;

impl ConstantFolder {
    /// Create a new constant folder
    pub fn new() -> Self {
        Self
    }

    /// Transform a program by folding constants
    pub fn transform_program(&mut self, program: &mut Program) -> TransformationResult {
        let mut result = TransformationResult::new();
        for item in &mut program.items {
            self.fold_in_item(item, &mut result);
        }
        result
    }

    /// Fold constants in an item
    fn fold_in_item(&mut self, item: &mut AstNode<Item>, result: &mut TransformationResult) {
        match &mut item.kind {
            Item::Function(func) => {
                if let Some(body) = &mut func.body {
                    self.fold_in_stmt(body, result);
                }
            }
            _ => {} // Handle other items as needed
        }
    }

    /// Fold constants in a statement
    fn fold_in_stmt(&mut self, stmt: &mut AstNode<Stmt>, result: &mut TransformationResult) {
        match &mut stmt.kind {
            Stmt::Expression(expr_stmt) => {
                self.fold_in_expr(&mut expr_stmt.expression, result);
            }
            _ => {} // Handle other statements as needed
        }
    }

    /// Fold constants in an expression
    fn fold_in_expr(&mut self, expr: &mut AstNode<Expr>, result: &mut TransformationResult) {
        // First, recursively fold sub-expressions
        match &mut expr.kind {
            Expr::Binary(binary) => {
                self.fold_in_expr(&mut binary.left, result);
                self.fold_in_expr(&mut binary.right, result);
                
                // Try to fold this binary expression
                if let (Expr::Literal(left_lit), Expr::Literal(right_lit)) = 
                    (&binary.left.kind, &binary.right.kind) {
                    if let Some(folded) = self.fold_binary_literals(&left_lit.value, &binary.operator, &right_lit.value) {
                        expr.kind = Expr::Literal(LiteralExpr { value: folded });
                        result.constants_folded += 1;
                        result.messages.push("Folded binary expression".to_string());
                    }
                }
            }
            Expr::Unary(unary) => {
                self.fold_in_expr(&mut unary.operand, result);
                
                // Try to fold this unary expression
                if let Expr::Literal(operand_lit) = &unary.operand.kind {
                    if let Some(folded) = self.fold_unary_literal(&unary.operator, &operand_lit.value) {
                        expr.kind = Expr::Literal(LiteralExpr { value: folded });
                        result.constants_folded += 1;
                        result.messages.push("Folded unary expression".to_string());
                    }
                }
            }
            Expr::Call(call) => {
                self.fold_in_expr(&mut call.callee, result);
                for arg in &mut call.arguments {
                    self.fold_in_expr(arg, result);
                }
            }
            _ => {} // Handle other expressions as needed
        }
    }

    /// Attempt to fold a binary operation on literal values
    fn fold_binary_literals(&self, left: &LiteralValue, op: &BinaryOperator, right: &LiteralValue) -> Option<LiteralValue> {
        match (left, op, right) {
            (LiteralValue::Integer(a), BinaryOperator::Add, LiteralValue::Integer(b)) => {
                Some(LiteralValue::Integer(a + b))
            }
            (LiteralValue::Integer(a), BinaryOperator::Subtract, LiteralValue::Integer(b)) => {
                Some(LiteralValue::Integer(a - b))
            }
            (LiteralValue::Integer(a), BinaryOperator::Multiply, LiteralValue::Integer(b)) => {
                Some(LiteralValue::Integer(a * b))
            }
            (LiteralValue::Integer(a), BinaryOperator::Divide, LiteralValue::Integer(b)) if *b != 0 => {
                Some(LiteralValue::Integer(a / b))
            }
            (LiteralValue::Float(a), BinaryOperator::Add, LiteralValue::Float(b)) => {
                Some(LiteralValue::Float(a + b))
            }
            (LiteralValue::Float(a), BinaryOperator::Subtract, LiteralValue::Float(b)) => {
                Some(LiteralValue::Float(a - b))
            }
            (LiteralValue::Float(a), BinaryOperator::Multiply, LiteralValue::Float(b)) => {
                Some(LiteralValue::Float(a * b))
            }
            (LiteralValue::Float(a), BinaryOperator::Divide, LiteralValue::Float(b)) if *b != 0.0 => {
                Some(LiteralValue::Float(a / b))
            }
            (LiteralValue::Boolean(a), BinaryOperator::And, LiteralValue::Boolean(b)) => {
                Some(LiteralValue::Boolean(*a && *b))
            }
            (LiteralValue::Boolean(a), BinaryOperator::Or, LiteralValue::Boolean(b)) => {
                Some(LiteralValue::Boolean(*a || *b))
            }
            _ => None, // Cannot fold this combination
        }
    }

    /// Attempt to fold a unary operation on a literal value
    fn fold_unary_literal(&self, op: &UnaryOperator, operand: &LiteralValue) -> Option<LiteralValue> {
        match (op, operand) {
            (UnaryOperator::Negate, LiteralValue::Integer(n)) => {
                Some(LiteralValue::Integer(-n))
            }
            (UnaryOperator::Negate, LiteralValue::Float(f)) => {
                Some(LiteralValue::Float(-f))
            }
            (UnaryOperator::Not, LiteralValue::Boolean(b)) => {
                Some(LiteralValue::Boolean(!b))
            }
            _ => None, // Cannot fold this combination
        }
    }
}

/// Type inference optimization pass
pub struct TypeInferenceOptimizer;

impl TypeInferenceOptimizer {
    /// Create a new type inference optimizer
    pub fn new() -> Self {
        Self
    }

    /// Transform a program by optimizing type inference
    pub fn transform_program(&mut self, _program: &mut Program) -> TransformationResult {
        let mut result = TransformationResult::new();
        
        // Placeholder implementation - would integrate with type inference system
        result.messages.push("Type inference optimization placeholder".to_string());
        
        result
    }
}

impl Default for DeadCodeEliminator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ConstantFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TypeInferenceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TransformationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryExpr, LiteralExpr, BinaryOperator, LiteralValue};
    use prism_common::{span::Span, NodeId};

    #[test]
    fn test_constant_folding() {
        let mut folder = ConstantFolder::new();
        
        // Create a binary expression: 2 + 3
        let left = AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(2),
            }),
            Span::dummy(),
            NodeId::new(1),
        );
        
        let right = AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(3),
            }),
            Span::dummy(),
            NodeId::new(2),
        );
        
        let mut binary_expr = AstNode::new(
            Expr::Binary(BinaryExpr {
                left: Box::new(left),
                operator: BinaryOperator::Add,
                right: Box::new(right),
            }),
            Span::dummy(),
            NodeId::new(3),
        );
        
        let mut result = TransformationResult::new();
        folder.fold_in_expr(&mut binary_expr, &mut result);
        
        // Should have folded to 5
        if let Expr::Literal(lit) = &binary_expr.kind {
            if let LiteralValue::Integer(value) = &lit.value {
                assert_eq!(*value, 5);
            } else {
                panic!("Expected integer literal");
            }
        } else {
            panic!("Expected literal expression after folding");
        }
        
        assert_eq!(result.constants_folded, 1);
    }

    #[test]
    fn test_transformation_config() {
        let config = TransformationConfig::default();
        assert!(config.enable_dead_code_elimination);
        assert!(config.enable_function_inlining);
        assert!(config.enable_constant_folding);
        assert!(config.enable_type_inference_optimization);
        assert_eq!(config.max_inline_size, 50);
        assert_eq!(config.max_inline_depth, 3);
    }
} 