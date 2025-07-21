//! Expression evaluator for constraint predicates
//!
//! This module provides evaluation of expressions used in constraint definitions,
//! allowing for complex constraint predicates and business rules.

use crate::{ConstraintValue, ConstraintResult, ConstraintError, ValidationContext};
use prism_ast::{Expr, LiteralValue, BinaryOperator, UnaryOperator, AstNode};
use prism_common::{span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Operation counter for tracking evaluation complexity
#[derive(Debug, Clone)]
struct OperationCounter {
    count: usize,
}

impl OperationCounter {
    /// Create a new operation counter
    fn new() -> Self {
        Self { count: 0 }
    }

    /// Increment the operation count
    fn increment(&mut self) {
        self.count += 1;
    }

    /// Get the current operation count
    fn count(&self) -> usize {
        self.count
    }
}

/// Expression evaluator for constraint predicates
#[derive(Debug)]
pub struct ExpressionEvaluator {
    /// Configuration for evaluation
    config: EvaluationConfig,
    /// Built-in function registry
    builtin_functions: HashMap<String, BuiltinFunction>,
}

/// Configuration for expression evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Maximum evaluation time
    pub max_duration: Duration,
    /// Enable overflow checking
    pub check_overflow: bool,
    /// Enable type coercion
    pub enable_coercion: bool,
}

/// Built-in function type
type BuiltinFunction = fn(&[ConstraintValue]) -> ConstraintResult<ConstraintValue>;

/// Context for expression evaluation
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    /// Variable bindings
    variables: HashMap<String, ConstraintValue>,
    /// Function bindings
    functions: HashMap<String, BuiltinFunction>,
    /// Current evaluation depth
    depth: usize,
    /// Start time for timeout checking
    start_time: Instant,
    /// Current span context for error reporting
    current_span: Option<Span>,
}

/// Result of expression evaluation
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// The evaluated value
    pub value: ConstraintValue,
    /// Any warnings generated during evaluation
    pub warnings: Vec<String>,
    /// Performance metrics
    pub performance: EvaluationPerformance,
}

/// Performance metrics for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationPerformance {
    /// Time taken for evaluation
    pub duration: Duration,
    /// Number of operations performed
    pub operations: usize,
    /// Maximum depth reached
    pub max_depth: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            max_duration: Duration::from_millis(1000),
            check_overflow: true,
            enable_coercion: true,
        }
    }
}

impl EvaluationContext {
    /// Create a new evaluation context
    pub fn new() -> Self {
        let mut context = Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            depth: 0,
            start_time: Instant::now(),
            current_span: None,
        };
        
        // Add built-in functions
        context.add_builtin_functions();
        context
    }

    /// Create context from validation context
    pub fn from_validation_context(validation_context: &ValidationContext) -> Self {
        let mut context = Self::new();
        
        // Copy variables from validation context
        for (name, value) in validation_context.iter_variables() {
            context.variables.insert(name.to_string(), value.clone());
        }
        
        context
    }

    /// Set current span context for error reporting
    pub fn with_span(mut self, span: Span) -> Self {
        self.current_span = Some(span);
        self
    }

    /// Add a variable binding
    pub fn add_variable(&mut self, name: impl Into<String>, value: ConstraintValue) {
        self.variables.insert(name.into(), value);
    }

    /// Get a variable value
    pub fn get_variable(&self, name: &str) -> Option<&ConstraintValue> {
        self.variables.get(name)
    }

    /// Get a function
    pub fn get_function(&self, name: &str) -> Option<&BuiltinFunction> {
        self.functions.get(name)
    }

    /// Get available variable names
    pub fn available_variables(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Get available function names
    pub fn available_functions(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }

    /// Add built-in functions to the context
    fn add_builtin_functions(&mut self) {
        self.functions.insert("length".to_string(), builtin_length);
        self.functions.insert("abs".to_string(), builtin_abs);
        self.functions.insert("min".to_string(), builtin_min);
        self.functions.insert("max".to_string(), builtin_max);
        self.functions.insert("is_string".to_string(), builtin_is_string);
        self.functions.insert("is_number".to_string(), builtin_is_number);
        self.functions.insert("is_boolean".to_string(), builtin_is_boolean);
        self.functions.insert("is_null".to_string(), builtin_is_null);
    }
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpressionEvaluator {
    /// Create a new expression evaluator
    pub fn new() -> Self {
        Self::with_config(EvaluationConfig::default())
    }

    /// Create evaluator with configuration
    pub fn with_config(config: EvaluationConfig) -> Self {
        let mut evaluator = Self {
            config,
            builtin_functions: HashMap::new(),
        };
        evaluator.register_builtin_functions();
        evaluator
    }

    /// Evaluate an expression in the given context
    pub fn evaluate(&self, expr: &AstNode<Expr>, context: &EvaluationContext) -> ConstraintResult<EvaluationResult> {
        let start_time = Instant::now();
        let mut context = context.clone().with_span(expr.span);
        let mut operation_counter = OperationCounter::new();
        
        let result = self.evaluate_with_depth_and_counter(&expr.kind, &mut context, 0, &mut operation_counter)?;
        
        let duration = start_time.elapsed();
        Ok(EvaluationResult {
            value: result.value,
            warnings: result.warnings,
            performance: EvaluationPerformance {
                duration,
                operations: operation_counter.count(),
                max_depth: context.depth,
            },
        })
    }

    /// Evaluate with depth tracking
    fn evaluate_with_depth(&self, expr: &Expr, context: &mut EvaluationContext, depth: usize) -> ConstraintResult<EvaluationResult> {
        let mut operation_counter = OperationCounter::new();
        self.evaluate_with_depth_and_counter(expr, context, depth, &mut operation_counter)
    }

    /// Evaluate with depth tracking and operation counting
    fn evaluate_with_depth_and_counter(&self, expr: &Expr, context: &mut EvaluationContext, depth: usize, operations: &mut OperationCounter) -> ConstraintResult<EvaluationResult> {
        operations.increment();
        
        // Check depth limit
        if depth > self.config.max_depth {
            return Err(ConstraintError::EvaluationError {
                location: context.current_span.unwrap_or_else(Span::dummy),
                message: format!("Maximum evaluation depth {} exceeded", self.config.max_depth),
            });
        }

        // Check timeout
        if context.start_time.elapsed() > self.config.max_duration {
            return Err(ConstraintError::EvaluationError {
                location: context.current_span.unwrap_or_else(Span::dummy),
                message: "Evaluation timeout exceeded".to_string(),
            });
        }

        context.depth = context.depth.max(depth);

        match expr {
            Expr::Literal(literal) => self.evaluate_literal(literal, context),
            Expr::Variable(var) => self.evaluate_variable(&var.name, context),
            Expr::Binary(binary) => self.evaluate_binary_with_counter(binary, context, depth, operations),
            Expr::Unary(unary) => self.evaluate_unary_with_counter(unary, context, depth, operations),
            Expr::Call(call) => self.evaluate_call_with_counter(call, context, depth, operations),
            _ => Err(ConstraintError::EvaluationError {
                location: context.current_span.unwrap_or_else(Span::dummy),
                message: format!("Unsupported expression type for constraint evaluation: {:?}", expr),
            }),
        }
    }

    /// Evaluate a literal expression
    fn evaluate_literal(&self, literal: &prism_ast::LiteralExpr, _context: &EvaluationContext) -> ConstraintResult<EvaluationResult> {
        let value = match &literal.value {
            LiteralValue::Integer(i) => ConstraintValue::Integer(*i),
            LiteralValue::Float(f) => ConstraintValue::Float(*f),
            LiteralValue::String(s) => ConstraintValue::String(s.clone()),
            LiteralValue::Boolean(b) => ConstraintValue::Boolean(*b),
            LiteralValue::Null => ConstraintValue::Null,
            _ => return Err(ConstraintError::EvaluationError {
                location: _context.current_span.unwrap_or_else(Span::dummy),
                message: format!("Unsupported literal type: {:?}", literal.value),
            }),
        };
        
        Ok(EvaluationResult::new(value))
    }

    /// Evaluate a variable expression
    fn evaluate_variable(&self, name: &Symbol, context: &EvaluationContext) -> ConstraintResult<EvaluationResult> {
        let var_name = name.resolve().unwrap_or_else(|| format!("sym:{:?}", name.raw()));
        
        if let Some(value) = context.get_variable(&var_name) {
            Ok(EvaluationResult::new(value.clone()))
        } else {
            Err(ConstraintError::UndefinedVariable {
                location: context.current_span.unwrap_or_else(Span::dummy),
                variable: var_name,
                available_variables: context.available_variables(),
            })
        }
    }

    /// Evaluate a binary expression with operation counting
    fn evaluate_binary_with_counter(&self, binary: &prism_ast::BinaryExpr, context: &mut EvaluationContext, depth: usize, operations: &mut OperationCounter) -> ConstraintResult<EvaluationResult> {
        // Update context with left operand span for better error reporting
        let mut left_context = context.clone().with_span(binary.left.span);
        let left_result = self.evaluate_with_depth_and_counter(&binary.left.kind, &mut left_context, depth + 1, operations)?;
        
        // Update context with right operand span
        let mut right_context = context.clone().with_span(binary.right.span);
        let right_result = self.evaluate_with_depth_and_counter(&binary.right.kind, &mut right_context, depth + 1, operations)?;
        
        operations.increment(); // Count the binary operation
        
        let left_val = &left_result.value;
        let right_val = &right_result.value;
        
        let result_value = match binary.operator {
            BinaryOperator::Add => self.add_values(left_val, right_val, context)?,
            BinaryOperator::Subtract => self.subtract_values(left_val, right_val, context)?,
            BinaryOperator::Multiply => self.multiply_values(left_val, right_val, context)?,
            BinaryOperator::Divide => self.divide_values(left_val, right_val, context)?,
            BinaryOperator::Modulo => self.modulo_values(left_val, right_val, context)?,
            BinaryOperator::Equal => ConstraintValue::Boolean(self.values_equal(left_val, right_val)),
            BinaryOperator::NotEqual => ConstraintValue::Boolean(!self.values_equal(left_val, right_val)),
            BinaryOperator::Less => ConstraintValue::Boolean(self.compare_values(left_val, right_val, context)? < 0),
            BinaryOperator::LessEqual => ConstraintValue::Boolean(self.compare_values(left_val, right_val, context)? <= 0),
            BinaryOperator::Greater => ConstraintValue::Boolean(self.compare_values(left_val, right_val, context)? > 0),
            BinaryOperator::GreaterEqual => ConstraintValue::Boolean(self.compare_values(left_val, right_val, context)? >= 0),
            BinaryOperator::And => ConstraintValue::Boolean(left_val.is_truthy() && right_val.is_truthy()),
            BinaryOperator::Or => ConstraintValue::Boolean(left_val.is_truthy() || right_val.is_truthy()),
            _ => {
                return Err(ConstraintError::EvaluationError {
                    location: context.current_span.unwrap_or_else(Span::dummy),
                    message: format!("Unsupported binary operator: {:?}", binary.operator),
                });
            }
        };

        let mut result = EvaluationResult::new(result_value);
        result.warnings.extend(left_result.warnings);
        result.warnings.extend(right_result.warnings);
        
        Ok(result)
    }

    /// Evaluate a unary expression with operation counting
    fn evaluate_unary_with_counter(&self, unary: &prism_ast::UnaryExpr, context: &mut EvaluationContext, depth: usize, operations: &mut OperationCounter) -> ConstraintResult<EvaluationResult> {
        let mut operand_context = context.clone().with_span(unary.operand.span);
        let operand_result = self.evaluate_with_depth_and_counter(&unary.operand.kind, &mut operand_context, depth + 1, operations)?;
        
        operations.increment(); // Count the unary operation
        
        let operand_val = &operand_result.value;
        
        let result_value = match unary.operator {
            UnaryOperator::Not => ConstraintValue::Boolean(!operand_val.is_truthy()),
            UnaryOperator::Negate => {
                match operand_val {
                    ConstraintValue::Integer(i) => ConstraintValue::Integer(-i),
                    ConstraintValue::Float(f) => ConstraintValue::Float(-f),
                    _ => {
                        return Err(ConstraintError::TypeMismatch {
                            location: context.current_span.unwrap_or_else(Span::dummy),
                            expected: "numeric".to_string(),
                            found: operand_val.type_name().to_string(),
                            value: format!("{:?}", operand_val),
                        });
                    }
                }
            }
            _ => {
                return Err(ConstraintError::EvaluationError {
                    location: context.current_span.unwrap_or_else(Span::dummy),
                    message: format!("Unsupported unary operator: {:?}", unary.operator),
                });
            }
        };

        let mut result = EvaluationResult::new(result_value);
        result.warnings.extend(operand_result.warnings);
        
        Ok(result)
    }

    /// Evaluate a function call expression with operation counting
    fn evaluate_call_with_counter(&self, call: &prism_ast::CallExpr, context: &mut EvaluationContext, depth: usize, operations: &mut OperationCounter) -> ConstraintResult<EvaluationResult> {
        // For now, only support simple function calls where the function is a variable
        if let Expr::Variable(var) = &call.callee.kind {
            let func_name = var.name.resolve().unwrap_or_else(|| format!("sym:{:?}", var.name.raw()));
            
            if let Some(func) = context.get_function(&func_name) {
                // Evaluate arguments
                let mut args = Vec::new();
                let mut warnings = Vec::new();
                
                for arg in &call.arguments {
                    let mut arg_context = context.clone().with_span(arg.span);
                    let arg_result = self.evaluate_with_depth_and_counter(&arg.kind, &mut arg_context, depth + 1, operations)?;
                    args.push(arg_result.value);
                    warnings.extend(arg_result.warnings);
                }
                
                operations.increment(); // Count the function call
                
                // Call the function
                let result_value = func(&args)?;
                
                Ok(EvaluationResult::with_warnings(result_value, warnings))
            } else {
                Err(ConstraintError::InvalidFunctionCall {
                    location: context.current_span.unwrap_or_else(Span::dummy),
                    function: func_name,
                    reason: "Function not found".to_string(),
                    available_functions: context.available_functions(),
                })
            }
        } else {
            Err(ConstraintError::EvaluationError {
                location: context.current_span.unwrap_or_else(Span::dummy),
                message: "Only simple function calls are supported in constraints".to_string(),
            })
        }
    }

    /// Evaluate a binary expression (legacy method for backward compatibility)
    fn evaluate_binary(&self, binary: &prism_ast::BinaryExpr, context: &mut EvaluationContext, depth: usize) -> ConstraintResult<EvaluationResult> {
        let mut operations = OperationCounter::new();
        self.evaluate_binary_with_counter(binary, context, depth, &mut operations)
    }

    /// Evaluate a unary expression (legacy method for backward compatibility)
    fn evaluate_unary(&self, unary: &prism_ast::UnaryExpr, context: &mut EvaluationContext, depth: usize) -> ConstraintResult<EvaluationResult> {
        let mut operations = OperationCounter::new();
        self.evaluate_unary_with_counter(unary, context, depth, &mut operations)
    }

    /// Evaluate a function call expression (legacy method for backward compatibility)
    fn evaluate_call(&self, call: &prism_ast::CallExpr, context: &mut EvaluationContext, depth: usize) -> ConstraintResult<EvaluationResult> {
        let mut operations = OperationCounter::new();
        self.evaluate_call_with_counter(call, context, depth, &mut operations)
    }

    // Helper methods for binary operations

    /// Add two constraint values
    fn add_values(&self, left: &ConstraintValue, right: &ConstraintValue, context: &EvaluationContext) -> ConstraintResult<ConstraintValue> {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => {
                if self.config.check_overflow {
                    l.checked_add(*r).map(ConstraintValue::Integer).ok_or_else(|| {
                        ConstraintError::EvaluationError {
                            location: context.current_span.unwrap_or_else(Span::dummy),
                            message: "Integer overflow in addition".to_string(),
                        }
                    })
                } else {
                    Ok(ConstraintValue::Integer(l.wrapping_add(*r)))
                }
            }
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => Ok(ConstraintValue::Float(l + r)),
            (ConstraintValue::Integer(l), ConstraintValue::Float(r)) => Ok(ConstraintValue::Float(*l as f64 + r)),
            (ConstraintValue::Float(l), ConstraintValue::Integer(r)) => Ok(ConstraintValue::Float(l + *r as f64)),
            (ConstraintValue::String(l), ConstraintValue::String(r)) => Ok(ConstraintValue::String(format!("{}{}", l, r))),
            _ => Err(ConstraintError::TypeMismatch {
                location: context.current_span.unwrap_or_else(Span::dummy),
                expected: "numeric or string".to_string(),
                found: format!("{} and {}", left.type_name(), right.type_name()),
                value: format!("{:?} + {:?}", left, right),
            }),
        }
    }

    /// Subtract two constraint values
    fn subtract_values(&self, left: &ConstraintValue, right: &ConstraintValue, context: &EvaluationContext) -> ConstraintResult<ConstraintValue> {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => {
                if self.config.check_overflow {
                    l.checked_sub(*r).map(ConstraintValue::Integer).ok_or_else(|| {
                        ConstraintError::EvaluationError {
                            location: context.current_span.unwrap_or_else(Span::dummy),
                            message: "Integer overflow in subtraction".to_string(),
                        }
                    })
                } else {
                    Ok(ConstraintValue::Integer(l.wrapping_sub(*r)))
                }
            }
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => Ok(ConstraintValue::Float(l - r)),
            (ConstraintValue::Integer(l), ConstraintValue::Float(r)) => Ok(ConstraintValue::Float(*l as f64 - r)),
            (ConstraintValue::Float(l), ConstraintValue::Integer(r)) => Ok(ConstraintValue::Float(l - *r as f64)),
            _ => Err(ConstraintError::TypeMismatch {
                location: context.current_span.unwrap_or_else(Span::dummy),
                expected: "numeric".to_string(),
                found: format!("{} and {}", left.type_name(), right.type_name()),
                value: format!("{:?} - {:?}", left, right),
            }),
        }
    }

    /// Multiply two constraint values
    fn multiply_values(&self, left: &ConstraintValue, right: &ConstraintValue, context: &EvaluationContext) -> ConstraintResult<ConstraintValue> {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => {
                if self.config.check_overflow {
                    l.checked_mul(*r).map(ConstraintValue::Integer).ok_or_else(|| {
                        ConstraintError::EvaluationError {
                            location: context.current_span.unwrap_or_else(Span::dummy),
                            message: "Integer overflow in multiplication".to_string(),
                        }
                    })
                } else {
                    Ok(ConstraintValue::Integer(l.wrapping_mul(*r)))
                }
            }
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => Ok(ConstraintValue::Float(l * r)),
            (ConstraintValue::Integer(l), ConstraintValue::Float(r)) => Ok(ConstraintValue::Float(*l as f64 * r)),
            (ConstraintValue::Float(l), ConstraintValue::Integer(r)) => Ok(ConstraintValue::Float(l * *r as f64)),
            _ => Err(ConstraintError::TypeMismatch {
                location: context.current_span.unwrap_or_else(Span::dummy),
                expected: "numeric".to_string(),
                found: format!("{} and {}", left.type_name(), right.type_name()),
                value: format!("{:?} * {:?}", left, right),
            }),
        }
    }

    /// Divide two constraint values
    fn divide_values(&self, left: &ConstraintValue, right: &ConstraintValue, context: &EvaluationContext) -> ConstraintResult<ConstraintValue> {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => {
                if *r == 0 {
                    return Err(ConstraintError::EvaluationError {
                        location: context.current_span.unwrap_or_else(Span::dummy),
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(ConstraintValue::Float(*l as f64 / *r as f64))
            }
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => {
                if *r == 0.0 {
                    return Err(ConstraintError::EvaluationError {
                        location: context.current_span.unwrap_or_else(Span::dummy),
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(ConstraintValue::Float(l / r))
            }
            (ConstraintValue::Integer(l), ConstraintValue::Float(r)) => {
                if *r == 0.0 {
                    return Err(ConstraintError::EvaluationError {
                        location: context.current_span.unwrap_or_else(Span::dummy),
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(ConstraintValue::Float(*l as f64 / r))
            }
            (ConstraintValue::Float(l), ConstraintValue::Integer(r)) => {
                if *r == 0 {
                    return Err(ConstraintError::EvaluationError {
                        location: context.current_span.unwrap_or_else(Span::dummy),
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(ConstraintValue::Float(l / *r as f64))
            }
            _ => Err(ConstraintError::TypeMismatch {
                location: context.current_span.unwrap_or_else(Span::dummy),
                expected: "numeric".to_string(),
                found: format!("{} and {}", left.type_name(), right.type_name()),
                value: format!("{:?} / {:?}", left, right),
            }),
        }
    }

    /// Modulo two constraint values
    fn modulo_values(&self, left: &ConstraintValue, right: &ConstraintValue, context: &EvaluationContext) -> ConstraintResult<ConstraintValue> {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => {
                if *r == 0 {
                    return Err(ConstraintError::EvaluationError {
                        location: context.current_span.unwrap_or_else(Span::dummy),
                        message: "Modulo by zero".to_string(),
                    });
                }
                Ok(ConstraintValue::Integer(l % r))
            }
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => {
                if *r == 0.0 {
                    return Err(ConstraintError::EvaluationError {
                        location: context.current_span.unwrap_or_else(Span::dummy),
                        message: "Modulo by zero".to_string(),
                    });
                }
                Ok(ConstraintValue::Float(l % r))
            }
            _ => Err(ConstraintError::TypeMismatch {
                location: context.current_span.unwrap_or_else(Span::dummy),
                expected: "numeric".to_string(),
                found: format!("{} and {}", left.type_name(), right.type_name()),
                value: format!("{:?} % {:?}", left, right),
            }),
        }
    }

    /// Check if two values are equal
    fn values_equal(&self, left: &ConstraintValue, right: &ConstraintValue) -> bool {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => l == r,
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => (l - r).abs() < f64::EPSILON,
            (ConstraintValue::Integer(l), ConstraintValue::Float(r)) => (*l as f64 - r).abs() < f64::EPSILON,
            (ConstraintValue::Float(l), ConstraintValue::Integer(r)) => (l - *r as f64).abs() < f64::EPSILON,
            (ConstraintValue::String(l), ConstraintValue::String(r)) => l == r,
            (ConstraintValue::Boolean(l), ConstraintValue::Boolean(r)) => l == r,
            (ConstraintValue::Null, ConstraintValue::Null) => true,
            _ => false,
        }
    }

    /// Compare two values (-1 if left < right, 0 if equal, 1 if left > right)
    fn compare_values(&self, left: &ConstraintValue, right: &ConstraintValue, context: &EvaluationContext) -> ConstraintResult<i32> {
        match (left, right) {
            (ConstraintValue::Integer(l), ConstraintValue::Integer(r)) => Ok(l.cmp(r) as i32),
            (ConstraintValue::Float(l), ConstraintValue::Float(r)) => {
                if l < r { Ok(-1) } else if l > r { Ok(1) } else { Ok(0) }
            }
            (ConstraintValue::Integer(l), ConstraintValue::Float(r)) => {
                let l_f = *l as f64;
                if l_f < *r { Ok(-1) } else if l_f > *r { Ok(1) } else { Ok(0) }
            }
            (ConstraintValue::Float(l), ConstraintValue::Integer(r)) => {
                let r_f = *r as f64;
                if *l < r_f { Ok(-1) } else if *l > r_f { Ok(1) } else { Ok(0) }
            }
            (ConstraintValue::String(l), ConstraintValue::String(r)) => Ok(l.cmp(r) as i32),
            _ => Err(ConstraintError::TypeMismatch {
                location: context.current_span.unwrap_or_else(Span::dummy),
                expected: "comparable types".to_string(),
                found: format!("{} and {}", left.type_name(), right.type_name()),
                value: format!("{:?} <=> {:?}", left, right),
            }),
        }
    }

    /// Register built-in functions
    fn register_builtin_functions(&mut self) {
        self.builtin_functions.insert("length".to_string(), builtin_length);
        self.builtin_functions.insert("abs".to_string(), builtin_abs);
        self.builtin_functions.insert("min".to_string(), builtin_min);
        self.builtin_functions.insert("max".to_string(), builtin_max);
        self.builtin_functions.insert("is_string".to_string(), builtin_is_string);
        self.builtin_functions.insert("is_number".to_string(), builtin_is_number);
        self.builtin_functions.insert("is_boolean".to_string(), builtin_is_boolean);
        self.builtin_functions.insert("is_null".to_string(), builtin_is_null);
    }
}

impl EvaluationResult {
    /// Create a new evaluation result
    pub fn new(value: ConstraintValue) -> Self {
        Self {
            value,
            warnings: Vec::new(),
            performance: EvaluationPerformance {
                duration: Duration::from_nanos(0),
                operations: 1,
                max_depth: 0,
            },
        }
    }

    /// Create evaluation result with warnings
    pub fn with_warnings(value: ConstraintValue, warnings: Vec<String>) -> Self {
        Self {
            value,
            warnings,
            performance: EvaluationPerformance {
                duration: Duration::from_nanos(0),
                operations: 1,
                max_depth: 0,
            },
        }
    }
}

impl Default for EvaluationResult {
    fn default() -> Self {
        Self::new(ConstraintValue::Null)
    }
}

// Built-in functions
fn builtin_length(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.len() != 1 {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "length".to_string(),
            reason: format!("Expected 1 argument, got {}", args.len()),
            available_functions: vec!["length".to_string()],
        });
    }

    match &args[0] {
        ConstraintValue::String(s) => Ok(ConstraintValue::Integer(s.len() as i64)),
        ConstraintValue::Array(arr) => Ok(ConstraintValue::Integer(arr.len() as i64)),
        _ => Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "length".to_string(),
            reason: format!("Cannot get length of {:?}", args[0]),
            available_functions: vec!["length".to_string()],
        }),
    }
}

fn builtin_abs(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.len() != 1 {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "abs".to_string(),
            reason: format!("Expected 1 argument, got {}", args.len()),
            available_functions: vec!["abs".to_string()],
        });
    }

    match &args[0] {
        ConstraintValue::Integer(i) => Ok(ConstraintValue::Integer(i.abs())),
        ConstraintValue::Float(f) => Ok(ConstraintValue::Float(f.abs())),
        _ => Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "abs".to_string(),
            reason: format!("Cannot get absolute value of {:?}", args[0]),
            available_functions: vec!["abs".to_string()],
        }),
    }
}

fn builtin_min(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.is_empty() {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "min".to_string(),
            reason: "Expected at least 1 argument".to_string(),
            available_functions: vec!["min".to_string()],
        });
    }

    let mut min_val = &args[0];
    for arg in &args[1..] {
        match (min_val, arg) {
            (ConstraintValue::Integer(a), ConstraintValue::Integer(b)) => {
                if b < a { min_val = arg; }
            }
            (ConstraintValue::Float(a), ConstraintValue::Float(b)) => {
                if b < a { min_val = arg; }
            }
            (ConstraintValue::Integer(a), ConstraintValue::Float(b)) => {
                if *b < *a as f64 { min_val = arg; }
            }
            (ConstraintValue::Float(a), ConstraintValue::Integer(b)) => {
                if (*b as f64) < *a { min_val = arg; }
            }
            _ => return Err(ConstraintError::InvalidFunctionCall {
                location: Span::dummy(),
                function: "min".to_string(),
                reason: "All arguments must be numeric".to_string(),
                available_functions: vec!["min".to_string()],
            }),
        }
    }

    Ok(min_val.clone())
}

fn builtin_max(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.is_empty() {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "max".to_string(),
            reason: "Expected at least 1 argument".to_string(),
            available_functions: vec!["max".to_string()],
        });
    }

    let mut max_val = &args[0];
    for arg in &args[1..] {
        match (max_val, arg) {
            (ConstraintValue::Integer(a), ConstraintValue::Integer(b)) => {
                if b > a { max_val = arg; }
            }
            (ConstraintValue::Float(a), ConstraintValue::Float(b)) => {
                if b > a { max_val = arg; }
            }
            (ConstraintValue::Integer(a), ConstraintValue::Float(b)) => {
                if *b > *a as f64 { max_val = arg; }
            }
            (ConstraintValue::Float(a), ConstraintValue::Integer(b)) => {
                if (*b as f64) > *a { max_val = arg; }
            }
            _ => return Err(ConstraintError::InvalidFunctionCall {
                location: Span::dummy(),
                function: "max".to_string(),
                reason: "All arguments must be numeric".to_string(),
                available_functions: vec!["max".to_string()],
            }),
        }
    }

    Ok(max_val.clone())
}

fn builtin_is_string(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.len() != 1 {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "is_string".to_string(),
            reason: format!("Expected 1 argument, got {}", args.len()),
            available_functions: vec!["is_string".to_string()],
        });
    }

    Ok(ConstraintValue::Boolean(matches!(args[0], ConstraintValue::String(_))))
}

fn builtin_is_number(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.len() != 1 {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "is_number".to_string(),
            reason: format!("Expected 1 argument, got {}", args.len()),
            available_functions: vec!["is_number".to_string()],
        });
    }

    Ok(ConstraintValue::Boolean(matches!(args[0], ConstraintValue::Integer(_) | ConstraintValue::Float(_))))
}

fn builtin_is_boolean(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.len() != 1 {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "is_boolean".to_string(),
            reason: format!("Expected 1 argument, got {}", args.len()),
            available_functions: vec!["is_boolean".to_string()],
        });
    }

    Ok(ConstraintValue::Boolean(matches!(args[0], ConstraintValue::Boolean(_))))
}

fn builtin_is_null(args: &[ConstraintValue]) -> ConstraintResult<ConstraintValue> {
    if args.len() != 1 {
        return Err(ConstraintError::InvalidFunctionCall {
            location: Span::dummy(),
            function: "is_null".to_string(),
            reason: format!("Expected 1 argument, got {}", args.len()),
            available_functions: vec!["is_null".to_string()],
        });
    }

    Ok(ConstraintValue::Boolean(matches!(args[0], ConstraintValue::Null)))
} 