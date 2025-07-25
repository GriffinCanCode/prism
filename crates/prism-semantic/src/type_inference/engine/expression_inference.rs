//! Expression Type Inference Engine
//!
//! This module implements type inference specifically for expressions.
//! It handles all expression types and their specific inference rules.
//!
//! **Single Responsibility**: Type inference for expressions
//! **What it does**: Infer types for all expression constructs
//! **What it doesn't do**: Handle statements, patterns, or orchestration

use super::{InferenceEngine, EffectAware, ASTTypeResolver};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        InferredType, InferenceSource, TypeVar, TypeVarGenerator,
        constraints::{ConstraintSet, TypeConstraint, ConstraintType, ConstraintReason},
        environment::{TypeEnvironment, TypeBinding, ScopeKind},
        errors::TypeError,
    },
};
use prism_ast::{Expr, LiteralValue, BinaryOperator, UnaryOperator, AstNode, Type as AstType, 
    Pattern, SpawnMode, ExprContext, Generator};
use prism_ast::expr::{Parameter, MatchArm, CatchClause};
use prism_common::{NodeId, Span};
use std::collections::HashMap;

/// Engine specifically for expression type inference
#[derive(Debug)]
pub struct ExpressionInferenceEngine {
    /// Type variable generator
    type_var_gen: TypeVarGenerator,
    /// Current type environment
    environment: TypeEnvironment,
    /// AST type resolver
    ast_resolver: ASTTypeResolver,
    /// Current inference depth
    current_depth: usize,
    /// Maximum inference depth
    max_depth: usize,
}

/// Result of expression inference
#[derive(Debug, Clone)]
pub struct ExpressionInferenceResult {
    /// The inferred type
    pub inferred_type: InferredType,
    /// Constraints generated during inference
    pub constraints: ConstraintSet,
    /// Updated type environment
    pub environment: TypeEnvironment,
    /// Effects inferred from the expression
    pub effects: Vec<String>,
}

impl ExpressionInferenceEngine {
    /// Create a new expression inference engine
    pub fn new() -> SemanticResult<Self> {
        Ok(Self {
            type_var_gen: TypeVarGenerator::new(),
            environment: TypeEnvironment::new(),
            ast_resolver: ASTTypeResolver::new(),
            current_depth: 0,
            max_depth: 1000,
        })
    }

    /// Infer type for an expression
    pub fn infer_expression(&mut self, expr: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        self.check_depth_limit()?;
        self.current_depth += 1;
        
        let result = match expr {
            Expr::Literal(literal_expr) => {
                self.infer_literal(&literal_expr.value, span)?
            }
            Expr::Variable(var_expr) => {
                let name = var_expr.name.resolve().unwrap_or_else(|| "unknown".to_string());
                self.infer_variable(&name, span)?
            }
            Expr::Binary(binary_expr) => {
                self.infer_binary_expr(&binary_expr.operator, &binary_expr.left.kind, &binary_expr.right.kind, span)?
            }
            Expr::Unary(unary_expr) => {
                self.infer_unary_expr(&unary_expr.operator, &unary_expr.operand.kind, span)?
            }
            Expr::Call(call_expr) => {
                let args: Vec<&Expr> = call_expr.arguments.iter().map(|arg| &arg.kind).collect();
                self.infer_function_call(&call_expr.callee.kind, &args, span)?
            }
            Expr::Array(array_expr) => {
                let elements: Vec<&Expr> = array_expr.elements.iter().map(|elem| &elem.kind).collect();
                self.infer_array_expr(&elements, span)?
            }
            Expr::Index(index_expr) => {
                self.infer_index_expr(&index_expr.object.kind, &index_expr.index.kind, span)?
            }
            Expr::Member(member_expr) => {
                let field_name = member_expr.member.resolve().unwrap_or_else(|| "unknown".to_string());
                self.infer_field_access(&member_expr.object.kind, &field_name, span)?
            }
            Expr::Lambda(lambda_expr) => {
                self.infer_function_expression(&lambda_expr.parameters, &lambda_expr.body.kind, span)?
            }
            Expr::If(if_expr) => {
                let else_expr = if_expr.else_branch.as_ref().map(|e| &e.kind);
                self.infer_if_expression(&if_expr.condition.kind, &if_expr.then_branch.kind, else_expr, span)?
            }
            Expr::Match(match_expr) => {
                self.infer_match_expression(&match_expr.scrutinee.kind, &match_expr.arms, span)?
            }
            Expr::Object(object_expr) => {
                let field_exprs: Vec<(String, &Expr)> = object_expr.fields.iter()
                    .map(|field| {
                        let key_name = match &field.key {
                            prism_ast::ObjectKey::Identifier(sym) => sym.resolve().unwrap_or_else(|| "unknown".to_string()),
                            prism_ast::ObjectKey::String(s) => s.clone(),
                            prism_ast::ObjectKey::Computed(_) => "computed".to_string(),
                        };
                        (key_name, &field.value.kind)
                    })
                    .collect();
                self.infer_record_expression(&field_exprs, span)?
            }
            Expr::TypeAssertion(type_assertion_expr) => {
                self.infer_type_annotation(&type_assertion_expr.expression.kind, &type_assertion_expr.target_type.kind, span)?
            }
            Expr::While(while_expr) => {
                self.infer_while_expression(&while_expr.condition.kind, &while_expr.body.kind, span)?
            }
            Expr::For(for_expr) => {
                let var_name = for_expr.variable.resolve().unwrap_or_else(|| "iterator".to_string());
                self.infer_for_expression(&var_name, &for_expr.iterable.kind, &for_expr.body.kind, span)?
            }
            Expr::Try(try_expr) => {
                let finally_expr = try_expr.finally_block.as_ref().map(|e| &e.kind);
                self.infer_try_expression(&try_expr.try_block.kind, &try_expr.catch_clauses, finally_expr, span)?
            }
            Expr::Await(await_expr) => {
                self.infer_await_expression(&await_expr.expression.kind, span)?
            }
            Expr::Yield(yield_expr) => {
                let value_expr = yield_expr.value.as_ref().map(|v| &v.kind);
                self.infer_yield_expression(value_expr, span)?
            }
            Expr::Actor(actor_expr) => {
                let capabilities: Vec<&Expr> = actor_expr.capabilities.iter().map(|c| &c.kind).collect();
                let config_expr = actor_expr.config.as_ref().map(|c| &c.kind);
                self.infer_actor_expression(&actor_expr.actor_impl.kind, &capabilities, config_expr, span)?
            }
            Expr::Spawn(spawn_expr) => {
                let capabilities: Vec<&Expr> = spawn_expr.capabilities.iter().map(|c| &c.kind).collect();
                let priority_expr = spawn_expr.priority.as_ref().map(|p| &p.kind);
                self.infer_spawn_expression(&spawn_expr.expression.kind, &spawn_expr.spawn_mode, &capabilities, priority_expr, span)?
            }
            Expr::Channel(channel_expr) => {
                self.infer_channel_expression(&channel_expr.channel_type, span)?
            }
            Expr::Select(select_expr) => {
                self.infer_select_expression(&select_expr.arms, span)?
            }
            Expr::Range(range_expr) => {
                let end_expr = Some(&range_expr.end.kind);
                self.infer_range_expression(&range_expr.start.kind, end_expr, range_expr.inclusive, span)?
            }
            Expr::Tuple(tuple_expr) => {
                let elements: Vec<&Expr> = tuple_expr.elements.iter().map(|elem| &elem.kind).collect();
                self.infer_tuple_expression(&elements, span)?
            }
            Expr::Block(block_expr) => {
                let statements: Vec<&prism_ast::Stmt> = block_expr.statements.iter().map(|stmt| &stmt.kind).collect();
                let final_expr = block_expr.final_expr.as_ref().map(|e| &e.kind);
                self.infer_block_expression(&statements, final_expr, span)?
            }
            Expr::Return(return_expr) => {
                let value_expr = return_expr.value.as_ref().map(|v| &v.kind);
                self.infer_return_expression(value_expr, span)?
            }
            Expr::Break(break_expr) => {
                let value_expr = break_expr.value.as_ref().map(|v| &v.kind);
                self.infer_break_expression(value_expr, span)?
            }
            Expr::Continue(_) => {
                self.infer_continue_expression(span)?
            }
            Expr::Throw(throw_expr) => {
                self.infer_throw_expression(&throw_expr.exception.kind, span)?
            }
            Expr::FormattedString(fmt_expr) => {
                let parts: Vec<&prism_ast::FStringPart> = fmt_expr.parts.iter().collect();
                self.infer_formatted_string_expression(&parts, span)?
            }
            Expr::ListComprehension(list_comp) => {
                self.infer_list_comprehension(&list_comp.element.kind, &list_comp.generators, span)?
            }
            Expr::SetComprehension(set_comp) => {
                self.infer_set_comprehension(&set_comp.element.kind, &set_comp.generators, span)?
            }
            Expr::DictComprehension(dict_comp) => {
                self.infer_dict_comprehension(&dict_comp.key.kind, &dict_comp.value.kind, &dict_comp.generators, span)?
            }
            Expr::GeneratorExpression(gen_expr) => {
                self.infer_generator_expression(&gen_expr.element.kind, &gen_expr.generators, span)?
            }
            Expr::NamedExpression(named_expr) => {
                let var_name = named_expr.target.resolve().unwrap_or_else(|| "named".to_string());
                self.infer_named_expression(&var_name, &named_expr.value.kind, span)?
            }
            Expr::StarredExpression(starred_expr) => {
                self.infer_starred_expression(&starred_expr.value.kind, &starred_expr.context, span)?
            }
            _ => {
                // Default case for unsupported expressions
                self.create_default_result(span)?
            }
        };
        
        self.current_depth -= 1;
        Ok(result)
    }

    /// Infer type for a literal value
    fn infer_literal(&mut self, value: &LiteralValue, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let literal_type = match value {
            LiteralValue::Integer(_) => self.create_semantic_type("Integer", span),
            LiteralValue::Float(_) => self.create_semantic_type("Float", span),
            LiteralValue::String(_) => self.create_semantic_type("String", span),
            LiteralValue::Boolean(_) => self.create_semantic_type("Boolean", span),
            LiteralValue::Null => self.create_semantic_type("Null", span),
            _ => self.create_semantic_type("Unknown", span),
        };

        let inferred_type = InferredType {
            type_info: literal_type,
            confidence: 1.0,
            inference_source: InferenceSource::Literal,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            environment: self.environment.clone(),
            effects: Vec::new(), // Literals have no effects
        })
    }

    /// Infer type for a variable reference
    fn infer_variable(&mut self, name: &str, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        match self.environment.lookup_and_instantiate(name, &mut self.type_var_gen)? {
            Some(instantiated_type) => {
                Ok(ExpressionInferenceResult {
                    inferred_type: instantiated_type,
                    constraints: ConstraintSet::new(),
                    environment: self.environment.clone(),
                    effects: Vec::new(), // Variable access has no effects
                })
            }
            None => {
                Err(SemanticError::TypeInferenceError {
                    message: format!("Undefined variable: {}", name),
                })
            }
        }
    }

    /// Infer type for a binary expression
    fn infer_binary_expr(
        &mut self, 
        op: &BinaryOperator, 
        left: &Expr, 
        right: &Expr, 
        span: Span
    ) -> SemanticResult<ExpressionInferenceResult> {
        // Infer left and right operand types
        let left_result = self.infer_expression(left, span)?;
        let right_result = self.infer_expression(right, span)?;
        
        let mut constraints = ConstraintSet::new();
        constraints.extend(left_result.constraints);
        constraints.extend(right_result.constraints);
        
        // Determine the expected types and result type based on the operator
        let (expected_left_type, expected_right_type, result_type) = match op {
            BinaryOperator::Add | BinaryOperator::Subtract | 
            BinaryOperator::Multiply | BinaryOperator::Divide => {
                // Arithmetic operations: both operands and result are numeric
                let numeric_type = self.create_semantic_type("Integer", span);
                (numeric_type.clone(), numeric_type.clone(), numeric_type)
            }
            BinaryOperator::Less | BinaryOperator::LessEqual |
            BinaryOperator::Greater | BinaryOperator::GreaterEqual => {
                // Comparison operations: numeric operands, boolean result
                let numeric_type = self.create_semantic_type("Integer", span);
                let bool_type = self.create_semantic_type("Boolean", span);
                (numeric_type.clone(), numeric_type, bool_type)
            }
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                // Equality operations: same types for operands, boolean result
                let var = self.fresh_type_var(span);
                let var_type = SemanticType::Variable(var.id.to_string());
                let bool_type = self.create_semantic_type("Boolean", span);
                (var_type.clone(), var_type, bool_type)
            }
            BinaryOperator::And | BinaryOperator::Or => {
                // Logical operations: boolean operands and result
                let bool_type = self.create_semantic_type("Boolean", span);
                (bool_type.clone(), bool_type.clone(), bool_type)
            }
            _ => {
                // Default case
                let default_type = self.create_semantic_type("Unknown", span);
                (default_type.clone(), default_type.clone(), default_type)
            }
        };
        
        // Add constraints for operand types
        constraints.add_constraint(TypeConstraint {
            lhs: ConstraintType::Concrete(left_result.inferred_type.type_info),
            rhs: ConstraintType::Concrete(expected_left_type),
            origin: span,
            reason: ConstraintReason::BinaryOperation { operator: format!("{:?}", op) },
            priority: 90,
        });
        
        constraints.add_constraint(TypeConstraint {
            lhs: ConstraintType::Concrete(right_result.inferred_type.type_info),
            rhs: ConstraintType::Concrete(expected_right_type),
            origin: span,
            reason: ConstraintReason::BinaryOperation { operator: format!("{:?}", op) },
            priority: 90,
        });
        
        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.9,
            inference_source: InferenceSource::Operator,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        // Combine effects from both operands
        let mut effects = left_result.effects;
        effects.extend(right_result.effects);
        
        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    /// Infer type for a unary expression
    fn infer_unary_expr(
        &mut self, 
        op: &UnaryOperator, 
        expr: &Expr, 
        span: Span
    ) -> SemanticResult<ExpressionInferenceResult> {
        let operand_result = self.infer_expression(expr, span)?;
        let mut constraints = operand_result.constraints;
        
        // Determine expected operand type and result type based on operator
        let (expected_operand_type, result_type) = match op {
            UnaryOperator::Not => {
                // Logical not: boolean operand and result
                let bool_type = self.create_semantic_type("Boolean", span);
                (bool_type.clone(), bool_type)
            }
            UnaryOperator::Negate => {
                // Numeric negation: numeric operand and result
                let numeric_type = self.create_semantic_type("Integer", span);
                (numeric_type.clone(), numeric_type)
            }
            _ => {
                // Default case
                let default_type = self.create_semantic_type("Unknown", span);
                (default_type.clone(), default_type)
            }
        };
        
        // Add constraint that operand must have expected type
        constraints.add_constraint(TypeConstraint {
            lhs: ConstraintType::Concrete(operand_result.inferred_type.type_info),
            rhs: ConstraintType::Concrete(expected_operand_type),
            origin: span,
            reason: ConstraintReason::UnaryOperation { operator: format!("{:?}", op) },
            priority: 90,
        });
        
        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.9,
            inference_source: InferenceSource::Operator,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: operand_result.effects,
        })
    }

    /// Infer type for a function call
    fn infer_function_call(
        &mut self, 
        func: &Expr, 
        args: &[&Expr], 
        span: Span
    ) -> SemanticResult<ExpressionInferenceResult> {
        // Infer the function type
        let func_result = self.infer_expression(func, span)?;
        let mut constraints = func_result.constraints;
        let mut effects = func_result.effects;
        
        // Infer argument types
        let mut arg_types = Vec::new();
        for arg in args {
            let arg_result = self.infer_expression(arg, span)?;
            constraints.extend(arg_result.constraints);
            effects.extend(arg_result.effects);
            arg_types.push(arg_result.inferred_type.type_info);
        }
        
        // Create a fresh type variable for the return type
        let return_type_var = self.fresh_type_var(span);
        let return_type = SemanticType::Variable(return_type_var.id.to_string());
        
        // Create the expected function type
        let expected_func_type = SemanticType::Function {
            params: arg_types,
            return_type: Box::new(return_type.clone()),
            effects: Vec::new(), // Effects will be inferred separately
        };
        
        // Add constraint that func_type = expected_func_type
        constraints.add_constraint(TypeConstraint {
            lhs: ConstraintType::Concrete(func_result.inferred_type.type_info),
            rhs: ConstraintType::Concrete(expected_func_type),
            origin: span,
            reason: ConstraintReason::FunctionApplication,
            priority: 100,
        });
        
        let inferred_type = InferredType {
            type_info: return_type,
            confidence: 0.8,
            inference_source: InferenceSource::Application,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        // Function calls may have effects
        effects.push("FunctionCall".to_string());
        
        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    // Additional methods for other expression types...
    
    /// Create a default result for unsupported expressions
    fn create_default_result(&self, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let default_type = self.create_semantic_type("Unknown", span);
        let inferred_type = InferredType {
            type_info: default_type,
            confidence: 0.1,
            inference_source: InferenceSource::Default,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            environment: self.environment.clone(),
            effects: Vec::new(),
        })
    }

    /// Create a semantic type for basic types
    fn create_semantic_type(&self, name: &str, span: Span) -> SemanticType {
        let primitive_type = match name {
            "Integer" => crate::types::PrimitiveType::Custom { 
                name: "Integer".to_string(), 
                base: "i64".to_string() 
            },
            "Float" => crate::types::PrimitiveType::Custom { 
                name: "Float".to_string(), 
                base: "f64".to_string() 
            },
            "String" => crate::types::PrimitiveType::Custom { 
                name: "String".to_string(), 
                base: "string".to_string() 
            },
            "Boolean" => crate::types::PrimitiveType::Custom { 
                name: "Boolean".to_string(), 
                base: "bool".to_string() 
            },
            "Unit" => crate::types::PrimitiveType::Custom { 
                name: "Unit".to_string(), 
                base: "unit".to_string() 
            },
            _ => crate::types::PrimitiveType::Custom { 
                name: name.to_string(), 
                base: "unknown".to_string() 
            },
        };

        SemanticType::primitive(name, self.convert_primitive_type(primitive_type), span)
    }

    /// Generate a fresh type variable
    pub fn fresh_type_var(&mut self, span: Span) -> TypeVar {
        self.type_var_gen.fresh(span)
    }

    /// Check depth limit to prevent infinite recursion
    fn check_depth_limit(&self) -> SemanticResult<()> {
        if self.current_depth >= self.max_depth {
            return Err(SemanticError::TypeInferenceError {
                message: format!("Maximum inference depth exceeded: {}", self.current_depth),
            });
        }
        Ok(())
    }

    /// Convert prism_ast::PrimitiveType to prism_ast::PrimitiveType (identity for now)
    fn convert_primitive_type(&self, prim_type: crate::types::PrimitiveType) -> prism_ast::PrimitiveType {
        match prim_type {
            crate::types::PrimitiveType::Custom { base, .. } => {
                match base.as_str() {
                    "i64" => prism_ast::PrimitiveType::Int64,
                    "f64" => prism_ast::PrimitiveType::Float64,
                    "string" => prism_ast::PrimitiveType::String,
                    "bool" => prism_ast::PrimitiveType::Boolean,
                    "unit" => prism_ast::PrimitiveType::Unit,
                    _ => prism_ast::PrimitiveType::Unit, // fallback
                }
            }
            _ => prism_ast::PrimitiveType::Unit, // fallback for other variants
        }
    }

    // Placeholder methods for additional expression types
    fn infer_array_expr(&mut self, elements: &[&Expr], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        if elements.is_empty() {
            // Empty array - create generic array type
            let element_type_var = self.fresh_type_var(span);
            let element_type = SemanticType::Variable(element_type_var.id.to_string());
            let array_type = SemanticType::List(Box::new(element_type));
            
            let inferred_type = InferredType {
                type_info: array_type,
                confidence: 0.8,
                inference_source: InferenceSource::Structural,
                constraints: Vec::new(),
                ai_metadata: None,
                span,
            };
            
            return Ok(ExpressionInferenceResult {
                inferred_type,
                constraints: ConstraintSet::new(),
                environment: self.environment.clone(),
                effects: Vec::new(),
            });
        }

        // Infer types of all elements
        let mut element_results = Vec::new();
        let mut all_constraints = ConstraintSet::new();
        let mut all_effects = Vec::new();

        for element in elements {
            let element_result = self.infer_expression(element, span)?;
            element_results.push(element_result.inferred_type.clone());
            all_constraints.merge(element_result.constraints);
            all_effects.extend(element_result.effects);
        }

        // Unify all element types
        let first_element_type = &element_results[0].type_info;
        let mut unified_type = first_element_type.clone();

        for element_result in &element_results[1..] {
            // Create constraint for type unification
            let constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(unified_type.clone()),
                rhs: ConstraintType::Concrete(element_result.type_info.clone()),
                origin: span,
                reason: ConstraintReason::ArrayElementUnification,
                priority: 80,
            };
            all_constraints.add_constraint(constraint);
            
            // For now, use the first type as unified type
            // In a full implementation, this would perform proper unification
        }

        let array_type = SemanticType::List(Box::new(unified_type));
        let inferred_type = InferredType {
            type_info: array_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: all_constraints,
            environment: self.environment.clone(),
            effects: all_effects,
        })
    }

    fn infer_index_expr(&mut self, object: &Expr, index: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Infer object type
        let object_result = self.infer_expression(object, span)?;
        let index_result = self.infer_expression(index, span)?;

        let mut constraints = ConstraintSet::new();
        constraints.merge(object_result.constraints);
        constraints.merge(index_result.constraints);

        let mut effects = object_result.effects;
        effects.extend(index_result.effects);
        effects.push("ArrayAccess".to_string());

        // Determine result type based on object type
        let result_type = match &object_result.inferred_type.type_info {
            SemanticType::List(element_type) => {
                // Array/List access - ensure index is integer
                let int_constraint = TypeConstraint {
                    lhs: ConstraintType::Concrete(index_result.inferred_type.type_info.clone()),
                    rhs: ConstraintType::Concrete(self.create_semantic_type("Integer", span)),
                    origin: span,
                    reason: ConstraintReason::IndexTypeCheck,
                    priority: 90,
                };
                constraints.add_constraint(int_constraint);
                (**element_type).clone()
            }
            SemanticType::Record(fields) => {
                // Record access with string index
                if let SemanticType::Primitive(_) = &index_result.inferred_type.type_info {
                    // For now, return a type variable for the field type
                    let field_type_var = self.fresh_type_var(span);
                    SemanticType::Variable(field_type_var.id.to_string())
                } else {
                    let string_constraint = TypeConstraint {
                        lhs: ConstraintType::Concrete(index_result.inferred_type.type_info.clone()),
                        rhs: ConstraintType::Concrete(self.create_semantic_type("String", span)),
                        origin: span,
                        reason: ConstraintReason::IndexTypeCheck,
                        priority: 90,
                    };
                    constraints.add_constraint(string_constraint);
                    
                    let field_type_var = self.fresh_type_var(span);
                    SemanticType::Variable(field_type_var.id.to_string())
                }
            }
            _ => {
                // Generic indexable type - create type variable for result
                let result_type_var = self.fresh_type_var(span);
                SemanticType::Variable(result_type_var.id.to_string())
            }
        };

        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.8,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    fn infer_field_access(&mut self, object: &Expr, field: &str, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let object_result = self.infer_expression(object, span)?;
        
        let mut effects = object_result.effects;
        effects.push("FieldAccess".to_string());

        let result_type = match &object_result.inferred_type.type_info {
            SemanticType::Record(fields) => {
                // Try to find the field in the record
                fields.get(field)
                    .cloned()
                    .unwrap_or_else(|| {
                        let field_type_var = self.fresh_type_var(span);
                        SemanticType::Variable(field_type_var.id.to_string())
                    })
            }
            SemanticType::Complex { name: _, base_type: _, constraints: _, business_rules: _, metadata, ai_context: _, verification_properties: _, location: _ } => {
                // For now, just create a type variable for the field
                let field_type_var = self.fresh_type_var(span);
                SemanticType::Variable(field_type_var.id.to_string())
            }
            _ => {
                // Unknown object type - create type variable for field
                let field_type_var = self.fresh_type_var(span);
                SemanticType::Variable(field_type_var.id.to_string())
            }
        };

        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.7,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: object_result.constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    fn infer_function_expression(&mut self, params: &[Parameter], body: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Create new scope for function parameters
        self.environment.enter_scope(ScopeKind::Function);

        let mut param_types = Vec::new();
        let mut all_constraints = ConstraintSet::new();
        let mut bindings = Vec::new();

        // Infer parameter types
        for (i, param) in params.iter().enumerate() {
            let param_type = if let Some(type_annotation) = &param.type_annotation {
                // Use annotated type if available
                self.ast_resolver.resolve_ast_type(&type_annotation.kind)?
            } else {
                // Create type variable if no annotation
                let param_type_var = self.fresh_type_var(span);
                SemanticType::Variable(param_type_var.id.to_string())
            };
            
            let param_name = param.name.resolve().unwrap_or_else(|| format!("param_{}", i));
            let binding = TypeBinding::new(
                param_name,
                InferredType {
                    type_info: param_type.clone(),
                    confidence: 0.8,
                    inference_source: InferenceSource::Parameter,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span,
                },
                param.is_mutable,
                self.environment.current_level(),
            );
            self.environment.add_binding(binding.clone());
            bindings.push(binding);
            
            param_types.push(param_type);
        }

        // Infer body type
        let body_result = self.infer_expression(body, span)?;
        all_constraints.merge(body_result.constraints);

        // Pop function scope
        self.environment.exit_scope()?;

        let function_type = SemanticType::Function {
            params: param_types,
            return_type: Box::new(body_result.inferred_type.type_info.clone()),
            effects: body_result.effects.clone(),
        };

        let inferred_type = InferredType {
            type_info: function_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: all_constraints,
            environment: self.environment.clone(),
            effects: body_result.effects,
        })
    }

    fn infer_let_expression(&mut self, name: &str, value: &Expr, body: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Infer value type
        let value_result = self.infer_expression(value, span)?;
        
        // Create binding for the let variable
        let binding = TypeBinding::new(
            name.to_string(),
            value_result.inferred_type.clone(),
            false, // let bindings are immutable
            self.environment.current_level(),
        );
        
        // Add binding to environment
        self.environment.add_binding(binding);
        
        // Infer body type with the new binding
        let body_result = self.infer_expression(body, span)?;
        
        // Merge constraints
        let mut all_constraints = value_result.constraints;
        all_constraints.merge(body_result.constraints);
        
        // Merge effects
        let mut all_effects = value_result.effects;
        all_effects.extend(body_result.effects);

        // The type of a let expression is the type of its body
        Ok(ExpressionInferenceResult {
            inferred_type: body_result.inferred_type,
            constraints: all_constraints,
            environment: self.environment.clone(),
            effects: all_effects,
        })
    }

    fn infer_if_expression(&mut self, condition: &Expr, then_branch: &Expr, else_branch: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Infer condition type
        let condition_result = self.infer_expression(condition, span)?;
        
        // Create constraint that condition must be boolean
        let mut constraints = condition_result.constraints;
        let bool_constraint = TypeConstraint {
            lhs: ConstraintType::Concrete(condition_result.inferred_type.type_info.clone()),
            rhs: ConstraintType::Concrete(self.create_semantic_type("Boolean", span)),
            origin: span,
            reason: ConstraintReason::ConditionalCheck,
            priority: 95,
        };
        constraints.add_constraint(bool_constraint);

        // Infer then branch
        let then_result = self.infer_expression(then_branch, span)?;
        constraints.merge(then_result.constraints);

        let mut all_effects = condition_result.effects;
        all_effects.extend(then_result.effects);

        let result_type = if let Some(else_expr) = else_branch {
            // If-else expression - both branches must have compatible types
            let else_result = self.infer_expression(else_expr, span)?;
            constraints.merge(else_result.constraints);
            all_effects.extend(else_result.effects);

            // Create constraint for branch type unification
            let branch_constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(then_result.inferred_type.type_info.clone()),
                rhs: ConstraintType::Concrete(else_result.inferred_type.type_info.clone()),
                origin: span,
                reason: ConstraintReason::BranchUnification,
                priority: 90,
            };
            constraints.add_constraint(branch_constraint);

            // Return then branch type (unification will ensure compatibility)
            then_result.inferred_type.type_info
        } else {
            // If without else - result type is Unit
            self.create_semantic_type("Unit", span)
        };

        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.85,
            inference_source: InferenceSource::Conditional,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: all_effects,
        })
    }

    fn infer_match_expression(&mut self, expr: &Expr, arms: &[MatchArm], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Infer the expression being matched
        let expr_result = self.infer_expression(expr, span)?;
        let mut all_constraints = expr_result.constraints;
        let mut all_effects = expr_result.effects;

        if arms.is_empty() {
            return Err(SemanticError::TypeInferenceError {
                message: "Match expression must have at least one arm".to_string(),
            });
        }

        // Infer types for all match arms
        let mut arm_types = Vec::new();
        
        for arm in arms {
            // Create new scope for pattern bindings
            self.environment.enter_scope(ScopeKind::Match);
            
            // For now, simplified pattern handling - would use pattern inference engine
            if let Some(guard) = &arm.guard {
                let guard_result = self.infer_expression(&guard.kind, guard.span)?;
                all_constraints.merge(guard_result.constraints);
                all_effects.extend(guard_result.effects);
                
                // Guard must be boolean
                let guard_constraint = TypeConstraint {
                    lhs: ConstraintType::Concrete(guard_result.inferred_type.type_info.clone()),
                    rhs: ConstraintType::Concrete(self.create_semantic_type("Boolean", span)),
                    origin: guard.span,
                    reason: ConstraintReason::GuardCheck,
                    priority: 95,
                };
                all_constraints.add_constraint(guard_constraint);
            }
            
            // Infer arm body type
            let body_result = self.infer_expression(&arm.body.kind, arm.body.span)?;
            all_constraints.merge(body_result.constraints);
            all_effects.extend(body_result.effects);
            
            arm_types.push(body_result.inferred_type.type_info.clone());
            
            // Pop match scope
            self.environment.exit_scope()?;
        }

        // Unify all arm types
        let first_arm_type = &arm_types[0];
        for arm_type in &arm_types[1..] {
            let unification_constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(first_arm_type.clone()),
                rhs: ConstraintType::Concrete(arm_type.clone()),
                origin: span,
                reason: ConstraintReason::MatchArmUnification,
                priority: 90,
            };
            all_constraints.add_constraint(unification_constraint);
        }

        let inferred_type = InferredType {
            type_info: first_arm_type.clone(),
            confidence: 0.85,
            inference_source: InferenceSource::PatternMatch,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: all_constraints,
            environment: self.environment.clone(),
            effects: all_effects,
        })
    }

    fn infer_record_expression(&mut self, fields: &[(String, &Expr)], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let mut field_types = HashMap::new();
        let mut all_constraints = ConstraintSet::new();
        let mut all_effects = Vec::new();

        // Infer type for each field
        for (field_name, field_expr) in fields {
            let field_result = self.infer_expression(field_expr, span)?;
            field_types.insert(field_name.clone(), field_result.inferred_type.type_info.clone());
            all_constraints.merge(field_result.constraints);
            all_effects.extend(field_result.effects);
        }

        let record_type = SemanticType::Record(field_types);
        let inferred_type = InferredType {
            type_info: record_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: all_constraints,
            environment: self.environment.clone(),
            effects: all_effects,
        })
    }

    fn infer_type_annotation(&mut self, expr: &Expr, type_annotation: &AstType, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Resolve the type annotation
        let annotated_type = self.ast_resolver.resolve_ast_type(type_annotation)?;
        
        // Infer the expression type
        let expr_result = self.infer_expression(expr, span)?;
        
        // Create constraint that expression type must match annotation
        let mut constraints = expr_result.constraints;
        let annotation_constraint = TypeConstraint {
            lhs: ConstraintType::Concrete(expr_result.inferred_type.type_info.clone()),
            rhs: ConstraintType::Concrete(annotated_type.clone()),
            origin: span,
            reason: ConstraintReason::TypeAnnotation,
            priority: 100, // Highest priority
        };
        constraints.add_constraint(annotation_constraint);

        // The result type is the annotated type
        let inferred_type = InferredType {
            type_info: annotated_type,
            confidence: 1.0, // High confidence due to explicit annotation
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: expr_result.effects,
        })
    }

    /// Generate a fresh type variable with a name
    pub fn fresh_type_var_named(&mut self, name: String, span: Span) -> TypeVar {
        self.type_var_gen.fresh_named(name, span)
    }

    // Additional expression inference methods for newly added expression types
    fn infer_while_expression(&mut self, condition: &Expr, body: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let condition_result = self.infer_expression(condition, span)?;
        let body_result = self.infer_expression(body, span)?;
        
        let mut constraints = condition_result.constraints;
        constraints.merge(body_result.constraints);
        
        // Condition must be boolean
        let bool_constraint = TypeConstraint {
            lhs: ConstraintType::Concrete(condition_result.inferred_type.type_info.clone()),
            rhs: ConstraintType::Concrete(self.create_semantic_type("Boolean", span)),
            origin: span,
            reason: ConstraintReason::ConditionalCheck,
            priority: 95,
        };
        constraints.add_constraint(bool_constraint);

        let inferred_type = InferredType {
            type_info: self.create_semantic_type("Unit", span),
            confidence: 0.8,
            inference_source: InferenceSource::Loop,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["Loop".to_string()],
        })
    }

    fn infer_for_expression(&mut self, var_name: &str, iterable: &Expr, body: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let iterable_result = self.infer_expression(iterable, span)?;
        
        // Create binding for iterator variable
        let element_type = match &iterable_result.inferred_type.type_info {
            SemanticType::List(element_type) => (**element_type).clone(),
            _ => {
                let var = self.fresh_type_var(span);
                SemanticType::Variable(var.id.to_string())
            }
        };
        
        let binding = TypeBinding::new(
            var_name.to_string(),
            InferredType {
                type_info: element_type,
                confidence: 0.8,
                inference_source: InferenceSource::Iterator,
                constraints: Vec::new(),
                ai_metadata: None,
                span,
            },
            false,
            self.environment.current_level(),
        );
        self.environment.add_binding(binding);
        
        let body_result = self.infer_expression(body, span)?;
        
        let mut constraints = iterable_result.constraints;
        constraints.merge(body_result.constraints);

        let inferred_type = InferredType {
            type_info: self.create_semantic_type("Unit", span),
            confidence: 0.8,
            inference_source: InferenceSource::Loop,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["ForLoop".to_string()],
        })
    }

    fn infer_try_expression(&mut self, try_block: &Expr, catch_clauses: &[CatchClause], finally_block: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let try_result = self.infer_expression(try_block, span)?;
        let mut constraints = try_result.constraints;
        let mut effects = vec!["TryBlock".to_string()];

        // Infer catch clauses
        for catch_clause in catch_clauses {
            let catch_result = self.infer_expression(&catch_clause.body.kind, catch_clause.body.span)?;
            constraints.merge(catch_result.constraints);
            effects.push("CatchBlock".to_string());
        }

        // Infer finally block if present
        if let Some(finally_expr) = finally_block {
            let finally_result = self.infer_expression(finally_expr, span)?;
            constraints.merge(finally_result.constraints);
            effects.push("FinallyBlock".to_string());
        }

        Ok(ExpressionInferenceResult {
            inferred_type: try_result.inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    fn infer_await_expression(&mut self, expr: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let expr_result = self.infer_expression(expr, span)?;
        
        // Await unwraps the Future/Promise type
        let result_type = match &expr_result.inferred_type.type_info {
            SemanticType::Generic { name, parameters } if name == "Future" || name == "Promise" => {
                parameters.get(0).cloned().unwrap_or_else(|| {
                    let var = self.fresh_type_var(span);
                    SemanticType::Variable(var.id.to_string())
                })
            }
            _ => {
                let var = self.fresh_type_var(span);
                SemanticType::Variable(var.id.to_string())
            }
        };

        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.8,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: expr_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["Await".to_string()],
        })
    }

    fn infer_yield_expression(&mut self, value: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let (value_type, constraints) = if let Some(value_expr) = value {
            let value_result = self.infer_expression(value_expr, span)?;
            (value_result.inferred_type.type_info.clone(), value_result.constraints)
        } else {
            (self.create_semantic_type("Unit", span), ConstraintSet::new())
        };

        let generator_type = SemanticType::Generic {
            name: "Generator".to_string(),
            parameters: vec![value_type],
        };

        let inferred_type = InferredType {
            type_info: generator_type,
            confidence: 0.8,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["Yield".to_string()],
        })
    }

    fn infer_actor_expression(&mut self, actor_impl: &Expr, capabilities: &[&Expr], config: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let impl_result = self.infer_expression(actor_impl, span)?;
        let mut constraints = impl_result.constraints;
        let mut effects = vec!["ActorCreation".to_string()];

        // Infer capability expressions
        for cap_expr in capabilities {
            let cap_result = self.infer_expression(cap_expr, span)?;
            constraints.merge(cap_result.constraints);
        }

        // Infer config if present
        if let Some(config_expr) = config {
            let config_result = self.infer_expression(config_expr, span)?;
            constraints.merge(config_result.constraints);
        }

        let actor_type = SemanticType::Generic {
            name: "Actor".to_string(),
            parameters: vec![impl_result.inferred_type.type_info.clone()],
        };

        let inferred_type = InferredType {
            type_info: actor_type,
            confidence: 0.8,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    fn infer_spawn_expression(&mut self, expr: &Expr, spawn_mode: &SpawnMode, capabilities: &[&Expr], priority: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let expr_result = self.infer_expression(expr, span)?;
        let mut constraints = expr_result.constraints;

        // Infer capability expressions
        for cap_expr in capabilities {
            let cap_result = self.infer_expression(cap_expr, span)?;
            constraints.merge(cap_result.constraints);
        }

        // Infer priority if present
        if let Some(priority_expr) = priority {
            let priority_result = self.infer_expression(priority_expr, span)?;
            constraints.merge(priority_result.constraints);
        }

        let task_type = SemanticType::Generic {
            name: "Task".to_string(),
            parameters: vec![expr_result.inferred_type.type_info.clone()],
        };

        let inferred_type = InferredType {
            type_info: task_type,
            confidence: 0.8,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["TaskSpawn".to_string()],
        })
    }

    fn infer_channel_expression(&mut self, channel_type: &Option<Box<AstNode<AstType>>>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let element_semantic_type = if let Some(channel_type) = channel_type {
            self.ast_resolver.resolve_ast_type(&channel_type.kind)?
        } else {
            // Default to a type variable if no channel type specified
            let type_var = self.fresh_type_var(span);
            SemanticType::Variable(type_var.id.to_string())
        };
        
        let channel_semantic_type = SemanticType::Generic {
            name: "Channel".to_string(),
            parameters: vec![element_semantic_type],
        };

        let inferred_type = InferredType {
            type_info: channel_semantic_type,
            confidence: 0.9,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            environment: self.environment.clone(),
            effects: vec!["ChannelCreation".to_string()],
        })
    }

    fn infer_select_expression(&mut self, arms: &[prism_ast::SelectArm], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let mut constraints = ConstraintSet::new();
        let mut arm_types = Vec::new();

        for arm in arms {
            // For now, just infer the body type - we'd need more sophisticated channel handling
            let body_result = self.infer_expression(&arm.body.kind, span)?;
            
            constraints.merge(body_result.constraints);
            arm_types.push(body_result.inferred_type.type_info.clone());
        }

        // Unify all arm types
        let result_type = if let Some(first_type) = arm_types.first() {
            first_type.clone()
        } else {
            self.create_semantic_type("Unit", span)
        };

        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.8,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["SelectExpression".to_string()],
        })
    }

    fn infer_range_expression(&mut self, start: &Expr, end: Option<&Expr>, inclusive: bool, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let start_result = self.infer_expression(start, span)?;
        let mut constraints = start_result.constraints;

        if let Some(end_expr) = end {
            let end_result = self.infer_expression(end_expr, span)?;
            constraints.merge(end_result.constraints);
        }

        let range_type = SemanticType::Generic {
            name: "Range".to_string(),
            parameters: vec![start_result.inferred_type.type_info.clone()],
        };

        let inferred_type = InferredType {
            type_info: range_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: Vec::new(),
        })
    }

    fn infer_tuple_expression(&mut self, elements: &[&Expr], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let mut element_types = Vec::new();
        let mut constraints = ConstraintSet::new();

        for element in elements {
            let element_result = self.infer_expression(element, span)?;
            element_types.push(element_result.inferred_type.type_info.clone());
            constraints.merge(element_result.constraints);
        }

        let tuple_type = SemanticType::Generic {
            name: "Tuple".to_string(),
            parameters: element_types,
        };

        let inferred_type = InferredType {
            type_info: tuple_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: Vec::new(),
        })
    }

    fn infer_block_expression(&mut self, statements: &[&prism_ast::Stmt], final_expr: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        self.environment.enter_scope(ScopeKind::Block);
        
        let mut constraints = ConstraintSet::new();
        let mut effects = Vec::new();

        // Process statements
        for stmt in statements {
            // Simplified statement processing
            effects.push("Statement".to_string());
        }

        // Process final expression
        let result_type = if let Some(final_expression) = final_expr {
            let final_result = self.infer_expression(final_expression, span)?;
            constraints.merge(final_result.constraints);
            effects.extend(final_result.effects);
            final_result.inferred_type.type_info
        } else {
            self.create_semantic_type("Unit", span)
        };

        self.environment.exit_scope()?;

        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.8,
            inference_source: InferenceSource::Block,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects,
        })
    }

    fn infer_return_expression(&mut self, value: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let (return_type, constraints) = if let Some(value_expr) = value {
            let value_result = self.infer_expression(value_expr, span)?;
            (value_result.inferred_type.type_info.clone(), value_result.constraints)
        } else {
            (self.create_semantic_type("Unit", span), ConstraintSet::new())
        };

        let inferred_type = InferredType {
            type_info: return_type,
            confidence: 0.9,
            inference_source: InferenceSource::Return,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["Return".to_string()],
        })
    }

    fn infer_break_expression(&mut self, value: Option<&Expr>, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let (break_type, constraints) = if let Some(value_expr) = value {
            let value_result = self.infer_expression(value_expr, span)?;
            (value_result.inferred_type.type_info.clone(), value_result.constraints)
        } else {
            (self.create_semantic_type("Unit", span), ConstraintSet::new())
        };

        let inferred_type = InferredType {
            type_info: break_type,
            confidence: 0.9,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["Break".to_string()],
        })
    }

    fn infer_continue_expression(&mut self, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let inferred_type = InferredType {
            type_info: self.create_semantic_type("Unit", span),
            confidence: 1.0,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            environment: self.environment.clone(),
            effects: vec!["Continue".to_string()],
        })
    }

    fn infer_throw_expression(&mut self, exception: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let exception_result = self.infer_expression(exception, span)?;

        let inferred_type = InferredType {
            type_info: self.create_semantic_type("Unit", span), // Throw never returns
            confidence: 1.0,
            inference_source: InferenceSource::Context,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: exception_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["Throw".to_string()],
        })
    }

    fn infer_formatted_string_expression(&mut self, parts: &[&prism_ast::FStringPart], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let mut constraints = ConstraintSet::new();

        // Infer types for expression parts
        for part in parts {
            match part {
                prism_ast::FStringPart::Literal(_) => {
                    // Literal parts don't need type inference
                }
                prism_ast::FStringPart::Expression { expression, .. } => {
                    let part_result = self.infer_expression(&expression.kind, span)?;
                    constraints.merge(part_result.constraints);
                }
            }
        }

        let inferred_type = InferredType {
            type_info: self.create_semantic_type("String", span),
            confidence: 1.0,
            inference_source: InferenceSource::Literal,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: Vec::new(),
        })
    }

    fn infer_list_comprehension(&mut self, element: &Expr, generators: &[Generator], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        // Simplified comprehension inference
        let element_result = self.infer_expression(element, span)?;
        
        let list_type = SemanticType::List(Box::new(element_result.inferred_type.type_info.clone()));

        let inferred_type = InferredType {
            type_info: list_type,
            confidence: 0.8,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: element_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["ListComprehension".to_string()],
        })
    }

    fn infer_set_comprehension(&mut self, element: &Expr, generators: &[Generator], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let element_result = self.infer_expression(element, span)?;
        
        let set_type = SemanticType::Generic {
            name: "Set".to_string(),
            parameters: vec![element_result.inferred_type.type_info.clone()],
        };

        let inferred_type = InferredType {
            type_info: set_type,
            confidence: 0.8,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: element_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["SetComprehension".to_string()],
        })
    }

    fn infer_dict_comprehension(&mut self, key: &Expr, value: &Expr, generators: &[Generator], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let key_result = self.infer_expression(key, span)?;
        let value_result = self.infer_expression(value, span)?;
        
        let mut constraints = key_result.constraints;
        constraints.merge(value_result.constraints);
        
        let dict_type = SemanticType::Generic {
            name: "Dict".to_string(),
            parameters: vec![key_result.inferred_type.type_info.clone(), value_result.inferred_type.type_info.clone()],
        };

        let inferred_type = InferredType {
            type_info: dict_type,
            confidence: 0.8,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
            effects: vec!["DictComprehension".to_string()],
        })
    }

    fn infer_generator_expression(&mut self, element: &Expr, generators: &[Generator], span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let element_result = self.infer_expression(element, span)?;
        
        let generator_type = SemanticType::Generic {
            name: "Generator".to_string(),
            parameters: vec![element_result.inferred_type.type_info.clone()],
        };

        let inferred_type = InferredType {
            type_info: generator_type,
            confidence: 0.8,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: element_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["GeneratorExpression".to_string()],
        })
    }

    fn infer_named_expression(&mut self, name: &str, value: &Expr, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let value_result = self.infer_expression(value, span)?;
        
        // Create binding for the named variable
        let binding = TypeBinding::new(
            name.to_string(),
            value_result.inferred_type.clone(),
            false,
            self.environment.current_level(),
        );
        self.environment.add_binding(binding);

        Ok(ExpressionInferenceResult {
            inferred_type: value_result.inferred_type,
            constraints: value_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["NamedExpression".to_string()],
        })
    }

    fn infer_starred_expression(&mut self, value: &Expr, context: &prism_ast::ExpressionContext, span: Span) -> SemanticResult<ExpressionInferenceResult> {
        let value_result = self.infer_expression(value, span)?;

        // Starred expressions typically unpack sequences
        let starred_type = match &value_result.inferred_type.type_info {
            SemanticType::List(element_type) => (**element_type).clone(),
            _ => value_result.inferred_type.type_info.clone(),
        };

        let inferred_type = InferredType {
            type_info: starred_type,
            confidence: 0.8,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(ExpressionInferenceResult {
            inferred_type,
            constraints: value_result.constraints,
            environment: self.environment.clone(),
            effects: vec!["StarredExpression".to_string()],
        })
    }
}

impl InferenceEngine for ExpressionInferenceEngine {
    type Input = (Expr, Span);
    type Output = ExpressionInferenceResult;
    
    fn infer(&mut self, input: Self::Input) -> SemanticResult<Self::Output> {
        let (expr, span) = input;
        self.infer_expression(&expr, span)
    }
    
    fn engine_name(&self) -> &'static str {
        "ExpressionInferenceEngine"
    }
    
    fn reset(&mut self) {
        self.type_var_gen = TypeVarGenerator::new();
        self.environment = TypeEnvironment::new();
        self.current_depth = 0;
    }
}

impl EffectAware for ExpressionInferenceEngine {
    fn infer_effects(&self, input: &Self::Input) -> SemanticResult<Vec<String>> {
        let (expr, _span) = input;
        // Basic effect inference - would be more sophisticated in practice
        match expr {
            Expr::Call(_) => Ok(vec!["FunctionCall".to_string()]),
            Expr::Member(_) => Ok(vec!["FieldAccess".to_string()]),
            Expr::Index(_) => Ok(vec!["ArrayAccess".to_string()]),
            _ => Ok(Vec::new()),
        }
    }
} 