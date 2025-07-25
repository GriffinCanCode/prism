//! Statement Type Inference Engine
//!
//! This module implements type inference specifically for statements.
//! It handles function declarations, let bindings, type definitions, and other statements.
//!
//! **Single Responsibility**: Type inference for statements
//! **What it does**: Infer types for statement constructs, manage bindings
//! **What it doesn't do**: Handle expressions, patterns, or orchestration

use super::{InferenceEngine, ExpressionInferenceEngine, PatternInferenceEngine, ASTTypeResolver};
use super::expression_inference::ExpressionInferenceResult;
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        TypeVar, InferredType, InferenceSource, TypeVarGenerator, TypeInferenceResult,
        environment::{TypeEnvironment, TypeBinding, ScopeKind},
        constraints::{TypeConstraint, ConstraintType, ConstraintReason, ConstraintSet},
        unification::Substitution,
    },
};
use prism_ast::{Stmt, Pattern, Type as AstType, FunctionDecl, TypeDecl, ModuleDecl, 
    LetDecl, LetStmt, FunctionStmt, TypeStmt};
use prism_common::{NodeId, Span};
use std::collections::HashMap;

/// Engine specifically for statement type inference
#[derive(Debug)]
pub struct StatementInferenceEngine {
    /// Type variable generator
    type_var_gen: TypeVarGenerator,
    /// Current type environment
    environment: TypeEnvironment,
    /// Expression inference engine for nested expressions
    expression_engine: ExpressionInferenceEngine,
    /// Pattern inference engine for patterns
    pattern_engine: PatternInferenceEngine,
    /// AST type resolver
    ast_resolver: ASTTypeResolver,
    /// Current inference depth
    current_depth: usize,
    /// Maximum inference depth
    max_depth: usize,
}

/// Result of statement inference
#[derive(Debug, Clone)]
pub struct StatementInferenceResult {
    /// Main inference result
    pub inference_result: TypeInferenceResult,
    /// New bindings created by the statement
    pub new_bindings: Vec<TypeBinding>,
    /// Effects inferred from the statement
    pub effects: Vec<String>,
}

impl StatementInferenceEngine {
    /// Create a new statement inference engine
    pub fn new() -> SemanticResult<Self> {
        Ok(Self {
            type_var_gen: TypeVarGenerator::new(),
            environment: TypeEnvironment::new(),
            expression_engine: ExpressionInferenceEngine::new()?,
            pattern_engine: PatternInferenceEngine::new()?,
            ast_resolver: ASTTypeResolver::new(),
            current_depth: 0,
            max_depth: 1000,
        })
    }

    /// Infer type for a statement
    pub fn infer_statement(&mut self, stmt: &Stmt) -> SemanticResult<TypeInferenceResult> {
        self.check_depth_limit()?;
        self.current_depth += 1;
        
        let result = match stmt {
            Stmt::Expression(expr_stmt) => {
                let expr_result = self.expression_engine.infer_expression(&expr_stmt.expression.kind, expr_stmt.expression.span)?;
                let mut result = TypeInferenceResult::empty();
                result.add_type(expr_stmt.expression.id, expr_result.inferred_type);
                result.constraints = expr_result.constraints;
                result.global_env = expr_result.environment;
                result
            }
            Stmt::Variable(var_decl) => {
                self.infer_variable_declaration(var_decl)?
            }
            Stmt::Function(func_stmt) => {
                self.infer_function_statement(func_stmt)?
            }
            Stmt::Type(type_decl) => {
                self.infer_type_declaration(type_decl)?
            }
            Stmt::Return(return_stmt) => {
                self.infer_return_statement(return_stmt)?
            }
            Stmt::If(if_stmt) => {
                self.infer_if_statement(if_stmt)?
            }
            Stmt::While(while_stmt) => {
                self.infer_while_statement(while_stmt)?
            }
            Stmt::For(for_stmt) => {
                self.infer_for_statement(for_stmt)?
            }
            Stmt::Match(match_stmt) => {
                self.infer_match_statement(match_stmt)?
            }
            Stmt::Block(block_stmt) => {
                self.infer_block_statement(block_stmt)?
            }
            _ => {
                // Default case for unsupported statements
                TypeInferenceResult::empty()
            }
        };
        
        self.current_depth -= 1;
        Ok(result)
    }

    /// Infer type for a function declaration
    pub fn infer_function_declaration(&mut self, func_decl: &FunctionDecl) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Enter function scope
        self.environment.enter_scope(ScopeKind::Function);
        
        // Create type variables for parameters
        let mut param_types = Vec::new();
        for param in &func_decl.parameters {
            // Create a simple identifier pattern from parameter name
            let param_pattern = prism_ast::AstNode {
                kind: prism_ast::Pattern::Identifier(prism_ast::IdentifierPattern {
                    name: param.name.clone(),
                    type_annotation: param.type_annotation.clone(),
                    is_mutable: param.is_mutable,
                }),
                span: prism_common::Span::dummy(),
                id: prism_common::NodeId::new(0),
                metadata: prism_ast::NodeMetadata::default(),
            };
            let param_result = self.pattern_engine.infer_pattern(&param_pattern, None)?;
            param_types.push(param_result.inferred_type.type_info.clone());
            
            // Add parameter binding to environment
            let param_binding = TypeBinding {
                name: param.name.resolve().unwrap_or_else(|| "param".to_string()),
                type_info: param_result.inferred_type,
                is_mutable: param.is_mutable,
                scope_level: self.environment.current_level(),
                definition_node: None, // Symbol doesn't have node_id method
                is_polymorphic: false,
                type_scheme: None,
            };
            self.environment.add_binding(param_binding);
        }
        
        // Infer body type
        let body_result = if let Some(body) = &func_decl.body {
            // Function body is a statement, so we need to infer it as a statement
            let stmt_result = self.infer_statement(&body.kind)?;
            
            // Convert statement result to expression result format
            ExpressionInferenceResult {
                inferred_type: InferredType {
                    type_info: self.create_semantic_type("Unit", prism_common::Span::dummy()),
                    confidence: 0.8,
                    inference_source: InferenceSource::LetBinding,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: prism_common::Span::dummy(),
                },
                constraints: stmt_result.constraints,
                environment: stmt_result.global_env,
                effects: Vec::new(),
            }
        } else {
            // Function has no body (e.g., external declaration)
            let default_type = self.create_semantic_type("Unit", prism_common::Span::dummy());
            ExpressionInferenceResult {
                inferred_type: InferredType {
                    type_info: default_type,
                    confidence: 1.0,
                    inference_source: InferenceSource::Default,
                                    constraints: Vec::new(),
                ai_metadata: None,
                span: prism_common::Span::dummy(),
            },
            constraints: ConstraintSet::new(),
                environment: self.environment.clone(),
                effects: Vec::new(),
            }
        };
        result.constraints.extend(body_result.constraints);
        
        // Handle return type
        let function_return_type = if let Some(return_annotation) = &func_decl.return_type {
            let annotated_return_type = self.ast_resolver.resolve_ast_type(&return_annotation.kind)?;
            
            // Add constraint that body type must match return annotation
            result.constraints.add_constraint(TypeConstraint {
                lhs: ConstraintType::Concrete(body_result.inferred_type.type_info),
                rhs: ConstraintType::Concrete(annotated_return_type.clone()),
                origin: return_annotation.span,
                reason: ConstraintReason::TypeAnnotation,
                priority: 95,
            });
            
            annotated_return_type
        } else {
            body_result.inferred_type.type_info
        };
        
        // Infer effects from function body
        let effects = Vec::new(); // Simplified for now
        
        // Create function type
        let function_type = SemanticType::Function {
            params: param_types,
            return_type: Box::new(function_return_type),
            effects,
        };
        
        // Exit function scope
        self.environment.exit_scope();
        
        // Add function binding to global environment
        let inferred_func_type = InferredType {
            type_info: function_type.clone(),
            confidence: 1.0,
            inference_source: InferenceSource::Function,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        };
        
        let function_binding = TypeBinding::new(
            func_decl.name.to_string(),
            inferred_func_type,
            false, // functions are not mutable
            0, // global scope
        );
        self.environment.add_binding(function_binding);
        
        // Store the inferred type for this node
        let node_id = NodeId::new(0); // Use default since FunctionDecl doesn't have span
        result.add_type(node_id, InferredType {
            type_info: function_type,
            confidence: 0.9,
            inference_source: InferenceSource::Function,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    /// Infer type for a type declaration
    pub fn infer_type_declaration(&mut self, type_decl: &TypeDecl) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Resolve the type definition
        let resolved_type = match &type_decl.kind {
            prism_ast::TypeKind::Alias(type_node) => {
                self.ast_resolver.resolve_ast_type(&type_node.kind)?
            }
            prism_ast::TypeKind::Semantic(semantic_type) => {
                // Convert from prism_ast::SemanticType to types::SemanticType
                self.convert_ast_semantic_type(semantic_type)?
            }
            _ => {
                // For other type kinds, create a default semantic type
                self.create_semantic_type(&type_decl.name.resolve().unwrap_or_else(|| "Unknown".to_string()), prism_common::Span::dummy())
            }
        };
        
        // Create a type alias binding
        let type_binding = TypeBinding::new(
            type_decl.name.to_string(),
            InferredType {
                type_info: resolved_type.clone(),
                confidence: 1.0,
                inference_source: InferenceSource::Explicit,
                constraints: Vec::new(),
                ai_metadata: None,
                span: prism_common::Span::dummy(),
            },
            false, // type aliases are not mutable
            0, // global scope
        );
        
        self.environment.add_global_binding(type_binding);
        
        // Store the type definition in the result
        let dummy_span = prism_common::Span::dummy();
        let node_id = NodeId::new(0); // Use dummy since TypeDecl doesn't have span
        result.add_type(node_id, InferredType {
            type_info: resolved_type,
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    /// Infer type for a variable declaration
    pub fn infer_variable_declaration(&mut self, var_decl: &prism_ast::VariableDecl) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer the type of the initializer if present
        let binding_type = if let Some(initializer) = &var_decl.initializer {
            let value_result = self.expression_engine.infer_expression(&initializer.kind, initializer.span)?;
            result.constraints.extend(value_result.constraints);
            
            // Check if there's a type annotation
            if let Some(annotation) = &var_decl.type_annotation {
                // Resolve the annotated type
                let annotated_type = self.ast_resolver.resolve_ast_type(&annotation.kind)?;
                
                // Add constraint that value type must match annotation
                result.constraints.add_constraint(TypeConstraint {
                    lhs: ConstraintType::Concrete(value_result.inferred_type.type_info),
                    rhs: ConstraintType::Concrete(annotated_type.clone()),
                    origin: prism_common::Span::dummy(), // Use dummy span since VariableDecl doesn't have one
                    reason: ConstraintReason::TypeAnnotation,
                    priority: 100,
                });
                
                annotated_type
            } else {
                // Use inferred type
                value_result.inferred_type.type_info.clone()
            }
        } else {
            // No initializer, must have type annotation
            if let Some(annotation) = &var_decl.type_annotation {
                self.ast_resolver.resolve_ast_type(&annotation.kind)?
            } else {
                return Err(SemanticError::TypeInferenceError {
                    message: "Variable declaration must have either initializer or type annotation".to_string(),
                });
            }
        };
        
        // Create a type binding for the variable
        let inferred_type = InferredType {
            type_info: binding_type.clone(),
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        };
        
        let binding = TypeBinding::new(
            var_decl.name.to_string(),
            inferred_type,
            var_decl.is_mutable,
            self.environment.current_level(),
        );
        self.environment.add_binding(binding);

        let node_id = NodeId::new(0); // Use dummy since VariableDecl doesn't have span
        result.add_type(node_id, InferredType {
            type_info: binding_type,
            confidence: 1.0,
            inference_source: InferenceSource::Variable,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    /// Infer type for a let declaration
    pub fn infer_let_declaration(&mut self, let_decl: &prism_ast::LetDecl) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer the type of the value expression
        let value_result = if let Some(initializer) = &let_decl.initializer {
            self.expression_engine.infer_expression(&initializer.kind, initializer.span)?
        } else {
            // If no initializer, use unit type
            let unit_type = self.create_semantic_type("Unit", prism_common::Span::dummy());
            super::expression_inference::ExpressionInferenceResult {
                inferred_type: InferredType {
                    type_info: unit_type,
                    confidence: 1.0,
                    inference_source: InferenceSource::Default,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: prism_common::Span::dummy(),
                },
                constraints: ConstraintSet::new(),
                environment: self.environment.clone(),
                effects: Vec::new(),
            }
        };
        result.constraints.extend(value_result.constraints);
        
        // Check if there's a type annotation
        let binding_type = if let Some(annotation) = &let_decl.type_annotation {
            // Resolve the annotated type
            let annotated_type = self.ast_resolver.resolve_ast_type(&annotation.kind)?;
            
            // Add constraint that value type must match annotation
            result.constraints.add_constraint(TypeConstraint {
                lhs: ConstraintType::Concrete(value_result.inferred_type.type_info),
                rhs: ConstraintType::Concrete(annotated_type.clone()),
                origin: prism_common::Span::dummy(),
                reason: ConstraintReason::TypeAnnotation,
                priority: 100,
            });
            
            annotated_type
        } else {
            // Use inferred type
            value_result.inferred_type.type_info.clone()
        };
        
        // Create a type binding for the variable
        let inferred_type = InferredType {
            type_info: binding_type.clone(),
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        };
        
        let binding = TypeBinding::new(
            let_decl.name.to_string(),
            inferred_type,
            let_decl.is_mutable,
            self.environment.current_level(),
        );
        self.environment.add_binding(binding);

        let node_id = NodeId::new(0);
        result.add_type(node_id, InferredType {
            type_info: binding_type,
            confidence: if let_decl.type_annotation.is_some() { 1.0 } else { value_result.inferred_type.confidence },
            inference_source: InferenceSource::Variable,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    /// Infer type for a const declaration
    pub fn infer_const_declaration(&mut self, const_decl: &prism_ast::ConstDecl) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer the type of the value expression
        let value_result = self.expression_engine.infer_expression(&const_decl.value.kind, const_decl.value.span)?;
        result.constraints.extend(value_result.constraints);
        
        // Check if there's a type annotation
        let binding_type = if let Some(annotation) = &const_decl.type_annotation {
            // Resolve the annotated type
            let annotated_type = self.ast_resolver.resolve_ast_type(&annotation.kind)?;
            
            // Add constraint that value type must match annotation
            result.constraints.add_constraint(TypeConstraint {
                lhs: ConstraintType::Concrete(value_result.inferred_type.type_info),
                rhs: ConstraintType::Concrete(annotated_type.clone()),
                origin: prism_common::Span::dummy(),
                reason: ConstraintReason::TypeAnnotation,
                priority: 100,
            });
            
            annotated_type
        } else {
            // Use inferred type
            value_result.inferred_type.type_info.clone()
        };
        
        // Create a type binding for the constant (immutable)
        let inferred_type = InferredType {
            type_info: binding_type.clone(),
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        };
        
        let binding = TypeBinding::new(
            const_decl.name.to_string(),
            inferred_type,
            false, // Constants are never mutable
            self.environment.current_level(),
        );
        self.environment.add_binding(binding);

        let node_id = NodeId::new(0);
        result.add_type(node_id, InferredType {
            type_info: binding_type,
            confidence: if const_decl.type_annotation.is_some() { 1.0 } else { value_result.inferred_type.confidence },
            inference_source: InferenceSource::Variable,
            constraints: Vec::new(),
            ai_metadata: None,
            span: prism_common::Span::dummy(),
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    /// Infer type for a module declaration
    pub fn infer_module_declaration(&mut self, module_decl: &ModuleDecl) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Enter module scope (use Block since Module doesn't exist)
        self.environment.enter_scope(ScopeKind::Block);
        
        // Process module sections
        for section in &module_decl.sections {
            for item in &section.kind.items {
                let item_result = self.infer_statement(&item.kind)?;
                result.merge(item_result);
            }
        }
        
        // Exit module scope
        self.environment.exit_scope();
        
        // Create module type (simplified for now)
        let dummy_span = prism_common::Span::dummy();
        let module_type = SemanticType::Complex {
            name: module_decl.name.to_string(),
            base_type: crate::types::BaseType::Composite(crate::types::CompositeType {
                kind: crate::types::CompositeKind::Record,
                fields: Vec::new(),
                methods: Vec::new(),
                inheritance: Vec::new(),
            }),
            constraints: Vec::new(),
            business_rules: Vec::new(),
            metadata: crate::types::SemanticTypeMetadata::default(),
            ai_context: None,
            verification_properties: Vec::new(),
            location: dummy_span,
        };
        
        let node_id = NodeId::new(0); // Use dummy since ModuleDecl doesn't have span
        result.add_type(node_id, InferredType {
            type_info: module_type,
            confidence: 1.0,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
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

    /// Infer effects from expression (simplified)
    fn infer_effects_from_body(&self, _expr: &prism_ast::Expr) -> SemanticResult<Vec<String>> {
        // Simplified effect inference - would be more sophisticated in practice
        Ok(Vec::new())
    }

    /// Generate a fresh type variable
    fn fresh_type_var(&mut self, span: Span) -> TypeVar {
        self.type_var_gen.fresh(span)
    }

    // Placeholder methods for other statement types
    fn infer_let_statement(&mut self, let_stmt: &prism_ast::LetStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer the value type
        let value_result = self.expression_engine.infer_expression(&let_stmt.value.kind, let_stmt.value.span)?;
        
        // Handle type annotation if present
        let declared_type = if let Some(type_annotation) = &let_stmt.type_annotation {
            Some(self.ast_resolver.resolve_ast_type(&type_annotation.kind)?)
        } else {
            None
        };
        
        // Create constraint if type annotation exists
        if let Some(declared_type) = &declared_type {
            let constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(value_result.inferred_type.type_info.clone()),
                rhs: ConstraintType::Concrete(declared_type.clone()),
                origin: let_stmt.value.span,
                reason: ConstraintReason::TypeAnnotation,
                priority: 100,
            };
            result.constraints.add_constraint(constraint);
        }
        
        // Determine final type
        let final_type = declared_type.as_ref().map(|t| t.clone()).unwrap_or_else(|| value_result.inferred_type.type_info.clone());
        
        // Create binding for the variable
        // Extract variable name from pattern (simplified - assumes identifier pattern)
        let var_name = match &let_stmt.pattern.kind {
            prism_ast::Pattern::Identifier(id_pattern) => id_pattern.name.resolve().unwrap_or_else(|| "unknown".to_string()),
            _ => "unknown".to_string(), // For now, only handle identifier patterns
        };
        let binding = TypeBinding::new(
            var_name.clone(),
            InferredType {
                type_info: final_type.clone(),
                confidence: if declared_type.is_some() { 1.0 } else { value_result.inferred_type.confidence },
                inference_source: if declared_type.is_some() { InferenceSource::Explicit } else { value_result.inferred_type.inference_source },
                constraints: Vec::new(),
                ai_metadata: None,
                span: let_stmt.pattern.span,
            },
            // Check if pattern is mutable (simplified)
            match &let_stmt.pattern.kind {
                prism_ast::Pattern::Identifier(id_pattern) => id_pattern.is_mutable,
                _ => false,
            },
            self.environment.current_level(),
        );
        
        // Add binding to environment
        self.environment.add_binding(binding);
        
        // Add type to result
        let node_id = NodeId::new(let_stmt.pattern.span.start.offset);
        result.add_type(node_id, InferredType {
            type_info: final_type,
            confidence: 0.9,
            inference_source: InferenceSource::LetBinding,
            constraints: Vec::new(),
            ai_metadata: None,
            span: let_stmt.pattern.span,
        });
        
        // Merge constraints from value inference
        result.constraints.merge(value_result.constraints);
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    fn infer_function_statement(&mut self, func_stmt: &prism_ast::FunctionStmt) -> SemanticResult<TypeInferenceResult> {
        // This would be similar to infer_function_declaration but for statement form
        // For now, delegate to the declaration inference
        let func_decl = FunctionDecl {
            name: func_stmt.name.clone(),
            parameters: func_stmt.parameters.clone(),
            return_type: func_stmt.return_type.clone(),
            body: func_stmt.body.clone(),
            visibility: prism_ast::Visibility::Private, // Default visibility
            attributes: Vec::new(),
            contracts: None,
            is_async: false,
        };
        
        self.infer_function_declaration(&func_decl)
    }

    fn infer_type_statement(&mut self, type_stmt: &prism_ast::TypeStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Resolve the type definition
        let type_name = type_stmt.name.resolve().unwrap_or_else(|| "unknown".to_string());
        let defined_type = self.ast_resolver.resolve_ast_type(&type_stmt.type_expr.kind)?;
        
        // Create type alias binding
        let binding = TypeBinding::new(
            type_name.clone(),
            InferredType {
                type_info: defined_type.clone(),
                confidence: 1.0,
                inference_source: InferenceSource::TypeDefinition,
                constraints: Vec::new(),
                ai_metadata: None,
                span: type_stmt.type_expr.span,
            },
            false, // Type aliases are immutable
            self.environment.current_level(),
        );
        
        self.environment.add_binding(binding);
        
        // Add type to result
        let node_id = NodeId::new(type_stmt.type_expr.span.start.offset);
        result.add_type(node_id, InferredType {
            type_info: defined_type,
            confidence: 1.0,
            inference_source: InferenceSource::TypeDefinition,
            constraints: Vec::new(),
            ai_metadata: None,
            span: type_stmt.type_expr.span,
        });
        
        result.global_env = self.environment.clone();
        Ok(result)
    }

    fn infer_return_statement(&mut self, return_stmt: &prism_ast::ReturnStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        let dummy_span = prism_common::Span::dummy();
        let return_type = if let Some(value) = &return_stmt.value {
            // Infer the return value type
            let value_result = self.expression_engine.infer_expression(&value.kind, value.span)?;
            result.constraints.merge(value_result.constraints);
            value_result.inferred_type.type_info
        } else {
            // No return value - Unit type
            self.create_semantic_type("Unit", dummy_span)
        };
        
        // Add type to result
        let node_id = NodeId::new(0); // Use dummy since ReturnStmt doesn't have span
        result.add_type(node_id, InferredType {
            type_info: return_type,
            confidence: 0.9,
            inference_source: InferenceSource::Return,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        result.global_env = self.environment.clone();
        Ok(result)
    }

    fn infer_if_statement(&mut self, if_stmt: &prism_ast::IfStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer condition type
        let condition_result = self.expression_engine.infer_expression(&if_stmt.condition.kind, if_stmt.condition.span)?;
        result.constraints.merge(condition_result.constraints);
        
        // Condition must be boolean
        let bool_constraint = TypeConstraint {
            lhs: ConstraintType::Concrete(condition_result.inferred_type.type_info.clone()),
            rhs: ConstraintType::Concrete(self.create_semantic_type("Boolean", if_stmt.condition.span)),
            origin: if_stmt.condition.span,
            reason: ConstraintReason::ConditionalCheck,
            priority: 95,
        };
        result.constraints.add_constraint(bool_constraint);
        
        // Infer then branch
        let then_result = self.infer_statement(&if_stmt.then_branch.kind)?;
        result.merge(then_result);
        
        // Infer else branch if present
        if let Some(else_branch) = &if_stmt.else_branch {
            let else_result = self.infer_statement(&else_branch.kind)?;
            result.merge(else_result);
        }
        
        // If statement itself has Unit type
        let dummy_span = prism_common::Span::dummy();
        let node_id = NodeId::new(0); // Use dummy since IfStmt doesn't have span
        result.add_type(node_id, InferredType {
            type_info: self.create_semantic_type("Unit", dummy_span),
            confidence: 0.9,
            inference_source: InferenceSource::Conditional,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        Ok(result)
    }

    fn infer_while_statement(&mut self, while_stmt: &prism_ast::WhileStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer condition type
        let condition_result = self.expression_engine.infer_expression(&while_stmt.condition.kind, while_stmt.condition.span)?;
        result.constraints.merge(condition_result.constraints);
        
        // Condition must be boolean
        let bool_constraint = TypeConstraint {
            lhs: ConstraintType::Concrete(condition_result.inferred_type.type_info.clone()),
            rhs: ConstraintType::Concrete(self.create_semantic_type("Boolean", while_stmt.condition.span)),
            origin: while_stmt.condition.span,
            reason: ConstraintReason::ConditionalCheck,
            priority: 95,
        };
        result.constraints.add_constraint(bool_constraint);
        
        // Create new scope for loop body
        self.environment.enter_scope(ScopeKind::Loop);
        
        // Infer body
        let body_result = self.infer_statement(&while_stmt.body.kind)?;
        result.merge(body_result);
        
        // Pop loop scope
        self.environment.exit_scope()?;
        
        // While statement has Unit type
        let dummy_span = prism_common::Span::dummy();
        let node_id = NodeId::new(0); // Use dummy since WhileStmt doesn't have span
        result.add_type(node_id, InferredType {
            type_info: self.create_semantic_type("Unit", dummy_span),
            confidence: 0.9,
            inference_source: InferenceSource::Loop,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        Ok(result)
    }

    fn infer_for_statement(&mut self, for_stmt: &prism_ast::ForStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer iterable type
        let iterable_result = self.expression_engine.infer_expression(&for_stmt.iterable.kind, for_stmt.iterable.span)?;
        result.constraints.merge(iterable_result.constraints);
        
        // Create new scope for loop
        self.environment.enter_scope(ScopeKind::Loop);
        
        // Infer iterator variable type based on iterable
        let dummy_span = prism_common::Span::dummy();
        let iterator_type = match &iterable_result.inferred_type.type_info {
            SemanticType::List(element_type) => (**element_type).clone(),
            SemanticType::Generic { name, parameters } if name == "Array" || name == "List" => {
                parameters.get(0).cloned().unwrap_or_else(|| {
                    let var = self.fresh_type_var(dummy_span);
                    SemanticType::Variable(var.id.to_string())
                })
            }
            _ => {
                // Create type variable for unknown iterable
                let var = self.fresh_type_var(dummy_span);
                SemanticType::Variable(var.id.to_string())
            }
        };
        
        // Create binding for iterator variable
        let iterator_name = for_stmt.variable.resolve().unwrap_or_else(|| "it".to_string());
        let binding = TypeBinding::new(
            iterator_name,
            InferredType {
                type_info: iterator_type,
                confidence: 0.8,
                inference_source: InferenceSource::Iterator,
                constraints: Vec::new(),
                ai_metadata: None,
                span: dummy_span,
            },
            false, // Iterator variables are immutable
            self.environment.current_level(),
        );
        self.environment.add_binding(binding);
        
        // Infer body
        let body_result = self.infer_statement(&for_stmt.body.kind)?;
        result.merge(body_result);
        
        // Pop loop scope
        self.environment.exit_scope()?;
        
        // For statement has Unit type
        let node_id = NodeId::new(0); // Use dummy since ForStmt doesn't have span
        result.add_type(node_id, InferredType {
            type_info: self.create_semantic_type("Unit", dummy_span),
            confidence: 0.9,
            inference_source: InferenceSource::Loop,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        Ok(result)
    }

    fn infer_match_statement(&mut self, match_stmt: &prism_ast::MatchStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Infer the expression being matched
        let expr_result = self.expression_engine.infer_expression(&match_stmt.expression.kind, match_stmt.expression.span)?;
        result.constraints.merge(expr_result.constraints);
        
        // Infer each match arm
        for arm in &match_stmt.arms {
            // Create new scope for pattern bindings
            self.environment.enter_scope(ScopeKind::Match);
            
            // For now, simplified pattern handling
            // In a full implementation, would use pattern inference engine
            
            // Infer guard if present
            if let Some(guard) = &arm.guard {
                let guard_result = self.expression_engine.infer_expression(&guard.kind, guard.span)?;
                result.constraints.merge(guard_result.constraints);
                
                // Guard must be boolean
                let guard_constraint = TypeConstraint {
                    lhs: ConstraintType::Concrete(guard_result.inferred_type.type_info.clone()),
                    rhs: ConstraintType::Concrete(self.create_semantic_type("Boolean", guard.span)),
                    origin: guard.span,
                    reason: ConstraintReason::GuardCheck,
                    priority: 95,
                };
                result.constraints.add_constraint(guard_constraint);
            }
            
            // Infer arm body
            let body_result = self.infer_statement(&arm.body.kind)?;
            result.merge(body_result);
            
            // Pop match scope
            self.environment.exit_scope()?;
        }
        
        // Match statement has Unit type
        let dummy_span = prism_common::Span::dummy();
        let node_id = NodeId::new(0); // Use dummy since MatchStmt doesn't have span
        result.add_type(node_id, InferredType {
            type_info: self.create_semantic_type("Unit", dummy_span),
            confidence: 0.9,
            inference_source: InferenceSource::PatternMatch,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        Ok(result)
    }

    fn infer_block_statement(&mut self, block_stmt: &prism_ast::BlockStmt) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Create new scope for block
        self.environment.enter_scope(ScopeKind::Block);
        
        // Infer all statements in the block
        for stmt in &block_stmt.statements {
            let stmt_result = self.infer_statement(&stmt.kind)?;
            result.merge(stmt_result);
        }
        
        // Pop block scope
        self.environment.exit_scope()?;
        
        // Block statement has Unit type - use dummy span since BlockStmt doesn't have one
        let dummy_span = prism_common::span::Span::dummy();
        let node_id = NodeId::new(0); // Use dummy offset since we don't have span
        result.add_type(node_id, InferredType {
            type_info: self.create_semantic_type("Unit", dummy_span),
            confidence: 0.9,
            inference_source: InferenceSource::Block,
            constraints: Vec::new(),
            ai_metadata: None,
            span: dummy_span,
        });
        
        Ok(result)
    }

    /// Create a semantic type for basic types
    fn create_semantic_type(&self, name: &str, span: Span) -> SemanticType {
        let primitive_type = match name {
            "Integer" => prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(64)),
            "Float" => prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64),
            "String" => prism_ast::PrimitiveType::String,
            "Boolean" => prism_ast::PrimitiveType::Boolean,
            "Unit" => prism_ast::PrimitiveType::Unit,
            _ => prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(64)), // Default fallback
        };

        SemanticType::primitive(name, primitive_type, span)
    }

    /// Convert AST semantic type to semantic type
    fn convert_ast_semantic_type(&mut self, ast_semantic_type: &prism_ast::SemanticType) -> SemanticResult<SemanticType> {
        // For now, create a simple conversion - in practice this would be more sophisticated
        let base_type = self.ast_resolver.resolve_ast_type(&ast_semantic_type.base_type.kind)?;
        Ok(base_type)
    }
}

impl InferenceEngine for StatementInferenceEngine {
    type Input = Stmt;
    type Output = TypeInferenceResult;
    
    fn infer(&mut self, input: Self::Input) -> SemanticResult<Self::Output> {
        self.infer_statement(&input)
    }
    
    fn engine_name(&self) -> &'static str {
        "StatementInferenceEngine"
    }
    
    fn reset(&mut self) {
        self.type_var_gen = TypeVarGenerator::new();
        self.environment = TypeEnvironment::new();
        self.expression_engine.reset();
        self.pattern_engine.reset();
        self.current_depth = 0;
    }
} 