//! Pattern Type Inference Engine
//!
//! This module implements type inference specifically for patterns.
//! It handles pattern matching, destructuring, and pattern type checking.
//!
//! **Single Responsibility**: Type inference for patterns
//! **What it does**: Infer types for pattern constructs, handle destructuring
//! **What it doesn't do**: Handle expressions, statements, or orchestration

use super::{InferenceEngine, ASTTypeResolver};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        TypeVar, InferredType, InferenceSource, TypeVarGenerator,
        environment::{TypeEnvironment, TypeBinding},
        constraints::{TypeConstraint, ConstraintType, ConstraintReason, ConstraintSet},
        unification::Substitution,
    },
};
use prism_ast::{Pattern, Type as AstType};
use prism_ast::expr::LiteralValue;
use prism_ast::pattern::ObjectPatternKey;
use prism_common::Span;
use std::collections::HashMap;

/// Engine specifically for pattern type inference
#[derive(Debug)]
pub struct PatternInferenceEngine {
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

/// Result of pattern inference
#[derive(Debug, Clone)]
pub struct PatternInferenceResult {
    /// The inferred type for the pattern
    pub inferred_type: InferredType,
    /// Constraints generated during inference
    pub constraints: ConstraintSet,
    /// New bindings created by the pattern
    pub bindings: Vec<TypeBinding>,
    /// Updated type environment
    pub environment: TypeEnvironment,
}

impl PatternInferenceEngine {
    /// Create a new pattern inference engine
    pub fn new() -> SemanticResult<Self> {
        Ok(Self {
            type_var_gen: TypeVarGenerator::new(),
            environment: TypeEnvironment::new(),
            ast_resolver: ASTTypeResolver::new(),
            current_depth: 0,
            max_depth: 1000,
        })
    }

    /// Infer type for a pattern with optional expected type
    pub fn infer_pattern(
        &mut self, 
        pattern: &prism_ast::AstNode<Pattern>, 
        expected_type: Option<&SemanticType>
    ) -> SemanticResult<PatternInferenceResult> {
        self.check_depth_limit()?;
        self.current_depth += 1;
        
        let result = match &pattern.kind {
            Pattern::Identifier(ident_pattern) => {
                self.infer_identifier_pattern(ident_pattern, expected_type, pattern.span)?
            }
            Pattern::Literal(literal_pattern) => {
                self.infer_literal_pattern(&literal_pattern.value, expected_type, pattern.span)?
            }
            Pattern::Wildcard => {
                self.infer_wildcard_pattern(expected_type, pattern.span)?
            }
            Pattern::Tuple(tuple_pattern) => {
                self.infer_tuple_pattern(tuple_pattern, expected_type, pattern.span)?
            }
            Pattern::Array(array_pattern) => {
                self.infer_array_pattern(array_pattern, expected_type, pattern.span)?
            }
            Pattern::Object(object_pattern) => {
                self.infer_object_pattern(object_pattern, expected_type, pattern.span)?
            }
            Pattern::Constructor(constructor_pattern) => {
                self.infer_constructor_pattern(constructor_pattern, expected_type, pattern.span)?
            }
            Pattern::Guard(guard_pattern) => {
                self.infer_guard_pattern(guard_pattern, expected_type, pattern.span)?
            }
            _ => {
                // Default case for unsupported patterns
                self.create_default_result(expected_type, pattern.span)?
            }
        };
        
        self.current_depth -= 1;
        Ok(result)
    }

    /// Infer type for an identifier pattern
    fn infer_identifier_pattern(
        &mut self,
        ident_pattern: &prism_ast::IdentifierPattern,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let pattern_type = expected_type
            .cloned()
            .unwrap_or_else(|| {
                // Create a fresh type variable if no expected type
                let type_var = self.fresh_type_var(span);
                SemanticType::Variable(type_var.id.to_string())
            });

        let inferred_type = InferredType {
            type_info: pattern_type.clone(),
            confidence: if expected_type.is_some() { 1.0 } else { 0.8 },
            inference_source: InferenceSource::Pattern,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        // Create a binding for the identifier
        let binding = TypeBinding::new(
            ident_pattern.name.to_string(),
            inferred_type.clone(),
            ident_pattern.is_mutable,
            self.environment.current_level(),
        );

        let mut environment = self.environment.clone();
        environment.add_binding(binding.clone());

        Ok(PatternInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            bindings: vec![binding],
            environment,
        })
    }

    /// Infer type for a literal pattern
    fn infer_literal_pattern(
        &mut self,
        literal: &LiteralValue,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let literal_type = match literal {
            LiteralValue::Integer(_) => self.create_semantic_type("Integer", span),
            LiteralValue::Float(_) => self.create_semantic_type("Float", span),
            LiteralValue::String(_) => self.create_semantic_type("String", span),
            LiteralValue::Boolean(_) => self.create_semantic_type("Boolean", span),
            LiteralValue::Null => self.create_semantic_type("Unit", span),
            LiteralValue::Money { .. } => self.create_semantic_type("Money", span),
            LiteralValue::Duration { .. } => self.create_semantic_type("Duration", span),
            LiteralValue::Regex(_) => self.create_semantic_type("Regex", span),
        };

        let mut constraints = ConstraintSet::new();

        // If there's an expected type, add a constraint
        if let Some(expected) = expected_type {
            constraints.add_constraint(TypeConstraint {
                lhs: ConstraintType::Concrete(literal_type.clone()),
                rhs: ConstraintType::Concrete(expected.clone()),
                origin: span,
                reason: ConstraintReason::PatternMatch,
                priority: 95,
            });
        }

        let inferred_type = InferredType {
            type_info: literal_type,
            confidence: 1.0,
            inference_source: InferenceSource::Literal,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints,
            bindings: Vec::new(), // Literal patterns don't create bindings
            environment: self.environment.clone(),
        })
    }

    /// Infer type for a wildcard pattern
    fn infer_wildcard_pattern(
        &mut self,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let pattern_type = expected_type
            .cloned()
            .unwrap_or_else(|| {
                // Create a fresh type variable if no expected type
                let type_var = self.fresh_type_var(span);
                SemanticType::Variable(type_var.id.to_string())
            });

        let inferred_type = InferredType {
            type_info: pattern_type,
            confidence: if expected_type.is_some() { 1.0 } else { 0.5 },
            inference_source: InferenceSource::Pattern,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            bindings: Vec::new(), // Wildcard patterns don't create bindings
            environment: self.environment.clone(),
        })
    }

    /// Infer type for a tuple pattern
    fn infer_tuple_pattern(
        &mut self,
        tuple_pattern: &prism_ast::TuplePattern,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let mut constraints = ConstraintSet::new();
        let mut all_bindings = Vec::new();
        let mut element_types = Vec::new();

        // Extract expected element types if available
        let expected_elements = if let Some(SemanticType::Record(fields)) = expected_type {
            // For now, treat tuples as records with numeric keys
            (0..tuple_pattern.elements.len())
                .map(|i| fields.get(&i.to_string()).cloned())
                .collect()
        } else {
            vec![None; tuple_pattern.elements.len()]
        };

        // Infer each element pattern
        for (i, element_pattern) in tuple_pattern.elements.iter().enumerate() {
            let expected_element = expected_elements.get(i).and_then(|opt| opt.as_ref());
            let element_result = self.infer_pattern(element_pattern, expected_element)?;
            
            constraints.extend(element_result.constraints);
            all_bindings.extend(element_result.bindings);
            element_types.push(element_result.inferred_type.type_info);
        }

        // Create tuple type as a record with numeric keys
        let mut tuple_fields = HashMap::new();
        for (i, element_type) in element_types.iter().enumerate() {
            tuple_fields.insert(i.to_string(), element_type.clone());
        }
        let tuple_type = SemanticType::Record(tuple_fields);

        let inferred_type = InferredType {
            type_info: tuple_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints,
            bindings: all_bindings,
            environment: self.environment.clone(),
        })
    }

    /// Create a default result for unsupported patterns
    fn create_default_result(
        &mut self,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let pattern_type = expected_type
            .cloned()
            .unwrap_or_else(|| self.create_semantic_type("Unknown", span));

        let inferred_type = InferredType {
            type_info: pattern_type,
            confidence: 0.1,
            inference_source: InferenceSource::Default,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            bindings: Vec::new(),
            environment: self.environment.clone(),
        })
    }

    /// Create a semantic type for basic types
    fn create_semantic_type(&self, name: &str, span: Span) -> SemanticType {
        let primitive_type = match name {
            "Integer" => prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(64)),
            "Float" => prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64),
            "String" => prism_ast::PrimitiveType::String,
            "Boolean" => prism_ast::PrimitiveType::Boolean,
            "Char" => prism_ast::PrimitiveType::Char,
            "Unit" => prism_ast::PrimitiveType::Unit,
            _ => prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(64)), // Default fallback
        };

        SemanticType::primitive(name, primitive_type, span)
    }

    /// Generate a fresh type variable
    fn fresh_type_var(&mut self, span: Span) -> TypeVar {
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

    // Placeholder methods for additional pattern types
    fn infer_array_pattern(
        &mut self,
        array_pattern: &prism_ast::ArrayPattern,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let mut all_constraints = ConstraintSet::new();
        let mut all_bindings = Vec::new();
        let mut element_types = Vec::new();

        // Determine expected element type
        let expected_element_type = match expected_type {
            Some(SemanticType::List(element_type)) => Some(element_type.as_ref()),
            Some(SemanticType::Generic { name, parameters }) if name == "Array" || name == "List" => {
                parameters.get(0)
            }
            _ => None,
        };

        // Infer each element pattern
        for element_pattern in &array_pattern.elements {
            let element_result = self.infer_pattern(element_pattern, expected_element_type)?;
            element_types.push(element_result.inferred_type.type_info.clone());
            all_constraints.merge(element_result.constraints);
            all_bindings.extend(element_result.bindings);
        }

        // Note: ArrayPattern doesn't have a rest field, so no rest pattern handling needed

        // Unify all element types
        let unified_element_type = if element_types.is_empty() {
            // Empty array pattern
            let type_var = self.fresh_type_var(span);
            SemanticType::Variable(type_var.id.to_string())
        } else {
            let first_type = &element_types[0];
            for element_type in &element_types[1..] {
                let constraint = TypeConstraint {
                    lhs: ConstraintType::Concrete(first_type.clone()),
                    rhs: ConstraintType::Concrete(element_type.clone()),
                    origin: span,
                    reason: ConstraintReason::PatternUnification,
                    priority: 85,
                };
                all_constraints.add_constraint(constraint);
            }
            first_type.clone()
        };

        let array_type = SemanticType::List(Box::new(unified_element_type));

        // Create constraint with expected type if provided
        if let Some(expected) = expected_type {
            let constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(array_type.clone()),
                rhs: ConstraintType::Concrete(expected.clone()),
                origin: span,
                reason: ConstraintReason::PatternTypeCheck,
                priority: 90,
            };
            all_constraints.add_constraint(constraint);
        }

        let inferred_type = InferredType {
            type_info: array_type,
            confidence: 0.85,
            inference_source: InferenceSource::PatternMatch,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints: all_constraints,
            bindings: all_bindings,
            environment: self.environment.clone(),
        })
    }

    fn infer_object_pattern(
        &mut self,
        object_pattern: &prism_ast::ObjectPattern,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let mut all_constraints = ConstraintSet::new();
        let mut all_bindings = Vec::new();
        let mut field_types = HashMap::new();

        // Extract expected field types
        let expected_fields = match expected_type {
            Some(SemanticType::Record(fields)) => Some(fields),
            Some(SemanticType::Complex { metadata, .. }) => {
                // Extract field information from metadata if available
                None // Simplified for now
            }
            _ => None,
        };

        // Infer each field pattern
        for field_pattern in &object_pattern.fields {
            let field_name = match &field_pattern.key {
                ObjectPatternKey::Identifier(symbol) => symbol.resolve().unwrap_or_else(|| "unknown".to_string()),
                ObjectPatternKey::String(s) => s.clone(),
                ObjectPatternKey::Computed(_) => "computed".to_string(), // Simplified for computed keys
            };
            
            let expected_field_type = expected_fields
                .and_then(|fields| fields.get(&field_name));

            let field_result = self.infer_pattern(&field_pattern.pattern, expected_field_type)?;
            field_types.insert(field_name.clone(), field_result.inferred_type.type_info.clone());
            all_constraints.merge(field_result.constraints);
            all_bindings.extend(field_result.bindings);
        }

        // Note: ObjectPattern doesn't have a rest field, so no rest pattern handling needed

        let record_type = SemanticType::Record(field_types);

        // Create constraint with expected type if provided
        if let Some(expected) = expected_type {
            let constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(record_type.clone()),
                rhs: ConstraintType::Concrete(expected.clone()),
                origin: span,
                reason: ConstraintReason::PatternTypeCheck,
                priority: 90,
            };
            all_constraints.add_constraint(constraint);
        }

        let inferred_type = InferredType {
            type_info: record_type,
            confidence: 0.85,
            inference_source: InferenceSource::PatternMatch,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints: all_constraints,
            bindings: all_bindings,
            environment: self.environment.clone(),
        })
    }

    fn infer_constructor_pattern(
        &mut self,
        constructor_pattern: &prism_ast::ConstructorPattern,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        let mut all_constraints = ConstraintSet::new();
        let mut all_bindings = Vec::new();

        let constructor_name = constructor_pattern.constructor.resolve()
            .unwrap_or_else(|| "Unknown".to_string());

        // For now, create a simple constructor type
        // In a full implementation, this would look up the constructor in the type environment
        let constructor_type = SemanticType::Complex {
            name: constructor_name.clone(),
            base_type: crate::types::BaseType::Composite(crate::types::CompositeType {
                kind: crate::types::CompositeKind::Struct,
                fields: Vec::new(),
                methods: Vec::new(),
                inheritance: Vec::new(),
            }),
            constraints: Vec::new(),
            business_rules: Vec::new(),
            metadata: crate::types::SemanticTypeMetadata::default(),
            ai_context: None,
            verification_properties: Vec::new(),
            location: span,
        };

        // Infer argument patterns
        for arg_pattern in &constructor_pattern.arguments {
            // For constructor arguments, we'd need to look up the constructor signature
            // For now, use a type variable
            let arg_type_var = self.fresh_type_var(span);
            let expected_arg_type = SemanticType::Variable(arg_type_var.id.to_string());
            
            let arg_result = self.infer_pattern(arg_pattern, Some(&expected_arg_type))?;
            all_constraints.merge(arg_result.constraints);
            all_bindings.extend(arg_result.bindings);
        }

        // Create constraint with expected type if provided
        if let Some(expected) = expected_type {
            let constraint = TypeConstraint {
                lhs: ConstraintType::Concrete(constructor_type.clone()),
                rhs: ConstraintType::Concrete(expected.clone()),
                origin: span,
                reason: ConstraintReason::PatternTypeCheck,
                priority: 90,
            };
            all_constraints.add_constraint(constraint);
        }

        let inferred_type = InferredType {
            type_info: constructor_type,
            confidence: 0.8,
            inference_source: InferenceSource::PatternMatch,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };

        Ok(PatternInferenceResult {
            inferred_type,
            constraints: all_constraints,
            bindings: all_bindings,
            environment: self.environment.clone(),
        })
    }

    fn infer_guard_pattern(
        &mut self,
        guard_pattern: &prism_ast::GuardPattern,
        expected_type: Option<&SemanticType>,
        span: Span,
    ) -> SemanticResult<PatternInferenceResult> {
        // First infer the base pattern
        let pattern_result = self.infer_pattern(&guard_pattern.pattern, expected_type)?;
        
        // The guard condition must be boolean
        // For now, we'll assume the guard is valid
        // In a full implementation, we'd need to infer the guard expression type
        
        // Guard patterns have the same type as their base pattern
        Ok(PatternInferenceResult {
            inferred_type: pattern_result.inferred_type,
            constraints: pattern_result.constraints,
            bindings: pattern_result.bindings,
            environment: pattern_result.environment,
        })
    }
}

impl InferenceEngine for PatternInferenceEngine {
    type Input = (prism_ast::AstNode<Pattern>, Option<SemanticType>);
    type Output = PatternInferenceResult;
    
    fn infer(&mut self, input: Self::Input) -> SemanticResult<Self::Output> {
        let (pattern, expected_type) = input;
        self.infer_pattern(&pattern, expected_type.as_ref())
    }
    
    fn engine_name(&self) -> &'static str {
        "PatternInferenceEngine"
    }
    
    fn reset(&mut self) {
        self.type_var_gen = TypeVarGenerator::new();
        self.environment = TypeEnvironment::new();
        self.current_depth = 0;
    }
} 