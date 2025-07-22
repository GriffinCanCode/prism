//! Type inference engine for semantic types
//!
//! This module provides comprehensive type inference capabilities for the Prism language,
//! including semantic type inference, constraint propagation, and AI-assisted type suggestions.

use crate::{
    AstNode, Expr, Type, TypeConstraint, Effect, EffectType,
    EffectMetadata, EffectCategory, AiContext, LiteralValue, BinaryOperator, UnaryOperator,
    SafetyLevel, RangeConstraint, BusinessRuleConstraint, PrimitiveType,
    ComputationEffect, ComputationComplexity, ComplexityBound,
    CustomEffect, NodeMetadata, 
    CompositeType, CompositeField, CompositeTypeKind, FieldVisibility, TypeParameter,
};
use crate::types::FunctionType as TypesFunctionType;
use prism_common::{span::Span, symbol::Symbol, NodeId};
use std::collections::{HashMap, HashSet};
use std::time::SystemTime;
use thiserror::Error;

/// Semantic type information for AI-assisted type inference
#[derive(Debug, Clone)]
pub struct SemanticTypeInfo {
    /// Business domain this type belongs to
    pub business_domain: Option<String>,
    /// Type constraints derived from semantic analysis
    pub constraints: Vec<TypeConstraint>,
    /// Usage patterns observed for this type
    pub usage_patterns: Vec<String>,
    /// Confidence level in this type inference (0.0 to 1.0)
    pub confidence: f64,
}

/// Enhanced type with semantic information and constraints
#[derive(Debug, Clone)]
pub struct EnhancedType {
    /// Base type being enhanced
    pub base_type: Box<InferredType>,
    /// Semantic type information
    pub semantic_info: SemanticTypeInfo,
    /// Effects associated with this type
    pub effects: Vec<Effect>,
    /// Additional constraints
    pub constraints: Vec<TypeConstraint>,
}

/// Inferred type from type inference engine
#[derive(Debug, Clone)]
pub enum InferredType {
    /// Primitive type
    Primitive(PrimitiveType),
    /// Composite type
    Composite(CompositeType),
    /// Function type
    Function(Box<FunctionType>),
    /// Enhanced type with semantic information
    Enhanced(EnhancedType),
    /// Unknown type
    Unknown,
}

/// Type inference engine
#[derive(Debug)]
pub struct TypeInferenceEngine {
    /// Type environment
    pub type_env: TypeEnvironment,
    /// Constraint solver
    pub constraint_solver: ConstraintSolver,
    /// Effect tracker
    pub effect_tracker: EffectTracker,
    /// AI assistant for type suggestions
    pub ai_assistant: AITypeAssistant,
    /// Inference configuration
    pub config: InferenceConfig,
    /// Node ID counter for generating unique IDs
    next_node_id: u32,
}

/// Type environment for inference
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    /// Variable types
    pub variables: HashMap<Symbol, InferredType>,
    /// Function types
    pub functions: HashMap<Symbol, FunctionType>,
    /// Type aliases
    pub type_aliases: HashMap<Symbol, AstNode<Type>>,
    /// Scoped environments
    pub scopes: Vec<HashMap<Symbol, InferredType>>,
}

/// Function type for inference
#[derive(Debug, Clone)]
pub struct FunctionType {
    /// Parameter types
    pub parameters: Vec<InferredType>,
    /// Return type
    pub return_type: Box<InferredType>,
    /// Effect type
    pub effects: Vec<Effect>,
}

/// Source of type inference
#[derive(Debug, Clone)]
pub enum InferenceSource {
    /// Explicitly declared
    Explicit,
    /// Inferred from usage
    Usage,
    /// Inferred from context
    Context,
    /// Inferred from constraints
    Constraints,
    /// AI-suggested
    AISuggested,
    /// Unified from multiple sources
    Unified(Vec<InferenceSource>),
}

/// AI type suggestion
#[derive(Debug, Clone)]
pub struct AISuggestion {
    /// Suggested type
    pub suggested_type: AstNode<Type>,
    /// Confidence level
    pub confidence: f64,
    /// Reasoning
    pub reasoning: String,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Source of the suggestion
    pub source: String,
}

/// Constraint solver for type inference
#[derive(Debug)]
pub struct ConstraintSolver {
    /// Active constraints
    pub constraints: Vec<TypeConstraintSet>,
    /// Solver configuration
    pub config: SolverConfig,
}

/// Set of related type constraints
#[derive(Debug, Clone)]
pub struct TypeConstraintSet {
    /// Constraint identifier
    pub id: u64,
    /// Constraints in this set
    pub constraints: Vec<TypeConstraint>,
    /// Variables involved
    pub variables: HashSet<Symbol>,
    /// Constraint relationships
    pub relationships: Vec<ConstraintRelationship>,
}

/// Relationship between constraints
#[derive(Debug, Clone)]
pub struct ConstraintRelationship {
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Related constraint IDs
    pub related_constraints: Vec<u64>,
}

/// Type of constraint relationship
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationshipType {
    /// Constraints are mutually exclusive
    MutuallyExclusive,
    /// Constraints imply each other
    Implication,
    /// Constraints are equivalent
    Equivalence,
    /// Constraints are in conflict
    Conflict,
}

/// Effect tracker for type inference
#[derive(Debug)]
pub struct EffectTracker {
    /// Current effect context
    pub current_effects: Vec<Effect>,
    /// Effect history
    pub effect_history: Vec<EffectContext>,
    /// Effect composition rules
    pub composition_rules: Vec<EffectCompositionRule>,
}

/// Effect context
#[derive(Debug, Clone)]
pub struct EffectContext {
    /// Context identifier
    pub id: u64,
    /// Effects in this context
    pub effects: Vec<Effect>,
    /// Context span
    pub span: Span,
}

/// Effect composition rule
#[derive(Debug, Clone)]
pub struct EffectCompositionRule {
    /// Rule name
    pub name: String,
    /// Input effects
    pub input_effects: Vec<Effect>,
    /// Output effect
    pub output_effect: Effect,
    /// Composition conditions
    pub conditions: Vec<AstNode<Expr>>,
}

/// AI assistant for type suggestions
#[derive(Debug)]
pub struct AITypeAssistant {
    /// Knowledge base of type patterns
    pub type_patterns: Vec<TypePattern>,
    /// Historical inference data
    pub inference_history: Vec<InferenceRecord>,
    /// AI configuration
    pub config: AIConfig,
}

/// Type pattern for AI assistance
#[derive(Debug, Clone)]
pub struct TypePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Pattern conditions
    pub conditions: Vec<PatternCondition>,
    /// Suggested type
    pub suggested_type: AstNode<Type>,
    /// Pattern confidence
    pub confidence: f64,
}

/// Condition for type pattern matching
#[derive(Debug, Clone)]
pub struct PatternCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition value
    pub value: String,
}

/// Type of pattern condition
#[derive(Debug, Clone)]
pub enum ConditionType {
    /// Variable name matches pattern
    VariableName,
    /// Function name matches pattern
    FunctionName,
    /// Usage context matches pattern
    UsageContext,
    /// Constraint matches pattern
    Constraint,
    /// Custom condition
    Custom(String),
}

/// Historical inference record
#[derive(Debug, Clone)]
pub struct InferenceRecord {
    /// Record identifier
    pub id: u64,
    /// Input context
    pub input_context: InferenceContext,
    /// Inferred type
    pub inferred_type: InferredType,
    /// Inference success
    pub success: bool,
    /// Feedback score
    pub feedback_score: Option<f64>,
}

/// Context for type inference
#[derive(Debug, Clone)]
pub struct InferenceContext {
    /// Expression being inferred
    pub expression: AstNode<Expr>,
    /// Surrounding context
    pub context: Vec<AstNode<Expr>>,
    /// Available type information
    pub type_info: HashMap<Symbol, AstNode<Type>>,
}

/// Configuration for type inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Enable AI assistance
    pub enable_ai_assistance: bool,
    /// Maximum inference depth
    pub max_inference_depth: usize,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f64,
    /// Enable constraint propagation
    pub enable_constraint_propagation: bool,
    /// Enable effect inference
    pub enable_effect_inference: bool,
}

/// Configuration for constraint solver
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum solver iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Enable constraint simplification
    pub enable_simplification: bool,
}

/// Configuration for AI assistant
#[derive(Debug, Clone)]
pub struct AIConfig {
    /// Enable pattern matching
    pub enable_pattern_matching: bool,
    /// Enable learning from feedback
    pub enable_learning: bool,
    /// Maximum suggestions per inference
    pub max_suggestions: usize,
}

/// Historical inference entry
#[derive(Debug, Clone)]
pub struct InferenceHistoryEntry {
    /// Expression context
    pub expression_context: String,
    /// Base type
    pub base_type: InferredType,
    /// Suggestions made
    pub suggestions: Vec<AISuggestion>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Type inference error
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Cannot infer type for expression at {span:?}")]
    CannotInferType { span: Span },
    
    #[error("Type constraint conflict: {message}")]
    ConstraintConflict { message: String },
    
    #[error("Effect inference failed: {message}")]
    EffectInferenceFailed { message: String },
    
    #[error("Circular type dependency detected")]
    CircularDependency,
    
    #[error("Insufficient type information")]
    InsufficientInformation,
    
    #[error("AI assistance unavailable: {reason}")]
    AIUnavailable { reason: String },
    
    #[error("Argument count mismatch: expected {expected}, found {found}")]
    ArgumentCountMismatch { expected: usize, found: usize },
    
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },
}

/// Result type for inference operations
pub type InferenceResult<T> = Result<T, InferenceError>;

impl TypeInferenceEngine {
    /// Helper function to convert InferredType to AstNode<Type>
    fn inferred_type_to_ast_node(&mut self, inferred_type: &InferredType) -> AstNode<Type> {
        let type_data = match inferred_type {
            InferredType::Primitive(prim_type) => Type::Primitive(prim_type.clone()),
            InferredType::Composite(comp_type) => Type::Composite(comp_type.clone()),
            InferredType::Function(func_type) => {
                // Convert function type parameters properly
                let param_types = func_type.parameters.iter()
                    .map(|param| self.inferred_type_to_ast_node(param))
                    .collect();
                    
                let return_type = Box::new(self.inferred_type_to_ast_node(&func_type.return_type));
                
                Type::Function(TypesFunctionType {
                    parameters: param_types,
                    return_type,
                    effects: func_type.effects.clone(),
                })
            },
            InferredType::Enhanced(enh_type) => {
                // Use the base type for now
                return self.inferred_type_to_ast_node(&enh_type.base_type);
            },
            InferredType::Unknown => Type::Primitive(PrimitiveType::Unit),
        };
        
        AstNode {
            kind: type_data,
            span: Span::dummy(),
            id: self.generate_node_id(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Generate a new unique node ID
    fn generate_node_id(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Create a proper expression node for constraint values
    fn create_constraint_expr(&mut self, value: i64) -> AstNode<Expr> {
        AstNode {
            kind: Expr::Literal(crate::LiteralExpr {
                value: LiteralValue::Integer(value),
            }),
            span: Span::dummy(),
            id: self.generate_node_id(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Create a new type inference engine
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            type_env: TypeEnvironment::new(),
            constraint_solver: ConstraintSolver::new(SolverConfig::default()),
            effect_tracker: EffectTracker::new(),
            ai_assistant: AITypeAssistant::new(AIConfig::default()),
            config,
            next_node_id: 1,
        }
    }
    
    /// Infer the type of an expression
    pub fn infer_type(&mut self, expr: &AstNode<Expr>) -> InferenceResult<InferredType> {
        // Start with basic type inference
        let base_type = self.infer_base_type(expr)?;
        
        // Apply constraint propagation
        let constrained_type = if self.config.enable_constraint_propagation {
            self.apply_constraints(base_type)?
        } else {
            base_type
        };
        
        // Apply effect inference
        let effect_type = if self.config.enable_effect_inference {
            self.infer_effects(constrained_type)?
        } else {
            constrained_type
        };
        
        // Apply AI assistance
        let ai_enhanced_type = if self.config.enable_ai_assistance {
            self.apply_ai_assistance(effect_type, expr)?
        } else {
            effect_type
        };
        
        Ok(ai_enhanced_type)
    }
    
    /// Infer base type without constraints or effects
    fn infer_base_type(&mut self, expr: &AstNode<Expr>) -> InferenceResult<InferredType> {
        match &expr.kind {
            Expr::Literal(literal) => self.infer_literal_type(&literal.value),
            Expr::Variable(var) => self.infer_identifier_type(&var.name),
            Expr::Binary(binary) => self.infer_binary_op_type(&binary.left, &binary.operator, &binary.right),
            Expr::Unary(unary) => self.infer_unary_op_type(&unary.operator, &unary.operand),
            Expr::Call(call) => self.infer_function_call_type(&call.callee, &call.arguments),
            _ => Err(InferenceError::CannotInferType { span: expr.span }),
        }
    }
    
    /// Apply constraints to refine type
    fn apply_constraints(&mut self, base_type: InferredType) -> InferenceResult<InferredType> {
        // Collect applicable constraints
        let constraints = self.collect_applicable_constraints(&base_type);
        
        // Solve constraints
        let solved_constraints = self.constraint_solver.solve(&constraints)?;
        
        // Apply solved constraints to refine type
        self.apply_solved_constraints(base_type, &solved_constraints)
    }
    
    /// Infer effects for the type
    fn infer_effects(&mut self, base_type: InferredType) -> InferenceResult<InferredType> {
        let effects = self.effect_tracker.infer_effects(&base_type)?;
        
        // Wrap type in effect type if effects were found
        if !effects.is_empty() {
            let base_type_node = self.inferred_type_to_ast_node(&base_type);
            let effect_type = EffectType {
                base_type: Box::new(base_type_node),
                effects,
                composition_rules: Vec::new(),
                capability_requirements: Vec::new(),
                metadata: Default::default(),
            };
            
            Ok(InferredType::Enhanced(EnhancedType {
                base_type: Box::new(base_type),
                semantic_info: SemanticTypeInfo {
                    business_domain: None,
                    constraints: vec![],
                    usage_patterns: vec!["effect_type".to_string()],
                    confidence: 0.95,
                },
                effects: vec![],
                constraints: vec![],
            }))
        } else {
            Ok(base_type)
        }
    }
    
    /// Apply AI assistance to enhance type inference
    fn apply_ai_assistance(
        &mut self,
        base_type: InferredType,
        expr: &AstNode<Expr>,
    ) -> InferenceResult<InferredType> {
        let suggestions = self.ai_assistant.suggest_types(expr, &base_type)?;
        
        // Merge AI suggestions with base type
        let mut enhanced_type = base_type;
        
        // Apply highest confidence suggestion if it exceeds threshold
        if let Some(best_suggestion) = suggestions.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
        {
            if best_suggestion.confidence > self.config.min_confidence_threshold {
                // Convert the suggested AstNode<Type> back to InferredType
                enhanced_type = self.ast_type_to_inferred_type(&best_suggestion.suggested_type)?;
            }
        }
        
        Ok(enhanced_type)
    }
    
    /// Convert AstNode<Type> back to InferredType
    fn ast_type_to_inferred_type(&self, ast_type: &AstNode<Type>) -> InferenceResult<InferredType> {
        match &ast_type.kind {
            Type::Primitive(prim_type) => Ok(InferredType::Primitive(prim_type.clone())),
            Type::Function(func_type) => {
                let param_types: Result<Vec<_>, _> = func_type.parameters.iter()
                    .map(|param| self.ast_type_to_inferred_type(param))
                    .collect();
                let param_types = param_types?;
                let return_type = Box::new(self.ast_type_to_inferred_type(&func_type.return_type)?);
                
                Ok(InferredType::Function(Box::new(FunctionType {
                    parameters: param_types,
                    return_type,
                    effects: func_type.effects.clone(),
                })))
            }
            _ => Ok(InferredType::Unknown),
        }
    }
    
    // Additional helper methods would be implemented here...
    
    fn infer_literal_type(&self, literal: &crate::LiteralValue) -> InferenceResult<InferredType> {
        use crate::LiteralValue;
        
        let (base_type, semantic_info) = match literal {
            LiteralValue::Integer(value) => {
                let base_type = if *value >= i32::MIN as i64 && *value <= i32::MAX as i64 {
                    InferredType::Primitive(PrimitiveType::Int32)
                } else {
                    InferredType::Primitive(PrimitiveType::Int64)
                };
                let semantic_info = SemanticTypeInfo {
                    business_domain: None,
                    constraints: vec![],
                    usage_patterns: vec!["literal_value".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::Float(value) => {
                let base_type = InferredType::Primitive(PrimitiveType::Float64);
                let semantic_info = SemanticTypeInfo {
                    business_domain: None,
                    constraints: vec![],
                    usage_patterns: vec!["literal_value".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::String(_) => {
                let base_type = InferredType::Primitive(PrimitiveType::String);
                let semantic_info = SemanticTypeInfo {
                    business_domain: None,
                    constraints: vec![],
                    usage_patterns: vec!["literal_value".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::Boolean(_) => {
                let base_type = InferredType::Primitive(PrimitiveType::Boolean);
                let semantic_info = SemanticTypeInfo {
                    business_domain: None,
                    constraints: vec![],
                    usage_patterns: vec!["literal_value".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::Null => {
                let base_type = InferredType::Unknown;
                let semantic_info = SemanticTypeInfo {
                    business_domain: None,
                    constraints: vec![],
                    usage_patterns: vec!["null_literal".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::Money { amount: _, currency } => {
                let base_type = InferredType::Enhanced(EnhancedType {
                    base_type: Box::new(InferredType::Primitive(PrimitiveType::Float64)),
                    semantic_info: SemanticTypeInfo {
                        business_domain: Some("Financial".to_string()),
                        constraints: vec![],
                        usage_patterns: vec!["money_literal".to_string()],
                        confidence: 1.0,
                    },
                    effects: vec![],
                    constraints: vec![],
                });
                let semantic_info = SemanticTypeInfo {
                    business_domain: Some("Financial".to_string()),
                    constraints: vec![],
                    usage_patterns: vec!["money_literal".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::Duration { value: _, unit: _ } => {
                let base_type = InferredType::Enhanced(EnhancedType {
                    base_type: Box::new(InferredType::Primitive(PrimitiveType::Float64)),
                    semantic_info: SemanticTypeInfo {
                        business_domain: Some("Temporal".to_string()),
                        constraints: vec![],
                        usage_patterns: vec!["duration_literal".to_string()],
                        confidence: 1.0,
                    },
                    effects: vec![],
                    constraints: vec![],
                });
                let semantic_info = SemanticTypeInfo {
                    business_domain: Some("Temporal".to_string()),
                    constraints: vec![],
                    usage_patterns: vec!["duration_literal".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
            LiteralValue::Regex(_) => {
                let base_type = InferredType::Enhanced(EnhancedType {
                    base_type: Box::new(InferredType::Primitive(PrimitiveType::String)),
                    semantic_info: SemanticTypeInfo {
                        business_domain: Some("Pattern".to_string()),
                        constraints: vec![],
                        usage_patterns: vec!["regex_literal".to_string()],
                        confidence: 1.0,
                    },
                    effects: vec![],
                    constraints: vec![],
                });
                let semantic_info = SemanticTypeInfo {
                    business_domain: Some("Pattern".to_string()),
                    constraints: vec![],
                    usage_patterns: vec!["regex_literal".to_string()],
                    confidence: 1.0,
                };
                (base_type, semantic_info)
            }
        };

        Ok(InferredType::Enhanced(EnhancedType {
            base_type: Box::new(base_type),
            semantic_info,
            effects: vec![],
            constraints: vec![],
        }))
    }

    fn infer_identifier_type(&self, name: &Symbol) -> InferenceResult<InferredType> {
        // Look up the identifier in the environment
        if let Some(var_type) = self.type_env.variables.get(name) {
            return Ok(var_type.clone());
        }

        // Check if it's a function
        if let Some(func_type) = self.type_env.functions.get(name) {
            return Ok(InferredType::Function(Box::new(func_type.clone())));
        }

        // Check if it's a type alias
        if let Some(_alias_type) = self.type_env.type_aliases.get(name) {
            // For now, return unknown - in a full implementation we'd convert the alias
            return Ok(InferredType::Unknown);
        }

        // If not found, create an unknown type with the identifier name
        Ok(InferredType::Enhanced(EnhancedType {
            base_type: Box::new(InferredType::Unknown),
            semantic_info: SemanticTypeInfo {
                business_domain: None,
                constraints: vec![],
                usage_patterns: vec!["identifier_reference".to_string()],
                confidence: 0.5, // Lower confidence for undefined identifiers
            },
            effects: vec![],
            constraints: vec![],
        }))
    }

    fn infer_binary_op_type(
        &mut self,
        left: &AstNode<Expr>,
        op: &crate::BinaryOperator,
        right: &AstNode<Expr>,
    ) -> InferenceResult<InferredType> {
        let left_type = self.infer_base_type(left)?;
        let right_type = self.infer_base_type(right)?;

        use crate::BinaryOperator;
        
        let result_type = match op {
            // Arithmetic operations
            BinaryOperator::Add | BinaryOperator::Subtract | 
            BinaryOperator::Multiply | BinaryOperator::Divide => {
                self.infer_arithmetic_result(&left_type, &right_type)?
            }
            
            // Comparison operations
            BinaryOperator::Equal | BinaryOperator::NotEqual |
            BinaryOperator::Less | BinaryOperator::LessEqual |
            BinaryOperator::Greater | BinaryOperator::GreaterEqual |
            BinaryOperator::LessThan | BinaryOperator::LessThanOrEqual |
            BinaryOperator::GreaterThan | BinaryOperator::GreaterThanOrEqual => {
                InferredType::Primitive(PrimitiveType::Boolean)
            }
            
            // Logical operations
            BinaryOperator::And | BinaryOperator::Or |
            BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr => {
                InferredType::Primitive(PrimitiveType::Boolean)
            }
            
            // Bitwise operations
            BinaryOperator::BitAnd | BinaryOperator::BitOr | 
            BinaryOperator::BitXor | BinaryOperator::LeftShift | 
            BinaryOperator::RightShift |
            BinaryOperator::BitwiseAnd | BinaryOperator::BitwiseOr |
            BinaryOperator::BitwiseXor => {
                self.infer_bitwise_result(&left_type, &right_type)?
            }
            
            // Assignment operations
            BinaryOperator::Assign => left_type,
            
            // Modulo operation
            BinaryOperator::Modulo => {
                self.infer_arithmetic_result(&left_type, &right_type)?
            }
            
            // Other operations
            _ => InferredType::Unknown,
        };

        Ok(InferredType::Enhanced(EnhancedType {
            base_type: Box::new(result_type),
            semantic_info: SemanticTypeInfo {
                business_domain: None,
                constraints: vec![],
                usage_patterns: vec![format!("binary_operation_{:?}", op)],
                confidence: 0.8,
            },
            effects: vec![],
            constraints: vec![],
        }))
    }

    fn infer_unary_op_type(
        &mut self,
        op: &crate::UnaryOperator,
        operand: &AstNode<Expr>,
    ) -> InferenceResult<InferredType> {
        let operand_type = self.infer_base_type(operand)?;

        use crate::UnaryOperator;
        
        let result_type = match op {
            UnaryOperator::Negate => operand_type,
            UnaryOperator::Not | UnaryOperator::LogicalNot => InferredType::Primitive(PrimitiveType::Boolean),
            UnaryOperator::BitNot | UnaryOperator::BitwiseNot => operand_type,
            UnaryOperator::PreIncrement | UnaryOperator::PostIncrement |
            UnaryOperator::PreDecrement | UnaryOperator::PostDecrement => operand_type,
            _ => InferredType::Unknown,
        };

        Ok(InferredType::Enhanced(EnhancedType {
            base_type: Box::new(result_type),
            semantic_info: SemanticTypeInfo {
                business_domain: None,
                constraints: vec![],
                usage_patterns: vec![format!("unary_operation_{:?}", op)],
                confidence: 0.8,
            },
            effects: vec![],
            constraints: vec![],
        }))
    }

    fn infer_function_call_type(
        &mut self,
        function: &AstNode<Expr>,
        args: &[AstNode<Expr>],
    ) -> InferenceResult<InferredType> {
        let function_type = self.infer_base_type(function)?;
        
        // Infer argument types
        let arg_types: Result<Vec<_>, _> = args.iter()
            .map(|arg| self.infer_base_type(arg))
            .collect();
        let _arg_types = arg_types?;

        // Extract return type from function type
        let return_type = match &function_type {
            InferredType::Function(func_type) => {
                func_type.return_type.as_ref().clone()
            }
            InferredType::Enhanced(enhanced) => {
                if let InferredType::Function(func_type) = enhanced.base_type.as_ref() {
                    func_type.return_type.as_ref().clone()
                } else {
                    InferredType::Unknown
                }
            }
            _ => InferredType::Unknown,
        };

        Ok(InferredType::Enhanced(EnhancedType {
            base_type: Box::new(return_type),
            semantic_info: SemanticTypeInfo {
                business_domain: None,
                constraints: vec![],
                usage_patterns: vec!["function_call".to_string()],
                confidence: 0.7,
            },
            effects: vec![],
            constraints: vec![],
        }))
    }

    fn collect_applicable_constraints(&mut self, base_type: &InferredType) -> Vec<TypeConstraint> {
        let mut constraints = Vec::new();

        // Add basic type constraints
        match base_type {
            InferredType::Primitive(PrimitiveType::Int32) => {
                // Create proper range constraints with actual expression nodes
                let min_expr = self.create_constraint_expr(i32::MIN as i64);
                let max_expr = self.create_constraint_expr(i32::MAX as i64);
                
                constraints.push(TypeConstraint::Range(RangeConstraint {
                    min: Some(min_expr),
                    max: Some(max_expr),
                    inclusive: true,
                }));
            }
            InferredType::Primitive(PrimitiveType::Int64) => {
                let min_expr = self.create_constraint_expr(i64::MIN);
                let max_expr = self.create_constraint_expr(i64::MAX);
                
                constraints.push(TypeConstraint::Range(RangeConstraint {
                    min: Some(min_expr),
                    max: Some(max_expr),
                    inclusive: true,
                }));
            }
            InferredType::Primitive(PrimitiveType::String) => {
                // Create proper BusinessRuleConstraint with expression
                let rule_expr = AstNode {
                    kind: Expr::Literal(crate::LiteralExpr {
                        value: LiteralValue::Boolean(true),
                    }),
                    span: Span::dummy(),
                    id: self.generate_node_id(),
                    metadata: NodeMetadata::default(),
                };
                
                constraints.push(TypeConstraint::BusinessRule(BusinessRuleConstraint {
                    description: "String must be valid UTF-8".to_string(),
                    expression: rule_expr,
                    priority: 1,
                }));
            }
            InferredType::Enhanced(enhanced) => {
                constraints.extend(enhanced.constraints.clone());
            }
            _ => {}
        }

        constraints
    }

    fn apply_solved_constraints(
        &self,
        mut base_type: InferredType,
        constraints: &[TypeConstraint],
    ) -> InferenceResult<InferredType> {
        // Apply each constraint to refine the type
        for constraint in constraints {
            base_type = self.apply_single_constraint(base_type, constraint)?;
        }

        Ok(base_type)
    }

    // Helper methods for binary operations
    fn infer_arithmetic_result(&self, left: &InferredType, right: &InferredType) -> InferenceResult<InferredType> {
        match (left, right) {
            (InferredType::Primitive(PrimitiveType::Int32), InferredType::Primitive(PrimitiveType::Int32)) => {
                Ok(InferredType::Primitive(PrimitiveType::Int32))
            }
            (InferredType::Primitive(PrimitiveType::Int64), _) | 
            (_, InferredType::Primitive(PrimitiveType::Int64)) => {
                Ok(InferredType::Primitive(PrimitiveType::Int64))
            }
            (InferredType::Primitive(PrimitiveType::Float64), _) | 
            (_, InferredType::Primitive(PrimitiveType::Float64)) => {
                Ok(InferredType::Primitive(PrimitiveType::Float64))
            }
            _ => Ok(InferredType::Unknown)
        }
    }

    fn infer_bitwise_result(&self, left: &InferredType, right: &InferredType) -> InferenceResult<InferredType> {
        match (left, right) {
            (InferredType::Primitive(PrimitiveType::Int32), InferredType::Primitive(PrimitiveType::Int32)) => {
                Ok(InferredType::Primitive(PrimitiveType::Int32))
            }
            (InferredType::Primitive(PrimitiveType::Int64), InferredType::Primitive(PrimitiveType::Int64)) => {
                Ok(InferredType::Primitive(PrimitiveType::Int64))
            }
            _ => Err(InferenceError::TypeMismatch {
                expected: "integer type".to_string(),
                found: format!("{:?}", left),
            })
        }
    }

    fn apply_single_constraint(&self, base_type: InferredType, _constraint: &TypeConstraint) -> InferenceResult<InferredType> {
        // Apply the constraint to refine the type
        // For now, just return the base type
        // In a full implementation, we'd create a more specific range type
        Ok(base_type)
    }
}

// Implementation for other structs...

impl TypeEnvironment {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            type_aliases: HashMap::new(),
            scopes: Vec::new(),
        }
    }
}

impl ConstraintSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            constraints: Vec::new(),
            config,
        }
    }
    
    pub fn solve(&mut self, constraints: &[TypeConstraint]) -> InferenceResult<Vec<TypeConstraint>> {
        let mut solved_constraints = Vec::new();
        let mut remaining_constraints = constraints.to_vec();
        
        // Iteratively solve constraints
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;
        
        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;
            
            let mut new_remaining = Vec::new();
            
            for constraint in remaining_constraints {
                match self.try_solve_constraint(&constraint) {
                    Ok(Some(solved)) => {
                        solved_constraints.push(solved);
                        changed = true;
                    }
                    Ok(None) => {
                        // Constraint couldn't be solved yet, keep it for next iteration
                        new_remaining.push(constraint);
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
            
            remaining_constraints = new_remaining;
        }
        
        // Add any remaining unsolved constraints
        solved_constraints.extend(remaining_constraints);
        
        Ok(solved_constraints)
    }
    
    fn try_solve_constraint(&self, constraint: &TypeConstraint) -> InferenceResult<Option<TypeConstraint>> {
        match constraint {
            TypeConstraint::Range(_range_constraint) => {
                // Range constraints are already solved
                Ok(Some(constraint.clone()))
            }
            TypeConstraint::BusinessRule(_business_rule) => {
                // Business rule constraints are already solved
                Ok(Some(constraint.clone()))
            }
            _ => Ok(Some(constraint.clone())),
        }
    }
}

impl EffectTracker {
    pub fn new() -> Self {
        Self {
            current_effects: Vec::new(),
            effect_history: Vec::new(),
            composition_rules: Vec::new(),
        }
    }
    
    pub fn infer_effects(&mut self, base_type: &InferredType) -> InferenceResult<Vec<Effect>> {
        let mut effects = Vec::new();
        
        match base_type {
            InferredType::Function(_func_type) => {
                // Function calls may have computation effects
                effects.push(Effect::Computation(ComputationEffect {
                    complexity: ComputationComplexity {
                        time_complexity: ComplexityBound::Linear, // Default assumption
                        space_complexity: ComplexityBound::Constant,
                    },
                    resource_requirements: vec![],
                }));
            }
            InferredType::Enhanced(enhanced) => {
                // Enhanced types may already have effects
                effects.extend(enhanced.effects.clone());
                
                // Check for domain-specific effects
                if let Some(domain) = &enhanced.semantic_info.business_domain {
                    match domain.as_str() {
                        "Financial" => {
                            effects.push(Effect::Custom(CustomEffect {
                                name: "FinancialOperation".to_string(),
                                parameters: std::collections::HashMap::new(),
                                constraints: vec![],
                            }));
                        }
                        "Temporal" => {
                            effects.push(Effect::Computation(ComputationEffect {
                                complexity: ComputationComplexity {
                                    time_complexity: ComplexityBound::Linear,
                                    space_complexity: ComplexityBound::Constant,
                                },
                                resource_requirements: vec![],
                            }));
                        }
                        _ => {}
                    }
                }
            }
            InferredType::Composite(_) => {
                // Composite types may involve construction effects
                effects.push(Effect::Computation(ComputationEffect {
                    complexity: ComputationComplexity {
                        time_complexity: ComplexityBound::Linear,
                        space_complexity: ComplexityBound::Linear,
                    },
                    resource_requirements: vec![],
                }));
            }
            _ => {
                // Other types generally have no effects
            }
        }
        
        // Record effects in history
        let history_id = self.effect_history.len() as u64;
        self.effect_history.extend(effects.iter().cloned().map(|effect| EffectContext {
            id: history_id,
            effects: vec![effect],
            span: Span::dummy(),
        }));
        self.current_effects = effects.clone();
        
        Ok(effects)
    }
}

impl AITypeAssistant {
    pub fn new(config: AIConfig) -> Self {
        Self {
            type_patterns: Vec::new(),
            inference_history: Vec::new(),
            config,
        }
    }
    
    pub fn suggest_types(
        &mut self,
        expr: &AstNode<Expr>,
        base_type: &InferredType,
    ) -> InferenceResult<Vec<AISuggestion>> {
        let mut suggestions = Vec::new();
        
        // Analyze the expression context
        let context = self.analyze_expression_context(expr);
        
        // Generate suggestions based on patterns
        match base_type {
            InferredType::Unknown => {
                if context.contains("name") || context.contains("title") || context.contains("description") {
                    let inferred_type = InferredType::Primitive(PrimitiveType::String);
                    suggestions.push(AISuggestion {
                        suggested_type: self.create_ast_type_node(&inferred_type),
                        confidence: 0.8,
                        reasoning: "Context suggests this is a textual value".to_string(),
                        source: "pattern_matching".to_string(),
                        evidence: vec!["Context contains textual keywords".to_string()],
                    });
                }
                
                if context.contains("price") || context.contains("amount") || context.contains("cost") {
                    let inferred_type = InferredType::Primitive(PrimitiveType::Float64);
                    suggestions.push(AISuggestion {
                        suggested_type: self.create_ast_type_node(&inferred_type),
                        confidence: 0.9,
                        reasoning: "Context suggests this is a monetary value".to_string(),
                        source: "business_domain_analysis".to_string(),
                        evidence: vec!["Context contains monetary keywords".to_string()],
                    });
                }
                
                if context.contains("time") || context.contains("duration") || context.contains("delay") {
                    let inferred_type = InferredType::Primitive(PrimitiveType::Float64);
                    suggestions.push(AISuggestion {
                        suggested_type: self.create_ast_type_node(&inferred_type),
                        confidence: 0.8,
                        reasoning: "Context suggests this is a temporal value".to_string(),
                        source: "business_domain_analysis".to_string(),
                        evidence: vec!["Context contains temporal keywords".to_string()],
                    });
                }
            }
            InferredType::Primitive(prim_type) => {
                // Suggest refinements for primitive types
                match prim_type {
                    PrimitiveType::String => {
                        if context.contains("email") {
                            let inferred_type = InferredType::Primitive(PrimitiveType::String);
                            suggestions.push(AISuggestion {
                                suggested_type: self.create_ast_type_node(&inferred_type),
                                confidence: 0.9,
                                reasoning: "Context suggests this is an email address".to_string(),
                                source: "semantic_analysis".to_string(),
                                evidence: vec!["Context contains email pattern".to_string()],
                            });
                        }
                    }
                    PrimitiveType::Int32 => {
                        if context.contains("id") || context.contains("identifier") {
                            let inferred_type = InferredType::Primitive(PrimitiveType::Int32);
                            suggestions.push(AISuggestion {
                                suggested_type: self.create_ast_type_node(&inferred_type),
                                confidence: 0.8,
                                reasoning: "Context suggests this is an identifier".to_string(),
                                source: "semantic_analysis".to_string(),
                                evidence: vec!["Context contains identifier pattern".to_string()],
                            });
                        }
                    }
                    _ => {}
                }
            }
            _ => {
                // For other types, suggest potential refinements
                suggestions.push(AISuggestion {
                    suggested_type: self.create_ast_type_node(base_type),
                    confidence: 0.6,
                    reasoning: "Type appears correct based on current analysis".to_string(),
                    source: "baseline_confirmation".to_string(),
                    evidence: vec!["Base type analysis".to_string()],
                });
            }
        }
        
        // Record suggestions in history
        self.inference_history.push(InferenceRecord {
            id: self.inference_history.len() as u64,
            input_context: InferenceContext {
                expression: expr.clone(),
                context: vec![],
                type_info: HashMap::new(),
            },
            inferred_type: base_type.clone(),
            success: true,
            feedback_score: None,
        });
        
        Ok(suggestions)
    }
    
    fn create_ast_type_node(&self, inferred_type: &InferredType) -> AstNode<Type> {
        let type_data = match inferred_type {
            InferredType::Primitive(prim_type) => Type::Primitive(prim_type.clone()),
            InferredType::Composite(comp_type) => Type::Composite(comp_type.clone()),
            InferredType::Function(func_type) => {
                // Create a simplified function type
                Type::Function(TypesFunctionType {
                    parameters: vec![],
                    return_type: Box::new(AstNode {
                        kind: Type::Primitive(PrimitiveType::Unit),
                        span: Span::dummy(),
                        id: NodeId::new(0),
                        metadata: NodeMetadata::default(),
                    }),
                    effects: func_type.effects.clone(),
                })
            },
            InferredType::Enhanced(enhanced) => {
                return self.create_ast_type_node(&enhanced.base_type);
            },
            InferredType::Unknown => Type::Primitive(PrimitiveType::Unit),
        };
        
        AstNode {
            kind: type_data,
            span: Span::dummy(),
            id: NodeId::new(0),
            metadata: NodeMetadata::default(),
        }
    }
    
    fn analyze_expression_context(&self, expr: &AstNode<Expr>) -> String {
        // Extract context information from the expression
        // This is a simplified implementation - in practice, this would be much more sophisticated
        match &expr.kind {
            Expr::Variable(var_expr) => var_expr.name.to_string(),
            Expr::Literal(_) => "literal_value".to_string(),
            Expr::Binary(_) => "binary_operation".to_string(),
            Expr::Unary(_) => "unary_operation".to_string(),
            Expr::Call(_) => "function_call".to_string(),
            Expr::Member(member_expr) => format!("field_access_{}", member_expr.member),
            Expr::Index(_) => "array_access".to_string(),
            Expr::Array(_) => "array_literal".to_string(),
            Expr::Object(_) => "object_literal".to_string(),
            Expr::Lambda(_) => "lambda_expression".to_string(),
            Expr::Block(_) => "block_expression".to_string(),
            Expr::If(_) => "conditional_expression".to_string(),
            Expr::Match(_) => "match_expression".to_string(),
            _ => "unknown_expression".to_string(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enable_ai_assistance: true,
            max_inference_depth: 10,
            min_confidence_threshold: 0.7,
            enable_constraint_propagation: true,
            enable_effect_inference: true,
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 0.001,
            enable_simplification: true,
        }
    }
}

impl Default for AIConfig {
    fn default() -> Self {
        Self {
            enable_pattern_matching: true,
            enable_learning: true,
            max_suggestions: 5,
        }
    }
}

impl Default for EffectMetadata {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            category: EffectCategory::Pure,
            ai_context: None,
        }
    }
}

impl Default for SemanticTypeInfo {
    fn default() -> Self {
        Self {
            business_domain: None,
            constraints: Vec::new(),
            usage_patterns: Vec::new(),
            confidence: 0.0,
        }
    }
}

impl Default for EnhancedType {
    fn default() -> Self {
        Self {
            base_type: Box::new(InferredType::Unknown),
            semantic_info: SemanticTypeInfo::default(),
            effects: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

impl Default for InferredType {
    fn default() -> Self {
        Self::Unknown
    }
} 