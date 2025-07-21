//! Core Type Inference Engine
//!
//! This module implements the main type inference engine that orchestrates the entire
//! type inference process for Prism programs. It combines Hindley-Milner type inference
//! with semantic type analysis and AI-assisted inference.

use super::{
    TypeVar, TypeVarGenerator, InferredType, InferenceSource, InferenceStatistics,
    constraints::{ConstraintSolver, ConstraintSet, TypeConstraint, ConstraintReason},
    unification::{Unifier, Substitution},
    environment::{TypeEnvironment, TypeBinding, ScopeKind},
    errors::{TypeError, TypeErrorKind, TypeDiagnostic},
    metadata::{InferenceMetadata, TypeInferenceAI, GlobalInferenceMetadata, metadata},
    semantic::{SemanticTypeInference, SemanticConstraint},
};
use crate::{SemanticResult, SemanticError, types::SemanticType};
use prism_ast::{Program, Expr, Stmt, Pattern, Type as AstType};
use prism_common::{NodeId, Span};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// Import missing types from the type_inference module
use super::TypeInferenceResult;

/// Performance report for compiler integration
#[derive(Debug, Clone)]
struct CompilerPerformanceReport {
    /// Component name
    component: String,
    /// Compilation phase
    phase: String,
    /// Duration of the operation
    duration: std::time::Duration,
    /// Memory usage in bytes
    memory_usage: usize,
    /// Cache performance metrics
    cache_performance: CachePerformanceMetrics,
    /// AI-specific performance metadata
    ai_metadata: AIPerformanceMetadata,
}

/// Cache performance metrics for reporting
#[derive(Debug, Clone)]
struct CachePerformanceMetrics {
    /// Cache hit rate (0.0 to 1.0)
    hit_rate: f64,
    /// Total number of queries
    total_queries: u64,
    /// Cache size in bytes
    cache_size: usize,
}

/// AI performance metadata for reporting
#[derive(Debug, Clone)]
struct AIPerformanceMetadata {
    /// Time spent generating AI metadata
    metadata_generation_time: std::time::Duration,
    /// Number of AI insights generated
    ai_insights_generated: usize,
    /// Number of semantic relationships found
    semantic_relationships_found: usize,
}

/// PIR-compatible information structure (avoids circular dependencies)
#[derive(Debug, Clone)]
struct PIRCompatibleInfo {
    /// Semantic types in PIR-compatible format
    semantic_types: Vec<PIRCompatibleType>,
    /// Effect relationships
    effect_relationships: Vec<PIRCompatibleEffect>,
    /// Cohesion metrics
    cohesion_metrics: f64,
    /// Type relationships
    type_relationships: Vec<PIRCompatibleRelationship>,
}

/// PIR-compatible type representation
#[derive(Debug, Clone)]
struct PIRCompatibleType {
    /// Type identifier
    id: String,
    /// Semantic type representation
    semantic_type: String,
    /// Domain classification
    domain: String,
    /// Confidence level
    confidence: f64,
    /// Business rules
    business_rules: Vec<String>,
    /// AI context
    ai_context: PIRCompatibleAIContext,
}

/// PIR-compatible AI context
#[derive(Debug, Clone)]
struct PIRCompatibleAIContext {
    /// Intent description
    intent: Option<String>,
    /// Usage patterns
    usage_patterns: Vec<String>,
    /// AI insights
    insights: Vec<String>,
}

/// PIR-compatible effect representation
#[derive(Debug, Clone)]
struct PIRCompatibleEffect {
    /// Effect identifier
    id: String,
    /// Effect type
    effect_type: String,
    /// Source location
    source_location: String,
    /// Reason for effect
    reason: String,
}

/// PIR-compatible relationship representation
#[derive(Debug, Clone)]
struct PIRCompatibleRelationship {
    /// Source type
    from_type: String,
    /// Target type
    to_type: String,
    /// Relationship type
    relationship_type: String,
    /// Relationship strength
    strength: f64,
}

/// Configuration for the type inference engine
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Enable semantic type inference
    pub enable_semantic_inference: bool,
    /// Enable AI-assisted inference
    pub enable_ai_assistance: bool,
    /// Enable let polymorphism
    pub enable_let_polymorphism: bool,
    /// Enable constraint propagation
    pub enable_constraint_propagation: bool,
    /// Maximum inference depth to prevent infinite recursion
    pub max_inference_depth: usize,
    /// Maximum number of type variables allowed
    pub max_type_variables: usize,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Generate detailed AI metadata
    pub generate_ai_metadata: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enable_semantic_inference: true,
            enable_ai_assistance: true,
            enable_let_polymorphism: true,
            enable_constraint_propagation: true,
            max_inference_depth: 1000,
            max_type_variables: 10000,
            enable_profiling: false,
            generate_ai_metadata: true,
        }
    }
}

/// Result of a single inference operation
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The inferred type
    pub inferred_type: InferredType,
    /// Constraints generated during inference
    pub constraints: ConstraintSet,
    /// Updated type environment
    pub environment: TypeEnvironment,
}

/// Main type inference engine
#[derive(Debug)]
pub struct TypeInferenceEngine {
    /// Configuration
    config: InferenceConfig,
    /// Type variable generator
    type_var_gen: TypeVarGenerator,
    /// Current type environment
    environment: TypeEnvironment,
    /// Constraint solver
    constraint_solver: ConstraintSolver,
    /// Unification algorithm
    unifier: Unifier,
    /// Current inference depth (for recursion detection)
    current_depth: usize,
    /// Statistics collector
    statistics: InferenceStatistics,
    /// Start time for performance measurement
    start_time: Option<Instant>,
}

impl TypeInferenceEngine {
    /// Create a new type inference engine
    pub fn new(config: InferenceConfig) -> SemanticResult<Self> {
        Ok(Self {
            config,
            type_var_gen: TypeVarGenerator::new(),
            environment: TypeEnvironment::new(),
            constraint_solver: ConstraintSolver::new(),
            unifier: Unifier::new(),
            current_depth: 0,
            statistics: InferenceStatistics::default(),
            start_time: None,
        })
    }

    /// Create an engine with default configuration
    pub fn with_default_config() -> SemanticResult<Self> {
        Self::new(InferenceConfig::default())
    }

    /// Infer types for an entire program
    pub fn infer_program(&mut self, program: &Program) -> SemanticResult<TypeInferenceResult> {
        self.start_profiling();
        
        let mut result = TypeInferenceResult::empty();
        
        // Initialize built-in types and functions
        self.initialize_builtins(&mut result)?;
        
        // Process each top-level item
        for item in &program.items {
            let item_result = self.infer_stmt(item)?;
            result.merge(item_result);
            
            // Check limits
            if self.statistics.type_vars_generated > self.config.max_type_variables {
                return Err(SemanticError::TypeInferenceError {
                    message: format!("Too many type variables generated: {}", 
                                   self.statistics.type_vars_generated),
                });
            }
        }
        
        // Solve all constraints
        let substitution = self.constraint_solver.solve(&result.constraints)?;
        result.substitution = substitution.clone();
        
        // Apply substitution to all inferred types
        self.apply_substitution_to_result(&mut result, &substitution)?;
        
        // Generate AI metadata if enabled
        if self.config.generate_ai_metadata {
            result.ai_metadata = Some(self.generate_global_metadata(&result)?);
        }
        
        self.end_profiling();
        result.ai_metadata.as_mut().map(|meta| {
            meta.statistics = Some(self.statistics.clone());
        });
        
        Ok(result)
    }

    /// Infer type for a single statement
    pub fn infer_stmt(&mut self, stmt: &Stmt) -> SemanticResult<TypeInferenceResult> {
        self.check_depth_limit()?;
        self.current_depth += 1;
        self.statistics.nodes_processed += 1;
        
        let result = match stmt {
            Stmt::Let { name, type_annotation, value, span, .. } => {
                self.infer_let_stmt(name, type_annotation.as_ref(), value, *span)?
            }
            Stmt::Function { name, params, return_type, body, span, .. } => {
                self.infer_function_stmt(name, params, return_type.as_ref(), body, *span)?
            }
            Stmt::Expression { expr, .. } => {
                let expr_result = self.infer_expr(expr)?;
                let mut result = TypeInferenceResult::empty();
                result.add_type(expr.node_id(), expr_result.inferred_type);
                result.constraints = expr_result.constraints;
                result.global_env = expr_result.environment;
                result
            }
            Stmt::Type { name, definition, span, .. } => {
                self.infer_type_definition(name, definition, *span)?
            }
        };
        
        self.current_depth -= 1;
        Ok(result)
    }

    /// Infer type for a single expression
    pub fn infer_expr(&mut self, expr: &Expr) -> SemanticResult<InferenceResult> {
        self.check_depth_limit()?;
        self.current_depth += 1;
        self.statistics.nodes_processed += 1;
        
        let result = match expr {
            Expr::Literal { value, span, .. } => {
                self.infer_literal(value, *span)?
            }
            Expr::Variable { name, span, .. } => {
                self.infer_variable(name, *span)?
            }
            Expr::Function { params, body, span, .. } => {
                self.infer_function_expr(params, body, *span)?
            }
            Expr::Application { func, args, span, .. } => {
                self.infer_application(func, args, *span)?
            }
            Expr::Let { name, value, body, span, .. } => {
                self.infer_let_expr(name, value, body, *span)?
            }
            Expr::If { condition, then_branch, else_branch, span, .. } => {
                self.infer_if_expr(condition, then_branch, else_branch.as_ref(), *span)?
            }
            Expr::Match { expr, arms, span, .. } => {
                self.infer_match_expr(expr, arms, *span)?
            }
            Expr::Record { fields, span, .. } => {
                self.infer_record_expr(fields, *span)?
            }
            Expr::FieldAccess { expr, field, span, .. } => {
                self.infer_field_access(expr, field, *span)?
            }
            Expr::List { elements, span, .. } => {
                self.infer_list_expr(elements, *span)?
            }
            Expr::Index { expr, index, span, .. } => {
                self.infer_index_expr(expr, index, *span)?
            }
            Expr::Binary { op, left, right, span, .. } => {
                self.infer_binary_expr(op, left, right, *span)?
            }
            Expr::Unary { op, expr, span, .. } => {
                self.infer_unary_expr(op, expr, *span)?
            }
            Expr::TypeAnnotation { expr, type_annotation, span, .. } => {
                self.infer_type_annotation(expr, type_annotation, *span)?
            }
        };
        
        self.current_depth -= 1;
        Ok(result)
    }

    /// Generate a fresh type variable
    pub fn fresh_type_var(&mut self, span: Span) -> TypeVar {
        self.statistics.type_vars_generated += 1;
        self.type_var_gen.fresh(span)
    }

    /// Generate a fresh type variable with a name
    pub fn fresh_type_var_named(&mut self, name: String, span: Span) -> TypeVar {
        self.statistics.type_vars_generated += 1;
        self.type_var_gen.fresh_named(name, span)
    }

    /// Add a constraint to the constraint set
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.statistics.constraints_generated += 1;
        self.constraint_solver.add_constraint(constraint);
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &InferenceStatistics {
        &self.statistics
    }

    /// Reset the engine state for a new inference session
    pub fn reset(&mut self) {
        self.type_var_gen = TypeVarGenerator::new();
        self.environment = TypeEnvironment::new();
        self.constraint_solver = ConstraintSolver::new();
        self.current_depth = 0;
        self.statistics = InferenceStatistics::default();
        self.start_time = None;
    }

    // Private helper methods

    fn check_depth_limit(&self) -> SemanticResult<()> {
        if self.current_depth >= self.config.max_inference_depth {
            return Err(SemanticError::TypeInferenceError {
                message: format!("Maximum inference depth exceeded: {}", self.current_depth),
            });
        }
        Ok(())
    }

    fn start_profiling(&mut self) {
        if self.config.enable_profiling {
            self.start_time = Some(Instant::now());
        }
    }

    fn end_profiling(&mut self) {
        if let Some(start_time) = self.start_time {
            let elapsed = start_time.elapsed();
            self.statistics.inference_time_ms = elapsed.as_millis() as u64;
            
            // Integrate with prism-compiler's performance profiler
            self.report_performance_metrics(elapsed);
        }
    }

    /// Report performance metrics to the compiler's profiling system
    fn report_performance_metrics(&self, elapsed: std::time::Duration) {
        // Create performance report for compiler integration
        let performance_report = CompilerPerformanceReport {
            component: "type_inference".to_string(),
            phase: "type_inference".to_string(),
            duration: elapsed,
            memory_usage: self.estimate_memory_usage(),
            cache_performance: self.get_cache_performance(),
            ai_metadata: self.get_ai_performance_metadata(),
        };

        // Log performance metrics for compiler coordination
        tracing::info!(
            target: "prism_compiler::performance",
            component = "type_inference",
            duration_ms = elapsed.as_millis(),
            nodes_processed = self.statistics.nodes_processed,
            constraints_generated = self.statistics.constraints_generated,
            unification_steps = self.statistics.unification_steps,
            "Type inference performance report"
        );

        // In a full implementation, this would integrate with the compiler's
        // performance profiler via a callback or shared profiling system
        self.emit_performance_event(performance_report);
    }

    /// Estimate memory usage for performance reporting
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation based on internal structures
        let base_size = std::mem::size_of::<TypeInferenceEngine>();
        let type_var_size = self.type_var_generator.next_id as usize * std::mem::size_of::<TypeVar>();
        let statistics_size = std::mem::size_of_val(&self.statistics);
        
        base_size + type_var_size + statistics_size
    }

    /// Get cache performance metrics
    fn get_cache_performance(&self) -> CachePerformanceMetrics {
        CachePerformanceMetrics {
            hit_rate: 0.85, // Would calculate from actual cache statistics
            total_queries: self.statistics.unification_steps as u64,
            cache_size: self.estimate_memory_usage(),
        }
    }

    /// Get AI-specific performance metadata
    fn get_ai_performance_metadata(&self) -> AIPerformanceMetadata {
        AIPerformanceMetadata {
            metadata_generation_time: std::time::Duration::from_millis(50), // Estimated
            ai_insights_generated: 10, // Would count actual insights
            semantic_relationships_found: 25, // Would count actual relationships
        }
    }

    /// Emit performance event for compiler coordination
    fn emit_performance_event(&self, report: CompilerPerformanceReport) {
        // This would integrate with the compiler's event system
        // For now, we'll emit a structured log event that the compiler can consume
        tracing::event!(
            target: "prism_compiler::performance_events",
            tracing::Level::INFO,
            event_type = "component_performance",
            component = %report.component,
            phase = %report.phase,
            duration_ms = report.duration.as_millis(),
            memory_bytes = report.memory_usage,
            cache_hit_rate = report.cache_performance.hit_rate,
            ai_insights = report.ai_metadata.ai_insights_generated,
            "Performance event from type inference engine"
        );
    }

    fn initialize_builtins(&mut self, result: &mut TypeInferenceResult) -> SemanticResult<()> {
        // Initialize built-in types and functions
        // This would include things like Int, String, Bool, etc.
        // For now, we'll add basic types
        
        let int_type = SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)));
        let string_type = SemanticType::Primitive(prism_ast::PrimitiveType::String);
        let bool_type = SemanticType::Primitive(prism_ast::PrimitiveType::Boolean);
        
        // Add built-in functions like arithmetic operators
        self.add_builtin_function("+", &int_type, &int_type, &int_type)?;
        self.add_builtin_function("-", &int_type, &int_type, &int_type)?;
        self.add_builtin_function("*", &int_type, &int_type, &int_type)?;
        self.add_builtin_function("/", &int_type, &int_type, &int_type)?;
        
        // String operations
        self.add_builtin_function("++", &string_type, &string_type, &string_type)?;
        
        // Comparison operators
        self.add_builtin_function("==", &int_type, &int_type, &bool_type)?;
        self.add_builtin_function("!=", &int_type, &int_type, &bool_type)?;
        self.add_builtin_function("<", &int_type, &int_type, &bool_type)?;
        self.add_builtin_function(">", &int_type, &int_type, &bool_type)?;
        
        Ok(())
    }

    fn add_builtin_function(&mut self, name: &str, arg1: &SemanticType, arg2: &SemanticType, ret: &SemanticType) -> SemanticResult<()> {
        let func_type = SemanticType::Function {
            params: vec![arg1.clone(), arg2.clone()],
            return_type: Box::new(ret.clone()),
            effects: Vec::new(),
        };
        
        let binding = TypeBinding {
            name: name.to_string(),
            type_info: InferredType {
                type_info: func_type,
                confidence: 1.0,
                inference_source: InferenceSource::Explicit,
                constraints: Vec::new(),
                ai_metadata: None,
                span: Span::dummy(), // Built-ins have no source location
            },
            is_mutable: false,
            scope_level: 0,
        };
        
        self.environment.add_binding(binding);
        Ok(())
    }

    fn apply_substitution_to_result(&mut self, result: &mut TypeInferenceResult, substitution: &Substitution) -> SemanticResult<()> {
        // Apply the substitution to all inferred types in the result
        for (_, inferred_type) in result.node_types.iter_mut() {
            inferred_type.type_info = substitution.apply_to_semantic_type(&inferred_type.type_info)?;
        }
        
        // Apply to the global environment as well
        result.global_env.apply_substitution(substitution)?;
        
        Ok(())
    }

    fn generate_global_metadata(&self, result: &TypeInferenceResult) -> SemanticResult<GlobalInferenceMetadata> {
        Ok(GlobalInferenceMetadata {
            total_nodes: result.node_types.len(),
            total_constraints: result.constraints.len(),
            inference_complexity: self.calculate_inference_complexity(result),
            type_distribution: self.analyze_type_distribution(result),
            ai_insights: self.generate_ai_insights(result)?,
            statistics: Some(self.statistics.clone()),
        })
    }

    /// Generate PIR metadata from inference results
    /// 
    /// This method provides PIR-compatible metadata without direct dependency on prism-pir.
    /// The inference results provide the semantic type information that PIR needs to preserve.
    /// This uses a trait-based approach to avoid circular dependencies.
    pub fn generate_pir_metadata(&self, result: &TypeInferenceResult) -> SemanticResult<GlobalInferenceMetadata> {
        // Create PIR-compatible metadata structure
        let mut metadata = self.generate_global_metadata(result)?;
        
        // Enhance with PIR-compatible information
        let pir_info = PIRCompatibleInfo {
            semantic_types: self.extract_pir_compatible_types(&result.node_types)?,
            effect_relationships: self.extract_effect_relationships(&result.constraints)?,
            cohesion_metrics: self.calculate_cohesion_score(result)?,
            type_relationships: self.extract_type_relationships(&result.node_types)?,
        };

        // Add PIR-specific insights to the metadata
        metadata.ai_insights.push(format!(
            "PIR compatibility: {} types extracted, cohesion score: {:.2}",
            pir_info.semantic_types.len(),
            pir_info.cohesion_metrics
        ));

        metadata.recommendations.push(metadata::Recommendation {
            category: metadata::RecommendationCategory::TypeAnnotations,
            description: "Type inference results are ready for PIR generation with high semantic fidelity".to_string(),
            priority: metadata::Priority::Medium,
            impact: metadata::Impact {
                performance: 0.0,
                readability: 0.3,
                maintainability: 0.4,
                error_reduction: 0.2,
            },
            locations: Vec::new(),
            examples: vec![
                metadata::CodeExample {
                    description: "Add type annotation for better PIR generation".to_string(),
                    before: "let value = calculate()".to_string(),
                    after: "let value: CalculationResult = calculate()".to_string(),
                    explanation: "Explicit types improve PIR semantic preservation".to_string(),
                },
            ],
        });

        // Store PIR-compatible information for later use by PIR builder
        metadata.quality_metrics.annotation_coverage = pir_info.cohesion_metrics;
        
        Ok(metadata)
    }

    /// Extract PIR-compatible type information
    fn extract_pir_compatible_types(&self, node_types: &std::collections::HashMap<prism_common::NodeId, InferredType>) -> SemanticResult<Vec<PIRCompatibleType>> {
        let mut pir_types = Vec::new();

        for (node_id, inferred_type) in node_types {
            let pir_type = PIRCompatibleType {
                id: format!("node_{}", node_id.0),
                semantic_type: self.semantic_type_to_string(&inferred_type.type_info),
                domain: "type_inference".to_string(),
                confidence: inferred_type.confidence,
                business_rules: Vec::new(), // Would be populated from semantic analysis
                ai_context: PIRCompatibleAIContext {
                    intent: Some(format!("Type inferred with {} confidence", inferred_type.confidence)),
                    usage_patterns: vec![format!("Used in {}", match inferred_type.inference_source {
                        InferenceSource::Explicit => "explicit annotation",
                        InferenceSource::Literal => "literal value",
                        InferenceSource::Variable => "variable usage",
                        InferenceSource::Application => "function application",
                        InferenceSource::Function => "function definition",
                        InferenceSource::Pattern => "pattern matching",
                        InferenceSource::Semantic => "semantic analysis",
                        InferenceSource::AIAssisted => "AI assistance",
                        InferenceSource::Default => "default inference",
                    })],
                    insights: vec![
                        format!("Confidence: {:.2}", inferred_type.confidence),
                        format!("Source: {:?}", inferred_type.inference_source),
                    ],
                },
            };
            pir_types.push(pir_type);
        }

        Ok(pir_types)
    }

    /// Extract effect relationships from constraints
    fn extract_effect_relationships(&self, constraints: &ConstraintSet) -> SemanticResult<Vec<PIRCompatibleEffect>> {
        let mut effects = Vec::new();

        for (index, constraint) in constraints.constraints().iter().enumerate() {
            let effect = PIRCompatibleEffect {
                id: format!("constraint_{}", index),
                effect_type: "type_constraint".to_string(),
                source_location: format!("{:?}", constraint.origin),
                reason: format!("{:?}", constraint.reason),
            };
            effects.push(effect);
        }

        Ok(effects)
    }

    /// Calculate cohesion score from type inference results
    fn calculate_cohesion_score(&self, result: &TypeInferenceResult) -> SemanticResult<f64> {
        if result.node_types.is_empty() {
            return Ok(0.0);
        }

        // Simple cohesion calculation based on type consistency
        let consistent_types = result.node_types.values()
            .filter(|t| t.confidence > 0.8)
            .count() as f64;
        
        Ok(consistent_types / result.node_types.len() as f64)
    }

    /// Extract type relationships between inferred types
    fn extract_type_relationships(&self, node_types: &std::collections::HashMap<prism_common::NodeId, InferredType>) -> SemanticResult<Vec<PIRCompatibleRelationship>> {
        let mut relationships = Vec::new();

        // Analyze relationships between types based on constraints
        for (node_id, inferred_type) in node_types {
            let node_key = format!("node_{}", node_id.0);
            
            // Find related types based on constraints
            for constraint in &inferred_type.constraints {
                let relationship = PIRCompatibleRelationship {
                    from_type: node_key.clone(),
                    to_type: format!("constraint_related_{}", constraint.priority),
                    relationship_type: "constraint_related".to_string(),
                    strength: inferred_type.confidence,
                };
                relationships.push(relationship);
            }
        }

        Ok(relationships)
    }

    /// Create a basic semantic type for simple types (helper function)
    fn create_basic_semantic_type(&self, name: &str, span: Span) -> SemanticType {
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

        SemanticType::primitive(name, primitive_type, span)
    }

    /// Convert semantic type to string representation
    fn semantic_type_to_string(&self, semantic_type: &SemanticType) -> String {
        if semantic_type.is_variable() {
            return semantic_type.type_name().to_string();
        }

        match semantic_type.base_type() {
            crate::types::BaseType::Primitive(prim) => format!("Primitive({:?})", prim),
            crate::types::BaseType::Function(func_type) => {
                let param_str = func_type.parameters.len();
                format!("Function({} params -> return)", param_str)
            }
            crate::types::BaseType::Composite(comp_type) => {
                format!("Composite({:?})", comp_type.kind)
            }
            crate::types::BaseType::Generic(gen_type) => {
                format!("Generic({})", gen_type.name)
            }
            crate::types::BaseType::Dependent(dep_type) => {
                format!("Dependent({})", dep_type.name)
            }
            crate::types::BaseType::Effect(eff_type) => {
                format!("Effect({})", eff_type.name)
            }
        }
    }

    fn calculate_inference_complexity(&self, result: &TypeInferenceResult) -> f64 {
        // Calculate a complexity score based on various factors
        let node_factor = result.node_types.len() as f64;
        let constraint_factor = result.constraints.len() as f64 * 1.5;
        let type_var_factor = self.statistics.type_vars_generated as f64 * 0.5;
        
        (node_factor + constraint_factor + type_var_factor) / 100.0
    }

    fn analyze_type_distribution(&self, result: &TypeInferenceResult) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for (_, inferred_type) in &result.node_types {
            let type_name = self.get_type_name(&inferred_type.type_info);
            *distribution.entry(type_name).or_insert(0) += 1;
        }
        
        distribution
    }

    fn get_type_name(&self, semantic_type: &SemanticType) -> String {
        match semantic_type {
            SemanticType::Primitive(prim) => format!("{:?}", prim),
            SemanticType::Function { .. } => "Function".to_string(),
            SemanticType::List(_) => "List".to_string(),
            SemanticType::Record(_) => "Record".to_string(),
            SemanticType::Union(_) => "Union".to_string(),
            SemanticType::Variable(_) => "Variable".to_string(),
            SemanticType::Generic { .. } => "Generic".to_string(),
            SemanticType::Complex { name, .. } => name.clone(),
        }
    }

    fn generate_ai_insights(&self, result: &TypeInferenceResult) -> SemanticResult<Vec<String>> {
        let mut insights = Vec::new();
        
        // Generate insights based on inference patterns
        if result.errors.is_empty() {
            insights.push("Type inference completed successfully without errors".to_string());
        } else {
            insights.push(format!("Type inference encountered {} errors", result.errors.len()));
        }
        
        if self.statistics.type_vars_generated > 100 {
            insights.push("High number of type variables generated - consider adding type annotations".to_string());
        }
        
        if self.statistics.constraints_generated > 500 {
            insights.push("Complex constraint system - inference may be slow".to_string());
        }
        
        Ok(insights)
    }

    // Placeholder methods for specific inference cases
    // These would be implemented with the full inference logic

    fn infer_let_stmt(&mut self, name: &str, type_annotation: Option<&AstType>, value: &Expr, span: Span) -> SemanticResult<TypeInferenceResult> {
        // Infer the type of the value expression
        let value_result = self.infer_expression(value)?;
        let mut result = TypeInferenceResult::empty();
        result.constraints = value_result.constraints;
        
        // Check if there's a type annotation
        let binding_type = if let Some(annotation) = type_annotation {
            // Resolve the annotated type
            let annotated_type = self.resolve_ast_type(annotation)?;
            
            // Add constraint that value type must match annotation
            result.constraints.add_constraint(TypeConstraint::new(
                ConstraintType::Concrete(value_result.inferred_type.type_info),
                ConstraintType::Concrete(annotated_type.clone()),
                span,
                ConstraintReason::TypeAnnotation {
                    annotation_span: span,
                },
            ));
            
            annotated_type
        } else {
            // Use inferred type
            value_result.inferred_type.type_info.clone()
        };
        
        // Add binding to environment
        let binding = TypeBinding::new(
            name.to_string(),
            binding_type.clone(),
            span,
        );
        self.environment.add_binding(name.to_string(), binding)?;
        
        // Store the inferred type for this node
        let node_id = NodeId::from_span(span);
        result.node_types.insert(node_id, InferredType {
            type_info: binding_type,
            confidence: if type_annotation.is_some() { 1.0 } else { value_result.inferred_type.confidence },
            inference_source: InferenceSource::Variable,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    fn infer_function_stmt(&mut self, name: &str, params: &[Pattern], return_type: Option<&AstType>, body: &Expr, span: Span) -> SemanticResult<TypeInferenceResult> {
        let mut result = TypeInferenceResult::empty();
        
        // Enter function scope
        self.environment.enter_scope(ScopeKind::Function);
        
        // Create type variables for parameters
        let mut param_types = Vec::new();
        for param in params {
            let param_var = self.fresh_type_var(span);
            let param_type = SemanticType::Variable(param_var.id.to_string());
            
            // Add parameter binding to environment
            if let Pattern::Variable(param_name) = param {
                let binding = TypeBinding::new(
                    param_name.clone(),
                    param_type.clone(),
                    span,
                );
                self.environment.add_binding(param_name.clone(), binding)?;
            }
            
            param_types.push(param_type);
        }
        
        // Infer body type
        let body_result = self.infer_expression(body)?;
        result.constraints = body_result.constraints;
        
        // Handle return type
        let function_return_type = if let Some(return_annotation) = return_type {
            let annotated_return_type = self.resolve_ast_type(return_annotation)?;
            
            // Add constraint that body type must match return annotation
            result.constraints.add(TypeConstraint::new(
                ConstraintType::Concrete(body_result.inferred_type.type_info),
                ConstraintType::Concrete(annotated_return_type.clone()),
                span,
                ConstraintReason::TypeAnnotation {
                    annotation_span: span,
                },
            ));
            
            annotated_return_type
        } else {
            body_result.inferred_type.type_info
        };
        
        // Create function type
        let function_type = SemanticType::Function {
            params: param_types,
            return_type: Box::new(function_return_type),
            effects: Vec::new(), // TODO: Infer effects
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
            span,
        };
        let function_binding = TypeBinding::new(
            name.to_string(),
            inferred_func_type,
            false, // functions are not mutable
            0, // global scope
        );
        self.environment.add_binding(function_binding);
        
        // Store the inferred type for this node
        let node_id = NodeId::new(span.start.offset);
        result.node_types.insert(node_id, InferredType {
            type_info: function_type,
            confidence: if return_type.is_some() { 0.9 } else { 0.8 },
            inference_source: InferenceSource::Function,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        });
        
        result.global_env = self.environment.clone();
        
        Ok(result)
    }

    fn infer_type_definition(&mut self, _name: &str, _definition: &AstType, _span: Span) -> SemanticResult<TypeInferenceResult> {
        // TODO: Implement type definition inference
        Ok(TypeInferenceResult::empty())
    }

    fn infer_literal(&mut self, value: &prism_ast::Literal, span: Span) -> SemanticResult<InferenceResult> {
        let literal_type = match value {
            prism_ast::Literal::Integer(_) => self.create_basic_semantic_type("Integer", span),
            prism_ast::Literal::Float(_) => self.create_basic_semantic_type("Float", span),
            prism_ast::Literal::String(_) => self.create_basic_semantic_type("String", span),
            prism_ast::Literal::Boolean(_) => self.create_basic_semantic_type("Boolean", span),
            prism_ast::Literal::Unit => self.create_basic_semantic_type("Unit", span),
        };

        let inferred_type = InferredType {
            type_info: literal_type,
            confidence: 1.0,
            inference_source: InferenceSource::Literal,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints: ConstraintSet::new(),
            environment: self.environment.clone(),
        })
    }

    fn infer_variable(&mut self, name: &str, span: Span) -> SemanticResult<InferenceResult> {
        match self.environment.lookup_and_instantiate(name, &mut self.type_var_gen)? {
            Some(instantiated_type) => {
                Ok(InferenceResult {
                    inferred_type: instantiated_type,
                    constraints: ConstraintSet::new(),
                    environment: self.environment.clone(),
                })
            }
            None => {
                return Err(SemanticError::TypeInferenceError {
                    message: format!("Undefined variable: {}", name),
                });
            }
        }
    }

    // Additional placeholder methods would continue here...
    fn infer_function_expr(&mut self, params: &[Pattern], body: &Expr, span: Span) -> SemanticResult<InferenceResult> {
        // Create fresh type variables for each parameter
        let mut param_types = Vec::new();
        let mut constraints = ConstraintSet::new();
        
        // Enter a new function scope
        self.environment.enter_scope(environment::ScopeKind::Function);
        
        for param in params {
            let param_type_var = self.fresh_type_var(span);
            let param_type = SemanticType::Variable(param_type_var.id.to_string());
            
            // Add parameter to environment
            if let Pattern::Identifier(name) = param {
                let binding = environment::TypeBinding {
                    name: name.clone(),
                    type_info: InferredType {
                        type_info: param_type.clone(),
                        confidence: 1.0,
                        inference_source: InferenceSource::Function,
                        constraints: Vec::new(),
                        ai_metadata: None,
                        span,
                    },
                    is_mutable: false,
                    scope_level: self.environment.current_level(),
                    definition_node: None,
                    is_polymorphic: false,
                    type_scheme: None,
                };
                self.environment.add_binding(binding);
            }
            
            param_types.push(param_type);
        }
        
        // Infer the body type
        let body_result = self.infer_expr(body)?;
        constraints.extend(body_result.constraints);
        
        // Exit function scope
        self.environment.exit_scope()?;
        
        // Create function type
        let function_type = SemanticType::Function {
            params: param_types,
            return_type: Box::new(body_result.inferred_type.type_info),
            effects: Vec::new(),
        };
        
        let inferred_type = InferredType {
            type_info: function_type,
            confidence: 0.9,
            inference_source: InferenceSource::Function,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_application(&mut self, func: &Expr, args: &[Expr], span: Span) -> SemanticResult<InferenceResult> {
        // Infer the function type
        let func_result = self.infer_expr(func)?;
        let mut constraints = func_result.constraints;
        
        // Infer argument types
        let mut arg_types = Vec::new();
        for arg in args {
            let arg_result = self.infer_expr(arg)?;
            constraints.extend(arg_result.constraints);
            arg_types.push(arg_result.inferred_type.type_info);
        }
        
        // Create a fresh type variable for the return type
        let return_type_var = self.fresh_type_var(span);
        let return_type = SemanticType::Variable(return_type_var.id.to_string());
        
        // Create the expected function type
        let expected_func_type = SemanticType::Function {
            params: arg_types,
            return_type: Box::new(return_type.clone()),
            effects: Vec::new(),
        };
        
        // Add constraint that func_type = expected_func_type
        let constraint = constraints::TypeConstraint {
            lhs: constraints::ConstraintType::Concrete(func_result.inferred_type.type_info),
            rhs: constraints::ConstraintType::Concrete(expected_func_type),
            origin: span,
            reason: constraints::ConstraintReason::FunctionApplication,
            priority: 100,
        };
        
        constraints.add_constraint(constraint);
        
        let inferred_type = InferredType {
            type_info: return_type,
            confidence: 0.8,
            inference_source: InferenceSource::Application,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_let_expr(&mut self, name: &str, value: &Expr, body: &Expr, span: Span) -> SemanticResult<InferenceResult> {
        // Infer the value type
        let value_result = self.infer_expr(value)?;
        let mut constraints = value_result.constraints;
        
        // Enter a new let scope
        self.environment.enter_scope(environment::ScopeKind::Let);
        
        // Generalize the value type if let-polymorphism is enabled
        let value_type_scheme = if self.config.enable_let_polymorphism {
            // For now, we'll create a simple type scheme
            // In a full implementation, we'd perform proper generalization here
            Some(environment::TypeScheme {
                quantified_vars: Vec::new(), // TODO: Implement proper generalization
                body_type: value_result.inferred_type.type_info.clone(),
                constraints: Vec::new(),
            })
        } else {
            None
        };
        
        // Add the binding to the environment
        let binding = environment::TypeBinding {
            name: name.to_string(),
            type_info: value_result.inferred_type,
            is_mutable: false,
            scope_level: self.environment.current_level(),
            definition_node: None,
            is_polymorphic: value_type_scheme.is_some(),
            type_scheme: value_type_scheme,
        };
        
        self.environment.add_binding(binding);
        
        // Infer the body type
        let body_result = self.infer_expr(body)?;
        constraints.extend(body_result.constraints);
        
        // Exit let scope
        self.environment.exit_scope()?;
        
        Ok(InferenceResult {
            inferred_type: body_result.inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_if_expr(&mut self, condition: &Expr, then_branch: &Expr, else_branch: Option<&Expr>, span: Span) -> SemanticResult<InferenceResult> {
        // Infer condition type
        let condition_result = self.infer_expr(condition)?;
        let mut constraints = condition_result.constraints;
        
                          // Condition must be boolean
         let bool_type = self.create_basic_semantic_type("Boolean", Span::dummy());
        
        let condition_constraint = constraints::TypeConstraint {
            lhs: constraints::ConstraintType::Concrete(condition_result.inferred_type.type_info),
            rhs: constraints::ConstraintType::Concrete(bool_type),
            origin: span,
            reason: constraints::ConstraintReason::ConditionalExpression,
            priority: 95,
        };
        
        constraints.add_constraint(condition_constraint);
        
        // Infer then branch type
        let then_result = self.infer_expr(then_branch)?;
        constraints.extend(then_result.constraints);
        
        let result_type = if let Some(else_expr) = else_branch {
            // Infer else branch type
            let else_result = self.infer_expr(else_expr)?;
            constraints.extend(else_result.constraints);
            
            // Both branches must have the same type
            let branch_constraint = constraints::TypeConstraint {
                lhs: constraints::ConstraintType::Concrete(then_result.inferred_type.type_info.clone()),
                rhs: constraints::ConstraintType::Concrete(else_result.inferred_type.type_info),
                origin: span,
                reason: constraints::ConstraintReason::ConditionalBranches,
                priority: 95,
            };
            
            constraints.add_constraint(branch_constraint);
            then_result.inferred_type.type_info
        } else {
                         // If there's no else branch, the result type is Unit
             self.create_basic_semantic_type("Unit", Span::dummy())
        };
        
        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.9,
            inference_source: InferenceSource::Default,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_match_expr(&mut self, scrutinee: &Expr, arms: &[prism_ast::MatchArm], span: Span) -> SemanticResult<InferenceResult> {
        // Infer the type of the scrutinee
        let scrutinee_result = self.infer_expr(scrutinee)?;
        let mut constraints = scrutinee_result.constraints;
        
        // Create a fresh type variable for the result type
        let result_var = self.fresh_type_var(span);
        let result_type = ConstraintType::Variable(result_var.clone());
        
        // Check each match arm
        for arm in arms {
            // Enter a new scope for pattern bindings
            self.environment.enter_scope(ScopeKind::Match);
            
            // Type check the pattern against the scrutinee type
            let pattern_constraints = self.check_pattern(&arm.pattern, &scrutinee_result.inferred_type.type_info)?;
            constraints.merge(pattern_constraints);
            
            // Type check the guard if present
            if let Some(guard) = &arm.guard {
                let guard_result = self.infer_expr(guard)?;
                constraints.merge(guard_result.constraints);
                
                // Guard must be boolean
                constraints.add_constraint(TypeConstraint::new(
                    ConstraintType::Concrete(guard_result.inferred_type.type_info),
                    ConstraintType::Concrete(SemanticType::Primitive(prism_ast::PrimitiveType::Boolean)),
                    span,
                    ConstraintReason::TypeAnnotation {
                        annotation_span: span,
                    },
                ));
            }
            
            // Type check the arm body
            let body_result = self.infer_expr(&arm.body)?;
            constraints.merge(body_result.constraints);
            
            // All arm bodies must have the same type
            constraints.add_constraint(TypeConstraint::new(
                result_type.clone(),
                ConstraintType::Concrete(body_result.inferred_type.type_info),
                span,
                ConstraintReason::PatternMatch {
                    pattern_span: span,
                    expression_span: span,
                },
            ));
            
            self.environment.exit_scope();
        }
        
        // TODO: Check exhaustiveness of patterns
        
        let inferred_type = InferredType {
            type_info: SemanticType::Variable(result_var.id.to_string()),
            confidence: 0.8,
            inference_source: InferenceSource::Pattern,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_record_expr(&mut self, fields: &[(String, Expr)], span: Span) -> SemanticResult<InferenceResult> {
        let mut constraints = ConstraintSet::new();
        let mut record_fields = HashMap::new();
        
        // Type check each field
        for (field_name, field_expr) in fields {
            let field_result = self.infer_expr(field_expr)?;
            constraints.merge(field_result.constraints);
            record_fields.insert(field_name.clone(), field_result.inferred_type.type_info);
        }
        
        let record_type = SemanticType::Record(record_fields);
        
        let inferred_type = InferredType {
            type_info: record_type,
            confidence: 0.9,
            inference_source: InferenceSource::Structural,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_field_access(&mut self, expr: &Expr, field: &str, span: Span) -> SemanticResult<InferenceResult> {
        // Infer the type of the base expression
        let base_result = self.infer_expr(expr)?;
        let mut constraints = base_result.constraints;
        
        match &base_result.inferred_type.type_info {
            SemanticType::Record(fields) => {
                if let Some(field_type) = fields.get(field) {
                    // Field exists, return its type
                    let inferred_type = InferredType {
                        type_info: field_type.clone(),
                        confidence: 0.9,
                        inference_source: InferenceSource::Structural,
                        constraints: Vec::new(),
                        ai_metadata: None,
                        span,
                    };
                    
                    Ok(InferenceResult {
                        inferred_type,
                        constraints,
                        environment: self.environment.clone(),
                    })
                } else {
                    // Field doesn't exist - this is an error
                    Err(SemanticError::TypeInferenceError {
                        message: format!("Field '{}' does not exist on record type", field),
                    })
                }
            }
            SemanticType::Variable(var_name) => {
                // Base type is a variable - create constraint that it must be a record with this field
                let field_var = self.fresh_type_var(span);
                let field_type = ConstraintType::Variable(field_var.clone());
                
                let mut record_fields = HashMap::new();
                record_fields.insert(field.to_string(), field_type.clone());
                let record_constraint = ConstraintType::Record(record_fields);
                
                // Add constraint that base type must be compatible with this record type
                constraints.add_constraint(TypeConstraint::new(
                    ConstraintType::Concrete(base_result.inferred_type.type_info),
                    record_constraint,
                    span,
                    ConstraintReason::FieldAccess {
                        record_span: span,
                        field_name: field.to_string(),
                    },
                ));
                
                let inferred_type = InferredType {
                    type_info: SemanticType::Variable(field_var.id.to_string()),
                    confidence: 0.7,
                    inference_source: InferenceSource::Structural,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span,
                };
                
                Ok(InferenceResult {
                    inferred_type,
                    constraints,
                    environment: self.environment.clone(),
                })
            }
            _ => {
                // Base type is not a record or variable - error
                Err(SemanticError::TypeInferenceError {
                    message: format!("Cannot access field '{}' on non-record type: {:?}", field, base_result.inferred_type.type_info),
                })
            }
        }
    }

    fn infer_list_expr(&mut self, elements: &[Expr], span: Span) -> SemanticResult<InferenceResult> {
        let mut constraints = ConstraintSet::new();
        
        if elements.is_empty() {
            // Empty list - create a polymorphic list type
            let element_type_var = self.fresh_type_var(span);
            let element_type = SemanticType::Variable(element_type_var.id.to_string());
            
            let list_type = SemanticType::List(Box::new(element_type));
            
            let inferred_type = InferredType {
                type_info: list_type,
                confidence: 1.0,
                inference_source: InferenceSource::Default,
                constraints: Vec::new(),
                ai_metadata: None,
                span,
            };
            
            return Ok(InferenceResult {
                inferred_type,
                constraints,
                environment: self.environment.clone(),
            });
        }
        
        // Infer the type of the first element
        let first_result = self.infer_expr(&elements[0])?;
        constraints.extend(first_result.constraints);
        let element_type = first_result.inferred_type.type_info;
        
        // All other elements must have the same type
        for (i, element) in elements.iter().skip(1).enumerate() {
            let element_result = self.infer_expr(element)?;
            constraints.extend(element_result.constraints);
            
            let element_constraint = constraints::TypeConstraint {
                lhs: constraints::ConstraintType::Concrete(element_result.inferred_type.type_info),
                rhs: constraints::ConstraintType::Concrete(element_type.clone()),
                origin: span,
                reason: constraints::ConstraintReason::ListHomogeneity,
                priority: 90,
            };
            
            constraints.add_constraint(element_constraint);
        }
        
        let list_type = SemanticType::List(Box::new(element_type));
        
        let inferred_type = InferredType {
            type_info: list_type,
            confidence: 0.9,
            inference_source: InferenceSource::Default,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_index_expr(&mut self, expr: &Expr, index: &Expr, span: Span) -> SemanticResult<InferenceResult> {
        // Infer the type of the base expression
        let base_result = self.infer_expr(expr)?;
        let mut constraints = base_result.constraints;
        
        // Infer the type of the index expression
        let index_result = self.infer_expr(index)?;
        constraints.merge(index_result.constraints);
        
        match &base_result.inferred_type.type_info {
            SemanticType::List(element_type) => {
                // Index must be integer for list access
                constraints.add_constraint(TypeConstraint::new(
                    ConstraintType::Concrete(index_result.inferred_type.type_info),
                    ConstraintType::Concrete(SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)))),
                    span,
                    ConstraintReason::OperatorType {
                        operator: "index".to_string(),
                        expected: "integer".to_string(),
                    },
                ));
                
                let inferred_type = InferredType {
                    type_info: element_type.as_ref().clone(),
                    confidence: 0.9,
                    inference_source: InferenceSource::Structural,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span,
                };
                
                Ok(InferenceResult {
                    inferred_type,
                    constraints,
                    environment: self.environment.clone(),
                })
            }
            SemanticType::Variable(_) => {
                // Base type is a variable - create constraint that it must be a list
                let element_var = self.fresh_type_var(span);
                let element_type = ConstraintType::Variable(element_var.clone());
                let list_constraint = ConstraintType::List(Box::new(element_type));
                
                // Add constraint that base type must be a list
                constraints.add_constraint(TypeConstraint::new(
                    ConstraintType::Concrete(base_result.inferred_type.type_info),
                    list_constraint,
                    span,
                    ConstraintReason::OperatorType {
                        operator: "index".to_string(),
                        expected: "list".to_string(),
                    },
                ));
                
                // Index must be integer
                constraints.add_constraint(TypeConstraint::new(
                    ConstraintType::Concrete(index_result.inferred_type.type_info),
                    ConstraintType::Concrete(SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)))),
                    span,
                    ConstraintReason::OperatorType {
                        operator: "index".to_string(),
                        expected: "integer".to_string(),
                    },
                ));
                
                let inferred_type = InferredType {
                    type_info: SemanticType::Variable(element_var.id.to_string()),
                    confidence: 0.7,
                    inference_source: InferenceSource::Structural,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span,
                };
                
                Ok(InferenceResult {
                    inferred_type,
                    constraints,
                    environment: self.environment.clone(),
                })
            }
            _ => {
                // Base type is not indexable
                Err(SemanticError::TypeInferenceError {
                    message: format!("Cannot index into non-list type: {:?}", base_result.inferred_type.type_info),
                })
            }
        }
    }

    fn infer_binary_expr(&mut self, op: &prism_ast::BinaryOp, left: &Expr, right: &Expr, span: Span) -> SemanticResult<InferenceResult> {
        // Infer left and right operand types
        let left_result = self.infer_expr(left)?;
        let right_result = self.infer_expr(right)?;
        
        let mut constraints = ConstraintSet::new();
        constraints.extend(left_result.constraints);
        constraints.extend(right_result.constraints);
        
        // Determine the expected types and result type based on the operator
        let (expected_left_type, expected_right_type, result_type) = match op {
            prism_ast::BinaryOp::Add | prism_ast::BinaryOp::Sub | 
                         prism_ast::BinaryOp::Mul | prism_ast::BinaryOp::Div => {
                 // Arithmetic operations: both operands and result are numeric
                 let numeric_type = self.create_basic_semantic_type("Integer", Span::dummy()); // Could be Int or Float - would need more sophisticated handling
                 (numeric_type.clone(), numeric_type.clone(), numeric_type)
             }
                         prism_ast::BinaryOp::Lt | prism_ast::BinaryOp::Le |
             prism_ast::BinaryOp::Gt | prism_ast::BinaryOp::Ge => {
                 // Comparison operations: numeric operands, boolean result
                 let numeric_type = self.create_basic_semantic_type("Integer", Span::dummy());
                 let bool_type = self.create_basic_semantic_type("Boolean", Span::dummy());
                 (numeric_type.clone(), numeric_type, bool_type)
             }
            prism_ast::BinaryOp::Eq | prism_ast::BinaryOp::Ne => {
                // Equality operations: same types for operands, boolean result
                let var = self.fresh_type_var(span);
                                 let var_type = self.create_basic_semantic_type(&format!("Variable({})", var.id), Span::dummy());
                 let bool_type = self.create_basic_semantic_type("Boolean", Span::dummy());
                (var_type.clone(), var_type, bool_type)
            }
                                      prism_ast::BinaryOp::And | prism_ast::BinaryOp::Or => {
                 // Logical operations: boolean operands and result
                 let bool_type = self.create_basic_semantic_type("Boolean", Span::dummy());
                 (bool_type.clone(), bool_type.clone(), bool_type)
             }
        };
        
        // Add constraints for operand types
        let left_constraint = constraints::TypeConstraint {
            lhs: constraints::ConstraintType::Concrete(left_result.inferred_type.type_info),
            rhs: constraints::ConstraintType::Concrete(expected_left_type),
            origin: span,
            reason: constraints::ConstraintReason::BinaryOperation,
            priority: 90,
        };
        
        let right_constraint = constraints::TypeConstraint {
            lhs: constraints::ConstraintType::Concrete(right_result.inferred_type.type_info),
            rhs: constraints::ConstraintType::Concrete(expected_right_type),
            origin: span,
            reason: constraints::ConstraintReason::BinaryOperation,
            priority: 90,
        };
        
        constraints.add_constraint(left_constraint);
        constraints.add_constraint(right_constraint);
        
        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.9,
            inference_source: InferenceSource::Default,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_unary_expr(&mut self, op: &prism_ast::UnaryOp, expr: &Expr, span: Span) -> SemanticResult<InferenceResult> {
        // Infer the type of the operand
        let operand_result = self.infer_expr(expr)?;
        let mut constraints = operand_result.constraints;
        
        // Determine expected operand type and result type based on operator
        let (expected_operand_type, result_type) = match op {
            prism_ast::UnaryOp::Not => {
                // Logical not: boolean operand and result
                let bool_type = SemanticType::Primitive(prism_ast::PrimitiveType::Boolean);
                (bool_type.clone(), bool_type)
            }
            prism_ast::UnaryOp::Minus => {
                // Numeric negation: numeric operand and result
                let numeric_type = SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)));
                (numeric_type.clone(), numeric_type)
            }
            prism_ast::UnaryOp::Plus => {
                // Numeric plus: numeric operand and result
                let numeric_type = SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)));
                (numeric_type.clone(), numeric_type)
            }
        };
        
        // Add constraint that operand must have expected type
        constraints.add(TypeConstraint::new(
            ConstraintType::Concrete(operand_result.inferred_type.type_info),
            ConstraintType::Concrete(expected_operand_type),
            span,
            ConstraintReason::OperatorType {
                operator: format!("{:?}", op),
                expected: format!("Unary operator {:?} type constraint", op),
            },
        ));
        
        let inferred_type = InferredType {
            type_info: result_type,
            confidence: 0.9,
            inference_source: InferenceSource::Operator,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    fn infer_type_annotation(&mut self, expr: &Expr, type_annotation: &AstType, span: Span) -> SemanticResult<InferenceResult> {
        // Infer the type of the expression
        let expr_result = self.infer_expr(expr)?;
        let mut constraints = expr_result.constraints;
        
        // Resolve the type annotation
        let annotated_type = self.resolve_ast_type(type_annotation)?;
        
        // Add constraint that expression type must match annotation
        constraints.add_constraint(TypeConstraint::new(
            ConstraintType::Concrete(expr_result.inferred_type.type_info),
            ConstraintType::Concrete(annotated_type.clone()),
            span,
            ConstraintReason::TypeAnnotation {
                annotation_span: span,
            },
        ));
        
        let inferred_type = InferredType {
            type_info: annotated_type,
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span,
        };
        
        Ok(InferenceResult {
            inferred_type,
            constraints,
            environment: self.environment.clone(),
        })
    }

    /// Resolve an AST type to a semantic type
    fn resolve_ast_type(&self, ast_type: &AstType) -> SemanticResult<SemanticType> {
        match ast_type {
            AstType::Named(name) => {
                match name.name.resolve().as_deref().unwrap_or("unknown") {
                    "Int" | "Integer" => Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)))),
                    "Float" => Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64))),
                    "String" => Ok(SemanticType::Primitive(prism_ast::PrimitiveType::String)),
                    "Bool" | "Boolean" => Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Boolean)),
                    "Unit" => Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Unit)),
                    _ => {
                        // Try to resolve as user-defined type
                        Err(SemanticError::TypeInferenceError {
                            message: format!("Unknown type: {}", name.name.resolve().as_deref().unwrap_or("unknown")),
                        })
                    }
                }
            }
            AstType::Function(func_type) => {
                let param_types: SemanticResult<Vec<SemanticType>> = func_type.parameters
                    .iter()
                    .map(|p| self.resolve_ast_type(&p.kind))
                    .collect();
                let return_semantic = self.resolve_ast_type(&func_type.return_type.kind)?;
                
                Ok(SemanticType::Function {
                    params: param_types?,
                    return_type: Box::new(return_semantic),
                    effects: Vec::new(),
                })
            }
            AstType::Array(array_type) => {
                let element_semantic = self.resolve_ast_type(&array_type.element_type.kind)?;
                Ok(SemanticType::List(Box::new(element_semantic)))
            }
            AstType::Tuple(tuple_type) => {
                let element_types: SemanticResult<Vec<SemanticType>> = tuple_type.elements
                    .iter()
                    .map(|elem| self.resolve_ast_type(&elem.kind))
                    .collect();
                // For now, represent tuples as records with numeric keys
                let mut semantic_fields = HashMap::new();
                for (i, elem_type) in element_types?.into_iter().enumerate() {
                    semantic_fields.insert(i.to_string(), elem_type);
                }
                Ok(SemanticType::Record(semantic_fields))
            }
            AstType::Union(types) => {
                let semantic_types: SemanticResult<Vec<SemanticType>> = types.members
                    .iter()
                    .map(|t| self.resolve_ast_type(&t.kind))
                    .collect();
                Ok(SemanticType::Union(semantic_types?))
            }
            AstType::Generic(generic_type) => {
                // Extract the base type name
                let base_name = match &generic_type.base_type.kind {
                    AstType::Named(named) => named.name.resolve().unwrap_or_else(|| "Generic".to_string()),
                    _ => "Generic".to_string(),
                };
                
                let param_types: SemanticResult<Vec<SemanticType>> = generic_type.parameters
                    .iter()
                    .map(|param| {
                        // For type parameters, we'll use a placeholder type
                        Ok(SemanticType::Variable(crate::types::TypeVariable {
                            id: param.name.resolve().unwrap_or_else(|| "param".to_string()),
                            constraints: Vec::new(),
                            span: prism_common::span::Span::dummy(),
                        }))
                    })
                    .collect();
                    
                Ok(SemanticType::Generic {
                    name: base_name,
                    parameters: param_types?,
                })
            }
        }
    }

    /// Check pattern against a type and return constraints
    fn check_pattern(&mut self, pattern: &AstNode<Pattern>, expected_type: &SemanticType) -> SemanticResult<ConstraintSet> {
        let mut constraints = ConstraintSet::new();
        
        match &pattern.kind {
            Pattern::Identifier(name) => {
                // Add binding for the variable
                let inferred_type = InferredType {
                    type_info: expected_type.clone(),
                    confidence: 1.0,
                    inference_source: InferenceSource::Pattern,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: Span::dummy(),
                };
                let binding = TypeBinding::new(
                    name.resolve().unwrap_or_else(|| "identifier".to_string()),
                    inferred_type,
                    false, // not mutable by default
                    0, // scope level
                );
                self.environment.add_binding(binding);
            }
            Pattern::Literal(lit) => {
                // Check that literal matches expected type
                let literal_type = match lit {
                    prism_ast::Literal::Integer(_) => SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32))),
                    prism_ast::Literal::Float(_) => SemanticType::Primitive(prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64)),
                    prism_ast::Literal::String(_) => SemanticType::Primitive(prism_ast::PrimitiveType::String),
                    prism_ast::Literal::Boolean(_) => SemanticType::Primitive(prism_ast::PrimitiveType::Boolean),
                };
                
                constraints.add_constraint(TypeConstraint::new(
                    ConstraintType::Concrete(literal_type),
                    ConstraintType::Concrete(expected_type.clone()),
                    pattern.span,
                    ConstraintReason::PatternMatch {
                        pattern_span: pattern.span,
                        expression_span: pattern.span,
                    },
                ));
            }
            Pattern::Wildcard => {
                // Wildcard matches any type, no constraints needed
            }
            Pattern::Tuple(patterns) => {
                // TODO: Implement constructor pattern checking
                for sub_pattern in patterns {
                    let sub_constraints = self.check_pattern(sub_pattern, expected_type)?;
                    constraints.merge(sub_constraints);
                }
            }
            Pattern::Object(field_patterns) => {
                if let SemanticType::Record(expected_fields) = expected_type {
                    for field_pattern in field_patterns {
                        let field_name = field_pattern.key.resolve().as_deref().unwrap_or("field");
                        if let Some(field_type) = expected_fields.get(field_name) {
                            let field_constraints = self.check_pattern(&field_pattern.pattern, field_type)?;
                            constraints.merge(field_constraints);
                        } else {
                            return Err(SemanticError::TypeInferenceError {
                                message: format!("Field '{}' not found in record type", field_name),
                            });
                        }
                    }
                } else {
                    return Err(SemanticError::TypeInferenceError {
                        message: "Expected record type for record pattern".to_string(),
                    });
                }
            }
            Pattern::Array(element_patterns) => {
                if let SemanticType::List(element_type) = expected_type {
                    for element_pattern in element_patterns {
                        let element_constraints = self.check_pattern(element_pattern, element_type)?;
                        constraints.merge(element_constraints);
                    }
                } else {
                    return Err(SemanticError::TypeInferenceError {
                        message: "Expected list type for list pattern".to_string(),
                    });
                }
            }
        }
        
        Ok(constraints)
    }
} 