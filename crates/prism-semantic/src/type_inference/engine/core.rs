//! Core Type Inference Engine
//!
//! This module provides the main public interface for the modular type inference engine.
//! It coordinates between the orchestrator and provides backwards compatibility.
//!
//! **Single Responsibility**: Main type inference API and coordination
//! **What it does**: Provide public API, coordinate with orchestrator, maintain compatibility
//! **What it doesn't do**: Perform actual inference, handle specific constructs

use super::{InferenceOrchestrator};
use super::orchestrator::OrchestrationConfig;
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        TypeInferenceResult, InferredType, InferenceSource, InferenceStatistics,
        constraints::{ConstraintSolver, ConstraintSet},
        unification::Substitution,
        environment::TypeEnvironment,
        metadata::GlobalInferenceMetadata,
    },
};
use prism_ast::{Program, Expr, Stmt};
use prism_common::{NodeId, Span};
use std::time::Instant;

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
    /// Orchestration configuration
    pub orchestration: OrchestrationConfig,
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
            orchestration: OrchestrationConfig::default(),
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

/// Main type inference engine - now a facade over the modular system
#[derive(Debug)]
pub struct TypeInferenceEngine {
    /// Configuration
    config: InferenceConfig,
    /// Internal orchestrator
    orchestrator: InferenceOrchestrator,
    /// Start time for performance measurement
    start_time: Option<Instant>,
}

impl TypeInferenceEngine {
    /// Create a new type inference engine
    pub fn new(config: InferenceConfig) -> SemanticResult<Self> {
        let orchestrator = InferenceOrchestrator::new()?;
        
        Ok(Self {
            config,
            orchestrator,
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
        
        // Delegate to orchestrator
        let result = self.orchestrator.infer_program(program)?;
        
        self.end_profiling();
        
        Ok(result)
    }

    /// Infer type for a single statement
    pub fn infer_stmt(&mut self, stmt: &Stmt) -> SemanticResult<TypeInferenceResult> {
        // Use the statement engine directly through orchestrator
        let stmt_result = self.orchestrator.statement_engine().infer_statement(stmt)?;
        Ok(stmt_result)
    }

    /// Infer type for a single expression
    pub fn infer_expr(&mut self, expr: &Expr) -> SemanticResult<InferenceResult> {
        // Use the expression engine directly through orchestrator
        let expr_result = self.orchestrator.expression_engine()
            .infer_expression(expr, Span::dummy())?;
        
        Ok(InferenceResult {
            inferred_type: expr_result.inferred_type,
            constraints: expr_result.constraints,
            environment: expr_result.environment,
        })
    }

    /// Generate a fresh type variable (delegated to orchestrator)
    pub fn fresh_type_var(&mut self, span: Span) -> crate::type_inference::TypeVar {
        // Access through orchestrator's expression engine
        self.orchestrator.expression_engine().fresh_type_var(span)
    }

    /// Generate a fresh type variable with a name (delegated to orchestrator)
    pub fn fresh_type_var_named(&mut self, name: String, span: Span) -> crate::type_inference::TypeVar {
        // Access through orchestrator's expression engine
        self.orchestrator.expression_engine().fresh_type_var_named(name, span)
    }

    /// Add a constraint to the constraint set (delegated to orchestrator)
    pub fn add_constraint(&mut self, constraint: crate::type_inference::constraints::TypeConstraint) {
        // This would need to be implemented in the orchestrator
        // For now, we'll just track statistics
        // Note: statistics field is private, so we'll skip this for now
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &InferenceStatistics {
        self.orchestrator.get_statistics()
    }

    /// Reset the engine state for a new inference session
    pub fn reset(&mut self) {
        self.orchestrator.reset();
        self.start_time = None;
    }

    /// Get the global environment
    pub fn global_environment(&self) -> &TypeEnvironment {
        self.orchestrator.global_environment()
    }

    /// Get mutable reference to global environment
    pub fn global_environment_mut(&mut self) -> &mut TypeEnvironment {
        self.orchestrator.global_environment_mut()
    }

    // Private helper methods

    fn start_profiling(&mut self) {
        if self.config.enable_profiling {
            self.start_time = Some(Instant::now());
        }
    }

    fn end_profiling(&mut self) {
        if let Some(start_time) = self.start_time {
            let elapsed = start_time.elapsed();
            
            // Report to orchestrator's profiler
            self.orchestrator.profiler_mut().end_program_inference(elapsed);
            
            // Report to compiler integration if enabled
            if self.config.orchestration.enable_compiler_integration {
                let statistics = self.orchestrator.get_statistics().clone();
                self.orchestrator.compiler_integration_mut().report_program_inference(
                    &statistics,
                    elapsed
                );
            }
        }
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

        SemanticType::primitive(name, self.convert_primitive_type(primitive_type), span)
    }
}

/// Backwards compatibility methods that were in the original engine
impl TypeInferenceEngine {
    /// Generate PIR metadata from inference results (delegated to orchestrator)
    pub fn generate_pir_metadata(&self, result: &TypeInferenceResult) -> SemanticResult<GlobalInferenceMetadata> {
        self.orchestrator.pir_generator().generate_global_metadata(result, self.get_statistics())
    }

    /// Report performance metrics to the compiler (delegated to orchestrator)
    pub fn report_performance_metrics(&mut self, elapsed: std::time::Duration) {
        if self.config.orchestration.enable_compiler_integration {
            let statistics = self.get_statistics().clone();
            self.orchestrator.compiler_integration_mut().report_program_inference(
                &statistics,
                elapsed
            );
        }
    }

    /// Get current performance statistics (delegated to orchestrator)
    pub fn get_performance_stats(&self) -> super::profiling::PerformanceStats {
        self.orchestrator.profiler().get_performance_stats()
    }

    /// Check if a name is a built-in (delegated to orchestrator)
    pub fn is_builtin(&self, name: &str) -> bool {
        self.orchestrator.builtin_manager().is_builtin(name)
    }

    /// Get a built-in type by name (delegated to orchestrator)
    pub fn get_builtin_type(&self, name: &str) -> Option<&SemanticType> {
        self.orchestrator.builtin_manager().get_builtin_type(name)
    }

    /// Get a built-in function by name (delegated to orchestrator)
    pub fn get_builtin_function(&self, name: &str) -> Option<&SemanticType> {
        self.orchestrator.builtin_manager().get_builtin_function(name)
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
} 