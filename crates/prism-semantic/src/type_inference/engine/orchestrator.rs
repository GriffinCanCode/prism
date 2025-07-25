//! Type Inference Orchestrator
//!
//! This module implements the high-level orchestration of the type inference process.
//! It coordinates between different inference engines and manages the overall workflow.
//!
//! **Single Responsibility**: Orchestrate the type inference workflow
//! **What it does**: Coordinate inference phases, manage dependencies, handle errors
//! **What it doesn't do**: Perform actual inference, manage performance, handle compiler integration

use super::{
    InferenceEngine, CompilerIntegratable, PIRIntegratable, EffectAware,
    ExpressionInferenceEngine, StatementInferenceEngine, PatternInferenceEngine,
    BuiltinTypeManager, InferenceProfiler, CompilerIntegration, PIRMetadataGenerator,
    EffectInferenceEngine, ASTTypeResolver,
};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        TypeInferenceResult, InferredType, InferenceSource, InferenceStatistics,
        constraints::{ConstraintSolver, ConstraintSet},
        errors::{TypeError, TypeErrorKind},
        unification::Substitution,
        environment::TypeEnvironment,
        metadata::GlobalInferenceMetadata,
    },
};
use prism_ast::{Program, Expr, Stmt, Pattern, Type as AstType, Item};
use prism_common::{NodeId, Span};
use std::collections::HashMap;
use std::time::Instant;

/// High-level orchestrator for type inference workflow
#[derive(Debug)]
pub struct InferenceOrchestrator {
    /// Expression inference engine
    expression_engine: ExpressionInferenceEngine,
    /// Statement inference engine  
    statement_engine: StatementInferenceEngine,
    /// Pattern inference engine
    pattern_engine: PatternInferenceEngine,
    /// Built-in type manager
    builtin_manager: BuiltinTypeManager,
    /// Constraint solver
    constraint_solver: ConstraintSolver,
    /// Performance profiler
    profiler: InferenceProfiler,
    /// Compiler integration
    compiler_integration: CompilerIntegration,
    /// PIR metadata generator
    pir_generator: PIRMetadataGenerator,
    /// Effect inference engine
    effect_engine: EffectInferenceEngine,
    /// AST type resolver
    ast_resolver: ASTTypeResolver,
    /// Global type environment
    global_env: TypeEnvironment,
    /// Inference statistics
    statistics: InferenceStatistics,
}

impl InferenceOrchestrator {
    /// Create a new inference orchestrator
    pub fn new() -> SemanticResult<Self> {
        Ok(Self {
            expression_engine: ExpressionInferenceEngine::new()?,
            statement_engine: StatementInferenceEngine::new()?,
            pattern_engine: PatternInferenceEngine::new()?,
            builtin_manager: BuiltinTypeManager::new()?,
            constraint_solver: ConstraintSolver::new(),
            profiler: InferenceProfiler::new(),
            compiler_integration: CompilerIntegration::new(),
            pir_generator: PIRMetadataGenerator::new(),
            effect_engine: EffectInferenceEngine::new()?,
            ast_resolver: ASTTypeResolver::new(),
            global_env: TypeEnvironment::new(),
            statistics: InferenceStatistics::default(),
        })
    }

    /// Orchestrate type inference for an entire program
    pub fn infer_program(&mut self, program: &Program) -> SemanticResult<TypeInferenceResult> {
        let start_time = Instant::now();
        self.profiler.start_program_inference();
        
        // Phase 1: Initialize built-in types
        let mut result = TypeInferenceResult::empty();
        self.builtin_manager.initialize_builtins(&mut result)?;
        
        // Phase 2: Process top-level items
        for item_node in &program.items {
            let item_result = self.process_program_item(item_node)?;
            result.merge(item_result);
            
            // Check resource limits
            if let Err(e) = self.check_resource_limits() {
                result.add_error(TypeError::Custom {
                    message: e.to_string(),
                    location: prism_common::Span::dummy(),
                    kind: TypeErrorKind::Internal,
                });
                break;
            }
        }
        
        // Phase 3: Solve constraints
        let substitution = self.constraint_solver.solve(&result.constraints)?;
        result.substitution = substitution.clone();
        
        // Phase 4: Apply substitution
        self.apply_substitution_to_result(&mut result, &substitution)?;
        
        // Phase 5: Generate metadata
        result.ai_metadata = Some(self.generate_global_metadata(&result)?);
        
        // Phase 6: Report performance
        let elapsed = start_time.elapsed();
        self.profiler.end_program_inference(elapsed);
        self.compiler_integration.report_program_inference(&self.statistics, elapsed);
        
        Ok(result)
    }

    /// Process a single program item
    fn process_program_item(&mut self, item_node: &prism_ast::AstNode<Item>) -> SemanticResult<TypeInferenceResult> {
        match &item_node.kind {
            Item::Statement(stmt) => {
                self.statement_engine.infer_statement(stmt)
            }
            Item::Function(func_decl) => {
                self.statement_engine.infer_function_declaration(func_decl)
            }
            Item::Type(type_decl) => {
                self.statement_engine.infer_type_declaration(type_decl)
            }
            Item::Const(const_decl) => {
                self.statement_engine.infer_const_declaration(const_decl)
            }
            Item::Variable(var_decl) => {
                self.statement_engine.infer_variable_declaration(var_decl)
            }
            Item::Module(module_decl) => {
                self.statement_engine.infer_module_declaration(module_decl)
            }
            _ => {
                // For other item types, return empty result
                Ok(TypeInferenceResult::empty())
            }
        }
    }

    /// Check resource usage limits
    fn check_resource_limits(&self) -> SemanticResult<()> {
        if self.statistics.type_vars_generated > 10000 {
            return Err(SemanticError::TypeInferenceError {
                message: format!("Too many type variables generated: {}", 
                               self.statistics.type_vars_generated),
            });
        }
        
        if self.statistics.constraints_generated > 50000 {
            return Err(SemanticError::TypeInferenceError {
                message: format!("Too many constraints generated: {}", 
                               self.statistics.constraints_generated),
            });
        }
        
        Ok(())
    }

    /// Apply substitution to all types in the result
    fn apply_substitution_to_result(
        &mut self, 
        result: &mut TypeInferenceResult, 
        substitution: &Substitution
    ) -> SemanticResult<()> {
        // Apply to all node types
        for (_, inferred_type) in result.node_types.iter_mut() {
            inferred_type.type_info = substitution.apply_to_semantic_type(&inferred_type.type_info)?;
        }
        
        // Apply to global environment
        result.global_env.apply_substitution(substitution)?;
        
        Ok(())
    }

    /// Generate global metadata for the inference result
    fn generate_global_metadata(&self, result: &TypeInferenceResult) -> SemanticResult<GlobalInferenceMetadata> {
        self.pir_generator.generate_global_metadata(result, &self.statistics)
    }

    /// Get current inference statistics
    pub fn get_statistics(&self) -> &InferenceStatistics {
        &self.statistics
    }

    /// Reset orchestrator state
    pub fn reset(&mut self) {
        self.expression_engine.reset();
        self.statement_engine.reset();
        self.pattern_engine.reset();
        self.builtin_manager.reset();
        self.profiler.reset();
        self.statistics = InferenceStatistics::default();
        self.global_env = TypeEnvironment::new();
    }

    /// Get reference to expression engine for direct access
    pub fn expression_engine(&mut self) -> &mut ExpressionInferenceEngine {
        &mut self.expression_engine
    }

    /// Get reference to statement engine for direct access
    pub fn statement_engine(&mut self) -> &mut StatementInferenceEngine {
        &mut self.statement_engine
    }

    /// Get reference to pattern engine for direct access
    pub fn pattern_engine(&mut self) -> &mut PatternInferenceEngine {
        &mut self.pattern_engine
    }

    /// Get reference to global environment
    pub fn global_environment(&self) -> &TypeEnvironment {
        &self.global_env
    }

    /// Get mutable reference to global environment
    pub fn global_environment_mut(&mut self) -> &mut TypeEnvironment {
        &mut self.global_env
    }

    /// Get reference to built-in manager
    pub fn builtin_manager(&self) -> &BuiltinTypeManager {
        &self.builtin_manager
    }

    /// Get mutable reference to built-in manager
    pub fn builtin_manager_mut(&mut self) -> &mut BuiltinTypeManager {
        &mut self.builtin_manager
    }

    /// Get reference to profiler
    pub fn profiler(&self) -> &InferenceProfiler {
        &self.profiler
    }

    /// Get mutable reference to profiler
    pub fn profiler_mut(&mut self) -> &mut InferenceProfiler {
        &mut self.profiler
    }

    /// Get reference to compiler integration
    pub fn compiler_integration(&self) -> &CompilerIntegration {
        &self.compiler_integration
    }

    /// Get mutable reference to compiler integration
    pub fn compiler_integration_mut(&mut self) -> &mut CompilerIntegration {
        &mut self.compiler_integration
    }

    /// Get reference to PIR generator
    pub fn pir_generator(&self) -> &PIRMetadataGenerator {
        &self.pir_generator
    }

    /// Get mutable reference to PIR generator
    pub fn pir_generator_mut(&mut self) -> &mut PIRMetadataGenerator {
        &mut self.pir_generator
    }
}

impl Default for InferenceOrchestrator {
    fn default() -> Self {
        Self::new().expect("Failed to create default InferenceOrchestrator")
    }
}

/// Orchestration configuration
#[derive(Debug, Clone)]
pub struct OrchestrationConfig {
    /// Maximum type variables allowed
    pub max_type_variables: usize,
    /// Maximum constraints allowed
    pub max_constraints: usize,
    /// Enable parallel processing of items
    pub enable_parallel_processing: bool,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Enable compiler integration
    pub enable_compiler_integration: bool,
    /// Enable PIR metadata generation
    pub enable_pir_metadata: bool,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            max_type_variables: 10000,
            max_constraints: 50000,
            enable_parallel_processing: false, // Disabled by default for safety
            enable_profiling: true,
            enable_compiler_integration: true,
            enable_pir_metadata: true,
        }
    }
}

/// Result of orchestrated inference
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    /// Main inference result
    pub inference_result: TypeInferenceResult,
    /// Performance metrics
    pub performance_metrics: crate::type_inference::engine::profiling::PerformanceMetrics,
    /// PIR metadata if generated
    pub pir_metadata: Option<crate::type_inference::engine::pir_integration::PIRMetadata>,
    /// Orchestration statistics
    pub orchestration_stats: OrchestrationStats,
}

/// Statistics about the orchestration process
#[derive(Debug, Clone)]
pub struct OrchestrationStats {
    /// Number of items processed
    pub items_processed: usize,
    /// Number of phases completed
    pub phases_completed: usize,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Error recovery attempts
    pub error_recovery_attempts: usize,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Peak memory usage (estimated)
    pub peak_memory_bytes: usize,
    /// Total processing time
    pub total_time_ms: u64,
    /// Type variables created
    pub type_vars_created: usize,
    /// Constraints generated
    pub constraints_generated: usize,
} 