//! VM Semantics Integration
//!
//! This module provides seamless integration between the Prism VM and the prism-semantic crate,
//! ensuring semantic information is properly preserved and accessible at runtime without
//! duplicating logic from the semantic analysis system.
//!
//! ## Design Principles
//!
//! 1. **No Logic Duplication**: Uses prism-semantic interfaces, doesn't reimplement
//! 2. **Separation of Concerns**: Only handles VM-specific semantic integration
//! 3. **Conceptual Cohesion**: Focused on VM semantic preservation and execution
//! 4. **Proper Interfaces**: Uses existing prism-semantic APIs and types

pub mod compiler_integration;
pub mod runtime_integration;
pub mod debug_support;
pub mod jit_integration;

use prism_semantic::{
    SemanticEngine, SemanticConfig, SemanticResult, SemanticError,
    types::{SemanticType, BusinessRule, TypeConstraint},
    database::{SemanticDatabase, SemanticInfo},
    analyzer::AnalysisResult,
    type_inference::engine::pir_integration::{PIRMetadata, PIRSemanticType},
    validation::{ValidationResult, ValidationError},
};
use prism_vm::bytecode::{
    BytecodeSemanticMetadata, CompiledBusinessRule, CompiledValidationPredicate,
    ValidationConfig, SemanticInformationRegistry,
};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// VM-specific semantic integration coordinator
/// 
/// This struct coordinates between prism-semantic and prism-vm systems,
/// ensuring semantic information flows properly through compilation and runtime.
#[derive(Debug)]
pub struct VMSemanticIntegrator {
    /// Reference to the semantic engine
    semantic_engine: Arc<SemanticEngine>,
    /// VM-specific semantic configuration
    vm_config: VMSemanticConfig,
    /// Semantic-to-bytecode compiler
    bytecode_compiler: compiler_integration::SemanticBytecodeCompiler,
    /// Runtime semantic validator
    runtime_validator: validator::RuntimeSemanticValidator,
    /// Debug support system
    debug_support: debug_support::SemanticDebugSupport,
}

/// VM-specific semantic configuration
#[derive(Debug, Clone)]
pub struct VMSemanticConfig {
    /// Enable semantic preservation in bytecode
    pub preserve_semantics_in_bytecode: bool,
    /// Enable runtime business rule validation
    pub enable_runtime_validation: bool,
    /// Enable semantic debugging features
    pub enable_semantic_debugging: bool,
    /// Enable JIT semantic optimizations
    pub enable_jit_semantic_optimizations: bool,
    /// Maximum semantic metadata size per function
    pub max_metadata_size_bytes: usize,
    /// Enable semantic introspection at runtime
    pub enable_runtime_introspection: bool,
}

/// Result of VM semantic integration
#[derive(Debug)]
pub struct VMSemanticIntegrationResult {
    /// Enhanced bytecode with semantic metadata
    pub enhanced_bytecode: prism_vm::PrismBytecode,
    /// Semantic information registry for runtime access
    pub semantic_registry: SemanticInformationRegistry,
    /// PIR metadata for further processing
    pub pir_metadata: Option<PIRMetadata>,
    /// Integration statistics
    pub integration_stats: IntegrationStatistics,
}

/// Statistics about semantic integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStatistics {
    /// Number of semantic types preserved
    pub semantic_types_preserved: usize,
    /// Number of business rules compiled
    pub business_rules_compiled: usize,
    /// Number of validation predicates compiled
    pub validation_predicates_compiled: usize,
    /// Semantic metadata size in bytes
    pub metadata_size_bytes: usize,
    /// Integration time in milliseconds
    pub integration_time_ms: u64,
    /// Memory overhead percentage
    pub memory_overhead_percent: f64,
}

impl VMSemanticIntegrator {
    /// Create a new VM semantic integrator
    pub fn new(semantic_config: SemanticConfig, vm_config: VMSemanticConfig) -> SemanticResult<Self> {
        let semantic_engine = Arc::new(SemanticEngine::new(semantic_config)?);
        
        let bytecode_compiler = compiler_integration::SemanticBytecodeCompiler::new(
            Arc::clone(&semantic_engine),
            vm_config.clone(),
        )?;
        
        let runtime_validator = validator::RuntimeSemanticValidator::new(
            Arc::clone(&semantic_engine),
            vm_config.clone(),
        )?;
        
        let debug_support = debug_support::SemanticDebugSupport::new(
            Arc::clone(&semantic_engine),
            vm_config.clone(),
        )?;

        Ok(Self {
            semantic_engine,
            vm_config,
            bytecode_compiler,
            runtime_validator,
            debug_support,
        })
    }

    /// Integrate semantic information into VM bytecode compilation
    pub fn integrate_semantics_into_bytecode(
        &mut self,
        program: &prism_ast::Program,
        base_bytecode: prism_vm::PrismBytecode,
    ) -> SemanticResult<VMSemanticIntegrationResult> {
        let start_time = std::time::Instant::now();

        // Step 1: Perform comprehensive semantic analysis using prism-semantic
        let semantic_info = self.semantic_engine.analyze_program(program)?;

        // Step 2: Generate PIR metadata for semantic preservation
        let pir_metadata = if self.vm_config.preserve_semantics_in_bytecode {
            Some(self.extract_pir_metadata(&semantic_info)?)
        } else {
            None
        };

        // Step 3: Compile semantic information into bytecode-compatible format
        let (enhanced_bytecode, semantic_registry) = self.bytecode_compiler
            .compile_semantic_info_to_bytecode(base_bytecode, &semantic_info, pir_metadata.as_ref())?;

        // Step 4: Set up runtime validation if enabled
        if self.vm_config.enable_runtime_validation {
            self.runtime_validator.prepare_runtime_validation(&semantic_info, &semantic_registry)?;
        }

        // Step 5: Set up debug support if enabled
        if self.vm_config.enable_semantic_debugging {
            self.debug_support.prepare_debug_support(&semantic_info, &semantic_registry)?;
        }

        let integration_time = start_time.elapsed();
        let integration_stats = self.calculate_integration_statistics(
            &semantic_info,
            &semantic_registry,
            integration_time,
        );

        Ok(VMSemanticIntegrationResult {
            enhanced_bytecode,
            semantic_registry,
            pir_metadata,
            integration_stats,
        })
    }

    /// Extract PIR metadata from semantic information
    fn extract_pir_metadata(&self, semantic_info: &SemanticInfo) -> SemanticResult<PIRMetadata> {
        // Use prism-semantic's PIR integration capabilities
        let pir_generator = prism_semantic::type_inference::engine::pir_integration::PIRMetadataGenerator::new();
        
        // Convert semantic info to format expected by PIR generator
        let type_inference_result = self.convert_semantic_info_to_inference_result(semantic_info)?;
        let inference_stats = self.create_inference_statistics(semantic_info);
        
        pir_generator.generate_pir_metadata(&type_inference_result, &inference_stats)
    }

    /// Convert SemanticInfo to TypeInferenceResult for PIR generation
    fn convert_semantic_info_to_inference_result(
        &self,
        semantic_info: &SemanticInfo,
    ) -> SemanticResult<prism_semantic::type_inference::TypeInferenceResult> {
        use prism_semantic::type_inference::{TypeInferenceResult, InferredType, InferenceSource};
        use prism_semantic::type_inference::environment::TypeEnvironment;
        use prism_semantic::type_inference::constraints::ConstraintSet;
        use prism_semantic::type_inference::unification::Substitution;

        let mut node_types = HashMap::new();
        
        // Convert type information to inferred types
        for (node_id, type_info) in &semantic_info.types {
            let inferred_type = InferredType {
                type_info: semantic_info.symbols.get(&Symbol::from_str(&type_info.ai_description.clone().unwrap_or_default()))
                    .and_then(|symbol_info| {
                        // Try to extract semantic type from symbol info
                        // This is a simplified conversion - in practice, we'd have better type mapping
                        Some(prism_semantic::types::SemanticType::primitive(
                            &symbol_info.name,
                            prism_ast::PrimitiveType::String, // Simplified
                            symbol_info.location,
                        ))
                    })
                    .unwrap_or_else(|| {
                        prism_semantic::types::SemanticType::primitive(
                            "unknown",
                            prism_ast::PrimitiveType::String,
                            type_info.location,
                        )
                    }),
                confidence: 0.8, // Default confidence
                inference_source: InferenceSource::Context,
                constraints: Vec::new(), // Would extract from semantic info
                ai_metadata: None, // Could be populated from semantic info
                span: type_info.location,
            };
            
            node_types.insert(*node_id, inferred_type);
        }

        Ok(TypeInferenceResult {
            node_types,
            global_env: TypeEnvironment::new(),
            constraints: ConstraintSet::new(),
            substitution: Substitution::empty(),
            errors: Vec::new(),
            ai_metadata: None,
        })
    }

    /// Create inference statistics from semantic info
    fn create_inference_statistics(
        &self,
        semantic_info: &SemanticInfo,
    ) -> prism_semantic::type_inference::InferenceStatistics {
        use prism_semantic::type_inference::{InferenceStatistics, MemoryStatistics};

        InferenceStatistics {
            nodes_processed: semantic_info.types.len(),
            type_vars_generated: 0, // Not available from semantic info
            constraints_generated: 0, // Not available from semantic info
            unification_steps: 0, // Not available from semantic info
            inference_time_ms: semantic_info.analysis_metadata.duration_ms,
            constraint_solving_time_ms: 0, // Not available from semantic info
            memory_stats: MemoryStatistics {
                peak_memory_bytes: 0, // Not available from semantic info
                environments_created: 1,
                substitutions_created: 0,
            },
        }
    }

    /// Calculate integration statistics
    fn calculate_integration_statistics(
        &self,
        semantic_info: &SemanticInfo,
        semantic_registry: &SemanticInformationRegistry,
        integration_time: std::time::Duration,
    ) -> IntegrationStatistics {
        let business_rules_count = semantic_info.symbols.values()
            .filter(|symbol| symbol.business_context.is_some())
            .count();

        let validation_predicates_count = semantic_info.validation_result
            .as_ref()
            .map(|vr| vr.errors.len() + vr.warnings.len())
            .unwrap_or(0);

        // Estimate metadata size (simplified calculation)
        let estimated_metadata_size = semantic_info.symbols.len() * 200 + // Rough estimate per symbol
            semantic_info.types.len() * 150 + // Rough estimate per type
            business_rules_count * 300; // Rough estimate per business rule

        IntegrationStatistics {
            semantic_types_preserved: semantic_info.types.len(),
            business_rules_compiled: business_rules_count,
            validation_predicates_compiled: validation_predicates_count,
            metadata_size_bytes: estimated_metadata_size,
            integration_time_ms: integration_time.as_millis() as u64,
            memory_overhead_percent: (estimated_metadata_size as f64 / (1024.0 * 1024.0)) * 100.0, // Convert to MB percentage
        }
    }

    /// Get reference to semantic engine for external use
    pub fn semantic_engine(&self) -> &SemanticEngine {
        &self.semantic_engine
    }

    /// Get VM semantic configuration
    pub fn vm_config(&self) -> &VMSemanticConfig {
        &self.vm_config
    }

    /// Update VM semantic configuration
    pub fn update_vm_config(&mut self, new_config: VMSemanticConfig) -> SemanticResult<()> {
        self.vm_config = new_config.clone();
        
        // Update sub-components with new configuration
        self.bytecode_compiler.update_config(new_config.clone())?;
        self.runtime_validator.update_config(new_config.clone())?;
        self.debug_support.update_config(new_config)?;
        
        Ok(())
    }
}

impl Default for VMSemanticConfig {
    fn default() -> Self {
        Self {
            preserve_semantics_in_bytecode: true,
            enable_runtime_validation: true,
            enable_semantic_debugging: false, // Disabled by default for performance
            enable_jit_semantic_optimizations: true,
            max_metadata_size_bytes: 10 * 1024 * 1024, // 10MB limit
            enable_runtime_introspection: false, // Disabled by default for security
        }
    }
}

// Helper trait for Symbol creation (simplified)
trait SymbolFromStr {
    fn from_str(s: &str) -> Self;
}

impl SymbolFromStr for Symbol {
    fn from_str(s: &str) -> Self {
        // Simplified symbol creation - in practice, this would use proper symbol resolution
        Symbol::new(s.to_string())
    }
}

// Re-export JIT integration types
pub use jit_integration::{
    SemanticJITOptimizer, SemanticJITConfig, SemanticOptimizationPlan,
    SemanticOptimization, AppliedOptimizations, SemanticJITStats,
}; 