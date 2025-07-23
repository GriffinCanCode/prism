//! Core Prism VM Backend Implementation
//!
//! This module provides the main Prism VM backend implementation that integrates
//! all the modular components (compiler, optimization, validation).

use super::{PrismVMBackendConfig, VMBackendResult, VMBackendError};
use super::compiler::PIRToBytecodeCompiler;
use super::optimization::BytecodeOptimizer;
use super::validation::BytecodeValidator;

use crate::backends::{
    CompilationContext, CompilationTarget, CodeGenBackend, CodeArtifact, 
    CodeGenConfig, CodeGenStats, BackendCapabilities, AIMetadataLevel, AIMetadata,
    PrismIR, PIRModule, PIRFunction, PIRSemanticType,
};
use crate::CodeGenResult;
use async_trait::async_trait;
use prism_ast::Program;
use prism_vm::{PrismBytecode, bytecode::serialization::utils};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, span, Level};

/// Prism VM backend with modular architecture
#[derive(Debug)]
pub struct PrismVMBackend {
    /// Backend configuration
    config: PrismVMBackendConfig,
    /// PIR to bytecode compiler
    compiler: PIRToBytecodeCompiler,
    /// Bytecode optimizer
    optimizer: BytecodeOptimizer,
    /// Bytecode validator
    validator: BytecodeValidator,
}

impl PrismVMBackend {
    /// Create new Prism VM backend with configuration
    pub fn new(config: CodeGenConfig) -> VMBackendResult<Self> {
        let vm_config = PrismVMBackendConfig::from_codegen_config(&config);
        Self::with_vm_config(vm_config)
    }

    /// Create new Prism VM backend with VM-specific configuration
    pub fn with_vm_config(config: PrismVMBackendConfig) -> VMBackendResult<Self> {
        let compiler = PIRToBytecodeCompiler::new(config.compiler_config.clone())?;
        let optimizer = BytecodeOptimizer::new(config.optimization_config.clone());
        let validator = BytecodeValidator::new(config.validation_config.clone());

        Ok(Self {
            config,
            compiler,
            optimizer,
            validator,
        })
    }

    /// Generate bytecode from PIR
    async fn generate_bytecode(&mut self, pir: &PrismIR) -> VMBackendResult<PrismBytecode> {
        let _span = span!(Level::INFO, "generate_bytecode").entered();
        let start_time = Instant::now();

        info!("Generating Prism bytecode from PIR");

        // Compile PIR to bytecode
        let mut bytecode = self.compiler.compile_pir(pir)
            .map_err(|e| VMBackendError::CompilationError {
                message: format!("PIR compilation failed: {}", e),
            })?;

        let compilation_time = start_time.elapsed();
        debug!("PIR compilation completed in {:?}", compilation_time);

        // Apply optimizations if enabled
        if self.config.optimization_level > 0 {
            let opt_start = Instant::now();
            bytecode = self.optimizer.optimize(bytecode)
                .map_err(|e| VMBackendError::OptimizationError {
                    message: format!("Bytecode optimization failed: {}", e),
                })?;
            let opt_time = opt_start.elapsed();
            debug!("Bytecode optimization completed in {:?}", opt_time);
        }

        // Validate bytecode if enabled
        if self.config.validate_bytecode {
            let val_start = Instant::now();
            self.validator.validate(&bytecode)
                .map_err(|e| VMBackendError::ValidationError {
                    message: format!("Bytecode validation failed: {}", e),
                })?;
            let val_time = val_start.elapsed();
            debug!("Bytecode validation completed in {:?}", val_time);
        }

        let total_time = start_time.elapsed();
        info!("Bytecode generation completed in {:?}", total_time);

        Ok(bytecode)
    }

    /// Generate AI metadata for bytecode
    fn generate_ai_metadata(&self, pir: &PrismIR, bytecode: &PrismBytecode) -> AIMetadata {
        let mut metadata = AIMetadata {
            intents: std::collections::HashMap::new(),
            examples: Vec::new(),
            patterns: Vec::new(),
            performance_hints: Vec::new(),
        };

        // Extract semantic information
        for (name, _semantic_type) in &pir.type_registry.types {
            metadata.intents.insert(
                format!("type_{}", name),
                format!("Semantic type: {}", name),
            );
        }

        // Extract business domains
        for module in &pir.modules {
            metadata.intents.insert(
                format!("module_{}", module.name),
                module.business_context.domain.clone(),
            );
        }

        // Extract performance hints
        for function in &bytecode.functions {
            for characteristic in &function.performance_characteristics {
                metadata.performance_hints.push(format!(
                    "Function {}: {}",
                    function.name, characteristic
                ));
            }
        }

        // Extract patterns from capabilities
        for function in &bytecode.functions {
            if !function.capabilities.is_empty() {
                metadata.patterns.push(format!(
                    "Function {} requires capabilities: {:?}",
                    function.name, function.capabilities
                ));
            }
        }

        metadata
    }

    /// Get backend capabilities
    fn get_backend_capabilities() -> BackendCapabilities {
        BackendCapabilities {
            source_maps: false, // VM doesn't generate source maps
            debug_info: true,
            incremental: false, // Not yet implemented
            parallel: false,    // Not yet implemented
            optimization_levels: vec![0, 1, 2, 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{PrismIR, PIRModule};
    use prism_pir::{SemanticTypeRegistry, business::BusinessContext};

    #[tokio::test]
    async fn test_basic_pir_compilation() {
        // Create a simple PIR for testing using the correct structure
        let pir = PrismIR {
            modules: vec![PIRModule {
                name: "test_module".to_string(),
                capability: "testing".to_string(),
                sections: Vec::new(),
                dependencies: Vec::new(),
                business_context: BusinessContext {
                    domain: "Testing".to_string(),
                    entities: Vec::new(),
                    relationships: Vec::new(),
                    constraints: Vec::new(),
                },
                smart_module_metadata: prism_pir::semantic::SmartModuleMetadata {
                    purpose: "Test module".to_string(),
                    patterns: Vec::new(),
                    quality_score: 0.8,
                    dependency_analysis: prism_pir::semantic::DependencyAnalysis {
                        incoming: Vec::new(),
                        outgoing: Vec::new(),
                        circular: Vec::new(),
                    },
                },
                domain_rules: Vec::new(),
                effects: Vec::new(),
                capabilities: Vec::new(),
                performance_profile: prism_pir::quality::PerformanceProfile::default(),
                cohesion_score: 0.85,
            }],
            type_registry: SemanticTypeRegistry {
                types: std::collections::HashMap::new(),
                relationships: std::collections::HashMap::new(),
                global_constraints: Vec::new(),
            },
            effect_graph: prism_pir::semantic::EffectGraph {
                nodes: std::collections::HashMap::new(),
                edges: Vec::new(),
            },
            cohesion_metrics: prism_pir::semantic::CohesionMetrics {
                overall_score: 0.85,
                module_scores: std::collections::HashMap::new(),
                coupling_metrics: prism_pir::semantic::CouplingMetrics {
                    afferent: std::collections::HashMap::new(),
                    efferent: std::collections::HashMap::new(),
                    instability: std::collections::HashMap::new(),
                },
            },
            ai_metadata: prism_pir::ai_integration::AIMetadata::default(),
            metadata: prism_pir::semantic::PIRMetadata {
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                source_hash: 0,
                optimization_level: 0,
                target_platforms: Vec::new(),
            },
        };

        // Create backend
        let config = CodeGenConfig::default();
        let mut backend = PrismVMBackend::new(config).unwrap();

        // Test compilation
        let result = backend.generate_bytecode(&pir).await;
        assert!(result.is_ok(), "PIR compilation should succeed: {:?}", result);

        let bytecode = result.unwrap();
        assert_eq!(bytecode.metadata.name, "test_module");
        assert!(bytecode.functions.len() >= 1); // Should have at least main function
    }
}

#[async_trait]
impl CodeGenBackend for PrismVMBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::PrismVM
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let _span = span!(Level::INFO, "prism_vm_codegen").entered();
        let start_time = Instant::now();

        info!("Generating Prism VM bytecode from PIR");

        // Clone self to make it mutable for generation
        let mut backend = Self::with_vm_config(
            PrismVMBackendConfig::from_codegen_config(config)
        ).map_err(|e| VMBackendError::CompilationError {
            message: format!("Failed to create VM backend: {}", e),
        })?;

        // Generate bytecode
        let bytecode = backend.generate_bytecode(pir).await?;

        let generation_time = start_time.elapsed();

        // Serialize bytecode to bytes
        let bytecode_bytes = utils::serialize_bytecode(&bytecode)
            .map_err(|e| VMBackendError::BytecodeError {
                message: format!("Bytecode serialization failed: {}", e),
            })?;

        // Generate AI metadata
        let ai_metadata = if config.ai_metadata_level != AIMetadataLevel::None {
            backend.generate_ai_metadata(pir, &bytecode)
        } else {
            AIMetadata::default()
        };

        // Create code artifact
        let artifact = CodeArtifact {
            target: CompilationTarget::PrismVM,
            content: String::from_utf8_lossy(&bytecode_bytes).to_string(),
            source_map: None, // VM doesn't generate source maps
            ai_metadata,
            output_path: PathBuf::from("output.pvm"),
            stats: CodeGenStats {
                lines_generated: 0, // Bytecode doesn't have lines
                generation_time: generation_time.as_millis() as u64,
                optimizations_applied: if backend.config.optimization_level > 0 { 1 } else { 0 },
                memory_usage: bytecode_bytes.len(),
            },
        };

        info!("Prism VM bytecode generation completed successfully");
        Ok(artifact)
    }

    async fn generate_code(
        &self,
        program: &Program,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        // For now, we'll return an error since we need PIR
        // In a complete implementation, we'd convert Program to PIR first
        Err(crate::CodeGenError::CodeGenerationError {
            target: "PrismVM".to_string(),
            message: "PrismVM backend requires PIR input. Use generate_code_from_pir instead.".to_string(),
        })
    }

    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        // Generate bytecode representation of a semantic type
        Ok(format!("// Semantic type: {}\n// Domain: {}\n// Business rules: {:?}",
            semantic_type.name,
            semantic_type.domain,
            semantic_type.business_rules.len()
        ))
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        // Generate bytecode representation of a function with effects
        Ok(format!("// Function: {}\n// Effects: {:?}\n// Capabilities: {:?}",
            function.name,
            function.signature.effects,
            function.capabilities_required
        ))
    }

    async fn generate_validation_logic(
        &self,
        semantic_type: &PIRSemanticType,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        // Generate bytecode for validation logic
        Ok(format!("// Validation for: {}\n// Rules: {:?}",
            semantic_type.name,
            semantic_type.validation_predicates.len()
        ))
    }

    async fn generate_runtime_support(
        &self,
        pir: &PrismIR,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        // Generate runtime support code
        Ok(format!("// Runtime support for {} modules\n// Capabilities: effect system, GC, concurrency",
            pir.modules.len()
        ))
    }

    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        // Optimization is handled during bytecode generation
        // This is a no-op for now
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        // Validation is handled during bytecode generation
        // Return empty validation issues
        Ok(Vec::new())
    }

    fn capabilities(&self) -> BackendCapabilities {
        Self::get_backend_capabilities()
    }
} 