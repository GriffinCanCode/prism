//! WebAssembly Core Backend Implementation
//!
//! This module provides the main WebAssembly backend implementation that integrates
//! all the modular components (types, memory, strings, instructions, runtime, optimization, validation).

use super::{WasmResult, WasmError, WasmBackendConfig};
use super::types::{WasmType, WasmTypeConverter, WasmOptimizationLevel, WasmRuntimeTarget, WasmFeatures};
use super::string_handler::{StringConstantManager, StringManagerConfig};
use super::memory::{WasmMemoryManager, WasmMemoryLayout};
use super::instructions::WasmInstructionGenerator;
use super::runtime::{WasmRuntimeIntegration, WasmRuntimeConfig, WasmCapabilitySystem};
use super::optimization::{WasmOptimizer, WasmOptimizerConfig};
use super::validation::{WasmValidator, WasmValidatorConfig};

use crate::backends::{
    CompilationContext, CompilationTarget, CodeGenBackend, CodeArtifact, 
    CodeGenConfig, CodeGenStats, BackendCapabilities, AIMetadataLevel,
    PrismIR, PIRModule, PIRFunction, PIRSemanticType, PIRExpression,
    Effect, Capability,
};
use crate::CodeGenResult;
use async_trait::async_trait;
use prism_ast::Program;
use std::path::PathBuf;
use tracing::{debug, info, span, Level};

/// WebAssembly backend with modular architecture
pub struct WebAssemblyBackend {
    /// Backend configuration
    config: WasmBackendConfig,
    /// Type converter for PIR to WASM type mapping
    type_converter: WasmTypeConverter,
    /// String constant manager
    string_manager: StringConstantManager,
    /// Memory layout manager
    memory_manager: WasmMemoryManager,
    /// Instruction generator
    instruction_generator: WasmInstructionGenerator,
    /// Runtime integration
    runtime_integration: WasmRuntimeIntegration,
    /// Code optimizer
    optimizer: WasmOptimizer,
    /// Code validator
    validator: WasmValidator,
}

impl WebAssemblyBackend {
    /// Create new WebAssembly backend with configuration
    pub fn new(config: CodeGenConfig) -> Self {
        let wasm_config = WasmBackendConfig::from_codegen_config(&config);
        
        // Create type converter
        let type_converter = WasmTypeConverter::new(
            wasm_config.features.clone(),
            wasm_config.runtime_target,
        );

        // Create string manager
        let string_manager = StringConstantManager::new(StringManagerConfig {
            enable_deduplication: true,
            memory_alignment: 4,
            max_string_length: 65536,
            enable_compression: false, // Can be enabled for size optimization
        });

        // Create memory manager
        let memory_layout = WasmMemoryLayout {
            initial_pages: 16, // 1MB initial
            max_pages: Some(256), // 16MB max
            type_registry_offset: 0x1000,
            effect_registry_offset: 0x2000,
            string_constants_offset: 0x3000,
            capability_registry_offset: 0x4000,
            business_rule_registry_offset: 0x5000,
            heap_start_offset: 0x10000,
        };
        let memory_manager = WasmMemoryManager::new(memory_layout);

        // Create instruction generator
        let instruction_generator = WasmInstructionGenerator::new(type_converter.clone());

        // Create runtime integration
        let runtime_config = WasmRuntimeConfig {
            target: wasm_config.runtime_target,
            features: wasm_config.features.clone(),
            enable_capability_validation: true,
            enable_effect_tracking: true,
            enable_performance_monitoring: config.ai_metadata_level != AIMetadataLevel::None,
            enable_debug_support: config.optimization_level == 0,
        };
        let runtime_integration = WasmRuntimeIntegration::new(runtime_config);

        // Create optimizer
        let optimizer_config = WasmOptimizerConfig::from(wasm_config.optimization_level);
        let optimizer = WasmOptimizer::new(optimizer_config);

        // Create validator
        let validator_config = WasmValidatorConfig {
            target: wasm_config.runtime_target,
            features: wasm_config.features.clone(),
            validate_prism_semantics: true,
            ..WasmValidatorConfig::default()
        };
        let validator = WasmValidator::new(validator_config);

        Self {
            config: wasm_config,
            type_converter,
            string_manager,
            memory_manager,
            instruction_generator,
            runtime_integration,
            optimizer,
            validator,
        }
    }

    /// Configure runtime target
    pub fn with_runtime_target(mut self, target: WasmRuntimeTarget) -> Self {
        self.config.runtime_target = target;
        // Update dependent components
        self.type_converter = WasmTypeConverter::new(self.config.features.clone(), target);
        self
    }

    /// Configure WebAssembly features
    pub fn with_features(mut self, features: WasmFeatures) -> Self {
        self.config.features = features.clone();
        // Update dependent components
        self.type_converter = WasmTypeConverter::new(features, self.config.runtime_target);
        self
    }

    /// Generate complete WebAssembly module from PIR
    async fn generate_wasm_module(&mut self, pir: &PrismIR, config: &CodeGenConfig) -> WasmResult<String> {
        let mut output = String::new();

        // Generate module header with metadata
        output.push_str(&self.generate_module_header(pir)?);

        // Generate memory declaration
        output.push_str(&self.memory_manager.generate_memory_declaration());

        // Generate runtime imports
        output.push_str(&self.runtime_integration.generate_import_declarations());

        // Register capabilities and effects
        for module in &pir.modules {
            for section in &module.sections {
                if let crate::backends::PIRSection::Functions(func_section) = section {
                    for function in &func_section.functions {
                        // Register function capabilities
                        for capability in &function.capabilities_required {
                            self.runtime_integration.register_capability(capability)?;
                        }
                        
                        // Register function effects
                        for effect in &function.signature.effects.input_effects {
                            self.runtime_integration.register_effect(effect)?;
                        }
                        for effect in &function.signature.effects.output_effects {
                            self.runtime_integration.register_effect(effect)?;
                        }
                    }
                }
            }
        }

        // Generate capability functions
        output.push_str(&self.runtime_integration.generate_capability_functions());

        // Generate effect tracking functions
        output.push_str(&self.runtime_integration.generate_effect_functions());

        // Generate semantic type integration
        let semantic_types: Vec<_> = pir.type_registry.types.values().collect();
        output.push_str(&self.runtime_integration.generate_semantic_type_integration(&semantic_types));

        // Generate runtime initialization
        output.push_str(&self.runtime_integration.generate_runtime_initialization());

        // Generate PIR modules
        for module in &pir.modules {
            output.push_str(&self.generate_pir_module(module, config).await?);
        }

        // Generate string constants data section
        output.push_str(&self.string_manager.generate_data_section());

        // Generate memory layout documentation
        output.push_str(&self.memory_manager.generate_layout_documentation());

        // Generate main function
        output.push_str(&self.generate_main_function());

        // Close module
        output.push_str(")\n");

        Ok(output)
    }

    /// Generate module header with comprehensive metadata
    fn generate_module_header(&self, pir: &PrismIR) -> WasmResult<String> {
        let target_info = match self.config.runtime_target {
            WasmRuntimeTarget::Browser => "browser",
            WasmRuntimeTarget::WASI => "wasi",
            WasmRuntimeTarget::Wasmtime => "wasmtime",
            WasmRuntimeTarget::Wasmer => "wasmer",
            WasmRuntimeTarget::NodeJS => "nodejs",
        };

        Ok(format!(
            r#";; Generated by Prism Compiler - Modular WebAssembly Backend
;; PIR Version: {}
;; Generated at: {}
;; Optimization Level: {:?}
;; Runtime Target: {}
;; Features: Multi-value={}, Bulk-memory={}, SIMD={}, Threads={}, Tail-calls={}, Reference-types={}
;; 
;; Semantic Metadata:
;; - Cohesion Score: {:.2}
;; - Module Count: {}
;; - Type Registry: {} types
;; - Effect Registry: {} effects
;;
;; Business Context:
;; - AI Metadata Level: {:?}
;; - Security Classification: Capability-based
;; - Performance Profile: Portable, sandboxed execution
;;
;; Memory Layout:
{}

(module
"#,
            pir.metadata.version,
            pir.metadata.created_at.as_deref().unwrap_or("unknown"),
            self.config.optimization_level,
            target_info,
            self.config.features.multi_value,
            self.config.features.bulk_memory,
            self.config.features.simd,
            self.config.features.threads,
            self.config.features.tail_calls,
            self.config.features.reference_types,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            pir.type_registry.types.len(),
            pir.effect_graph.nodes.len(),
            self.config.ai_metadata_level,
            self.memory_manager.generate_memory_layout_info(),
        ))
    }

    /// Generate PIR module as WASM functions and data
    async fn generate_pir_module(&mut self, module: &PIRModule, _config: &CodeGenConfig) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            "  ;; === MODULE: {} ===\n",
            module.name
        ));
        output.push_str(&format!(
            "  ;; Business Capability: {}\n",
            module.capability
        ));
        output.push_str(&format!(
            "  ;; Cohesion Score: {:.2}\n",
            module.cohesion_score
        ));
        output.push('\n');

        // Generate module sections
        for section in &module.sections {
            match section {
                crate::backends::PIRSection::Functions(function_section) => {
                    for function in &function_section.functions {
                        output.push_str(&self.generate_wasm_function(function).await?);
                    }
                }
                crate::backends::PIRSection::Constants(constant_section) => {
                    output.push_str("  ;; Constants (stored in data section)\n");
                    for constant in &constant_section.constants {
                        // Add constants to string manager if they're strings
                        if let PIRExpression::Literal(crate::backends::PIRLiteral::String(s)) = &constant.value {
                            self.string_manager.add_string(s, None)?;
                        }
                        
                        output.push_str(&format!(
                            "  ;; Constant: {} = {} ({})\n",
                            constant.name,
                            self.format_constant_value(&constant.value)?,
                            constant.business_meaning.as_deref().unwrap_or("No description")
                        ));
                    }
                    output.push('\n');
                }
                _ => {
                    output.push_str("  ;; Other section types handled elsewhere\n");
                }
            }
        }

        Ok(output)
    }

    /// Generate WebAssembly function from PIR function
    async fn generate_wasm_function(&mut self, function: &PIRFunction) -> WasmResult<String> {
        let mut output = String::new();
        
        // Reset instruction generator for new function
        self.instruction_generator.reset_locals();
        
        // Convert parameter types
        let mut param_types = Vec::new();
        for param in &function.signature.parameters {
            let wasm_type = self.type_converter.convert_pir_type_to_wasm(&param.param_type)?;
            param_types.push((param.name.clone(), wasm_type));
        }
        
        // Convert return type
        let return_type = self.type_converter.convert_pir_type_to_wasm(&function.signature.return_type)?;
        
        // Generate function signature with comprehensive metadata
        output.push_str(&format!(
            "  ;; === FUNCTION: {} ===\n", function.name
        ));
        if let Some(responsibility) = &function.responsibility {
            output.push_str(&format!(
                "  ;; Business Responsibility: {}\n", responsibility
            ));
        }
        
        output.push_str(&format!(
            "  ;; Required Capabilities: [{}]\n",
            function.capabilities_required.iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        
        // Generate function declaration
        let params_str = param_types.iter()
            .map(|(name, typ)| format!("(param ${} {})", name, typ))
            .collect::<Vec<_>>()
            .join(" ");
        
        let result_str = if !matches!(function.signature.return_type.as_ref(), 
            crate::backends::PIRTypeInfo::Primitive(crate::backends::PIRPrimitiveType::Unit)) {
            format!(" (result {})", return_type)
        } else {
            String::new()
        };
        
        output.push_str(&format!(
            "  (func ${} (export \"{}\") {}{}\n",
            function.name, function.name, params_str, result_str
        ));
        
        // Generate local variables
        let required_locals = self.instruction_generator.get_required_locals();
        for (local_name, local_type) in required_locals {
            output.push_str(&format!("    (local {} {})\n", local_name, local_type));
        }
        output.push('\n');
        
        // Generate capability validation using the capability system
        let capability_system = WasmCapabilitySystem::new(self.runtime_integration.get_config().clone());
        let capability_code = capability_system.generate_capability_check(&function.capabilities_required);
        output.push_str(&capability_code);
        
        // Generate effect tracking
        let all_effects: Vec<_> = function.signature.effects.input_effects.iter()
            .chain(function.signature.effects.output_effects.iter())
            .collect();
        let (begin_effects, end_effects) = capability_system.generate_effect_tracking(&all_effects);
        output.push_str(&begin_effects);
        
        // Generate function body
        output.push_str("    ;; === FUNCTION BODY ===\n");
        let body_code = self.instruction_generator.generate_expression(&function.body)?;
        // Indent the body code properly
        for line in body_code.lines() {
            output.push_str("  ");
            output.push_str(line);
            output.push('\n');
        }
        
        // Generate effect cleanup
        output.push_str(&end_effects);
        
        // Generate return handling if needed
        if !matches!(function.signature.return_type.as_ref(), 
            crate::backends::PIRTypeInfo::Primitive(crate::backends::PIRPrimitiveType::Unit)) {
            // The function body should leave the return value on the stack
        }
        
        output.push_str("  )\n\n");
        
        Ok(output)
    }

    /// Generate main function
    fn generate_main_function(&self) -> String {
        match self.config.runtime_target {
            WasmRuntimeTarget::WASI => {
                format!(
                    r#"  ;; === MAIN FUNCTION ===
  (func $main (export "_start") ;; WASI entry point
    ;; Initialize Prism runtime
    call $prism_runtime_init
    
    ;; Call user main function (if exists)
    ;; TODO: Generate call to user main
    
    ;; Clean exit
    i32.const 0
    call $proc_exit ;; WASI exit
  )

"#
                )
            }
            _ => {
                format!(
                    r#"  ;; === MAIN FUNCTION ===
  (func $main (export "main")
    ;; Initialize Prism runtime
    call $prism_runtime_init
    
    ;; Call user main function (if exists)
    ;; TODO: Generate call to user main
  )

"#
                )
            }
        }
    }

    /// Format constant value for display
    fn format_constant_value(&self, expr: &PIRExpression) -> WasmResult<String> {
        match expr {
            PIRExpression::Literal(lit) => {
                match lit {
                    crate::backends::PIRLiteral::Integer(i) => Ok(i.to_string()),
                    crate::backends::PIRLiteral::Float(f) => Ok(f.to_string()),
                    crate::backends::PIRLiteral::Boolean(b) => Ok(if *b { "true" } else { "false" }.to_string()),
                    crate::backends::PIRLiteral::String(s) => Ok(format!("\"{}\"", s)),
                    crate::backends::PIRLiteral::Unit => Ok("()".to_string()),
                }
            }
            _ => Ok("complex_expression".to_string())
        }
    }
}

impl WasmBackendConfig {
    /// Create WASM backend config from CodeGen config
    pub fn from_codegen_config(config: &CodeGenConfig) -> Self {
        Self {
            optimization_level: WasmOptimizationLevel::from(config.optimization_level),
            runtime_target: WasmRuntimeTarget::default(),
            features: WasmFeatures::default(),
            ai_metadata_level: config.ai_metadata_level.clone(),
            preserve_debug_info: config.optimization_level == 0,
            enable_validation: true,
        }
    }
}

#[async_trait]
impl CodeGenBackend for WebAssemblyBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::WebAssembly
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let _span = span!(Level::INFO, "wasm_pir_codegen").entered();
        let start_time = std::time::Instant::now();

        info!("Generating WebAssembly from PIR with modular architecture");

        // Clone self to make it mutable for generation
        let mut backend = self.clone();
        
        // Generate WASM code
        let wasm_content = backend.generate_wasm_module(pir, config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "WebAssembly".to_string(),
                message: format!("WASM generation failed: {:?}", e),
            })?;

        let generation_time = start_time.elapsed().as_millis() as u64;

        // Apply optimizations
        let optimized_content = if config.optimization_level > 0 {
            let mut optimizer = backend.optimizer;
            optimizer.optimize(&wasm_content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "WebAssembly".to_string(),
                    message: format!("WASM optimization failed: {:?}", e),
                })?
        } else {
            wasm_content
        };

        // Validate the generated code
        let validation_results = backend.validator.validate(&optimized_content)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "WebAssembly".to_string(),
                message: format!("WASM validation failed: {:?}", e),
            })?;

        // Convert validation warnings to our format
        let warnings: Vec<String> = validation_results.warnings.iter()
            .map(|w| format!("[{}] {}", w.code, w.message))
            .collect();

        // Check if validation failed
        if !validation_results.is_valid {
            let errors: Vec<String> = validation_results.errors.iter()
                .map(|e| format!("[{}] {}", e.code, e.message))
                .collect();
            return Err(crate::CodeGenError::ValidationError {
                target: "WebAssembly".to_string(),
                errors,
            });
        }

        Ok(CodeArtifact {
            target: CompilationTarget::WebAssembly,
            content: optimized_content,
            source_map: None, // WASM source maps are complex and target-specific
            ai_metadata: pir.ai_metadata.clone(),
            output_path: PathBuf::from("output.wasm"),
            stats: CodeGenStats {
                lines_generated: optimized_content.lines().count(),
                generation_time,
                optimizations_applied: if config.optimization_level > 0 { 
                    backend.optimizer.get_stats().constant_foldings +
                    backend.optimizer.get_stats().dead_code_eliminations +
                    backend.optimizer.get_stats().instruction_combinations
                } else { 0 },
                memory_usage: optimized_content.len(),
            },
        })
    }

    async fn generate_code(
        &self,
        program: &Program,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        // Convert AST to PIR first, then use PIR generation
        let mut pir_builder = crate::backends::PIRConstructionBuilder::new();
        let pir = pir_builder.build_from_program(program)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "WebAssembly".to_string(),
                message: format!("PIR construction failed: {:?}", e),
            })?;
        self.generate_code_from_pir(&pir, context, config).await
    }

    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let wasm_type = self.type_converter.convert_pir_type_to_wasm(&semantic_type.base_type)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "WebAssembly".to_string(),
                message: format!("Type conversion failed: {:?}", e),
            })?;
        
        Ok(format!(
            ";; Semantic Type: {} (Domain: {}, Security: {:?})\n;; WASM Type: {}\n",
            semantic_type.name, semantic_type.domain, semantic_type.security_classification, wasm_type
        ))
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut backend = self.clone();
        backend.generate_wasm_function(function).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "WebAssembly".to_string(),
                message: format!("Function generation failed: {:?}", e),
            })
    }

    async fn generate_validation_logic(
        &self,
        semantic_type: &PIRSemanticType,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        self.generate_semantic_type(semantic_type, config).await
    }

    async fn generate_runtime_support(
        &self,
        pir: &PrismIR,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        Ok(self.runtime_integration.generate_runtime_initialization())
    }

    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        if config.optimization_level > 0 {
            let mut optimizer = self.optimizer.clone();
            artifact.content = optimizer.optimize(&artifact.content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "WebAssembly".to_string(),
                    message: format!("Optimization failed: {:?}", e),
                })?;
            artifact.stats.optimizations_applied += optimizer.get_stats().constant_foldings +
                optimizer.get_stats().dead_code_eliminations +
                optimizer.get_stats().instruction_combinations;
        }
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        let mut validator = self.validator.clone();
        let results = validator.validate(&artifact.content)
            .map_err(|e| crate::CodeGenError::ValidationError {
                target: "WebAssembly".to_string(),
                errors: vec![format!("Validation failed: {:?}", e)],
            })?;

        Ok(results.warnings.iter()
            .map(|w| format!("[{}] {}", w.code, w.message))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            source_maps: false, // WASM source maps are complex and runtime-specific
            debug_info: true,   // WASM has excellent debug support
            incremental: false, // WASM compilation is typically full-module
            parallel: true,     // WASM generation can be parallelized
            optimization_levels: vec![0, 1, 2, 3], // Standard optimization levels
        }
    }
}

impl Clone for WebAssemblyBackend {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            type_converter: self.type_converter.clone(),
            string_manager: self.string_manager.clone(),
            memory_manager: self.memory_manager.clone(),
            instruction_generator: WasmInstructionGenerator::new(self.type_converter.clone()),
            runtime_integration: WasmRuntimeIntegration::new(self.runtime_integration.get_config().clone()),
            optimizer: WasmOptimizer::new(self.optimizer.get_config().clone()),
            validator: WasmValidator::new(WasmValidatorConfig {
                target: self.config.runtime_target,
                features: self.config.features.clone(),
                ..WasmValidatorConfig::default()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let config = CodeGenConfig::default();
        let backend = WebAssemblyBackend::new(config);
        
        assert_eq!(backend.target(), CompilationTarget::WebAssembly);
        assert!(backend.capabilities().debug_info);
        assert!(backend.capabilities().parallel);
    }

    #[test]
    fn test_configuration() {
        let config = CodeGenConfig::default();
        let backend = WebAssemblyBackend::new(config)
            .with_runtime_target(WasmRuntimeTarget::Browser)
            .with_features(WasmFeatures {
                simd: true,
                threads: true,
                ..WasmFeatures::default()
            });
        
        assert_eq!(backend.config.runtime_target, WasmRuntimeTarget::Browser);
        assert!(backend.config.features.simd);
        assert!(backend.config.features.threads);
    }
} 