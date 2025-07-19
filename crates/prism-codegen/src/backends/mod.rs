//! Multi-target code generation backends
//!
//! This module provides code generation for multiple target platforms:
//! TypeScript, WebAssembly (WASM), LLVM native code, and JavaScript.

pub mod typescript;
pub mod llvm;

// Re-export main backend implementations
pub use typescript::TypeScriptBackend;
pub use llvm::LLVMBackend;

// Additional backends defined in this module
pub use self::{WasmBackend, JavaScriptBackend};

use crate::CodeGenResult;
use async_trait::async_trait;
use prism_ast::Program;
use prism_common::{NodeId, span::Span};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// Import PIR types from the dedicated crate - this maintains proper SoC
// Codegen consumes PIR but doesn't produce it
pub use prism_pir::{
    PrismIR, PIRModule, PIRFunction, PIRSemanticType, PIRExpression, PIRStatement,
    PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRCompositeKind, PIRParameter,
    PIRBinaryOp, PIRUnaryOp, PIRLiteral, PIRTypeConstraint, PIRPerformanceContract,
    PIRComplexityAnalysis, SemanticTypeRegistry, BusinessRule, ValidationPredicate,
    PIRTypeAIContext, SecurityClassification, PIRBuilder, PIRBuilderConfig,
    PIRSection, TypeSection, FunctionSection, ConstantSection, InterfaceSection,
    ImplementationSection, PIRConstant, PIRInterface, PIRImplementationItem,
    PIRTypeImplementation, PIRMethod, PIRField, PIRVisibility, PIRCondition,
    PIRPerformanceGuarantee, PIRPerformanceType, EffectSignature, Effect, Capability,
    CohesionMetrics, PIRMetadata, EffectGraph, EffectNode, EffectEdge, EffectEdgeType
};

/// Compilation target platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilationTarget {
    /// TypeScript transpilation for rapid prototyping
    TypeScript,
    /// WebAssembly for portable execution
    WebAssembly,
    /// Native code via LLVM for performance
    LLVM,
    /// JavaScript for web deployment
    JavaScript,
}

/// Compilation context - provided by compiler orchestration
#[derive(Debug, Clone)]
pub struct CompilationContext {
    /// Current compilation phase
    pub current_phase: String,
    /// Target configurations
    pub targets: Vec<CompilationTarget>,
}

/// AI metadata for generated code
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Semantic type information
    pub semantic_types: HashMap<String, String>,
    /// Business context
    pub business_context: HashMap<String, String>,
    /// Performance hints
    pub performance_hints: Vec<String>,
}

/// Generated code artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeArtifact {
    /// Target platform
    pub target: CompilationTarget,
    /// Generated code content
    pub content: String,
    /// Source map for debugging
    pub source_map: Option<String>,
    /// AI metadata for the generated code
    pub ai_metadata: AIMetadata,
    /// Output file path
    pub output_path: PathBuf,
    /// Code generation statistics
    pub stats: CodeGenStats,
}

/// Code generation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeGenStats {
    /// Lines of code generated
    pub lines_generated: usize,
    /// Generation time in milliseconds
    pub generation_time: u64,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Memory usage during generation
    pub memory_usage: usize,
}

/// Code generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGenConfig {
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable debug information
    pub debug_info: bool,
    /// Enable source maps
    pub source_maps: bool,
    /// Target-specific options
    pub target_options: HashMap<String, String>,
    /// AI metadata generation level
    pub ai_metadata_level: AIMetadataLevel,
}

/// AI metadata generation level
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AIMetadataLevel {
    /// No AI metadata
    None,
    /// Basic metadata (types, functions)
    Basic,
    /// Full metadata (includes semantic analysis)
    Full,
    /// Comprehensive metadata (includes optimization hints)
    Comprehensive,
}

impl Default for CodeGenConfig {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            debug_info: true,
            source_maps: true,
            target_options: HashMap::new(),
            ai_metadata_level: AIMetadataLevel::Full,
        }
    }
}

/// Code generation backend trait - Enhanced for PIR-based semantic preservation
#[async_trait]
pub trait CodeGenBackend: Send + Sync {
    /// Target platform this backend generates code for
    fn target(&self) -> CompilationTarget;

    /// Generate code from PIR (Prism Intermediate Representation)
    /// This is the primary method - all backends should implement PIR consumption
    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact>;

    /// Generate code from AST (legacy support)
    /// This should convert AST to PIR first, then use PIR generation
    async fn generate_code(
        &self,
        program: &Program,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact>;

    /// Generate semantic type with business rules preservation
    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String>;

    /// Generate function with effect tracking
    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String>;

    /// Generate validation logic for business rules
    async fn generate_validation_logic(
        &self,
        semantic_type: &PIRSemanticType,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String>;

    /// Generate runtime support for effects and capabilities
    async fn generate_runtime_support(
        &self,
        pir: &PrismIR,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String>;

    /// Optimize generated code
    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()>;

    /// Validate generated code
    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>>;

    /// Get backend-specific capabilities
    fn capabilities(&self) -> BackendCapabilities;
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Supports source maps
    pub source_maps: bool,
    /// Supports debugging information
    pub debug_info: bool,
    /// Supports incremental compilation
    pub incremental: bool,
    /// Supports parallel compilation
    pub parallel: bool,
    /// Optimization levels supported
    pub optimization_levels: Vec<u8>,
}

/// WebAssembly code generation backend (simplified implementation)
pub struct WasmBackend {
    config: CodeGenConfig,
}

impl WasmBackend {
    /// Create a new WebAssembly backend
    pub fn new(config: CodeGenConfig) -> Self {
        Self { config }
    }

    async fn generate_wasm_function_from_pir(
        &self,
        function: &PIRFunction,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            "(func ${} (export \"{}\")\n",
            function.name, function.name
        ));
        
        // Generate capability validation
        for capability in &function.capabilities_required {
            output.push_str(&format!(
                "  ;; Validate capability: {}\n",
                capability.name
            ));
            output.push_str("  ;; TODO: call $prism_validate_capability\n");
        }
        
        // Generate effect tracking
        for effect in &function.signature.effects.effects {
            output.push_str(&format!(
                "  ;; Track effect: {}\n",
                effect.name
            ));
            output.push_str("  ;; TODO: call $prism_track_effect\n");
        }
        
        // Generate function body (placeholder)
        output.push_str("  ;; Function body would be generated here\n");
        
        output.push_str(")\n\n");
        
        Ok(output)
    }

    fn convert_pir_type_to_wasm(&self, pir_type: &PIRTypeInfo) -> String {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => {
                match prim {
                    PIRPrimitiveType::Integer { signed: _, width } => {
                        match width {
                            32 => "i32".to_string(),
                            64 => "i64".to_string(),
                            _ => "i32".to_string(),
                        }
                    }
                    PIRPrimitiveType::Float { width } => {
                        match width {
                            32 => "f32".to_string(),
                            64 => "f64".to_string(),
                            _ => "f64".to_string(),
                        }
                    }
                    PIRPrimitiveType::Boolean => "i32".to_string(), // WASM bool is i32
                    PIRPrimitiveType::String => "i32".to_string(), // WASM string is i32 pointer
                    PIRPrimitiveType::Unit => "void".to_string(),
                }
            }
            PIRTypeInfo::Composite(_) => "i32".to_string(), // Pointer to struct
            PIRTypeInfo::Function(_) => "i32".to_string(), // Function pointer
            PIRTypeInfo::Generic(_) => "i32".to_string(), // Generic pointer
            PIRTypeInfo::Effect(_) => "i32".to_string(), // Effect handle
        }
    }

    fn generate_wasm_constant(&self, expr: &PIRExpression) -> CodeGenResult<String> {
        match expr {
            PIRExpression::Literal(lit) => {
                match lit {
                    PIRLiteral::Integer(i) => Ok(format!("i32.const {}", i)),
                    PIRLiteral::Float(f) => Ok(format!("f64.const {}", f)),
                    PIRLiteral::Boolean(b) => Ok(format!("i32.const {}", if *b { 1 } else { 0 })),
                    PIRLiteral::String(s) => Ok(format!("i32.const {}", s.len())), // Placeholder for string literal
                    PIRLiteral::Unit => Ok("i32.const 0".to_string()),
                }
            }
            _ => Ok("i32.const 0 ; Complex expression not supported".to_string()),
        }
    }

    fn generate_js_constant(&self, expr: &PIRExpression) -> CodeGenResult<String> {
        match expr {
            PIRExpression::Literal(lit) => {
                match lit {
                    PIRLiteral::Integer(i) => Ok(i.to_string()),
                    PIRLiteral::Float(f) => Ok(f.to_string()),
                    PIRLiteral::Boolean(b) => Ok(b.to_string()),
                    PIRLiteral::String(s) => Ok(format!("\"{}\"", s)),
                    PIRLiteral::Unit => Ok("undefined".to_string()),
                }
            }
            _ => Ok("null /* Complex expression not supported */".to_string()),
        }
    }
}

#[async_trait]
impl CodeGenBackend for WasmBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::WebAssembly
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let start_time = std::time::Instant::now();

        // Generate WebAssembly Text (WAT) format with semantic preservation
        let mut output = String::new();
        output.push_str("(module\n");
        output.push_str(&format!("  ;; Generated by Prism Compiler from PIR v{}\n", pir.metadata.version));
        output.push_str(&format!("  ;; Cohesion Score: {:.2}\n", pir.cohesion_metrics.overall_score));
        
        // Memory and exports
        output.push_str("  (memory 1)\n");
        output.push_str("  (export \"memory\" (memory 0))\n\n");
        
        // Import Prism runtime functions
        output.push_str("  ;; Prism Runtime Imports\n");
        output.push_str("  (import \"prism\" \"validate_capability\" (func $prism_validate_capability (param i32) (result i32)))\n");
        output.push_str("  (import \"prism\" \"track_effect\" (func $prism_track_effect (param i32)))\n");
        output.push_str("  (import \"prism\" \"validate_business_rule\" (func $prism_validate_business_rule (param i32 i32) (result i32)))\n\n");

        // Generate semantic type metadata as data sections
        output.push_str("  ;; Semantic Type Registry\n");
        for (type_name, semantic_type) in &pir.type_registry.types {
            output.push_str(&format!(
                "  (data (i32.const {}) \"{}\")\n", 
                type_name.len(), type_name
            ));
            output.push_str(&format!(
                "  ;; Type: {} - Domain: {} - Security: {:?}\n",
                type_name, semantic_type.domain, semantic_type.security_classification
            ));
        }
        output.push('\n');

        // Generate functions from PIR modules
        for module in &pir.modules {
            output.push_str(&format!("  ;; Module: {} (Capability: {})\n", module.name, module.capability));
            
            for section in &module.sections {
                match section {
                    PIRSection::Functions(function_section) => {
                        for function in &function_section.functions {
                            output.push_str(&self.generate_wasm_function_from_pir(function, config).await?);
                        }
                    }
                    PIRSection::Constants(constant_section) => {
                        for constant in &constant_section.constants {
                            output.push_str(&format!(
                                "  ;; Constant: {} = {}\n",
                                constant.name,
                                self.generate_wasm_constant(&constant.value)?
                            ));
                        }
                    }
                    _ => {
                        output.push_str(&format!("  ;; Section type not yet implemented in WASM\n"));
                    }
                }
            }
        }

        // Generate main export function
        output.push_str("  (func $main (export \"main\") (result i32)\n");
        output.push_str("    ;; Initialize Prism runtime\n");
        output.push_str("    ;; TODO: Call user main function\n");
        output.push_str("    i32.const 0\n");
        output.push_str("  )\n");

        output.push_str(")\n");

        let generation_time = start_time.elapsed().as_millis() as u64;

        Ok(CodeArtifact {
            target: CompilationTarget::WebAssembly,
            content: output.clone(),
            source_map: None, // WASM source maps are complex
            ai_metadata: pir.ai_metadata.clone(),
            output_path: PathBuf::from("output.wasm"),
            stats: CodeGenStats {
                lines_generated: output.lines().count(),
                generation_time,
                optimizations_applied: if config.optimization_level > 0 { 1 } else { 0 },
                memory_usage: output.len(),
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
        let mut pir_builder = PIRBuilder::new();
        let pir = pir_builder.build_from_program(program)?;
        self.generate_code_from_pir(&pir, context, config).await
    }

    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut output = String::new();
        
        // Generate WASM struct type representation
        output.push_str(&format!(
            ";; Semantic Type: {} (Domain: {})\n",
            semantic_type.name, semantic_type.domain
        ));
        
        // WASM doesn't have native struct types, so we use memory layout
        let wasm_type = self.convert_pir_type_to_wasm(&semantic_type.base_type);
        output.push_str(&format!(
            ";; Base type maps to WASM: {}\n",
            wasm_type
        ));
        
        // Generate validation function
        output.push_str(&format!(
            "(func $validate_{} (param ${} {}) (result i32)\n",
            semantic_type.name.to_lowercase(), 
            semantic_type.name.to_lowercase(),
            wasm_type
        ));
        
        // Add business rule validation
        for rule in &semantic_type.business_rules {
            output.push_str(&format!(
                "  ;; Business rule: {} - {}\n",
                rule.name, rule.description
            ));
        }
        
        output.push_str("  ;; TODO: Implement validation logic\n");
        output.push_str("  i32.const 1 ;; Return true for now\n");
        output.push_str(")\n\n");
        
        Ok(output)
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            "(func ${} (export \"{}\")\n",
            function.name, function.name
        ));
        
        // Generate capability validation
        for capability in &function.capabilities_required {
            output.push_str(&format!(
                "  ;; Validate capability: {}\n",
                capability.name
            ));
            output.push_str("  ;; TODO: call $prism_validate_capability\n");
        }
        
        // Generate effect tracking
        for effect in &function.signature.effects.effects {
            output.push_str(&format!(
                "  ;; Track effect: {}\n",
                effect.name
            ));
            output.push_str("  ;; TODO: call $prism_track_effect\n");
        }
        
        // Generate function body (placeholder)
        output.push_str("  ;; Function body would be generated here\n");
        
        output.push_str(")\n\n");
        
        Ok(output)
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
        let mut output = String::new();
        
        output.push_str(";; Prism Runtime Support for WebAssembly\n");
        output.push_str(&format!(";; Effect registry: {} effects\n", pir.effect_graph.nodes.len()));
        output.push_str(&format!(";; Type registry: {} types\n", pir.type_registry.types.len()));
        
        // Generate effect registry as data
        output.push_str(";; Effect Registry Data\n");
        for (i, (effect_name, effect_node)) in pir.effect_graph.nodes.iter().enumerate() {
            output.push_str(&format!(
                "(data (i32.const {}) \"{}\")\n",
                1000 + i * 32, // Simple offset calculation
                effect_name
            ));
        }
        
        Ok(output)
    }

    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        if config.optimization_level > 1 {
            // Simple WASM optimization: remove comments
            artifact.content = artifact.content.lines()
                .filter(|line| !line.trim().starts_with(";;") || line.contains("Generated by"))
                .collect::<Vec<_>>()
                .join("\n");
            artifact.stats.optimizations_applied += 1;
        }
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        let mut warnings = Vec::new();
        
        if !artifact.content.contains("(module") {
            warnings.push("Invalid WASM module structure".to_string());
        }
        
        if artifact.content.contains("TODO:") {
            warnings.push("Generated WASM contains incomplete implementations".to_string());
        }
        
        Ok(warnings)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            source_maps: false,
            debug_info: true,
            incremental: false,
            parallel: true,
            optimization_levels: vec![0, 1, 2, 3],
        }
    }
}

/// JavaScript code generation backend (simplified implementation)
pub struct JavaScriptBackend {
    config: CodeGenConfig,
}

impl JavaScriptBackend {
    /// Create a new JavaScript backend
    pub fn new(config: CodeGenConfig) -> Self {
        Self { config }
    }

    async fn generate_js_code(&self, _program: &Program) -> CodeGenResult<String> {
        Ok("// Program code would be generated here\n".to_string())
    }

    async fn generate_source_map(&self, _code: &str) -> CodeGenResult<String> {
        Ok(r#"{"version":3,"sources":[],"names":[],"mappings":""}"#.to_string())
    }

    async fn optimize_js_code(&self, code: &str, _level: u8) -> CodeGenResult<String> {
        Ok(code.to_string())
    }

    fn convert_pir_type_to_js(&self, pir_type: &PIRTypeInfo) -> String {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => {
                match prim {
                    PIRPrimitiveType::Integer { signed: _, width } => {
                        match width {
                            32 => "number".to_string(),
                            64 => "bigint".to_string(),
                            _ => "number".to_string(),
                        }
                    }
                    PIRPrimitiveType::Float { width: _ } => "number".to_string(),
                    PIRPrimitiveType::Boolean => "boolean".to_string(),
                    PIRPrimitiveType::String => "string".to_string(),
                    PIRPrimitiveType::Unit => "void".to_string(),
                }
            }
            PIRTypeInfo::Composite(_) => "object".to_string(),
            PIRTypeInfo::Function(_) => "function".to_string(),
            PIRTypeInfo::Generic(_) => "any".to_string(),
            PIRTypeInfo::Effect(_) => "Effect".to_string(),
        }
    }

         async fn generate_js_module_from_pir(
         &self,
         module: &PIRModule,
         _config: &CodeGenConfig,
     ) -> CodeGenResult<String> {
         let mut output = String::new();
         output.push_str(&format!("// Module: {}\n", module.name));
         output.push_str(&format!("// Capability: {}\n", module.capability));
         output.push_str(&format!("// Cohesion Score: {:.2}\n\n", module.cohesion_score));

         // Generate imports from prism-runtime
         output.push_str("const { PrismRuntime, SemanticType, EffectTracker, CapabilityManager } = require('@prism/runtime');\n\n");

         // Generate module sections
         for section in &module.sections {
             match section {
                 PIRSection::Functions(function_section) => {
                     output.push_str("// Functions\n");
                     for function in &function_section.functions {
                         output.push_str(&format!(
                             "// Function: {} - {}\n",
                             function.name,
                             function.responsibility.as_deref().unwrap_or("N/A")
                         ));
                         output.push_str(&format!(
                             "async function {}() {{\n",
                             function.name
                         ));
                         output.push_str("  // Function implementation would be generated here\n");
                         output.push_str("  throw new Error('Function not yet implemented');\n");
                         output.push_str("}\n\n");
                     }
                 }
                 PIRSection::Constants(constant_section) => {
                     output.push_str("// Constants\n");
                     for constant in &constant_section.constants {
                         output.push_str(&format!(
                             "const {} = {}; // {}\n",
                             constant.name,
                             self.generate_js_constant(&constant.value)?,
                             constant.business_meaning.as_deref().unwrap_or("No description")
                         ));
                     }
                     output.push('\n');
                 }
                 _ => {
                     output.push_str(&format!("// Section type not yet implemented in JS: {:?}\n", section));
                 }
             }
         }

         // Generate module exports
         output.push_str("module.exports = {\n");
         output.push_str(&format!("  moduleName: '{}',\n", module.name));
         output.push_str(&format!("  capability: '{}',\n", module.capability));
         output.push_str("  // Functions and constants would be exported here\n");
         output.push_str("};\n");

         Ok(output)
     }
}

#[async_trait]
impl CodeGenBackend for JavaScriptBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::JavaScript
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let start_time = std::time::Instant::now();

        let mut output = String::new();
        
        // Generate header with metadata
        output.push_str("// Generated by Prism Compiler\n");
        output.push_str(&format!("// PIR Version: {}\n", pir.metadata.version));
        output.push_str(&format!("// Cohesion Score: {:.2}\n", pir.cohesion_metrics.overall_score));
        output.push_str("// JavaScript target with semantic preservation\n\n");
        
        // Runtime imports
        output.push_str("const { PrismRuntime, SemanticType, EffectTracker, CapabilityManager } = require('@prism/runtime');\n\n");
        
        // Generate semantic type registry
        output.push_str("// Semantic Type Registry\n");
        output.push_str("const SEMANTIC_TYPES = new Map();\n");
        for (type_name, semantic_type) in &pir.type_registry.types {
            output.push_str(&format!(
                "SEMANTIC_TYPES.set('{}', new SemanticType({{\n",
                type_name
            ));
            output.push_str(&format!("  name: '{}',\n", semantic_type.name));
            output.push_str(&format!("  domain: '{}',\n", semantic_type.domain));
            output.push_str(&format!("  baseType: '{}',\n", self.convert_pir_type_to_js(&semantic_type.base_type)));
            output.push_str("  businessRules: [\n");
            for rule in &semantic_type.business_rules {
                output.push_str(&format!(
                    "    {{ name: '{}', description: '{}', expression: '{}' }},\n",
                    rule.name, rule.description, rule.expression
                ));
            }
            output.push_str("  ],\n");
            output.push_str("  validationPredicates: [\n");
            for predicate in &semantic_type.validation_predicates {
                output.push_str(&format!(
                    "    {{ name: '{}', expression: '{}' }},\n",
                    predicate.name, predicate.expression
                ));
            }
            output.push_str("  ]\n");
            output.push_str("}));\n\n");
        }

        // Generate effect registry
        output.push_str("// Effect Registry\n");
        output.push_str("const EFFECT_REGISTRY = new Map();\n");
        for (effect_name, effect_node) in &pir.effect_graph.nodes {
            output.push_str(&format!(
                "EFFECT_REGISTRY.set('{}', {{\n",
                effect_name
            ));
            output.push_str(&format!("  name: '{}',\n", effect_node.name));
            output.push_str(&format!("  type: '{}',\n", effect_node.effect_type));
            output.push_str("  capabilities: [");
            output.push_str(&effect_node.capabilities.iter()
                .map(|c| format!("'{}'", c))
                .collect::<Vec<_>>()
                .join(", "));
            output.push_str("],\n");
            output.push_str("  sideEffects: [");
            output.push_str(&effect_node.side_effects.iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", "));
            output.push_str("]\n");
            output.push_str("});\n");
        }
        output.push('\n');

        // Generate modules
        for module in &pir.modules {
            output.push_str(&self.generate_js_module_from_pir(module, config).await?);
        }

        // Generate runtime initialization
        output.push_str("// Initialize Prism runtime\n");
        output.push_str("PrismRuntime.initialize({\n");
        output.push_str("  semanticTypes: SEMANTIC_TYPES,\n");
        output.push_str("  effectRegistry: EFFECT_REGISTRY,\n");
        output.push_str(&format!("  cohesionScore: {},\n", pir.cohesion_metrics.overall_score));
        output.push_str("  debugMode: true\n");
        output.push_str("});\n\n");

        // Export main module
        output.push_str("module.exports = { PrismRuntime, SEMANTIC_TYPES, EFFECT_REGISTRY };\n");

        let generation_time = start_time.elapsed().as_millis() as u64;

        Ok(CodeArtifact {
            target: CompilationTarget::JavaScript,
            content: output.clone(),
            source_map: if config.source_maps {
                Some(self.generate_source_map(&output).await?)
            } else {
                None
            },
            ai_metadata: pir.ai_metadata.clone(),
            output_path: PathBuf::from("output.js"),
            stats: CodeGenStats {
                lines_generated: output.lines().count(),
                generation_time,
                optimizations_applied: if config.optimization_level > 0 { 1 } else { 0 },
                memory_usage: output.len(),
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
        let mut pir_builder = PIRBuilder::new();
        let pir = pir_builder.build_from_program(program)?;
        self.generate_code_from_pir(&pir, context, config).await
    }

    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            "/**\n * Semantic Type: {}\n * Domain: {}\n * Security: {:?}\n */\n",
            semantic_type.name, semantic_type.domain, semantic_type.security_classification
        ));
        
        // Generate branded type class
        output.push_str(&format!(
            "class {} extends SemanticType {{\n",
            semantic_type.name
        ));
        output.push_str("  constructor(value) {\n");
        output.push_str("    super({\n");
        output.push_str(&format!("      name: '{}',\n", semantic_type.name));
        output.push_str(&format!("      domain: '{}',\n", semantic_type.domain));
        output.push_str(&format!("      baseType: '{}',\n", self.convert_pir_type_to_js(&semantic_type.base_type)));
        output.push_str("      value\n");
        output.push_str("    });\n");
        output.push_str("    this.validate();\n");
        output.push_str("  }\n\n");
        
        // Generate validation method
        output.push_str("  validate() {\n");
        for rule in &semantic_type.business_rules {
            output.push_str(&format!(
                "    // Business rule: {} - {}\n",
                rule.name, rule.description
            ));
            output.push_str(&format!(
                "    if (!this.validateBusinessRule('{}', this.value)) {{\n",
                rule.name
            ));
            output.push_str(&format!(
                "      throw new ValidationError('Business rule violation: {}');\n",
                rule.description
            ));
            output.push_str("    }\n");
        }
        
        for predicate in &semantic_type.validation_predicates {
            output.push_str(&format!(
                "    // Validation predicate: {}\n",
                predicate.name
            ));
            output.push_str(&format!(
                "    if (!this.validatePredicate('{}', this.value)) {{\n",
                predicate.expression
            ));
            output.push_str(&format!(
                "      throw new ValidationError('Predicate validation failed: {}');\n",
                predicate.expression
            ));
            output.push_str("    }\n");
        }
        
        output.push_str("    return true;\n");
        output.push_str("  }\n");
        output.push_str("}\n\n");
        
        Ok(output)
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut output = String::new();
        
        // Generate function documentation
        output.push_str("/**\n");
        output.push_str(&format!(" * Function: {}\n", function.name));
        if let Some(responsibility) = &function.responsibility {
            output.push_str(&format!(" * Responsibility: {}\n", responsibility));
        }
        if let Some(algorithm) = &function.algorithm {
            output.push_str(&format!(" * Algorithm: {}\n", algorithm));
        }
        output.push_str(" * Capabilities Required: [");
        output.push_str(&function.capabilities_required.iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", "));
        output.push_str("]\n");
        output.push_str(" * Effects: [");
        output.push_str(&function.signature.effects.effects.iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", "));
        output.push_str("]\n");
        output.push_str(" */\n");
        
        // Generate async function
        output.push_str(&format!("async function {}(", function.name));
        
        // Generate parameters
        let params = function.signature.parameters.iter()
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&params);
        output.push_str(") {\n");
        
        // Generate capability validation
        if !function.capabilities_required.is_empty() {
            output.push_str("  // Capability validation\n");
            output.push_str("  await CapabilityManager.validateCapabilities([\n");
            for capability in &function.capabilities_required {
                output.push_str(&format!("    '{}',\n", capability.name));
            }
            output.push_str("  ]);\n\n");
        }
        
        // Generate effect tracking
        if !function.signature.effects.effects.is_empty() {
            output.push_str("  // Effect tracking\n");
            output.push_str("  const effectTracker = new EffectTracker();\n");
            output.push_str("  effectTracker.trackEffects([\n");
            for effect in &function.signature.effects.effects {
                output.push_str(&format!("    '{}',\n", effect.name));
            }
            output.push_str("  ]);\n\n");
        }
        
        output.push_str("  try {\n");
        output.push_str("    // Function implementation would be generated here\n");
        output.push_str("    const result = undefined; // Placeholder\n");
        output.push_str("    \n");
        
        if !function.signature.effects.effects.is_empty() {
            output.push_str("    effectTracker.complete();\n");
        }
        
        output.push_str("    return result;\n");
        output.push_str("  } catch (error) {\n");
        
        if !function.signature.effects.effects.is_empty() {
            output.push_str("    effectTracker.abort();\n");
        }
        
        output.push_str("    throw error;\n");
        output.push_str("  }\n");
        output.push_str("}\n\n");
        
        Ok(output)
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
        let mut output = String::new();
        
        output.push_str("// Prism Runtime Support for JavaScript\n");
        output.push_str(&format!("// Effect registry: {} effects\n", pir.effect_graph.nodes.len()));
        output.push_str(&format!("// Type registry: {} types\n", pir.type_registry.types.len()));
        output.push_str(&format!("// Cohesion score: {:.2}\n", pir.cohesion_metrics.overall_score));
        
        // Generate validation helpers
        output.push_str("\n// Validation Helper Functions\n");
        output.push_str("function validateBusinessRule(ruleName, value) {\n");
        output.push_str("  // Business rule validation implementation\n");
        output.push_str("  return PrismRuntime.validateBusinessRule(ruleName, value);\n");
        output.push_str("}\n\n");
        
        output.push_str("function validatePredicate(expression, value) {\n");
        output.push_str("  // Predicate validation implementation\n");
        output.push_str("  return PrismRuntime.validatePredicate(expression, value);\n");
        output.push_str("}\n\n");
        
        output.push_str("class ValidationError extends Error {\n");
        output.push_str("  constructor(message) {\n");
        output.push_str("    super(message);\n");
        output.push_str("    this.name = 'ValidationError';\n");
        output.push_str("  }\n");
        output.push_str("}\n\n");
        
        Ok(output)
    }

    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        if config.optimization_level > 0 {
            artifact.content = self.optimize_js_code(&artifact.content, config.optimization_level).await?;
            artifact.stats.optimizations_applied += 1;
        }
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        let mut warnings = Vec::new();
        
        if !artifact.content.contains("'use strict'") && !artifact.content.contains("\"use strict\"") {
            warnings.push("Consider adding 'use strict' directive".to_string());
        }
        
        if artifact.content.contains("undefined; // Placeholder") {
            warnings.push("Generated JavaScript contains placeholder implementations".to_string());
        }
        
        if !artifact.content.contains("PrismRuntime") {
            warnings.push("No Prism runtime integration found".to_string());
        }
        
        Ok(warnings)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            source_maps: true,
            debug_info: true,
            incremental: true,
            parallel: true,
            optimization_levels: vec![0, 1, 2, 3],
        }
    }
}

/// Multi-target code generator that manages all backends
pub struct MultiTargetCodeGen {
    backends: HashMap<CompilationTarget, Box<dyn CodeGenBackend>>,
}

impl MultiTargetCodeGen {
    /// Create a new multi-target code generator with all backends
    pub fn new() -> Self {
        let mut backends: HashMap<CompilationTarget, Box<dyn CodeGenBackend>> = HashMap::new();
        
        let config = CodeGenConfig::default();
        
        // Register all available backends
        backends.insert(
            CompilationTarget::TypeScript,
            Box::new(TypeScriptBackend::new(config.clone()))
        );
        backends.insert(
            CompilationTarget::LLVM,
            Box::new(LLVMBackend::new(config.clone()))
        );
        backends.insert(
            CompilationTarget::WebAssembly,
            Box::new(WasmBackend::new(config.clone()))
        );
        backends.insert(
            CompilationTarget::JavaScript,
            Box::new(JavaScriptBackend::new(config))
        );

        Self { backends }
    }

    /// Generate code for all configured targets
    pub async fn generate_all_targets(
        &self,
        program: &Program,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<HashMap<CompilationTarget, CodeArtifact>> {
        let mut results = HashMap::new();

        for (target, backend) in &self.backends {
            let artifact = backend.generate_code(program, context, config).await?;
            results.insert(*target, artifact);
        }

        Ok(results)
    }

    /// Generate code from PIR for all configured targets
    pub async fn generate_all_targets_from_pir(
        &self,
        pir: &PrismIR,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<HashMap<CompilationTarget, CodeArtifact>> {
        let mut results = HashMap::new();

        for (target, backend) in &self.backends {
            let artifact = backend.generate_code_from_pir(pir, context, config).await?;
            results.insert(*target, artifact);
        }

        Ok(results)
    }

    /// Generate code for specific target
    pub async fn generate_target(
        &self,
        target: CompilationTarget,
        program: &Program,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let backend = self.backends.get(&target)
            .ok_or_else(|| crate::CodeGenError::UnsupportedTarget { 
                target: format!("{:?}", target) 
            })?;

        backend.generate_code(program, context, config).await
    }

    /// Generate code from PIR for specific target
    pub async fn generate_target_from_pir(
        &self,
        target: CompilationTarget,
        pir: &PrismIR,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let backend = self.backends.get(&target)
            .ok_or_else(|| crate::CodeGenError::UnsupportedTarget { 
                target: format!("{:?}", target) 
            })?;

        backend.generate_code_from_pir(pir, context, config).await
    }

    /// Get available backends
    pub fn get_available_targets(&self) -> Vec<CompilationTarget> {
        self.backends.keys().cloned().collect()
    }

    /// Get backend capabilities for a target
    pub fn get_target_capabilities(&self, target: CompilationTarget) -> Option<BackendCapabilities> {
        self.backends.get(&target).map(|backend| backend.capabilities())
    }
}

impl Default for MultiTargetCodeGen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ast::Program;

    #[tokio::test]
    async fn test_multi_target_codegen_creation() {
        let codegen = MultiTargetCodeGen::new();
        let targets = codegen.get_available_targets();
        
        assert!(targets.contains(&CompilationTarget::TypeScript));
        assert!(targets.contains(&CompilationTarget::LLVM));
        assert!(targets.contains(&CompilationTarget::WebAssembly));
        assert!(targets.contains(&CompilationTarget::JavaScript));
        assert_eq!(targets.len(), 4);
    }

    #[tokio::test]
    async fn test_backend_capabilities() {
        let codegen = MultiTargetCodeGen::new();
        
        let ts_caps = codegen.get_target_capabilities(CompilationTarget::TypeScript);
        assert!(ts_caps.is_some());
        assert!(ts_caps.unwrap().source_maps);
        
        let llvm_caps = codegen.get_target_capabilities(CompilationTarget::LLVM);
        assert!(llvm_caps.is_some());
        assert!(llvm_caps.unwrap().debug_info);
    }

    #[test]
    fn test_codegen_config_defaults() {
        let config = CodeGenConfig::default();
        assert_eq!(config.optimization_level, 2);
        assert!(config.debug_info);
        assert!(config.source_maps);
        assert!(matches!(config.ai_metadata_level, AIMetadataLevel::Full));
    }
} 