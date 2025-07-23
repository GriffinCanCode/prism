//! Multi-target code generation backends
//!
//! This module provides code generation for multiple target platforms:
//! TypeScript, WebAssembly (WASM), LLVM native code, and JavaScript.

pub mod typescript;
pub mod llvm;
pub mod webassembly;
pub mod python;
pub mod javascript;
pub mod prism_vm;

// Re-export main backend implementations
// TypeScript backend now uses modular architecture with 2025 best practices
pub use typescript::{TypeScriptBackend, TypeScriptBackendConfig, TypeScriptFeatures, TypeScriptTarget};
pub use llvm::{LLVMBackend, LLVMBackendConfig};
pub use webassembly::WebAssemblyBackend;
pub use python::{PythonBackend, PythonBackendConfig, PythonFeatures, PythonTarget};
pub use javascript::{JavaScriptBackend, JavaScriptBackendConfig, JavaScriptFeatures, JavaScriptTarget};
pub use prism_vm::{PrismVMBackend, PrismVMBackendConfig};

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
    PIRTypeAIContext, SecurityClassification, PIRConstructionBuilder, ConstructionConfig,
    PIRSection, TypeSection, FunctionSection, ConstantSection, InterfaceSection,
    ImplementationSection, PIRConstant, PIRInterface, PIRImplementationItem,
    PIRTypeImplementation, PIRMethod, PIRField, PIRVisibility, PIRCondition,
    PIRPerformanceGuarantee, PIRPerformanceType, EffectSignature, Effect, Capability,
    CohesionMetrics, PIRMetadata, EffectGraph, EffectNode, EffectEdge, EffectEdgeType,
    ResourceLimits, ResourceUsageDelta,
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
    /// Python for AI/ML and data science integration
    Python,
    /// Prism VM for unified debugging and runtime optimization
    PrismVM,
    
    // Potential future targets:
    // /// Rust for systems programming with memory safety
    // Rust,
    // /// C# for enterprise .NET integration  
    // CSharp,
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
    /// Semantic types with their business domains
    pub semantic_types: HashMap<String, String>,
    /// Business context for each module
    pub business_context: HashMap<String, String>,
    /// Performance optimization hints
    pub performance_hints: Vec<String>,
}

/// AI metadata generation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AIMetadataLevel {
    /// No AI metadata
    None,
    /// Basic metadata (types, functions)
    Basic,
    /// Full metadata (includes business context, performance hints)
    Full,
}

/// Code generation configuration
#[derive(Debug, Clone)]
pub struct CodeGenConfig {
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Generate debugging information
    pub debug_info: bool,
    /// Generate source maps
    pub source_maps: bool,
    /// Target-specific options
    pub target_options: HashMap<CompilationTarget, HashMap<String, String>>,
    /// AI metadata generation level
    pub ai_metadata_level: AIMetadataLevel,
}

/// Generated code artifact
#[derive(Debug, Clone)]
pub struct CodeArtifact {
    /// Target platform
    pub target: CompilationTarget,
    /// Generated code content
    pub content: String,
    /// Source map (if generated)
    pub source_map: Option<String>,
    /// AI metadata
    pub ai_metadata: AIMetadata,
    /// Output file path
    pub output_path: PathBuf,
    /// Generation statistics
    pub stats: CodeGenStats,
}

/// Code generation statistics
#[derive(Debug, Clone)]
pub struct CodeGenStats {
    /// Number of lines generated
    pub lines_generated: usize,
    /// Generation time in milliseconds
    pub generation_time: u64,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
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

/// Multi-target code generator
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
            Box::new(LLVMBackend::new(LLVMBackendConfig::default()).unwrap())
        );
        backends.insert(
            CompilationTarget::WebAssembly,
            Box::new(WebAssemblyBackend::new(config.clone()))
        );
        backends.insert(
            CompilationTarget::JavaScript,
            Box::new(javascript::JavaScriptBackend::new(javascript::JavaScriptBackendConfig::default()))
        );
        backends.insert(
            CompilationTarget::Python,
            Box::new(PythonBackend::new(config.clone()))
        );
        backends.insert(
            CompilationTarget::PrismVM,
            Box::new(PrismVMBackend::new(config).expect("Failed to create PrismVM backend"))
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