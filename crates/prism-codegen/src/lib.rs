//! Prism Code Generation
//!
//! This crate provides multi-target code generation for the Prism programming language.
//! It supports TypeScript, JavaScript, WebAssembly, and LLVM native code generation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod backends;

// Re-export main types for easy access
pub use backends::{
    CodeGenBackend, MultiTargetCodeGen, CodeArtifact, CodeGenConfig, CodeGenStats,
    BackendCapabilities, AIMetadataLevel,
    TypeScriptBackend, LLVMBackend, WasmBackend, JavaScriptBackend
};

use prism_common::span::Span;
use prism_ast::Program;

// Import PIR types that codegen needs to consume
// This maintains proper separation: codegen consumes PIR, doesn't produce it
use prism_pir::{
    PrismIR, PIRModule, PIRFunction, PIRSemanticType, PIRExpression, PIRStatement,
    PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRCompositeKind, PIRParameter,
    PIRBinaryOp, PIRUnaryOp, PIRLiteral, PIRTypeConstraint, PIRPerformanceContract,
    PIRComplexityAnalysis, SemanticTypeRegistry, BusinessRule, ValidationPredicate,
    PIRTypeAIContext, SecurityClassification, PIRBuilder, PIRBuilderConfig,
    PIRSection, TypeSection, FunctionSection, ConstantSection, InterfaceSection,
    ImplementationSection, PIRConstant, PIRInterface, PIRImplementationItem,
    PIRTypeImplementation, PIRMethod, PIRField, PIRVisibility, PIRCondition,
    PIRPerformanceGuarantee, PIRPerformanceType, EffectSignature, Effect, Capability,
    CohesionMetrics, PIRMetadata
};

use std::path::PathBuf;
use thiserror::Error;

/// Result type for code generation operations
pub type CodeGenResult<T> = Result<T, CodeGenError>;

/// Code generation error types
#[derive(Debug, Error)]
pub enum CodeGenError {
    /// Code generation failed
    #[error("Code generation failed for target {target}: {message}")]
    CodeGenerationError { target: String, message: String },

    /// Unsupported target
    #[error("Unsupported compilation target: {target}")]
    UnsupportedTarget { target: String },

    /// Invalid configuration
    #[error("Invalid code generation configuration: {message}")]
    InvalidConfig { message: String },

    /// Optimization failed
    #[error("Optimization failed: {message}")]
    OptimizationError { message: String },

    /// Validation failed
    #[error("Generated code validation failed: {errors:?}")]
    ValidationError { errors: Vec<String> },

    /// PIR builder error during AST to PIR conversion
    #[error("PIR builder error: {source}")]
    PIRBuilderError {
        #[from]
        source: prism_pir::PIRError,
    },

    /// I/O error during code generation
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// Serialization error
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        source: serde_json::Error,
    },

    /// Internal error
    #[error("Internal code generation error: {message}")]
    InternalError { message: String },
}

impl From<CodeGenError> for prism_common::diagnostics::Diagnostic {
    fn from(error: CodeGenError) -> Self {
        use prism_common::diagnostics::{Diagnostic, DiagnosticLevel};
        
        Diagnostic {
            level: DiagnosticLevel::Error,
            message: error.to_string(),
            span: Span::default(),
            source: Some("codegen".to_string()),
            help: None,
        }
    }
}
