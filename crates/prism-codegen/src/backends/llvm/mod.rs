//! Modular LLVM Native Code Generation Backend
//!
//! This module implements a comprehensive LLVM backend with proper separation
//! of concerns, following 2025 LLVM best practices and Prism's design principles.
//!
//! ## Architecture
//!
//! The LLVM backend is organized into focused modules:
//!
//! - [`core`] - Core LLVM backend implementation and orchestration
//! - [`types`] - LLVM type system and PIR type conversion
//! - [`instructions`] - LLVM IR instruction generation from PIR expressions
//! - [`optimization`] - LLVM-specific optimization passes and hints
//! - [`validation`] - LLVM IR validation and verification
//! - [`runtime`] - Runtime integration and capability system
//! - [`debug_info`] - Debug information generation and metadata
//! - [`target_machine`] - Target machine configuration and management
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Each module handles a specific aspect of LLVM code generation
//! 2. **Modularity**: Components can be used independently and configured separately
//! 3. **Extensibility**: Easy to add new optimizations, target architectures, or runtime features
//! 4. **Performance**: Optimized for both compile-time and runtime performance
//! 5. **Safety**: Comprehensive validation and error handling throughout
//! 6. **Debugging**: Full debug information support with source location tracking
//!
//! ## Usage
//!
//! ```rust
//! use prism_codegen::backends::llvm::{LLVMBackend, LLVMBackendConfig};
//!
//! let config = LLVMBackendConfig::default();
//! let mut backend = LLVMBackend::new(config)?;
//! let result = backend.generate_code(&pir_module)?;
//! ```
//!
//! ## Integration with Prism Runtime
//!
//! The LLVM backend integrates seamlessly with the Prism runtime system,
//! providing capability validation, effect tracking, and security enforcement
//! at the LLVM IR level.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Core modules
pub mod core;
pub mod types;
pub mod instructions;
pub mod optimization;
pub mod validation;
pub mod runtime;
pub mod debug_info;
pub mod target_machine;

// Tests module
#[cfg(test)]
pub mod tests;

// Re-export main types and functions
pub use core::{LLVMBackend, LLVMBackendConfig, LLVMModule, LLVMFunction, CodeGenResult};
pub use types::{LLVMType, LLVMTypeSystem, LLVMTypeConfig, LLVMTargetArch, LLVMOptimizationLevel};
pub use instructions::{LLVMInstructionGenerator, LLVMInstructionConfig, PIRExpression, PIRStatement};
pub use optimization::{LLVMOptimizer, LLVMOptimizerConfig, OptimizationResults};
pub use validation::{LLVMValidator, LLVMValidatorConfig, ValidationResults};
pub use runtime::{LLVMRuntime, LLVMRuntimeConfig, RuntimeFunction, AllocatorType};
pub use debug_info::{LLVMDebugInfo, LLVMDebugConfig, DebugFunction, DebugVariable, SourceLocation};
pub use target_machine::{LLVMTargetMachine, LLVMTargetConfig};

/// LLVM backend error types
#[derive(Debug, Clone)]
pub enum LLVMError {
    /// Target architecture not supported
    UnsupportedTarget(String),
    /// Type conversion error
    TypeConversionError(String),
    /// Unsupported type
    UnsupportedType(String),
    /// Undefined variable
    UndefinedVariable(String),
    /// Invalid assignment target
    InvalidAssignmentTarget,
    /// Immutable assignment
    ImmutableAssignment(String),
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Missing runtime function
    MissingRuntimeFunction(String),
    /// Invalid capability type
    InvalidCapabilityType(String),
    /// Invalid effect type
    InvalidEffectType(String),
    /// Invalid security check
    InvalidSecurityCheck(String),
    /// Missing compile unit
    MissingCompileUnit,
    /// Validation failed
    ValidationFailed(ValidationResults),
    /// Optimization failed
    OptimizationFailed(String),
    /// Target machine error
    TargetMachineError(String),
    /// Assembly generation error
    AssemblyGenerationError(String),
    /// Object file generation error
    ObjectFileGenerationError(String),
    /// IO error
    IoError(String),
    /// Generic error
    Generic(String),
}

impl std::fmt::Display for LLVMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLVMError::UnsupportedTarget(target) => write!(f, "Unsupported target architecture: {}", target),
            LLVMError::TypeConversionError(msg) => write!(f, "Type conversion error: {}", msg),
            LLVMError::UnsupportedType(type_name) => write!(f, "Unsupported type: {}", type_name),
            LLVMError::UndefinedVariable(var) => write!(f, "Undefined variable: {}", var),
            LLVMError::InvalidAssignmentTarget => write!(f, "Invalid assignment target"),
            LLVMError::ImmutableAssignment(var) => write!(f, "Cannot assign to immutable variable: {}", var),
            LLVMError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            LLVMError::MissingRuntimeFunction(func) => write!(f, "Missing runtime function: {}", func),
            LLVMError::InvalidCapabilityType(cap) => write!(f, "Invalid capability type: {}", cap),
            LLVMError::InvalidEffectType(effect) => write!(f, "Invalid effect type: {}", effect),
            LLVMError::InvalidSecurityCheck(check) => write!(f, "Invalid security check: {}", check),
            LLVMError::MissingCompileUnit => write!(f, "Missing compile unit for debug information"),
            LLVMError::ValidationFailed(results) => write!(f, "LLVM IR validation failed: {} errors", results.errors.len()),
            LLVMError::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
            LLVMError::TargetMachineError(msg) => write!(f, "Target machine error: {}", msg),
            LLVMError::AssemblyGenerationError(msg) => write!(f, "Assembly generation error: {}", msg),
            LLVMError::ObjectFileGenerationError(msg) => write!(f, "Object file generation error: {}", msg),
            LLVMError::IoError(msg) => write!(f, "IO error: {}", msg),
            LLVMError::Generic(msg) => write!(f, "LLVM backend error: {}", msg),
        }
    }
}

impl std::error::Error for LLVMError {}

/// LLVM backend result type
pub type LLVMResult<T> = Result<T, LLVMError>;

// Compatibility with existing error system
impl From<LLVMError> for crate::CodeGenError {
    fn from(err: LLVMError) -> Self {
        crate::CodeGenError::CodeGenerationError {
            target: "LLVM".to_string(),
            message: err.to_string(),
        }
    }
} 