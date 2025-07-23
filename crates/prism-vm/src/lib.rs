//! Prism Virtual Machine - Bytecode Compilation and Execution
//!
//! This crate implements the Prism Virtual Machine (PVM) that provides:
//! - Bytecode compilation from PIR
//! - Stack-based bytecode interpreter
//! - JIT compilation for performance optimization
//! - Unified debugging and profiling capabilities
//! - Capability-based security enforcement
//!
//! ## Architecture
//!
//! The PVM follows a modular architecture with clear separation of concerns:
//!
//! - [`bytecode`] - Bytecode format definitions and serialization
//! - [`execution`] - Execution engine (interpreter and JIT compiler)
//! - [`runtime`] - Runtime services (GC, threads, I/O)
//! - [`tools`] - Development tools (disassembler, debugger, profiler)
//!
//! ## Design Principles
//!
//! 1. **Semantic Preservation**: Maintains all semantic information from PIR
//! 2. **Capability Enforcement**: Integrates with Prism's capability system
//! 3. **Effect Tracking**: Monitors all computational effects
//! 4. **Performance Optimization**: Supports both interpretation and JIT compilation
//! 5. **Debugging Support**: Rich debugging and profiling capabilities
//! 6. **Modular Design**: Each component has a single, clear responsibility

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Core VM modules
pub mod bytecode;
pub mod execution;
pub mod runtime;
pub mod tools;

// Re-export main types for easy access
pub use bytecode::{
    PrismBytecode, PrismOpcode, Instruction, ConstantPool, FunctionDefinition,
    BytecodeVersion, ModuleMetadata,
};
pub use execution::{
    PrismVM, Interpreter, ExecutionResult, VMError, VMConfig,
};
pub use runtime::{
    VMMemory, GarbageCollector, ThreadManager, IOManager,
};
pub use tools::{
    Disassembler, Debugger, Profiler, ProfilerReport,
};

// Error types
use thiserror::Error;

/// VM-specific error types
#[derive(Error, Debug)]
pub enum PrismVMError {
    /// Bytecode compilation error
    #[error("Bytecode compilation failed: {message}")]
    CompilationError { message: String },

    /// Bytecode execution error
    #[error("Execution failed: {message}")]
    ExecutionError { message: String },

    /// Runtime error
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    /// Invalid bytecode format
    #[error("Invalid bytecode: {message}")]
    InvalidBytecode { message: String },

    /// Capability violation
    #[error("Capability violation: {message}")]
    CapabilityViolation { message: String },

    /// Effect system error
    #[error("Effect system error: {message}")]
    EffectError { message: String },

    /// JIT compilation error
    #[error("JIT compilation failed: {message}")]
    JITError { message: String },

    /// I/O error
    #[error("I/O error: {source}")]
    IOError {
        #[from]
        source: std::io::Error,
    },

    /// Serialization error
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        source: bincode::Error,
    },
}

/// Result type for VM operations
pub type VMResult<T> = Result<T, PrismVMError>;

/// VM version information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VMVersion {
    /// Major version
    pub major: u16,
    /// Minor version
    pub minor: u16,
    /// Patch version
    pub patch: u16,
    /// Pre-release identifier
    pub pre_release: Option<String>,
}

impl VMVersion {
    /// Current VM version
    pub const CURRENT: VMVersion = VMVersion {
        major: 0,
        minor: 1,
        patch: 0,
        pre_release: Some("alpha".to_string()),
    };

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &VMVersion) -> bool {
        // For now, require exact major version match
        self.major == other.major
    }
}

impl std::fmt::Display for VMVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre_release {
            write!(f, "-{}", pre)?;
        }
        Ok(())
    }
} 