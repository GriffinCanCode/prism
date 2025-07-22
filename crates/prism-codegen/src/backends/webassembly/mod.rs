//! Modular WebAssembly Code Generation Backend
//!
//! This module implements a comprehensive WebAssembly backend with proper separation
//! of concerns, following 2025 WebAssembly best practices and Prism's design principles.
//!
//! ## Architecture
//!
//! The WebAssembly backend is organized into focused modules:
//!
//! - [`core`] - Core WebAssembly backend implementation
//! - [`string_handler`] - String constant management and optimization
//! - [`memory`] - Memory layout and management utilities
//! - [`types`] - WebAssembly type system and PIR type conversion
//! - [`instructions`] - WASM instruction generation from PIR expressions
//! - [`runtime`] - Runtime integration and capability system
//! - [`optimization`] - WASM-specific optimization passes
//! - [`validation`] - WASM code validation and verification
//!
//! ## Design Principles
//!
//! 1. **Conceptual Cohesion**: Each module has a single, clear responsibility
//! 2. **Loose Coupling**: Modules interact through well-defined interfaces
//! 3. **Shared Utilities**: Common functionality is abstracted for reuse
//! 4. **Target-Specific Logic**: WASM-specific details are properly encapsulated
//! 5. **Testability**: Each module can be tested independently

pub mod core;
pub mod string_handler;
pub mod memory;
pub mod types;
pub mod instructions;
pub mod runtime;
pub mod optimization;
pub mod validation;

// Re-export the main backend for backward compatibility
pub use core::WebAssemblyBackend;

// Re-export commonly used types
pub use types::{WasmType, WasmOptimizationLevel, WasmRuntimeTarget, WasmFeatures};
pub use memory::WasmMemoryLayout;
pub use string_handler::StringConstantManager;
pub use runtime::{WasmRuntimeIntegration, WasmCapabilitySystem};
pub use optimization::WasmOptimizer;
pub use validation::WasmValidator;

/// WebAssembly backend configuration combining all module configurations
#[derive(Debug, Clone)]
pub struct WasmBackendConfig {
    /// Core backend configuration
    pub core_config: crate::backends::CodeGenConfig,
    /// Runtime target configuration
    pub runtime_target: WasmRuntimeTarget,
    /// WebAssembly features to enable
    pub features: WasmFeatures,
    /// Memory layout configuration
    pub memory_layout: WasmMemoryLayout,
    /// String management configuration
    pub string_config: string_handler::StringManagerConfig,
    /// Optimization configuration
    pub optimization_config: optimization::WasmOptimizationConfig,
}

impl Default for WasmBackendConfig {
    fn default() -> Self {
        Self {
            core_config: crate::backends::CodeGenConfig::default(),
            runtime_target: WasmRuntimeTarget::default(),
            features: WasmFeatures::default(),
            memory_layout: WasmMemoryLayout::default(),
            string_config: string_handler::StringManagerConfig::default(),
            optimization_config: optimization::WasmOptimizationConfig::default(),
        }
    }
}

/// Result type for WebAssembly backend operations
pub type WasmResult<T> = Result<T, WasmError>;

/// WebAssembly backend specific errors
#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    #[error("WASM type conversion error: {message}")]
    TypeConversion { message: String },
    
    #[error("WASM instruction generation error: {message}")]
    InstructionGeneration { message: String },
    
    #[error("WASM runtime integration error: {message}")]
    RuntimeIntegration { message: String },
    
    #[error("WASM memory layout error: {message}")]
    MemoryLayout { message: String },
    
    #[error("WASM string management error: {message}")]
    StringManagement { message: String },
    
    #[error("WASM optimization error: {message}")]
    Optimization { message: String },
    
    #[error("WASM validation error: {message}")]
    Validation { message: String },
    
    #[error("General WASM backend error: {message}")]
    General { message: String },
}

impl From<WasmError> for crate::CodeGenError {
    fn from(err: WasmError) -> Self {
        crate::CodeGenError::CodeGenerationError {
            target: "WebAssembly".to_string(),
            message: err.to_string(),
        }
    }
} 