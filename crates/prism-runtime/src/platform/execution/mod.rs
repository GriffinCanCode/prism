//! Multi-Target Execution System
//!
//! This module implements the execution system that can run Prism code across
//! multiple targets (TypeScript, WebAssembly, Native, PrismVM) while maintaining
//! capability-based security and effect tracking.
//!
//! ## Design Principles
//!
//! 1. **Target Agnostic**: Core execution logic independent of target platform
//! 2. **Adapter Pattern**: Target-specific adapters handle platform details
//! 3. **Capability Preservation**: Security guarantees maintained across all targets
//! 4. **Performance Optimized**: Target-specific optimizations where beneficial
//! 5. **AI-Comprehensible**: Structured execution metadata for AI analysis
//!
//! ## Module Structure
//!
//! - [`context`] - Execution context and configuration types
//! - [`adapters`] - Target-specific execution adapters
//! - [`monitoring`] - Execution monitoring and metrics collection
//! - [`manager`] - Main execution orchestration
//! - [`errors`] - Error types for execution operations

pub mod context;
pub mod adapters;
pub mod monitoring;
pub mod manager;
pub mod errors;
pub mod integration;

// Re-export commonly used types for convenience
pub use context::{
    ExecutionContext, ExecutionId, ExecutionTarget, ExecutionConfig, 
    MonitoringLevel, TargetConfig, ExecutionAIContext, SemanticState,
    TypeScriptConfig, WebAssemblyConfig, NativeConfig, PrismVMConfig,
};

pub use adapters::{
    TargetAdapter, TargetAdapterImpl, TypeScriptAdapter, WebAssemblyAdapter, 
    NativeAdapter, PrismVMAdapter, AdapterRegistry,
};

pub use monitoring::{
    ExecutionMonitor, MonitoringHandle, ExecutionMetrics, ExecutionEvent,
    ResourceUsage, MonitorConfig, MonitoringStats,
};

pub use manager::{
    ExecutionManager, GeneratedCode, GeneratedContent, GenerationMetadata,
    ExecutionStats,
};

pub use errors::{ExecutionError, ExecutionResult};

pub use integration::{
    BackendIntegrationBridge, BackendIntegration, IntegrationCapabilities,
    ContentType, PerformanceTier,
}; 