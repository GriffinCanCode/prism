//! Async Runtime System - Structured Concurrency for I/O Operations
//!
//! This module implements the async/await runtime as specified in PLD-005, providing:
//! - **Structured concurrency**: All async operations have clear lifetimes
//! - **Cancellation propagation**: Cancellation tokens flow through async operations
//! - **Effect integration**: Async operations declare their effects
//! - **Capability checking**: All async operations require explicit capabilities
//! - **AI metadata**: Rich metadata for AI comprehension of async patterns
//!
//! ## Module Organization
//!
//! The async runtime is organized into focused, cohesive modules:
//! - `types` - Core async types and identifiers
//! - `cancellation` - Cancellation token system for structured cancellation
//! - `context` - Execution context for async operations
//! - `scheduler` - Task scheduling and priority management
//! - `runtime` - Main async runtime coordinator
//!
//! ## Design Principles
//!
//! 1. **Structured Concurrency**: All async operations have clear parent-child relationships
//! 2. **Capability-Based Security**: All operations require explicit capabilities
//! 3. **Effect Tracking**: All async operations declare their effects
//! 4. **Resource Management**: Proper cleanup and resource tracking
//! 5. **AI Comprehensible**: Rich metadata for external AI analysis

pub mod types;
pub mod cancellation;
pub mod context;
pub mod scheduler;
pub mod runtime;

// Re-export public API
pub use types::{TaskId, TaskPriority, AsyncTaskMetadata, AsyncResult, AsyncError};
pub use cancellation::CancellationToken;
pub use context::AsyncContext;
pub use scheduler::{AsyncScheduler, SchedulerMetrics};
pub use runtime::{AsyncRuntime, AsyncRuntimeMetrics};
pub use types::AsyncHandle; 