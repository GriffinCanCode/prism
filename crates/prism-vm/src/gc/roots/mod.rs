//! Modular Root Set Management for Prism VM Garbage Collection
//!
//! This module provides comprehensive root set management for the Prism VM garbage collector.
//! The design is modular and performance-oriented, with specialized components for:
//!
//! - **Root Manager**: Central coordinator for all root sources
//! - **Stack Scanner**: Precise scanning of execution stacks and JIT frames  
//! - **Global Roots**: Management of global variables and static data
//! - **Root Types**: Type definitions and data structures
//! - **Root Interfaces**: Clean abstractions for root operations
//! - **Platform Scanner**: Platform-specific stack boundary detection
//! - **Security Integration**: Capability-aware root management
//! - **Performance Analytics**: Root scanning performance monitoring
//!
//! ## Architecture
//!
//! The root set subsystem follows a layered architecture:
//! ```
//! ┌─────────────────┐
//! │   RootManager   │  ← Main coordinator and public interface
//! ├─────────────────┤
//! │ Stack │ Global  │  ← Root source implementations
//! │Scanner│ Roots   │
//! ├─────────────────┤  
//! │ Platform Scanner│  ← Platform-specific stack detection
//! ├─────────────────┤
//! │ Security Mgr    │  ← Capability-based access control
//! ├─────────────────┤
//! │ Analytics       │  ← Performance monitoring
//! └─────────────────┘
//! ```
//!
//! ## Root Sources
//!
//! The system identifies roots from multiple sources:
//! - **Execution Stack**: Local variables, temporaries, and upvalues in stack frames
//! - **JIT Compiled Code**: Stack maps and deoptimization state
//! - **Global Variables**: Module-level constants and static data
//! - **Capability Tokens**: Security capability and effect handles
//! - **Thread-Local Storage**: Per-thread root sets
//!
//! ## Design Principles
//!
//! 1. **Precise Scanning**: Uses Prism's well-defined type system for accurate root identification
//! 2. **Security-First**: All operations respect capability-based security constraints
//! 3. **Performance-Optimized**: Minimizes GC pause time through efficient scanning
//! 4. **Mixed Execution**: Handles both interpreter and JIT compiled code seamlessly
//! 5. **Modular Design**: Each component can be tested and optimized independently

pub mod types;
pub mod interfaces;
pub mod manager;
pub mod stack_scanner;
pub mod global_roots;
pub mod platform_scanner;
pub mod security;
pub mod analytics;

#[cfg(test)]
mod tests;

// Re-export key types and traits for convenience
pub use types::*;
pub use interfaces::*;
pub use manager::RootManager;
pub use stack_scanner::StackScanner;
pub use global_roots::GlobalRootManager;
pub use platform_scanner::PlatformStackScanner;
pub use security::RootSecurityManager;
pub use analytics::RootAnalytics;

use crate::{VMResult, PrismVMError};
use std::collections::HashSet;

/// Legacy compatibility - re-export the old RootSet for existing code
pub use manager::RootManager as RootSet;

/// Factory for creating different types of root managers with safety validation
pub struct RootManagerFactory;

impl RootManagerFactory {
    /// Create a new root manager with default settings
    pub fn create_default() -> VMResult<RootManager> {
        RootManager::new()
    }
    
    /// Create a root manager with custom configuration
    pub fn create_with_config(config: RootManagerConfig) -> VMResult<RootManager> {
        RootManager::with_config(config)
    }
    
    /// Create a root manager optimized for low-latency scenarios
    pub fn create_low_latency() -> VMResult<RootManager> {
        let config = RootManagerConfig {
            enable_concurrent_scanning: true,
            enable_incremental_scanning: true,
            max_scan_time_us: 100, // 100 microseconds max
            enable_analytics: false, // Minimal overhead
            security_level: SecurityLevel::Basic,
            ..Default::default()
        };
        RootManager::with_config(config)
    }
    
    /// Create a root manager optimized for debugging and analysis
    pub fn create_debug() -> VMResult<RootManager> {
        let config = RootManagerConfig {
            enable_analytics: true,
            enable_validation: true,
            enable_detailed_logging: true,
            security_level: SecurityLevel::Strict,
            ..Default::default()
        };
        RootManager::with_config(config)
    }
} 