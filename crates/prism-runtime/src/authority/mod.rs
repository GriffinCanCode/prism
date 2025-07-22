//! Authority Management - Capability-Based Security System
//!
//! This module implements the authority management system that provides capability-based
//! security for the Prism runtime. It embodies the business capability of **access control
//! and authorization** through explicit, time-bounded, attenuatable capabilities.
//!
//! ## Business Capability: Authority Management
//!
//! **Core Responsibility**: Manage who can do what, when, and under what constraints.
//!
//! **Key Business Functions**:
//! - **Capability Issuance**: Grant explicit authority to perform operations
//! - **Authority Validation**: Verify operations are authorized by valid capabilities  
//! - **Capability Attenuation**: Weaken capabilities without strengthening them
//! - **Audit Trail**: Track all capability usage for security analysis
//! - **Time-Bounded Authority**: Ensure capabilities expire and cannot persist indefinitely
//!
//! ## Conceptual Cohesion
//!
//! This module maintains high conceptual cohesion by focusing solely on **authority management**.
//! It does NOT handle:
//! - Resource allocation (handled by `resources` module)
//! - Platform execution (handled by `platform` module)  
//! - Policy enforcement (handled by `security` module)
//! - Intelligence collection (handled by `intelligence` module)
//!
//! ## Usage Example
//!
//! ```rust
//! use prism_runtime::authority::{Capability, Authority, ConstraintSet};
//! use std::time::Duration;
//!
//! // Create a capability for file reading with constraints
//! let file_read_authority = Authority::FileSystem(FileSystemAuthority {
//!     operations: vec![FileOperation::Read].into_iter().collect(),
//!     allowed_paths: vec![PathPattern::new("/tmp/*")],
//! });
//!
//! let constraints = ConstraintSet::new()
//!     .with_time_limit(Duration::from_secs(3600))  // 1 hour limit
//!     .with_rate_limit(100);  // 100 operations per second
//!
//! let capability = Capability::new(
//!     file_read_authority,
//!     constraints,
//!     Duration::from_secs(3600),
//!     component_id,
//! );
//! ```

// Re-export the capability system with the same API
mod capability;

pub use capability::*;

/// Authority management system that coordinates all capability operations
#[derive(Debug)]
pub struct AuthoritySystem {
    capability_manager: CapabilityManager,
}

impl AuthoritySystem {
    /// Create a new authority system
    pub fn new() -> Result<Self, CapabilityError> {
        Ok(Self {
            capability_manager: CapabilityManager::new()?,
        })
    }

    /// Get the capability manager
    pub fn capability_manager(&self) -> &CapabilityManager {
        &self.capability_manager
    }
} 