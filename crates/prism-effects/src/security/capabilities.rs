//! Capability Management
//!
//! Object capabilities and capability-based security

use std::collections::{HashMap, HashSet};
use thiserror::Error;

// Re-export from existing capability.rs for now
pub use crate::capability::Capability;

/// Set of capabilities
pub type CapabilitySet = Vec<Capability>;

/// Capability management system
#[derive(Debug)]
pub struct CapabilityManager {
    /// Active capability sets
    capability_sets: HashMap<String, CapabilitySet>,
    /// Capability grants and revocations
    grant_log: Vec<CapabilityGrant>,
    /// Active contexts
    active_contexts: HashMap<String, CapabilityContext>,
}

impl CapabilityManager {
    /// Create new capability manager
    pub fn new() -> Self {
        Self {
            capability_sets: HashMap::new(),
            grant_log: Vec::new(),
            active_contexts: HashMap::new(),
        }
    }

    /// Validate capabilities for an operation
    pub fn validate_capabilities(
        &self,
        required: &[String],
        available: &[Capability],
    ) -> Result<CapabilityValidationResult, CapabilityError> {
        let available_names: HashSet<String> = available.iter().map(|c| c.definition.clone()).collect();
        let missing: Vec<String> = required.iter()
            .filter(|req| !available_names.contains(*req))
            .cloned()
            .collect();

        Ok(CapabilityValidationResult {
            valid: missing.is_empty(),
            missing_capabilities: missing,
            available_capabilities: available.len(),
        })
    }

    /// Get capability count
    pub fn get_capability_count(&self) -> usize {
        self.capability_sets.values().map(|set| set.len()).sum()
    }

    /// Get active context count
    pub fn get_active_context_count(&self) -> usize {
        self.active_contexts.len()
    }
}

impl Default for CapabilityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of capability validation
#[derive(Debug)]
pub struct CapabilityValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Missing capabilities
    pub missing_capabilities: Vec<String>,
    /// Number of available capabilities
    pub available_capabilities: usize,
}

/// A capability grant record
#[derive(Debug)]
pub struct CapabilityGrant {
    /// Capability that was granted
    pub capability: String,
    /// When it was granted
    pub granted_at: std::time::SystemTime,
    /// Who granted it
    pub granted_by: String,
    /// Who received it
    pub granted_to: String,
}

/// Capability context for execution
#[derive(Debug)]
pub struct CapabilityContext {
    /// Context identifier
    pub id: String,
    /// Available capabilities
    pub capabilities: Vec<Capability>,
    /// Context creation time
    pub created_at: std::time::SystemTime,
}

/// Capability-related errors
#[derive(Debug, Error)]
pub enum CapabilityError {
    #[error("Capability not found: {0}")]
    NotFound(String),
    
    #[error("Insufficient capabilities: missing {0:?}")]
    Insufficient(Vec<String>),
    
    #[error("Capability revoked: {0}")]
    Revoked(String),
} 