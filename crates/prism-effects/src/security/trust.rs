//! Trust Management
//!
//! Trust levels and trust-based security policies

use std::collections::HashMap;
use thiserror::Error;

// Re-export from existing security_trust.rs for now
pub use crate::security_trust::{TrustLevel, TrustPolicy};

/// Trust management system
#[derive(Debug)]
pub struct TrustManager {
    /// Trust policies
    policies: Vec<TrustPolicy>,
    /// Trust contexts
    contexts: HashMap<String, TrustContext>,
    /// Violation counter
    violation_count: usize,
}

impl TrustManager {
    /// Create new trust manager
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            contexts: HashMap::new(),
            violation_count: 0,
        }
    }

    /// Validate trust requirements
    pub fn validate_trust(
        &mut self,
        requirements: &TrustRequirements,
        context: &TrustContext,
    ) -> Result<TrustValidationResult, TrustError> {
        let mut valid = true;
        let mut violations = Vec::new();

        // Check minimum trust level
        if context.level < requirements.minimum_level {
            valid = false;
            violations.push(format!(
                "Context trust level {:?} below required minimum {:?}",
                context.level, requirements.minimum_level
            ));
            self.violation_count += 1;
        }

        // Check required trust categories
        for category in &requirements.required_categories {
            if !context.categories.contains(category) {
                valid = false;
                violations.push(format!(
                    "Context missing required trust category: {}",
                    category
                ));
                self.violation_count += 1;
            }
        }

        Ok(TrustValidationResult {
            valid,
            violations,
            context_level: context.level.clone(),
        })
    }

    /// Get violation count
    pub fn get_violation_count(&self) -> usize {
        self.violation_count
    }
}

impl Default for TrustManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Trust requirements for an operation
#[derive(Debug, Clone)]
pub struct TrustRequirements {
    /// Minimum trust level required
    pub minimum_level: TrustLevel,
    /// Required trust categories
    pub required_categories: Vec<String>,
}

/// Trust context for validation
#[derive(Debug, Clone)]
pub struct TrustContext {
    /// Trust level
    pub level: TrustLevel,
    /// Trust categories
    pub categories: Vec<String>,
    /// Context metadata
    pub metadata: HashMap<String, String>,
}

/// Result of trust validation
#[derive(Debug)]
pub struct TrustValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Any violations found
    pub violations: Vec<String>,
    /// Context trust level
    pub context_level: TrustLevel,
}

/// Trust-related errors
#[derive(Debug, Error)]
pub enum TrustError {
    #[error("Trust level too low: {:?}", .0)]
    InsufficientTrust(TrustLevel),
    
    #[error("Trust category missing: {0}")]
    MissingCategory(String),
    
    #[error("Trust policy violation: {0}")]
    PolicyViolation(String),
} 