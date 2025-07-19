//! Security System
//!
//! Complete security system including capability management, information flow control,
//! and trust management

pub mod capabilities;
pub mod information_flow;
pub mod trust;
pub mod context;

use prism_common::span::Span;
use std::collections::HashMap;
use thiserror::Error;

// Re-exports for convenience
pub use capabilities::{Capability, Capability as ObjectCapability, CapabilitySet};
pub use information_flow::{InformationFlowControl, SecurityLattice, SecurityLevel};
pub use trust::{TrustManager, TrustLevel, TrustPolicy};
pub use context::{SecureExecutionContext, SecurityEvent, SecurityAuditLog};

/// Complete security system for Prism effects
#[derive(Debug)]
pub struct SecuritySystem {
    /// Capability management
    pub capabilities: capabilities::CapabilityManager,
    /// Information flow control
    pub information_flow: information_flow::InformationFlowControl,
    /// Trust management
    pub trust: trust::TrustManager,
    /// Security audit log
    pub audit_log: context::SecurityAuditLog,
}

impl SecuritySystem {
    /// Create new security system
    pub fn new() -> Self {
        Self {
            capabilities: capabilities::CapabilityManager::new(),
            information_flow: information_flow::InformationFlowControl::new(),
            trust: trust::TrustManager::new(),
            audit_log: context::SecurityAuditLog::new(),
        }
    }

    /// Validate a security operation
    pub fn validate_operation(
        &mut self,
        operation: &SecurityOperation,
        context: &SecureExecutionContext,
    ) -> Result<SecurityValidationResult, SecurityError> {
        // Check capabilities
        let capability_result = self.capabilities.validate_capabilities(
            &operation.required_capabilities,
            &context.available_capabilities,
        )?;

        // Check information flow
        let flow_result = self.information_flow.validate_flow(
            &operation.information_flows,
            &context.security_level,
        )?;

        // Check trust requirements
        let trust_result = self.trust.validate_trust(
            &operation.trust_requirements,
            &context.trust_context,
        )?;

        // Log the validation
        self.audit_log.log_validation(SecurityEvent {
            operation_id: operation.id.clone(),
            timestamp: std::time::SystemTime::now(),
            result: if capability_result.valid && flow_result.valid && trust_result.valid {
                "ALLOWED".to_string()
            } else {
                "DENIED".to_string()
            },
            context: context.clone(),
        });

        Ok(SecurityValidationResult {
            valid: capability_result.valid && flow_result.valid && trust_result.valid,
            capability_result,
            flow_result,
            trust_result,
            audit_entry_id: self.audit_log.entries.len() - 1,
        })
    }

    /// Get security statistics
    pub fn get_stats(&self) -> SecurityStats {
        SecurityStats {
            total_capabilities: self.capabilities.get_capability_count(),
            active_contexts: self.capabilities.get_active_context_count(),
            flow_violations: self.information_flow.get_violation_count(),
            trust_violations: self.trust.get_violation_count(),
            total_validations: self.audit_log.entries.len(),
        }
    }
}

impl Default for SecuritySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// A security operation that needs validation
#[derive(Debug, Clone)]
pub struct SecurityOperation {
    /// Unique operation identifier
    pub id: String,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Information flows involved
    pub information_flows: Vec<information_flow::InformationFlow>,
    /// Trust requirements
    pub trust_requirements: trust::TrustRequirements,
    /// Operation metadata
    pub metadata: HashMap<String, String>,
}

/// Result of security validation
#[derive(Debug)]
pub struct SecurityValidationResult {
    /// Whether the operation is valid
    pub valid: bool,
    /// Capability validation result
    pub capability_result: capabilities::CapabilityValidationResult,
    /// Information flow validation result
    pub flow_result: information_flow::FlowValidationResult,
    /// Trust validation result
    pub trust_result: trust::TrustValidationResult,
    /// ID of the audit log entry
    pub audit_entry_id: usize,
}

/// Security system statistics
#[derive(Debug)]
pub struct SecurityStats {
    /// Total number of capabilities
    pub total_capabilities: usize,
    /// Number of active security contexts
    pub active_contexts: usize,
    /// Number of information flow violations
    pub flow_violations: usize,
    /// Number of trust violations
    pub trust_violations: usize,
    /// Total number of security validations performed
    pub total_validations: usize,
}

/// Security system errors
#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("Capability error: {0}")]
    Capability(#[from] capabilities::CapabilityError),
    
    #[error("Information flow error: {0}")]
    InformationFlow(#[from] information_flow::FlowError),
    
    #[error("Trust error: {0}")]
    Trust(#[from] trust::TrustError),
    
    #[error("Security context error: {0}")]
    Context(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
} 