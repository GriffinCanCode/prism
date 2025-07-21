//! Security Enforcement - Policy & Isolation Management
//!
//! This module implements comprehensive security management that coordinates policy enforcement
//! with component isolation. It embodies the business capability of **security governance**
//! by ensuring all runtime operations comply with security policies and isolation boundaries.
//!
//! ## Business Capability: Security Enforcement
//!
//! **Core Responsibility**: Enforce security policies and maintain isolation boundaries.
//!
//! **Key Business Functions**:
//! - **Policy Enforcement**: Apply and monitor security policies in real-time
//! - **Component Isolation**: Maintain secure boundaries between components
//! - **Threat Detection**: Identify and respond to security threats
//! - **Incident Response**: Handle security violations with appropriate responses
//! - **Security Auditing**: Maintain comprehensive security audit trails
//!
//! ## Conceptual Cohesion
//!
//! This module maintains high conceptual cohesion by focusing on **security governance**.
//! Policy enforcement and component isolation are closely related because:
//! - Isolation policies define component boundaries
//! - Policy violations may require component isolation
//! - Both require real-time monitoring and response
//! - Both contribute to security audit trails
//!
//! The module does NOT handle:
//! - Authority management (handled by `authority` module)
//! - Resource allocation (handled by `resources` module)
//! - Platform execution (handled by `platform` module)
//! - Business intelligence (handled by `intelligence` module)

use crate::{authority, platform::execution::ExecutionContext};
use std::sync::Arc;
use thiserror::Error;

pub mod enforcement;
pub mod isolation;

/// Unified security enforcer that coordinates policy enforcement and isolation
#[derive(Debug)]
pub struct SecurityEnforcer {
    /// Policy enforcement system
    policy_enforcer: Arc<enforcement::SecurityPolicyEnforcer>,
    
    /// Component isolation manager
    isolation_manager: Arc<isolation::ComponentIsolationManager>,
    
    /// Security coordination system
    security_coordinator: Arc<SecurityCoordinator>,
}

impl SecurityEnforcer {
    /// Create a new security enforcer
    pub fn new() -> Result<Self, SecurityError> {
        let policy_enforcer = Arc::new(enforcement::SecurityPolicyEnforcer::new()?);
        let isolation_manager = Arc::new(isolation::ComponentIsolationManager::new()?);
        let security_coordinator = Arc::new(SecurityCoordinator::new());

        Ok(Self {
            policy_enforcer,
            isolation_manager,
            security_coordinator,
        })
    }

    /// Get the number of isolated components
    pub fn component_count(&self) -> usize {
        self.isolation_manager.component_count()
    }

    /// Get the number of security violations
    pub fn violation_count(&self) -> usize {
        self.policy_enforcer.violation_count()
    }

    /// Create an isolated component with security policies
    pub fn create_secure_component(
        &self,
        spec: &isolation::ComponentSpec,
        capabilities: authority::CapabilitySet,
        security_policies: Vec<enforcement::SecurityPolicy>,
    ) -> Result<isolation::ComponentHandle, SecurityError> {
        // Create isolated component
        let component_handle = self.isolation_manager
            .create_component(spec, capabilities)
            .map_err(SecurityError::Isolation)?;

        // Apply security policies to the component
        for policy in security_policies {
            self.policy_enforcer
                .apply_policy_to_component(&component_handle, policy)
                .map_err(SecurityError::Enforcement)?;
        }

        // Register with security coordinator
        self.security_coordinator.register_secure_component(&component_handle);

        Ok(component_handle)
    }

    /// Enforce security policies on a runtime operation
    pub fn enforce_security(
        &self,
        operation: &enforcement::RuntimeOperation,
        context: &ExecutionContext,
    ) -> Result<enforcement::PolicyDecision, SecurityError> {
        // Check isolation boundaries
        self.isolation_manager
            .validate_operation_boundaries(operation, context)
            .map_err(SecurityError::Isolation)?;

        // Enforce policies
        let decision = self.policy_enforcer
            .enforce_policies(operation, context)
            .map_err(SecurityError::Enforcement)?;

        // Coordinate security response if needed
        if decision.requires_action() {
            self.security_coordinator.coordinate_security_response(&decision, operation, context);
        }

        Ok(decision)
    }
}

/// Security coordinator that manages interactions between enforcement and isolation
#[derive(Debug)]
struct SecurityCoordinator {
    // Implementation would coordinate between policy enforcement and isolation
}

impl SecurityCoordinator {
    fn new() -> Self {
        Self {}
    }

    fn register_secure_component(&self, _handle: &isolation::ComponentHandle) {
        // Register component with security coordination
    }

    fn coordinate_security_response(
        &self,
        _decision: &enforcement::PolicyDecision,
        _operation: &enforcement::RuntimeOperation,
        _context: &ExecutionContext,
    ) {
        // Coordinate response between enforcement and isolation systems
    }
}

/// Security management errors
#[derive(Debug, Error)]
pub enum SecurityError {
    /// Policy enforcement error
    #[error("Policy enforcement error: {0}")]
    Enforcement(#[from] enforcement::SecurityError),

    /// Component isolation error
    #[error("Isolation error: {0}")]
    Isolation(#[from] isolation::IsolationError),

    /// Security coordination error
    #[error("Security coordination error: {message}")]
    Coordination { message: String },

    /// Generic security error
    #[error("Security error: {message}")]
    Generic { message: String },
} 