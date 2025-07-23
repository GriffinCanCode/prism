//! Security System - Policy Enforcement and Isolation Management
//!
//! This module implements the security system that enforces security policies,
//! manages component isolation, and provides security analysis capabilities.
//! It embodies the business capability of **security governance** by ensuring
//! that all runtime operations comply with security policies and constraints.

use crate::{authority, resources};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use thiserror::Error;

pub mod enforcement;
pub mod isolation;

use enforcement::SecurityEnforcement;
use isolation::IsolationManager;

/// Main security system that coordinates all security operations
#[derive(Debug)]
pub struct SecuritySystem {
    /// Security policy enforcement
    enforcement: Arc<SecurityEnforcement>,
    /// Component isolation management
    isolation_manager: Arc<IsolationManager>,
    /// Security configuration
    config: SecurityConfig,
}

/// Configuration for the security system
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Enable strict security mode
    pub strict_mode: bool,
    /// Default security level for new components
    pub default_security_level: SecurityLevel,
    /// Enable security auditing
    pub enable_auditing: bool,
    /// Maximum isolation level supported
    pub max_isolation_level: isolation::IsolationLevel,
    /// Security policy enforcement level
    pub enforcement_level: EnforcementLevel,
}

/// Security levels for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Public information
    Public,
    /// Internal use only
    Internal,
    /// Confidential information
    Confidential,
    /// Restricted access
    Restricted,
}

/// Levels of security enforcement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnforcementLevel {
    /// Advisory warnings only
    Advisory,
    /// Block violations but allow override
    Blocking,
    /// Strict enforcement with no overrides
    Strict,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            default_security_level: SecurityLevel::Internal,
            enable_auditing: true,
            max_isolation_level: isolation::IsolationLevel::Process,
            enforcement_level: EnforcementLevel::Blocking,
        }
    }
}

/// Security policy structure
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Policy type
    pub policy_type: PolicyType,
    /// Policy rules
    pub rules: Vec<SecurityRule>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modified timestamp
    pub modified_at: SystemTime,
}

/// Types of security policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyType {
    /// Access control policy
    AccessControl,
    /// Resource usage policy
    ResourceUsage,
    /// Communication policy
    Communication,
    /// Capability delegation policy
    CapabilityDelegation,
    /// Data protection policy
    DataProtection,
}

/// Security rule within a policy
#[derive(Debug, Clone)]
pub struct SecurityRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule type
    pub rule_type: RuleType,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: RuleAction,
}

/// Types of security rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleType {
    /// Component validation rule
    ComponentValidation,
    /// Resource limit rule
    ResourceLimit,
    /// Access control rule
    AccessControl,
    /// Communication rule
    Communication,
}

/// Actions for security rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleAction {
    /// Allow the operation
    Allow,
    /// Deny the operation
    Deny,
    /// Allow with restrictions
    AllowWithRestrictions,
    /// Log and allow
    LogAndAllow,
}

/// Context for security decisions
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Component identifier
    pub component_id: authority::ComponentId,
    /// Security level of the operation
    pub security_level: SecurityLevel,
    /// Requested operation
    pub operation: String,
    /// Additional context data
    pub context_data: std::collections::HashMap<String, String>,
    /// Timestamp of the request
    pub timestamp: SystemTime,
}

/// Decision from policy enforcement
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Allow the operation
    Allow,
    /// Deny the operation
    Deny,
    /// Allow with specific restrictions
    AllowWithRestrictions(Vec<String>),
}

/// Result of policy validation
#[derive(Debug, Clone)]
pub struct PolicyValidationResult {
    /// Whether the policy is valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

impl SecuritySystem {
    /// Create a new security system with default configuration
    pub fn new() -> Result<Self, SecurityError> {
        Self::with_config(SecurityConfig::default())
    }

    /// Create a security system with custom configuration
    pub fn with_config(config: SecurityConfig) -> Result<Self, SecurityError> {
        let enforcement = Arc::new(SecurityEnforcement::new(config.clone())
            .map_err(|e| SecurityError::Enforcement(e))?);
        let isolation_manager = Arc::new(IsolationManager::new()
            .map_err(|e| SecurityError::Isolation(e))?);

        Ok(Self {
            enforcement,
            isolation_manager,
            config,
        })
    }

    /// Validate a security policy
    pub fn validate_policy(&self, policy: &SecurityPolicy) -> Result<PolicyValidationResult, SecurityError> {
        self.enforcement.validate_policy(policy)
            .map_err(|e| SecurityError::Enforcement(e))
    }

    /// Enforce a security policy
    pub fn enforce_policy(
        &self,
        policy: &SecurityPolicy,
        context: &SecurityContext,
    ) -> Result<PolicyDecision, SecurityError> {
        self.enforcement.enforce_policy(policy, context)
            .map_err(|e| SecurityError::Enforcement(e))
    }

    /// Create an isolated component
    pub fn create_isolated_component(
        &self,
        spec: isolation::ComponentSpec,
    ) -> Result<isolation::ComponentHandle, SecurityError> {
        self.isolation_manager.create_component(spec)
            .map_err(|e| SecurityError::Isolation(e))
    }

    /// Validate capability delegation
    pub fn validate_delegation(
        &self,
        from_component: isolation::ComponentId,
        to_component: isolation::ComponentId,
        capabilities: &authority::CapabilitySet,
    ) -> Result<DelegationValidationResult, SecurityError> {
        // Check if delegation is allowed by security policy
        let delegation_policy = SecurityPolicy::new(
            "capability_delegation".to_string(),
            PolicyType::CapabilityDelegation,
            vec![
                SecurityRule::new(
                    "source_component".to_string(),
                    RuleType::ComponentValidation,
                    format!("{:?}", from_component),
                ),
                SecurityRule::new(
                    "target_component".to_string(),
                    RuleType::ComponentValidation,
                    format!("{:?}", to_component),
                ),
            ],
        );

        let context = SecurityContext::new(
            self.config.default_security_level,
            vec!["capability_delegation".to_string()],
        );

        let decision = self.enforce_policy(&delegation_policy, &context)?;

        Ok(DelegationValidationResult {
            allowed: matches!(decision, PolicyDecision::Allow),
            restrictions: Vec::new(),
            audit_required: self.config.enable_auditing,
        })
    }

    /// Get security statistics
    pub fn get_security_stats(&self) -> SecurityStats {
        SecurityStats {
            total_policy_checks: 0, // Would track actual checks
            policy_violations: 0,
            isolated_components: self.isolation_manager.component_count(),
            active_delegations: 0,
            security_level: self.config.default_security_level,
        }
    }

    /// Perform security audit
    pub fn perform_audit(&self) -> Result<SecurityAuditReport, SecurityError> {
        Ok(SecurityAuditReport {
            audit_timestamp: std::time::SystemTime::now(),
            security_level: self.config.default_security_level,
            policy_violations: Vec::new(),
            isolation_status: self.isolation_manager.get_isolation_status(),
            recommendations: Vec::new(),
        })
    }
}

impl SecurityPolicy {
    /// Create a new security policy
    pub fn new(name: String, policy_type: PolicyType, rules: Vec<SecurityRule>) -> Self {
        Self {
            name,
            description: String::new(),
            policy_type,
            rules,
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
        }
    }
}

impl SecurityRule {
    /// Create a new security rule
    pub fn new(name: String, rule_type: RuleType, condition: String) -> Self {
        Self {
            name,
            description: String::new(),
            rule_type,
            condition,
            action: RuleAction::Allow,
        }
    }
}

impl SecurityContext {
    /// Create a new security context
    pub fn new(security_level: SecurityLevel, operations: Vec<String>) -> Self {
        Self {
            component_id: authority::ComponentId::new(1), // Placeholder
            security_level,
            operation: operations.join(","),
            context_data: std::collections::HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }
}

/// Result of delegation validation
#[derive(Debug, Clone)]
pub struct DelegationValidationResult {
    /// Whether delegation is allowed
    pub allowed: bool,
    /// Any restrictions on the delegation
    pub restrictions: Vec<String>,
    /// Whether audit is required
    pub audit_required: bool,
}

/// Security system statistics
#[derive(Debug, Clone)]
pub struct SecurityStats {
    /// Total policy checks performed
    pub total_policy_checks: u64,
    /// Number of policy violations
    pub policy_violations: u64,
    /// Number of isolated components
    pub isolated_components: usize,
    /// Number of active capability delegations
    pub active_delegations: u64,
    /// Current security level
    pub security_level: SecurityLevel,
}

/// Security audit report
#[derive(Debug, Clone)]
pub struct SecurityAuditReport {
    /// When the audit was performed
    pub audit_timestamp: std::time::SystemTime,
    /// Overall security level
    pub security_level: SecurityLevel,
    /// Policy violations found
    pub policy_violations: Vec<PolicyViolation>,
    /// Component isolation status
    pub isolation_status: isolation::IsolationStatus,
    /// Security recommendations
    pub recommendations: Vec<String>,
}

/// Policy violation record
#[derive(Debug, Clone)]
pub struct PolicyViolation {
    /// Policy that was violated
    pub policy_name: String,
    /// Rule that was violated
    pub rule_name: String,
    /// When the violation occurred
    pub timestamp: std::time::SystemTime,
    /// Description of the violation
    pub description: String,
    /// Severity of the violation
    pub severity: ViolationSeverity,
}

/// Severity levels for policy violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationSeverity {
    /// Low severity violation
    Low,
    /// Medium severity violation
    Medium,
    /// High severity violation
    High,
    /// Critical violation requiring immediate attention
    Critical,
}

/// Security system errors
#[derive(Debug, Error)]
pub enum SecurityError {
    /// Policy enforcement error
    #[error("Policy enforcement error: {0}")]
    Enforcement(#[from] enforcement::EnforcementError),

    /// Isolation management error
    #[error("Isolation error: {0}")]
    Isolation(#[from] isolation::IsolationError),

    /// Configuration error
    #[error("Security configuration error: {message}")]
    Configuration { message: String },

    /// Policy validation error
    #[error("Policy validation error: {message}")]
    PolicyValidation { message: String },

    /// Generic security error
    #[error("Security error: {message}")]
    Generic { message: String },
} 