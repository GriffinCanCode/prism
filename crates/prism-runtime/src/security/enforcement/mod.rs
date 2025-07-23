//! Security Policy Enforcement - Focused Implementation
//!
//! This module implements security policy enforcement following the same modular
//! philosophy as other runtime components. It focuses solely on policy validation
//! and enforcement without duplicating logic from other systems.

use crate::security::{SecurityConfig, SecurityPolicy, SecurityContext, PolicyDecision, PolicyValidationResult};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::SystemTime;
use thiserror::Error;

/// Security enforcement system that validates and enforces policies
#[derive(Debug)]
pub struct SecurityEnforcement {
    /// Configuration for enforcement behavior
    config: SecurityConfig,
    /// Policy validation statistics
    policy_checks: AtomicU64,
    /// Policy violations count
    policy_violations: AtomicU64,
    /// Enforcement start time
    started_at: SystemTime,
}

impl SecurityEnforcement {
    /// Create a new security enforcement system
    pub fn new(config: SecurityConfig) -> Result<Self, EnforcementError> {
        Ok(Self {
            config,
            policy_checks: AtomicU64::new(0),
            policy_violations: AtomicU64::new(0),
            started_at: SystemTime::now(),
        })
    }

    /// Validate a security policy structure and content
    pub fn validate_policy(&self, policy: &SecurityPolicy) -> Result<PolicyValidationResult, EnforcementError> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Basic structural validation
        if policy.name.is_empty() {
            errors.push("Policy name cannot be empty".to_string());
        }

        if policy.name.len() > 255 {
            errors.push("Policy name too long (max 255 characters)".to_string());
        }

        if policy.rules.is_empty() {
            warnings.push("Policy has no rules - it will have no effect".to_string());
        }

        // Rule validation
        for (i, rule) in policy.rules.iter().enumerate() {
            if rule.name.is_empty() {
                errors.push(format!("Rule {} has empty name", i));
            }
            if rule.condition.is_empty() {
                errors.push(format!("Rule {} has empty condition", i));
            }
        }

        // Policy type specific validation
        match policy.policy_type {
            crate::security::PolicyType::CapabilityDelegation => {
                if !policy.rules.iter().any(|r| r.rule_type == crate::security::RuleType::ComponentValidation) {
                    warnings.push("Capability delegation policy should include component validation rules".to_string());
                }
            }
            crate::security::PolicyType::ResourceUsage => {
                if !policy.rules.iter().any(|r| r.rule_type == crate::security::RuleType::ResourceLimit) {
                    warnings.push("Resource usage policy should include resource limit rules".to_string());
                }
            }
            _ => {} // Other policy types don't have specific requirements yet
        }

        Ok(PolicyValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Enforce a security policy against a context
    pub fn enforce_policy(
        &self,
        policy: &SecurityPolicy,
        context: &SecurityContext,
    ) -> Result<PolicyDecision, EnforcementError> {
        // Increment policy check counter
        self.policy_checks.fetch_add(1, Ordering::Relaxed);

        // Enforcement logic based on configuration and context
        let decision = match self.config.enforcement_level {
            crate::security::EnforcementLevel::Strict => {
                self.enforce_strict_policy(policy, context)
            }
            crate::security::EnforcementLevel::Blocking => {
                self.enforce_blocking_policy(policy, context)
            }
            crate::security::EnforcementLevel::Advisory => {
                self.enforce_advisory_policy(policy, context)
            }
        };

        // Track violations
        if matches!(decision, PolicyDecision::Deny) {
            self.policy_violations.fetch_add(1, Ordering::Relaxed);
        }

        Ok(decision)
    }

    /// Get enforcement statistics
    pub fn get_stats(&self) -> EnforcementStats {
        EnforcementStats {
            policy_checks: self.policy_checks.load(Ordering::Relaxed),
            policy_violations: self.policy_violations.load(Ordering::Relaxed),
            uptime: self.started_at.elapsed().unwrap_or_default(),
            enforcement_level: self.config.enforcement_level,
        }
    }

    /// Strict enforcement - most restrictive
    fn enforce_strict_policy(
        &self,
        policy: &SecurityPolicy,
        context: &SecurityContext,
    ) -> PolicyDecision {
        // In strict mode, require high security level for sensitive operations
        match policy.policy_type {
            crate::security::PolicyType::CapabilityDelegation => {
                if (context.security_level as u8) < (crate::security::SecurityLevel::Confidential as u8) {
                    PolicyDecision::Deny
                } else {
                    PolicyDecision::Allow
                }
            }
            crate::security::PolicyType::AccessControl => {
                if (context.security_level as u8) < (crate::security::SecurityLevel::Internal as u8) {
                    PolicyDecision::AllowWithRestrictions(vec![
                        "Enhanced monitoring required".to_string(),
                        "Limited access duration".to_string(),
                    ])
                } else {
                    PolicyDecision::Allow
                }
            }
            _ => PolicyDecision::Allow
        }
    }

    /// Blocking enforcement - balanced approach
    fn enforce_blocking_policy(
        &self,
        policy: &SecurityPolicy,
        context: &SecurityContext,
    ) -> PolicyDecision {
        // Blocking mode allows most operations but adds restrictions for lower security levels
        match context.security_level {
            crate::security::SecurityLevel::Public => {
                PolicyDecision::AllowWithRestrictions(vec![
                    "Public access monitoring".to_string()
                ])
            }
            crate::security::SecurityLevel::Internal |
            crate::security::SecurityLevel::Confidential |
            crate::security::SecurityLevel::Restricted => {
                PolicyDecision::Allow
            }
        }
    }

    /// Advisory enforcement - most permissive
    fn enforce_advisory_policy(
        &self,
        _policy: &SecurityPolicy,
        _context: &SecurityContext,
    ) -> PolicyDecision {
        // Advisory mode always allows but may add monitoring
        PolicyDecision::Allow
    }
}

/// Statistics about security enforcement
#[derive(Debug, Clone)]
pub struct EnforcementStats {
    /// Total policy checks performed
    pub policy_checks: u64,
    /// Total policy violations detected
    pub policy_violations: u64,
    /// How long enforcement has been running
    pub uptime: std::time::Duration,
    /// Current enforcement level
    pub enforcement_level: crate::security::EnforcementLevel,
}

/// Security enforcement errors
#[derive(Debug, Error)]
pub enum EnforcementError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Policy validation error  
    #[error("Policy validation error: {message}")]
    PolicyValidation { message: String },

    /// Enforcement operation error
    #[error("Enforcement operation error: {message}")]
    Operation { message: String },

    /// Generic enforcement error
    #[error("Enforcement error: {message}")]
    Generic { message: String },
} 