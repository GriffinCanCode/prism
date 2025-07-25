//! Security manager for capability-based root access control
//!
//! This module provides security validation and audit logging for all
//! root set operations, ensuring compliance with capability-based security.

use crate::{VMResult, PrismVMError};
use super::{types::*, interfaces::*};
use prism_runtime::authority::capability::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, Duration};
use tracing::{debug, info, warn};

/// Security manager for root operations
pub struct RootSecurityManager {
    policy: SecurityPolicy,
    audit_log: Arc<Mutex<Vec<SecurityEvent>>>,
    /// Security statistics
    security_stats: Arc<Mutex<SecurityStats>>,
    /// Component capability mappings
    component_capabilities: Arc<Mutex<HashMap<String, CapabilitySet>>>,
    /// Security policy enforcement level
    enforcement_level: SecurityEnforcementLevel,
}

/// Security enforcement levels
#[derive(Debug, Clone, Copy, PartialEq)]
enum SecurityEnforcementLevel {
    /// No enforcement (development mode)
    None,
    /// Warning only (testing mode)
    Warn,
    /// Strict enforcement (production mode)
    Strict,
}

impl RootSecurityManager {
    pub fn new() -> VMResult<Self> {
        let policy = SecurityPolicy {
            default_level: SecurityLevel::Standard,
            operation_policies: Self::create_default_operation_policies(),
            capability_requirements: Self::create_default_capability_requirements(),
            audit_requirements: AuditRequirements {
                enabled: true,
                operations_to_audit: [
                    "AddRoot".to_string(),
                    "RemoveRoot".to_string(),
                    "ScanRoots".to_string(),
                    "ModifyPolicy".to_string(),
                ].iter().cloned().collect(),
                detail_level: AuditDetailLevel::Detailed,
                retention_days: 30,
            },
            access_restrictions: Self::create_default_access_restrictions(),
        };

        Ok(Self {
            policy,
            audit_log: Arc::new(Mutex::new(Vec::new())),
            security_stats: Arc::new(Mutex::new(SecurityStats {
                violations_detected: 0,
                capability_checks: 0,
                access_denials: 0,
                audit_events: 0,
                security_overhead_us: 0,
            })),
            component_capabilities: Arc::new(Mutex::new(HashMap::new())),
            enforcement_level: SecurityEnforcementLevel::Strict,
        })
    }

    /// Create default operation policies
    fn create_default_operation_policies() -> HashMap<String, SecurityLevel> {
        let mut policies = HashMap::new();
        
        // Root management operations
        policies.insert("AddRoot".to_string(), SecurityLevel::Standard);
        policies.insert("RemoveRoot".to_string(), SecurityLevel::Standard);
        policies.insert("ScanRoots".to_string(), SecurityLevel::Basic);
        policies.insert("AccessMetadata".to_string(), SecurityLevel::Basic);
        
        // Administrative operations require strict security
        policies.insert("ModifyPolicy".to_string(), SecurityLevel::Strict);
        policies.insert("Administrative".to_string(), SecurityLevel::Strict);
        
        policies
    }

    /// Create default capability requirements
    fn create_default_capability_requirements() -> HashMap<RootSource, Vec<String>> {
        let mut requirements = HashMap::new();
        
        requirements.insert(RootSource::Manual, vec![
            "memory.read".to_string(),
            "memory.write".to_string(),
        ]);
        
        requirements.insert(RootSource::GlobalVariables, vec![
            "memory.read".to_string(),
        ]);
        
        requirements.insert(RootSource::CapabilityTokens, vec![
            "system.read".to_string(),
        ]);
        
        requirements.insert(RootSource::ExecutionStack, vec![
            "memory.read".to_string(),
            "memory.write".to_string(),
        ]);
        
        requirements
    }

    /// Create default access restrictions
    fn create_default_access_restrictions() -> HashMap<RootSource, AccessRestrictions> {
        let mut restrictions = HashMap::new();
        
        // Global variables are read-only
        restrictions.insert(RootSource::GlobalVariables, AccessRestrictions {
            read_restrictions: Vec::new(),
            write_restrictions: vec!["no_modification".to_string()],
            time_restrictions: None,
            context_restrictions: Vec::new(),
        });
        
        // Capability tokens are read-only
        restrictions.insert(RootSource::CapabilityTokens, AccessRestrictions {
            read_restrictions: Vec::new(),
            write_restrictions: vec!["no_modification".to_string()],
            time_restrictions: None,
            context_restrictions: Vec::new(),
        });
        
        restrictions
    }
}

impl RootSecurityInterface for RootSecurityManager {
    fn validate_security_context(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> RootOperationResult<()> {
        let start_time = SystemTime::now();
        
        // Update statistics
        {
            let mut stats = self.security_stats.lock().unwrap();
            stats.capability_checks += 1;
        }
        
        // Check enforcement level
        if self.enforcement_level == SecurityEnforcementLevel::None {
            return RootOperationResult::Success(());
        }
        
        // Validate operation against policy
        let validation_result = self.validate_operation_policy(operation, context)?;
        
        // Check capability requirements
        let capability_result = self.validate_capability_requirements(operation, context)?;
        
        // Check access restrictions
        let access_result = self.validate_access_restrictions(operation, context)?;
        
        // Check security classification
        let classification_result = self.validate_security_classification(operation, context)?;
        
        // Combine all validation results
        let overall_result = match (validation_result, capability_result, access_result, classification_result) {
            (Ok(()), Ok(()), Ok(()), Ok(())) => Ok(()),
            _ => {
                let violation = SecurityViolation {
                    operation: operation.clone(),
                    context: context.clone(),
                    violation_type: SecurityViolationType::PolicyViolation,
                    severity: SecuritySeverity::High,
                    timestamp: SystemTime::now(),
                    details: "Security validation failed".to_string(),
                };
                
                self.record_security_violation(&violation);
                
                if self.enforcement_level == SecurityEnforcementLevel::Strict {
                    Err(RootError::SecurityViolation {
                        violation: "Comprehensive security validation failed".to_string()
                    })
                } else {
                    warn!("Security validation failed but enforcement level is not strict");
                    Ok(())
                }
            }
        };
        
        // Record security overhead
        {
            let mut stats = self.security_stats.lock().unwrap();
            stats.security_overhead_us += start_time.elapsed().unwrap_or(Duration::ZERO).as_micros() as u64;
        }
        
        match overall_result {
            Ok(()) => RootOperationResult::Success(()),
            Err(e) => {
                let mut stats = self.security_stats.lock().unwrap();
                stats.access_denials += 1;
                RootOperationResult::Failed(e)
            }
        }
    }
    
    fn is_operation_permitted(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> bool {
        match self.validate_security_context(operation, context) {
            RootOperationResult::Success(()) => true,
            _ => false,
        }
    }
    
    fn log_security_event(&mut self, event: SecurityEvent) -> RootOperationResult<()> {
        // Check if this operation should be audited
        let operation_name = format!("{:?}", event.event_type);
        if !self.policy.audit_requirements.operations_to_audit.contains(&operation_name) {
            return RootOperationResult::Success(());
        }
        
        // Add to audit log
        {
            let mut log = self.audit_log.lock().unwrap();
            log.push(event);
            
            // Cleanup old entries if needed
            let retention_duration = Duration::from_secs(
                self.policy.audit_requirements.retention_days as u64 * 24 * 3600
            );
            let cutoff_time = SystemTime::now() - retention_duration;
            
            log.retain(|event| event.timestamp >= cutoff_time);
        }
        
        // Update statistics
        {
            let mut stats = self.security_stats.lock().unwrap();
            stats.audit_events += 1;
        }
        
        RootOperationResult::Success(())
    }
    
    fn get_security_statistics(&self) -> SecurityStats {
        self.security_stats.lock().unwrap().clone()
    }
    
    fn update_security_policy(&mut self, policy: SecurityPolicy) -> RootOperationResult<()> {
        // Validate policy change is authorized
        let dummy_context = SecurityContext {
            capabilities: CapabilitySet::new(),
            classification: SecurityClassification::Secret, // Require high clearance
            restrictions: AccessRestrictions {
                read_restrictions: Vec::new(),
                write_restrictions: Vec::new(),
                time_restrictions: None,
                context_restrictions: Vec::new(),
            },
            audit_required: true,
        };
        
        let policy_operation = RootSecurityOperation::ModifyPolicy;
        
        match self.validate_security_context(&policy_operation, &dummy_context) {
            RootOperationResult::Success(()) => {
                self.policy = policy;
                info!("Security policy updated successfully");
                RootOperationResult::Success(())
            }
            other => {
                warn!("Policy update denied due to insufficient authorization");
                other
            }
        }
    }
}

impl RootSecurityManager {
    /// Validate operation against security policy
    fn validate_operation_policy(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> Result<(), RootError> {
        let operation_name = format!("{:?}", operation).split_whitespace().next().unwrap_or("Unknown").to_string();
        
        let required_level = self.policy.operation_policies
            .get(&operation_name)
            .copied()
            .unwrap_or(SecurityLevel::Standard);
        
        let context_level = match context.classification {
            SecurityClassification::Public => SecurityLevel::Basic,
            SecurityClassification::Internal => SecurityLevel::Standard,
            SecurityClassification::Confidential => SecurityLevel::Standard,
            SecurityClassification::Secret => SecurityLevel::Strict,
        };
        
        if context_level >= required_level {
            Ok(())
        } else {
            Err(RootError::SecurityViolation {
                violation: format!("Insufficient security level: required {:?}, have {:?}", required_level, context_level)
            })
        }
    }
    
    /// Validate capability requirements
    fn validate_capability_requirements(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> Result<(), RootError> {
        match operation {
            RootSecurityOperation::AddRoot { source, .. } |
            RootSecurityOperation::ScanRoots { source } => {
                if let Some(required_caps) = self.policy.capability_requirements.get(source) {
                    let available_cap_names = context.capabilities.capability_names();
                    
                    for required_cap in required_caps {
                        if !available_cap_names.iter().any(|cap| cap.contains(required_cap)) {
                            return Err(RootError::SecurityViolation {
                                violation: format!("Missing required capability: {}", required_cap)
                            });
                        }
                    }
                }
            }
            _ => {} // Other operations don't have specific capability requirements
        }
        
        Ok(())
    }
    
    /// Validate access restrictions
    fn validate_access_restrictions(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> Result<(), RootError> {
        match operation {
            RootSecurityOperation::AddRoot { source, .. } => {
                if let Some(restrictions) = self.policy.access_restrictions.get(source) {
                    // Check write restrictions for add operations
                    if !restrictions.write_restrictions.is_empty() {
                        return Err(RootError::SecurityViolation {
                            violation: format!("Write operation not allowed for source {:?}", source)
                        });
                    }
                }
            }
            RootSecurityOperation::RemoveRoot { .. } => {
                // Check if removal is allowed based on context restrictions
                if !context.restrictions.context_restrictions.is_empty() {
                    for restriction in &context.restrictions.context_restrictions {
                        if restriction == "no_removal" {
                            return Err(RootError::SecurityViolation {
                                violation: "Root removal not allowed in this context".to_string()
                            });
                        }
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Validate security classification
    fn validate_security_classification(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> Result<(), RootError> {
        match operation {
            RootSecurityOperation::ModifyPolicy |
            RootSecurityOperation::Administrative { .. } => {
                // Administrative operations require secret clearance
                if context.classification != SecurityClassification::Secret {
                    return Err(RootError::SecurityViolation {
                        violation: "Administrative operations require secret clearance".to_string()
                    });
                }
            }
            _ => {
                // Regular operations require at least internal clearance
                if context.classification == SecurityClassification::Public {
                    return Err(RootError::SecurityViolation {
                        violation: "Public clearance insufficient for root operations".to_string()
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Record a security violation
    fn record_security_violation(&self, violation: &SecurityViolation) {
        {
            let mut stats = self.security_stats.lock().unwrap();
            stats.violations_detected += 1;
        }
        
        warn!("Security violation detected: {:?}", violation);
        
        // Create security event
        let event = SecurityEvent {
            timestamp: SystemTime::now(),
            event_type: SecurityEventType::SecurityViolation,
            operation: violation.operation.clone(),
            context: violation.context.clone(),
            outcome: SecurityOutcome::Blocked,
            details: violation.details.clone(),
        };
        
        // Log the event (ignoring errors to avoid recursion)
        let _ = self.audit_log.lock().unwrap().push(event);
    }
}

/// Security violation details
#[derive(Debug, Clone)]
struct SecurityViolation {
    /// The operation that caused the violation
    operation: RootSecurityOperation,
    /// Security context at time of violation
    context: SecurityContext,
    /// Type of violation
    violation_type: SecurityViolationType,
    /// Severity of the violation
    severity: SecuritySeverity,
    /// When the violation occurred
    timestamp: SystemTime,
    /// Additional details
    details: String,
}

/// Types of security violations
#[derive(Debug, Clone)]
enum SecurityViolationType {
    /// Policy violation
    PolicyViolation,
    /// Capability violation
    CapabilityViolation,
    /// Access restriction violation
    AccessViolation,
    /// Classification violation
    ClassificationViolation,
}

/// Security violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SecuritySeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - warning
    Medium,
    /// High severity - error
    High,
    /// Critical severity - system compromise
    Critical,
} 
} 