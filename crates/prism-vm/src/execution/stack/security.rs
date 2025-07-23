//! Stack Security Management
//!
//! This module provides capability-aware security management for stack operations,
//! integrating with the existing prism-runtime authority system to ensure all
//! stack operations respect capability-based security constraints.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use prism_runtime::{
    authority::capability::{
        CapabilitySet, Capability, Authority, Operation, ConstraintSet,
        ComponentId, CapabilityManager,
    },
    platform::execution::ExecutionContext,
};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn, span, Level};

/// Stack security manager that enforces capability-based security
#[derive(Debug)]
pub struct StackSecurityManager {
    /// Available capabilities
    capabilities: CapabilitySet,
    
    /// Capability manager integration
    capability_manager: Arc<CapabilityManager>,
    
    /// Security policy enforcement
    security_policy: StackSecurityPolicy,
    
    /// Security audit log
    audit_log: Arc<RwLock<Vec<StackSecurityEvent>>>,
    
    /// Security statistics
    stats: Arc<RwLock<StackSecurityStats>>,
}

/// Stack security policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityPolicy {
    /// Require capabilities for stack frame creation
    pub require_frame_capability: bool,
    
    /// Require capabilities for local variable access
    pub require_local_access_capability: bool,
    
    /// Require capabilities for upvalue access
    pub require_upvalue_capability: bool,
    
    /// Maximum stack depth without elevated capabilities
    pub max_depth_without_elevation: usize,
    
    /// Enable stack overflow protection
    pub stack_overflow_protection: bool,
    
    /// Enable capability inheritance validation
    pub validate_capability_inheritance: bool,
}

impl Default for StackSecurityPolicy {
    fn default() -> Self {
        Self {
            require_frame_capability: true,
            require_local_access_capability: false, // Too restrictive for normal operation
            require_upvalue_capability: true,
            max_depth_without_elevation: 1000,
            stack_overflow_protection: true,
            validate_capability_inheritance: true,
        }
    }
}

/// Stack security event for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityEvent {
    /// When the event occurred
    pub timestamp: SystemTime,
    
    /// Type of security event
    pub event_type: StackSecurityEventType,
    
    /// Function context
    pub function_context: Option<String>,
    
    /// Stack depth at time of event
    pub stack_depth: usize,
    
    /// Capabilities involved
    pub capabilities: Vec<String>,
    
    /// Whether the operation was allowed
    pub allowed: bool,
    
    /// Additional context
    pub context: String,
}

/// Types of stack security events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StackSecurityEventType {
    /// Stack frame creation
    FrameCreation,
    
    /// Local variable access
    LocalAccess,
    
    /// Upvalue access
    UpvalueAccess,
    
    /// Stack depth check
    DepthCheck,
    
    /// Capability validation
    CapabilityValidation,
    
    /// Security violation
    SecurityViolation,
}

/// Stack security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityStats {
    /// Total security checks performed
    pub total_checks: u64,
    
    /// Security checks passed
    pub checks_passed: u64,
    
    /// Security violations detected
    pub violations_detected: u64,
    
    /// Capability validations performed
    pub capability_validations: u64,
    
    /// Average check time in microseconds
    pub avg_check_time_us: f64,
    
    /// Security events by type
    pub events_by_type: HashMap<String, u64>,
}

impl Default for StackSecurityStats {
    fn default() -> Self {
        Self {
            total_checks: 0,
            checks_passed: 0,
            violations_detected: 0,
            capability_validations: 0,
            avg_check_time_us: 0.0,
            events_by_type: HashMap::new(),
        }
    }
}

/// Stack security analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityAnalysis {
    /// Overall security status
    pub security_status: SecurityStatus,
    
    /// Active capabilities
    pub active_capabilities: Vec<String>,
    
    /// Security violations found
    pub violations: Vec<SecurityViolation>,
    
    /// Capability coverage analysis
    pub capability_coverage: CapabilityCoverage,
    
    /// Risk assessment
    pub risk_level: RiskLevel,
    
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Security status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityStatus {
    /// All security checks passed
    Secure,
    
    /// Minor security concerns
    Warning,
    
    /// Significant security issues
    Critical,
    
    /// Security violations detected
    Violated,
}

/// Security violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    
    /// Function where violation occurred
    pub function_name: Option<String>,
    
    /// Stack depth at violation
    pub stack_depth: usize,
    
    /// Description of the violation
    pub description: String,
    
    /// Severity level
    pub severity: Severity,
}

/// Types of security violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Missing required capability
    MissingCapability,
    
    /// Capability expired
    ExpiredCapability,
    
    /// Stack depth exceeded
    StackDepthExceeded,
    
    /// Unauthorized access attempt
    UnauthorizedAccess,
    
    /// Capability inheritance violation
    InheritanceViolation,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Capability coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCoverage {
    /// Required capabilities
    pub required: HashSet<String>,
    
    /// Available capabilities
    pub available: HashSet<String>,
    
    /// Missing capabilities
    pub missing: HashSet<String>,
    
    /// Coverage percentage
    pub coverage_percentage: f64,
}

/// Risk assessment levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl StackSecurityManager {
    /// Create a new stack security manager
    pub fn new(capabilities: CapabilitySet) -> VMResult<Self> {
        let _span = span!(Level::INFO, "stack_security_init").entered();
        info!("Initializing stack security management");

        let capability_manager = Arc::new(
            CapabilityManager::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create capability manager: {}", e),
            })?
        );

        Ok(Self {
            capabilities,
            capability_manager,
            security_policy: StackSecurityPolicy::default(),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(StackSecurityStats::default())),
        })
    }

    /// Validate stack frame creation
    pub fn validate_frame_creation(
        &self,
        function_name: &str,
        context: &ExecutionContext,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "validate_frame_creation", 
            function = %function_name
        ).entered();

        let start_time = Instant::now();
        
        if !self.security_policy.require_frame_capability {
            return Ok(());
        }

        // Check if we have the required capability for frame creation
        let operation = Operation::Memory(MemoryOperation::Allocate);
        let authorized = self.capabilities.authorizes(&operation, context);

        // Log the security event
        self.log_security_event(StackSecurityEvent {
            timestamp: SystemTime::now(),
            event_type: StackSecurityEventType::FrameCreation,
            function_context: Some(function_name.to_string()),
            stack_depth: 0, // Will be updated by caller
            capabilities: self.capabilities.capability_names(),
            allowed: authorized,
            context: format!("Frame creation for function: {}", function_name),
        });

        // Update statistics
        self.update_stats(start_time, authorized);

        if !authorized {
            return Err(PrismVMError::CapabilityViolation {
                message: format!(
                    "Missing capability for stack frame creation in function: {}",
                    function_name
                ),
            });
        }

        debug!("Stack frame creation authorized for: {}", function_name);
        Ok(())
    }

    /// Validate stack depth
    pub fn validate_stack_depth(
        &self,
        current_depth: usize,
        context: &ExecutionContext,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "validate_stack_depth", depth = current_depth).entered();

        if !self.security_policy.stack_overflow_protection {
            return Ok(());
        }

        let start_time = Instant::now();
        let allowed = if current_depth > self.security_policy.max_depth_without_elevation {
            // Check for elevated capabilities
            let elevated_operation = Operation::System(SystemOperation::ElevatedExecution);
            self.capabilities.authorizes(&elevated_operation, context)
        } else {
            true
        };

        // Log the security event
        self.log_security_event(StackSecurityEvent {
            timestamp: SystemTime::now(),
            event_type: StackSecurityEventType::DepthCheck,
            function_context: None,
            stack_depth: current_depth,
            capabilities: self.capabilities.capability_names(),
            allowed,
            context: format!("Stack depth validation: {}", current_depth),
        });

        // Update statistics
        self.update_stats(start_time, allowed);

        if !allowed {
            return Err(PrismVMError::CapabilityViolation {
                message: format!(
                    "Stack depth {} exceeds limit {} without elevated capabilities",
                    current_depth, self.security_policy.max_depth_without_elevation
                ),
            });
        }

        debug!("Stack depth {} validated", current_depth);
        Ok(())
    }

    /// Validate upvalue access
    pub fn validate_upvalue_access(
        &self,
        function_name: &str,
        slot: u8,
        context: &ExecutionContext,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "validate_upvalue_access", 
            function = %function_name, 
            slot = slot
        ).entered();

        if !self.security_policy.require_upvalue_capability {
            return Ok(());
        }

        let start_time = Instant::now();
        
        // Check for closure access capability
        let operation = Operation::Memory(MemoryOperation::Access);
        let authorized = self.capabilities.authorizes(&operation, context);

        // Log the security event
        self.log_security_event(StackSecurityEvent {
            timestamp: SystemTime::now(),
            event_type: StackSecurityEventType::UpvalueAccess,
            function_context: Some(function_name.to_string()),
            stack_depth: 0, // Will be updated by caller
            capabilities: self.capabilities.capability_names(),
            allowed: authorized,
            context: format!("Upvalue access in {}, slot {}", function_name, slot),
        });

        // Update statistics
        self.update_stats(start_time, authorized);

        if !authorized {
            return Err(PrismVMError::CapabilityViolation {
                message: format!(
                    "Missing capability for upvalue access in function: {}, slot: {}",
                    function_name, slot
                ),
            });
        }

        debug!("Upvalue access authorized for: {} slot {}", function_name, slot);
        Ok(())
    }

    /// Analyze stack security
    pub fn analyze(&self, stack: &ExecutionStack) -> StackSecurityAnalysis {
        let _span = span!(Level::DEBUG, "analyze_security").entered();

        let mut violations = Vec::new();
        let mut recommendations = Vec::new();

        // Check stack depth
        let current_depth = stack.frame_count();
        if current_depth > self.security_policy.max_depth_without_elevation {
            violations.push(SecurityViolation {
                violation_type: ViolationType::StackDepthExceeded,
                function_name: None,
                stack_depth: current_depth,
                description: format!("Stack depth {} exceeds safe limit", current_depth),
                severity: if current_depth > self.security_policy.max_depth_without_elevation * 2 {
                    Severity::Critical
                } else {
                    Severity::Medium
                },
            });
            recommendations.push("Consider reducing recursion depth or obtaining elevated capabilities".to_string());
        }

        // Analyze capability coverage
        let capability_coverage = self.analyze_capability_coverage(stack);
        if capability_coverage.coverage_percentage < 80.0 {
            recommendations.push("Consider obtaining additional capabilities for better security coverage".to_string());
        }

        // Determine overall security status
        let security_status = if violations.is_empty() {
            SecurityStatus::Secure
        } else if violations.iter().any(|v| matches!(v.severity, Severity::Critical)) {
            SecurityStatus::Critical
        } else if violations.iter().any(|v| matches!(v.severity, Severity::High)) {
            SecurityStatus::Violated
        } else {
            SecurityStatus::Warning
        };

        // Determine risk level
        let risk_level = match security_status {
            SecurityStatus::Secure => RiskLevel::Low,
            SecurityStatus::Warning => RiskLevel::Medium,
            SecurityStatus::Critical => RiskLevel::Critical,
            SecurityStatus::Violated => RiskLevel::High,
        };

        StackSecurityAnalysis {
            security_status,
            active_capabilities: self.capabilities.capability_names(),
            violations,
            capability_coverage,
            risk_level,
            recommendations,
        }
    }

    /// Analyze capability coverage
    fn analyze_capability_coverage(&self, _stack: &ExecutionStack) -> CapabilityCoverage {
        // This would analyze what capabilities are required vs available
        // For now, provide a basic implementation
        let available: HashSet<String> = self.capabilities.capability_names().into_iter().collect();
        let required: HashSet<String> = [
            "memory_access".to_string(),
            "stack_frame_creation".to_string(),
        ].into_iter().collect();
        
        let missing: HashSet<String> = required.difference(&available).cloned().collect();
        let coverage_percentage = if required.is_empty() {
            100.0
        } else {
            ((required.len() - missing.len()) as f64 / required.len() as f64) * 100.0
        };

        CapabilityCoverage {
            required,
            available,
            missing,
            coverage_percentage,
        }
    }

    /// Log a security event
    fn log_security_event(&self, event: StackSecurityEvent) {
        let mut audit_log = self.audit_log.write().unwrap();
        audit_log.push(event.clone());

        // Keep only recent events (last 1000)
        if audit_log.len() > 1000 {
            audit_log.drain(0..100);
        }

        // Update event type statistics
        let mut stats = self.stats.write().unwrap();
        let event_type_key = format!("{:?}", event.event_type);
        *stats.events_by_type.entry(event_type_key).or_insert(0) += 1;
    }

    /// Update security statistics
    fn update_stats(&self, start_time: Instant, allowed: bool) {
        let mut stats = self.stats.write().unwrap();
        stats.total_checks += 1;
        
        if allowed {
            stats.checks_passed += 1;
        } else {
            stats.violations_detected += 1;
        }

        // Update average check time
        let check_time_us = start_time.elapsed().as_micros() as f64;
        stats.avg_check_time_us = (stats.avg_check_time_us * (stats.total_checks - 1) as f64 + check_time_us) / stats.total_checks as f64;
    }

    /// Get security statistics
    pub fn stats(&self) -> StackSecurityStats {
        self.stats.read().unwrap().clone()
    }

    /// Get recent security events
    pub fn recent_events(&self, count: usize) -> Vec<StackSecurityEvent> {
        let audit_log = self.audit_log.read().unwrap();
        audit_log.iter().rev().take(count).cloned().collect()
    }
}

// Define operation types for capability checking
#[derive(Debug, Clone)]
pub enum MemoryOperation {
    Allocate,
    Access,
    Deallocate,
}

#[derive(Debug, Clone)]
pub enum SystemOperation {
    ElevatedExecution,
    DebugAccess,
    ProfilerAccess,
} 