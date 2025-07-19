//! Security Context
//!
//! Secure execution contexts and security audit logging

use super::{Capability, SecurityLevel};
use super::trust::TrustContext;
use std::collections::HashMap;

/// Secure execution context
#[derive(Debug, Clone)]
pub struct SecureExecutionContext {
    /// Context identifier
    pub id: String,
    /// Available capabilities
    pub available_capabilities: Vec<Capability>,
    /// Security level for this context
    pub security_level: SecurityLevel,
    /// Trust context
    pub trust_context: TrustContext,
    /// Context metadata
    pub metadata: HashMap<String, String>,
    /// When the context was created
    pub created_at: std::time::SystemTime,
}

impl SecureExecutionContext {
    /// Create a new secure execution context
    pub fn new(id: String, security_level: SecurityLevel, trust_context: TrustContext) -> Self {
        Self {
            id,
            available_capabilities: Vec::new(),
            security_level,
            trust_context,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now(),
        }
    }

    /// Add a capability to this context
    pub fn add_capability(&mut self, capability: Capability) {
        self.available_capabilities.push(capability);
    }

    /// Check if this context has a specific capability
    pub fn has_capability(&self, capability_name: &str) -> bool {
        self.available_capabilities
            .iter()
            .any(|cap| cap.definition == capability_name)
    }
}

/// Security event for audit logging
#[derive(Debug)]
pub struct SecurityEvent {
    /// Operation identifier
    pub operation_id: String,
    /// When the event occurred
    pub timestamp: std::time::SystemTime,
    /// Result of the operation
    pub result: String,
    /// Security context at time of event
    pub context: SecureExecutionContext,
}

/// Security audit log
#[derive(Debug)]
pub struct SecurityAuditLog {
    /// Log entries
    pub entries: Vec<SecurityEvent>,
    /// Maximum log size
    max_entries: usize,
}

impl SecurityAuditLog {
    /// Create new audit log
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: 10000, // Default limit
        }
    }

    /// Log a security validation event
    pub fn log_validation(&mut self, event: SecurityEvent) {
        self.entries.push(event);
        
        // Trim log if it exceeds maximum size
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }

    /// Get recent entries
    pub fn get_recent_entries(&self, count: usize) -> &[SecurityEvent] {
        let start = if self.entries.len() > count {
            self.entries.len() - count
        } else {
            0
        };
        &self.entries[start..]
    }

    /// Get entries for a specific operation
    pub fn get_entries_for_operation(&self, operation_id: &str) -> Vec<&SecurityEvent> {
        self.entries
            .iter()
            .filter(|entry| entry.operation_id == operation_id)
            .collect()
    }
}

impl Default for SecurityAuditLog {
    fn default() -> Self {
        Self::new()
    }
} 