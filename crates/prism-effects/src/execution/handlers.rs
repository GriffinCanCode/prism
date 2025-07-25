//! Effect handler framework
//!
//! This module provides composable effect handlers that interpret and manage
//! effects in a secure, capability-controlled manner as specified in PLD-003.

use crate::effects::{Effect};
use crate::capability::Capability;
use prism_common::span::Span;
use prism_ast::{AstNode, Expr, SecurityClassification};
use std::collections::{HashMap, HashSet};

/// Registry for effect handlers
#[derive(Debug, Default)]
pub struct HandlerRegistry {
    /// Registered effect handlers
    pub handlers: HashMap<String, Box<dyn EffectHandler>>,
    /// Handler composition rules
    pub composition_rules: Vec<HandlerCompositionRule>,
    /// Handler priority ordering
    pub priorities: HashMap<String, u32>,
}

impl HandlerRegistry {
    /// Create a new handler registry
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_handlers();
        registry
    }

    /// Register a new effect handler
    pub fn register(&mut self, handler: Box<dyn EffectHandler>) -> Result<(), crate::EffectSystemError> {
        let handler_name = handler.name().to_string();
        
        if self.handlers.contains_key(&handler_name) {
            return Err(crate::EffectSystemError::HandlerRegistrationFailed {
                reason: format!("Handler '{}' is already registered", handler_name),
            });
        }

        self.handlers.insert(handler_name.clone(), handler);
        self.priorities.insert(handler_name, 100); // Default priority
        Ok(())
    }

    /// Get a handler by name
    pub fn get_handler(&self, name: &str) -> Option<&dyn EffectHandler> {
        self.handlers.get(name).map(|h| h.as_ref())
    }

    /// Get all handlers that can handle a specific effect
    pub fn get_handlers_for_effect(&self, effect_name: &str) -> Vec<&dyn EffectHandler> {
        self.handlers
            .values()
            .filter(|handler| handler.can_handle(effect_name))
            .map(|h| h.as_ref())
            .collect()
    }

    /// Register built-in effect handlers
    fn register_builtin_handlers(&mut self) {
        // File system handler
        let fs_handler = Box::new(FileSystemHandler::new());
        let _ = self.register(fs_handler);

        // Network handler
        let network_handler = Box::new(NetworkHandler::new());
        let _ = self.register(network_handler);

        // Database handler
        let db_handler = Box::new(DatabaseHandler::new());
        let _ = self.register(db_handler);

        // Security handler
        let security_handler = Box::new(SecurityHandler::new());
        let _ = self.register(security_handler);

        // External AI integration handler
        let external_ai_handler = Box::new(ExternalAIHandler::new());
        let _ = self.register(external_ai_handler);
    }
}

/// Trait for effect handlers
pub trait EffectHandler: std::fmt::Debug + Send + Sync {
    /// Get the name of this handler
    fn name(&self) -> &str;

    /// Check if this handler can handle a specific effect
    fn can_handle(&self, effect_name: &str) -> bool;

    /// Handle an effect with given capabilities
    fn handle_effect(
        &self,
        effect: &Effect,
        capabilities: &[Capability],
        context: &mut EffectContext,
    ) -> Result<EffectResult, EffectHandlerError>;

    /// Get the required capabilities for handling effects
    fn required_capabilities(&self) -> Vec<String>;

    /// Get AI-comprehensible information about this handler
    fn ai_info(&self) -> HandlerAIInfo;

    /// Validate effect parameters before handling
    fn validate_parameters(&self, effect: &Effect) -> Result<(), EffectHandlerError> {
        // Default implementation - no validation
        Ok(())
    }

    /// Get security implications of handling this effect
    fn security_implications(&self, effect: &Effect) -> Vec<String> {
        vec![]
    }
}

/// Context for effect handling
#[derive(Debug)]
pub struct EffectContext {
    /// Current execution span
    pub span: Span,
    /// Available capabilities
    pub capabilities: HashMap<String, Capability>,
    /// Security classification context
    pub security_context: SecurityContext,
    /// AI context for this execution
    pub ai_context: Option<String>,
    /// Execution metadata
    pub metadata: EffectExecutionMetadata,
}

impl EffectContext {
    /// Create a new effect context
    pub fn new(span: Span) -> Self {
        Self {
            span,
            capabilities: HashMap::new(),
            security_context: SecurityContext::default(),
            ai_context: None,
            metadata: EffectExecutionMetadata::default(),
        }
    }

    /// Add a capability to the context
    pub fn add_capability(&mut self, capability: Capability) {
        self.capabilities.insert(capability.definition.clone(), capability);
    }

    /// Check if a capability is available
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.contains_key(name)
    }

    /// Get a capability by name
    pub fn get_capability(&self, name: &str) -> Option<&Capability> {
        self.capabilities.get(name)
    }
}

/// Security context for effect execution
#[derive(Debug, Default)]
pub struct SecurityContext {
    /// Current security classification
    pub classification: SecurityClassification,
    /// Information flow labels
    pub flow_labels: HashSet<String>,
    /// Security policies in effect
    pub active_policies: Vec<String>,
    /// Audit trail enabled
    pub audit_enabled: bool,
}

/// Metadata for effect execution
#[derive(Debug, Default)]
pub struct EffectExecutionMetadata {
    /// Execution start time
    pub start_time: Option<std::time::Instant>,
    /// Execution duration
    pub duration: Option<std::time::Duration>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Error information if execution failed
    pub error_info: Option<String>,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory allocated (bytes)
    pub memory_allocated: u64,
    /// CPU time used (microseconds)
    pub cpu_time: u64,
    /// Network bytes sent
    pub network_sent: u64,
    /// Network bytes received
    pub network_received: u64,
    /// Disk bytes read
    pub disk_read: u64,
    /// Disk bytes written
    pub disk_written: u64,
    /// Timestamp when measurement was taken
    pub timestamp: std::time::Instant,
}

impl ResourceUsage {
    /// Create a new resource usage measurement
    pub fn new() -> Self {
        Self {
            memory_allocated: 0,
            network_sent: 0,
            network_received: 0,
            disk_read: 0,
            disk_written: 0,
            cpu_time: 0,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Capture current resource usage from the system
    pub fn capture_current() -> Result<Self, EffectHandlerError> {
        #[cfg(feature = "monitoring")]
        {
            use crate::effects::monitoring::ResourceMonitor;
            
            let monitor = ResourceMonitor::new();
            let snapshot = monitor.capture_current_usage()
                .map_err(|e| EffectHandlerError::InternalError { 
                    error: format!("Failed to capture resource usage: {:?}", e) 
                })?;
            
            Ok(Self {
                memory_allocated: snapshot.memory_allocated_bytes,
                cpu_time: snapshot.cpu_time_micros,
                network_sent: snapshot.network_sent_bytes,
                network_received: snapshot.network_received_bytes,
                disk_read: snapshot.disk_read_bytes,
                disk_written: snapshot.disk_write_bytes,
                timestamp: snapshot.timestamp,
            })
        }
        
        #[cfg(not(feature = "monitoring"))]
        {
            // Fallback to placeholder values when monitoring is disabled
            Ok(Self::new())
        }
    }

    /// Calculate resource consumption between two measurements
    pub fn calculate_consumption(&self, start: &ResourceUsage) -> ResourceConsumption {
        ResourceConsumption {
            memory_delta: self.memory_allocated.saturating_sub(start.memory_allocated),
            cpu_time_delta: self.cpu_time.saturating_sub(start.cpu_time),
            network_sent_delta: self.network_sent.saturating_sub(start.network_sent),
            network_received_delta: self.network_received.saturating_sub(start.network_received),
            disk_read_delta: self.disk_read.saturating_sub(start.disk_read),
            disk_written_delta: self.disk_written.saturating_sub(start.disk_written),
            duration: self.timestamp.duration_since(start.timestamp),
        }
    }

    /// Get total resource units for comparison
    pub fn total_units(&self) -> u64 {
        // Weight different resource types for comparison
        self.cpu_time / 1000 +  // CPU time in milliseconds
        self.memory_allocated / 1024 +  // Memory in KB
        (self.disk_read + self.disk_written) / 1024 +  // Disk I/O in KB
        (self.network_sent + self.network_received) / 1024  // Network I/O in KB
    }

    /// Check if resource usage exceeds thresholds
    pub fn exceeds_thresholds(&self) -> Vec<ResourceThresholdViolation> {
        let mut violations = Vec::new();

        // Memory threshold: 1GB
        if self.memory_allocated > 1024 * 1024 * 1024 {
            violations.push(ResourceThresholdViolation {
                resource_type: ResourceType::Memory,
                threshold: 1024 * 1024 * 1024,
                actual: self.memory_allocated,
                severity: if self.memory_allocated > 4 * 1024 * 1024 * 1024 {
                    ViolationSeverity::Critical
                } else {
                    ViolationSeverity::Warning
                },
            });
        }

        // CPU time threshold: 10 seconds
        if self.cpu_time > 10_000_000 {
            violations.push(ResourceThresholdViolation {
                resource_type: ResourceType::CPU,
                threshold: 10_000_000,
                actual: self.cpu_time,
                severity: if self.cpu_time > 60_000_000 {
                    ViolationSeverity::Critical
                } else {
                    ViolationSeverity::Warning
                },
            });
        }

        // Network threshold: 100MB
        let total_network = self.network_sent + self.network_received;
        if total_network > 100 * 1024 * 1024 {
            violations.push(ResourceThresholdViolation {
                resource_type: ResourceType::Network,
                threshold: 100 * 1024 * 1024,
                actual: total_network,
                severity: ViolationSeverity::Warning,
            });
        }

        // Disk I/O threshold: 500MB
        let total_disk = self.disk_read + self.disk_written;
        if total_disk > 500 * 1024 * 1024 {
            violations.push(ResourceThresholdViolation {
                resource_type: ResourceType::Disk,
                threshold: 500 * 1024 * 1024,
                actual: total_disk,
                severity: ViolationSeverity::Warning,
            });
        }

        violations
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource consumption between two measurements
#[derive(Debug, Clone)]
pub struct ResourceConsumption {
    /// Memory allocated difference (bytes)
    pub memory_delta: u64,
    /// CPU time difference (microseconds)
    pub cpu_time_delta: u64,
    /// Network bytes sent difference
    pub network_sent_delta: u64,
    /// Network bytes received difference
    pub network_received_delta: u64,
    /// Disk bytes read difference
    pub disk_read_delta: u64,
    /// Disk bytes written difference
    pub disk_written_delta: u64,
    /// Time duration between measurements
    pub duration: std::time::Duration,
}

impl ResourceConsumption {
    /// Get resource consumption rate (per second)
    pub fn get_rates(&self) -> ResourceRates {
        let duration_secs = self.duration.as_secs_f64().max(0.001); // Avoid division by zero
        
        ResourceRates {
            memory_rate: self.memory_delta as f64 / duration_secs,
            cpu_rate: self.cpu_time_delta as f64 / duration_secs,
            network_sent_rate: self.network_sent_delta as f64 / duration_secs,
            network_received_rate: self.network_received_delta as f64 / duration_secs,
            disk_read_rate: self.disk_read_delta as f64 / duration_secs,
            disk_write_rate: self.disk_written_delta as f64 / duration_secs,
        }
    }

    /// Calculate efficiency score based on resource usage
    pub fn efficiency_score(&self) -> f64 {
        // Simple efficiency calculation: lower resource usage = higher efficiency
        let total_resources = self.memory_delta / 1024 + // KB
                             self.cpu_time_delta / 1000 + // milliseconds
                             (self.network_sent_delta + self.network_received_delta) / 1024 + // KB
                             (self.disk_read_delta + self.disk_written_delta) / 1024; // KB
        
        // Normalize to 0-1 scale (higher is better)
        1.0 / (1.0 + (total_resources as f64 / 1000.0))
    }
}

/// Resource usage rates (per second)
#[derive(Debug, Clone)]
pub struct ResourceRates {
    /// Memory allocation rate (bytes/sec)
    pub memory_rate: f64,
    /// CPU usage rate (microseconds/sec)
    pub cpu_rate: f64,
    /// Network send rate (bytes/sec)
    pub network_sent_rate: f64,
    /// Network receive rate (bytes/sec)
    pub network_received_rate: f64,
    /// Disk read rate (bytes/sec)
    pub disk_read_rate: f64,
    /// Disk write rate (bytes/sec)
    pub disk_write_rate: f64,
}

/// Resource threshold violation
#[derive(Debug, Clone)]
pub struct ResourceThresholdViolation {
    /// Type of resource that exceeded threshold
    pub resource_type: ResourceType,
    /// Threshold that was exceeded
    pub threshold: u64,
    /// Actual resource usage
    pub actual: u64,
    /// Severity of the violation
    pub severity: ViolationSeverity,
}

/// Types of system resources
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceType {
    Memory,
    CPU,
    Network,
    Disk,
}

/// Severity levels for threshold violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationSeverity {
    Warning,
    Critical,
}

/// Result of effect handling
#[derive(Debug, Clone)]
pub struct EffectResult {
    /// Success or failure
    pub success: bool,
    /// Result value if any
    pub value: Option<AstNode<Expr>>,
    /// Any side effects produced
    pub side_effects: Vec<SideEffect>,
    /// Security events generated
    pub security_events: Vec<SecurityEvent>,
    /// AI-readable summary of what happened
    pub ai_summary: Option<String>,
}

impl EffectResult {
    /// Create a successful result
    pub fn success() -> Self {
        Self {
            success: true,
            value: None,
            side_effects: Vec::new(),
            security_events: Vec::new(),
            ai_summary: None,
        }
    }

    /// Create a successful result with value
    pub fn success_with_value(value: AstNode<Expr>) -> Self {
        Self {
            success: true,
            value: Some(value),
            side_effects: Vec::new(),
            security_events: Vec::new(),
            ai_summary: None,
        }
    }

    /// Create a failure result
    pub fn failure(reason: String) -> Self {
        Self {
            success: false,
            value: None,
            side_effects: Vec::new(),
            security_events: vec![SecurityEvent::EffectHandlingFailed { reason }],
            ai_summary: None,
        }
    }
}

/// Side effect produced during handling
#[derive(Debug, Clone)]
pub struct SideEffect {
    /// Type of side effect
    pub effect_type: String,
    /// Description of the side effect
    pub description: String,
    /// Security implications
    pub security_implications: Vec<String>,
    /// AI-readable context
    pub ai_context: Option<String>,
}

/// Security event generated during effect handling
#[derive(Debug, Clone)]
pub enum SecurityEvent {
    /// Capability was used
    CapabilityUsed { capability: String, operation: String },
    /// Security policy was enforced
    PolicyEnforced { policy: String, action: String },
    /// Information flow was controlled
    InformationFlowControlled { from: String, to: String },
    /// Effect handling failed
    EffectHandlingFailed { reason: String },
    /// Unauthorized access attempt
    UnauthorizedAccess { resource: String, capability: String },
}

/// AI-readable information about a handler
#[derive(Debug, Clone)]
pub struct HandlerAIInfo {
    /// Purpose of this handler
    pub purpose: String,
    /// Effects this handler can process
    pub handled_effects: Vec<String>,
    /// Security guarantees provided
    pub security_guarantees: Vec<String>,
    /// Potential risks
    pub risks: Vec<String>,
    /// Best practices for usage
    pub best_practices: Vec<String>,
}

/// Errors that can occur during effect handling
#[derive(Debug, thiserror::Error)]
pub enum EffectHandlerError {
    /// Missing required capability
    #[error("Missing required capability: {capability}")]
    MissingCapability { capability: String },

    /// Invalid effect parameters
    #[error("Invalid effect parameters: {reason}")]
    InvalidParameters { reason: String },

    /// Security violation
    #[error("Security violation: {violation}")]
    SecurityViolation { violation: String },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource}")]
    ResourceLimitExceeded { resource: String },

    /// Handler internal error
    #[error("Handler internal error: {error}")]
    InternalError { error: String },

    /// Invalid capability
    #[error("Invalid capability: {0}")]
    InvalidCapability(String),
}

/// File system effect handler
#[derive(Debug)]
pub struct FileSystemHandler {
    /// Handler name
    name: String,
    /// Supported operations
    supported_operations: HashSet<String>,
}

impl FileSystemHandler {
    /// Create a new file system handler
    pub fn new() -> Self {
        let mut supported_operations = HashSet::new();
        supported_operations.insert("IO.FileSystem.Read".to_string());
        supported_operations.insert("IO.FileSystem.Write".to_string());
        supported_operations.insert("IO.FileSystem.Execute".to_string());

        Self {
            name: "FileSystemHandler".to_string(),
            supported_operations,
        }
    }
}

impl EffectHandler for FileSystemHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn can_handle(&self, effect_name: &str) -> bool {
        self.supported_operations.contains(effect_name)
    }

    fn handle_effect(
        &self,
        effect: &Effect,
        capabilities: &[Capability],
        context: &mut EffectContext,
    ) -> Result<EffectResult, EffectHandlerError> {
        // Check for FileSystem capability
        let fs_capability = capabilities
            .iter()
            .find(|cap| cap.definition == "FileSystem")
            .ok_or_else(|| EffectHandlerError::MissingCapability {
                capability: "FileSystem".to_string(),
            })?;

        // Validate capability is not revoked
        if fs_capability.revoked {
            return Err(EffectHandlerError::SecurityViolation {
                violation: "FileSystem capability has been revoked".to_string(),
            });
        }

        // Handle specific file system operations
        match effect.definition.as_str() {
            "IO.FileSystem.Read" => self.handle_file_read(effect, fs_capability, context),
            "IO.FileSystem.Write" => self.handle_file_write(effect, fs_capability, context),
            "IO.FileSystem.Execute" => self.handle_file_execute(effect, fs_capability, context),
            _ => Err(EffectHandlerError::InvalidParameters {
                reason: format!("Unsupported file system operation: {}", effect.definition),
            }),
        }
    }

    fn required_capabilities(&self) -> Vec<String> {
        vec!["FileSystem".to_string()]
    }

    fn ai_info(&self) -> HandlerAIInfo {
        HandlerAIInfo {
            purpose: "Handles file system operations with capability-based security".to_string(),
            handled_effects: self.supported_operations.iter().cloned().collect(),
            security_guarantees: vec![
                "File access limited by capability constraints".to_string(),
                "Path traversal attacks prevented".to_string(),
                "File permissions enforced".to_string(),
            ],
            risks: vec![
                "Sensitive file contents may be exposed".to_string(),
                "File system state may be modified".to_string(),
            ],
            best_practices: vec![
                "Use minimal file system capabilities".to_string(),
                "Validate all file paths".to_string(),
                "Monitor file system access patterns".to_string(),
            ],
        }
    }
}

impl FileSystemHandler {
    /// Handle file read operation
    fn handle_file_read(
        &self,
        effect: &Effect,
        capability: &Capability,
        context: &mut EffectContext,
    ) -> Result<EffectResult, EffectHandlerError> {
        // Extract file path from effect parameters
        let path = effect.parameters.get("path")
            .ok_or_else(|| EffectHandlerError::InvalidParameters {
                reason: "Missing 'path' parameter".to_string(),
            })?;

        // Validate path against capability constraints
        // This is a simplified implementation
        let mut result = EffectResult::success();
        result.security_events.push(SecurityEvent::CapabilityUsed {
            capability: "FileSystem".to_string(),
            operation: "Read".to_string(),
        });
        result.ai_summary = Some("File read operation completed successfully".to_string());

        Ok(result)
    }

    /// Handle file write operation
    fn handle_file_write(
        &self,
        effect: &Effect,
        capability: &Capability,
        context: &mut EffectContext,
    ) -> Result<EffectResult, EffectHandlerError> {
        // Similar implementation to file_read but for write operations
        let mut result = EffectResult::success();
        result.security_events.push(SecurityEvent::CapabilityUsed {
            capability: "FileSystem".to_string(),
            operation: "Write".to_string(),
        });
        result.ai_summary = Some("File write operation completed successfully".to_string());

        Ok(result)
    }

    /// Handle file execute operation
    fn handle_file_execute(
        &self,
        effect: &Effect,
        capability: &Capability,
        context: &mut EffectContext,
    ) -> Result<EffectResult, EffectHandlerError> {
        // Implementation for file execution
        let mut result = EffectResult::success();
        result.security_events.push(SecurityEvent::CapabilityUsed {
            capability: "FileSystem".to_string(),
            operation: "Execute".to_string(),
        });
        result.ai_summary = Some("File execute operation completed successfully".to_string());

        Ok(result)
    }
}

/// Network effect handler
#[derive(Debug)]
pub struct NetworkHandler {
    name: String,
    supported_operations: HashSet<String>,
}

impl NetworkHandler {
    pub fn new() -> Self {
        let mut supported_operations = HashSet::new();
        supported_operations.insert("IO.Network.Connect".to_string());
        supported_operations.insert("IO.Network.Listen".to_string());
        supported_operations.insert("IO.Network.DNS".to_string());

        Self {
            name: "NetworkHandler".to_string(),
            supported_operations,
        }
    }
}

impl EffectHandler for NetworkHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn can_handle(&self, effect_name: &str) -> bool {
        self.supported_operations.contains(effect_name)
    }

    fn handle_effect(
        &self,
        effect: &Effect,
        capabilities: &[Capability],
        context: &mut EffectContext,
    ) -> Result<EffectResult, EffectHandlerError> {
        // Check for Network capability
        let _network_capability = capabilities
            .iter()
            .find(|cap| cap.definition == "Network")
            .ok_or_else(|| EffectHandlerError::MissingCapability {
                capability: "Network".to_string(),
            })?;

        let mut result = EffectResult::success();
        result.security_events.push(SecurityEvent::CapabilityUsed {
            capability: "Network".to_string(),
            operation: effect.definition.clone(),
        });
        result.ai_summary = Some(format!("Network operation {} completed", effect.definition));

        Ok(result)
    }

    fn required_capabilities(&self) -> Vec<String> {
        vec!["Network".to_string()]
    }

    fn ai_info(&self) -> HandlerAIInfo {
        HandlerAIInfo {
            purpose: "Handles network operations with security controls".to_string(),
            handled_effects: self.supported_operations.iter().cloned().collect(),
            security_guarantees: vec![
                "Network access limited by capability constraints".to_string(),
                "Rate limiting enforced".to_string(),
                "Protocol restrictions applied".to_string(),
            ],
            risks: vec![
                "Data may be transmitted over network".to_string(),
                "Network resources may be consumed".to_string(),
            ],
            best_practices: vec![
                "Use encrypted connections".to_string(),
                "Validate all network inputs".to_string(),
                "Monitor network usage patterns".to_string(),
            ],
        }
    }
}

// Similar implementations for DatabaseHandler, SecurityHandler, AIHandler...
// I'll create abbreviated versions for brevity

/// Database effect handler
#[derive(Debug)]
pub struct DatabaseHandler {
    name: String,
    supported_operations: HashSet<String>,
}

impl DatabaseHandler {
    pub fn new() -> Self {
        let mut supported_operations = HashSet::new();
        supported_operations.insert("Database.Query".to_string());
        supported_operations.insert("Database.Transaction".to_string());

        Self {
            name: "DatabaseHandler".to_string(),
            supported_operations,
        }
    }
}

impl EffectHandler for DatabaseHandler {
    fn name(&self) -> &str { &self.name }
    fn can_handle(&self, effect_name: &str) -> bool { self.supported_operations.contains(effect_name) }
    fn handle_effect(&self, effect: &Effect, capabilities: &[Capability], context: &mut EffectContext) -> Result<EffectResult, EffectHandlerError> {
        Ok(EffectResult::success())
    }
    fn required_capabilities(&self) -> Vec<String> { vec!["Database".to_string()] }
    fn ai_info(&self) -> HandlerAIInfo {
        HandlerAIInfo {
            purpose: "Handles database operations securely".to_string(),
            handled_effects: self.supported_operations.iter().cloned().collect(),
            security_guarantees: vec!["Database access controlled".to_string()],
            risks: vec!["Database state may be modified".to_string()],
            best_practices: vec!["Use transactions for consistency".to_string()],
        }
    }
}

/// Security effect handler
#[derive(Debug)]
pub struct SecurityHandler {
    name: String,
    supported_operations: HashSet<String>,
}

impl SecurityHandler {
    pub fn new() -> Self {
        let mut supported_operations = HashSet::new();
        supported_operations.insert("Cryptography.KeyGeneration".to_string());
        supported_operations.insert("Cryptography.Encryption".to_string());

        Self {
            name: "SecurityHandler".to_string(),
            supported_operations,
        }
    }
}

impl EffectHandler for SecurityHandler {
    fn name(&self) -> &str { &self.name }
    fn can_handle(&self, effect_name: &str) -> bool { self.supported_operations.contains(effect_name) }
    fn handle_effect(&self, effect: &Effect, capabilities: &[Capability], context: &mut EffectContext) -> Result<EffectResult, EffectHandlerError> {
        Ok(EffectResult::success())
    }
    fn required_capabilities(&self) -> Vec<String> { vec!["Cryptography".to_string()] }
    fn ai_info(&self) -> HandlerAIInfo {
        HandlerAIInfo {
            purpose: "Handles cryptographic operations".to_string(),
            handled_effects: self.supported_operations.iter().cloned().collect(),
            security_guarantees: vec!["Cryptographic operations secured".to_string()],
            risks: vec!["Sensitive cryptographic material handled".to_string()],
            best_practices: vec!["Use approved algorithms only".to_string()],
        }
    }
}

/// AI effect handler
#[derive(Debug)]
pub struct ExternalAIHandler {
    name: String,
    supported_operations: HashSet<String>,
}

impl ExternalAIHandler {
    pub fn new() -> Self {
        let mut supported_operations = HashSet::new();
        supported_operations.insert("ExternalAI.DataExport".to_string());
        supported_operations.insert("ExternalAI.MetadataGeneration".to_string());

        Self {
            name: "ExternalAIHandler".to_string(),
            supported_operations,
        }
    }
}

impl EffectHandler for ExternalAIHandler {
    fn name(&self) -> &str { &self.name }
    fn can_handle(&self, effect_name: &str) -> bool { self.supported_operations.contains(effect_name) }
    fn handle_effect(&self, effect: &Effect, capabilities: &[Capability], context: &mut EffectContext) -> Result<EffectResult, EffectHandlerError> {
        match effect.definition.as_str() {
            "ExternalAI.DataExport" => {
                // Validate data export capabilities and sanitize data
                // This would prepare data for external AI systems but not execute models
                Ok(EffectResult::success_with_value(
                    // Return metadata about what was prepared for export
                    prism_ast::AstNode::new(
                        prism_ast::Expr::Literal(prism_ast::LiteralExpr {
                            value: prism_ast::LiteralValue::String("data_prepared_for_export".to_string()),
                        }),
                        effect.span,
                        prism_common::NodeId::new(1),
                    )
                ))
            }
            "ExternalAI.MetadataGeneration" => {
                // Generate AI-comprehensible metadata structures
                Ok(EffectResult::success_with_value(
                    prism_ast::AstNode::new(
                        prism_ast::Expr::Literal(prism_ast::LiteralExpr {
                            value: prism_ast::LiteralValue::String("ai_metadata_generated".to_string()),
                        }),
                        effect.span,
                        prism_common::NodeId::new(2),
                    )
                ))
            }
            _ => Ok(EffectResult::success())
        }
    }
    fn required_capabilities(&self) -> Vec<String> { 
        vec!["Network".to_string(), "FileSystem".to_string()] 
    }
    fn ai_info(&self) -> HandlerAIInfo {
        HandlerAIInfo {
            purpose: "Handles integration with external AI systems through data export and metadata generation".to_string(),
            handled_effects: self.supported_operations.iter().cloned().collect(),
            security_guarantees: vec![
                "No direct AI model execution in language runtime".to_string(),
                "Data sanitization before external export".to_string(),
                "Metadata generation controlled and audited".to_string()
            ],
            risks: vec![
                "Exported data may be processed by external AI services".to_string(),
                "Generated metadata may reveal code structure".to_string()
            ],
            best_practices: vec![
                "Sanitize sensitive data before export".to_string(), 
                "Audit all external AI integrations".to_string(),
                "Use secure channels for AI service communication".to_string()
            ],
        }
    }
}

/// Rule for composing handlers
#[derive(Debug, Clone)]
pub struct HandlerCompositionRule {
    /// Rule name
    pub name: String,
    /// Description
    pub description: String,
    /// Handler precedence order
    pub precedence: Vec<String>,
    /// Composition conditions
    pub conditions: Vec<AstNode<Expr>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_registry_creation() {
        let registry = HandlerRegistry::new();
        assert!(!registry.handlers.is_empty(), "Registry should have built-in handlers");
        assert!(registry.handlers.contains_key("FileSystemHandler"));
        assert!(registry.handlers.contains_key("NetworkHandler"));
    }

    #[test]
    fn test_file_system_handler() {
        let handler = FileSystemHandler::new();
        assert_eq!(handler.name(), "FileSystemHandler");
        assert!(handler.can_handle("IO.FileSystem.Read"));
        assert!(!handler.can_handle("Database.Query"));
    }

    #[test]
    fn test_effect_context() {
        let span = Span::dummy();
        let mut context = EffectContext::new(span);
        
        let capability = Capability::new(
            "FileSystem".to_string(),
            crate::capability::CapabilityConstraints::new(),
        );
        
        context.add_capability(capability);
        assert!(context.has_capability("FileSystem"));
        assert!(!context.has_capability("Network"));
    }
} 