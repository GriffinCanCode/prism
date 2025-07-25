//! Clean interfaces and traits for root set management
//!
//! This module defines the core traits and interfaces that all root management
//! components must implement, ensuring consistency and interoperability across
//! the root set subsystem.

use crate::{VMResult, PrismVMError};
use crate::execution::{StackValue, StackFrame, ExecutionStack};
use super::types::*;
use std::collections::HashSet;
use std::sync::Arc;

/// Main interface for root set management
/// 
/// This trait provides a unified interface for root set operations across
/// different implementations, ensuring consistency and enabling testing.
pub trait RootSetInterface: Send + Sync {
    /// Add a root object to the set
    fn add_root(&mut self, ptr: *const u8, root_type: RootType, source: RootSource) -> RootOperationResult<()>;
    
    /// Remove a root object from the set
    fn remove_root(&mut self, ptr: *const u8) -> RootOperationResult<()>;
    
    /// Check if a pointer is registered as a root
    fn contains_root(&self, ptr: *const u8) -> bool;
    
    /// Get all root objects for garbage collection
    fn get_all_roots(&self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Get roots by source type
    fn get_roots_by_source(&self, source: RootSource) -> RootOperationResult<Vec<*const u8>>;
    
    /// Scan for additional roots (e.g., stack scanning)
    fn scan_for_roots(&mut self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Clear all roots (used for testing)
    fn clear_all_roots(&mut self) -> RootOperationResult<()>;
    
    /// Get root set statistics
    fn get_statistics(&self) -> RootStatistics;
    
    /// Validate root set integrity
    fn validate_integrity(&self) -> RootOperationResult<()>;
}

/// Interface for stack scanning operations
/// 
/// This trait abstracts stack scanning functionality, allowing for different
/// scanning strategies (precise, conservative, hybrid) while maintaining
/// a consistent interface.
pub trait StackScannerInterface: Send + Sync {
    /// Scan the execution stack for root objects
    fn scan_execution_stack(&self, stack: &ExecutionStack) -> RootOperationResult<Vec<RootEntry>>;
    
    /// Scan a specific stack frame
    fn scan_stack_frame(&self, frame: &StackFrame) -> RootOperationResult<Vec<RootEntry>>;
    
    /// Scan a stack value for nested references
    fn scan_stack_value(&self, value: &StackValue) -> RootOperationResult<Vec<*const u8>>;
    
    /// Get platform-specific stack bounds
    fn get_stack_bounds(&self) -> RootOperationResult<Option<(usize, usize)>>;
    
    /// Scan native stack (conservative approach)
    fn scan_native_stack(&self, start: *const u8, end: *const u8) -> RootOperationResult<Vec<*const u8>>;
    
    /// Configure scanning strategy
    fn set_scan_strategy(&mut self, strategy: StackScanStrategy) -> RootOperationResult<()>;
    
    /// Get scanning statistics
    fn get_scan_statistics(&self) -> ScanStatistics;
}

/// Interface for global root management
/// 
/// This trait handles global variables, static data, and other non-stack roots
/// that need to be tracked across the entire VM lifetime.
pub trait GlobalRootInterface: Send + Sync {
    /// Register a global variable as a root
    fn register_global(&mut self, ptr: *const u8, name: String, var_type: String) -> RootOperationResult<()>;
    
    /// Unregister a global variable
    fn unregister_global(&mut self, ptr: *const u8) -> RootOperationResult<()>;
    
    /// Register a capability token
    fn register_capability(&mut self, ptr: *const u8, name: String) -> RootOperationResult<()>;
    
    /// Register an effect handle
    fn register_effect(&mut self, ptr: *const u8, name: String) -> RootOperationResult<()>;
    
    /// Get all global roots
    fn get_global_roots(&self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Get capability roots
    fn get_capability_roots(&self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Get effect roots
    fn get_effect_roots(&self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Clear all global roots
    fn clear_global_roots(&mut self) -> RootOperationResult<()>;
}

/// Interface for security-aware root operations
/// 
/// This trait ensures all root operations respect capability-based security
/// constraints and provide proper audit logging.
pub trait RootSecurityInterface: Send + Sync {
    /// Validate security context for root operation
    fn validate_security_context(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> RootOperationResult<()>;
    
    /// Check if operation is permitted
    fn is_operation_permitted(&self, operation: &RootSecurityOperation, context: &SecurityContext) -> bool;
    
    /// Log security event
    fn log_security_event(&mut self, event: SecurityEvent) -> RootOperationResult<()>;
    
    /// Get security statistics
    fn get_security_statistics(&self) -> SecurityStats;
    
    /// Update security policy
    fn update_security_policy(&mut self, policy: SecurityPolicy) -> RootOperationResult<()>;
}

/// Interface for root analytics and monitoring
/// 
/// This trait provides performance monitoring, trend analysis, and optimization
/// recommendations for root set management.
pub trait RootAnalyticsInterface: Send + Sync {
    /// Record a root operation for analytics
    fn record_operation(&mut self, operation: &RootAnalyticsOperation) -> RootOperationResult<()>;
    
    /// Get performance trends
    fn get_performance_trends(&self) -> PerformanceTrends;
    
    /// Get optimization recommendations
    fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation>;
    
    /// Generate performance report
    fn generate_performance_report(&self) -> PerformanceReport;
    
    /// Reset analytics data
    fn reset_analytics(&mut self) -> RootOperationResult<()>;
}

/// Interface for platform-specific stack operations
/// 
/// This trait abstracts platform-specific functionality like stack bounds
/// detection and register scanning.
pub trait PlatformStackInterface: Send + Sync {
    /// Detect stack bounds for current thread
    fn detect_stack_bounds(&self) -> RootOperationResult<Option<(usize, usize)>>;
    
    /// Scan CPU registers for potential roots
    fn scan_registers(&self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Get thread-local storage roots
    fn scan_thread_local_storage(&self) -> RootOperationResult<Vec<*const u8>>;
    
    /// Check if pointer could be a valid heap reference
    fn is_valid_heap_pointer(&self, ptr: *const u8) -> bool;
    
    /// Get platform-specific configuration
    fn get_platform_config(&self) -> PlatformConfig;
    
    /// Update platform configuration
    fn set_platform_config(&mut self, config: PlatformConfig) -> RootOperationResult<()>;
}

/// Security operations for root management
#[derive(Debug, Clone)]
pub enum RootSecurityOperation {
    /// Adding a new root
    AddRoot { ptr: *const u8, source: RootSource },
    /// Removing a root
    RemoveRoot { ptr: *const u8 },
    /// Scanning for roots
    ScanRoots { source: RootSource },
    /// Accessing root metadata
    AccessMetadata { ptr: *const u8 },
    /// Modifying security policy
    ModifyPolicy,
    /// Administrative operation
    Administrative { operation: String },
}

/// Security events for audit logging
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    /// Event timestamp
    pub timestamp: std::time::SystemTime,
    /// Event type
    pub event_type: SecurityEventType,
    /// Operation that triggered the event
    pub operation: RootSecurityOperation,
    /// Security context
    pub context: SecurityContext,
    /// Event outcome
    pub outcome: SecurityOutcome,
    /// Additional details
    pub details: String,
}

/// Types of security events
#[derive(Debug, Clone)]
pub enum SecurityEventType {
    /// Access granted
    AccessGranted,
    /// Access denied
    AccessDenied,
    /// Security violation detected
    SecurityViolation,
    /// Policy change
    PolicyChange,
    /// Audit trail event
    AuditTrail,
}

/// Security event outcomes
#[derive(Debug, Clone)]
pub enum SecurityOutcome {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation was blocked
    Blocked,
    /// Operation was modified
    Modified,
}

/// Security policy configuration
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Default security level
    pub default_level: SecurityLevel,
    /// Operation-specific policies
    pub operation_policies: std::collections::HashMap<String, SecurityLevel>,
    /// Capability requirements
    pub capability_requirements: std::collections::HashMap<RootSource, Vec<String>>,
    /// Audit requirements
    pub audit_requirements: AuditRequirements,
    /// Access restrictions
    pub access_restrictions: std::collections::HashMap<RootSource, AccessRestrictions>,
}

/// Audit requirements configuration
#[derive(Debug, Clone)]
pub struct AuditRequirements {
    /// Enable audit logging
    pub enabled: bool,
    /// Operations to audit
    pub operations_to_audit: HashSet<String>,
    /// Audit detail level
    pub detail_level: AuditDetailLevel,
    /// Retention policy
    pub retention_days: u32,
}

/// Audit detail levels
#[derive(Debug, Clone, Copy)]
pub enum AuditDetailLevel {
    /// Basic audit information
    Basic,
    /// Detailed audit information
    Detailed,
    /// Full audit information
    Full,
}

/// Analytics operations for monitoring
#[derive(Debug, Clone)]
pub struct RootAnalyticsOperation {
    /// Operation timestamp
    pub timestamp: std::time::Instant,
    /// Operation type
    pub operation_type: AnalyticsOperationType,
    /// Duration of operation
    pub duration: std::time::Duration,
    /// Number of roots affected
    pub roots_affected: usize,
    /// Memory usage delta
    pub memory_delta: i64,
    /// Success status
    pub success: bool,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Types of analytics operations
#[derive(Debug, Clone)]
pub enum AnalyticsOperationType {
    /// Root addition
    AddRoot,
    /// Root removal
    RemoveRoot,
    /// Stack scanning
    StackScan,
    /// Global root scanning
    GlobalScan,
    /// Root validation
    Validation,
    /// Cache operation
    CacheOperation,
    /// Security check
    SecurityCheck,
}

/// Performance optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: OptimizationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Implementation difficulty
    pub difficulty: ImplementationDifficulty,
    /// Recommendation description
    pub description: String,
    /// Specific actions to take
    pub actions: Vec<String>,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Reduce scanning time
    ReduceScanTime,
    /// Improve cache efficiency
    ImproveCacheEfficiency,
    /// Reduce memory overhead
    ReduceMemoryOverhead,
    /// Optimize scanning order
    OptimizeScanOrder,
    /// Improve security performance
    ImproveSecurity,
    /// Configuration optimization
    Configuration,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Critical optimization needed
    Critical,
    /// High priority optimization
    High,
    /// Medium priority optimization
    Medium,
    /// Low priority optimization
    Low,
    /// Nice to have optimization
    Optional,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy)]
pub enum ImplementationDifficulty {
    /// Easy to implement
    Easy,
    /// Moderate difficulty
    Moderate,
    /// Difficult to implement
    Difficult,
    /// Very difficult to implement
    VeryDifficult,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Report generation timestamp
    pub generated_at: std::time::SystemTime,
    /// Report period
    pub period: std::time::Duration,
    /// Overall performance summary
    pub summary: PerformanceSummary,
    /// Detailed statistics
    pub detailed_stats: RootStatistics,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Alerts and warnings
    pub alerts: Vec<PerformanceAlert>,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Overall performance score (0-100)
    pub performance_score: f64,
    /// Key performance indicators
    pub kpis: std::collections::HashMap<String, f64>,
    /// Performance grade
    pub grade: PerformanceGrade,
    /// Summary description
    pub description: String,
}

/// Performance grades
#[derive(Debug, Clone, Copy)]
pub enum PerformanceGrade {
    /// Excellent performance
    Excellent,
    /// Good performance
    Good,
    /// Fair performance
    Fair,
    /// Poor performance
    Poor,
    /// Critical performance issues
    Critical,
}

/// Performance alerts
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Recommended actions
    pub actions: Vec<String>,
    /// Alert timestamp
    pub timestamp: std::time::SystemTime,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Information only
    Info,
    /// Warning condition
    Warning,
    /// Error condition
    Error,
    /// Critical condition
    Critical,
}

/// Types of performance alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    /// High scan time
    HighScanTime,
    /// Memory usage high
    HighMemoryUsage,
    /// Low cache efficiency
    LowCacheEfficiency,
    /// Security overhead high
    HighSecurityOverhead,
    /// Root count high
    HighRootCount,
    /// Performance degradation
    PerformanceDegradation,
    /// Configuration issue
    ConfigurationIssue,
} 