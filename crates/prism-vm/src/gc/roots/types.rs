//! Core types and data structures for the root set management subsystem
//!
//! This module defines all the shared types used across the root set subsystem,
//! with clear separation from other GC components:
//!
//! **Root Set Responsibilities:**
//! - Root source identification and categorization
//! - Stack frame and value scanning coordination
//! - Global variable and static data tracking
//! - Security context and capability integration
//! - Performance monitoring and statistics
//!
//! **NOT Root Set Responsibilities (delegated to other components):**
//! - Object marking/coloring (handled by collectors::*)
//! - Heap object management (handled by heap::*)
//! - Memory allocation (handled by allocators::*)
//! - Write barriers (handled by barriers::*)

use crate::{VMResult, PrismVMError};
use crate::execution::{StackValue, StackFrame};
use prism_runtime::authority::capability::CapabilitySet;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, AtomicBool};
use std::time::{Duration, Instant, SystemTime};
use std::thread::ThreadId;

/// Configuration for root manager behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootManagerConfig {
    /// Enable concurrent root scanning
    pub enable_concurrent_scanning: bool,
    /// Enable incremental scanning to reduce pause times
    pub enable_incremental_scanning: bool,
    /// Maximum time to spend scanning roots (microseconds)
    pub max_scan_time_us: u64,
    /// Enable detailed analytics and monitoring
    pub enable_analytics: bool,
    /// Enable root validation (debug builds)
    pub enable_validation: bool,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
    /// Security level for root operations
    pub security_level: SecurityLevel,
    /// Thread-local root cache size
    pub thread_cache_size: usize,
    /// Maximum number of roots before triggering cleanup
    pub max_roots_before_cleanup: usize,
    /// Stack scanning strategy
    pub stack_scan_strategy: StackScanStrategy,
    /// Platform-specific configuration
    pub platform_config: PlatformConfig,
}

impl Default for RootManagerConfig {
    fn default() -> Self {
        Self {
            enable_concurrent_scanning: false,
            enable_incremental_scanning: true,
            max_scan_time_us: 1000, // 1ms default
            enable_analytics: true,
            enable_validation: cfg!(debug_assertions),
            enable_detailed_logging: false,
            security_level: SecurityLevel::Standard,
            thread_cache_size: 128,
            max_roots_before_cleanup: 10000,
            stack_scan_strategy: StackScanStrategy::Precise,
            platform_config: PlatformConfig::default(),
        }
    }
}

/// Security levels for root operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Basic security checks
    Basic,
    /// Standard security with capability validation
    Standard,
    /// Strict security with full audit logging
    Strict,
}

/// Stack scanning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StackScanStrategy {
    /// Precise scanning using type information
    Precise,
    /// Conservative scanning for safety
    Conservative,
    /// Hybrid approach combining both
    Hybrid,
}

/// Platform-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    /// Enable platform-specific stack boundary detection
    pub enable_stack_bounds_detection: bool,
    /// Stack alignment requirements
    pub stack_alignment: usize,
    /// Register scanning configuration
    pub register_scan_config: RegisterScanConfig,
    /// Thread-local storage scanning
    pub enable_tls_scanning: bool,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            enable_stack_bounds_detection: true,
            stack_alignment: std::mem::align_of::<usize>(),
            register_scan_config: RegisterScanConfig::default(),
            enable_tls_scanning: false, // Disabled by default for security
        }
    }
}

/// Register scanning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterScanConfig {
    /// Enable register scanning
    pub enabled: bool,
    /// Registers to scan (platform-specific)
    pub registers_to_scan: Vec<String>,
    /// Conservative register scanning
    pub conservative_scan: bool,
}

impl Default for RegisterScanConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default - complex and platform-specific
            registers_to_scan: Vec::new(),
            conservative_scan: true,
        }
    }
}

/// Root source types in Prism VM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RootSource {
    /// Execution stack roots (local variables, temporaries)
    ExecutionStack,
    /// JIT compiled code roots (stack maps, deopt state)
    JitCompiled,
    /// Global variables and static data
    GlobalVariables,
    /// Capability tokens and effect handles
    CapabilityTokens,
    /// Thread-local storage
    ThreadLocal,
    /// Manually registered roots
    Manual,
    /// Pinned objects (cannot be moved)
    Pinned,
}

/// Individual root entry with metadata
#[derive(Debug, Clone)]
pub struct RootEntry {
    /// Pointer to the root object
    pub ptr: *const u8,
    /// Source of this root
    pub source: RootSource,
    /// Root type for precise scanning
    pub root_type: RootType,
    /// When this root was registered
    pub registered_at: Instant,
    /// Thread that registered this root
    pub thread_id: ThreadId,
    /// Security context
    pub security_context: SecurityContext,
    /// Additional metadata
    pub metadata: RootMetadata,
}

/// Root type information for precise scanning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootType {
    /// Direct heap object pointer
    DirectObject { type_id: u32, size: usize },
    /// Stack value (may contain nested references)
    StackValue { value_type: StackValueType },
    /// Array of stack values
    StackValueArray { element_count: usize },
    /// Global variable
    GlobalVariable { name: String, var_type: String },
    /// Capability token
    Capability { name: String },
    /// Effect handle
    Effect { name: String },
    /// JIT compiled frame reference
    JitFrame { function_id: u32, frame_size: usize },
    /// Unknown/opaque root (conservative scanning)
    Unknown,
}

/// Stack value type information for root scanning
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StackValueType {
    /// Null value (no references)
    Null,
    /// Boolean value (no references)
    Boolean,
    /// Integer value (no references)
    Integer,
    /// Float value (no references)
    Float,
    /// String value (heap reference)
    String,
    /// Byte array (heap reference)
    Bytes,
    /// Array of values (heap reference + nested references)
    Array { element_count: usize },
    /// Object with fields (heap reference + field references)
    Object { field_count: usize, field_names: Vec<String> },
    /// Function with upvalues (heap reference + upvalue references)
    Function { upvalue_count: usize },
    /// Type reference (no heap reference)
    Type,
    /// Capability token (string reference)
    Capability,
    /// Effect handle (string reference)
    Effect,
}

impl StackValueType {
    /// Check if this stack value type contains heap references
    pub fn contains_heap_references(&self) -> bool {
        match self {
            StackValueType::Null
            | StackValueType::Boolean
            | StackValueType::Integer
            | StackValueType::Float
            | StackValueType::Type => false,
            
            StackValueType::String
            | StackValueType::Bytes
            | StackValueType::Array { .. }
            | StackValueType::Object { .. }
            | StackValueType::Function { .. }
            | StackValueType::Capability
            | StackValueType::Effect => true,
        }
    }
    
    /// Extract type information from a StackValue
    pub fn from_stack_value(value: &StackValue) -> Self {
        match value {
            StackValue::Null => StackValueType::Null,
            StackValue::Boolean(_) => StackValueType::Boolean,
            StackValue::Integer(_) => StackValueType::Integer,
            StackValue::Float(_) => StackValueType::Float,
            StackValue::String(_) => StackValueType::String,
            StackValue::Bytes(_) => StackValueType::Bytes,
            StackValue::Array(arr) => StackValueType::Array { 
                element_count: arr.len() 
            },
            StackValue::Object(obj) => StackValueType::Object { 
                field_count: obj.len(),
                field_names: obj.keys().cloned().collect()
            },
            StackValue::Function { upvalues, .. } => StackValueType::Function { 
                upvalue_count: upvalues.len() 
            },
            StackValue::Type(_) => StackValueType::Type,
            StackValue::Capability(_) => StackValueType::Capability,
            StackValue::Effect(_) => StackValueType::Effect,
        }
    }
}

/// Security context for root operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Available capabilities
    pub capabilities: CapabilitySet,
    /// Security classification
    pub classification: SecurityClassification,
    /// Access restrictions
    pub restrictions: AccessRestrictions,
    /// Audit requirements
    pub audit_required: bool,
}

/// Security classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityClassification {
    /// Public data
    Public,
    /// Internal data
    Internal,
    /// Confidential data
    Confidential,
    /// Secret data
    Secret,
}

/// Access restrictions for roots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRestrictions {
    /// Read access restrictions
    pub read_restrictions: Vec<String>,
    /// Write access restrictions
    pub write_restrictions: Vec<String>,
    /// Time-based restrictions
    pub time_restrictions: Option<TimeRestrictions>,
    /// Context-based restrictions
    pub context_restrictions: Vec<String>,
}

/// Time-based access restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRestrictions {
    /// Access allowed from this time
    pub valid_from: SystemTime,
    /// Access allowed until this time
    pub valid_until: SystemTime,
    /// Time zone restrictions
    pub timezone_restrictions: Vec<String>,
}

/// Additional metadata for root entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootMetadata {
    /// Human-readable description
    pub description: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Business context
    pub business_context: Option<String>,
    /// Performance hints
    pub performance_hints: PerformanceHints,
    /// Debug information
    pub debug_info: Option<DebugInfo>,
}

/// Performance hints for root scanning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHints {
    /// Expected scanning frequency
    pub scan_frequency: ScanFrequency,
    /// Access pattern information
    pub access_pattern: AccessPattern,
    /// Cache locality hints
    pub locality_hints: LocalityHints,
    /// Priority for scanning order
    pub scan_priority: ScanPriority,
}

/// Expected scanning frequency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScanFrequency {
    /// Scanned every GC cycle
    EveryGC,
    /// Scanned frequently
    High,
    /// Scanned occasionally
    Medium,
    /// Scanned rarely
    Low,
    /// Scanned once and cached
    Once,
}

/// Access pattern information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Temporal locality
    Temporal,
    /// Spatial locality
    Spatial,
    /// Unknown pattern
    Unknown,
}

/// Cache locality hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalityHints {
    /// Prefer cache-friendly scanning order
    pub prefer_cache_friendly: bool,
    /// NUMA node affinity
    pub numa_node: Option<usize>,
    /// Expected cache level
    pub cache_level: CacheLevel,
}

/// Cache level preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheLevel {
    /// L1 cache
    L1,
    /// L2 cache
    L2,
    /// L3 cache
    L3,
    /// Main memory
    Memory,
}

/// Scanning priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScanPriority {
    /// Critical roots (must be scanned first)
    Critical = 0,
    /// High priority roots
    High = 1,
    /// Normal priority roots
    Normal = 2,
    /// Low priority roots
    Low = 3,
    /// Deferred roots (can be scanned later)
    Deferred = 4,
}

/// Debug information for root entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    /// Source location where root was registered
    pub source_location: Option<SourceLocation>,
    /// Call stack at registration
    pub call_stack: Vec<String>,
    /// Additional debug data
    pub debug_data: HashMap<String, String>,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File name
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u32,
    /// Function name
    pub function: String,
}

/// Statistics for root set management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootStatistics {
    /// Total number of roots
    pub total_roots: usize,
    /// Roots by source type
    pub roots_by_source: HashMap<RootSource, usize>,
    /// Roots by type
    pub roots_by_type: HashMap<String, usize>,
    /// Scanning performance statistics
    pub scan_stats: ScanStatistics,
    /// Memory usage statistics
    pub memory_stats: RootMemoryStats,
    /// Security statistics
    pub security_stats: SecurityStats,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
}

/// Root scanning performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanStatistics {
    /// Total scans performed
    pub total_scans: u64,
    /// Average scan time (microseconds)
    pub average_scan_time_us: f64,
    /// Maximum scan time (microseconds)
    pub max_scan_time_us: u64,
    /// Minimum scan time (microseconds)
    pub min_scan_time_us: u64,
    /// Roots scanned per second
    pub roots_per_second: f64,
    /// Scan efficiency (roots per microsecond)
    pub scan_efficiency: f64,
    /// Cache hit rate for root lookups
    pub cache_hit_rate: f64,
    /// Last scan timestamp
    pub last_scan_time: Option<Instant>,
}

/// Memory usage statistics for root management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootMemoryStats {
    /// Memory used by root entries
    pub root_entries_bytes: usize,
    /// Memory used by metadata
    pub metadata_bytes: usize,
    /// Memory used by caches
    pub cache_bytes: usize,
    /// Total memory overhead
    pub total_overhead_bytes: usize,
    /// Memory efficiency (roots per byte)
    pub memory_efficiency: f64,
}

/// Security-related statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStats {
    /// Security violations detected
    pub violations_detected: u64,
    /// Capability checks performed
    pub capability_checks: u64,
    /// Access denials
    pub access_denials: u64,
    /// Audit events generated
    pub audit_events: u64,
    /// Security overhead (time spent on security)
    pub security_overhead_us: u64,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Scanning time trend
    pub scan_time_trend: TrendDirection,
    /// Root count trend
    pub root_count_trend: TrendDirection,
    /// Memory usage trend
    pub memory_usage_trend: TrendDirection,
    /// Cache efficiency trend
    pub cache_efficiency_trend: TrendDirection,
    /// Overall performance trend
    pub overall_trend: TrendDirection,
}

/// Trend direction indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Improving performance
    Improving,
    /// Stable performance
    Stable,
    /// Degrading performance
    Degrading,
    /// Insufficient data
    Unknown,
}

/// Root set operation results
#[derive(Debug, Clone)]
pub enum RootOperationResult<T> {
    /// Operation completed successfully
    Success(T),
    /// Operation completed with warnings
    SuccessWithWarnings(T, Vec<RootWarning>),
    /// Operation failed
    Failed(RootError),
    /// Operation was denied by security policy
    SecurityDenied(String),
    /// Operation timed out
    TimedOut(Duration),
}

/// Root set operation warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RootWarning {
    /// Root count approaching limit
    ApproachingRootLimit { current: usize, limit: usize },
    /// Scanning time approaching limit
    SlowScanning { time_us: u64, limit_us: u64 },
    /// Memory usage high
    HighMemoryUsage { usage_bytes: usize, limit_bytes: usize },
    /// Security context degraded
    DegradedSecurity { reason: String },
    /// Cache efficiency low
    LowCacheEfficiency { efficiency: f64, threshold: f64 },
}

/// Root set operation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RootError {
    /// Invalid root pointer
    InvalidPointer { ptr: usize, reason: String },
    /// Root already exists
    DuplicateRoot { ptr: usize },
    /// Root not found
    RootNotFound { ptr: usize },
    /// Security violation
    SecurityViolation { violation: String },
    /// Resource exhaustion
    ResourceExhaustion { resource: String },
    /// Platform error
    PlatformError { error: String },
    /// Configuration error
    ConfigurationError { error: String },
}

impl std::fmt::Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootError::InvalidPointer { ptr, reason } => {
                write!(f, "Invalid root pointer 0x{:x}: {}", ptr, reason)
            }
            RootError::DuplicateRoot { ptr } => {
                write!(f, "Root already exists: 0x{:x}", ptr)
            }
            RootError::RootNotFound { ptr } => {
                write!(f, "Root not found: 0x{:x}", ptr)
            }
            RootError::SecurityViolation { violation } => {
                write!(f, "Security violation: {}", violation)
            }
            RootError::ResourceExhaustion { resource } => {
                write!(f, "Resource exhaustion: {}", resource)
            }
            RootError::PlatformError { error } => {
                write!(f, "Platform error: {}", error)
            }
            RootError::ConfigurationError { error } => {
                write!(f, "Configuration error: {}", error)
            }
        }
    }
}

impl std::error::Error for RootError {}

/// Convert RootError to PrismVMError
impl From<RootError> for PrismVMError {
    fn from(error: RootError) -> Self {
        PrismVMError::RuntimeError {
            message: error.to_string(),
        }
    }
} 