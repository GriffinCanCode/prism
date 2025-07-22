//! Multi-Target Execution System
//!
//! This module implements the execution system that can run Prism code across
//! multiple targets (TypeScript, WebAssembly, Native) while maintaining
//! capability-based security and effect tracking.
//!
//! ## Design Principles
//!
//! 1. **Target Agnostic**: Core execution logic independent of target platform
//! 2. **Adapter Pattern**: Target-specific adapters handle platform details
//! 3. **Capability Preservation**: Security guarantees maintained across all targets
//! 4. **Performance Optimized**: Target-specific optimizations where beneficial
//! 5. **AI-Comprehensible**: Structured execution metadata for AI analysis

use crate::{authority::capability, resources::effects, RuntimeError, Executable};
use crate::resources::effects::Effect;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use uuid::Uuid;
use std::time::Instant;

/// Execution context providing environment and metadata for code execution
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Unique execution ID
    pub execution_id: ExecutionId,
    
    /// Target platform for execution
    pub target: ExecutionTarget,
    
    /// Component executing the code
    pub component_id: capability::ComponentId,
    
    /// Available capabilities for this execution
    pub capabilities: capability::CapabilitySet,
    
    /// Execution configuration
    pub config: ExecutionConfig,
    
    /// AI-readable context metadata
    pub ai_context: ExecutionAIContext,
    
    /// Execution timestamp
    pub timestamp: SystemTime,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new(
        target: ExecutionTarget,
        component_id: capability::ComponentId,
        capabilities: capability::CapabilitySet,
    ) -> Self {
        Self {
            execution_id: ExecutionId::new(),
            target,
            component_id,
            capabilities,
            config: ExecutionConfig::default(),
            ai_context: ExecutionAIContext::default(),
            timestamp: SystemTime::now(),
        }
    }

    /// Get the current execution target
    pub fn target(&self) -> ExecutionTarget {
        self.target
    }

    /// Get the current capabilities
    pub fn current_capabilities(&self) -> &capability::CapabilitySet {
        &self.capabilities
    }

    /// Get the current effects (placeholder for now)
    pub fn current_effects(&self) -> Vec<Effect> {
        // This would track currently active effects
        Vec::new()
    }

    /// Get the semantic state (placeholder for now)
    pub fn semantic_state(&self) -> SemanticState {
        SemanticState::default()
    }

    /// Get the current component
    pub fn current_component(&self) -> capability::ComponentId {
        self.component_id
    }
}

/// Unique identifier for an execution instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionId(u64);

impl ExecutionId {
    /// Generate a new unique execution ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// Target platforms for code execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionTarget {
    /// TypeScript/JavaScript target
    TypeScript,
    /// WebAssembly target
    WebAssembly,
    /// Native LLVM target
    Native,
}

/// TypeScript execution configuration
#[derive(Debug, Clone)]
pub struct TypeScriptConfig {
    pub enable_strict_mode: bool,
    pub target_version: String,
}

/// WebAssembly execution configuration
#[derive(Debug, Clone, Default)]
pub struct WebAssemblyConfig {
    pub enable_simd: bool,
    pub memory_pages: u32,
}

/// Native execution configuration
#[derive(Debug, Clone)]
pub struct NativeConfig {
    pub optimization_level: u8,
    pub enable_debug_info: bool,
}

/// Configuration for code execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Maximum memory usage
    pub max_memory: usize,
    /// Enable debug mode
    pub debug_mode: bool,
    /// Performance monitoring level
    pub monitoring_level: MonitoringLevel,
    /// Target-specific configuration
    pub target_config: TargetConfig,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(30),
            max_memory: 1024 * 1024 * 1024, // 1GB
            debug_mode: false,
            monitoring_level: MonitoringLevel::Standard,
            target_config: TargetConfig::default(),
        }
    }
}

/// Levels of performance monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitoringLevel {
    /// Minimal monitoring
    Minimal,
    /// Standard monitoring
    Standard,
    /// Detailed monitoring
    Detailed,
    /// Comprehensive monitoring
    Comprehensive,
}

/// Target-specific configuration
#[derive(Debug, Clone)]
pub enum TargetConfig {
    /// TypeScript-specific configuration
    TypeScript(TypeScriptConfig),
    /// WebAssembly-specific configuration
    WebAssembly(WebAssemblyConfig),
    /// Native-specific configuration
    Native(NativeConfig),
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self::TypeScript(TypeScriptConfig::default())
    }
}

/// AI-readable execution context
#[derive(Debug, Clone, Default)]
pub struct ExecutionAIContext {
    /// Business domain of the execution
    pub business_domain: Option<String>,
    /// Architectural context
    pub architectural_context: Vec<String>,
    /// Performance expectations
    pub performance_expectations: PerformanceExpectations,
    /// Security context
    pub security_context: SecurityContext,
}

/// Performance expectations for execution
#[derive(Debug, Clone, Default)]
pub struct PerformanceExpectations {
    /// Expected latency category
    pub latency_category: LatencyCategory,
    /// Expected throughput requirements
    pub throughput_requirements: ThroughputRequirements,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// Latency categories for execution
#[derive(Debug, Clone, Copy, Default)]
pub enum LatencyCategory {
    /// Real-time (< 1ms)
    RealTime,
    /// Interactive (< 100ms)
    Interactive,
    /// Standard (< 1s)
    #[default]
    Standard,
    /// Batch (> 1s)
    Batch,
}

/// Throughput requirements
#[derive(Debug, Clone, Default)]
pub struct ThroughputRequirements {
    /// Operations per second
    pub ops_per_second: Option<u64>,
    /// Data processing rate (bytes/sec)
    pub bytes_per_second: Option<u64>,
}

/// Resource constraints for execution
#[derive(Debug, Clone, Default)]
pub struct ResourceConstraints {
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// CPU limit as percentage (0.0-1.0)
    pub cpu_limit: Option<f64>,
    /// Network bandwidth limit (bytes/sec)
    pub network_limit: Option<u64>,
}

/// Security context for execution
#[derive(Debug, Clone, Default)]
pub struct SecurityContext {
    /// Security classification level
    pub classification_level: SecurityLevel,
    /// Required security features
    pub required_features: Vec<SecurityFeature>,
    /// Threat model considerations
    pub threat_model: Vec<ThreatCategory>,
}

/// Security levels
#[derive(Debug, Clone, Copy, Default)]
pub enum SecurityLevel {
    /// Public information
    Public,
    /// Internal use
    #[default]
    Internal,
    /// Confidential
    Confidential,
    /// Restricted
    Restricted,
}

/// Security features
#[derive(Debug, Clone)]
pub enum SecurityFeature {
    /// Encryption at rest
    EncryptionAtRest,
    /// Encryption in transit
    EncryptionInTransit,
    /// Access logging
    AccessLogging,
    /// Capability isolation
    CapabilityIsolation,
}

/// Threat categories
#[derive(Debug, Clone)]
pub enum ThreatCategory {
    /// Data exfiltration
    DataExfiltration,
    /// Code injection
    CodeInjection,
    /// Privilege escalation
    PrivilegeEscalation,
    /// Denial of service
    DenialOfService,
}

/// Placeholder for semantic state
#[derive(Debug, Clone, Default)]
pub struct SemanticState {
    /// Current semantic context
    pub context: HashMap<String, String>,
}

/// Execution manager that coordinates target-specific execution
#[derive(Debug)]
pub struct ExecutionManager {
    /// Target-specific execution adapters
    target_adapters: HashMap<ExecutionTarget, TargetAdapterImpl>,
    /// Execution monitoring system
    execution_monitor: Arc<ExecutionMonitor>,
    /// Execution metrics collector
    metrics_collector: Arc<ExecutionMetricsCollector>,
}

/// Concrete implementation of target adapters
#[derive(Debug)]
pub enum TargetAdapterImpl {
    TypeScript(TypeScriptAdapter),
    WebAssembly(WebAssemblyAdapter),
    Native(NativeAdapter),
}

impl TargetAdapterImpl {
    /// Execute code on this target adapter
    pub fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &crate::authority::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.execute(code, capabilities, context),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.execute(code, capabilities, context),
            TargetAdapterImpl::Native(adapter) => adapter.execute(code, capabilities, context),
        }
    }
    
    /// Get adapter name
    pub fn name(&self) -> &'static str {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.name(),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.name(),
            TargetAdapterImpl::Native(adapter) => adapter.name(),
        }
    }
    
    /// Check if adapter is available
    pub fn is_available(&self) -> bool {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.is_available(),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.is_available(),
            TargetAdapterImpl::Native(adapter) => adapter.is_available(),
        }
    }
}

/// Trait for target-specific execution adapters
pub trait TargetAdapter: Send + Sync {
    /// Execute code on this target
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &crate::authority::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError>;
    
    /// Get adapter name
    fn name(&self) -> &'static str;
    
    /// Check if adapter is available
    fn is_available(&self) -> bool { true }
}

/// TypeScript execution adapter
#[derive(Debug)]
pub struct TypeScriptAdapter {
    /// Adapter configuration
    config: TypeScriptConfig,
}

impl TypeScriptAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: TypeScriptConfig::default(),
        })
    }
}

impl TargetAdapter for TypeScriptAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &crate::authority::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Transpile code to JavaScript
        // 2. Set up Node.js/V8 execution environment
        // 3. Execute with capability restrictions
        // 4. Return result
        
        // For now, delegate to the code's execute method
        code.execute(capabilities, context)
            .map_err(|e| match e {
                crate::RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                crate::RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("TypeScript execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "TypeScript"
    }
}

/// WebAssembly execution adapter
#[derive(Debug)]
pub struct WebAssemblyAdapter {
    /// WASM runtime configuration
    config: WasmConfig,
}

impl WebAssemblyAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: WasmConfig::default(),
        })
    }
}

impl TargetAdapter for WebAssemblyAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &crate::authority::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Compile code to WebAssembly
        // 2. Set up WASM runtime (wasmtime, wasmer, etc.)
        // 3. Execute with capability restrictions
        // 4. Return result
        
        // For now, delegate to the code's execute method
        code.execute(capabilities, context)
            .map_err(|e| match e {
                crate::RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                crate::RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("WebAssembly execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "WebAssembly"
    }
}

/// Native execution adapter
#[derive(Debug)]
pub struct NativeAdapter {
    /// Native execution configuration
    config: NativeConfig,
}

impl NativeAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: NativeConfig::default(),
        })
    }
}

impl TargetAdapter for NativeAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &crate::authority::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Compile code to native binary
        // 2. Set up native execution environment
        // 3. Execute with capability restrictions
        // 4. Return result
        
        // For now, delegate to the code's execute method
        code.execute(capabilities, context)
            .map_err(|e| match e {
                crate::RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                crate::RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("Native execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "Native"
    }
}

/// Execution monitoring system
#[derive(Debug)]
pub struct ExecutionMonitor {
    /// Active monitoring sessions
    active_monitors: Arc<RwLock<HashMap<MonitoringHandle, MonitoringSession>>>,
    /// Monitor configuration
    config: MonitorConfig,
}

impl ExecutionMonitor {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            active_monitors: Arc::new(RwLock::new(HashMap::new())),
            config: MonitorConfig::default(),
        })
    }
    
    pub fn start_monitoring(&self, context: &ExecutionContext) -> Result<MonitoringHandle, ExecutionError> {
        let handle = MonitoringHandle::new();
        let session = MonitoringSession::new(context.clone());
        
        {
            let mut monitors = self.active_monitors.write().unwrap();
            monitors.insert(handle, session);
        }
        
        Ok(handle)
    }
    
    pub fn stop_monitoring(&self, handle: MonitoringHandle) -> Result<ExecutionMetrics, ExecutionError> {
        let session = {
            let mut monitors = self.active_monitors.write().unwrap();
            monitors.remove(&handle)
                .ok_or_else(|| ExecutionError::MonitoringFailed {
                    reason: "Monitor session not found".to_string(),
                })?
        };
        
        Ok(session.finalize())
    }
}

/// Monitoring handle for tracking execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MonitoringHandle(Uuid);

impl MonitoringHandle {
    fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Active monitoring session
#[derive(Debug)]
pub struct MonitoringSession {
    /// Execution context being monitored
    context: ExecutionContext,
    /// Start time
    start_time: Instant,
    /// Resource usage at start
    initial_resources: ResourceSnapshot,
}

impl MonitoringSession {
    fn new(context: ExecutionContext) -> Self {
        Self {
            context,
            start_time: Instant::now(),
            initial_resources: ResourceSnapshot::current(),
        }
    }
    
    fn finalize(self) -> ExecutionMetrics {
        let duration = self.start_time.elapsed();
        let final_resources = ResourceSnapshot::current();
        
        ExecutionMetrics {
            duration,
            resource_usage: final_resources.diff(&self.initial_resources),
            success: true, // Would be set based on actual execution result
        }
    }
}

/// Snapshot of system resources
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Memory usage in bytes
    memory_bytes: u64,
    /// CPU time in nanoseconds
    cpu_time_ns: u64,
}

impl ResourceSnapshot {
    fn current() -> Self {
        Self {
            memory_bytes: crate::resources::effects::EffectTracker::get_memory_usage(),
            cpu_time_ns: crate::resources::effects::EffectTracker::get_cpu_time_ns(),
        }
    }
    
    fn diff(&self, other: &ResourceSnapshot) -> ResourceUsage {
        ResourceUsage {
            memory_delta: self.memory_bytes.saturating_sub(other.memory_bytes),
            cpu_time_delta: self.cpu_time_ns.saturating_sub(other.cpu_time_ns),
        }
    }
}

/// Resource usage during execution
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage change in bytes
    pub memory_delta: u64,
    /// CPU time used in nanoseconds
    pub cpu_time_delta: u64,
}

/// Execution metrics collected during monitoring
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Duration of execution
    pub duration: Duration,
    /// Resource usage during execution
    pub resource_usage: ResourceUsage,
    /// Whether execution was successful
    pub success: bool,
}

/// Metrics collector for execution data
#[derive(Debug)]
pub struct ExecutionMetricsCollector {
    /// Collected metrics
    metrics: Arc<RwLock<Vec<ExecutionRecord>>>,
}

impl ExecutionMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub fn record_execution(&self, metrics: &ExecutionMetrics, context: &ExecutionContext) {
        let record = ExecutionRecord {
            execution_id: context.execution_id,
            target: context.target.clone(),
            metrics: metrics.clone(),
            timestamp: SystemTime::now(),
        };
        
        let mut metrics_store = self.metrics.write().unwrap();
        metrics_store.push(record);
        
        // Keep only last 1000 records to prevent unbounded growth
        if metrics_store.len() > 1000 {
            metrics_store.remove(0);
        }
    }
}

/// Record of execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Execution ID
    pub execution_id: ExecutionId,
    /// Target platform
    pub target: ExecutionTarget,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Timestamp
    pub timestamp: SystemTime,
}



impl Default for TypeScriptConfig {
    fn default() -> Self {
        Self {
            node_version: "18.0.0".to_string(),
            ts_options: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// WASM runtime to use
    pub runtime: String,
    /// Memory limits
    pub memory_limit_mb: usize,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            runtime: "wasmtime".to_string(),
            memory_limit_mb: 64,
        }
    }
}



impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            opt_level: 2,
            debug_symbols: false,
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enable_resource_monitoring: true,
            monitoring_interval: Duration::from_millis(100),
        }
    }
} 

/// Execution-related errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// Unsupported target
    #[error("Unsupported execution target: {target:?}")]
    UnsupportedTarget {
        /// Target that is not supported
        target: ExecutionTarget,
    },

    /// Insufficient capabilities for execution
    #[error("Insufficient capabilities - required: {required:?}, available: {available:?}")]
    InsufficientCapabilities {
        /// Required capabilities
        required: Vec<String>,
        /// Available capabilities
        available: Vec<String>,
    },

    /// Monitoring failed
    #[error("Execution monitoring failed: {reason}")]
    MonitoringFailed {
        /// Reason for failure
        reason: String,
    },

    /// Capability error during execution
    #[error("Capability error: {0}")]
    Capability(#[from] crate::authority::CapabilityError),

    /// Resource error during execution
    #[error("Resource error: {0}")]
    Resource(#[from] crate::resources::ResourceError),

    /// Generic execution error
    #[error("Execution error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
} 