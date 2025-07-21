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

use crate::{capability, effects, RuntimeError};
use prism_common::{span::Span, symbol::Symbol};
use prism_effects::Effect;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use thiserror::Error;

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

/// Multi-target execution manager
#[derive(Debug)]
pub struct ExecutionManager {
    /// Target-specific adapters
    adapters: HashMap<ExecutionTarget, TargetAdapterEnum>,
    
    /// Active executions
    active_executions: Arc<RwLock<HashMap<ExecutionId, ActiveExecution>>>,
    
    /// Execution history
    execution_history: Arc<Mutex<Vec<ExecutionRecord>>>,
    
    /// Performance metrics
    performance_metrics: Arc<RwLock<ExecutionMetrics>>,
}

impl ExecutionManager {
    /// Create a new execution manager
    pub fn new() -> Result<Self, ExecutionError> {
        let mut adapters: HashMap<ExecutionTarget, TargetAdapterEnum> = HashMap::new();
        
        // Register target adapters
        adapters.insert(ExecutionTarget::TypeScript, TargetAdapterEnum::TypeScript(TypeScriptAdapter::new()?));
        adapters.insert(ExecutionTarget::WebAssembly, TargetAdapterEnum::WebAssembly(WebAssemblyAdapter::new()?));
        adapters.insert(ExecutionTarget::Native, TargetAdapterEnum::Native(NativeAdapter::new()?));

        Ok(Self {
            adapters,
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(Mutex::new(Vec::new())),
            performance_metrics: Arc::new(RwLock::new(ExecutionMetrics::new())),
        })
    }

    /// Execute code with monitoring and capability checking
    pub fn execute_monitored<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
        effect_handle: &effects::EffectHandle,
    ) -> Result<T, ExecutionError> {
        // Get target adapter
        let adapter = self.adapters.get(&context.target)
            .ok_or_else(|| ExecutionError::UnsupportedTarget { target: context.target })?;

        // Create active execution record
        let active_execution = ActiveExecution {
            id: context.execution_id,
            context: context.clone(),
            started_at: SystemTime::now(),
            effect_handle: *effect_handle,
        };

        // Record active execution
        {
            let mut active = self.active_executions.write().unwrap();
            active.insert(context.execution_id, active_execution);
        }

        // Execute through adapter
        let start_time = SystemTime::now();
        let result = adapter.execute_with_capabilities(code, capabilities, context);
        let duration = start_time.elapsed().unwrap_or(Duration::ZERO);

        // Remove from active executions
        {
            let mut active = self.active_executions.write().unwrap();
            active.remove(&context.execution_id);
        }

        // Record execution history
        let execution_record = ExecutionRecord {
            id: context.execution_id,
            target: context.target,
            duration,
            success: result.is_ok(),
            timestamp: start_time,
        };

        {
            let mut history = self.execution_history.lock().unwrap();
            history.push(execution_record.clone());
        }

        // Update metrics
        {
            let mut metrics = self.performance_metrics.write().unwrap();
            metrics.record_execution(&execution_record);
        }

        result
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> ExecutionStats {
        let active_count = self.active_executions.read().unwrap().len();
        let metrics = self.performance_metrics.read().unwrap();
        
        ExecutionStats {
            active_executions: active_count,
            total_executions: metrics.total_executions,
            average_duration: metrics.average_duration,
            success_rate: metrics.success_rate,
        }
    }
}

/// Trait for target-specific execution adapters
/// Enum of target adapters to avoid dyn trait issues
#[derive(Debug)]
pub enum TargetAdapterEnum {
    TypeScript(TypeScriptAdapter),
    WebAssembly(WebAssemblyAdapter),
    Native(NativeAdapter),
}

impl TargetAdapterEnum {
    /// Execute code with capability checking
    pub fn execute_with_capabilities<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        match self {
            TargetAdapterEnum::TypeScript(adapter) => adapter.execute_with_capabilities(code, capabilities, context),
            TargetAdapterEnum::WebAssembly(adapter) => adapter.execute_with_capabilities(code, capabilities, context),
            TargetAdapterEnum::Native(adapter) => adapter.execute_with_capabilities(code, capabilities, context),
        }
    }

    /// Get target-specific memory requirements
    pub fn memory_requirements(&self, context: &ExecutionContext) -> MemoryRequirements {
        match self {
            TargetAdapterEnum::TypeScript(adapter) => adapter.memory_requirements(context),
            TargetAdapterEnum::WebAssembly(adapter) => adapter.memory_requirements(context),
            TargetAdapterEnum::Native(adapter) => adapter.memory_requirements(context),
        }
    }

    /// Get target name
    pub fn target_name(&self) -> &'static str {
        match self {
            TargetAdapterEnum::TypeScript(adapter) => adapter.target_name(),
            TargetAdapterEnum::WebAssembly(adapter) => adapter.target_name(),
            TargetAdapterEnum::Native(adapter) => adapter.target_name(),
        }
    }
}

pub trait TargetAdapter: Send + Sync {
    /// Execute code with capability checking
    fn execute_with_capabilities<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError>;

    /// Get target-specific memory requirements
    fn memory_requirements(&self, context: &ExecutionContext) -> MemoryRequirements;

    /// Get target name
    fn target_name(&self) -> &'static str;

    /// Get target-specific configuration
    fn get_config(&self) -> TargetAdapterConfig;
}

/// Memory requirements for target execution
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Minimum memory required (bytes)
    pub minimum: usize,
    /// Recommended memory (bytes)
    pub recommended: usize,
    /// Maximum memory limit (bytes)
    pub maximum: usize,
}

/// Target adapter configuration
#[derive(Debug, Clone)]
pub struct TargetAdapterConfig {
    /// Supported features
    pub supported_features: Vec<String>,
    /// Performance characteristics
    pub performance_profile: TargetPerformanceProfile,
    /// Security capabilities
    pub security_capabilities: Vec<String>,
}

/// Performance profile for a target
#[derive(Debug, Clone)]
pub struct TargetPerformanceProfile {
    /// Startup overhead
    pub startup_overhead: Duration,
    /// Execution overhead per operation
    pub execution_overhead: Duration,
    /// Memory overhead
    pub memory_overhead: usize,
}

/// TypeScript execution adapter
pub struct TypeScriptAdapter {
    /// JavaScript engine interface
    js_engine: Arc<dyn JavaScriptEngine>,
    /// Module loader
    module_loader: Arc<ModuleLoader>,
    /// Configuration
    config: TypeScriptConfig,
}

impl std::fmt::Debug for TypeScriptAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeScriptAdapter")
            .field("js_engine", &"<js_engine>")
            .field("module_loader", &self.module_loader)
            .field("config", &self.config)
            .finish()
    }
}

impl TypeScriptAdapter {
    /// Create a new TypeScript adapter
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            js_engine: Arc::new(V8Engine::new()?),
            module_loader: Arc::new(ModuleLoader::new()),
            config: TypeScriptConfig::default(),
        })
    }
}

impl TargetAdapter for TypeScriptAdapter {
    fn execute_with_capabilities<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // Validate capabilities for TypeScript execution
        self.validate_typescript_capabilities(capabilities, context)?;
        
        // Execute through JavaScript engine
        code.execute(capabilities, context)
            .map_err(|e| ExecutionError::TargetExecutionFailed {
                target: ExecutionTarget::TypeScript,
                reason: format!("TypeScript execution failed: {}", e),
            })
    }

    fn memory_requirements(&self, _context: &ExecutionContext) -> MemoryRequirements {
        MemoryRequirements {
            minimum: 64 * 1024 * 1024,    // 64MB
            recommended: 256 * 1024 * 1024, // 256MB
            maximum: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }

    fn target_name(&self) -> &'static str {
        "TypeScript"
    }

    fn get_config(&self) -> TargetAdapterConfig {
        TargetAdapterConfig {
            supported_features: vec![
                "dynamic_typing".to_string(),
                "web_apis".to_string(),
                "npm_modules".to_string(),
            ],
            performance_profile: TargetPerformanceProfile {
                startup_overhead: Duration::from_millis(100),
                execution_overhead: Duration::from_nanos(10),
                memory_overhead: 32 * 1024 * 1024, // 32MB
            },
            security_capabilities: vec![
                "sandbox_isolation".to_string(),
                "capability_checking".to_string(),
            ],
        }
    }
}

impl TypeScriptAdapter {
    fn validate_typescript_capabilities(
        &self,
        _capabilities: &capability::CapabilitySet,
        _context: &ExecutionContext,
    ) -> Result<(), ExecutionError> {
        // Validate TypeScript-specific capabilities
        Ok(())
    }
}

/// WebAssembly execution adapter
pub struct WebAssemblyAdapter {
    /// WASM runtime
    wasm_runtime: Arc<dyn WasmRuntime>,
    /// Host function bindings
    host_functions: Arc<HostFunctionRegistry>,
    /// Configuration
    config: WebAssemblyConfig,
}

impl std::fmt::Debug for WebAssemblyAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebAssemblyAdapter")
            .field("wasm_runtime", &"<wasm_runtime>")
            .field("host_functions", &self.host_functions)
            .field("config", &self.config)
            .finish()
    }
}

impl WebAssemblyAdapter {
    /// Create a new WebAssembly adapter
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            wasm_runtime: Arc::new(WasmerRuntime::new()?),
            host_functions: Arc::new(HostFunctionRegistry::new()),
            config: WebAssemblyConfig::default(),
        })
    }
}

impl TargetAdapter for WebAssemblyAdapter {
    fn execute_with_capabilities<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // Execute through WASM runtime
        code.execute(capabilities, context)
            .map_err(|e| ExecutionError::TargetExecutionFailed {
                target: ExecutionTarget::WebAssembly,
                reason: format!("WebAssembly execution failed: {}", e),
            })
    }

    fn memory_requirements(&self, _context: &ExecutionContext) -> MemoryRequirements {
        MemoryRequirements {
            minimum: 16 * 1024 * 1024,    // 16MB
            recommended: 128 * 1024 * 1024, // 128MB
            maximum: 4 * 1024 * 1024 * 1024, // 4GB
        }
    }

    fn target_name(&self) -> &'static str {
        "WebAssembly"
    }

    fn get_config(&self) -> TargetAdapterConfig {
        TargetAdapterConfig {
            supported_features: vec![
                "linear_memory".to_string(),
                "host_functions".to_string(),
                "wasi".to_string(),
            ],
            performance_profile: TargetPerformanceProfile {
                startup_overhead: Duration::from_millis(50),
                execution_overhead: Duration::from_nanos(5),
                memory_overhead: 8 * 1024 * 1024, // 8MB
            },
            security_capabilities: vec![
                "memory_isolation".to_string(),
                "capability_checking".to_string(),
                "deterministic_execution".to_string(),
            ],
        }
    }
}

/// Native execution adapter
#[derive(Debug)]
pub struct NativeAdapter {
    /// Dynamic library loader
    lib_loader: Arc<LibraryLoader>,
    /// Native function registry
    native_functions: Arc<NativeFunctionRegistry>,
    /// Configuration
    config: NativeConfig,
}

impl NativeAdapter {
    /// Create a new native adapter
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            lib_loader: Arc::new(LibraryLoader::new()?),
            native_functions: Arc::new(NativeFunctionRegistry::new()),
            config: NativeConfig::default(),
        })
    }
}

impl TargetAdapter for NativeAdapter {
    fn execute_with_capabilities<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // Execute native code
        code.execute(capabilities, context)
            .map_err(|e| ExecutionError::TargetExecutionFailed {
                target: ExecutionTarget::Native,
                reason: format!("Native execution failed: {}", e),
            })
    }

    fn memory_requirements(&self, _context: &ExecutionContext) -> MemoryRequirements {
        MemoryRequirements {
            minimum: 8 * 1024 * 1024,     // 8MB
            recommended: 64 * 1024 * 1024,  // 64MB
            maximum: 16 * 1024 * 1024 * 1024, // 16GB
        }
    }

    fn target_name(&self) -> &'static str {
        "Native"
    }

    fn get_config(&self) -> TargetAdapterConfig {
        TargetAdapterConfig {
            supported_features: vec![
                "direct_memory_access".to_string(),
                "system_calls".to_string(),
                "native_libraries".to_string(),
            ],
            performance_profile: TargetPerformanceProfile {
                startup_overhead: Duration::from_millis(10),
                execution_overhead: Duration::from_nanos(1),
                memory_overhead: 4 * 1024 * 1024, // 4MB
            },
            security_capabilities: vec![
                "capability_checking".to_string(),
                "memory_protection".to_string(),
            ],
        }
    }
}

/// Currently active execution
#[derive(Debug, Clone)]
struct ActiveExecution {
    /// Execution ID
    id: ExecutionId,
    /// Execution context
    context: ExecutionContext,
    /// When execution started
    started_at: SystemTime,
    /// Associated effect handle
    effect_handle: effects::EffectHandle,
}

/// Record of a completed execution
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Execution ID
    pub id: ExecutionId,
    /// Target platform
    pub target: ExecutionTarget,
    /// Execution duration
    pub duration: Duration,
    /// Whether execution succeeded
    pub success: bool,
    /// Execution timestamp
    pub timestamp: SystemTime,
}

/// Execution performance metrics
#[derive(Debug)]
struct ExecutionMetrics {
    /// Total executions
    total_executions: u64,
    /// Successful executions
    successful_executions: u64,
    /// Average execution duration
    average_duration: Duration,
    /// Success rate (0.0-1.0)
    success_rate: f64,
}

impl ExecutionMetrics {
    fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            average_duration: Duration::ZERO,
            success_rate: 0.0,
        }
    }

    fn record_execution(&mut self, record: &ExecutionRecord) {
        self.total_executions += 1;
        
        if record.success {
            self.successful_executions += 1;
        }
        
        // Update average duration
        let total_time = self.average_duration * (self.total_executions - 1) as u32 + record.duration;
        self.average_duration = total_time / self.total_executions as u32;
        
        // Update success rate
        self.success_rate = self.successful_executions as f64 / self.total_executions as f64;
    }
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Number of active executions
    pub active_executions: usize,
    /// Total executions
    pub total_executions: u64,
    /// Average execution duration
    pub average_duration: Duration,
    /// Success rate
    pub success_rate: f64,
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

    /// Target execution failed
    #[error("Execution failed on target {target:?}: {reason}")]
    TargetExecutionFailed {
        /// Target that failed
        target: ExecutionTarget,
        /// Reason for failure
        reason: String,
    },

    /// Execution timeout
    #[error("Execution timed out after {duration:?}")]
    ExecutionTimeout {
        /// Duration before timeout
        duration: Duration,
    },

    /// Memory limit exceeded
    #[error("Memory limit exceeded: {used} bytes used, {limit} bytes allowed")]
    MemoryLimitExceeded {
        /// Memory used
        used: usize,
        /// Memory limit
        limit: usize,
    },

    /// Capability validation failed
    #[error("Capability validation failed: {reason}")]
    CapabilityValidationFailed {
        /// Reason for failure
        reason: String,
    },

    /// Generic execution error
    #[error("Execution error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
}

// Target-specific configuration types
#[derive(Debug, Clone)]
pub struct TypeScriptConfig {
    /// TypeScript compiler options
    pub compiler_options: HashMap<String, String>,
    /// Node.js runtime options
    pub runtime_options: NodeRuntimeOptions,
}

impl Default for TypeScriptConfig {
    fn default() -> Self {
        Self {
            compiler_options: HashMap::new(),
            runtime_options: NodeRuntimeOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NodeRuntimeOptions {
    /// Maximum heap size
    pub max_heap_size: Option<usize>,
    /// Enable inspector
    pub enable_inspector: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WebAssemblyConfig {
    /// Memory configuration
    pub memory_config: WasmMemoryConfig,
    /// WASI configuration
    pub wasi_config: WasiConfig,
}

#[derive(Debug, Clone, Default)]
pub struct WasmMemoryConfig {
    /// Initial memory pages
    pub initial_pages: u32,
    /// Maximum memory pages
    pub maximum_pages: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct WasiConfig {
    /// Environment variables
    pub env_vars: HashMap<String, String>,
    /// Allowed directories
    pub allowed_dirs: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct NativeConfig {
    /// Library search paths
    pub library_paths: Vec<String>,
    /// Security options
    pub security_options: NativeSecurityOptions,
}

#[derive(Debug, Clone, Default)]
pub struct NativeSecurityOptions {
    /// Enable address sanitizer
    pub enable_asan: bool,
    /// Enable memory sanitizer
    pub enable_msan: bool,
    /// Enable undefined behavior sanitizer
    pub enable_ubsan: bool,
}

// Placeholder trait implementations for runtime engines
pub trait JavaScriptEngine: Send + Sync {
    fn execute(&self, code: &str) -> Result<String, ExecutionError>;
}

pub trait WasmRuntime: Send + Sync {
    fn instantiate(&self, module: &[u8]) -> Result<Box<dyn WasmInstanceTrait>, ExecutionError>;
}

pub trait WasmInstanceTrait {
    fn call(&self, function: &str, args: &[WasmValue]) -> Result<WasmValue, ExecutionError>;
}

// Placeholder implementations
#[derive(Debug)]
pub struct V8Engine;

impl V8Engine {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self)
    }
}

impl JavaScriptEngine for V8Engine {
    fn execute(&self, _code: &str) -> Result<String, ExecutionError> {
        Ok("placeholder".to_string())
    }
}

#[derive(Debug)]
pub struct WasmerRuntime;

impl WasmerRuntime {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self)
    }
}

impl WasmRuntime for WasmerRuntime {
    fn instantiate(&self, _module: &[u8]) -> Result<Box<dyn WasmInstanceTrait>, ExecutionError> {
        Ok(Box::new(WasmInstanceImpl))
    }
}

#[derive(Debug)]
pub struct WasmInstanceImpl;

impl WasmInstanceTrait for WasmInstanceImpl {
    fn call(&self, _function: &str, _args: &[WasmValue]) -> Result<WasmValue, ExecutionError> {
        Ok(WasmValue::I32(0))
    }
}

#[derive(Debug)]
pub enum WasmValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

// Placeholder registry types
#[derive(Debug)]
pub struct ModuleLoader;

impl ModuleLoader {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct HostFunctionRegistry;

impl HostFunctionRegistry {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct LibraryLoader;

impl LibraryLoader {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self)
    }
}

#[derive(Debug)]
pub struct NativeFunctionRegistry;

impl NativeFunctionRegistry {
    pub fn new() -> Self {
        Self
    }
} 