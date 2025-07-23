//! Execution Context and Configuration
//!
//! This module defines the execution context that provides environment and metadata
//! for code execution across different target platforms.

use crate::authority::capability;
use crate::resources::effects::Effect;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

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
    /// Prism VM target
    PrismVM,
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
    /// Prism VM-specific configuration
    PrismVM(PrismVMConfig),
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self::TypeScript(TypeScriptConfig::default())
    }
}

/// TypeScript execution configuration
#[derive(Debug, Clone)]
pub struct TypeScriptConfig {
    pub enable_strict_mode: bool,
    pub target_version: String,
}

impl Default for TypeScriptConfig {
    fn default() -> Self {
        Self {
            enable_strict_mode: true,
            target_version: "ES2020".to_string(),
        }
    }
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

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            enable_debug_info: false,
        }
    }
}

/// Prism VM configuration
#[derive(Debug, Clone)]
pub struct PrismVMConfig {
    /// Enable JIT compilation
    pub enable_jit: bool,
    /// Maximum stack size
    pub max_stack_size: usize,
    /// Enable debugging
    pub enable_debugging: bool,
}

impl Default for PrismVMConfig {
    fn default() -> Self {
        Self {
            enable_jit: false,  // Disabled by default for now
            max_stack_size: 1024 * 1024,  // 1MB
            enable_debugging: true,
        }
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