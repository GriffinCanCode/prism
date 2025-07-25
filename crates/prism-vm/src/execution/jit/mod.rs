//! Prism VM JIT Compilation System
//!
//! This module implements an innovative tiered JIT compiler that combines proven techniques
//! from leading JIT implementations with novel optimizations tailored for Prism's unique
//! semantic preservation and capability-based security model.
//!
//! ## Architecture Overview
//!
//! The Prism JIT uses a **3-tier adaptive compilation system**:
//!
//! 1. **Interpreter Tier**: Fast startup, collects profiling data
//! 2. **Baseline JIT**: Template-based compilation for hot functions
//! 3. **Optimizing JIT**: Advanced optimizations for critical hot paths
//!
//! ## Key Innovations
//!
//! - **Semantic-Aware Optimization**: Preserves business logic semantics during compilation
//! - **Capability-Integrated Codegen**: Security checks compiled into generated code
//! - **Effect-Guided Optimization**: Uses effect information for advanced optimizations
//! - **Adaptive Profiling**: ML-inspired profiling for tier decision making
//! - **Copy-and-Patch Foundation**: Fast baseline compilation using template patching
//! - **Futamura-Style Specialization**: Partial evaluation for interpreter specialization
//!
//! ## Integration with Existing Infrastructure
//!
//! This JIT system is designed to integrate seamlessly with existing Prism infrastructure:
//! - **prism-runtime**: Uses existing performance profiling and concurrency systems
//! - **prism-codegen**: Leverages existing code generation backends
//! - **prism-vm**: Extends existing VM execution capabilities
//! - **No Logic Duplication**: Interfaces with rather than reimplements existing functionality
//!
//! ## Module Structure
//!
//! - [`profiler`] - Integration with prism-runtime performance profiling
//! - [`baseline`] - Fast template-based baseline JIT compiler
//! - [`optimizing`] - Advanced optimizing JIT compiler
//! - [`codegen`] - Integration with prism-codegen infrastructure
//! - [`runtime`] - JIT runtime support and integration
//! - [`cache`] - Compiled code caching and management
//! - [`analysis`] - Static analysis for optimization opportunities
//! - [`security`] - Capability and effect integration for JIT code

pub mod profiler;
pub mod baseline;
pub mod optimizing;
pub mod codegen;
pub mod runtime;
pub mod cache;
pub mod analysis;
pub mod security;
pub mod capability_guards;
pub mod security_constraint_optimizer;
pub mod egraph_optimizer;
pub mod profile_guided_optimizer;

// Re-export main types with integration-focused naming
pub use profiler::{
    JitProfilerIntegration as AdaptiveProfiler, 
    JitProfileData as ProfileData, 
    TierDecision, 
    CompilationTier,
    JitProfilingConfig as ProfilerConfig, 
    ProfilingEvent
};
pub use baseline::{
    BaselineJIT, BaselineCompiler, TemplateEngine, PatchPoint,
    BaselineConfig, CompilationResult
};
pub use optimizing::{
    OptimizingJIT, OptimizingCompiler, OptimizationPipeline, IRBuilder,
    OptimizingConfig, OptimizationLevel, CompilationMetadata
};
pub use codegen::{
    JitCodeGenerator as CodeGenerator, 
    TargetISA, 
    MachineCode, 
    CodeBuffer,
    JitCodeGenConfig as GeneratorConfig, 
    ISAFeatures
};
pub use runtime::{
    JITRuntime, CompiledFunction, ExecutionEngine, RuntimeConfig,
    DeoptimizationPoint, OSREntry, CompilationMetadata as RuntimeCompilationMetadata,
    OptimizationSummary, OptimizationCategory
};
pub use cache::{
    CodeCache, CacheEntry, CachePolicy, CacheStats,
    EvictionStrategy, CacheConfig
};
pub use analysis::{
    StaticAnalyzer, AnalysisConfig, AnalysisResult,
    control_flow::ControlFlowGraph, 
    data_flow::DataFlowAnalysis, 
    loop_analysis::LoopAnalysis,
    optimization_opportunities::OptimizationOpportunity
};
pub use security::{
    SecurityCompiler, CapabilityChecker, EffectTracker, SecurityPolicy,
    SecurityConfig, SecurityViolation
};
pub use capability_guards::{
    CapabilityGuardGenerator, GuardGeneratorConfig, CapabilityGuardAnalysis,
    RequiredGuard, GuardType, GuardExecutionResult, GuardFailureResponse,
    RuntimeCapabilityValidator, DeoptimizationManager, SecurityAuditLogger
};
pub use security_constraint_optimizer::{
    SecurityConstraintOptimizer, ConstraintOptimizerConfig, OptimizedConstraintValidation,
    HotConstraintPath, ConstraintType, OptimizedValidationResult, SpecializedValidator
};
pub use egraph_optimizer::{
    EGraphOptimizer, EGraphConfig, OptimizedFunction
};
pub use profile_guided_optimizer::{
    ProfileGuidedOptimizer, PGOConfig, RuntimeProfiler, HotSpotDetector
};

use crate::{VMResult, PrismVMError, bytecode::PrismBytecode};
use prism_runtime::{
    authority::capability::CapabilitySet,
    concurrency::performance::PerformanceProfiler,
};
use prism_codegen::backends::PrismVMBackend;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};

/// Main JIT compiler system coordinating all tiers with existing infrastructure integration
#[derive(Debug)]
pub struct JitCompiler {
    /// JIT configuration
    config: JitConfig,
    
    /// Integration with runtime performance profiler
    profiler_integration: Arc<RwLock<profiler::JitProfilerIntegration>>,
    
    /// Baseline JIT compiler
    baseline_jit: Arc<BaselineJIT>,
    
    /// Optimizing JIT compiler
    optimizing_jit: Arc<OptimizingJIT>,
    
    /// JIT runtime system
    runtime: Arc<JITRuntime>,
    
    /// Compiled code cache
    code_cache: Arc<RwLock<CodeCache>>,
    
    /// Security integration
    security_compiler: Arc<SecurityCompiler>,
    
    /// Security constraint optimizer
    constraint_optimizer: Arc<Mutex<SecurityConstraintOptimizer>>,
    
    /// Integration with prism-codegen VM backend
    vm_backend: Arc<PrismVMBackend>,
    
    /// Compilation statistics
    stats: Arc<RwLock<JitStats>>,
}

/// JIT compiler configuration with integration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitConfig {
    /// Enable JIT compilation
    pub enabled: bool,
    
    /// Profiler integration configuration
    pub profiler_config: profiler::JitProfilingConfig,
    
    /// Baseline compiler configuration
    pub baseline_config: BaselineConfig,
    
    /// Optimizing compiler configuration
    pub optimizing_config: OptimizingConfig,
    
    /// Runtime configuration
    pub runtime_config: RuntimeConfig,
    
    /// Cache configuration
    pub cache_config: CacheConfig,
    
    /// Security configuration
    pub security_config: SecurityConfig,
    
    /// Constraint optimizer configuration
    pub constraint_optimizer_config: ConstraintOptimizerConfig,
    
    /// Code generation configuration
    pub codegen_config: codegen::JitCodeGenConfig,
    
    /// Integration settings
    pub integration_config: IntegrationConfig,
}

/// Integration configuration for existing infrastructure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Use prism-runtime performance profiler
    pub use_runtime_profiler: bool,
    
    /// Use prism-codegen VM backend
    pub use_codegen_backend: bool,
    
    /// Integration with prism-runtime concurrency system
    pub integrate_concurrency: bool,
    
    /// Share capability system with runtime
    pub share_capabilities: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            use_runtime_profiler: true,
            use_codegen_backend: true,
            integrate_concurrency: true,
            share_capabilities: true,
        }
    }
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            profiler_config: profiler::JitProfilingConfig::default(),
            baseline_config: BaselineConfig::default(),
            optimizing_config: OptimizingConfig::default(),
            runtime_config: RuntimeConfig::default(),
            cache_config: CacheConfig::default(),
            security_config: SecurityConfig::default(),
            constraint_optimizer_config: ConstraintOptimizerConfig::default(),
            codegen_config: codegen::JitCodeGenConfig::default(),
            integration_config: IntegrationConfig::default(),
        }
    }
}

/// JIT compilation statistics
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    /// Functions compiled with baseline JIT
    pub baseline_compilations: u64,
    /// Functions compiled with optimizing JIT
    pub optimizing_compilations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total compilation time
    pub total_compilation_time: Duration,
    /// Average compilation time
    pub avg_compilation_time: Duration,
    /// Performance improvements achieved
    pub performance_improvements: Vec<f64>,
}

impl JitCompiler {
    /// Create a new JIT compiler with default configuration
    pub fn new() -> VMResult<Self> {
        Self::with_config(JitConfig::default())
    }

    /// Create a new JIT compiler with custom configuration and infrastructure integration
    pub fn with_config(config: JitConfig) -> VMResult<Self> {
        let _span = span!(Level::INFO, "jit_init").entered();
        info!("Initializing Prism JIT compiler with infrastructure integration");
        
        // Create runtime performance profiler if integration is enabled
        let runtime_profiler = if config.integration_config.use_runtime_profiler {
            Arc::new(PerformanceProfiler::new().map_err(|e| PrismVMError::JITError {
                message: format!("Failed to create runtime profiler: {}", e),
            })?)
        } else {
            // Create a minimal profiler for standalone operation
            Arc::new(PerformanceProfiler::new().map_err(|e| PrismVMError::JITError {
                message: format!("Failed to create profiler: {}", e),
            })?)
        };
        
        let profiler_integration = Arc::new(RwLock::new(
            profiler::JitProfilerIntegration::new(
                runtime_profiler,
                config.profiler_config.clone()
            )?
        ));
        
        let baseline_jit = Arc::new(
            BaselineJIT::new(config.baseline_config.clone())?
        );
        
        let optimizing_jit = Arc::new(
            OptimizingJIT::new(config.optimizing_config.clone())?
        );
        
        let runtime = Arc::new(
            JITRuntime::new(config.runtime_config.clone())?
        );
        
        let code_cache = Arc::new(RwLock::new(
            CodeCache::new(config.cache_config.clone())?
        ));
        
        let security_compiler = Arc::new(
            SecurityCompiler::new(config.security_config.clone())?
        );
        
        let constraint_optimizer = Arc::new(Mutex::new(
            SecurityConstraintOptimizer::new(config.constraint_optimizer_config.clone())?
        ));
        
        // Create VM backend integration if enabled
        let vm_backend = if config.integration_config.use_codegen_backend {
            Arc::new(PrismVMBackend::new(
                prism_codegen::backends::CodeGenConfig::default()
            ).map_err(|e| PrismVMError::JITError {
                message: format!("Failed to create VM backend: {}", e),
            })?)
        } else {
            // Create a minimal backend for standalone operation
            Arc::new(PrismVMBackend::new(
                prism_codegen::backends::CodeGenConfig::default()
            ).map_err(|e| PrismVMError::JITError {
                message: format!("Failed to create VM backend: {}", e),
            })?)
        };
        
        Ok(Self {
            config,
            profiler_integration,
            baseline_jit,
            optimizing_jit,
            runtime,
            code_cache,
            security_compiler,
            constraint_optimizer,
            vm_backend,
            stats: Arc::new(RwLock::new(JitStats::default())),
        })
    }
    
    /// Attempt to compile a function using the appropriate tier with infrastructure integration
    pub fn try_compile(
        &self,
        bytecode: &PrismBytecode,
        function_id: u32,
        capabilities: &CapabilitySet,
    ) -> VMResult<Option<CompiledFunction>> {
        let _span = span!(Level::DEBUG, "jit_try_compile", function_id = function_id).entered();
        
        if !self.config.enabled {
            return Ok(None);
        }
        
        // Check if already compiled and cached
        if let Some(cached) = self.code_cache.read().unwrap().get(function_id) {
            self.stats.write().unwrap().cache_hits += 1;
            return Ok(Some(cached.clone()));
        }
        
        self.stats.write().unwrap().cache_misses += 1;
        
        // Use integrated profiling to decide compilation tier
        let tier_decision = self.profiler_integration.write().unwrap().decide_tier(function_id);
        
        debug!("Tier decision for function {}: {:?}", function_id, tier_decision);
        
        let compiled_function = match tier_decision {
            TierDecision::Interpret => {
                // Continue interpreting, no compilation
                return Ok(None);
            }
            TierDecision::BaselineCompile => {
                self.compile_baseline(bytecode, function_id, capabilities)?
            }
            TierDecision::OptimizingCompile => {
                self.compile_optimizing(bytecode, function_id, capabilities)?
            }
        };
        
        // Cache the compiled function
        if let Some(ref compiled) = compiled_function {
            self.code_cache.write().unwrap().insert(function_id, compiled.clone());
        }
        
        Ok(compiled_function)
    }
    
    /// Compile using baseline JIT with infrastructure integration
    fn compile_baseline(
        &self,
        bytecode: &PrismBytecode,
        function_id: u32,
        capabilities: &CapabilitySet,
    ) -> VMResult<Option<CompiledFunction>> {
        let start_time = Instant::now();
        
        let function = bytecode.functions.iter()
            .find(|f| f.id == function_id)
            .ok_or_else(|| PrismVMError::JITError {
                message: format!("Function {} not found", function_id),
            })?;
        
        let compiled = self.baseline_jit.compile(bytecode, function)?;
        let compilation_time = start_time.elapsed();
        
        // Record compilation attempt with integrated profiler
        self.profiler_integration.write().unwrap().record_compilation_attempt(
            function_id,
            CompilationTier::Baseline,
            true,
            compilation_time,
            None, // Performance improvement measured later
        );
        
        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.baseline_compilations += 1;
        stats.total_compilation_time += compilation_time;
        stats.avg_compilation_time = stats.total_compilation_time / 
            (stats.baseline_compilations + stats.optimizing_compilations);
        
        Ok(Some(compiled))
    }
    
    /// Compile using optimizing JIT with infrastructure integration
    fn compile_optimizing(
        &self,
        bytecode: &PrismBytecode,
        function_id: u32,
        capabilities: &CapabilitySet,
    ) -> VMResult<Option<CompiledFunction>> {
        let start_time = Instant::now();
        
        let function = bytecode.functions.iter()
            .find(|f| f.id == function_id)
            .ok_or_else(|| PrismVMError::JITError {
                message: format!("Function {} not found", function_id),
            })?;
        
        // Get runtime optimization hints from integrated profiler
        let optimization_hints = self.profiler_integration.read().unwrap().get_runtime_hints();
        
        let compiled = self.optimizing_jit.compile_with_hints(
            bytecode, 
            function, 
            capabilities,
            &optimization_hints
        )?;
        
        let compilation_time = start_time.elapsed();
        
        // Record compilation attempt with integrated profiler
        self.profiler_integration.write().unwrap().record_compilation_attempt(
            function_id,
            CompilationTier::Optimizing,
            true,
            compilation_time,
            None, // Performance improvement measured later
        );
        
        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.optimizing_compilations += 1;
        stats.total_compilation_time += compilation_time;
        stats.avg_compilation_time = stats.total_compilation_time / 
            (stats.baseline_compilations + stats.optimizing_compilations);
        
        Ok(Some(compiled))
    }
    
    /// Record function execution for profiling integration
    pub fn record_execution(&self, function_id: u32, execution_time: Duration) {
        self.profiler_integration.write().unwrap().record_execution(function_id, execution_time);
    }
    
    /// Get JIT compilation statistics
    pub fn get_stats(&self) -> JitStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Get integrated profiling statistics
    pub fn get_profiling_stats(&self) -> profiler::JitProfilingStats {
        self.profiler_integration.read().unwrap().get_stats()
    }
    
    /// Check if JIT compilation is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
    
    /// Get integration configuration
    pub fn integration_config(&self) -> &IntegrationConfig {
        &self.config.integration_config
    }
} 