//! JIT Profiling Integration
//!
//! This module integrates the JIT compiler with the existing prism-runtime performance
//! monitoring infrastructure. Instead of duplicating profiling logic, it leverages
//! the comprehensive performance metrics system already available in prism-runtime.
//!
//! ## Integration Approach
//!
//! - **Leverages Existing Infrastructure**: Uses prism-runtime's PerformanceProfiler
//! - **VM-Specific Metrics**: Adds JIT-specific metrics on top of existing system
//! - **No Logic Duplication**: Interfaces with rather than reimplements profiling
//! - **Separation of Concerns**: JIT profiling focuses only on compilation decisions

use crate::{VMResult, PrismVMError, bytecode::FunctionDefinition};
use prism_runtime::concurrency::performance::{
    PerformanceProfiler, PerformanceMetrics, OptimizationHint
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, trace};

/// JIT-specific profiling integration that leverages prism-runtime infrastructure
#[derive(Debug)]
pub struct JitProfilerIntegration {
    /// Integration with runtime performance profiler
    runtime_profiler: Arc<PerformanceProfiler>,
    
    /// JIT-specific configuration
    config: JitProfilingConfig,
    
    /// Function-specific compilation history
    compilation_history: HashMap<u32, CompilationHistory>,
    
    /// Tier decision cache
    tier_decisions: HashMap<u32, CachedTierDecision>,
}

/// JIT profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitProfilingConfig {
    /// Baseline compilation threshold (execution count)
    pub baseline_threshold: u32,
    /// Optimizing compilation threshold (execution count)
    pub optimizing_threshold: u32,
    /// Cache tier decisions for this duration
    pub decision_cache_duration: Duration,
    /// Enable adaptive threshold adjustment
    pub adaptive_thresholds: bool,
}

impl Default for JitProfilingConfig {
    fn default() -> Self {
        Self {
            baseline_threshold: 10,
            optimizing_threshold: 100,
            decision_cache_duration: Duration::from_secs(60),
            adaptive_thresholds: true,
        }
    }
}

/// Compilation history for a function
#[derive(Debug, Clone)]
pub struct CompilationHistory {
    /// Function ID
    pub function_id: u32,
    /// Number of times executed
    pub execution_count: u32,
    /// Last execution time
    pub last_execution: Instant,
    /// Current compilation tier
    pub current_tier: CompilationTier,
    /// Compilation attempts
    pub compilation_attempts: Vec<CompilationAttempt>,
}

/// Compilation tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationTier {
    /// Interpreted execution
    Interpreter,
    /// Fast baseline compilation
    Baseline,
    /// Advanced optimizing compilation
    Optimizing,
}

/// Record of a compilation attempt
#[derive(Debug, Clone)]
pub struct CompilationAttempt {
    /// When compilation was attempted
    pub timestamp: Instant,
    /// Target tier
    pub target_tier: CompilationTier,
    /// Whether compilation succeeded
    pub succeeded: bool,
    /// Compilation time
    pub compilation_time: Duration,
    /// Performance improvement (if measurable)
    pub performance_improvement: Option<f64>,
}

/// Cached tier decision
#[derive(Debug, Clone)]
pub struct CachedTierDecision {
    /// The decision
    pub decision: TierDecision,
    /// When decision was made
    pub timestamp: Instant,
    /// Decision confidence (0.0 to 1.0)
    pub confidence: f64,
}

/// Tier compilation decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierDecision {
    /// Continue interpreting
    Interpret,
    /// Compile with baseline JIT
    BaselineCompile,
    /// Compile with optimizing JIT
    OptimizingCompile,
}

/// JIT-specific profile data extracted from runtime metrics
#[derive(Debug, Clone)]
pub struct JitProfileData {
    /// Function execution frequency
    pub execution_frequency: f64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Memory usage pattern
    pub memory_pattern: MemoryPattern,
    /// CPU utilization during execution
    pub cpu_utilization: f64,
    /// Call pattern analysis
    pub call_pattern: CallPattern,
}

/// Memory usage pattern
#[derive(Debug, Clone)]
pub enum MemoryPattern {
    /// Low memory usage, good for baseline
    LowMemory,
    /// Medium memory usage
    MediumMemory,
    /// High memory usage, may benefit from optimization
    HighMemory,
    /// Variable memory usage
    Variable,
}

/// Function call pattern
#[derive(Debug, Clone)]
pub enum CallPattern {
    /// Called frequently in tight loops
    HotLoop,
    /// Called regularly but not in loops
    Regular,
    /// Called occasionally
    Occasional,
    /// Called very rarely
    Rare,
}

impl JitProfilerIntegration {
    /// Create new JIT profiler integration
    pub fn new(
        runtime_profiler: Arc<PerformanceProfiler>,
        config: JitProfilingConfig,
    ) -> VMResult<Self> {
        Ok(Self {
            runtime_profiler,
            config,
            compilation_history: HashMap::new(),
            tier_decisions: HashMap::new(),
        })
    }

    /// Record function execution for profiling
    pub fn record_execution(&mut self, function_id: u32, execution_time: Duration) {
        let history = self.compilation_history
            .entry(function_id)
            .or_insert_with(|| CompilationHistory {
                function_id,
                execution_count: 0,
                last_execution: Instant::now(),
                current_tier: CompilationTier::Interpreter,
                compilation_attempts: Vec::new(),
            });

        history.execution_count += 1;
        history.last_execution = Instant::now();

        trace!("Recorded execution for function {}: count={}", function_id, history.execution_count);
    }

    /// Make tier decision based on runtime performance data and JIT-specific metrics
    pub fn decide_tier(&mut self, function_id: u32) -> TierDecision {
        // Check for cached decision first
        if let Some(cached) = self.tier_decisions.get(&function_id) {
            if cached.timestamp.elapsed() < self.config.decision_cache_duration {
                return cached.decision;
            }
        }

        // Get runtime performance metrics
        let runtime_metrics = self.runtime_profiler.get_current_metrics();
        
        // Extract JIT-specific profile data
        let profile_data = self.extract_jit_profile_data(function_id, &runtime_metrics);
        
        // Get compilation history
        let history = self.compilation_history.get(&function_id);
        
        // Make tier decision using integrated data
        let decision = self.make_tier_decision(&profile_data, history);
        
        // Cache the decision
        self.tier_decisions.insert(function_id, CachedTierDecision {
            decision,
            timestamp: Instant::now(),
            confidence: self.calculate_decision_confidence(&profile_data, history),
        });

        debug!("Tier decision for function {}: {:?}", function_id, decision);
        decision
    }

    /// Extract JIT-specific profile data from runtime metrics
    fn extract_jit_profile_data(
        &self,
        function_id: u32,
        runtime_metrics: &PerformanceMetrics,
    ) -> JitProfileData {
        // Extract relevant metrics from the comprehensive runtime performance data
        let execution_frequency = self.calculate_execution_frequency(function_id, runtime_metrics);
        let avg_execution_time = self.calculate_avg_execution_time(function_id, runtime_metrics);
        let memory_pattern = self.analyze_memory_pattern(function_id, runtime_metrics);
        let cpu_utilization = runtime_metrics.cpu_utilization.overall;
        let call_pattern = self.analyze_call_pattern(function_id, runtime_metrics);

        JitProfileData {
            execution_frequency,
            avg_execution_time,
            memory_pattern,
            cpu_utilization,
            call_pattern,
        }
    }

    /// Make tier decision based on integrated profiling data
    fn make_tier_decision(
        &self,
        profile_data: &JitProfileData,
        history: Option<&CompilationHistory>,
    ) -> TierDecision {
        let execution_count = history.map(|h| h.execution_count).unwrap_or(0);
        let current_tier = history.map(|h| h.current_tier).unwrap_or(CompilationTier::Interpreter);

        // Use runtime performance data to inform decisions
        match current_tier {
            CompilationTier::Interpreter => {
                if execution_count >= self.config.baseline_threshold {
                    // Check if system has capacity for compilation
                    if profile_data.cpu_utilization < 0.8 {
                        return TierDecision::BaselineCompile;
                    }
                }
            }
            CompilationTier::Baseline => {
                if execution_count >= self.config.optimizing_threshold {
                    // Only upgrade to optimizing if there's clear benefit
                    if matches!(profile_data.call_pattern, CallPattern::HotLoop) &&
                       profile_data.cpu_utilization < 0.9 {
                        return TierDecision::OptimizingCompile;
                    }
                }
            }
            CompilationTier::Optimizing => {
                // Already at highest tier
                return TierDecision::Interpret;
            }
        }

        TierDecision::Interpret
    }

    /// Calculate execution frequency from runtime metrics
    fn calculate_execution_frequency(&self, _function_id: u32, metrics: &PerformanceMetrics) -> f64 {
        // Use actor system metrics as a proxy for function execution frequency
        metrics.actor_system.messages_per_sec as f64 / 1000.0
    }

    /// Calculate average execution time from runtime metrics
    fn calculate_avg_execution_time(&self, _function_id: u32, metrics: &PerformanceMetrics) -> Duration {
        Duration::from_millis(metrics.actor_system.avg_message_latency_ms as u64)
    }

    /// Analyze memory usage pattern
    fn analyze_memory_pattern(&self, _function_id: u32, metrics: &PerformanceMetrics) -> MemoryPattern {
        match metrics.memory_utilization.usage {
            usage if usage < 0.3 => MemoryPattern::LowMemory,
            usage if usage < 0.6 => MemoryPattern::MediumMemory,
            usage if usage < 0.8 => MemoryPattern::HighMemory,
            _ => MemoryPattern::Variable,
        }
    }

    /// Analyze function call pattern
    fn analyze_call_pattern(&self, _function_id: u32, metrics: &PerformanceMetrics) -> CallPattern {
        // Use concurrency metrics to infer call patterns
        if metrics.concurrency.parallel_efficiency > 0.8 {
            CallPattern::HotLoop
        } else if metrics.concurrency.avg_task_execution_ms < 1.0 {
            CallPattern::Regular
        } else if metrics.concurrency.avg_task_execution_ms < 10.0 {
            CallPattern::Occasional
        } else {
            CallPattern::Rare
        }
    }

    /// Calculate confidence in tier decision
    fn calculate_decision_confidence(
        &self,
        profile_data: &JitProfileData,
        history: Option<&CompilationHistory>,
    ) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Increase confidence based on execution frequency
        confidence += (profile_data.execution_frequency.min(1.0)) * 0.3;

        // Increase confidence based on historical data
        if let Some(history) = history {
            if history.execution_count > 50 {
                confidence += 0.2;
            }
        }

        confidence.min(1.0)
    }

    /// Record compilation attempt
    pub fn record_compilation_attempt(
        &mut self,
        function_id: u32,
        target_tier: CompilationTier,
        succeeded: bool,
        compilation_time: Duration,
        performance_improvement: Option<f64>,
    ) {
        let history = self.compilation_history
            .entry(function_id)
            .or_insert_with(|| CompilationHistory {
                function_id,
                execution_count: 0,
                last_execution: Instant::now(),
                current_tier: CompilationTier::Interpreter,
                compilation_attempts: Vec::new(),
            });

        history.compilation_attempts.push(CompilationAttempt {
            timestamp: Instant::now(),
            target_tier,
            succeeded,
            compilation_time,
            performance_improvement,
        });

        if succeeded {
            history.current_tier = target_tier;
        }

        debug!("Recorded compilation attempt for function {}: tier={:?}, succeeded={}", 
               function_id, target_tier, succeeded);
    }

    /// Get runtime optimization hints that may affect JIT decisions
    pub fn get_runtime_hints(&self) -> Vec<OptimizationHint> {
        self.runtime_profiler.analyze_performance()
    }

    /// Get compilation statistics
    pub fn get_stats(&self) -> JitProfilingStats {
        JitProfilingStats {
            functions_profiled: self.compilation_history.len(),
            total_executions: self.compilation_history.values()
                .map(|h| h.execution_count as u64)
                .sum(),
            baseline_compilations: self.count_compilations(CompilationTier::Baseline),
            optimizing_compilations: self.count_compilations(CompilationTier::Optimizing),
            cache_hit_rate: self.calculate_cache_hit_rate(),
        }
    }

    /// Count compilations of a specific tier
    fn count_compilations(&self, tier: CompilationTier) -> u64 {
        self.compilation_history.values()
            .flat_map(|h| &h.compilation_attempts)
            .filter(|a| a.target_tier == tier && a.succeeded)
            .count() as u64
    }

    /// Calculate cache hit rate for tier decisions
    fn calculate_cache_hit_rate(&self) -> f64 {
        if self.tier_decisions.is_empty() {
            0.0
        } else {
            let valid_decisions = self.tier_decisions.values()
                .filter(|d| d.timestamp.elapsed() < self.config.decision_cache_duration)
                .count();
            valid_decisions as f64 / self.tier_decisions.len() as f64
        }
    }
}

/// JIT profiling statistics
#[derive(Debug, Clone)]
pub struct JitProfilingStats {
    /// Number of functions being profiled
    pub functions_profiled: usize,
    /// Total function executions recorded
    pub total_executions: u64,
    /// Successful baseline compilations
    pub baseline_compilations: u64,
    /// Successful optimizing compilations
    pub optimizing_compilations: u64,
    /// Tier decision cache hit rate
    pub cache_hit_rate: f64,
}

// Re-export for backward compatibility with existing JIT modules
pub use JitProfilerIntegration as AdaptiveProfiler;
pub use JitProfileData as ProfileData;
pub use JitProfilingConfig as ProfilerConfig;

/// Profiling event (simplified interface)
#[derive(Debug, Clone)]
pub enum ProfilingEvent {
    FunctionExecuted { function_id: u32, duration: Duration },
    CompilationStarted { function_id: u32, tier: CompilationTier },
    CompilationCompleted { function_id: u32, tier: CompilationTier, success: bool },
}

/// Hotspot analysis (delegated to runtime profiler)
#[derive(Debug, Clone)]
pub struct HotspotAnalysis {
    /// Hot functions identified by runtime profiler
    pub hot_functions: Vec<u32>,
    /// Performance bottlenecks
    pub bottlenecks: Vec<String>,
    /// Optimization opportunities
    pub opportunities: Vec<OptimizationHint>,
} 