//! Write Barriers Subsystem for Prism VM Garbage Collector
//! 
//! This module provides a comprehensive write barrier implementation designed for
//! high-performance concurrent garbage collection. The design is inspired by:
//! - Go's hybrid write barrier for concurrent marking
//! - JVM G1/ZGC's region-based barriers and card marking
//! - Rust's memory safety and zero-cost abstractions
//!
//! ## Architecture
//!
//! The barriers subsystem is organized into focused modules:
//! - `types`: Core types, enums, and data structures
//! - `implementations`: Specific barrier algorithms (Dijkstra, Yuasa, Hybrid)
//! - `performance`: Optimizations like thread-local buffering and SIMD
//! - `integration`: Clean interfaces with allocator, collectors, and heap
//! - `safety`: Memory ordering, race condition prevention, and validation

pub mod types;
pub mod implementations;
pub mod performance;
pub mod integration;
pub mod safety;

// Re-export main types for convenience
pub use types::{
    WriteBarrierType, ObjectColor, BarrierConfig, BarrierStats,
    ColorTransition, BarrierEvent, ThreadLocalState
};

pub use implementations::{
    WriteBarrier, DijkstraBarrier, YuasaBarrier, HybridBarrier,
    BarrierImplementation
};

pub use performance::{
    ThreadLocalBuffer, SIMDBarrierOps, CardTable, RememberedSet,
    PerformanceHints
};

pub use integration::{
    BarrierManager, AllocatorIntegration, CollectorIntegration,
    HeapIntegration, BarrierHooks
};

pub use safety::{
    MemoryOrdering, RaceDetection, BarrierValidator,
    SafetyChecks, ConcurrencyGuards
};

use std::sync::Arc;
use super::{ObjectHeader, HeapStats, GcConfig};

/// Main entry point for the write barriers subsystem
/// 
/// This provides a unified interface for all barrier operations while
/// maintaining clean separation between different concerns.
pub struct BarrierSubsystem {
    /// The active barrier implementation
    barrier: Arc<dyn BarrierImplementation>,
    /// Performance optimization layer
    performance: Arc<performance::PerformanceLayer>,
    /// Integration layer with other GC components
    integration: Arc<integration::IntegrationLayer>,
    /// Safety validation and monitoring
    safety: Arc<safety::SafetyLayer>,
    /// Current configuration
    config: BarrierConfig,
}

impl BarrierSubsystem {
    /// Create a new barriers subsystem with the given configuration
    pub fn new(config: BarrierConfig) -> Self {
        let barrier = Self::create_barrier_implementation(&config);
        let performance = Arc::new(performance::PerformanceLayer::new(&config));
        let integration = Arc::new(integration::IntegrationLayer::new(&config));
        let safety = Arc::new(safety::SafetyLayer::new(&config));

        Self {
            barrier,
            performance,
            integration,
            safety,
            config,
        }
    }

    /// Create the appropriate barrier implementation based on configuration
    fn create_barrier_implementation(config: &BarrierConfig) -> Arc<dyn BarrierImplementation> {
        match config.barrier_type {
            WriteBarrierType::None => {
                Arc::new(implementations::NoBarrier::new())
            }
            WriteBarrierType::Incremental => {
                Arc::new(implementations::DijkstraBarrier::new(config.clone()))
            }
            WriteBarrierType::Snapshot => {
                Arc::new(implementations::YuasaBarrier::new(config.clone()))
            }
            WriteBarrierType::Hybrid => {
                Arc::new(implementations::HybridBarrier::new(config.clone()))
            }
        }
    }

    /// Execute a write barrier for a pointer update
    /// 
    /// This is the main entry point for all write barrier operations.
    /// It coordinates between the implementation, performance optimizations,
    /// and safety checks.
    pub fn write_barrier(
        &self,
        slot: *mut *const u8,
        new_value: *const u8,
        old_value: *const u8,
    ) {
        // Safety checks first
        if self.config.enable_safety_checks {
            self.safety.validate_barrier_call(slot, new_value, old_value);
        }

        // Performance layer handles buffering and optimizations
        self.performance.process_barrier_call(
            slot,
            new_value,
            old_value,
            &*self.barrier,
        );

        // Integration layer handles coordination with other GC components
        self.integration.notify_barrier_executed(slot, new_value, old_value);
    }

    /// Enable marking phase - barriers become active
    pub fn enable_marking(&self) {
        self.barrier.enable_marking();
        self.performance.enable_marking_optimizations();
        self.integration.notify_marking_started();
    }

    /// Disable marking phase - barriers become inactive
    pub fn disable_marking(&self) {
        self.barrier.disable_marking();
        self.performance.disable_marking_optimizations();
        self.integration.notify_marking_stopped();
    }

    /// Get current barrier statistics
    pub fn get_stats(&self) -> BarrierStats {
        let impl_stats = self.barrier.get_stats();
        let perf_stats = self.performance.get_stats();
        let integration_stats = self.integration.get_stats();
        let safety_stats = self.safety.get_stats();

        BarrierStats::combine(vec![impl_stats, perf_stats, integration_stats, safety_stats])
    }

    /// Reconfigure the barriers subsystem
    pub fn reconfigure(&mut self, new_config: BarrierConfig) {
        // Create new barrier implementation if type changed
        if new_config.barrier_type != self.config.barrier_type {
            self.barrier = Self::create_barrier_implementation(&new_config);
        }

        // Update all layers with new configuration
        self.performance.reconfigure(&new_config);
        self.integration.reconfigure(&new_config);
        self.safety.reconfigure(&new_config);

        self.config = new_config;
    }

    /// Prepare for garbage collection
    /// 
    /// This flushes all buffers and prepares the barriers for collection.
    pub fn prepare_for_gc(&self) {
        self.performance.flush_all_buffers();
        self.integration.prepare_for_collection();
        self.safety.validate_pre_collection_state();
    }

    /// Clean up after garbage collection
    pub fn post_collection_cleanup(&self) {
        self.performance.reset_after_collection();
        self.integration.post_collection_cleanup();
        self.safety.validate_post_collection_state();
    }

    /// Get the underlying barrier implementation for advanced usage
    pub fn get_implementation(&self) -> &dyn BarrierImplementation {
        &*self.barrier
    }
}

/// Factory for creating optimized barrier subsystems
pub struct BarrierFactory;

impl BarrierFactory {
    /// Create a barrier subsystem optimized for Prism VM's workload
    pub fn create_prism_optimized() -> BarrierSubsystem {
        let config = BarrierConfig {
            barrier_type: WriteBarrierType::Hybrid,
            enable_thread_local_buffering: true,
            enable_simd_optimizations: true,
            enable_card_marking: true,
            enable_safety_checks: cfg!(debug_assertions),
            buffer_size: 256,
            flush_threshold: 192, // 75% of buffer
            max_pause_contribution: std::time::Duration::from_micros(100),
            ..Default::default()
        };

        BarrierSubsystem::new(config)
    }

    /// Create a barrier subsystem optimized for low latency
    pub fn create_low_latency() -> BarrierSubsystem {
        let config = BarrierConfig {
            barrier_type: WriteBarrierType::Hybrid,
            enable_thread_local_buffering: true,
            enable_simd_optimizations: true,
            enable_card_marking: false, // Reduces pause time
            enable_safety_checks: false,
            buffer_size: 128, // Smaller buffers for lower latency
            flush_threshold: 96,
            max_pause_contribution: std::time::Duration::from_micros(50),
            ..Default::default()
        };

        BarrierSubsystem::new(config)
    }

    /// Create a barrier subsystem optimized for high throughput
    pub fn create_high_throughput() -> BarrierSubsystem {
        let config = BarrierConfig {
            barrier_type: WriteBarrierType::Hybrid,
            enable_thread_local_buffering: true,
            enable_simd_optimizations: true,
            enable_card_marking: true,
            enable_safety_checks: false,
            buffer_size: 512, // Larger buffers for better throughput
            flush_threshold: 384,
            max_pause_contribution: std::time::Duration::from_millis(1),
            ..Default::default()
        };

        BarrierSubsystem::new(config)
    }

    /// Create a debug barrier subsystem with extensive validation
    pub fn create_debug() -> BarrierSubsystem {
        let config = BarrierConfig {
            barrier_type: WriteBarrierType::Hybrid,
            enable_thread_local_buffering: false, // Simpler for debugging
            enable_simd_optimizations: false,
            enable_card_marking: true,
            enable_safety_checks: true,
            enable_race_detection: true,
            enable_barrier_validation: true,
            buffer_size: 64,
            flush_threshold: 48,
            max_pause_contribution: std::time::Duration::from_millis(10),
            ..Default::default()
        };

        BarrierSubsystem::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_subsystem_creation() {
        let subsystem = BarrierFactory::create_prism_optimized();
        let stats = subsystem.get_stats();
        assert_eq!(stats.barrier_calls, 0);
    }

    #[test]
    fn test_barrier_enable_disable() {
        let subsystem = BarrierFactory::create_prism_optimized();
        
        subsystem.enable_marking();
        // Would test that barriers are now active
        
        subsystem.disable_marking();
        // Would test that barriers are now inactive
    }

    #[test]
    fn test_barrier_reconfiguration() {
        let mut subsystem = BarrierFactory::create_prism_optimized();
        
        let new_config = BarrierConfig {
            barrier_type: WriteBarrierType::Incremental,
            ..Default::default()
        };
        
        subsystem.reconfigure(new_config);
        // Would verify that the barrier type changed
    }
} 