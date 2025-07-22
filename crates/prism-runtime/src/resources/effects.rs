//! Effect Tracking and Resource Coordination
//!
//! This module provides effect tracking capabilities for the resource management system,
//! allowing correlation between computational effects and resource consumption patterns.
//! This is a self-contained implementation that doesn't rely on external prism crates.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use uuid::Uuid;

/// Unique identifier for effect tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(Uuid);

impl EffectId {
    /// Generate a new effect ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Basic effect representation for resource tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effect {
    /// I/O operations (file, network, etc.)
    IO { operation: String, size: Option<usize> },
    /// Memory allocation/deallocation
    Memory { operation: String, size: usize },
    /// Computation effects
    Computation { operation: String, complexity: Option<String> },
    /// External system calls
    SystemCall { call: String, parameters: Vec<String> },
    /// Custom effect type
    Custom { name: String, metadata: HashMap<String, String> },
}

impl Effect {
    /// Get the name of this effect for identification
    pub fn name(&self) -> &str {
        match self {
            Effect::IO { operation, .. } => operation,
            Effect::Memory { operation, .. } => operation,
            Effect::Computation { operation, .. } => operation,
            Effect::SystemCall { call, .. } => call,
            Effect::Custom { name, .. } => name,
        }
    }
}

/// Resource measurement for effect tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMeasurement {
    /// CPU time used (nanoseconds)
    pub cpu_time_ns: u64,
    /// Memory allocated (bytes)
    pub memory_bytes: u64,
    /// Network bytes transferred
    pub network_bytes: u64,
    /// Disk bytes read/written
    pub disk_bytes: u64,
    /// System calls made
    pub system_calls: u64,
    /// Custom measurements
    pub custom: HashMap<String, f64>,
}

impl Default for ResourceMeasurement {
    fn default() -> Self {
        Self {
            cpu_time_ns: 0,
            memory_bytes: 0,
            network_bytes: 0,
            disk_bytes: 0,
            system_calls: 0,
            custom: HashMap::new(),
        }
    }
}

impl ResourceMeasurement {
    /// Add another measurement to this one
    pub fn add(&mut self, other: &ResourceMeasurement) {
        self.cpu_time_ns += other.cpu_time_ns;
        self.memory_bytes += other.memory_bytes;
        self.network_bytes += other.network_bytes;
        self.disk_bytes += other.disk_bytes;
        self.system_calls += other.system_calls;
        
        for (key, value) in &other.custom {
            *self.custom.entry(key.clone()).or_insert(0.0) += value;
        }
    }
    
    /// Get total resource "cost" as a simple metric
    pub fn total_cost(&self) -> f64 {
        // Simple weighted sum for demonstration
        (self.cpu_time_ns as f64 / 1_000_000.0) + // CPU in milliseconds
        (self.memory_bytes as f64 / 1_000_000.0) + // Memory in MB
        (self.network_bytes as f64 / 1_000_000.0) + // Network in MB
        (self.disk_bytes as f64 / 1_000_000.0) + // Disk in MB
        (self.system_calls as f64 * 0.1) // System calls with small weight
    }
}

/// An active effect allocation being tracked
#[derive(Debug, Clone)]
pub struct EffectAllocation {
    /// Unique allocation ID
    pub id: EffectId,
    /// The effect being tracked
    pub effect: Effect,
    /// When tracking started
    pub started_at: Instant,
    /// Expected duration (if known)
    pub expected_duration: Option<Duration>,
    /// Resources measured so far
    pub resources_used: ResourceMeasurement,
    /// Metadata for business correlation
    pub metadata: HashMap<String, String>,
}

/// Errors that can occur in effect tracking
#[derive(Debug, Clone, Error)]
pub enum EffectError {
    /// Effect not found
    #[error("Effect not found: {effect_id:?}")]
    EffectNotFound { effect_id: EffectId },
    
    /// Resource measurement failed
    #[error("Resource measurement failed: {message}")]
    MeasurementFailed { message: String },
    
    /// Invalid effect configuration
    #[error("Invalid effect configuration: {message}")]
    InvalidConfig { message: String },
    
    /// Generic effect error
    #[error("Effect error: {message}")]
    Generic { message: String },
}

/// Main effect tracking system
#[derive(Debug)]
pub struct EffectTracker {
    /// Active effect allocations
    allocations: RwLock<HashMap<EffectId, EffectAllocation>>,
    /// Historical effect data
    history: RwLock<Vec<CompletedEffect>>,
    /// Configuration
    config: EffectTrackerConfig,
}

/// Configuration for effect tracking
#[derive(Debug, Clone)]
pub struct EffectTrackerConfig {
    /// Maximum number of historical effects to keep
    pub max_history: usize,
    /// Whether to collect detailed resource measurements
    pub detailed_measurements: bool,
    /// Sampling interval for resource measurements
    pub measurement_interval: Duration,
}

impl Default for EffectTrackerConfig {
    fn default() -> Self {
        Self {
            max_history: 10000,
            detailed_measurements: true,
            measurement_interval: Duration::from_millis(100),
        }
    }
}

/// A completed effect with full resource measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedEffect {
    /// Effect ID
    pub id: EffectId,
    /// The effect that was executed
    pub effect: Effect,
    /// When it started
    pub started_at: SystemTime,
    /// Duration of execution
    pub duration: Duration,
    /// Total resources consumed
    pub resources_used: ResourceMeasurement,
    /// Business metadata
    pub metadata: HashMap<String, String>,
}

impl EffectTracker {
    /// Create a new effect tracker
    pub fn new() -> Result<Self, EffectError> {
        Self::with_config(EffectTrackerConfig::default())
    }
    
    /// Create effect tracker with custom configuration
    pub fn with_config(config: EffectTrackerConfig) -> Result<Self, EffectError> {
        Ok(Self {
            allocations: RwLock::new(HashMap::new()),
            history: RwLock::new(Vec::new()),
            config,
        })
    }
    
    /// Begin tracking an effect
    pub fn begin_effect(&self, effect: Effect, metadata: Option<HashMap<String, String>>) -> Result<EffectId, EffectError> {
        let effect_id = EffectId::new();
        
        let allocation = EffectAllocation {
            id: effect_id,
            effect,
            started_at: Instant::now(),
            expected_duration: None,
            resources_used: ResourceMeasurement::default(),
            metadata: metadata.unwrap_or_default(),
        };
        
        self.allocations.write().unwrap().insert(effect_id, allocation);
        
        Ok(effect_id)
    }
    
    /// End tracking an effect
    pub fn end_effect(&self, effect_id: EffectId) -> Result<CompletedEffect, EffectError> {
        let allocation = {
            let mut allocations = self.allocations.write().unwrap();
            allocations.remove(&effect_id)
                .ok_or(EffectError::EffectNotFound { effect_id })?
        };
        
        let duration = allocation.started_at.elapsed();
        
        // Create completed effect
        let completed = CompletedEffect {
            id: effect_id,
            effect: allocation.effect,
            started_at: SystemTime::now() - duration, // Approximate
            duration,
            resources_used: allocation.resources_used,
            metadata: allocation.metadata,
        };
        
        // Add to history
        {
            let mut history = self.history.write().unwrap();
            history.push(completed.clone());
            
            // Trim history if needed
            if history.len() > self.config.max_history {
                history.remove(0);
            }
        }
        
        Ok(completed)
    }
    
    /// Update resource measurements for an active effect
    pub fn update_effect_resources(&self, effect_id: EffectId, additional_resources: ResourceMeasurement) -> Result<(), EffectError> {
        let mut allocations = self.allocations.write().unwrap();
        
        if let Some(allocation) = allocations.get_mut(&effect_id) {
            allocation.resources_used.add(&additional_resources);
            Ok(())
        } else {
            Err(EffectError::EffectNotFound { effect_id })
        }
    }
    
    /// Get current active effects
    pub fn active_effects(&self) -> Vec<EffectAllocation> {
        self.allocations.read().unwrap().values().cloned().collect()
    }
    
    /// Get effect history
    pub fn effect_history(&self) -> Vec<CompletedEffect> {
        self.history.read().unwrap().clone()
    }
    
    /// Get statistics about effect tracking
    pub fn statistics(&self) -> EffectStatistics {
        let allocations = self.allocations.read().unwrap();
        let history = self.history.read().unwrap();
        
        let active_count = allocations.len();
        let total_completed = history.len();
        
        let total_resources = history.iter().fold(ResourceMeasurement::default(), |mut acc, effect| {
            acc.add(&effect.resources_used);
            acc
        });
        
        let avg_duration = if !history.is_empty() {
            history.iter().map(|e| e.duration.as_nanos() as f64).sum::<f64>() / history.len() as f64
        } else {
            0.0
        };
        
        EffectStatistics {
            active_effects: active_count,
            completed_effects: total_completed,
            total_resources_used: total_resources,
            average_duration_ns: avg_duration,
        }
    }
    
    /// Measure current resource usage using simple system APIs
    fn measure_current_resources(&self) -> ResourceMeasurement {
        use std::process;
        use std::time::Instant;
        
        // Get current memory usage from the process
        let memory_bytes = Self::get_memory_usage();
        
        // CPU time is harder to measure accurately in real-time, 
        // so we'll track it via timing intervals
        let cpu_time_ns = Self::get_cpu_time_ns();
        
        ResourceMeasurement {
            cpu_time_ns,
            memory_bytes,
            network_bytes: 0, // Would require OS-specific network monitoring
            disk_bytes: 0,    // Would require OS-specific disk I/O monitoring  
            system_calls: 0,  // Would require OS-specific syscall monitoring
            custom: HashMap::new(),
        }
    }
    
    /// Get memory usage for current process (cross-platform)
    fn get_memory_usage() -> u64 {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/self/status on Linux
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // Use ps command on macOS as a fallback
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output() 
            {
                if let Ok(rss_str) = String::from_utf8(output.stdout) {
                    if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                        return rss_kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        
        // Fallback: return 0 if we can't measure
        0
    }
    
    /// Get CPU time for current process (simplified)
    fn get_cpu_time_ns() -> u64 {
        // This is a simplified implementation
        // In a real system, you'd use platform-specific APIs
        
        #[cfg(unix)]
        {
            // On Unix systems, we could read from /proc/self/stat
            // For now, return a placeholder
        }
        
        // Fallback: return 0 if we can't measure
        0
    }
}

/// Statistics about effect tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectStatistics {
    /// Number of currently active effects
    pub active_effects: usize,
    /// Number of completed effects
    pub completed_effects: usize,
    /// Total resources used by all completed effects
    pub total_resources_used: ResourceMeasurement,
    /// Average duration of completed effects (nanoseconds)
    pub average_duration_ns: f64,
}

/// Convenience function to create a memory allocation effect
pub fn memory_allocation_effect(size: usize) -> Effect {
    Effect::Memory {
        operation: "allocate".to_string(),
        size,
    }
}

/// Convenience function to create a memory deallocation effect
pub fn memory_deallocation_effect(size: usize) -> Effect {
    Effect::Memory {
        operation: "deallocate".to_string(),
        size,
    }
}

/// Convenience function to create an I/O effect
pub fn io_effect(operation: &str, size: Option<usize>) -> Effect {
    Effect::IO {
        operation: operation.to_string(),
        size,
    }
}

/// Convenience function to create a computation effect
pub fn computation_effect(operation: &str, complexity: Option<&str>) -> Effect {
    Effect::Computation {
        operation: operation.to_string(),
        complexity: complexity.map(|s| s.to_string()),
    }
}

/// Result of an effect execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectResult {
    /// Effect ID
    pub effect_id: EffectId,
    /// Whether the effect succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Resources consumed during execution
    pub resources_consumed: ResourceMeasurement,
    /// Duration of execution
    pub duration: Duration,
}

/// Resource consumption summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConsumption {
    /// CPU time consumed
    pub cpu_time: Duration,
    /// Memory allocated
    pub memory_bytes: u64,
    /// Network bytes transferred
    pub network_bytes: u64,
    /// Disk I/O bytes
    pub disk_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_effect_tracker_creation() {
        let tracker = EffectTracker::new();
        assert!(tracker.is_ok());
    }
    
    #[test]
    fn test_begin_end_effect() {
        let tracker = EffectTracker::new().unwrap();
        
        let effect = memory_allocation_effect(1024);
        let effect_id = tracker.begin_effect(effect, None).unwrap();
        
        // Add some resource usage
        let resources = ResourceMeasurement {
            memory_bytes: 1024,
            cpu_time_ns: 1000000,
            ..Default::default()
        };
        tracker.update_effect_resources(effect_id, resources).unwrap();
        
        // End the effect
        let completed = tracker.end_effect(effect_id).unwrap();
        assert_eq!(completed.resources_used.memory_bytes, 1024);
        assert_eq!(completed.resources_used.cpu_time_ns, 1000000);
    }
    
    #[test]
    fn test_effect_statistics() {
        let tracker = EffectTracker::new().unwrap();
        
        let effect1 = memory_allocation_effect(512);
        let effect2 = io_effect("read", Some(1024));
        
        let id1 = tracker.begin_effect(effect1, None).unwrap();
        let id2 = tracker.begin_effect(effect2, None).unwrap();
        
        let stats = tracker.statistics();
        assert_eq!(stats.active_effects, 2);
        assert_eq!(stats.completed_effects, 0);
        
        tracker.end_effect(id1).unwrap();
        tracker.end_effect(id2).unwrap();
        
        let stats = tracker.statistics();
        assert_eq!(stats.active_effects, 0);
        assert_eq!(stats.completed_effects, 2);
    }
    
    #[test]
    fn test_resource_measurement_add() {
        let mut measurement1 = ResourceMeasurement {
            cpu_time_ns: 1000,
            memory_bytes: 2000,
            ..Default::default()
        };
        
        let measurement2 = ResourceMeasurement {
            cpu_time_ns: 500,
            memory_bytes: 1000,
            network_bytes: 100,
            ..Default::default()
        };
        
        measurement1.add(&measurement2);
        
        assert_eq!(measurement1.cpu_time_ns, 1500);
        assert_eq!(measurement1.memory_bytes, 3000);
        assert_eq!(measurement1.network_bytes, 100);
    }
    
    #[test]
    fn test_effect_convenience_functions() {
        let mem_effect = memory_allocation_effect(4096);
        match mem_effect {
            Effect::Memory { operation, size } => {
                assert_eq!(operation, "allocate");
                assert_eq!(size, 4096);
            }
            _ => panic!("Expected memory effect"),
        }
        
        let io_effect = io_effect("write", Some(1024));
        match io_effect {
            Effect::IO { operation, size } => {
                assert_eq!(operation, "write");
                assert_eq!(size, Some(1024));
            }
            _ => panic!("Expected I/O effect"),
        }
    }
} 