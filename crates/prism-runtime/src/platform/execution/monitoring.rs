//! Execution Monitoring and Metrics
//!
//! This module provides comprehensive monitoring and metrics collection
//! for code execution across different target platforms.

use super::context::{ExecutionContext, ExecutionId, MonitoringLevel};
use super::errors::{ExecutionError, ExecutionResult};
use crate::resources::{ResourceTracker, EffectTracker};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Execution monitoring system
#[derive(Debug)]
pub struct ExecutionMonitor {
    /// Active monitoring sessions
    active_monitors: Arc<RwLock<HashMap<MonitoringHandle, MonitoringSession>>>,
    /// Monitor configuration
    config: MonitorConfig,
    /// Metrics collector
    metrics_collector: Arc<ExecutionMetricsCollector>,
    /// Resource tracker for system-level metrics
    resource_tracker: Arc<ResourceTracker>,
    /// Effect tracker for business-level tracking
    effect_tracker: Arc<EffectTracker>,
}

impl ExecutionMonitor {
    pub fn new() -> ExecutionResult<Self> {
        let resource_tracker = Arc::new(ResourceTracker::new()
            .map_err(|e| ExecutionError::Generic { 
                message: format!("Failed to create resource tracker: {}", e) 
            })?);
        
        let effect_tracker = Arc::new(EffectTracker::new()
            .map_err(|e| ExecutionError::Generic { 
                message: format!("Failed to create effect tracker: {}", e) 
            })?);
        
        Ok(Self {
            active_monitors: Arc::new(RwLock::new(HashMap::new())),
            config: MonitorConfig::default(),
            metrics_collector: Arc::new(ExecutionMetricsCollector::new()),
            resource_tracker,
            effect_tracker,
        })
    }
    
    pub fn with_config(config: MonitorConfig) -> ExecutionResult<Self> {
        let resource_tracker = Arc::new(ResourceTracker::new()
            .map_err(|e| ExecutionError::Generic { 
                message: format!("Failed to create resource tracker: {}", e) 
            })?);
        
        let effect_tracker = Arc::new(EffectTracker::new()
            .map_err(|e| ExecutionError::Generic { 
                message: format!("Failed to create effect tracker: {}", e) 
            })?);
        
        Ok(Self {
            active_monitors: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics_collector: Arc::new(ExecutionMetricsCollector::new()),
            resource_tracker,
            effect_tracker,
        })
    }
    
    /// Start monitoring an execution
    pub fn start_monitoring(&self, context: &ExecutionContext) -> ExecutionResult<MonitoringHandle> {
        let handle = MonitoringHandle::new();
        let session = MonitoringSession::new(context.clone(), &self.config);
        
        {
            let mut monitors = self.active_monitors.write()
                .map_err(|_| ExecutionError::MonitoringFailed {
                    reason: "Failed to acquire write lock on monitors".to_string(),
                })?;
            monitors.insert(handle, session);
        }
        
        Ok(handle)
    }
    
    /// Stop monitoring and collect metrics
    pub fn stop_monitoring(&self, handle: MonitoringHandle) -> ExecutionResult<ExecutionMetrics> {
        let session = {
            let mut monitors = self.active_monitors.write()
                .map_err(|_| ExecutionError::MonitoringFailed {
                    reason: "Failed to acquire write lock on monitors".to_string(),
                })?;
            monitors.remove(&handle)
                .ok_or_else(|| ExecutionError::MonitoringFailed {
                    reason: "Monitor session not found".to_string(),
                })?
        };
        
        let context = session.context.clone();
        let metrics = session.finalize();
        
        // Record metrics for analytics
        self.metrics_collector.record_execution(&metrics, &context);
        
        Ok(metrics)
    }
    
    /// Get current monitoring statistics
    pub fn get_monitoring_stats(&self) -> MonitoringStats {
        let active_count = self.active_monitors.read()
            .map(|monitors| monitors.len())
            .unwrap_or(0);
            
        MonitoringStats {
            active_sessions: active_count,
            total_sessions: self.metrics_collector.total_executions(),
            average_execution_time: self.metrics_collector.average_execution_time(),
        }
    }
    
    /// Get current resource snapshot
    pub fn get_resource_snapshot(&self) -> crate::resources::ResourceSnapshot {
        self.resource_tracker.current_snapshot()
    }
    
    /// Get resource statistics for a resource type
    pub fn get_resource_stats(&self, resource_type: &crate::resources::ResourceType) -> Option<crate::resources::ResourceStats> {
        self.resource_tracker.get_resource_stats(resource_type)
    }
    
    /// Begin tracking an effect during execution
    pub fn begin_effect(&self, effect: crate::resources::Effect) -> Result<crate::resources::EffectId, ExecutionError> {
        self.effect_tracker.begin_effect(effect, None)
            .map_err(|e| ExecutionError::Generic { 
                message: format!("Failed to begin effect tracking: {}", e) 
            })
    }
    
    /// End tracking an effect
    pub fn end_effect(&self, effect_id: crate::resources::EffectId) -> Result<crate::resources::CompletedEffect, ExecutionError> {
        self.effect_tracker.end_effect(effect_id)
            .map_err(|e| ExecutionError::Generic { 
                message: format!("Failed to end effect tracking: {}", e) 
            })
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
    /// Monitoring configuration
    config: MonitorConfig,
    /// Collected events during execution
    events: Vec<ExecutionEvent>,
}

impl MonitoringSession {
    fn new(context: ExecutionContext, config: &MonitorConfig) -> Self {
        Self {
            context,
            start_time: Instant::now(),
            initial_resources: ResourceSnapshot::current(),
            config: config.clone(),
            events: Vec::new(),
        }
    }
    
    fn finalize(self) -> ExecutionMetrics {
        let duration = self.start_time.elapsed();
        let final_resources = ResourceSnapshot::current();
        
        ExecutionMetrics {
            execution_id: self.context.execution_id,
            target: self.context.target,
            duration,
            resource_usage: final_resources.diff(&self.initial_resources),
            success: true, // Would be set based on actual execution result
            events: self.events,
            monitoring_level: self.config.monitoring_level,
            timestamp: SystemTime::now(),
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
    /// Network bytes sent
    network_bytes_sent: u64,
    /// Network bytes received
    network_bytes_received: u64,
    /// Disk bytes read
    disk_bytes_read: u64,
    /// Disk bytes written
    disk_bytes_written: u64,
}

impl ResourceSnapshot {
    fn current() -> Self {
        Self {
            memory_bytes: crate::resources::effects::EffectTracker::get_memory_usage(),
            cpu_time_ns: crate::resources::effects::EffectTracker::get_cpu_time_ns(),
            network_bytes_sent: 0, // Would be implemented with actual network tracking
            network_bytes_received: 0,
            disk_bytes_read: 0, // Would be implemented with actual disk tracking
            disk_bytes_written: 0,
        }
    }
    
    fn diff(&self, other: &ResourceSnapshot) -> ResourceUsage {
        ResourceUsage {
            memory_delta: self.memory_bytes.saturating_sub(other.memory_bytes),
            cpu_time_delta: self.cpu_time_ns.saturating_sub(other.cpu_time_ns),
            network_bytes_sent: self.network_bytes_sent.saturating_sub(other.network_bytes_sent),
            network_bytes_received: self.network_bytes_received.saturating_sub(other.network_bytes_received),
            disk_bytes_read: self.disk_bytes_read.saturating_sub(other.disk_bytes_read),
            disk_bytes_written: self.disk_bytes_written.saturating_sub(other.disk_bytes_written),
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
    /// Network bytes sent
    pub network_bytes_sent: u64,
    /// Network bytes received
    pub network_bytes_received: u64,
    /// Disk bytes read
    pub disk_bytes_read: u64,
    /// Disk bytes written
    pub disk_bytes_written: u64,
}

/// Execution event for detailed monitoring
#[derive(Debug, Clone)]
pub struct ExecutionEvent {
    /// When the event occurred
    pub timestamp: Instant,
    /// Type of event
    pub event_type: ExecutionEventType,
    /// Event description
    pub description: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of execution events
#[derive(Debug, Clone)]
pub enum ExecutionEventType {
    /// Execution started
    ExecutionStarted,
    /// Execution completed
    ExecutionCompleted,
    /// Error occurred
    Error,
    /// Warning issued
    Warning,
    /// Capability checked
    CapabilityCheck,
    /// Resource allocated
    ResourceAllocation,
    /// Performance milestone
    PerformanceMilestone,
}

/// Execution metrics collected during monitoring
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Execution ID
    pub execution_id: ExecutionId,
    /// Target platform
    pub target: super::context::ExecutionTarget,
    /// Duration of execution
    pub duration: Duration,
    /// Resource usage during execution
    pub resource_usage: ResourceUsage,
    /// Whether execution was successful
    pub success: bool,
    /// Events that occurred during execution
    pub events: Vec<ExecutionEvent>,
    /// Level of monitoring used
    pub monitoring_level: MonitoringLevel,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

/// Metrics collector for execution data
#[derive(Debug)]
pub struct ExecutionMetricsCollector {
    /// Collected metrics
    metrics: Arc<RwLock<Vec<ExecutionRecord>>>,
    /// Aggregated statistics
    stats: Arc<RwLock<AggregatedStats>>,
}

impl ExecutionMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(AggregatedStats::default())),
        }
    }
    
    pub fn record_execution(&self, metrics: &ExecutionMetrics, context: &ExecutionContext) {
        let record = ExecutionRecord {
            execution_id: context.execution_id,
            target: context.target,
            metrics: metrics.clone(),
            timestamp: SystemTime::now(),
        };
        
        // Update metrics
        if let Ok(mut metrics_store) = self.metrics.write() {
            metrics_store.push(record);
            
            // Keep only last 1000 records to prevent unbounded growth
            if metrics_store.len() > 1000 {
                metrics_store.remove(0);
            }
        }
        
        // Update aggregated statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.update(metrics);
        }
    }
    
    pub fn total_executions(&self) -> usize {
        self.metrics.read()
            .map(|metrics| metrics.len())
            .unwrap_or(0)
    }
    
    pub fn average_execution_time(&self) -> Duration {
        self.stats.read()
            .map(|stats| stats.average_duration)
            .unwrap_or(Duration::ZERO)
    }
    
    pub fn get_recent_metrics(&self, count: usize) -> Vec<ExecutionRecord> {
        self.metrics.read()
            .map(|metrics| {
                let start = metrics.len().saturating_sub(count);
                metrics[start..].to_vec()
            })
            .unwrap_or_default()
    }
}

/// Record of execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Execution ID
    pub execution_id: ExecutionId,
    /// Target platform
    pub target: super::context::ExecutionTarget,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Aggregated statistics across all executions
#[derive(Debug, Clone, Default)]
pub struct AggregatedStats {
    /// Total number of executions
    pub total_executions: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Average execution duration
    pub average_duration: Duration,
    /// Total CPU time used
    pub total_cpu_time: Duration,
    /// Total memory allocated
    pub total_memory_allocated: u64,
}

impl AggregatedStats {
    fn update(&mut self, metrics: &ExecutionMetrics) {
        self.total_executions += 1;
        if metrics.success {
            self.successful_executions += 1;
        }
        
        // Update average duration using incremental calculation
        let old_avg_nanos = self.average_duration.as_nanos() as u64;
        let new_duration_nanos = metrics.duration.as_nanos() as u64;
        let new_avg_nanos = (old_avg_nanos * (self.total_executions - 1) + new_duration_nanos) / self.total_executions;
        self.average_duration = Duration::from_nanos(new_avg_nanos);
        
        self.total_cpu_time += Duration::from_nanos(metrics.resource_usage.cpu_time_delta);
        self.total_memory_allocated += metrics.resource_usage.memory_delta;
    }
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Level of monitoring detail
    pub monitoring_level: MonitoringLevel,
    /// Enable event collection
    pub enable_event_collection: bool,
    /// Maximum number of events to collect per execution
    pub max_events_per_execution: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enable_resource_monitoring: true,
            monitoring_interval: Duration::from_millis(100),
            monitoring_level: MonitoringLevel::Standard,
            enable_event_collection: true,
            max_events_per_execution: 100,
        }
    }
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitoringStats {
    /// Number of active monitoring sessions
    pub active_sessions: usize,
    /// Total number of sessions created
    pub total_sessions: usize,
    /// Average execution time across all sessions
    pub average_execution_time: Duration,
} 