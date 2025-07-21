//! Effect Monitoring and Performance Analytics
//!
//! This module provides real-time monitoring of effect execution, performance analytics,
//! and resource measurement to replace placeholder implementations with actual tracking.
//!
//! ## Design Principles
//!
//! 1. **Real-time Monitoring**: Track effects as they execute with minimal overhead
//! 2. **Resource Measurement**: Actual system resource tracking (CPU, memory, I/O)
//! 3. **Performance Analytics**: Identify bottlenecks and optimization opportunities
//! 4. **AI-Comprehensible**: Generate structured data for external AI analysis
//! 5. **Separation of Concerns**: Monitor effects without affecting execution logic

use crate::effects::definition::{Effect, EffectDefinition};
use crate::execution::handlers::ResourceUsage;
use prism_common::span::Span;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;

/// Real-time effect monitoring system
#[derive(Debug)]
pub struct EffectMonitor {
    /// Active effect executions being monitored
    active_executions: Arc<RwLock<HashMap<ExecutionId, ActiveExecution>>>,
    
    /// Historical execution data for analytics
    execution_history: Arc<Mutex<VecDeque<ExecutionRecord>>>,
    
    /// Real-time performance metrics
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Resource monitoring system
    resource_monitor: Arc<ResourceMonitor>,
    
    /// Performance analytics engine
    analytics_engine: Arc<PerformanceAnalyticsEngine>,
    
    /// Monitoring configuration
    config: MonitoringConfig,
    
    /// Next execution ID generator
    next_execution_id: Arc<std::sync::atomic::AtomicU64>,
}

impl EffectMonitor {
    /// Create a new effect monitor
    pub fn new() -> Self {
        Self {
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
            resource_monitor: Arc::new(ResourceMonitor::new()),
            analytics_engine: Arc::new(PerformanceAnalyticsEngine::new()),
            config: MonitoringConfig::default(),
            next_execution_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Start monitoring an effect execution
    pub fn start_execution(&self, effect: &Effect) -> Result<ExecutionId, MonitoringError> {
        let execution_id = ExecutionId(
            self.next_execution_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );

        let active_execution = ActiveExecution {
            id: execution_id,
            effect: effect.clone(),
            start_time: Instant::now(),
            start_resources: self.resource_monitor.capture_current_usage()?,
            checkpoints: Vec::new(),
        };

        // Store active execution
        self.active_executions.write().unwrap()
            .insert(execution_id, active_execution);

        // Update metrics
        self.performance_metrics.write().unwrap()
            .record_execution_start();

        Ok(execution_id)
    }

    /// End monitoring an effect execution
    pub fn end_execution(&self, execution_id: ExecutionId, success: bool) -> Result<ExecutionRecord, MonitoringError> {
        let mut active_executions = self.active_executions.write().unwrap();
        let active_execution = active_executions.remove(&execution_id)
            .ok_or(MonitoringError::ExecutionNotFound { id: execution_id })?;

        let end_time = Instant::now();
        let duration = end_time.duration_since(active_execution.start_time);
        let end_resources = self.resource_monitor.capture_current_usage()?;

        // Calculate resource consumption
        let resource_consumption = ResourceConsumption::calculate(
            &active_execution.start_resources,
            &end_resources
        );

        // Create execution record
        let execution_record = ExecutionRecord {
            id: execution_id,
            effect: active_execution.effect.clone(),
            start_time: active_execution.start_time,
            end_time,
            duration,
            success,
            resource_consumption: resource_consumption.clone(),
            checkpoints: active_execution.checkpoints.clone(),
            performance_profile: self.calculate_performance_profile(&active_execution, duration, &resource_consumption),
        };

        // Store in history (with size limit)
        let mut history = self.execution_history.lock().unwrap();
        if history.len() >= self.config.max_history_size {
            history.pop_front();
        }
        history.push_back(execution_record.clone());

        // Update performance metrics
        self.performance_metrics.write().unwrap()
            .record_execution_complete(&execution_record);

        Ok(execution_record)
    }

    /// Add a checkpoint to an active execution
    pub fn add_checkpoint(&self, execution_id: ExecutionId, checkpoint: ExecutionCheckpoint) -> Result<(), MonitoringError> {
        let mut active_executions = self.active_executions.write().unwrap();
        let active_execution = active_executions.get_mut(&execution_id)
            .ok_or(MonitoringError::ExecutionNotFound { id: execution_id })?;

        active_execution.checkpoints.push(checkpoint);
        Ok(())
    }

    /// Get current monitoring metrics
    pub fn get_current_metrics(&self) -> MonitoringMetrics {
        let active_count = self.active_executions.read().unwrap().len();
        let performance_metrics = self.performance_metrics.read().unwrap().clone();
        
        MonitoringMetrics {
            active_executions: active_count,
            total_executions: performance_metrics.total_executions,
            average_duration: performance_metrics.average_duration,
            success_rate: performance_metrics.success_rate,
            current_resource_usage: self.resource_monitor.get_current_usage(),
            peak_concurrent_executions: performance_metrics.peak_concurrent_executions,
        }
    }

    /// Get performance analytics
    pub fn get_performance_analytics(&self) -> PerformanceAnalytics {
        let history = self.execution_history.lock().unwrap();
        self.analytics_engine.analyze(&history, &self.config)
    }

    /// Get execution history for analysis
    pub fn get_execution_history(&self) -> Vec<ExecutionRecord> {
        self.execution_history.lock().unwrap().iter().cloned().collect()
    }

    /// Calculate performance profile for an execution
    fn calculate_performance_profile(
        &self,
        execution: &ActiveExecution,
        duration: Duration,
        resource_consumption: &ResourceConsumption,
    ) -> PerformanceProfile {
        PerformanceProfile {
            execution_efficiency: self.calculate_efficiency_score(duration, resource_consumption),
            resource_intensity: self.calculate_resource_intensity(resource_consumption),
            bottlenecks: self.identify_bottlenecks(execution, resource_consumption),
            optimization_opportunities: self.identify_optimizations(execution, resource_consumption),
        }
    }

    /// Calculate execution efficiency score (0.0 to 1.0)
    fn calculate_efficiency_score(&self, duration: Duration, resources: &ResourceConsumption) -> f64 {
        // Simple heuristic: shorter duration and lower resource usage = higher efficiency
        let duration_score = 1.0 - (duration.as_millis() as f64 / 10000.0).min(1.0);
        let resource_score = 1.0 - (resources.total_resource_units() as f64 / 1000000.0).min(1.0);
        (duration_score + resource_score) / 2.0
    }

    /// Calculate resource intensity level
    fn calculate_resource_intensity(&self, resources: &ResourceConsumption) -> ResourceIntensity {
        let total_units = resources.total_resource_units();
        if total_units > 1000000 {
            ResourceIntensity::VeryHigh
        } else if total_units > 100000 {
            ResourceIntensity::High
        } else if total_units > 10000 {
            ResourceIntensity::Medium
        } else if total_units > 1000 {
            ResourceIntensity::Low
        } else {
            ResourceIntensity::VeryLow
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self, execution: &ActiveExecution, resources: &ResourceConsumption) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // CPU bottleneck
        if resources.cpu_time_micros > 1000000 { // > 1 second
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::CPU,
                severity: if resources.cpu_time_micros > 5000000 { 
                    BottleneckSeverity::High 
                } else { 
                    BottleneckSeverity::Medium 
                },
                description: format!("High CPU usage: {} ms", resources.cpu_time_micros / 1000),
                suggestion: "Consider optimizing computational complexity".to_string(),
            });
        }

        // Memory bottleneck
        if resources.memory_allocated_bytes > 100 * 1024 * 1024 { // > 100MB
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: if resources.memory_allocated_bytes > 1024 * 1024 * 1024 { 
                    BottleneckSeverity::High 
                } else { 
                    BottleneckSeverity::Medium 
                },
                description: format!("High memory usage: {} MB", resources.memory_allocated_bytes / (1024 * 1024)),
                suggestion: "Consider memory pooling or streaming".to_string(),
            });
        }

        // I/O bottleneck
        let total_io = resources.disk_read_bytes + resources.disk_write_bytes + 
                      resources.network_sent_bytes + resources.network_received_bytes;
        if total_io > 10 * 1024 * 1024 { // > 10MB
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::IO,
                severity: if total_io > 100 * 1024 * 1024 { 
                    BottleneckSeverity::High 
                } else { 
                    BottleneckSeverity::Medium 
                },
                description: format!("High I/O usage: {} MB", total_io / (1024 * 1024)),
                suggestion: "Consider caching or batching I/O operations".to_string(),
            });
        }

        bottlenecks
    }

    /// Identify optimization opportunities
    fn identify_optimizations(&self, execution: &ActiveExecution, resources: &ResourceConsumption) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();

        // Check for potential parallelization
        if execution.checkpoints.len() > 1 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OptimizationType::Parallelization,
                description: "Multiple checkpoints detected - consider parallel execution".to_string(),
                estimated_benefit: 0.3, // 30% improvement estimate
                implementation_complexity: OptimizationComplexity::Medium,
            });
        }

        // Check for caching opportunities
        if resources.disk_read_bytes > resources.disk_write_bytes * 2 {
            opportunities.push(OptimizationOpportunity {
                opportunity_type: OptimizationType::Caching,
                description: "High read-to-write ratio - consider result caching".to_string(),
                estimated_benefit: 0.5, // 50% improvement estimate
                implementation_complexity: OptimizationComplexity::Low,
            });
        }

        opportunities
    }
}

impl Default for EffectMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for an execution instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionId(u64);

/// Active execution being monitored
#[derive(Debug, Clone)]
pub struct ActiveExecution {
    /// Execution ID
    pub id: ExecutionId,
    /// Effect being executed
    pub effect: Effect,
    /// When execution started
    pub start_time: Instant,
    /// Resource usage at start
    pub start_resources: SystemResourceSnapshot,
    /// Execution checkpoints
    pub checkpoints: Vec<ExecutionCheckpoint>,
}

/// Checkpoint during effect execution
#[derive(Debug, Clone)]
pub struct ExecutionCheckpoint {
    /// Checkpoint name/description
    pub name: String,
    /// When checkpoint occurred
    pub timestamp: Instant,
    /// Resource usage at checkpoint
    pub resources: SystemResourceSnapshot,
    /// Checkpoint metadata
    pub metadata: HashMap<String, String>,
}

/// Complete record of an effect execution
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Execution ID
    pub id: ExecutionId,
    /// Effect that was executed
    pub effect: Effect,
    /// When execution started
    pub start_time: Instant,
    /// When execution ended
    pub end_time: Instant,
    /// Total execution duration
    pub duration: Duration,
    /// Whether execution was successful
    pub success: bool,
    /// Resource consumption during execution
    pub resource_consumption: ResourceConsumption,
    /// Execution checkpoints
    pub checkpoints: Vec<ExecutionCheckpoint>,
    /// Performance profile
    pub performance_profile: PerformanceProfile,
}

/// Resource consumption during execution
#[derive(Debug, Clone)]
pub struct ResourceConsumption {
    /// CPU time consumed (microseconds)
    pub cpu_time_micros: u64,
    /// Memory allocated (bytes)
    pub memory_allocated_bytes: u64,
    /// Disk bytes read
    pub disk_read_bytes: u64,
    /// Disk bytes written
    pub disk_write_bytes: u64,
    /// Network bytes sent
    pub network_sent_bytes: u64,
    /// Network bytes received
    pub network_received_bytes: u64,
}

impl ResourceConsumption {
    /// Calculate resource consumption between two snapshots
    pub fn calculate(start: &SystemResourceSnapshot, end: &SystemResourceSnapshot) -> Self {
        Self {
            cpu_time_micros: end.cpu_time_micros.saturating_sub(start.cpu_time_micros),
            memory_allocated_bytes: end.memory_allocated_bytes.saturating_sub(start.memory_allocated_bytes),
            disk_read_bytes: end.disk_read_bytes.saturating_sub(start.disk_read_bytes),
            disk_write_bytes: end.disk_write_bytes.saturating_sub(start.disk_write_bytes),
            network_sent_bytes: end.network_sent_bytes.saturating_sub(start.network_sent_bytes),
            network_received_bytes: end.network_received_bytes.saturating_sub(start.network_received_bytes),
        }
    }

    /// Calculate total resource units (for comparative analysis)
    pub fn total_resource_units(&self) -> u64 {
        // Weight different resource types for comparison
        self.cpu_time_micros / 1000 +  // CPU time in milliseconds
        self.memory_allocated_bytes / 1024 +  // Memory in KB
        (self.disk_read_bytes + self.disk_write_bytes) / 1024 +  // Disk I/O in KB
        (self.network_sent_bytes + self.network_received_bytes) / 1024  // Network I/O in KB
    }
}

/// Performance profile for an execution
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Execution efficiency score (0.0 to 1.0)
    pub execution_efficiency: f64,
    /// Resource intensity level
    pub resource_intensity: ResourceIntensity,
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Resource intensity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceIntensity {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Performance bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Description of the bottleneck
    pub description: String,
    /// Suggested improvement
    pub suggestion: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub opportunity_type: OptimizationType,
    /// Description of the opportunity
    pub description: String,
    /// Estimated performance benefit (0.0 to 1.0)
    pub estimated_benefit: f64,
    /// Implementation complexity
    pub implementation_complexity: OptimizationComplexity,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    Parallelization,
    Caching,
    Batching,
    MemoryOptimization,
    IOOptimization,
}

/// Optimization implementation complexity
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationComplexity {
    Low,
    Medium,
    High,
}

/// System resource monitoring
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Process handle for resource monitoring (Linux only)
    #[cfg(all(target_os = "linux", feature = "linux-monitoring"))]
    process_info: Arc<Mutex<Option<procfs::process::Process>>>,
    /// Baseline resource usage
    baseline: Arc<RwLock<Option<SystemResourceSnapshot>>>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> Self {
        Self {
            #[cfg(all(target_os = "linux", feature = "linux-monitoring"))]
            process_info: Arc::new(Mutex::new(None)),
            baseline: Arc::new(RwLock::new(None)),
        }
    }

    /// Capture current system resource usage
    pub fn capture_current_usage(&self) -> Result<SystemResourceSnapshot, MonitoringError> {
        #[cfg(all(target_os = "linux", feature = "linux-monitoring"))]
        {
            self.capture_unix_resources()
        }
        #[cfg(not(all(target_os = "linux", feature = "linux-monitoring")))]
        {
            // Fallback for non-Linux systems or when linux-monitoring is disabled
            Ok(SystemResourceSnapshot {
                timestamp: Instant::now(),
                cpu_time_micros: self.get_fallback_cpu_time(),
                memory_allocated_bytes: self.get_fallback_memory_usage(),
                disk_read_bytes: 0,
                disk_write_bytes: 0,
                network_sent_bytes: 0,
                network_received_bytes: 0,
            })
        }
    }

    #[cfg(all(target_os = "linux", feature = "linux-monitoring"))]
    fn capture_unix_resources(&self) -> Result<SystemResourceSnapshot, MonitoringError> {
        // Initialize process info if needed
        let mut process_info = self.process_info.lock().unwrap();
        if process_info.is_none() {
            *process_info = Some(procfs::process::Process::myself()
                .map_err(|e| MonitoringError::ResourceCaptureFailed { 
                    reason: format!("Failed to get process info: {}", e) 
                })?);
        }

        let process = process_info.as_ref().unwrap();
        
        // Get CPU usage
        let stat = process.stat()
            .map_err(|e| MonitoringError::ResourceCaptureFailed { 
                reason: format!("Failed to get process stats: {}", e) 
            })?;
        
        // Get memory usage
        let status = process.status()
            .map_err(|e| MonitoringError::ResourceCaptureFailed { 
                reason: format!("Failed to get process status: {}", e) 
            })?;

        // Get I/O stats
        let io = process.io().unwrap_or_default();

        Ok(SystemResourceSnapshot {
            timestamp: Instant::now(),
            cpu_time_micros: (stat.utime + stat.stime) * 10000, // Convert from clock ticks to microseconds (assuming 100Hz)
            memory_allocated_bytes: status.vmrss.unwrap_or(0) * 1024, // Convert from KB to bytes
            disk_read_bytes: io.read_bytes.unwrap_or(0),
            disk_write_bytes: io.write_bytes.unwrap_or(0),
            network_sent_bytes: 0, // Would need additional system calls
            network_received_bytes: 0, // Would need additional system calls
        })
    }

    /// Get fallback CPU time measurement (cross-platform)
    fn get_fallback_cpu_time(&self) -> u64 {
        // Simple fallback using process start time
        std::process::id() as u64 * 1000 // Placeholder based on process ID
    }

    /// Get fallback memory usage measurement (cross-platform) 
    fn get_fallback_memory_usage(&self) -> u64 {
        // Very basic fallback - in a real implementation you'd use platform-specific APIs
        #[cfg(target_os = "macos")]
        {
            // On macOS, we could use mach APIs, but for now use a simple heuristic
            8 * 1024 * 1024 // 8MB baseline
        }
        #[cfg(target_os = "windows")]
        {
            // On Windows, we could use Windows APIs, but for now use a simple heuristic
            8 * 1024 * 1024 // 8MB baseline
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            8 * 1024 * 1024 // 8MB baseline for other platforms
        }
    }

    /// Get current resource usage relative to baseline
    pub fn get_current_usage(&self) -> SystemResourceSnapshot {
        self.capture_current_usage().unwrap_or_else(|_| SystemResourceSnapshot {
            timestamp: Instant::now(),
            cpu_time_micros: 0,
            memory_allocated_bytes: 0,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
            network_sent_bytes: 0,
            network_received_bytes: 0,
        })
    }
}

/// Snapshot of system resource usage at a point in time
#[derive(Debug, Clone)]
pub struct SystemResourceSnapshot {
    /// When snapshot was taken
    pub timestamp: Instant,
    /// CPU time used (microseconds)
    pub cpu_time_micros: u64,
    /// Memory allocated (bytes)
    pub memory_allocated_bytes: u64,
    /// Disk bytes read
    pub disk_read_bytes: u64,
    /// Disk bytes written
    pub disk_write_bytes: u64,
    /// Network bytes sent
    pub network_sent_bytes: u64,
    /// Network bytes received
    pub network_received_bytes: u64,
}

/// Real-time performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total number of executions
    pub total_executions: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Average execution duration
    pub average_duration: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Peak concurrent executions
    pub peak_concurrent_executions: usize,
    /// Current concurrent executions
    pub current_concurrent_executions: usize,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            average_duration: Duration::from_millis(0),
            success_rate: 0.0,
            peak_concurrent_executions: 0,
            current_concurrent_executions: 0,
        }
    }

    /// Record the start of an execution
    pub fn record_execution_start(&mut self) {
        self.current_concurrent_executions += 1;
        if self.current_concurrent_executions > self.peak_concurrent_executions {
            self.peak_concurrent_executions = self.current_concurrent_executions;
        }
    }

    /// Record the completion of an execution
    pub fn record_execution_complete(&mut self, record: &ExecutionRecord) {
        self.current_concurrent_executions = self.current_concurrent_executions.saturating_sub(1);
        self.total_executions += 1;
        
        if record.success {
            self.successful_executions += 1;
        }
        
        // Update average duration (exponential moving average)
        if self.total_executions == 1 {
            self.average_duration = record.duration;
        } else {
            let alpha = 0.1; // Smoothing factor
            let current_avg_nanos = self.average_duration.as_nanos() as f64;
            let new_duration_nanos = record.duration.as_nanos() as f64;
            let updated_avg_nanos = (1.0 - alpha) * current_avg_nanos + alpha * new_duration_nanos;
            self.average_duration = Duration::from_nanos(updated_avg_nanos as u64);
        }
        
        // Update success rate
        self.success_rate = self.successful_executions as f64 / self.total_executions as f64;
    }
}

/// Current monitoring metrics snapshot
#[derive(Debug, Clone)]
pub struct MonitoringMetrics {
    /// Number of currently active executions
    pub active_executions: usize,
    /// Total executions processed
    pub total_executions: u64,
    /// Average execution duration
    pub average_duration: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Current resource usage
    pub current_resource_usage: SystemResourceSnapshot,
    /// Peak concurrent executions
    pub peak_concurrent_executions: usize,
}

/// Performance analytics engine
#[derive(Debug)]
pub struct PerformanceAnalyticsEngine {
    /// Analytics configuration
    config: AnalyticsConfig,
}

impl PerformanceAnalyticsEngine {
    /// Create new analytics engine
    pub fn new() -> Self {
        Self {
            config: AnalyticsConfig::default(),
        }
    }

    /// Analyze execution history for performance insights
    pub fn analyze(&self, history: &VecDeque<ExecutionRecord>, _config: &MonitoringConfig) -> PerformanceAnalytics {
        if history.is_empty() {
            return PerformanceAnalytics::empty();
        }

        let total_records = history.len();
        let successful_records: Vec<_> = history.iter().filter(|r| r.success).collect();
        
        // Calculate aggregate metrics
        let total_duration: Duration = history.iter().map(|r| r.duration).sum();
        let average_duration = total_duration / total_records as u32;
        
        let success_rate = successful_records.len() as f64 / total_records as f64;
        
        // Identify trends
        let duration_trend = self.analyze_duration_trend(history);
        let resource_trend = self.analyze_resource_trend(history);
        
        // Find common bottlenecks
        let common_bottlenecks = self.analyze_common_bottlenecks(history);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(history, &common_bottlenecks);

        PerformanceAnalytics {
            total_executions: total_records,
            success_rate,
            average_duration,
            duration_trend,
            resource_trend,
            common_bottlenecks,
            recommendations,
            efficiency_score: self.calculate_overall_efficiency(history),
        }
    }

    /// Analyze duration trends over time
    fn analyze_duration_trend(&self, history: &VecDeque<ExecutionRecord>) -> TrendAnalysis {
        if history.len() < 2 {
            return TrendAnalysis::Stable;
        }

        let mid_point = history.len() / 2;
        let first_half_avg: Duration = history.iter().take(mid_point).map(|r| r.duration).sum::<Duration>() / mid_point as u32;
        let second_half_avg: Duration = history.iter().skip(mid_point).map(|r| r.duration).sum::<Duration>() / (history.len() - mid_point) as u32;

        let change_ratio = second_half_avg.as_nanos() as f64 / first_half_avg.as_nanos() as f64;

        if change_ratio > 1.2 {
            TrendAnalysis::Increasing
        } else if change_ratio < 0.8 {
            TrendAnalysis::Decreasing
        } else {
            TrendAnalysis::Stable
        }
    }

    /// Analyze resource usage trends
    fn analyze_resource_trend(&self, history: &VecDeque<ExecutionRecord>) -> TrendAnalysis {
        if history.len() < 2 {
            return TrendAnalysis::Stable;
        }

        let mid_point = history.len() / 2;
        let first_half_avg: u64 = history.iter().take(mid_point)
            .map(|r| r.resource_consumption.total_resource_units())
            .sum::<u64>() / mid_point as u64;
        let second_half_avg: u64 = history.iter().skip(mid_point)
            .map(|r| r.resource_consumption.total_resource_units())
            .sum::<u64>() / (history.len() - mid_point) as u64;

        let change_ratio = second_half_avg as f64 / first_half_avg.max(1) as f64;

        if change_ratio > 1.2 {
            TrendAnalysis::Increasing
        } else if change_ratio < 0.8 {
            TrendAnalysis::Decreasing
        } else {
            TrendAnalysis::Stable
        }
    }

    /// Analyze common bottlenecks across executions
    fn analyze_common_bottlenecks(&self, history: &VecDeque<ExecutionRecord>) -> Vec<CommonBottleneck> {
        let mut bottleneck_counts: HashMap<BottleneckType, usize> = HashMap::new();
        let mut bottleneck_examples: HashMap<BottleneckType, Vec<String>> = HashMap::new();

        for record in history.iter() {
            for bottleneck in &record.performance_profile.bottlenecks {
                *bottleneck_counts.entry(bottleneck.bottleneck_type.clone()).or_insert(0) += 1;
                bottleneck_examples.entry(bottleneck.bottleneck_type.clone())
                    .or_default()
                    .push(bottleneck.description.clone());
            }
        }

        bottleneck_counts.into_iter()
            .filter(|(_, count)| *count > 1) // Only include bottlenecks that occur multiple times
            .map(|(bottleneck_type, frequency)| CommonBottleneck {
                bottleneck_type: bottleneck_type.clone(),
                frequency,
                percentage: frequency as f64 / history.len() as f64,
                examples: bottleneck_examples.get(&bottleneck_type)
                    .map(|examples| examples.iter().take(3).cloned().collect())
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self, history: &VecDeque<ExecutionRecord>, bottlenecks: &[CommonBottleneck]) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        // Recommend based on common bottlenecks
        for bottleneck in bottlenecks {
            if bottleneck.percentage > 0.5 { // More than 50% of executions
                let recommendation = match bottleneck.bottleneck_type {
                    BottleneckType::CPU => PerformanceRecommendation {
                        category: RecommendationCategory::Performance,
                        priority: RecommendationPriority::High,
                        title: "Optimize CPU Usage".to_string(),
                        description: "High CPU usage detected in multiple executions".to_string(),
                        actions: vec![
                            "Profile CPU-intensive operations".to_string(),
                            "Consider algorithmic optimizations".to_string(),
                            "Evaluate parallel processing opportunities".to_string(),
                        ],
                        estimated_impact: 0.4,
                    },
                    BottleneckType::Memory => PerformanceRecommendation {
                        category: RecommendationCategory::Memory,
                        priority: RecommendationPriority::Medium,
                        title: "Optimize Memory Usage".to_string(),
                        description: "High memory usage detected in multiple executions".to_string(),
                        actions: vec![
                            "Implement memory pooling".to_string(),
                            "Review object lifecycles".to_string(),
                            "Consider streaming for large datasets".to_string(),
                        ],
                        estimated_impact: 0.3,
                    },
                    BottleneckType::IO => PerformanceRecommendation {
                        category: RecommendationCategory::IO,
                        priority: RecommendationPriority::High,
                        title: "Optimize I/O Operations".to_string(),
                        description: "High I/O usage detected in multiple executions".to_string(),
                        actions: vec![
                            "Implement caching strategies".to_string(),
                            "Batch I/O operations".to_string(),
                            "Consider asynchronous I/O".to_string(),
                        ],
                        estimated_impact: 0.5,
                    },
                    BottleneckType::Network => PerformanceRecommendation {
                        category: RecommendationCategory::Network,
                        priority: RecommendationPriority::Medium,
                        title: "Optimize Network Usage".to_string(),
                        description: "High network usage detected in multiple executions".to_string(),
                        actions: vec![
                            "Implement response caching".to_string(),
                            "Compress network payloads".to_string(),
                            "Use connection pooling".to_string(),
                        ],
                        estimated_impact: 0.3,
                    },
                };
                recommendations.push(recommendation);
            }
        }

        // Check for low success rate
        let success_rate = history.iter().filter(|r| r.success).count() as f64 / history.len() as f64;
        if success_rate < 0.9 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::Reliability,
                priority: RecommendationPriority::Critical,
                title: "Improve Success Rate".to_string(),
                description: format!("Success rate is {:.1}%, below recommended 90%", success_rate * 100.0),
                actions: vec![
                    "Analyze failure patterns".to_string(),
                    "Implement better error handling".to_string(),
                    "Add retry mechanisms where appropriate".to_string(),
                ],
                estimated_impact: 0.6,
            });
        }

        recommendations
    }

    /// Calculate overall system efficiency
    fn calculate_overall_efficiency(&self, history: &VecDeque<ExecutionRecord>) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        let efficiency_sum: f64 = history.iter()
            .map(|r| r.performance_profile.execution_efficiency)
            .sum();
        
        efficiency_sum / history.len() as f64
    }
}

/// Performance analytics results
#[derive(Debug, Clone)]
pub struct PerformanceAnalytics {
    /// Total number of executions analyzed
    pub total_executions: usize,
    /// Overall success rate
    pub success_rate: f64,
    /// Average execution duration
    pub average_duration: Duration,
    /// Duration trend analysis
    pub duration_trend: TrendAnalysis,
    /// Resource usage trend analysis
    pub resource_trend: TrendAnalysis,
    /// Common bottlenecks across executions
    pub common_bottlenecks: Vec<CommonBottleneck>,
    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Overall efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

impl PerformanceAnalytics {
    /// Create empty analytics for when no data is available
    pub fn empty() -> Self {
        Self {
            total_executions: 0,
            success_rate: 0.0,
            average_duration: Duration::from_millis(0),
            duration_trend: TrendAnalysis::Stable,
            resource_trend: TrendAnalysis::Stable,
            common_bottlenecks: Vec::new(),
            recommendations: Vec::new(),
            efficiency_score: 0.0,
        }
    }
}

/// Trend analysis results
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendAnalysis {
    Increasing,
    Decreasing,
    Stable,
}

/// Common bottleneck analysis
#[derive(Debug, Clone)]
pub struct CommonBottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Percentage of executions affected
    pub percentage: f64,
    /// Example descriptions
    pub examples: Vec<String>,
}

/// Performance recommendation
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Specific actions to take
    pub actions: Vec<String>,
    /// Estimated performance impact (0.0 to 1.0)
    pub estimated_impact: f64,
}

/// Recommendation categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationCategory {
    Performance,
    Memory,
    IO,
    Network,
    Reliability,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Maximum number of execution records to keep in history
    pub max_history_size: usize,
    /// Whether to enable detailed resource monitoring
    pub enable_detailed_monitoring: bool,
    /// Minimum execution duration to track (filter out very short executions)
    pub min_tracking_duration: Duration,
    /// Whether to track checkpoints
    pub enable_checkpoints: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            enable_detailed_monitoring: true,
            min_tracking_duration: Duration::from_micros(100),
            enable_checkpoints: true,
        }
    }
}

/// Analytics configuration
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Minimum number of samples needed for trend analysis
    pub min_samples_for_trends: usize,
    /// Threshold for identifying significant bottlenecks
    pub bottleneck_threshold: f64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            min_samples_for_trends: 10,
            bottleneck_threshold: 0.1, // 10% of executions
        }
    }
}

/// Monitoring errors
#[derive(Debug, Error)]
pub enum MonitoringError {
    /// Execution not found
    #[error("Execution not found: {id:?}")]
    ExecutionNotFound {
        /// Execution ID
        id: ExecutionId,
    },

    /// Resource capture failed
    #[error("Resource capture failed: {reason}")]
    ResourceCaptureFailed {
        /// Failure reason
        reason: String,
    },

    /// Monitoring system error
    #[error("Monitoring system error: {message}")]
    SystemError {
        /// Error message
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_monitor_creation() {
        let monitor = EffectMonitor::new();
        let metrics = monitor.get_current_metrics();
        assert_eq!(metrics.active_executions, 0);
        assert_eq!(metrics.total_executions, 0);
    }

    #[test]
    fn test_execution_lifecycle() {
        let monitor = EffectMonitor::new();
        let effect = Effect::new("TestEffect".to_string(), Span::dummy());
        
        // Start execution
        let execution_id = monitor.start_execution(&effect).unwrap();
        assert_eq!(monitor.get_current_metrics().active_executions, 1);
        
        // End execution
        let record = monitor.end_execution(execution_id, true).unwrap();
        assert!(record.success);
        assert_eq!(monitor.get_current_metrics().active_executions, 0);
        assert_eq!(monitor.get_current_metrics().total_executions, 1);
    }

    #[test]
    fn test_resource_consumption_calculation() {
        let start = SystemResourceSnapshot {
            timestamp: Instant::now(),
            cpu_time_micros: 1000,
            memory_allocated_bytes: 1024,
            disk_read_bytes: 512,
            disk_write_bytes: 256,
            network_sent_bytes: 128,
            network_received_bytes: 64,
        };

        let end = SystemResourceSnapshot {
            timestamp: Instant::now(),
            cpu_time_micros: 2000,
            memory_allocated_bytes: 2048,
            disk_read_bytes: 1024,
            disk_write_bytes: 512,
            network_sent_bytes: 256,
            network_received_bytes: 128,
        };

        let consumption = ResourceConsumption::calculate(&start, &end);
        assert_eq!(consumption.cpu_time_micros, 1000);
        assert_eq!(consumption.memory_allocated_bytes, 1024);
        assert_eq!(consumption.disk_read_bytes, 512);
    }

    #[test]
    fn test_performance_metrics_updates() {
        let mut metrics = PerformanceMetrics::new();
        
        // Start execution
        metrics.record_execution_start();
        assert_eq!(metrics.current_concurrent_executions, 1);
        assert_eq!(metrics.peak_concurrent_executions, 1);
        
        // Complete execution
        let record = ExecutionRecord {
            id: ExecutionId(1),
            effect: Effect::new("Test".to_string(), Span::dummy()),
            start_time: Instant::now(),
            end_time: Instant::now(),
            duration: Duration::from_millis(100),
            success: true,
            resource_consumption: ResourceConsumption {
                cpu_time_micros: 1000,
                memory_allocated_bytes: 1024,
                disk_read_bytes: 0,
                disk_write_bytes: 0,
                network_sent_bytes: 0,
                network_received_bytes: 0,
            },
            checkpoints: Vec::new(),
            performance_profile: PerformanceProfile {
                execution_efficiency: 0.8,
                resource_intensity: ResourceIntensity::Low,
                bottlenecks: Vec::new(),
                optimization_opportunities: Vec::new(),
            },
        };
        
        metrics.record_execution_complete(&record);
        assert_eq!(metrics.current_concurrent_executions, 0);
        assert_eq!(metrics.total_executions, 1);
        assert_eq!(metrics.successful_executions, 1);
        assert_eq!(metrics.success_rate, 1.0);
    }
} 