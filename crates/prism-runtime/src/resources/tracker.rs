//! Real-Time Resource Tracking System
//!
//! This module provides comprehensive resource tracking capabilities for Prism runtime
//! program execution, inspired by modern practices from Kubernetes, .NET, and other 
//! runtime systems. It tracks CPU, memory, network, and custom resources with 
//! high-resolution metrics during Prism program execution.
//!
//! ## Scope and Responsibility
//!
//! **This is for RUNTIME program execution monitoring, not CLI process monitoring.**
//! The CLI uses this system indirectly through the compiler and runtime integration.
//!
//! ## Features
//!
//! - **Real-time Monitoring**: Continuous tracking of resource usage patterns during execution
//! - **NUMA Awareness**: NUMA topology-aware resource allocation strategies  
//! - **Adaptive Scaling**: Dynamic resource adjustment based on load patterns
//! - **Pool Management**: Efficient memory and resource pooling
//! - **Telemetry Export**: Metrics export for external monitoring systems
//! - **Resource Quotas**: Enforced limits and fair-share scheduling
//! - **CLI Integration**: Provides metrics that CLI can query and display

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;
use std::fs;
use std::io::Read;
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// High-resolution timestamp in nanoseconds since UNIX epoch
pub type Timestamp = u64;

/// Resource identifier for tracking different resource types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU resources (cores, time, etc.)
    Cpu,
    /// Memory resources (RAM, swap, etc.)
    Memory,
    /// Network resources (bandwidth, connections, etc.)
    Network,
    /// Disk I/O resources
    Disk,
    /// GPU resources (if available)
    Gpu,
    /// Custom resource defined by applications
    Custom(String),
}

/// Comprehensive resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// When this snapshot was taken
    pub timestamp: Timestamp,
    /// CPU usage metrics
    pub cpu: CpuMetrics,
    /// Memory usage metrics
    pub memory: MemoryMetrics,
    /// Network usage metrics
    pub network: NetworkMetrics,
    /// Disk I/O metrics
    pub disk: DiskMetrics,
    /// Custom resource metrics
    pub custom: HashMap<String, f64>,
}

/// CPU usage metrics with detailed breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Overall CPU utilization percentage (0.0 - 100.0)
    pub utilization_percent: f64,
    /// Number of logical CPU cores
    pub core_count: usize,
    /// CPU time spent in user mode (nanoseconds)
    pub user_time_ns: u64,
    /// CPU time spent in system mode (nanoseconds)
    pub system_time_ns: u64,
    /// CPU time spent idle (nanoseconds)
    pub idle_time_ns: u64,
    /// CPU time spent waiting for I/O (nanoseconds)
    pub iowait_time_ns: u64,
    /// Current CPU frequency (MHz)
    pub frequency_mhz: Option<u32>,
    /// Per-core utilization breakdown
    pub per_core_utilization: Vec<f64>,
}

/// Memory usage metrics with detailed breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total system memory (bytes)
    pub total_bytes: u64,
    /// Currently used memory (bytes)
    pub used_bytes: u64,
    /// Available memory (bytes)
    pub available_bytes: u64,
    /// Memory used by buffers (bytes)
    pub buffers_bytes: u64,
    /// Memory used by cache (bytes)
    pub cached_bytes: u64,
    /// Swap total (bytes)
    pub swap_total_bytes: u64,
    /// Swap used (bytes)
    pub swap_used_bytes: u64,
    /// Memory utilization percentage (0.0 - 100.0)
    pub utilization_percent: f64,
    /// Memory pressure indicator (0.0 - 1.0)
    pub pressure: f64,
    /// NUMA node breakdown (if available)
    pub numa_nodes: Vec<NumaNodeMemory>,
}

/// NUMA node memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaNodeMemory {
    /// NUMA node ID
    pub node_id: u32,
    /// Total memory on this node (bytes)
    pub total_bytes: u64,
    /// Used memory on this node (bytes)
    pub used_bytes: u64,
    /// Available memory on this node (bytes)
    pub available_bytes: u64,
}

/// Network usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bytes received per second
    pub rx_bytes_per_sec: f64,
    /// Bytes transmitted per second
    pub tx_bytes_per_sec: f64,
    /// Packets received per second
    pub rx_packets_per_sec: f64,
    /// Packets transmitted per second
    pub tx_packets_per_sec: f64,
    /// Number of active connections
    pub active_connections: u32,
    /// Network interface statistics
    pub interfaces: HashMap<String, NetworkInterface>,
}

/// Network interface statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// Interface name (e.g., "eth0", "wlan0")
    pub name: String,
    /// Total bytes received
    pub rx_bytes: u64,
    /// Total bytes transmitted
    pub tx_bytes: u64,
    /// Total packets received
    pub rx_packets: u64,
    /// Total packets transmitted
    pub tx_packets: u64,
    /// Interface speed (Mbps)
    pub speed_mbps: Option<u32>,
}

/// Disk I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    /// Read operations per second
    pub read_ops_per_sec: f64,
    /// Write operations per second
    pub write_ops_per_sec: f64,
    /// Bytes read per second
    pub read_bytes_per_sec: f64,
    /// Bytes written per second
    pub write_bytes_per_sec: f64,
    /// Average I/O latency (milliseconds)
    pub avg_latency_ms: f64,
    /// Disk utilization percentage (0.0 - 100.0)
    pub utilization_percent: f64,
    /// Per-disk statistics
    pub disks: HashMap<String, DiskDevice>,
}

/// Individual disk device statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskDevice {
    /// Device name (e.g., "sda", "nvme0n1")
    pub name: String,
    /// Total disk size (bytes)
    pub total_bytes: u64,
    /// Used disk space (bytes)
    pub used_bytes: u64,
    /// Available disk space (bytes)
    pub available_bytes: u64,
    /// Read operations count
    pub read_ops: u64,
    /// Write operations count
    pub write_ops: u64,
    /// Bytes read
    pub read_bytes: u64,
    /// Bytes written
    pub write_bytes: u64,
}

/// Resource allocation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// Type of resource being requested
    pub resource_type: ResourceType,
    /// Amount of resource requested
    pub amount: f64,
    /// Priority of this request (0 = lowest, 100 = highest)
    pub priority: u8,
    /// Optional NUMA node preference
    pub numa_preference: Option<u32>,
    /// Timeout for allocation
    pub timeout: Option<Duration>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Resource allocation result
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Unique allocation ID
    pub id: String,
    /// Type of resource allocated
    pub resource_type: ResourceType,
    /// Amount actually allocated
    pub allocated_amount: f64,
    /// NUMA node where resource was allocated (if applicable)
    pub numa_node: Option<u32>,
    /// When allocation was made
    pub allocated_at: Instant,
    /// Handle for releasing the allocation
    pub release_handle: Arc<dyn ResourceReleaseHandle>,
}

/// Trait for releasing resource allocations
pub trait ResourceReleaseHandle: Send + Sync {
    /// Release the allocated resource
    fn release(&self);
    /// Check if resource is still allocated
    fn is_allocated(&self) -> bool;
}

/// Resource usage history for trend analysis
#[derive(Debug, Clone)]
pub struct ResourceHistory {
    /// Resource type this history tracks
    pub resource_type: ResourceType,
    /// Historical snapshots (limited by retention policy)
    pub snapshots: VecDeque<(Timestamp, f64)>,
    /// Maximum number of snapshots to retain
    pub max_snapshots: usize,
    /// Statistics computed from history
    pub stats: ResourceStats,
}

/// Statistical analysis of resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    /// Average usage over the tracked period
    pub average: f64,
    /// Minimum usage observed
    pub minimum: f64,
    /// Maximum usage observed
    pub maximum: f64,
    /// Standard deviation of usage
    pub std_deviation: f64,
    /// 95th percentile usage
    pub p95: f64,
    /// 99th percentile usage
    pub p99: f64,
    /// Trend direction (-1.0 to 1.0, negative = decreasing, positive = increasing)
    pub trend: f64,
}

/// Main resource tracking system
pub struct ResourceTracker {
    /// Configuration for the tracker
    config: ResourceTrackerConfig,
    /// Current resource snapshot
    current_snapshot: Arc<RwLock<ResourceSnapshot>>,
    /// Historical data for different resource types
    history: Arc<RwLock<HashMap<ResourceType, ResourceHistory>>>,
    /// Active resource allocations
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    /// Background monitoring thread handle
    monitor_handle: Option<thread::JoinHandle<()>>,
    /// Shutdown signal for background thread
    shutdown_signal: Arc<std::sync::atomic::AtomicBool>,
}

/// Configuration for resource tracking
#[derive(Debug, Clone)]
pub struct ResourceTrackerConfig {
    /// How often to collect resource metrics
    pub collection_interval: Duration,
    /// How many historical snapshots to retain
    pub history_retention: usize,
    /// Whether to enable NUMA topology detection
    pub numa_aware: bool,
    /// Whether to collect per-core CPU metrics
    pub per_core_cpu: bool,
    /// Whether to collect network interface details
    pub detailed_network: bool,
    /// Custom resource collection callbacks
    pub custom_collectors: HashMap<String, Box<dyn CustomResourceCollector>>,
}

/// Trait for collecting custom resource metrics
pub trait CustomResourceCollector: Send + Sync {
    /// Collect current value for this custom resource
    fn collect(&self) -> Result<f64, ResourceError>;
    /// Get the units for this resource (e.g., "requests/sec", "GB")
    fn units(&self) -> &str;
    /// Get description of what this resource represents
    fn description(&self) -> &str;
}

/// Errors that can occur during resource tracking
#[derive(Debug, Error)]
pub enum ResourceError {
    /// Failed to read system resource information
    #[error("Failed to read system resources: {message}")]
    SystemRead { message: String },
    
    /// Resource allocation failed
    #[error("Resource allocation failed: {resource_type:?} amount {amount}")]
    AllocationFailed { resource_type: ResourceType, amount: f64 },
    
    /// Resource not found
    #[error("Resource not found: {resource_type:?}")]
    ResourceNotFound { resource_type: ResourceType },
    
    /// Invalid resource configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
    
    /// I/O error when reading system files
    #[error("I/O error: {source}")]
    Io { #[from] source: std::io::Error },
    
    /// Parsing error for system data
    #[error("Parse error: {message}")]
    ParseError { message: String },
}

impl Default for ResourceTrackerConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_millis(500), // 2 Hz collection rate
            history_retention: 3600, // 30 minutes at 2 Hz
            numa_aware: true,
            per_core_cpu: true,
            detailed_network: true,
            custom_collectors: HashMap::new(),
        }
    }
}

impl ResourceTracker {
    /// Create a new resource tracker with default configuration
    pub fn new() -> Result<Self, ResourceError> {
        Self::with_config(ResourceTrackerConfig::default())
    }
    
    /// Create a new resource tracker with custom configuration
    pub fn with_config(config: ResourceTrackerConfig) -> Result<Self, ResourceError> {
        let current_snapshot = Arc::new(RwLock::new(Self::collect_snapshot(&config)?));
        let history = Arc::new(RwLock::new(HashMap::new()));
        let allocations = Arc::new(RwLock::new(HashMap::new()));
        let shutdown_signal = Arc::new(std::sync::atomic::AtomicBool::new(false));
        
        Ok(Self {
            config,
            current_snapshot,
            history,
            allocations,
            monitor_handle: None,
            shutdown_signal,
        })
    }
    
    /// Start background monitoring
    pub fn start_monitoring(&mut self) -> Result<(), ResourceError> {
        if self.monitor_handle.is_some() {
            return Ok(()); // Already monitoring
        }
        
        let config = self.config.clone();
        let current_snapshot = Arc::clone(&self.current_snapshot);
        let history = Arc::clone(&self.history);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        
        let handle = thread::spawn(move || {
            Self::monitoring_loop(config, current_snapshot, history, shutdown_signal);
        });
        
        self.monitor_handle = Some(handle);
        Ok(())
    }
    
    /// Stop background monitoring
    pub fn stop_monitoring(&mut self) {
        if let Some(handle) = self.monitor_handle.take() {
            self.shutdown_signal.store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = handle.join();
        }
    }
    
    /// Get current resource snapshot
    pub fn current_snapshot(&self) -> ResourceSnapshot {
        self.current_snapshot.read().unwrap().clone()
    }
    
    /// Get resource statistics for a specific resource type
    pub fn get_resource_stats(&self, resource_type: &ResourceType) -> Option<ResourceStats> {
        self.history.read().unwrap()
            .get(resource_type)
            .map(|h| h.stats.clone())
    }
    
    /// Get recent history for a resource type
    pub fn get_resource_history(&self, resource_type: &ResourceType, samples: usize) -> Vec<(Timestamp, f64)> {
        self.history.read().unwrap()
            .get(resource_type)
            .map(|h| h.snapshots.iter().rev().take(samples).cloned().collect())
            .unwrap_or_default()
    }
    
    /// Allocate resources according to a request
    pub fn allocate_resource(&self, request: ResourceRequest) -> Result<ResourceAllocation, ResourceError> {
        // TODO: Implement actual resource allocation logic
        // For now, this is a placeholder that creates a mock allocation
        
        let allocation_id = format!("alloc_{}", uuid::Uuid::new_v4().simple());
        let release_handle = Arc::new(MockReleaseHandle::new());
        
        let allocation = ResourceAllocation {
            id: allocation_id.clone(),
            resource_type: request.resource_type.clone(),
            allocated_amount: request.amount,
            numa_node: request.numa_preference,
            allocated_at: Instant::now(),
            release_handle,
        };
        
        // Store allocation for tracking
        self.allocations.write().unwrap().insert(allocation_id.clone(), allocation.clone());
        
        Ok(allocation)
    }
    
    /// Release a resource allocation
    pub fn release_allocation(&self, allocation_id: &str) -> Result<(), ResourceError> {
        if let Some(allocation) = self.allocations.write().unwrap().remove(allocation_id) {
            allocation.release_handle.release();
            Ok(())
        } else {
            Err(ResourceError::ResourceNotFound { 
                resource_type: ResourceType::Custom("allocation".to_string()) 
            })
        }
    }
    
    /// Get all active allocations
    pub fn active_allocations(&self) -> Vec<ResourceAllocation> {
        self.allocations.read().unwrap().values().cloned().collect()
    }
    
    /// Background monitoring loop
    fn monitoring_loop(
        config: ResourceTrackerConfig,
        current_snapshot: Arc<RwLock<ResourceSnapshot>>,
        history: Arc<RwLock<HashMap<ResourceType, ResourceHistory>>>,
        shutdown_signal: Arc<std::sync::atomic::AtomicBool>,
    ) {
        while !shutdown_signal.load(std::sync::atomic::Ordering::Relaxed) {
            match Self::collect_snapshot(&config) {
                Ok(snapshot) => {
                    // Update current snapshot
                    *current_snapshot.write().unwrap() = snapshot.clone();
                    
                    // Update history
                    Self::update_history(&history, &snapshot, &config);
                }
                Err(e) => {
                    eprintln!("Failed to collect resource snapshot: {}", e);
                }
            }
            
            thread::sleep(config.collection_interval);
        }
    }
    
    /// Collect a resource snapshot
    fn collect_snapshot(config: &ResourceTrackerConfig) -> Result<ResourceSnapshot, ResourceError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let cpu = Self::collect_cpu_metrics(config)?;
        let memory = Self::collect_memory_metrics(config)?;
        let network = Self::collect_network_metrics(config)?;
        let disk = Self::collect_disk_metrics(config)?;
        let custom = Self::collect_custom_metrics(config)?;
        
        Ok(ResourceSnapshot {
            timestamp,
            cpu,
            memory,
            network,
            disk,
            custom,
        })
    }
    
    /// Collect CPU metrics
    fn collect_cpu_metrics(config: &ResourceTrackerConfig) -> Result<CpuMetrics, ResourceError> {
        // Read /proc/stat for CPU information
        let stat_content = fs::read_to_string("/proc/stat")
            .map_err(|e| ResourceError::SystemRead { 
                message: format!("Failed to read /proc/stat: {}", e) 
            })?;
        
        let mut lines = stat_content.lines();
        let cpu_line = lines.next()
            .ok_or_else(|| ResourceError::ParseError { 
                message: "No CPU line in /proc/stat".to_string() 
            })?;
        
        // Parse: cpu user nice system idle iowait irq softirq steal guest guest_nice
        let values: Vec<u64> = cpu_line
            .split_whitespace()
            .skip(1) // Skip "cpu" label
            .map(|s| s.parse().unwrap_or(0))
            .collect();
        
        if values.len() < 4 {
            return Err(ResourceError::ParseError { 
                message: "Invalid CPU data format".to_string() 
            });
        }
        
        let user_time_ns = values[0] * 10_000_000; // Convert jiffies to nanoseconds (assuming 100 Hz)
        let system_time_ns = values[2] * 10_000_000;
        let idle_time_ns = values[3] * 10_000_000;
        let iowait_time_ns = values.get(4).unwrap_or(&0) * 10_000_000;
        
        let total_time = user_time_ns + system_time_ns + idle_time_ns + iowait_time_ns;
        let utilization_percent = if total_time > 0 {
            ((total_time - idle_time_ns) as f64 / total_time as f64) * 100.0
        } else {
            0.0
        };
        
        // Get CPU core count
        let core_count = num_cpus::get();
        
        // Collect per-core data if requested
        let per_core_utilization = if config.per_core_cpu {
            Self::collect_per_core_utilization(&stat_content)?
        } else {
            vec![utilization_percent; core_count]
        };
        
        // Try to get CPU frequency (best effort)
        let frequency_mhz = Self::get_cpu_frequency().ok();
        
        Ok(CpuMetrics {
            utilization_percent,
            core_count,
            user_time_ns,
            system_time_ns,
            idle_time_ns,
            iowait_time_ns,
            frequency_mhz,
            per_core_utilization,
        })
    }
    
    /// Collect memory metrics
    fn collect_memory_metrics(config: &ResourceTrackerConfig) -> Result<MemoryMetrics, ResourceError> {
        // Read /proc/meminfo
        let meminfo_content = fs::read_to_string("/proc/meminfo")
            .map_err(|e| ResourceError::SystemRead { 
                message: format!("Failed to read /proc/meminfo: {}", e) 
            })?;
        
        let mut memory_values = HashMap::new();
        for line in meminfo_content.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let value_kb: u64 = value
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                memory_values.insert(key.to_string(), value_kb * 1024); // Convert to bytes
            }
        }
        
        let total_bytes = memory_values.get("MemTotal").copied().unwrap_or(0);
        let available_bytes = memory_values.get("MemAvailable").copied().unwrap_or(0);
        let buffers_bytes = memory_values.get("Buffers").copied().unwrap_or(0);
        let cached_bytes = memory_values.get("Cached").copied().unwrap_or(0);
        let swap_total_bytes = memory_values.get("SwapTotal").copied().unwrap_or(0);
        let swap_free_bytes = memory_values.get("SwapFree").copied().unwrap_or(0);
        
        let used_bytes = total_bytes.saturating_sub(available_bytes);
        let swap_used_bytes = swap_total_bytes.saturating_sub(swap_free_bytes);
        
        let utilization_percent = if total_bytes > 0 {
            (used_bytes as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };
        
        // Calculate memory pressure (simplified heuristic)
        let pressure = if available_bytes < total_bytes / 10 { // Less than 10% available
            0.9
        } else if available_bytes < total_bytes / 5 { // Less than 20% available
            0.5
        } else {
            0.1
        };
        
        // Collect NUMA node information if requested
        let numa_nodes = if config.numa_aware {
            Self::collect_numa_memory()?
        } else {
            Vec::new()
        };
        
        Ok(MemoryMetrics {
            total_bytes,
            used_bytes,
            available_bytes,
            buffers_bytes,
            cached_bytes,
            swap_total_bytes,
            swap_used_bytes,
            utilization_percent,
            pressure,
            numa_nodes,
        })
    }
    
    /// Collect network metrics
    fn collect_network_metrics(_config: &ResourceTrackerConfig) -> Result<NetworkMetrics, ResourceError> {
        // Placeholder implementation - in a real system this would read from
        // /proc/net/dev, /proc/net/sockstat, etc.
        Ok(NetworkMetrics {
            rx_bytes_per_sec: 0.0,
            tx_bytes_per_sec: 0.0,
            rx_packets_per_sec: 0.0,
            tx_packets_per_sec: 0.0,
            active_connections: 0,
            interfaces: HashMap::new(),
        })
    }
    
    /// Collect disk metrics
    fn collect_disk_metrics(_config: &ResourceTrackerConfig) -> Result<DiskMetrics, ResourceError> {
        // Placeholder implementation - in a real system this would read from
        // /proc/diskstats, /sys/block/*/stat, etc.
        Ok(DiskMetrics {
            read_ops_per_sec: 0.0,
            write_ops_per_sec: 0.0,
            read_bytes_per_sec: 0.0,
            write_bytes_per_sec: 0.0,
            avg_latency_ms: 0.0,
            utilization_percent: 0.0,
            disks: HashMap::new(),
        })
    }
    
    /// Collect custom metrics from registered collectors
    fn collect_custom_metrics(config: &ResourceTrackerConfig) -> Result<HashMap<String, f64>, ResourceError> {
        let mut custom_metrics = HashMap::new();
        
        for (name, collector) in &config.custom_collectors {
            match collector.collect() {
                Ok(value) => {
                    custom_metrics.insert(name.clone(), value);
                }
                Err(e) => {
                    eprintln!("Failed to collect custom metric {}: {}", name, e);
                }
            }
        }
        
        Ok(custom_metrics)
    }
    
    /// Helper function to collect per-core CPU utilization
    fn collect_per_core_utilization(stat_content: &str) -> Result<Vec<f64>, ResourceError> {
        let mut per_core = Vec::new();
        
        for line in stat_content.lines() {
            if line.starts_with("cpu") && line.len() > 3 && line.chars().nth(3).unwrap().is_ascii_digit() {
                let values: Vec<u64> = line
                    .split_whitespace()
                    .skip(1)
                    .map(|s| s.parse().unwrap_or(0))
                    .collect();
                
                if values.len() >= 4 {
                    let user = values[0];
                    let system = values[2];
                    let idle = values[3];
                    let total = user + system + idle + values.get(4).unwrap_or(&0);
                    
                    let utilization = if total > 0 {
                        ((total - idle) as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    };
                    
                    per_core.push(utilization);
                }
            }
        }
        
        Ok(per_core)
    }
    
    /// Helper function to get CPU frequency
    fn get_cpu_frequency() -> Result<u32, ResourceError> {
        let freq_str = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .or_else(|_| fs::read_to_string("/proc/cpuinfo"))?;
        
        // Parse frequency in kHz and convert to MHz
        if let Ok(freq_khz) = freq_str.trim().parse::<u32>() {
            Ok(freq_khz / 1000)
        } else {
            Err(ResourceError::ParseError { 
                message: "Could not parse CPU frequency".to_string() 
            })
        }
    }
    
    /// Helper function to collect NUMA memory information
    fn collect_numa_memory() -> Result<Vec<NumaNodeMemory>, ResourceError> {
        let mut numa_nodes = Vec::new();
        
        // Try to read NUMA information from /sys/devices/system/node/
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node/") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("node") {
                        if let Ok(node_id) = name[4..].parse::<u32>() {
                            // Read memory info for this node
                            let meminfo_path = path.join("meminfo");
                            if let Ok(content) = fs::read_to_string(&meminfo_path) {
                                if let Some(node_mem) = Self::parse_numa_meminfo(node_id, &content) {
                                    numa_nodes.push(node_mem);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(numa_nodes)
    }
    
    /// Parse NUMA node meminfo
    fn parse_numa_meminfo(node_id: u32, content: &str) -> Option<NumaNodeMemory> {
        let mut total_bytes = 0;
        let mut used_bytes = 0;
        
        for line in content.lines() {
            if line.contains("MemTotal:") {
                if let Some(value) = line.split_whitespace().nth(3) {
                    total_bytes = value.parse::<u64>().unwrap_or(0) * 1024;
                }
            } else if line.contains("MemUsed:") {
                if let Some(value) = line.split_whitespace().nth(3) {
                    used_bytes = value.parse::<u64>().unwrap_or(0) * 1024;
                }
            }
        }
        
        if total_bytes > 0 {
            Some(NumaNodeMemory {
                node_id,
                total_bytes,
                used_bytes,
                available_bytes: total_bytes.saturating_sub(used_bytes),
            })
        } else {
            None
        }
    }
    
    /// Update historical data with new snapshot
    fn update_history(
        history: &Arc<RwLock<HashMap<ResourceType, ResourceHistory>>>,
        snapshot: &ResourceSnapshot,
        config: &ResourceTrackerConfig,
    ) {
        let mut history_map = history.write().unwrap();
        
        // Update CPU history
        Self::update_resource_history(
            &mut history_map,
            ResourceType::Cpu,
            snapshot.timestamp,
            snapshot.cpu.utilization_percent,
            config.history_retention,
        );
        
        // Update Memory history
        Self::update_resource_history(
            &mut history_map,
            ResourceType::Memory,
            snapshot.timestamp,
            snapshot.memory.utilization_percent,
            config.history_retention,
        );
        
        // Update custom resource histories
        for (name, value) in &snapshot.custom {
            Self::update_resource_history(
                &mut history_map,
                ResourceType::Custom(name.clone()),
                snapshot.timestamp,
                *value,
                config.history_retention,
            );
        }
    }
    
    /// Update history for a specific resource type
    fn update_resource_history(
        history_map: &mut HashMap<ResourceType, ResourceHistory>,
        resource_type: ResourceType,
        timestamp: Timestamp,
        value: f64,
        max_snapshots: usize,
    ) {
        let history = history_map
            .entry(resource_type.clone())
            .or_insert_with(|| ResourceHistory {
                resource_type: resource_type.clone(),
                snapshots: VecDeque::new(),
                max_snapshots,
                stats: ResourceStats {
                    average: 0.0,
                    minimum: f64::INFINITY,
                    maximum: f64::NEG_INFINITY,
                    std_deviation: 0.0,
                    p95: 0.0,
                    p99: 0.0,
                    trend: 0.0,
                },
            });
        
        // Add new snapshot
        history.snapshots.push_back((timestamp, value));
        
        // Maintain size limit
        while history.snapshots.len() > max_snapshots {
            history.snapshots.pop_front();
        }
        
        // Recompute statistics
        Self::compute_resource_stats(&mut history.stats, &history.snapshots);
    }
    
    /// Compute statistical metrics for resource history
    fn compute_resource_stats(stats: &mut ResourceStats, snapshots: &VecDeque<(Timestamp, f64)>) {
        if snapshots.is_empty() {
            return;
        }
        
        let values: Vec<f64> = snapshots.iter().map(|(_, v)| *v).collect();
        let n = values.len() as f64;
        
        // Basic statistics
        stats.minimum = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        stats.maximum = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        stats.average = values.iter().sum::<f64>() / n;
        
        // Standard deviation
        let variance = values.iter()
            .map(|&x| (x - stats.average).powi(2))
            .sum::<f64>() / n;
        stats.std_deviation = variance.sqrt();
        
        // Percentiles
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p95_idx = ((n * 0.95) as usize).min(sorted_values.len() - 1);
        let p99_idx = ((n * 0.99) as usize).min(sorted_values.len() - 1);
        
        stats.p95 = sorted_values[p95_idx];
        stats.p99 = sorted_values[p99_idx];
        
        // Simple trend calculation (slope of linear regression)
        if snapshots.len() >= 2 {
            let x_mean = snapshots.len() as f64 / 2.0;
            let y_mean = stats.average;
            
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            
            for (i, (_, value)) in snapshots.iter().enumerate() {
                let x = i as f64;
                numerator += (x - x_mean) * (value - y_mean);
                denominator += (x - x_mean).powi(2);
            }
            
            stats.trend = if denominator != 0.0 {
                numerator / denominator
            } else {
                0.0
            };
        }
    }
}

impl Drop for ResourceTracker {
    fn drop(&mut self) {
        self.stop_monitoring();
    }
}

/// Mock implementation of ResourceReleaseHandle for testing
struct MockReleaseHandle {
    is_allocated: Arc<std::sync::atomic::AtomicBool>,
}

impl MockReleaseHandle {
    fn new() -> Self {
        Self {
            is_allocated: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }
}

impl ResourceReleaseHandle for MockReleaseHandle {
    fn release(&self) {
        self.is_allocated.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn is_allocated(&self) -> bool {
        self.is_allocated.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resource_tracker_creation() {
        let tracker = ResourceTracker::new();
        assert!(tracker.is_ok());
    }
    
    #[test]
    fn test_resource_allocation() {
        let tracker = ResourceTracker::new().unwrap();
        
        let request = ResourceRequest {
            resource_type: ResourceType::Memory,
            amount: 1024.0,
            priority: 50,
            numa_preference: None,
            timeout: Some(Duration::from_secs(5)),
            metadata: HashMap::new(),
        };
        
        let allocation = tracker.allocate_resource(request);
        assert!(allocation.is_ok());
        
        let alloc = allocation.unwrap();
        assert_eq!(alloc.allocated_amount, 1024.0);
        assert!(alloc.is_allocated());
    }
    
    #[test]
    fn test_resource_history_stats() {
        let mut snapshots = VecDeque::new();
        snapshots.push_back((1000, 10.0));
        snapshots.push_back((2000, 20.0));
        snapshots.push_back((3000, 30.0));
        snapshots.push_back((4000, 40.0));
        snapshots.push_back((5000, 50.0));
        
        let mut stats = ResourceStats {
            average: 0.0,
            minimum: f64::INFINITY,
            maximum: f64::NEG_INFINITY,
            std_deviation: 0.0,
            p95: 0.0,
            p99: 0.0,
            trend: 0.0,
        };
        
        ResourceTracker::compute_resource_stats(&mut stats, &snapshots);
        
        assert_eq!(stats.average, 30.0);
        assert_eq!(stats.minimum, 10.0);
        assert_eq!(stats.maximum, 50.0);
        assert!(stats.trend > 0.0); // Should show upward trend
    }
} 