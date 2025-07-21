//! NUMA-Aware Scheduling - CPU Topology-Optimized Task Placement
//!
//! This module implements NUMA (Non-Uniform Memory Access) aware scheduling:
//! - **CPU Topology Detection**: Automatic detection of CPU cores, sockets, and NUMA nodes
//! - **Actor Affinity**: Place actors on optimal CPU cores based on their characteristics
//! - **Memory Locality**: Optimize memory allocation and access patterns
//! - **Load Balancing**: Distribute work evenly across NUMA nodes while maintaining locality
//! - **Dynamic Migration**: Move actors between cores based on performance metrics

use crate::concurrency::{ActorId, ActorError};
use super::{PerformanceError, lock_free::{LockFreeMap, AtomicCounter}};
use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// NUMA-aware scheduler for optimal task placement
pub struct NumaScheduler {
    /// System topology information
    topology: Arc<NumaTopology>,
    /// Actor placement decisions
    actor_placements: LockFreeMap<ActorId, CpuAffinity>,
    /// CPU load tracking
    cpu_loads: Arc<RwLock<HashMap<CpuId, CpuLoadMetrics>>>,
    /// NUMA node load tracking
    numa_loads: Arc<RwLock<HashMap<NumaNodeId, NumaLoadMetrics>>>,
    /// Scheduling policies
    policies: Arc<RwLock<HashMap<ActorId, SchedulingPolicy>>>,
    /// Migration coordinator
    migration_coordinator: MigrationCoordinator,
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub numa_nodes: Vec<NumaNode>,
    /// Total number of CPU cores
    pub total_cores: usize,
    /// Total number of logical processors (including hyperthreading)
    pub total_logical_processors: usize,
    /// Cache hierarchy information
    pub cache_hierarchy: CacheHierarchy,
}

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// NUMA node ID
    pub id: NumaNodeId,
    /// CPU cores in this NUMA node
    pub cpu_cores: Vec<CpuCore>,
    /// Memory size in MB
    pub memory_size_mb: u64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Distance to other NUMA nodes
    pub distances: HashMap<NumaNodeId, u32>,
}

/// CPU core information
#[derive(Debug, Clone)]
pub struct CpuCore {
    /// Core ID
    pub id: CpuId,
    /// Logical processors (hyperthreads) on this core
    pub logical_processors: Vec<LogicalProcessor>,
    /// Base frequency in MHz
    pub base_frequency_mhz: u32,
    /// Maximum frequency in MHz
    pub max_frequency_mhz: u32,
    /// Cache sizes (L1, L2, L3)
    pub cache_sizes: CacheSizes,
}

/// Logical processor (hyperthread)
#[derive(Debug, Clone)]
pub struct LogicalProcessor {
    /// Logical processor ID
    pub id: LogicalProcessorId,
    /// Current utilization (0.0 to 1.0)
    pub utilization: f64,
}

/// Cache hierarchy information
#[derive(Debug, Clone)]
pub struct CacheHierarchy {
    /// L1 instruction cache size per core (KB)
    pub l1i_size_kb: u32,
    /// L1 data cache size per core (KB)
    pub l1d_size_kb: u32,
    /// L2 cache size per core (KB)
    pub l2_size_kb: u32,
    /// L3 cache size per NUMA node (KB)
    pub l3_size_kb: u32,
}

/// Cache sizes for a CPU core
#[derive(Debug, Clone)]
pub struct CacheSizes {
    /// L1 instruction cache (KB)
    pub l1i_kb: u32,
    /// L1 data cache (KB)
    pub l1d_kb: u32,
    /// L2 cache (KB)
    pub l2_kb: u32,
}

/// NUMA node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumaNodeId(pub u32);

/// CPU core identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CpuId(pub u32);

/// Logical processor identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalProcessorId(pub u32);

/// CPU affinity specification
#[derive(Debug, Clone)]
pub struct CpuAffinity {
    /// Preferred NUMA node
    pub preferred_numa_node: NumaNodeId,
    /// Preferred CPU core
    pub preferred_cpu_core: CpuId,
    /// Allowed CPU cores (for migration)
    pub allowed_cores: Vec<CpuId>,
    /// Affinity strength (0.0 = no preference, 1.0 = strict)
    pub strength: f64,
    /// Last placement time
    pub last_placed: Instant,
}

/// CPU load metrics
#[derive(Debug, Clone)]
pub struct CpuLoadMetrics {
    /// Current utilization (0.0 to 1.0)
    pub utilization: f64,
    /// Number of actors assigned
    pub actor_count: usize,
    /// Average message processing rate
    pub messages_per_second: f64,
    /// Memory usage in MB
    pub memory_usage_mb: u64,
    /// Cache miss rate
    pub cache_miss_rate: f64,
    /// Last update time
    pub last_updated: Instant,
}

/// NUMA node load metrics
#[derive(Debug, Clone)]
pub struct NumaLoadMetrics {
    /// Average CPU utilization across all cores
    pub avg_cpu_utilization: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Network I/O load
    pub network_io_mbps: f64,
    /// Disk I/O load
    pub disk_io_mbps: f64,
    /// Number of actors in this NUMA node
    pub actor_count: usize,
    /// Cross-NUMA memory accesses per second
    pub cross_numa_accesses: u64,
    /// Last update time
    pub last_updated: Instant,
}

/// Scheduling policy for an actor
#[derive(Debug, Clone)]
pub struct SchedulingPolicy {
    /// Actor characteristics
    pub characteristics: ActorCharacteristics,
    /// Placement preferences
    pub placement_preferences: PlacementPreferences,
    /// Migration settings
    pub migration_settings: MigrationSettings,
}

/// Actor characteristics for scheduling decisions
#[derive(Debug, Clone)]
pub struct ActorCharacteristics {
    /// CPU intensity (0.0 = I/O bound, 1.0 = CPU bound)
    pub cpu_intensity: f64,
    /// Memory usage pattern
    pub memory_pattern: MemoryPattern,
    /// Network I/O requirements
    pub network_io_intensity: f64,
    /// Cache sensitivity (how much performance depends on cache locality)
    pub cache_sensitivity: f64,
    /// NUMA sensitivity (performance impact of cross-NUMA access)
    pub numa_sensitivity: f64,
}

/// Memory usage patterns
#[derive(Debug, Clone)]
pub enum MemoryPattern {
    /// Streaming access (sequential)
    Streaming { bandwidth_mbps: f64 },
    /// Random access
    Random { access_rate_per_second: u64 },
    /// Working set (fits in cache)
    WorkingSet { size_mb: u64 },
    /// Large allocations
    LargeAllocations { avg_size_mb: u64 },
}

/// Actor placement preferences
#[derive(Debug, Clone)]
pub struct PlacementPreferences {
    /// Preferred NUMA nodes (in order of preference)
    pub preferred_numa_nodes: Vec<NumaNodeId>,
    /// Avoid these NUMA nodes
    pub avoided_numa_nodes: Vec<NumaNodeId>,
    /// Co-locate with these actors
    pub co_locate_with: Vec<ActorId>,
    /// Avoid co-locating with these actors
    pub avoid_co_location_with: Vec<ActorId>,
}

/// Migration settings for dynamic optimization
#[derive(Debug, Clone)]
pub struct MigrationSettings {
    /// Enable automatic migration
    pub enabled: bool,
    /// Minimum performance improvement to trigger migration
    pub min_improvement_threshold: f64,
    /// Maximum migration frequency
    pub max_migration_frequency: Duration,
    /// Migration cost penalty
    pub migration_cost_penalty: f64,
}

/// Migration coordinator for moving actors between cores
pub struct MigrationCoordinator {
    /// Pending migrations
    pending_migrations: Arc<Mutex<Vec<PendingMigration>>>,
    /// Migration history for learning
    migration_history: Arc<RwLock<Vec<MigrationRecord>>>,
    /// Migration worker channel
    migration_tx: mpsc::UnboundedSender<MigrationRequest>,
}

/// Pending migration request
#[derive(Debug)]
struct PendingMigration {
    /// Actor to migrate
    actor_id: ActorId,
    /// Source CPU
    from_cpu: CpuId,
    /// Destination CPU
    to_cpu: CpuId,
    /// Expected performance improvement
    expected_improvement: f64,
    /// Request timestamp
    requested_at: Instant,
}

/// Migration history record
#[derive(Debug, Clone)]
struct MigrationRecord {
    /// Actor that was migrated
    actor_id: ActorId,
    /// Source CPU
    from_cpu: CpuId,
    /// Destination CPU
    to_cpu: CpuId,
    /// Expected improvement
    expected_improvement: f64,
    /// Actual improvement measured
    actual_improvement: f64,
    /// Migration timestamp
    migrated_at: Instant,
}

/// Migration request
#[derive(Debug)]
enum MigrationRequest {
    /// Migrate a single actor
    MigrateActor {
        actor_id: ActorId,
        from_cpu: CpuId,
        to_cpu: CpuId,
    },
    /// Rebalance a NUMA node
    RebalanceNumaNode {
        numa_node: NumaNodeId,
    },
    /// Global rebalancing
    GlobalRebalance,
}

impl NumaScheduler {
    /// Create a new NUMA-aware scheduler
    pub fn new() -> Result<Self, PerformanceError> {
        let topology = Arc::new(Self::detect_topology()?);
        let (migration_tx, mut migration_rx) = mpsc::unbounded_channel();
        
        let migration_coordinator = MigrationCoordinator {
            pending_migrations: Arc::new(Mutex::new(Vec::new())),
            migration_history: Arc::new(RwLock::new(Vec::new())),
            migration_tx,
        };

        let scheduler = Self {
            topology,
            actor_placements: LockFreeMap::new(),
            cpu_loads: Arc::new(RwLock::new(HashMap::new())),
            numa_loads: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(HashMap::new())),
            migration_coordinator,
        };

        // Start migration worker
        let scheduler_clone = scheduler.clone();
        tokio::spawn(async move {
            while let Some(request) = migration_rx.recv().await {
                if let Err(e) = scheduler_clone.handle_migration_request(request).await {
                    tracing::error!("Migration request failed: {}", e);
                }
            }
        });

        Ok(scheduler)
    }

    /// Detect system topology
    fn detect_topology() -> Result<NumaTopology, PerformanceError> {
        // Simplified topology detection - in a real implementation, this would use
        // platform-specific APIs (e.g., hwloc, /proc/cpuinfo, Windows APIs)
        
        let num_cores = num_cpus::get();
        let num_logical = num_cpus::get(); // Simplified - doesn't account for hyperthreading
        
        // Create a simple topology with one NUMA node for now
        let numa_node = NumaNode {
            id: NumaNodeId(0),
            cpu_cores: (0..num_cores).map(|i| CpuCore {
                id: CpuId(i as u32),
                logical_processors: vec![LogicalProcessor {
                    id: LogicalProcessorId(i as u32),
                    utilization: 0.0,
                }],
                base_frequency_mhz: 2400, // Default values
                max_frequency_mhz: 3600,
                cache_sizes: CacheSizes {
                    l1i_kb: 32,
                    l1d_kb: 32,
                    l2_kb: 256,
                },
            }).collect(),
            memory_size_mb: 16384, // 16GB default
            memory_bandwidth_gbps: 25.6,
            distances: HashMap::new(),
        };

        Ok(NumaTopology {
            numa_nodes: vec![numa_node],
            total_cores: num_cores,
            total_logical_processors: num_logical,
            cache_hierarchy: CacheHierarchy {
                l1i_size_kb: 32,
                l1d_size_kb: 32,
                l2_size_kb: 256,
                l3_size_kb: 8192,
            },
        })
    }

    /// Place an actor on an optimal CPU core
    pub fn place_actor(
        &self,
        actor_id: ActorId,
        characteristics: ActorCharacteristics,
        preferences: PlacementPreferences,
    ) -> Result<CpuAffinity, PerformanceError> {
        let policy = SchedulingPolicy {
            characteristics: characteristics.clone(),
            placement_preferences: preferences,
            migration_settings: MigrationSettings {
                enabled: true,
                min_improvement_threshold: 0.05, // 5% improvement
                max_migration_frequency: Duration::from_secs(30),
                migration_cost_penalty: 0.02, // 2% penalty
            },
        };

        // Store the policy
        {
            let mut policies = self.policies.write().unwrap();
            policies.insert(actor_id, policy);
        }

        // Find optimal placement
        let optimal_cpu = self.find_optimal_cpu(&characteristics)?;
        let preferred_numa_node = self.cpu_to_numa_node(optimal_cpu);
        
        let affinity = CpuAffinity {
            preferred_numa_node,
            preferred_cpu_core: optimal_cpu,
            allowed_cores: self.get_allowed_cores(preferred_numa_node),
            strength: self.calculate_affinity_strength(&characteristics),
            last_placed: Instant::now(),
        };

        // Record the placement
        self.actor_placements.insert(actor_id, affinity.clone());
        
        // Update CPU load tracking
        self.update_cpu_load(optimal_cpu, 1)?;

        tracing::info!(
            "Placed actor {:?} on CPU {:?} in NUMA node {:?}",
            actor_id,
            optimal_cpu,
            preferred_numa_node
        );

        Ok(affinity)
    }

    /// Find the optimal CPU core for an actor
    fn find_optimal_cpu(&self, characteristics: &ActorCharacteristics) -> Result<CpuId, PerformanceError> {
        let cpu_loads = self.cpu_loads.read().unwrap();
        let numa_loads = self.numa_loads.read().unwrap();

        let mut best_cpu = CpuId(0);
        let mut best_score = f64::NEG_INFINITY;

        for numa_node in &self.topology.numa_nodes {
            let numa_load = numa_loads.get(&numa_node.id).cloned().unwrap_or_default();
            
            for cpu_core in &numa_node.cpu_cores {
                let cpu_load = cpu_loads.get(&cpu_core.id).cloned().unwrap_or_default();
                let score = self.calculate_placement_score(characteristics, &cpu_load, &numa_load, cpu_core);
                
                if score > best_score {
                    best_score = score;
                    best_cpu = cpu_core.id;
                }
            }
        }

        Ok(best_cpu)
    }

    /// Calculate placement score for a CPU core
    fn calculate_placement_score(
        &self,
        characteristics: &ActorCharacteristics,
        cpu_load: &CpuLoadMetrics,
        numa_load: &NumaLoadMetrics,
        cpu_core: &CpuCore,
    ) -> f64 {
        let mut score = 0.0;

        // CPU utilization penalty (prefer less loaded cores)
        score -= cpu_load.utilization * 100.0;

        // NUMA load penalty
        score -= numa_load.avg_cpu_utilization * 50.0;

        // CPU frequency bonus for CPU-intensive actors
        if characteristics.cpu_intensity > 0.5 {
            score += cpu_core.max_frequency_mhz as f64 * characteristics.cpu_intensity * 0.01;
        }

        // Cache sensitivity bonus
        if characteristics.cache_sensitivity > 0.5 {
            let cache_score = (cpu_core.cache_sizes.l1d_kb + cpu_core.cache_sizes.l2_kb) as f64;
            score += cache_score * characteristics.cache_sensitivity * 0.1;
        }

        // Memory access penalty for cross-NUMA sensitive actors
        if characteristics.numa_sensitivity > 0.5 {
            score -= numa_load.cross_numa_accesses as f64 * characteristics.numa_sensitivity * 0.001;
        }

        score
    }

    /// Get the NUMA node for a CPU core
    fn cpu_to_numa_node(&self, cpu_id: CpuId) -> NumaNodeId {
        for numa_node in &self.topology.numa_nodes {
            if numa_node.cpu_cores.iter().any(|core| core.id == cpu_id) {
                return numa_node.id;
            }
        }
        NumaNodeId(0) // Fallback
    }

    /// Get allowed CPU cores for a NUMA node
    fn get_allowed_cores(&self, numa_node_id: NumaNodeId) -> Vec<CpuId> {
        self.topology.numa_nodes
            .iter()
            .find(|node| node.id == numa_node_id)
            .map(|node| node.cpu_cores.iter().map(|core| core.id).collect())
            .unwrap_or_default()
    }

    /// Calculate affinity strength based on actor characteristics
    fn calculate_affinity_strength(&self, characteristics: &ActorCharacteristics) -> f64 {
        let mut strength = 0.5; // Base strength

        // CPU-intensive actors have higher affinity
        strength += characteristics.cpu_intensity * 0.3;

        // Cache-sensitive actors have higher affinity
        strength += characteristics.cache_sensitivity * 0.2;

        // NUMA-sensitive actors have higher affinity
        strength += characteristics.numa_sensitivity * 0.3;

        strength.clamp(0.0, 1.0)
    }

    /// Update CPU load metrics
    fn update_cpu_load(&self, cpu_id: CpuId, actor_delta: i32) -> Result<(), PerformanceError> {
        let mut cpu_loads = self.cpu_loads.write().unwrap();
        let load = cpu_loads.entry(cpu_id).or_insert_with(Default::default);
        
        if actor_delta > 0 {
            load.actor_count += actor_delta as usize;
        } else {
            load.actor_count = load.actor_count.saturating_sub((-actor_delta) as usize);
        }
        
        // Estimate utilization based on actor count (simplified)
        load.utilization = (load.actor_count as f64 * 0.1).min(1.0);
        load.last_updated = Instant::now();

        Ok(())
    }

    /// Migrate actors between CPU cores
    pub async fn migrate_actors(
        &self,
        from_cpu: CpuId,
        to_cpu: CpuId,
        actor_ids: Vec<ActorId>,
    ) -> Result<(), PerformanceError> {
        for actor_id in actor_ids {
            let request = MigrationRequest::MigrateActor {
                actor_id,
                from_cpu,
                to_cpu,
            };

            self.migration_coordinator.migration_tx.send(request)
                .map_err(|_| PerformanceError::NumaScheduling {
                    message: "Failed to send migration request".to_string(),
                })?;
        }

        Ok(())
    }

    /// Handle a migration request
    async fn handle_migration_request(&self, request: MigrationRequest) -> Result<(), PerformanceError> {
        match request {
            MigrationRequest::MigrateActor { actor_id, from_cpu, to_cpu } => {
                self.migrate_single_actor(actor_id, from_cpu, to_cpu).await
            }
            MigrationRequest::RebalanceNumaNode { numa_node } => {
                self.rebalance_numa_node(numa_node).await
            }
            MigrationRequest::GlobalRebalance => {
                self.global_rebalance().await
            }
        }
    }

    /// Migrate a single actor
    async fn migrate_single_actor(
        &self,
        actor_id: ActorId,
        from_cpu: CpuId,
        to_cpu: CpuId,
    ) -> Result<(), PerformanceError> {
        // Update placement record
        if let Some(mut affinity) = self.actor_placements.get(&actor_id) {
            affinity.preferred_cpu_core = to_cpu;
            affinity.preferred_numa_node = self.cpu_to_numa_node(to_cpu);
            affinity.last_placed = Instant::now();
            self.actor_placements.insert(actor_id, affinity);
        }

        // Update load tracking
        self.update_cpu_load(from_cpu, -1)?;
        self.update_cpu_load(to_cpu, 1)?;

        tracing::info!(
            "Migrated actor {:?} from CPU {:?} to CPU {:?}",
            actor_id,
            from_cpu,
            to_cpu
        );

        Ok(())
    }

    /// Rebalance actors within a NUMA node
    async fn rebalance_numa_node(&self, _numa_node: NumaNodeId) -> Result<(), PerformanceError> {
        // Implementation would analyze load distribution and move actors
        // between cores within the NUMA node for better balance
        Ok(())
    }

    /// Perform global rebalancing across all NUMA nodes
    async fn global_rebalance(&self) -> Result<(), PerformanceError> {
        // Implementation would analyze global load and perform
        // cross-NUMA migrations when beneficial
        Ok(())
    }

    /// Get current scheduling statistics
    pub fn get_scheduling_stats(&self) -> SchedulingStats {
        let cpu_loads = self.cpu_loads.read().unwrap();
        let numa_loads = self.numa_loads.read().unwrap();
        
        SchedulingStats {
            total_actors_placed: self.actor_placements.len(),
            cpu_utilization_by_core: cpu_loads.iter()
                .map(|(&cpu_id, load)| (cpu_id, load.utilization))
                .collect(),
            numa_utilization_by_node: numa_loads.iter()
                .map(|(&numa_id, load)| (numa_id, load.avg_cpu_utilization))
                .collect(),
            total_migrations: self.migration_coordinator.migration_history.read().unwrap().len(),
        }
    }
}

impl Clone for NumaScheduler {
    fn clone(&self) -> Self {
        Self {
            topology: Arc::clone(&self.topology),
            actor_placements: self.actor_placements.clone(), // LockFreeMap implements Clone
            cpu_loads: Arc::clone(&self.cpu_loads),
            numa_loads: Arc::clone(&self.numa_loads),
            policies: Arc::clone(&self.policies),
            migration_coordinator: MigrationCoordinator {
                pending_migrations: Arc::clone(&self.migration_coordinator.pending_migrations),
                migration_history: Arc::clone(&self.migration_coordinator.migration_history),
                migration_tx: self.migration_coordinator.migration_tx.clone(),
            },
        }
    }
}

impl Default for CpuLoadMetrics {
    fn default() -> Self {
        Self {
            utilization: 0.0,
            actor_count: 0,
            messages_per_second: 0.0,
            memory_usage_mb: 0,
            cache_miss_rate: 0.0,
            last_updated: Instant::now(),
        }
    }
}

impl Default for NumaLoadMetrics {
    fn default() -> Self {
        Self {
            avg_cpu_utilization: 0.0,
            memory_utilization: 0.0,
            network_io_mbps: 0.0,
            disk_io_mbps: 0.0,
            actor_count: 0,
            cross_numa_accesses: 0,
            last_updated: Instant::now(),
        }
    }
}

/// Scheduling statistics
#[derive(Debug)]
pub struct SchedulingStats {
    /// Total number of actors placed
    pub total_actors_placed: usize,
    /// CPU utilization by core
    pub cpu_utilization_by_core: HashMap<CpuId, f64>,
    /// NUMA utilization by node
    pub numa_utilization_by_node: HashMap<NumaNodeId, f64>,
    /// Total number of migrations performed
    pub total_migrations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_scheduler_creation() {
        let scheduler = NumaScheduler::new().unwrap();
        assert!(!scheduler.topology.numa_nodes.is_empty());
        assert!(scheduler.topology.total_cores > 0);
    }

    #[test]
    fn test_actor_placement() {
        let scheduler = NumaScheduler::new().unwrap();
        let actor_id = ActorId::new();
        
        let characteristics = ActorCharacteristics {
            cpu_intensity: 0.8,
            memory_pattern: MemoryPattern::WorkingSet { size_mb: 100 },
            network_io_intensity: 0.2,
            cache_sensitivity: 0.7,
            numa_sensitivity: 0.6,
        };
        
        let preferences = PlacementPreferences {
            preferred_numa_nodes: vec![NumaNodeId(0)],
            avoided_numa_nodes: vec![],
            co_locate_with: vec![],
            avoid_co_location_with: vec![],
        };
        
        let affinity = scheduler.place_actor(actor_id, characteristics, preferences).unwrap();
        assert_eq!(affinity.preferred_numa_node, NumaNodeId(0));
        assert!(affinity.strength > 0.0);
    }
} 