//! Resource Quotas and Limits System
//!
//! This module implements a comprehensive resource quota and limits enforcement
//! system inspired by Kubernetes ResourceQuotas, Docker container limits, and
//! modern container orchestration platforms. It provides fair-share scheduling,
//! burst capabilities, and hierarchical resource management.
//!
//! ## Features
//!
//! - **Hard and Soft Limits**: Enforceable caps and advisory warnings
//! - **Fair-Share Scheduling**: Proportional resource allocation
//! - **Burst Allowances**: Temporary over-allocation for bursty workloads
//! - **Hierarchical Quotas**: Nested resource hierarchies for teams/projects
//! - **Priority Classes**: Different QoS levels for different workloads
//! - **Time-based Quotas**: Rate limiting and time window controls

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use uuid::Uuid;

use crate::resources::tracker::{ResourceType, ResourceSnapshot};

/// Unique identifier for resource quotas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuotaId(Uuid);

impl QuotaId {
    /// Generate a new quota ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create from string representation
    pub fn from_string(s: &str) -> Result<Self, QuotaError> {
        let uuid = Uuid::parse_str(s)
            .map_err(|_| QuotaError::InvalidFormat { 
                field: "quota_id".to_string(),
                value: s.to_string() 
            })?;
        Ok(Self(uuid))
    }
}

/// Errors that can occur in quota management
#[derive(Debug, Error)]
pub enum QuotaError {
    /// Resource request exceeds quota limits
    #[error("Quota exceeded: {resource_type:?} requested {requested}, limit {limit}")]
    QuotaExceeded { resource_type: ResourceType, requested: f64, limit: f64 },
    
    /// Insufficient burst capacity
    #[error("Burst capacity exceeded: {resource_type:?} burst_used {burst_used}, burst_limit {burst_limit}")]
    BurstExceeded { resource_type: ResourceType, burst_used: f64, burst_limit: f64 },
    
    /// Rate limit exceeded
    #[error("Rate limit exceeded: {operations} operations in {window:?}, limit {limit}")]
    RateLimitExceeded { operations: u64, window: Duration, limit: u64 },
    
    /// Quota not found
    #[error("Quota not found: {quota_id:?}")]
    QuotaNotFound { quota_id: QuotaId },
    
    /// Invalid quota configuration
    #[error("Invalid quota configuration: {message}")]
    InvalidConfig { message: String },
    
    /// Invalid format for field
    #[error("Invalid format for {field}: {value}")]
    InvalidFormat { field: String, value: String },
    
    /// Priority class not found
    #[error("Priority class not found: {class_name}")]
    PriorityClassNotFound { class_name: String },
    
    /// Hierarchical quota conflict
    #[error("Hierarchical quota conflict: child quota exceeds parent limits")]
    HierarchyConflict,
}

/// Priority class for workloads
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PriorityClass {
    /// Critical system workloads (highest priority)
    System,
    /// High priority production workloads
    High,
    /// Normal priority workloads
    Normal,
    /// Low priority batch workloads
    Low,
    /// Best effort workloads (lowest priority)
    BestEffort,
}

impl PriorityClass {
    /// Get numeric priority value (higher = more important)
    pub fn priority_value(&self) -> u32 {
        match self {
            PriorityClass::System => 1000,
            PriorityClass::High => 800,
            PriorityClass::Normal => 500,
            PriorityClass::Low => 200,
            PriorityClass::BestEffort => 100,
        }
    }
    
    /// Get default resource multiplier for this priority class
    pub fn resource_multiplier(&self) -> f64 {
        match self {
            PriorityClass::System => 2.0,
            PriorityClass::High => 1.5,
            PriorityClass::Normal => 1.0,
            PriorityClass::Low => 0.75,
            PriorityClass::BestEffort => 0.5,
        }
    }
}

/// Resource limit specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit {
    /// Type of resource this limit applies to
    pub resource_type: ResourceType,
    /// Hard limit (cannot be exceeded)
    pub hard_limit: f64,
    /// Soft limit (warning threshold)
    pub soft_limit: f64,
    /// Burst limit (temporary overage allowed)
    pub burst_limit: Option<f64>,
    /// Burst duration (how long burst is allowed)
    pub burst_duration: Option<Duration>,
    /// Units for this resource (e.g., "bytes", "cores", "requests/sec")
    pub units: String,
}

/// Resource quota specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Unique identifier for this quota
    pub id: QuotaId,
    /// Human-readable name
    pub name: String,
    /// Description of what this quota covers
    pub description: String,
    /// Resource limits for different resource types
    pub limits: HashMap<ResourceType, ResourceLimit>,
    /// Priority class for workloads using this quota
    pub priority_class: PriorityClass,
    /// Parent quota ID (for hierarchical quotas)
    pub parent_quota: Option<QuotaId>,
    /// Child quota IDs
    pub child_quotas: Vec<QuotaId>,
    /// Whether this quota is currently active
    pub active: bool,
    /// When this quota was created
    pub created_at: SystemTime,
    /// When this quota expires (if any)
    pub expires_at: Option<SystemTime>,
    /// Metadata tags
    pub tags: HashMap<String, String>,
}

/// Current usage against a quota
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaUsage {
    /// Quota this usage is for
    pub quota_id: QuotaId,
    /// Current usage by resource type
    pub current_usage: HashMap<ResourceType, f64>,
    /// Peak usage observed
    pub peak_usage: HashMap<ResourceType, f64>,
    /// Burst usage (usage above hard limit)
    pub burst_usage: HashMap<ResourceType, f64>,
    /// When burst started (if any)
    pub burst_started_at: HashMap<ResourceType, SystemTime>,
    /// Number of times limits were hit
    pub limit_violations: HashMap<ResourceType, u64>,
    /// Last time usage was updated
    pub last_updated: SystemTime,
}

/// Resource request for quota checking
#[derive(Debug, Clone)]
pub struct ResourceRequest {
    /// Type of resource being requested
    pub resource_type: ResourceType,
    /// Amount being requested
    pub amount: f64,
    /// Priority class for this request
    pub priority_class: PriorityClass,
    /// How long the resource will be held
    pub duration: Option<Duration>,
    /// Whether this request can use burst capacity
    pub allow_burst: bool,
    /// Metadata for tracking
    pub metadata: HashMap<String, String>,
}

/// Result of quota checking
#[derive(Debug, Clone)]
pub struct QuotaCheckResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Quota that was checked
    pub quota_id: QuotaId,
    /// Resource type that was checked
    pub resource_type: ResourceType,
    /// Amount that was requested
    pub requested_amount: f64,
    /// Current usage before this request
    pub current_usage: f64,
    /// Limit that applies
    pub applicable_limit: f64,
    /// Whether burst capacity was used
    pub used_burst: bool,
    /// Warning message if approaching limits
    pub warning: Option<String>,
    /// Handle for tracking this allocation
    pub allocation_id: Option<String>,
}

/// Rate limiting window for quota enforcement
#[derive(Debug, Clone)]
struct RateWindow {
    /// Window duration
    duration: Duration,
    /// Maximum operations in window
    max_operations: u64,
    /// Operation timestamps in current window
    operations: VecDeque<Instant>,
}

impl RateWindow {
    fn new(duration: Duration, max_operations: u64) -> Self {
        Self {
            duration,
            max_operations,
            operations: VecDeque::new(),
        }
    }
    
    /// Check if an operation can be performed
    fn check_rate_limit(&mut self) -> bool {
        let now = Instant::now();
        let window_start = now - self.duration;
        
        // Remove old operations outside the window
        while let Some(&front_time) = self.operations.front() {
            if front_time < window_start {
                self.operations.pop_front();
            } else {
                break;
            }
        }
        
        // Check if we can add another operation
        self.operations.len() < self.max_operations as usize
    }
    
    /// Record an operation
    fn record_operation(&mut self) {
                    self.operations.push_back(Instant::now());
    }
}

/// Main quota management system
#[derive(Debug)]
pub struct QuotaManager {
    /// All configured quotas
    quotas: RwLock<HashMap<QuotaId, ResourceQuota>>,
    /// Current usage for each quota
    usage: RwLock<HashMap<QuotaId, QuotaUsage>>,
    /// Active allocations
    allocations: RwLock<HashMap<String, ActiveAllocation>>,
    /// Rate limiting windows
    rate_limits: Mutex<HashMap<QuotaId, HashMap<ResourceType, RateWindow>>>,
    /// Fair-share scheduler state
    fair_share: Mutex<FairShareScheduler>,
}

/// Active resource allocation
#[derive(Debug, Clone)]
struct ActiveAllocation {
    /// Unique allocation ID
    id: String,
    /// Quota this allocation is charged against
    quota_id: QuotaId,
    /// Resource type allocated
    resource_type: ResourceType,
    /// Amount allocated
    amount: f64,
    /// When allocation was made
    allocated_at: Instant,
    /// Expected duration
    duration: Option<Duration>,
    /// Whether this used burst capacity
    is_burst: bool,
}

/// Fair-share scheduler for proportional resource allocation
#[derive(Debug)]
struct FairShareScheduler {
    /// Share assignments by quota ID
    shares: HashMap<QuotaId, f64>,
    /// Current usage for fair-share calculation
    current_usage: HashMap<QuotaId, HashMap<ResourceType, f64>>,
    /// Pending requests by priority
    pending_requests: BTreeMap<u32, Vec<PendingRequest>>,
}

/// Pending resource request in fair-share queue
#[derive(Debug, Clone)]
struct PendingRequest {
    /// Request ID
    id: String,
    /// Quota requesting resources
    quota_id: QuotaId,
    /// Resource request details
    request: ResourceRequest,
    /// When request was made
    requested_at: Instant,
}

impl QuotaManager {
    /// Create a new quota manager
    pub fn new() -> Self {
        Self {
            quotas: RwLock::new(HashMap::new()),
            usage: RwLock::new(HashMap::new()),
            allocations: RwLock::new(HashMap::new()),
            rate_limits: Mutex::new(HashMap::new()),
            fair_share: Mutex::new(FairShareScheduler {
                shares: HashMap::new(),
                current_usage: HashMap::new(),
                pending_requests: BTreeMap::new(),
            }),
        }
    }
    
    /// Create a new resource quota
    pub fn create_quota(&self, quota: ResourceQuota) -> Result<(), QuotaError> {
        // Validate quota configuration
        self.validate_quota(&quota)?;
        
        let quota_id = quota.id;
        
        // Insert quota
        self.quotas.write().unwrap().insert(quota_id, quota.clone());
        
        // Initialize usage tracking
        let usage = QuotaUsage {
            quota_id,
            current_usage: HashMap::new(),
            peak_usage: HashMap::new(),
            burst_usage: HashMap::new(),
            burst_started_at: HashMap::new(),
            limit_violations: HashMap::new(),
            last_updated: SystemTime::now(),
        };
        self.usage.write().unwrap().insert(quota_id, usage);
        
        // Initialize rate limits
        let mut rate_limits = self.rate_limits.lock().unwrap();
        rate_limits.insert(quota_id, HashMap::new());
        
        // Initialize fair-share
        let mut fair_share = self.fair_share.lock().unwrap();
        fair_share.shares.insert(quota_id, quota.priority_class.resource_multiplier());
        fair_share.current_usage.insert(quota_id, HashMap::new());
        
        Ok(())
    }
    
    /// Update an existing quota
    pub fn update_quota(&self, quota: ResourceQuota) -> Result<(), QuotaError> {
        self.validate_quota(&quota)?;
        
        let mut quotas = self.quotas.write().unwrap();
        if quotas.contains_key(&quota.id) {
            quotas.insert(quota.id, quota);
            Ok(())
        } else {
            Err(QuotaError::QuotaNotFound { quota_id: quota.id })
        }
    }
    
    /// Delete a quota
    pub fn delete_quota(&self, quota_id: QuotaId) -> Result<(), QuotaError> {
        let mut quotas = self.quotas.write().unwrap();
        if quotas.remove(&quota_id).is_some() {
            // Clean up usage tracking
            self.usage.write().unwrap().remove(&quota_id);
            self.rate_limits.lock().unwrap().remove(&quota_id);
            
            let mut fair_share = self.fair_share.lock().unwrap();
            fair_share.shares.remove(&quota_id);
            fair_share.current_usage.remove(&quota_id);
            
            Ok(())
        } else {
            Err(QuotaError::QuotaNotFound { quota_id })
        }
    }
    
    /// Check if a resource request is allowed under quota
    pub fn check_quota(&self, quota_id: QuotaId, request: &ResourceRequest) -> Result<QuotaCheckResult, QuotaError> {
        let quotas = self.quotas.read().unwrap();
        let quota = quotas.get(&quota_id)
            .ok_or(QuotaError::QuotaNotFound { quota_id })?;
        
        let usage = self.usage.read().unwrap();
        let current_usage = usage.get(&quota_id)
            .ok_or(QuotaError::QuotaNotFound { quota_id })?;
        
        // Get current usage for this resource type
        let current_amount = current_usage.current_usage
            .get(&request.resource_type)
            .copied()
            .unwrap_or(0.0);
        
        // Get applicable limit
        let limit = quota.limits.get(&request.resource_type)
            .ok_or_else(|| QuotaError::InvalidConfig { 
                message: format!("No limit configured for resource type {:?}", request.resource_type) 
            })?;
        
        let new_usage = current_amount + request.amount;
        
        // Check hard limit first
        if new_usage <= limit.hard_limit {
            // Within hard limit - check soft limit for warnings
            let warning = if new_usage > limit.soft_limit {
                Some(format!("Approaching limit: {}% of soft limit used", 
                    (new_usage / limit.soft_limit * 100.0) as u32))
            } else {
                None
            };
            
            return Ok(QuotaCheckResult {
                allowed: true,
                quota_id,
                resource_type: request.resource_type.clone(),
                requested_amount: request.amount,
                current_usage: current_amount,
                applicable_limit: limit.hard_limit,
                used_burst: false,
                warning,
                allocation_id: Some(Uuid::new_v4().to_string()),
            });
        }
        
        // Check if burst is allowed and configured
        if request.allow_burst && limit.burst_limit.is_some() {
            let burst_limit = limit.burst_limit.unwrap();
            
            if new_usage <= burst_limit {
                // Check burst duration
                if let Some(burst_duration) = limit.burst_duration {
                    let burst_start = current_usage.burst_started_at
                        .get(&request.resource_type)
                        .copied()
                        .unwrap_or_else(SystemTime::now);
                    
                    if SystemTime::now().duration_since(burst_start).unwrap_or_default() > burst_duration {
                        return Err(QuotaError::BurstExceeded { 
                            resource_type: request.resource_type.clone(),
                            burst_used: new_usage - limit.hard_limit,
                            burst_limit: burst_limit - limit.hard_limit,
                        });
                    }
                }
                
                return Ok(QuotaCheckResult {
                    allowed: true,
                    quota_id,
                    resource_type: request.resource_type.clone(),
                    requested_amount: request.amount,
                    current_usage: current_amount,
                    applicable_limit: burst_limit,
                    used_burst: true,
                    warning: Some("Using burst capacity".to_string()),
                    allocation_id: Some(Uuid::new_v4().to_string()),
                });
            }
        }
        
        // Request exceeds all limits
        Err(QuotaError::QuotaExceeded { 
            resource_type: request.resource_type.clone(),
            requested: request.amount,
            limit: limit.hard_limit,
        })
    }
    
    /// Allocate resources against a quota
    pub fn allocate(&self, quota_id: QuotaId, request: ResourceRequest) -> Result<String, QuotaError> {
        // Check quota first
        let check_result = self.check_quota(quota_id, &request)?;
        
        if !check_result.allowed {
            return Err(QuotaError::QuotaExceeded { 
                resource_type: request.resource_type.clone(),
                requested: request.amount,
                limit: check_result.applicable_limit,
            });
        }
        
        let allocation_id = check_result.allocation_id.unwrap();
        
        // Create allocation record
        let allocation = ActiveAllocation {
            id: allocation_id.clone(),
            quota_id,
            resource_type: request.resource_type.clone(),
            amount: request.amount,
                            allocated_at: Instant::now(),
            duration: request.duration,
            is_burst: check_result.used_burst,
        };
        
        // Update usage
        {
            let mut usage = self.usage.write().unwrap();
            if let Some(quota_usage) = usage.get_mut(&quota_id) {
                let current = quota_usage.current_usage
                    .entry(request.resource_type.clone())
                    .or_insert(0.0);
                *current += request.amount;
                
                // Update peak usage
                let peak = quota_usage.peak_usage
                    .entry(request.resource_type.clone())
                    .or_insert(0.0);
                if *current > *peak {
                    *peak = *current;
                }
                
                // Update burst usage if applicable
                if check_result.used_burst {
                    let quotas = self.quotas.read().unwrap();
                    if let Some(quota) = quotas.get(&quota_id) {
                        if let Some(limit) = quota.limits.get(&request.resource_type) {
                            if *current > limit.hard_limit {
                                quota_usage.burst_usage
                                    .entry(request.resource_type.clone())
                                    .and_modify(|v| *v += request.amount)
                                    .or_insert(request.amount);
                                
                                quota_usage.burst_started_at
                                    .entry(request.resource_type.clone())
                                    .or_insert(SystemTime::now());
                            }
                        }
                    }
                }
                
                quota_usage.last_updated = SystemTime::now();
            }
        }
        
        // Store allocation
        self.allocations.write().unwrap().insert(allocation_id.clone(), allocation);
        
        Ok(allocation_id)
    }
    
    /// Release a resource allocation
    pub fn deallocate(&self, allocation_id: &str) -> Result<(), QuotaError> {
        let allocation = {
            let mut allocations = self.allocations.write().unwrap();
            allocations.remove(allocation_id)
                .ok_or_else(|| QuotaError::InvalidFormat { 
                    field: "allocation_id".to_string(),
                    value: allocation_id.to_string() 
                })?
        };
        
        // Update usage
        let mut usage = self.usage.write().unwrap();
        if let Some(quota_usage) = usage.get_mut(&allocation.quota_id) {
            let current = quota_usage.current_usage
                .entry(allocation.resource_type.clone())
                .or_insert(0.0);
            *current = (*current - allocation.amount).max(0.0);
            
            // Update burst usage if applicable
            if allocation.is_burst {
                let burst = quota_usage.burst_usage
                    .entry(allocation.resource_type.clone())
                    .or_insert(0.0);
                *burst = (*burst - allocation.amount).max(0.0);
                
                // Clear burst start time if no longer using burst
                if *burst <= 0.0 {
                    quota_usage.burst_started_at.remove(&allocation.resource_type);
                }
            }
            
            quota_usage.last_updated = SystemTime::now();
        }
        
        Ok(())
    }
    
    /// Get current usage for a quota
    pub fn get_usage(&self, quota_id: QuotaId) -> Result<QuotaUsage, QuotaError> {
        let usage = self.usage.read().unwrap();
        usage.get(&quota_id)
            .cloned()
            .ok_or(QuotaError::QuotaNotFound { quota_id })
    }
    
    /// Get all quotas
    pub fn list_quotas(&self) -> Vec<ResourceQuota> {
        self.quotas.read().unwrap().values().cloned().collect()
    }
    
    /// Update usage from resource snapshot
    pub fn update_from_snapshot(&self, snapshot: &ResourceSnapshot) {
        // This would be called by the resource tracker to update current usage
        // based on actual system resource consumption
        let timestamp = snapshot.timestamp;
        
        // For each quota, update usage based on actual resource consumption
        // This is a simplified implementation - in practice, you'd need to
        // correlate system usage with specific quotas/workloads
        
        let mut usage = self.usage.write().unwrap();
        for (quota_id, quota_usage) in usage.iter_mut() {
            // Update CPU usage
            quota_usage.current_usage.insert(
                ResourceType::Cpu, 
                snapshot.cpu.utilization_percent / 100.0
            );
            
            // Update memory usage
            quota_usage.current_usage.insert(
                ResourceType::Memory,
                snapshot.memory.utilization_percent / 100.0
            );
            
            quota_usage.last_updated = SystemTime::now();
        }
    }
    
    /// Validate quota configuration
    fn validate_quota(&self, quota: &ResourceQuota) -> Result<(), QuotaError> {
        // Check that soft limits <= hard limits
        for limit in quota.limits.values() {
            if limit.soft_limit > limit.hard_limit {
                return Err(QuotaError::InvalidConfig { 
                    message: "Soft limit cannot exceed hard limit".to_string() 
                });
            }
            
            if let Some(burst_limit) = limit.burst_limit {
                if burst_limit < limit.hard_limit {
                    return Err(QuotaError::InvalidConfig { 
                        message: "Burst limit cannot be less than hard limit".to_string() 
                    });
                }
            }
        }
        
        // Check parent-child hierarchy
        if let Some(parent_id) = quota.parent_quota {
            let quotas = self.quotas.read().unwrap();
            if let Some(parent) = quotas.get(&parent_id) {
                // Validate that child limits don't exceed parent limits
                for (resource_type, child_limit) in &quota.limits {
                    if let Some(parent_limit) = parent.limits.get(resource_type) {
                        if child_limit.hard_limit > parent_limit.hard_limit {
                            return Err(QuotaError::HierarchyConflict);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Create a default system quota
pub fn create_system_quota() -> ResourceQuota {
    let mut limits = HashMap::new();
    
    // CPU limit: 100% (all cores)
    limits.insert(ResourceType::Cpu, ResourceLimit {
        resource_type: ResourceType::Cpu,
        hard_limit: 1.0, // 100%
        soft_limit: 0.8, // 80%
        burst_limit: Some(1.2), // 120% for short periods
        burst_duration: Some(Duration::from_secs(30)),
        units: "ratio".to_string(),
    });
    
    // Memory limit: 16 GB
    limits.insert(ResourceType::Memory, ResourceLimit {
        resource_type: ResourceType::Memory,
        hard_limit: 16_000_000_000.0, // 16 GB
        soft_limit: 12_000_000_000.0,  // 12 GB
        burst_limit: Some(18_000_000_000.0), // 18 GB burst
        burst_duration: Some(Duration::from_secs(60)),
        units: "bytes".to_string(),
    });
    
    ResourceQuota {
        id: QuotaId::new(),
        name: "system".to_string(),
        description: "Default system resource quota".to_string(),
        limits,
        priority_class: PriorityClass::System,
        parent_quota: None,
        child_quotas: Vec::new(),
        active: true,
        created_at: SystemTime::now(),
        expires_at: None,
        tags: [("type".to_string(), "system".to_string())].into_iter().collect(),
    }
}

/// Create a user workload quota
pub fn create_user_quota(name: String, cpu_cores: f64, memory_gb: f64) -> ResourceQuota {
    let mut limits = HashMap::new();
    
    limits.insert(ResourceType::Cpu, ResourceLimit {
        resource_type: ResourceType::Cpu,
        hard_limit: cpu_cores,
        soft_limit: cpu_cores * 0.8,
        burst_limit: Some(cpu_cores * 1.2),
        burst_duration: Some(Duration::from_secs(60)),
        units: "cores".to_string(),
    });
    
    limits.insert(ResourceType::Memory, ResourceLimit {
        resource_type: ResourceType::Memory,
        hard_limit: memory_gb * 1_000_000_000.0,
        soft_limit: memory_gb * 0.8 * 1_000_000_000.0,
        burst_limit: Some(memory_gb * 1.1 * 1_000_000_000.0),
        burst_duration: Some(Duration::from_secs(120)),
        units: "bytes".to_string(),
    });
    
    ResourceQuota {
        id: QuotaId::new(),
        name,
        description: "User workload quota".to_string(),
        limits,
        priority_class: PriorityClass::Normal,
        parent_quota: None,
        child_quotas: Vec::new(),
        active: true,
        created_at: SystemTime::now(),
        expires_at: None,
        tags: [("type".to_string(), "user".to_string())].into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quota_creation() {
        let manager = QuotaManager::new();
        let quota = create_system_quota();
        
        let result = manager.create_quota(quota);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_quota_checking() {
        let manager = QuotaManager::new();
        let quota = create_user_quota("test".to_string(), 2.0, 4.0);
        let quota_id = quota.id;
        
        manager.create_quota(quota).unwrap();
        
        let request = ResourceRequest {
            resource_type: ResourceType::Cpu,
            amount: 1.0,
            priority_class: PriorityClass::Normal,
            duration: Some(Duration::from_secs(60)),
            allow_burst: false,
            metadata: HashMap::new(),
        };
        
        let result = manager.check_quota(quota_id, &request);
        assert!(result.is_ok());
        assert!(result.unwrap().allowed);
    }
    
    #[test]
    fn test_quota_exceeded() {
        let manager = QuotaManager::new();
        let quota = create_user_quota("test".to_string(), 1.0, 1.0);
        let quota_id = quota.id;
        
        manager.create_quota(quota).unwrap();
        
        let request = ResourceRequest {
            resource_type: ResourceType::Cpu,
            amount: 2.0, // Exceeds 1.0 limit
            priority_class: PriorityClass::Normal,
            duration: Some(Duration::from_secs(60)),
            allow_burst: false,
            metadata: HashMap::new(),
        };
        
        let result = manager.check_quota(quota_id, &request);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QuotaError::QuotaExceeded { .. }));
    }
    
    #[test]
    fn test_allocation_deallocation() {
        let manager = QuotaManager::new();
        let quota = create_user_quota("test".to_string(), 4.0, 8.0);
        let quota_id = quota.id;
        
        manager.create_quota(quota).unwrap();
        
        let request = ResourceRequest {
            resource_type: ResourceType::Memory,
            amount: 2_000_000_000.0, // 2 GB
            priority_class: PriorityClass::Normal,
            duration: Some(Duration::from_secs(300)),
            allow_burst: false,
            metadata: HashMap::new(),
        };
        
        // Allocate
        let allocation_id = manager.allocate(quota_id, request).unwrap();
        
        // Check usage increased
        let usage = manager.get_usage(quota_id).unwrap();
        assert_eq!(usage.current_usage.get(&ResourceType::Memory), Some(&2_000_000_000.0));
        
        // Deallocate
        manager.deallocate(&allocation_id).unwrap();
        
        // Check usage decreased
        let usage = manager.get_usage(quota_id).unwrap();
        assert_eq!(usage.current_usage.get(&ResourceType::Memory), Some(&0.0));
    }
    
    #[test]
    fn test_burst_capacity() {
        let manager = QuotaManager::new();
        let quota = create_user_quota("test".to_string(), 1.0, 1.0);
        let quota_id = quota.id;
        
        manager.create_quota(quota).unwrap();
        
        let request = ResourceRequest {
            resource_type: ResourceType::Cpu,
            amount: 1.1, // Exceeds hard limit but within burst
            priority_class: PriorityClass::Normal,
            duration: Some(Duration::from_secs(30)),
            allow_burst: true,
            metadata: HashMap::new(),
        };
        
        let result = manager.check_quota(quota_id, &request);
        assert!(result.is_ok());
        
        let check_result = result.unwrap();
        assert!(check_result.allowed);
        assert!(check_result.used_burst);
    }
} 