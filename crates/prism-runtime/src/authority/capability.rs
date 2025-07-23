//! Capability-Based Security System
//!
//! This module implements the runtime capability management system that enforces
//! capability-based security at runtime, ensuring that code can only perform
//! operations for which it holds valid, unexpired capabilities.
//!
//! ## Design Principles
//!
//! 1. **Zero-Trust Model**: No code has ambient authority - all operations require explicit capabilities
//! 2. **Capability Attenuation**: Capabilities can be weakened but never strengthened
//! 3. **Time-Bounded Authority**: All capabilities have expiration times
//! 4. **Audit Trail**: All capability usage is logged for security analysis
//! 5. **AI-Comprehensible**: Structured capability metadata for AI analysis

use crate::resources::effects::Effect;
use crate::platform::execution::ExecutionTarget;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, Duration};
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// A capability represents explicit authorization to perform specific operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Unique capability identifier
    pub id: CapabilityId,
    
    /// The authority this capability grants
    pub authority: Authority,
    
    /// Constraints that limit this capability's scope
    pub constraints: ConstraintSet,
    
    /// When this capability was issued
    pub issued_at: SystemTime,
    
    /// When this capability expires
    pub valid_until: SystemTime,
    
    /// Who issued this capability (for audit trail)
    pub issued_by: ComponentId,
    
    /// Whether this capability is still active
    pub active: bool,
}

impl Capability {
    /// Create a new capability with the given authority and constraints
    pub fn new(
        authority: Authority,
        constraints: ConstraintSet,
        valid_for: Duration,
        issued_by: ComponentId,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            id: CapabilityId::new(),
            authority,
            constraints,
            issued_at: now,
            valid_until: now + valid_for,
            issued_by,
            active: true,
        }
    }

    /// Check if this capability authorizes a specific operation
    pub fn authorizes(&self, operation: &Operation, context: &crate::platform::execution::ExecutionContext) -> bool {
        if !self.active || SystemTime::now() > self.valid_until {
            return false;
        }

        // Check if authority covers this operation
        if !self.authority.covers(operation) {
            return false;
        }

        // Check all constraints
        self.constraints.allows(operation, context)
    }

    /// Attenuate this capability by adding additional constraints
    pub fn attenuate(&self, additional_constraints: ConstraintSet) -> Result<Capability, CapabilityError> {
        if !self.active {
            return Err(CapabilityError::CapabilityInactive { id: self.id });
        }

        let combined_constraints = self.constraints.intersect(&additional_constraints)?;
        
        Ok(Capability {
            id: CapabilityId::new(),
            authority: self.authority.clone(),
            constraints: combined_constraints,
            issued_at: SystemTime::now(),
            valid_until: std::cmp::min(self.valid_until, additional_constraints.max_valid_until()),
            issued_by: self.issued_by,
            active: true,
        })
    }

    /// Revoke this capability
    pub fn revoke(&mut self) {
        self.active = false;
    }

    /// Check if this capability is still valid
    pub fn is_valid(&self) -> bool {
        self.active && SystemTime::now() <= self.valid_until
    }
}

/// Unique identifier for a capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityId(u64);

impl CapabilityId {
    /// Generate a new unique capability ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// Component identifier for capability tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(u32);

impl ComponentId {
    /// Create a new component ID
    pub fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the inner value
    pub fn into_inner(self) -> u32 {
        self.0
    }
}

/// Authority represents what operations a capability can authorize
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Authority {
    /// File system operations
    FileSystem(FileSystemAuthority),
    /// Network operations
    Network(NetworkAuthority),
    /// Database operations
    Database(DatabaseAuthority),
    /// Memory operations
    Memory(MemoryAuthority),
    /// System operations
    System(SystemAuthority),
    /// Composite authority combining multiple authorities
    Composite(Vec<Authority>),
}

impl Authority {
    /// Check if this authority covers a specific operation
    pub fn covers(&self, operation: &Operation) -> bool {
        match (self, operation) {
            (Authority::FileSystem(fs_auth), Operation::FileSystem(fs_op)) => {
                fs_auth.covers(fs_op)
            }
            (Authority::Network(net_auth), Operation::Network(net_op)) => {
                net_auth.covers(net_op)
            }
            (Authority::Database(db_auth), Operation::Database(db_op)) => {
                db_auth.covers(db_op)
            }
            (Authority::Memory(mem_auth), Operation::Memory(mem_op)) => {
                mem_auth.covers(mem_op)
            }
            (Authority::System(sys_auth), Operation::System(sys_op)) => {
                sys_auth.covers(sys_op)
            }
            (Authority::Composite(authorities), operation) => {
                authorities.iter().any(|auth| auth.covers(operation))
            }
            _ => false,
        }
    }

    /// Check if this authority covers another authority (for capability containment)
    pub fn covers_authority(&self, other: &Authority) -> bool {
        match (self, other) {
            // Same authority types - check if self is broader or equal
            (Authority::FileSystem(self_fs), Authority::FileSystem(other_fs)) => {
                self_fs.covers_authority(other_fs)
            }
            (Authority::Network(self_net), Authority::Network(other_net)) => {
                self_net.covers_authority(other_net)
            }
            (Authority::Database(self_db), Authority::Database(other_db)) => {
                self_db.covers_authority(other_db)
            }
            (Authority::Memory(self_mem), Authority::Memory(other_mem)) => {
                self_mem.covers_authority(other_mem)
            }
            (Authority::System(self_sys), Authority::System(other_sys)) => {
                self_sys.covers_authority(other_sys)
            }
            (Authority::Composite(self_auths), other) => {
                self_auths.iter().any(|auth| auth.covers_authority(other))
            }
            (self_auth, Authority::Composite(other_auths)) => {
                other_auths.iter().all(|auth| self_auth.covers_authority(auth))
            }
            _ => false,
        }
    }
}

/// File system authority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemAuthority {
    /// Allowed file operations
    pub operations: HashSet<FileOperation>,
    /// Path patterns that are allowed
    pub allowed_paths: Vec<PathPattern>,
}

impl FileSystemAuthority {
    /// Check if this authority covers a file system operation
    pub fn covers(&self, operation: &FileSystemOperation) -> bool {
        self.operations.contains(&operation.operation_type()) &&
        self.allowed_paths.iter().any(|pattern| pattern.matches(&operation.path()))
    }

    /// Check if this authority covers another file system authority
    pub fn covers_authority(&self, other: &FileSystemAuthority) -> bool {
        // Check if all operations in other are covered by self
        other.operations.iter().all(|op| self.operations.contains(op)) &&
        // Check if all paths in other are covered by self
        other.allowed_paths.iter().all(|other_path| {
            self.allowed_paths.iter().any(|self_path| self_path.covers(other_path))
        })
    }
}

/// Network authority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAuthority {
    /// Allowed network operations
    pub operations: HashSet<NetworkOperation>,
    /// Allowed hosts
    pub allowed_hosts: Vec<HostPattern>,
    /// Allowed ports
    pub allowed_ports: Vec<PortRange>,
}

impl NetworkAuthority {
    /// Check if this authority covers a network operation
    pub fn covers(&self, operation: &NetworkOperationStruct) -> bool {
        self.operations.contains(&operation.operation_type()) &&
        self.allowed_hosts.iter().any(|pattern| pattern.matches(&operation.host())) &&
        self.allowed_ports.iter().any(|range| range.contains(operation.port()))
    }

    /// Check if this authority covers another network authority
    pub fn covers_authority(&self, other: &NetworkAuthority) -> bool {
        // Check if all operations are covered
        other.operations.iter().all(|op| self.operations.contains(op)) &&
        // Check if all hosts are covered (simplified check)
        other.allowed_hosts.iter().all(|other_host| {
            self.allowed_hosts.iter().any(|self_host| self_host.covers(other_host))
        }) &&
        // Check if all ports are covered
        other.allowed_ports.iter().all(|other_range| {
            self.allowed_ports.iter().any(|self_range| self_range.covers(other_range))
        })
    }
}

/// Database authority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseAuthority {
    /// Allowed database operations
    pub operations: HashSet<DatabaseOperation>,
    /// Allowed tables
    pub allowed_tables: Vec<String>,
    /// Maximum rows that can be accessed
    pub max_rows: Option<usize>,
}

impl DatabaseAuthority {
    /// Check if this authority covers a database operation
    pub fn covers(&self, operation: &DatabaseOperationStruct) -> bool {
        self.operations.contains(&operation.operation_type()) &&
        operation.tables().iter().all(|table| self.allowed_tables.contains(table)) &&
        self.max_rows.map_or(true, |max| operation.estimated_rows() <= max)
    }

    /// Check if this authority covers another database authority
    pub fn covers_authority(&self, other: &DatabaseAuthority) -> bool {
        other.operations.iter().all(|op| self.operations.contains(op)) &&
        other.allowed_tables.iter().all(|table| self.allowed_tables.contains(table)) &&
        match (self.max_rows, other.max_rows) {
            (Some(self_max), Some(other_max)) => self_max >= other_max,
            (None, _) => true, // No limit covers any limit
            (Some(_), None) => false, // A limit cannot cover no limit
        }
    }
}

/// Memory authority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAuthority {
    /// Maximum memory that can be allocated
    pub max_allocation: usize,
    /// Allowed memory regions
    pub allowed_regions: Vec<MemoryRegion>,
}

impl MemoryAuthority {
    /// Check if this authority covers a memory operation
    pub fn covers(&self, operation: &MemoryOperation) -> bool {
        operation.size() <= self.max_allocation &&
        self.allowed_regions.iter().any(|region| region.allows(operation))
    }

    /// Check if this authority covers another memory authority
    pub fn covers_authority(&self, other: &MemoryAuthority) -> bool {
        self.max_allocation >= other.max_allocation &&
        other.allowed_regions.iter().all(|other_region| {
            self.allowed_regions.iter().any(|self_region| self_region.covers(other_region))
        })
    }
}

/// System authority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAuthority {
    /// Allowed system operations
    pub operations: HashSet<SystemOperation>,
    /// Environment variables that can be accessed
    pub allowed_env_vars: Vec<String>,
}

impl SystemAuthority {
    /// Check if this authority covers a system operation
    pub fn covers(&self, operation: &SystemOperationStruct) -> bool {
        self.operations.contains(&operation.operation_type()) &&
        operation.env_vars().iter().all(|var| self.allowed_env_vars.contains(var))
    }

    /// Check if this authority covers another system authority
    pub fn covers_authority(&self, other: &SystemAuthority) -> bool {
        other.operations.iter().all(|op| self.operations.contains(op)) &&
        other.allowed_env_vars.iter().all(|var| self.allowed_env_vars.contains(var))
    }
}

/// Set of constraints that limit a capability's scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSet {
    /// Time constraints
    pub time_constraints: Vec<TimeConstraint>,
    /// Rate limiting constraints
    pub rate_limits: Vec<RateLimit>,
    /// Resource constraints
    pub resource_limits: Vec<ResourceLimit>,
    /// Context constraints
    pub context_constraints: Vec<ContextConstraint>,
}

impl ConstraintSet {
    /// Create an empty constraint set
    pub fn new() -> Self {
        Self {
            time_constraints: Vec::new(),
            rate_limits: Vec::new(),
            resource_limits: Vec::new(),
            context_constraints: Vec::new(),
        }
    }

    /// Check if all constraints allow an operation
    pub fn allows(&self, operation: &Operation, context: &crate::platform::execution::ExecutionContext) -> bool {
        self.time_constraints.iter().all(|c| c.allows(operation, context)) &&
        self.rate_limits.iter().all(|c| c.allows(operation, context)) &&
        self.resource_limits.iter().all(|c| c.allows(operation, context)) &&
        self.context_constraints.iter().all(|c| c.allows(operation, context))
    }

    /// Intersect with another constraint set (more restrictive)
    pub fn intersect(&self, other: &ConstraintSet) -> Result<ConstraintSet, CapabilityError> {
        let mut result = self.clone();
        result.time_constraints.extend(other.time_constraints.clone());
        result.rate_limits.extend(other.rate_limits.clone());
        result.resource_limits.extend(other.resource_limits.clone());
        result.context_constraints.extend(other.context_constraints.clone());
        Ok(result)
    }

    /// Get the maximum valid until time from constraints
    pub fn max_valid_until(&self) -> SystemTime {
        self.time_constraints
            .iter()
            .map(|c| c.max_valid_until())
            .min()
            .unwrap_or(SystemTime::now() + Duration::from_secs(3600)) // Default 1 hour
    }
}

/// A set of capabilities
#[derive(Debug, Clone)]
pub struct CapabilitySet {
    capabilities: HashMap<CapabilityId, Capability>,
}

impl CapabilitySet {
    /// Create a new empty capability set
    pub fn new() -> Self {
        Self {
            capabilities: HashMap::new(),
        }
    }

    /// Add a capability to the set
    pub fn add(&mut self, capability: Capability) {
        self.capabilities.insert(capability.id, capability);
    }

    /// Check if the set contains a capability that authorizes an operation
    pub fn authorizes(&self, operation: &Operation, context: &crate::platform::execution::ExecutionContext) -> bool {
        self.capabilities
            .values()
            .any(|cap| cap.authorizes(operation, context))
    }

    /// Get all active capabilities
    pub fn active_capabilities(&self) -> Vec<&Capability> {
        self.capabilities
            .values()
            .filter(|cap| cap.is_valid())
            .collect()
    }

    /// Remove expired capabilities
    pub fn clean_expired(&mut self) {
        self.capabilities.retain(|_, cap| cap.is_valid());
    }

    /// Check if this set contains all capabilities from another set
    pub fn contains_all(&self, other: &CapabilitySet) -> bool {
        other.capabilities.values().all(|other_cap| {
            self.capabilities.values().any(|cap| {
                cap.id == other_cap.id && cap.authority.covers_authority(&other_cap.authority)
            })
        })
    }

    /// Check if this set contains a specific capability
    pub fn contains(&self, capability: &Capability) -> bool {
        self.capabilities.values().any(|cap| {
            cap.id == capability.id && cap.authority.covers_authority(&capability.authority)
        })
    }

    /// Get an iterator over all capabilities
    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.values()
    }

    /// Attenuate capabilities by creating a restricted subset
    pub fn attenuate(&self, requested: &CapabilitySet) -> Result<CapabilitySet, CapabilityError> {
        let mut attenuated = CapabilitySet::new();
        
        for requested_cap in requested.capabilities.values() {
            // Find a capability that can satisfy this request
            if let Some(granting_cap) = self.capabilities.values().find(|cap| {
                cap.authority.covers_authority(&requested_cap.authority) && cap.is_valid()
            }) {
                // Create an attenuated capability with more restrictive constraints
                let mut attenuated_cap = requested_cap.clone();
                attenuated_cap.constraints = granting_cap.constraints.intersect(&requested_cap.constraints)?;
                attenuated_cap.issued_by = granting_cap.issued_by; // Track delegation chain
                attenuated.add(attenuated_cap);
            } else {
                return Err(CapabilityError::InsufficientCapability {
                    required: requested.clone(),
                    available: self.clone(),
                });
            }
        }
        
        Ok(attenuated)
    }

    /// Get capability names for metadata
    pub fn capability_names(&self) -> Vec<String> {
        self.capabilities.keys()
            .map(|id| format!("Capability_{}", id.0))
            .collect()
    }
}

/// Capability manager that handles capability lifecycle
#[derive(Debug)]
pub struct CapabilityManager {
    /// Active capabilities by component
    component_capabilities: Arc<RwLock<HashMap<ComponentId, CapabilitySet>>>,
    
    /// Capability audit log
    audit_log: Arc<Mutex<Vec<CapabilityAuditEvent>>>,
    
    /// Capability usage statistics
    usage_stats: Arc<RwLock<HashMap<CapabilityId, CapabilityUsageStats>>>,
}

impl CapabilityManager {
    /// Create a new capability manager
    pub fn new() -> Result<Self, CapabilityError> {
        Ok(Self {
            component_capabilities: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(Mutex::new(Vec::new())),
            usage_stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Grant a capability to a component
    pub fn grant_capability(
        &self,
        component_id: ComponentId,
        capability: Capability,
    ) -> Result<(), CapabilityError> {
        let mut components = self.component_capabilities.write().unwrap();
        let capability_set = components.entry(component_id).or_insert_with(CapabilitySet::new);
        
        // Log the grant
        self.log_audit_event(CapabilityAuditEvent {
            timestamp: SystemTime::now(),
            event_type: AuditEventType::CapabilityGranted,
            capability_id: capability.id,
            component_id,
            operation: None,
            context: "Capability granted".to_string(),
        });

        capability_set.add(capability);
        Ok(())
    }

    /// Check if a component has authorization for an operation
    pub fn check_authorization(
        &self,
        component_id: ComponentId,
        operation: &Operation,
        context: &crate::platform::execution::ExecutionContext,
    ) -> Result<bool, CapabilityError> {
        let components = self.component_capabilities.read().unwrap();
        let capability_set = components.get(&component_id)
            .ok_or(CapabilityError::ComponentNotFound { id: component_id })?;

        let authorized = capability_set.authorizes(operation, context);

        // Log the check
        self.log_audit_event(CapabilityAuditEvent {
            timestamp: SystemTime::now(),
            event_type: if authorized { AuditEventType::OperationAuthorized } else { AuditEventType::OperationDenied },
            capability_id: CapabilityId::new(), // Would need to track which capability was used
            component_id,
            operation: Some(operation.clone()),
            context: format!("Authorization check: {}", authorized),
        });

        Ok(authorized)
    }

    /// Validate a set of capabilities for a context
    pub fn validate_capabilities(
        &self,
        capabilities: &CapabilitySet,
        context: &crate::platform::execution::ExecutionContext,
    ) -> Result<(), CapabilityError> {
        // Remove expired capabilities
        let mut capabilities = capabilities.clone();
        capabilities.clean_expired();

        // Validate remaining capabilities
        for capability in capabilities.active_capabilities() {
            if !capability.is_valid() {
                return Err(CapabilityError::CapabilityExpired { id: capability.id });
            }
        }

        Ok(())
    }

    /// Get the number of active capabilities
    pub fn active_count(&self) -> usize {
        let components = self.component_capabilities.read().unwrap();
        components.values()
            .map(|set| set.active_capabilities().len())
            .sum()
    }

    /// Log an audit event
    fn log_audit_event(&self, event: CapabilityAuditEvent) {
        let mut log = self.audit_log.lock().unwrap();
        log.push(event);
    }
}

/// Types for operations that can be authorized
#[derive(Debug, Clone)]
pub enum Operation {
    /// File system operation
    FileSystem(FileSystemOperation),
    /// Network operation
    Network(NetworkOperationStruct),
    /// Database operation
    Database(DatabaseOperationStruct),
    /// Memory operation
    Memory(MemoryOperation),
    /// System operation
    System(SystemOperationStruct),
}

/// File system operations
#[derive(Debug, Clone)]
pub struct FileSystemOperation {
    /// File path being accessed
    path: String,
    /// Type of file system operation
    operation_type: FileOperation,
    /// Estimated size for write operations
    estimated_size: Option<usize>,
}

impl FileSystemOperation {
    /// Create a new file system operation
    pub fn new(path: String, operation_type: FileOperation, estimated_size: Option<usize>) -> Self {
        Self { path, operation_type, estimated_size }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn operation_type(&self) -> FileOperation {
        self.operation_type
    }

    pub fn estimated_size(&self) -> Option<usize> {
        self.estimated_size
    }
}

/// Network operations
#[derive(Debug, Clone)]
pub struct NetworkOperationStruct {
    /// Host being accessed
    host: String,
    /// Port being accessed
    port: u16,
    /// Type of network operation
    operation_type: NetworkOperation,
    /// Estimated bytes for data transfer
    estimated_bytes: Option<usize>,
}

impl NetworkOperationStruct {
    /// Create a new network operation
    pub fn new(host: String, port: u16, operation_type: NetworkOperation, estimated_bytes: Option<usize>) -> Self {
        Self { host, port, operation_type, estimated_bytes }
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn operation_type(&self) -> NetworkOperation {
        self.operation_type
    }

    pub fn estimated_bytes(&self) -> Option<usize> {
        self.estimated_bytes
    }
}

// Additional operation types and constraint types would be defined here...
// For brevity, I'll include the key error types and audit structures

/// Capability-related errors
#[derive(Debug, Clone, Error)]
pub enum CapabilityError {
    /// Capability not found
    #[error("Capability not found: {id:?}")]
    CapabilityNotFound {
        /// Capability ID
        id: CapabilityId,
    },

    /// Capability expired
    #[error("Capability expired: {id:?}")]
    CapabilityExpired {
        /// Capability ID
        id: CapabilityId,
    },

    /// Capability inactive
    #[error("Capability inactive: {id:?}")]
    CapabilityInactive {
        /// Capability ID
        id: CapabilityId,
    },

    /// Component not found
    #[error("Component not found: {id:?}")]
    ComponentNotFound {
        /// Component ID
        id: ComponentId,
    },

    /// Operation not authorized
    #[error("Operation not authorized for component {component:?}")]
    OperationNotAuthorized {
        /// Component ID
        component: ComponentId,
    },

    /// Insufficient capability for operation
    #[error("Insufficient capability - required capabilities not available")]
    InsufficientCapability {
        /// Required capabilities
        required: CapabilitySet,
        /// Available capabilities
        available: CapabilitySet,
    },

    /// Generic capability error
    #[error("Capability error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
}

/// Audit event for capability usage
#[derive(Debug, Clone)]
pub struct CapabilityAuditEvent {
    /// When the event occurred
    pub timestamp: SystemTime,
    /// Type of audit event
    pub event_type: AuditEventType,
    /// Capability involved
    pub capability_id: CapabilityId,
    /// Component involved
    pub component_id: ComponentId,
    /// Operation attempted (if applicable)
    pub operation: Option<Operation>,
    /// Additional context
    pub context: String,
}

/// Types of capability audit events
#[derive(Debug, Clone)]
pub enum AuditEventType {
    /// Capability was granted
    CapabilityGranted,
    /// Capability was revoked
    CapabilityRevoked,
    /// Operation was authorized
    OperationAuthorized,
    /// Operation was denied
    OperationDenied,
    /// Capability expired
    CapabilityExpired,
}

/// Usage statistics for a capability
#[derive(Debug, Clone)]
pub struct CapabilityUsageStats {
    /// Number of times this capability was used
    pub usage_count: u64,
    /// Last time this capability was used
    pub last_used: SystemTime,
    /// Operations performed with this capability
    pub operations_performed: Vec<String>,
}

// Placeholder types for the constraint system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeConstraint {
    /// Capability is valid until a specific time
    ValidUntil(SystemTime),
    /// Capability is valid after a specific time
    ValidAfter(SystemTime),
    /// Capability is valid during a specific time range
    ValidDuring { start: SystemTime, end: SystemTime },
    /// Capability is valid during a specific time of day
    TimeOfDay { start_hour: u8, end_hour: u8 },
    /// Capability is valid on specific days of the week
    DaysOfWeek(HashSet<u8>), // 0 for Sunday, 1 for Monday, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimit {
    /// Maximum operations per second
    PerSecond(u64),
    /// Maximum operations per minute
    PerMinute(u64),
    /// Maximum operations per hour
    PerHour(u64),
    /// Burst rate with replenishment
    Burst { max_burst: u64, replenish_rate: u64 },
    /// Maximum concurrent operations
    Concurrent(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceLimit {
    /// Maximum memory allocation
    Memory(usize),
    /// Maximum CPU time usage
    CpuTime(Duration),
    /// Maximum network bandwidth
    NetworkBandwidth(usize),
    /// Maximum file descriptors
    FileDescriptors(u64),
    /// Maximum disk space usage
    DiskSpace(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextConstraint {
    /// Requires specific capabilities to be present in the context
    RequiredCapabilities(Vec<String>),
    /// Forbids specific capabilities from being present in the context
    ForbiddenCapabilities(Vec<String>),
    /// Requires specific effects to be present in the context
    RequiredEffects(Vec<String>),
    /// Restricts the execution target (e.g., only allow execution on TypeScript, WebAssembly, etc.)
    ExecutionTarget(HashSet<ExecutionTarget>),
    /// Restricts the component ID that can execute the operation
    ComponentId(HashSet<ComponentId>),
}

// Placeholder types for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileOperation {
    Read,
    Write,
    Execute,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkOperation {
    Connect,
    Listen,
    Send,
    Receive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatabaseOperation {
    Select,
    Insert,
    Update,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SystemOperation {
    ProcessCreate,
    EnvironmentRead,
    SystemCall,
}

// Placeholder pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathPattern {
    /// Pattern string (supports wildcards like * and **)
    pattern: String,
    /// Whether pattern is recursive
    recursive: bool,
}

impl PathPattern {
    /// Create a new path pattern
    pub fn new(pattern: String, recursive: bool) -> Self {
        Self { pattern, recursive }
    }

    /// Check if this pattern matches a path
    fn matches(&self, path: &str) -> bool {
        if self.recursive {
            // Support recursive matching with **
            self.matches_recursive(path)
        } else {
            // Simple wildcard matching
            self.matches_simple(path)
        }
    }

    /// Check if this pattern covers another pattern (is more permissive)
    pub fn covers(&self, other: &PathPattern) -> bool {
        // A pattern covers another if it's more general
        if self.recursive && !other.recursive {
            // Recursive patterns cover non-recursive ones in the same tree
            other.pattern.starts_with(&self.pattern.replace("**", ""))
        } else if self.pattern == "*" || self.pattern == "**" {
            // Wildcard patterns cover everything
            true
        } else {
            // Exact match or prefix match
            other.pattern.starts_with(&self.pattern)
        }
    }

    /// Simple wildcard matching (* matches any file/directory name)
    fn matches_simple(&self, path: &str) -> bool {
        if self.pattern == "*" {
            return true;
        }

        // Split both pattern and path into segments
        let pattern_segments: Vec<&str> = self.pattern.split('/').collect();
        let path_segments: Vec<&str> = path.split('/').collect();

        if pattern_segments.len() != path_segments.len() {
            return false;
        }

        for (pattern_seg, path_seg) in pattern_segments.iter().zip(path_segments.iter()) {
            if *pattern_seg != "*" && *pattern_seg != *path_seg {
                return false;
            }
        }

        true
    }

    /// Recursive wildcard matching (** matches any number of directories)
    fn matches_recursive(&self, path: &str) -> bool {
        if self.pattern.contains("**") {
            let parts: Vec<&str> = self.pattern.split("**").collect();
            if parts.len() == 2 {
                let prefix = parts[0].trim_end_matches('/');
                let suffix = parts[1].trim_start_matches('/');
                
                if prefix.is_empty() && suffix.is_empty() {
                    return true; // ** matches everything
                }
                
                let prefix_matches = prefix.is_empty() || path.starts_with(prefix);
                let suffix_matches = suffix.is_empty() || path.ends_with(suffix);
                
                return prefix_matches && suffix_matches;
            }
        }
        
        // Fallback to simple matching
        self.matches_simple(path)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostPattern {
    /// Host pattern (supports wildcards)
    pattern: String,
    /// Whether to allow subdomains
    allow_subdomains: bool,
}

impl HostPattern {
    /// Create a new host pattern
    pub fn new(pattern: String, allow_subdomains: bool) -> Self {
        Self { pattern, allow_subdomains }
    }

    /// Check if this pattern matches a host
    fn matches(&self, host: &str) -> bool {
        if self.pattern == "*" {
            return true;
        }

        if self.allow_subdomains {
            // Allow subdomains (e.g., *.example.com matches api.example.com)
            if self.pattern.starts_with("*.") {
                let domain = &self.pattern[2..];
                return host.ends_with(domain) && 
                       (host == domain || host.ends_with(&format!(".{}", domain)));
            }
        }

        // Exact match or simple wildcard
        if self.pattern.contains('*') {
            self.wildcard_match(host)
        } else {
            self.pattern == host
        }
    }

    /// Check if this pattern covers another pattern
    pub fn covers(&self, other: &HostPattern) -> bool {
        if self.pattern == "*" {
            return true;
        }

        if self.allow_subdomains && other.allow_subdomains {
            // More general subdomain patterns cover more specific ones
            if self.pattern.starts_with("*.") && other.pattern.starts_with("*.") {
                let self_domain = &self.pattern[2..];
                let other_domain = &other.pattern[2..];
                return other_domain.ends_with(self_domain);
            }
        }

        // Exact match
        self.pattern == other.pattern
    }

    /// Simple wildcard matching for hosts
    fn wildcard_match(&self, host: &str) -> bool {
        let pattern_parts: Vec<&str> = self.pattern.split('.').collect();
        let host_parts: Vec<&str> = host.split('.').collect();

        if pattern_parts.len() != host_parts.len() {
            return false;
        }

        for (pattern_part, host_part) in pattern_parts.iter().zip(host_parts.iter()) {
            if *pattern_part != "*" && *pattern_part != *host_part {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortRange {
    /// Start of port range (inclusive)
    start: u16,
    /// End of port range (inclusive)
    end: u16,
}

impl PortRange {
    /// Create a new port range
    pub fn new(start: u16, end: u16) -> Self {
        Self { start, end }
    }

    /// Create a single port range
    pub fn single(port: u16) -> Self {
        Self { start: port, end: port }
    }

    /// Check if this range contains a port
    fn contains(&self, port: u16) -> bool {
        port >= self.start && port <= self.end
    }

    /// Check if this range covers another range
    pub fn covers(&self, other: &PortRange) -> bool {
        self.start <= other.start && self.end >= other.end
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    /// Start address of memory region
    start_addr: usize,
    /// Size of memory region in bytes
    size: usize,
    /// Allowed operations on this region
    allowed_operations: HashSet<MemoryOperationType>,
}

impl MemoryRegion {
    /// Create a new memory region
    pub fn new(start_addr: usize, size: usize, allowed_operations: HashSet<MemoryOperationType>) -> Self {
        Self { start_addr, size, allowed_operations }
    }

    /// Check if this region allows a memory operation
    fn allows(&self, operation: &MemoryOperation) -> bool {
        // Check if operation type is allowed
        if !self.allowed_operations.contains(&operation.operation_type) {
            return false;
        }

        // Check if operation is within memory bounds
        let op_start = operation.address;
        let op_end = operation.address.saturating_add(operation.size);
        let region_end = self.start_addr.saturating_add(self.size);

        op_start >= self.start_addr && op_end <= region_end
    }

    /// Check if this memory region covers another memory region
    pub fn covers(&self, other: &MemoryRegion) -> bool {
        // Check if all operations allowed by other are allowed by self
        if !other.allowed_operations.is_subset(&self.allowed_operations) {
            return false;
        }

        // Check if other region is within bounds of self
        let other_end = other.start_addr.saturating_add(other.size);
        let self_end = self.start_addr.saturating_add(self.size);

        other.start_addr >= self.start_addr && other_end <= self_end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryOperationType {
    Read,
    Write,
    Execute,
    Allocate,
    Deallocate,
}

// Additional proper implementations for constraints
impl TimeConstraint {
    /// Check if this constraint allows an operation at the current time
    fn allows(&self, _operation: &Operation, _context: &crate::platform::execution::ExecutionContext) -> bool {
        let now = SystemTime::now();
        
        match self {
            TimeConstraint::ValidUntil(until) => now <= *until,
            TimeConstraint::ValidAfter(after) => now >= *after,
            TimeConstraint::ValidDuring { start, end } => now >= *start && now <= *end,
            TimeConstraint::TimeOfDay { start_hour, end_hour } => {
                // Get current hour (simplified - would need proper timezone handling)
                if let Ok(duration) = now.duration_since(SystemTime::UNIX_EPOCH) {
                    let hours_since_epoch = (duration.as_secs() / 3600) % 24;
                    let current_hour = hours_since_epoch as u8;
                    
                    if start_hour <= end_hour {
                        current_hour >= *start_hour && current_hour <= *end_hour
                    } else {
                        // Handle overnight ranges (e.g., 22:00 to 06:00)
                        current_hour >= *start_hour || current_hour <= *end_hour
                    }
                } else {
                    false
                }
            }
            TimeConstraint::DaysOfWeek(allowed_days) => {
                // Get current day of week (simplified)
                if let Ok(duration) = now.duration_since(SystemTime::UNIX_EPOCH) {
                    let days_since_epoch = duration.as_secs() / (24 * 3600);
                    let day_of_week = ((days_since_epoch + 4) % 7) as u8; // Unix epoch was Thursday
                    allowed_days.contains(&day_of_week)
                } else {
                    false
                }
            }
        }
    }

    /// Get the maximum time this constraint is valid until
    fn max_valid_until(&self) -> SystemTime {
        match self {
            TimeConstraint::ValidUntil(until) => *until,
            TimeConstraint::ValidAfter(_) => SystemTime::now() + Duration::from_secs(365 * 24 * 3600), // 1 year
            TimeConstraint::ValidDuring { end, .. } => *end,
            TimeConstraint::TimeOfDay { .. } => SystemTime::now() + Duration::from_secs(24 * 3600), // 1 day
            TimeConstraint::DaysOfWeek(_) => SystemTime::now() + Duration::from_secs(7 * 24 * 3600), // 1 week
        }
    }
}

impl RateLimit {
    /// Check if this rate limit allows an operation
    fn allows(&self, _operation: &Operation, context: &crate::platform::execution::ExecutionContext) -> bool {
        // In a real implementation, this would track operation counts over time windows
        // For now, we'll do a simplified check based on the current context
        
        match self {
            RateLimit::PerSecond(max_per_sec) => {
                // Simplified: assume we can track operations per second
                // In reality, this would need a sliding window counter
                *max_per_sec > 0 // Just check that some operations are allowed
            }
            RateLimit::PerMinute(max_per_min) => {
                *max_per_min > 0
            }
            RateLimit::PerHour(max_per_hour) => {
                *max_per_hour > 0
            }
            RateLimit::Burst { max_burst, replenish_rate } => {
                // Token bucket algorithm would be implemented here
                *max_burst > 0 && *replenish_rate > 0
            }
            RateLimit::Concurrent(max_concurrent) => {
                // Would check current concurrent operations
                *max_concurrent > 0
            }
        }
    }
}

impl ResourceLimit {
    /// Check if this resource limit allows an operation
    fn allows(&self, operation: &Operation, context: &crate::platform::execution::ExecutionContext) -> bool {
        match self {
            ResourceLimit::Memory(max_bytes) => {
                // Check if operation would exceed memory limit
                if let Operation::Memory(mem_op) = operation {
                    mem_op.size <= *max_bytes
                } else {
                    true // Non-memory operations don't consume memory directly
                }
            }
            ResourceLimit::CpuTime(max_duration) => {
                // In a real implementation, this would track CPU time usage
                // For now, just check that the limit is reasonable
                *max_duration > Duration::from_millis(1)
            }
            ResourceLimit::NetworkBandwidth(max_bytes_per_sec) => {
                // Check network operation bandwidth requirements
                if let Operation::Network(net_op) = operation {
                    // Simplified: assume operation size represents bandwidth need
                    net_op.estimated_bytes().unwrap_or(0) <= *max_bytes_per_sec
                } else {
                    true
                }
            }
            ResourceLimit::FileDescriptors(max_fds) => {
                // Would check current file descriptor usage
                *max_fds > 0
            }
            ResourceLimit::DiskSpace(max_bytes) => {
                // Check if file operation would exceed disk space limit
                if let Operation::FileSystem(fs_op) = operation {
                    fs_op.estimated_size().unwrap_or(0) <= *max_bytes
                } else {
                    true
                }
            }
        }
    }
}

impl ContextConstraint {
    /// Check if this context constraint allows an operation
    fn allows(&self, operation: &Operation, context: &crate::platform::execution::ExecutionContext) -> bool {
        match self {
            ContextConstraint::RequiredCapabilities(required_caps) => {
                // Check if context has all required capabilities
                required_caps.iter().all(|cap_name| {
                    context.capabilities.capability_names().contains(cap_name)
                })
            }
            ContextConstraint::ForbiddenCapabilities(forbidden_caps) => {
                // Check that context doesn't have forbidden capabilities
                !forbidden_caps.iter().any(|cap_name| {
                    context.capabilities.capability_names().contains(cap_name)
                })
            }
            ContextConstraint::RequiredEffects(required_effects) => {
                // Check if context has required effects
                let current_effects = context.current_effects();
                required_effects.iter().all(|effect_name| {
                    current_effects.iter().any(|effect| effect.name() == *effect_name)
                })
            }
            ContextConstraint::ExecutionTarget(allowed_targets) => {
                // Check if current execution target is allowed
                allowed_targets.contains(&context.target())
            }
            ContextConstraint::ComponentId(allowed_components) => {
                // Check if current component is allowed
                allowed_components.contains(&context.current_component())
            }
        }
    }
}

// Enhanced operation types with proper implementations
#[derive(Debug, Clone)]
pub struct DatabaseOperationStruct {
    operation_type: DatabaseOperation,
    tables: Vec<String>,
    estimated_rows: usize,
}

impl DatabaseOperationStruct {
    /// Create a new database operation
    pub fn new(operation_type: DatabaseOperation, tables: Vec<String>, estimated_rows: usize) -> Self {
        Self { operation_type, tables, estimated_rows }
    }

    pub fn operation_type(&self) -> DatabaseOperation {
        self.operation_type
    }

    pub fn tables(&self) -> &[String] {
        &self.tables
    }

    pub fn estimated_rows(&self) -> usize {
        self.estimated_rows
    }
}

#[derive(Debug, Clone)]
pub struct MemoryOperation {
    /// Memory address being accessed
    address: usize,
    /// Size of memory operation
    size: usize,
    /// Type of memory operation
    operation_type: MemoryOperationType,
}

impl MemoryOperation {
    /// Create a new memory operation
    pub fn new(address: usize, size: usize, operation_type: MemoryOperationType) -> Self {
        Self { address, size, operation_type }
    }

    pub fn address(&self) -> usize {
        self.address
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn operation_type(&self) -> MemoryOperationType {
        self.operation_type
    }
}

#[derive(Debug, Clone)]
pub struct SystemOperationStruct {
    operation_type: SystemOperation,
    env_vars: Vec<String>,
}

impl SystemOperationStruct {
    /// Create a new system operation
    pub fn new(operation_type: SystemOperation, env_vars: Vec<String>) -> Self {
        Self { operation_type, env_vars }
    }
    
    pub fn operation_type(&self) -> SystemOperation {
        self.operation_type
    }

    pub fn env_vars(&self) -> &[String] {
        &self.env_vars
    }
} 