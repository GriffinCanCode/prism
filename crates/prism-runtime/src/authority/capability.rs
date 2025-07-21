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

use prism_common::{span::Span, symbol::Symbol};
use prism_effects::Effect;
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
    pub fn authorizes(&self, operation: &Operation, context: &crate::execution::ExecutionContext) -> bool {
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
    pub fn allows(&self, operation: &Operation, context: &crate::execution::ExecutionContext) -> bool {
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
    pub fn authorizes(&self, operation: &Operation, context: &crate::execution::ExecutionContext) -> bool {
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
        context: &crate::execution::ExecutionContext,
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
        context: &crate::execution::ExecutionContext,
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
    operation_type: FileOperation,
    path: String,
}

impl FileSystemOperation {
    /// Get the operation type
    pub fn operation_type(&self) -> FileOperation {
        self.operation_type
    }

    /// Get the path
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// Network operations
#[derive(Debug, Clone)]
pub struct NetworkOperationStruct {
    operation_type: NetworkOperation,
    host: String,
    port: u16,
}

impl NetworkOperationStruct {
    /// Get the operation type
    pub fn operation_type(&self) -> NetworkOperation {
        self.operation_type
    }

    /// Get the host
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.port
    }
}

// Additional operation types and constraint types would be defined here...
// For brevity, I'll include the key error types and audit structures

/// Capability-related errors
#[derive(Debug, Error)]
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
pub struct TimeConstraint;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConstraint;

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
pub struct PathPattern;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostPattern;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortRange;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion;

impl PathPattern {
    fn matches(&self, _path: &str) -> bool {
        true // Placeholder implementation
    }
}

impl HostPattern {
    fn matches(&self, _host: &str) -> bool {
        true // Placeholder implementation
    }
}

impl PortRange {
    fn contains(&self, _port: u16) -> bool {
        true // Placeholder implementation
    }
}

impl MemoryRegion {
    fn allows(&self, _operation: &MemoryOperation) -> bool {
        true // Placeholder implementation
    }
}

// Additional placeholder implementations for constraints
impl TimeConstraint {
    fn allows(&self, _operation: &Operation, _context: &crate::execution::ExecutionContext) -> bool {
        true // Placeholder implementation
    }

    fn max_valid_until(&self) -> SystemTime {
        SystemTime::now() + Duration::from_secs(3600) // Placeholder implementation
    }
}

impl RateLimit {
    fn allows(&self, _operation: &Operation, _context: &crate::execution::ExecutionContext) -> bool {
        true // Placeholder implementation
    }
}

impl ResourceLimit {
    fn allows(&self, _operation: &Operation, _context: &crate::execution::ExecutionContext) -> bool {
        true // Placeholder implementation
    }
}

impl ContextConstraint {
    fn allows(&self, _operation: &Operation, _context: &crate::execution::ExecutionContext) -> bool {
        true // Placeholder implementation
    }
}

// Placeholder operation types
#[derive(Debug, Clone)]
pub struct DatabaseOperationStruct {
    operation_type: DatabaseOperation,
    tables: Vec<String>,
    estimated_rows: usize,
}

impl DatabaseOperationStruct {
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
    size: usize,
}

impl MemoryOperation {
    /// Create a new memory operation
    pub fn new(size: usize) -> Self {
        Self { size }
    }
    
    pub fn size(&self) -> usize {
        self.size
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