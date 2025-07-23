//! Component Isolation - Focused Implementation  
//!
//! This module implements component isolation following the same modular
//! philosophy as other runtime components. It focuses solely on managing
//! component boundaries and secure communication without duplicating logic.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicUsize, Ordering}};
use std::time::{SystemTime, Duration};
use thiserror::Error;
use uuid::Uuid;

/// Simple isolation manager that handles component boundaries
#[derive(Debug)]
pub struct IsolationManager {
    /// Active isolated components
    components: Arc<RwLock<HashMap<ComponentId, IsolatedComponent>>>,
    /// Isolation statistics
    stats: Arc<RwLock<IsolationStats>>,
    /// Manager start time
    started_at: SystemTime,
}

impl IsolationManager {
    /// Create a new isolation manager
    pub fn new() -> Result<Self, IsolationError> {
        Ok(Self {
            components: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(IsolationStats::default())),
            started_at: SystemTime::now(),
        })
    }

    /// Create a new isolated component
    pub fn create_component(&self, spec: ComponentSpec) -> Result<ComponentHandle, IsolationError> {
        let component_id = ComponentId::new();
        
        // Validate component specification
        self.validate_component_spec(&spec)?;

        // Create isolated component
        let component = IsolatedComponent {
            id: component_id,
            spec: spec.clone(),
            state: ComponentState::Created,
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
        };

        // Store component
        {
            let mut components = self.components.write().unwrap();
            components.insert(component_id, component);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_components_created += 1;
            stats.active_components += 1;
        }

        Ok(ComponentHandle {
            component_id,
            name: spec.name.clone(),
            created_at: SystemTime::now(),
        })
    }

    /// Get the number of active components
    pub fn component_count(&self) -> usize {
        self.components.read().unwrap().len()
    }

    /// Get isolation status
    pub fn get_isolation_status(&self) -> IsolationStatus {
        let components = self.components.read().unwrap();
        let stats = self.stats.read().unwrap();
        
        IsolationStatus {
            active_components: components.len(),
            total_created: stats.total_components_created,
            isolation_violations: stats.isolation_violations,
            uptime: self.started_at.elapsed().unwrap_or_default(),
        }
    }

    /// Remove a component
    pub fn remove_component(&self, component_id: ComponentId) -> Result<(), IsolationError> {
        let mut components = self.components.write().unwrap();
        if components.remove(&component_id).is_some() {
            let mut stats = self.stats.write().unwrap();
            stats.active_components = stats.active_components.saturating_sub(1);
            Ok(())
        } else {
            Err(IsolationError::ComponentNotFound { component_id })
        }
    }

    /// Validate component specification
    fn validate_component_spec(&self, spec: &ComponentSpec) -> Result<(), IsolationError> {
        if spec.name.is_empty() {
            return Err(IsolationError::InvalidSpec {
                reason: "Component name cannot be empty".to_string(),
            });
        }

        if spec.name.len() > 255 {
            return Err(IsolationError::InvalidSpec {
                reason: "Component name too long (max 255 characters)".to_string(),
            });
        }

        Ok(())
    }
}

/// Unique identifier for components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(u64);

impl ComponentId {
    /// Generate a new unique component ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the inner ID value
    pub fn into_inner(self) -> u64 {
        self.0
    }
}

/// Handle for managing a component
#[derive(Debug, Clone)]
pub struct ComponentHandle {
    /// Component identifier
    pub component_id: ComponentId,
    /// Component name
    pub name: String,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Specification for creating a component
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Component name
    pub name: String,
    /// Component description
    pub description: String,
    /// Isolation level required
    pub isolation_level: IsolationLevel,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Communication policy
    pub communication_policy: CommunicationPolicy,
    /// Security domain
    pub security_domain: SecurityDomain,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
}

/// Levels of component isolation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// No isolation (for testing)
    None,
    /// Process-level isolation
    Process,
    /// Container-level isolation
    Container,
    /// VM-level isolation
    VirtualMachine,
    /// Hardware-level isolation
    Hardware,
}

/// Resource limits for components
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    /// Maximum CPU percentage (0.0-1.0)
    pub max_cpu_percent: f64,
    /// Maximum I/O operations per second
    pub max_io_ops_per_sec: u64,
    /// Maximum network bandwidth in bytes per second
    pub max_network_bandwidth: u64,
    /// Maximum execution time
    pub max_execution_time: Duration,
}

/// Communication policy for components
#[derive(Debug, Clone)]
pub struct CommunicationPolicy {
    /// Allowed communication patterns
    pub allowed_patterns: Vec<CommunicationPattern>,
    /// Message size limits
    pub max_message_size: usize,
    /// Message rate limits
    pub max_messages_per_second: u64,
    /// Allowed message types
    pub allowed_message_types: Vec<MessageType>,
}

/// Communication patterns between components
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommunicationPattern {
    /// One-way communication
    OneWay,
    /// Request-response pattern
    RequestResponse,
    /// Publish-subscribe pattern
    PublishSubscribe,
    /// Event streaming
    EventStreaming,
}

/// Message types for component communication
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    /// Control message
    Control,
    /// Data message
    Data,
    /// Event message
    Event,
    /// Request message
    Request,
    /// Response message
    Response,
}

/// Security domain for component isolation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityDomain {
    /// Domain name
    pub name: String,
    /// Security level
    pub level: SecurityLevel,
    /// Trust boundaries
    pub trust_boundaries: Vec<TrustBoundary>,
}

/// Security levels for domains
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Public domain
    Public,
    /// Internal domain
    Internal,
    /// Confidential domain
    Confidential,
    /// Restricted domain
    Restricted,
    /// Top secret domain
    TopSecret,
}

/// Trust boundaries between domains
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustBoundary {
    /// Boundary name
    pub name: String,
    /// Required security controls
    pub required_controls: Vec<SecurityControl>,
}

/// Security controls for trust boundaries
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityControl {
    /// Encryption in transit
    EncryptionInTransit,
    /// Mutual authentication
    MutualAuthentication,
    /// Message signing
    MessageSigning,
    /// Access logging
    AccessLogging,
    /// Rate limiting
    RateLimiting,
}

/// Isolated component with its execution context
#[derive(Debug, Clone)]
pub struct IsolatedComponent {
    /// Component identifier
    pub id: ComponentId,
    /// Component specification
    pub spec: ComponentSpec,
    /// Current state
    pub state: ComponentState,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last activity timestamp
    pub last_activity: SystemTime,
}

/// States of a component
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentState {
    /// Component created but not started
    Created,
    /// Component is starting up
    Starting,
    /// Component is running
    Running,
    /// Component is paused
    Paused,
    /// Component is stopping
    Stopping,
    /// Component has stopped
    Stopped,
    /// Component has failed
    Failed,
}

/// Isolation system statistics
#[derive(Debug, Clone, Default)]
pub struct IsolationStats {
    /// Total components created
    pub total_components_created: usize,
    /// Currently active components
    pub active_components: usize,
    /// Isolation violations detected
    pub isolation_violations: u64,
}

/// Current isolation status
#[derive(Debug, Clone)]
pub struct IsolationStatus {
    /// Number of active components
    pub active_components: usize,
    /// Total components created since start
    pub total_created: usize,
    /// Number of isolation violations
    pub isolation_violations: u64,
    /// How long isolation manager has been running
    pub uptime: Duration,
}

/// Isolation management errors
#[derive(Debug, Error)]
pub enum IsolationError {
    /// Component not found
    #[error("Component not found: {component_id:?}")]
    ComponentNotFound { component_id: ComponentId },

    /// Invalid component specification
    #[error("Invalid component specification: {reason}")]
    InvalidSpec { reason: String },

    /// Resource allocation failed
    #[error("Resource allocation failed: {reason}")]
    ResourceAllocation { reason: String },

    /// Communication setup failed
    #[error("Communication setup failed: {reason}")]
    Communication { reason: String },

    /// Generic isolation error
    #[error("Isolation error: {message}")]
    Generic { message: String },
} 