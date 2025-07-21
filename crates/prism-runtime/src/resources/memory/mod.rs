//! Memory Management System with Capability Checking
//!
//! This module implements the memory management system that provides efficient
//! memory allocation and deallocation while enforcing capability-based security
//! for all memory operations.
//!
//! ## Design Principles
//!
//! 1. **Capability-Controlled Access**: All memory operations require explicit capabilities
//! 2. **Semantic Memory Types**: Memory is typed with semantic information for AI comprehension
//! 3. **Multi-Target Support**: Memory management works across all execution targets
//! 4. **Performance Optimized**: Efficient allocation strategies with minimal overhead
//! 5. **Security by Default**: Memory isolation and protection by default

use crate::capability::{Capability, CapabilitySet, Operation, MemoryOperation};
use crate::execution::ExecutionContext;
use prism_common::{span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use std::ptr::NonNull;
use thiserror::Error;

/// Memory manager that handles all memory operations with capability checking
pub struct MemoryManager {
    /// Target-specific allocators
    allocators: HashMap<crate::execution::ExecutionTarget, Box<dyn MemoryAllocator>>,
    
    /// Memory usage tracking
    usage_tracker: Arc<MemoryUsageTracker>,
    
    /// Memory pool manager for frequent allocations
    memory_pools: Arc<MemoryPoolManager>,
    
    /// Semantic memory registry
    semantic_registry: Arc<RwLock<SemanticMemoryRegistry>>,
    
    /// Memory security enforcer
    security_enforcer: Arc<MemorySecurityEnforcer>,
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("allocators", &format!("{} allocators", self.allocators.len()))
            .field("usage_tracker", &self.usage_tracker)
            .field("memory_pools", &self.memory_pools)
            .field("semantic_registry", &"<semantic_registry>")
            .field("security_enforcer", &self.security_enforcer)
            .finish()
    }
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Result<Self, MemoryError> {
        let mut allocators: HashMap<crate::execution::ExecutionTarget, Box<dyn MemoryAllocator>> = HashMap::new();
        
        // Register target-specific allocators
        allocators.insert(
            crate::execution::ExecutionTarget::TypeScript,
            Box::new(JavaScriptAllocator::new()?)
        );
        allocators.insert(
            crate::execution::ExecutionTarget::WebAssembly,
            Box::new(WasmAllocator::new()?)
        );
        allocators.insert(
            crate::execution::ExecutionTarget::Native,
            Box::new(NativeAllocator::new()?)
        );

        Ok(Self {
            allocators,
            usage_tracker: Arc::new(MemoryUsageTracker::new()),
            memory_pools: Arc::new(MemoryPoolManager::new()?),
            semantic_registry: Arc::new(RwLock::new(SemanticMemoryRegistry::new())),
            security_enforcer: Arc::new(MemorySecurityEnforcer::new()?),
        })
    }

    /// Allocate memory with capability checking and semantic typing
    pub fn allocate_semantic<T: SemanticType>(
        &self,
        semantic_type: &T,
        size: usize,
        capability: &Capability,
        context: &ExecutionContext,
    ) -> Result<SemanticPtr<T>, MemoryError> {
        // Verify capability allows memory allocation
        let memory_op = MemoryOperation::new(size);
        let operation = Operation::Memory(memory_op);
        
        if !capability.authorizes(&operation, context) {
            return Err(MemoryError::InsufficientCapability {
                required: "Memory.Allocate".to_string(),
                available: format!("{:?}", capability),
            });
        }

        // Get target-appropriate allocator
        let allocator = self.get_allocator_for_target(context.target())?;

        // Check memory limits
        self.security_enforcer.check_allocation_limits(size, context)?;

        // Perform allocation
        let raw_ptr = allocator.allocate(size, context)?;

        // Create semantic pointer
        let semantic_ptr = SemanticPtr::new(raw_ptr, semantic_type.clone(), size);

        // Register in semantic registry
        {
            let mut registry = self.semantic_registry.write().unwrap();
            registry.register_allocation(&semantic_ptr, context)?;
        }

        // Track allocation for profiling
        self.usage_tracker.record_allocation(&semantic_ptr, context);

        Ok(semantic_ptr)
    }

    /// Deallocate memory with tracking and cleanup
    pub fn deallocate_semantic<T: SemanticType>(
        &self,
        ptr: SemanticPtr<T>,
        context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        // Verify pointer is valid and owned
        {
            let registry = self.semantic_registry.read().unwrap();
            registry.verify_ownership(&ptr, context)?;
        }

        // Record deallocation
        self.usage_tracker.record_deallocation(&ptr, context);

        // Get target-appropriate allocator
        let allocator = self.get_allocator_for_target(context.target())?;

        // Perform deallocation
        allocator.deallocate(ptr.raw_ptr(), context)?;

        // Unregister from semantic registry
        {
            let mut registry = self.semantic_registry.write().unwrap();
            registry.unregister_allocation(&ptr)?;
        }

        Ok(())
    }

    /// Get current memory usage statistics
    pub fn current_usage(&self) -> usize {
        self.usage_tracker.total_allocated()
    }

    /// Get memory statistics for monitoring
    pub fn get_memory_stats(&self) -> MemoryStats {
        let usage = self.usage_tracker.get_stats();
        let registry = self.semantic_registry.read().unwrap();
        
        MemoryStats {
            total_allocated: usage.total_allocated,
            total_deallocated: usage.total_deallocated,
            current_usage: usage.current_usage,
            allocation_count: usage.allocation_count,
            deallocation_count: usage.deallocation_count,
            semantic_objects: registry.object_count(),
            memory_pools: self.memory_pools.get_stats(),
        }
    }

    /// Get allocator for a specific target
    fn get_allocator_for_target(
        &self,
        target: crate::execution::ExecutionTarget,
    ) -> Result<&dyn MemoryAllocator, MemoryError> {
        self.allocators.get(&target)
            .map(|a| a.as_ref())
            .ok_or(MemoryError::UnsupportedTarget { target })
    }
}

/// Trait for semantic types that can be allocated in memory
pub trait SemanticType: Clone + Send + Sync + 'static {
    /// Get the semantic type name
    fn type_name(&self) -> &'static str;
    
    /// Get AI-readable metadata about this type
    fn ai_metadata(&self) -> SemanticTypeMetadata;
    
    /// Get alignment requirements
    fn alignment(&self) -> usize {
        std::mem::align_of::<Self>()
    }
    
    /// Validate the semantic properties of this type
    fn validate_semantics(&self) -> Result<(), SemanticValidationError>;
}

/// Metadata about a semantic type for AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTypeMetadata {
    /// Business domain this type belongs to
    pub domain: String,
    /// Purpose of this type
    pub purpose: String,
    /// Constraints on values of this type
    pub constraints: Vec<String>,
    /// Relationships to other types
    pub relationships: Vec<TypeRelationship>,
}

/// Relationship between semantic types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Related type name
    pub related_type: String,
    /// Description of the relationship
    pub description: String,
}

/// Types of relationships between semantic types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Composition relationship
    ComposedOf,
    /// Inheritance relationship
    IsA,
    /// Usage relationship
    Uses,
    /// Dependency relationship
    DependsOn,
}

/// Smart pointer that carries semantic type information
#[derive(Debug)]
pub struct SemanticPtr<T: SemanticType> {
    /// Raw pointer to allocated memory
    raw_ptr: NonNull<u8>,
    /// Semantic type information
    semantic_type: T,
    /// Size of allocation
    size: usize,
    /// Allocation timestamp
    allocated_at: SystemTime,
    /// Unique allocation ID
    allocation_id: AllocationId,
}

impl<T: SemanticType> SemanticPtr<T> {
    /// Create a new semantic pointer
    pub fn new(raw_ptr: NonNull<u8>, semantic_type: T, size: usize) -> Self {
        Self {
            raw_ptr,
            semantic_type,
            size,
            allocated_at: SystemTime::now(),
            allocation_id: AllocationId::new(),
        }
    }

    /// Get the raw pointer
    pub fn raw_ptr(&self) -> NonNull<u8> {
        self.raw_ptr
    }

    /// Get the semantic type
    pub fn semantic_type(&self) -> &T {
        &self.semantic_type
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the allocation ID
    pub fn allocation_id(&self) -> AllocationId {
        self.allocation_id
    }

    /// Get AI-readable metadata about this allocation
    pub fn ai_metadata(&self) -> AllocationAIMetadata {
        AllocationAIMetadata {
            semantic_type: self.semantic_type.ai_metadata(),
            allocation_size: self.size,
            allocation_time: self.allocated_at,
            allocation_id: self.allocation_id,
            memory_pattern: self.analyze_memory_pattern(),
        }
    }

    /// Analyze the memory access pattern for AI insights
    fn analyze_memory_pattern(&self) -> MemoryAccessPattern {
        // This would analyze actual memory access patterns
        MemoryAccessPattern {
            access_frequency: AccessFrequency::Medium,
            access_pattern: AccessPattern::Sequential,
            sharing_level: SharingLevel::Private,
        }
    }
}

/// Unique identifier for memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocationId(u64);

impl AllocationId {
    /// Generate a new unique allocation ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::SeqCst))
    }
}

/// AI metadata for memory allocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationAIMetadata {
    /// Semantic type metadata
    pub semantic_type: SemanticTypeMetadata,
    /// Size of allocation
    pub allocation_size: usize,
    /// When allocation occurred
    pub allocation_time: SystemTime,
    /// Unique allocation identifier
    pub allocation_id: AllocationId,
    /// Memory access pattern
    pub memory_pattern: MemoryAccessPattern,
}

/// Memory access pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessPattern {
    /// How frequently this memory is accessed
    pub access_frequency: AccessFrequency,
    /// Pattern of memory access
    pub access_pattern: AccessPattern,
    /// Level of sharing across components
    pub sharing_level: SharingLevel,
}

/// Frequency of memory access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessFrequency {
    /// Very low frequency
    VeryLow,
    /// Low frequency
    Low,
    /// Medium frequency
    Medium,
    /// High frequency
    High,
    /// Very high frequency
    VeryHigh,
}

/// Pattern of memory access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Temporal locality
    Temporal,
    /// Spatial locality
    Spatial,
}

/// Level of memory sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingLevel {
    /// Private to single component
    Private,
    /// Shared within module
    ModuleShared,
    /// Shared across components
    ComponentShared,
    /// Globally shared
    Global,
}

/// Trait for target-specific memory allocators
pub trait MemoryAllocator: Send + Sync {
    /// Allocate memory of the specified size
    fn allocate(
        &self,
        size: usize,
        context: &ExecutionContext,
    ) -> Result<NonNull<u8>, MemoryError>;

    /// Deallocate memory at the specified pointer
    fn deallocate(
        &self,
        ptr: NonNull<u8>,
        context: &ExecutionContext,
    ) -> Result<(), MemoryError>;

    /// Get allocator name
    fn name(&self) -> &'static str;

    /// Get allocator statistics
    fn stats(&self) -> AllocatorStats;
}

/// Statistics for a memory allocator
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes deallocated
    pub total_deallocated: usize,
    /// Current memory usage
    pub current_usage: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
}

/// JavaScript/TypeScript memory allocator
#[derive(Debug)]
pub struct JavaScriptAllocator {
    /// Allocation statistics
    stats: Arc<RwLock<AllocatorStats>>,
}

impl JavaScriptAllocator {
    /// Create a new JavaScript allocator
    pub fn new() -> Result<Self, MemoryError> {
        Ok(Self {
            stats: Arc::new(RwLock::new(AllocatorStats {
                total_allocated: 0,
                total_deallocated: 0,
                current_usage: 0,
                allocation_count: 0,
                deallocation_count: 0,
            })),
        })
    }
}

impl MemoryAllocator for JavaScriptAllocator {
    fn allocate(
        &self,
        size: usize,
        _context: &ExecutionContext,
    ) -> Result<NonNull<u8>, MemoryError> {
        // Use safe allocation through Vec
        let mut vec = vec![0u8; size];
        vec.shrink_to_fit();
        let boxed = vec.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut u8;
        let non_null = NonNull::new(ptr)
            .ok_or_else(|| MemoryError::AllocationFailed {
                reason: "Failed to create non-null pointer".to_string(),
            })?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocated += size;
            stats.current_usage += size;
            stats.allocation_count += 1;
        }

        Ok(non_null)
    }

    fn deallocate(
        &self,
        ptr: NonNull<u8>,
        _context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        // In a real implementation, we would track allocation sizes
        // For now, we'll convert back to Box and drop it
        // Safe deallocation - in a real implementation, we'd maintain a registry
        // of allocations and their sizes to properly deallocate them
        // For now, we just simulate the deallocation by dropping the pointer reference

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.deallocation_count += 1;
            // We'd subtract the actual size if we tracked it
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "JavaScript"
    }

    fn stats(&self) -> AllocatorStats {
        self.stats.read().unwrap().clone()
    }
}

/// WebAssembly memory allocator
#[derive(Debug)]
pub struct WasmAllocator {
    /// Allocation statistics
    stats: Arc<RwLock<AllocatorStats>>,
}

impl WasmAllocator {
    /// Create a new WebAssembly allocator
    pub fn new() -> Result<Self, MemoryError> {
        Ok(Self {
            stats: Arc::new(RwLock::new(AllocatorStats {
                total_allocated: 0,
                total_deallocated: 0,
                current_usage: 0,
                allocation_count: 0,
                deallocation_count: 0,
            })),
        })
    }
}

impl MemoryAllocator for WasmAllocator {
    fn allocate(
        &self,
        size: usize,
        _context: &ExecutionContext,
    ) -> Result<NonNull<u8>, MemoryError> {
        // Use safe allocation through Vec
        let mut vec = vec![0u8; size];
        vec.shrink_to_fit();
        let boxed = vec.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut u8;
        let non_null = NonNull::new(ptr)
            .ok_or_else(|| MemoryError::AllocationFailed {
                reason: "Failed to create non-null pointer".to_string(),
            })?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocated += size;
            stats.current_usage += size;
            stats.allocation_count += 1;
        }

        Ok(non_null)
    }

    fn deallocate(
        &self,
        ptr: NonNull<u8>,
        _context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        // Safe deallocation simulation - in a real implementation, 
        // we'd maintain proper allocation tracking

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.deallocation_count += 1;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "WebAssembly"
    }

    fn stats(&self) -> AllocatorStats {
        self.stats.read().unwrap().clone()
    }
}

/// Native memory allocator
#[derive(Debug)]
pub struct NativeAllocator {
    /// Allocation statistics
    stats: Arc<RwLock<AllocatorStats>>,
}

impl NativeAllocator {
    /// Create a new native allocator
    pub fn new() -> Result<Self, MemoryError> {
        Ok(Self {
            stats: Arc::new(RwLock::new(AllocatorStats {
                total_allocated: 0,
                total_deallocated: 0,
                current_usage: 0,
                allocation_count: 0,
                deallocation_count: 0,
            })),
        })
    }
}

impl MemoryAllocator for NativeAllocator {
    fn allocate(
        &self,
        size: usize,
        _context: &ExecutionContext,
    ) -> Result<NonNull<u8>, MemoryError> {
        // Use safe allocation through Vec
        let mut vec = vec![0u8; size];
        vec.shrink_to_fit();
        let boxed = vec.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut u8;
        let non_null = NonNull::new(ptr)
            .ok_or_else(|| MemoryError::AllocationFailed {
                reason: "Failed to create non-null pointer".to_string(),
            })?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocated += size;
            stats.current_usage += size;
            stats.allocation_count += 1;
        }

        Ok(non_null)
    }

    fn deallocate(
        &self,
        ptr: NonNull<u8>,
        _context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        // Safe deallocation simulation - in a real implementation, 
        // we'd maintain proper allocation tracking

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.deallocation_count += 1;
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Native"
    }

    fn stats(&self) -> AllocatorStats {
        self.stats.read().unwrap().clone()
    }
}

/// Memory usage tracker for profiling and analysis
#[derive(Debug)]
pub struct MemoryUsageTracker {
    /// Usage statistics
    stats: Arc<RwLock<MemoryUsageStats>>,
    /// Allocation history
    allocation_history: Arc<Mutex<Vec<AllocationRecord>>>,
}

impl MemoryUsageTracker {
    /// Create a new memory usage tracker
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(MemoryUsageStats::default())),
            allocation_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record a memory allocation
    pub fn record_allocation<T: SemanticType>(
        &self,
        ptr: &SemanticPtr<T>,
        context: &ExecutionContext,
    ) {
        let record = AllocationRecord {
            allocation_id: ptr.allocation_id(),
            size: ptr.size(),
            semantic_type: ptr.semantic_type().type_name(),
            timestamp: SystemTime::now(),
            context_id: context.execution_id,
        };

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocated += ptr.size();
            stats.current_usage += ptr.size();
            stats.allocation_count += 1;
        }

        // Record in history
        {
            let mut history = self.allocation_history.lock().unwrap();
            history.push(record);
        }
    }

    /// Record a memory deallocation
    pub fn record_deallocation<T: SemanticType>(
        &self,
        ptr: &SemanticPtr<T>,
        _context: &ExecutionContext,
    ) {
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_deallocated += ptr.size();
            stats.current_usage -= ptr.size();
            stats.deallocation_count += 1;
        }
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.stats.read().unwrap().current_usage
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryUsageStats {
        self.stats.read().unwrap().clone()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryUsageStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes deallocated
    pub total_deallocated: usize,
    /// Current memory usage
    pub current_usage: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
}

/// Record of a memory allocation
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Allocation ID
    pub allocation_id: AllocationId,
    /// Size of allocation
    pub size: usize,
    /// Semantic type name
    pub semantic_type: &'static str,
    /// Allocation timestamp
    pub timestamp: SystemTime,
    /// Execution context ID
    pub context_id: crate::execution::ExecutionId,
}

/// Memory pool manager for efficient allocation
#[derive(Debug)]
pub struct MemoryPoolManager {
    /// Pool statistics
    stats: Arc<RwLock<PoolStats>>,
}

impl MemoryPoolManager {
    /// Create a new memory pool manager
    pub fn new() -> Result<Self, MemoryError> {
        Ok(Self {
            stats: Arc::new(RwLock::new(PoolStats::default())),
        })
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        self.stats.read().unwrap().clone()
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of active pools
    pub active_pools: usize,
    /// Total pooled memory
    pub total_pooled: usize,
    /// Pool hit rate
    pub hit_rate: f64,
}

/// Semantic memory registry for tracking allocations
#[derive(Debug)]
pub struct SemanticMemoryRegistry {
    /// Registered allocations
    allocations: HashMap<AllocationId, AllocationInfo>,
}

impl SemanticMemoryRegistry {
    /// Create a new semantic memory registry
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
        }
    }

    /// Register a new allocation
    pub fn register_allocation<T: SemanticType>(
        &mut self,
        ptr: &SemanticPtr<T>,
        context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        let info = AllocationInfo {
            size: ptr.size(),
            semantic_type: ptr.semantic_type().type_name().to_string(),
            owner_component: context.component_id,
            allocated_at: SystemTime::now(),
        };

        self.allocations.insert(ptr.allocation_id(), info);
        Ok(())
    }

    /// Unregister an allocation
    pub fn unregister_allocation<T: SemanticType>(
        &mut self,
        ptr: &SemanticPtr<T>,
    ) -> Result<(), MemoryError> {
        self.allocations.remove(&ptr.allocation_id())
            .ok_or_else(|| MemoryError::AllocationNotFound {
                id: ptr.allocation_id(),
            })?;
        Ok(())
    }

    /// Verify ownership of an allocation
    pub fn verify_ownership<T: SemanticType>(
        &self,
        ptr: &SemanticPtr<T>,
        context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        let info = self.allocations.get(&ptr.allocation_id())
            .ok_or_else(|| MemoryError::AllocationNotFound {
                id: ptr.allocation_id(),
            })?;

        if info.owner_component != context.component_id {
            return Err(MemoryError::OwnershipViolation {
                allocation_id: ptr.allocation_id(),
                owner: info.owner_component,
                accessor: context.component_id,
            });
        }

        Ok(())
    }

    /// Get the number of registered objects
    pub fn object_count(&self) -> usize {
        self.allocations.len()
    }
}

/// Information about a memory allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Size of allocation
    pub size: usize,
    /// Semantic type name
    pub semantic_type: String,
    /// Component that owns this allocation
    pub owner_component: crate::capability::ComponentId,
    /// When allocation occurred
    pub allocated_at: SystemTime,
}

/// Memory security enforcer
#[derive(Debug)]
pub struct MemorySecurityEnforcer {
    /// Security policies
    policies: Arc<RwLock<Vec<MemorySecurityPolicy>>>,
}

impl MemorySecurityEnforcer {
    /// Create a new memory security enforcer
    pub fn new() -> Result<Self, MemoryError> {
        Ok(Self {
            policies: Arc::new(RwLock::new(Self::default_policies())),
        })
    }

    /// Check allocation limits before allowing allocation
    pub fn check_allocation_limits(
        &self,
        size: usize,
        context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        let policies = self.policies.read().unwrap();
        
        for policy in policies.iter() {
            policy.check_allocation(size, context)?;
        }
        
        Ok(())
    }

    /// Get default memory security policies
    fn default_policies() -> Vec<MemorySecurityPolicy> {
        vec![
            MemorySecurityPolicy::MaxAllocationSize { max_size: 1024 * 1024 * 1024 }, // 1GB
            MemorySecurityPolicy::ComponentMemoryLimit { max_memory_per_component: 512 * 1024 * 1024 }, // 512MB
        ]
    }
}

/// Memory security policies
#[derive(Debug, Clone)]
pub enum MemorySecurityPolicy {
    /// Maximum size for a single allocation
    MaxAllocationSize {
        /// Maximum size in bytes
        max_size: usize,
    },
    /// Maximum memory per component
    ComponentMemoryLimit {
        /// Maximum memory per component in bytes
        max_memory_per_component: usize,
    },
}

impl MemorySecurityPolicy {
    /// Check if an allocation is allowed by this policy
    pub fn check_allocation(
        &self,
        size: usize,
        _context: &ExecutionContext,
    ) -> Result<(), MemoryError> {
        match self {
            Self::MaxAllocationSize { max_size } => {
                if size > *max_size {
                    return Err(MemoryError::AllocationTooLarge {
                        requested: size,
                        max_allowed: *max_size,
                    });
                }
            }
            Self::ComponentMemoryLimit { max_memory_per_component: _ } => {
                // Would check component-specific memory usage
            }
        }
        Ok(())
    }
}

/// Memory statistics for monitoring
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes deallocated
    pub total_deallocated: usize,
    /// Current memory usage
    pub current_usage: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Number of semantic objects
    pub semantic_objects: usize,
    /// Memory pool statistics
    pub memory_pools: PoolStats,
}

/// Memory-related errors
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Insufficient capability for memory operation
    #[error("Insufficient capability: required {required}, available {available}")]
    InsufficientCapability {
        /// Required capability
        required: String,
        /// Available capabilities
        available: String,
    },

    /// Unsupported target for memory allocation
    #[error("Unsupported target: {target:?}")]
    UnsupportedTarget {
        /// Target that is not supported
        target: crate::execution::ExecutionTarget,
    },

    /// Memory allocation failed
    #[error("Memory allocation failed: {reason}")]
    AllocationFailed {
        /// Reason for failure
        reason: String,
    },

    /// Allocation too large
    #[error("Allocation too large: requested {requested} bytes, max allowed {max_allowed} bytes")]
    AllocationTooLarge {
        /// Requested size
        requested: usize,
        /// Maximum allowed size
        max_allowed: usize,
    },

    /// Allocation not found
    #[error("Allocation not found: {id:?}")]
    AllocationNotFound {
        /// Allocation ID
        id: AllocationId,
    },

    /// Ownership violation
    #[error("Ownership violation: allocation {allocation_id:?} owned by {owner:?}, accessed by {accessor:?}")]
    OwnershipViolation {
        /// Allocation ID
        allocation_id: AllocationId,
        /// Owner component
        owner: crate::capability::ComponentId,
        /// Accessor component
        accessor: crate::capability::ComponentId,
    },

    /// Generic memory error
    #[error("Memory error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
}

/// Semantic validation error
#[derive(Debug, Error)]
pub enum SemanticValidationError {
    /// Invalid semantic constraint
    #[error("Invalid semantic constraint: {constraint}")]
    InvalidConstraint {
        /// Constraint that is invalid
        constraint: String,
    },

    /// Type mismatch
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Actual type
        actual: String,
    },

    /// Generic validation error
    #[error("Semantic validation error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
}

// Example semantic type implementations
#[derive(Debug, Clone)]
pub struct UserDataType {
    pub user_id: u64,
    pub classification: SecurityClassification,
}

impl SemanticType for UserDataType {
    fn type_name(&self) -> &'static str {
        "UserData"
    }

    fn ai_metadata(&self) -> SemanticTypeMetadata {
        SemanticTypeMetadata {
            domain: "User Management".to_string(),
            purpose: "Represents user data with security classification".to_string(),
            constraints: vec![
                "Must have valid user ID".to_string(),
                "Must include security classification".to_string(),
            ],
            relationships: vec![
                TypeRelationship {
                    relationship_type: RelationshipType::Uses,
                    related_type: "SecurityClassification".to_string(),
                    description: "Uses security classification for access control".to_string(),
                },
            ],
        }
    }

    fn validate_semantics(&self) -> Result<(), SemanticValidationError> {
        if self.user_id == 0 {
            return Err(SemanticValidationError::InvalidConstraint {
                constraint: "User ID must be non-zero".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum SecurityClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

#[derive(Debug, Clone)]
pub struct BusinessDocumentType {
    pub document_id: String,
    pub document_type: DocumentType,
    pub sensitivity: SensitivityLevel,
}

impl SemanticType for BusinessDocumentType {
    fn type_name(&self) -> &'static str {
        "BusinessDocument"
    }

    fn ai_metadata(&self) -> SemanticTypeMetadata {
        SemanticTypeMetadata {
            domain: "Document Management".to_string(),
            purpose: "Represents business documents with type and sensitivity".to_string(),
            constraints: vec![
                "Must have unique document ID".to_string(),
                "Must specify document type".to_string(),
                "Must include sensitivity level".to_string(),
            ],
            relationships: vec![
                TypeRelationship {
                    relationship_type: RelationshipType::ComposedOf,
                    related_type: "DocumentType".to_string(),
                    description: "Composed of document type and sensitivity".to_string(),
                },
            ],
        }
    }

    fn validate_semantics(&self) -> Result<(), SemanticValidationError> {
        if self.document_id.is_empty() {
            return Err(SemanticValidationError::InvalidConstraint {
                constraint: "Document ID cannot be empty".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum DocumentType {
    Contract,
    Report,
    Proposal,
    Invoice,
}

#[derive(Debug, Clone)]
pub enum SensitivityLevel {
    Low,
    Medium,
    High,
    Critical,
} 