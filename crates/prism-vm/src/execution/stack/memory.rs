//! Stack Memory Management Integration
//!
//! This module provides advanced memory management for stack operations by integrating
//! with the existing prism-runtime resource management system. It follows the principle
//! of leveraging existing infrastructure rather than duplicating logic.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackValue, StackFrame};
use prism_runtime::{
    resources::{
        ResourceManager, MemoryPool, PooledBuffer, MemoryManager,
        ResourceType, ResourceRequest, QuotaId, PriorityClass,
    },
    resource_management::{SemanticPtr, SemanticType, AllocationId},
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, span, Level};

/// Stack memory manager that integrates with prism-runtime
#[derive(Debug)]
pub struct StackMemoryManager {
    /// Resource manager integration
    resource_manager: Arc<ResourceManager>,
    
    /// Stack-specific memory pool
    stack_pool: Arc<dyn MemoryPool>,
    
    /// Memory allocation tracking
    allocations: Arc<RwLock<HashMap<AllocationId, StackAllocation>>>,
    
    /// Memory usage statistics
    stats: Arc<RwLock<StackMemoryStats>>,
    
    /// Configuration
    config: StackMemoryConfig,
}

/// Stack memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackMemoryConfig {
    /// Use NUMA-aware allocation
    pub numa_aware: bool,
    
    /// Stack frame pool size
    pub frame_pool_size: usize,
    
    /// Value pool size
    pub value_pool_size: usize,
    
    /// Enable memory pressure monitoring
    pub pressure_monitoring: bool,
    
    /// Memory pressure threshold (0.0 - 1.0)
    pub pressure_threshold: f64,
}

impl Default for StackMemoryConfig {
    fn default() -> Self {
        Self {
            numa_aware: true,
            frame_pool_size: 1024,
            value_pool_size: 8192,
            pressure_monitoring: true,
            pressure_threshold: 0.8,
        }
    }
}

/// Stack allocation tracking
#[derive(Debug, Clone)]
pub struct StackAllocation {
    /// Allocation ID
    pub id: AllocationId,
    
    /// Size in bytes
    pub size: usize,
    
    /// Purpose of allocation
    pub purpose: StackAllocationPurpose,
    
    /// When allocated
    pub allocated_at: Instant,
    
    /// Business context
    pub business_context: Option<String>,
}

/// Purpose of stack allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StackAllocationPurpose {
    /// Stack frame allocation
    StackFrame { function_name: String },
    
    /// Local variable storage
    LocalVariable { frame_id: u32, slot: u8 },
    
    /// Temporary value storage
    TemporaryValue { operation: String },
    
    /// Upvalue storage for closures
    UpvalueStorage { closure_id: u32 },
    
    /// Exception handler data
    ExceptionHandler { handler_id: u32 },
}

/// Stack memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackMemoryStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    
    /// Current active allocations
    pub active_allocations: usize,
    
    /// Peak memory usage
    pub peak_usage: usize,
    
    /// Pool hit rate
    pub pool_hit_rate: f64,
    
    /// NUMA node distribution
    pub numa_distribution: HashMap<u32, usize>,
    
    /// Memory pressure level (0.0 - 1.0)
    pub pressure_level: f64,
    
    /// Allocation by purpose
    pub allocation_by_purpose: HashMap<String, usize>,
}

impl Default for StackMemoryStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            active_allocations: 0,
            peak_usage: 0,
            pool_hit_rate: 0.0,
            numa_distribution: HashMap::new(),
            pressure_level: 0.0,
            allocation_by_purpose: HashMap::new(),
        }
    }
}

impl StackMemoryManager {
    /// Create a new stack memory manager
    pub fn new(resource_manager: Arc<ResourceManager>) -> VMResult<Self> {
        let _span = span!(Level::INFO, "stack_memory_init").entered();
        info!("Initializing stack memory management");

        let config = StackMemoryConfig::default();
        
        // Get the stack-specific memory pool from resource manager
        let stack_pool = resource_manager.pool_manager()
            .get_pool("stack_pool")
            .unwrap_or_else(|| {
                // Create a high-performance pool for stack operations
                resource_manager.pool_manager().default_pool()
            });

        Ok(Self {
            resource_manager,
            stack_pool,
            allocations: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StackMemoryStats::default())),
            config,
        })
    }

    /// Allocate memory for a stack frame with business context
    pub fn allocate_frame(
        &self,
        function_name: &str,
        frame_size: usize,
        business_context: Option<String>,
    ) -> VMResult<SemanticPtr<StackFrameAllocation>> {
        let _span = span!(Level::DEBUG, "allocate_frame", 
            function = %function_name, 
            size = frame_size
        ).entered();

        // Check memory pressure
        self.check_memory_pressure()?;

        // Allocate through resource manager with quota checking
        let buffer = self.resource_manager.allocate_memory(
            frame_size,
            None, // No specific quota for now
            Some("stack_pool"),
        ).map_err(|e| PrismVMError::RuntimeError {
            message: format!("Failed to allocate stack frame: {}", e),
        })?;

        let allocation = StackFrameAllocation::new(
            function_name.to_string(),
            buffer,
            business_context,
        );

        let semantic_ptr = SemanticPtr::new(allocation);
        
        // Track the allocation
        self.track_allocation(StackAllocation {
            id: semantic_ptr.allocation_id(),
            size: frame_size,
            purpose: StackAllocationPurpose::StackFrame {
                function_name: function_name.to_string(),
            },
            allocated_at: Instant::now(),
            business_context,
        });

        debug!("Allocated stack frame for function: {}", function_name);
        Ok(semantic_ptr)
    }

    /// Allocate memory for stack values with semantic context
    pub fn allocate_values(
        &self,
        count: usize,
        operation_context: &str,
    ) -> VMResult<PooledBuffer> {
        let _span = span!(Level::DEBUG, "allocate_values", 
            count = count, 
            context = %operation_context
        ).entered();

        let size = count * std::mem::size_of::<StackValue>();
        
        // Check memory pressure
        self.check_memory_pressure()?;

        let buffer = self.resource_manager.allocate_memory(
            size,
            None,
            Some("stack_pool"),
        ).map_err(|e| PrismVMError::RuntimeError {
            message: format!("Failed to allocate stack values: {}", e),
        })?;

        // Track the allocation
        self.track_allocation(StackAllocation {
            id: AllocationId::new(),
            size,
            purpose: StackAllocationPurpose::TemporaryValue {
                operation: operation_context.to_string(),
            },
            allocated_at: Instant::now(),
            business_context: Some(format!("Stack values for: {}", operation_context)),
        });

        debug!("Allocated {} stack values for: {}", count, operation_context);
        Ok(buffer)
    }

    /// Check memory pressure and trigger optimization if needed
    fn check_memory_pressure(&self) -> VMResult<()> {
        if !self.config.pressure_monitoring {
            return Ok(());
        }

        let current_usage = self.resource_manager.system_snapshot().memory.utilization_percent;
        let pressure_level = current_usage / 100.0;

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.pressure_level = pressure_level;
        }

        if pressure_level > self.config.pressure_threshold {
            warn!("High memory pressure detected: {:.1}%", current_usage);
            
            // Trigger memory optimization
            self.optimize_memory_usage()?;
        }

        Ok(())
    }

    /// Optimize memory usage under pressure
    fn optimize_memory_usage(&self) -> VMResult<()> {
        let _span = span!(Level::INFO, "optimize_memory").entered();
        info!("Optimizing stack memory usage due to pressure");

        // Trim unused allocations
        let trimmed = self.trim_unused_allocations();
        
        // Suggest garbage collection if available
        // This would integrate with the VM's GC system
        
        info!("Memory optimization complete, trimmed {} allocations", trimmed);
        Ok(())
    }

    /// Trim unused allocations
    fn trim_unused_allocations(&self) -> usize {
        let mut allocations = self.allocations.write().unwrap();
        let initial_count = allocations.len();
        
        // Remove allocations older than a threshold
        let threshold = Instant::now() - Duration::from_secs(300); // 5 minutes
        allocations.retain(|_, alloc| alloc.allocated_at > threshold);
        
        let trimmed = initial_count - allocations.len();
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.active_allocations = allocations.len();
        }
        
        trimmed
    }

    /// Track an allocation
    fn track_allocation(&self, allocation: StackAllocation) {
        let mut allocations = self.allocations.write().unwrap();
        allocations.insert(allocation.id, allocation.clone());

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.total_allocated += allocation.size;
        stats.active_allocations = allocations.len();
        stats.peak_usage = stats.peak_usage.max(stats.total_allocated);
        
        let purpose_key = match &allocation.purpose {
            StackAllocationPurpose::StackFrame { .. } => "stack_frame",
            StackAllocationPurpose::LocalVariable { .. } => "local_variable",
            StackAllocationPurpose::TemporaryValue { .. } => "temporary_value",
            StackAllocationPurpose::UpvalueStorage { .. } => "upvalue_storage",
            StackAllocationPurpose::ExceptionHandler { .. } => "exception_handler",
        };
        
        *stats.allocation_by_purpose.entry(purpose_key.to_string()).or_insert(0) += allocation.size;
    }

    /// Get current memory statistics
    pub fn stats(&self) -> StackMemoryStats {
        self.stats.read().unwrap().clone()
    }

    /// Get memory usage for AI analysis
    pub fn ai_memory_context(&self) -> StackMemoryAIContext {
        let stats = self.stats();
        let allocations = self.allocations.read().unwrap();
        
        StackMemoryAIContext {
            total_allocated_bytes: stats.total_allocated,
            allocation_count: stats.active_allocations,
            pressure_level: stats.pressure_level,
            allocation_purposes: stats.allocation_by_purpose,
            recent_allocations: allocations.values()
                .filter(|a| a.allocated_at.elapsed() < Duration::from_secs(60))
                .map(|a| format!("{:?}", a.purpose))
                .collect(),
            optimization_suggestions: self.generate_optimization_suggestions(&stats),
        }
    }

    /// Generate optimization suggestions for AI
    fn generate_optimization_suggestions(&self, stats: &StackMemoryStats) -> Vec<String> {
        let mut suggestions = Vec::new();

        if stats.pressure_level > 0.7 {
            suggestions.push("Consider reducing stack frame size".to_string());
        }

        if stats.pool_hit_rate < 0.8 {
            suggestions.push("Pool configuration may need tuning".to_string());
        }

        if stats.allocation_by_purpose.get("temporary_value").unwrap_or(&0) > &(stats.total_allocated / 2) {
            suggestions.push("High temporary value allocation - consider value reuse".to_string());
        }

        suggestions
    }
}

/// Stack frame allocation with semantic information
#[derive(Debug)]
pub struct StackFrameAllocation {
    /// Function name for business context
    pub function_name: String,
    
    /// Underlying memory buffer
    pub buffer: PooledBuffer,
    
    /// Business context
    pub business_context: Option<String>,
    
    /// Allocation timestamp
    pub allocated_at: Instant,
}

impl StackFrameAllocation {
    /// Create a new stack frame allocation
    pub fn new(
        function_name: String,
        buffer: PooledBuffer,
        business_context: Option<String>,
    ) -> Self {
        Self {
            function_name,
            buffer,
            business_context,
            allocated_at: Instant::now(),
        }
    }
}

impl SemanticType for StackFrameAllocation {
    fn type_name(&self) -> &'static str {
        "StackFrameAllocation"
    }

    fn business_purpose(&self) -> Option<&str> {
        self.business_context.as_deref()
    }

    fn expected_lifetime(&self) -> Option<Duration> {
        Some(Duration::from_secs(60)) // Stack frames are typically short-lived
    }
}

/// AI context for stack memory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackMemoryAIContext {
    /// Total bytes allocated
    pub total_allocated_bytes: usize,
    
    /// Number of active allocations
    pub allocation_count: usize,
    
    /// Memory pressure level
    pub pressure_level: f64,
    
    /// Allocation breakdown by purpose
    pub allocation_purposes: HashMap<String, usize>,
    
    /// Recent allocation descriptions
    pub recent_allocations: Vec<String>,
    
    /// AI-generated optimization suggestions
    pub optimization_suggestions: Vec<String>,
} 