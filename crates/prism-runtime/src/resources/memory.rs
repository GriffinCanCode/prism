//! Memory Management Utilities
//!
//! This module provides memory management utilities that complement the pool system,
//! offering additional abstractions for memory allocation patterns and tracking.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Memory management errors
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Allocation failed
    #[error("Memory allocation failed: {size} bytes")]
    AllocationFailed { size: usize },
    
    /// Invalid memory access
    #[error("Invalid memory access: {message}")]
    InvalidAccess { message: String },
    
    /// Memory limit exceeded  
    #[error("Memory limit exceeded: {current} > {limit}")]
    LimitExceeded { current: usize, limit: usize },
    
    /// Generic memory error
    #[error("Memory error: {message}")]
    Generic { message: String },
}

/// Memory allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Size of allocation in bytes
    pub size: usize,
    /// When allocated
    pub allocated_at: Instant,
    /// Purpose/context of allocation
    pub purpose: String,
    /// Stack trace or source information
    pub source: Option<String>,
}

/// Memory manager for tracking allocations
#[derive(Debug)]
pub struct MemoryManager {
    /// Active allocations
    allocations: Mutex<HashMap<usize, MemoryAllocation>>,
    /// Total bytes allocated
    total_allocated: Mutex<usize>,
    /// Configuration
    config: MemoryManagerConfig,
}

/// Configuration for the memory manager
#[derive(Debug, Clone)]
pub struct MemoryManagerConfig {
    /// Maximum memory that can be allocated
    pub max_memory: Option<usize>,
    /// Whether to track allocation sources
    pub track_sources: bool,
    /// Warning threshold as a fraction of max memory
    pub warning_threshold: f64,
}

impl Default for MemoryManagerConfig {
    fn default() -> Self {
        Self {
            max_memory: None,
            track_sources: false,
            warning_threshold: 0.8, // 80%
        }
    }
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Result<Self, MemoryError> {
        Self::with_config(MemoryManagerConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: MemoryManagerConfig) -> Result<Self, MemoryError> {
        Ok(Self {
            allocations: Mutex::new(HashMap::new()),
            total_allocated: Mutex::new(0),
            config,
        })
    }
    
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        *self.total_allocated.lock().unwrap()
    }
    
    /// Get number of active allocations
    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }
    
    /// Check if within memory limits
    pub fn check_limits(&self, additional_size: usize) -> Result<(), MemoryError> {
        let current = self.current_usage();
        let new_total = current + additional_size;
        
        if let Some(max) = self.config.max_memory {
            if new_total > max {
                return Err(MemoryError::LimitExceeded { 
                    current: new_total, 
                    limit: max 
                });
            }
        }
        
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn statistics(&self) -> MemoryStatistics {
        let allocations = self.allocations.lock().unwrap();
        let total = *self.total_allocated.lock().unwrap();
        
        let count = allocations.len();
        let avg_size = if count > 0 { total / count } else { 0 };
        
        let (min_size, max_size) = allocations.values()
            .map(|a| a.size)
            .fold((usize::MAX, 0), |(min, max), size| {
                (min.min(size), max.max(size))
            });
        
        MemoryStatistics {
            total_allocated: total,
            active_allocations: count,
            average_allocation_size: avg_size,
            min_allocation_size: if min_size == usize::MAX { 0 } else { min_size },
            max_allocation_size: max_size,
            memory_limit: self.config.max_memory,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Total bytes currently allocated
    pub total_allocated: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Average allocation size
    pub average_allocation_size: usize,
    /// Smallest allocation size
    pub min_allocation_size: usize,
    /// Largest allocation size
    pub max_allocation_size: usize,
    /// Memory limit (if any)
    pub memory_limit: Option<usize>,
}

/// Trait for types that have semantic meaning in memory allocation
pub trait SemanticType: Send + Sync + 'static {
    /// Get the semantic type name
    fn type_name(&self) -> &'static str;
    /// Get the business purpose of this allocation
    fn business_purpose(&self) -> Option<&str> { None }
    /// Get the expected lifetime
    fn expected_lifetime(&self) -> Option<Duration> { None }
}

/// Smart pointer that tracks semantic information about memory allocations
#[derive(Debug)]
pub struct SemanticPtr<T: SemanticType> {
    /// The actual data
    data: Box<T>,
    /// Allocation ID for tracking
    allocation_id: AllocationId,
    /// When this was allocated
    allocated_at: Instant,
}

/// Unique identifier for memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocationId(u64);

impl AllocationId {
    /// Generate a new allocation ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::SeqCst))
    }
}

impl<T: SemanticType> SemanticPtr<T> {
    /// Create a new semantic pointer
    pub fn new(data: T) -> Self {
        Self {
            data: Box::new(data),
            allocation_id: AllocationId::new(),
            allocated_at: Instant::now(),
        }
    }
    
    /// Get the allocation ID
    pub fn allocation_id(&self) -> AllocationId {
        self.allocation_id
    }
    
    /// Get when this was allocated
    pub fn allocated_at(&self) -> Instant {
        self.allocated_at
    }
}

impl<T: SemanticType> std::ops::Deref for SemanticPtr<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: SemanticType> std::ops::DerefMut for SemanticPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new();
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_memory_limits() {
        let config = MemoryManagerConfig {
            max_memory: Some(1024),
            ..Default::default()
        };
        
        let manager = MemoryManager::with_config(config).unwrap();
        
        // Should be OK within limit
        assert!(manager.check_limits(512).is_ok());
        
        // Should fail when exceeding limit
        assert!(manager.check_limits(2048).is_err());
    }
    
    #[test]
    fn test_memory_statistics() {
        let manager = MemoryManager::new().unwrap();
        let stats = manager.statistics();
        
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.active_allocations, 0);
    }
} 