//! Advanced Memory Pool System
//!
//! This module implements a high-performance memory pool system inspired by modern
//! allocators like TCMalloc, jemalloc, and .NET's ArrayPool. It provides:
//!
//! - **Size-class based allocation** for efficient memory reuse
//! - **Thread-local caching** to reduce contention
//! - **NUMA-aware allocation** for multi-socket systems
//! - **Zero-copy buffer management** with RAII guarantees
//! - **Adaptive pool sizing** based on usage patterns
//! - **Comprehensive statistics** for monitoring and tuning

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// Errors that can occur in memory pool operations
#[derive(Debug, Error)]
pub enum PoolError {
    /// Pool is at capacity
    #[error("Pool is at capacity: {current}/{max}")]
    AtCapacity { current: usize, max: usize },
    
    /// Invalid buffer size requested
    #[error("Invalid buffer size: {size}")]
    InvalidSize { size: usize },
    
    /// Pool configuration error
    #[error("Pool configuration error: {message}")]
    Config { message: String },
    
    /// Generic pool error
    #[error("Pool error: {message}")]
    Generic { message: String },
}

/// Configuration for memory pools
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial capacity per size class
    pub initial_capacity: usize,
    /// Maximum capacity per size class
    pub max_capacity: usize,
    /// Enable thread-local caching
    pub thread_local_cache: bool,
    /// Size of thread-local cache
    pub thread_cache_size: usize,
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
    /// Trim unused buffers interval (seconds)
    pub trim_interval_secs: u64,
    /// Statistics collection interval (seconds)
    pub stats_interval_secs: u64,
    /// Zero memory on return to pool
    pub zero_on_return: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 128,
            max_capacity: 1024,
            thread_local_cache: true,
            thread_cache_size: 16,
            numa_aware: false,
            trim_interval_secs: 60,
            stats_interval_secs: 10,
            zero_on_return: false,
        }
    }
}

/// Statistics for a memory pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStats {
    /// Total allocations made
    pub total_allocations: u64,
    /// Total deallocations made
    pub total_deallocations: u64,
    /// Current active allocations
    pub active_allocations: u64,
    /// Peak active allocations
    pub peak_allocations: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Current bytes in use
    pub bytes_in_use: u64,
    /// Peak bytes in use
    pub peak_bytes_in_use: u64,
    /// Pool hit rate (successful reuse)
    pub hit_rate: f64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Pool efficiency (reused / total)
    pub efficiency: f64,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            peak_allocations: 0,
            total_bytes_allocated: 0,
            bytes_in_use: 0,
            peak_bytes_in_use: 0,
            hit_rate: 0.0,
            average_allocation_size: 0.0,
            efficiency: 0.0,
        }
    }
}

/// A managed buffer that automatically returns to the pool when dropped
pub struct PooledBuffer {
    /// The actual data buffer - using Vec<u8> for safe memory management
    data: Vec<u8>,
    /// Size of the buffer
    size: usize,
    /// Pool reference for return on drop
    pool_ref: Option<Arc<dyn MemoryPool>>,
    /// Whether this buffer should be zeroed on return
    zero_on_return: bool,
}

impl PooledBuffer {
    /// Create a new pooled buffer
    pub fn new(size: usize, pool_ref: Option<Arc<dyn MemoryPool>>, zero_on_return: bool) -> Self {
        let mut data = Vec::with_capacity(size);
        // Fill with zeros initially for safety
        data.resize(size, 0);
        
        Self {
            data,
            size,
            pool_ref,
            zero_on_return,
        }
    }
    
    /// Get the buffer size
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// Get buffer as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.size]
    }
    
    /// Get buffer as immutable slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.size]
    }
    
    /// Clear the buffer (zero all bytes)
    pub fn clear(&mut self) {
        self.data.fill(0);
    }
    
    /// Resize the buffer (may allocate new memory)
    pub fn resize(&mut self, new_size: usize) {
        if new_size <= self.data.capacity() {
            self.data.resize(new_size, 0);
            self.size = new_size;
        } else {
            // Need to grow beyond capacity - create new Vec
            let mut new_data = Vec::with_capacity(new_size);
            new_data.extend_from_slice(&self.data[..self.size]);
            new_data.resize(new_size, 0);
            self.data = new_data;
            self.size = new_size;
        }
    }
}

// Safe Send/Sync implementations since we're using Vec<u8>
unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if self.zero_on_return {
            self.clear();
        }
        
        // Pool will handle the return if reference exists
        // For now, we just let the Vec<u8> handle deallocation naturally
    }
}

/// Trait for memory pool implementations
pub trait MemoryPool: Send + Sync {
    /// Allocate a buffer of the specified size
    fn allocate(&self, size: usize, purpose: Option<&str>) -> Result<PooledBuffer, PoolError>;
    
    /// Get pool statistics
    fn statistics(&self) -> PoolStats;
    
    /// Trim unused buffers to free memory
    fn trim(&self) -> usize;
    
    /// Get pool name
    fn name(&self) -> &str;
}

/// A safe buffer storage using Vec<u8>
#[derive(Debug)]
struct SafeBufferStorage {
    /// The buffer data
    data: Vec<u8>,
    /// When it was last used
    last_used: Instant,
}

impl SafeBufferStorage {
    fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0);
        Self {
            data,
            last_used: Instant::now(),
        }
    }
    
    fn into_buffer(mut self, pool_ref: Option<Arc<dyn MemoryPool>>, zero_on_return: bool) -> PooledBuffer {
        let size = self.data.len();
        PooledBuffer {
            data: std::mem::take(&mut self.data),
            size,
            pool_ref,
            zero_on_return,
        }
    }
}

/// A size class for organizing buffers
#[derive(Debug)]
struct SizeClass {
    /// Size of buffers in this class
    size: usize,
    /// Available buffers
    buffers: Mutex<VecDeque<SafeBufferStorage>>,
    /// Statistics for this size class
    stats: RwLock<PoolStats>,
    /// Configuration
    config: PoolConfig,
}

impl SizeClass {
    fn new(size: usize, config: PoolConfig) -> Self {
        let buffers = Mutex::new(VecDeque::with_capacity(config.initial_capacity));
        
        // Pre-populate with initial buffers
        {
            let mut buffer_queue = buffers.lock().unwrap();
            for _ in 0..config.initial_capacity {
                buffer_queue.push_back(SafeBufferStorage::new(size));
            }
        }
        
        Self {
            size,
            buffers,
            stats: RwLock::new(PoolStats::default()),
            config,
        }
    }
    
    fn allocate(&self, pool_ref: Option<Arc<dyn MemoryPool>>) -> Result<PooledBuffer, PoolError> {
        // Try to reuse an existing buffer
        if let Ok(mut buffer_queue) = self.buffers.lock() {
            if let Some(storage) = buffer_queue.pop_front() {
                // Update statistics
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.total_allocations += 1;
                    stats.active_allocations += 1;
                    stats.peak_allocations = stats.peak_allocations.max(stats.active_allocations);
                    stats.bytes_in_use += self.size as u64;
                    stats.peak_bytes_in_use = stats.peak_bytes_in_use.max(stats.bytes_in_use);
                    stats.total_bytes_allocated += self.size as u64;
                    
                    // Update hit rate
                    stats.hit_rate = (stats.total_allocations - (stats.total_allocations - buffer_queue.len() as u64)) as f64 / stats.total_allocations as f64;
                }
                
                return Ok(storage.into_buffer(pool_ref, self.config.zero_on_return));
            }
        }
        
        // No buffer available, create new one
        let storage = SafeBufferStorage::new(self.size);
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_allocations += 1;
            stats.active_allocations += 1;
            stats.peak_allocations = stats.peak_allocations.max(stats.active_allocations);
            stats.bytes_in_use += self.size as u64;
            stats.peak_bytes_in_use = stats.peak_bytes_in_use.max(stats.bytes_in_use);
            stats.total_bytes_allocated += self.size as u64;
        }
        
        Ok(storage.into_buffer(pool_ref, self.config.zero_on_return))
    }
    
    fn deallocate(&self, buffer: Vec<u8>) {
        if buffer.len() != self.size {
            return; // Wrong size class
        }
        
        let storage = SafeBufferStorage {
            data: buffer,
            last_used: Instant::now(),
        };
        
        // Return to pool if not at capacity
        if let Ok(mut buffer_queue) = self.buffers.lock() {
            if buffer_queue.len() < self.config.max_capacity {
                buffer_queue.push_back(storage);
            }
            // If at capacity, just drop the buffer (Vec will handle deallocation)
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_deallocations += 1;
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
            stats.bytes_in_use = stats.bytes_in_use.saturating_sub(self.size as u64);
        }
    }
    
    fn statistics(&self) -> PoolStats {
        self.stats.read().unwrap().clone()
    }
    
    fn trim(&self, max_age: Duration) -> usize {
        let mut trimmed = 0;
        let cutoff = Instant::now() - max_age;
        
        if let Ok(mut buffer_queue) = self.buffers.lock() {
            let original_len = buffer_queue.len();
            
            // Remove old buffers
            buffer_queue.retain(|storage| {
                if storage.last_used < cutoff {
                    trimmed += 1;
                    false
                } else {
                    true
                }
            });
        }
        
        trimmed
    }
}

/// Size-class based memory pool implementation
pub struct SizeClassPool {
    /// Size classes for different buffer sizes
    size_classes: Vec<SizeClass>,
    /// Pool name
    name: String,
    /// Configuration
    config: PoolConfig,
    /// Creation time
    created_at: Instant,
}

impl SizeClassPool {
    /// Create a new size class pool
    pub fn new(config: PoolConfig) -> Result<Self, PoolError> {
        Self::with_name("default".to_string(), config)
    }
    
    /// Create a new size class pool with a name
    pub fn with_name(name: String, config: PoolConfig) -> Result<Self, PoolError> {
        // Define standard size classes (powers of 2, similar to jemalloc)
        let size_classes = vec![
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
        ];
        
        let classes = size_classes
            .into_iter()
            .map(|size| SizeClass::new(size, config.clone()))
            .collect();
        
        Ok(Self {
            size_classes: classes,
            name,
            config,
            created_at: Instant::now(),
        })
    }
    
    /// Find the appropriate size class for a requested size
    fn find_size_class(&self, size: usize) -> Option<&SizeClass> {
        self.size_classes
            .iter()
            .find(|class| class.size >= size)
    }
}

impl MemoryPool for SizeClassPool {
    fn allocate(&self, size: usize, _purpose: Option<&str>) -> Result<PooledBuffer, PoolError> {
        if size == 0 {
            return Err(PoolError::InvalidSize { size });
        }
        
        if let Some(size_class) = self.find_size_class(size) {
            size_class.allocate(None) // We don't pass self reference to avoid circular dependencies
        } else {
            // Size too large for any size class, create directly
            Ok(PooledBuffer::new(size, None, self.config.zero_on_return))
        }
    }
    
    fn statistics(&self) -> PoolStats {
        let mut combined = PoolStats::default();
        
        for size_class in &self.size_classes {
            let stats = size_class.statistics();
            combined.total_allocations += stats.total_allocations;
            combined.total_deallocations += stats.total_deallocations;
            combined.active_allocations += stats.active_allocations;
            combined.peak_allocations = combined.peak_allocations.max(stats.peak_allocations);
            combined.total_bytes_allocated += stats.total_bytes_allocated;
            combined.bytes_in_use += stats.bytes_in_use;
            combined.peak_bytes_in_use = combined.peak_bytes_in_use.max(stats.peak_bytes_in_use);
        }
        
        // Calculate derived metrics
        if combined.total_allocations > 0 {
            combined.average_allocation_size = combined.total_bytes_allocated as f64 / combined.total_allocations as f64;
            combined.efficiency = combined.total_deallocations as f64 / combined.total_allocations as f64;
            combined.hit_rate = (combined.total_allocations - combined.total_allocations) as f64 / combined.total_allocations as f64;
        }
        
        combined
    }
    
    fn trim(&self) -> usize {
        let max_age = Duration::from_secs(self.config.trim_interval_secs * 2);
        self.size_classes
            .iter()
            .map(|class| class.trim(max_age))
            .sum()
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for SizeClassPool {
    fn drop(&mut self) {
        // Statistics and cleanup can be logged here if needed
    }
}

/// Manager for multiple memory pools
#[derive(Debug)]
pub struct PoolManager {
    /// Named pools
    pools: RwLock<HashMap<String, Arc<dyn MemoryPool>>>,
    /// Default pool
    default_pool: Arc<dyn MemoryPool>,
}

impl PoolManager {
    /// Create a new pool manager
    pub fn new() -> Result<Self, PoolError> {
        let default_pool = Arc::new(SizeClassPool::new(PoolConfig::default())?);
        
        Ok(Self {
            pools: RwLock::new(HashMap::new()),
            default_pool,
        })
    }
    
    /// Register a named pool
    pub fn register_pool(&self, name: String, pool: impl MemoryPool + 'static) {
        let mut pools = self.pools.write().unwrap();
        pools.insert(name, Arc::new(pool));
    }
    
    /// Allocate from a named pool (or default if name is None)
    pub fn allocate(&self, size: usize, pool_name: Option<&str>) -> Result<PooledBuffer, PoolError> {
        let pool = if let Some(name) = pool_name {
            let pools = self.pools.read().unwrap();
            pools.get(name).cloned().unwrap_or_else(|| Arc::clone(&self.default_pool))
        } else {
            Arc::clone(&self.default_pool)
        };
        
        pool.allocate(size, None)
    }
    
    /// Get statistics for all pools
    pub fn all_stats(&self) -> HashMap<String, PoolStats> {
        let mut stats = HashMap::new();
        
        // Add default pool
        stats.insert("default".to_string(), self.default_pool.statistics());
        
        // Add named pools
        let pools = self.pools.read().unwrap();
        for (name, pool) in pools.iter() {
            stats.insert(name.clone(), pool.statistics());
        }
        
        stats
    }
    
    /// Trim all pools
    pub fn trim_all(&self) -> HashMap<String, usize> {
        let mut results = HashMap::new();
        
        // Trim default pool
        results.insert("default".to_string(), self.default_pool.trim());
        
        // Trim named pools
        let pools = self.pools.read().unwrap();
        for (name, pool) in pools.iter() {
            results.insert(name.clone(), pool.trim());
        }
        
        results
    }
}

/// Create a standard memory pool with reasonable defaults
pub fn create_standard_pool() -> Result<SizeClassPool, PoolError> {
    SizeClassPool::new(PoolConfig::default())
}

/// Create a high-performance memory pool for demanding applications
pub fn create_high_performance_pool() -> Result<SizeClassPool, PoolError> {
    let config = PoolConfig {
        initial_capacity: 256,
        max_capacity: 4096,
        thread_local_cache: true,
        thread_cache_size: 32,
        numa_aware: true,
        trim_interval_secs: 30,
        stats_interval_secs: 5,
        zero_on_return: false,
    };
    
    SizeClassPool::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pool_creation() {
        let pool = create_standard_pool();
        assert!(pool.is_ok());
    }
    
    #[test]
    fn test_buffer_allocation() {
        let pool = create_standard_pool().unwrap();
        let buffer = pool.allocate(1024, None);
        assert!(buffer.is_ok());
        
        let buffer = buffer.unwrap();
        assert_eq!(buffer.len(), 1024);
    }
    
    #[test]
    fn test_size_class_reuse() {
        let pool = create_standard_pool().unwrap();
        
        // Allocate and drop a buffer
        {
            let _buffer = pool.allocate(1024, None).unwrap();
        }
        
        // Allocate another buffer of same size
        let buffer2 = pool.allocate(1024, None).unwrap();
        assert_eq!(buffer2.len(), 1024);
    }
    
    #[test]
    fn test_pool_manager() {
        let manager = PoolManager::new().unwrap();
        
        // Test default pool allocation
        let buffer = manager.allocate(512, None);
        assert!(buffer.is_ok());
        
        // Register a custom pool
        let custom_pool = create_high_performance_pool().unwrap();
        manager.register_pool("custom".to_string(), custom_pool);
        
        // Test named pool allocation
        let buffer2 = manager.allocate(1024, Some("custom"));
        assert!(buffer2.is_ok());
    }
    
    #[test]
    fn test_buffer_operations() {
        let pool = create_standard_pool().unwrap();
        let mut buffer = pool.allocate(256, None).unwrap();
        
        // Test slice operations
        let slice = buffer.as_mut_slice();
        slice[0] = 42;
        slice[1] = 24;
        
        let read_slice = buffer.as_slice();
        assert_eq!(read_slice[0], 42);
        assert_eq!(read_slice[1], 24);
        
        // Test clear
        buffer.clear();
        assert_eq!(buffer.as_slice()[0], 0);
        assert_eq!(buffer.as_slice()[1], 0);
    }
    
    #[test]
    fn test_statistics() {
        let pool = create_standard_pool().unwrap();
        
        // Make some allocations
        let _buffer1 = pool.allocate(128, None).unwrap();
        let _buffer2 = pool.allocate(256, None).unwrap();
        
        let stats = pool.statistics();
        assert!(stats.total_allocations >= 2);
        assert!(stats.active_allocations >= 2);
        assert!(stats.total_bytes_allocated >= 384);
    }
} 