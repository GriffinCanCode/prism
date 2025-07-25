//! Common types, traits, and configurations for the allocator subsystem
//!
//! This module contains shared types and constants used across different
//! allocator implementations.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, AtomicBool};
use std::sync::{Mutex, Arc};
use std::alloc::Layout;
use std::time::Instant;
use std::thread;
use std::collections::HashMap;

/// Size classes optimized for common allocation patterns
/// Based on research from Go's runtime, TCMalloc, and jemalloc
pub const SIZE_CLASSES: &[usize] = &[
    8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 896, 1024, 1152, 1280,
    1408, 1536, 1792, 2048, 2304, 2688, 3072, 3200, 3456, 4096, 4864, 5376, 6144, 6528,
    6784, 6912, 8192, 9472, 9728, 10240, 10880, 12288, 13568, 14336, 16384, 18432, 19072,
    20480, 21760, 24576, 27264, 28672, 32768,
];

/// Large object threshold - objects larger than this bypass size classes
pub const LARGE_OBJECT_THRESHOLD: usize = 32768; // 32KB

/// Thread-local cache size per size class
pub const THREAD_CACHE_SIZE: usize = 64;

/// Central cache batch size for refill/drain operations
pub const CENTRAL_CACHE_BATCH: usize = 32;

/// Page size for memory management
pub const PAGE_SIZE: usize = 8192; // 8KB

/// Memory page representation
#[derive(Debug, Clone)]
pub struct Page {
    /// Start address
    pub addr: NonNull<u8>,
    /// Page count
    pub count: usize,
    /// NUMA node (future use)
    pub numa_node: usize,
}

/// Memory span - a contiguous region of memory for a size class
#[derive(Debug, Clone)]
pub struct Span {
    /// Start address of the span
    pub start: NonNull<u8>,
    /// Size in pages
    pub page_count: usize,
    /// Size class this span serves
    pub size_class_index: usize,
    /// Number of allocated objects in this span
    pub allocated_objects: AtomicUsize,
    /// Total objects this span can hold
    pub total_objects: usize,
    /// Free objects linked list head
    pub free_objects: std::sync::atomic::AtomicPtr<u8>,
}

/// Information about a large allocation
#[derive(Debug, Clone)]
pub struct LargeAllocation {
    /// Size of the allocation
    pub size: usize,
    /// Layout used for allocation
    pub layout: Layout,
    /// Allocation timestamp
    pub allocated_at: Instant,
}

/// Single allocation region for bump allocation
#[derive(Debug)]
pub struct AllocationRegion {
    /// Start of the region
    pub start: NonNull<u8>,
    /// Current allocation pointer
    pub current: std::sync::atomic::AtomicPtr<u8>,
    /// End of the region
    pub end: *mut u8,
    /// Size of this region
    pub size: usize,
    /// Number of objects allocated in this region
    pub object_count: AtomicUsize,
    /// Whether this region is active for allocation
    pub active: AtomicBool,
}

/// General allocator configuration
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Enable thread-local caching
    pub enable_thread_cache: bool,
    /// Maximum memory before triggering GC
    pub gc_trigger_threshold: usize,
    /// Enable NUMA awareness
    pub numa_aware: bool,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            enable_thread_cache: true,
            gc_trigger_threshold: 64 * 1024 * 1024, // 64MB
            numa_aware: false, // Disable by default for simplicity
        }
    }
}

/// Configuration specific to PrismAllocator
#[derive(Debug, Clone)]
pub struct PrismAllocatorConfig {
    /// Base allocator configuration
    pub base: AllocatorConfig,
    /// Enable size class segregation
    pub enable_size_classes: bool,
    /// Maximum thread cache size per size class
    pub max_thread_cache_size: usize,
    /// Central cache batch size for refill operations
    pub central_cache_batch_size: usize,
}

impl Default for PrismAllocatorConfig {
    fn default() -> Self {
        Self {
            base: AllocatorConfig::default(),
            enable_size_classes: true,
            max_thread_cache_size: THREAD_CACHE_SIZE,
            central_cache_batch_size: CENTRAL_CACHE_BATCH,
        }
    }
}

/// Configuration for bump allocator
#[derive(Debug, Clone)]
pub struct BumpAllocatorConfig {
    /// Initial region size
    pub initial_region_size: usize,
    /// Maximum region size
    pub max_region_size: usize,
    /// Alignment for all allocations
    pub default_alignment: usize,
    /// Whether to zero memory on allocation
    pub zero_memory: bool,
}

impl Default for BumpAllocatorConfig {
    fn default() -> Self {
        Self {
            initial_region_size: 1024 * 1024, // 1MB
            max_region_size: 16 * 1024 * 1024, // 16MB
            default_alignment: 8,
            zero_memory: false, // GC will handle initialization
        }
    }
}

/// General allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub total_allocated: usize,
    pub total_deallocated: usize,
    pub live_bytes: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub peak_memory: usize,
    pub large_object_count: usize,
    pub large_object_bytes: usize,
    pub page_bytes: usize,
    pub metadata_memory: usize,
    pub memory_overhead: usize,
    pub barrier_calls: usize,
}

impl AllocationStats {
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            total_deallocated: 0,
            live_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            peak_memory: 0,
            large_object_count: 0,
            large_object_bytes: 0,
            page_bytes: 0,
            metadata_memory: 0,
            memory_overhead: 0,
            barrier_calls: 0,
        }
    }
    
    pub fn combine(stats_list: Vec<AllocationStats>) -> Self {
        let mut combined = AllocationStats::new();
        
        for stats in stats_list {
            combined.total_allocated += stats.total_allocated;
            combined.total_deallocated += stats.total_deallocated;
            combined.live_bytes += stats.live_bytes;
            combined.allocation_count += stats.allocation_count;
            combined.deallocation_count += stats.deallocation_count;
            combined.peak_memory = combined.peak_memory.max(stats.peak_memory);
            combined.large_object_count += stats.large_object_count;
            combined.large_object_bytes += stats.large_object_bytes;
            combined.page_bytes += stats.page_bytes;
            combined.metadata_memory += stats.metadata_memory;
            combined.memory_overhead += stats.memory_overhead;
            combined.barrier_calls += stats.barrier_calls;
        }
        
        combined
    }
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub total_allocated: usize,
    pub live_bytes: usize,
    pub metadata_overhead: usize,
    pub fragmentation_ratio: f64,
}

/// Generation for generational garbage collection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    /// Young generation (nursery) - for newly allocated objects
    Young,
    /// Old generation (tenured) - for long-lived objects
    Old,
    /// Permanent generation - for objects that never die
    Permanent,
}

/// Statistics for bump allocator performance monitoring
#[derive(Debug, Default, Clone)]
pub struct BumpAllocationStats {
    /// Total number of allocations
    pub total_allocations: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Number of allocation regions created
    pub regions_created: usize,
    /// Number of failed allocations (requiring GC)
    pub allocation_failures: usize,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Last allocation timestamp
    pub last_allocation_time: Instant,
}

/// Statistics for central cache
#[derive(Debug, Default)]
pub struct CentralCacheStats {
    /// Total allocations from this cache
    pub allocations: AtomicUsize,
    /// Total deallocations to this cache
    pub deallocations: AtomicUsize,
    /// Cache hits
    pub cache_hits: AtomicUsize,
    /// Cache misses
    pub cache_misses: AtomicUsize,
}

/// Object header used by GC system
/// This is referenced from the original allocator but defined here for clarity
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    pub size: usize,
    pub type_id: u32,
    pub mark_bits: u8,
    pub age: u8,
}

impl ObjectHeader {
    pub fn new(size: usize, type_id: u32) -> Self {
        Self {
            size,
            type_id,
            mark_bits: 0,
            age: 0,
        }
    }
}

/// Allocation information for tracking
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub address: *const u8,
    pub size: usize,
    pub timestamp: Instant,
    pub thread_id: thread::ThreadId,
}

/// Utility functions for size class management
pub struct SizeClassUtils;

impl SizeClassUtils {
    /// Find the appropriate size class for the given size
    pub fn find_size_class(size: usize) -> Option<usize> {
        SIZE_CLASSES.iter().position(|&class_size| size <= class_size)
    }
    
    /// Get the actual size for a given size class index
    pub fn get_size_for_class(index: usize) -> Option<usize> {
        SIZE_CLASSES.get(index).copied()
    }
    
    /// Calculate number of objects that fit in a page for a given size class
    pub fn objects_per_page(size_class_index: usize) -> usize {
        if let Some(object_size) = Self::get_size_for_class(size_class_index) {
            PAGE_SIZE / object_size
        } else {
            0
        }
    }
    
    /// Check if a size should be handled as a large object
    pub fn is_large_object(size: usize) -> bool {
        size > LARGE_OBJECT_THRESHOLD
    }
}

/// Error types for allocator operations
#[derive(Debug, Clone)]
pub enum AllocatorError {
    OutOfMemory,
    InvalidSize,
    InvalidAlignment,
    InvalidPointer,
    ConfigurationError(String),
}

impl std::fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocatorError::OutOfMemory => write!(f, "Out of memory"),
            AllocatorError::InvalidSize => write!(f, "Invalid allocation size"),
            AllocatorError::InvalidAlignment => write!(f, "Invalid alignment requirement"),
            AllocatorError::InvalidPointer => write!(f, "Invalid pointer for deallocation"),
            AllocatorError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for AllocatorError {}

/// Result type for allocator operations
pub type AllocatorResult<T> = Result<T, AllocatorError>;

/// Detailed memory breakdown for bump allocator
#[derive(Debug, Clone)]
pub struct BumpMemoryBreakdown {
    /// Total number of regions
    pub total_regions: usize,
    /// Index of currently active region
    pub active_region_index: usize,
    /// Total capacity across all regions
    pub total_capacity: usize,
    /// Currently allocated bytes
    pub allocated_bytes: usize,
    /// Peak allocated bytes
    pub peak_bytes: usize,
    /// Details for each region
    pub region_details: Vec<RegionDetail>,
}

/// Details for a specific allocation region
#[derive(Debug, Clone)]
pub struct RegionDetail {
    /// Region index
    pub region_index: usize,
    /// Total size of the region
    pub size: usize,
    /// Bytes allocated in this region
    pub allocated: usize,
    /// Bytes free in this region
    pub free: usize,
    /// Number of objects in this region
    pub object_count: usize,
    /// Whether this region is active for allocation
    pub is_active: bool,
} 