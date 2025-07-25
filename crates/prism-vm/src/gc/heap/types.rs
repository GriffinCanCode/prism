//! Core types and data structures for the heap subsystem
//!
//! This module defines all the shared types used across the heap subsystem,
//! with clear separation from other GC components:
//!
//! **Heap Responsibilities:**
//! - Object tracking and metadata management
//! - Free list organization by size classes
//! - Memory region management for cache locality
//! - Heap statistics and monitoring
//! - Fragmentation tracking and compaction coordination
//!
//! **NOT Heap Responsibilities (delegated to other components):**
//! - Raw memory allocation (handled by allocators::*)
//! - Write barriers (handled by barriers::*)
//! - Object marking/tracing (handled by collectors::*)
//! - Root set management (handled by roots::*)

use std::sync::atomic::{AtomicUsize, AtomicPtr, AtomicU8, Ordering};
use std::collections::HashMap;
use std::time::Instant;
use std::ptr::NonNull;

// Re-export ObjectHeader and ObjectColor from the main GC module
// These are shared across all GC components
pub use crate::gc::{ObjectHeader, ObjectColor};

/// Configuration for heap behavior
#[derive(Debug, Clone)]
pub struct HeapConfig {
    /// Type of heap to create
    pub heap_type: HeapType,
    /// Total heap capacity in bytes
    pub capacity: usize,
    /// Large object threshold (objects above this size get special handling)
    pub large_object_threshold: usize,
    /// Size classes for segregated free lists
    pub size_classes: Vec<usize>,
    /// Enable card table for generational GC
    pub enable_card_table: bool,
    /// Card size for card table (if enabled)
    pub card_size: usize,
    /// Enable memory region tracking
    pub enable_memory_regions: bool,
    /// Maximum number of memory regions
    pub max_memory_regions: usize,
    /// Enable fragmentation monitoring
    pub enable_fragmentation_monitoring: bool,
    /// Fragmentation compaction threshold (0.0-1.0)
    pub compaction_threshold: f64,
}

impl Default for HeapConfig {
    fn default() -> Self {
        Self {
            heap_type: HeapType::Standard,
            capacity: 64 * 1024 * 1024, // 64MB
            large_object_threshold: 32 * 1024, // 32KB
            size_classes: DEFAULT_SIZE_CLASSES.to_vec(),
            enable_card_table: false,
            card_size: 512, // 512 bytes per card
            enable_memory_regions: true,
            max_memory_regions: 256,
            enable_fragmentation_monitoring: true,
            compaction_threshold: 0.25, // Compact when 25% fragmented
        }
    }
}

/// Types of heaps available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeapType {
    /// Standard heap with size class segregation
    Standard,
    /// Regional heap for generational collection
    Regional,
}

/// Default size classes optimized for common allocation patterns
/// These are aligned with the allocator size classes to avoid conflicts
pub const DEFAULT_SIZE_CLASSES: &[usize] = &[
    8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768
];

/// Heap statistics for monitoring and decision making
#[derive(Debug, Clone)]
pub struct HeapStats {
    /// Total bytes allocated in the heap
    pub total_allocated: usize,
    /// Number of live objects currently tracked
    pub live_objects: usize,
    /// Free space available (not necessarily contiguous)
    pub free_space: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = maximum fragmentation)
    pub fragmentation_ratio: f64,
    /// Current allocation rate (bytes per second)
    pub allocation_rate: f64,
    /// GC overhead as percentage of total execution time
    pub gc_overhead: f64,
    /// Memory region statistics
    pub region_stats: Option<RegionStats>,
    /// Card table statistics (if enabled)
    pub card_table_stats: Option<CardTableStats>,
}

/// Statistics for memory regions
#[derive(Debug, Clone)]
pub struct RegionStats {
    /// Number of active regions
    pub active_regions: usize,
    /// Average region utilization (0.0-1.0)
    pub average_utilization: f64,
    /// Hottest region utilization
    pub max_utilization: f64,
    /// Coldest region utilization
    pub min_utilization: f64,
}

/// Statistics for card table
#[derive(Debug, Clone)]
pub struct CardTableStats {
    /// Total number of cards
    pub total_cards: usize,
    /// Number of dirty cards
    pub dirty_cards: usize,
    /// Card table memory overhead in bytes
    pub memory_overhead: usize,
}

/// Memory region for organizing objects with similar characteristics
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Unique identifier for this region
    pub id: usize,
    /// Starting address of the region
    pub start: NonNull<u8>,
    /// Size of the region in bytes
    pub size: usize,
    /// Currently allocated bytes in this region
    pub allocated: usize,
    /// Number of objects in this region
    pub object_count: usize,
    /// Average object age in this region (for generational promotion)
    pub average_age: f64,
    /// Region type (young, old, large, etc.)
    pub region_type: RegionType,
    /// Last access timestamp for LRU tracking
    pub last_access: Instant,
}

/// Types of memory regions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegionType {
    /// Young generation objects (frequently collected)
    Young,
    /// Old generation objects (infrequently collected)
    Old,
    /// Large objects (special handling)
    Large,
    /// Permanent objects (never collected)
    Permanent,
}

/// Allocation statistics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct AllocationStats {
    /// Total number of allocations
    pub total_allocations: usize,
    /// Total bytes allocated over time
    pub total_bytes_allocated: usize,
    /// Number of allocations served from free lists
    pub free_list_hits: usize,
    /// Number of allocations requiring new memory
    pub new_allocations: usize,
    /// Current fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Last allocation timestamp
    pub last_allocation: Option<Instant>,
}

impl AllocationStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record a new allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocations += 1;
        self.total_bytes_allocated += size;
        self.last_allocation = Some(Instant::now());
        self.update_average_allocation_size();
    }
    
    /// Record a free list hit
    pub fn record_free_list_hit(&mut self) {
        self.free_list_hits += 1;
    }
    
    /// Record a new memory allocation
    pub fn record_new_allocation(&mut self) {
        self.new_allocations += 1;
    }
    
    /// Update the average allocation size
    fn update_average_allocation_size(&mut self) {
        if self.total_allocations > 0 {
            self.average_allocation_size = 
                self.total_bytes_allocated as f64 / self.total_allocations as f64;
        }
    }
    
    /// Update fragmentation ratio
    pub fn update_fragmentation_ratio(&mut self, ratio: f64) {
        self.fragmentation_ratio = ratio.clamp(0.0, 1.0);
    }
    
    /// Update peak memory usage
    pub fn update_peak_memory(&mut self, current_usage: usize) {
        if current_usage > self.peak_memory_usage {
            self.peak_memory_usage = current_usage;
        }
    }
}

/// Free list entry for size class management
#[derive(Debug, Clone)]
pub struct FreeListEntry {
    /// Pointer to the free memory block
    pub ptr: NonNull<u8>,
    /// Size of the free block
    pub size: usize,
}

/// Size class information
#[derive(Debug, Clone)]
pub struct SizeClass {
    /// Size of objects in this class
    pub object_size: usize,
    /// Index of this size class
    pub index: usize,
    /// Free list for this size class
    pub free_list: Vec<FreeListEntry>,
    /// Statistics for this size class
    pub stats: SizeClassStats,
}

/// Statistics for a specific size class
#[derive(Debug, Default, Clone)]
pub struct SizeClassStats {
    /// Number of allocations from this size class
    pub allocations: usize,
    /// Number of deallocations to this size class
    pub deallocations: usize,
    /// Current number of free objects
    pub free_objects: usize,
    /// Total bytes managed by this size class
    pub total_bytes: usize,
    /// Utilization ratio (0.0-1.0)
    pub utilization: f64,
}

/// Large object entry for special handling
#[derive(Debug, Clone)]
pub struct LargeObjectEntry {
    /// Pointer to the large object
    pub ptr: NonNull<u8>,
    /// Size of the large object
    pub size: usize,
    /// Allocation timestamp
    pub allocated_at: Instant,
    /// Last access timestamp
    pub last_access: Instant,
}

/// Fragmentation information
#[derive(Debug, Clone)]
pub struct FragmentationInfo {
    /// Total free space
    pub total_free_space: usize,
    /// Number of free blocks
    pub free_block_count: usize,
    /// Largest free block size
    pub largest_free_block: usize,
    /// Average free block size
    pub average_free_block_size: f64,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = maximum)
    pub fragmentation_ratio: f64,
}

impl FragmentationInfo {
    /// Calculate fragmentation ratio based on free space distribution
    pub fn calculate_fragmentation_ratio(
        total_free_space: usize,
        free_block_count: usize,
    ) -> f64 {
        if total_free_space == 0 || free_block_count <= 1 {
            return 0.0;
        }
        
        let average_block_size = total_free_space as f64 / free_block_count as f64;
        let max_possible_block_size = total_free_space as f64;
        
        // Fragmentation increases as we have more, smaller blocks
        1.0 - (average_block_size / max_possible_block_size)
    }
}

/// Card table entry for generational GC
#[derive(Debug, Clone, Copy)]
pub struct Card {
    /// Card state (clean, dirty, etc.)
    pub state: CardState,
    /// Number of references from this card to younger generations
    pub cross_generation_refs: u16,
}

/// States a card can be in
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CardState {
    /// Card is clean (no writes since last GC)
    Clean,
    /// Card is dirty (has been written to)
    Dirty,
    /// Card is being processed
    Processing,
}

/// Span information for memory organization
#[derive(Debug)]
pub struct Span {
    /// Starting address of the span
    pub start: NonNull<u8>,
    /// Number of pages in this span
    pub page_count: usize,
    /// Size class index for objects in this span
    pub size_class_index: usize,
    /// Number of allocated objects in this span
    pub allocated_objects: AtomicUsize,
    /// Total number of objects this span can hold
    pub total_objects: usize,
    /// Free objects linked list head
    pub free_objects: AtomicPtr<u8>,
}

/// Utility functions for size class management
pub struct SizeClassUtils;

impl SizeClassUtils {
    /// Find the appropriate size class for a given size
    pub fn find_size_class(size: usize) -> Option<usize> {
        DEFAULT_SIZE_CLASSES
            .iter()
            .position(|&class_size| size <= class_size)
    }
    
    /// Get the size for a given size class index
    pub fn get_size_class_size(index: usize) -> Option<usize> {
        DEFAULT_SIZE_CLASSES.get(index).copied()
    }
    
    /// Calculate the number of objects that fit in a page for a size class
    pub fn objects_per_page(size_class_size: usize, page_size: usize) -> usize {
        if size_class_size == 0 {
            0
        } else {
            page_size / size_class_size
        }
    }
}

/// Configuration constants
pub const PAGE_SIZE: usize = 8192; // 8KB pages
pub const LARGE_OBJECT_THRESHOLD: usize = 32 * 1024; // 32KB
pub const DEFAULT_CARD_SIZE: usize = 512; // 512 byte cards
pub const MAX_SIZE_CLASSES: usize = 64;

/// Validation functions for heap integrity
pub struct HeapValidator;

impl HeapValidator {
    /// Validate that a pointer is within heap bounds
    pub fn is_valid_heap_pointer(ptr: *const u8, heap_start: *const u8, heap_size: usize) -> bool {
        let ptr_addr = ptr as usize;
        let heap_start_addr = heap_start as usize;
        let heap_end_addr = heap_start_addr + heap_size;
        
        ptr_addr >= heap_start_addr && ptr_addr < heap_end_addr
    }
    
    /// Validate size class index
    pub fn is_valid_size_class_index(index: usize) -> bool {
        index < DEFAULT_SIZE_CLASSES.len()
    }
    
    /// Validate object size
    pub fn is_valid_object_size(size: usize) -> bool {
        size > 0 && size <= LARGE_OBJECT_THRESHOLD * 2 // Allow some headroom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_utils() {
        assert_eq!(SizeClassUtils::find_size_class(1), Some(0)); // Maps to 8
        assert_eq!(SizeClassUtils::find_size_class(8), Some(0)); // Exact match
        assert_eq!(SizeClassUtils::find_size_class(9), Some(1)); // Maps to 16
        assert_eq!(SizeClassUtils::find_size_class(32768), Some(23)); // Largest class
        assert_eq!(SizeClassUtils::find_size_class(32769), None); // Too large
    }

    #[test]
    fn test_fragmentation_calculation() {
        // No fragmentation - single large block
        let ratio = FragmentationInfo::calculate_fragmentation_ratio(1000, 1);
        assert_eq!(ratio, 0.0);
        
        // Maximum fragmentation - many tiny blocks
        let ratio = FragmentationInfo::calculate_fragmentation_ratio(1000, 1000);
        assert!(ratio > 0.9); // Should be close to 1.0
        
        // Moderate fragmentation
        let ratio = FragmentationInfo::calculate_fragmentation_ratio(1000, 10);
        assert!(ratio > 0.0 && ratio < 1.0);
    }

    #[test]
    fn test_allocation_stats() {
        let mut stats = AllocationStats::new();
        
        stats.record_allocation(64);
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes_allocated, 64);
        assert_eq!(stats.average_allocation_size, 64.0);
        
        stats.record_allocation(128);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_bytes_allocated, 192);
        assert_eq!(stats.average_allocation_size, 96.0);
    }

    #[test]
    fn test_heap_validator() {
        let heap_start = 0x1000 as *const u8;
        let heap_size = 0x1000; // 4KB
        
        // Valid pointer within heap
        let valid_ptr = 0x1500 as *const u8;
        assert!(HeapValidator::is_valid_heap_pointer(valid_ptr, heap_start, heap_size));
        
        // Invalid pointer before heap
        let invalid_ptr = 0x500 as *const u8;
        assert!(!HeapValidator::is_valid_heap_pointer(invalid_ptr, heap_start, heap_size));
        
        // Invalid pointer after heap
        let invalid_ptr = 0x3000 as *const u8;
        assert!(!HeapValidator::is_valid_heap_pointer(invalid_ptr, heap_start, heap_size));
    }
} 