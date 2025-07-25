//! Core Heap Implementation
//!
//! This module provides the main Heap implementation that coordinates
//! all the heap subsystem components.

use super::*;
use super::types::*;
use super::size_class::SizeClassAllocator;
use super::large_object::LargeObjectAllocator as HeapLargeObjectAllocator;
use super::memory_regions::MemoryRegionManager;
use super::fragmentation::FragmentationManager;
use super::card_table::CardTable;
use super::statistics::StatisticsCollector;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::ptr::NonNull;

/// Main heap implementation using modular components
pub struct Heap {
    /// Map from object pointer to object header
    objects: RwLock<HashMap<*const u8, crate::ObjectHeader>>,
    
    /// Total heap capacity
    capacity: usize,
    
    /// Currently allocated bytes
    allocated: AtomicUsize,
    
    /// Size class allocator for segregated free lists
    size_class_allocator: SizeClassAllocator,
    
    /// Large object allocator
    large_object_allocator: HeapLargeObjectAllocator,
    
    /// Memory region manager (optional)
    region_manager: Option<MemoryRegionManager>,
    
    /// Fragmentation manager
    fragmentation_manager: FragmentationManager,
    
    /// Card table (optional)
    card_table: Option<CardTable>,
    
    /// Statistics collector
    statistics: StatisticsCollector,
    
    /// Configuration
    config: HeapConfig,
}

impl Heap {
    /// Create a new heap with default configuration
    pub fn new(capacity: usize) -> Self {
        let config = HeapConfig {
            capacity,
            ..Default::default()
        };
        Self::with_config(config)
    }
    
    /// Create a heap with custom configuration
    pub fn with_config(config: HeapConfig) -> Self {
        let size_class_allocator = SizeClassAllocator::new();
        let large_object_allocator = HeapLargeObjectAllocator::new();
        
        let region_manager = if config.enable_memory_regions {
            Some(MemoryRegionManager::new(config.max_memory_regions))
        } else {
            None
        };
        
        let fragmentation_manager = FragmentationManager::new(config.compaction_threshold);
        
        // Initialize card table with proper heap base address
        let card_table = if config.enable_card_table {
            // For now, use a placeholder address that will be updated when actual heap memory is allocated
            // In a real implementation, this would be coordinated with the memory allocator
            let heap_base = 0x10000000 as *const u8; // Placeholder base address
            Some(CardTable::new(
                heap_base,
                config.capacity,
                config.card_size,
            ))
        } else {
            None
        };
        
        let statistics = StatisticsCollector::new();
        
        Self {
            objects: RwLock::new(HashMap::new()),
            capacity: config.capacity,
            allocated: AtomicUsize::new(0),
            size_class_allocator,
            large_object_allocator,
            region_manager,
            fragmentation_manager,
            card_table,
            statistics,
            config,
        }
    }
}

impl HeapInterface for Heap {
    /// Register a newly allocated object with enhanced tracking
    fn register_object(&mut self, ptr: *const u8, header: crate::ObjectHeader) {
        // Add to object map
        {
            let mut objects = self.objects.write().unwrap();
            objects.insert(ptr, header);
        }
        
        // Update allocated bytes
        self.allocated.fetch_add(header.size, Ordering::Relaxed);
        
        // Update memory regions if enabled
        if let Some(ref region_manager) = self.region_manager {
            region_manager.track_allocation(ptr, header.size);
        }
        
        // Mark card table if enabled
        if let Some(ref card_table) = self.card_table {
            card_table.mark_dirty(ptr);
        }
        
        // Update statistics
        self.statistics.record_object_registration(header.size);
    }
    
    /// Get object header for a given pointer
    fn get_header(&self, ptr: *const u8) -> Option<&crate::ObjectHeader> {
        let objects = self.objects.read().unwrap();
        objects.get(&ptr)
    }
    
    /// Get mutable object header for a given pointer
    fn get_header_mut(&mut self, ptr: *const u8) -> Option<&mut crate::ObjectHeader> {
        let mut objects = self.objects.write().unwrap();
        objects.get_mut(&ptr)
    }
    
    /// Find all white (unmarked) objects for sweeping
    fn find_white_objects(&self) -> Vec<*const u8> {
        let objects = self.objects.read().unwrap();
        objects
            .iter()
            .filter_map(|(&ptr, header)| {
                if header.get_color() == ObjectColor::White {
                    Some(ptr)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Reset all object colors to white for next collection cycle
    fn reset_colors_to_white(&mut self) {
        let mut objects = self.objects.write().unwrap();
        for header in objects.values_mut() {
            header.set_color(ObjectColor::White);
        }
    }
    
    /// Enhanced deallocation with better free list management
    fn deallocate(&mut self, ptr: *const u8) {
        let header = {
            let mut objects = self.objects.write().unwrap();
            objects.remove(&ptr)
        };
        
        if let Some(header) = header {
            // Update allocated bytes
            self.allocated.fetch_sub(header.size, Ordering::Relaxed);
            
            // Add to appropriate free list based on size
            if header.size >= self.config.large_object_threshold {
                // Large objects go to large object allocator
                if let Some(ptr_nn) = NonNull::new(ptr as *mut u8) {
                    self.large_object_allocator.deallocate(ptr_nn, header.size);
                }
            } else {
                // Small/medium objects go to size class allocator
                if let Some(ptr_nn) = NonNull::new(ptr as *mut u8) {
                    self.size_class_allocator.deallocate(ptr_nn, header.size);
                }
            }
            
            // Update memory regions if enabled
            if let Some(ref region_manager) = self.region_manager {
                region_manager.track_deallocation(ptr, header.size);
            }
            
            // Update statistics
            self.statistics.record_object_deallocation(header.size);
        }
    }
    
    /// Try to allocate from free lists with size class optimization
    fn try_allocate_from_free_list(&mut self, size: usize, align: usize) -> Option<*const u8> {
        let result = if size >= self.config.large_object_threshold {
            // Try large object allocator
            self.large_object_allocator.allocate(size)
                .map(|ptr| ptr.as_ptr() as *const u8)
        } else {
            // Try size class allocator
            self.size_class_allocator.allocate(size)
                .map(|ptr| ptr.as_ptr() as *const u8)
        };
        
        if result.is_some() {
            self.statistics.record_free_list_hit();
        } else {
            self.statistics.record_free_list_miss();
        }
        
        result
    }
    
    /// Get heap statistics
    fn get_stats(&self) -> HeapStats {
        let allocated = self.allocated.load(Ordering::Relaxed);
        let objects = self.objects.read().unwrap();
        let live_objects = objects.len();
        let free_space = self.capacity.saturating_sub(allocated);
        
        // Get fragmentation information
        let size_class_frag = self.size_class_allocator.get_fragmentation_info();
        let large_obj_stats = self.large_object_allocator.get_statistics();
        
        // Calculate overall fragmentation ratio
        let fragmentation_ratio = if size_class_frag.total_free_space > 0 || large_obj_stats.fragmentation_waste > 0 {
            let total_free = size_class_frag.total_free_space + large_obj_stats.fragmentation_waste;
            let total_blocks = size_class_frag.free_block_count + 1; // +1 for large objects
            FragmentationInfo::calculate_fragmentation_ratio(total_free, total_blocks)
        } else {
            0.0
        };
        
        // Get region statistics if enabled
        let region_stats = self.region_manager.as_ref().map(|rm| rm.get_statistics());
        
        // Get card table statistics if enabled
        let card_table_stats = self.card_table.as_ref().map(|ct| ct.get_statistics());
        
        let mut stats = HeapStats {
            total_allocated: allocated,
            live_objects,
            free_space,
            fragmentation_ratio,
            allocation_rate: 0.0, // Will be updated by statistics collector
            gc_overhead: 0.0,     // Will be updated by statistics collector
            region_stats,
            card_table_stats,
        };
        
        // Update with statistics collector data
        let current_stats = self.statistics.get_current_stats();
        stats.allocation_rate = current_stats.allocation_rate;
        stats.gc_overhead = current_stats.gc_overhead;
        
        // Update fragmentation manager
        let frag_info = FragmentationInfo {
            total_free_space: size_class_frag.total_free_space,
            free_block_count: size_class_frag.free_block_count,
            largest_free_block: size_class_frag.largest_free_block,
            average_free_block_size: size_class_frag.average_free_block_size,
            fragmentation_ratio,
        };
        self.fragmentation_manager.update_metrics(frag_info);
        
        stats
    }
    
    /// Check if heap is nearly full and needs collection
    fn needs_collection(&self, threshold: f64) -> bool {
        let allocated = self.allocated.load(Ordering::Relaxed);
        let usage_ratio = allocated as f64 / self.capacity as f64;
        usage_ratio > threshold
    }
    
    /// Get memory pressure indicator (0.0 = no pressure, 1.0 = full)
    fn memory_pressure(&self) -> f64 {
        let allocated = self.allocated.load(Ordering::Relaxed);
        allocated as f64 / self.capacity as f64
    }
    
    /// Compact the heap to reduce fragmentation
    fn compact(&mut self) -> usize {
        let mut total_compacted = 0;
        
        // Compact size class free lists
        total_compacted += self.size_class_allocator.compact_free_lists();
        
        // Compact large object free lists
        total_compacted += self.large_object_allocator.compact();
        
        // Perform fragmentation manager compaction
        if let Some(target) = self.fragmentation_manager.get_next_compaction_request() {
            let _result = self.fragmentation_manager.perform_compaction();
        }
        
        total_compacted
    }
    
    /// Verify heap integrity (comprehensive debug function)
    #[cfg(debug_assertions)]
    fn verify_integrity(&self) -> Result<(), String> {
        // Verify object map consistency
        self.verify_object_map()?;
        
        // Verify allocation tracking consistency
        self.verify_allocation_tracking()?;
        
        // Verify size class allocator integrity
        self.verify_size_class_integrity()?;
        
        // Verify large object allocator integrity
        self.verify_large_object_integrity()?;
        
        // Verify memory regions if enabled
        if let Some(ref region_manager) = self.region_manager {
            self.verify_memory_regions(region_manager)?;
        }
        
        // Verify card table if enabled
        if let Some(ref card_table) = self.card_table {
            self.verify_card_table(card_table)?;
        }
        
        // Verify fragmentation manager consistency
        self.verify_fragmentation_consistency()?;
        
        Ok(())
    }
    
    /// Verify object map consistency
    #[cfg(debug_assertions)]
    fn verify_object_map(&self) -> Result<(), String> {
        let objects = self.objects.read().unwrap();
        let mut total_size = 0;
        let mut size_distribution = std::collections::HashMap::new();
        
        for (&ptr, header) in objects.iter() {
            // Check that pointer is valid
            if ptr.is_null() {
                return Err("Null pointer in object map".to_string());
            }
            
            // Check pointer alignment (should be at least 8-byte aligned)
            if (ptr as usize) % 8 != 0 {
                return Err(format!("Unaligned object pointer: {:p}", ptr));
            }
            
            // Check that size is reasonable
            if header.size == 0 {
                return Err("Zero-sized object in heap".to_string());
            }
            
            if header.size > self.capacity {
                return Err(format!(
                    "Object larger than heap capacity: {} > {}",
                    header.size, self.capacity
                ));
            }
            
            // Check object header consistency
            if header.generation > 7 {
                return Err(format!("Invalid generation: {}", header.generation));
            }
            
            if header.type_id == 0 {
                return Err("Invalid type_id: 0".to_string());
            }
            
            total_size += header.size;
            
            // Track size distribution for analysis
            let size_class = (header.size / 64) * 64; // Group by 64-byte classes
            *size_distribution.entry(size_class).or_insert(0) += 1;
        }
        
        // Check that total size doesn't exceed capacity
        if total_size > self.capacity {
            return Err(format!(
                "Total object size exceeds heap capacity: {} > {}",
                total_size, self.capacity
            ));
        }
        
        // Verify that allocated counter matches actual allocation
        let recorded_allocated = self.allocated.load(Ordering::Relaxed);
        if total_size != recorded_allocated {
            return Err(format!(
                "Allocated counter mismatch: recorded={}, actual={}",
                recorded_allocated, total_size
            ));
        }
        
        Ok(())
    }
    
    /// Verify allocation tracking consistency
    #[cfg(debug_assertions)]
    fn verify_allocation_tracking(&self) -> Result<(), String> {
        let objects = self.objects.read().unwrap();
        let object_count = objects.len();
        
        // Verify statistics consistency
        let stats = self.statistics.get_current_stats();
        if stats.live_objects != object_count {
            return Err(format!(
                "Live object count mismatch: stats={}, actual={}",
                stats.live_objects, object_count
            ));
        }
        
        let calculated_free_space = self.capacity.saturating_sub(stats.total_allocated);
        if (stats.free_space as i64 - calculated_free_space as i64).abs() > 1024 {
            return Err(format!(
                "Free space calculation mismatch: reported={}, calculated={}",
                stats.free_space, calculated_free_space
            ));
        }
        
        Ok(())
    }
    
    /// Verify size class allocator integrity
    #[cfg(debug_assertions)]
    fn verify_size_class_integrity(&self) -> Result<(), String> {
        let size_class_stats = self.size_class_allocator.get_statistics();
        
        // Verify that size class statistics are reasonable
        for class_stat in &size_class_stats.per_class_stats {
            if class_stat.allocations < class_stat.deallocations {
                return Err(format!(
                    "Size class {} has more deallocations than allocations: {} < {}",
                    class_stat.class_index, class_stat.allocations, class_stat.deallocations
                ));
            }
            
            if class_stat.utilization > 1.0 {
                return Err(format!(
                    "Size class {} has utilization > 100%: {}",
                    class_stat.class_index, class_stat.utilization
                ));
            }
        }
        
        Ok(())
    }
    
    /// Verify large object allocator integrity
    #[cfg(debug_assertions)]
    fn verify_large_object_integrity(&self) -> Result<(), String> {
        let large_obj_stats = self.large_object_allocator.get_statistics();
        
        // Verify large object statistics consistency
        if large_obj_stats.total_allocated < large_obj_stats.total_deallocated {
            return Err(format!(
                "Large object allocator has more deallocations than allocations: {} < {}",
                large_obj_stats.total_allocated, large_obj_stats.total_deallocated
            ));
        }
        
        let expected_live = large_obj_stats.total_allocated - large_obj_stats.total_deallocated;
        if large_obj_stats.live_objects != expected_live {
            return Err(format!(
                "Large object live count mismatch: reported={}, expected={}",
                large_obj_stats.live_objects, expected_live
            ));
        }
        
        Ok(())
    }
    
    /// Verify memory regions consistency
    #[cfg(debug_assertions)]
    fn verify_memory_regions(&self, region_manager: &MemoryRegionManager) -> Result<(), String> {
        let region_stats = region_manager.get_statistics();
        let regions = region_manager.get_all_regions();
        
        // Verify region count consistency
        if regions.len() != region_stats.active_regions {
            return Err(format!(
                "Region count mismatch: actual={}, reported={}",
                regions.len(), region_stats.active_regions
            ));
        }
        
        // Verify region utilization calculations
        let mut total_allocated = 0;
        let mut total_capacity = 0;
        
        for region in &regions {
            if region.allocated > region.size {
                return Err(format!(
                    "Region {} has allocated > size: {} > {}",
                    region.id, region.allocated, region.size
                ));
            }
            
            if region.average_age < 0.0 {
                return Err(format!(
                    "Region {} has negative average age: {}",
                    region.id, region.average_age
                ));
            }
            
            total_allocated += region.allocated;
            total_capacity += region.size;
        }
        
        let calculated_utilization = if total_capacity > 0 {
            total_allocated as f64 / total_capacity as f64
        } else {
            0.0
        };
        
        if (region_stats.average_utilization - calculated_utilization).abs() > 0.01 {
            return Err(format!(
                "Region utilization mismatch: reported={:.3}, calculated={:.3}",
                region_stats.average_utilization, calculated_utilization
            ));
        }
        
        Ok(())
    }
    
    /// Verify card table consistency
    #[cfg(debug_assertions)]
    fn verify_card_table(&self, card_table: &CardTable) -> Result<(), String> {
        let card_stats = card_table.get_statistics();
        let detailed_stats = card_table.get_detailed_statistics();
        
        // Verify card count consistency
        if card_stats.total_cards != detailed_stats.total_cards {
            return Err(format!(
                "Card count mismatch: basic={}, detailed={}",
                card_stats.total_cards, detailed_stats.total_cards
            ));
        }
        
        // Verify dirty card count is reasonable
        if card_stats.dirty_cards > card_stats.total_cards {
            return Err(format!(
                "More dirty cards than total cards: {} > {}",
                card_stats.dirty_cards, card_stats.total_cards
            ));
        }
        
        // Verify memory overhead calculation
        let expected_overhead = card_stats.total_cards; // 1 byte per card minimum
        if card_stats.memory_overhead < expected_overhead {
            return Err(format!(
                "Card table memory overhead too low: {} < {}",
                card_stats.memory_overhead, expected_overhead
            ));
        }
        
        Ok(())
    }
    
    /// Verify fragmentation manager consistency
    #[cfg(debug_assertions)]
    fn verify_fragmentation_consistency(&self) -> Result<(), String> {
        let frag_metrics = self.fragmentation_manager.get_current_metrics();
        
        // Verify fragmentation ratio is in valid range
        if frag_metrics.overall_fragmentation < 0.0 || frag_metrics.overall_fragmentation > 1.0 {
            return Err(format!(
                "Invalid fragmentation ratio: {}",
                frag_metrics.overall_fragmentation
            ));
        }
        
        // Verify free space consistency
        if frag_metrics.largest_free_block > frag_metrics.total_free_space {
            return Err(format!(
                "Largest free block larger than total free space: {} > {}",
                frag_metrics.largest_free_block, frag_metrics.total_free_space
            ));
        }
        
        // Verify wasted bytes calculation
        if frag_metrics.wasted_bytes > frag_metrics.total_free_space {
            return Err(format!(
                "Wasted bytes larger than total free space: {} > {}",
                frag_metrics.wasted_bytes, frag_metrics.total_free_space
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_creation() {
        let heap = Heap::new(64 * 1024 * 1024);
        let stats = heap.get_stats();
        
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.live_objects, 0);
        assert!(stats.free_space > 0);
    }

    #[test]
    fn test_object_registration() {
        let mut heap = Heap::new(64 * 1024 * 1024);
        
        let header = crate::ObjectHeader {
            size: 1024,
            type_id: 1,
            mark_bits: 0,
            generation: 0,
            ref_count: 1,
        };
        
        let ptr = 0x1000 as *const u8;
        heap.register_object(ptr, header);
        
        let stats = heap.get_stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.live_objects, 1);
        
        let retrieved_header = heap.get_header(ptr);
        assert!(retrieved_header.is_some());
        assert_eq!(retrieved_header.unwrap().size, 1024);
    }

    #[test]
    fn test_object_deallocation() {
        let mut heap = Heap::new(64 * 1024 * 1024);
        
        let header = crate::ObjectHeader {
            size: 1024,
            type_id: 1,
            mark_bits: 0,
            generation: 0,
            ref_count: 1,
        };
        
        let ptr = 0x1000 as *const u8;
        heap.register_object(ptr, header);
        
        // Verify object is registered
        assert!(heap.get_header(ptr).is_some());
        
        // Deallocate object
        heap.deallocate(ptr);
        
        // Verify object is removed
        assert!(heap.get_header(ptr).is_none());
        
        let stats = heap.get_stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.live_objects, 0);
    }

    #[test]
    fn test_free_list_allocation() {
        let mut heap = Heap::new(64 * 1024 * 1024);
        
        // First, deallocate something to populate free lists
        let header = crate::ObjectHeader {
            size: 1024,
            type_id: 1,
            mark_bits: 0,
            generation: 0,
            ref_count: 1,
        };
        
        let ptr = 0x1000 as *const u8;
        heap.register_object(ptr, header);
        heap.deallocate(ptr);
        
        // Now try to allocate from free list
        let result = heap.try_allocate_from_free_list(1024, 8);
        assert!(result.is_some());
    }

    #[test]
    fn test_color_management() {
        let mut heap = Heap::new(64 * 1024 * 1024);
        
        let header = crate::ObjectHeader {
            size: 1024,
            type_id: 1,
            mark_bits: 0,
            generation: 0,
            ref_count: 1,
        };
        
        let ptr = 0x1000 as *const u8;
        heap.register_object(ptr, header);
        
        // Initially should be white
        let white_objects = heap.find_white_objects();
        assert_eq!(white_objects.len(), 1);
        assert_eq!(white_objects[0], ptr);
        
        // Reset colors (should still be white)
        heap.reset_colors_to_white();
        let white_objects = heap.find_white_objects();
        assert_eq!(white_objects.len(), 1);
    }

    #[test]
    fn test_memory_pressure() {
        let mut heap = Heap::new(1024); // Small heap for testing
        
        // Initially no pressure
        assert_eq!(heap.memory_pressure(), 0.0);
        
        // Add some objects
        for i in 0..5 {
            let header = crate::ObjectHeader {
                size: 100,
                type_id: 1,
                mark_bits: 0,
                generation: 0,
                ref_count: 1,
            };
            
            let ptr = (0x1000 + i * 100) as *const u8;
            heap.register_object(ptr, header);
        }
        
        // Should have some pressure now
        let pressure = heap.memory_pressure();
        assert!(pressure > 0.0);
        assert!(pressure < 1.0);
        
        // Check if collection is needed
        assert!(heap.needs_collection(0.3)); // 30% threshold
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_heap_integrity() {
        let heap = Heap::new(64 * 1024 * 1024);
        
        // Empty heap should pass integrity check
        assert!(heap.verify_integrity().is_ok());
    }
} 