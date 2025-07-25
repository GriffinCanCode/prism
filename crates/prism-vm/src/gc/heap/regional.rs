//! Regional Heap for Generational Garbage Collection
//!
//! This module provides specialized heap regions for different object types
//! and generations, optimizing for generational collection patterns.

use super::*;
use super::types::*;
use super::core::Heap;
use std::sync::{Arc, RwLock};

/// Regional heap with specialized regions for different object types
pub struct RegionalHeap {
    /// Small object heap (< 1KB)
    small_heap: Arc<RwLock<Heap>>,
    /// Medium object heap (1KB - 8KB)  
    medium_heap: Arc<RwLock<Heap>>,
    /// Large object heap (> 8KB)
    large_heap: Arc<RwLock<Heap>>,
    /// Nursery for young objects (generational GC)
    nursery: Option<Arc<RwLock<Heap>>>,
    /// Configuration
    config: RegionalHeapConfig,
}

/// Configuration for regional heap
#[derive(Debug, Clone)]
pub struct RegionalHeapConfig {
    /// Enable generational collection
    pub enable_generational: bool,
    /// Small object threshold
    pub small_object_threshold: usize,
    /// Large object threshold  
    pub large_object_threshold: usize,
    /// Nursery size (if generational enabled)
    pub nursery_size: usize,
}

impl Default for RegionalHeapConfig {
    fn default() -> Self {
        Self {
            enable_generational: false,
            small_object_threshold: 1024,
            large_object_threshold: 8192,
            nursery_size: 16 * 1024 * 1024, // 16MB
        }
    }
}

impl RegionalHeap {
    /// Create a new regional heap
    pub fn new(total_capacity: usize, enable_generational: bool) -> Self {
        let config = RegionalHeapConfig {
            enable_generational,
            ..Default::default()
        };
        Self::with_config_and_capacity(total_capacity, config)
    }
    
    /// Create with custom configuration
    pub fn with_config(config: HeapConfig) -> Self {
        let regional_config = RegionalHeapConfig {
            enable_generational: config.enable_card_table, // Use card table as proxy for generational
            small_object_threshold: 1024,
            large_object_threshold: config.large_object_threshold,
            nursery_size: config.capacity / 8, // 1/8 of total capacity
        };
        Self::with_config_and_capacity(config.capacity, regional_config)
    }
    
    /// Create with configuration and capacity
    fn with_config_and_capacity(total_capacity: usize, config: RegionalHeapConfig) -> Self {
        let small_capacity = total_capacity / 4;
        let medium_capacity = total_capacity / 2;  
        let large_capacity = total_capacity / 4;
        
        let small_config = HeapConfig {
            capacity: small_capacity,
            large_object_threshold: config.small_object_threshold,
            ..Default::default()
        };
        
        let medium_config = HeapConfig {
            capacity: medium_capacity,
            large_object_threshold: config.large_object_threshold,
            ..Default::default()
        };
        
        let large_config = HeapConfig {
            capacity: large_capacity,
            large_object_threshold: usize::MAX, // All objects are "small" for large heap
            ..Default::default()
        };
        
        let nursery = if config.enable_generational {
            let nursery_config = HeapConfig {
                capacity: config.nursery_size,
                large_object_threshold: config.small_object_threshold,
                ..Default::default()
            };
            Some(Arc::new(RwLock::new(Heap::with_config(nursery_config))))
        } else {
            None
        };
        
        Self {
            small_heap: Arc::new(RwLock::new(Heap::with_config(small_config))),
            medium_heap: Arc::new(RwLock::new(Heap::with_config(medium_config))),
            large_heap: Arc::new(RwLock::new(Heap::with_config(large_config))),
            nursery,
            config,
        }
    }
    
    /// Choose appropriate heap region for allocation
    pub fn choose_heap(&self, size: usize) -> Arc<RwLock<Heap>> {
        if size < self.config.small_object_threshold {
            if let Some(ref nursery) = self.nursery {
                nursery.clone()
            } else {
                self.small_heap.clone()
            }
        } else if size < self.config.large_object_threshold {
            self.medium_heap.clone()
        } else {
            self.large_heap.clone()
        }
    }
    
    /// Get combined statistics from all regions
    pub fn get_combined_stats(&self) -> HeapStats {
        let small_stats = self.small_heap.read().unwrap().get_stats();
        let medium_stats = self.medium_heap.read().unwrap().get_stats();
        let large_stats = self.large_heap.read().unwrap().get_stats();
        
        let mut combined = HeapStats {
            total_allocated: small_stats.total_allocated + medium_stats.total_allocated + large_stats.total_allocated,
            live_objects: small_stats.live_objects + medium_stats.live_objects + large_stats.live_objects,
            free_space: small_stats.free_space + medium_stats.free_space + large_stats.free_space,
            fragmentation_ratio: (small_stats.fragmentation_ratio + medium_stats.fragmentation_ratio + large_stats.fragmentation_ratio) / 3.0,
            allocation_rate: small_stats.allocation_rate + medium_stats.allocation_rate + large_stats.allocation_rate,
            gc_overhead: (small_stats.gc_overhead + medium_stats.gc_overhead + large_stats.gc_overhead) / 3.0,
            region_stats: None,
            card_table_stats: None,
        };
        
        if let Some(ref nursery) = self.nursery {
            let nursery_stats = nursery.read().unwrap().get_stats();
            combined.total_allocated += nursery_stats.total_allocated;
            combined.live_objects += nursery_stats.live_objects;
            combined.free_space += nursery_stats.free_space;
        }
        
        combined
    }
}

impl HeapInterface for RegionalHeap {
    fn register_object(&mut self, ptr: *const u8, header: crate::ObjectHeader) {
        let heap = self.choose_heap(header.size);
        heap.write().unwrap().register_object(ptr, header);
    }
    
    fn get_header(&self, ptr: *const u8) -> Option<&crate::ObjectHeader> {
        // Try each heap region
        if let Some(header) = self.small_heap.read().unwrap().get_header(ptr) {
            return Some(header);
        }
        if let Some(header) = self.medium_heap.read().unwrap().get_header(ptr) {
            return Some(header);
        }
        if let Some(header) = self.large_heap.read().unwrap().get_header(ptr) {
            return Some(header);
        }
        if let Some(ref nursery) = self.nursery {
            if let Some(header) = nursery.read().unwrap().get_header(ptr) {
                return Some(header);
            }
        }
        None
    }
    
    fn get_header_mut(&mut self, ptr: *const u8) -> Option<&mut crate::ObjectHeader> {
        // Try each heap region
        if let Some(header) = self.small_heap.write().unwrap().get_header_mut(ptr) {
            return Some(header);
        }
        if let Some(header) = self.medium_heap.write().unwrap().get_header_mut(ptr) {
            return Some(header);
        }
        if let Some(header) = self.large_heap.write().unwrap().get_header_mut(ptr) {
            return Some(header);
        }
        if let Some(ref nursery) = self.nursery {
            if let Some(header) = nursery.write().unwrap().get_header_mut(ptr) {
                return Some(header);
            }
        }
        None
    }
    
    fn find_white_objects(&self) -> Vec<*const u8> {
        let mut white_objects = Vec::new();
        
        white_objects.extend(self.small_heap.read().unwrap().find_white_objects());
        white_objects.extend(self.medium_heap.read().unwrap().find_white_objects());
        white_objects.extend(self.large_heap.read().unwrap().find_white_objects());
        
        if let Some(ref nursery) = self.nursery {
            white_objects.extend(nursery.read().unwrap().find_white_objects());
        }
        
        white_objects
    }
    
    fn reset_colors_to_white(&mut self) {
        self.small_heap.write().unwrap().reset_colors_to_white();
        self.medium_heap.write().unwrap().reset_colors_to_white();
        self.large_heap.write().unwrap().reset_colors_to_white();
        
        if let Some(ref nursery) = self.nursery {
            nursery.write().unwrap().reset_colors_to_white();
        }
    }
    
    fn deallocate(&mut self, ptr: *const u8) {
        // Try each heap region
        if self.small_heap.read().unwrap().get_header(ptr).is_some() {
            self.small_heap.write().unwrap().deallocate(ptr);
            return;
        }
        if self.medium_heap.read().unwrap().get_header(ptr).is_some() {
            self.medium_heap.write().unwrap().deallocate(ptr);
            return;
        }
        if self.large_heap.read().unwrap().get_header(ptr).is_some() {
            self.large_heap.write().unwrap().deallocate(ptr);
            return;
        }
        if let Some(ref nursery) = self.nursery {
            if nursery.read().unwrap().get_header(ptr).is_some() {
                nursery.write().unwrap().deallocate(ptr);
                return;
            }
        }
    }
    
    fn try_allocate_from_free_list(&mut self, size: usize, align: usize) -> Option<*const u8> {
        let heap = self.choose_heap(size);
        heap.write().unwrap().try_allocate_from_free_list(size, align)
    }
    
    fn get_stats(&self) -> HeapStats {
        self.get_combined_stats()
    }
    
    fn needs_collection(&self, threshold: f64) -> bool {
        self.small_heap.read().unwrap().needs_collection(threshold) ||
        self.medium_heap.read().unwrap().needs_collection(threshold) ||
        self.large_heap.read().unwrap().needs_collection(threshold) ||
        self.nursery.as_ref().map_or(false, |n| n.read().unwrap().needs_collection(threshold))
    }
    
    fn memory_pressure(&self) -> f64 {
        let small_pressure = self.small_heap.read().unwrap().memory_pressure();
        let medium_pressure = self.medium_heap.read().unwrap().memory_pressure();
        let large_pressure = self.large_heap.read().unwrap().memory_pressure();
        
        let nursery_pressure = self.nursery.as_ref()
            .map(|n| n.read().unwrap().memory_pressure())
            .unwrap_or(0.0);
        
        // Return maximum pressure across all regions
        small_pressure.max(medium_pressure).max(large_pressure).max(nursery_pressure)
    }
    
    fn compact(&mut self) -> usize {
        let mut total_compacted = 0;
        
        total_compacted += self.small_heap.write().unwrap().compact();
        total_compacted += self.medium_heap.write().unwrap().compact();
        total_compacted += self.large_heap.write().unwrap().compact();
        
        if let Some(ref nursery) = self.nursery {
            total_compacted += nursery.write().unwrap().compact();
        }
        
        total_compacted
    }
    
    #[cfg(debug_assertions)]
    fn verify_integrity(&self) -> Result<(), String> {
        self.small_heap.read().unwrap().verify_integrity()?;
        self.medium_heap.read().unwrap().verify_integrity()?;
        self.large_heap.read().unwrap().verify_integrity()?;
        
        if let Some(ref nursery) = self.nursery {
            nursery.read().unwrap().verify_integrity()?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regional_heap_creation() {
        let heap = RegionalHeap::new(64 * 1024 * 1024, false);
        let stats = heap.get_combined_stats();
        
        assert_eq!(stats.live_objects, 0);
        assert!(stats.free_space > 0);
    }

    #[test]
    fn test_regional_heap_with_generational() {
        let heap = RegionalHeap::new(64 * 1024 * 1024, true);
        assert!(heap.nursery.is_some());
        
        let stats = heap.get_combined_stats();
        assert_eq!(stats.live_objects, 0);
    }

    #[test]
    fn test_heap_selection() {
        let heap = RegionalHeap::new(64 * 1024 * 1024, false);
        
        // Small object should go to small heap
        let small_heap = heap.choose_heap(512);
        assert!(Arc::ptr_eq(&small_heap, &heap.small_heap));
        
        // Medium object should go to medium heap
        let medium_heap = heap.choose_heap(4096);
        assert!(Arc::ptr_eq(&medium_heap, &heap.medium_heap));
        
        // Large object should go to large heap
        let large_heap = heap.choose_heap(16384);
        assert!(Arc::ptr_eq(&large_heap, &heap.large_heap));
    }

    #[test]
    fn test_nursery_selection() {
        let heap = RegionalHeap::new(64 * 1024 * 1024, true);
        
        // Small object should go to nursery when generational is enabled
        let selected_heap = heap.choose_heap(512);
        assert!(Arc::ptr_eq(&selected_heap, heap.nursery.as_ref().unwrap()));
    }
} 