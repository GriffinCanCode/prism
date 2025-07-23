use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Heap management for garbage collection
/// Tracks allocated objects and their metadata
pub struct Heap {
    /// Map from object pointer to object header
    objects: HashMap<*const u8, ObjectHeader>,
    /// Total heap capacity
    capacity: usize,
    /// Currently allocated bytes
    allocated: AtomicUsize,
    /// Free list for reusing deallocated objects
    free_list: Vec<(*const u8, usize)>, // (pointer, size)
    /// Large object threshold - objects larger than this get special treatment
    large_object_threshold: usize,
}

impl Heap {
    pub fn new(capacity: usize) -> Self {
        Self {
            objects: HashMap::new(),
            capacity,
            allocated: AtomicUsize::new(0),
            free_list: Vec::new(),
            large_object_threshold: 8192, // 8KB threshold
        }
    }
    
    /// Register a newly allocated object
    pub fn register_object(&mut self, ptr: *const u8, header: ObjectHeader) {
        self.objects.insert(ptr, header);
        self.allocated.fetch_add(header.size, Ordering::Relaxed);
    }
    
    /// Get object header for a given pointer
    pub fn get_header(&self, ptr: *const u8) -> Option<&ObjectHeader> {
        self.objects.get(&ptr)
    }
    
    /// Get mutable object header for a given pointer
    pub fn get_header_mut(&mut self, ptr: *const u8) -> Option<&mut ObjectHeader> {
        self.objects.get_mut(&ptr)
    }
    
    /// Find all white (unmarked) objects for sweeping
    pub fn find_white_objects(&self) -> Vec<*const u8> {
        self.objects
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
    pub fn reset_colors_to_white(&mut self) {
        for header in self.objects.values_mut() {
            header.set_color(ObjectColor::White);
        }
    }
    
    /// Deallocate an object and add it to free list
    pub fn deallocate(&mut self, ptr: *const u8) {
        if let Some(header) = self.objects.remove(&ptr) {
            self.allocated.fetch_sub(header.size, Ordering::Relaxed);
            
            // Add to free list for potential reuse
            if header.size < self.large_object_threshold {
                self.free_list.push((ptr, header.size));
            }
            // Large objects are not reused to avoid fragmentation
        }
    }
    
    /// Try to allocate from free list before going to system allocator
    pub fn try_allocate_from_free_list(&mut self, size: usize, align: usize) -> Option<*const u8> {
        // Find a suitable free block
        for i in 0..self.free_list.len() {
            let (ptr, free_size) = self.free_list[i];
            
            // Check if this block is suitable
            if free_size >= size && (ptr as usize) % align == 0 {
                // Remove from free list
                self.free_list.swap_remove(i);
                
                // If the block is much larger, split it
                if free_size > size + 64 {
                    let remaining_ptr = unsafe { ptr.add(size) };
                    let remaining_size = free_size - size;
                    self.free_list.push((remaining_ptr, remaining_size));
                }
                
                return Some(ptr);
            }
        }
        
        None
    }
    
    /// Get heap statistics
    pub fn get_stats(&self) -> HeapStats {
        let allocated = self.allocated.load(Ordering::Relaxed);
        let free_space = self.capacity.saturating_sub(allocated);
        
        // Calculate fragmentation ratio
        let free_list_space: usize = self.free_list.iter().map(|(_, size)| *size).sum();
        let fragmentation_ratio = if free_space > 0 {
            1.0 - (free_list_space as f64 / free_space as f64)
        } else {
            0.0
        };
        
        HeapStats {
            total_allocated: allocated,
            live_objects: self.objects.len(),
            free_space,
            fragmentation_ratio,
            allocation_rate: 0.0, // Would be calculated elsewhere
            gc_overhead: 0.0,     // Would be calculated elsewhere
        }
    }
    
    /// Compact the heap to reduce fragmentation
    pub fn compact(&mut self) -> usize {
        // Sort free blocks by address to enable coalescing
        self.free_list.sort_by_key(|(ptr, _)| *ptr as usize);
        
        let mut coalesced = 0;
        let mut i = 0;
        
        while i < self.free_list.len() - 1 {
            let (ptr1, size1) = self.free_list[i];
            let (ptr2, size2) = self.free_list[i + 1];
            
            // Check if blocks are adjacent
            if unsafe { ptr1.add(size1) } == ptr2 {
                // Coalesce the blocks
                self.free_list[i] = (ptr1, size1 + size2);
                self.free_list.remove(i + 1);
                coalesced += 1;
            } else {
                i += 1;
            }
        }
        
        coalesced
    }
    
    /// Get all objects in a specific generation (for generational GC)
    pub fn get_objects_in_generation(&self, generation: u8) -> Vec<*const u8> {
        self.objects
            .iter()
            .filter_map(|(&ptr, header)| {
                if header.generation == generation {
                    Some(ptr)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Promote objects to next generation
    pub fn promote_objects(&mut self, objects: &[*const u8]) {
        for &ptr in objects {
            if let Some(header) = self.objects.get_mut(&ptr) {
                if header.generation < 255 {
                    header.generation += 1;
                }
            }
        }
    }
    
    /// Check if heap is nearly full and needs collection
    pub fn needs_collection(&self, threshold: f64) -> bool {
        let allocated = self.allocated.load(Ordering::Relaxed);
        let usage_ratio = allocated as f64 / self.capacity as f64;
        usage_ratio > threshold
    }
    
    /// Get memory pressure indicator (0.0 = no pressure, 1.0 = full)
    pub fn memory_pressure(&self) -> f64 {
        let allocated = self.allocated.load(Ordering::Relaxed);
        allocated as f64 / self.capacity as f64
    }
    
    /// Verify heap integrity (debug function)
    #[cfg(debug_assertions)]
    pub fn verify_integrity(&self) -> Result<(), String> {
        let mut total_size = 0;
        
        for (&ptr, header) in &self.objects {
            // Check that pointer is valid
            if ptr.is_null() {
                return Err("Null pointer in object map".to_string());
            }
            
            // Check that size is reasonable
            if header.size == 0 {
                return Err("Zero-sized object in heap".to_string());
            }
            
            if header.size > self.capacity {
                return Err("Object larger than heap capacity".to_string());
            }
            
            total_size += header.size;
        }
        
        // Check that total size doesn't exceed capacity
        if total_size > self.capacity {
            return Err("Total object size exceeds heap capacity".to_string());
        }
        
        // Verify free list doesn't contain live objects
        for &(free_ptr, _) in &self.free_list {
            if self.objects.contains_key(&free_ptr) {
                return Err("Free list contains live object".to_string());
            }
        }
        
        Ok(())
    }
}

/// Specialized heap regions for different object types
pub struct RegionalHeap {
    /// Small object heap (< 1KB)
    small_heap: Heap,
    /// Medium object heap (1KB - 8KB)
    medium_heap: Heap,
    /// Large object heap (> 8KB)
    large_heap: Heap,
    /// Nursery for young objects (generational GC)
    nursery: Option<Heap>,
}

impl RegionalHeap {
    pub fn new(total_capacity: usize, enable_generational: bool) -> Self {
        let small_capacity = total_capacity / 4;
        let medium_capacity = total_capacity / 2;
        let large_capacity = total_capacity / 4;
        
        Self {
            small_heap: Heap::new(small_capacity),
            medium_heap: Heap::new(medium_capacity),
            large_heap: Heap::new(large_capacity),
            nursery: if enable_generational {
                Some(Heap::new(small_capacity))
            } else {
                None
            },
        }
    }
    
    /// Choose appropriate heap region for allocation
    pub fn choose_heap(&mut self, size: usize) -> &mut Heap {
        if size < 1024 {
            if let Some(ref mut nursery) = self.nursery {
                nursery
            } else {
                &mut self.small_heap
            }
        } else if size < 8192 {
            &mut self.medium_heap
        } else {
            &mut self.large_heap
        }
    }
    
    /// Collect statistics from all regions
    pub fn get_combined_stats(&self) -> HeapStats {
        let small_stats = self.small_heap.get_stats();
        let medium_stats = self.medium_heap.get_stats();
        let large_stats = self.large_heap.get_stats();
        
        let mut combined = HeapStats {
            total_allocated: small_stats.total_allocated + medium_stats.total_allocated + large_stats.total_allocated,
            live_objects: small_stats.live_objects + medium_stats.live_objects + large_stats.live_objects,
            free_space: small_stats.free_space + medium_stats.free_space + large_stats.free_space,
            fragmentation_ratio: (small_stats.fragmentation_ratio + medium_stats.fragmentation_ratio + large_stats.fragmentation_ratio) / 3.0,
            allocation_rate: small_stats.allocation_rate + medium_stats.allocation_rate + large_stats.allocation_rate,
            gc_overhead: (small_stats.gc_overhead + medium_stats.gc_overhead + large_stats.gc_overhead) / 3.0,
        };
        
        if let Some(ref nursery) = self.nursery {
            let nursery_stats = nursery.get_stats();
            combined.total_allocated += nursery_stats.total_allocated;
            combined.live_objects += nursery_stats.live_objects;
            combined.free_space += nursery_stats.free_space;
        }
        
        combined
    }
}

/// Card table for tracking dirty regions in the heap
/// Used for generational garbage collection
pub struct CardTable {
    /// Each byte represents a 512-byte card in the heap
    cards: Vec<u8>,
    /// Size of each card in bytes
    card_size: usize,
    /// Base address of the heap
    heap_base: *const u8,
    /// Size of the heap
    heap_size: usize,
}

impl CardTable {
    pub fn new(heap_base: *const u8, heap_size: usize, card_size: usize) -> Self {
        let num_cards = (heap_size + card_size - 1) / card_size;
        
        Self {
            cards: vec![0; num_cards],
            card_size,
            heap_base,
            heap_size,
        }
    }
    
    /// Mark a card as dirty
    pub fn mark_dirty(&mut self, address: *const u8) {
        if let Some(card_index) = self.get_card_index(address) {
            self.cards[card_index] = 1;
        }
    }
    
    /// Check if a card is dirty
    pub fn is_dirty(&self, address: *const u8) -> bool {
        if let Some(card_index) = self.get_card_index(address) {
            self.cards[card_index] != 0
        } else {
            false
        }
    }
    
    /// Clear all dirty cards
    pub fn clear_all(&mut self) {
        self.cards.fill(0);
    }
    
    /// Get all dirty card indices
    pub fn get_dirty_cards(&self) -> Vec<usize> {
        self.cards
            .iter()
            .enumerate()
            .filter_map(|(i, &dirty)| if dirty != 0 { Some(i) } else { None })
            .collect()
    }
    
    /// Get card index for a given address
    fn get_card_index(&self, address: *const u8) -> Option<usize> {
        if address < self.heap_base || address >= unsafe { self.heap_base.add(self.heap_size) } {
            return None;
        }
        
        let offset = address as usize - self.heap_base as usize;
        Some(offset / self.card_size)
    }
    
    /// Get the address range for a card
    pub fn get_card_range(&self, card_index: usize) -> Option<(*const u8, *const u8)> {
        if card_index >= self.cards.len() {
            return None;
        }
        
        let start_offset = card_index * self.card_size;
        let end_offset = ((card_index + 1) * self.card_size).min(self.heap_size);
        
        let start_addr = unsafe { self.heap_base.add(start_offset) };
        let end_addr = unsafe { self.heap_base.add(end_offset) };
        
        Some((start_addr, end_addr))
    }
} 