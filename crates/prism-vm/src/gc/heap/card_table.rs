//! Card Table for Generational Garbage Collection
//!
//! This module provides card table functionality for tracking cross-generational
//! references in support of generational garbage collection.
//!
//! **Card Table Responsibilities:**
//! - Track dirty regions of memory for generational GC
//! - Provide efficient cross-generational reference detection
//! - Coordinate with write barriers for dirty card marking
//! - Support incremental and concurrent collection
//!
//! **NOT Card Table Responsibilities (delegated):**
//! - Write barrier implementation (handled by barriers::*)
//! - Object scanning and marking (handled by collectors::*)
//! - Memory allocation (handled by allocators::*)

use super::types::*;
use std::sync::{RwLock, Mutex};
use std::sync::atomic::{AtomicUsize, AtomicU8, Ordering};
use std::ptr::NonNull;

/// Card table for tracking dirty memory regions
pub struct CardTable {
    /// The card table itself - each byte represents one card
    cards: Vec<AtomicU8>,
    
    /// Size of each card in bytes
    card_size: usize,
    
    /// Base address of the heap
    heap_base: *const u8,
    
    /// Size of the heap
    heap_size: usize,
    
    /// Configuration
    config: CardTableConfig,
    
    /// Statistics
    stats: CardTableStatistics,
    
    /// Dirty card tracking for incremental processing
    dirty_card_tracker: Mutex<DirtyCardTracker>,
    
    /// Card aging for optimization
    card_aging: RwLock<CardAging>,
}

/// Configuration for card table
#[derive(Debug, Clone)]
pub struct CardTableConfig {
    /// Size of each card in bytes
    pub card_size: usize,
    
    /// Enable card aging for optimization
    pub enable_aging: bool,
    
    /// Enable batch processing of dirty cards
    pub enable_batch_processing: bool,
    
    /// Batch size for processing dirty cards
    pub batch_size: usize,
    
    /// Enable concurrent card scanning
    pub enable_concurrent_scanning: bool,
    
    /// Card clearing strategy
    pub clearing_strategy: CardClearingStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CardClearingStrategy {
    /// Clear all cards at once
    ClearAll,
    /// Clear cards incrementally
    Incremental,
    /// Clear cards lazily as they're processed
    Lazy,
}

impl Default for CardTableConfig {
    fn default() -> Self {
        Self {
            card_size: DEFAULT_CARD_SIZE,
            enable_aging: true,
            enable_batch_processing: true,
            batch_size: 64,
            enable_concurrent_scanning: true,
            clearing_strategy: CardClearingStrategy::Incremental,
        }
    }
}

/// Statistics for card table operations
#[derive(Debug, Default)]
pub struct CardTableStatistics {
    /// Total cards in the table
    total_cards: usize,
    
    /// Currently dirty cards
    dirty_cards: AtomicUsize,
    
    /// Total cards marked dirty (cumulative)
    total_dirty_markings: AtomicUsize,
    
    /// Total cards cleared
    total_cards_cleared: AtomicUsize,
    
    /// Total card scans performed
    total_scans: AtomicUsize,
    
    /// Average cards per scan
    average_cards_per_scan: AtomicUsize,
    
    /// Card table memory overhead
    memory_overhead: usize,
}

/// Dirty card tracker for incremental processing
#[derive(Debug, Default)]
struct DirtyCardTracker {
    /// Queue of dirty card indices for processing
    dirty_queue: Vec<usize>,
    
    /// Cards currently being processed
    processing_cards: Vec<usize>,
    
    /// Batch processing state
    current_batch: Vec<usize>,
    
    /// Last processed card index
    last_processed: usize,
}

/// Card aging for optimization
#[derive(Debug, Default)]
struct CardAging {
    /// Age of each card (number of GC cycles since last dirty)
    card_ages: Vec<AtomicU8>,
    
    /// Current GC cycle number
    current_cycle: AtomicUsize,
    
    /// Age threshold for optimization decisions
    age_threshold: u8,
}

impl CardTable {
    /// Create a new card table
    pub fn new(heap_base: *const u8, heap_size: usize, card_size: usize) -> Self {
        let config = CardTableConfig {
            card_size,
            ..Default::default()
        };
        
        Self::with_config(heap_base, heap_size, config)
    }
    
    /// Create card table with custom configuration
    pub fn with_config(heap_base: *const u8, heap_size: usize, config: CardTableConfig) -> Self {
        let num_cards = (heap_size + config.card_size - 1) / config.card_size;
        let cards = (0..num_cards).map(|_| AtomicU8::new(0)).collect();
        
        let card_ages = if config.enable_aging {
            (0..num_cards).map(|_| AtomicU8::new(0)).collect()
        } else {
            Vec::new()
        };
        
        let stats = CardTableStatistics {
            total_cards: num_cards,
            memory_overhead: num_cards + if config.enable_aging { num_cards } else { 0 },
            ..Default::default()
        };
        
        Self {
            cards,
            card_size: config.card_size,
            heap_base,
            heap_size,
            config,
            stats,
            dirty_card_tracker: Mutex::new(DirtyCardTracker::default()),
            card_aging: RwLock::new(CardAging {
                card_ages,
                current_cycle: AtomicUsize::new(0),
                age_threshold: 5, // Cards older than 5 cycles are considered stable
            }),
        }
    }
    
    /// Mark a card as dirty
    pub fn mark_dirty(&self, address: *const u8) {
        if let Some(card_index) = self.get_card_index(address) {
            let old_value = self.cards[card_index].swap(CardState::Dirty as u8, Ordering::Relaxed);
            
            // If card wasn't already dirty, update statistics and tracking
            if old_value != CardState::Dirty as u8 {
                self.stats.dirty_cards.fetch_add(1, Ordering::Relaxed);
                self.stats.total_dirty_markings.fetch_add(1, Ordering::Relaxed);
                
                // Add to dirty queue for processing
                if let Ok(mut tracker) = self.dirty_card_tracker.lock() {
                    tracker.dirty_queue.push(card_index);
                }
                
                // Reset card age if aging is enabled
                if self.config.enable_aging {
                    if let Ok(aging) = self.card_aging.read() {
                        if card_index < aging.card_ages.len() {
                            aging.card_ages[card_index].store(0, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }
    
    /// Check if a card is dirty
    pub fn is_dirty(&self, address: *const u8) -> bool {
        if let Some(card_index) = self.get_card_index(address) {
            self.cards[card_index].load(Ordering::Relaxed) == CardState::Dirty as u8
        } else {
            false
        }
    }
    
    /// Clear all dirty cards
    pub fn clear_all(&self) {
        match self.config.clearing_strategy {
            CardClearingStrategy::ClearAll => self.clear_all_immediate(),
            CardClearingStrategy::Incremental => self.clear_incremental(),
            CardClearingStrategy::Lazy => {}, // Cards cleared during processing
        }
    }
    
    /// Clear all cards immediately
    fn clear_all_immediate(&self) {
        let mut cleared_count = 0;
        
        for card in &self.cards {
            let old_value = card.swap(CardState::Clean as u8, Ordering::Relaxed);
            if old_value == CardState::Dirty as u8 {
                cleared_count += 1;
            }
        }
        
        // Update statistics
        self.stats.dirty_cards.store(0, Ordering::Relaxed);
        self.stats.total_cards_cleared.fetch_add(cleared_count, Ordering::Relaxed);
        
        // Clear dirty queue
        if let Ok(mut tracker) = self.dirty_card_tracker.lock() {
            tracker.dirty_queue.clear();
            tracker.processing_cards.clear();
            tracker.current_batch.clear();
        }
        
        // Age all cards if aging is enabled
        if self.config.enable_aging {
            self.age_all_cards();
        }
    }
    
    /// Clear cards incrementally
    fn clear_incremental(&self) {
        if let Ok(mut tracker) = self.dirty_card_tracker.lock() {
            let batch_size = self.config.batch_size.min(tracker.dirty_queue.len());
            
            for _ in 0..batch_size {
                if let Some(card_index) = tracker.dirty_queue.pop() {
                    let old_value = self.cards[card_index].swap(CardState::Clean as u8, Ordering::Relaxed);
                    if old_value == CardState::Dirty as u8 {
                        self.stats.dirty_cards.fetch_sub(1, Ordering::Relaxed);
                        self.stats.total_cards_cleared.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    
    /// Get all dirty card indices
    pub fn get_dirty_cards(&self) -> Vec<usize> {
        let mut dirty_cards = Vec::new();
        
        for (index, card) in self.cards.iter().enumerate() {
            if card.load(Ordering::Relaxed) == CardState::Dirty as u8 {
                dirty_cards.push(index);
            }
        }
        
        dirty_cards
    }
    
    /// Get the next batch of dirty cards for processing
    pub fn get_next_dirty_batch(&self) -> Vec<usize> {
        if let Ok(mut tracker) = self.dirty_card_tracker.lock() {
            if tracker.current_batch.is_empty() {
                // Fill batch from dirty queue
                let batch_size = self.config.batch_size.min(tracker.dirty_queue.len());
                for _ in 0..batch_size {
                    if let Some(card_index) = tracker.dirty_queue.pop() {
                        tracker.current_batch.push(card_index);
                    }
                }
            }
            
            // Move current batch to processing and return it
            let batch = std::mem::take(&mut tracker.current_batch);
            tracker.processing_cards.extend(&batch);
            batch
        } else {
            Vec::new()
        }
    }
    
    /// Mark cards as processed (clear them)
    pub fn mark_cards_processed(&self, card_indices: &[usize]) {
        let mut cleared_count = 0;
        
        for &card_index in card_indices {
            if card_index < self.cards.len() {
                let old_value = self.cards[card_index].swap(CardState::Clean as u8, Ordering::Relaxed);
                if old_value == CardState::Dirty as u8 {
                    cleared_count += 1;
                }
            }
        }
        
        // Update statistics
        self.stats.dirty_cards.fetch_sub(cleared_count, Ordering::Relaxed);
        self.stats.total_cards_cleared.fetch_add(cleared_count, Ordering::Relaxed);
        
        // Remove from processing list
        if let Ok(mut tracker) = self.dirty_card_tracker.lock() {
            for &card_index in card_indices {
                tracker.processing_cards.retain(|&x| x != card_index);
            }
        }
        
        // Age processed cards if aging is enabled
        if self.config.enable_aging {
            self.age_cards(card_indices);
        }
    }
    
    /// Get card index for a given address
    fn get_card_index(&self, address: *const u8) -> Option<usize> {
        if address < self.heap_base || 
           address >= unsafe { self.heap_base.add(self.heap_size) } {
            return None;
        }
        
        let offset = address as usize - self.heap_base as usize;
        let card_index = offset / self.card_size;
        
        if card_index < self.cards.len() {
            Some(card_index)
        } else {
            None
        }
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
    
    /// Age all cards (increment their age)
    fn age_all_cards(&self) {
        if let Ok(aging) = self.card_aging.read() {
            aging.current_cycle.fetch_add(1, Ordering::Relaxed);
            
            for card_age in &aging.card_ages {
                let current_age = card_age.load(Ordering::Relaxed);
                if current_age < 255 { // Prevent overflow
                    card_age.store(current_age + 1, Ordering::Relaxed);
                }
            }
        }
    }
    
    /// Age specific cards
    fn age_cards(&self, card_indices: &[usize]) {
        if let Ok(aging) = self.card_aging.read() {
            for &card_index in card_indices {
                if card_index < aging.card_ages.len() {
                    let current_age = aging.card_ages[card_index].load(Ordering::Relaxed);
                    if current_age < 255 {
                        aging.card_ages[card_index].store(current_age + 1, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    
    /// Get cards that are likely stable (old and not recently dirtied)
    pub fn get_stable_cards(&self) -> Vec<usize> {
        if !self.config.enable_aging {
            return Vec::new();
        }
        
        let mut stable_cards = Vec::new();
        
        if let Ok(aging) = self.card_aging.read() {
            for (index, card_age) in aging.card_ages.iter().enumerate() {
                let age = card_age.load(Ordering::Relaxed);
                if age >= aging.age_threshold &&
                   self.cards[index].load(Ordering::Relaxed) == CardState::Clean as u8 {
                    stable_cards.push(index);
                }
            }
        }
        
        stable_cards
    }
    
    /// Get statistics for the card table
    pub fn get_statistics(&self) -> CardTableStats {
        CardTableStats {
            total_cards: self.stats.total_cards,
            dirty_cards: self.stats.dirty_cards.load(Ordering::Relaxed),
            memory_overhead: self.stats.memory_overhead,
        }
    }
    
    /// Get detailed statistics
    pub fn get_detailed_statistics(&self) -> DetailedCardTableStats {
        let dirty_queue_size = self.dirty_card_tracker
            .lock()
            .map(|tracker| tracker.dirty_queue.len())
            .unwrap_or(0);
        
        let processing_cards_count = self.dirty_card_tracker
            .lock()
            .map(|tracker| tracker.processing_cards.len())
            .unwrap_or(0);
        
        let current_cycle = if self.config.enable_aging {
            self.card_aging.read().unwrap().current_cycle.load(Ordering::Relaxed)
        } else {
            0
        };
        
        DetailedCardTableStats {
            total_cards: self.stats.total_cards,
            dirty_cards: self.stats.dirty_cards.load(Ordering::Relaxed),
            total_dirty_markings: self.stats.total_dirty_markings.load(Ordering::Relaxed),
            total_cards_cleared: self.stats.total_cards_cleared.load(Ordering::Relaxed),
            total_scans: self.stats.total_scans.load(Ordering::Relaxed),
            average_cards_per_scan: self.stats.average_cards_per_scan.load(Ordering::Relaxed),
            memory_overhead: self.stats.memory_overhead,
            dirty_queue_size,
            processing_cards_count,
            current_gc_cycle: current_cycle,
            card_size: self.card_size,
        }
    }
    
    /// Perform maintenance operations
    pub fn perform_maintenance(&self) {
        // Age cards if enabled
        if self.config.enable_aging {
            self.age_all_cards();
        }
        
        // Clean up processing queues
        if let Ok(mut tracker) = self.dirty_card_tracker.lock() {
            // Remove stale entries from processing cards
            tracker.processing_cards.retain(|&card_index| {
                self.cards[card_index].load(Ordering::Relaxed) == CardState::Processing as u8
            });
        }
        
        // Update scan statistics
        self.stats.total_scans.fetch_add(1, Ordering::Relaxed);
        
        let dirty_count = self.stats.dirty_cards.load(Ordering::Relaxed);
        let total_scans = self.stats.total_scans.load(Ordering::Relaxed);
        if total_scans > 0 {
            self.stats.average_cards_per_scan.store(
                dirty_count / total_scans,
                Ordering::Relaxed,
            );
        }
    }
    
    /// Reset the card table
    pub fn reset(&self) {
        self.clear_all_immediate();
        
        // Reset aging if enabled
        if self.config.enable_aging {
            if let Ok(aging) = self.card_aging.write() {
                aging.current_cycle.store(0, Ordering::Relaxed);
                for card_age in &aging.card_ages {
                    card_age.store(0, Ordering::Relaxed);
                }
            }
        }
        
        // Reset statistics
        self.stats.total_dirty_markings.store(0, Ordering::Relaxed);
        self.stats.total_cards_cleared.store(0, Ordering::Relaxed);
        self.stats.total_scans.store(0, Ordering::Relaxed);
        self.stats.average_cards_per_scan.store(0, Ordering::Relaxed);
    }
    
    /// Update heap base address (called when actual heap memory is allocated)
    pub fn update_heap_base(&mut self, new_heap_base: *const u8, new_heap_size: usize) {
        // Update heap parameters
        self.heap_base = new_heap_base;
        self.heap_size = new_heap_size;
        
        // Recalculate number of cards needed
        let num_cards = (new_heap_size + self.card_size - 1) / self.card_size;
        
        // Resize card table if necessary
        if num_cards != self.cards.len() {
            self.cards = (0..num_cards).map(|_| AtomicU8::new(0)).collect();
            
            // Update aging if enabled
            if self.config.enable_aging {
                if let Ok(mut aging) = self.card_aging.write() {
                    aging.card_ages = (0..num_cards).map(|_| AtomicU8::new(0)).collect();
                }
            }
            
            // Update statistics
            self.stats.total_cards = num_cards;
            self.stats.memory_overhead = num_cards + if self.config.enable_aging { num_cards } else { 0 };
        }
        
        // Clear all cards since heap layout has changed
        self.clear_all_immediate();
    }
}

/// Detailed statistics for card table analysis
#[derive(Debug, Clone)]
pub struct DetailedCardTableStats {
    pub total_cards: usize,
    pub dirty_cards: usize,
    pub total_dirty_markings: usize,
    pub total_cards_cleared: usize,
    pub total_scans: usize,
    pub average_cards_per_scan: usize,
    pub memory_overhead: usize,
    pub dirty_queue_size: usize,
    pub processing_cards_count: usize,
    pub current_gc_cycle: usize,
    pub card_size: usize,
}

// Ensure CardState enum values for atomic operations
impl CardState {
    const Clean: u8 = 0;
    const Dirty: u8 = 1;
    const Processing: u8 = 2;
}

unsafe impl Send for CardTable {}
unsafe impl Sync for CardTable {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_table_creation() {
        let heap_base = 0x1000 as *const u8;
        let heap_size = 64 * 1024; // 64KB
        let card_size = 512;
        
        let card_table = CardTable::new(heap_base, heap_size, card_size);
        let stats = card_table.get_statistics();
        
        assert_eq!(stats.total_cards, heap_size / card_size);
        assert_eq!(stats.dirty_cards, 0);
    }

    #[test]
    fn test_card_marking() {
        let heap_base = 0x1000 as *const u8;
        let heap_size = 64 * 1024;
        let card_size = 512;
        
        let card_table = CardTable::new(heap_base, heap_size, card_size);
        
        // Mark a card as dirty
        let test_addr = unsafe { heap_base.add(1024) };
        card_table.mark_dirty(test_addr);
        
        // Check that the card is dirty
        assert!(card_table.is_dirty(test_addr));
        
        let stats = card_table.get_statistics();
        assert_eq!(stats.dirty_cards, 1);
    }

    #[test]
    fn test_card_clearing() {
        let heap_base = 0x1000 as *const u8;
        let heap_size = 64 * 1024;
        let card_size = 512;
        
        let card_table = CardTable::new(heap_base, heap_size, card_size);
        
        // Mark several cards as dirty
        for i in 0..10 {
            let test_addr = unsafe { heap_base.add(i * 1024) };
            card_table.mark_dirty(test_addr);
        }
        
        assert_eq!(card_table.get_statistics().dirty_cards, 10);
        
        // Clear all cards
        card_table.clear_all();
        
        assert_eq!(card_table.get_statistics().dirty_cards, 0);
    }

    #[test]
    fn test_batch_processing() {
        let heap_base = 0x1000 as *const u8;
        let heap_size = 64 * 1024;
        let card_size = 512;
        
        let config = CardTableConfig {
            batch_size: 5,
            ..Default::default()
        };
        
        let card_table = CardTable::with_config(heap_base, heap_size, config);
        
        // Mark many cards as dirty
        for i in 0..20 {
            let test_addr = unsafe { heap_base.add(i * 1024) };
            card_table.mark_dirty(test_addr);
        }
        
        // Get a batch for processing
        let batch = card_table.get_next_dirty_batch();
        assert_eq!(batch.len(), 5); // Should return batch_size cards
        
        // Mark batch as processed
        card_table.mark_cards_processed(&batch);
        
        let stats = card_table.get_statistics();
        assert_eq!(stats.dirty_cards, 15); // 20 - 5 processed
    }

    #[test]
    fn test_card_aging() {
        let heap_base = 0x1000 as *const u8;
        let heap_size = 64 * 1024;
        let card_size = 512;
        
        let config = CardTableConfig {
            enable_aging: true,
            ..Default::default()
        };
        
        let card_table = CardTable::with_config(heap_base, heap_size, config);
        
        // Mark a card as dirty
        let test_addr = unsafe { heap_base.add(1024) };
        card_table.mark_dirty(test_addr);
        
        // Age cards several times
        for _ in 0..10 {
            card_table.perform_maintenance();
        }
        
        // Clear the card and age it more
        card_table.clear_all();
        for _ in 0..10 {
            card_table.perform_maintenance();
        }
        
        // Should have stable cards now
        let stable_cards = card_table.get_stable_cards();
        assert!(!stable_cards.is_empty());
    }
} 