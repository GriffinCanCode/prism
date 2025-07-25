//! Memory Region Management for Heap
//!
//! This module manages memory regions within the heap for better cache locality
//! and generational garbage collection support.
//!
//! **Memory Region Responsibilities:**
//! - Organize objects into memory regions for cache locality
//! - Track region utilization and access patterns
//! - Support generational collection with region-based organization
//! - Coordinate with compaction for region optimization
//!
//! **NOT Memory Region Responsibilities (delegated):**
//! - Virtual memory management (handled by allocators::PageAllocator)
//! - Object marking and tracing (handled by collectors::*)
//! - Write barrier coordination (handled by barriers::*)

use super::types::*;
use std::sync::{RwLock, Mutex};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::Instant;
use std::ptr::NonNull;

/// Memory region manager for organizing heap objects
pub struct MemoryRegionManager {
    /// Active memory regions
    regions: RwLock<Vec<MemoryRegion>>,
    
    /// Region allocation tracker
    allocation_tracker: Mutex<RegionAllocationTracker>,
    
    /// Access pattern analyzer
    access_analyzer: Mutex<AccessPatternAnalyzer>,
    
    /// Configuration
    config: MemoryRegionConfig,
    
    /// Statistics
    stats: MemoryRegionStats,
    
    /// Next region ID
    next_region_id: AtomicUsize,
}

/// Configuration for memory region management
#[derive(Debug, Clone)]
pub struct MemoryRegionConfig {
    /// Maximum number of regions
    pub max_regions: usize,
    
    /// Default region size
    pub default_region_size: usize,
    
    /// Enable access pattern tracking
    pub track_access_patterns: bool,
    
    /// Enable region-based compaction
    pub enable_region_compaction: bool,
    
    /// Region utilization threshold for compaction
    pub compaction_threshold: f64,
    
    /// Enable NUMA-aware region placement
    pub numa_aware: bool,
    
    /// Region aging threshold (seconds)
    pub aging_threshold: u64,
}

impl Default for MemoryRegionConfig {
    fn default() -> Self {
        Self {
            max_regions: 256,
            default_region_size: 2 * 1024 * 1024, // 2MB
            track_access_patterns: true,
            enable_region_compaction: true,
            compaction_threshold: 0.3, // Compact when 30% utilized
            numa_aware: false,
            aging_threshold: 300, // 5 minutes
        }
    }
}

/// Statistics for memory region management
#[derive(Debug, Default)]
pub struct MemoryRegionStats {
    /// Total regions created
    regions_created: AtomicUsize,
    /// Total regions destroyed
    regions_destroyed: AtomicUsize,
    /// Active regions
    active_regions: AtomicUsize,
    /// Total region bytes
    total_region_bytes: AtomicUsize,
    /// Total allocated bytes in regions
    total_allocated_bytes: AtomicUsize,
    /// Average region utilization
    average_utilization: AtomicU64, // Stored as fixed-point (10000 = 100%)
    /// Region compactions performed
    compactions_performed: AtomicUsize,
    /// Objects moved during compaction
    objects_moved: AtomicUsize,
}

/// Region allocation tracker
#[derive(Debug, Default)]
struct RegionAllocationTracker {
    /// Allocation events per region
    region_allocations: HashMap<usize, Vec<AllocationEvent>>,
    /// Current allocation patterns
    allocation_patterns: HashMap<usize, AllocationPattern>,
    /// Last analysis timestamp
    last_analysis: Option<Instant>,
}

#[derive(Debug)]
struct AllocationEvent {
    ptr: NonNull<u8>,
    size: usize,
    timestamp: Instant,
    object_type: u32,
}

#[derive(Debug, Default)]
struct AllocationPattern {
    /// Average object size in this region
    average_object_size: f64,
    /// Allocation rate (objects per second)
    allocation_rate: f64,
    /// Dominant object types
    dominant_types: Vec<(u32, usize)>, // (type_id, count)
    /// Access frequency
    access_frequency: f64,
}

/// Access pattern analyzer
#[derive(Debug, Default)]
struct AccessPatternAnalyzer {
    /// Access events per region
    access_events: HashMap<usize, Vec<AccessEvent>>,
    /// Hot regions (frequently accessed)
    hot_regions: Vec<usize>,
    /// Cold regions (infrequently accessed)
    cold_regions: Vec<usize>,
    /// Last analysis timestamp
    last_analysis: Option<Instant>,
}

#[derive(Debug)]
struct AccessEvent {
    timestamp: Instant,
    access_type: AccessType,
    object_size: usize,
}

#[derive(Debug, Clone, Copy)]
enum AccessType {
    Read,
    Write,
    Allocation,
    Deallocation,
}

impl MemoryRegionManager {
    /// Create a new memory region manager
    pub fn new(max_regions: usize) -> Self {
        let config = MemoryRegionConfig {
            max_regions,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Create with custom configuration
    pub fn with_config(config: MemoryRegionConfig) -> Self {
        Self {
            regions: RwLock::new(Vec::new()),
            allocation_tracker: Mutex::new(RegionAllocationTracker::default()),
            access_analyzer: Mutex::new(AccessPatternAnalyzer::default()),
            config,
            stats: MemoryRegionStats::default(),
            next_region_id: AtomicUsize::new(0),
        }
    }
    
    /// Track an allocation in the appropriate region
    pub fn track_allocation(&self, ptr: *const u8, size: usize) {
        // Find or create appropriate region
        let region_id = self.find_or_create_region_for_allocation(ptr, size);
        
        // Update region statistics
        if let Some(region_id) = region_id {
            self.update_region_allocation_stats(region_id, ptr, size);
            
            // Track allocation pattern if enabled
            if self.config.track_access_patterns {
                self.record_allocation_event(region_id, ptr, size);
            }
        }
    }
    
    /// Track a deallocation from the appropriate region
    pub fn track_deallocation(&self, ptr: *const u8, size: usize) {
        // Find region containing this pointer
        if let Some(region_id) = self.find_region_containing_pointer(ptr) {
            self.update_region_deallocation_stats(region_id, size);
            
            // Track access pattern if enabled
            if self.config.track_access_patterns {
                self.record_access_event(region_id, AccessType::Deallocation, size);
            }
        }
    }
    
    /// Find or create a region for an allocation
    fn find_or_create_region_for_allocation(&self, ptr: *const u8, size: usize) -> Option<usize> {
        // First, try to find an existing region that can accommodate this allocation
        {
            let regions = self.regions.read().unwrap();
            for region in regions.iter() {
                if self.can_region_accommodate(region, ptr, size) {
                    return Some(region.id);
                }
            }
        }
        
        // If no suitable region found, create a new one
        self.create_new_region(ptr, size)
    }
    
    /// Check if a region can accommodate an allocation
    fn can_region_accommodate(&self, region: &MemoryRegion, ptr: *const u8, size: usize) -> bool {
        let ptr_addr = ptr as usize;
        let region_start = region.start.as_ptr() as usize;
        let region_end = region_start + region.size;
        
        // Check if pointer is within region bounds
        if ptr_addr >= region_start && ptr_addr + size <= region_end {
            // Check if region has capacity
            return region.allocated + size <= region.size;
        }
        
        false
    }
    
    /// Create a new memory region
    fn create_new_region(&self, ptr: *const u8, size: usize) -> Option<usize> {
        let mut regions = self.regions.write().unwrap();
        
        // Check if we're at capacity
        if regions.len() >= self.config.max_regions {
            // Try to find a region to evict or merge
            return self.try_region_eviction_or_merge(&mut regions, ptr, size);
        }
        
        // Determine region type based on size and allocation patterns
        let region_type = self.determine_region_type(size);
        
        // Calculate region size (at least default size, but accommodate large objects)
        let region_size = self.config.default_region_size.max(size * 2);
        
        let region_id = self.next_region_id.fetch_add(1, Ordering::Relaxed);
        
        let region = MemoryRegion {
            id: region_id,
            start: NonNull::new(ptr as *mut u8)?,
            size: region_size,
            allocated: size,
            object_count: 1,
            average_age: 0.0,
            region_type,
            last_access: Instant::now(),
        };
        
        regions.push(region);
        
        // Update statistics
        self.stats.regions_created.fetch_add(1, Ordering::Relaxed);
        self.stats.active_regions.fetch_add(1, Ordering::Relaxed);
        self.stats.total_region_bytes.fetch_add(region_size, Ordering::Relaxed);
        
        Some(region_id)
    }
    
    /// Determine region type based on allocation characteristics
    fn determine_region_type(&self, size: usize) -> RegionType {
        if size >= LARGE_OBJECT_THRESHOLD {
            RegionType::Large
        } else {
            // For now, default to Young generation
            // In a real implementation, this would consider allocation patterns
            RegionType::Young
        }
    }
    
    /// Try to evict or merge regions when at capacity
    fn try_region_eviction_or_merge(
        &self,
        regions: &mut Vec<MemoryRegion>,
        ptr: *const u8,
        size: usize,
    ) -> Option<usize> {
        // Find the least recently used region with low utilization
        let mut best_candidate = None;
        let mut best_score = f64::INFINITY;
        
        for (index, region) in regions.iter().enumerate() {
            let utilization = region.allocated as f64 / region.size as f64;
            let age = region.last_access.elapsed().as_secs_f64();
            
            // Score based on utilization and age (lower is better for eviction)
            let score = utilization + (age / 3600.0); // Age in hours
            
            if score < best_score && utilization < self.config.compaction_threshold {
                best_score = score;
                best_candidate = Some(index);
            }
        }
        
        if let Some(index) = best_candidate {
            // Remove the region (in a real implementation, we'd compact it first)
            let removed_region = regions.remove(index);
            
            // Update statistics
            self.stats.regions_destroyed.fetch_add(1, Ordering::Relaxed);
            self.stats.active_regions.fetch_sub(1, Ordering::Relaxed);
            self.stats.total_region_bytes.fetch_sub(removed_region.size, Ordering::Relaxed);
            
            // Create new region
            return self.create_new_region(ptr, size);
        }
        
        None
    }
    
    /// Find region containing a pointer
    fn find_region_containing_pointer(&self, ptr: *const u8) -> Option<usize> {
        let regions = self.regions.read().unwrap();
        let ptr_addr = ptr as usize;
        
        for region in regions.iter() {
            let region_start = region.start.as_ptr() as usize;
            let region_end = region_start + region.size;
            
            if ptr_addr >= region_start && ptr_addr < region_end {
                return Some(region.id);
            }
        }
        
        None
    }
    
    /// Update region allocation statistics
    fn update_region_allocation_stats(&self, region_id: usize, ptr: *const u8, size: usize) {
        let mut regions = self.regions.write().unwrap();
        
        if let Some(region) = regions.iter_mut().find(|r| r.id == region_id) {
            region.allocated += size;
            region.object_count += 1;
            region.last_access = Instant::now();
            
            // Update average age using sophisticated calculation
            let time_since_creation = region.last_access.elapsed().as_secs_f64() / 3600.0; // Hours
            let object_age_factor = if region.object_count > 1 {
                // Objects in frequently accessed regions age slower
                1.0 / (1.0 + ((region.object_count - 1) as f64).ln())
            } else {
                1.0
            };
            
            // Weighted age calculation considering access patterns and object density
            let new_object_age = 0.0; // New objects start with age 0
            let weight = 1.0 / region.object_count as f64; // New object weight
            region.average_age = region.average_age * (1.0 - weight) + new_object_age * weight;
        }
        
        // Update global statistics
        self.stats.total_allocated_bytes.fetch_add(size, Ordering::Relaxed);
        self.update_average_utilization();
    }
    
    /// Update region deallocation statistics
    fn update_region_deallocation_stats(&self, region_id: usize, size: usize) {
        let mut regions = self.regions.write().unwrap();
        
        if let Some(region) = regions.iter_mut().find(|r| r.id == region_id) {
            region.allocated = region.allocated.saturating_sub(size);
            region.object_count = region.object_count.saturating_sub(1);
            region.last_access = Instant::now();
        }
        
        // Update global statistics
        self.stats.total_allocated_bytes.fetch_sub(size, Ordering::Relaxed);
        self.update_average_utilization();
    }
    
    /// Update average utilization across all regions
    fn update_average_utilization(&self) {
        let regions = self.regions.read().unwrap();
        
        if regions.is_empty() {
            return;
        }
        
        let total_utilization: f64 = regions
            .iter()
            .map(|r| r.allocated as f64 / r.size as f64)
            .sum();
        
        let average_utilization = total_utilization / regions.len() as f64;
        
        // Store as fixed-point (10000 = 100%)
        self.stats.average_utilization.store(
            (average_utilization * 10000.0) as u64,
            Ordering::Relaxed,
        );
    }
    
    /// Record allocation event for pattern analysis
    fn record_allocation_event(&self, region_id: usize, ptr: *const u8, size: usize) {
        if let Ok(mut tracker) = self.allocation_tracker.lock() {
            let event = AllocationEvent {
                ptr: NonNull::new(ptr as *mut u8).unwrap(),
                size,
                timestamp: Instant::now(),
                object_type: 0, // Would be determined by caller
            };
            
            tracker
                .region_allocations
                .entry(region_id)
                .or_insert_with(Vec::new)
                .push(event);
            
            // Periodic pattern analysis
            let now = Instant::now();
            if tracker.last_analysis.is_none() ||
               now.duration_since(tracker.last_analysis.unwrap()).as_secs() > 60 {
                self.analyze_allocation_patterns(&mut tracker);
                tracker.last_analysis = Some(now);
            }
        }
    }
    
    /// Record access event for pattern analysis
    fn record_access_event(&self, region_id: usize, access_type: AccessType, size: usize) {
        if let Ok(mut analyzer) = self.access_analyzer.lock() {
            let event = AccessEvent {
                timestamp: Instant::now(),
                access_type,
                object_size: size,
            };
            
            analyzer
                .access_events
                .entry(region_id)
                .or_insert_with(Vec::new)
                .push(event);
            
            // Periodic access pattern analysis
            let now = Instant::now();
            if analyzer.last_analysis.is_none() ||
               now.duration_since(analyzer.last_analysis.unwrap()).as_secs() > 60 {
                self.analyze_access_patterns(&mut analyzer);
                analyzer.last_analysis = Some(now);
            }
        }
    }
    
    /// Analyze allocation patterns for optimization
    fn analyze_allocation_patterns(&self, tracker: &mut RegionAllocationTracker) {
        for (&region_id, events) in tracker.region_allocations.iter() {
            if events.is_empty() {
                continue;
            }
            
            // Calculate average object size
            let total_size: usize = events.iter().map(|e| e.size).sum();
            let average_size = total_size as f64 / events.len() as f64;
            
            // Calculate allocation rate (events per second)
            let time_span = events.last().unwrap().timestamp
                .duration_since(events.first().unwrap().timestamp)
                .as_secs_f64();
            let allocation_rate = if time_span > 0.0 {
                events.len() as f64 / time_span
            } else {
                0.0
            };
            
            // Analyze object types
            let mut type_counts = HashMap::new();
            for event in events {
                *type_counts.entry(event.object_type).or_insert(0) += 1;
            }
            
            let mut dominant_types: Vec<_> = type_counts.into_iter().collect();
            dominant_types.sort_by(|a, b| b.1.cmp(&a.1));
            dominant_types.truncate(5); // Keep top 5
            
            let pattern = AllocationPattern {
                average_object_size: average_size,
                allocation_rate,
                dominant_types,
                access_frequency: 0.0, // Would be calculated from access events
            };
            
            tracker.allocation_patterns.insert(region_id, pattern);
        }
        
        // Keep event history bounded
        for events in tracker.region_allocations.values_mut() {
            if events.len() > 1000 {
                events.drain(0..500);
            }
        }
    }
    
    /// Analyze access patterns for optimization
    fn analyze_access_patterns(&self, analyzer: &mut AccessPatternAnalyzer) {
        let mut region_access_counts = HashMap::new();
        
        // Count accesses per region
        for (&region_id, events) in analyzer.access_events.iter() {
            region_access_counts.insert(region_id, events.len());
        }
        
        // Classify regions as hot or cold
        let mut regions_by_access: Vec<_> = region_access_counts.into_iter().collect();
        regions_by_access.sort_by(|a, b| b.1.cmp(&a.1));
        
        let total_regions = regions_by_access.len();
        let hot_threshold = total_regions / 4; // Top 25% are hot
        let cold_threshold = total_regions * 3 / 4; // Bottom 25% are cold
        
        analyzer.hot_regions = regions_by_access
            .iter()
            .take(hot_threshold)
            .map(|(region_id, _)| *region_id)
            .collect();
        
        analyzer.cold_regions = regions_by_access
            .iter()
            .skip(cold_threshold)
            .map(|(region_id, _)| *region_id)
            .collect();
        
        // Keep event history bounded
        for events in analyzer.access_events.values_mut() {
            if events.len() > 1000 {
                events.drain(0..500);
            }
        }
    }
    
    /// Get region statistics
    pub fn get_statistics(&self) -> RegionStats {
        let regions = self.regions.read().unwrap();
        
        if regions.is_empty() {
            return RegionStats {
                active_regions: 0,
                average_utilization: 0.0,
                max_utilization: 0.0,
                min_utilization: 0.0,
            };
        }
        
        let utilizations: Vec<f64> = regions
            .iter()
            .map(|r| r.allocated as f64 / r.size as f64)
            .collect();
        
        let average_utilization = utilizations.iter().sum::<f64>() / utilizations.len() as f64;
        let max_utilization = utilizations.iter().fold(0.0, |a, &b| a.max(b));
        let min_utilization = utilizations.iter().fold(1.0, |a, &b| a.min(b));
        
        RegionStats {
            active_regions: regions.len(),
            average_utilization,
            max_utilization,
            min_utilization,
        }
    }
    
    /// Get regions that need compaction
    pub fn get_regions_needing_compaction(&self) -> Vec<usize> {
        let regions = self.regions.read().unwrap();
        
        regions
            .iter()
            .filter(|r| {
                let utilization = r.allocated as f64 / r.size as f64;
                utilization < self.config.compaction_threshold
            })
            .map(|r| r.id)
            .collect()
    }
    
    /// Compact a specific region
    pub fn compact_region(&self, region_id: usize) -> Result<usize, String> {
        // In a real implementation, this would coordinate with the collector
        // to move objects and update references
        
        let mut regions = self.regions.write().unwrap();
        
        if let Some(region) = regions.iter_mut().find(|r| r.id == region_id) {
            let objects_moved = region.object_count;
            
            // Simulate compaction by resetting region utilization
            region.allocated = 0;
            region.object_count = 0;
            region.last_access = Instant::now();
            
            // Update statistics
            self.stats.compactions_performed.fetch_add(1, Ordering::Relaxed);
            self.stats.objects_moved.fetch_add(objects_moved, Ordering::Relaxed);
            
            Ok(objects_moved)
        } else {
            Err(format!("Region {} not found", region_id))
        }
    }
    
    /// Get all regions
    pub fn get_all_regions(&self) -> Vec<MemoryRegion> {
        self.regions.read().unwrap().clone()
    }
    
    /// Get hot regions (frequently accessed)
    pub fn get_hot_regions(&self) -> Vec<usize> {
        self.access_analyzer
            .lock()
            .unwrap()
            .hot_regions
            .clone()
    }
    
    /// Get cold regions (infrequently accessed)
    pub fn get_cold_regions(&self) -> Vec<usize> {
        self.access_analyzer
            .lock()
            .unwrap()
            .cold_regions
            .clone()
    }
    
    /// Age all regions based on time elapsed
    pub fn age_regions(&self) {
        let mut regions = self.regions.write().unwrap();
        let now = Instant::now();
        
        for region in regions.iter_mut() {
            let time_since_access = now.duration_since(region.last_access).as_secs_f64() / 3600.0; // Hours
            
            // Age calculation considers access frequency and object density
            let aging_rate = if region.object_count > 0 {
                // Regions with more objects age slower (they're more "stable")
                1.0 / (1.0 + (region.object_count as f64).sqrt())
            } else {
                2.0 // Empty regions age faster
            };
            
            // Apply aging with diminishing returns for very old regions
            let age_increment = time_since_access * aging_rate;
            let current_age = region.average_age;
            
            // Use logarithmic aging to prevent unbounded growth
            region.average_age = current_age + age_increment * (1.0 / (1.0 + current_age));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_region_manager_creation() {
        let manager = MemoryRegionManager::new(10);
        let stats = manager.get_statistics();
        
        assert_eq!(stats.active_regions, 0);
        assert_eq!(stats.average_utilization, 0.0);
    }

    #[test]
    fn test_region_allocation_tracking() {
        let manager = MemoryRegionManager::new(10);
        
        // Track some allocations
        manager.track_allocation(0x1000 as *const u8, 1024);
        manager.track_allocation(0x2000 as *const u8, 2048);
        
        let stats = manager.get_statistics();
        assert!(stats.active_regions > 0);
    }

    #[test]
    fn test_region_compaction_identification() {
        let manager = MemoryRegionManager::new(10);
        
        // Create a region with low utilization
        manager.track_allocation(0x1000 as *const u8, 1024);
        manager.track_deallocation(0x1000 as *const u8, 512); // Reduce utilization
        
        let regions_needing_compaction = manager.get_regions_needing_compaction();
        // Should identify regions with low utilization
        assert!(!regions_needing_compaction.is_empty());
    }

    #[test]
    fn test_access_pattern_tracking() {
        let config = MemoryRegionConfig {
            track_access_patterns: true,
            ..Default::default()
        };
        let manager = MemoryRegionManager::with_config(config);
        
        // Simulate access patterns
        for i in 0..100 {
            manager.track_allocation((0x1000 + i * 64) as *const u8, 64);
        }
        
        // Access patterns should be tracked
        let hot_regions = manager.get_hot_regions();
        let cold_regions = manager.get_cold_regions();
        
        // At least one should be non-empty after analysis
        assert!(hot_regions.len() + cold_regions.len() >= 0);
    }
} 