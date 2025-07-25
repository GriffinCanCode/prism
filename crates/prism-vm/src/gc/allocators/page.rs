//! Page Allocator - Low-level memory page management
//!
//! This allocator manages memory pages that are used by other allocators
//! for their internal memory management. It provides:
//!
//! - Page-aligned memory allocation
//! - Page tracking and reuse
//! - NUMA-aware allocation (future)
//! - Statistics and monitoring

use super::types::*;
use super::Allocator;

use std::ptr::NonNull;
use std::sync::{Mutex, Arc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::VecDeque;

/// NUMA node detection and management
#[derive(Debug, Clone)]
pub struct NumaInfo {
    /// Number of NUMA nodes in the system
    pub node_count: usize,
    /// Current preferred node for allocations
    pub preferred_node: usize,
    /// Whether NUMA is available on this system
    pub numa_available: bool,
    /// Node topology information
    pub node_topology: Vec<NumaNodeInfo>,
}

#[derive(Debug, Clone)]
pub struct NumaNodeInfo {
    /// Node ID
    pub node_id: usize,
    /// Available memory on this node (bytes)
    pub available_memory: usize,
    /// CPU cores associated with this node
    pub cpu_cores: Vec<usize>,
    /// Distance to other nodes (for affinity)
    pub distances: Vec<u32>,
}

impl Default for NumaInfo {
    fn default() -> Self {
        Self {
            node_count: 1,
            preferred_node: 0,
            numa_available: false,
            node_topology: vec![NumaNodeInfo {
                node_id: 0,
                available_memory: 0,
                cpu_cores: Vec::new(),
                distances: vec![0],
            }],
        }
    }
}

impl NumaInfo {
    /// Detect NUMA topology on the current system
    pub fn detect() -> Self {
        // On Unix systems, we can check /sys/devices/system/node/
        // On Windows, we'd use GetNumaHighestNodeNumber
        // For now, implement a cross-platform detection approach
        
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // Fallback for other systems
            Self::default()
        }
    }
    
    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;
        use std::path::Path;
        
        let node_path = Path::new("/sys/devices/system/node/");
        if !node_path.exists() {
            return Self::default();
        }
        
        let mut nodes = Vec::new();
        let mut node_count = 0;
        
        // Read available nodes
        if let Ok(entries) = fs::read_dir(node_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                
                if name_str.starts_with("node") {
                    if let Ok(node_id) = name_str[4..].parse::<usize>() {
                        node_count = node_count.max(node_id + 1);
                        
                        // Try to read memory info
                        let meminfo_path = entry.path().join("meminfo");
                        let available_memory = if let Ok(content) = fs::read_to_string(&meminfo_path) {
                            Self::parse_node_memory(&content)
                        } else {
                            0
                        };
                        
                        // Try to read CPU list
                        let cpulist_path = entry.path().join("cpulist");
                        let cpu_cores = if let Ok(content) = fs::read_to_string(&cpulist_path) {
                            Self::parse_cpu_list(&content)
                        } else {
                            Vec::new()
                        };
                        
                        nodes.push(NumaNodeInfo {
                            node_id,
                            available_memory,
                            cpu_cores,
                            distances: vec![0; node_count], // Will be filled later
                        });
                    }
                }
            }
        }
        
        if nodes.is_empty() {
            return Self::default();
        }
        
        // Sort nodes by ID
        nodes.sort_by_key(|n| n.node_id);
        
        // Fill in distance information
        for (i, node) in nodes.iter_mut().enumerate() {
            node.distances = vec![0; node_count];
            
            // Try to read distance information
            let distance_path = format!("/sys/devices/system/node/node{}/distance", node.node_id);
            if let Ok(content) = fs::read_to_string(&distance_path) {
                let distances: Vec<u32> = content
                    .trim()
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                
                if distances.len() == node_count {
                    node.distances = distances;
                }
            }
        }
        
        Self {
            node_count,
            preferred_node: 0, // Start with node 0
            numa_available: true,
            node_topology: nodes,
        }
    }
    
    #[cfg(target_os = "windows")]
    fn detect_windows() -> Self {
        // On Windows, we'd use GetNumaHighestNodeNumber and related APIs
        // For now, provide a simplified implementation
        
        // This would require Windows API bindings
        // extern "system" {
        //     fn GetNumaHighestNodeNumber(highest_node_number: *mut u32) -> i32;
        //     fn GetNumaAvailableMemoryNode(node: u8, available_bytes: *mut u64) -> i32;
        // }
        
        // For now, assume single node on Windows
        Self::default()
    }
    
    #[cfg(target_os = "linux")]
    fn parse_node_memory(content: &str) -> usize {
        // Parse /sys/devices/system/node/nodeN/meminfo
        // Look for "Node N MemFree:" line
        for line in content.lines() {
            if line.contains("MemFree:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(kb) = parts[3].parse::<usize>() {
                        return kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        0
    }
    
    #[cfg(target_os = "linux")]
    fn parse_cpu_list(content: &str) -> Vec<usize> {
        // Parse CPU list like "0-3,8-11" or "0,2,4,6"
        let mut cpus = Vec::new();
        
        for part in content.trim().split(',') {
            if part.contains('-') {
                // Range like "0-3"
                let range_parts: Vec<&str> = part.split('-').collect();
                if range_parts.len() == 2 {
                    if let (Ok(start), Ok(end)) = (range_parts[0].parse::<usize>(), range_parts[1].parse::<usize>()) {
                        for cpu in start..=end {
                            cpus.push(cpu);
                        }
                    }
                }
            } else {
                // Single CPU
                if let Ok(cpu) = part.parse::<usize>() {
                    cpus.push(cpu);
                }
            }
        }
        
        cpus
    }
    
    /// Get the optimal NUMA node for allocation
    pub fn get_optimal_node(&self) -> usize {
        if !self.numa_available {
            return 0;
        }
        
        // Simple policy: prefer node with most available memory
        self.node_topology
            .iter()
            .max_by_key(|node| node.available_memory)
            .map(|node| node.node_id)
            .unwrap_or(0)
    }
    
    /// Update preferred node based on current thread affinity
    pub fn update_preferred_node(&mut self) {
        if !self.numa_available {
            return;
        }
        
        // Try to get current CPU and map to NUMA node
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            
            // Get current CPU from /proc/self/stat
            if let Ok(stat) = fs::read_to_string("/proc/self/stat") {
                let fields: Vec<&str> = stat.split_whitespace().collect();
                if fields.len() > 38 {
                    if let Ok(cpu) = fields[38].parse::<usize>() {
                        // Find which NUMA node this CPU belongs to
                        for node in &self.node_topology {
                            if node.cpu_cores.contains(&cpu) {
                                self.preferred_node = node.node_id;
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Page allocator for managing memory pages
pub struct PageAllocator {
    /// Available pages organized by size
    free_pages: Mutex<FreePageLists>,
    /// Total allocated pages
    allocated_pages: AtomicUsize,
    /// Total page bytes allocated
    total_page_bytes: AtomicUsize,
    /// Statistics
    stats: Arc<Mutex<PageAllocationStats>>,
    /// Configuration
    config: PageAllocatorConfig,
    /// NUMA information
    numa_info: Arc<Mutex<NumaInfo>>,
}

/// Configuration for page allocator
#[derive(Debug, Clone)]
pub struct PageAllocatorConfig {
    /// Enable page reuse
    pub enable_page_reuse: bool,
    /// Maximum pages to keep in free lists
    pub max_free_pages: usize,
    /// Enable NUMA awareness
    pub numa_aware: bool,
    /// Page size (should be system page size)
    pub page_size: usize,
}

impl Default for PageAllocatorConfig {
    fn default() -> Self {
        Self {
            enable_page_reuse: true,
            max_free_pages: 1000,
            numa_aware: false,
            page_size: PAGE_SIZE,
        }
    }
}

/// Free page lists organized by page count
#[derive(Debug)]
struct FreePageLists {
    /// Single pages (most common)
    single_pages: VecDeque<Page>,
    /// Small multi-page allocations (2-4 pages)
    small_multi_pages: VecDeque<Page>,
    /// Large multi-page allocations (5+ pages)
    large_multi_pages: VecDeque<Page>,
    /// Total pages in free lists
    total_free_pages: usize,
}

impl Default for FreePageLists {
    fn default() -> Self {
        Self {
            single_pages: VecDeque::new(),
            small_multi_pages: VecDeque::new(),
            large_multi_pages: VecDeque::new(),
            total_free_pages: 0,
        }
    }
}

/// Statistics for page allocation
#[derive(Debug, Default, Clone)]
pub struct PageAllocationStats {
    /// Total page allocations
    pub total_allocations: usize,
    /// Total page deallocations
    pub total_deallocations: usize,
    /// Current allocated pages
    pub allocated_pages: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Current live bytes
    pub live_bytes: usize,
    /// Pages reused from free lists
    pub pages_reused: usize,
    /// Pages returned to free lists
    pub pages_freed: usize,
    /// System allocations (not from free lists)
    pub system_allocations: usize,
    /// Peak page usage
    pub peak_pages: usize,
    /// Peak memory usage
    pub peak_memory: usize,
}

impl FreePageLists {
    /// Find a suitable page from free lists
    fn find_suitable_page(&mut self, count: usize) -> Option<Page> {
        let page = match count {
            1 => self.single_pages.pop_front(),
            2..=4 => {
                // Try small multi-pages first, then large
                self.small_multi_pages.iter()
                    .position(|p| p.count >= count)
                    .and_then(|idx| self.small_multi_pages.remove(idx))
                    .or_else(|| {
                        self.large_multi_pages.iter()
                            .position(|p| p.count >= count)
                            .and_then(|idx| self.large_multi_pages.remove(idx))
                    })
            }
            _ => {
                // Large allocation - check large multi-pages
                self.large_multi_pages.iter()
                    .position(|p| p.count >= count)
                    .and_then(|idx| self.large_multi_pages.remove(idx))
            }
        };
        
        if page.is_some() {
            self.total_free_pages = self.total_free_pages.saturating_sub(count);
        }
        
        page
    }
    
    /// Add a page to appropriate free list
    fn add_page(&mut self, page: Page, max_free_pages: usize) -> bool {
        if self.total_free_pages >= max_free_pages {
            return false; // Don't add if we have too many free pages
        }
        
        match page.count {
            1 => self.single_pages.push_back(page),
            2..=4 => self.small_multi_pages.push_back(page),
            _ => self.large_multi_pages.push_back(page),
        }
        
        self.total_free_pages += page.count;
        true
    }
    
    /// Get total number of free pages
    fn total_pages(&self) -> usize {
        self.total_free_pages
    }
    
    /// Clear all free pages (for cleanup)
    fn clear(&mut self) {
        self.single_pages.clear();
        self.small_multi_pages.clear();
        self.large_multi_pages.clear();
        self.total_free_pages = 0;
    }
}

impl PageAllocator {
    pub fn new() -> Self {
        Self::with_config(PageAllocatorConfig::default())
    }
    
    pub fn with_config(config: PageAllocatorConfig) -> Self {
        // Detect NUMA topology if NUMA awareness is enabled
        let numa_info = if config.numa_aware {
            Arc::new(Mutex::new(NumaInfo::detect()))
        } else {
            Arc::new(Mutex::new(NumaInfo::default()))
        };
        
        Self {
            free_pages: Mutex::new(FreePageLists::default()),
            allocated_pages: AtomicUsize::new(0),
            total_page_bytes: AtomicUsize::new(0),
            stats: Arc::new(Mutex::new(PageAllocationStats::default())),
            config,
            numa_info,
        }
    }
    
    /// Allocate pages with specified count
    pub fn allocate_pages(&self, count: usize) -> Option<Page> {
        if count == 0 {
            return None;
        }
        
        // Try to reuse existing pages first
        if self.config.enable_page_reuse {
            if let Ok(mut free_pages) = self.free_pages.lock() {
                if let Some(page) = free_pages.find_suitable_page(count) {
                    // Update statistics
                    self.update_reuse_stats(count);
                    return Some(page);
                }
            }
        }
        
        // Allocate new pages from system
        self.allocate_new_pages(count)
    }
    
    /// Allocate new pages from system
    fn allocate_new_pages(&self, count: usize) -> Option<Page> {
        let total_size = count * self.config.page_size;
        let layout = Layout::from_size_align(total_size, self.config.page_size).ok()?;
        
        // Get optimal NUMA node
        let numa_node = if self.config.numa_aware {
            let numa_info = self.numa_info.lock().unwrap();
            numa_info.get_optimal_node()
        } else {
            0
        };
        
        // Allocate memory (with NUMA binding if available)
        let ptr = if self.config.numa_aware && numa_node > 0 {
            self.allocate_on_numa_node(layout, numa_node)
        } else {
            unsafe { alloc(layout) }
        };
        
        if let Some(addr) = NonNull::new(ptr) {
            // Update counters
            self.allocated_pages.fetch_add(count, Ordering::Relaxed);
            self.total_page_bytes.fetch_add(total_size, Ordering::Relaxed);
            
            // Update statistics
            self.update_allocation_stats(count, total_size);
            
            Some(Page {
                addr,
                count,
                numa_node,
            })
        } else {
            None
        }
    }
    
    /// Deallocate pages (potentially returning to free list)
    pub fn deallocate_pages(&self, page: Page) {
        let should_reuse = self.config.enable_page_reuse;
        let mut should_free_to_system = true;
        
        if should_reuse {
            if let Ok(mut free_pages) = self.free_pages.lock() {
                if free_pages.add_page(page.clone(), self.config.max_free_pages) {
                    should_free_to_system = false;
                    
                    // Update statistics
                    self.update_free_stats(page.count);
                }
            }
        }
        
        if should_free_to_system {
            // Free to system
            let total_size = page.count * self.config.page_size;
            let layout = Layout::from_size_align(total_size, self.config.page_size)
                .expect("Invalid layout for page deallocation");
            
            unsafe {
                dealloc(page.addr.as_ptr(), layout);
            }
            
            // Update counters
            self.allocated_pages.fetch_sub(page.count, Ordering::Relaxed);
            self.total_page_bytes.fetch_sub(total_size, Ordering::Relaxed);
            
            // Update statistics
            self.update_deallocation_stats(page.count);
        }
    }
    
    /// Update statistics for allocation
    fn update_allocation_stats(&self, page_count: usize, bytes: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocations += 1;
            stats.allocated_pages += page_count;
            stats.total_bytes_allocated += bytes;
            stats.live_bytes += bytes;
            stats.system_allocations += 1;
            
            // Update peaks
            if stats.allocated_pages > stats.peak_pages {
                stats.peak_pages = stats.allocated_pages;
            }
            if stats.live_bytes > stats.peak_memory {
                stats.peak_memory = stats.live_bytes;
            }
        }
    }
    
    /// Update statistics for deallocation
    fn update_deallocation_stats(&self, page_count: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_deallocations += 1;
            stats.allocated_pages = stats.allocated_pages.saturating_sub(page_count);
            let bytes = page_count * self.config.page_size;
            stats.live_bytes = stats.live_bytes.saturating_sub(bytes);
        }
    }
    
    /// Update statistics for page reuse
    fn update_reuse_stats(&self, page_count: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocations += 1;
            stats.allocated_pages += page_count;
            stats.pages_reused += page_count;
            let bytes = page_count * self.config.page_size;
            stats.live_bytes += bytes;
            
            // Update peaks
            if stats.allocated_pages > stats.peak_pages {
                stats.peak_pages = stats.allocated_pages;
            }
            if stats.live_bytes > stats.peak_memory {
                stats.peak_memory = stats.live_bytes;
            }
        }
    }
    
    /// Update statistics for freeing to free list
    fn update_free_stats(&self, page_count: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_deallocations += 1;
            stats.pages_freed += page_count;
            stats.allocated_pages = stats.allocated_pages.saturating_sub(page_count);
            let bytes = page_count * self.config.page_size;
            stats.live_bytes = stats.live_bytes.saturating_sub(bytes);
        }
    }
    
    /// Get detailed page allocation statistics
    pub fn get_detailed_stats(&self) -> PageAllocationStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Get number of free pages available for reuse
    pub fn free_page_count(&self) -> usize {
        self.free_pages.lock().unwrap().total_pages()
    }
    
    /// Clear all free pages (useful for cleanup or memory pressure)
    pub fn clear_free_pages(&self) {
        if let Ok(mut free_pages) = self.free_pages.lock() {
            // Free all pages to system
            let total_pages = free_pages.total_pages();
            let total_bytes = total_pages * self.config.page_size;
            
            // Free single pages
            while let Some(page) = free_pages.single_pages.pop_front() {
                let layout = Layout::from_size_align(
                    page.count * self.config.page_size, 
                    self.config.page_size
                ).expect("Invalid layout");
                unsafe {
                    dealloc(page.addr.as_ptr(), layout);
                }
            }
            
            // Free small multi-pages
            while let Some(page) = free_pages.small_multi_pages.pop_front() {
                let layout = Layout::from_size_align(
                    page.count * self.config.page_size, 
                    self.config.page_size
                ).expect("Invalid layout");
                unsafe {
                    dealloc(page.addr.as_ptr(), layout);
                }
            }
            
            // Free large multi-pages
            while let Some(page) = free_pages.large_multi_pages.pop_front() {
                let layout = Layout::from_size_align(
                    page.count * self.config.page_size, 
                    self.config.page_size
                ).expect("Invalid layout");
                unsafe {
                    dealloc(page.addr.as_ptr(), layout);
                }
            }
            
            free_pages.clear();
            
            // Update counters
            self.allocated_pages.fetch_sub(total_pages, Ordering::Relaxed);
            self.total_page_bytes.fetch_sub(total_bytes, Ordering::Relaxed);
        }
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> PageAllocatorConfig {
        self.config.clone()
    }
    
    /// Update configuration
    pub fn set_config(&self, new_config: PageAllocatorConfig) {
        // Update configuration with proper synchronization
        let old_config = self.config.clone();
        
        // Check if NUMA awareness changed
        if old_config.numa_aware != new_config.numa_aware {
            if new_config.numa_aware {
                // Enable NUMA awareness - detect topology
                let mut numa_info = self.numa_info.lock().unwrap();
                *numa_info = NumaInfo::detect();
            } else {
                // Disable NUMA awareness - reset to default
                let mut numa_info = self.numa_info.lock().unwrap();
                *numa_info = NumaInfo::default();
            }
        }
        
        // If page reuse setting changed, handle existing free pages
        if old_config.enable_page_reuse != new_config.enable_page_reuse {
            if !new_config.enable_page_reuse {
                // Disabled page reuse - clear all free pages
                self.clear_free_pages();
            }
        }
        
        // If max_free_pages was reduced, trim excess pages
        if new_config.max_free_pages < old_config.max_free_pages {
            if let Ok(mut free_pages) = self.free_pages.lock() {
                while free_pages.total_pages() > new_config.max_free_pages {
                    // Remove pages from largest category first
                    if let Some(page) = free_pages.large_multi_pages.pop_front() {
                        self.deallocate_page_to_system(page);
                        free_pages.total_free_pages = free_pages.total_free_pages.saturating_sub(page.count);
                    } else if let Some(page) = free_pages.small_multi_pages.pop_front() {
                        self.deallocate_page_to_system(page);
                        free_pages.total_free_pages = free_pages.total_free_pages.saturating_sub(page.count);
                    } else if let Some(page) = free_pages.single_pages.pop_front() {
                        self.deallocate_page_to_system(page);
                        free_pages.total_free_pages = free_pages.total_free_pages.saturating_sub(page.count);
                    } else {
                        break;
                    }
                }
            }
        }
        
        // Update the configuration atomically
        // In a real implementation, this would use atomic operations or RwLock
        // For now, we'll use unsafe to update the config field
        unsafe {
            let config_ptr = &self.config as *const PageAllocatorConfig as *mut PageAllocatorConfig;
            std::ptr::write(config_ptr, new_config);
        }
    }
    
    /// Deallocate a page directly to system (helper for config changes)
    fn deallocate_page_to_system(&self, page: Page) {
        let total_size = page.count * self.config.page_size;
        let layout = Layout::from_size_align(total_size, self.config.page_size)
            .expect("Invalid layout for page deallocation");
        
        unsafe {
            dealloc(page.addr.as_ptr(), layout);
        }
        
        // Update counters
        self.allocated_pages.fetch_sub(page.count, Ordering::Relaxed);
        self.total_page_bytes.fetch_sub(total_size, Ordering::Relaxed);
    }

    /// Get current NUMA information
    pub fn get_numa_info(&self) -> NumaInfo {
        self.numa_info.lock().unwrap().clone()
    }

    /// Update NUMA node preferences
    pub fn update_numa_preferences(&self) {
        if self.config.numa_aware {
            let mut numa_info = self.numa_info.lock().unwrap();
            numa_info.update_preferred_node();
        }
    }
    
    /// Allocate memory on specific NUMA node (platform-specific)
    fn allocate_on_numa_node(&self, layout: Layout, numa_node: usize) -> *mut u8 {
        #[cfg(target_os = "linux")]
        {
            // On Linux, we could use mbind() or numa_alloc_onnode()
            // For now, fall back to regular allocation
            // In a full implementation, you'd use libnuma bindings
            unsafe { alloc(layout) }
        }
        
        #[cfg(target_os = "windows")]
        {
            // On Windows, we could use VirtualAllocExNuma()
            // For now, fall back to regular allocation
            unsafe { alloc(layout) }
        }
        
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            unsafe { alloc(layout) }
        }
    }
}

impl Allocator for PageAllocator {
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Calculate number of pages needed
        let pages_needed = (size + self.config.page_size - 1) / self.config.page_size;
        
        // Ensure alignment is compatible with page alignment
        if align > self.config.page_size {
            return None; // Cannot satisfy alignment requirement
        }
        
        // Allocate pages
        let page = self.allocate_pages(pages_needed)?;
        Some(page.addr)
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        // Calculate number of pages
        let pages_needed = (size + self.config.page_size - 1) / self.config.page_size;
        
        // Create page structure for deallocation
        let page = Page {
            addr: ptr,
            count: pages_needed,
            numa_node: 0, // We don't track NUMA node for deallocations in this simple case
        };
        
        self.deallocate_pages(page);
    }
    
    fn stats(&self) -> AllocationStats {
        let page_stats = self.get_detailed_stats();
        
        AllocationStats {
            total_allocated: page_stats.total_bytes_allocated,
            total_deallocated: page_stats.total_bytes_allocated - page_stats.live_bytes,
            live_bytes: page_stats.live_bytes,
            allocation_count: page_stats.total_allocations,
            deallocation_count: page_stats.total_deallocations,
            peak_memory: page_stats.peak_memory,
            large_object_count: 0,
            large_object_bytes: 0,
            page_bytes: page_stats.live_bytes,
            metadata_memory: 0,
            memory_overhead: 0,
            barrier_calls: 0,
        }
    }
    
    fn should_trigger_gc(&self) -> bool {
        // Page allocator doesn't directly trigger GC
        // It's used by other allocators
        false
    }
    
    fn prepare_for_gc(&self) {
        // Could clear free pages to reduce memory pressure
        if self.free_page_count() > self.config.max_free_pages / 2 {
            self.clear_free_pages();
        }
    }
    
    fn get_config(&self) -> AllocatorConfig {
        AllocatorConfig {
            enable_thread_cache: false,
            gc_trigger_threshold: 0, // Page allocator doesn't trigger GC
            numa_aware: self.config.numa_aware,
        }
    }
    
    fn reconfigure(&self, config: AllocatorConfig) {
        // Update relevant settings
        let mut new_config = self.config.clone();
        new_config.numa_aware = config.numa_aware;
        self.set_config(new_config);
    }
}

impl Default for PageAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_page_allocator_creation() {
        let allocator = PageAllocator::new();
        let stats = allocator.get_detailed_stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.allocated_pages, 0);
    }
    
    #[test]
    fn test_single_page_allocation() {
        let allocator = PageAllocator::new();
        
        let page = allocator.allocate_pages(1).expect("Failed to allocate single page");
        assert_eq!(page.count, 1);
        assert!(!page.addr.as_ptr().is_null());
        
        let stats = allocator.get_detailed_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.allocated_pages, 1);
        assert_eq!(stats.system_allocations, 1);
        
        // Clean up
        allocator.deallocate_pages(page);
        
        let stats_after = allocator.get_detailed_stats();
        assert_eq!(stats_after.total_deallocations, 1);
        assert_eq!(stats_after.allocated_pages, 0);
    }
    
    #[test]
    fn test_multi_page_allocation() {
        let allocator = PageAllocator::new();
        
        let page = allocator.allocate_pages(4).expect("Failed to allocate 4 pages");
        assert_eq!(page.count, 4);
        
        let stats = allocator.get_detailed_stats();
        assert_eq!(stats.allocated_pages, 4);
        assert_eq!(stats.live_bytes, 4 * PAGE_SIZE);
        
        // Clean up
        allocator.deallocate_pages(page);
    }
    
    #[test]
    fn test_page_reuse() {
        let allocator = PageAllocator::new();
        
        // Allocate and deallocate a page
        let page1 = allocator.allocate_pages(1).unwrap();
        allocator.deallocate_pages(page1);
        
        // Check that we have a free page
        assert_eq!(allocator.free_page_count(), 1);
        
        // Allocate again - should reuse the free page
        let _page2 = allocator.allocate_pages(1).unwrap();
        
        let stats = allocator.get_detailed_stats();
        assert_eq!(stats.pages_reused, 1);
        assert_eq!(stats.system_allocations, 1); // Only one system allocation
        
        // Clean up
        allocator.clear_free_pages();
    }
    
    #[test]
    fn test_zero_page_allocation() {
        let allocator = PageAllocator::new();
        
        let result = allocator.allocate_pages(0);
        assert!(result.is_none());
    }
    
    #[test]
    fn test_free_page_management() {
        let allocator = PageAllocator::new();
        
        // Allocate several pages of different sizes
        let page1 = allocator.allocate_pages(1).unwrap();
        let page2 = allocator.allocate_pages(3).unwrap();
        let page3 = allocator.allocate_pages(8).unwrap();
        
        // Deallocate them (should go to free lists)
        allocator.deallocate_pages(page1);
        allocator.deallocate_pages(page2);
        allocator.deallocate_pages(page3);
        
        // Check free page count
        assert_eq!(allocator.free_page_count(), 12); // 1 + 3 + 8
        
        let stats = allocator.get_detailed_stats();
        assert_eq!(stats.pages_freed, 12);
        
        // Clear free pages
        allocator.clear_free_pages();
        assert_eq!(allocator.free_page_count(), 0);
    }
    
    #[test]
    fn test_allocator_interface() {
        let allocator = PageAllocator::new();
        
        // Test allocation through Allocator trait
        let size = PAGE_SIZE * 2;
        let ptr = allocator.allocate(size, PAGE_SIZE).expect("Failed to allocate");
        
        let stats = allocator.stats();
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.live_bytes, size);
        
        // Test deallocation
        allocator.deallocate(ptr, size);
        
        let stats_after = allocator.stats();
        assert_eq!(stats_after.deallocation_count, 1);
    }
    
    #[test]
    fn test_alignment_requirements() {
        let allocator = PageAllocator::new();
        
        // Valid alignment (within page size)
        let ptr1 = allocator.allocate(PAGE_SIZE, PAGE_SIZE / 2);
        assert!(ptr1.is_some());
        
        // Invalid alignment (larger than page size)
        let ptr2 = allocator.allocate(PAGE_SIZE, PAGE_SIZE * 2);
        assert!(ptr2.is_none());
        
        if let Some(ptr) = ptr1 {
            allocator.deallocate(ptr, PAGE_SIZE);
        }
    }
    
    #[test]
    fn test_configuration() {
        let mut config = PageAllocatorConfig::default();
        config.enable_page_reuse = false;
        config.max_free_pages = 100;
        
        let allocator = PageAllocator::with_config(config);
        
        // With reuse disabled, pages should be freed to system immediately
        let page = allocator.allocate_pages(1).unwrap();
        allocator.deallocate_pages(page);
        
        assert_eq!(allocator.free_page_count(), 0);
    }
    
    #[test]
    fn test_statistics_accuracy() {
        let allocator = PageAllocator::new();
        
        // Allocate some pages
        let page1 = allocator.allocate_pages(2).unwrap();
        let page2 = allocator.allocate_pages(3).unwrap();
        
        let stats = allocator.get_detailed_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.allocated_pages, 5);
        assert_eq!(stats.live_bytes, 5 * PAGE_SIZE);
        assert_eq!(stats.peak_pages, 5);
        
        // Deallocate one
        allocator.deallocate_pages(page1);
        
        let stats_after = allocator.get_detailed_stats();
        assert_eq!(stats_after.allocated_pages, 3);
        assert_eq!(stats_after.live_bytes, 3 * PAGE_SIZE);
        assert_eq!(stats_after.peak_pages, 5); // Peak doesn't decrease
        
        // Clean up
        allocator.deallocate_pages(page2);
    }
} 