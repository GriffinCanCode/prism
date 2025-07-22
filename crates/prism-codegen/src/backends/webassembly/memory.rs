//! WebAssembly Memory Layout and Management
//!
//! This module handles WebAssembly memory layout, allocation strategies,
//! and memory-related utilities for semantic metadata preservation.

use super::{WasmResult, WasmError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// WebAssembly memory layout with semantic regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMemoryLayout {
    /// Initial memory size in pages (64KB each)
    pub initial_pages: u32,
    /// Maximum memory size in pages
    pub max_pages: Option<u32>,
    /// Semantic type registry offset in memory
    pub type_registry_offset: u32,
    /// Effect registry offset in memory
    pub effect_registry_offset: u32,
    /// String constants offset in memory
    pub string_constants_offset: u32,
    /// Capability registry offset in memory
    pub capability_registry_offset: u32,
    /// Business rule registry offset
    pub business_rule_registry_offset: u32,
    /// Heap start offset for dynamic allocation
    pub heap_start_offset: u32,
}

impl Default for WasmMemoryLayout {
    fn default() -> Self {
        Self {
            initial_pages: 16, // 1MB initial memory
            max_pages: Some(256), // 16MB maximum
            type_registry_offset: 0x1000,  // 4KB offset
            effect_registry_offset: 0x2000, // 8KB offset
            string_constants_offset: 0x3000, // 12KB offset
            capability_registry_offset: 0x4000, // 16KB offset
            business_rule_registry_offset: 0x5000, // 20KB offset
            heap_start_offset: 0x10000, // 64KB offset
        }
    }
}

/// Memory region descriptor for semantic data
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Region name for debugging
    pub name: String,
    /// Start offset in memory
    pub start_offset: u32,
    /// Size of the region in bytes
    pub size: u32,
    /// Alignment requirements
    pub alignment: u32,
    /// Whether the region is read-only
    pub read_only: bool,
    /// Region purpose/description
    pub description: String,
}

/// WebAssembly memory manager
pub struct WasmMemoryManager {
    /// Memory layout configuration
    layout: WasmMemoryLayout,
    /// Registered memory regions
    regions: HashMap<String, MemoryRegion>,
    /// Next available offset for dynamic allocation
    next_dynamic_offset: u32,
    /// Memory usage statistics
    stats: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory allocated (bytes)
    pub total_allocated: u32,
    /// Memory used by semantic metadata
    pub semantic_metadata_bytes: u32,
    /// Memory used by string constants
    pub string_constants_bytes: u32,
    /// Memory used by business rules
    pub business_rules_bytes: u32,
    /// Memory fragmentation percentage
    pub fragmentation_percent: f64,
}

impl WasmMemoryManager {
    /// Create a new memory manager with layout
    pub fn new(layout: WasmMemoryLayout) -> Self {
        let mut manager = Self {
            next_dynamic_offset: layout.heap_start_offset,
            layout,
            regions: HashMap::new(),
            stats: MemoryStats::default(),
        };

        // Register standard memory regions
        manager.register_standard_regions();
        manager
    }

    /// Register standard memory regions for Prism runtime
    fn register_standard_regions(&mut self) {
        self.register_region(MemoryRegion {
            name: "type_registry".to_string(),
            start_offset: self.layout.type_registry_offset,
            size: 0x1000, // 4KB
            alignment: 8,
            read_only: false,
            description: "Semantic type registry with business rules".to_string(),
        }).expect("Failed to register type registry region");

        self.register_region(MemoryRegion {
            name: "effect_registry".to_string(),
            start_offset: self.layout.effect_registry_offset,
            size: 0x1000, // 4KB
            alignment: 8,
            read_only: false,
            description: "Effect tracking and capability registry".to_string(),
        }).expect("Failed to register effect registry region");

        self.register_region(MemoryRegion {
            name: "string_constants".to_string(),
            start_offset: self.layout.string_constants_offset,
            size: 0x1000, // 4KB (can grow)
            alignment: 4,
            read_only: true,
            description: "String literal constants".to_string(),
        }).expect("Failed to register string constants region");

        self.register_region(MemoryRegion {
            name: "capability_registry".to_string(),
            start_offset: self.layout.capability_registry_offset,
            size: 0x1000, // 4KB
            alignment: 8,
            read_only: false,
            description: "Active capability instances".to_string(),
        }).expect("Failed to register capability registry region");

        self.register_region(MemoryRegion {
            name: "business_rules".to_string(),
            start_offset: self.layout.business_rule_registry_offset,
            size: 0x1000, // 4KB
            alignment: 8,
            read_only: false,
            description: "Business rule validation data".to_string(),
        }).expect("Failed to register business rules region");
    }

    /// Register a memory region
    pub fn register_region(&mut self, region: MemoryRegion) -> WasmResult<()> {
        // Check for overlaps with existing regions
        for (_, existing_region) in &self.regions {
            if self.regions_overlap(&region, existing_region) {
                return Err(WasmError::MemoryLayout {
                    message: format!(
                        "Region '{}' overlaps with existing region '{}'",
                        region.name, existing_region.name
                    ),
                });
            }
        }

        // Validate alignment
        if region.start_offset % region.alignment != 0 {
            return Err(WasmError::MemoryLayout {
                message: format!(
                    "Region '{}' start offset 0x{:X} is not aligned to {} bytes",
                    region.name, region.start_offset, region.alignment
                ),
            });
        }

        self.regions.insert(region.name.clone(), region);
        self.update_stats();
        Ok(())
    }

    /// Allocate memory in the dynamic heap region
    pub fn allocate_dynamic(&mut self, size: u32, alignment: u32) -> WasmResult<u32> {
        // Align the current offset
        let aligned_offset = self.align_offset(self.next_dynamic_offset, alignment);
        
        // Check if allocation fits in memory
        let end_offset = aligned_offset + size;
        let max_memory = self.layout.max_pages.unwrap_or(u32::MAX) * 65536; // 64KB pages
        
        if end_offset > max_memory {
            return Err(WasmError::MemoryLayout {
                message: format!(
                    "Dynamic allocation of {} bytes would exceed memory limit",
                    size
                ),
            });
        }

        // Update next offset
        self.next_dynamic_offset = end_offset;
        self.stats.total_allocated += size;
        
        Ok(aligned_offset)
    }

    /// Get memory region by name
    pub fn get_region(&self, name: &str) -> Option<&MemoryRegion> {
        self.regions.get(name)
    }

    /// Generate WebAssembly memory declaration
    pub fn generate_memory_declaration(&self) -> String {
        let max_clause = if let Some(max_pages) = self.layout.max_pages {
            format!(" {}", max_pages)
        } else {
            String::new()
        };

        format!(
            "  ;; Memory configuration for semantic preservation\n  (memory (export \"memory\") {}{})\n",
            self.layout.initial_pages,
            max_clause
        )
    }

    /// Generate memory layout documentation
    pub fn generate_layout_documentation(&self) -> String {
        let mut output = String::new();
        
        output.push_str(";; === MEMORY LAYOUT DOCUMENTATION ===\n");
        output.push_str(&format!(
            ";; Total memory: {} pages ({} KB), Max: {} pages ({} KB)\n",
            self.layout.initial_pages,
            self.layout.initial_pages * 64,
            self.layout.max_pages.unwrap_or(u32::MAX),
            self.layout.max_pages.unwrap_or(u32::MAX) * 64
        ));
        output.push_str(";;\n");

        // Sort regions by start offset
        let mut sorted_regions: Vec<_> = self.regions.values().collect();
        sorted_regions.sort_by_key(|r| r.start_offset);

        for region in sorted_regions {
            output.push_str(&format!(
                ";; 0x{:08X} - 0x{:08X}: {} ({} bytes, align {})\n",
                region.start_offset,
                region.start_offset + region.size,
                region.name,
                region.size,
                region.alignment
            ));
            output.push_str(&format!(";; Description: {}\n", region.description));
            output.push_str(&format!(";; Access: {}\n", if region.read_only { "Read-only" } else { "Read-write" }));
            output.push_str(";;\n");
        }

        output.push_str(&format!(
            ";; Dynamic heap starts at: 0x{:08X}\n",
            self.layout.heap_start_offset
        ));
        output.push_str(&format!(
            ";; Next allocation offset: 0x{:08X}\n",
            self.next_dynamic_offset
        ));
        
        output.push('\n');
        output
    }

    /// Generate data initialization for memory regions
    pub fn generate_data_initialization(&self) -> String {
        let mut output = String::new();
        
        output.push_str("  ;; === MEMORY REGION INITIALIZATION ===\n");
        
        // Initialize type registry header
        output.push_str(&format!(
            "  (data (i32.const {}) \"\\00\\00\\00\\00\") ;; Type registry header\n",
            self.layout.type_registry_offset
        ));
        
        // Initialize effect registry header
        output.push_str(&format!(
            "  (data (i32.const {}) \"\\00\\00\\00\\00\") ;; Effect registry header\n",
            self.layout.effect_registry_offset
        ));
        
        // Initialize capability registry header
        output.push_str(&format!(
            "  (data (i32.const {}) \"\\00\\00\\00\\00\") ;; Capability registry header\n",
            self.layout.capability_registry_offset
        ));
        
        // Initialize business rules header
        output.push_str(&format!(
            "  (data (i32.const {}) \"\\00\\00\\00\\00\") ;; Business rules header\n",
            self.layout.business_rule_registry_offset
        ));
        
        output.push('\n');
        output
    }

    /// Get memory usage statistics
    pub fn get_statistics(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get memory layout
    pub fn get_layout(&self) -> &WasmMemoryLayout {
        &self.layout
    }

    /// Check if two regions overlap
    fn regions_overlap(&self, region1: &MemoryRegion, region2: &MemoryRegion) -> bool {
        let r1_end = region1.start_offset + region1.size;
        let r2_end = region2.start_offset + region2.size;
        
        !(r1_end <= region2.start_offset || r2_end <= region1.start_offset)
    }

    /// Align an offset to the specified alignment
    fn align_offset(&self, offset: u32, alignment: u32) -> u32 {
        ((offset + alignment - 1) / alignment) * alignment
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        self.stats.total_allocated = self.regions.values()
            .map(|r| r.size)
            .sum::<u32>() + (self.next_dynamic_offset - self.layout.heap_start_offset);

        // Calculate semantic metadata usage
        self.stats.semantic_metadata_bytes = self.regions.get("type_registry")
            .map(|r| r.size)
            .unwrap_or(0);

        self.stats.string_constants_bytes = self.regions.get("string_constants")
            .map(|r| r.size)
            .unwrap_or(0);

        self.stats.business_rules_bytes = self.regions.get("business_rules")
            .map(|r| r.size)
            .unwrap_or(0);

        // Calculate fragmentation (simplified)
        let used_space = self.stats.total_allocated;
        let total_space = self.next_dynamic_offset;
        if total_space > 0 {
            self.stats.fragmentation_percent = 
                (1.0 - (used_space as f64 / total_space as f64)) * 100.0;
        }
    }
}

impl Default for WasmMemoryManager {
    fn default() -> Self {
        Self::new(WasmMemoryLayout::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_layout_default() {
        let layout = WasmMemoryLayout::default();
        
        assert_eq!(layout.initial_pages, 16);
        assert_eq!(layout.max_pages, Some(256));
        assert_eq!(layout.type_registry_offset, 0x1000);
        assert_eq!(layout.heap_start_offset, 0x10000);
    }

    #[test]
    fn test_memory_manager_creation() {
        let manager = WasmMemoryManager::default();
        
        // Should have standard regions registered
        assert!(manager.get_region("type_registry").is_some());
        assert!(manager.get_region("effect_registry").is_some());
        assert!(manager.get_region("string_constants").is_some());
        assert!(manager.get_region("capability_registry").is_some());
        assert!(manager.get_region("business_rules").is_some());
    }

    #[test]
    fn test_dynamic_allocation() {
        let mut manager = WasmMemoryManager::default();
        
        // Allocate 100 bytes with 8-byte alignment
        let offset1 = manager.allocate_dynamic(100, 8).unwrap();
        assert_eq!(offset1, 0x10000); // Should be at heap start
        
        // Allocate another 50 bytes
        let offset2 = manager.allocate_dynamic(50, 4).unwrap();
        assert_eq!(offset2, 0x10000 + 100); // Should follow first allocation
    }

    #[test]
    fn test_region_overlap_detection() {
        let mut manager = WasmMemoryManager::new(WasmMemoryLayout::default());
        
        // Try to register an overlapping region
        let overlapping_region = MemoryRegion {
            name: "overlap_test".to_string(),
            start_offset: 0x1500, // Overlaps with type_registry at 0x1000
            size: 0x1000,
            alignment: 4,
            read_only: false,
            description: "Test overlap".to_string(),
        };
        
        assert!(manager.register_region(overlapping_region).is_err());
    }

    #[test]
    fn test_memory_declaration_generation() {
        let manager = WasmMemoryManager::default();
        let declaration = manager.generate_memory_declaration();
        
        assert!(declaration.contains("(memory"));
        assert!(declaration.contains("16")); // Initial pages
        assert!(declaration.contains("256")); // Max pages
    }
} 