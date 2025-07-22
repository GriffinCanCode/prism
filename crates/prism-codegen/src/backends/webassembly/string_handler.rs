//! WebAssembly String Constant Management
//!
//! This module provides efficient string constant management for WebAssembly code generation,
//! including deduplication, memory layout optimization, and data section generation.
//!
//! ## Features
//!
//! - **Deduplication**: Automatic deduplication of identical string constants
//! - **Memory Layout**: Optimized memory layout with alignment considerations
//! - **Encoding Support**: UTF-8 encoding with proper null termination
//! - **Debug Information**: Optional debug metadata for string constants
//! - **Size Optimization**: String interning and compression for large programs

use super::{WasmResult, WasmError};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Configuration for string constant management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringManagerConfig {
    /// Base offset for string constants in memory
    pub base_offset: u32,
    /// Alignment for string constants (must be power of 2)
    pub alignment: u32,
    /// Enable string deduplication
    pub enable_deduplication: bool,
    /// Enable debug metadata for strings
    pub enable_debug_metadata: bool,
    /// Maximum string length before warning
    pub max_string_length: usize,
    /// Enable string compression for large strings
    pub enable_compression: bool,
}

impl Default for StringManagerConfig {
    fn default() -> Self {
        Self {
            base_offset: 0x3000, // 12KB offset by default
            alignment: 4,        // 4-byte alignment
            enable_deduplication: true,
            enable_debug_metadata: true,
            max_string_length: 1024,
            enable_compression: false, // Disabled by default for simplicity
        }
    }
}

/// String constant entry with metadata
#[derive(Debug, Clone)]
pub struct StringConstantEntry {
    /// The string content
    pub content: String,
    /// Memory offset where this string is stored
    pub offset: u32,
    /// Aligned size in memory (including null terminator and padding)
    pub aligned_size: u32,
    /// Usage count (for optimization decisions)
    pub usage_count: u32,
    /// Source location information (for debugging)
    pub source_info: Option<SourceInfo>,
}

/// Source information for debugging
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// File where the string was defined
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u32,
}

/// String constant manager for WebAssembly backend
pub struct StringConstantManager {
    /// Configuration
    config: StringManagerConfig,
    /// String constants table
    constants: HashMap<String, StringConstantEntry>,
    /// Next available offset
    next_offset: u32,
    /// Total memory used by strings
    total_memory_used: u32,
    /// String usage statistics
    stats: StringManagerStats,
}

/// Statistics for string constant management
#[derive(Debug, Clone, Default)]
pub struct StringManagerStats {
    /// Total number of strings
    pub total_strings: usize,
    /// Number of unique strings (after deduplication)
    pub unique_strings: usize,
    /// Total memory saved by deduplication
    pub memory_saved_bytes: u32,
    /// Average string length
    pub average_length: f64,
    /// Largest string length
    pub max_length: usize,
}

impl StringConstantManager {
    /// Create a new string constant manager
    pub fn new(config: StringManagerConfig) -> Self {
        Self {
            next_offset: config.base_offset,
            config,
            constants: HashMap::new(),
            total_memory_used: 0,
            stats: StringManagerStats::default(),
        }
    }

    /// Add a string constant and return its memory offset
    pub fn add_string(&mut self, content: &str, source_info: Option<SourceInfo>) -> WasmResult<u32> {
        // Check string length
        if content.len() > self.config.max_string_length {
            return Err(WasmError::StringManagement {
                message: format!(
                    "String length {} exceeds maximum {}", 
                    content.len(), 
                    self.config.max_string_length
                ),
            });
        }

        // Check for existing string (deduplication)
        if self.config.enable_deduplication {
            if let Some(entry) = self.constants.get_mut(content) {
                entry.usage_count += 1;
                self.stats.memory_saved_bytes += self.calculate_aligned_size(content.len()) as u32;
                return Ok(entry.offset);
            }
        }

        // Calculate aligned size (content + null terminator + padding)
        let aligned_size = self.calculate_aligned_size(content.len());
        
        // Create new entry
        let entry = StringConstantEntry {
            content: content.to_string(),
            offset: self.next_offset,
            aligned_size,
            usage_count: 1,
            source_info,
        };

        // Update offsets and statistics
        self.constants.insert(content.to_string(), entry);
        self.next_offset += aligned_size;
        self.total_memory_used += aligned_size;
        self.update_stats();

        Ok(self.next_offset - aligned_size)
    }

    /// Get offset for an existing string constant
    pub fn get_string_offset(&self, content: &str) -> Option<u32> {
        self.constants.get(content).map(|entry| entry.offset)
    }

    /// Generate WebAssembly data section for all string constants
    pub fn generate_data_section(&self) -> String {
        let mut output = String::new();
        
        output.push_str("  ;; === STRING CONSTANTS ===\n");
        output.push_str(&format!(
            "  ;; {} unique strings, {} total memory, {} saved by deduplication\n",
            self.stats.unique_strings,
            self.total_memory_used,
            self.stats.memory_saved_bytes
        ));
        output.push('\n');

        // Sort strings by offset for consistent output
        let mut sorted_constants: Vec<_> = self.constants.values().collect();
        sorted_constants.sort_by_key(|entry| entry.offset);

        for entry in sorted_constants {
            // Generate data directive
            output.push_str(&format!(
                "  ;; String: \"{}\" (used {} times)\n",
                self.escape_string_for_comment(&entry.content),
                entry.usage_count
            ));
            
            if self.config.enable_debug_metadata {
                if let Some(ref source) = entry.source_info {
                    output.push_str(&format!(
                        "  ;; Source: {}:{}:{}\n",
                        source.file, source.line, source.column
                    ));
                }
            }
            
            output.push_str(&format!(
                "  (data (i32.const {}) \"{}\\00\")\n",
                entry.offset,
                self.escape_string_for_wasm(&entry.content)
            ));
            output.push('\n');
        }

        output
    }

    /// Generate memory layout information
    pub fn generate_memory_layout_info(&self) -> String {
        format!(
            r#"  ;; === STRING MEMORY LAYOUT ===
  ;; Base offset: 0x{:X}
  ;; Next offset: 0x{:X}  
  ;; Total memory: {} bytes
  ;; Alignment: {} bytes
  ;; Unique strings: {}
  ;; Total references: {}
  ;; Memory efficiency: {:.1}%

"#,
            self.config.base_offset,
            self.next_offset,
            self.total_memory_used,
            self.config.alignment,
            self.stats.unique_strings,
            self.stats.total_strings,
            if self.stats.total_strings > 0 {
                (self.stats.unique_strings as f64 / self.stats.total_strings as f64) * 100.0
            } else {
                100.0
            }
        )
    }

    /// Get statistics about string usage
    pub fn get_statistics(&self) -> &StringManagerStats {
        &self.stats
    }

    /// Get total memory used by string constants
    pub fn get_total_memory_used(&self) -> u32 {
        self.total_memory_used
    }

    /// Get next available offset (useful for memory layout planning)
    pub fn get_next_offset(&self) -> u32 {
        self.next_offset
    }

    /// Calculate aligned size for a string of given length
    fn calculate_aligned_size(&self, content_length: usize) -> u32 {
        // Content + null terminator
        let size_with_null = content_length + 1;
        
        // Align to configured boundary
        let aligned_size = ((size_with_null + self.config.alignment as usize - 1) 
                           / self.config.alignment as usize) 
                           * self.config.alignment as usize;
        
        aligned_size as u32
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        self.stats.unique_strings = self.constants.len();
        self.stats.total_strings = self.constants.values().map(|e| e.usage_count as usize).sum();
        
        if !self.constants.is_empty() {
            let total_length: usize = self.constants.values().map(|e| e.content.len()).sum();
            self.stats.average_length = total_length as f64 / self.constants.len() as f64;
            self.stats.max_length = self.constants.values().map(|e| e.content.len()).max().unwrap_or(0);
        }
    }

    /// Escape string for WebAssembly data section
    fn escape_string_for_wasm(&self, s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if c.is_control() => format!("\\{:02x}", c as u8),
                c => c.to_string(),
            })
            .collect()
    }

    /// Escape string for comments (more readable)
    fn escape_string_for_comment(&self, s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if c.is_control() => format!("\\x{:02x}", c as u8),
                c => c.to_string(),
            })
            .collect()
    }
}

impl Default for StringConstantManager {
    fn default() -> Self {
        Self::new(StringManagerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_deduplication() {
        let mut manager = StringConstantManager::new(StringManagerConfig::default());
        
        // Add the same string twice
        let offset1 = manager.add_string("hello", None).unwrap();
        let offset2 = manager.add_string("hello", None).unwrap();
        
        // Should return the same offset
        assert_eq!(offset1, offset2);
        assert_eq!(manager.get_statistics().unique_strings, 1);
        assert_eq!(manager.get_statistics().total_strings, 2);
    }

    #[test]
    fn test_string_alignment() {
        let config = StringManagerConfig {
            alignment: 8,
            ..StringManagerConfig::default()
        };
        let mut manager = StringConstantManager::new(config);
        
        // Add a 3-character string (+ null = 4 bytes)
        // Should be aligned to 8 bytes
        manager.add_string("abc", None).unwrap();
        
        let stats = manager.get_statistics();
        assert_eq!(manager.get_total_memory_used(), 8); // Aligned to 8 bytes
    }

    #[test]
    fn test_string_length_limit() {
        let config = StringManagerConfig {
            max_string_length: 5,
            ..StringManagerConfig::default()
        };
        let mut manager = StringConstantManager::new(config);
        
        // This should succeed
        assert!(manager.add_string("hello", None).is_ok());
        
        // This should fail (6 characters > 5 limit)
        assert!(manager.add_string("hello!", None).is_err());
    }

    #[test]
    fn test_data_section_generation() {
        let mut manager = StringConstantManager::new(StringManagerConfig::default());
        
        manager.add_string("hello", None).unwrap();
        manager.add_string("world", None).unwrap();
        
        let data_section = manager.generate_data_section();
        
        // Should contain both strings
        assert!(data_section.contains("hello"));
        assert!(data_section.contains("world"));
        assert!(data_section.contains("(data"));
    }
} 