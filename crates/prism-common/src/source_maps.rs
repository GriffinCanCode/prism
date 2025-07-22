//! Unified Source Map Generation
//!
//! This module provides source map generation functionality that can be shared
//! across different code generation backends, implementing proper VLQ encoding
//! according to the Source Map v3 specification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{PrismError, Result as PrismResult};

/// Source map configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMapConfig {
    /// Enable source map generation
    pub enabled: bool,
    /// Include source content in map
    pub include_sources: bool,
    /// Source map file name
    pub filename: Option<String>,
    /// Source root URL
    pub source_root: Option<String>,
}

impl Default for SourceMapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            include_sources: true,
            filename: None,
            source_root: None,
        }
    }
}

/// Source map mapping entry
#[derive(Debug, Clone)]
pub struct Mapping {
    /// Generated line number (0-based)
    pub generated_line: u32,
    /// Generated column number (0-based)
    pub generated_column: u32,
    /// Source file index (optional)
    pub source_index: Option<usize>,
    /// Original line number (0-based, optional)
    pub original_line: Option<u32>,
    /// Original column number (0-based, optional)
    pub original_column: Option<u32>,
    /// Name index (optional)
    pub name_index: Option<usize>,
}

/// Source map structure (v3 specification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMap {
    /// Source map version (always 3)
    pub version: u8,
    /// Generated file name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Source root URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_root: Option<String>,
    /// Source file names
    pub sources: Vec<String>,
    /// Source file contents
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources_content: Option<Vec<String>>,
    /// Symbol names
    pub names: Vec<String>,
    /// VLQ-encoded mappings
    pub mappings: String,
}

/// Source map generator with proper VLQ encoding
pub struct SourceMapGenerator {
    config: SourceMapConfig,
    mappings: Vec<Mapping>,
    sources: Vec<String>,
    names: Vec<String>,
    source_contents: HashMap<String, String>,
}

impl SourceMapGenerator {
    /// Create a new source map generator
    pub fn new(config: SourceMapConfig) -> Self {
        Self {
            config,
            mappings: Vec::new(),
            sources: Vec::new(),
            names: Vec::new(),
            source_contents: HashMap::new(),
        }
    }

    /// Add a mapping entry
    pub fn add_mapping(&mut self, mapping: Mapping) {
        self.mappings.push(mapping);
    }

    /// Add a source file
    pub fn add_source(&mut self, source: String) -> usize {
        if let Some(index) = self.sources.iter().position(|s| s == &source) {
            index
        } else {
            self.sources.push(source);
            self.sources.len() - 1
        }
    }

    /// Add source content for a source file
    pub fn add_source_content(&mut self, source: String, content: String) {
        self.source_contents.insert(source, content);
    }

    /// Add a name
    pub fn add_name(&mut self, name: String) -> usize {
        if let Some(index) = self.names.iter().position(|n| n == &name) {
            index
        } else {
            self.names.push(name);
            self.names.len() - 1
        }
    }

    /// Generate the source map
    pub fn generate(&self) -> PrismResult<String> {
        if !self.config.enabled {
            return Ok(String::new());
        }

        let sources_content = if self.config.include_sources {
            Some(
                self.sources
                    .iter()
                    .map(|source| {
                        self.source_contents
                            .get(source)
                            .cloned()
                            .unwrap_or_else(|| format!("// Source content for {}", source))
                    })
                    .collect(),
            )
        } else {
            None
        };

        let source_map = SourceMap {
            version: 3,
            file: self.config.filename.clone(),
            source_root: self.config.source_root.clone(),
            sources: self.sources.clone(),
            sources_content,
            names: self.names.clone(),
            mappings: self.encode_mappings()?,
        };

        serde_json::to_string_pretty(&source_map)
            .map_err(|e| PrismError::Generic { message: format!("Failed to serialize source map: {}", e) })
    }

    /// Encode mappings using VLQ encoding
    fn encode_mappings(&self) -> PrismResult<String> {
        let mut result = String::new();
        let mut previous_generated_line = 0;
        let mut previous_generated_column = 0;
        let mut previous_source = 0i32;
        let mut previous_original_line = 0i32;
        let mut previous_original_column = 0i32;
        let mut previous_name = 0i32;

        // Group mappings by generated line
        let mut mappings_by_line: HashMap<u32, Vec<&Mapping>> = HashMap::new();
        for mapping in &self.mappings {
            mappings_by_line
                .entry(mapping.generated_line)
                .or_default()
                .push(mapping);
        }

        // Sort lines
        let mut lines: Vec<_> = mappings_by_line.keys().collect();
        lines.sort();

        for (line_index, &line_number) in lines.iter().enumerate() {
            if line_index > 0 {
                result.push(';');
            }

            // Add empty segments for skipped lines
            while previous_generated_line < *line_number {
                if previous_generated_line > 0 {
                    result.push(';');
                }
                previous_generated_line += 1;
            }

            let mut line_mappings = mappings_by_line[&line_number].clone();
            line_mappings.sort_by_key(|m| m.generated_column);

            for (mapping_index, mapping) in line_mappings.iter().enumerate() {
                if mapping_index > 0 {
                    result.push(',');
                }

                // Generated column (relative to previous)
                let generated_column_delta = mapping.generated_column as i32 - previous_generated_column as i32;
                result.push_str(&encode_vlq(generated_column_delta));
                previous_generated_column = mapping.generated_column;

                if let Some(source_index) = mapping.source_index {
                    // Source index (relative to previous)
                    let source_delta = source_index as i32 - previous_source;
                    result.push_str(&encode_vlq(source_delta));
                    previous_source = source_index as i32;

                    if let (Some(original_line), Some(original_column)) = 
                        (mapping.original_line, mapping.original_column) {
                        // Original line (relative to previous)
                        let original_line_delta = original_line as i32 - previous_original_line;
                        result.push_str(&encode_vlq(original_line_delta));
                        previous_original_line = original_line as i32;

                        // Original column (relative to previous)
                        let original_column_delta = original_column as i32 - previous_original_column;
                        result.push_str(&encode_vlq(original_column_delta));
                        previous_original_column = original_column as i32;

                        if let Some(name_index) = mapping.name_index {
                            // Name index (relative to previous)
                            let name_delta = name_index as i32 - previous_name;
                            result.push_str(&encode_vlq(name_delta));
                            previous_name = name_index as i32;
                        }
                    }
                }
            }
            
            // Update previous line for next iteration
            previous_generated_line = *line_number;
        }

        Ok(result)
    }
}

/// VLQ Base64 alphabet
const VLQ_BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// VLQ continuation bit
const VLQ_CONTINUATION_BIT: u32 = 0x20;

/// VLQ base shift
const VLQ_BASE_SHIFT: u32 = 5;

/// VLQ base mask
const VLQ_BASE_MASK: u32 = 0x1F;

/// Encode a signed integer using Variable Length Quantity (VLQ) encoding
fn encode_vlq(value: i32) -> String {
    let mut vlq = if value < 0 {
        ((-value) << 1) | 1
    } else {
        (value << 1) & !1
    } as u32;

    let mut result = String::new();

    loop {
        let mut digit = vlq & VLQ_BASE_MASK;
        vlq >>= VLQ_BASE_SHIFT;

        if vlq > 0 {
            digit |= VLQ_CONTINUATION_BIT;
        }

        result.push(VLQ_BASE64_CHARS[digit as usize] as char);

        if vlq == 0 {
            break;
        }
    }

    result
}

/// Decode a VLQ-encoded value (for testing/validation)
#[allow(dead_code)]
fn decode_vlq(encoded: &str) -> PrismResult<i32> {
    let mut result = 0u32;
    let mut shift = 0;

    for ch in encoded.chars() {
        let digit = VLQ_BASE64_CHARS
            .iter()
            .position(|&c| c as char == ch)
            .ok_or_else(|| PrismError::Generic { message: format!("Invalid VLQ character: {}", ch) })? as u32;

        let continuation = (digit & VLQ_CONTINUATION_BIT) != 0;
        result |= (digit & VLQ_BASE_MASK) << shift;
        shift += VLQ_BASE_SHIFT;

        if !continuation {
            break;
        }
    }

    // Convert back to signed integer
    let value = if (result & 1) != 0 {
        -((result >> 1) as i32)
    } else {
        (result >> 1) as i32
    };

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlq_encoding() {
        assert_eq!(encode_vlq(0), "A");
        assert_eq!(encode_vlq(1), "C");
        assert_eq!(encode_vlq(-1), "D");
        assert_eq!(encode_vlq(123), "2H");
        assert_eq!(encode_vlq(-123), "3H");
    }

    #[test]
    fn test_vlq_roundtrip() {
        let values = vec![0, 1, -1, 15, -15, 16, -16, 123, -123, 1000, -1000];
        for value in values {
            let encoded = encode_vlq(value);
            let decoded = decode_vlq(&encoded).unwrap();
            assert_eq!(value, decoded, "Failed roundtrip for value {}", value);
        }
    }

    #[test]
    fn test_source_map_generation() {
        let config = SourceMapConfig::default();
        let mut generator = SourceMapGenerator::new(config);

        let source_index = generator.add_source("test.ts".to_string());
        generator.add_source_content("test.ts".to_string(), "const x = 1;".to_string());

        generator.add_mapping(Mapping {
            generated_line: 0,
            generated_column: 0,
            source_index: Some(source_index),
            original_line: Some(0),
            original_column: Some(0),
            name_index: None,
        });

        let source_map = generator.generate().unwrap();
        assert!(!source_map.is_empty());
        assert!(source_map.contains("\"version\":3"));
    }
} 