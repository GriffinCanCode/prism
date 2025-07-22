//! Source Map Generation for JavaScript Backend
//!
//! This module handles source map generation for debugging support,
//! following the Source Map v3 specification. Uses the unified source map
//! implementation from prism-common.

use super::{JavaScriptResult, JavaScriptError};
use prism_common::{SourceMapGenerator, SourceMapConfig, SourceMapMapping};
use serde::{Deserialize, Serialize};

/// JavaScript-specific source map generator that wraps the unified implementation
pub struct JavaScriptSourceMapGenerator {
    generator: SourceMapGenerator,
}

impl JavaScriptSourceMapGenerator {
    pub fn new(config: SourceMapConfig) -> Self {
        Self {
            generator: SourceMapGenerator::new(config),
        }
    }

    pub fn add_mapping(&mut self, generated_line: u32, generated_column: u32, source: Option<String>, original_line: Option<u32>, original_column: Option<u32>, name: Option<String>) {
        let source_index = source.map(|s| self.generator.add_source(s));
        let name_index = name.map(|n| self.generator.add_name(n));
        
        self.generator.add_mapping(SourceMapMapping {
            generated_line,
            generated_column,
            source_index,
            original_line,
            original_column,
            name_index,
        });
    }

    pub fn add_source_content(&mut self, source: String, content: String) {
        self.generator.add_source_content(source, content);
    }

    pub fn generate(&self, _code: &str, filename: &str) -> JavaScriptResult<String> {
        self.generator.generate()
            .map_err(|e| JavaScriptError::SourceMapError(format!("Source map generation failed: {}", e)))
    }
}

// Re-export unified types for backward compatibility
pub use prism_common::{SourceMapConfig, SourceMapMapping as Mapping, SourceMap}; 