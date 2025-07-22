//! Template Literal Types for TypeScript
//!
//! This module handles template literal type generation

use super::{TypeScriptResult, TypeScriptError};
use serde::{Serialize, Deserialize};

/// Template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// Enable template literal types
    pub enabled: bool,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Template literal generator
pub struct TemplateLiteralGenerator {
    config: TemplateConfig,
}

impl TemplateLiteralGenerator {
    /// Create new template literal generator
    pub fn new(config: TemplateConfig) -> Self {
        Self { config }
    }
} 