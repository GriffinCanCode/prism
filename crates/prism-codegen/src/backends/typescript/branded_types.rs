//! Branded Types for TypeScript
//!
//! This module handles branded type generation for semantic safety

use super::{TypeScriptResult, TypeScriptError};
use serde::{Serialize, Deserialize};

/// Branding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandingConfig {
    /// Enable branded types
    pub enabled: bool,
}

impl Default for BrandingConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Branded type generator
pub struct BrandedTypeGenerator {
    config: BrandingConfig,
}

impl BrandedTypeGenerator {
    /// Create new branded type generator
    pub fn new(config: BrandingConfig) -> Self {
        Self { config }
    }
} 