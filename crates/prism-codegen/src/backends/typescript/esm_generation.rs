//! ESM Generation for Modern TypeScript
//!
//! This module handles modern ESM import/export generation

use super::{TypeScriptResult, TypeScriptError};
use serde::{Serialize, Deserialize};

/// ESM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ESMConfig {
    /// Use ESM imports
    pub use_esm: bool,
}

impl Default for ESMConfig {
    fn default() -> Self {
        Self { use_esm: true }
    }
}

/// ESM generator
pub struct ESMGenerator {
    config: ESMConfig,
}

impl ESMGenerator {
    /// Create new ESM generator
    pub fn new(config: ESMConfig) -> Self {
        Self { config }
    }
}

/// Module system
pub struct ModuleSystem {
    config: ESMConfig,
}

impl ModuleSystem {
    /// Create new module system
    pub fn new(config: ESMConfig) -> Self {
        Self { config }
    }
} 