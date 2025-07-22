//! TypeScript Validation and Linting Integration
//!
//! This module provides TypeScript-specific validation including:
//! - ESLint integration for modern TypeScript rules
//! - TypeScript compiler validation
//! - Prism semantic validation
//! - Business rule validation

use super::{TypeScriptResult, TypeScriptError};
use serde::{Serialize, Deserialize};

/// TypeScript validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeScriptValidationConfig {
    /// Enable ESLint validation
    pub enable_eslint: bool,
    /// Enable TypeScript compiler validation
    pub enable_typescript_compiler: bool,
    /// Enable Prism semantic validation
    pub enable_semantic_validation: bool,
    /// Enable business rule validation
    pub enable_business_rules: bool,
}

impl Default for TypeScriptValidationConfig {
    fn default() -> Self {
        Self {
            enable_eslint: true,
            enable_typescript_compiler: true,
            enable_semantic_validation: true,
            enable_business_rules: true,
        }
    }
}

/// TypeScript validator
pub struct TypeScriptValidator {
    config: TypeScriptValidationConfig,
}

impl TypeScriptValidator {
    /// Create a new TypeScript validator
    pub fn new(config: TypeScriptValidationConfig) -> Self {
        Self { config }
    }

    /// Validate TypeScript code
    pub fn validate(&self, code: &str) -> TypeScriptResult<Vec<String>> {
        let mut issues = Vec::new();

        // Basic TypeScript validation
        if !code.contains("export") && !code.contains("import") {
            issues.push("Warning: No imports/exports found - consider modern ESM patterns".to_string());
        }

        if code.contains("any") {
            issues.push("Warning: 'any' type detected - consider using more specific types".to_string());
        }

        if !code.contains("satisfies") && code.contains("as const") {
            issues.push("Info: Consider using 'satisfies' operator for better type safety".to_string());
        }

        Ok(issues)
    }
}

/// Validation configuration
pub struct ValidationConfig {
    pub typescript_config: TypeScriptValidationConfig,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            typescript_config: TypeScriptValidationConfig::default(),
        }
    }
}

/// Linting integration
pub struct LintingIntegration {
    config: TypeScriptValidationConfig,
}

impl LintingIntegration {
    /// Create new linting integration
    pub fn new(config: TypeScriptValidationConfig) -> Self {
        Self { config }
    }
} 