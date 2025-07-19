//! Bridge to prism-ast for AST conversion.

use crate::normalization::CanonicalForm;
use thiserror::Error;

/// Bridge for AST conversion
#[derive(Debug)]
pub struct AstBridge {
    /// Configuration
    config: String, // Placeholder
}

/// Result of AST conversion
#[derive(Debug)]
pub struct AstConversionResult {
    /// Success status
    pub success: bool,
}

/// AST integration errors
#[derive(Debug, Error)]
pub enum AstIntegrationError {
    /// Conversion failed
    #[error("AST conversion failed: {reason}")]
    ConversionFailed { reason: String },
}

impl AstBridge {
    /// Create new AST bridge
    pub fn new() -> Self {
        Self { config: String::new() }
    }
    
    /// Convert canonical form to AST
    pub fn to_ast(&self, _canonical: &CanonicalForm) -> Result<AstConversionResult, AstIntegrationError> {
        // TODO: Implement AST conversion
        Ok(AstConversionResult { success: true })
    }
}

impl Default for AstBridge {
    fn default() -> Self {
        Self::new()
    }
} 