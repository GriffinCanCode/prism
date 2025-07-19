//! Validation diagnostics and error reporting.

/// Validation diagnostic message
#[derive(Debug, Clone)]
pub struct ValidationDiagnostic {
    /// Diagnostic message
    pub message: String,
    
    /// Severity level
    pub severity: DiagnosticSeverity,
    
    /// Diagnostic code
    pub code: DiagnosticCode,
}

/// Severity levels for diagnostics
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticSeverity {
    /// Error level
    Error,
    /// Warning level
    Warning,
    /// Information level
    Info,
    /// Hint level
    Hint,
}

/// Diagnostic codes for categorization
#[derive(Debug, Clone)]
pub enum DiagnosticCode {
    /// Structure-related diagnostic
    Structure(String),
    /// Documentation-related diagnostic
    Documentation(String),
    /// Semantic-related diagnostic
    Semantic(String),
}

impl ValidationDiagnostic {
    /// Create a new diagnostic
    pub fn new(message: String, severity: DiagnosticSeverity, code: DiagnosticCode) -> Self {
        Self { message, severity, code }
    }
} 