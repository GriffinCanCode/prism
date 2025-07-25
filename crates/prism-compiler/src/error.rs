//! Comprehensive error handling for the Prism compiler
//!
//! This module provides detailed error types with AI-friendly messages and
//! comprehensive context information for debugging and development.

use prism_common::{span::Span, NodeId};
use std::path::PathBuf;
use thiserror::Error;
/// Result type for compiler operations
pub type CompilerResult<T> = Result<T, CompilerError>;

/// Comprehensive compiler error types
#[derive(Debug, Error)]
pub enum CompilerError {
    // I/O and file system errors
    #[error("Failed to read file '{path}': {source}")]
    FileReadError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to write file '{path}': {source}")]
    FileWriteError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Project file not found: {path}")]
    ProjectFileNotFound { path: PathBuf },

    // Documentation errors
    #[error("Documentation error: {0}")]
    DocumentationError(String),

    // Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),

    // I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    // Lexing errors
    #[error("Lexical error at {location}: {message}")]
    LexError { message: String, location: Span },

    // Parsing and lexing errors
    #[error("Syntax error at {location}: {message}")]
    SyntaxError { location: Span, message: String },

    #[error("Parse error at {location}: {message}")]
    ParseError {
        message: String,
        location: Span,
        suggestions: Vec<String>,
    },

    #[error("Unexpected token '{token}' at {location}, expected {expected}")]
    UnexpectedToken {
        token: String,
        location: Span,
        expected: Vec<String>,
    },

    #[error("Unterminated string literal at {location}")]
    UnterminatedString { location: Span },

    #[error("Invalid character '{character}' at {location}")]
    InvalidCharacter { character: char, location: Span },

    // Semantic analysis errors
    #[error("Undefined symbol '{symbol}' at {location}")]
    UndefinedSymbol { symbol: String, location: Span },

    #[error("Symbol '{symbol}' is already defined at {original_location}, redefined at {redefinition_location}")]
    SymbolRedefinition {
        symbol: String,
        original_location: Span,
        redefinition_location: Span,
    },

    #[error("Circular dependency detected: {cycle}")]
    CircularDependency { cycle: Vec<String> },

    // Type system errors
    #[error("Type mismatch at {location}: expected {expected}, found {found}")]
    TypeMismatch {
        location: Span,
        expected: String,
        found: String,
    },

    #[error("Cannot infer type for '{symbol}' at {location}")]
    TypeInferenceError { symbol: String, location: Span },

    #[error("Invalid type constraint at {location}: {constraint}")]
    InvalidTypeConstraint { location: Span, constraint: String },

    #[error("Semantic type violation at {location}: {rule}")]
    SemanticTypeViolation { location: Span, rule: String },

    #[error("Semantic analysis error: {message}")]
    SemanticAnalysisError { message: String },

    #[error("Missing dependency '{dependency}' in context '{context}'")]
    MissingDependency { dependency: String, context: String },

    #[error("Business rule violation at {location}: {rule} - {explanation}")]
    BusinessRuleViolation {
        location: Span,
        rule: String,
        explanation: String,
    },

    // Effect system errors
    #[error("Effect error at {location}: {message}")]
    EffectError { location: Span, message: String },

    #[error("Missing capability '{capability}' for effect '{effect}' at {location}")]
    MissingCapability {
        capability: String,
        effect: String,
        location: Span,
    },

    #[error("Effect conflict at {location}: {conflicting_effects}")]
    EffectConflict {
        location: Span,
        conflicting_effects: Vec<String>,
    },

    // Module system errors
    #[error("Module '{module}' not found")]
    ModuleNotFound { module: String },

    #[error("Import error at {location}: {message}")]
    ImportError { location: Span, message: String },

    #[error("Export error at {location}: {message}")]
    ExportError { location: Span, message: String },

    #[error("Module cohesion violation in '{module}': {issue}")]
    ModuleCohesionViolation { module: String, issue: String },

    // Code generation errors
    #[error("Code generation failed for target {target}: {message}")]
    CodeGenError { target: String, message: String },

    #[error("Code generation coordination failed: {message}")]
    CodeGenerationFailed { target: String, message: String },

    #[error("Cross-target validation failed: {message}")]
    CrossTargetValidationFailed { message: String },

    #[error("Unsupported feature '{feature}' for target {target}")]
    UnsupportedFeature { feature: String, target: String },

    #[error("Optimization error: {message}")]
    OptimizationError { message: String },

    // Query system errors
    #[error("Query execution failed: {message}")]
    QueryExecutionError { message: String },

    #[error("Cache serialization error: {message}")]
    CacheSerializationError { message: String },

    #[error("Cache deserialization error: {message}")]
    CacheDeserializationError { message: String },

    #[error("Dependency resolution error: {message}")]
    DependencyResolutionError { message: String },

    // AI integration errors
    #[error("AI context generation failed: {message}")]
    AIContextError { message: String },

    #[error("AI metadata export failed: {message}")]
    AIMetadataExportError { message: String },

    #[error("AI suggestion generation failed: {message}")]
    AISuggestionError { message: String },

    // Language server errors
    #[error("Language server error: {message}")]
    LanguageServerError { message: String },

    #[error("Protocol error: {message}")]
    ProtocolError { message: String },

    // Configuration errors
    #[error("Invalid configuration: {message}")]
    ConfigurationError { message: String },

    #[error("Missing required configuration: {key}")]
    MissingConfiguration { key: String },

    // Runtime errors
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    #[error("Memory allocation error: {message}")]
    MemoryError { message: String },

    #[error("Thread synchronization error: {message}")]
    SynchronizationError { message: String },

    // Generic errors
    #[error("Internal compiler error: {message}")]
    InternalError { message: String },

    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },
}

/// Error context for enhanced debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Source file path
    pub file_path: Option<PathBuf>,
    /// Function or method name
    pub function_name: Option<String>,
    /// Module name
    pub module_name: Option<String>,
    /// Line number
    pub line_number: Option<u32>,
    /// Column number
    pub column_number: Option<u32>,
    /// Additional context information
    pub context_info: Vec<String>,
    /// Related spans for multi-location errors
    pub related_spans: Vec<Span>,
    /// AI-generated help text
    pub ai_help: Option<String>,
}

/// Enhanced error with additional context
#[derive(Debug)]
pub struct EnhancedError {
    /// Base error
    pub error: CompilerError,
    /// Additional context
    pub context: ErrorContext,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error category for filtering
    pub category: ErrorCategory,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Fatal error (compilation cannot continue)
    Fatal,
    /// Error (compilation fails but can continue for more errors)
    Error,
    /// Warning (compilation succeeds but issue noted)
    Warning,
    /// Info (informational message)
    Info,
}

/// Error categories for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Syntax and parsing errors
    Syntax,
    /// Type system errors
    Type,
    /// Semantic analysis errors
    Semantic,
    /// Effect system errors
    Effect,
    /// Module system errors
    Module,
    /// Code generation errors
    CodeGen,
    /// AI integration errors
    AI,
    /// Performance and optimization
    Performance,
    /// Security issues
    Security,
    /// Configuration issues
    Configuration,
    /// Internal compiler issues
    Internal,
}

/// Error builder for constructing detailed errors
pub struct ErrorBuilder {
    error: CompilerError,
    context: ErrorContext,
    severity: ErrorSeverity,
    category: ErrorCategory,
}

impl CompilerError {
    /// Create an enhanced error with context
    pub fn with_context(self, context: ErrorContext) -> EnhancedError {
        let severity = self.default_severity();
        let category = self.default_category();
        
        EnhancedError {
            error: self,
            context,
            severity,
            category,
        }
    }

    /// Create an error builder
    pub fn builder(self) -> ErrorBuilder {
        ErrorBuilder {
            error: self.clone(),
            context: ErrorContext::default(),
            severity: self.default_severity(),
            category: self.default_category(),
        }
    }

    /// Get default severity for this error type
    pub fn default_severity(&self) -> ErrorSeverity {
        match self {
            // Fatal errors
            CompilerError::InternalError { .. } => ErrorSeverity::Fatal,
            CompilerError::MemoryError { .. } => ErrorSeverity::Fatal,
            
            // Regular errors
            CompilerError::SyntaxError { .. } |
            CompilerError::TypeMismatch { .. } |
            CompilerError::UndefinedSymbol { .. } |
            CompilerError::BusinessRuleViolation { .. } => ErrorSeverity::Error,
            
            // Warnings
            CompilerError::ModuleCohesionViolation { .. } => ErrorSeverity::Warning,
            
            // Info
            _ => ErrorSeverity::Info,
        }
    }

    /// Get default category for this error type
    pub fn default_category(&self) -> ErrorCategory {
        match self {
            CompilerError::SyntaxError { .. } |
            CompilerError::UnexpectedToken { .. } |
            CompilerError::UnterminatedString { .. } |
            CompilerError::InvalidCharacter { .. } => ErrorCategory::Syntax,

            CompilerError::TypeMismatch { .. } |
            CompilerError::TypeInferenceError { .. } |
            CompilerError::InvalidTypeConstraint { .. } |
            CompilerError::SemanticTypeViolation { .. } => ErrorCategory::Type,

            CompilerError::UndefinedSymbol { .. } |
            CompilerError::SymbolRedefinition { .. } |
            CompilerError::BusinessRuleViolation { .. } => ErrorCategory::Semantic,

            CompilerError::EffectError { .. } |
            CompilerError::MissingCapability { .. } |
            CompilerError::EffectConflict { .. } => ErrorCategory::Effect,

            CompilerError::ModuleNotFound { .. } |
            CompilerError::ImportError { .. } |
            CompilerError::ExportError { .. } |
            CompilerError::ModuleCohesionViolation { .. } => ErrorCategory::Module,

            CompilerError::CodeGenError { .. } |
            CompilerError::UnsupportedFeature { .. } |
            CompilerError::OptimizationError { .. } => ErrorCategory::CodeGen,

            CompilerError::AIContextError { .. } |
            CompilerError::AIMetadataExportError { .. } |
            CompilerError::AISuggestionError { .. } => ErrorCategory::AI,

            CompilerError::ConfigurationError { .. } |
            CompilerError::MissingConfiguration { .. } => ErrorCategory::Configuration,

            CompilerError::InternalError { .. } |
            CompilerError::NotImplemented { .. } => ErrorCategory::Internal,

            _ => ErrorCategory::Internal,
        }
    }

    /// Check if this error should halt compilation
    pub fn is_fatal(&self) -> bool {
        matches!(self.default_severity(), ErrorSeverity::Fatal)
    }

    /// Get AI-friendly error explanation
    pub fn ai_explanation(&self) -> String {
        match self {
            CompilerError::TypeMismatch { expected, found, .. } => {
                format!("The code expects a value of type '{}' but received a value of type '{}'. This typically happens when you pass the wrong type of argument to a function or assign a value to a variable of a different type.", expected, found)
            }
            
            CompilerError::UndefinedSymbol { symbol, .. } => {
                format!("The symbol '{}' is used but not defined. Make sure you've declared the variable, function, or type before using it, or check for typos in the name.", symbol)
            }
            
            CompilerError::BusinessRuleViolation { rule, explanation, .. } => {
                format!("A business rule was violated: {}. {}", rule, explanation)
            }
            
            CompilerError::MissingCapability { capability, effect, .. } => {
                format!("The effect '{}' requires the capability '{}' but it's not available in the current context. You may need to add the capability to your function signature or module.", effect, capability)
            }
            
            _ => "An error occurred during compilation. Check the error message for specific details.".to_string(),
        }
    }

    /// Get suggested fixes
    pub fn suggested_fixes(&self) -> Vec<String> {
        match self {
            CompilerError::UndefinedSymbol { symbol, .. } => {
                vec![
                    format!("Define the symbol '{}' before using it", symbol),
                    "Check for typos in the symbol name".to_string(),
                    "Make sure the symbol is imported if it's from another module".to_string(),
                ]
            }
            
            CompilerError::TypeMismatch { expected, .. } => {
                vec![
                    format!("Convert the value to type '{}'", expected),
                    "Check the function signature or variable declaration".to_string(),
                    "Use explicit type casting if appropriate".to_string(),
                ]
            }
            
            CompilerError::MissingCapability { capability, .. } => {
                vec![
                    format!("Add the '{}' capability to your function or module", capability),
                    "Check if you have the necessary permissions".to_string(),
                    "Consider using a different approach that doesn't require this capability".to_string(),
                ]
            }
            
            _ => vec!["Refer to the documentation for more information".to_string()],
        }
    }
}

impl ErrorContext {
    /// Create a new error context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add context information
    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.context_info.push(info.into());
        self
    }

    /// Set file path
    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Set function name
    pub fn with_function(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    /// Set module name
    pub fn with_module(mut self, name: impl Into<String>) -> Self {
        self.module_name = Some(name.into());
        self
    }

    /// Add related span
    pub fn with_related_span(mut self, span: Span) -> Self {
        self.related_spans.push(span);
        self
    }

    /// Set AI help text
    pub fn with_ai_help(mut self, help: impl Into<String>) -> Self {
        self.ai_help = Some(help.into());
        self
    }
}

impl ErrorBuilder {
    /// Set error context
    pub fn context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    /// Set error severity
    pub fn severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set error category
    pub fn category(mut self, category: ErrorCategory) -> Self {
        self.category = category;
        self
    }

    /// Build the enhanced error
    pub fn build(self) -> EnhancedError {
        EnhancedError {
            error: self.error,
            context: self.context,
            severity: self.severity,
            category: self.category,
        }
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self {
            file_path: None,
            function_name: None,
            module_name: None,
            line_number: None,
            column_number: None,
            context_info: Vec::new(),
            related_spans: Vec::new(),
            ai_help: None,
        }
    }
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Fatal => write!(f, "fatal"),
            ErrorSeverity::Error => write!(f, "error"),
            ErrorSeverity::Warning => write!(f, "warning"),
            ErrorSeverity::Info => write!(f, "info"),
        }
    }
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorCategory::Syntax => write!(f, "syntax"),
            ErrorCategory::Type => write!(f, "type"),
            ErrorCategory::Semantic => write!(f, "semantic"),
            ErrorCategory::Effect => write!(f, "effect"),
            ErrorCategory::Module => write!(f, "module"),
            ErrorCategory::CodeGen => write!(f, "codegen"),
            ErrorCategory::AI => write!(f, "ai"),
            ErrorCategory::Performance => write!(f, "performance"),
            ErrorCategory::Security => write!(f, "security"),
            ErrorCategory::Configuration => write!(f, "config"),
            ErrorCategory::Internal => write!(f, "internal"),
        }
    }
}

// Convenience macros for creating errors
#[macro_export]
macro_rules! syntax_error {
    ($location:expr, $message:expr) => {
        CompilerError::SyntaxError {
            location: $location,
            message: $message.to_string(),
        }
    };
}

#[macro_export]
macro_rules! type_mismatch {
    ($location:expr, $expected:expr, $found:expr) => {
        CompilerError::TypeMismatch {
            location: $location,
            expected: $expected.to_string(),
            found: $found.to_string(),
        }
    };
}

#[macro_export]
macro_rules! undefined_symbol {
    ($symbol:expr, $location:expr) => {
        CompilerError::UndefinedSymbol {
            symbol: $symbol.to_string(),
            location: $location,
        }
    };
}

#[macro_export]
macro_rules! internal_error {
    ($message:expr) => {
        CompilerError::InternalError {
            message: $message.to_string(),
        }
    };
} 