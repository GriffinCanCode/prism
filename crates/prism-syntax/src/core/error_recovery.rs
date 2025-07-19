//! Error recovery for robust parsing.
//!
//! This module implements intelligent error recovery strategies to continue parsing
//! even when syntax errors are encountered, maintaining conceptual cohesion around
//! "error recovery and parsing resilience".

use prism_common::span::Span;
use thiserror::Error;

/// Error recovery engine for robust parsing
#[derive(Debug)]
pub struct ErrorRecovery {
    /// Recovery strategy configuration
    strategy: RecoveryStrategy,
    
    /// Recovery points for backtracking
    recovery_points: Vec<RecoveryPoint>,
}

/// Strategy for error recovery
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Conservative recovery (minimal changes)
    Conservative,
    
    /// Aggressive recovery (maximum continuation)
    Aggressive,
    
    /// Balanced recovery (practical middle ground)
    Balanced,
    
    /// Custom recovery with specific rules
    Custom(CustomRecoveryRules),
}

/// Custom rules for error recovery
#[derive(Debug, Clone)]
pub struct CustomRecoveryRules {
    /// Rules for different error types
    pub rules: Vec<RecoveryRule>,
    
    /// Maximum recovery attempts
    pub max_attempts: usize,
    
    /// Whether to generate synthetic tokens
    pub allow_synthetic_tokens: bool,
}

/// A specific recovery rule
#[derive(Debug, Clone)]
pub struct RecoveryRule {
    /// Pattern to match for this rule
    pub pattern: String,
    
    /// Action to take when pattern matches
    pub action: RecoveryAction,
    
    /// Priority of this rule
    pub priority: u8,
}

/// Action to take during error recovery
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Skip the problematic token
    Skip,
    
    /// Insert a synthetic token
    Insert(String),
    
    /// Replace with a different token
    Replace(String),
    
    /// Restart parsing from a recovery point
    Restart,
    
    /// Custom recovery function
    Custom(String), // Function name for now
}

/// Point in parsing where recovery can restart
#[derive(Debug, Clone)]
pub struct RecoveryPoint {
    /// Position in token stream
    pub position: usize,
    
    /// Parser state at this point
    pub state: ParserState,
    
    /// Span where recovery point was created
    pub span: Span,
}

/// Parser state for recovery
#[derive(Debug, Clone)]
pub struct ParserState {
    /// Current parsing context
    pub context: String,
    
    /// Nesting level
    pub nesting_level: usize,
    
    /// Expected tokens at this point
    pub expected_tokens: Vec<String>,
}

/// Error recovery result
#[derive(Debug)]
pub struct RecoveryResult {
    /// Whether recovery was successful
    pub success: bool,
    
    /// Actions taken during recovery
    pub actions_taken: Vec<RecoveryAction>,
    
    /// New position after recovery
    pub new_position: usize,
    
    /// Synthetic tokens generated
    pub synthetic_tokens: Vec<String>,
}

/// Errors during recovery
#[derive(Debug, Error)]
pub enum RecoveryError {
    /// Recovery failed completely
    #[error("Recovery failed: {reason}")]
    RecoveryFailed { reason: String },
    
    /// Maximum recovery attempts exceeded
    #[error("Maximum recovery attempts ({max}) exceeded")]
    MaxAttemptsExceeded { max: usize },
    
    /// No recovery strategy available
    #[error("No recovery strategy available for error: {error}")]
    NoStrategyAvailable { error: String },
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        RecoveryStrategy::Balanced
    }
}

impl ErrorRecovery {
    /// Create a new error recovery engine
    pub fn new() -> Self {
        Self {
            strategy: RecoveryStrategy::default(),
            recovery_points: Vec::new(),
        }
    }
    
    /// Create error recovery with specific strategy
    pub fn with_strategy(strategy: RecoveryStrategy) -> Self {
        Self {
            strategy,
            recovery_points: Vec::new(),
        }
    }
    
    /// Attempt to recover from a parsing error
    pub fn recover_from_error(
        &mut self,
        _error: &str,
        _position: usize
    ) -> Result<RecoveryResult, RecoveryError> {
        // TODO: Implement actual error recovery
        Ok(RecoveryResult {
            success: true,
            actions_taken: vec![RecoveryAction::Skip],
            new_position: _position + 1,
            synthetic_tokens: Vec::new(),
        })
    }
    
    /// Add a recovery point
    pub fn add_recovery_point(&mut self, point: RecoveryPoint) {
        self.recovery_points.push(point);
    }
    
    /// Get the most recent recovery point
    pub fn latest_recovery_point(&self) -> Option<&RecoveryPoint> {
        self.recovery_points.last()
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new()
    }
} 