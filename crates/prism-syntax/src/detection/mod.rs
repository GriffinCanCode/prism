//! Syntax style detection with confidence scoring.
//! 
//! This module analyzes source code to determine which syntax style is being
//! used (C-like, Python-like, Rust-like, or Canonical). It uses pattern
//! matching and heuristics to provide confidence scores for each style,
//! maintaining conceptual cohesion around the single responsibility of
//! "intelligent syntax style recognition and confidence assessment".

pub mod detector;
pub mod patterns;
pub mod heuristics;
pub mod confidence;

pub use detector::{SyntaxDetector, DetectionResult, SyntaxStyle, DetectionWarning};
pub use patterns::{PatternMatcher, SyntaxPattern, PatternEvidence, SyntaxEvidence};
pub use heuristics::{HeuristicEngine, HeuristicRule, HeuristicWeight};
pub use confidence::{ConfidenceScorer, ConfidenceLevel, DetectionConfidence}; 