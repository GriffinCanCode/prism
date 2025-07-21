//! Main syntax detection engine with intelligent pattern recognition.
//!
//! This module implements the core syntax detection logic that analyzes source code
//! to determine which syntax style is being used. It maintains conceptual cohesion
//! by focusing solely on the responsibility of "syntax style identification with
//! confidence scoring and evidence collection".

use crate::detection::{PatternMatcher, HeuristicEngine, ConfidenceScorer};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use rustc_hash::FxHashMap;

/// Intelligent syntax style detector with confidence scoring.
/// 
/// The SyntaxDetector analyzes source code to determine which syntax style
/// is being used. It uses multiple analysis techniques:
/// - Pattern matching for characteristic syntax elements
/// - Heuristic rules based on language conventions
/// - Confidence scoring to handle ambiguous cases
/// - Caching for performance optimization
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing exclusively on syntax
/// detection. It coordinates between pattern matching, heuristic analysis, and
/// confidence scoring, but does not implement parsing or normalization logic.
/// 
/// # Examples
/// 
/// ```rust
/// use prism_syntax::detection::SyntaxDetector;
/// 
/// let mut detector = SyntaxDetector::new();
/// 
/// let c_like_source = r#"
///     module UserAuth {
///         function authenticate(user: User) -> Result<Session, Error> {
///             if (user.isActive && user.hasPermission) {
///                 return Ok(createSession(user));
///             }
///         }
///     }
/// "#;
/// 
/// let result = detector.detect_syntax(c_like_source);
/// assert_eq!(result.detected_style, SyntaxStyle::CLike);
/// assert!(result.confidence > 0.8);
/// ```
#[derive(Debug)]
pub struct SyntaxDetector {
    /// Pattern matcher for syntax-specific elements
    pattern_matcher: PatternMatcher,
    
    /// Heuristic engine for rule-based analysis
    heuristic_engine: HeuristicEngine,
    
    /// Confidence scorer for final determination
    confidence_scorer: ConfidenceScorer,
    
    /// Detection cache for performance
    detection_cache: FxHashMap<u64, DetectionResult>,
    
    /// Configuration for detection behavior
    config: DetectionConfig,
}

/// Configuration for syntax detection behavior
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Minimum confidence threshold for detection
    pub min_confidence_threshold: f64,
    
    /// Whether to cache detection results
    pub enable_caching: bool,
    
    /// Maximum cache size
    pub max_cache_size: usize,
    
    /// Whether to detect mixed styles
    pub detect_mixed_styles: bool,
    
    /// Custom pattern weights
    pub custom_pattern_weights: HashMap<String, f64>,
}

/// Cache for detection results to improve performance
#[derive(Debug)]
struct DetectionCache {
    /// Cached results by source hash
    cache: FxHashMap<u64, DetectionResult>,
    
    /// Maximum cache size
    max_size: usize,
    
    /// Access order for LRU eviction
    access_order: Vec<u64>,
}

/// Supported syntax styles in Prism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyntaxStyle {
    /// C-like syntax (C/C++/Java/JavaScript style)
    /// Characteristics: braces {}, semicolons, parentheses in conditions
    CLike,
    
    /// Python-like syntax (Python/CoffeeScript style)
    /// Characteristics: colons :, indentation-based, no braces
    PythonLike,
    
    /// Rust-like syntax (Rust/Go style)
    /// Characteristics: explicit keywords (fn, mod), snake_case
    RustLike,
    
    /// Canonical Prism syntax (semantic delimiters)
    /// Characteristics: full English keywords, semantic operators
    Canonical,
}

/// Complete detection result with confidence and evidence
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// The detected syntax style
    pub detected_style: SyntaxStyle,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Evidence supporting the detection
    pub evidence: Vec<SyntaxEvidence>,
    
    /// Alternative styles with their confidence scores
    pub alternatives: Vec<AlternativeStyle>,
    
    /// Warnings about mixed styles or ambiguities
    pub warnings: Vec<DetectionWarning>,
}

/// Evidence supporting a particular syntax style detection
#[derive(Debug, Clone)]
pub struct SyntaxEvidence {
    /// The pattern that was matched
    pub pattern: String,
    
    /// Which style this evidence supports
    pub style: SyntaxStyle,
    
    /// Weight/importance of this evidence
    pub weight: f64,
    
    /// Location where the evidence was found
    pub location: Span,
    
    /// Human-readable description
    pub description: String,
    
    /// Type of evidence
    pub evidence_type: EvidenceType,
}

/// Types of evidence for syntax detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Keyword usage (function, fn, def, etc.)
    Keyword,
    
    /// Punctuation patterns (braces, colons, semicolons)
    Punctuation,
    
    /// Operator style (&&, and, etc.)
    Operator,
    
    /// Naming convention (camelCase, snake_case)
    NamingConvention,
    
    /// Structural pattern (indentation, block delimiters)
    Structure,
    
    /// Comment style (// vs #)
    Comments,
}

/// Alternative syntax style with confidence
#[derive(Debug, Clone)]
pub struct AlternativeStyle {
    /// The alternative style
    pub style: SyntaxStyle,
    
    /// Confidence score for this alternative
    pub confidence: f64,
    
    /// Reason why this wasn't chosen as primary
    pub rejection_reason: String,
}

/// Warnings about detection issues
#[derive(Debug, Clone)]
pub struct DetectionWarning {
    /// Type of warning
    pub warning_type: WarningType,
    
    /// Human-readable message
    pub message: String,
    
    /// Location of the issue (if applicable)
    pub location: Option<Span>,
    
    /// Suggestion for resolving the issue
    pub suggestion: Option<String>,
}

/// Types of detection warnings
#[derive(Debug, Clone)]
pub enum WarningType {
    /// Multiple syntax styles detected in same file
    MixedStyles,
    
    /// Low confidence in detection
    LowConfidence,
    
    /// Ambiguous syntax that could match multiple styles
    Ambiguous,
    
    /// Unusual or non-standard syntax patterns
    NonStandard,
}

/// Mixed style warning for files with inconsistent syntax
#[derive(Debug, Clone)]
pub struct MixedStyleWarning {
    /// Line number where inconsistency was found
    pub line: usize,
    
    /// Expected style based on file majority
    pub expected_style: SyntaxStyle,
    
    /// Actually found style on this line
    pub found_style: SyntaxStyle,
    
    /// Suggestion for fixing the inconsistency
    pub suggestion: String,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            enable_caching: true,
            max_cache_size: 1000,
            detect_mixed_styles: true,
            custom_pattern_weights: HashMap::new(),
        }
    }
}

impl DetectionCache {
    /// Create a new detection cache
    fn new(max_size: usize) -> Self {
        Self {
            cache: FxHashMap::default(),
            max_size,
            access_order: Vec::new(),
        }
    }
    
    /// Get cached result for source code
    fn get(&mut self, source_hash: u64) -> Option<DetectionResult> {
        if let Some(result) = self.cache.get(&source_hash) {
            // Update access order for LRU
            self.access_order.retain(|&h| h != source_hash);
            self.access_order.push(source_hash);
            Some(result.clone())
        } else {
            None
        }
    }
    
    /// Insert result into cache
    fn insert(&mut self, source_hash: u64, result: DetectionResult) {
        // Evict oldest entry if at capacity
        if self.cache.len() >= self.max_size {
            if let Some(&oldest) = self.access_order.first() {
                self.cache.remove(&oldest);
                self.access_order.remove(0);
            }
        }
        
        self.cache.insert(source_hash, result);
        self.access_order.push(source_hash);
    }
}

impl SyntaxDetector {
    /// Create a new syntax detector with default configuration
    pub fn new() -> Self {
        Self {
            pattern_matcher: PatternMatcher::new(),
            heuristic_engine: HeuristicEngine::new(),
            confidence_scorer: ConfidenceScorer::new(),
            detection_cache: FxHashMap::default(),
        }
    }
    
    /// Detect the syntax style of source code
    pub fn detect_syntax(&mut self, source: &str) -> Result<DetectionResult, DetectionError> {
        // Check cache first
        let source_hash = self.calculate_source_hash(source);
        if let Some(cached_result) = self.detection_cache.get(&source_hash) {
            return Ok(cached_result.clone());
        }
        
        // Gather evidence from multiple sources
        let pattern_evidence = self.pattern_matcher.find_patterns(source)?;
        let heuristic_evidence = self.heuristic_engine.analyze_heuristics(source);
        
        // Combine all evidence
        let mut all_evidence = pattern_evidence;
        all_evidence.extend(heuristic_evidence);
        
        // Score the evidence and determine the most likely syntax style
        let result = self.confidence_scorer.calculate_confidence(&all_evidence)?;
        
        // Cache the result
        self.detection_cache.insert(source_hash, result.clone());
        
        Ok(result)
    }
    
    /// Calculate a simple hash of the source for caching
    fn calculate_source_hash(&self, source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Clear the detection cache
    pub fn clear_cache(&mut self) {
        self.detection_cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.detection_cache.len(), self.detection_cache.capacity())
    }
} 