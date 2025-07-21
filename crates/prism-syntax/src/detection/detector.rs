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
    detection_cache: DetectionCache,
    
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
        Self::with_config(DetectionConfig::default())
    }
    
    /// Create a syntax detector with custom configuration
    pub fn with_config(config: DetectionConfig) -> Self {
        Self {
            pattern_matcher: PatternMatcher::new(),
            heuristic_engine: HeuristicEngine::new(),
            confidence_scorer: ConfidenceScorer::new(),
            detection_cache: DetectionCache::new(config.max_cache_size),
            config,
        }
    }
    
    /// Detects syntax style from source code with comprehensive analysis.
    /// 
    /// This method analyzes the source code using multiple techniques:
    /// 1. Pattern matching for syntax-specific elements
    /// 2. Heuristic analysis of code structure
    /// 3. Confidence scoring for final determination
    /// 4. Caching for performance optimization
    /// 
    /// # Arguments
    /// 
    /// * `source` - The source code to analyze
    /// 
    /// # Returns
    /// 
    /// A `DetectionResult` containing the detected style, confidence score,
    /// supporting evidence, alternatives, and any warnings.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use prism_syntax::detection::SyntaxDetector;
    /// 
    /// let mut detector = SyntaxDetector::new();
    /// 
    /// let python_like_source = r#"
    ///     module UserAuth:
    ///         function authenticate(user: User) -> Result<Session, Error>:
    ///             if user.isActive and user.hasPermission:
    ///                 return Ok(createSession(user))
    /// "#;
    /// 
    /// let result = detector.detect_syntax(python_like_source);
    /// assert_eq!(result.detected_style, SyntaxStyle::PythonLike);
    /// assert!(result.confidence > 0.8);
    /// ```
    pub fn detect_syntax(&mut self, source: &str) -> DetectionResult {
        // Check cache first if enabled
        if self.config.enable_caching {
            let source_hash = self.calculate_source_hash(source);
            if let Some(cached) = self.detection_cache.get(source_hash) {
                return cached;
            }
        }
        
        // Step 1: Analyze patterns for each syntax style
        let pattern_evidence = self.pattern_matcher.analyze_patterns(source);
        
        // Step 2: Apply heuristic rules
        let heuristic_evidence = self.heuristic_engine.analyze_heuristics(source);
        
        // Step 3: Combine all evidence
        let combined_evidence = self.combine_evidence(pattern_evidence, heuristic_evidence);
        
        // Step 4: Score confidence for each style
        let style_scores = self.confidence_scorer.score_styles(&combined_evidence);
        
        // Step 5: Determine primary style and alternatives
        let (detected_style, confidence) = self.determine_primary_style(&style_scores);
        let alternatives = self.generate_alternatives(&style_scores, detected_style);
        
        // Step 6: Generate warnings
        let warnings = self.generate_warnings(source, &combined_evidence, confidence);
        
        let result = DetectionResult {
            detected_style,
            confidence,
            evidence: combined_evidence,
            alternatives,
            warnings,
        };
        
        // Cache result if enabled
        if self.config.enable_caching {
            let source_hash = self.calculate_source_hash(source);
            self.detection_cache.insert(source_hash, result.clone());
        }
        
        result
    }
    
    /// Detects mixed syntax styles within the same file.
    /// 
    /// This method analyzes each significant line to detect inconsistent syntax
    /// styles within the same file, which can indicate copy-paste errors or
    /// inconsistent coding practices.
    /// 
    /// # Arguments
    /// 
    /// * `source` - The source code to analyze
    /// 
    /// # Returns
    /// 
    /// A vector of warnings about mixed syntax styles found in the file.
    pub fn detect_mixed_styles(&mut self, source: &str) -> Vec<MixedStyleWarning> {
        if !self.config.detect_mixed_styles {
            return Vec::new();
        }
        
        let lines = source.lines().enumerate();
        let mut warnings = Vec::new();
        let mut line_styles = Vec::new();
        
        // Analyze each significant line
        for (line_num, line) in lines {
            if line.trim().is_empty() || line.trim_start().starts_with("//") {
                continue;
            }
            
            let line_detection = self.detect_syntax(line);
            line_styles.push((line_num, line_detection.detected_style));
        }
        
        // Find the dominant style
        let dominant_style = self.find_dominant_style(&line_styles);
        
        // Find inconsistencies
        for (line_num, style) in line_styles {
            if style != dominant_style {
                warnings.push(MixedStyleWarning {
                    line: line_num + 1, // 1-indexed for user display
                    expected_style: dominant_style,
                    found_style: style,
                    suggestion: format!(
                        "Consider using {} style consistently throughout the file", 
                        self.style_name(dominant_style)
                    ),
                });
            }
        }
        
        warnings
    }
    
    /// Calculate a hash for source code for caching
    fn calculate_source_hash(&self, source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Combine evidence from patterns and heuristics
    fn combine_evidence(
        &self,
        pattern_evidence: Vec<SyntaxEvidence>,
        heuristic_evidence: Vec<SyntaxEvidence>
    ) -> Vec<SyntaxEvidence> {
        let mut combined = pattern_evidence;
        combined.extend(heuristic_evidence);
        
        // Apply custom weights if configured
        for evidence in &mut combined {
            if let Some(&weight) = self.config.custom_pattern_weights.get(&evidence.pattern) {
                evidence.weight *= weight;
            }
        }
        
        combined
    }
    
    /// Determine the primary style from confidence scores
    fn determine_primary_style(
        &self,
        style_scores: &HashMap<SyntaxStyle, f64>
    ) -> (SyntaxStyle, f64) {
        style_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&style, &confidence)| (style, confidence))
            .unwrap_or((SyntaxStyle::Canonical, 0.0))
    }
    
    /// Generate alternative style suggestions
    fn generate_alternatives(
        &self,
        style_scores: &HashMap<SyntaxStyle, f64>,
        primary_style: SyntaxStyle
    ) -> Vec<AlternativeStyle> {
        let mut alternatives = Vec::new();
        
        for (&style, &confidence) in style_scores {
            if style != primary_style && confidence > 0.3 {
                alternatives.push(AlternativeStyle {
                    style,
                    confidence,
                    rejection_reason: format!(
                        "Lower confidence ({:.2}) than primary style", 
                        confidence
                    ),
                });
            }
        }
        
        // Sort by confidence descending
        alternatives.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        alternatives
    }
    
    /// Generate warnings about detection issues
    fn generate_warnings(
        &mut self,
        source: &str,
        evidence: &[SyntaxEvidence],
        confidence: f64
    ) -> Vec<DetectionWarning> {
        let mut warnings = Vec::new();
        
        // Low confidence warning
        if confidence < self.config.min_confidence_threshold {
            warnings.push(DetectionWarning {
                warning_type: WarningType::LowConfidence,
                message: format!(
                    "Low confidence ({:.2}) in syntax detection. Consider using explicit style annotation.",
                    confidence
                ),
                location: None,
                suggestion: Some("Add explicit syntax style annotation to clarify intent".to_string()),
            });
        }
        
        // Check for mixed styles if enabled
        if self.config.detect_mixed_styles {
            let mixed_warnings = self.detect_mixed_styles(source);
            if !mixed_warnings.is_empty() {
                warnings.push(DetectionWarning {
                    warning_type: WarningType::MixedStyles,
                    message: format!("Found {} mixed style issues in file", mixed_warnings.len()),
                    location: None,
                    suggestion: Some("Use consistent syntax style throughout the file".to_string()),
                });
            }
        }
        
        warnings
    }
    
    /// Find the dominant style from line-by-line analysis
    fn find_dominant_style(&self, line_styles: &[(usize, SyntaxStyle)]) -> SyntaxStyle {
        let mut style_counts = HashMap::new();
        
        for (_, style) in line_styles {
            *style_counts.entry(*style).or_insert(0) += 1;
        }
        
        style_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(style, _)| style)
            .unwrap_or(SyntaxStyle::Canonical)
    }
    
    /// Get human-readable name for a syntax style
    fn style_name(&self, style: SyntaxStyle) -> &'static str {
        match style {
            SyntaxStyle::CLike => "C-like",
            SyntaxStyle::PythonLike => "Python-like",
            SyntaxStyle::RustLike => "Rust-like",
            SyntaxStyle::Canonical => "Canonical",
        }
    }
}

impl Default for SyntaxDetector {
    fn default() -> Self {
        Self::new()
    }
} 