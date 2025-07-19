//! Pattern matching for syntax style detection.
//!
//! This module implements pattern recognition for characteristic syntax elements
//! of different programming styles, maintaining conceptual cohesion around the
//! single responsibility of "syntax pattern identification and evidence collection".

use super::{SyntaxEvidence, SyntaxStyle, EvidenceType};
use prism_common::span::Span;
use regex::Regex;
use once_cell::sync::Lazy;

/// Pattern matcher for syntax-specific elements
#[derive(Debug)]
pub struct PatternMatcher {
    /// Compiled regex patterns for efficient matching
    patterns: Vec<CompiledPattern>,
}

/// A compiled pattern for fast matching
#[derive(Debug)]
struct CompiledPattern {
    /// Compiled regex
    regex: Regex,
    /// Original pattern metadata
    pattern: SyntaxPattern,
}

/// A pattern that identifies a specific syntax style
#[derive(Debug, Clone)]
pub struct SyntaxPattern {
    /// Regular expression pattern
    pub pattern: String,
    
    /// Syntax style this pattern identifies
    pub style: SyntaxStyle,
    
    /// Weight/importance of this pattern
    pub weight: f64,
    
    /// Type of evidence this pattern provides
    pub evidence_type: EvidenceType,
    
    /// Human-readable description
    pub description: String,
}

/// Evidence found by pattern matching
pub type PatternEvidence = SyntaxEvidence;

// Pre-compiled regex patterns for performance
static C_LIKE_BRACE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\{\s*$").unwrap());
static PYTHON_COLON: Lazy<Regex> = Lazy::new(|| Regex::new(r":\s*$").unwrap());
static RUST_FN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bfn\s+\w+").unwrap());
static CANONICAL_FUNCTION: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bfunction\s+\w+").unwrap());
static C_LIKE_SEMICOLON: Lazy<Regex> = Lazy::new(|| Regex::new(r";\s*$").unwrap());
static PYTHON_INDENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(\s{4}|\t)+\w+").unwrap());
static RUST_MATCH: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bmatch\s+\w+\s*\{").unwrap());
static LOGICAL_AND: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(and|&&)\b").unwrap());

impl PatternMatcher {
    /// Create a new pattern matcher with compiled patterns
    pub fn new() -> Self {
        let patterns = Self::create_default_patterns();
        let compiled = patterns
            .into_iter()
            .filter_map(|pattern| {
                match Regex::new(&pattern.pattern) {
                    Ok(regex) => Some(CompiledPattern { regex, pattern }),
                    Err(_) => {
                        // Log warning in production, skip invalid patterns
                        None
                    }
                }
            })
            .collect();
            
        Self {
            patterns: compiled,
        }
    }
    
    /// Analyze source code for syntax patterns
    pub fn analyze_patterns(&self, source: &str) -> Vec<PatternEvidence> {
        let mut evidence = Vec::new();
        let lines: Vec<&str> = source.lines().collect();
        
        // Use pre-compiled patterns for performance
        self.analyze_with_precompiled_patterns(source, &lines, &mut evidence);
        
        // Additional contextual analysis
        self.analyze_contextual_patterns(source, &lines, &mut evidence);
        
        evidence
    }
    
    /// Fast analysis using pre-compiled regex patterns
    fn analyze_with_precompiled_patterns(
        &self,
        source: &str,
        lines: &[&str],
        evidence: &mut Vec<PatternEvidence>
    ) {
        // C-like patterns
        if C_LIKE_BRACE.is_match(source) {
            let count = lines.iter().filter(|line| C_LIKE_BRACE.is_match(line)).count();
            evidence.push(SyntaxEvidence {
                pattern: "c_like_braces".to_string(),
                style: SyntaxStyle::CLike,
                weight: 0.8 * (count as f64 / lines.len() as f64).min(1.0),
                location: Span::dummy(),
                description: "C-like opening braces".to_string(),
                evidence_type: EvidenceType::Punctuation,
            });
        }
        
        if C_LIKE_SEMICOLON.is_match(source) {
            let count = lines.iter().filter(|line| C_LIKE_SEMICOLON.is_match(line)).count();
            evidence.push(SyntaxEvidence {
                pattern: "c_like_semicolons".to_string(),
                style: SyntaxStyle::CLike,
                weight: 0.6 * (count as f64 / lines.len() as f64).min(1.0),
                location: Span::dummy(),
                description: "C-like semicolons".to_string(),
                evidence_type: EvidenceType::Punctuation,
            });
        }
        
        // Python-like patterns
        if PYTHON_COLON.is_match(source) {
            let count = lines.iter().filter(|line| PYTHON_COLON.is_match(line)).count();
            evidence.push(SyntaxEvidence {
                pattern: "python_colons".to_string(),
                style: SyntaxStyle::PythonLike,
                weight: 0.9 * (count as f64 / lines.len() as f64).min(1.0),
                location: Span::dummy(),
                description: "Python-like colon delimiters".to_string(),
                evidence_type: EvidenceType::Punctuation,
            });
        }
        
        if PYTHON_INDENT.is_match(source) {
            let count = lines.iter().filter(|line| PYTHON_INDENT.is_match(line)).count();
            evidence.push(SyntaxEvidence {
                pattern: "python_indentation".to_string(),
                style: SyntaxStyle::PythonLike,
                weight: 0.7 * (count as f64 / lines.len() as f64).min(1.0),
                location: Span::dummy(),
                description: "Python-like indentation".to_string(),
                evidence_type: EvidenceType::Structure,
            });
        }
        
        // Rust-like patterns
        if RUST_FN.is_match(source) {
            let count = RUST_FN.find_iter(source).count();
            evidence.push(SyntaxEvidence {
                pattern: "rust_fn_keyword".to_string(),
                style: SyntaxStyle::RustLike,
                weight: 0.9 * (count as f64).min(1.0),
                location: Span::dummy(),
                description: "Rust-like fn keyword".to_string(),
                evidence_type: EvidenceType::Keyword,
            });
        }
        
        if RUST_MATCH.is_match(source) {
            let count = RUST_MATCH.find_iter(source).count();
            evidence.push(SyntaxEvidence {
                pattern: "rust_match".to_string(),
                style: SyntaxStyle::RustLike,
                weight: 0.8 * (count as f64).min(1.0),
                location: Span::dummy(),
                description: "Rust-like match expressions".to_string(),
                evidence_type: EvidenceType::Keyword,
            });
        }
        
        // Canonical patterns
        if CANONICAL_FUNCTION.is_match(source) {
            let count = CANONICAL_FUNCTION.find_iter(source).count();
            evidence.push(SyntaxEvidence {
                pattern: "canonical_function".to_string(),
                style: SyntaxStyle::Canonical,
                weight: 0.8 * (count as f64).min(1.0),
                location: Span::dummy(),
                description: "Canonical function keyword".to_string(),
                evidence_type: EvidenceType::Keyword,
            });
        }
    }
    
    /// Analyze contextual patterns that require more sophisticated logic
    fn analyze_contextual_patterns(
        &self,
        source: &str,
        lines: &[&str],
        evidence: &mut Vec<PatternEvidence>
    ) {
        // Logical operators analysis
        if let Some(captures) = LOGICAL_AND.captures(source) {
            if let Some(matched) = captures.get(1) {
                match matched.as_str() {
                    "and" => {
                        evidence.push(SyntaxEvidence {
                            pattern: "python_logical_and".to_string(),
                            style: SyntaxStyle::PythonLike,
                            weight: 0.7,
                            location: Span::dummy(),
                            description: "Python-like 'and' operator".to_string(),
                            evidence_type: EvidenceType::Operator,
                        });
                    }
                    "&&" => {
                        evidence.push(SyntaxEvidence {
                            pattern: "c_like_logical_and".to_string(),
                            style: SyntaxStyle::CLike,
                            weight: 0.6,
                            location: Span::dummy(),
                            description: "C-like '&&' operator".to_string(),
                            evidence_type: EvidenceType::Operator,
                        });
                    }
                    _ => {}
                }
            }
        }
        
        // Block structure analysis
        let brace_count = source.matches('{').count();
        let colon_count = lines.iter().filter(|line| line.trim_end().ends_with(':')).count();
        
        if brace_count > 0 && colon_count > 0 {
            // Mixed style - reduce confidence for both
            evidence.iter_mut().for_each(|e| {
                if matches!(e.style, SyntaxStyle::CLike | SyntaxStyle::PythonLike) {
                    e.weight *= 0.7; // Reduce confidence for mixed patterns
                }
            });
        }
    }
    
    /// Create default patterns for syntax detection
    fn create_default_patterns() -> Vec<SyntaxPattern> {
        vec![
            // C-like patterns
            SyntaxPattern {
                pattern: r"\{\s*$".to_string(),
                style: SyntaxStyle::CLike,
                weight: 0.8,
                evidence_type: EvidenceType::Punctuation,
                description: "C-like opening brace".to_string(),
            },
            SyntaxPattern {
                pattern: r";\s*$".to_string(),
                style: SyntaxStyle::CLike,
                weight: 0.6,
                evidence_type: EvidenceType::Punctuation,
                description: "C-like semicolon".to_string(),
            },
            
            // Python-like patterns
            SyntaxPattern {
                pattern: r":\s*$".to_string(),
                style: SyntaxStyle::PythonLike,
                weight: 0.9,
                evidence_type: EvidenceType::Punctuation,
                description: "Python-like colon delimiter".to_string(),
            },
            SyntaxPattern {
                pattern: r"^(\s{4}|\t)+\w+".to_string(),
                style: SyntaxStyle::PythonLike,
                weight: 0.7,
                evidence_type: EvidenceType::Structure,
                description: "Python-like indentation".to_string(),
            },
            
            // Rust-like patterns
            SyntaxPattern {
                pattern: r"\bfn\s+\w+".to_string(),
                style: SyntaxStyle::RustLike,
                weight: 0.9,
                evidence_type: EvidenceType::Keyword,
                description: "Rust-like function keyword".to_string(),
            },
            SyntaxPattern {
                pattern: r"\bmatch\s+\w+\s*\{".to_string(),
                style: SyntaxStyle::RustLike,
                weight: 0.8,
                evidence_type: EvidenceType::Keyword,
                description: "Rust-like match expression".to_string(),
            },
            
            // Canonical patterns
            SyntaxPattern {
                pattern: r"\bfunction\s+\w+".to_string(),
                style: SyntaxStyle::Canonical,
                weight: 0.8,
                evidence_type: EvidenceType::Keyword,
                description: "Canonical function keyword".to_string(),
            },
        ]
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_c_like_pattern_detection() {
        let matcher = PatternMatcher::new();
        
        let c_like_source = r#"
            module UserAuth {
                function authenticate(user: User) -> Result<Session, Error> {
                    return processAuth(user);
                }
            }
        "#;
        
        let evidence = matcher.analyze_patterns(c_like_source);
        
        // Should find C-like evidence
        assert!(!evidence.is_empty(), "Should find some evidence");
        
        let c_like_evidence: Vec<_> = evidence.iter()
            .filter(|e| e.style == SyntaxStyle::CLike)
            .collect();
            
        assert!(!c_like_evidence.is_empty(), "Should find C-like evidence");
        
        // Check for specific patterns
        let has_braces = c_like_evidence.iter().any(|e| e.pattern == "c_like_braces");
        let has_semicolons = c_like_evidence.iter().any(|e| e.pattern == "c_like_semicolons");
        
        assert!(has_braces, "Should detect C-like braces");
        assert!(has_semicolons, "Should detect C-like semicolons");
    }
    
    #[test]
    fn test_python_like_pattern_detection() {
        let matcher = PatternMatcher::new();
        
        let python_like_source = r#"
            module UserAuth:
                function authenticate(user: User) -> Result<Session, Error>:
                    if user.isActive and user.hasPermission:
                        return processAuth(user)
        "#;
        
        let evidence = matcher.analyze_patterns(python_like_source);
        
        let python_evidence: Vec<_> = evidence.iter()
            .filter(|e| e.style == SyntaxStyle::PythonLike)
            .collect();
            
        assert!(!python_evidence.is_empty(), "Should find Python-like evidence");
        
        // Check for specific patterns
        let has_colons = python_evidence.iter().any(|e| e.pattern == "python_colons");
        let has_logical_and = python_evidence.iter().any(|e| e.pattern == "python_logical_and");
        
        assert!(has_colons, "Should detect Python-like colons");
        assert!(has_logical_and, "Should detect Python-like 'and' operator");
    }
    
    #[test]
    fn test_rust_like_pattern_detection() {
        let matcher = PatternMatcher::new();
        
        let rust_like_source = r#"
            mod user_auth {
                fn authenticate(user: User) -> Result<Session, Error> {
                    match user.status {
                        Active => process_auth(user),
                        _ => Err(AuthError::Inactive)
                    }
                }
            }
        "#;
        
        let evidence = matcher.analyze_patterns(rust_like_source);
        
        let rust_evidence: Vec<_> = evidence.iter()
            .filter(|e| e.style == SyntaxStyle::RustLike)
            .collect();
            
        assert!(!rust_evidence.is_empty(), "Should find Rust-like evidence");
        
        // Check for specific patterns
        let has_fn_keyword = rust_evidence.iter().any(|e| e.pattern == "rust_fn_keyword");
        let has_match = rust_evidence.iter().any(|e| e.pattern == "rust_match");
        
        assert!(has_fn_keyword, "Should detect Rust 'fn' keyword");
        assert!(has_match, "Should detect Rust 'match' expression");
    }
    
    #[test]
    fn test_mixed_style_penalty() {
        let matcher = PatternMatcher::new();
        
        // Source with mixed C-like and Python-like elements
        let mixed_source = r#"
            module UserAuth {
                function authenticate(user: User):
                    return processAuth(user);
            }
        "#;
        
        let evidence = matcher.analyze_patterns(mixed_source);
        
        // Should have both C-like and Python-like evidence
        let c_like_count = evidence.iter().filter(|e| e.style == SyntaxStyle::CLike).count();
        let python_count = evidence.iter().filter(|e| e.style == SyntaxStyle::PythonLike).count();
        
        assert!(c_like_count > 0, "Should find C-like evidence");
        assert!(python_count > 0, "Should find Python-like evidence");
        
        // Weights should be reduced due to mixed style penalty
        let avg_weight: f64 = evidence.iter().map(|e| e.weight).sum::<f64>() / evidence.len() as f64;
        assert!(avg_weight < 0.8, "Average weight should be reduced due to mixed styles: {}", avg_weight);
    }
    
    #[test]
    fn test_pattern_matcher_performance() {
        let matcher = PatternMatcher::new();
        
        // Large source code for performance test
        let large_source = r#"
            module UserAuth {
                function authenticate(user: User) -> Result<Session, Error> {
                    if (user.isActive && user.hasPermission) {
                        return Ok(createSession(user));
                    } else {
                        return Err(AuthError::InvalidUser);
                    }
                }
                
                function createSession(user: User) -> Session {
                    return Session {
                        userId: user.id,
                        timestamp: now(),
                        permissions: user.permissions
                    };
                }
            }
        "#.repeat(100); // Repeat 100 times for performance test
        
        let start = std::time::Instant::now();
        let evidence = matcher.analyze_patterns(&large_source);
        let duration = start.elapsed();
        
        assert!(!evidence.is_empty(), "Should find evidence in large source");
        assert!(duration.as_millis() < 100, "Should complete quickly: {}ms", duration.as_millis());
    }
} 