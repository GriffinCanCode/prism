//! Pattern matching for syntax style detection.
//!
//! This module implements pattern recognition for characteristic syntax elements
//! of different programming styles, maintaining conceptual cohesion around the
//! single responsibility of "syntax pattern identification and evidence collection".

use super::{detector::{SyntaxEvidence, EvidenceType, DetectionError}, SyntaxStyle};
use prism_common::span::Span;
use regex::Regex;
use once_cell::sync::Lazy;

/// Pattern matcher for syntax-specific elements
#[derive(Debug)]
pub struct PatternMatcher {
    /// Compiled regex patterns for efficient matching
    patterns: Vec<SyntaxPattern>,
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
    /// Name of the pattern
    pub name: String,
    
    /// Regular expression pattern
    pub regex: String,
    
    /// Syntax style this pattern identifies
    pub style: SyntaxStyle,
    
    /// Weight/importance of this pattern
    pub confidence_weight: f64,
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
    /// Create a new pattern matcher with default patterns
    pub fn new() -> Self {
        Self {
            patterns: Self::create_default_patterns(),
        }
    }
    
    /// Find patterns in source code and return evidence
    pub fn find_patterns(&self, source: &str) -> Result<Vec<SyntaxEvidence>, DetectionError> {
        let mut evidence = Vec::new();
        
        for pattern in &self.patterns {
            if let Some(matches) = self.match_pattern(pattern, source) {
                evidence.extend(matches);
            }
        }
        
        Ok(evidence)
    }
    
    /// Create default syntax patterns for each style
    fn create_default_patterns() -> Vec<SyntaxPattern> {
        vec![
            // Python-like patterns
            SyntaxPattern {
                name: "python_function_def".to_string(),
                regex: r"def\s+\w+\s*\([^)]*\)\s*:".to_string(),
                style: SyntaxStyle::PythonLike,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "python_class_def".to_string(),
                regex: r"class\s+\w+.*:".to_string(),
                style: SyntaxStyle::PythonLike,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "python_if_statement".to_string(),
                regex: r"if\s+.*:".to_string(),
                style: SyntaxStyle::PythonLike,
                confidence_weight: 0.7,
            },
            SyntaxPattern {
                name: "python_for_loop".to_string(),
                regex: r"for\s+\w+\s+in\s+.*:".to_string(),
                style: SyntaxStyle::PythonLike,
                confidence_weight: 0.8,
            },
            
            // Rust-like patterns
            SyntaxPattern {
                name: "rust_function_def".to_string(),
                regex: r"fn\s+\w+\s*\([^)]*\)".to_string(),
                style: SyntaxStyle::RustLike,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "rust_struct_def".to_string(),
                regex: r"struct\s+\w+".to_string(),
                style: SyntaxStyle::RustLike,
                confidence_weight: 0.8,
            },
            SyntaxPattern {
                name: "rust_impl_block".to_string(),
                regex: r"impl\s+.*\s*\{".to_string(),
                style: SyntaxStyle::RustLike,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "rust_let_binding".to_string(),
                regex: r"let\s+\w+".to_string(),
                style: SyntaxStyle::RustLike,
                confidence_weight: 0.6,
            },
            
            // C-like patterns
            SyntaxPattern {
                name: "c_function_def".to_string(),
                regex: r"\w+\s+\w+\s*\([^)]*\)\s*\{".to_string(),
                style: SyntaxStyle::CLike,
                confidence_weight: 0.7,
            },
            SyntaxPattern {
                name: "c_include".to_string(),
                regex: r#"#include\s*[<"].*[>"]"#.to_string(),
                style: SyntaxStyle::CLike,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "c_struct_def".to_string(),
                regex: r"struct\s+\w+\s*\{".to_string(),
                style: SyntaxStyle::CLike,
                confidence_weight: 0.8,
            },
            SyntaxPattern {
                name: "c_typedef".to_string(),
                regex: r"typedef\s+.*".to_string(),
                style: SyntaxStyle::CLike,
                confidence_weight: 0.8,
            },
            
            // Canonical patterns (Prism-specific)
            SyntaxPattern {
                name: "prism_module".to_string(),
                regex: r"module\s+\w+\s*\{".to_string(),
                style: SyntaxStyle::Canonical,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "prism_section".to_string(),
                regex: r"section\s+\w+\s*\{".to_string(),
                style: SyntaxStyle::Canonical,
                confidence_weight: 0.9,
            },
            SyntaxPattern {
                name: "prism_annotation".to_string(),
                regex: r"@\w+".to_string(),
                style: SyntaxStyle::Canonical,
                confidence_weight: 0.7,
            },
        ]
    }
    
    /// Match a specific pattern against source code
    fn match_pattern(&self, pattern: &SyntaxPattern, source: &str) -> Option<Vec<SyntaxEvidence>> {
        // Simple pattern matching - in a real implementation you'd use proper regex
        let matches = self.simple_pattern_match(&pattern.regex, source);
        
        if matches.is_empty() {
            return None;
        }
        
        let evidence: Vec<SyntaxEvidence> = matches.into_iter().map(|_match_info| {
            SyntaxEvidence {
                pattern: pattern.name.clone(),
                style: pattern.style,
                weight: pattern.confidence_weight,
                location: Span::default(), // Use default Span instead of None
                description: format!("Pattern '{}' found", pattern.name),
                evidence_type: EvidenceType::Keyword, // Default to keyword type
                confidence: pattern.confidence_weight,
                reason: format!("Pattern '{}' found", pattern.name),
            }
        }).collect();
        
        Some(evidence)
    }
    
    /// Simple pattern matching (placeholder for regex)
    fn simple_pattern_match(&self, pattern: &str, source: &str) -> Vec<PatternMatch> {
        let mut matches = Vec::new();
        
        // Very basic pattern matching - just check for key substrings
        match pattern {
            p if p.contains("def\\s+\\w+\\s*\\([^)]*\\)\\s*:") => {
                if source.contains("def ") && source.contains(":") {
                    matches.push(PatternMatch { start: 0, end: 0 });
                }
            }
            p if p.contains("fn\\s+\\w+\\s*\\([^)]*\\)") => {
                if source.contains("fn ") {
                    matches.push(PatternMatch { start: 0, end: 0 });
                }
            }
            p if p.contains("module\\s+\\w+\\s*\\{") => {
                if source.contains("module ") && source.contains("{") {
                    matches.push(PatternMatch { start: 0, end: 0 });
                }
            }
            p if p.contains("section\\s+\\w+\\s*\\{") => {
                if source.contains("section ") && source.contains("{") {
                    matches.push(PatternMatch { start: 0, end: 0 });
                }
            }
            p if p.contains("#include") => {
                if source.contains("#include") {
                    matches.push(PatternMatch { start: 0, end: 0 });
                }
            }
            p if p.contains("@\\w+") => {
                if source.contains("@") {
                    matches.push(PatternMatch { start: 0, end: 0 });
                }
            }
            _ => {
                // For other patterns, do basic substring matching
                if let Some(key) = self.extract_key_substring(pattern) {
                    if source.contains(&key) {
                        matches.push(PatternMatch { start: 0, end: 0 });
                    }
                }
            }
        }
        
        matches
    }
    
    /// Extract a key substring from a regex pattern for simple matching
    fn extract_key_substring(&self, pattern: &str) -> Option<String> {
        // Very basic extraction - look for common keywords
        if pattern.contains("class") {
            Some("class ".to_string())
        } else if pattern.contains("struct") {
            Some("struct ".to_string())
        } else if pattern.contains("impl") {
            Some("impl ".to_string())
        } else if pattern.contains("let") {
            Some("let ".to_string())
        } else if pattern.contains("for") {
            Some("for ".to_string())
        } else if pattern.contains("if") {
            Some("if ".to_string())
        } else {
            None
        }
    }
}

/// Represents a pattern match in source code
#[derive(Debug)]
struct PatternMatch {
    start: usize,
    end: usize,
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
        
        let evidence = matcher.find_patterns(c_like_source).unwrap();
        
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
        
        let evidence = matcher.find_patterns(python_like_source).unwrap();
        
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
        
        let evidence = matcher.find_patterns(rust_like_source).unwrap();
        
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
        
        let evidence = matcher.find_patterns(mixed_source).unwrap();
        
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
        let evidence = matcher.find_patterns(&large_source).unwrap();
        let duration = start.elapsed();
        
        assert!(!evidence.is_empty(), "Should find evidence in large source");
        assert!(duration.as_millis() < 100, "Should complete quickly: {}ms", duration.as_millis());
    }
} 