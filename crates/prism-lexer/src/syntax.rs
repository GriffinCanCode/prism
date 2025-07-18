//! Syntax detection and style recognition for multi-syntax support
//!
//! This module provides automatic detection of syntax styles and 
//! style-specific lexing rules for the Prism language.

use crate::token::SyntaxStyle;
use prism_common::span::Span;

/// Evidence for syntax style detection
#[derive(Debug, Clone)]
pub struct SyntaxEvidence {
    /// The pattern that was detected
    pub pattern: String,
    /// The syntax style this evidence points to
    pub style: SyntaxStyle,
    /// Weight of this evidence (0.0 to 1.0)
    pub weight: f64,
    /// Location where this evidence was found
    pub location: Span,
}

/// Warning about mixed syntax styles
#[derive(Debug, Clone)]
pub struct MixedStyleWarning {
    /// Description of the mixed style issue
    pub message: String,
    /// Location of the mixed style
    pub location: Span,
    /// Conflicting styles detected
    pub conflicting_styles: Vec<SyntaxStyle>,
}

/// Syntax detector that analyzes source code to determine syntax style
#[derive(Debug, Clone)]
pub struct SyntaxDetector {
    /// The detected syntax style
    pub detected_style: SyntaxStyle,
    /// Confidence in the detection (0.0 to 1.0)
    pub confidence: f64,
    /// Evidence supporting the detection
    pub evidence: Vec<SyntaxEvidence>,
    /// Warnings about mixed styles
    pub mixed_style_warnings: Vec<MixedStyleWarning>,
}

impl SyntaxDetector {
    /// Detect syntax style from source code
    pub fn detect_syntax(source: &str) -> Self {
        let mut detector = Self {
            detected_style: SyntaxStyle::Canonical,
            confidence: 0.0,
            evidence: Vec::new(),
            mixed_style_warnings: Vec::new(),
        };
        
        // Analyze different patterns
        detector.analyze_indentation_patterns(source);
        detector.analyze_brace_patterns(source);
        detector.analyze_keyword_patterns(source);
        detector.analyze_operator_patterns(source);
        
        // Calculate final confidence and detect mixed styles
        detector.calculate_confidence();
        detector.detect_mixed_styles();
        
        detector
    }
    
    /// Analyze indentation patterns for Python-like syntax
    fn analyze_indentation_patterns(&mut self, source: &str) {
        let lines: Vec<&str> = source.lines().collect();
        let mut indented_lines = 0;
        let mut significant_indentation = 0;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                continue;
            }
            
            let indent_level = line.len() - trimmed.len();
            if indent_level > 0 {
                indented_lines += 1;
                
                // Check if indentation follows colon (Python-like)
                if i > 0 && lines[i - 1].trim_end().ends_with(':') {
                    significant_indentation += 1;
                }
            }
        }
        
        if indented_lines > 0 {
            let indent_ratio = indented_lines as f64 / lines.len() as f64;
            let significance_ratio = significant_indentation as f64 / indented_lines as f64;
            
            if indent_ratio > 0.3 && significance_ratio > 0.5 {
                self.evidence.push(SyntaxEvidence {
                    pattern: "Significant indentation with colons".to_string(),
                    style: SyntaxStyle::PythonLike,
                    weight: (indent_ratio * significance_ratio).min(1.0),
                    location: Span::entire_file(),
                });
            }
        }
    }
    
    /// Analyze brace patterns for C-like syntax
    fn analyze_brace_patterns(&mut self, source: &str) {
        let brace_count = source.chars().filter(|&c| c == '{' || c == '}').count();
        let line_count = source.lines().count();
        
        if brace_count > 0 && line_count > 0 {
            let brace_ratio = brace_count as f64 / line_count as f64;
            
            // Check for C-like brace style (opening brace on same line)
            let same_line_braces = source.lines()
                .filter(|line| line.contains('{') && !line.trim().starts_with('{'))
                .count();
            
            if brace_ratio > 0.1 {
                let weight = if same_line_braces > 0 {
                    brace_ratio * 0.8
                } else {
                    brace_ratio * 0.5
                };
                
                self.evidence.push(SyntaxEvidence {
                    pattern: "Frequent brace usage".to_string(),
                    style: SyntaxStyle::CLike,
                    weight: weight.min(1.0),
                    location: Span::entire_file(),
                });
            }
        }
    }
    
    /// Analyze keyword patterns for different syntax styles
    fn analyze_keyword_patterns(&mut self, source: &str) {
        // Rust-like keywords
        let rust_keywords = ["fn", "mod", "impl", "trait", "enum", "struct", "pub"];
        let rust_count = rust_keywords.iter()
            .map(|&keyword| count_word_occurrences(source, keyword))
            .sum::<usize>();
            
        if rust_count > 0 {
            self.evidence.push(SyntaxEvidence {
                pattern: "Rust-like keywords".to_string(),
                style: SyntaxStyle::RustLike,
                weight: (rust_count as f64 * 0.1).min(1.0),
                location: Span::entire_file(),
            });
        }
        
        // Prism canonical keywords
        let prism_keywords = ["function", "module", "section", "capability"];
        let prism_count = prism_keywords.iter()
            .map(|&keyword| count_word_occurrences(source, keyword))
            .sum::<usize>();
            
        if prism_count > 0 {
            self.evidence.push(SyntaxEvidence {
                pattern: "Prism canonical keywords".to_string(),
                style: SyntaxStyle::Canonical,
                weight: (prism_count as f64 * 0.2).min(1.0),
                location: Span::entire_file(),
            });
        }
        
        // C-like keywords
        let c_keywords = ["class", "interface", "namespace", "using"];
        let c_count = c_keywords.iter()
            .map(|&keyword| count_word_occurrences(source, keyword))
            .sum::<usize>();
            
        if c_count > 0 {
            self.evidence.push(SyntaxEvidence {
                pattern: "C-like keywords".to_string(),
                style: SyntaxStyle::CLike,
                weight: (c_count as f64 * 0.15).min(1.0),
                location: Span::entire_file(),
            });
        }
    }
    
    /// Analyze operator patterns
    fn analyze_operator_patterns(&mut self, source: &str) {
        // Symbolic operators (C-like/Rust-like)
        let symbolic_ops = ["&&", "||", "!=", "=="];
        let symbolic_count = symbolic_ops.iter()
            .map(|&op| source.matches(op).count())
            .sum::<usize>();
        
        // English operators (Python-like/Canonical)
        let english_ops = ["and", "or", "not"];
        let english_count = english_ops.iter()
            .map(|&op| count_word_occurrences(source, op))
            .sum::<usize>();
        
        if symbolic_count > english_count && symbolic_count > 0 {
            self.evidence.push(SyntaxEvidence {
                pattern: "Symbolic operators".to_string(),
                style: SyntaxStyle::CLike,
                weight: (symbolic_count as f64 * 0.1).min(1.0),
                location: Span::entire_file(),
            });
        } else if english_count > symbolic_count && english_count > 0 {
            self.evidence.push(SyntaxEvidence {
                pattern: "English operators".to_string(),
                style: SyntaxStyle::Canonical,
                weight: (english_count as f64 * 0.1).min(1.0),
                location: Span::entire_file(),
            });
        }
    }
    
    /// Calculate final confidence based on evidence
    fn calculate_confidence(&mut self) {
        if self.evidence.is_empty() {
            self.confidence = 0.0;
            return;
        }
        
        // Group evidence by style
        let mut style_weights: std::collections::HashMap<SyntaxStyle, f64> = std::collections::HashMap::new();
        
        for evidence in &self.evidence {
            *style_weights.entry(evidence.style.clone()).or_insert(0.0) += evidence.weight;
        }
        
        // Find the style with highest weight
        let (best_style, best_weight) = style_weights.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(style, weight)| (style.clone(), *weight))
            .unwrap_or((SyntaxStyle::Canonical, 0.0));
        
        self.detected_style = best_style;
        
        // Calculate confidence based on evidence strength and consensus
        let total_weight: f64 = style_weights.values().sum();
        self.confidence = if total_weight > 0.0 {
            (best_weight / total_weight).min(1.0)
        } else {
            0.0
        };
    }
    
    /// Detect mixed styles and generate warnings
    fn detect_mixed_styles(&mut self) {
        let mut style_counts: std::collections::HashMap<SyntaxStyle, usize> = std::collections::HashMap::new();
        
        for evidence in &self.evidence {
            if evidence.weight > 0.3 {  // Only consider significant evidence
                *style_counts.entry(evidence.style.clone()).or_insert(0) += 1;
            }
        }
        
        if style_counts.len() > 1 {
            let conflicting_styles: Vec<SyntaxStyle> = style_counts.keys().cloned().collect();
            
            self.mixed_style_warnings.push(MixedStyleWarning {
                message: format!("Mixed syntax styles detected: {:?}", conflicting_styles),
                location: Span::entire_file(),
                conflicting_styles,
            });
        }
    }
}

/// Style-specific lexing rules
#[derive(Debug, Clone)]
pub struct StyleRules {
    /// Whether indentation has semantic meaning
    pub indentation_semantic: bool,
    /// Whether semicolons are required
    pub semicolon_required: bool,
    /// Brace style preference
    pub brace_style: BraceStyle,
    /// Operator style preference
    pub operator_style: OperatorStyle,
    /// Naming convention preference
    pub naming_convention: NamingStyle,
}

/// Brace style preferences
#[derive(Debug, Clone, PartialEq)]
pub enum BraceStyle {
    /// Braces required for structure (C-like)
    Required,
    /// Braces optional, indentation used (Python-like)
    Optional,
    /// Braces are semantic delimiters (Canonical)
    Semantic,
}

/// Operator style preferences
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorStyle {
    /// Symbolic operators (&&, ||, !)
    Symbolic,
    /// English operators (and, or, not)
    English,
    /// Both allowed
    Mixed,
}

/// Naming convention preferences
#[derive(Debug, Clone, PartialEq)]
pub enum NamingStyle {
    /// snake_case
    SnakeCase,
    /// camelCase
    CamelCase,
    /// PascalCase
    PascalCase,
    /// Context-dependent
    Mixed,
}

impl StyleRules {
    /// Get style rules for a specific syntax style
    pub fn for_style(style: &SyntaxStyle) -> Self {
        match style {
            SyntaxStyle::CLike => Self {
                indentation_semantic: false,
                semicolon_required: true,
                brace_style: BraceStyle::Required,
                operator_style: OperatorStyle::Symbolic,
                naming_convention: NamingStyle::CamelCase,
            },
            SyntaxStyle::PythonLike => Self {
                indentation_semantic: true,
                semicolon_required: false,
                brace_style: BraceStyle::Optional,
                operator_style: OperatorStyle::English,
                naming_convention: NamingStyle::SnakeCase,
            },
            SyntaxStyle::RustLike => Self {
                indentation_semantic: false,
                semicolon_required: false,
                brace_style: BraceStyle::Required,
                operator_style: OperatorStyle::Symbolic,
                naming_convention: NamingStyle::SnakeCase,
            },
            SyntaxStyle::Canonical => Self {
                indentation_semantic: false,
                semicolon_required: false,
                brace_style: BraceStyle::Semantic,
                operator_style: OperatorStyle::English,
                naming_convention: NamingStyle::Mixed,
            },
            SyntaxStyle::Mixed => Self::for_style(&SyntaxStyle::Canonical),
        }
    }
}

/// Count word occurrences (not just substring matches)
fn count_word_occurrences(source: &str, word: &str) -> usize {
    let word_regex = regex::Regex::new(&format!(r"\b{}\b", regex::escape(word)))
        .expect("Failed to create regex");
    word_regex.find_iter(source).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_like_detection() {
        let source = r#"
        module Test {
            function foo() {
                if (condition) {
                    return value;
                }
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::CLike);
        assert!(detector.confidence > 0.3);
    }
    
    #[test]
    fn test_python_like_detection() {
        let source = r#"
        module Test:
            function foo():
                if condition:
                    return value
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::PythonLike);
        assert!(detector.confidence > 0.3);
    }
    
    #[test]
    fn test_rust_like_detection() {
        let source = r#"
        mod test {
            fn foo() -> i32 {
                if condition {
                    return 42;
                }
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::RustLike);
        assert!(detector.confidence > 0.3);
    }
    
    #[test]
    fn test_canonical_detection() {
        let source = r#"
        module Test {
            section types {
                type User = String
            }
            
            function createUser() -> User {
                return "test"
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::Canonical);
        assert!(detector.confidence > 0.3);
    }
    
    #[test]
    fn test_mixed_style_warning() {
        let source = r#"
        module Test {
            fn foo() {  // Rust-like fn
                if condition and other:  // English operators
                    return value
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert!(!detector.mixed_style_warnings.is_empty());
    }
} 