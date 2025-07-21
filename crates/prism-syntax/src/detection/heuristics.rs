//! Heuristic analysis for syntax style detection.
//!
//! This module implements rule-based heuristic analysis to identify syntax styles
//! based on structural patterns and conventions, maintaining conceptual cohesion
//! around "heuristic rule application and evidence generation".

use super::{detector::{SyntaxEvidence, EvidenceType}, SyntaxStyle};
use std::collections::HashMap;

/// Heuristic engine for rule-based syntax analysis
#[derive(Debug)]
pub struct HeuristicEngine {
    /// Collection of heuristic rules
    rules: Vec<HeuristicRule>,
}

/// A heuristic rule for syntax detection
#[derive(Debug, Clone)]
pub struct HeuristicRule {
    /// Name of the rule
    pub name: String,
    
    /// Weight of this rule's evidence
    pub weight: HeuristicWeight,
    
    /// Function to apply the rule
    pub apply_fn: fn(&str) -> Vec<SyntaxEvidence>,
}

/// Weight classification for heuristic rules
#[derive(Debug, Clone)]
pub enum HeuristicWeight {
    /// Low importance (0.0 - 0.3)
    Low(f64),
    /// Medium importance (0.3 - 0.7)
    Medium(f64),
    /// High importance (0.7 - 1.0)
    High(f64),
}

impl HeuristicEngine {
    /// Create a new heuristic engine with default rules
    pub fn new() -> Self {
        Self {
            rules: Self::create_default_rules(),
        }
    }
    
    /// Apply heuristic analysis to source code
    pub fn analyze_heuristics(&self, source: &str) -> Vec<SyntaxEvidence> {
        let mut evidence = Vec::new();
        
        // Apply all heuristic rules
        for rule in &self.rules {
            let rule_evidence = (rule.apply_fn)(source);
            evidence.extend(rule_evidence);
        }
        
        evidence
    }
    
    /// Create default heuristic rules
    fn create_default_rules() -> Vec<HeuristicRule> {
        vec![
            HeuristicRule {
                name: "indentation_analysis".to_string(),
                weight: HeuristicWeight::High(0.8),
                apply_fn: Self::analyze_indentation,
            },
            HeuristicRule {
                name: "operator_style".to_string(),
                weight: HeuristicWeight::Medium(0.6),
                apply_fn: Self::analyze_operators,
            },
            HeuristicRule {
                name: "naming_convention".to_string(),
                weight: HeuristicWeight::Medium(0.5),
                apply_fn: Self::analyze_naming,
            },
            HeuristicRule {
                name: "delimiter_patterns".to_string(),
                weight: HeuristicWeight::High(0.9),
                apply_fn: Self::analyze_delimiters,
            },
            HeuristicRule {
                name: "keyword_patterns".to_string(),
                weight: HeuristicWeight::High(0.7),
                apply_fn: Self::analyze_keywords,
            },
        ]
    }
    
    /// Analyze indentation patterns
    fn analyze_indentation(source: &str) -> Vec<SyntaxEvidence> {
        let mut evidence = Vec::new();
        let lines: Vec<&str> = source.lines().collect();
        
        let mut significant_indentation = 0;
        let mut brace_delimited = 0;
        let mut colon_after_statement = 0;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }
            
            // Count leading whitespace
            let leading_spaces = line.len() - line.trim_start().len();
            
            // Check for significant indentation (Python-like)
            if leading_spaces > 0 && !trimmed.starts_with('{') && !trimmed.starts_with('}') {
                significant_indentation += 1;
            }
            
            // Check for braces (C-like/Rust-like)
            if trimmed.contains('{') || trimmed.contains('}') {
                brace_delimited += 1;
            }
            
            // Check for colon at end of statement (Python-like)
            if trimmed.ends_with(':') && 
               (trimmed.starts_with("if ") || 
                trimmed.starts_with("def ") || 
                trimmed.starts_with("class ") ||
                trimmed.starts_with("for ") ||
                trimmed.starts_with("while ")) {
                colon_after_statement += 1;
            }
        }
        
        let total_lines = lines.len().max(1);
        
        // Generate evidence based on patterns
        if significant_indentation as f64 / total_lines as f64 > 0.3 {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::PythonLike,
                confidence: 0.7,
                reason: format!("Significant indentation found in {}% of lines", 
                    (significant_indentation * 100 / total_lines)),
                location: None,
            });
        }
        
        if brace_delimited as f64 / total_lines as f64 > 0.2 {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::CLike,
                confidence: 0.6,
                reason: format!("Brace delimiters found in {}% of lines", 
                    (brace_delimited * 100 / total_lines)),
                location: None,
            });
        }
        
        if colon_after_statement > 0 {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::PythonLike,
                confidence: 0.8,
                reason: format!("{} Python-like control structures found", colon_after_statement),
                location: None,
            });
        }
        
        evidence
    }
    
    /// Analyze operator styles
    fn analyze_operators(source: &str) -> Vec<SyntaxEvidence> {
        let mut evidence = Vec::new();
        
        // Count different operator styles
        let and_count = source.matches(" and ").count();
        let or_count = source.matches(" or ").count();
        let logical_and_count = source.matches("&&").count();
        let logical_or_count = source.matches("||").count();
        let not_count = source.matches(" not ").count();
        let bang_count = source.matches('!').count() - source.matches("!=").count();
        
        // Python-like operators
        if and_count > 0 || or_count > 0 || not_count > 0 {
            let total_python_ops = and_count + or_count + not_count;
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::PythonLike,
                confidence: (total_python_ops as f64 * 0.1).min(0.8),
                reason: format!("Python-style logical operators found: {} instances", total_python_ops),
                location: None,
            });
        }
        
        // C-like/Rust-like operators
        if logical_and_count > 0 || logical_or_count > 0 || bang_count > 0 {
            let total_c_ops = logical_and_count + logical_or_count + bang_count;
            let style = if source.contains("fn ") || source.contains("let ") {
                crate::detection::SyntaxStyle::RustLike
            } else {
                crate::detection::SyntaxStyle::CLike
            };
            
            evidence.push(SyntaxEvidence {
                style,
                confidence: (total_c_ops as f64 * 0.1).min(0.7),
                reason: format!("C-style logical operators found: {} instances", total_c_ops),
                location: None,
            });
        }
        
        evidence
    }
    
    /// Analyze naming conventions
    fn analyze_naming(source: &str) -> Vec<SyntaxEvidence> {
        let mut evidence = Vec::new();
        
        // Simple regex-like pattern matching for naming conventions
        let snake_case_count = source.matches(char::is_lowercase)
            .filter(|_| source.contains('_'))
            .count();
        let camel_case_indicators = source.matches(|c: char| c.is_uppercase()).count();
        
        // This is a simplified analysis - in practice, you'd use proper regex
        if snake_case_count > camel_case_indicators {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::RustLike,
                confidence: 0.4,
                reason: "Snake_case naming convention detected".to_string(),
                location: None,
            });
        }
        
        evidence
    }
    
    /// Analyze delimiter patterns
    fn analyze_delimiters(source: &str) -> Vec<SyntaxEvidence> {
        let mut evidence = Vec::new();
        
        let brace_pairs = source.matches('{').count().min(source.matches('}').count());
        let semicolon_count = source.matches(';').count();
        let colon_count = source.matches(':').count();
        
        // Strong indicators for different styles
        if brace_pairs > 0 {
            if semicolon_count > brace_pairs {
                evidence.push(SyntaxEvidence {
                    style: crate::detection::SyntaxStyle::CLike,
                    confidence: 0.8,
                    reason: format!("C-like pattern: {} brace pairs with {} semicolons", 
                        brace_pairs, semicolon_count),
                    location: None,
                });
            } else {
                evidence.push(SyntaxEvidence {
                    style: crate::detection::SyntaxStyle::RustLike,
                    confidence: 0.7,
                    reason: format!("Rust-like pattern: {} brace pairs with few semicolons", brace_pairs),
                    location: None,
                });
            }
        }
        
        if colon_count > brace_pairs && source.contains("def ") {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::PythonLike,
                confidence: 0.9,
                reason: format!("Python-like pattern: {} colons with function definitions", colon_count),
                location: None,
            });
        }
        
        evidence
    }
    
    /// Analyze keyword patterns
    fn analyze_keywords(source: &str) -> Vec<SyntaxEvidence> {
        let mut evidence = Vec::new();
        
        // Python keywords
        let python_keywords = ["def ", "class ", "import ", "from ", "elif ", "lambda ", "yield "];
        let python_count: usize = python_keywords.iter()
            .map(|&kw| source.matches(kw).count())
            .sum();
        
        // Rust keywords
        let rust_keywords = ["fn ", "struct ", "enum ", "impl ", "trait ", "mod ", "use ", "pub "];
        let rust_count: usize = rust_keywords.iter()
            .map(|&kw| source.matches(kw).count())
            .sum();
        
        // C-like keywords
        let c_keywords = ["int ", "char ", "void ", "struct ", "typedef ", "#include", "malloc"];
        let c_count: usize = c_keywords.iter()
            .map(|&kw| source.matches(kw).count())
            .sum();
        
        if python_count > 0 {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::PythonLike,
                confidence: (python_count as f64 * 0.15).min(0.9),
                reason: format!("Python keywords found: {} instances", python_count),
                location: None,
            });
        }
        
        if rust_count > 0 {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::RustLike,
                confidence: (rust_count as f64 * 0.15).min(0.9),
                reason: format!("Rust keywords found: {} instances", rust_count),
                location: None,
            });
        }
        
        if c_count > 0 {
            evidence.push(SyntaxEvidence {
                style: crate::detection::SyntaxStyle::CLike,
                confidence: (c_count as f64 * 0.15).min(0.9),
                reason: format!("C-like keywords found: {} instances", c_count),
                location: None,
            });
        }
        
        evidence
    }
}

impl HeuristicWeight {
    /// Get the numeric weight value
    pub fn value(&self) -> f64 {
        match self {
            HeuristicWeight::Low(w) | HeuristicWeight::Medium(w) | HeuristicWeight::High(w) => *w,
        }
    }
}

impl Default for HeuristicEngine {
    fn default() -> Self {
        Self::new()
    }
} 