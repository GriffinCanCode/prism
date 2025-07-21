//! Specialized validation checkers for documentation analysis.
//!
//! This module provides specialized checkers that validate specific
//! aspects of documentation such as JSDoc compatibility, AI context
//! completeness, and global consistency.

use crate::{DocumentationResult, DocumentationError};
use crate::extraction::{DocumentationElement, ExtractedDocumentation};
use crate::validation::{ValidationViolation, ViolationType, ViolationSeverity};
use prism_common::span::Span;
use std::collections::{HashMap, HashSet};

/// Collection of specialized validation checkers
#[derive(Debug)]
pub struct ValidationCheckers {
    /// JSDoc compatibility checker
    pub jsdoc_checker: JSDocCompatibilityChecker,
    /// AI context completeness checker
    pub ai_context_checker: AIContextChecker,
    /// Global consistency checker
    pub consistency_checker: ConsistencyChecker,
    /// Content quality checker
    pub content_checker: ContentQualityChecker,
    /// Business rule alignment checker
    pub business_rule_checker: BusinessRuleChecker,
}

/// Result from a checker validation
#[derive(Debug)]
pub struct CheckerResult {
    /// Validation violations found
    pub violations: Vec<ValidationViolation>,
    /// Warnings generated
    pub warnings: Vec<String>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
}

impl ValidationCheckers {
    /// Create new validation checkers
    pub fn new() -> Self {
        Self {
            jsdoc_checker: JSDocCompatibilityChecker::new(),
            ai_context_checker: AIContextChecker::new(),
            consistency_checker: ConsistencyChecker::new(),
            content_checker: ContentQualityChecker::new(),
            business_rule_checker: BusinessRuleChecker::new(),
        }
    }
}

/// Checker for JSDoc compatibility
#[derive(Debug)]
pub struct JSDocCompatibilityChecker {
    /// Supported JSDoc tags
    supported_tags: HashSet<String>,
}

impl JSDocCompatibilityChecker {
    pub fn new() -> Self {
        let mut supported_tags = HashSet::new();
        supported_tags.extend([
            "param", "returns", "throws", "example", "since", "deprecated",
            "author", "version", "see", "todo", "note", "warning", "internal",
            "override", "abstract", "static", "readonly", "async", "generator",
            "callback", "event", "fires", "listens", "mixes", "namespace"
        ].iter().map(|s| s.to_string()));
        
        Self { supported_tags }
    }
    
    /// Check JSDoc compatibility for an element
    pub fn check_element(&self, element: &DocumentationElement) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check if annotations use JSDoc-compatible tags
        for annotation in &element.annotations {
            if !self.supported_tags.contains(&annotation.name) {
                warnings.push(format!(
                    "Annotation '@{}' is not JSDoc compatible. Consider using standard JSDoc tags for better tool support.",
                    annotation.name
                ));
                
                // Suggest JSDoc alternatives
                let suggestion = match annotation.name.as_str() {
                    "responsibility" => Some("Consider adding a standard description comment in addition to @responsibility"),
                    "effects" => Some("Consider documenting side effects in the main description or @note"),
                    "capability" => Some("Consider documenting capabilities in @note or custom @requires"),
                    _ => None,
                };
                
                if let Some(sugg) = suggestion {
                    suggestions.push(sugg.to_string());
                }
            }
        }
        
        // Check for missing JSDoc essentials
        if matches!(element.element_type, crate::extraction::DocumentationElementType::Function) {
            if matches!(element.visibility, crate::extraction::ElementVisibility::Public) {
                let has_param = element.annotations.iter().any(|a| a.name == "param");
                let has_returns = element.annotations.iter().any(|a| a.name == "returns");
                
                if !has_param {
                    suggestions.push("Consider adding @param annotations for JSDoc compatibility".to_string());
                }
                
                if !has_returns {
                    suggestions.push("Consider adding @returns annotation for JSDoc compatibility".to_string());
                }
            }
        }
        
        Ok(CheckerResult {
            violations,
            warnings,
            suggestions,
        })
    }
}

/// Checker for AI context completeness
#[derive(Debug)]
pub struct AIContextChecker;

impl AIContextChecker {
    pub fn new() -> Self {
        Self
    }
    
    /// Check AI context completeness for an element
    pub fn check_element(&self, element: &DocumentationElement) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check if element has AI context information
        let has_ai_context = element.ai_context.is_some();
        
        if !has_ai_context && matches!(element.visibility, crate::extraction::ElementVisibility::Public) {
            suggestions.push("Consider adding AI context information to improve code comprehensibility".to_string());
        }
        
        // Check for business context in complex functions/types
        let is_complex = matches!(element.element_type, 
            crate::extraction::DocumentationElementType::Function | 
            crate::extraction::DocumentationElementType::Type
        );
        
        if is_complex && !self.has_business_context(element) {
            suggestions.push("Consider adding business context for better AI understanding".to_string());
        }
        
        // Check for usage examples in public APIs
        if matches!(element.visibility, crate::extraction::ElementVisibility::Public) {
            let has_examples = element.annotations.iter().any(|a| a.name == "example");
            if !has_examples {
                suggestions.push("Consider adding usage examples to improve AI comprehension".to_string());
            }
        }
        
        Ok(CheckerResult {
            violations,
            warnings: Vec::new(),
            suggestions,
        })
    }
    
    fn has_business_context(&self, element: &DocumentationElement) -> bool {
        // Check if element has business-related annotations or context
        element.annotations.iter().any(|a| {
            matches!(a.name.as_str(), "business" | "domain" | "capability" | "compliance")
        }) || element.ai_context.as_ref().map_or(false, |ctx| {
            ctx.business_context.is_some()
        })
    }
}

/// Checker for global documentation consistency
#[derive(Debug)]
pub struct ConsistencyChecker;

impl ConsistencyChecker {
    pub fn new() -> Self {
        Self
    }
    
    /// Check global consistency across all documentation
    pub fn check_global_consistency(&self, docs: &ExtractedDocumentation) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check for consistent annotation usage
        let annotation_usage = self.analyze_annotation_usage(&docs.elements);
        
        // Check for inconsistent responsibility patterns
        let responsibility_patterns = self.analyze_responsibility_patterns(&docs.elements);
        if responsibility_patterns.len() > 3 {
            warnings.push("Multiple responsibility statement patterns detected. Consider standardizing on one format.".to_string());
        }
        
        // Check for missing cross-references
        let cross_refs = self.analyze_cross_references(&docs.elements);
        if cross_refs.missing_refs > 0 {
            suggestions.push(format!(
                "Found {} potentially missing cross-references. Consider adding links between related elements.",
                cross_refs.missing_refs
            ));
        }
        
        // Check module-level consistency
        if let Some(module_doc) = &docs.module_documentation {
            let module_consistency = self.check_module_consistency(module_doc, &docs.elements)?;
            violations.extend(module_consistency.violations);
            warnings.extend(module_consistency.warnings);
            suggestions.extend(module_consistency.suggestions);
        }
        
        Ok(CheckerResult {
            violations,
            warnings,
            suggestions,
        })
    }
    
    fn analyze_annotation_usage(&self, elements: &[DocumentationElement]) -> HashMap<String, usize> {
        let mut usage = HashMap::new();
        for element in elements {
            for annotation in &element.annotations {
                *usage.entry(annotation.name.clone()).or_insert(0) += 1;
            }
        }
        usage
    }
    
    fn analyze_responsibility_patterns(&self, elements: &[DocumentationElement]) -> HashSet<String> {
        let mut patterns = HashSet::new();
        for element in elements {
            for annotation in &element.annotations {
                if annotation.name == "responsibility" {
                    if let Some(value) = &annotation.value {
                        // Extract pattern (first few words)
                        let words: Vec<&str> = value.split_whitespace().take(3).collect();
                        if !words.is_empty() {
                            patterns.insert(words.join(" "));
                        }
                    }
                }
            }
        }
        patterns
    }
    
    fn analyze_cross_references(&self, elements: &[DocumentationElement]) -> CrossReferenceAnalysis {
        let mut element_names = HashSet::new();
        let mut referenced_names = HashSet::new();
        
        // Collect element names
        for element in elements {
            element_names.insert(element.name.clone());
        }
        
        // Find references in documentation content
        for element in elements {
            if let Some(content) = &element.content {
                // Simple reference detection (could be enhanced with proper parsing)
                for name in &element_names {
                    if name != &element.name && content.contains(name) {
                        referenced_names.insert(name.clone());
                    }
                }
            }
        }
        
        CrossReferenceAnalysis {
            total_elements: element_names.len(),
            referenced_elements: referenced_names.len(),
            missing_refs: element_names.len().saturating_sub(referenced_names.len()),
        }
    }
    
    fn check_module_consistency(
        &self,
        module_doc: &crate::extraction::ModuleDocumentation,
        elements: &[DocumentationElement],
    ) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        
        // Check if module name matches declared name
        let module_elements: Vec<_> = elements.iter()
            .filter(|e| matches!(e.element_type, crate::extraction::DocumentationElementType::Module))
            .collect();
            
        if module_elements.len() > 1 {
            warnings.push("Multiple module declarations found. Consider consolidating.".to_string());
        }
        
        // Check responsibility alignment
        if let Some(module_responsibility) = &module_doc.responsibility {
            let element_responsibilities: Vec<String> = elements.iter()
                .filter_map(|e| {
                    e.annotations.iter()
                        .find(|a| a.name == "responsibility")
                        .and_then(|a| a.value.as_ref())
                        .cloned()
                })
                .collect();
                
            // Simple semantic check - could be enhanced
            let related_count = element_responsibilities.iter()
                .filter(|resp| self.are_semantically_related(module_responsibility, resp))
                .count();
                
            if related_count < element_responsibilities.len() / 2 {
                warnings.push("Some element responsibilities may not align with module responsibility".to_string());
            }
        }
        
        Ok(CheckerResult {
            violations,
            warnings,
            suggestions: Vec::new(),
        })
    }
    
    fn are_semantically_related(&self, module_resp: &str, element_resp: &str) -> bool {
        // Simple semantic relatedness check
        // In a full implementation, this could use NLP techniques
        let module_words: HashSet<_> = module_resp.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter out common words
            .collect();
            
        let element_words: HashSet<_> = element_resp.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
            
        let common_words = module_words.intersection(&element_words).count();
        common_words > 0 || module_words.len().min(element_words.len()) == 0
    }
}

#[derive(Debug)]
struct CrossReferenceAnalysis {
    total_elements: usize,
    referenced_elements: usize,
    missing_refs: usize,
}

/// Checker for content quality
#[derive(Debug)]
pub struct ContentQualityChecker;

impl ContentQualityChecker {
    pub fn new() -> Self {
        Self
    }
    
    /// Check content quality for an element
    pub fn check_element(&self, element: &DocumentationElement) -> DocumentationResult<CheckerResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check content length and quality
        if let Some(content) = &element.content {
            if content.trim().is_empty() {
                violations.push(ValidationViolation {
                    violation_type: ViolationType::ContentQualityIssue,
                    severity: ViolationSeverity::Warning,
                    message: "Documentation content is empty".to_string(),
                    location: element.location,
                    suggested_fix: Some("Add meaningful documentation content".to_string()),
                    rule_id: "CONTENT_EMPTY".to_string(),
                    context: HashMap::new(),
                });
            } else if content.len() < 10 {
                warnings.push("Documentation content is very brief. Consider adding more detail.".to_string());
            }
            
            // Check for common quality issues
            if content.to_lowercase().contains("todo") || content.to_lowercase().contains("fixme") {
                warnings.push("Documentation contains TODO/FIXME markers".to_string());
            }
            
            // Check for placeholder content
            if content.contains("...") || content.to_lowercase().contains("placeholder") {
                violations.push(ValidationViolation {
                    violation_type: ViolationType::ContentQualityIssue,
                    severity: ViolationSeverity::Warning,
                    message: "Documentation appears to contain placeholder content".to_string(),
                    location: element.location,
                    suggested_fix: Some("Replace placeholder content with actual documentation".to_string()),
                    rule_id: "CONTENT_PLACEHOLDER".to_string(),
                    context: HashMap::new(),
                });
            }
        }
        
        // Check annotation quality
        for annotation in &element.annotations {
            if let Some(value) = &annotation.value {
                if value.trim().is_empty() {
                    violations.push(ValidationViolation {
                        violation_type: ViolationType::InvalidAnnotationFormat,
                        severity: ViolationSeverity::Error,
                        message: format!("Annotation @{} has empty value", annotation.name),
                        location: annotation.location,
                        suggested_fix: Some("Provide a meaningful value for the annotation".to_string()),
                        rule_id: "ANNOTATION_EMPTY_VALUE".to_string(),
                        context: [("annotation".to_string(), annotation.name.clone())].into(),
                    });
                }
            }
        }
        
        Ok(CheckerResult {
            violations,
            warnings,
            suggestions,
        })
    }
}

/// Checker for business rule alignment
#[derive(Debug)]
pub struct BusinessRuleChecker;

impl BusinessRuleChecker {
    pub fn new() -> Self {
        Self
    }
    
    /// Check business rule alignment for an element
    pub fn check_element(&self, element: &DocumentationElement) -> DocumentationResult<CheckerResult> {
        let mut suggestions = Vec::new();
        
        // Check for business context indicators
        let has_business_annotations = element.annotations.iter()
            .any(|a| matches!(a.name.as_str(), "business" | "domain" | "capability" | "compliance"));
            
        if !has_business_annotations && self.seems_business_related(element) {
            suggestions.push("Consider adding business context annotations for this element".to_string());
        }
        
        // Check for consistency in business terminology
        if let Some(content) = &element.content {
            if self.contains_business_terms(content) && !has_business_annotations {
                suggestions.push("Documentation mentions business concepts. Consider adding @business or @domain annotations".to_string());
            }
        }
        
        Ok(CheckerResult {
            violations: Vec::new(),
            warnings: Vec::new(),
            suggestions,
        })
    }
    
    fn seems_business_related(&self, element: &DocumentationElement) -> bool {
        let business_indicators = [
            "user", "customer", "order", "payment", "account", "transaction",
            "business", "process", "workflow", "policy", "rule", "compliance"
        ];
        
        let name_lower = element.name.to_lowercase();
        business_indicators.iter().any(|indicator| name_lower.contains(indicator))
    }
    
    fn contains_business_terms(&self, content: &str) -> bool {
        let business_terms = [
            "business rule", "policy", "compliance", "regulation", "customer",
            "user", "transaction", "payment", "order", "account", "workflow"
        ];
        
        let content_lower = content.to_lowercase();
        business_terms.iter().any(|term| content_lower.contains(term))
    }
}

impl Default for ValidationCheckers {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckerResult {
    /// Create empty checker result
    pub fn empty() -> Self {
        Self {
            violations: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        }
    }
    
    /// Check if result has any violations
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }
    
    /// Check if result has any errors
    pub fn has_errors(&self) -> bool {
        self.violations.iter().any(|v| v.severity == ViolationSeverity::Error)
    }
} 