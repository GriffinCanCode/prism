//! Business Cohesion Analysis
//!
//! This module focuses on analyzing business cohesion by examining
//! capabilities, business focus, and domain-specific organization.

use crate::CohesionResult;
use prism_ast::{AstNode, Item, ModuleDecl};

/// Specialized analyzer for business cohesion
#[derive(Debug)]
pub struct BusinessCohesionAnalyzer {
    /// Domain knowledge for business analysis
    domain_patterns: Vec<BusinessPattern>,
}

impl BusinessCohesionAnalyzer {
    /// Create a new business cohesion analyzer
    pub fn new() -> Self {
        Self {
            domain_patterns: Self::initialize_business_patterns(),
        }
    }
    
    /// Analyze business cohesion across multiple modules
    pub fn analyze_program(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> CohesionResult<f64> {
        let mut total_focus = 0.0;
        let mut module_count = 0;
        
        for (_, module_decl) in modules {
            let business_focus = self.calculate_business_focus(module_decl)?;
            total_focus += business_focus;
            module_count += 1;
        }
        
        Ok(if module_count > 0 { total_focus / module_count as f64 } else { 0.0 })
    }
    
    /// Analyze business cohesion within a single module
    pub fn analyze_module(&self, _module_item: &AstNode<Item>, module_decl: &ModuleDecl) -> CohesionResult<f64> {
        self.calculate_business_focus(module_decl)
    }
    
    /// Calculate business focus for a module
    fn calculate_business_focus(&self, module_decl: &ModuleDecl) -> CohesionResult<f64> {
        let mut score: f64 = 60.0; // Base score
        
        // Capability analysis (major factor)
        if let Some(ref capability) = module_decl.capability {
            score += 20.0; // Bonus for having capability
            
            // Analyze capability clarity
            let capability_score = self.analyze_capability_clarity(capability);
            score += capability_score;
        } else {
            score -= 15.0; // Penalty for missing capability
        }
        
        // Documentation analysis
        if let Some(ref description) = module_decl.description {
            score += 15.0; // Bonus for documentation
            
            // Analyze description quality
            let description_score = self.analyze_description_quality(description);
            score += description_score;
        } else {
            score -= 10.0; // Penalty for missing description
        }
        
        // Structural focus analysis
        let structure_score = self.analyze_structural_focus(module_decl);
        score += structure_score;
        
        // Business domain consistency
        let domain_score = self.analyze_domain_consistency(module_decl);
        score += domain_score;
        
        // AI context analysis (if available)
        if let Some(ref ai_context) = module_decl.ai_context {
            let ai_score = self.analyze_ai_context(ai_context);
            score += ai_score;
        }
        
        Ok(score.clamp(0.0, 100.0))
    }
    
    /// Analyze the clarity and specificity of a capability statement
    fn analyze_capability_clarity(&self, capability: &str) -> f64 {
        let mut clarity_score = 0.0;
        
        // Length analysis - not too short, not too long
        let word_count = capability.split_whitespace().count();
        match word_count {
            1..=2 => clarity_score -= 5.0, // Too brief
            3..=8 => clarity_score += 10.0, // Good length
            9..=15 => clarity_score += 5.0, // Acceptable length
            _ => clarity_score -= 3.0, // Too verbose
        }
        
        // Check for action verbs (indicates clear capability)
        let action_verbs = [
            "manages", "handles", "processes", "provides", "enables",
            "coordinates", "validates", "transforms", "generates", "monitors"
        ];
        
        if action_verbs.iter().any(|verb| capability.to_lowercase().contains(verb)) {
            clarity_score += 8.0;
        }
        
        // Check for domain-specific terminology
        for pattern in &self.domain_patterns {
            if pattern.keywords.iter().any(|keyword| capability.to_lowercase().contains(keyword)) {
                clarity_score += 5.0;
                break;
            }
        }
        
        // Check for business value indicators
        let value_indicators = ["user", "customer", "business", "data", "security", "performance"];
        if value_indicators.iter().any(|indicator| capability.to_lowercase().contains(indicator)) {
            clarity_score += 3.0;
        }
        
        clarity_score.min(15.0) // Cap the bonus
    }
    
    /// Analyze the quality of module description
    fn analyze_description_quality(&self, description: &str) -> f64 {
        let mut quality_score = 0.0;
        
        // Basic quality indicators
        if description.len() > 20 {
            quality_score += 3.0; // Sufficient detail
        }
        
        if description.contains('.') {
            quality_score += 2.0; // Proper sentences
        }
        
        // Check for technical depth
        let technical_terms = [
            "implementation", "algorithm", "protocol", "interface", "api",
            "service", "component", "architecture", "pattern", "design"
        ];
        
        if technical_terms.iter().any(|term| description.to_lowercase().contains(term)) {
            quality_score += 3.0;
        }
        
        // Check for business context
        let business_terms = [
            "requirement", "business", "user", "customer", "workflow",
            "process", "operation", "functionality", "feature", "capability"
        ];
        
        if business_terms.iter().any(|term| description.to_lowercase().contains(term)) {
            quality_score += 4.0;
        }
        
        quality_score.min(10.0) // Cap the bonus
    }
    
    /// Analyze structural focus of the module
    fn analyze_structural_focus(&self, module_decl: &ModuleDecl) -> f64 {
        let mut focus_score = 0.0;
        
        let section_count = module_decl.sections.len();
        
        // Prefer focused modules (not too many sections)
        match section_count {
            0 => focus_score -= 20.0, // Empty module
            1..=3 => focus_score += 10.0, // Well focused
            4..=6 => focus_score += 5.0, // Acceptable focus
            7..=10 => focus_score += 0.0, // Neutral
            _ => focus_score -= 10.0, // Too many responsibilities
        }
        
        // Analyze section organization
        let has_interface = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Interface));
        let has_types = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Types));
        let has_config = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Config));
        
        // Bonus for good architectural patterns
        if has_interface {
            focus_score += 5.0; // Clear public API
        }
        
        if has_types && has_interface {
            focus_score += 3.0; // Type-driven design
        }
        
        if has_config {
            focus_score += 2.0; // Configurable design
        }
        
        focus_score
    }
    
    /// Analyze consistency within business domain
    fn analyze_domain_consistency(&self, module_decl: &ModuleDecl) -> f64 {
        let module_name = module_decl.name.to_string().to_lowercase();
        
        // Determine likely business domain
        let detected_domain = self.detect_business_domain(&module_name, module_decl);
        
        if detected_domain.is_none() {
            return 0.0; // No clear domain detected
        }
        
        let domain = detected_domain.unwrap();
        let mut consistency_score = 5.0; // Base bonus for having a domain
        
        // Check if sections align with the domain
        let domain_alignment = self.calculate_domain_alignment(module_decl, &domain);
        consistency_score += domain_alignment;
        
        // Check naming consistency with domain
        let naming_alignment = self.calculate_naming_alignment(&module_name, &domain);
        consistency_score += naming_alignment;
        
        consistency_score.min(15.0) // Cap the bonus
    }
    
    /// Analyze AI context for business insights
    fn analyze_ai_context(&self, _ai_context: &prism_ast::AiContext) -> f64 {
        // AI context indicates thoughtful design
        5.0 // Base bonus for having AI context
        
        // TODO: Could analyze AI context content for business relevance
        // For now, just give a bonus for having it
    }
    
    /// Detect the business domain of a module
    fn detect_business_domain(&self, module_name: &str, module_decl: &ModuleDecl) -> Option<&BusinessPattern> {
        // Check patterns against module name and capability
        for pattern in &self.domain_patterns {
            if pattern.keywords.iter().any(|keyword| module_name.contains(keyword)) {
                return Some(pattern);
            }
            
            if let Some(ref capability) = module_decl.capability {
                if pattern.keywords.iter().any(|keyword| capability.to_lowercase().contains(keyword)) {
                    return Some(pattern);
                }
            }
        }
        
        None
    }
    
    /// Calculate alignment of sections with business domain
    fn calculate_domain_alignment(&self, module_decl: &ModuleDecl, domain: &BusinessPattern) -> f64 {
        let mut alignment_score = 0.0;
        
        // Check if sections match domain expectations
        for expected_section in &domain.expected_sections {
            let has_section = module_decl.sections.iter()
                .any(|s| self.section_matches_expectation(s, expected_section));
            
            if has_section {
                alignment_score += 2.0;
            }
        }
        
        alignment_score.min(8.0) // Cap the bonus
    }
    
    /// Calculate naming alignment with business domain
    fn calculate_naming_alignment(&self, module_name: &str, domain: &BusinessPattern) -> f64 {
        // Simple heuristic: if module name contains domain keywords
        if domain.keywords.iter().any(|keyword| module_name.contains(keyword)) {
            3.0
        } else {
            0.0
        }
    }
    
    /// Check if a section matches domain expectations
    fn section_matches_expectation(&self, _section: &AstNode<prism_ast::SectionDecl>, expected: &str) -> bool {
        // Simple matching for now
        // In a real implementation, this would be more sophisticated
        expected == "any" // Placeholder
    }
    
    /// Initialize business domain patterns
    fn initialize_business_patterns() -> Vec<BusinessPattern> {
        vec![
            BusinessPattern {
                name: "User Management".to_string(),
                keywords: vec!["user", "account", "profile", "authentication", "auth"],
                expected_sections: vec!["types", "interface", "internal"],
            },
            BusinessPattern {
                name: "Data Processing".to_string(),
                keywords: vec!["data", "process", "transform", "parse", "format"],
                expected_sections: vec!["types", "interface", "internal"],
            },
            BusinessPattern {
                name: "Business Logic".to_string(),
                keywords: vec!["business", "logic", "rule", "workflow", "operation"],
                expected_sections: vec!["types", "interface", "business"],
            },
            BusinessPattern {
                name: "Infrastructure".to_string(),
                keywords: vec!["config", "setup", "infrastructure", "system", "platform"],
                expected_sections: vec!["config", "interface", "internal"],
            },
            BusinessPattern {
                name: "API/Service".to_string(),
                keywords: vec!["api", "service", "endpoint", "handler", "controller"],
                expected_sections: vec!["interface", "types", "internal"],
            },
            BusinessPattern {
                name: "Security".to_string(),
                keywords: vec!["security", "encryption", "validation", "authorization", "permission"],
                expected_sections: vec!["types", "interface", "security"],
            },
        ]
    }
}

/// Business domain pattern for analysis
#[derive(Debug, Clone)]
struct BusinessPattern {
    /// Pattern name
    name: String,
    /// Keywords that identify this domain
    keywords: Vec<&'static str>,
    /// Expected section types for this domain
    expected_sections: Vec<&'static str>,
} 