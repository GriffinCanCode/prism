//! Cohesion Improvement Suggestions
//!
//! This module generates actionable suggestions for improving code cohesion
//! based on analysis results and detected violations.

use crate::{CohesionResult, CohesionMetrics, CohesionViolation};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Suggestion engine for cohesion improvements
#[derive(Debug)]
pub struct SuggestionEngine {
    /// Suggestion generation configuration
    config: SuggestionConfig,
    
    /// Suggestion templates
    templates: SuggestionTemplates,
}

/// Configuration for suggestion generation
#[derive(Debug, Clone)]
pub struct SuggestionConfig {
    /// Maximum number of suggestions to generate
    pub max_suggestions: usize,
    
    /// Minimum impact threshold for suggestions
    pub min_impact_threshold: f64,
    
    /// Enable contextual suggestions
    pub enable_contextual: bool,
    
    /// Enable architectural suggestions
    pub enable_architectural: bool,
    
    /// Suggestion verbosity level
    pub verbosity: SuggestionVerbosity,
}

/// Suggestion verbosity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionVerbosity {
    /// Brief suggestions
    Brief,
    /// Detailed suggestions with rationale
    Detailed,
    /// Comprehensive suggestions with examples
    Comprehensive,
}

/// Cohesion improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionSuggestion {
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    
    /// Priority level
    pub priority: SuggestionPriority,
    
    /// Suggestion description
    pub description: String,
    
    /// Detailed rationale
    pub rationale: String,
    
    /// Location where suggestion applies
    pub location: Option<Span>,
    
    /// Estimated impact of implementing this suggestion
    pub estimated_impact: f64,
    
    /// Effort level required
    pub effort_level: EffortLevel,
    
    /// Related metrics that would improve
    pub related_metrics: Vec<String>,
    
    /// Example implementation (if available)
    pub example: Option<String>,
    
    /// Prerequisites for this suggestion
    pub prerequisites: Vec<String>,
}

/// Types of cohesion suggestions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Split module into multiple modules
    SplitModule,
    
    /// Merge related modules
    MergeModules,
    
    /// Reorganize sections within module
    ReorganizeSections,
    
    /// Improve naming consistency
    ImproveNaming,
    
    /// Add missing capability definition
    AddCapability,
    
    /// Reduce dependencies
    ReduceDependencies,
    
    /// Extract common functionality
    ExtractCommon,
    
    /// Add documentation
    AddDocumentation,
    
    /// Clarify responsibilities
    ClarifyResponsibilities,
    
    /// Improve type organization
    ImproveTypeOrganization,
    
    /// Optimize data flow
    OptimizeDataFlow,
    
    /// Architectural refactoring
    ArchitecturalRefactoring,
    
    /// Custom suggestion
    Custom(String),
}

/// Suggestion priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuggestionPriority {
    /// Critical - should be addressed immediately
    Critical,
    
    /// High - important improvement
    High,
    
    /// Medium - beneficial improvement
    Medium,
    
    /// Low - minor enhancement
    Low,
    
    /// Optional - nice to have
    Optional,
}

/// Effort level required for suggestion
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffortLevel {
    /// Minimal effort (< 1 hour)
    Minimal,
    
    /// Low effort (1-4 hours)
    Low,
    
    /// Medium effort (0.5-2 days)
    Medium,
    
    /// High effort (3-7 days)
    High,
    
    /// Very high effort (> 1 week)
    VeryHigh,
}

/// Suggestion templates for different scenarios
#[derive(Debug)]
struct SuggestionTemplates {
    /// Templates by suggestion type
    templates: HashMap<SuggestionType, SuggestionTemplate>,
}

/// Template for generating suggestions
#[derive(Debug, Clone)]
struct SuggestionTemplate {
    /// Description template
    pub description_template: String,
    
    /// Rationale template
    pub rationale_template: String,
    
    /// Default priority
    pub default_priority: SuggestionPriority,
    
    /// Default effort level
    pub default_effort: EffortLevel,
    
    /// Example template
    pub example_template: Option<String>,
}

impl SuggestionEngine {
    /// Create new suggestion engine
    pub fn new() -> Self {
        Self {
            config: SuggestionConfig::default(),
            templates: SuggestionTemplates::new(),
        }
    }
    
    /// Create suggestion engine with custom configuration
    pub fn with_config(config: SuggestionConfig) -> Self {
        Self {
            config,
            templates: SuggestionTemplates::new(),
        }
    }
    
    /// Generate suggestions for a complete program
    pub fn generate_program_suggestions(
        &self,
        program: &Program,
        metrics: &CohesionMetrics,
        violations: &[CohesionViolation],
    ) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Generate suggestions based on violations
        suggestions.extend(self.generate_violation_based_suggestions(violations)?);
        
        // Generate suggestions based on metrics
        suggestions.extend(self.generate_metric_based_suggestions(metrics)?);
        
        // Generate architectural suggestions if enabled
        if self.config.enable_architectural {
            suggestions.extend(self.generate_architectural_suggestions(program, metrics)?);
        }
        
        // Generate contextual suggestions if enabled
        if self.config.enable_contextual {
            suggestions.extend(self.generate_contextual_suggestions(program, metrics)?);
        }
        
        // Filter and prioritize suggestions
        self.filter_and_prioritize_suggestions(suggestions)
    }
    
    /// Generate suggestions for a single module
    pub fn generate_module_suggestions(
        &self,
        module_item: &AstNode<Item>,
        metrics: &CohesionMetrics,
        violations: &[CohesionViolation],
    ) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Item::Module(module_decl) = &module_item.kind {
            // Module-specific suggestions
            suggestions.extend(self.generate_module_specific_suggestions(module_decl, metrics)?);
            
            // Violation-based suggestions
            suggestions.extend(self.generate_violation_based_suggestions(violations)?);
        }
        
        self.filter_and_prioritize_suggestions(suggestions)
    }
    
    /// Generate suggestions based on violations
    fn generate_violation_based_suggestions(&self, violations: &[CohesionViolation]) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        for violation in violations {
            let suggestion = self.violation_to_suggestion(violation)?;
            suggestions.push(suggestion);
        }
        
        Ok(suggestions)
    }
    
    /// Convert violation to suggestion
    fn violation_to_suggestion(&self, violation: &CohesionViolation) -> CohesionResult<CohesionSuggestion> {
        let suggestion_type = match violation.violation_type {
            crate::ViolationType::LowCohesion => SuggestionType::ClarifyResponsibilities,
            crate::ViolationType::MultipleResponsibilities => SuggestionType::SplitModule,
            crate::ViolationType::InconsistentNaming => SuggestionType::ImproveNaming,
            crate::ViolationType::ExcessiveDependencies => SuggestionType::ReduceDependencies,
            crate::ViolationType::MissingCapability => SuggestionType::AddCapability,
            crate::ViolationType::PoorSectionOrganization => SuggestionType::ReorganizeSections,
            crate::ViolationType::GodModule => SuggestionType::SplitModule,
            crate::ViolationType::AnemicModule => SuggestionType::MergeModules,
            _ => SuggestionType::ClarifyResponsibilities,
        };
        
        let priority = match violation.severity {
            crate::ViolationSeverity::Error => SuggestionPriority::Critical,
            crate::ViolationSeverity::Warning => SuggestionPriority::High,
            crate::ViolationSeverity::Info => SuggestionPriority::Medium,
            crate::ViolationSeverity::Hint => SuggestionPriority::Low,
        };
        
        let template = self.templates.get_template(&suggestion_type);
        
        Ok(CohesionSuggestion {
            suggestion_type: suggestion_type.clone(),
            priority,
            description: violation.suggested_fix.clone(),
            rationale: violation.description.clone(),
            location: violation.location,
            estimated_impact: violation.impact_score,
            effort_level: self.estimate_effort_level(&suggestion_type, violation.impact_score),
            related_metrics: violation.related_metrics.clone(),
            example: template.map(|t| t.example_template.clone()).flatten(),
            prerequisites: Vec::new(),
        })
    }
    
    /// Generate suggestions based on metrics
    fn generate_metric_based_suggestions(&self, metrics: &CohesionMetrics) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Identify the weakest metrics and suggest improvements
        let weak_metrics = [
            ("type_cohesion", metrics.type_cohesion, SuggestionType::ImproveTypeOrganization),
            ("data_flow_cohesion", metrics.data_flow_cohesion, SuggestionType::OptimizeDataFlow),
            ("semantic_cohesion", metrics.semantic_cohesion, SuggestionType::ImproveNaming),
            ("business_cohesion", metrics.business_cohesion, SuggestionType::ClarifyResponsibilities),
            ("dependency_cohesion", metrics.dependency_cohesion, SuggestionType::ReduceDependencies),
        ];
        
        for (metric_name, score, suggestion_type) in weak_metrics {
            if score < 70.0 {
                let template = self.templates.get_template(&suggestion_type);
                
                suggestions.push(CohesionSuggestion {
                    suggestion_type: suggestion_type.clone(),
                    priority: if score < 50.0 { SuggestionPriority::High } else { SuggestionPriority::Medium },
                    description: format!("Improve {} (current: {:.1})", metric_name.replace('_', " "), score),
                    rationale: format!("The {} metric is below the recommended threshold", metric_name.replace('_', " ")),
                    location: None,
                    estimated_impact: (70.0 - score) / 100.0,
                    effort_level: self.estimate_effort_level(&suggestion_type, (70.0 - score) / 100.0),
                    related_metrics: vec![metric_name.to_string()],
                    example: template.and_then(|t| t.example_template.clone()),
                    prerequisites: Vec::new(),
                });
            }
        }
        
        Ok(suggestions)
    }
    
    /// Generate architectural suggestions
    fn generate_architectural_suggestions(&self, _program: &Program, metrics: &CohesionMetrics) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Overall architecture assessment
        if metrics.overall_score < 60.0 {
            suggestions.push(CohesionSuggestion {
                suggestion_type: SuggestionType::ArchitecturalRefactoring,
                priority: SuggestionPriority::High,
                description: "Consider architectural refactoring to improve overall cohesion".to_string(),
                rationale: format!("Overall cohesion score ({:.1}) indicates systemic architectural issues", metrics.overall_score),
                location: None,
                estimated_impact: 0.8,
                effort_level: EffortLevel::High,
                related_metrics: vec!["overall_score".to_string()],
                example: Some("Consider applying domain-driven design principles to reorganize modules around business capabilities".to_string()),
                prerequisites: vec!["Architectural review".to_string(), "Business domain analysis".to_string()],
            });
        }
        
        Ok(suggestions)
    }
    
    /// Generate contextual suggestions based on program structure
    fn generate_contextual_suggestions(&self, program: &Program, _metrics: &CohesionMetrics) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Count modules
        let module_count = program.items.iter()
            .filter(|item| matches!(item.kind, Item::Module(_)))
            .count();
        
        if module_count == 0 {
            suggestions.push(CohesionSuggestion {
                suggestion_type: SuggestionType::AddDocumentation,
                priority: SuggestionPriority::High,
                description: "No modules found - consider organizing code into modules".to_string(),
                rationale: "Modules provide better code organization and conceptual clarity".to_string(),
                location: None,
                estimated_impact: 0.9,
                effort_level: EffortLevel::Medium,
                related_metrics: vec!["overall_score".to_string()],
                example: Some("Create modules that represent business capabilities or technical concerns".to_string()),
                prerequisites: vec!["Analyze code organization needs".to_string()],
            });
        } else if module_count == 1 {
            suggestions.push(CohesionSuggestion {
                suggestion_type: SuggestionType::SplitModule,
                priority: SuggestionPriority::Medium,
                description: "Consider if the single module could be split for better organization".to_string(),
                rationale: "Multiple focused modules often provide better conceptual clarity than one large module".to_string(),
                location: None,
                estimated_impact: 0.6,
                effort_level: EffortLevel::Medium,
                related_metrics: vec!["business_cohesion".to_string()],
                example: None,
                prerequisites: vec!["Analyze module responsibilities".to_string()],
            });
        }
        
        Ok(suggestions)
    }
    
    /// Generate module-specific suggestions
    fn generate_module_specific_suggestions(&self, module_decl: &ModuleDecl, _metrics: &CohesionMetrics) -> CohesionResult<Vec<CohesionSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Check for missing elements
        if module_decl.description.is_none() {
            suggestions.push(CohesionSuggestion {
                suggestion_type: SuggestionType::AddDocumentation,
                priority: SuggestionPriority::Medium,
                description: format!("Add description to module {}", module_decl.name),
                rationale: "Module descriptions help communicate purpose and improve maintainability".to_string(),
                location: None,
                estimated_impact: 0.3,
                effort_level: EffortLevel::Minimal,
                related_metrics: vec!["business_cohesion".to_string()],
                example: Some("@description \"Manages user authentication and authorization\"".to_string()),
                prerequisites: Vec::new(),
            });
        }
        
        if module_decl.ai_context.is_none() {
            suggestions.push(CohesionSuggestion {
                suggestion_type: SuggestionType::AddDocumentation,
                priority: SuggestionPriority::Low,
                description: format!("Consider adding AI context to module {}", module_decl.name),
                rationale: "AI context helps automated tools better understand the module's purpose".to_string(),
                location: None,
                estimated_impact: 0.2,
                effort_level: EffortLevel::Low,
                related_metrics: vec!["business_cohesion".to_string()],
                example: Some("@aiContext \"Critical authentication module - handles user login security\"".to_string()),
                prerequisites: Vec::new(),
            });
        }
        
        Ok(suggestions)
    }
    
    /// Filter and prioritize suggestions
    fn filter_and_prioritize_suggestions(&self, mut suggestions: Vec<CohesionSuggestion>) -> CohesionResult<Vec<CohesionSuggestion>> {
        // Filter by impact threshold
        suggestions.retain(|s| s.estimated_impact >= self.config.min_impact_threshold);
        
        // Remove duplicates (simple deduplication by description)
        let mut seen_descriptions = std::collections::HashSet::new();
        suggestions.retain(|s| seen_descriptions.insert(s.description.clone()));
        
        // Sort by priority and impact
        suggestions.sort_by(|a, b| {
            match a.priority.cmp(&b.priority) {
                std::cmp::Ordering::Equal => b.estimated_impact.partial_cmp(&a.estimated_impact).unwrap_or(std::cmp::Ordering::Equal),
                other => other,
            }
        });
        
        // Limit to max suggestions
        suggestions.truncate(self.config.max_suggestions);
        
        Ok(suggestions)
    }
    
    /// Estimate effort level for a suggestion
    fn estimate_effort_level(&self, suggestion_type: &SuggestionType, impact: f64) -> EffortLevel {
        match suggestion_type {
            SuggestionType::AddDocumentation | SuggestionType::AddCapability => EffortLevel::Minimal,
            SuggestionType::ImproveNaming => EffortLevel::Low,
            SuggestionType::ReorganizeSections | SuggestionType::ReduceDependencies => EffortLevel::Medium,
            SuggestionType::SplitModule | SuggestionType::MergeModules => {
                if impact > 0.7 { EffortLevel::High } else { EffortLevel::Medium }
            },
            SuggestionType::ArchitecturalRefactoring => EffortLevel::VeryHigh,
            _ => EffortLevel::Medium,
        }
    }
}

impl SuggestionTemplates {
    /// Create new suggestion templates
    fn new() -> Self {
        let mut templates = HashMap::new();
        
        // Initialize default templates
        templates.insert(SuggestionType::SplitModule, SuggestionTemplate {
            description_template: "Split module into focused, single-responsibility modules".to_string(),
            rationale_template: "Large modules with multiple responsibilities are harder to maintain".to_string(),
            default_priority: SuggestionPriority::High,
            default_effort: EffortLevel::High,
            example_template: Some("Split UserManagement into UserAuthentication and UserProfile modules".to_string()),
        });
        
        templates.insert(SuggestionType::ImproveNaming, SuggestionTemplate {
            description_template: "Improve naming consistency throughout the module".to_string(),
            rationale_template: "Consistent naming improves code readability and conceptual clarity".to_string(),
            default_priority: SuggestionPriority::Medium,
            default_effort: EffortLevel::Low,
            example_template: Some("Use consistent prefixes: getUserById, updateUserProfile, deleteUserAccount".to_string()),
        });
        
        templates.insert(SuggestionType::AddCapability, SuggestionTemplate {
            description_template: "Add capability definition to clarify module purpose".to_string(),
            rationale_template: "Capability definitions help establish clear module boundaries".to_string(),
            default_priority: SuggestionPriority::Medium,
            default_effort: EffortLevel::Minimal,
            example_template: Some("@capability \"User Authentication\"".to_string()),
        });
        
        Self { templates }
    }
    
    /// Get template for suggestion type
    fn get_template(&self, suggestion_type: &SuggestionType) -> Option<&SuggestionTemplate> {
        self.templates.get(suggestion_type)
    }
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 20,
            min_impact_threshold: 0.1,
            enable_contextual: true,
            enable_architectural: true,
            verbosity: SuggestionVerbosity::Detailed,
        }
    }
}

impl Default for SuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
} 