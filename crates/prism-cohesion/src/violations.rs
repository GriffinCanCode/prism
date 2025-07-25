//! Cohesion Violation Detection
//!
//! This module detects violations of cohesion principles and identifies
//! anti-patterns that reduce code maintainability and conceptual clarity.

use crate::{CohesionResult, CohesionMetrics, ViolationThresholds};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Violation detector for cohesion analysis
#[derive(Debug)]
pub struct ViolationDetector {
    /// Violation detection thresholds
    thresholds: ViolationThresholds,
    
    /// Custom violation rules
    custom_rules: Vec<ViolationRule>,
}

/// Detected cohesion violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    
    /// Severity level
    pub severity: ViolationSeverity,
    
    /// Violation description
    pub description: String,
    
    /// Location of the violation
    pub location: Option<Span>,
    
    /// Evidence for this violation
    pub evidence: Vec<String>,
    
    /// Suggested fix
    pub suggested_fix: String,
    
    /// Violation impact score
    pub impact_score: f64,
    
    /// Related metrics
    pub related_metrics: Vec<String>,
}

/// Types of cohesion violations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationType {
    /// Low overall cohesion
    LowCohesion,
    
    /// Multiple responsibilities in single module
    MultipleResponsibilities,
    
    /// Scattered related functionality
    ScatteredFunctionality,
    
    /// Poor naming consistency
    InconsistentNaming,
    
    /// Excessive dependencies
    ExcessiveDependencies,
    
    /// Missing capability definition
    MissingCapability,
    
    /// Inappropriate section organization
    PoorSectionOrganization,
    
    /// Circular dependencies
    CircularDependencies,
    
    /// God module (too large)
    GodModule,
    
    /// Anemic module (too small)
    AnemicModule,
    
    /// Custom violation type
    Custom(String),
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Critical violation that must be fixed
    Error,
    
    /// Important violation that should be fixed
    Warning,
    
    /// Minor issue or improvement opportunity
    Info,
    
    /// Suggestion for enhancement
    Hint,
}

/// Custom violation rule
#[derive(Debug, Clone)]
pub struct ViolationRule {
    /// Rule name
    pub name: String,
    
    /// Rule description
    pub description: String,
    
    /// Rule evaluation function
    pub evaluate: fn(&ModuleDecl, &CohesionMetrics) -> Option<CohesionViolation>,
    
    /// Rule severity
    pub severity: ViolationSeverity,
}

impl ViolationDetector {
    /// Create new violation detector
    pub fn new(thresholds: ViolationThresholds) -> Self {
        Self {
            thresholds,
            custom_rules: Vec::new(),
        }
    }
    
    /// Detect violations in a complete program
    pub fn detect_program_violations(
        &self, 
        program: &Program, 
        metrics: &CohesionMetrics
    ) -> CohesionResult<Vec<CohesionViolation>> {
        let mut violations = Vec::new();
        
        // Extract modules
        let modules: Vec<_> = program.items.iter()
            .filter_map(|item| {
                if let Item::Module(module_decl) = &item.kind {
                    Some((item, module_decl))
                } else {
                    None
                }
            })
            .collect();
        
        // Check overall program violations
        violations.extend(self.detect_overall_violations(metrics)?);
        
        // Check individual module violations
        for (module_item, module_decl) in modules {
            let module_violations = self.detect_module_violations(module_item, metrics)?;
            violations.extend(module_violations);
        }
        
        // Sort by severity and impact
        violations.sort_by(|a, b| {
            match a.severity.cmp(&b.severity) {
                std::cmp::Ordering::Equal => b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal),
                other => other,
            }
        });
        
        Ok(violations)
    }
    
    /// Detect violations in a single module
    pub fn detect_module_violations(
        &self, 
        module_item: &AstNode<Item>, 
        metrics: &CohesionMetrics
    ) -> CohesionResult<Vec<CohesionViolation>> {
        let mut violations = Vec::new();
        
        if let Item::Module(module_decl) = &module_item.kind {
            // Check all built-in violation types
            violations.extend(self.check_low_cohesion(module_decl, metrics));
            violations.extend(self.check_multiple_responsibilities(module_decl));
            violations.extend(self.check_inconsistent_naming(module_decl));
            violations.extend(self.check_excessive_dependencies(module_decl));
            violations.extend(self.check_missing_capability(module_decl));
            violations.extend(self.check_poor_section_organization(module_decl));
            violations.extend(self.check_module_size(module_decl));
            
            // Check custom rules
            for rule in &self.custom_rules {
                if let Some(violation) = (rule.evaluate)(module_decl, metrics) {
                    violations.push(violation);
                }
            }
        }
        
        Ok(violations)
    }
    
    /// Detect overall program-level violations
    fn detect_overall_violations(&self, metrics: &CohesionMetrics) -> CohesionResult<Vec<CohesionViolation>> {
        let mut violations = Vec::new();
        
        // Check overall cohesion score
        if metrics.overall_score < self.thresholds.error_threshold {
            violations.push(CohesionViolation {
                violation_type: ViolationType::LowCohesion,
                severity: ViolationSeverity::Error,
                description: format!("Overall cohesion score ({:.1}) is below error threshold ({:.1})", 
                                   metrics.overall_score, self.thresholds.error_threshold),
                location: None,
                evidence: vec![
                    format!("Type cohesion: {:.1}", metrics.type_cohesion),
                    format!("Data flow cohesion: {:.1}", metrics.data_flow_cohesion),
                    format!("Semantic cohesion: {:.1}", metrics.semantic_cohesion),
                    format!("Business cohesion: {:.1}", metrics.business_cohesion),
                    format!("Dependency cohesion: {:.1}", metrics.dependency_cohesion),
                ],
                suggested_fix: "Review module organization and split or merge modules as needed".to_string(),
                impact_score: (self.thresholds.error_threshold - metrics.overall_score) / 100.0,
                related_metrics: vec!["overall_score".to_string()],
            });
        } else if metrics.overall_score < self.thresholds.warning_threshold {
            violations.push(CohesionViolation {
                violation_type: ViolationType::LowCohesion,
                severity: ViolationSeverity::Warning,
                description: format!("Overall cohesion score ({:.1}) is below warning threshold ({:.1})", 
                                   metrics.overall_score, self.thresholds.warning_threshold),
                location: None,
                evidence: vec![format!("Overall score: {:.1}", metrics.overall_score)],
                suggested_fix: "Consider improving module organization".to_string(),
                impact_score: (self.thresholds.warning_threshold - metrics.overall_score) / 100.0,
                related_metrics: vec!["overall_score".to_string()],
            });
        }
        
        Ok(violations)
    }
    
    /// Check for low cohesion violations
    fn check_low_cohesion(&self, module_decl: &ModuleDecl, metrics: &CohesionMetrics) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        // Check individual metric violations
        let metric_checks = [
            ("type_cohesion", metrics.type_cohesion, "Type cohesion"),
            ("data_flow_cohesion", metrics.data_flow_cohesion, "Data flow cohesion"),
            ("semantic_cohesion", metrics.semantic_cohesion, "Semantic cohesion"),
            ("business_cohesion", metrics.business_cohesion, "Business cohesion"),
            ("dependency_cohesion", metrics.dependency_cohesion, "Dependency cohesion"),
        ];
        
        for (metric_name, score, description) in metric_checks {
            if score < self.thresholds.error_threshold {
                violations.push(CohesionViolation {
                    violation_type: ViolationType::LowCohesion,
                    severity: ViolationSeverity::Error,
                    description: format!("{} is critically low ({:.1}) in module {}", 
                                       description, score, module_decl.name),
                    location: None,
                    evidence: vec![format!("{}: {:.1}", description, score)],
                    suggested_fix: self.get_metric_specific_suggestion(metric_name),
                    impact_score: (self.thresholds.error_threshold - score) / 100.0,
                    related_metrics: vec![metric_name.to_string()],
                });
            } else if score < self.thresholds.warning_threshold {
                violations.push(CohesionViolation {
                    violation_type: ViolationType::LowCohesion,
                    severity: ViolationSeverity::Warning,
                    description: format!("{} is low ({:.1}) in module {}", 
                                       description, score, module_decl.name),
                    location: None,
                    evidence: vec![format!("{}: {:.1}", description, score)],
                    suggested_fix: self.get_metric_specific_suggestion(metric_name),
                    impact_score: (self.thresholds.warning_threshold - score) / 100.0,
                    related_metrics: vec![metric_name.to_string()],
                });
            }
        }
        
        violations
    }
    
    /// Check for multiple responsibilities
    fn check_multiple_responsibilities(&self, module_decl: &ModuleDecl) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        // Heuristic: if module has many different section types and no clear capability, 
        // it might have multiple responsibilities
        let section_types: std::collections::HashSet<_> = module_decl.sections.iter()
            .map(|s| std::mem::discriminant(&s.kind.kind))
            .collect();
        
        if section_types.len() > 5 && module_decl.capability.is_none() {
            violations.push(CohesionViolation {
                violation_type: ViolationType::MultipleResponsibilities,
                severity: ViolationSeverity::Warning,
                description: format!("Module {} may have multiple responsibilities", module_decl.name),
                location: None,
                evidence: vec![
                    format!("Has {} different section types", section_types.len()),
                    "No capability definition".to_string(),
                ],
                suggested_fix: "Define a clear capability and split into focused modules".to_string(),
                impact_score: 0.6,
                related_metrics: vec!["business_cohesion".to_string()],
            });
        }
        
        violations
    }
    
    /// Check for inconsistent naming
    fn check_inconsistent_naming(&self, module_decl: &ModuleDecl) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        // TODO: Implement more sophisticated naming analysis
        // For now, just check if module name doesn't match capability
        if let Some(capability) = &module_decl.capability {
            let name_str = module_decl.name.to_string();
            let similarity = strsim::jaro_winkler(&name_str, capability);
            
            if similarity < 0.5 {
                violations.push(CohesionViolation {
                    violation_type: ViolationType::InconsistentNaming,
                    severity: ViolationSeverity::Info,
                    description: format!("Module name '{}' doesn't match capability '{}'", name_str, capability),
                    location: None,
                    evidence: vec![format!("Name similarity: {:.2}", similarity)],
                    suggested_fix: "Consider renaming module to match its capability".to_string(),
                    impact_score: 0.3,
                    related_metrics: vec!["semantic_cohesion".to_string()],
                });
            }
        }
        
        violations
    }
    
    /// Check for excessive dependencies
    fn check_excessive_dependencies(&self, module_decl: &ModuleDecl) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        let dependency_count = module_decl.dependencies.len();
        
        if dependency_count > 10 {
            violations.push(CohesionViolation {
                violation_type: ViolationType::ExcessiveDependencies,
                severity: ViolationSeverity::Warning,
                description: format!("Module {} has too many dependencies ({})", 
                                   module_decl.name, dependency_count),
                location: None,
                evidence: vec![format!("Dependency count: {}", dependency_count)],
                suggested_fix: "Consider reducing dependencies or splitting the module".to_string(),
                impact_score: (dependency_count as f64 - 10.0) / 20.0,
                related_metrics: vec!["dependency_cohesion".to_string()],
            });
        } else if dependency_count > 7 {
            violations.push(CohesionViolation {
                violation_type: ViolationType::ExcessiveDependencies,
                severity: ViolationSeverity::Info,
                description: format!("Module {} has many dependencies ({})", 
                                   module_decl.name, dependency_count),
                location: None,
                evidence: vec![format!("Dependency count: {}", dependency_count)],
                suggested_fix: "Review dependencies and consider if all are necessary".to_string(),
                impact_score: (dependency_count as f64 - 7.0) / 20.0,
                related_metrics: vec!["dependency_cohesion".to_string()],
            });
        }
        
        violations
    }
    
    /// Check for missing capability definition
    fn check_missing_capability(&self, module_decl: &ModuleDecl) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        if module_decl.capability.is_none() {
            violations.push(CohesionViolation {
                violation_type: ViolationType::MissingCapability,
                severity: ViolationSeverity::Info,
                description: format!("Module {} lacks capability definition", module_decl.name),
                location: None,
                evidence: vec!["No @capability annotation".to_string()],
                suggested_fix: "Add @capability annotation to clarify module purpose".to_string(),
                impact_score: 0.4,
                related_metrics: vec!["business_cohesion".to_string()],
            });
        }
        
        violations
    }
    
    /// Check for poor section organization
    fn check_poor_section_organization(&self, module_decl: &ModuleDecl) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        if module_decl.sections.is_empty() {
            violations.push(CohesionViolation {
                violation_type: ViolationType::PoorSectionOrganization,
                severity: ViolationSeverity::Warning,
                description: format!("Module {} has no sections", module_decl.name),
                location: None,
                evidence: vec!["Zero sections defined".to_string()],
                suggested_fix: "Organize module content into logical sections".to_string(),
                impact_score: 0.5,
                related_metrics: vec!["overall_score".to_string()],
            });
        }
        
        // TODO: Check for logical section order and completeness
        
        violations
    }
    
    /// Check for module size issues
    fn check_module_size(&self, module_decl: &ModuleDecl) -> Vec<CohesionViolation> {
        let mut violations = Vec::new();
        
        let section_count = module_decl.sections.len();
        
        // God module check
        if section_count > 10 {
            violations.push(CohesionViolation {
                violation_type: ViolationType::GodModule,
                severity: ViolationSeverity::Warning,
                description: format!("Module {} is very large ({} sections)", 
                                   module_decl.name, section_count),
                location: None,
                evidence: vec![format!("Section count: {}", section_count)],
                suggested_fix: "Consider splitting into smaller, focused modules".to_string(),
                impact_score: (section_count as f64 - 10.0) / 10.0,
                related_metrics: vec!["overall_score".to_string()],
            });
        }
        
        // Anemic module check (only for modules with capability but very few sections)
        if module_decl.capability.is_some() && section_count < 2 {
            violations.push(CohesionViolation {
                violation_type: ViolationType::AnemicModule,
                severity: ViolationSeverity::Info,
                description: format!("Module {} may be too small ({} sections)", 
                                   module_decl.name, section_count),
                location: None,
                evidence: vec![
                    format!("Section count: {}", section_count),
                    "Has capability but minimal content".to_string(),
                ],
                suggested_fix: "Consider merging with related modules or adding missing functionality".to_string(),
                impact_score: 0.3,
                related_metrics: vec!["overall_score".to_string()],
            });
        }
        
        violations
    }
    
    /// Get metric-specific suggestions
    fn get_metric_specific_suggestion(&self, metric_name: &str) -> String {
        match metric_name {
            "type_cohesion" => "Review type relationships and consider grouping related types together".to_string(),
            "data_flow_cohesion" => "Improve data flow by organizing functions in logical order".to_string(),
            "semantic_cohesion" => "Use consistent naming conventions throughout the module".to_string(),
            "business_cohesion" => "Clarify the module's business purpose and focus on single responsibility".to_string(),
            "dependency_cohesion" => "Reduce or better organize external dependencies".to_string(),
            _ => "Review module organization and improve conceptual cohesion".to_string(),
        }
    }
    
    /// Add custom violation rule
    pub fn add_custom_rule(&mut self, rule: ViolationRule) {
        self.custom_rules.push(rule);
    }
    
    /// Get violation statistics
    pub fn get_violation_stats(&self, violations: &[CohesionViolation]) -> ViolationStats {
        let mut stats = ViolationStats::default();
        
        for violation in violations {
            match violation.severity {
                ViolationSeverity::Error => stats.error_count += 1,
                ViolationSeverity::Warning => stats.warning_count += 1,
                ViolationSeverity::Info => stats.info_count += 1,
                ViolationSeverity::Hint => stats.hint_count += 1,
            }
            
            *stats.violation_types.entry(violation.violation_type.clone()).or_insert(0) += 1;
            stats.total_impact_score += violation.impact_score;
        }
        
        stats.total_count = violations.len();
        stats
    }
}

/// Violation statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ViolationStats {
    /// Total number of violations
    pub total_count: usize,
    
    /// Number of errors
    pub error_count: usize,
    
    /// Number of warnings
    pub warning_count: usize,
    
    /// Number of info messages
    pub info_count: usize,
    
    /// Number of hints
    pub hint_count: usize,
    
    /// Total impact score
    pub total_impact_score: f64,
    
    /// Violation types and their counts
    pub violation_types: HashMap<ViolationType, usize>,
}

impl Default for ViolationDetector {
    fn default() -> Self {
        Self::new(ViolationThresholds::default())
    }
} 