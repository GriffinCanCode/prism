//! Confidence Calculation for Cohesion Analysis
//!
//! This module provides sophisticated confidence calculation that considers
//! multiple factors to determine the reliability of cohesion analysis results.

use crate::{CohesionResult, CohesionMetrics};
use prism_ast::{AstNode, Item, ModuleDecl};
use serde::{Serialize, Deserialize};

/// Confidence calculator for cohesion analysis
#[derive(Debug)]
pub struct ConfidenceCalculator {
    /// Configuration for confidence calculation
    config: ConfidenceConfig,
}

/// Configuration for confidence calculation
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Minimum data points needed for high confidence
    pub min_data_points: usize,
    /// Weight for data quality factor
    pub data_quality_weight: f64,
    /// Weight for analysis depth factor
    pub analysis_depth_weight: f64,
    /// Weight for consistency factor
    pub consistency_weight: f64,
    /// Weight for completeness factor
    pub completeness_weight: f64,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            min_data_points: 5,
            data_quality_weight: 0.3,
            analysis_depth_weight: 0.25,
            consistency_weight: 0.25,
            completeness_weight: 0.2,
        }
    }
}

/// Detailed confidence breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBreakdown {
    /// Overall confidence score (0.0 to 1.0)
    pub overall_confidence: f64,
    
    /// Individual confidence factors
    pub data_quality: f64,
    pub analysis_depth: f64,
    pub consistency: f64,
    pub completeness: f64,
    
    /// Confidence level classification
    pub confidence_level: ConfidenceLevel,
    
    /// Factors contributing to confidence
    pub confidence_factors: Vec<String>,
    
    /// Factors reducing confidence
    pub uncertainty_factors: Vec<String>,
    
    /// Recommendations for improving confidence
    pub improvement_recommendations: Vec<String>,
}

/// Confidence level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Very high confidence (>= 0.9)
    VeryHigh,
    /// High confidence (>= 0.8)
    High,
    /// Medium confidence (>= 0.6)
    Medium,
    /// Low confidence (>= 0.4)
    Low,
    /// Very low confidence (< 0.4)
    VeryLow,
}

impl ConfidenceLevel {
    /// Create confidence level from score
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 0.9 => ConfidenceLevel::VeryHigh,
            s if s >= 0.8 => ConfidenceLevel::High,
            s if s >= 0.6 => ConfidenceLevel::Medium,
            s if s >= 0.4 => ConfidenceLevel::Low,
            _ => ConfidenceLevel::VeryLow,
        }
    }
    
    /// Get description of confidence level
    pub fn description(&self) -> &'static str {
        match self {
            ConfidenceLevel::VeryHigh => "Very high confidence - results are highly reliable",
            ConfidenceLevel::High => "High confidence - results are reliable",
            ConfidenceLevel::Medium => "Medium confidence - results are generally reliable",
            ConfidenceLevel::Low => "Low confidence - results should be interpreted cautiously",
            ConfidenceLevel::VeryLow => "Very low confidence - results may be unreliable",
        }
    }
}

impl ConfidenceCalculator {
    /// Create a new confidence calculator
    pub fn new() -> Self {
        Self {
            config: ConfidenceConfig::default(),
        }
    }
    
    /// Create confidence calculator with custom configuration
    pub fn with_config(config: ConfidenceConfig) -> Self {
        Self { config }
    }
    
    /// Calculate confidence for program-level analysis
    pub fn calculate_program_confidence(
        &self,
        modules: &[(&AstNode<Item>, &ModuleDecl)],
        metrics: &CohesionMetrics,
    ) -> CohesionResult<ConfidenceBreakdown> {
        // Calculate individual confidence factors
        let data_quality = self.calculate_data_quality_confidence(modules);
        let analysis_depth = self.calculate_analysis_depth_confidence(modules, metrics);
        let consistency = self.calculate_consistency_confidence(modules, metrics);
        let completeness = self.calculate_completeness_confidence(modules);
        
        // Calculate weighted overall confidence
        let overall_confidence = (
            data_quality * self.config.data_quality_weight +
            analysis_depth * self.config.analysis_depth_weight +
            consistency * self.config.consistency_weight +
            completeness * self.config.completeness_weight
        ).clamp(0.0, 1.0);
        
        let confidence_level = ConfidenceLevel::from_score(overall_confidence);
        
        // Generate insights
        let confidence_factors = self.identify_confidence_factors(modules, metrics);
        let uncertainty_factors = self.identify_uncertainty_factors(modules, metrics);
        let improvement_recommendations = self.generate_improvement_recommendations(
            &confidence_factors, &uncertainty_factors, confidence_level
        );
        
        Ok(ConfidenceBreakdown {
            overall_confidence,
            data_quality,
            analysis_depth,
            consistency,
            completeness,
            confidence_level,
            confidence_factors,
            uncertainty_factors,
            improvement_recommendations,
        })
    }
    
    /// Calculate confidence for module-level analysis
    pub fn calculate_module_confidence(
        &self,
        module_decl: &ModuleDecl,
        metrics: &CohesionMetrics,
    ) -> CohesionResult<ConfidenceBreakdown> {
        // Simpler calculation for single module
        let data_quality = self.calculate_module_data_quality(module_decl);
        let analysis_depth = self.calculate_module_analysis_depth(module_decl);
        let consistency = self.calculate_module_consistency(module_decl, metrics);
        let completeness = self.calculate_module_completeness(module_decl);
        
        let overall_confidence = (
            data_quality * self.config.data_quality_weight +
            analysis_depth * self.config.analysis_depth_weight +
            consistency * self.config.consistency_weight +
            completeness * self.config.completeness_weight
        ).clamp(0.0, 1.0);
        
        let confidence_level = ConfidenceLevel::from_score(overall_confidence);
        
        Ok(ConfidenceBreakdown {
            overall_confidence,
            data_quality,
            analysis_depth,
            consistency,
            completeness,
            confidence_level,
            confidence_factors: self.identify_module_confidence_factors(module_decl),
            uncertainty_factors: self.identify_module_uncertainty_factors(module_decl),
            improvement_recommendations: self.generate_module_improvement_recommendations(module_decl),
        })
    }
    
    // PROGRAM-LEVEL CONFIDENCE FACTORS
    
    /// Calculate data quality confidence
    fn calculate_data_quality_confidence(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> f64 {
        if modules.is_empty() {
            return 0.0;
        }
        
        let mut quality_score = 0.7; // Base score
        
        // Module count factor
        let module_count = modules.len();
        if module_count >= self.config.min_data_points {
            quality_score += 0.2; // Sufficient data
        } else {
            quality_score -= 0.1 * (self.config.min_data_points - module_count) as f64 / self.config.min_data_points as f64;
        }
        
        // Documentation quality
        let documented_modules = modules.iter()
            .filter(|(_, m)| m.description.is_some())
            .count();
        let documentation_ratio = documented_modules as f64 / module_count as f64;
        quality_score += documentation_ratio * 0.15;
        
        // Capability definition quality
        let capability_modules = modules.iter()
            .filter(|(_, m)| m.capability.is_some())
            .count();
        let capability_ratio = capability_modules as f64 / module_count as f64;
        quality_score += capability_ratio * 0.15;
        
        quality_score.clamp(0.0, 1.0)
    }
    
    /// Calculate analysis depth confidence
    fn calculate_analysis_depth_confidence(&self, modules: &[(&AstNode<Item>, &ModuleDecl)], _metrics: &CohesionMetrics) -> f64 {
        let mut depth_score = 0.6; // Base score
        
        // Section organization depth
        let well_organized_modules = modules.iter()
            .filter(|(_, m)| m.sections.len() >= 2)
            .count();
        let organization_ratio = well_organized_modules as f64 / modules.len() as f64;
        depth_score += organization_ratio * 0.2;
        
        // Type definition depth
        let modules_with_types = modules.iter()
            .filter(|(_, m)| m.sections.iter().any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Types)))
            .count();
        let types_ratio = modules_with_types as f64 / modules.len() as f64;
        depth_score += types_ratio * 0.15;
        
        // Interface definition depth
        let modules_with_interfaces = modules.iter()
            .filter(|(_, m)| m.sections.iter().any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Interface)))
            .count();
        let interface_ratio = modules_with_interfaces as f64 / modules.len() as f64;
        depth_score += interface_ratio * 0.15;
        
        depth_score.clamp(0.0, 1.0)
    }
    
    /// Calculate consistency confidence
    fn calculate_consistency_confidence(&self, modules: &[(&AstNode<Item>, &ModuleDecl)], metrics: &CohesionMetrics) -> f64 {
        let mut consistency_score = 0.5; // Base score
        
        // Metric consistency (low variance in scores indicates consistency)
        let scores = [
            metrics.type_cohesion,
            metrics.semantic_cohesion,
            metrics.business_cohesion,
            metrics.dependency_cohesion,
        ];
        
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter()
            .map(|s| (s - mean_score).powi(2))
            .sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();
        
        // Lower standard deviation = higher consistency
        if std_dev < 10.0 {
            consistency_score += 0.3; // Very consistent
        } else if std_dev < 20.0 {
            consistency_score += 0.2; // Reasonably consistent
        } else if std_dev < 30.0 {
            consistency_score += 0.1; // Moderately consistent
        }
        
        // Naming consistency across modules
        let naming_consistency = self.calculate_cross_module_naming_consistency(modules);
        consistency_score += naming_consistency * 0.2;
        
        consistency_score.clamp(0.0, 1.0)
    }
    
    /// Calculate completeness confidence
    fn calculate_completeness_confidence(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> f64 {
        if modules.is_empty() {
            return 0.0;
        }
        
        let mut completeness_score = 0.4; // Base score
        
        // Essential elements present
        let total_modules = modules.len();
        
        // Documentation completeness
        let documented_count = modules.iter()
            .filter(|(_, m)| m.description.is_some())
            .count();
        completeness_score += (documented_count as f64 / total_modules as f64) * 0.2;
        
        // Capability completeness
        let capability_count = modules.iter()
            .filter(|(_, m)| m.capability.is_some())
            .count();
        completeness_score += (capability_count as f64 / total_modules as f64) * 0.2;
        
        // Section completeness
        let complete_modules = modules.iter()
            .filter(|(_, m)| m.sections.len() >= 2)
            .count();
        completeness_score += (complete_modules as f64 / total_modules as f64) * 0.2;
        
        completeness_score.clamp(0.0, 1.0)
    }
    
    // MODULE-LEVEL CONFIDENCE FACTORS
    
    /// Calculate module data quality
    fn calculate_module_data_quality(&self, module_decl: &ModuleDecl) -> f64 {
        let mut quality_score: f64 = 0.5; // Base score
        
        if module_decl.description.is_some() {
            quality_score += 0.2;
        }
        
        if module_decl.capability.is_some() {
            quality_score += 0.2;
        }
        
        if !module_decl.sections.is_empty() {
            quality_score += 0.1;
        }
        
        quality_score.clamp(0.0, 1.0)
    }
    
    /// Calculate module analysis depth
    fn calculate_module_analysis_depth(&self, module_decl: &ModuleDecl) -> f64 {
        let mut depth_score: f64 = 0.4; // Base score
        
        let section_count = module_decl.sections.len();
        if section_count >= 3 {
            depth_score += 0.3;
        } else if section_count >= 2 {
            depth_score += 0.2;
        } else if section_count >= 1 {
            depth_score += 0.1;
        }
        
        // Check for different types of sections
        let has_types = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Types));
        let has_interface = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Interface));
        
        if has_types { depth_score += 0.15; }
        if has_interface { depth_score += 0.15; }
        
        depth_score.clamp(0.0, 1.0)
    }
    
    /// Calculate module consistency
    fn calculate_module_consistency(&self, _module_decl: &ModuleDecl, _metrics: &CohesionMetrics) -> f64 {
        // For single module, consistency is harder to measure
        // Base it on internal consistency indicators
        0.7 // Neutral score
    }
    
    /// Calculate module completeness
    fn calculate_module_completeness(&self, module_decl: &ModuleDecl) -> f64 {
        let mut completeness_score: f64 = 0.3; // Base score
        
        // Essential elements
        if module_decl.description.is_some() { completeness_score += 0.2; }
        if module_decl.capability.is_some() { completeness_score += 0.2; }
        if !module_decl.sections.is_empty() { completeness_score += 0.2; }
        if !module_decl.dependencies.is_empty() { completeness_score += 0.1; }
        
        completeness_score.clamp(0.0, 1.0)
    }
    
    // HELPER METHODS
    
    /// Calculate naming consistency across modules
    fn calculate_cross_module_naming_consistency(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> f64 {
        if modules.len() < 2 {
            return 0.5; // Neutral for single module
        }
        
        // Simple heuristic: check for consistent naming patterns
        let module_names: Vec<String> = modules.iter()
            .map(|(_, m)| m.name.to_string())
            .collect();
        
        // Check for common patterns (simplified)
        let snake_case_count = module_names.iter()
            .filter(|name| name.contains('_'))
            .count();
        
        let consistency_ratio = if snake_case_count == module_names.len() || snake_case_count == 0 {
            1.0 // All follow same pattern
        } else {
            snake_case_count as f64 / module_names.len() as f64
        };
        
        consistency_ratio
    }
    
    /// Identify factors contributing to confidence
    fn identify_confidence_factors(&self, modules: &[(&AstNode<Item>, &ModuleDecl)], _metrics: &CohesionMetrics) -> Vec<String> {
        let mut factors = Vec::new();
        
        if modules.len() >= self.config.min_data_points {
            factors.push(format!("Sufficient modules for analysis ({})", modules.len()));
        }
        
        let documented_ratio = modules.iter()
            .filter(|(_, m)| m.description.is_some())
            .count() as f64 / modules.len() as f64;
        
        if documented_ratio >= 0.8 {
            factors.push("Most modules are well-documented".to_string());
        }
        
        let capability_ratio = modules.iter()
            .filter(|(_, m)| m.capability.is_some())
            .count() as f64 / modules.len() as f64;
        
        if capability_ratio >= 0.7 {
            factors.push("Most modules have clear capabilities defined".to_string());
        }
        
        factors
    }
    
    /// Identify factors reducing confidence
    fn identify_uncertainty_factors(&self, modules: &[(&AstNode<Item>, &ModuleDecl)], _metrics: &CohesionMetrics) -> Vec<String> {
        let mut factors = Vec::new();
        
        if modules.len() < self.config.min_data_points {
            factors.push(format!("Limited modules for analysis ({})", modules.len()));
        }
        
        let empty_modules = modules.iter()
            .filter(|(_, m)| m.sections.is_empty())
            .count();
        
        if empty_modules > 0 {
            factors.push(format!("{} modules have no sections", empty_modules));
        }
        
        let undocumented_modules = modules.iter()
            .filter(|(_, m)| m.description.is_none())
            .count();
        
        if undocumented_modules > modules.len() / 2 {
            factors.push("Many modules lack documentation".to_string());
        }
        
        factors
    }
    
    /// Generate improvement recommendations
    fn generate_improvement_recommendations(
        &self,
        _confidence_factors: &[String],
        uncertainty_factors: &[String],
        confidence_level: ConfidenceLevel,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match confidence_level {
            ConfidenceLevel::VeryLow | ConfidenceLevel::Low => {
                recommendations.push("Add more modules to increase analysis reliability".to_string());
                recommendations.push("Improve module documentation and capability definitions".to_string());
            }
            ConfidenceLevel::Medium => {
                recommendations.push("Consider adding more detailed documentation".to_string());
            }
            _ => {
                recommendations.push("Analysis confidence is good".to_string());
            }
        }
        
        // Address specific uncertainty factors
        for factor in uncertainty_factors {
            if factor.contains("modules have no sections") {
                recommendations.push("Add meaningful sections to empty modules".to_string());
            }
            if factor.contains("lack documentation") {
                recommendations.push("Add descriptions to undocumented modules".to_string());
            }
        }
        
        recommendations
    }
    
    /// Identify module-specific confidence factors
    fn identify_module_confidence_factors(&self, module_decl: &ModuleDecl) -> Vec<String> {
        let mut factors = Vec::new();
        
        if module_decl.description.is_some() {
            factors.push("Module has documentation".to_string());
        }
        
        if module_decl.capability.is_some() {
            factors.push("Module has clear capability defined".to_string());
        }
        
        if module_decl.sections.len() >= 3 {
            factors.push("Module has well-organized sections".to_string());
        }
        
        factors
    }
    
    /// Identify module-specific uncertainty factors
    fn identify_module_uncertainty_factors(&self, module_decl: &ModuleDecl) -> Vec<String> {
        let mut factors = Vec::new();
        
        if module_decl.description.is_none() {
            factors.push("Module lacks documentation".to_string());
        }
        
        if module_decl.capability.is_none() {
            factors.push("Module lacks capability definition".to_string());
        }
        
        if module_decl.sections.is_empty() {
            factors.push("Module has no sections".to_string());
        }
        
        factors
    }
    
    /// Generate module-specific improvement recommendations
    fn generate_module_improvement_recommendations(&self, module_decl: &ModuleDecl) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if module_decl.description.is_none() {
            recommendations.push("Add module description for better analysis".to_string());
        }
        
        if module_decl.capability.is_none() {
            recommendations.push("Define module capability for business context".to_string());
        }
        
        if module_decl.sections.is_empty() {
            recommendations.push("Add sections to organize module content".to_string());
        }
        
        recommendations
    }
}

impl Default for ConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
} 