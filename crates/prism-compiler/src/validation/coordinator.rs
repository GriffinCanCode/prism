//! Validation Coordinator - Cross-Target Result Coordination
//!
//! This module implements coordination of validation results from different validators,
//! focusing on cross-target consistency analysis without duplicating domain logic.
//!
//! **Conceptual Responsibility**: Result coordination and consistency analysis
//! **What it does**: Analyzes consistency across validation results, coordinates findings
//! **What it doesn't do**: Implement domain validation logic (uses results from existing validators)

use crate::error::{CompilerError, CompilerResult};
use crate::validation::{AggregatedValidationResults};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Validation coordinator for cross-target consistency analysis
/// 
/// This coordinator analyzes results from existing validators to determine
/// cross-target consistency without reimplementing domain validation logic.
#[derive(Debug)]
pub struct ValidationCoordinator {
    /// Coordination configuration
    config: CoordinationConfig,
}

/// Configuration for result coordination
#[derive(Debug, Clone)]
pub struct CoordinationConfig {
    /// Minimum consistency threshold (0.0 to 1.0)
    pub min_consistency_threshold: f64,
    /// Enable detailed consistency analysis
    pub detailed_analysis: bool,
    /// Consistency check weights
    pub consistency_weights: ConsistencyWeights,
}

/// Weights for different consistency checks
#[derive(Debug, Clone)]
pub struct ConsistencyWeights {
    /// PIR validation weight
    pub pir_weight: f64,
    /// Semantic validation weight
    pub semantic_weight: f64,
    /// Constraint validation weight
    pub constraint_weight: f64,
    /// Effect validation weight
    pub effect_weight: f64,
    /// Codegen validation weight
    pub codegen_weight: f64,
}

/// Consistency analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyAnalysis {
    /// Overall consistency status
    pub overall_consistency: bool,
    /// Consistency score (0.0 to 1.0)
    pub consistency_score: f64,
    /// Detailed consistency findings
    pub findings: Vec<ConsistencyFinding>,
    /// Cross-target inconsistencies
    pub inconsistencies: Vec<CrossTargetInconsistency>,
    /// Consistency recommendations
    pub recommendations: Vec<String>,
}

/// Individual consistency finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyFinding {
    /// Finding category
    pub category: String,
    /// Finding severity
    pub severity: ConsistencySeverity,
    /// Finding description
    pub description: String,
    /// Affected validation areas
    pub affected_areas: Vec<String>,
}

/// Cross-target inconsistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossTargetInconsistency {
    /// Inconsistency type
    pub inconsistency_type: InconsistencyType,
    /// Description of the inconsistency
    pub description: String,
    /// Affected targets
    pub affected_targets: Vec<String>,
    /// Severity level
    pub severity: ConsistencySeverity,
}

/// Types of cross-target inconsistencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InconsistencyType {
    /// Validation results differ between targets
    ValidationMismatch,
    /// Warning patterns differ between targets
    WarningPatternMismatch,
    /// Success/failure status differs
    StatusMismatch,
    /// Quality metrics vary significantly
    QualityMetricVariation,
}

/// Consistency severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencySeverity {
    /// Critical consistency issue
    Critical,
    /// High priority consistency issue
    High,
    /// Medium priority consistency issue
    Medium,
    /// Low priority consistency issue
    Low,
    /// Informational finding
    Info,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            min_consistency_threshold: 0.8,
            detailed_analysis: true,
            consistency_weights: ConsistencyWeights::default(),
        }
    }
}

impl Default for ConsistencyWeights {
    fn default() -> Self {
        Self {
            pir_weight: 0.3,        // PIR validation is critical
            semantic_weight: 0.25,   // Business rules are important
            constraint_weight: 0.2,  // Constraints matter
            effect_weight: 0.15,     // Effects are important for security
            codegen_weight: 0.1,     // Codegen warnings are less critical
        }
    }
}

impl ValidationCoordinator {
    /// Create new validation coordinator
    pub fn new() -> Self {
        Self {
            config: CoordinationConfig::default(),
        }
    }

    /// Create coordinator with custom configuration
    pub fn with_config(config: CoordinationConfig) -> Self {
        Self { config }
    }

    /// Analyze consistency across validation results
    /// 
    /// This method coordinates results from existing validators to determine
    /// cross-target consistency without reimplementing validation logic.
    pub fn analyze_consistency(
        &self,
        results: &AggregatedValidationResults,
    ) -> CompilerResult<ConsistencyAnalysis> {
        let mut findings = Vec::new();
        let mut inconsistencies = Vec::new();
        let mut recommendations = Vec::new();

        // Analyze PIR validation consistency
        let pir_consistency = self.analyze_pir_consistency(&results.pir_result)?;
        findings.extend(pir_consistency.findings);

        // Analyze semantic validation consistency
        let semantic_consistency = self.analyze_semantic_consistency(&results.semantic_result)?;
        findings.extend(semantic_consistency.findings);

        // Analyze constraint validation consistency
        let constraint_consistency = self.analyze_constraint_consistency(&results.constraint_result)?;
        findings.extend(constraint_consistency.findings);

        // Analyze effect validation consistency
        let effect_consistency = self.analyze_effect_consistency(&results.effect_result)?;
        findings.extend(effect_consistency.findings);

        // Analyze cross-target codegen consistency
        let codegen_inconsistencies = self.analyze_codegen_consistency(&results.codegen_results)?;
        inconsistencies.extend(codegen_inconsistencies);

        // Calculate overall consistency score
        let consistency_score = self.calculate_consistency_score(results);

        // Generate recommendations based on findings
        recommendations.extend(self.generate_consistency_recommendations(&findings, &inconsistencies));

        let overall_consistency = consistency_score >= self.config.min_consistency_threshold;

        Ok(ConsistencyAnalysis {
            overall_consistency,
            consistency_score,
            findings,
            inconsistencies,
            recommendations,
        })
    }

    /// Analyze PIR validation consistency
    fn analyze_pir_consistency(
        &self,
        pir_result: &crate::validation::PIRValidationResultSummary,
    ) -> CompilerResult<ConsistencySubAnalysis> {
        let mut findings = Vec::new();

        if !pir_result.passed {
            findings.push(ConsistencyFinding {
                category: "PIR Validation".to_string(),
                severity: ConsistencySeverity::Critical,
                description: "PIR semantic preservation validation failed".to_string(),
                affected_areas: vec!["semantic_preservation".to_string()],
            });
        }

        if pir_result.score < 0.8 {
            findings.push(ConsistencyFinding {
                category: "PIR Quality".to_string(),
                severity: ConsistencySeverity::High,
                description: format!("PIR validation score ({:.2}) below recommended threshold", pir_result.score),
                affected_areas: vec!["quality_metrics".to_string()],
            });
        }

        Ok(ConsistencySubAnalysis { findings })
    }

    /// Analyze semantic validation consistency
    fn analyze_semantic_consistency(
        &self,
        semantic_result: &crate::validation::SemanticValidationResultSummary,
    ) -> CompilerResult<ConsistencySubAnalysis> {
        let mut findings = Vec::new();

        if semantic_result.error_count > 0 {
            findings.push(ConsistencyFinding {
                category: "Semantic Validation".to_string(),
                severity: ConsistencySeverity::High,
                description: format!("Semantic validation found {} errors", semantic_result.error_count),
                affected_areas: vec!["business_rules".to_string()],
            });
        }

        if semantic_result.rule_violation_count > 0 {
            findings.push(ConsistencyFinding {
                category: "Business Rules".to_string(),
                severity: ConsistencySeverity::High,
                description: format!("Found {} business rule violations", semantic_result.rule_violation_count),
                affected_areas: vec!["business_rules".to_string()],
            });
        }

        Ok(ConsistencySubAnalysis { findings })
    }

    /// Analyze constraint validation consistency
    fn analyze_constraint_consistency(
        &self,
        constraint_result: &crate::validation::ConstraintValidationResultSummary,
    ) -> CompilerResult<ConsistencySubAnalysis> {
        let mut findings = Vec::new();

        if constraint_result.error_count > 0 {
            findings.push(ConsistencyFinding {
                category: "Constraint Validation".to_string(),
                severity: ConsistencySeverity::High,
                description: format!("Constraint validation found {} errors", constraint_result.error_count),
                affected_areas: vec!["type_constraints".to_string()],
            });
        }

        Ok(ConsistencySubAnalysis { findings })
    }

    /// Analyze effect validation consistency
    fn analyze_effect_consistency(
        &self,
        effect_result: &crate::validation::EffectValidationResultSummary,
    ) -> CompilerResult<ConsistencySubAnalysis> {
        let mut findings = Vec::new();

        if effect_result.violation_count > 0 {
            findings.push(ConsistencyFinding {
                category: "Effect Validation".to_string(),
                severity: ConsistencySeverity::High,
                description: format!("Effect validation found {} violations", effect_result.violation_count),
                affected_areas: vec!["effect_system".to_string()],
            });
        }

        Ok(ConsistencySubAnalysis { findings })
    }

    /// Analyze cross-target codegen consistency
    fn analyze_codegen_consistency(
        &self,
        codegen_results: &HashMap<crate::context::CompilationTarget, crate::validation::CodegenValidationResultSummary>,
    ) -> CompilerResult<Vec<CrossTargetInconsistency>> {
        let mut inconsistencies = Vec::new();

        // Check for status mismatches across targets
        let passed_targets: Vec<_> = codegen_results.iter()
            .filter(|(_, result)| result.passed)
            .map(|(target, _)| format!("{:?}", target))
            .collect();

        let failed_targets: Vec<_> = codegen_results.iter()
            .filter(|(_, result)| !result.passed)
            .map(|(target, _)| format!("{:?}", target))
            .collect();

        if !passed_targets.is_empty() && !failed_targets.is_empty() {
            inconsistencies.push(CrossTargetInconsistency {
                inconsistency_type: InconsistencyType::StatusMismatch,
                description: format!("Validation passed for {} but failed for {}", 
                                   passed_targets.join(", "), failed_targets.join(", ")),
                affected_targets: codegen_results.keys().map(|t| format!("{:?}", t)).collect(),
                severity: ConsistencySeverity::High,
            });
        }

        // Check for warning pattern differences
        let warning_counts: HashMap<_, _> = codegen_results.iter()
            .map(|(target, result)| (target, result.warnings.len()))
            .collect();

        if warning_counts.values().max() != warning_counts.values().min() {
            inconsistencies.push(CrossTargetInconsistency {
                inconsistency_type: InconsistencyType::WarningPatternMismatch,
                description: "Warning counts vary significantly across targets".to_string(),
                affected_targets: codegen_results.keys().map(|t| format!("{:?}", t)).collect(),
                severity: ConsistencySeverity::Medium,
            });
        }

        Ok(inconsistencies)
    }

    /// Calculate overall consistency score
    fn calculate_consistency_score(&self, results: &AggregatedValidationResults) -> f64 {
        let weights = &self.config.consistency_weights;
        let mut weighted_score = 0.0;

        // PIR validation contribution
        let pir_score = if results.pir_result.passed { results.pir_result.score } else { 0.0 };
        weighted_score += pir_score * weights.pir_weight;

        // Semantic validation contribution
        let semantic_score = if results.semantic_result.passed { 1.0 } else { 0.0 };
        weighted_score += semantic_score * weights.semantic_weight;

        // Constraint validation contribution
        let constraint_score = if results.constraint_result.passed { 1.0 } else { 0.0 };
        weighted_score += constraint_score * weights.constraint_weight;

        // Effect validation contribution
        let effect_score = if results.effect_result.passed { 1.0 } else { 0.0 };
        weighted_score += effect_score * weights.effect_weight;

        // Codegen validation contribution
        let codegen_passed_count = results.codegen_results.values().filter(|r| r.passed).count();
        let codegen_total_count = results.codegen_results.len();
        let codegen_score = if codegen_total_count > 0 {
            codegen_passed_count as f64 / codegen_total_count as f64
        } else {
            1.0
        };
        weighted_score += codegen_score * weights.codegen_weight;

        weighted_score
    }

    /// Generate consistency recommendations
    fn generate_consistency_recommendations(
        &self,
        findings: &[ConsistencyFinding],
        inconsistencies: &[CrossTargetInconsistency],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Recommendations based on findings
        let critical_findings = findings.iter().filter(|f| matches!(f.severity, ConsistencySeverity::Critical)).count();
        if critical_findings > 0 {
            recommendations.push(format!("Address {} critical consistency issues before proceeding", critical_findings));
        }

        let high_findings = findings.iter().filter(|f| matches!(f.severity, ConsistencySeverity::High)).count();
        if high_findings > 0 {
            recommendations.push(format!("Review {} high-priority consistency issues", high_findings));
        }

        // Recommendations based on inconsistencies
        if inconsistencies.iter().any(|i| matches!(i.inconsistency_type, InconsistencyType::StatusMismatch)) {
            recommendations.push("Investigate validation status differences between targets".to_string());
        }

        if inconsistencies.iter().any(|i| matches!(i.inconsistency_type, InconsistencyType::WarningPatternMismatch)) {
            recommendations.push("Review warning patterns across targets for consistency".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Validation consistency looks good across all targets".to_string());
        }

        recommendations
    }
}

/// Sub-analysis result for individual validation areas
struct ConsistencySubAnalysis {
    findings: Vec<ConsistencyFinding>,
}

impl Default for ValidationCoordinator {
    fn default() -> Self {
        Self::new()
    }
} 