//! Validation Reporting - Result Formatting and Presentation
//!
//! This module implements validation result reporting and formatting,
//! focusing on presenting results from existing validators in a unified format.
//!
//! **Conceptual Responsibility**: Result formatting and reporting
//! **What it does**: Formats validation results, generates reports, presents findings
//! **What it doesn't do**: Implement validation logic (presents results from existing validators)

use crate::validation::{AggregatedValidationResults, ConsistencyAnalysis};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Comprehensive validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Report summary
    pub summary: ValidationSummary,
    /// Consistency analysis results
    pub consistency_analysis: ConsistencyAnalysis,
    /// Aggregated results from all validators
    pub aggregated_results: AggregatedValidationResults,
    /// Recommendations for improvement
    pub recommendations: Vec<ValidationRecommendation>,
    /// Report generation timestamp
    pub timestamp: DateTime<Utc>,
}

/// Validation summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of validations performed
    pub total_validations: usize,
    /// Number of validations that passed
    pub passed_validations: usize,
    /// Total issues found across all validators
    pub total_issues: usize,
    /// Overall consistency score (0.0 to 1.0)
    pub consistency_score: f64,
    /// Total validation duration in milliseconds
    pub validation_duration_ms: u64,
}

/// Validation recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRecommendation {
    /// Recommendation category
    pub category: String,
    /// Priority level
    pub priority: String,
    /// Recommendation message
    pub message: String,
    /// Suggested action
    pub action: String,
}

impl ValidationReport {
    /// Generate a human-readable summary of the validation report
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str("=== PRISM VALIDATION REPORT ===\n");
        summary.push_str(&format!("Generated: {}\n", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        summary.push_str("\n");

        // Overall summary
        summary.push_str("=== OVERALL SUMMARY ===\n");
        summary.push_str(&format!("Total Validations: {}\n", self.summary.total_validations));
        summary.push_str(&format!("Passed Validations: {}\n", self.summary.passed_validations));
        summary.push_str(&format!("Total Issues: {}\n", self.summary.total_issues));
        summary.push_str(&format!("Consistency Score: {:.2}/1.0\n", self.summary.consistency_score));
        summary.push_str(&format!("Validation Duration: {}ms\n", self.summary.validation_duration_ms));
        summary.push_str("\n");

        // Validation results breakdown
        summary.push_str("=== VALIDATION RESULTS ===\n");
        
        // PIR validation
        summary.push_str(&format!("PIR Validation: {} (Score: {:.2})\n", 
                                 if self.aggregated_results.pir_result.passed { "PASSED" } else { "FAILED" },
                                 self.aggregated_results.pir_result.score));
        
        // Semantic validation
        summary.push_str(&format!("Semantic Validation: {} ({} errors, {} warnings)\n",
                                 if self.aggregated_results.semantic_result.passed { "PASSED" } else { "FAILED" },
                                 self.aggregated_results.semantic_result.error_count,
                                 self.aggregated_results.semantic_result.warning_count));
        
        // Constraint validation
        summary.push_str(&format!("Constraint Validation: {} ({} errors)\n",
                                 if self.aggregated_results.constraint_result.passed { "PASSED" } else { "FAILED" },
                                 self.aggregated_results.constraint_result.error_count));
        
        // Effect validation
        summary.push_str(&format!("Effect Validation: {} ({} violations)\n",
                                 if self.aggregated_results.effect_result.passed { "PASSED" } else { "FAILED" },
                                 self.aggregated_results.effect_result.violation_count));
        
        // Codegen validation
        summary.push_str("Codegen Validation:\n");
        for (target, result) in &self.aggregated_results.codegen_results {
            summary.push_str(&format!("  {:?}: {} ({} warnings)\n",
                                     target,
                                     if result.passed { "PASSED" } else { "FAILED" },
                                     result.warnings.len()));
        }
        summary.push_str("\n");

        // Consistency analysis
        summary.push_str("=== CONSISTENCY ANALYSIS ===\n");
        summary.push_str(&format!("Overall Consistency: {}\n", 
                                 if self.consistency_analysis.overall_consistency { "CONSISTENT" } else { "INCONSISTENT" }));
        summary.push_str(&format!("Consistency Score: {:.2}/1.0\n", self.consistency_analysis.consistency_score));
        
        if !self.consistency_analysis.findings.is_empty() {
            summary.push_str("Consistency Findings:\n");
            for finding in &self.consistency_analysis.findings {
                summary.push_str(&format!("  [{:?}] {}: {}\n", 
                                         finding.severity, finding.category, finding.description));
            }
        }
        
        if !self.consistency_analysis.inconsistencies.is_empty() {
            summary.push_str("Cross-Target Inconsistencies:\n");
            for inconsistency in &self.consistency_analysis.inconsistencies {
                summary.push_str(&format!("  [{:?}] {:?}: {}\n", 
                                         inconsistency.severity, inconsistency.inconsistency_type, inconsistency.description));
            }
        }
        summary.push_str("\n");

        // Recommendations
        if !self.recommendations.is_empty() {
            summary.push_str("=== RECOMMENDATIONS ===\n");
            for rec in &self.recommendations {
                summary.push_str(&format!("[{}] {}: {}\n", rec.priority, rec.category, rec.message));
                summary.push_str(&format!("  Action: {}\n", rec.action));
            }
            summary.push_str("\n");
        }

        summary.push_str("=== END REPORT ===\n");
        summary
    }

    /// Generate a JSON representation of the report
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Generate a condensed status report
    pub fn generate_status_report(&self) -> String {
        let status = if self.consistency_analysis.overall_consistency && 
                       self.summary.total_issues == 0 {
            "✅ VALIDATION PASSED"
        } else if self.consistency_analysis.overall_consistency {
            "⚠️  VALIDATION PASSED WITH WARNINGS"
        } else {
            "❌ VALIDATION FAILED"
        };

        format!("{} - Consistency: {:.2}, Issues: {}, Duration: {}ms",
                status, 
                self.summary.consistency_score,
                self.summary.total_issues,
                self.summary.validation_duration_ms)
    }

    /// Check if validation passed overall
    pub fn validation_passed(&self) -> bool {
        self.consistency_analysis.overall_consistency && self.summary.total_issues == 0
    }

    /// Check if validation passed with warnings
    pub fn validation_passed_with_warnings(&self) -> bool {
        self.consistency_analysis.overall_consistency && self.summary.total_issues > 0
    }

    /// Get critical issues count
    pub fn get_critical_issues_count(&self) -> usize {
        self.consistency_analysis.findings.iter()
            .filter(|f| matches!(f.severity, crate::validation::ConsistencySeverity::Critical))
            .count() +
        self.consistency_analysis.inconsistencies.iter()
            .filter(|i| matches!(i.severity, crate::validation::ConsistencySeverity::Critical))
            .count()
    }

    /// Get high priority issues count
    pub fn get_high_priority_issues_count(&self) -> usize {
        self.consistency_analysis.findings.iter()
            .filter(|f| matches!(f.severity, crate::validation::ConsistencySeverity::High))
            .count() +
        self.consistency_analysis.inconsistencies.iter()
            .filter(|i| matches!(i.severity, crate::validation::ConsistencySeverity::High))
            .count()
    }
}

impl ValidationSummary {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_validations == 0 {
            1.0
        } else {
            self.passed_validations as f64 / self.total_validations as f64
        }
    }

    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.passed_validations == self.total_validations
    }
}

/// Report formatter for different output formats
pub struct ValidationReportFormatter;

impl ValidationReportFormatter {
    /// Format report as markdown
    pub fn format_as_markdown(report: &ValidationReport) -> String {
        let mut md = String::new();
        
        md.push_str("# Prism Validation Report\n\n");
        md.push_str(&format!("**Generated:** {}\n\n", report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));

        // Summary table
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Total Validations | {} |\n", report.summary.total_validations));
        md.push_str(&format!("| Passed Validations | {} |\n", report.summary.passed_validations));
        md.push_str(&format!("| Total Issues | {} |\n", report.summary.total_issues));
        md.push_str(&format!("| Consistency Score | {:.2}/1.0 |\n", report.summary.consistency_score));
        md.push_str(&format!("| Duration | {}ms |\n", report.summary.validation_duration_ms));
        md.push_str("\n");

        // Validation results
        md.push_str("## Validation Results\n\n");
        md.push_str("| Validator | Status | Details |\n");
        md.push_str("|-----------|--------|---------|\n");
        
        let pir_status = if report.aggregated_results.pir_result.passed { "✅ PASSED" } else { "❌ FAILED" };
        md.push_str(&format!("| PIR | {} | Score: {:.2} |\n", pir_status, report.aggregated_results.pir_result.score));
        
        let semantic_status = if report.aggregated_results.semantic_result.passed { "✅ PASSED" } else { "❌ FAILED" };
        md.push_str(&format!("| Semantic | {} | {} errors, {} warnings |\n", 
                           semantic_status, 
                           report.aggregated_results.semantic_result.error_count,
                           report.aggregated_results.semantic_result.warning_count));
        
        let constraint_status = if report.aggregated_results.constraint_result.passed { "✅ PASSED" } else { "❌ FAILED" };
        md.push_str(&format!("| Constraint | {} | {} errors |\n", 
                           constraint_status, 
                           report.aggregated_results.constraint_result.error_count));
        
        let effect_status = if report.aggregated_results.effect_result.passed { "✅ PASSED" } else { "❌ FAILED" };
        md.push_str(&format!("| Effect | {} | {} violations |\n", 
                           effect_status, 
                           report.aggregated_results.effect_result.violation_count));

        for (target, result) in &report.aggregated_results.codegen_results {
            let status = if result.passed { "✅ PASSED" } else { "❌ FAILED" };
            md.push_str(&format!("| Codegen {:?} | {} | {} warnings |\n", target, status, result.warnings.len()));
        }
        md.push_str("\n");

        // Recommendations
        if !report.recommendations.is_empty() {
            md.push_str("## Recommendations\n\n");
            for rec in &report.recommendations {
                md.push_str(&format!("### {} - {}\n", rec.priority, rec.category));
                md.push_str(&format!("{}\n\n", rec.message));
                md.push_str(&format!("**Action:** {}\n\n", rec.action));
            }
        }

        md
    }

    /// Format report as CSV
    pub fn format_as_csv(report: &ValidationReport) -> String {
        let mut csv = String::new();
        
        csv.push_str("Validator,Status,Score,Errors,Warnings,Details\n");
        
        csv.push_str(&format!("PIR,{},{:.2},0,0,{} checks\n",
                             if report.aggregated_results.pir_result.passed { "PASSED" } else { "FAILED" },
                             report.aggregated_results.pir_result.score,
                             report.aggregated_results.pir_result.check_count));
        
        csv.push_str(&format!("Semantic,{},0,{},{},{} rule violations\n",
                             if report.aggregated_results.semantic_result.passed { "PASSED" } else { "FAILED" },
                             report.aggregated_results.semantic_result.error_count,
                             report.aggregated_results.semantic_result.warning_count,
                             report.aggregated_results.semantic_result.rule_violation_count));
        
        csv.push_str(&format!("Constraint,{},0,{},{},\n",
                             if report.aggregated_results.constraint_result.passed { "PASSED" } else { "FAILED" },
                             report.aggregated_results.constraint_result.error_count,
                             report.aggregated_results.constraint_result.warning_count));
        
        csv.push_str(&format!("Effect,{},0,0,{},{} violations\n",
                             if report.aggregated_results.effect_result.passed { "PASSED" } else { "FAILED" },
                             report.aggregated_results.effect_result.warning_count,
                             report.aggregated_results.effect_result.violation_count));

        for (target, result) in &report.aggregated_results.codegen_results {
            csv.push_str(&format!("Codegen-{:?},{},0,0,{},\n",
                                 target,
                                 if result.passed { "PASSED" } else { "FAILED" },
                                 result.warnings.len()));
        }
        
        csv
    }
} 