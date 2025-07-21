//! AI-Formatted Cohesion Insights
//!
//! This module generates AI-comprehensible analysis and insights from cohesion
//! analysis results. No actual ML integration is required - this provides
//! structured data formatted for AI tools to consume.
//!
//! **Conceptual Responsibility**: Generate AI-formatted cohesion insights

use crate::{CohesionResult, CohesionError, CohesionMetrics, CohesionViolation, ConceptualBoundary};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::{span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// AI insight generator for cohesion analysis
#[derive(Debug)]
pub struct AIInsightGenerator {
    /// Insight generation configuration
    config: AIInsightConfig,
    
    /// Pattern recognition database
    pattern_db: PatternDatabase,
}

/// Configuration for AI insight generation
#[derive(Debug, Clone)]
pub struct AIInsightConfig {
    /// Enable architectural pattern detection
    pub enable_pattern_detection: bool,
    
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    
    /// Enable contextual recommendations
    pub enable_contextual_recommendations: bool,
    
    /// Insight verbosity level
    pub verbosity: AIInsightVerbosity,
    
    /// Maximum insights to generate
    pub max_insights: usize,
}

/// AI insight verbosity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AIInsightVerbosity {
    /// Essential insights only
    Essential,
    /// Standard insight level
    Standard,
    /// Comprehensive insights
    Comprehensive,
}

/// Complete AI-formatted cohesion insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAIInsights {
    /// Executive summary for AI consumption
    pub executive_summary: AIExecutiveSummary,
    
    /// Detected architectural patterns
    pub architectural_patterns: Vec<ArchitecturalPattern>,
    
    /// Cohesion trend analysis
    pub trend_analysis: TrendAnalysis,
    
    /// Contextual recommendations
    pub contextual_recommendations: Vec<ContextualRecommendation>,
    
    /// Technical debt indicators
    pub technical_debt: TechnicalDebtAnalysis,
    
    /// Quality metrics for AI processing
    pub quality_metrics: AIQualityMetrics,
    
    /// Metadata for AI tools
    pub ai_metadata: AIAnalysisMetadata,
}

/// Executive summary formatted for AI consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIExecutiveSummary {
    /// Overall assessment
    pub overall_assessment: String,
    
    /// Key strengths (AI-formatted)
    pub key_strengths: Vec<String>,
    
    /// Primary concerns (AI-formatted)
    pub primary_concerns: Vec<String>,
    
    /// Recommended actions (prioritized for AI)
    pub recommended_actions: Vec<String>,
    
    /// Risk level assessment
    pub risk_level: RiskLevel,
    
    /// Confidence in analysis
    pub confidence_score: f64,
}

/// Risk level assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk - good cohesion
    Low,
    /// Medium risk - some issues
    Medium,
    /// High risk - significant problems
    High,
    /// Critical risk - major architectural issues
    Critical,
}

/// Detected architectural pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern type
    pub pattern_type: PatternType,
    
    /// Pattern confidence (0-1)
    pub confidence: f64,
    
    /// Pattern description (AI-formatted)
    pub description: String,
    
    /// Components involved in the pattern
    pub components: Vec<String>,
    
    /// Pattern quality assessment
    pub quality_assessment: PatternQuality,
    
    /// Suggestions for pattern improvement
    pub improvement_suggestions: Vec<String>,
}

/// Types of architectural patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Domain-driven design pattern
    DomainDriven,
    
    /// Layered architecture pattern
    Layered,
    
    /// Modular monolith pattern
    ModularMonolith,
    
    /// Microservice pattern
    Microservice,
    
    /// Event-driven pattern
    EventDriven,
    
    /// Repository pattern
    Repository,
    
    /// Factory pattern
    Factory,
    
    /// Anti-pattern (problematic)
    AntiPattern,
    
    /// Custom pattern
    Custom(String),
}

/// Pattern quality assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternQuality {
    /// Excellent implementation
    Excellent,
    /// Good implementation
    Good,
    /// Adequate implementation
    Adequate,
    /// Poor implementation
    Poor,
    /// Anti-pattern (should be refactored)
    AntiPattern,
}

/// Trend analysis for AI consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Cohesion trajectory
    pub cohesion_trajectory: TrajectoryAssessment,
    
    /// Complexity trends
    pub complexity_trends: Vec<ComplexityTrend>,
    
    /// Dependency evolution
    pub dependency_evolution: DependencyTrend,
    
    /// Predictive insights
    pub predictive_insights: Vec<String>,
}

/// Trajectory assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryAssessment {
    /// Improving over time
    Improving,
    /// Stable/maintaining quality
    Stable,
    /// Degrading over time
    Degrading,
    /// Insufficient data
    Unknown,
}

/// Complexity trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityTrend {
    /// Metric name
    pub metric: String,
    
    /// Trend direction
    pub trend: TrendDirection,
    
    /// Trend strength (0-1)
    pub strength: f64,
    
    /// AI-formatted explanation
    pub explanation: String,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
    /// Volatile trend
    Volatile,
}

/// Dependency evolution trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyTrend {
    /// Overall dependency health
    pub overall_health: DependencyHealth,
    
    /// Dependency growth rate
    pub growth_rate: f64,
    
    /// Coupling trends
    pub coupling_trends: Vec<String>,
    
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Dependency health assessment
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyHealth {
    /// Healthy dependency structure
    Healthy,
    /// Moderate dependency issues
    Moderate,
    /// Unhealthy dependency structure
    Unhealthy,
    /// Critical dependency problems
    Critical,
}

/// Contextual recommendation for AI tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualRecommendation {
    /// Recommendation context
    pub context: String,
    
    /// Recommendation text (AI-formatted)
    pub recommendation: String,
    
    /// Reasoning behind recommendation
    pub reasoning: String,
    
    /// Expected impact
    pub expected_impact: f64,
    
    /// Implementation complexity
    pub implementation_complexity: String,
    
    /// Related patterns or principles
    pub related_concepts: Vec<String>,
}

/// Technical debt analysis for AI consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDebtAnalysis {
    /// Overall debt level
    pub debt_level: DebtLevel,
    
    /// Debt categories and scores
    pub debt_categories: HashMap<String, f64>,
    
    /// Debt hotspots (AI-formatted)
    pub hotspots: Vec<DebtHotspot>,
    
    /// Repayment strategy
    pub repayment_strategy: Vec<String>,
    
    /// Debt trend
    pub debt_trend: TrendDirection,
}

/// Technical debt level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebtLevel {
    /// Low debt
    Low,
    /// Moderate debt
    Moderate,
    /// High debt
    High,
    /// Critical debt
    Critical,
}

/// Technical debt hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebtHotspot {
    /// Location description
    pub location: String,
    
    /// Debt type
    pub debt_type: String,
    
    /// Severity score (0-10)
    pub severity: f64,
    
    /// AI-formatted description
    pub description: String,
    
    /// Suggested remediation
    pub remediation: String,
}

/// Quality metrics formatted for AI processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIQualityMetrics {
    /// Maintainability index (0-100)
    pub maintainability_index: f64,
    
    /// Testability score (0-100)
    pub testability_score: f64,
    
    /// Readability score (0-100)
    pub readability_score: f64,
    
    /// Modularity score (0-100)
    pub modularity_score: f64,
    
    /// Reusability potential (0-100)
    pub reusability_potential: f64,
    
    /// Evolution capability (0-100)
    pub evolution_capability: f64,
}

/// Metadata for AI analysis tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIAnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: String,
    
    /// Analysis version
    pub analysis_version: String,
    
    /// Data confidence level
    pub confidence_level: f64,
    
    /// Analysis scope
    pub analysis_scope: String,
    
    /// Recommended AI processing approaches
    pub processing_recommendations: Vec<String>,
    
    /// Data format version
    pub format_version: String,
}

/// Pattern recognition database
#[derive(Debug)]
struct PatternDatabase {
    /// Known patterns and their signatures
    patterns: HashMap<String, PatternSignature>,
}

/// Pattern signature for recognition
#[derive(Debug, Clone)]
struct PatternSignature {
    /// Pattern indicators
    pub indicators: Vec<String>,
    
    /// Pattern confidence threshold
    pub confidence_threshold: f64,
    
    /// Pattern quality indicators
    pub quality_indicators: Vec<String>,
}

impl AIInsightGenerator {
    /// Create new AI insight generator
    pub fn new() -> Self {
        Self {
            config: AIInsightConfig::default(),
            pattern_db: PatternDatabase::new(),
        }
    }
    
    /// Create AI insight generator with custom configuration
    pub fn with_config(config: AIInsightConfig) -> Self {
        Self {
            config,
            pattern_db: PatternDatabase::new(),
        }
    }
    
    /// Generate comprehensive AI insights for a program
    pub fn generate_program_insights(
        &self,
        program: &Program,
        metrics: &CohesionMetrics,
        boundaries: &[ConceptualBoundary],
        violations: &[CohesionViolation],
    ) -> CohesionResult<CohesionAIInsights> {
        
        // Generate executive summary
        let executive_summary = self.generate_executive_summary(metrics, violations)?;
        
        // Detect architectural patterns
        let architectural_patterns = if self.config.enable_pattern_detection {
            self.detect_architectural_patterns(program, metrics)?
        } else {
            Vec::new()
        };
        
        // Generate trend analysis
        let trend_analysis = if self.config.enable_trend_analysis {
            self.generate_trend_analysis(metrics)?
        } else {
            TrendAnalysis::empty()
        };
        
        // Generate contextual recommendations
        let contextual_recommendations = if self.config.enable_contextual_recommendations {
            self.generate_contextual_recommendations(program, metrics, violations)?
        } else {
            Vec::new()
        };
        
        // Analyze technical debt
        let technical_debt = self.analyze_technical_debt(metrics, violations)?;
        
        // Generate quality metrics
        let quality_metrics = self.generate_quality_metrics(metrics)?;
        
        // Create AI metadata
        let ai_metadata = self.create_ai_metadata()?;
        
        Ok(CohesionAIInsights {
            executive_summary,
            architectural_patterns,
            trend_analysis,
            contextual_recommendations,
            technical_debt,
            quality_metrics,
            ai_metadata,
        })
    }
    
    /// Generate executive summary for AI consumption
    fn generate_executive_summary(&self, metrics: &CohesionMetrics, violations: &[CohesionViolation]) -> CohesionResult<AIExecutiveSummary> {
        let overall_assessment = self.assess_overall_quality(metrics);
        let key_strengths = self.identify_key_strengths(metrics);
        let primary_concerns = self.identify_primary_concerns(metrics, violations);
        let recommended_actions = self.generate_recommended_actions(violations);
        let risk_level = self.assess_risk_level(metrics, violations);
        let confidence_score = self.calculate_confidence_score(metrics);
        
        Ok(AIExecutiveSummary {
            overall_assessment,
            key_strengths,
            primary_concerns,
            recommended_actions,
            risk_level,
            confidence_score,
        })
    }
    
    /// Assess overall quality for AI
    fn assess_overall_quality(&self, metrics: &CohesionMetrics) -> String {
        let score = metrics.overall_score;
        
        if score >= 90.0 {
            "EXCELLENT: Code demonstrates exceptional cohesion with clear conceptual boundaries and strong architectural patterns.".to_string()
        } else if score >= 80.0 {
            "GOOD: Code shows strong cohesion with minor areas for improvement. Well-organized with clear responsibilities.".to_string()
        } else if score >= 70.0 {
            "ADEQUATE: Code has acceptable cohesion but would benefit from organizational improvements and clearer boundaries.".to_string()
        } else if score >= 60.0 {
            "CONCERNING: Code cohesion is below recommended standards. Multiple organizational issues need attention.".to_string()
        } else {
            "CRITICAL: Code suffers from poor cohesion with significant architectural and organizational problems requiring immediate attention.".to_string()
        }
    }
    
    /// Identify key strengths for AI
    fn identify_key_strengths(&self, metrics: &CohesionMetrics) -> Vec<String> {
        let mut strengths = Vec::new();
        
        if metrics.type_cohesion >= 80.0 {
            strengths.push("Strong type organization with well-related data structures".to_string());
        }
        
        if metrics.semantic_cohesion >= 80.0 {
            strengths.push("Excellent naming consistency and conceptual clarity".to_string());
        }
        
        if metrics.business_cohesion >= 80.0 {
            strengths.push("Clear business capability focus with well-defined responsibilities".to_string());
        }
        
        if metrics.data_flow_cohesion >= 80.0 {
            strengths.push("Smooth data flow patterns with logical function organization".to_string());
        }
        
        if metrics.dependency_cohesion >= 80.0 {
            strengths.push("Well-managed dependencies with focused external coupling".to_string());
        }
        
        if strengths.is_empty() {
            strengths.push("Analysis complete - ready for improvement recommendations".to_string());
        }
        
        strengths
    }
    
    /// Identify primary concerns for AI
    fn identify_primary_concerns(&self, metrics: &CohesionMetrics, violations: &[CohesionViolation]) -> Vec<String> {
        let mut concerns = Vec::new();
        
        // Metric-based concerns
        if metrics.type_cohesion < 60.0 {
            concerns.push("Type organization lacks coherence - types may not belong together".to_string());
        }
        
        if metrics.semantic_cohesion < 60.0 {
            concerns.push("Inconsistent naming patterns reduce code comprehension".to_string());
        }
        
        if metrics.business_cohesion < 60.0 {
            concerns.push("Unclear business focus - multiple responsibilities may be mixed".to_string());
        }
        
        // Violation-based concerns
        let error_count = violations.iter().filter(|v| matches!(v.severity, crate::ViolationSeverity::Error)).count();
        if error_count > 0 {
            concerns.push(format!("Critical violations detected: {} errors requiring immediate attention", error_count));
        }
        
        let warning_count = violations.iter().filter(|v| matches!(v.severity, crate::ViolationSeverity::Warning)).count();
        if warning_count > 3 {
            concerns.push(format!("Multiple organizational issues: {} warnings indicate systemic problems", warning_count));
        }
        
        concerns
    }
    
    /// Generate recommended actions for AI
    fn generate_recommended_actions(&self, violations: &[CohesionViolation]) -> Vec<String> {
        let mut actions = Vec::new();
        
        // Prioritize by violation severity and frequency
        let mut violation_types = std::collections::HashMap::new();
        for violation in violations {
            *violation_types.entry(&violation.violation_type).or_insert(0) += 1;
        }
        
        // Generate actions based on most common violations
        for (violation_type, count) in violation_types.iter() {
            if *count >= 2 {
                let action = match violation_type {
                    crate::ViolationType::LowCohesion => "PRIORITY: Improve module organization and clarify responsibilities".to_string(),
                    crate::ViolationType::MultipleResponsibilities => "PRIORITY: Split modules to focus on single responsibilities".to_string(),
                    crate::ViolationType::InconsistentNaming => "RECOMMENDED: Standardize naming conventions across modules".to_string(),
                    crate::ViolationType::ExcessiveDependencies => "RECOMMENDED: Reduce and organize external dependencies".to_string(),
                    crate::ViolationType::MissingCapability => "QUICK WIN: Add capability definitions to clarify module purposes".to_string(),
                    _ => "REVIEW: Address detected organizational issues".to_string(),
                };
                actions.push(format!("{} ({} instances)", action, count));
            }
        }
        
        if actions.is_empty() {
            actions.push("MAINTAIN: Continue current organizational practices".to_string());
        }
        
        actions
    }
    
    /// Assess risk level for AI
    fn assess_risk_level(&self, metrics: &CohesionMetrics, violations: &[CohesionViolation]) -> RiskLevel {
        let error_count = violations.iter().filter(|v| matches!(v.severity, crate::ViolationSeverity::Error)).count();
        
        if error_count > 0 || metrics.overall_score < 40.0 {
            RiskLevel::Critical
        } else if metrics.overall_score < 60.0 {
            RiskLevel::High
        } else if metrics.overall_score < 80.0 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }
    
    /// Calculate confidence score for AI
    fn calculate_confidence_score(&self, metrics: &CohesionMetrics) -> f64 {
        // Base confidence on analysis metadata if available
        metrics.metadata.confidence
    }
    
    /// Detect architectural patterns
    fn detect_architectural_patterns(&self, program: &Program, metrics: &CohesionMetrics) -> CohesionResult<Vec<ArchitecturalPattern>> {
        let mut patterns = Vec::new();
        
        // Analyze module organization patterns
        let modules: Vec<_> = program.items.iter()
            .filter_map(|item| {
                if let Item::Module(module_decl) = &item.kind {
                    Some(module_decl)
                } else {
                    None
                }
            })
            .collect();
        
        // Detect domain-driven design pattern
        if self.detect_domain_driven_pattern(&modules) {
            patterns.push(ArchitecturalPattern {
                name: "Domain-Driven Design".to_string(),
                pattern_type: PatternType::DomainDriven,
                confidence: 0.8,
                description: "Modules organized around business domains with clear capability definitions".to_string(),
                components: modules.iter().map(|m| m.name.to_string()).collect(),
                quality_assessment: self.assess_pattern_quality(metrics),
                improvement_suggestions: vec![
                    "Ensure all modules have clear capability definitions".to_string(),
                    "Verify domain boundaries align with business capabilities".to_string(),
                ],
            });
        }
        
        // Detect modular monolith pattern
        if modules.len() > 3 && metrics.dependency_cohesion > 70.0 {
            patterns.push(ArchitecturalPattern {
                name: "Modular Monolith".to_string(),
                pattern_type: PatternType::ModularMonolith,
                confidence: 0.7,
                description: "Well-organized modules within a single codebase with good dependency management".to_string(),
                components: modules.iter().map(|m| m.name.to_string()).collect(),
                quality_assessment: self.assess_pattern_quality(metrics),
                improvement_suggestions: vec![
                    "Monitor module coupling to prevent degradation".to_string(),
                    "Consider extraction of highly independent modules".to_string(),
                ],
            });
        }
        
        // Detect anti-patterns
        if metrics.overall_score < 50.0 {
            patterns.push(ArchitecturalPattern {
                name: "Big Ball of Mud".to_string(),
                pattern_type: PatternType::AntiPattern,
                confidence: 0.9,
                description: "Poorly organized code with unclear boundaries and mixed responsibilities".to_string(),
                components: vec!["Entire codebase".to_string()],
                quality_assessment: PatternQuality::AntiPattern,
                improvement_suggestions: vec![
                    "URGENT: Refactor to establish clear module boundaries".to_string(),
                    "Define and enforce architectural principles".to_string(),
                    "Implement domain-driven design practices".to_string(),
                ],
            });
        }
        
        Ok(patterns)
    }
    
    /// Detect domain-driven design pattern
    fn detect_domain_driven_pattern(&self, modules: &[&ModuleDecl]) -> bool {
        let capability_count = modules.iter()
            .filter(|m| m.capability.is_some())
            .count();
        
        // At least 70% of modules should have capabilities for DDD pattern
        capability_count as f64 / modules.len() as f64 > 0.7
    }
    
    /// Assess pattern quality
    fn assess_pattern_quality(&self, metrics: &CohesionMetrics) -> PatternQuality {
        if metrics.overall_score >= 90.0 {
            PatternQuality::Excellent
        } else if metrics.overall_score >= 80.0 {
            PatternQuality::Good
        } else if metrics.overall_score >= 70.0 {
            PatternQuality::Adequate
        } else if metrics.overall_score >= 50.0 {
            PatternQuality::Poor
        } else {
            PatternQuality::AntiPattern
        }
    }
    
    /// Generate trend analysis (simplified since we don't have historical data)
    fn generate_trend_analysis(&self, metrics: &CohesionMetrics) -> CohesionResult<TrendAnalysis> {
        Ok(TrendAnalysis {
            cohesion_trajectory: TrajectoryAssessment::Unknown,
            complexity_trends: vec![
                ComplexityTrend {
                    metric: "cohesion_score".to_string(),
                    trend: TrendDirection::Stable,
                    strength: 0.5,
                    explanation: "Current analysis baseline established".to_string(),
                },
            ],
            dependency_evolution: DependencyTrend {
                overall_health: if metrics.dependency_cohesion > 70.0 {
                    DependencyHealth::Healthy
                } else if metrics.dependency_cohesion > 50.0 {
                    DependencyHealth::Moderate
                } else {
                    DependencyHealth::Unhealthy
                },
                growth_rate: 0.0, // No historical data
                coupling_trends: vec!["Baseline measurement established".to_string()],
                recommendations: vec!["Monitor dependency evolution over time".to_string()],
            },
            predictive_insights: vec![
                "Establish baseline for future trend analysis".to_string(),
                "Monitor cohesion metrics over time to identify patterns".to_string(),
            ],
        })
    }
    
    /// Generate contextual recommendations
    fn generate_contextual_recommendations(&self, _program: &Program, metrics: &CohesionMetrics, violations: &[CohesionViolation]) -> CohesionResult<Vec<ContextualRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Generate recommendations based on current state
        if metrics.overall_score < 70.0 {
            recommendations.push(ContextualRecommendation {
                context: "Overall cohesion improvement".to_string(),
                recommendation: "Focus on improving the weakest cohesion metrics first for maximum impact".to_string(),
                reasoning: "Addressing fundamental organizational issues provides compound benefits".to_string(),
                expected_impact: 0.8,
                implementation_complexity: "Medium to High".to_string(),
                related_concepts: vec!["Single Responsibility Principle".to_string(), "Domain-Driven Design".to_string()],
            });
        }
        
        if !violations.is_empty() {
            recommendations.push(ContextualRecommendation {
                context: "Violation remediation".to_string(),
                recommendation: "Address critical violations before implementing new features".to_string(),
                reasoning: "Technical debt compounds over time and becomes harder to address".to_string(),
                expected_impact: 0.7,
                implementation_complexity: "Low to Medium".to_string(),
                related_concepts: vec!["Technical Debt Management".to_string(), "Code Quality".to_string()],
            });
        }
        
        Ok(recommendations)
    }
    
    /// Analyze technical debt
    fn analyze_technical_debt(&self, metrics: &CohesionMetrics, violations: &[CohesionViolation]) -> CohesionResult<TechnicalDebtAnalysis> {
        let debt_level = if metrics.overall_score < 50.0 {
            DebtLevel::Critical
        } else if metrics.overall_score < 70.0 {
            DebtLevel::High
        } else if metrics.overall_score < 85.0 {
            DebtLevel::Moderate
        } else {
            DebtLevel::Low
        };
        
        let mut debt_categories = HashMap::new();
        debt_categories.insert("Organizational Debt".to_string(), 100.0 - metrics.business_cohesion);
        debt_categories.insert("Semantic Debt".to_string(), 100.0 - metrics.semantic_cohesion);
        debt_categories.insert("Structural Debt".to_string(), 100.0 - metrics.type_cohesion);
        debt_categories.insert("Dependency Debt".to_string(), 100.0 - metrics.dependency_cohesion);
        
        let hotspots: Vec<_> = violations.iter()
            .filter(|v| matches!(v.severity, crate::ViolationSeverity::Error | crate::ViolationSeverity::Warning))
            .map(|v| DebtHotspot {
                location: format!("{:?}", v.violation_type),
                debt_type: "Cohesion Violation".to_string(),
                severity: v.impact_score * 10.0,
                description: v.description.clone(),
                remediation: v.suggested_fix.clone(),
            })
            .collect();
        
        Ok(TechnicalDebtAnalysis {
            debt_level,
            debt_categories,
            hotspots,
            repayment_strategy: vec![
                "Address critical violations first".to_string(),
                "Improve weakest cohesion metrics".to_string(),
                "Establish architectural guidelines".to_string(),
            ],
            debt_trend: TrendDirection::Stable, // No historical data
        })
    }
    
    /// Generate quality metrics for AI
    fn generate_quality_metrics(&self, metrics: &CohesionMetrics) -> CohesionResult<AIQualityMetrics> {
        Ok(AIQualityMetrics {
            maintainability_index: metrics.overall_score,
            testability_score: metrics.business_cohesion, // Well-defined responsibilities are easier to test
            readability_score: metrics.semantic_cohesion, // Good naming improves readability
            modularity_score: (metrics.type_cohesion + metrics.dependency_cohesion) / 2.0,
            reusability_potential: metrics.business_cohesion, // Clear capabilities are more reusable
            evolution_capability: metrics.overall_score * 0.9, // Good cohesion enables evolution
        })
    }
    
    /// Create AI metadata
    fn create_ai_metadata(&self) -> CohesionResult<AIAnalysisMetadata> {
        Ok(AIAnalysisMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            analysis_version: env!("CARGO_PKG_VERSION").to_string(),
            confidence_level: 0.85,
            analysis_scope: "Program-level cohesion analysis".to_string(),
            processing_recommendations: vec![
                "Use structured data for automated decision making".to_string(),
                "Focus on risk_level and primary_concerns for prioritization".to_string(),
                "Leverage architectural_patterns for design guidance".to_string(),
            ],
            format_version: "1.0".to_string(),
        })
    }
}

impl TrendAnalysis {
    /// Create empty trend analysis
    fn empty() -> Self {
        Self {
            cohesion_trajectory: TrajectoryAssessment::Unknown,
            complexity_trends: Vec::new(),
            dependency_evolution: DependencyTrend {
                overall_health: DependencyHealth::Moderate,
                growth_rate: 0.0,
                coupling_trends: Vec::new(),
                recommendations: Vec::new(),
            },
            predictive_insights: Vec::new(),
        }
    }
}

impl PatternDatabase {
    /// Create new pattern database
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }
}

impl Default for AIInsightConfig {
    fn default() -> Self {
        Self {
            enable_pattern_detection: true,
            enable_trend_analysis: true,
            enable_contextual_recommendations: true,
            verbosity: AIInsightVerbosity::Standard,
            max_insights: 50,
        }
    }
}

impl Default for AIInsightGenerator {
    fn default() -> Self {
        Self::new()
    }
} 