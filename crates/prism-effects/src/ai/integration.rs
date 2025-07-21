//! AI integration for effect system
//!
//! This module provides AI-comprehensible analysis, explanations, and
//! metadata for the effect system. It generates structured data that
//! external AI systems (accessed via servers) can consume to understand
//! and reason about effects and capabilities.
//!
//! ## External AI Integration Model
//! 
//! This module does NOT execute AI models directly. Instead, it:
//! - Generates AI-readable metadata and analysis structures
//! - Provides interfaces for external AI systems to consume
//! - Enables AI tools to understand Prism code through structured data
//! - Supports AI-assisted development through external services

use crate::effects::{Effect, EffectRegistry, EffectDefinition};
use prism_ast::SecurityClassification;
use std::collections::{HashMap, HashSet};

/// AI integration component for the effect system
#[derive(Debug)]
pub struct AIEffectIntegration {
    /// Analysis engines
    pub analyzers: Vec<Box<dyn AIEffectAnalyzer>>,
    /// Explanation generators
    pub explainers: Vec<Box<dyn EffectExplainer>>,
    /// Recommendation engines
    pub recommenders: Vec<Box<dyn EffectRecommender>>,
    /// Configuration
    pub config: AIIntegrationConfig,
}

impl AIEffectIntegration {
    /// Create a new AI integration component
    pub fn new() -> Self {
        let mut integration = Self {
            analyzers: Vec::new(),
            explainers: Vec::new(),
            recommenders: Vec::new(),
            config: AIIntegrationConfig::default(),
        };
        integration.register_default_components();
        integration
    }

    /// Analyze effects for AI comprehension
    pub fn analyze_effects(
        &self,
        effects: &[Effect],
        registry: &EffectRegistry,
    ) -> AIEffectAnalysis {
        let mut analysis = AIEffectAnalysis::new();

        // Run all analyzers
        for analyzer in &self.analyzers {
            let analyzer_result = analyzer.analyze(effects, registry);
            analysis.merge_analysis(analyzer_result);
        }

        // Generate explanations
        for explainer in &self.explainers {
            let explanations = explainer.explain(effects, registry);
            analysis.add_explanations(explanations);
        }

        // Generate recommendations
        for recommender in &self.recommenders {
            let recommendations = recommender.recommend(effects, registry);
            analysis.add_recommendations(recommendations);
        }

        analysis
    }

    /// Register default AI components
    fn register_default_components(&mut self) {
        // Security analyzer
        self.analyzers.push(Box::new(SecurityAnalyzer::new()));

        // Performance analyzer
        self.analyzers.push(Box::new(PerformanceAnalyzer::new()));

        // Business impact analyzer
        self.analyzers.push(Box::new(BusinessImpactAnalyzer::new()));

        // Natural language explainer
        self.explainers.push(Box::new(NaturalLanguageExplainer::new()));

        // Technical explainer
        self.explainers.push(Box::new(TechnicalExplainer::new()));

        // Security recommender
        self.recommenders.push(Box::new(SecurityRecommender::new()));

        // Performance recommender
        self.recommenders.push(Box::new(PerformanceRecommender::new()));
    }
}

impl Default for AIEffectIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for AI integration
#[derive(Debug, Clone)]
pub struct AIIntegrationConfig {
    /// Enable natural language explanations
    pub enable_natural_language: bool,
    /// Enable technical explanations
    pub enable_technical_explanations: bool,
    /// Enable security analysis
    pub enable_security_analysis: bool,
    /// Enable performance analysis
    pub enable_performance_analysis: bool,
    /// Generate recommendations
    pub generate_recommendations: bool,
    /// Confidence threshold for including analysis
    pub confidence_threshold: f64,
}

impl Default for AIIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_natural_language: true,
            enable_technical_explanations: true,
            enable_security_analysis: true,
            enable_performance_analysis: true,
            generate_recommendations: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Comprehensive AI analysis of effects
#[derive(Debug, Default)]
pub struct AIEffectAnalysis {
    /// Security analysis results
    pub security_analysis: SecurityAnalysisResult,
    /// Performance analysis results
    pub performance_analysis: PerformanceAnalysisResult,
    /// Business impact analysis
    pub business_impact: BusinessImpactAnalysis,
    /// Natural language explanations
    pub explanations: Vec<EffectExplanation>,
    /// AI-generated recommendations
    pub recommendations: Vec<EffectRecommendation>,
    /// Overall risk assessment
    pub risk_assessment: RiskAssessment,
    /// Compliance information
    pub compliance: ComplianceAnalysis,
}

impl AIEffectAnalysis {
    /// Create a new AI effect analysis
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another analysis into this one
    pub fn merge_analysis(&mut self, other: AIEffectAnalysis) {
        self.security_analysis.merge(other.security_analysis);
        self.performance_analysis.merge(other.performance_analysis);
        self.business_impact.merge(other.business_impact);
        self.explanations.extend(other.explanations);
        self.recommendations.extend(other.recommendations);
        self.risk_assessment.merge(other.risk_assessment);
        self.compliance.merge(other.compliance);
    }

    /// Add explanations to the analysis
    pub fn add_explanations(&mut self, explanations: Vec<EffectExplanation>) {
        self.explanations.extend(explanations);
    }

    /// Add recommendations to the analysis
    pub fn add_recommendations(&mut self, recommendations: Vec<EffectRecommendation>) {
        self.recommendations.extend(recommendations);
    }

    /// Generate a summary for AI consumption
    pub fn generate_ai_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("Effect Analysis Summary:\n");
        summary.push_str(&format!("- Security Risk Level: {}\n", self.risk_assessment.security_risk_level));
        summary.push_str(&format!("- Performance Impact: {}\n", self.performance_analysis.overall_impact));
        summary.push_str(&format!("- Business Criticality: {}\n", self.business_impact.criticality_level));

        if !self.recommendations.is_empty() {
            summary.push_str("\nKey Recommendations:\n");
            for (i, rec) in self.recommendations.iter().take(3).enumerate() {
                summary.push_str(&format!("{}. {}\n", i + 1, rec.summary));
            }
        }

        summary
    }
}

/// Trait for AI effect analyzers
pub trait AIEffectAnalyzer: std::fmt::Debug + Send + Sync {
    /// Analyze effects and return AI-comprehensible results
    fn analyze(&self, effects: &[Effect], registry: &EffectRegistry) -> AIEffectAnalysis;

    /// Get the name of this analyzer
    fn name(&self) -> &str;

    /// Get the confidence level of this analyzer
    fn confidence(&self) -> f64;
}

/// Trait for effect explainers
pub trait EffectExplainer: std::fmt::Debug + Send + Sync {
    /// Generate explanations for effects
    fn explain(&self, effects: &[Effect], registry: &EffectRegistry) -> Vec<EffectExplanation>;

    /// Get the name of this explainer
    fn name(&self) -> &str;
}

/// Trait for effect recommenders
pub trait EffectRecommender: std::fmt::Debug + Send + Sync {
    /// Generate recommendations for effect usage
    fn recommend(&self, effects: &[Effect], registry: &EffectRegistry) -> Vec<EffectRecommendation>;

    /// Get the name of this recommender
    fn name(&self) -> &str;
}

/// Security analyzer for effects
#[derive(Debug)]
pub struct SecurityAnalyzer {
    /// Security patterns to detect
    patterns: Vec<SecurityPattern>,
}

impl SecurityAnalyzer {
    /// Create a new security analyzer
    pub fn new() -> Self {
        Self {
            patterns: vec![
                SecurityPattern {
                    name: "UnencryptedNetworkTraffic".to_string(),
                    description: "Network traffic without encryption".to_string(),
                    risk_level: RiskLevel::High,
                    detection_fn: Box::new(|effects, _registry| {
                        effects.iter().any(|e| e.definition.contains("Network")) &&
                        !effects.iter().any(|e| e.definition.contains("Encryption"))
                    }),
                },
                SecurityPattern {
                    name: "DatabaseWithoutTransaction".to_string(),
                    description: "Database operations without transaction protection".to_string(),
                    risk_level: RiskLevel::Medium,
                    detection_fn: Box::new(|effects, _registry| {
                        effects.iter().any(|e| e.definition.contains("Database.Query")) &&
                        !effects.iter().any(|e| e.definition.contains("Database.Transaction"))
                    }),
                },
                SecurityPattern {
                    name: "UnsafeOperationWithoutApproval".to_string(),
                    description: "Unsafe operations without explicit approval".to_string(),
                    risk_level: RiskLevel::Critical,
                    detection_fn: Box::new(|effects, _registry| {
                        effects.iter().any(|e| {
                            e.definition.starts_with("Unsafe") &&
                            !e.metadata.ai_context.as_ref()
                                .map(|ctx| ctx.contains("approved"))
                                .unwrap_or(false)
                        })
                    }),
                },
            ],
        }
    }
}

impl AIEffectAnalyzer for SecurityAnalyzer {
    fn analyze(&self, effects: &[Effect], registry: &EffectRegistry) -> AIEffectAnalysis {
        let mut analysis = AIEffectAnalysis::new();
        let mut security_issues = Vec::new();

        // Check each security pattern
        for pattern in &self.patterns {
            if (pattern.detection_fn)(effects, registry) {
                security_issues.push(SecurityIssue {
                    issue_type: pattern.name.clone(),
                    description: pattern.description.clone(),
                    risk_level: pattern.risk_level.clone(),
                    affected_effects: effects.iter()
                        .map(|e| e.definition.clone())
                        .collect(),
                    mitigation_suggestions: self.get_mitigation_suggestions(&pattern.name),
                });
            }
        }

        // Calculate overall security risk
        let max_risk = security_issues.iter()
            .map(|issue| &issue.risk_level)
            .max()
            .unwrap_or(&RiskLevel::Low);

        analysis.security_analysis = SecurityAnalysisResult {
            security_risk_level: max_risk.clone(),
            identified_issues: security_issues,
            security_score: self.calculate_security_score(effects),
            recommendations: self.generate_security_recommendations(effects),
        };

        analysis
    }

    fn name(&self) -> &str {
        "SecurityAnalyzer"
    }

    fn confidence(&self) -> f64 {
        0.85
    }
}

impl SecurityAnalyzer {
    /// Get mitigation suggestions for a security issue
    fn get_mitigation_suggestions(&self, issue_type: &str) -> Vec<String> {
        match issue_type {
            "UnencryptedNetworkTraffic" => vec![
                "Use HTTPS/TLS for network communications".to_string(),
                "Implement end-to-end encryption".to_string(),
                "Consider using VPN for sensitive data".to_string(),
            ],
            "DatabaseWithoutTransaction" => vec![
                "Wrap database operations in transactions".to_string(),
                "Implement proper rollback mechanisms".to_string(),
                "Use connection pooling with transaction support".to_string(),
            ],
            "UnsafeOperationWithoutApproval" => vec![
                "Obtain explicit security approval".to_string(),
                "Document safety justification".to_string(),
                "Implement additional safety checks".to_string(),
            ],
            _ => vec!["Review security implications".to_string()],
        }
    }

    /// Calculate overall security score
    fn calculate_security_score(&self, effects: &[Effect]) -> f64 {
        let mut score: f64 = 100.0;

        // Deduct points for risky effects
        for effect in effects {
            if effect.definition.starts_with("Unsafe") {
                score -= 30.0;
            } else if effect.definition.contains("Network") &&
                      effect.metadata.security_classification != SecurityClassification::Public {
                score -= 15.0;
            } else if effect.definition.contains("Database") {
                score -= 10.0;
            }
        }

        score.max(0.0)
    }

    /// Generate security recommendations
    fn generate_security_recommendations(&self, effects: &[Effect]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if effects.iter().any(|e| e.definition.contains("Network")) {
            recommendations.push("Consider implementing network security monitoring".to_string());
        }

        if effects.iter().any(|e| e.definition.contains("Database")) {
            recommendations.push("Implement database access auditing".to_string());
        }

        if effects.iter().any(|e| e.definition.starts_with("AI")) {
            recommendations.push("Enable AI safety monitoring and content filtering".to_string());
        }

        recommendations
    }
}

/// Performance analyzer for effects
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Performance patterns to analyze
    patterns: Vec<PerformancePattern>,
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                PerformancePattern {
                    name: "HighLatencyOperations".to_string(),
                    description: "Operations that may introduce high latency".to_string(),
                    impact_level: ImpactLevel::High,
                    detection_fn: Box::new(|effects, _registry| {
                        effects.iter().any(|e| {
                            e.definition.contains("Network") || 
                            e.definition.contains("Database") ||
                            e.definition.contains("ExternalAI.DataExport")
                        })
                    }),
                },
                PerformancePattern {
                    name: "ResourceIntensiveOperations".to_string(),
                    description: "Operations that consume significant resources".to_string(),
                    impact_level: ImpactLevel::Medium,
                    detection_fn: Box::new(|effects, _registry| {
                        effects.iter().any(|e| {
                            e.definition.contains("Cryptography") ||
                            e.definition.contains("ExternalAI.MetadataGeneration")
                        })
                    }),
                },
            ],
        }
    }
}

impl AIEffectAnalyzer for PerformanceAnalyzer {
    fn analyze(&self, effects: &[Effect], registry: &EffectRegistry) -> AIEffectAnalysis {
        let mut analysis = AIEffectAnalysis::new();
        let mut performance_issues = Vec::new();

        // Check performance patterns
        for pattern in &self.patterns {
            if (pattern.detection_fn)(effects, registry) {
                performance_issues.push(PerformanceIssue {
                    issue_type: pattern.name.clone(),
                    description: pattern.description.clone(),
                    impact_level: pattern.impact_level.clone(),
                    optimization_suggestions: self.get_optimization_suggestions(&pattern.name),
                });
            }
        }

        // Calculate performance impact
        let overall_impact = if performance_issues.iter().any(|i| i.impact_level == ImpactLevel::High) {
            ImpactLevel::High
        } else if performance_issues.iter().any(|i| i.impact_level == ImpactLevel::Medium) {
            ImpactLevel::Medium
        } else {
            ImpactLevel::Low
        };

        analysis.performance_analysis = PerformanceAnalysisResult {
            overall_impact,
            identified_issues: performance_issues,
            performance_score: self.calculate_performance_score(effects),
            optimization_recommendations: self.generate_optimization_recommendations(effects),
        };

        analysis
    }

    fn name(&self) -> &str {
        "PerformanceAnalyzer"
    }

    fn confidence(&self) -> f64 {
        0.80
    }
}

impl PerformanceAnalyzer {
    fn get_optimization_suggestions(&self, issue_type: &str) -> Vec<String> {
        match issue_type {
            "HighLatencyOperations" => vec![
                "Implement caching strategies".to_string(),
                "Use asynchronous processing".to_string(),
                "Consider connection pooling".to_string(),
            ],
            "ResourceIntensiveOperations" => vec![
                "Implement resource throttling".to_string(),
                "Use batch processing".to_string(),
                "Consider hardware acceleration".to_string(),
            ],
            _ => vec!["Review performance characteristics".to_string()],
        }
    }

    fn calculate_performance_score(&self, effects: &[Effect]) -> f64 {
        let mut score: f64 = 100.0;

        for effect in effects {
            if effect.definition.contains("ExternalAI.MetadataGeneration") {
                score -= 25.0;
            } else if effect.definition.contains("Network") {
                score -= 15.0;
            } else if effect.definition.contains("Database") {
                score -= 10.0;
            } else if effect.definition.contains("Cryptography") {
                score -= 8.0;
            }
        }

        score.max(0.0)
    }

    fn generate_optimization_recommendations(&self, effects: &[Effect]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if effects.len() > 5 {
            recommendations.push("Consider effect composition to reduce overhead".to_string());
        }

        if effects.iter().any(|e| e.definition.contains("Database")) {
            recommendations.push("Implement database connection pooling".to_string());
        }

        recommendations
    }
}

/// Business impact analyzer
#[derive(Debug)]
pub struct BusinessImpactAnalyzer;

impl BusinessImpactAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl AIEffectAnalyzer for BusinessImpactAnalyzer {
    fn analyze(&self, effects: &[Effect], _registry: &EffectRegistry) -> AIEffectAnalysis {
        let mut analysis = AIEffectAnalysis::new();

        // Analyze business criticality
        let criticality = if effects.iter().any(|e| e.definition.contains("Database") || e.definition.contains("Payment")) {
            CriticalityLevel::High
        } else if effects.iter().any(|e| e.definition.contains("Network") || e.definition.contains("AI")) {
            CriticalityLevel::Medium
        } else {
            CriticalityLevel::Low
        };

        analysis.business_impact = BusinessImpactAnalysis {
            criticality_level: criticality,
            affected_business_processes: self.identify_affected_processes(effects),
            compliance_requirements: self.identify_compliance_requirements(effects),
            business_risk_factors: self.identify_risk_factors(effects),
        };

        analysis
    }

    fn name(&self) -> &str {
        "BusinessImpactAnalyzer"
    }

    fn confidence(&self) -> f64 {
        0.75
    }
}

impl BusinessImpactAnalyzer {
    fn identify_affected_processes(&self, effects: &[Effect]) -> Vec<String> {
        let mut processes = Vec::new();

        for effect in effects {
            if effect.definition.contains("Database") {
                processes.push("Data Management".to_string());
            }
            if effect.definition.contains("Network") {
                processes.push("External Communications".to_string());
            }
            if effect.definition.contains("AI") {
                processes.push("Decision Support".to_string());
            }
            if effect.definition.contains("Cryptography") {
                processes.push("Security Operations".to_string());
            }
        }

        processes.sort();
        processes.dedup();
        processes
    }

    fn identify_compliance_requirements(&self, effects: &[Effect]) -> Vec<String> {
        let mut requirements = Vec::new();

        for effect in effects {
            match effect.metadata.security_classification {
                SecurityClassification::Confidential | SecurityClassification::TopSecret => {
                    requirements.push("Data Protection Compliance".to_string());
                }
                _ => {}
            }

            if effect.definition.contains("AI") {
                requirements.push("AI Ethics Guidelines".to_string());
            }
            if effect.definition.contains("Cryptography") {
                requirements.push("Cryptographic Standards".to_string());
            }
        }

        requirements.sort();
        requirements.dedup();
        requirements
    }

    fn identify_risk_factors(&self, effects: &[Effect]) -> Vec<String> {
        let mut risks = Vec::new();

        if effects.iter().any(|e| e.definition.starts_with("Unsafe")) {
            risks.push("System Stability Risk".to_string());
        }
        if effects.iter().any(|e| e.definition.contains("Network")) {
            risks.push("Data Breach Risk".to_string());
        }
        if effects.iter().any(|e| e.definition.contains("AI")) {
            risks.push("Algorithmic Bias Risk".to_string());
        }

        risks
    }
}

/// Natural language explainer
#[derive(Debug)]
pub struct NaturalLanguageExplainer;

impl NaturalLanguageExplainer {
    pub fn new() -> Self {
        Self
    }
}

impl EffectExplainer for NaturalLanguageExplainer {
    fn explain(&self, effects: &[Effect], registry: &EffectRegistry) -> Vec<EffectExplanation> {
        let mut explanations = Vec::new();

        // Generate overall explanation
        if !effects.is_empty() {
            let summary = self.generate_summary(effects);
            explanations.push(EffectExplanation {
                explanation_type: ExplanationType::Summary,
                title: "What this code does".to_string(),
                content: summary,
                confidence: 0.9,
                audience: ExplanationAudience::General,
            });
        }

        // Generate individual effect explanations
        for effect in effects {
            if let Some(effect_def) = registry.get_effect(&effect.definition) {
                let explanation = self.explain_single_effect(effect, effect_def);
                explanations.push(explanation);
            }
        }

        explanations
    }

    fn name(&self) -> &str {
        "NaturalLanguageExplainer"
    }
}

impl NaturalLanguageExplainer {
    fn generate_summary(&self, effects: &[Effect]) -> String {
        let mut summary = String::new();

        summary.push_str("This code performs the following operations:\n");

        let mut effect_categories: HashMap<String, Vec<String>> = HashMap::new();
        for effect in effects {
            let category = self.categorize_effect(&effect.definition);
            effect_categories.entry(category).or_default().push(effect.definition.clone());
        }

        for (category, effect_names) in effect_categories {
            summary.push_str(&format!("- {}: {}\n", category, effect_names.join(", ")));
        }

        summary
    }

    fn categorize_effect(&self, effect_name: &str) -> String {
        if effect_name.contains("Database") {
            "Database Operations".to_string()
        } else if effect_name.contains("Network") {
            "Network Communications".to_string()
        } else if effect_name.contains("FileSystem") {
            "File System Access".to_string()
        } else if effect_name.contains("AI") {
            "AI Operations".to_string()
        } else if effect_name.contains("Cryptography") {
            "Security Operations".to_string()
        } else {
            "Other Operations".to_string()
        }
    }

    fn explain_single_effect(&self, effect: &Effect, effect_def: &EffectDefinition) -> EffectExplanation {
        let content = format!(
            "{}\n\nThis operation {}",
            effect_def.description,
            effect_def.ai_context.as_deref().unwrap_or("performs system-level operations")
        );

        EffectExplanation {
            explanation_type: ExplanationType::DetailedDescription,
            title: format!("Effect: {}", effect.definition),
            content,
            confidence: 0.85,
            audience: ExplanationAudience::Developer,
        }
    }
}

/// Technical explainer for detailed technical information
#[derive(Debug)]
pub struct TechnicalExplainer;

impl TechnicalExplainer {
    pub fn new() -> Self {
        Self
    }
}

impl EffectExplainer for TechnicalExplainer {
    fn explain(&self, effects: &[Effect], registry: &EffectRegistry) -> Vec<EffectExplanation> {
        let mut explanations = Vec::new();

        // Generate technical overview
        let technical_overview = self.generate_technical_overview(effects, registry);
        explanations.push(technical_overview);

        explanations
    }

    fn name(&self) -> &str {
        "TechnicalExplainer"
    }
}

impl TechnicalExplainer {
    fn generate_technical_overview(&self, effects: &[Effect], registry: &EffectRegistry) -> EffectExplanation {
        let mut content = String::new();

        content.push_str("Technical Analysis:\n\n");

        // Effect composition analysis
        content.push_str("Effect Composition:\n");
        for effect in effects {
            content.push_str(&format!("- {} ({})\n", 
                effect.definition,
                if effect.metadata.inferred { "inferred" } else { "explicit" }
            ));
        }

        // Capability requirements
        content.push_str("\nCapability Requirements:\n");
        let mut all_capabilities = HashSet::new();
        for effect in effects {
            if let Some(effect_def) = registry.get_effect(&effect.definition) {
                for (cap_name, _permissions) in &effect_def.capability_requirements {
                    all_capabilities.insert(cap_name);
                }
            }
        }
        for capability in &all_capabilities {
            content.push_str(&format!("- {}\n", capability));
        }

        EffectExplanation {
            explanation_type: ExplanationType::TechnicalAnalysis,
            title: "Technical Overview".to_string(),
            content,
            confidence: 0.95,
            audience: ExplanationAudience::Technical,
        }
    }
}

/// Security recommender
#[derive(Debug)]
pub struct SecurityRecommender;

impl SecurityRecommender {
    pub fn new() -> Self {
        Self
    }
}

impl EffectRecommender for SecurityRecommender {
    fn recommend(&self, effects: &[Effect], _registry: &EffectRegistry) -> Vec<EffectRecommendation> {
        let mut recommendations = Vec::new();

        // Check for common security issues
        if effects.iter().any(|e| e.definition.contains("Network")) {
            recommendations.push(EffectRecommendation {
                recommendation_type: RecommendationType::Security,
                priority: RecommendationPriority::High,
                summary: "Implement network security measures".to_string(),
                description: "Network operations detected. Consider implementing encryption, authentication, and monitoring.".to_string(),
                implementation_steps: vec![
                    "Enable TLS/HTTPS for all network communications".to_string(),
                    "Implement proper authentication mechanisms".to_string(),
                    "Add network monitoring and logging".to_string(),
                ],
                estimated_effort: ImplementationEffort::Medium,
            });
        }

        if effects.iter().any(|e| e.definition.starts_with("Unsafe")) {
            recommendations.push(EffectRecommendation {
                recommendation_type: RecommendationType::Security,
                priority: RecommendationPriority::Critical,
                summary: "Review unsafe operations".to_string(),
                description: "Unsafe operations require careful review and approval.".to_string(),
                implementation_steps: vec![
                    "Document the necessity of unsafe operations".to_string(),
                    "Implement additional safety checks".to_string(),
                    "Get security team approval".to_string(),
                ],
                estimated_effort: ImplementationEffort::High,
            });
        }

        recommendations
    }

    fn name(&self) -> &str {
        "SecurityRecommender"
    }
}

/// Performance recommender
#[derive(Debug)]
pub struct PerformanceRecommender;

impl PerformanceRecommender {
    pub fn new() -> Self {
        Self
    }
}

impl EffectRecommender for PerformanceRecommender {
    fn recommend(&self, effects: &[Effect], _registry: &EffectRegistry) -> Vec<EffectRecommendation> {
        let mut recommendations = Vec::new();

        if effects.len() > 10 {
            recommendations.push(EffectRecommendation {
                recommendation_type: RecommendationType::Performance,
                priority: RecommendationPriority::Medium,
                summary: "Optimize effect composition".to_string(),
                description: "Large number of effects detected. Consider optimizing for better performance.".to_string(),
                implementation_steps: vec![
                    "Analyze effect dependencies".to_string(),
                    "Combine related effects".to_string(),
                    "Implement batching where possible".to_string(),
                ],
                estimated_effort: ImplementationEffort::Medium,
            });
        }

        recommendations
    }

    fn name(&self) -> &str {
        "PerformanceRecommender"
    }
}

// Supporting types and structures...

#[derive(Debug, Clone, Default)]
pub struct SecurityAnalysisResult {
    pub security_risk_level: RiskLevel,
    pub identified_issues: Vec<SecurityIssue>,
    pub security_score: f64,
    pub recommendations: Vec<String>,
}

impl SecurityAnalysisResult {
    pub fn merge(&mut self, other: SecurityAnalysisResult) {
        if other.security_risk_level > self.security_risk_level {
            self.security_risk_level = other.security_risk_level;
        }
        self.identified_issues.extend(other.identified_issues);
        self.security_score = (self.security_score + other.security_score) / 2.0;
        self.recommendations.extend(other.recommendations);
    }
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceAnalysisResult {
    pub overall_impact: ImpactLevel,
    pub identified_issues: Vec<PerformanceIssue>,
    pub performance_score: f64,
    pub optimization_recommendations: Vec<String>,
}

impl PerformanceAnalysisResult {
    pub fn merge(&mut self, other: PerformanceAnalysisResult) {
        if other.overall_impact > self.overall_impact {
            self.overall_impact = other.overall_impact;
        }
        self.identified_issues.extend(other.identified_issues);
        self.performance_score = (self.performance_score + other.performance_score) / 2.0;
        self.optimization_recommendations.extend(other.optimization_recommendations);
    }
}

#[derive(Debug, Clone, Default)]
pub struct BusinessImpactAnalysis {
    pub criticality_level: CriticalityLevel,
    pub affected_business_processes: Vec<String>,
    pub compliance_requirements: Vec<String>,
    pub business_risk_factors: Vec<String>,
}

impl BusinessImpactAnalysis {
    pub fn merge(&mut self, other: BusinessImpactAnalysis) {
        if other.criticality_level > self.criticality_level {
            self.criticality_level = other.criticality_level;
        }
        self.affected_business_processes.extend(other.affected_business_processes);
        self.compliance_requirements.extend(other.compliance_requirements);
        self.business_risk_factors.extend(other.business_risk_factors);
    }
}

#[derive(Debug, Clone, Default)]
pub struct RiskAssessment {
    pub security_risk_level: RiskLevel,
    pub performance_risk_level: RiskLevel,
    pub business_risk_level: RiskLevel,
    pub overall_risk_score: f64,
}

impl RiskAssessment {
    pub fn merge(&mut self, other: RiskAssessment) {
        if other.security_risk_level > self.security_risk_level {
            self.security_risk_level = other.security_risk_level;
        }
        if other.performance_risk_level > self.performance_risk_level {
            self.performance_risk_level = other.performance_risk_level;
        }
        if other.business_risk_level > self.business_risk_level {
            self.business_risk_level = other.business_risk_level;
        }
        self.overall_risk_score = (self.overall_risk_score + other.overall_risk_score) / 2.0;
    }
}

#[derive(Debug, Clone, Default)]
pub struct ComplianceAnalysis {
    pub required_standards: Vec<String>,
    pub compliance_gaps: Vec<String>,
    pub remediation_actions: Vec<String>,
}

impl ComplianceAnalysis {
    pub fn merge(&mut self, other: ComplianceAnalysis) {
        self.required_standards.extend(other.required_standards);
        self.compliance_gaps.extend(other.compliance_gaps);
        self.remediation_actions.extend(other.remediation_actions);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RiskLevel {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "Low"),
            RiskLevel::Medium => write!(f, "Medium"),
            RiskLevel::High => write!(f, "High"),
            RiskLevel::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ImpactLevel {
    #[default]
    Low,
    Medium,
    High,
}

impl std::fmt::Display for ImpactLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImpactLevel::Low => write!(f, "Low"),
            ImpactLevel::Medium => write!(f, "Medium"),
            ImpactLevel::High => write!(f, "High"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum CriticalityLevel {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for CriticalityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CriticalityLevel::Low => write!(f, "Low"),
            CriticalityLevel::Medium => write!(f, "Medium"),
            CriticalityLevel::High => write!(f, "High"),
            CriticalityLevel::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SecurityIssue {
    pub issue_type: String,
    pub description: String,
    pub risk_level: RiskLevel,
    pub affected_effects: Vec<String>,
    pub mitigation_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceIssue {
    pub issue_type: String,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub optimization_suggestions: Vec<String>,
}

pub struct SecurityPattern {
    pub name: String,
    pub description: String,
    pub risk_level: RiskLevel,
    pub detection_fn: Box<dyn Fn(&[Effect], &EffectRegistry) -> bool + Send + Sync>,
}

impl std::fmt::Debug for SecurityPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityPattern")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("risk_level", &self.risk_level)
            .field("detection_fn", &"<function>")
            .finish()
    }
}

impl Clone for SecurityPattern {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            risk_level: self.risk_level.clone(),
            detection_fn: Box::new(|_, _| true), // Simplified clone
        }
    }
}

pub struct PerformancePattern {
    pub name: String,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub detection_fn: Box<dyn Fn(&[Effect], &EffectRegistry) -> bool + Send + Sync>,
}

impl std::fmt::Debug for PerformancePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerformancePattern")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("impact_level", &self.impact_level)
            .field("detection_fn", &"<function>")
            .finish()
    }
}

impl Clone for PerformancePattern {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            impact_level: self.impact_level.clone(),
            detection_fn: Box::new(|_, _| true), // Simplified clone
        }
    }
}

#[derive(Debug, Clone)]
pub struct EffectExplanation {
    pub explanation_type: ExplanationType,
    pub title: String,
    pub content: String,
    pub confidence: f64,
    pub audience: ExplanationAudience,
}

#[derive(Debug, Clone)]
pub enum ExplanationType {
    Summary,
    DetailedDescription,
    TechnicalAnalysis,
    BusinessImpact,
    SecurityAnalysis,
    PerformanceAnalysis,
}

#[derive(Debug, Clone)]
pub enum ExplanationAudience {
    General,
    Developer,
    Technical,
    Business,
    Security,
}

#[derive(Debug, Clone)]
pub struct EffectRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: RecommendationPriority,
    pub summary: String,
    pub description: String,
    pub implementation_steps: Vec<String>,
    pub estimated_effort: ImplementationEffort,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    Security,
    Performance,
    Maintainability,
    Compliance,
    BestPractice,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_integration_creation() {
        let integration = AIEffectIntegration::new();
        assert!(!integration.analyzers.is_empty());
        assert!(!integration.explainers.is_empty());
        assert!(!integration.recommenders.is_empty());
    }

    #[test]
    fn test_security_analyzer() {
        let analyzer = SecurityAnalyzer::new();
        let registry = EffectRegistry::new();
        
        let effects = vec![
            Effect::new("IO.Network.Connect".to_string(), prism_common::span::Span::dummy()),
        ];
        
        let analysis = analyzer.analyze(&effects, &registry);
        assert!(!analysis.security_analysis.identified_issues.is_empty());
    }

    #[test]
    fn test_natural_language_explainer() {
        let explainer = NaturalLanguageExplainer::new();
        let registry = EffectRegistry::new();
        
        let effects = vec![
            Effect::new("Database.Query".to_string(), prism_common::span::Span::dummy()),
        ];
        
        let explanations = explainer.explain(&effects, &registry);
        assert!(!explanations.is_empty());
        assert!(explanations[0].content.contains("Database Operations"));
    }
} 