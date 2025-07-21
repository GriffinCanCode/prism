//! Symbol Metadata and AI Context
//!
//! This module provides comprehensive metadata handling for symbols, including
//! AI-comprehensible context, documentation, business rules, and semantic annotations.
//! It integrates with the broader AI metadata system from prism-common.
//!
//! ## Conceptual Responsibility
//! 
//! This module handles ONE thing: "Symbol Metadata and Context Management"
//! - AI-readable symbol context and descriptions
//! - Documentation and help text
//! - Business rule associations
//! - Semantic annotations and tags
//! - Performance and optimization metadata
//! 
//! It does NOT handle:
//! - Symbol storage (delegated to table.rs)
//! - Symbol classification (delegated to kinds.rs)
//! - Symbol resolution (delegated to resolution subsystem)

use prism_common::{span::Span, ai_metadata::{AIMetadata, SemanticContextEntry, BusinessRuleEntry}};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive symbol metadata with AI integration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolMetadata {
    /// AI-comprehensible context for this symbol
    pub ai_context: Option<AISymbolContext>,
    
    /// Human-readable documentation
    pub documentation: Option<SymbolDocumentation>,
    
    /// Business rules and domain context
    pub business_context: Option<BusinessContext>,
    
    /// Semantic annotations and tags
    pub semantic_annotations: Vec<SemanticAnnotation>,
    
    /// Performance and optimization metadata
    pub performance_metadata: Option<PerformanceMetadata>,
    
    /// Security and capability metadata
    pub security_metadata: Option<SecurityMetadata>,
    
    /// Quality metrics and assessments
    pub quality_metrics: Option<QualityMetrics>,
    
    /// Custom metadata fields
    pub custom_metadata: HashMap<String, String>,
    
    /// Metadata generation timestamp
    pub last_updated: Option<std::time::SystemTime>,
    
    /// Confidence in metadata accuracy (0.0 to 1.0)
    pub confidence: f64,
}

/// AI-comprehensible symbol context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISymbolContext {
    /// Primary purpose of this symbol
    pub purpose: String,
    
    /// Detailed description for AI understanding
    pub description: String,
    
    /// Usage examples and patterns
    pub usage_examples: Vec<UsageExample>,
    
    /// Common use cases
    pub use_cases: Vec<UseCase>,
    
    /// Related concepts and terminology
    pub related_concepts: Vec<String>,
    
    /// Domain-specific context
    pub domain_context: Option<DomainContext>,
    
    /// AI-generated insights
    pub ai_insights: Vec<AIInsight>,
    
    /// Complexity assessment
    pub complexity_assessment: Option<ComplexityAssessment>,
}

/// Human-readable documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDocumentation {
    /// Brief summary
    pub summary: String,
    
    /// Detailed description
    pub description: Option<String>,
    
    /// Parameter documentation (for functions)
    pub parameters: Vec<ParameterDoc>,
    
    /// Return value documentation
    pub returns: Option<ReturnDoc>,
    
    /// Examples of usage
    pub examples: Vec<CodeExample>,
    
    /// Notes and warnings
    pub notes: Vec<DocumentationNote>,
    
    /// See-also references
    pub see_also: Vec<String>,
    
    /// Author information
    pub author: Option<String>,
    
    /// Version information
    pub version: Option<String>,
    
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Business context and domain information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Business domain this symbol belongs to
    pub domain: String,
    
    /// Business rules that apply to this symbol
    pub business_rules: Vec<BusinessRule>,
    
    /// Stakeholders affected by this symbol
    pub stakeholders: Vec<String>,
    
    /// Compliance requirements
    pub compliance_requirements: Vec<ComplianceRequirement>,
    
    /// Business impact assessment
    pub business_impact: Option<BusinessImpact>,
    
    /// Cost and resource information
    pub resource_info: Option<ResourceInfo>,
}

/// Semantic annotation for symbol meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnnotation {
    /// Annotation type
    pub annotation_type: SemanticAnnotationType,
    
    /// Annotation value or description
    pub value: String,
    
    /// Confidence in this annotation (0.0 to 1.0)
    pub confidence: f64,
    
    /// Source of the annotation
    pub source: AnnotationSource,
    
    /// Context where this annotation applies
    pub context: Option<String>,
}

/// Types of semantic annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticAnnotationType {
    /// Intent or purpose annotation
    Intent,
    /// Behavior description
    Behavior,
    /// Precondition
    Precondition,
    /// Postcondition
    Postcondition,
    /// Invariant
    Invariant,
    /// Side effect description
    SideEffect,
    /// Performance characteristic
    Performance,
    /// Security implication
    Security,
    /// Business rule
    BusinessRule,
    /// Custom annotation
    Custom(String),
}

/// Source of semantic annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationSource {
    /// Explicitly written in code
    Explicit,
    /// Inferred by compiler
    Inferred,
    /// Generated by AI analysis
    AIGenerated,
    /// Added by external tool
    External(String),
    /// Derived from usage patterns
    UsagePattern,
}

/// Performance metadata for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetadata {
    /// Time complexity assessment
    pub time_complexity: Option<ComplexityClass>,
    
    /// Space complexity assessment
    pub space_complexity: Option<ComplexityClass>,
    
    /// Performance characteristics
    pub characteristics: Vec<PerformanceCharacteristic>,
    
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    
    /// Benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
    
    /// Resource usage patterns
    pub resource_usage: Vec<ResourceUsage>,
}

/// Security and capability metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetadata {
    /// Security classification
    pub security_level: SecurityLevel,
    
    /// Threat model considerations
    pub threat_model: Vec<ThreatConsideration>,
    
    /// Required permissions
    pub required_permissions: Vec<Permission>,
    
    /// Security implications
    pub security_implications: Vec<SecurityImplication>,
    
    /// Audit requirements
    pub audit_requirements: Vec<AuditRequirement>,
    
    /// Data sensitivity
    pub data_sensitivity: Option<DataSensitivity>,
}

/// Quality metrics and assessments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Code quality score (0.0 to 1.0)
    pub quality_score: f64,
    
    /// Maintainability assessment
    pub maintainability: QualityAssessment,
    
    /// Readability assessment
    pub readability: QualityAssessment,
    
    /// Testability assessment
    pub testability: QualityAssessment,
    
    /// Reusability assessment
    pub reusability: QualityAssessment,
    
    /// Technical debt indicators
    pub technical_debt: Vec<TechnicalDebtIndicator>,
    
    /// Code smells identified
    pub code_smells: Vec<CodeSmell>,
}

// Supporting types for comprehensive metadata

/// Usage example for AI understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    /// Example code or description
    pub example: String,
    /// Context where this example applies
    pub context: String,
    /// Explanation of the example
    pub explanation: Option<String>,
}

/// Use case description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseCase {
    /// Use case name
    pub name: String,
    /// Use case description
    pub description: String,
    /// Frequency of this use case
    pub frequency: UseCaseFrequency,
}

/// Use case frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UseCaseFrequency {
    VeryCommon,
    Common,
    Occasional,
    Rare,
}

/// Domain-specific context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainContext {
    /// Domain name (e.g., "finance", "healthcare", "gaming")
    pub domain: String,
    /// Domain-specific terminology
    pub terminology: HashMap<String, String>,
    /// Domain constraints
    pub constraints: Vec<String>,
    /// Domain best practices
    pub best_practices: Vec<String>,
}

/// AI-generated insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInsight {
    /// Type of insight
    pub insight_type: AIInsightType,
    /// Insight content
    pub content: String,
    /// Confidence in the insight (0.0 to 1.0)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Types of AI insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIInsightType {
    /// Pattern recognition
    PatternRecognition,
    /// Potential improvement
    Improvement,
    /// Risk identification
    Risk,
    /// Optimization opportunity
    Optimization,
    /// Design pattern match
    DesignPattern,
    /// Anti-pattern detection
    AntiPattern,
}

/// Complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAssessment {
    /// Cognitive complexity score
    pub cognitive_complexity: u32,
    /// Cyclomatic complexity
    pub cyclomatic_complexity: u32,
    /// Overall complexity rating
    pub complexity_rating: ComplexityRating,
    /// Complexity factors
    pub factors: Vec<ComplexityFactor>,
}

/// Complexity rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityRating {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Complexity factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactor {
    /// Factor name
    pub name: String,
    /// Factor contribution to complexity
    pub contribution: u32,
    /// Factor description
    pub description: String,
}

/// Parameter documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDoc {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: Option<String>,
    /// Parameter description
    pub description: String,
    /// Whether parameter is optional
    pub optional: bool,
    /// Default value if any
    pub default_value: Option<String>,
}

/// Return value documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnDoc {
    /// Return type
    pub return_type: Option<String>,
    /// Return value description
    pub description: String,
    /// Possible return values
    pub possible_values: Vec<String>,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Example title
    pub title: String,
    /// Example code
    pub code: String,
    /// Example explanation
    pub explanation: Option<String>,
    /// Example output or result
    pub output: Option<String>,
}

/// Documentation note
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationNote {
    /// Note type
    pub note_type: NoteType,
    /// Note content
    pub content: String,
}

/// Types of documentation notes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoteType {
    Note,
    Warning,
    Caution,
    Important,
    Tip,
    Example,
}

/// Business rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule enforcement level
    pub enforcement: crate::symbols::kinds::EnforcementLevel,
    /// Rule category
    pub category: String,
    /// Rule validation method
    pub validation: ValidationMethod,
}

/// Rule validation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMethod {
    Compile,
    Runtime,
    Manual,
    Automated,
}

/// Compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    /// Standard or regulation name
    pub standard: String,
    /// Specific requirement
    pub requirement: String,
    /// Compliance level required
    pub level: ComplianceLevel,
}

/// Compliance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceLevel {
    Required,
    Recommended,
    Optional,
}

/// Business impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    /// Impact level
    pub impact_level: ImpactLevel,
    /// Affected areas
    pub affected_areas: Vec<String>,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Mitigation strategies
    pub mitigations: Vec<String>,
}

/// Impact levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// Development cost
    pub development_cost: Option<String>,
    /// Maintenance cost
    pub maintenance_cost: Option<String>,
    /// Resource requirements
    pub resource_requirements: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

// Additional supporting types would continue here...
// For brevity, I'll include key types and leave placeholders for others

/// Complexity class for algorithmic complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    Linearithmic,
    Quadratic,
    Cubic,
    Exponential,
    Unknown,
}

/// Performance characteristic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristic {
    pub name: String,
    pub value: String,
    pub measurement_unit: Option<String>,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_type: String,
    pub description: String,
    pub estimated_improvement: Option<String>,
    pub implementation_cost: Option<String>,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub result: String,
    pub timestamp: std::time::SystemTime,
}

/// Resource usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub resource_type: String,
    pub usage_pattern: String,
    pub peak_usage: Option<String>,
}

/// Security level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Public,
    Internal,
    Confidential,
    Secret,
    TopSecret,
}

/// Threat consideration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatConsideration {
    pub threat_type: String,
    pub description: String,
    pub severity: RiskLevel,
    pub mitigation: Option<String>,
}

/// Permission requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub permission_type: String,
    pub description: String,
    pub required: bool,
}

/// Security implication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplication {
    pub implication_type: String,
    pub description: String,
    pub impact: ImpactLevel,
}

/// Audit requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequirement {
    pub requirement_type: String,
    pub description: String,
    pub frequency: AuditFrequency,
}

/// Audit frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditFrequency {
    Continuous,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
    OnDemand,
}

/// Data sensitivity classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSensitivity {
    pub sensitivity_level: SensitivityLevel,
    pub data_types: Vec<String>,
    pub handling_requirements: Vec<String>,
}

/// Sensitivity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensitivityLevel {
    Public,
    Internal,
    Confidential,
    Restricted,
    Secret,
}

/// Quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    pub score: f64, // 0.0 to 1.0
    pub factors: Vec<QualityFactor>,
    pub recommendations: Vec<String>,
}

/// Quality factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFactor {
    pub factor_name: String,
    pub score: f64,
    pub weight: f64,
    pub description: String,
}

/// Technical debt indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDebtIndicator {
    pub debt_type: String,
    pub severity: RiskLevel,
    pub description: String,
    pub estimated_cost: Option<String>,
}

/// Code smell identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSmell {
    pub smell_type: String,
    pub description: String,
    pub severity: RiskLevel,
    pub refactoring_suggestion: Option<String>,
}

impl SymbolMetadata {
    /// Create new metadata with minimal information
    pub fn new(responsibility: Option<String>) -> Self {
        Self {
            business_context: responsibility.map(|resp| BusinessContext {
                domain: resp,
                business_rules: Vec::new(),
                stakeholders: Vec::new(),
                compliance_requirements: Vec::new(),
                business_impact: None,
                resource_info: None,
            }),
            confidence: 1.0,
            last_updated: Some(std::time::SystemTime::now()),
            ..Default::default()
        }
    }
    
    /// Add AI context
    pub fn with_ai_context(mut self, ai_context: AISymbolContext) -> Self {
        self.ai_context = Some(ai_context);
        self
    }
    
    /// Add documentation
    pub fn with_documentation(mut self, documentation: SymbolDocumentation) -> Self {
        self.documentation = Some(documentation);
        self
    }
    
    /// Add semantic annotation
    pub fn add_semantic_annotation(&mut self, annotation: SemanticAnnotation) {
        self.semantic_annotations.push(annotation);
    }
    
    /// Get AI-readable summary
    pub fn ai_summary(&self) -> String {
        let mut summary = String::new();
        
        if let Some(business_context) = &self.business_context {
            summary.push_str(&format!("Responsibility: {}", business_context.domain));
        }
        
        if let Some(ai_context) = &self.ai_context {
            if !summary.is_empty() {
                summary.push_str(". ");
            }
            summary.push_str(&format!("Purpose: {}", ai_context.purpose));
        }
        
        if let Some(documentation) = &self.documentation {
            if !summary.is_empty() {
                summary.push_str(". ");
            }
            summary.push_str(&format!("Description: {}", documentation.summary));
        }
        
        summary
    }
    
    /// Update confidence based on metadata completeness
    pub fn calculate_confidence(&mut self) {
        let mut factors = 0;
        let mut score = 0.0;
        
        if self.ai_context.is_some() {
            factors += 1;
            score += 0.3;
        }
        
        if self.documentation.is_some() {
            factors += 1;
            score += 0.2;
        }
        
        if self.business_context.is_some() {
            factors += 1;
            score += 0.2;
        }
        
        if !self.semantic_annotations.is_empty() {
            factors += 1;
            score += 0.1;
        }
        
        if self.performance_metadata.is_some() {
            factors += 1;
            score += 0.1;
        }
        
        if self.security_metadata.is_some() {
            factors += 1;
            score += 0.1;
        }
        
        self.confidence = if factors > 0 { score } else { 0.0 };
    }
}

impl AISymbolContext {
    /// Create new AI context
    pub fn new(purpose: String) -> Self {
        Self {
            purpose,
            description: String::new(),
            usage_examples: Vec::new(),
            use_cases: Vec::new(),
            related_concepts: Vec::new(),
            domain_context: None,
            ai_insights: Vec::new(),
            complexity_assessment: None,
        }
    }
    
    /// Add usage example
    pub fn add_usage_example(&mut self, example: UsageExample) {
        self.usage_examples.push(example);
    }
    
    /// Add use case
    pub fn add_use_case(&mut self, use_case: UseCase) {
        self.use_cases.push(use_case);
    }
}

impl SymbolDocumentation {
    /// Create new documentation
    pub fn new(summary: String) -> Self {
        Self {
            summary,
            description: None,
            parameters: Vec::new(),
            returns: None,
            examples: Vec::new(),
            notes: Vec::new(),
            see_also: Vec::new(),
            author: None,
            version: None,
            tags: Vec::new(),
        }
    }
    
    /// Add parameter documentation
    pub fn add_parameter(&mut self, param_doc: ParameterDoc) {
        self.parameters.push(param_doc);
    }
    
    /// Add code example
    pub fn add_example(&mut self, example: CodeExample) {
        self.examples.push(example);
    }
}