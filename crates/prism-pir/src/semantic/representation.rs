//! Semantic Representation - Core PIR Types
//!
//! This module defines the core semantic representation types that preserve
//! all semantic information from the AST while adding business context.

use crate::{PIRError, PIRResult};
use prism_common::symbol::Symbol;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The complete Prism Intermediate Representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismIR {
    /// PIR modules containing business capabilities
    pub modules: Vec<PIRModule>,
    /// Global type registry with semantic information
    pub type_registry: SemanticTypeRegistry,
    /// Effect graph for capability analysis
    pub effect_graph: EffectGraph,
    /// Cohesion metrics for optimization guidance
    pub cohesion_metrics: CohesionMetrics,
    /// AI metadata for external tools
    pub ai_metadata: crate::ai_integration::AIMetadata,
    /// PIR metadata and versioning
    pub metadata: PIRMetadata,
}

/// PIR module representing a cohesive business capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRModule {
    /// Module name
    pub name: String,
    /// Business capability domain
    pub capability: String,
    /// Module sections organized by conceptual purpose
    pub sections: Vec<PIRSection>,
    /// Module dependencies
    pub dependencies: Vec<ModuleDependency>,
    /// Business context information
    pub business_context: crate::business::BusinessContext,
    /// Smart module metadata
    pub smart_module_metadata: SmartModuleMetadata,
    /// Domain-specific rules
    pub domain_rules: Vec<DomainRule>,
    /// Effects provided or required by this module
    pub effects: Vec<Effect>,
    /// Capabilities provided or required
    pub capabilities: Vec<Capability>,
    /// Performance characteristics
    pub performance_profile: crate::quality::PerformanceProfile,
    /// Conceptual cohesion score (0.0 to 1.0)
    pub cohesion_score: f64,
}

/// Comprehensive smart module metadata for AI comprehension and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartModuleMetadata {
    /// Module discovery information
    pub discovery_info: ModuleDiscoveryInfo,
    /// Capability mapping and compatibility
    pub capability_mapping: CapabilityMapping,
    /// Business context analysis
    pub business_analysis: BusinessAnalysis,
    /// Cohesion metrics and trends
    pub cohesion_analysis: CohesionAnalysis,
    /// AI-generated insights and recommendations
    pub ai_insights: AIInsights,
    /// Module lifecycle information
    pub lifecycle_info: ModuleLifecycleInfo,
    /// Integration patterns and relationships
    pub integration_patterns: IntegrationPatterns,
    /// Quality metrics and assessments
    pub quality_metrics: QualityMetrics,
}

/// Module discovery and identification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDiscoveryInfo {
    /// Primary module identifier
    pub module_id: String,
    /// Module version information
    pub version: Option<String>,
    /// Module stability level
    pub stability: StabilityLevel,
    /// Module visibility scope
    pub visibility: ModuleVisibility,
    /// Tags for categorization and search
    pub tags: Vec<String>,
    /// Module classification (domain, layer, etc.)
    pub classification: ModuleClassification,
    /// Search keywords for discovery
    pub search_keywords: Vec<String>,
}

/// Module stability levels for lifecycle management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityLevel {
    /// Experimental - may change significantly
    Experimental,
    /// Alpha - basic functionality, may have breaking changes
    Alpha,
    /// Beta - feature complete, API may change
    Beta,
    /// Stable - production ready, API stable
    Stable,
    /// Deprecated - marked for removal
    Deprecated,
    /// Legacy - maintained but not enhanced
    Legacy,
}

/// Module visibility levels for access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleVisibility {
    /// Public - accessible to all
    Public,
    /// Internal - accessible within the same project
    Internal,
    /// Private - accessible only within the same module
    Private,
    /// Protected - accessible within inheritance hierarchy
    Protected,
}

/// Module classification for organizational structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleClassification {
    /// Domain classification (business domain)
    pub domain: Option<String>,
    /// Layer classification (presentation, business, data, etc.)
    pub layer: Option<String>,
    /// Architectural pattern (MVC, hexagonal, etc.)
    pub pattern: Option<String>,
    /// Technology stack classification
    pub technology_stack: Vec<String>,
}

/// Capability mapping and compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMapping {
    /// Capabilities this module provides
    pub provides: Vec<ProvidedCapability>,
    /// Capabilities this module requires
    pub requires: Vec<RequiredCapability>,
    /// Optional capabilities (graceful degradation)
    pub optional: Vec<OptionalCapability>,
    /// Capability compatibility matrix
    pub compatibility_matrix: HashMap<String, CompatibilityInfo>,
    /// Version compatibility information
    pub version_compatibility: VersionCompatibility,
}

/// Information about a capability this module provides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidedCapability {
    /// Capability name
    pub name: String,
    /// Capability version
    pub version: String,
    /// Capability description
    pub description: Option<String>,
    /// Quality of service level
    pub qos_level: QoSLevel,
    /// Performance characteristics
    pub performance_hints: Vec<String>,
    /// Security requirements
    pub security_requirements: Vec<String>,
}

/// Information about a capability this module requires
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredCapability {
    /// Capability name
    pub name: String,
    /// Minimum required version
    pub min_version: String,
    /// Maximum compatible version
    pub max_version: Option<String>,
    /// Criticality level
    pub criticality: CriticalityLevel,
    /// Fallback behavior if unavailable
    pub fallback_strategy: Option<String>,
}

/// Information about an optional capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionalCapability {
    /// Capability name
    pub name: String,
    /// Preferred version
    pub preferred_version: Option<String>,
    /// Enhanced functionality when available
    pub enhanced_features: Vec<String>,
    /// Degraded behavior when unavailable
    pub degraded_behavior: Option<String>,
}

/// Compatibility information for capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    /// Compatible versions
    pub compatible_versions: Vec<String>,
    /// Incompatible versions
    pub incompatible_versions: Vec<String>,
    /// Migration paths for upgrades
    pub migration_paths: Vec<MigrationPath>,
    /// Breaking changes information
    pub breaking_changes: Vec<BreakingChange>,
}

/// Quality of Service levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QoSLevel {
    /// Best effort - no guarantees
    BestEffort,
    /// Standard - typical performance expectations
    Standard,
    /// High - above average performance requirements
    High,
    /// Critical - mission critical performance requirements
    Critical,
}

/// Criticality levels for requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CriticalityLevel {
    /// Low - nice to have
    Low,
    /// Medium - important for full functionality
    Medium,
    /// High - essential for core functionality
    High,
    /// Critical - module cannot function without this
    Critical,
}

/// Version compatibility information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionCompatibility {
    /// Semantic versioning compatibility
    pub semantic_versioning: bool,
    /// Backward compatibility guarantees
    pub backward_compatible_versions: Vec<String>,
    /// Forward compatibility information
    pub forward_compatible_versions: Vec<String>,
    /// API evolution strategy
    pub evolution_strategy: String,
}

/// Business context analysis for the module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessAnalysis {
    /// Primary business domain
    pub primary_domain: String,
    /// Secondary domains (if applicable)
    pub secondary_domains: Vec<String>,
    /// Business value proposition
    pub value_proposition: String,
    /// Key business entities handled
    pub business_entities: Vec<BusinessEntityInfo>,
    /// Business processes supported
    pub business_processes: Vec<BusinessProcessInfo>,
    /// Business rules enforced
    pub business_rules: Vec<BusinessRuleInfo>,
    /// Stakeholder information
    pub stakeholders: Vec<StakeholderInfo>,
}

/// Information about business entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessEntityInfo {
    /// Entity name
    pub name: String,
    /// Entity description
    pub description: Option<String>,
    /// Entity lifecycle states
    pub lifecycle_states: Vec<String>,
    /// Related entities
    pub relationships: Vec<String>,
    /// Business constraints
    pub constraints: Vec<String>,
}

/// Information about business processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessProcessInfo {
    /// Process name
    pub name: String,
    /// Process description
    pub description: Option<String>,
    /// Process steps
    pub steps: Vec<String>,
    /// Input requirements
    pub inputs: Vec<String>,
    /// Output deliverables
    pub outputs: Vec<String>,
    /// Success criteria
    pub success_criteria: Vec<String>,
}

/// Information about business rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRuleInfo {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule type (validation, calculation, etc.)
    pub rule_type: String,
    /// Enforcement level
    pub enforcement_level: EnforcementLevel,
    /// Related entities
    pub related_entities: Vec<String>,
}

/// Information about stakeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeholderInfo {
    /// Stakeholder role
    pub role: String,
    /// Stakeholder interests
    pub interests: Vec<String>,
    /// Interaction patterns
    pub interaction_patterns: Vec<String>,
}

/// Cohesion analysis and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysis {
    /// Overall cohesion score
    pub overall_score: f64,
    /// Detailed cohesion metrics
    pub detailed_metrics: DetailedCohesionMetrics,
    /// Cohesion trends over time
    pub trends: CohesionTrends,
    /// Cohesion violations and issues
    pub violations: Vec<CohesionViolation>,
    /// Improvement suggestions
    pub improvement_suggestions: Vec<CohesionSuggestion>,
}

/// Detailed cohesion metrics breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedCohesionMetrics {
    /// Functional cohesion score
    pub functional_cohesion: f64,
    /// Sequential cohesion score
    pub sequential_cohesion: f64,
    /// Communicational cohesion score
    pub communicational_cohesion: f64,
    /// Procedural cohesion score
    pub procedural_cohesion: f64,
    /// Temporal cohesion score
    pub temporal_cohesion: f64,
    /// Logical cohesion score
    pub logical_cohesion: f64,
    /// Coincidental cohesion score
    pub coincidental_cohesion: f64,
}

/// Cohesion trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionTrends {
    /// Historical cohesion scores
    pub historical_scores: Vec<CohesionSnapshot>,
    /// Trend direction (improving, declining, stable)
    pub trend_direction: TrendDirection,
    /// Rate of change
    pub change_rate: f64,
    /// Predicted future trend
    pub future_prediction: Option<f64>,
}

/// Cohesion snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionSnapshot {
    /// Timestamp of measurement
    pub timestamp: String,
    /// Cohesion score at that time
    pub score: f64,
    /// What triggered this measurement
    pub trigger: String,
    /// Context information
    pub context: Option<String>,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Cohesion is improving
    Improving,
    /// Cohesion is declining
    Declining,
    /// Cohesion is stable
    Stable,
    /// Insufficient data to determine trend
    Unknown,
}

/// Cohesion violation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionViolation {
    /// Violation type
    pub violation_type: String,
    /// Violation description
    pub description: String,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Location information
    pub location: Option<String>,
    /// Impact assessment
    pub impact: String,
}

/// Violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Low severity - minor issue
    Low,
    /// Medium severity - should be addressed
    Medium,
    /// High severity - important to fix
    High,
    /// Critical severity - must be fixed
    Critical,
}

/// Cohesion improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionSuggestion {
    /// Suggestion title
    pub title: String,
    /// Detailed suggestion description
    pub description: String,
    /// Estimated effort to implement
    pub effort_estimate: EffortLevel,
    /// Expected impact
    pub expected_impact: f64,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

/// Effort levels for improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    /// Minimal effort required
    Minimal,
    /// Low effort required
    Low,
    /// Medium effort required
    Medium,
    /// High effort required
    High,
    /// Significant effort required
    Significant,
}

/// AI-generated insights and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInsights {
    /// Generated summary of the module
    pub module_summary: String,
    /// Architectural insights
    pub architectural_insights: Vec<String>,
    /// Performance insights
    pub performance_insights: Vec<String>,
    /// Security insights
    pub security_insights: Vec<String>,
    /// Maintainability insights
    pub maintainability_insights: Vec<String>,
    /// Reusability assessment
    pub reusability_assessment: ReusabilityAssessment,
    /// Suggested improvements
    pub suggested_improvements: Vec<ImprovementSuggestion>,
    /// Related modules and patterns
    pub related_modules: Vec<String>,
    /// Learning resources
    pub learning_resources: Vec<LearningResource>,
}

/// Reusability assessment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReusabilityAssessment {
    /// Reusability score (0.0 to 1.0)
    pub score: f64,
    /// Factors that enhance reusability
    pub enhancing_factors: Vec<String>,
    /// Factors that limit reusability
    pub limiting_factors: Vec<String>,
    /// Suggested improvements for reusability
    pub improvement_suggestions: Vec<String>,
}

/// Improvement suggestion with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementSuggestion {
    /// Suggestion category
    pub category: String,
    /// Suggestion title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Priority level
    pub priority: Priority,
    /// Estimated effort
    pub effort: EffortLevel,
    /// Expected benefits
    pub benefits: Vec<String>,
    /// Implementation guidance
    pub implementation_guidance: Option<String>,
}

/// Priority levels for suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Learning resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResource {
    /// Resource title
    pub title: String,
    /// Resource type (documentation, tutorial, example, etc.)
    pub resource_type: String,
    /// Resource URL or location
    pub location: String,
    /// Resource description
    pub description: Option<String>,
    /// Relevance score
    pub relevance: f64,
}

/// Module lifecycle information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleLifecycleInfo {
    /// Current lifecycle stage
    pub current_stage: LifecycleStage,
    /// Creation timestamp
    pub created_at: String,
    /// Last modification timestamp
    pub last_modified: String,
    /// Version history
    pub version_history: Vec<VersionInfo>,
    /// Deprecation information
    pub deprecation_info: Option<DeprecationInfo>,
    /// Migration information
    pub migration_info: Option<MigrationInfo>,
}

/// Module lifecycle stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleStage {
    /// Planning stage
    Planning,
    /// Development stage
    Development,
    /// Testing stage
    Testing,
    /// Production stage
    Production,
    /// Maintenance stage
    Maintenance,
    /// Deprecated stage
    Deprecated,
    /// Retired stage
    Retired,
}

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Version number
    pub version: String,
    /// Release date
    pub release_date: String,
    /// Changes in this version
    pub changes: Vec<String>,
    /// Breaking changes
    pub breaking_changes: Vec<String>,
    /// Migration notes
    pub migration_notes: Option<String>,
}

/// Deprecation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Deprecation reason
    pub reason: String,
    /// Deprecation date
    pub deprecated_at: String,
    /// Planned removal date
    pub removal_date: Option<String>,
    /// Replacement module
    pub replacement: Option<String>,
    /// Migration guide
    pub migration_guide: Option<String>,
}

/// Migration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationInfo {
    /// Source version
    pub from_version: String,
    /// Target version
    pub to_version: String,
    /// Migration steps
    pub migration_steps: Vec<MigrationStep>,
    /// Automated migration available
    pub automated_migration: bool,
    /// Migration complexity
    pub complexity: MigrationComplexity,
}

/// Migration step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Step number
    pub step_number: u32,
    /// Step description
    pub description: String,
    /// Required actions
    pub actions: Vec<String>,
    /// Validation criteria
    pub validation: Vec<String>,
}

/// Migration complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationComplexity {
    /// Simple migration
    Simple,
    /// Moderate complexity
    Moderate,
    /// Complex migration
    Complex,
    /// Very complex migration
    VeryComplex,
}

/// Migration path information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPath {
    /// Source version
    pub from: String,
    /// Target version
    pub to: String,
    /// Migration steps
    pub steps: Vec<String>,
    /// Estimated effort
    pub effort: EffortLevel,
    /// Risk level
    pub risk: RiskLevel,
}

/// Risk levels for migrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Very high risk
    VeryHigh,
}

/// Breaking change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingChange {
    /// Change description
    pub description: String,
    /// Affected APIs
    pub affected_apis: Vec<String>,
    /// Migration instructions
    pub migration_instructions: String,
    /// Introduced in version
    pub introduced_in: String,
}

/// Integration patterns and relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPatterns {
    /// Common integration patterns used
    pub patterns: Vec<IntegrationPattern>,
    /// Module relationships
    pub relationships: Vec<ModuleRelationship>,
    /// Communication patterns
    pub communication_patterns: Vec<CommunicationPattern>,
    /// Data flow patterns
    pub data_flow_patterns: Vec<DataFlowPattern>,
}

/// Integration pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// When to use this pattern
    pub use_cases: Vec<String>,
    /// Benefits of this pattern
    pub benefits: Vec<String>,
    /// Drawbacks of this pattern
    pub drawbacks: Vec<String>,
    /// Implementation examples
    pub examples: Vec<String>,
}

/// Module relationship information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRelationship {
    /// Related module name
    pub module_name: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Relationship description
    pub description: Option<String>,
    /// Dependencies involved
    pub dependencies: Vec<String>,
}

/// Types of module relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Direct dependency
    Depends,
    /// Provides services to
    Provides,
    /// Collaborates with
    Collaborates,
    /// Extends functionality of
    Extends,
    /// Composes with
    Composes,
    /// Similar functionality
    Similar,
    /// Conflicting functionality
    Conflicts,
}

/// Communication pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    /// Pattern name
    pub name: String,
    /// Communication type (synchronous, asynchronous, etc.)
    pub communication_type: String,
    /// Protocol used
    pub protocol: Option<String>,
    /// Message format
    pub message_format: Option<String>,
    /// Error handling strategy
    pub error_handling: Option<String>,
}

/// Data flow pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowPattern {
    /// Pattern name
    pub name: String,
    /// Data source
    pub source: String,
    /// Data destination
    pub destination: String,
    /// Transformation applied
    pub transformation: Option<String>,
    /// Validation rules
    pub validation: Vec<String>,
}

/// Quality metrics and assessments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score
    pub overall_score: f64,
    /// Code quality metrics
    pub code_quality: CodeQualityMetrics,
    /// Design quality metrics
    pub design_quality: DesignQualityMetrics,
    /// Documentation quality
    pub documentation_quality: DocumentationQualityMetrics,
    /// Test coverage metrics
    pub test_coverage: TestCoverageMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Security metrics
    pub security_metrics: SecurityMetrics,
}

/// Code quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQualityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: f64,
    /// Lines of code
    pub lines_of_code: u32,
    /// Code duplication percentage
    pub duplication_percentage: f64,
    /// Technical debt ratio
    pub technical_debt_ratio: f64,
    /// Maintainability index
    pub maintainability_index: f64,
}

/// Design quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignQualityMetrics {
    /// Coupling metrics
    pub coupling: CouplingMetrics,
    /// Cohesion score
    pub cohesion: f64,
    /// Abstraction level
    pub abstraction_level: f64,
    /// Interface quality
    pub interface_quality: f64,
    /// Design pattern adherence
    pub pattern_adherence: f64,
}

/// Documentation quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationQualityMetrics {
    /// Documentation coverage percentage
    pub coverage_percentage: f64,
    /// Documentation quality score
    pub quality_score: f64,
    /// Up-to-date percentage
    pub up_to_date_percentage: f64,
    /// Completeness score
    pub completeness_score: f64,
}

/// Test coverage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCoverageMetrics {
    /// Line coverage percentage
    pub line_coverage: f64,
    /// Branch coverage percentage
    pub branch_coverage: f64,
    /// Function coverage percentage
    pub function_coverage: f64,
    /// Integration test coverage
    pub integration_coverage: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average response time (ms)
    pub avg_response_time: f64,
    /// Throughput (requests/second)
    pub throughput: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Error rate percentage
    pub error_rate: f64,
}

/// Security metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    /// Security score (0.0 to 1.0)
    pub security_score: f64,
    /// Vulnerability count
    pub vulnerability_count: u32,
    /// Security test coverage
    pub security_test_coverage: f64,
    /// Compliance score
    pub compliance_score: f64,
}

/// PIR section organized by conceptual purpose within a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRSection {
    /// Type definitions with semantic meaning
    Types(TypeSection),
    /// Function definitions with business logic
    Functions(FunctionSection),
    /// Constants with domain significance
    Constants(ConstantSection),
    /// Interface definitions for contracts
    Interface(InterfaceSection),
    /// Implementation details
    Implementation(ImplementationSection),
}

/// Type section containing semantically rich types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSection {
    /// Semantic types defined in this section
    pub types: Vec<PIRSemanticType>,
}

/// Function section with effect-aware functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSection {
    /// Functions with full semantic and business information
    pub functions: Vec<PIRFunction>,
}

/// Constant section with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantSection {
    /// Constants with semantic significance
    pub constants: Vec<PIRConstant>,
}

/// Interface section defining contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceSection {
    /// Interface definitions
    pub interfaces: Vec<PIRInterface>,
}

/// Implementation section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationSection {
    /// Implementation items
    pub items: Vec<PIRImplementationItem>,
}

/// PIR semantic type with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRSemanticType {
    /// Type name
    pub name: String,
    /// Base type information
    pub base_type: PIRTypeInfo,
    /// Semantic domain this type belongs to
    pub domain: String,
    /// Business rules associated with this type
    pub business_rules: Vec<crate::business::BusinessRule>,
    /// Validation predicates
    pub validation_predicates: Vec<ValidationPredicate>,
    /// Type constraints
    pub constraints: Vec<PIRTypeConstraint>,
    /// AI context for understanding
    pub ai_context: PIRTypeAIContext,
    /// Security classification
    pub security_classification: SecurityClassification,
}

/// PIR type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRTypeInfo {
    /// Primitive types
    Primitive(PIRPrimitiveType),
    /// Composite types (structs, enums, etc.)
    Composite(PIRCompositeType),
    /// Function types with effects
    Function(PIRFunctionType),
    /// Generic types with bounds
    Generic(PIRGenericType),
    /// Effect types for capability system
    Effect(PIREffectType),
}

/// PIR primitive types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRPrimitiveType {
    /// Integer with signedness and width
    Integer { signed: bool, width: u8 },
    /// Floating point with width
    Float { width: u8 },
    /// Boolean value
    Boolean,
    /// String value
    String,
    /// Unit type (void)
    Unit,
}

/// PIR composite type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRCompositeType {
    /// Kind of composite
    pub kind: PIRCompositeKind,
    /// Fields in the composite
    pub fields: Vec<PIRField>,
    /// Methods associated with the type
    pub methods: Vec<PIRMethod>,
}

/// Kind of composite type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRCompositeKind {
    /// Struct type
    Struct,
    /// Enum type
    Enum,
    /// Union type
    Union,
    /// Tuple type
    Tuple,
}

/// PIR field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: PIRTypeInfo,
    /// Visibility level
    pub visibility: PIRVisibility,
    /// Business meaning of this field
    pub business_meaning: Option<String>,
    /// Validation rules for this field
    pub validation_rules: Vec<String>,
}

/// PIR method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRMethod {
    /// Method name
    pub name: String,
    /// Method signature
    pub signature: PIRFunctionType,
    /// Method implementation (if available)
    pub implementation: Option<PIRExpression>,
    /// Business purpose of this method
    pub business_purpose: Option<String>,
}

/// PIR function type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRFunctionType {
    /// Function parameters
    pub parameters: Vec<PIRParameter>,
    /// Return type
    pub return_type: Box<PIRTypeInfo>,
    /// Effect signature
    pub effects: EffectSignature,
    /// Performance contracts
    pub contracts: PIRPerformanceContract,
}

/// PIR parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: PIRTypeInfo,
    /// Default value (if any)
    pub default_value: Option<PIRExpression>,
    /// Business meaning of this parameter
    pub business_meaning: Option<String>,
}

/// PIR generic type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRGenericType {
    /// Type parameter name
    pub name: String,
    /// Type bounds
    pub bounds: Vec<PIRTypeConstraint>,
    /// Default type (if any)
    pub default_type: Option<Box<PIRTypeInfo>>,
}

/// PIR effect type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIREffectType {
    /// Effect name
    pub name: String,
    /// Effect operations
    pub operations: Vec<PIREffectOperation>,
    /// Required capabilities
    pub capabilities: Vec<String>,
}

/// PIR effect operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIREffectOperation {
    /// Operation name
    pub name: String,
    /// Operation signature
    pub signature: PIRFunctionType,
    /// Side effects of this operation
    pub side_effects: Vec<String>,
}

/// PIR type constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRTypeConstraint {
    /// Range constraint for numeric types
    Range {
        /// Minimum value
        min: Option<PIRExpression>,
        /// Maximum value
        max: Option<PIRExpression>,
    },
    /// Pattern constraint for string types
    Pattern {
        /// Regular expression pattern
        pattern: String,
    },
    /// Business rule constraint
    BusinessRule {
        /// Business rule to enforce
        rule: crate::business::BusinessRule,
    },
    /// Custom constraint with predicate
    Custom {
        /// Custom predicate expression
        predicate: PIRExpression,
    },
}

/// PIR function with full semantic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRFunction {
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: PIRFunctionType,
    /// Function body
    pub body: PIRExpression,
    /// Business responsibility of this function
    pub responsibility: Option<String>,
    /// Algorithm description
    pub algorithm: Option<String>,
    /// Complexity analysis
    pub complexity: Option<PIRComplexityAnalysis>,
    /// Capabilities required to execute
    pub capabilities_required: Vec<Capability>,
    /// Performance characteristics
    pub performance_characteristics: Vec<String>,
    /// AI hints for understanding
    pub ai_hints: Vec<String>,
}

/// PIR constant with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRConstant {
    /// Constant name
    pub name: String,
    /// Constant type
    pub const_type: PIRTypeInfo,
    /// Constant value
    pub value: PIRExpression,
    /// Business meaning of this constant
    pub business_meaning: Option<String>,
}

/// PIR interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRInterface {
    /// Interface name
    pub name: String,
    /// Interface methods
    pub methods: Vec<PIRMethod>,
    /// Required capabilities
    pub capabilities: Vec<Capability>,
}

/// PIR implementation item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRImplementationItem {
    /// Function implementation
    Function(PIRFunction),
    /// Type implementation
    Type(PIRTypeImplementation),
}

/// PIR type implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRTypeImplementation {
    /// Type being implemented
    pub target_type: String,
    /// Interface being implemented
    pub interface: String,
    /// Method implementations
    pub methods: Vec<PIRMethod>,
}

/// PIR expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRExpression {
    /// Literal value
    Literal(PIRLiteral),
    /// Variable reference
    Variable(String),
    /// Function call with effects
    Call {
        /// Function being called
        function: Box<PIRExpression>,
        /// Arguments to the function
        arguments: Vec<PIRExpression>,
        /// Effects of this call
        effects: Vec<Effect>,
    },
    /// Binary operation
    Binary {
        /// Left operand
        left: Box<PIRExpression>,
        /// Binary operator
        operator: PIRBinaryOp,
        /// Right operand
        right: Box<PIRExpression>,
    },
    /// Unary operation
    Unary {
        /// Unary operator
        operator: PIRUnaryOp,
        /// Operand
        operand: Box<PIRExpression>,
    },
    /// Block expression
    Block {
        /// Statements in the block
        statements: Vec<PIRStatement>,
        /// Result expression (if any)
        result: Option<Box<PIRExpression>>,
    },
    /// Conditional expression
    If {
        /// Condition
        condition: Box<PIRExpression>,
        /// Then branch
        then_branch: Box<PIRExpression>,
        /// Else branch (if any)
        else_branch: Option<Box<PIRExpression>>,
    },
    /// Pattern matching expression
    Match {
        /// Expression being matched
        scrutinee: Box<PIRExpression>,
        /// Match arms
        arms: Vec<PIRMatchArm>,
    },
    /// Type assertion
    TypeAssertion {
        /// Expression to assert
        expression: Box<PIRExpression>,
        /// Target type
        target_type: PIRTypeInfo,
    },
}

/// PIR literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRLiteral {
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// Boolean literal
    Boolean(bool),
    /// String literal
    String(String),
    /// Unit literal
    Unit,
}

/// PIR binary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRBinaryOp {
    /// Addition
    Add,
    /// Subtraction
    Subtract,
    /// Multiplication
    Multiply,
    /// Division
    Divide,
    /// Modulo
    Modulo,
    /// Equality
    Equal,
    /// Inequality
    NotEqual,
    /// Less than
    Less,
    /// Less than or equal
    LessEqual,
    /// Greater than
    Greater,
    /// Greater than or equal
    GreaterEqual,
    /// Logical and
    And,
    /// Logical or
    Or,
    /// Prism-specific semantic equality
    SemanticEqual,
}

/// PIR unary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRUnaryOp {
    /// Logical not
    Not,
    /// Arithmetic negation
    Negate,
}

/// PIR statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRStatement {
    /// Expression statement
    Expression(PIRExpression),
    /// Variable binding
    Let {
        /// Variable name
        name: String,
        /// Type annotation (if any)
        type_annotation: Option<PIRTypeInfo>,
        /// Initial value
        value: PIRExpression,
    },
    /// Assignment
    Assignment {
        /// Assignment target
        target: PIRExpression,
        /// Value to assign
        value: PIRExpression,
    },
    /// Return statement
    Return(Option<PIRExpression>),
}

/// PIR match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRMatchArm {
    /// Pattern to match
    pub pattern: PIRPattern,
    /// Guard condition (if any)
    pub guard: Option<PIRExpression>,
    /// Arm body
    pub body: PIRExpression,
}

/// PIR pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRPattern {
    /// Wildcard pattern (matches anything)
    Wildcard,
    /// Variable pattern (binds to variable)
    Variable(String),
    /// Literal pattern (matches literal)
    Literal(PIRLiteral),
    /// Constructor pattern
    Constructor {
        /// Constructor name
        name: String,
        /// Field patterns
        fields: Vec<PIRPattern>,
    },
}

/// PIR visibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRVisibility {
    /// Public visibility
    Public,
    /// Private visibility
    Private,
    /// Internal visibility (module-level)
    Internal,
}

/// PIR type AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRTypeAIContext {
    /// Intent description
    pub intent: Option<String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common mistakes to avoid
    pub common_mistakes: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
}

/// PIR complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRComplexityAnalysis {
    /// Time complexity
    pub time_complexity: String,
    /// Space complexity
    pub space_complexity: String,
    /// Best case scenario
    pub best_case: Option<String>,
    /// Average case scenario
    pub average_case: Option<String>,
    /// Worst case scenario
    pub worst_case: Option<String>,
}

/// PIR performance contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRPerformanceContract {
    /// Preconditions
    pub preconditions: Vec<PIRCondition>,
    /// Postconditions
    pub postconditions: Vec<PIRCondition>,
    /// Performance guarantees
    pub performance_guarantees: Vec<PIRPerformanceGuarantee>,
}

/// PIR condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRCondition {
    /// Condition name
    pub name: String,
    /// Condition expression
    pub expression: PIRExpression,
    /// Error message if condition fails
    pub error_message: String,
}

/// PIR performance guarantee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRPerformanceGuarantee {
    /// Type of performance guarantee
    pub guarantee_type: PIRPerformanceType,
    /// Bound expression
    pub bound: PIRExpression,
    /// Description of the guarantee
    pub description: String,
}

/// PIR performance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRPerformanceType {
    /// Time complexity bound
    TimeComplexity,
    /// Space complexity bound
    SpaceComplexity,
    /// Maximum execution time
    MaxExecutionTime,
    /// Maximum memory usage
    MaxMemoryUsage,
}

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityClassification {
    /// Public information
    Public,
    /// Internal use only
    Internal,
    /// Confidential information
    Confidential,
    /// Restricted access
    Restricted,
    /// Top secret
    TopSecret,
}

/// Module dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDependency {
    /// Module name
    pub name: String,
    /// Type of dependency
    pub dependency_type: DependencyType,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
}

/// Dependency types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Direct dependency
    Direct,
    /// Transitive dependency
    Transitive,
    /// Optional dependency
    Optional,
    /// Development-only dependency
    Development,
}

/// Domain rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule expression
    pub expression: PIRExpression,
    /// Enforcement level
    pub enforcement: EnforcementLevel,
}

/// Enforcement level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced
    Required,
    /// Should be enforced
    Recommended,
    /// May be enforced
    Optional,
}

/// Effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    /// Effect name
    pub name: String,
    /// Effect type
    pub effect_type: String,
    /// Effect description
    pub description: Option<String>,
}

/// Capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Capability name
    pub name: String,
    /// Capability description
    pub description: Option<String>,
    /// Required permissions
    pub permissions: Vec<String>,
}

/// Validation predicate for semantic types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPredicate {
    /// Predicate name
    pub name: String,
    /// Predicate expression
    pub expression: String,
    /// Description of what this predicate validates
    pub description: Option<String>,
}

/// Effect signature for functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSignature {
    /// Input effects (effects consumed)
    pub input_effects: Vec<Effect>,
    /// Output effects (effects produced)
    pub output_effects: Vec<Effect>,
    /// Effect dependencies
    pub effect_dependencies: Vec<EffectDependency>,
}

/// Effect dependency relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDependency {
    /// Source effect
    pub source: String,
    /// Target effect
    pub target: String,
    /// Dependency type
    pub dependency_type: EffectDependencyType,
}

/// Effect dependency types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectDependencyType {
    /// Sequential dependency
    Sequential,
    /// Parallel dependency
    Parallel,
    /// Conditional dependency
    Conditional,
}

/// Smart module metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartModuleMetadata {
    /// Module conceptual purpose
    pub purpose: String,
    /// Architectural patterns used
    pub patterns: Vec<String>,
    /// Quality metrics
    pub quality_score: f64,
    /// Dependencies analysis
    pub dependency_analysis: DependencyAnalysis,
}

/// Dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    /// Incoming dependencies
    pub incoming: Vec<String>,
    /// Outgoing dependencies  
    pub outgoing: Vec<String>,
    /// Circular dependencies
    pub circular: Vec<Vec<String>>,
}

/// Semantic type registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTypeRegistry {
    /// Types in the registry
    pub types: HashMap<String, PIRSemanticType>,
    /// Type relationships
    pub relationships: HashMap<String, Vec<TypeRelationship>>,
    /// Global constraints
    pub global_constraints: Vec<PIRTypeConstraint>,
}

/// Type relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationship {
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Related type
    pub related_type: String,
    /// Relationship strength
    pub strength: f64,
}

/// Relationship types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Inheritance relationship
    Inheritance,
    /// Composition relationship
    Composition,
    /// Association relationship
    Association,
    /// Dependency relationship
    Dependency,
}

/// Effect graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectGraph {
    /// Graph nodes (effects)
    pub nodes: HashMap<String, EffectNode>,
    /// Graph edges (relationships)
    pub edges: Vec<EffectEdge>,
}

/// Effect graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectNode {
    /// Effect name
    pub name: String,
    /// Effect type
    pub effect_type: String,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Side effects
    pub side_effects: Vec<String>,
}

/// Effect graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectEdge {
    /// Source node
    pub source: String,
    /// Target node
    pub target: String,
    /// Edge type
    pub edge_type: EffectEdgeType,
}

/// Effect edge types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectEdgeType {
    /// Dependency edge
    Dependency,
    /// Composition edge
    Composition,
    /// Conflict edge
    Conflict,
}

/// Cohesion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionMetrics {
    /// Overall cohesion score
    pub overall_score: f64,
    /// Per-module scores
    pub module_scores: HashMap<String, f64>,
    /// Coupling metrics
    pub coupling_metrics: CouplingMetrics,
}

/// Coupling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    /// Afferent coupling (incoming dependencies)
    pub afferent: HashMap<String, u32>,
    /// Efferent coupling (outgoing dependencies)
    pub efferent: HashMap<String, u32>,
    /// Instability measure
    pub instability: HashMap<String, f64>,
}

/// PIR metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRMetadata {
    /// PIR version
    pub version: String,
    /// Creation timestamp
    pub created_at: String,
    /// Source AST hash
    pub source_hash: u64,
    /// Optimization level applied
    pub optimization_level: u8,
    /// Target platforms
    pub target_platforms: Vec<String>,
}

/// Resource limits for runtime integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory in MB
    pub max_memory_mb: Option<u64>,
    /// Maximum CPU time in milliseconds
    pub max_cpu_time_ms: Option<u64>,
    /// Maximum network connections
    pub max_network_connections: Option<u32>,
}

/// Resource usage delta for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageDelta {
    /// Memory delta in bytes
    pub memory_delta: i64,
    /// CPU time delta in microseconds
    pub cpu_time_delta: i64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(1024), // 1GB default
            max_cpu_time_ms: Some(30000), // 30 seconds default
            max_network_connections: Some(100),
        }
    }
}

impl Default for ResourceUsageDelta {
    fn default() -> Self {
        Self {
            memory_delta: 0,
            cpu_time_delta: 0,
        }
    }
}

impl PrismIR {
    /// Create a new empty PIR
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            type_registry: SemanticTypeRegistry {
                types: HashMap::new(),
                relationships: HashMap::new(),
                global_constraints: Vec::new(),
            },
            effect_graph: EffectGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
            },
            cohesion_metrics: CohesionMetrics {
                overall_score: 0.0,
                module_scores: HashMap::new(),
                coupling_metrics: CouplingMetrics {
                    afferent: HashMap::new(),
                    efferent: HashMap::new(),
                    instability: HashMap::new(),
                },
            },
            ai_metadata: crate::ai_integration::AIMetadata::default(),
            metadata: PIRMetadata {
                version: crate::PIRVersion::CURRENT.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                source_hash: 0,
                optimization_level: 0,
                target_platforms: Vec::new(),
            },
        }
    }

    /// Get a module by name
    pub fn get_module(&self, name: &str) -> Option<&PIRModule> {
        self.modules.iter().find(|module| module.name == name)
    }

    /// Get a mutable reference to a module by name
    pub fn get_module_mut(&mut self, name: &str) -> Option<&mut PIRModule> {
        self.modules.iter_mut().find(|module| module.name == name)
    }

    /// Add a module to the PIR
    pub fn add_module(&mut self, module: PIRModule) {
        self.modules.push(module);
    }

    /// Get the overall cohesion score
    pub fn cohesion_score(&self) -> f64 {
        self.cohesion_metrics.overall_score
    }
}

impl Default for PrismIR {
    fn default() -> Self {
        Self::new()
    }
} 