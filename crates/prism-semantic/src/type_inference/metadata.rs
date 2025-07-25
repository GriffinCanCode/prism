//! AI Metadata Generation for Type Inference
//!
//! This module generates comprehensive metadata about the type inference process
//! to enable AI understanding and assistance. It captures semantic meaning,
//! inference patterns, and provides insights for AI-driven development tools.

use super::{
    TypeInferenceResult, InferredType, InferenceSource, InferenceStatistics,
    TypeVar,
};
use crate::{SemanticResult, SemanticError, types::SemanticType};
use crate::type_inference::errors::TypeError;
use crate::type_inference::constraints::{TypeConstraint, ConstraintSet};
use prism_common::{NodeId, Span};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Helper function to get a simplified type description for metadata
fn get_type_description(semantic_type: &SemanticType) -> String {
    if semantic_type.is_variable() {
        return "Variable".to_string();
    }
    
    match semantic_type.base_type() {
        Some(crate::types::BaseType::Primitive(_)) => "Primitive".to_string(),
        Some(crate::types::BaseType::Function(_)) => "Function".to_string(),
        Some(crate::types::BaseType::Composite(comp)) => format!("Composite({:?})", comp.kind),
        Some(crate::types::BaseType::Generic(gen)) => format!("Generic({})", gen.base_type.type_name()),
        Some(crate::types::BaseType::Dependent(dep)) => format!("Dependent({})", dep.base_type.type_name()),
        Some(crate::types::BaseType::Effect(eff)) => format!("Effect({})", eff.name),
        None => semantic_type.type_name(),
    }
}

/// Helper function to estimate type complexity for metadata
fn estimate_type_complexity(semantic_type: &SemanticType) -> f64 {
    if semantic_type.is_variable() {
        return 0.5;
    }
    
    match semantic_type.base_type() {
        Some(crate::types::BaseType::Primitive(_)) => 1.0,
        Some(crate::types::BaseType::Function(func)) => {
            1.5 + (func.parameters.len() as f64 * 0.2)
        }
        Some(crate::types::BaseType::Composite(comp)) => {
            match comp.kind {
                crate::types::CompositeKind::Struct => 1.2,
                crate::types::CompositeKind::Enum => 1.1,
                crate::types::CompositeKind::Union => 1.4,
                _ => 1.0,
            }
        }
        Some(crate::types::BaseType::Generic(_)) => 1.2,
        Some(crate::types::BaseType::Dependent(_)) => 1.8,
        Some(crate::types::BaseType::Effect(_)) => 1.6,
        None => 1.0, // Default complexity for types without base_type
    }
}

/// AI metadata for a single type inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetadata {
    /// Semantic meaning of this type
    pub semantic_meaning: String,
    /// Business domain context
    pub domain_context: String,
    /// Confidence level in the inference
    pub confidence_level: ConfidenceLevel,
    /// Inference complexity score
    pub complexity_score: f64,
    /// Related patterns and idioms
    pub patterns: Vec<InferencePattern>,
    /// Suggested improvements
    pub suggestions: Vec<String>,
    /// Performance implications
    pub performance_notes: Vec<String>,
    /// Usage examples
    pub usage_examples: Vec<String>,
    /// Related documentation
    pub documentation_links: Vec<String>,
}

/// Confidence level in type inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Very high confidence (explicit annotations)
    VeryHigh,
    /// High confidence (clear from context)
    High,
    /// Medium confidence (reasonable inference)
    Medium,
    /// Low confidence (ambiguous context)
    Low,
    /// Very low confidence (fallback/default)
    VeryLow,
}

/// Pattern recognized during type inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePattern {
    /// Name of the pattern
    pub name: String,
    /// Description of the pattern
    pub description: String,
    /// Confidence in pattern recognition
    pub confidence: f64,
    /// Benefits of using this pattern
    pub benefits: Vec<String>,
    /// Potential issues with this pattern
    pub issues: Vec<String>,
}

/// Global metadata for the entire type inference process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalInferenceMetadata {
    /// Total number of nodes processed
    pub total_nodes: usize,
    /// Total number of constraints generated
    pub total_constraints: usize,
    /// Overall inference complexity
    pub inference_complexity: f64,
    /// Distribution of types found
    pub type_distribution: HashMap<String, usize>,
    /// AI-generated insights about the codebase
    pub ai_insights: Vec<String>,
    /// Performance statistics
    pub statistics: Option<InferenceStatistics>,
    /// Code quality metrics
    pub quality_metrics: QualityMetrics,
    /// Recommendations for improvement
    pub recommendations: Vec<Recommendation>,
}

/// Code quality metrics derived from type inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Type annotation coverage (0.0 to 1.0)
    pub annotation_coverage: f64,
    /// Polymorphism usage score
    pub polymorphism_score: f64,
    /// Type complexity average
    pub average_type_complexity: f64,
    /// Error proneness score
    pub error_proneness: f64,
    /// Maintainability score
    pub maintainability: f64,
}

/// Recommendation for code improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Type of recommendation
    pub category: RecommendationCategory,
    /// Description of the recommendation
    pub description: String,
    /// Priority level
    pub priority: Priority,
    /// Estimated impact
    pub impact: Impact,
    /// Specific locations to apply this recommendation
    pub locations: Vec<Span>,
    /// Code examples showing the improvement
    pub examples: Vec<CodeExample>,
}

/// Category of recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Type annotation recommendations
    TypeAnnotations,
    /// Performance improvements
    Performance,
    /// Code clarity improvements
    Clarity,
    /// Error prevention
    ErrorPrevention,
    /// Pattern usage
    Patterns,
    /// Refactoring suggestions
    Refactoring,
}

/// Priority level for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority (nice to have)
    Low,
    /// Medium priority (should consider)
    Medium,
    /// High priority (important)
    High,
    /// Critical priority (must fix)
    Critical,
}

/// Expected impact of a recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impact {
    /// Performance impact (-1.0 to 1.0, negative is worse)
    pub performance: f64,
    /// Readability impact (-1.0 to 1.0)
    pub readability: f64,
    /// Maintainability impact (-1.0 to 1.0)
    pub maintainability: f64,
    /// Error reduction impact (0.0 to 1.0)
    pub error_reduction: f64,
}

/// Code example for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Description of the example
    pub description: String,
    /// Original code
    pub before: String,
    /// Improved code
    pub after: String,
    /// Explanation of the improvement
    pub explanation: String,
}

/// AI integration interface for type inference
#[derive(Debug)]
pub struct TypeInferenceAI {
    /// Pattern recognition engine
    pattern_engine: PatternEngine,
    /// Semantic analyzer for business context
    semantic_analyzer: SemanticAnalyzer,
    /// Quality metrics calculator
    quality_calculator: QualityCalculator,
    /// Recommendation generator
    recommendation_generator: RecommendationGenerator,
}

/// Pattern recognition engine
#[derive(Debug)]
struct PatternEngine {
    /// Known patterns database
    known_patterns: Vec<PatternDefinition>,
    /// Pattern matching threshold
    threshold: f64,
}

/// Definition of a recognized pattern
#[derive(Debug, Clone)]
struct PatternDefinition {
    /// Pattern name
    name: String,
    /// Pattern description
    description: String,
    /// Matching criteria
    criteria: PatternCriteria,
    /// Benefits of this pattern
    benefits: Vec<String>,
    /// Potential issues
    issues: Vec<String>,
}

/// Criteria for pattern matching
#[derive(Debug, Clone)]
enum PatternCriteria {
    /// Function signature pattern
    FunctionSignature {
        param_count: Option<usize>,
        return_type: Option<String>,
        complexity: Option<f64>,
    },
    /// Type structure pattern
    TypeStructure {
        type_kind: String,
        nesting_level: Option<usize>,
    },
    /// Usage pattern
    Usage {
        frequency: f64,
        context: String,
    },
}

/// Semantic analyzer for extracting business context
#[derive(Debug)]
struct SemanticAnalyzer {
    /// Domain knowledge database
    domain_knowledge: HashMap<String, DomainInfo>,
    /// Semantic similarity threshold
    similarity_threshold: f64,
}

/// Domain information
#[derive(Debug, Clone)]
struct DomainInfo {
    /// Domain name
    name: String,
    /// Common types in this domain
    common_types: Vec<String>,
    /// Domain-specific patterns
    patterns: Vec<String>,
    /// Best practices
    best_practices: Vec<String>,
}

/// Quality metrics calculator
#[derive(Debug)]
struct QualityCalculator {
    /// Weights for different quality aspects
    weights: QualityWeights,
}

/// Weights for quality calculation
#[derive(Debug, Clone)]
struct QualityWeights {
    /// Weight for type annotation coverage
    annotation_weight: f64,
    /// Weight for polymorphism usage
    polymorphism_weight: f64,
    /// Weight for type complexity
    complexity_weight: f64,
    /// Weight for error proneness
    error_weight: f64,
}

/// Recommendation generator
#[derive(Debug)]
struct RecommendationGenerator {
    /// Recommendation rules
    rules: Vec<RecommendationRule>,
    /// Minimum confidence for recommendations
    min_confidence: f64,
}

/// Rule for generating recommendations
#[derive(Debug, Clone)]
struct RecommendationRule {
    /// Rule name
    name: String,
    /// Condition for triggering this rule
    condition: RuleCondition,
    /// Generated recommendation
    recommendation: RecommendationTemplate,
}

/// Condition for recommendation rules
#[derive(Debug, Clone)]
enum RuleCondition {
    /// Type complexity exceeds threshold
    TypeComplexity(f64),
    /// Missing type annotations
    MissingAnnotations(f64),
    /// Error-prone patterns
    ErrorPronePattern(String),
    /// Performance issues
    PerformanceIssue(String),
}

/// Template for generating recommendations
#[derive(Debug, Clone)]
struct RecommendationTemplate {
    /// Category of the recommendation
    category: RecommendationCategory,
    /// Description template
    description: String,
    /// Priority calculation
    priority: Priority,
    /// Impact estimation
    impact: Impact,
}

impl TypeInferenceAI {
    /// Create a new AI integration interface
    pub fn new() -> Self {
        Self {
            pattern_engine: PatternEngine::new(),
            semantic_analyzer: SemanticAnalyzer::new(),
            quality_calculator: QualityCalculator::new(),
            recommendation_generator: RecommendationGenerator::new(),
        }
    }

    /// Generate metadata for a single type inference
    pub fn generate_inference_metadata(
        &self,
        inferred_type: &InferredType,
        constraints: &[TypeConstraint],
        context: &InferenceContext,
    ) -> InferenceMetadata {
        // Extract semantic meaning
        let semantic_meaning = self.extract_semantic_meaning(&inferred_type.type_info, context);
        
        // Determine domain context
        let domain_context = self.semantic_analyzer.analyze_domain(&inferred_type.type_info);
        
        // Calculate confidence level
        let confidence_level = self.calculate_confidence_level(inferred_type, constraints);
        
        // Calculate complexity score
        let complexity_score = self.calculate_complexity_score(&inferred_type.type_info);
        
        // Recognize patterns
        let patterns = self.pattern_engine.recognize_patterns(inferred_type, constraints, context);
        
        // Generate suggestions
        let suggestions = self.generate_suggestions(inferred_type, constraints, context);
        
        // Generate performance notes
        let performance_notes = self.analyze_performance_implications(&inferred_type.type_info);
        
        // Generate usage examples
        let usage_examples = self.generate_usage_examples(&inferred_type.type_info, context);
        
        // Find relevant documentation
        let documentation_links = self.find_documentation_links(&inferred_type.type_info);

        InferenceMetadata {
            semantic_meaning,
            domain_context,
            confidence_level,
            complexity_score,
            patterns,
            suggestions,
            performance_notes,
            usage_examples,
            documentation_links,
        }
    }

    /// Generate global metadata for the entire inference process
    pub fn generate_global_metadata(
        &self,
        node_types: &HashMap<NodeId, InferredType>,
        constraints: &[TypeConstraint],
        errors: &[TypeError],
        statistics: &InferenceStatistics,
    ) -> GlobalInferenceMetadata {
        // Calculate quality metrics
        let quality_metrics = self.quality_calculator.calculate_metrics(
            node_types,
            constraints,
            errors,
        );
        
        // Generate AI insights
        let ai_insights = self.generate_ai_insights(node_types, constraints, errors, &quality_metrics);
        
        // Generate recommendations
        let recommendations = self.recommendation_generator.generate_recommendations(
            node_types,
            constraints,
            errors,
            &quality_metrics,
        );
        
        // Calculate type distribution
        let type_distribution = self.calculate_type_distribution(node_types);
        
        // Calculate overall complexity
        let inference_complexity = self.calculate_overall_complexity(node_types, constraints);

        GlobalInferenceMetadata {
            total_nodes: node_types.len(),
            total_constraints: constraints.len(),
            inference_complexity,
            type_distribution,
            ai_insights,
            statistics: Some(statistics.clone()),
            quality_metrics,
            recommendations,
        }
    }

    // Private helper methods

    fn extract_semantic_meaning(
        &self,
        semantic_type: &SemanticType,
        context: &InferenceContext,
    ) -> String {
        // Simplified implementation using helper functions
        let base_description = get_type_description(semantic_type);
        
        // Add context-specific information
        if let Some(business_context) = context.business_context.as_ref() {
            format!("{} ({})", base_description, business_context)
        } else {
            base_description
        }
    }

    fn calculate_confidence_level(&self, inferred_type: &InferredType, _constraints: &[TypeConstraint]) -> ConfidenceLevel {
        match inferred_type.inference_source {
            InferenceSource::Explicit => ConfidenceLevel::VeryHigh,
            InferenceSource::Literal => ConfidenceLevel::High,
            InferenceSource::Variable | InferenceSource::Function => ConfidenceLevel::Medium,
            InferenceSource::Application | InferenceSource::Pattern => ConfidenceLevel::Medium,
            InferenceSource::Semantic => ConfidenceLevel::High,
            InferenceSource::AIAssisted => ConfidenceLevel::Low,
            InferenceSource::Default => ConfidenceLevel::VeryLow,
            InferenceSource::Context | InferenceSource::Usage => ConfidenceLevel::Medium,
            InferenceSource::Structural => ConfidenceLevel::Medium,
            InferenceSource::Parameter => ConfidenceLevel::High,
            InferenceSource::LetBinding => ConfidenceLevel::High,
            InferenceSource::Return => ConfidenceLevel::High,
            InferenceSource::Conditional => ConfidenceLevel::Medium,
            InferenceSource::PatternMatch => ConfidenceLevel::High,
            InferenceSource::Loop => ConfidenceLevel::Medium,
            InferenceSource::Block => ConfidenceLevel::Medium,
            InferenceSource::Iterator => ConfidenceLevel::Medium,
            InferenceSource::TypeDefinition => ConfidenceLevel::VeryHigh,
            InferenceSource::Operator => ConfidenceLevel::High,
        }
    }

    fn calculate_complexity_score(&self, semantic_type: &SemanticType) -> f64 {
        match semantic_type {
            SemanticType::Primitive(_) => 1.0,
            SemanticType::Variable(_) => 0.5,
            SemanticType::Function { params, return_type, .. } => {
                let param_complexity: f64 = params.iter()
                    .map(|p| self.calculate_complexity_score(p))
                    .sum();
                let return_complexity = self.calculate_complexity_score(return_type);
                2.0 + param_complexity + return_complexity
            }
            SemanticType::List(element_type) => {
                1.5 + self.calculate_complexity_score(element_type)
            }
            SemanticType::Record(fields) => {
                let field_complexity: f64 = fields.values()
                    .map(|f| self.calculate_complexity_score(f))
                    .sum();
                2.0 + field_complexity
            }
            SemanticType::Union(types) => {
                let union_complexity: f64 = types.iter()
                    .map(|t| self.calculate_complexity_score(t))
                    .sum();
                1.5 + union_complexity
            }
            SemanticType::Generic { .. } => 1.2,
            SemanticType::Complex { .. } => 2.0, // Complex types are inherently complex
        }
    }

    fn generate_suggestions(&self, inferred_type: &InferredType, _constraints: &[TypeConstraint], _context: &InferenceContext) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Suggest type annotations for complex types
        if self.calculate_complexity_score(&inferred_type.type_info) > 3.0 {
            suggestions.push("Consider adding explicit type annotations for clarity".to_string());
        }

        // Suggest simplification for overly complex types
        if self.calculate_complexity_score(&inferred_type.type_info) > 5.0 {
            suggestions.push("This type is quite complex - consider simplifying the design".to_string());
        }

        // Suggest documentation for public functions
        if matches!(inferred_type.type_info, SemanticType::Function { .. }) {
            suggestions.push("Consider adding documentation for this function".to_string());
        }

        suggestions
    }

    fn analyze_performance_implications(&self, semantic_type: &SemanticType) -> Vec<String> {
        let mut notes = Vec::new();

        match semantic_type {
            SemanticType::Function { params, .. } if params.len() > 5 => {
                notes.push("Functions with many parameters may have performance implications".to_string());
            }
            SemanticType::List(_) => {
                notes.push("List operations have O(n) complexity for most operations".to_string());
            }
            SemanticType::Record(fields) if fields.len() > 10 => {
                notes.push("Large records may impact memory usage and access performance".to_string());
            }
            SemanticType::Union(types) if types.len() > 5 => {
                notes.push("Large union types may impact pattern matching performance".to_string());
            }
            _ => {}
        }

        notes
    }

    fn generate_usage_examples(&self, semantic_type: &SemanticType, _context: &InferenceContext) -> Vec<String> {
        match semantic_type {
            SemanticType::Primitive(prim) => {
                vec![format!("let value: {} = ...", format_primitive_type(prim))]
            }
            SemanticType::Function { params, return_type, .. } => {
                let param_placeholders = (0..params.len())
                    .map(|i| format!("arg{}", i))
                    .collect::<Vec<_>>()
                    .join(", ");
                vec![format!("function({}) -> {}", param_placeholders, self.type_to_string(return_type))]
            }
            SemanticType::List(element_type) => {
                vec![format!("[{}, {}, ...]", self.type_to_string(element_type), self.type_to_string(element_type))]
            }
            _ => Vec::new(),
        }
    }

    fn find_documentation_links(&self, _semantic_type: &SemanticType) -> Vec<String> {
        // In a real implementation, this would look up relevant documentation
        Vec::new()
    }

    fn generate_ai_insights(
        &self,
        node_types: &HashMap<NodeId, InferredType>,
        _constraints: &[TypeConstraint],
        errors: &[TypeError],
        quality_metrics: &QualityMetrics,
    ) -> Vec<String> {
        let mut insights = Vec::new();

        // Insights about type coverage
        if quality_metrics.annotation_coverage < 0.5 {
            insights.push("Low type annotation coverage - consider adding more explicit types".to_string());
        }

        // Insights about complexity
        if quality_metrics.average_type_complexity > 4.0 {
            insights.push("High average type complexity - consider simplifying type designs".to_string());
        }

        // Insights about errors
        let error_count = errors.len();
        let total_nodes = node_types.len();
        if error_count > 0 {
            let error_rate = error_count as f64 / total_nodes as f64;
            if error_rate > 0.1 {
                insights.push("High error rate detected - review type definitions and usage".to_string());
            }
        }

        // Insights about polymorphism
        if quality_metrics.polymorphism_score > 0.8 {
            insights.push("Good use of polymorphism enhances code reusability".to_string());
        } else if quality_metrics.polymorphism_score < 0.3 {
            insights.push("Consider using more generic/polymorphic types where appropriate".to_string());
        }

        insights
    }

    fn calculate_type_distribution(&self, node_types: &HashMap<NodeId, InferredType>) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for (_, inferred_type) in node_types {
            let type_name = self.get_type_category_name(&inferred_type.type_info);
            *distribution.entry(type_name).or_insert(0) += 1;
        }

        distribution
    }

    fn calculate_overall_complexity(&self, node_types: &HashMap<NodeId, InferredType>, constraints: &[TypeConstraint]) -> f64 {
        let type_complexity: f64 = node_types.values()
            .map(|t| self.calculate_complexity_score(&t.type_info))
            .sum();
        
        let constraint_complexity = constraints.len() as f64 * 0.1;
        let node_complexity = node_types.len() as f64 * 0.05;

        (type_complexity + constraint_complexity + node_complexity) / 100.0
    }

    fn get_type_category_name(&self, semantic_type: &SemanticType) -> String {
        match semantic_type {
            SemanticType::Primitive(prim) => format!("Primitive({})", format_primitive_type(prim)),
            SemanticType::Function { .. } => "Function".to_string(),
            SemanticType::List(_) => "List".to_string(),
            SemanticType::Record(_) => "Record".to_string(),
            SemanticType::Union(_) => "Union".to_string(),
            SemanticType::Variable(_) => "Variable".to_string(),
            SemanticType::Generic { .. } => "Generic".to_string(),
            SemanticType::Complex { name, .. } => format!("Complex({})", name),
        }
    }

    fn type_to_string(&self, semantic_type: &SemanticType) -> String {
        match semantic_type {
            SemanticType::Primitive(prim) => format_primitive_type(prim),
            SemanticType::Variable(var) => var.clone(),
            _ => "...".to_string(),
        }
    }
}

/// Context for type inference
#[derive(Debug, Clone)]
pub struct InferenceContext {
    /// Current function name (if any)
    pub function_name: Option<String>,
    /// Current module name
    pub module_name: String,
    /// Nesting level
    pub nesting_level: usize,
    /// Whether we're in a pattern context
    pub in_pattern: bool,
    /// Whether we're in a type annotation
    pub in_annotation: bool,
    /// Business context for the current inference
    pub business_context: Option<String>,
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self {
            function_name: None,
            module_name: "main".to_string(),
            nesting_level: 0,
            in_pattern: false,
            in_annotation: false,
            business_context: None,
        }
    }
}

// Implementation of helper components

impl PatternEngine {
    fn new() -> Self {
        Self {
            known_patterns: Self::initialize_patterns(),
            threshold: 0.7,
        }
    }

    fn initialize_patterns() -> Vec<PatternDefinition> {
        vec![
            PatternDefinition {
                name: "Simple Function".to_string(),
                description: "A function with 1-3 parameters".to_string(),
                criteria: PatternCriteria::FunctionSignature {
                    param_count: Some(2),
                    return_type: None,
                    complexity: Some(2.0),
                },
                benefits: vec!["Easy to understand and test".to_string()],
                issues: vec!["May be too simple for complex logic".to_string()],
            },
            // Add more patterns as needed
        ]
    }

    fn recognize_patterns(
        &self,
        inferred_type: &InferredType,
        constraints: &[TypeConstraint],
        context: &InferenceContext,
    ) -> Vec<InferencePattern> {
        let mut patterns = Vec::new();
        
        // Analyze the type structure for common patterns
        match &inferred_type.type_info {
            SemanticType::Function { params, return_type, effects } => {
                // Function patterns
                if params.is_empty() {
                    patterns.push(InferencePattern {
                        name: "Nullary Function".to_string(),
                        description: "Function with no parameters - likely a getter or constant".to_string(),
                        confidence: 0.8,
                        benefits: vec![
                            "Simple to call".to_string(),
                            "No parameter validation needed".to_string(),
                        ],
                        issues: vec![
                            "Limited flexibility".to_string(),
                        ],
                    });
                }
                
                if params.len() == 1 {
                    patterns.push(InferencePattern {
                        name: "Unary Function".to_string(),
                        description: "Function with single parameter - common transformation pattern".to_string(),
                        confidence: 0.7,
                        benefits: vec![
                            "Simple interface".to_string(),
                            "Easy to compose".to_string(),
                        ],
                        issues: vec![
                            "May need multiple overloads for different types".to_string(),
                        ],
                    });
                }
                
                if params.len() > 5 {
                    patterns.push(InferencePattern {
                        name: "High Arity Function".to_string(),
                        description: "Function with many parameters - consider parameter object".to_string(),
                        confidence: 0.9,
                        benefits: vec![],
                        issues: vec![
                            "Hard to remember parameter order".to_string(),
                            "Difficult to extend".to_string(),
                            "Consider using a configuration object".to_string(),
                        ],
                    });
                }
                
                if !effects.is_empty() {
                    patterns.push(InferencePattern {
                        name: "Effectful Function".to_string(),
                        description: format!("Function with {} effects: {}", effects.len(), effects.join(", ")),
                        confidence: 0.85,
                        benefits: vec![
                            "Clear about side effects".to_string(),
                        ],
                        issues: vec![
                            "Requires careful handling".to_string(),
                            "May need error handling".to_string(),
                        ],
                    });
                }
            }
            
            SemanticType::Record(fields) => {
                // Record patterns
                if fields.is_empty() {
                    patterns.push(InferencePattern {
                        name: "Empty Record".to_string(),
                        description: "Record with no fields - possibly a marker type".to_string(),
                        confidence: 0.6,
                        benefits: vec![
                            "Lightweight".to_string(),
                        ],
                        issues: vec![
                            "Limited utility".to_string(),
                            "Consider using unit type instead".to_string(),
                        ],
                    });
                } else if fields.len() > 10 {
                    patterns.push(InferencePattern {
                        name: "Large Record".to_string(),
                        description: "Record with many fields - consider breaking into smaller types".to_string(),
                        confidence: 0.8,
                        benefits: vec![],
                        issues: vec![
                            "May violate single responsibility principle".to_string(),
                            "Hard to maintain".to_string(),
                            "Consider composition over large structures".to_string(),
                        ],
                    });
                }
            }
            
            SemanticType::List(element_type) => {
                patterns.push(InferencePattern {
                    name: "Homogeneous Collection".to_string(),
                    description: "List of uniform elements - good for bulk operations".to_string(),
                    confidence: 0.7,
                    benefits: vec![
                        "Efficient iteration".to_string(),
                        "Predictable memory layout".to_string(),
                    ],
                    issues: vec![
                        "All elements must be same type".to_string(),
                    ],
                });
            }
            
            SemanticType::Union(types) => {
                if types.len() == 2 {
                    patterns.push(InferencePattern {
                        name: "Binary Union".to_string(),
                        description: "Union of two types - common for optional or result types".to_string(),
                        confidence: 0.8,
                        benefits: vec![
                            "Clear alternatives".to_string(),
                            "Type-safe error handling".to_string(),
                        ],
                        issues: vec![
                            "Requires pattern matching".to_string(),
                        ],
                    });
                }
            }
            
            _ => {}
        }
        
        // Pattern based on inference confidence
        if inferred_type.confidence < 0.5 {
            patterns.push(InferencePattern {
                name: "Low Confidence Type".to_string(),
                description: "Type inference has low confidence - consider adding annotations".to_string(),
                confidence: 1.0,
                benefits: vec![],
                issues: vec![
                    "May lead to runtime errors".to_string(),
                    "Consider explicit type annotations".to_string(),
                    "Review usage context".to_string(),
                ],
            });
        }
        
        // Pattern based on constraint complexity
        if constraints.len() > 5 {
            patterns.push(InferencePattern {
                name: "Complex Constraints".to_string(),
                description: "Many type constraints - complex type relationships".to_string(),
                confidence: 0.7,
                benefits: vec![
                    "Rich type relationships".to_string(),
                ],
                issues: vec![
                    "May be over-engineered".to_string(),
                    "Consider simplifying".to_string(),
                ],
            });
        }
        
        // Context-based patterns
        if context.in_pattern {
            patterns.push(InferencePattern {
                name: "Pattern Context".to_string(),
                description: "Type used in pattern matching context".to_string(),
                confidence: 0.6,
                benefits: vec![
                    "Supports exhaustive matching".to_string(),
                ],
                issues: vec![
                    "Must handle all cases".to_string(),
                ],
            });
        }
        
        patterns
    }
}

impl SemanticAnalyzer {
    fn new() -> Self {
        Self {
            domain_knowledge: Self::initialize_domain_knowledge(),
            similarity_threshold: 0.8,
        }
    }

    fn initialize_domain_knowledge() -> HashMap<String, DomainInfo> {
        let mut knowledge = HashMap::new();
        
        knowledge.insert("general".to_string(), DomainInfo {
            name: "General Programming".to_string(),
            common_types: vec!["Int".to_string(), "String".to_string(), "Bool".to_string()],
            patterns: vec!["Function composition".to_string()],
            best_practices: vec!["Use descriptive names".to_string()],
        });
        
        knowledge.insert("web".to_string(), DomainInfo {
            name: "Web Development".to_string(),
            common_types: vec![
                "String".to_string(), 
                "Number".to_string(), 
                "Boolean".to_string(),
                "HttpRequest".to_string(),
                "HttpResponse".to_string(),
                "URL".to_string(),
                "JSON".to_string(),
            ],
            patterns: vec![
                "Request-Response".to_string(), 
                "MVC".to_string(),
                "REST API".to_string(),
                "Middleware".to_string(),
                "Authentication".to_string(),
            ],
            best_practices: vec![
                "Validate input".to_string(), 
                "Handle errors gracefully".to_string(),
                "Use HTTPS".to_string(),
                "Sanitize output".to_string(),
                "Rate limiting".to_string(),
            ],
        });
        
        knowledge.insert("database".to_string(), DomainInfo {
            name: "Database".to_string(),
            common_types: vec![
                "Connection".to_string(),
                "Query".to_string(),
                "Result".to_string(),
                "Transaction".to_string(),
                "Schema".to_string(),
            ],
            patterns: vec![
                "Repository".to_string(),
                "Unit of Work".to_string(),
                "Active Record".to_string(),
                "Data Mapper".to_string(),
            ],
            best_practices: vec![
                "Use parameterized queries".to_string(),
                "Handle connection pooling".to_string(),
                "Implement proper indexing".to_string(),
                "Use transactions appropriately".to_string(),
            ],
        });
        
        knowledge.insert("math".to_string(), DomainInfo {
            name: "Mathematics".to_string(),
            common_types: vec![
                "Integer".to_string(),
                "Float".to_string(),
                "Complex".to_string(),
                "Vector".to_string(),
                "Matrix".to_string(),
                "Set".to_string(),
            ],
            patterns: vec![
                "Functional composition".to_string(),
                "Immutable operations".to_string(),
                "Lazy evaluation".to_string(),
            ],
            best_practices: vec![
                "Handle precision carefully".to_string(),
                "Check for division by zero".to_string(),
                "Use appropriate numeric types".to_string(),
                "Consider overflow/underflow".to_string(),
            ],
        });
        
        knowledge.insert("system".to_string(), DomainInfo {
            name: "System Programming".to_string(),
            common_types: vec![
                "FileHandle".to_string(),
                "Process".to_string(),
                "Thread".to_string(),
                "Memory".to_string(),
                "Socket".to_string(),
            ],
            patterns: vec![
                "RAII".to_string(),
                "Producer-Consumer".to_string(),
                "Observer".to_string(),
                "State Machine".to_string(),
            ],
            best_practices: vec![
                "Handle resource cleanup".to_string(),
                "Avoid race conditions".to_string(),
                "Use appropriate synchronization".to_string(),
                "Handle system errors".to_string(),
            ],
        });

        knowledge
    }

    fn analyze_domain(&self, _semantic_type: &SemanticType) -> String {
        // Domain analysis logic would go here
        "general".to_string()
    }
}

impl QualityCalculator {
    fn new() -> Self {
        Self {
            weights: QualityWeights {
                annotation_weight: 0.3,
                polymorphism_weight: 0.2,
                complexity_weight: 0.2,
                error_weight: 0.3,
            },
        }
    }

    fn calculate_metrics(
        &self,
        node_types: &HashMap<NodeId, InferredType>,
        _constraints: &[TypeConstraint],
        errors: &[TypeError],
    ) -> QualityMetrics {
        let total_nodes = node_types.len() as f64;
        
        // Calculate annotation coverage
        let explicit_annotations = node_types.values()
            .filter(|t| t.inference_source == InferenceSource::Explicit)
            .count() as f64;
        let annotation_coverage = if total_nodes > 0.0 {
            explicit_annotations / total_nodes
        } else {
            0.0
        };

        // Calculate average complexity
        let total_complexity: f64 = node_types.values()
            .map(|t| self.calculate_type_complexity(&t.type_info))
            .sum();
        let average_type_complexity = if total_nodes > 0.0 {
            total_complexity / total_nodes
        } else {
            0.0
        };

        // Calculate error proneness
        let error_count = errors.len() as f64;
        let error_proneness = if total_nodes > 0.0 {
            error_count / total_nodes
        } else {
            0.0
        };

        QualityMetrics {
            annotation_coverage,
            polymorphism_score: 0.5, // Placeholder
            average_type_complexity,
            error_proneness,
            maintainability: 0.8 - error_proneness, // Simple heuristic
        }
    }

    fn calculate_type_complexity(&self, semantic_type: &SemanticType) -> f64 {
        estimate_type_complexity(semantic_type)
    }
}

impl RecommendationGenerator {
    fn new() -> Self {
        Self {
            rules: Self::initialize_rules(),
            min_confidence: 0.6,
        }
    }

    fn initialize_rules() -> Vec<RecommendationRule> {
        vec![
            RecommendationRule {
                name: "Complex Type Annotation".to_string(),
                condition: RuleCondition::TypeComplexity(4.0),
                recommendation: RecommendationTemplate {
                    category: RecommendationCategory::TypeAnnotations,
                    description: "Consider adding explicit type annotations for complex types".to_string(),
                    priority: Priority::Medium,
                    impact: Impact {
                        performance: 0.0,
                        readability: 0.3,
                        maintainability: 0.4,
                        error_reduction: 0.2,
                    },
                },
            },
        ]
    }

    fn generate_recommendations(
        &self,
        node_types: &HashMap<NodeId, InferredType>,
        _constraints: &[TypeConstraint],
        _errors: &[TypeError],
        quality_metrics: &QualityMetrics,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Generate recommendations based on quality metrics
        if quality_metrics.annotation_coverage < 0.5 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::TypeAnnotations,
                description: "Increase type annotation coverage to improve code clarity".to_string(),
                priority: Priority::Medium,
                impact: Impact {
                    performance: 0.0,
                    readability: 0.4,
                    maintainability: 0.3,
                    error_reduction: 0.2,
                },
                locations: Vec::new(), // Would be populated with actual locations
                examples: vec![
                    CodeExample {
                        description: "Add type annotation to function parameter".to_string(),
                        before: "function process(data) { ... }".to_string(),
                        after: "function process(data: ProcessingData) { ... }".to_string(),
                        explanation: "Explicit type annotation improves code clarity and catches type errors early".to_string(),
                    },
                ],
            });
        }

        recommendations
    }
}

impl Default for TypeInferenceAI {
    fn default() -> Self {
        Self::new()
    }
}

// Helper function
fn format_primitive_type(prim: &prism_ast::PrimitiveType) -> String {
    match prim {
        prism_ast::PrimitiveType::Integer(_) => "Integer".to_string(),
        prism_ast::PrimitiveType::Float(_) => "Float".to_string(),
        prism_ast::PrimitiveType::String => "String".to_string(),
        prism_ast::PrimitiveType::Boolean => "Boolean".to_string(),
        prism_ast::PrimitiveType::Char => "Char".to_string(),
        prism_ast::PrimitiveType::Unit => "Unit".to_string(),
        prism_ast::PrimitiveType::Never => "Never".to_string(),
        prism_ast::PrimitiveType::Int32 => "Int32".to_string(),
        prism_ast::PrimitiveType::Int64 => "Int64".to_string(),
        prism_ast::PrimitiveType::Float64 => "Float64".to_string(),
    }
}