//! AI-readable metadata generation
//!
//! This module embodies the single concept of "AI Metadata Generation".
//! Following Prism's Conceptual Cohesion principle, this module is responsible
//! for ONE thing: generating AI-comprehensible metadata from documentation.

use crate::{DocumentationError, DocumentationResult, AIIntegrationConfig, AIDetailLevel};
use crate::extraction::{ExtractedDocumentation, DocumentationElement, DocumentationElementType, AIContextInfo};
use prism_ast::{Program, AstNode, Item};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// AI metadata generator for external AI tool consumption
#[derive(Debug)]
pub struct AIMetadataGenerator {
    /// Configuration for AI integration
    config: AIIntegrationConfig,
    /// Context extractors for different element types
    context_extractors: HashMap<DocumentationElementType, Box<dyn ContextExtractor>>,
}

/// AI context extractor trait
pub trait ContextExtractor: Send + Sync {
    /// Extract AI context from a documentation element
    fn extract_context(&self, element: &DocumentationElement) -> DocumentationResult<AIElementContext>;
    
    /// Get the element type this extractor handles
    fn element_type(&self) -> DocumentationElementType;
}

/// Comprehensive AI documentation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIDocumentationMetadata {
    /// Metadata version for compatibility
    pub version: String,
    /// Generation timestamp
    pub generated_at: String,
    /// Overall program context
    pub program_context: ProgramAIContext,
    /// Individual element contexts
    pub element_contexts: Vec<AIElementContext>,
    /// Cross-element relationships
    pub relationships: Vec<AIRelationship>,
    /// Business domain information
    pub business_domain: Option<BusinessDomainInfo>,
    /// Architectural patterns detected
    pub architectural_patterns: Vec<ArchitecturalPattern>,
    /// AI comprehension aids
    pub comprehension_aids: AIComprehensionAids,
    /// Quality metrics
    pub quality_metrics: DocumentationQualityMetrics,
}

/// Program-level AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramAIContext {
    /// Primary purpose of the program/module
    pub primary_purpose: Option<String>,
    /// Business capabilities provided
    pub capabilities: Vec<String>,
    /// Key responsibilities
    pub responsibilities: Vec<String>,
    /// Domain concepts used
    pub domain_concepts: Vec<String>,
    /// Integration points
    pub integration_points: Vec<IntegrationPoint>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Security considerations
    pub security_considerations: Vec<String>,
}

/// Individual element AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIElementContext {
    /// Element name
    pub name: String,
    /// Element type
    pub element_type: DocumentationElementType,
    /// Primary purpose
    pub purpose: Option<String>,
    /// Business context
    pub business_context: Option<String>,
    /// Functional description
    pub functional_description: Option<String>,
    /// Input/output specification
    pub io_specification: Option<IOSpecification>,
    /// Behavioral characteristics
    pub behavioral_characteristics: Vec<BehavioralCharacteristic>,
    /// Constraints and limitations
    pub constraints: Vec<String>,
    /// Usage examples
    pub usage_examples: Vec<UsageExample>,
    /// Common patterns
    pub common_patterns: Vec<String>,
    /// Anti-patterns to avoid
    pub anti_patterns: Vec<String>,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// AI-specific hints
    pub ai_hints: Vec<String>,
    /// Error conditions
    pub error_conditions: Vec<ErrorCondition>,
    /// Performance characteristics
    pub performance_characteristics: Vec<PerformanceCharacteristic>,
    /// Security implications
    pub security_implications: Vec<SecurityImplication>,
}

/// Input/output specification for functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSpecification {
    /// Input parameters
    pub inputs: Vec<ParameterSpec>,
    /// Output specification
    pub outputs: Vec<OutputSpec>,
    /// Side effects
    pub side_effects: Vec<SideEffectSpec>,
}

/// Parameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// Parameter name
    pub name: String,
    /// Parameter type (if available)
    pub param_type: Option<String>,
    /// Parameter description
    pub description: Option<String>,
    /// Whether parameter is optional
    pub optional: bool,
    /// Valid value ranges or constraints
    pub constraints: Vec<String>,
    /// Example values
    pub examples: Vec<String>,
}

/// Output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Output type (if available)
    pub output_type: Option<String>,
    /// Output description
    pub description: Option<String>,
    /// Possible values or ranges
    pub possible_values: Vec<String>,
    /// Success conditions
    pub success_conditions: Vec<String>,
}

/// Side effect specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffectSpec {
    /// Effect type
    pub effect_type: String,
    /// Effect description
    pub description: String,
    /// Conditions under which effect occurs
    pub conditions: Vec<String>,
}

/// Behavioral characteristic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralCharacteristic {
    /// Characteristic type
    pub characteristic_type: BehavioralCharacteristicType,
    /// Description
    pub description: String,
    /// Implications
    pub implications: Vec<String>,
}

/// Types of behavioral characteristics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BehavioralCharacteristicType {
    /// Pure function (no side effects)
    Pure,
    /// Idempotent operation
    Idempotent,
    /// Asynchronous operation
    Asynchronous,
    /// Stateful operation
    Stateful,
    /// Caching behavior
    Cached,
    /// Retry behavior
    Retryable,
    /// Thread-safe operation
    ThreadSafe,
}

/// Usage example with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExample {
    /// Example title
    pub title: String,
    /// Example code
    pub code: String,
    /// Example description
    pub description: Option<String>,
    /// Context where this example applies
    pub context: String,
    /// Expected outcome
    pub expected_outcome: Option<String>,
}

/// Error condition specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCondition {
    /// Error type
    pub error_type: String,
    /// Error description
    pub description: String,
    /// Conditions that trigger this error
    pub trigger_conditions: Vec<String>,
    /// Recovery strategies
    pub recovery_strategies: Vec<String>,
}

/// Performance characteristic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristic {
    /// Characteristic type
    pub characteristic_type: PerformanceCharacteristicType,
    /// Description
    pub description: String,
    /// Quantitative measures (if available)
    pub measures: Vec<String>,
}

/// Types of performance characteristics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceCharacteristicType {
    /// Time complexity
    TimeComplexity,
    /// Space complexity
    SpaceComplexity,
    /// Throughput
    Throughput,
    /// Latency
    Latency,
    /// Resource usage
    ResourceUsage,
    /// Scalability
    Scalability,
}

/// Security implication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplication {
    /// Implication type
    pub implication_type: SecurityImplicationType,
    /// Description
    pub description: String,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Types of security implications
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityImplicationType {
    /// Data access
    DataAccess,
    /// Authentication
    Authentication,
    /// Authorization
    Authorization,
    /// Input validation
    InputValidation,
    /// Output sanitization
    OutputSanitization,
    /// Cryptographic operation
    Cryptographic,
    /// Network communication
    NetworkCommunication,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

/// Relationship between AI elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRelationship {
    /// Source element
    pub source: String,
    /// Target element
    pub target: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Description of the relationship
    pub description: String,
}

/// Types of relationships
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Dependency relationship
    Dependency,
    /// Composition relationship
    Composition,
    /// Usage relationship
    Usage,
    /// Inheritance relationship
    Inheritance,
    /// Association relationship
    Association,
    /// Data flow relationship
    DataFlow,
    /// Control flow relationship
    ControlFlow,
}

/// Business domain information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessDomainInfo {
    /// Primary domain
    pub primary_domain: String,
    /// Sub-domains
    pub sub_domains: Vec<String>,
    /// Key entities
    pub key_entities: Vec<String>,
    /// Business processes
    pub business_processes: Vec<String>,
    /// Domain rules
    pub domain_rules: Vec<String>,
    /// Compliance frameworks
    pub compliance_frameworks: Vec<String>,
}

/// Architectural pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalPattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Elements that implement this pattern
    pub implementing_elements: Vec<String>,
    /// Pattern benefits
    pub benefits: Vec<String>,
    /// Pattern drawbacks
    pub drawbacks: Vec<String>,
}

/// Integration point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPoint {
    /// Integration point name
    pub name: String,
    /// Integration type
    pub integration_type: IntegrationType,
    /// Description
    pub description: String,
    /// External dependencies
    pub external_dependencies: Vec<String>,
}

/// Types of integration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationType {
    /// API integration
    API,
    /// Database integration
    Database,
    /// File system integration
    FileSystem,
    /// Network integration
    Network,
    /// External service integration
    ExternalService,
    /// Library integration
    Library,
}

/// AI comprehension aids
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIComprehensionAids {
    /// Key concepts glossary
    pub glossary: Vec<GlossaryEntry>,
    /// Mental models
    pub mental_models: Vec<MentalModel>,
    /// Decision trees
    pub decision_trees: Vec<DecisionTree>,
    /// Common workflows
    pub workflows: Vec<Workflow>,
}

/// Glossary entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryEntry {
    /// Term
    pub term: String,
    /// Definition
    pub definition: String,
    /// Context
    pub context: String,
    /// Related terms
    pub related_terms: Vec<String>,
}

/// Mental model for understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalModel {
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Key concepts
    pub key_concepts: Vec<String>,
    /// Relationships
    pub relationships: Vec<String>,
}

/// Decision tree for AI decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Tree name
    pub name: String,
    /// Root condition
    pub root_condition: String,
    /// Decision branches
    pub branches: Vec<DecisionBranch>,
}

/// Decision branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBranch {
    /// Condition
    pub condition: String,
    /// Action or outcome
    pub outcome: String,
    /// Sub-branches
    pub sub_branches: Vec<DecisionBranch>,
}

/// Workflow description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: String,
    /// Steps
    pub steps: Vec<WorkflowStep>,
}

/// Workflow step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Prerequisites
    pub prerequisites: Vec<String>,
    /// Outcomes
    pub outcomes: Vec<String>,
}

/// Documentation quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationQualityMetrics {
    /// Completeness score (0.0 to 1.0)
    pub completeness_score: f64,
    /// Clarity score (0.0 to 1.0)
    pub clarity_score: f64,
    /// AI comprehensibility score (0.0 to 1.0)
    pub ai_comprehensibility_score: f64,
    /// Consistency score (0.0 to 1.0)
    pub consistency_score: f64,
    /// Coverage metrics
    pub coverage_metrics: CoverageMetrics,
}

/// Coverage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics {
    /// Percentage of elements with documentation
    pub documented_elements_percentage: f64,
    /// Percentage of elements with examples
    pub elements_with_examples_percentage: f64,
    /// Percentage of elements with AI context
    pub elements_with_ai_context_percentage: f64,
    /// Percentage of functions with parameter documentation
    pub functions_with_param_docs_percentage: f64,
}

impl AIMetadataGenerator {
    /// Create a new AI metadata generator
    pub fn new(config: AIIntegrationConfig) -> Self {
        let mut generator = Self {
            config,
            context_extractors: HashMap::new(),
        };
        
        // Register default context extractors
        generator.register_default_extractors();
        generator
    }

    /// Generate AI metadata for a complete program
    pub fn generate_for_program(
        &self,
        program: &Program,
        docs: &ExtractedDocumentation,
    ) -> DocumentationResult<AIDocumentationMetadata> {
        let program_context = self.extract_program_context(program, docs)?;
        let element_contexts = self.extract_element_contexts(&docs.elements)?;
        let relationships = self.analyze_relationships(&element_contexts)?;
        let business_domain = self.extract_business_domain(docs)?;
        let architectural_patterns = self.detect_architectural_patterns(&element_contexts)?;
        let comprehension_aids = self.generate_comprehension_aids(&element_contexts, &program_context)?;
        let quality_metrics = self.calculate_quality_metrics(docs, &element_contexts)?;

        Ok(AIDocumentationMetadata {
            version: "1.0".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            program_context,
            element_contexts,
            relationships,
            business_domain,
            architectural_patterns,
            comprehension_aids,
            quality_metrics,
        })
    }

    /// Extract program-level AI context
    fn extract_program_context(
        &self,
        program: &Program,
        docs: &ExtractedDocumentation,
    ) -> DocumentationResult<ProgramAIContext> {
        let mut capabilities = Vec::new();
        let mut responsibilities = Vec::new();
        let mut domain_concepts = Vec::new();
        let mut integration_points = Vec::new();
        let mut compliance_requirements = Vec::new();
        let mut security_considerations = Vec::new();

        // Extract from module documentation if available
        if let Some(module_doc) = &docs.module_documentation {
            if let Some(responsibility) = &module_doc.responsibility {
                responsibilities.push(responsibility.clone());
            }
            
            // Extract domain concepts from description
            if let Some(description) = &module_doc.description {
                domain_concepts.extend(self.extract_domain_concepts(description));
            }
        }

        // Extract from individual elements
        for element in &docs.elements {
            // Extract capabilities from function responsibilities
            if element.element_type == DocumentationElementType::Function {
                for annotation in &element.annotations {
                    if annotation.name == "responsibility" {
                        if let Some(value) = &annotation.value {
                            capabilities.push(value.clone());
                        }
                    }
                }
            }

            // Extract security considerations
            if let Some(ai_context) = &element.ai_context {
                security_considerations.extend(ai_context.constraints.clone());
            }
        }

        Ok(ProgramAIContext {
            primary_purpose: docs.module_documentation.as_ref().and_then(|m| m.description.clone()),
            capabilities,
            responsibilities,
            domain_concepts,
            integration_points,
            compliance_requirements,
            security_considerations,
        })
    }

    /// Extract AI context for individual elements
    fn extract_element_contexts(
        &self,
        elements: &[DocumentationElement],
    ) -> DocumentationResult<Vec<AIElementContext>> {
        let mut contexts = Vec::new();

        for element in elements {
            let context = if let Some(extractor) = self.context_extractors.get(&element.element_type) {
                extractor.extract_context(element)?
            } else {
                self.extract_default_context(element)?
            };
            contexts.push(context);
        }

        Ok(contexts)
    }

    /// Extract default AI context for an element
    fn extract_default_context(&self, element: &DocumentationElement) -> DocumentationResult<AIElementContext> {
        let purpose = self.extract_purpose_from_element(element);
        let business_context = self.extract_business_context_from_element(element);
        let functional_description = element.content.clone();
        let constraints = self.extract_constraints_from_element(element);
        let usage_examples = self.extract_usage_examples_from_element(element);
        let ai_hints = self.extract_ai_hints_from_element(element);

        Ok(AIElementContext {
            name: element.name.clone(),
            element_type: element.element_type.clone(),
            purpose,
            business_context,
            functional_description,
            io_specification: None, // TODO: Extract from function signatures
            behavioral_characteristics: Vec::new(), // TODO: Infer from annotations
            constraints,
            usage_examples,
            common_patterns: Vec::new(), // TODO: Detect patterns
            anti_patterns: Vec::new(), // TODO: Detect anti-patterns
            related_concepts: Vec::new(), // TODO: Extract related concepts
            ai_hints,
            error_conditions: Vec::new(), // TODO: Extract from @throws annotations
            performance_characteristics: Vec::new(), // TODO: Extract from @performance annotations
            security_implications: Vec::new(), // TODO: Extract from @security annotations
        })
    }

    /// Extract purpose from element
    fn extract_purpose_from_element(&self, element: &DocumentationElement) -> Option<String> {
        // First try responsibility annotation
        for annotation in &element.annotations {
            if annotation.name == "responsibility" {
                return annotation.value.clone();
            }
        }

        // Then try description
        for annotation in &element.annotations {
            if annotation.name == "description" {
                return annotation.value.clone();
            }
        }

        // Finally try content
        element.content.clone()
    }

    /// Extract business context from element
    fn extract_business_context_from_element(&self, element: &DocumentationElement) -> Option<String> {
        element.ai_context.as_ref().and_then(|ctx| ctx.business_context.clone())
    }

    /// Extract constraints from element
    fn extract_constraints_from_element(&self, element: &DocumentationElement) -> Vec<String> {
        let mut constraints = Vec::new();

        if let Some(ai_context) = &element.ai_context {
            constraints.extend(ai_context.constraints.clone());
        }

        // Extract from annotations
        for annotation in &element.annotations {
            match annotation.name.as_str() {
                "requires" | "precondition" | "constraint" => {
                    if let Some(value) = &annotation.value {
                        constraints.push(value.clone());
                    }
                }
                _ => {}
            }
        }

        constraints
    }

    /// Extract usage examples from element
    fn extract_usage_examples_from_element(&self, element: &DocumentationElement) -> Vec<UsageExample> {
        let mut examples = Vec::new();

        for annotation in &element.annotations {
            if annotation.name == "example" {
                if let Some(value) = &annotation.value {
                    examples.push(UsageExample {
                        title: format!("Example for {}", element.name),
                        code: value.clone(),
                        description: None,
                        context: "General usage".to_string(),
                        expected_outcome: None,
                    });
                }
            }
        }

        // Extract from AI context
        if let Some(ai_context) = &element.ai_context {
            for example in &ai_context.examples {
                examples.push(UsageExample {
                    title: format!("AI Example for {}", element.name),
                    code: example.clone(),
                    description: None,
                    context: "AI-generated example".to_string(),
                    expected_outcome: None,
                });
            }
        }

        examples
    }

    /// Extract AI hints from element
    fn extract_ai_hints_from_element(&self, element: &DocumentationElement) -> Vec<String> {
        let mut hints = Vec::new();

        if let Some(ai_context) = &element.ai_context {
            hints.extend(ai_context.ai_hints.clone());
        }

        // Extract from annotations
        for annotation in &element.annotations {
            if annotation.name.starts_with("ai") || annotation.name == "hint" {
                if let Some(value) = &annotation.value {
                    hints.push(value.clone());
                }
            }
        }

        hints
    }

    /// Analyze relationships between elements
    fn analyze_relationships(&self, contexts: &[AIElementContext]) -> DocumentationResult<Vec<AIRelationship>> {
        let mut relationships = Vec::new();

        // Simple relationship analysis based on naming and usage patterns
        for (i, source) in contexts.iter().enumerate() {
            for (j, target) in contexts.iter().enumerate() {
                if i != j {
                    if let Some(relationship) = self.detect_relationship(source, target) {
                        relationships.push(relationship);
                    }
                }
            }
        }

        Ok(relationships)
    }

    /// Detect relationship between two elements
    fn detect_relationship(&self, source: &AIElementContext, target: &AIElementContext) -> Option<AIRelationship> {
        // Simple heuristics for relationship detection
        
        // Check if source mentions target in description or examples
        let source_text = format!("{} {} {}", 
            source.functional_description.as_deref().unwrap_or(""),
            source.purpose.as_deref().unwrap_or(""),
            source.usage_examples.iter().map(|ex| &ex.code).collect::<Vec<_>>().join(" ")
        );

        if source_text.contains(&target.name) {
            return Some(AIRelationship {
                source: source.name.clone(),
                target: target.name.clone(),
                relationship_type: RelationshipType::Usage,
                strength: 0.7, // Medium confidence
                description: format!("{} uses {}", source.name, target.name),
            });
        }

        // Check for naming patterns that suggest relationships
        if source.name.contains(&target.name) || target.name.contains(&source.name) {
            return Some(AIRelationship {
                source: source.name.clone(),
                target: target.name.clone(),
                relationship_type: RelationshipType::Association,
                strength: 0.5, // Lower confidence for naming patterns
                description: format!("{} is associated with {}", source.name, target.name),
            });
        }

        None
    }

    /// Extract business domain information
    fn extract_business_domain(&self, docs: &ExtractedDocumentation) -> DocumentationResult<Option<BusinessDomainInfo>> {
        // For now, return None - this would be enhanced with more sophisticated domain analysis
        Ok(None)
    }

    /// Detect architectural patterns
    fn detect_architectural_patterns(&self, contexts: &[AIElementContext]) -> DocumentationResult<Vec<ArchitecturalPattern>> {
        let mut patterns = Vec::new();

        // Simple pattern detection based on naming and structure
        let function_names: Vec<&str> = contexts.iter()
            .filter(|ctx| ctx.element_type == DocumentationElementType::Function)
            .map(|ctx| ctx.name.as_str())
            .collect();

        // Detect repository pattern
        if function_names.iter().any(|name| name.contains("Repository") || name.contains("repository")) {
            patterns.push(ArchitecturalPattern {
                name: "Repository Pattern".to_string(),
                description: "Data access abstraction pattern".to_string(),
                implementing_elements: function_names.iter()
                    .filter(|name| name.contains("Repository") || name.contains("repository"))
                    .map(|s| s.to_string())
                    .collect(),
                benefits: vec!["Separates data access logic".to_string(), "Testable".to_string()],
                drawbacks: vec!["Additional abstraction layer".to_string()],
            });
        }

        // Detect service pattern
        if function_names.iter().any(|name| name.contains("Service") || name.contains("service")) {
            patterns.push(ArchitecturalPattern {
                name: "Service Pattern".to_string(),
                description: "Business logic encapsulation pattern".to_string(),
                implementing_elements: function_names.iter()
                    .filter(|name| name.contains("Service") || name.contains("service"))
                    .map(|s| s.to_string())
                    .collect(),
                benefits: vec!["Encapsulates business logic".to_string(), "Reusable".to_string()],
                drawbacks: vec!["Can become too large".to_string()],
            });
        }

        Ok(patterns)
    }

    /// Generate comprehension aids
    fn generate_comprehension_aids(
        &self,
        contexts: &[AIElementContext],
        program_context: &ProgramAIContext,
    ) -> DocumentationResult<AIComprehensionAids> {
        let glossary = self.generate_glossary(contexts, program_context)?;
        let mental_models = self.generate_mental_models(contexts)?;
        let decision_trees = Vec::new(); // TODO: Generate decision trees
        let workflows = Vec::new(); // TODO: Generate workflows

        Ok(AIComprehensionAids {
            glossary,
            mental_models,
            decision_trees,
            workflows,
        })
    }

    /// Generate glossary from contexts
    fn generate_glossary(
        &self,
        contexts: &[AIElementContext],
        program_context: &ProgramAIContext,
    ) -> DocumentationResult<Vec<GlossaryEntry>> {
        let mut glossary = Vec::new();

        // Add domain concepts from program context
        for concept in &program_context.domain_concepts {
            glossary.push(GlossaryEntry {
                term: concept.clone(),
                definition: format!("Domain concept: {}", concept),
                context: "Program domain".to_string(),
                related_terms: Vec::new(),
            });
        }

        // Add key elements as glossary entries
        for context in contexts {
            if let Some(purpose) = &context.purpose {
                glossary.push(GlossaryEntry {
                    term: context.name.clone(),
                    definition: purpose.clone(),
                    context: format!("{:?}", context.element_type),
                    related_terms: context.related_concepts.clone(),
                });
            }
        }

        Ok(glossary)
    }

    /// Generate mental models
    fn generate_mental_models(&self, contexts: &[AIElementContext]) -> DocumentationResult<Vec<MentalModel>> {
        let mut models = Vec::new();

        // Generate a simple mental model based on element types
        let function_names: Vec<String> = contexts.iter()
            .filter(|ctx| ctx.element_type == DocumentationElementType::Function)
            .map(|ctx| ctx.name.clone())
            .collect();

        let type_names: Vec<String> = contexts.iter()
            .filter(|ctx| ctx.element_type == DocumentationElementType::Type)
            .map(|ctx| ctx.name.clone())
            .collect();

        if !function_names.is_empty() || !type_names.is_empty() {
            models.push(MentalModel {
                name: "System Structure".to_string(),
                description: "High-level view of system components".to_string(),
                key_concepts: [function_names, type_names].concat(),
                relationships: vec!["Functions operate on types".to_string()],
            });
        }

        Ok(models)
    }

    /// Calculate documentation quality metrics
    fn calculate_quality_metrics(
        &self,
        docs: &ExtractedDocumentation,
        contexts: &[AIElementContext],
    ) -> DocumentationResult<DocumentationQualityMetrics> {
        let completeness_score = self.calculate_completeness_score(&docs.statistics);
        let clarity_score = self.calculate_clarity_score(contexts);
        let ai_comprehensibility_score = self.calculate_ai_comprehensibility_score(contexts);
        let consistency_score = self.calculate_consistency_score(contexts);
        let coverage_metrics = self.calculate_coverage_metrics(&docs.statistics, contexts);

        Ok(DocumentationQualityMetrics {
            completeness_score,
            clarity_score,
            ai_comprehensibility_score,
            consistency_score,
            coverage_metrics,
        })
    }

    /// Calculate completeness score
    fn calculate_completeness_score(&self, stats: &crate::extraction::ExtractionStatistics) -> f64 {
        stats.documentation_coverage / 100.0
    }

    /// Calculate clarity score
    fn calculate_clarity_score(&self, contexts: &[AIElementContext]) -> f64 {
        if contexts.is_empty() {
            return 0.0;
        }

        let clear_descriptions = contexts.iter()
            .filter(|ctx| {
                ctx.purpose.is_some() || ctx.functional_description.is_some()
            })
            .count();

        clear_descriptions as f64 / contexts.len() as f64
    }

    /// Calculate AI comprehensibility score
    fn calculate_ai_comprehensibility_score(&self, contexts: &[AIElementContext]) -> f64 {
        if contexts.is_empty() {
            return 0.0;
        }

        let ai_friendly = contexts.iter()
            .filter(|ctx| {
                !ctx.ai_hints.is_empty() || 
                ctx.business_context.is_some() ||
                !ctx.usage_examples.is_empty()
            })
            .count();

        ai_friendly as f64 / contexts.len() as f64
    }

    /// Calculate consistency score
    fn calculate_consistency_score(&self, contexts: &[AIElementContext]) -> f64 {
        // Simple consistency check based on naming patterns and structure
        if contexts.is_empty() {
            return 1.0;
        }

        let consistent_elements = contexts.iter()
            .filter(|ctx| {
                // Check if element has consistent documentation structure
                match ctx.element_type {
                    DocumentationElementType::Function => ctx.purpose.is_some(),
                    DocumentationElementType::Type => ctx.purpose.is_some(),
                    DocumentationElementType::Module => ctx.purpose.is_some(),
                    _ => true,
                }
            })
            .count();

        consistent_elements as f64 / contexts.len() as f64
    }

    /// Calculate coverage metrics
    fn calculate_coverage_metrics(
        &self,
        stats: &crate::extraction::ExtractionStatistics,
        contexts: &[AIElementContext],
    ) -> CoverageMetrics {
        let documented_elements_percentage = stats.documentation_coverage;
        
        let elements_with_examples_percentage = if contexts.is_empty() {
            0.0
        } else {
            let with_examples = contexts.iter()
                .filter(|ctx| !ctx.usage_examples.is_empty())
                .count();
            (with_examples as f64 / contexts.len() as f64) * 100.0
        };

        let elements_with_ai_context_percentage = if contexts.is_empty() {
            0.0
        } else {
            let with_ai_context = contexts.iter()
                .filter(|ctx| !ctx.ai_hints.is_empty() || ctx.business_context.is_some())
                .count();
            (with_ai_context as f64 / contexts.len() as f64) * 100.0
        };

        let functions_with_param_docs_percentage = {
            let functions: Vec<_> = contexts.iter()
                .filter(|ctx| ctx.element_type == DocumentationElementType::Function)
                .collect();
            
            if functions.is_empty() {
                0.0
            } else {
                let with_param_docs = functions.iter()
                    .filter(|ctx| {
                        ctx.io_specification.as_ref()
                            .map_or(false, |io| !io.inputs.is_empty())
                    })
                    .count();
                (with_param_docs as f64 / functions.len() as f64) * 100.0
            }
        };

        CoverageMetrics {
            documented_elements_percentage,
            elements_with_examples_percentage,
            elements_with_ai_context_percentage,
            functions_with_param_docs_percentage,
        }
    }

    /// Extract domain concepts from text
    fn extract_domain_concepts(&self, text: &str) -> Vec<String> {
        // Simple keyword extraction - would be enhanced with NLP
        let common_domain_words = ["user", "account", "payment", "order", "product", "service", "data", "system"];
        
        common_domain_words.iter()
            .filter(|word| text.to_lowercase().contains(*word))
            .map(|word| word.to_string())
            .collect()
    }

    /// Register default context extractors
    fn register_default_extractors(&mut self) {
        // For now, we'll use the default extraction for all types
        // In a full implementation, we'd have specialized extractors for each type
    }
}

impl Default for AIIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            include_business_context: true,
            include_architectural_patterns: true,
            include_semantic_relationships: true,
            detail_level: AIDetailLevel::Standard,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::{DocumentationElement, DocumentationElementType, ExtractedAnnotation, ElementVisibility};

    #[test]
    fn test_ai_context_extraction() {
        let generator = AIMetadataGenerator::new(AIIntegrationConfig::default());
        
        let element = DocumentationElement {
            element_type: DocumentationElementType::Function,
            name: "authenticateUser".to_string(),
            content: Some("Authenticates a user with credentials".to_string()),
            annotations: vec![
                ExtractedAnnotation {
                    name: "responsibility".to_string(),
                    value: Some("Validates user credentials and creates session".to_string()),
                    arguments: vec![],
                    location: Span::dummy(),
                },
                ExtractedAnnotation {
                    name: "example".to_string(),
                    value: Some("authenticateUser(email, password)".to_string()),
                    arguments: vec![],
                    location: Span::dummy(),
                },
            ],
            location: Span::dummy(),
            visibility: ElementVisibility::Public,
            ai_context: Some(AIContextInfo {
                purpose: Some("User authentication".to_string()),
                business_context: Some("Security domain".to_string()),
                constraints: vec!["Must validate email format".to_string()],
                examples: vec!["auth.authenticate('user@example.com', 'password')".to_string()],
                ai_hints: vec!["Returns session token on success".to_string()],
            }),
            jsdoc_info: None,
        };

        let context = generator.extract_default_context(&element).unwrap();
        
        assert_eq!(context.name, "authenticateUser");
        assert_eq!(context.element_type, DocumentationElementType::Function);
        assert_eq!(context.purpose, Some("Validates user credentials and creates session".to_string()));
        assert_eq!(context.business_context, Some("Security domain".to_string()));
        assert!(!context.constraints.is_empty());
        assert!(!context.usage_examples.is_empty());
        assert!(!context.ai_hints.is_empty());
    }

    #[test]
    fn test_relationship_detection() {
        let generator = AIMetadataGenerator::new(AIIntegrationConfig::default());
        
        let source = AIElementContext {
            name: "UserService".to_string(),
            element_type: DocumentationElementType::Function,
            purpose: Some("Manages user operations".to_string()),
            functional_description: Some("Uses UserRepository to store user data".to_string()),
            // ... other fields with defaults
            business_context: None,
            io_specification: None,
            behavioral_characteristics: Vec::new(),
            constraints: Vec::new(),
            usage_examples: Vec::new(),
            common_patterns: Vec::new(),
            anti_patterns: Vec::new(),
            related_concepts: Vec::new(),
            ai_hints: Vec::new(),
            error_conditions: Vec::new(),
            performance_characteristics: Vec::new(),
            security_implications: Vec::new(),
        };

        let target = AIElementContext {
            name: "UserRepository".to_string(),
            element_type: DocumentationElementType::Function,
            purpose: Some("Handles user data persistence".to_string()),
            functional_description: Some("Repository for user data".to_string()),
            // ... other fields with defaults
            business_context: None,
            io_specification: None,
            behavioral_characteristics: Vec::new(),
            constraints: Vec::new(),
            usage_examples: Vec::new(),
            common_patterns: Vec::new(),
            anti_patterns: Vec::new(),
            related_concepts: Vec::new(),
            ai_hints: Vec::new(),
            error_conditions: Vec::new(),
            performance_characteristics: Vec::new(),
            security_implications: Vec::new(),
        };

        let relationship = generator.detect_relationship(&source, &target);
        
        assert!(relationship.is_some());
        let rel = relationship.unwrap();
        assert_eq!(rel.relationship_type, RelationshipType::Usage);
        assert_eq!(rel.source, "UserService");
        assert_eq!(rel.target, "UserRepository");
    }

    #[test]
    fn test_quality_metrics_calculation() {
        let generator = AIMetadataGenerator::new(AIIntegrationConfig::default());
        
        let contexts = vec![
            AIElementContext {
                name: "TestFunction".to_string(),
                element_type: DocumentationElementType::Function,
                purpose: Some("Test purpose".to_string()),
                business_context: Some("Test domain".to_string()),
                ai_hints: vec!["Test hint".to_string()],
                usage_examples: vec![UsageExample {
                    title: "Test example".to_string(),
                    code: "test()".to_string(),
                    description: None,
                    context: "test".to_string(),
                    expected_outcome: None,
                }],
                // ... other fields
                functional_description: None,
                io_specification: None,
                behavioral_characteristics: Vec::new(),
                constraints: Vec::new(),
                common_patterns: Vec::new(),
                anti_patterns: Vec::new(),
                related_concepts: Vec::new(),
                error_conditions: Vec::new(),
                performance_characteristics: Vec::new(),
                security_implications: Vec::new(),
            }
        ];

        let stats = crate::extraction::ExtractionStatistics {
            total_elements: 1,
            documented_elements: 1,
            elements_with_required_annotations: 1,
            missing_documentation_count: 0,
            missing_annotation_count: 0,
            documentation_coverage: 100.0,
        };

        let metrics = generator.calculate_quality_metrics(&stats, &contexts).unwrap();
        
        assert_eq!(metrics.completeness_score, 1.0);
        assert_eq!(metrics.clarity_score, 1.0);
        assert_eq!(metrics.ai_comprehensibility_score, 1.0);
        assert_eq!(metrics.coverage_metrics.documented_elements_percentage, 100.0);
        assert_eq!(metrics.coverage_metrics.elements_with_examples_percentage, 100.0);
        assert_eq!(metrics.coverage_metrics.elements_with_ai_context_percentage, 100.0);
    }
} 