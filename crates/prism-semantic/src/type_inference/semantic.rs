//! Semantic Type Inference Extensions
//!
//! This module extends the basic Hindley-Milner type inference with semantic
//! type awareness, business rule validation, and domain-specific type reasoning.

use super::{
    TypeVar, InferredType, InferenceSource, TypeInferenceResult,
    constraints::{TypeConstraint, ConstraintType, ConstraintReason, ConstraintSet},
    environment::{TypeEnvironment, TypeBinding},
    errors::{TypeError, TypeErrorKind, TypeContext},
};
use crate::{
    SemanticResult, SemanticError,
    types::{SemanticType, SemanticTypeMetadata},
    analyzer::BusinessRule,
};
use prism_ast::Expr;
use prism_ast::expr::{LiteralValue, BinaryOperator, UnaryOperator};
use prism_common::Span;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Null handling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NullHandling {
    /// Explicit null handling required
    Explicit,
    /// Implicit null handling allowed
    Implicit,
    /// Null values forbidden
    Forbidden,
}

/// Semantic type inference engine that extends basic inference
#[derive(Debug)]
pub struct SemanticTypeInference {
    /// Business rule validator
    business_validator: BusinessRuleValidator,
    /// Domain knowledge base
    domain_knowledge: DomainKnowledgeBase,
    /// Semantic constraint generator
    constraint_generator: SemanticConstraintGenerator,
    /// Type refinement engine
    refinement_engine: TypeRefinementEngine,
}

/// Semantic constraint that extends basic type constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConstraint {
    /// Base type constraint
    pub base_constraint: TypeConstraint,
    /// Business rules that must be satisfied
    pub business_rules: Vec<BusinessRule>,
    /// Semantic properties
    pub semantic_properties: Vec<SemanticProperty>,
    /// Domain context
    pub domain_context: String,
    /// Confidence in this constraint
    pub confidence: f64,
}

/// Semantic property of a type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticProperty {
    /// Property name
    pub name: String,
    /// Property value
    pub value: PropertyValue,
    /// Source of this property
    pub source: PropertySource,
}

/// Value of a semantic property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Boolean property
    Boolean(bool),
    /// Numeric property
    Numeric(f64),
    /// String property
    String(String),
    /// Range property
    Range { min: f64, max: f64 },
    /// Set of allowed values
    Enum(Vec<String>),
}

/// Source of a semantic property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertySource {
    /// Explicitly declared
    Explicit,
    /// Inferred from usage
    Inferred,
    /// Derived from business rules
    BusinessRule,
    /// Domain knowledge
    Domain,
    /// AI suggested
    AIGenerated,
}

/// Business rule validator
#[derive(Debug)]
struct BusinessRuleValidator {
    /// Active business rules
    rules: Vec<BusinessRule>,
    /// Rule evaluation cache
    cache: HashMap<String, bool>,
}

/// Domain knowledge base for semantic inference
#[derive(Debug)]
struct DomainKnowledgeBase {
    /// Known domain types and their properties
    domain_types: HashMap<String, DomainTypeInfo>,
    /// Type relationships
    relationships: Vec<TypeRelationship>,
    /// Common patterns
    patterns: Vec<DomainPattern>,
}

/// Information about a domain-specific type
#[derive(Debug, Clone)]
struct DomainTypeInfo {
    /// Type name
    name: String,
    /// Base semantic type
    base_type: SemanticType,
    /// Domain-specific properties
    properties: Vec<SemanticProperty>,
    /// Validation rules
    validation_rules: Vec<ValidationRule>,
    /// Usage patterns
    usage_patterns: Vec<String>,
}

/// Relationship between types
#[derive(Debug, Clone)]
struct TypeRelationship {
    /// Source type
    from_type: String,
    /// Target type
    to_type: String,
    /// Relationship kind
    relationship: RelationshipKind,
    /// Confidence in this relationship
    confidence: f64,
}

/// Kind of type relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelationshipKind {
    /// Subtype relationship
    Subtype,
    /// Conversion relationship
    Convertible,
    /// Composition relationship
    Contains,
    /// Usage relationship
    UsedWith,
    /// Mutual exclusion
    Excludes,
}

/// Domain-specific pattern
#[derive(Debug, Clone)]
struct DomainPattern {
    /// Pattern name
    name: String,
    /// Pattern description
    description: String,
    /// Trigger conditions
    conditions: Vec<PatternCondition>,
    /// Inferred properties
    inferred_properties: Vec<SemanticProperty>,
}

/// Condition for pattern matching
#[derive(Debug, Clone)]
enum PatternCondition {
    /// Type matches pattern
    TypeMatch(String),
    /// Function signature matches
    FunctionSignature { params: usize, returns: String },
    /// Usage context matches
    UsageContext(String),
    /// Name matches pattern
    NamePattern(String),
}

/// Validation rule for domain types
#[derive(Debug, Clone)]
struct ValidationRule {
    /// Rule name
    name: String,
    /// Rule expression
    expression: String,
    /// Error message if rule fails
    error_message: String,
    /// Rule severity
    severity: RuleSeverity,
}

/// Severity of validation rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleSeverity {
    /// Warning - doesn't prevent compilation
    Warning,
    /// Error - prevents compilation
    Error,
    /// Critical - major issue
    Critical,
}

/// Semantic constraint generator
#[derive(Debug)]
struct SemanticConstraintGenerator {
    /// Constraint generation rules
    rules: Vec<ConstraintGenerationRule>,
}

/// Rule for generating semantic constraints
#[derive(Debug, Clone)]
struct ConstraintGenerationRule {
    /// Rule name
    name: String,
    /// Trigger condition
    trigger: ConstraintTrigger,
    /// Generated constraint template
    constraint_template: SemanticConstraintTemplate,
}

/// Trigger for constraint generation
#[derive(Debug, Clone)]
enum ConstraintTrigger {
    /// Literal value
    Literal(LiteralPattern),
    /// Function application
    FunctionApplication(String),
    /// Binary operation
    BinaryOperation(BinaryOperator),
    /// Field access
    FieldAccess(String),
    /// Type annotation
    TypeAnnotation(String),
}

/// Pattern for literal values
#[derive(Debug, Clone)]
struct LiteralPattern {
    /// Type of literal
    literal_type: LiteralType,
    /// Value pattern (regex or exact match)
    pattern: String,
    /// Case sensitive matching
    case_sensitive: bool,
}

/// Type of literal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralType {
    String,
    Integer,
    Float,
    Boolean,
}

/// Template for semantic constraints
#[derive(Debug, Clone)]
struct SemanticConstraintTemplate {
    /// Business rules to add
    business_rules: Vec<BusinessRule>,
    /// Semantic properties to infer
    properties: Vec<SemanticProperty>,
    /// Domain context to assign
    domain_context: String,
}

/// Type refinement engine for improving inference
#[derive(Debug)]
struct TypeRefinementEngine {
    /// Refinement strategies
    strategies: Vec<RefinementStrategy>,
    /// Maximum refinement iterations
    max_iterations: usize,
}

/// Strategy for type refinement
#[derive(Debug)]
struct RefinementStrategy {
    /// Strategy name
    name: String,
    /// Applicability condition
    condition: RefinementCondition,
    /// Refinement action
    action: RefinementAction,
}

/// Condition for applying refinement
#[derive(Debug, Clone)]
enum RefinementCondition {
    /// Low confidence inference
    LowConfidence(f64),
    /// Ambiguous type
    AmbiguousType,
    /// Missing semantic information
    MissingSemantics,
    /// Inconsistent constraints
    InconsistentConstraints,
}

/// Action to take for refinement
#[derive(Debug, Clone)]
enum RefinementAction {
    /// Add semantic properties
    AddProperties(Vec<SemanticProperty>),
    /// Narrow type based on usage
    NarrowType(String),
    /// Add business rules
    AddBusinessRules(Vec<BusinessRule>),
    /// Request user annotation
    RequestAnnotation(String),
}

impl SemanticTypeInference {
    /// Create a new semantic type inference engine
    pub fn new() -> Self {
        Self {
            business_validator: BusinessRuleValidator::new(),
            domain_knowledge: DomainKnowledgeBase::new(),
            constraint_generator: SemanticConstraintGenerator::new(),
            refinement_engine: TypeRefinementEngine::new(),
        }
    }

    /// Enhance basic type inference with semantic awareness
    pub fn enhance_inference(
        &mut self,
        basic_result: &mut TypeInferenceResult,
        expressions: &[Expr],
    ) -> SemanticResult<()> {
        // Generate semantic constraints
        let semantic_constraints = self.generate_semantic_constraints(expressions, basic_result)?;
        
        // Validate business rules
        self.validate_business_rules(&semantic_constraints, basic_result)?;
        
        // Apply domain knowledge
        self.apply_domain_knowledge(basic_result)?;
        
        // Refine types based on semantic information
        self.refine_types(basic_result, &semantic_constraints)?;
        
        // Update AI metadata with semantic information
        self.update_ai_metadata(basic_result, &semantic_constraints)?;
        
        Ok(())
    }

    /// Infer semantic type for a literal value
    pub fn infer_literal_semantic_type(
        &self,
        literal: &LiteralValue,
        span: Span,
    ) -> SemanticResult<InferredType> {
        let base_type = match literal {
            LiteralValue::Integer(value) => {
                let semantic_type = self.infer_integer_semantics(*value, span)?;
                semantic_type
            }
            LiteralValue::Float(value) => {
                let semantic_type = self.infer_float_semantics(*value, span)?;
                semantic_type
            }
            LiteralValue::String(value) => {
                let semantic_type = self.infer_string_semantics(value, span)?;
                semantic_type
            }
            LiteralValue::Boolean(value) => {
                let semantic_type = self.infer_boolean_semantics(*value, span)?;
                semantic_type
            }
            LiteralValue::Null => {
                SemanticType::Primitive(prism_ast::PrimitiveType::Unit)
            }
            LiteralValue::Money { amount, currency } => {
                // Create a money semantic type
                SemanticType::Complex {
                    name: "Money".to_string(),
                    base_type: crate::types::BaseType::Primitive(crate::types::PrimitiveType::Money {
                        currency: currency.clone(),
                        precision: 2,
                    }),
                    constraints: Vec::new(),
                    business_rules: Vec::new(),
                    metadata: crate::types::SemanticTypeMetadata::default(),
                    ai_context: None,
                    verification_properties: Vec::new(),
                    location: span,
                }
            }
            LiteralValue::Duration { value, unit } => {
                // Create a duration semantic type
                SemanticType::Complex {
                    name: "Duration".to_string(),
                    base_type: crate::types::BaseType::Primitive(crate::types::PrimitiveType::Custom {
                        name: "Duration".to_string(),
                        base: format!("{}.{}", value, unit),
                    }),
                    constraints: Vec::new(),
                    business_rules: Vec::new(),
                    metadata: crate::types::SemanticTypeMetadata::default(),
                    ai_context: None,
                    verification_properties: Vec::new(),
                    location: span,
                }
            }
            LiteralValue::Regex(pattern) => {
                // Create a regex semantic type
                SemanticType::Complex {
                    name: "Regex".to_string(),
                    base_type: crate::types::BaseType::Primitive(crate::types::PrimitiveType::Custom {
                        name: "Regex".to_string(),
                        base: pattern.clone(),
                    }),
                    constraints: Vec::new(),
                    business_rules: Vec::new(),
                    metadata: crate::types::SemanticTypeMetadata::default(),
                    ai_context: None,
                    verification_properties: Vec::new(),
                    location: span,
                }
            }
        };

        // Generate semantic properties
        let properties = self.extract_literal_properties(literal)?;
        
        // Apply domain patterns
        let domain_context = self.domain_knowledge.classify_literal(literal);

        // Generate semantic constraints based on literal type and domain
        let _semantic_constraints = self.generate_literal_constraints(literal, &base_type, &Some(domain_context))?;
        
        // For now, just return empty constraints to get compilation working
        let constraints: Vec<TypeConstraint> = Vec::new();

        Ok(InferredType {
            type_info: base_type,
            confidence: 0.9, // High confidence for literals
            inference_source: InferenceSource::Semantic,
            constraints,
            ai_metadata: None, // Will be populated later
            span,
        })
    }

    /// Infer semantic constraints for binary operations
    pub fn infer_binary_operation_constraints(
        &self,
        op: &BinaryOperator,
        left_type: &SemanticType,
        right_type: &SemanticType,
        span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        let mut constraints = Vec::new();

        match op {
            BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Multiply | BinaryOperator::Divide => {
                // Arithmetic operations require numeric types
                constraints.extend(self.generate_numeric_constraints(left_type, right_type, span)?);
            }
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                // Equality operations require compatible types
                constraints.extend(self.generate_equality_constraints(left_type, right_type, span)?);
            }
            BinaryOperator::Less | BinaryOperator::Greater | BinaryOperator::LessEqual | BinaryOperator::GreaterEqual => {
                // Comparison operations require ordered types
                constraints.extend(self.generate_ordering_constraints(left_type, right_type, span)?);
            }
            BinaryOperator::And | BinaryOperator::Or => {
                // Logical operations require boolean types
                constraints.extend(self.generate_boolean_constraints(left_type, right_type, span)?);
            }
            _ => {
                // For other operators, generate generic constraints
                constraints.extend(self.generate_generic_constraints(left_type, right_type, span)?);
            }
            // String concatenation would likely use Add operator
        }

        Ok(constraints)
    }

    /// Validate that inferred types satisfy business rules
    pub fn validate_business_rules(
        &mut self,
        constraints: &[SemanticConstraint],
        result: &mut TypeInferenceResult,
    ) -> SemanticResult<()> {
        for constraint in constraints {
            for rule in &constraint.business_rules {
                if !self.business_validator.validate_rule(rule, &constraint.base_constraint)? {
                    let error = TypeError::ConstraintViolation {
                        constraint: rule.rule_name.clone(),
                        location: constraint.base_constraint.origin,
                        explanation: rule.description.clone(),
                    };
                    result.add_error(error);
                }
            }
        }
        Ok(())
    }

    // Private helper methods

    fn generate_semantic_constraints(
        &mut self,
        expressions: &[Expr],
        basic_result: &TypeInferenceResult,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        let mut semantic_constraints = Vec::new();

        for expr in expressions {
            let constraints = self.constraint_generator.generate_for_expression(expr, basic_result)?;
            semantic_constraints.extend(constraints);
        }

        Ok(semantic_constraints)
    }

    fn apply_domain_knowledge(&mut self, result: &mut TypeInferenceResult) -> SemanticResult<()> {
        for (node_id, inferred_type) in &mut result.node_types {
            // Apply domain patterns
            if let Some(pattern) = self.domain_knowledge.find_matching_pattern(&inferred_type.type_info) {
                // Add domain-specific properties to AI metadata
                if inferred_type.ai_metadata.is_none() {
                    inferred_type.ai_metadata = Some(super::metadata::InferenceMetadata {
                        semantic_meaning: pattern.description.clone(),
                        domain_context: pattern.name.clone(),
                        confidence_level: super::metadata::ConfidenceLevel::Medium,
                        complexity_score: 1.0,
                        patterns: Vec::new(),
                        suggestions: Vec::new(),
                        performance_notes: Vec::new(),
                        usage_examples: Vec::new(),
                        documentation_links: Vec::new(),
                    });
                }
                
                // Add pattern-specific properties
                if let Some(ref mut metadata) = inferred_type.ai_metadata {
                    for property in &pattern.inferred_properties {
                        let suggestion = format!("Pattern '{}': {}", pattern.name, property.name);
                        if !metadata.suggestions.contains(&suggestion) {
                            metadata.suggestions.push(suggestion);
                        }
                    }
                }
            }

            // Apply type relationships
            let type_relationships = self.domain_knowledge.get_relationships(&inferred_type.type_info);
            if let Some(relationships) = type_relationships {
                // Enhance confidence based on relationships
                let relationship_count = relationships.len();
                if relationship_count > 0 {
                    // Boost confidence slightly if the type has known relationships
                    inferred_type.confidence = (inferred_type.confidence + 0.1).min(1.0);
                    
                    // Add relationship information to AI metadata
                    if let Some(ref mut metadata) = inferred_type.ai_metadata {
                        for relationship in relationships {
                            let note = format!("Related to {} via {}", 
                                relationship.to_type, 
                                match relationship.relationship {
                                    RelationshipKind::Subtype => "subtyping",
                                    RelationshipKind::Convertible => "conversion",
                                    RelationshipKind::Contains => "composition",
                                    RelationshipKind::UsedWith => "usage",
                                    RelationshipKind::Excludes => "exclusion",
                                }
                            );
                            if !metadata.performance_notes.contains(&note) {
                                metadata.performance_notes.push(note);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn refine_types(
        &mut self,
        result: &mut TypeInferenceResult,
        semantic_constraints: &[SemanticConstraint],
    ) -> SemanticResult<()> {
        for _ in 0..self.refinement_engine.max_iterations {
            let mut refined = false;

            for (node_id, inferred_type) in &mut result.node_types {
                for strategy in &self.refinement_engine.strategies {
                    if strategy.applies_to(inferred_type, semantic_constraints) {
                        strategy.apply(inferred_type)?;
                        refined = true;
                    }
                }
            }

            if !refined {
                break; // No more refinements possible
            }
        }

        Ok(())
    }

    fn update_ai_metadata(
        &self,
        result: &mut TypeInferenceResult,
        semantic_constraints: &[SemanticConstraint],
    ) -> SemanticResult<()> {
        // Update AI metadata with semantic information
        for (node_id, inferred_type) in &mut result.node_types {
            if let Some(ref mut ai_metadata) = inferred_type.ai_metadata {
                // Add semantic insights
                ai_metadata.semantic_meaning = self.extract_semantic_meaning(&inferred_type.type_info);
                ai_metadata.domain_context = self.extract_domain_context(&inferred_type.type_info);
            }
        }

        Ok(())
    }

    fn infer_integer_semantics(&self, value: i64, span: Span) -> SemanticResult<SemanticType> {
        // Apply domain knowledge to classify integer values
        if value >= 0 && value <= 255 {
            // Could be a byte value
            if let Some(domain_type) = self.domain_knowledge.get_type("Byte") {
                return Ok(domain_type.base_type.clone());
            }
        }
        
        if value >= 1900 && value <= 2100 {
            // Could be a year
            if let Some(domain_type) = self.domain_knowledge.get_type("Year") {
                return Ok(domain_type.base_type.clone());
            }
        }

        // Default to primitive int
        Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32))))
    }

    fn infer_float_semantics(&self, value: f64, span: Span) -> SemanticResult<SemanticType> {
        // Apply domain knowledge for float values
        if value >= 0.0 && value <= 1.0 {
            // Could be a percentage or probability
            if let Some(domain_type) = self.domain_knowledge.get_type("Percentage") {
                return Ok(domain_type.base_type.clone());
            }
        }

        // Default to primitive float
        Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64)))
    }

    fn infer_string_semantics(&self, value: &str, span: Span) -> SemanticResult<SemanticType> {
        // Apply pattern matching for string values
        if self.looks_like_email(value) {
            if let Some(domain_type) = self.domain_knowledge.get_type("Email") {
                return Ok(domain_type.base_type.clone());
            }
        }
        
        if self.looks_like_url(value) {
            if let Some(domain_type) = self.domain_knowledge.get_type("URL") {
                return Ok(domain_type.base_type.clone());
            }
        }
        
        if self.looks_like_phone_number(value) {
            if let Some(domain_type) = self.domain_knowledge.get_type("PhoneNumber") {
                return Ok(domain_type.base_type.clone());
            }
        }

        // Default to primitive string
        Ok(SemanticType::Primitive(prism_ast::PrimitiveType::String))
    }

    fn infer_boolean_semantics(&self, value: bool, span: Span) -> SemanticResult<SemanticType> {
        // Boolean values are straightforward
        Ok(SemanticType::Primitive(prism_ast::PrimitiveType::Boolean))
    }

    fn extract_literal_properties(&self, literal: &LiteralValue) -> SemanticResult<Vec<SemanticProperty>> {
        let mut properties = Vec::new();

        match literal {
            LiteralValue::String(value) => {
                properties.push(SemanticProperty {
                    name: "length".to_string(),
                    value: PropertyValue::Numeric(value.len() as f64),
                    source: PropertySource::Inferred,
                });

                if !value.is_empty() {
                    properties.push(SemanticProperty {
                        name: "non_empty".to_string(),
                        value: PropertyValue::Boolean(true),
                        source: PropertySource::Inferred,
                    });
                }
            }
            LiteralValue::Integer(value) => {
                properties.push(SemanticProperty {
                    name: "sign".to_string(),
                    value: PropertyValue::String(if *value >= 0 { "positive".to_string() } else { "negative".to_string() }),
                    source: PropertySource::Inferred,
                });
            }
            _ => {}
        }

        Ok(properties)
    }

    fn generate_numeric_constraints(
        &self,
        left_type: &SemanticType,
        right_type: &SemanticType,
        span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        let mut constraints = Vec::new();

        // Both operands must be numeric
        let numeric_rule = BusinessRule {
            rule_name: "Arithmetic Operands Must Be Numeric".to_string(),
            rule_type: "arithmetic".to_string(),
            confidence: 0.95,
            description: "Arithmetic operations require numeric operands".to_string(),
            evidence: vec!["Type safety requirement for arithmetic operations".to_string()],
            location: Some(span),
        };

        let constraint = SemanticConstraint {
            base_constraint: TypeConstraint::new(
                ConstraintType::Concrete(left_type.clone()),
                ConstraintType::Concrete(right_type.clone()),
                span,
                ConstraintReason::LiteralType,
            ),
            business_rules: vec![numeric_rule],
            semantic_properties: vec![
                SemanticProperty {
                    name: "numeric".to_string(),
                    value: PropertyValue::Boolean(true),
                    source: PropertySource::BusinessRule,
                },
            ],
            domain_context: "arithmetic".to_string(),
            confidence: 0.9,
        };

        constraints.push(constraint);
        Ok(constraints)
    }

    fn generate_equality_constraints(
        &self,
        left_type: &SemanticType,
        right_type: &SemanticType,
        span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // Equality constraints are more permissive
        Ok(Vec::new())
    }

    fn generate_ordering_constraints(
        &self,
        left_type: &SemanticType,
        right_type: &SemanticType,
        span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // Ordering requires comparable types
        Ok(Vec::new())
    }

    fn generate_boolean_constraints(
        &self,
        left_type: &SemanticType,
        right_type: &SemanticType,
        span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // Boolean operations require boolean operands
        Ok(Vec::new())
    }

    fn generate_string_constraints(
        &self,
        left_type: &SemanticType,
        right_type: &SemanticType,
        span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // String concatenation constraints
        Ok(Vec::new())
    }

    fn generate_generic_constraints(
        &self,
        _left_type: &SemanticType,
        _right_type: &SemanticType,
        _span: Span,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // Generic constraints for operators not specifically handled
        Ok(Vec::new())
    }

    fn extract_semantic_meaning(&self, semantic_type: &SemanticType) -> String {
        match semantic_type {
            SemanticType::Primitive(prim) => format!("Primitive {} value", self.format_primitive_type(prim)),
            SemanticType::Function { params, .. } => format!("Function with {} parameters", params.len()),
            SemanticType::List(_) => "Collection of values".to_string(),
            SemanticType::Record(_) => "Structured data record".to_string(),
            _ => "Type with semantic meaning".to_string(),
        }
    }

    fn extract_domain_context(&self, semantic_type: &SemanticType) -> String {
        // Extract domain context based on type characteristics
        "general".to_string()
    }

    fn format_primitive_type(&self, prim: &prism_ast::PrimitiveType) -> &'static str {
        match prim {
            prism_ast::PrimitiveType::Integer(_) => "integer",
            prism_ast::PrimitiveType::Float(_) => "float",
            prism_ast::PrimitiveType::String => "string",
            prism_ast::PrimitiveType::Boolean => "boolean",
            prism_ast::PrimitiveType::Char => "char",
            prism_ast::PrimitiveType::Unit => "unit",
            prism_ast::PrimitiveType::Never => "never",
            prism_ast::PrimitiveType::Int32 => "int32",
            prism_ast::PrimitiveType::Int64 => "int64",
            prism_ast::PrimitiveType::Float64 => "float64",
        }
    }

    // Pattern matching helpers
    fn looks_like_email(&self, value: &str) -> bool {
        value.contains('@') && value.contains('.')
    }

    fn looks_like_url(&self, value: &str) -> bool {
        value.starts_with("http://") || value.starts_with("https://")
    }

    fn looks_like_phone_number(&self, value: &str) -> bool {
        value.chars().filter(|c| c.is_ascii_digit()).count() >= 7
    }
    
    /// Generate semantic constraints for literals
    fn generate_literal_constraints(
        &self,
        literal: &LiteralValue,
        base_type: &SemanticType,
        domain_context: &Option<String>,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // For now, return empty constraints to avoid compilation issues
        Ok(Vec::new())
    }
    
    // Helper methods for validation
    fn count_decimal_places(&self, value: f64) -> u8 {
        let s = format!("{}", value);
        if let Some(dot_pos) = s.find('.') {
            (s.len() - dot_pos - 1).min(255) as u8
        } else {
            0
        }
    }
    
    fn is_valid_email_format(&self, s: &str) -> bool {
        // Simple email validation - in practice, use a proper email validator
        s.contains('@') && s.contains('.') && s.len() > 5
    }
    
    fn is_valid_url_format(&self, s: &str) -> bool {
        // Simple URL validation - in practice, use a proper URL parser
        s.starts_with("http://") || s.starts_with("https://") || s.starts_with("ftp://")
    }
    
    fn is_valid_phone_format(&self, s: &str) -> bool {
        // Simple phone validation - in practice, use a proper phone validator
        s.chars().any(|c| c.is_ascii_digit()) && s.len() >= 10
    }
}

// Implementation of helper components

impl BusinessRuleValidator {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            cache: HashMap::new(),
        }
    }

    fn validate_rule(&mut self, rule: &BusinessRule, constraint: &TypeConstraint) -> SemanticResult<bool> {
        // Simple validation logic - in practice this would be more sophisticated
        Ok(true)
    }
}

impl DomainKnowledgeBase {
    fn new() -> Self {
        Self {
            domain_types: Self::initialize_domain_types(),
            relationships: Vec::new(),
            patterns: Vec::new(),
        }
    }

    fn initialize_domain_types() -> HashMap<String, DomainTypeInfo> {
        let mut types = HashMap::new();
        
        // Add common domain types
        types.insert("Email".to_string(), DomainTypeInfo {
            name: "Email".to_string(),
            base_type: SemanticType::Primitive(prism_ast::PrimitiveType::String),
            properties: vec![
                SemanticProperty {
                    name: "format".to_string(),
                    value: PropertyValue::String("email".to_string()),
                    source: PropertySource::Domain,
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    name: "email_format".to_string(),
                    expression: "matches_email_pattern(value)".to_string(),
                    error_message: "Invalid email format".to_string(),
                    severity: RuleSeverity::Error,
                },
            ],
            usage_patterns: vec!["user_input".to_string(), "contact_info".to_string()],
        });

        types
    }

    fn get_type(&self, name: &str) -> Option<&DomainTypeInfo> {
        self.domain_types.get(name)
    }

    fn classify_literal(&self, literal: &LiteralValue) -> String {
        match literal {
            LiteralValue::String(value) => {
                if value.contains('@') { "communication".to_string() }
                else if value.starts_with("http") { "web".to_string() }
                else { "general".to_string() }
            }
            LiteralValue::Integer(value) => {
                if *value >= 1900 && *value <= 2100 { "temporal".to_string() }
                else { "general".to_string() }
            }
            _ => "general".to_string(),
        }
    }

    fn find_matching_pattern(&self, semantic_type: &SemanticType) -> Option<&DomainPattern> {
        // Pattern matching logic would go here
        None
    }

    fn get_relationships(&self, semantic_type: &SemanticType) -> Option<&Vec<TypeRelationship>> {
        // Relationship lookup would go here
        None
    }
}

impl SemanticConstraintGenerator {
    fn new() -> Self {
        Self {
            rules: Self::initialize_rules(),
        }
    }

    fn initialize_rules() -> Vec<ConstraintGenerationRule> {
        Vec::new()
    }

    fn generate_for_expression(
        &self,
        expr: &Expr,
        basic_result: &TypeInferenceResult,
    ) -> SemanticResult<Vec<SemanticConstraint>> {
        // Constraint generation logic would go here
        Ok(Vec::new())
    }
}

impl TypeRefinementEngine {
    fn new() -> Self {
        Self {
            strategies: Self::initialize_strategies(),
            max_iterations: 5,
        }
    }

    fn initialize_strategies() -> Vec<RefinementStrategy> {
        Vec::new()
    }
}

impl RefinementStrategy {
    fn applies_to(&self, inferred_type: &InferredType, constraints: &[SemanticConstraint]) -> bool {
        match &self.condition {
            RefinementCondition::LowConfidence(threshold) => {
                inferred_type.confidence < *threshold
            }
            RefinementCondition::AmbiguousType => {
                matches!(inferred_type.type_info, SemanticType::Variable(_))
            }
            _ => false,
        }
    }

    fn apply(&self, inferred_type: &mut InferredType) -> SemanticResult<()> {
        match &self.action {
            RefinementAction::AddProperties(properties) => {
                // Add semantic properties to the AI metadata
                if inferred_type.ai_metadata.is_none() {
                    inferred_type.ai_metadata = Some(super::metadata::InferenceMetadata {
                        semantic_meaning: "Refined type".to_string(),
                        domain_context: "refinement".to_string(),
                        confidence_level: super::metadata::ConfidenceLevel::Medium,
                        complexity_score: 1.0,
                        patterns: Vec::new(),
                        suggestions: Vec::new(),
                        performance_notes: Vec::new(),
                        usage_examples: Vec::new(),
                        documentation_links: Vec::new(),
                    });
                }
                
                if let Some(ref mut metadata) = inferred_type.ai_metadata {
                    for property in properties {
                        let suggestion = format!("Property: {} = {:?}", property.name, property.value);
                        if !metadata.suggestions.contains(&suggestion) {
                            metadata.suggestions.push(suggestion);
                        }
                    }
                }
                
                // Increase confidence since we've added more information
                inferred_type.confidence = (inferred_type.confidence + 0.05).min(1.0);
            }
            RefinementAction::NarrowType(type_name) => {
                // Try to narrow the type based on usage context
                match &inferred_type.type_info {
                    SemanticType::Variable(_) => {
                        // For type variables, we could try to resolve to a more specific type
                        // This is a simplified implementation
                        if let Some(ref mut metadata) = inferred_type.ai_metadata {
                            metadata.suggestions.push(format!("Consider narrowing to type: {}", type_name));
                        }
                    }
                    SemanticType::Union(types) => {
                        // For union types, try to eliminate some alternatives
                        if types.len() > 1 {
                            if let Some(ref mut metadata) = inferred_type.ai_metadata {
                                metadata.suggestions.push(format!("Union type could be narrowed to: {}", type_name));
                            }
                        }
                    }
                    _ => {
                        // For other types, add a suggestion
                        if let Some(ref mut metadata) = inferred_type.ai_metadata {
                            metadata.suggestions.push(format!("Type refinement suggestion: {}", type_name));
                        }
                    }
                }
                
                inferred_type.confidence = (inferred_type.confidence + 0.03).min(1.0);
            }
            RefinementAction::AddBusinessRules(rules) => {
                // Add business rules to the constraints
                for rule in rules {
                    // Convert business rule to a type constraint (simplified)
                    let constraint = super::constraints::TypeConstraint {
                        lhs: super::constraints::ConstraintType::Concrete(inferred_type.type_info.clone()),
                        rhs: super::constraints::ConstraintType::Concrete(inferred_type.type_info.clone()),
                        origin: inferred_type.span,
                        reason: super::constraints::ConstraintReason::TypeAnnotation,
                        priority: 50,
                    };
                    
                    if !inferred_type.constraints.iter().any(|c| c.priority == constraint.priority) {
                        inferred_type.constraints.push(constraint);
                    }
                }
                
                if let Some(ref mut metadata) = inferred_type.ai_metadata {
                    metadata.suggestions.push(format!("Added {} business rules", rules.len()));
                }
                
                inferred_type.confidence = (inferred_type.confidence + 0.02).min(1.0);
            }
            RefinementAction::RequestAnnotation(message) => {
                // Add a suggestion for user annotation
                if let Some(ref mut metadata) = inferred_type.ai_metadata {
                    metadata.suggestions.push(format!("Consider adding type annotation: {}", message));
                }
                
                // Lower confidence since we need user input
                inferred_type.confidence = (inferred_type.confidence - 0.1).max(0.0);
            }
        }
        Ok(())
    }
}

impl Default for SemanticTypeInference {
    fn default() -> Self {
        Self::new()
    }
} 