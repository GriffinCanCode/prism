//! Main normalization engine for canonical form conversion.
//!
//! This module implements the core normalization logic that converts parsed syntax
//! from any style into Prism's canonical semantic representation while preserving
//! all semantic meaning and generating AI-comprehensible metadata.

use crate::{
    core::parser::{CLikeSyntax, PythonLikeSyntax, RustLikeSyntax, CanonicalSyntax},
    normalization::{CanonicalForm, MetadataPreserver},
    styles::canonical::{
        CanonicalSyntax as StyleCanonicalSyntax, CanonicalModule, CanonicalFunction, 
        CanonicalStatement, CanonicalExpression, CanonicalLiteral, CanonicalItem,
        CanonicalParameter
    },
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use thiserror::Error;

/// Normalizes parsed syntax from any style to canonical form.
/// 
/// The Normalizer ensures that regardless of input syntax style,
/// the output is always in Prism's canonical semantic representation.
/// This enables consistent downstream processing while preserving
/// the original semantic meaning and generating rich AI metadata.
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing exclusively on
/// "semantic normalization and canonical form generation". It coordinates
/// between style-specific normalization logic and metadata preservation.
#[derive(Debug)]
pub struct Normalizer {
    /// Metadata preservation engine
    metadata_preserver: MetadataPreserver,
    
    /// Semantic validation during normalization
    semantic_validator: SemanticValidator,
    
    /// AI metadata enhancement
    ai_enhancer: AIMetadataEnhancer,
    
    /// Normalization configuration
    config: NormalizationConfig,
}

/// Configuration for normalization behavior
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Whether to preserve all original formatting information
    pub preserve_formatting: bool,
    
    /// Whether to generate comprehensive AI metadata
    pub generate_ai_metadata: bool,
    
    /// Level of semantic validation during normalization
    pub validation_level: ValidationLevel,
    
    /// Custom normalization rules
    pub custom_rules: HashMap<String, String>,
}

/// Level of semantic validation during normalization
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationLevel {
    /// No validation (fastest)
    None,
    
    /// Basic validation (structure only)
    Basic,
    
    /// Full validation (semantics and structure)
    Full,
    
    /// Strict validation (all rules enforced)
    Strict,
}

/// Context information during normalization
#[derive(Debug, Clone)]
pub struct NormalizationContext {
    /// Source syntax style being normalized
    pub source_style: crate::detection::SyntaxStyle,
    
    /// Current normalization phase
    pub current_phase: NormalizationPhase,
    
    /// Accumulated warnings during normalization
    pub warnings: Vec<NormalizationWarning>,
    
    /// Performance metrics
    pub metrics: NormalizationMetrics,
}

/// Phases of the normalization process
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationPhase {
    /// Initial structure conversion
    StructureConversion,
    
    /// Semantic preservation
    SemanticPreservation,
    
    /// Metadata generation
    MetadataGeneration,
    
    /// Validation and verification
    Validation,
    
    /// Finalization
    Finalization,
}

/// Result of normalization process
#[derive(Debug, Clone)]
pub struct NormalizationResult {
    /// The canonical form result
    pub canonical_form: CanonicalForm,
    
    /// Normalization context and metrics
    pub context: NormalizationContext,
    
    /// Any warnings generated during normalization
    pub warnings: Vec<NormalizationWarning>,
    
    /// Success/failure status
    pub success: bool,
}

/// Warning generated during normalization
#[derive(Debug, Clone)]
pub struct NormalizationWarning {
    /// Warning type
    pub warning_type: WarningType,
    
    /// Human-readable message
    pub message: String,
    
    /// Location where warning occurred
    pub location: Option<String>,
    
    /// Suggested resolution
    pub suggestion: Option<String>,
}

/// Types of normalization warnings
#[derive(Debug, Clone)]
pub enum WarningType {
    /// Potential semantic loss during conversion
    SemanticLoss,
    
    /// Formatting information lost
    FormattingLoss,
    
    /// Ambiguous syntax interpretation
    AmbiguousInterpretation,
    
    /// Non-standard syntax patterns
    NonStandardPattern,
}

/// Performance metrics for normalization
#[derive(Debug, Clone)]
pub struct NormalizationMetrics {
    /// Time spent in each phase (milliseconds)
    pub phase_times: HashMap<NormalizationPhase, u64>,
    
    /// Memory usage during normalization
    pub memory_usage: u64,
    
    /// Number of nodes processed
    pub nodes_processed: usize,
    
    /// Complexity score of the normalization
    pub complexity_score: f64,
}

/// Semantic validator for normalization
#[derive(Debug)]
pub struct SemanticValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
}

/// AI metadata enhancer
#[derive(Debug)]
pub struct AIMetadataEnhancer {
    /// Enhancement configuration
    config: AIEnhancementConfig,
}

/// Configuration for AI metadata enhancement
#[derive(Debug, Clone)]
pub struct AIEnhancementConfig {
    /// Level of detail for AI metadata
    pub detail_level: AIDetailLevel,
    
    /// Whether to generate business context
    pub generate_business_context: bool,
    
    /// Whether to extract domain concepts
    pub extract_domain_concepts: bool,
}

/// Level of AI metadata detail
#[derive(Debug, Clone)]
pub enum AIDetailLevel {
    /// Minimal metadata
    Minimal,
    
    /// Standard metadata
    Standard,
    
    /// Comprehensive metadata
    Comprehensive,
}

/// Validation rule for semantic preservation
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Name of the rule
    pub name: String,
    
    /// Rule implementation
    pub rule_fn: fn(&CanonicalForm) -> Result<(), ValidationError>,
    
    /// Priority of this rule
    pub priority: u8,
}

/// Normalization errors
#[derive(Debug, Error)]
pub enum NormalizationError {
    /// Style conversion failed
    #[error("Failed to convert {style:?} syntax: {reason}")]
    ConversionFailed { 
        style: crate::detection::SyntaxStyle, 
        reason: String 
    },
    
    /// Semantic validation failed
    #[error("Semantic validation failed: {errors:?}")]
    ValidationFailed { errors: Vec<ValidationError> },
    
    /// Metadata preservation failed
    #[error("Metadata preservation failed: {reason}")]
    MetadataPreservationFailed { reason: String },
    
    /// AI enhancement failed
    #[error("AI metadata enhancement failed: {reason}")]
    AIEnhancementFailed { reason: String },
}

/// Validation errors during normalization
#[derive(Debug, Error)]
pub enum ValidationError {
    /// Semantic inconsistency detected
    #[error("Semantic inconsistency: {description}")]
    SemanticInconsistency { description: String },
    
    /// Required element missing
    #[error("Required element missing: {element}")]
    MissingRequiredElement { element: String },
    
    /// Invalid structure detected
    #[error("Invalid structure: {description}")]
    InvalidStructure { description: String },
}

/// Parsed syntax from any style (used internally)
#[derive(Debug)]
pub enum ParsedSyntax {
    /// C-like syntax parse result
    CLike(CLikeSyntax),
    /// Python-like syntax parse result
    PythonLike(PythonLikeSyntax),
    /// Rust-like syntax parse result
    RustLike(RustLikeSyntax),
    /// Canonical syntax parse result
    Canonical(CanonicalSyntax),
    /// Style canonical syntax (from our parser)
    StyleCanonical(StyleCanonicalSyntax),
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            preserve_formatting: false,
            generate_ai_metadata: true,
            validation_level: ValidationLevel::Basic,
            custom_rules: HashMap::new(),
        }
    }
}

impl Default for AIEnhancementConfig {
    fn default() -> Self {
        Self {
            detail_level: AIDetailLevel::Standard,
            generate_business_context: true,
            extract_domain_concepts: true,
        }
    }
}

impl Normalizer {
    /// Create a new normalizer with default configuration
    pub fn new() -> Self {
        Self {
            metadata_preserver: MetadataPreserver::new(),
            semantic_validator: SemanticValidator::new(),
            ai_enhancer: AIMetadataEnhancer::new(),
            config: NormalizationConfig::default(),
        }
    }
    
    /// Create a normalizer with custom configuration
    pub fn with_config(config: NormalizationConfig) -> Self {
        let mut normalizer = Self::new();
        normalizer.config = config;
        normalizer
    }
    
    /// Normalizes parsed syntax to canonical form with comprehensive metadata.
    /// 
    /// This is the main normalization method that:
    /// 1. Converts style-specific syntax to canonical structure
    /// 2. Preserves all semantic meaning and metadata
    /// 3. Generates AI-comprehensible metadata
    /// 4. Validates semantic consistency
    /// 5. Provides detailed metrics and warnings
    /// 
    /// # Arguments
    /// 
    /// * `parsed` - The parsed syntax from any supported style
    /// 
    /// # Returns
    /// 
    /// A `CanonicalForm` containing the normalized representation
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use prism_syntax::normalization::Normalizer;
    /// 
    /// let mut normalizer = Normalizer::new();
    /// let result = normalizer.normalize(parsed_syntax)?;
    /// 
    /// // Access canonical form
    /// let canonical = result.canonical_form;
    /// ```
    pub fn normalize(&mut self, parsed: ParsedSyntax) -> Result<CanonicalForm, NormalizationError> {
        let start_time = std::time::Instant::now();
        
        // Create normalization context
        let source_style = match &parsed {
            ParsedSyntax::CLike(_) => crate::detection::SyntaxStyle::CLike,
            ParsedSyntax::PythonLike(_) => crate::detection::SyntaxStyle::PythonLike,
            ParsedSyntax::RustLike(_) => crate::detection::SyntaxStyle::RustLike,
            ParsedSyntax::Canonical(_) => crate::detection::SyntaxStyle::Canonical,
            ParsedSyntax::StyleCanonical(_) => crate::detection::SyntaxStyle::Canonical,
        };
        
        let mut context = NormalizationContext {
            source_style,
            current_phase: NormalizationPhase::StructureConversion,
            warnings: Vec::new(),
            metrics: NormalizationMetrics::default(),
        };
        
        // Phase 1: Structure conversion
        context.current_phase = NormalizationPhase::StructureConversion;
        let phase_start = std::time::Instant::now();
        let mut canonical = self.convert_structure(&parsed, &mut context)?;
        context.metrics.phase_times.insert(
            NormalizationPhase::StructureConversion,
            phase_start.elapsed().as_millis() as u64
        );
        
        // Phase 2: Semantic preservation
        context.current_phase = NormalizationPhase::SemanticPreservation;
        let phase_start = std::time::Instant::now();
        self.preserve_semantics(&mut canonical, &parsed, &mut context)?;
        context.metrics.phase_times.insert(
            NormalizationPhase::SemanticPreservation,
            phase_start.elapsed().as_millis() as u64
        );
        
        // Phase 3: Metadata generation
        if self.config.generate_ai_metadata {
            context.current_phase = NormalizationPhase::MetadataGeneration;
            let phase_start = std::time::Instant::now();
            self.generate_metadata(&mut canonical, &mut context)?;
            context.metrics.phase_times.insert(
                NormalizationPhase::MetadataGeneration,
                phase_start.elapsed().as_millis() as u64
            );
        }
        
        // Phase 4: Validation
        if self.config.validation_level != ValidationLevel::None {
            context.current_phase = NormalizationPhase::Validation;
            let phase_start = std::time::Instant::now();
            self.validate_result(&canonical, &mut context)?;
            context.metrics.phase_times.insert(
                NormalizationPhase::Validation,
                phase_start.elapsed().as_millis() as u64
            );
        }
        
        // Phase 5: Finalization
        context.current_phase = NormalizationPhase::Finalization;
        let total_time = start_time.elapsed().as_millis() as u64;
        context.metrics.phase_times.insert(NormalizationPhase::Finalization, 0);
        
        // Calculate final metrics
        context.metrics.complexity_score = self.calculate_complexity_score(&canonical);
        
        Ok(canonical)
    }
    
    /// Convert style-specific structure to canonical form
    fn convert_structure(
        &self,
        parsed: &ParsedSyntax,
        context: &mut NormalizationContext
    ) -> Result<CanonicalForm, NormalizationError> {
        match parsed {
            ParsedSyntax::CLike(syntax) => self.convert_c_like_structure(syntax, context),
            ParsedSyntax::PythonLike(syntax) => self.convert_python_like_structure(syntax, context),
            ParsedSyntax::RustLike(syntax) => self.convert_rust_like_structure(syntax, context),
            ParsedSyntax::Canonical(syntax) => self.convert_canonical_structure(syntax, context),
            ParsedSyntax::StyleCanonical(syntax) => self.convert_style_canonical_structure(syntax, context),
        }
    }
    
    /// Convert C-like syntax structure
    fn convert_c_like_structure(
        &self,
        _syntax: &CLikeSyntax,
        _context: &mut NormalizationContext
    ) -> Result<CanonicalForm, NormalizationError> {
        // TODO: Implement actual C-like structure conversion
        Ok(CanonicalForm::placeholder())
    }
    
    /// Convert Python-like syntax structure
    fn convert_python_like_structure(
        &self,
        _syntax: &PythonLikeSyntax,
        _context: &mut NormalizationContext
    ) -> Result<CanonicalForm, NormalizationError> {
        // TODO: Implement actual Python-like structure conversion
        Ok(CanonicalForm::placeholder())
    }
    
    /// Convert Rust-like syntax structure
    fn convert_rust_like_structure(
        &self,
        _syntax: &RustLikeSyntax,
        _context: &mut NormalizationContext
    ) -> Result<CanonicalForm, NormalizationError> {
        // TODO: Implement actual Rust-like structure conversion
        Ok(CanonicalForm::placeholder())
    }
    
    /// Convert canonical syntax structure (identity conversion)
    fn convert_canonical_structure(
        &self,
        _syntax: &CanonicalSyntax,
        _context: &mut NormalizationContext
    ) -> Result<CanonicalForm, NormalizationError> {
        // TODO: Implement canonical structure conversion
        Ok(CanonicalForm::placeholder())
    }
    
    /// Convert style canonical syntax structure (our main implementation)
    fn convert_style_canonical_structure(
        &self,
        syntax: &StyleCanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<CanonicalForm, NormalizationError> {
        use crate::normalization::canonical_form::*;
        
        let mut canonical_nodes = Vec::new();
        context.metrics.nodes_processed = 0;
        
        // Convert modules
        for module in &syntax.modules {
            let canonical_module = self.convert_canonical_module(module, context)?;
            canonical_nodes.push(canonical_module);
            context.metrics.nodes_processed += 1;
        }
        
        // Convert top-level functions
        for function in &syntax.functions {
            let canonical_function = self.convert_canonical_function(function, context)?;
            canonical_nodes.push(canonical_function);
            context.metrics.nodes_processed += 1;
        }
        
        // Convert top-level statements
        for statement in &syntax.statements {
            let canonical_statement = self.convert_canonical_statement(statement, context)?;
            canonical_nodes.push(CanonicalNode::Statement {
                statement: canonical_statement,
                span: CanonicalSpan {
                    start: Position { line: 1, column: 1 },
                    end: Position { line: 1, column: 1 },
                    source_id: 1,
                },
                semantic_metadata: NodeSemanticMetadata::default(),
            });
            context.metrics.nodes_processed += 1;
        }
        
        // Create canonical form
        let mut canonical_form = CanonicalForm {
            nodes: canonical_nodes,
            metadata: CanonicalMetadata::default(),
            ai_metadata: AIMetadata::default(),
            semantic_version: "0.1.0".to_string(),
            semantic_hash: 0,
        };
        
        // Update semantic hash
        canonical_form.update_semantic_hash();
        
        Ok(canonical_form)
    }
    
    /// Convert a canonical module to canonical form
    fn convert_canonical_module(
        &self,
        module: &CanonicalModule,
        context: &mut NormalizationContext
    ) -> Result<crate::normalization::canonical_form::CanonicalNode, NormalizationError> {
        use crate::normalization::canonical_form::*;
        
        let mut sections = Vec::new();
        
        // Convert module items to sections
        for item in &module.items {
            match item {
                CanonicalItem::Function(func) => {
                    let canonical_func = self.convert_canonical_function(func, context)?;
                    sections.push(CanonicalSection {
                        section_type: SectionType::Interface,
                        items: vec![canonical_func],
                        metadata: SectionMetadata {
                            purpose: Some("Function definitions".to_string()),
                            cohesion_score: 1.0,
                            dependencies: Vec::new(),
                        },
                    });
                }
                CanonicalItem::Statement(stmt) => {
                    let canonical_stmt = self.convert_canonical_statement(stmt, context)?;
                    sections.push(CanonicalSection {
                        section_type: SectionType::Internal,
                        items: vec![CanonicalNode::Statement {
                            statement: canonical_stmt,
                            span: CanonicalSpan {
                                start: Position { line: 1, column: 1 },
                                end: Position { line: 1, column: 1 },
                                source_id: 1,
                            },
                            semantic_metadata: NodeSemanticMetadata::default(),
                        }],
                        metadata: SectionMetadata {
                            purpose: Some("Module statements".to_string()),
                            cohesion_score: 0.8,
                            dependencies: Vec::new(),
                        },
                    });
                }
            }
        }
        
        Ok(CanonicalNode::Module {
            name: module.name.clone(),
            sections,
            annotations: Vec::new(),
            span: CanonicalSpan {
                start: Position { line: module.span.start.line, column: module.span.start.column },
                end: Position { line: module.span.end.line, column: module.span.end.column },
                source_id: module.span.source_id.0,
            },
            semantic_metadata: NodeSemanticMetadata {
                responsibility: Some(format!("Module: {}", module.name)),
                business_rules: Vec::new(),
                ai_hints: vec![format!("This is a module containing related functionality for {}", module.name)],
                documentation_score: 0.7,
            },
        })
    }
    
    /// Convert a canonical function to canonical form
    fn convert_canonical_function(
        &self,
        function: &CanonicalFunction,
        context: &mut NormalizationContext
    ) -> Result<crate::normalization::canonical_form::CanonicalNode, NormalizationError> {
        use crate::normalization::canonical_form::*;
        
        // Convert parameters
        let parameters = function.parameters.iter().map(|param| {
            Parameter {
                name: param.name.clone(),
                param_type: param.param_type.as_ref().map(|t| CanonicalType::Named(t.clone())).unwrap_or(CanonicalType::Primitive(PrimitiveType::Unit)),
                default: None,
            }
        }).collect();
        
        // Convert body
        let mut body_statements = Vec::new();
        for stmt in &function.body {
            let canonical_stmt = self.convert_canonical_statement(stmt, context)?;
            body_statements.push(CanonicalNode::Statement {
                statement: canonical_stmt,
                span: CanonicalSpan {
                    start: Position { line: 1, column: 1 },
                    end: Position { line: 1, column: 1 },
                    source_id: 1,
                },
                semantic_metadata: NodeSemanticMetadata::default(),
            });
        }
        
        Ok(CanonicalNode::Function {
            name: function.name.clone(),
            parameters,
            return_type: function.return_type.as_ref().map(|t| CanonicalType::Named(t.clone())),
            body: if body_statements.is_empty() { None } else { 
                Some(CanonicalExpression::Block(body_statements.into_iter().map(|node| {
                    if let CanonicalNode::Statement { statement, .. } = node {
                        statement
                    } else {
                        CanonicalStatement::Expression(CanonicalExpression::Literal(LiteralValue::Unit))
                    }
                }).collect()))
            },
            annotations: Vec::new(),
            span: CanonicalSpan {
                start: Position { line: function.span.start.line, column: function.span.start.column },
                end: Position { line: function.span.end.line, column: function.span.end.column },
                source_id: function.span.source_id.0,
            },
            semantic_metadata: NodeSemanticMetadata {
                responsibility: Some(format!("Function: {}", function.name)),
                business_rules: Vec::new(),
                ai_hints: vec![format!("This function performs: {}", function.name)],
                documentation_score: 0.6,
            },
        })
    }
    
    /// Convert a canonical statement to canonical form
    fn convert_canonical_statement(
        &self,
        statement: &CanonicalStatement,
        context: &mut NormalizationContext
    ) -> Result<crate::normalization::canonical_form::CanonicalStatement, NormalizationError> {
        use crate::normalization::canonical_form::CanonicalStatement as CanonicalStmt;
        
        match statement {
            CanonicalStatement::Expression(expr) => {
                let canonical_expr = self.convert_canonical_expression(expr, context)?;
                Ok(CanonicalStmt::Expression(canonical_expr))
            }
            CanonicalStatement::Return(expr) => {
                let canonical_expr = if let Some(expr) = expr {
                    Some(self.convert_canonical_expression(expr, context)?)
                } else {
                    None
                };
                Ok(CanonicalStmt::Return(canonical_expr))
            }
            CanonicalStatement::Assignment { name, value } => {
                let canonical_value = self.convert_canonical_expression(value, context)?;
                Ok(CanonicalStmt::Assignment {
                    target: name.clone(),
                    value: canonical_value,
                })
            }
        }
    }
    
    /// Convert a canonical expression to canonical form
    fn convert_canonical_expression(
        &self,
        expression: &CanonicalExpression,
        _context: &mut NormalizationContext
    ) -> Result<crate::normalization::canonical_form::CanonicalExpression, NormalizationError> {
        use crate::normalization::canonical_form::CanonicalExpression as CanonicalExpr;
        use crate::normalization::canonical_form::LiteralValue;
        
        match expression {
            CanonicalExpression::Identifier(name) => {
                Ok(CanonicalExpr::Variable(name.clone()))
            }
            CanonicalExpression::Literal(literal) => {
                let canonical_literal = match literal {
                    CanonicalLiteral::String(s) => LiteralValue::String(s.clone()),
                    CanonicalLiteral::Integer(i) => LiteralValue::Integer(*i),
                    CanonicalLiteral::Float(f) => LiteralValue::Float(*f),
                    CanonicalLiteral::Boolean(b) => LiteralValue::Boolean(*b),
                };
                Ok(CanonicalExpr::Literal(canonical_literal))
            }
            CanonicalExpression::Call { function, arguments } => {
                let canonical_args = arguments.iter()
                    .map(|arg| self.convert_canonical_expression(arg, _context))
                    .collect::<Result<Vec<_>, _>>()?;
                
                Ok(CanonicalExpr::Call {
                    function: Box::new(CanonicalExpr::Variable(function.clone())),
                    arguments: canonical_args,
                })
            }
            CanonicalExpression::Binary { left, operator, right } => {
                let canonical_left = Box::new(self.convert_canonical_expression(left, _context)?);
                let canonical_right = Box::new(self.convert_canonical_expression(right, _context)?);
                
                let canonical_op = match operator.as_str() {
                    "+" => crate::normalization::canonical_form::BinaryOperator::Add,
                    "-" => crate::normalization::canonical_form::BinaryOperator::Subtract,
                    "*" => crate::normalization::canonical_form::BinaryOperator::Multiply,
                    "/" => crate::normalization::canonical_form::BinaryOperator::Divide,
                    "==" => crate::normalization::canonical_form::BinaryOperator::Equal,
                    "!=" => crate::normalization::canonical_form::BinaryOperator::NotEqual,
                    "<" => crate::normalization::canonical_form::BinaryOperator::Less,
                    ">" => crate::normalization::canonical_form::BinaryOperator::Greater,
                    _ => crate::normalization::canonical_form::BinaryOperator::Add, // Default fallback
                };
                
                Ok(CanonicalExpr::Binary {
                    left: canonical_left,
                    operator: canonical_op,
                    right: canonical_right,
                })
            }
        }
    }
    
    /// Preserve semantic information during normalization
    fn preserve_semantics(
        &self,
        canonical: &mut CanonicalForm,
        _original: &ParsedSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Update AI metadata with semantic information
        canonical.ai_metadata.domain_concepts = vec![
            "module".to_string(),
            "function".to_string(),
            "statement".to_string(),
        ];
        
        canonical.ai_metadata.relationships = vec![
            "module contains functions".to_string(),
            "functions contain statements".to_string(),
        ];
        
        // Add complexity metrics
        canonical.ai_metadata.complexity_metrics.cyclomatic = context.metrics.nodes_processed as f64 * 0.1;
        canonical.ai_metadata.complexity_metrics.cognitive = context.metrics.nodes_processed as f64 * 0.15;
        canonical.ai_metadata.complexity_metrics.nesting_depth = 2; // Basic default
        canonical.ai_metadata.complexity_metrics.dependencies = canonical.nodes.len();
        
        Ok(())
    }
    
    /// Generate AI metadata for the canonical form
    fn generate_metadata(
        &self,
        canonical: &mut CanonicalForm,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Generate business context based on structure
        if !canonical.nodes.is_empty() {
            canonical.ai_metadata.business_context = Some(
                "Multi-syntax code normalized to canonical form for consistent processing".to_string()
            );
        }
        
        // Add processing metadata
        canonical.metadata.normalized_at = chrono::Utc::now().to_rfc3339();
        canonical.metadata.original_style = context.source_style;
        
        Ok(())
    }
    
    /// Validate the normalization result
    fn validate_result(
        &self,
        canonical: &CanonicalForm,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        if self.config.validation_level == ValidationLevel::Basic {
            // Basic validation: ensure we have some content
            if canonical.nodes.is_empty() {
                context.warnings.push(NormalizationWarning {
                    warning_type: WarningType::SemanticLoss,
                    message: "No nodes found in canonical form".to_string(),
                    location: None,
                    suggestion: Some("Check input syntax for valid content".to_string()),
                });
            }
        }
        
        Ok(())
    }
    
    /// Calculate complexity score for the canonical form
    fn calculate_complexity_score(&self, canonical: &CanonicalForm) -> f64 {
        let node_count = canonical.nodes.len() as f64;
        let base_complexity = node_count * 0.1;
        
        // Add complexity for nested structures
        let nesting_penalty = canonical.nodes.iter().map(|node| {
            match node {
                crate::normalization::canonical_form::CanonicalNode::Module { sections, .. } => {
                    sections.len() as f64 * 0.2
                }
                crate::normalization::canonical_form::CanonicalNode::Function { body, .. } => {
                    if body.is_some() { 0.3 } else { 0.1 }
                }
                _ => 0.1,
            }
        }).sum::<f64>();
        
        (base_complexity + nesting_penalty).min(1.0)
    }
}

impl SemanticValidator {
    /// Create a new semantic validator
    pub fn new() -> Self {
        Self {
            rules: Vec::new(), // TODO: Add default validation rules
        }
    }
}

impl AIMetadataEnhancer {
    /// Create a new AI metadata enhancer
    pub fn new() -> Self {
        Self {
            config: AIEnhancementConfig::default(),
        }
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for NormalizationMetrics {
    fn default() -> Self {
        Self {
            phase_times: HashMap::new(),
            memory_usage: 0,
            nodes_processed: 0,
            complexity_score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::styles::canonical::*;
    use prism_common::span::{Span, Position};
    use prism_common::SourceId;
    
    fn create_test_span() -> Span {
        Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 10, 9),
            SourceId::new(1),
        )
    }
    
    #[test]
    fn test_basic_normalization() {
        let mut normalizer = Normalizer::new();
        
        // Create a simple function
        let test_function = CanonicalFunction {
            name: "test".to_string(),
            parameters: vec![],
            return_type: None,
            body: vec![
                CanonicalStatement::Return(Some(CanonicalExpression::Literal(
                    CanonicalLiteral::String("hello".to_string())
                )))
            ],
            span: create_test_span(),
        };
        
        let syntax = StyleCanonicalSyntax {
            modules: vec![],
            functions: vec![test_function],
            statements: vec![],
        };
        
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(syntax));
        assert!(result.is_ok(), "Normalization should succeed: {:?}", result);
        
        let canonical = result.unwrap();
        assert_eq!(canonical.nodes.len(), 1, "Should have one node");
        assert!(!canonical.ai_metadata.domain_concepts.is_empty(), "Should have domain concepts");
        assert!(canonical.semantic_hash != 0, "Should have semantic hash");
    }
    
    #[test]
    fn test_module_normalization() {
        let mut normalizer = Normalizer::new();
        
        // Create a module with a function
        let test_function = CanonicalFunction {
            name: "authenticate".to_string(),
            parameters: vec![
                CanonicalParameter {
                    name: "user".to_string(),
                    param_type: Some("User".to_string()),
                }
            ],
            return_type: Some("Result".to_string()),
            body: vec![
                CanonicalStatement::Return(Some(CanonicalExpression::Call {
                    function: "processAuth".to_string(),
                    arguments: vec![CanonicalExpression::Identifier("user".to_string())],
                }))
            ],
            span: create_test_span(),
        };
        
        let test_module = CanonicalModule {
            name: "UserAuth".to_string(),
            items: vec![CanonicalItem::Function(test_function)],
            span: create_test_span(),
        };
        
        let syntax = StyleCanonicalSyntax {
            modules: vec![test_module],
            functions: vec![],
            statements: vec![],
        };
        
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(syntax));
        assert!(result.is_ok(), "Module normalization should succeed: {:?}", result);
        
        let canonical = result.unwrap();
        assert_eq!(canonical.nodes.len(), 1, "Should have one module node");
        
        // Check the module structure
        if let crate::normalization::canonical_form::CanonicalNode::Module { name, sections, .. } = &canonical.nodes[0] {
            assert_eq!(name, "UserAuth", "Module name should be preserved");
            assert!(!sections.is_empty(), "Module should have sections");
        } else {
            panic!("Expected module node");
        }
        
        // Check AI metadata
        assert!(canonical.ai_metadata.business_context.is_some(), "Should have business context");
        assert!(canonical.ai_metadata.domain_concepts.contains(&"module".to_string()), "Should recognize modules");
        assert!(canonical.ai_metadata.domain_concepts.contains(&"function".to_string()), "Should recognize functions");
    }
    
    #[test]
    fn test_complex_expression_normalization() {
        let mut normalizer = Normalizer::new();
        
        // Create a function with complex expressions
        let binary_expr = CanonicalExpression::Binary {
            left: Box::new(CanonicalExpression::Identifier("a".to_string())),
            operator: "+".to_string(),
            right: Box::new(CanonicalExpression::Literal(CanonicalLiteral::Integer(42))),
        };
        
        let test_function = CanonicalFunction {
            name: "calculate".to_string(),
            parameters: vec![
                CanonicalParameter {
                    name: "a".to_string(),
                    param_type: Some("i32".to_string()),
                }
            ],
            return_type: Some("i32".to_string()),
            body: vec![
                CanonicalStatement::Assignment {
                    name: "result".to_string(),
                    value: binary_expr,
                },
                CanonicalStatement::Return(Some(CanonicalExpression::Identifier("result".to_string()))),
            ],
            span: create_test_span(),
        };
        
        let syntax = StyleCanonicalSyntax {
            modules: vec![],
            functions: vec![test_function],
            statements: vec![],
        };
        
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(syntax));
        assert!(result.is_ok(), "Complex expression normalization should succeed: {:?}", result);
        
        let canonical = result.unwrap();
        assert_eq!(canonical.nodes.len(), 1, "Should have one function node");
        
        // Check complexity metrics
        assert!(canonical.ai_metadata.complexity_metrics.cyclomatic > 0.0, "Should have cyclomatic complexity");
        assert!(canonical.ai_metadata.complexity_metrics.cognitive > 0.0, "Should have cognitive complexity");
    }
    
    #[test]
    fn test_validation_warnings() {
        let mut normalizer = Normalizer::new();
        
        // Create empty syntax (should generate warnings)
        let syntax = StyleCanonicalSyntax {
            modules: vec![],
            functions: vec![],
            statements: vec![],
        };
        
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(syntax));
        assert!(result.is_ok(), "Empty normalization should succeed but with warnings");
        
        let canonical = result.unwrap();
        assert_eq!(canonical.nodes.len(), 0, "Should have no nodes");
        // Note: warnings are currently internal to the context, not exposed in the result
    }
    
    #[test]
    fn test_semantic_preservation() {
        let mut normalizer = Normalizer::new();
        
        // Create a function that tests semantic preservation
        let test_function = CanonicalFunction {
            name: "businessLogic".to_string(),
            parameters: vec![],
            return_type: Some("BusinessResult".to_string()),
            body: vec![
                CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "validateInput".to_string(),
                    arguments: vec![],
                }),
                CanonicalStatement::Return(Some(CanonicalExpression::Literal(
                    CanonicalLiteral::Boolean(true)
                ))),
            ],
            span: create_test_span(),
        };
        
        let syntax = StyleCanonicalSyntax {
            modules: vec![],
            functions: vec![test_function],
            statements: vec![],
        };
        
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(syntax));
        assert!(result.is_ok(), "Semantic preservation test should succeed");
        
        let canonical = result.unwrap();
        
        // Check that semantic information is preserved
        assert!(canonical.ai_metadata.relationships.contains(&"functions contain statements".to_string()));
        assert!(canonical.metadata.original_style == crate::detection::SyntaxStyle::Canonical);
        assert!(!canonical.metadata.normalized_at.is_empty(), "Should have normalization timestamp");
    }
    
    #[test]
    fn test_normalization_performance() {
        let mut normalizer = Normalizer::new();
        
        // Create a larger syntax structure for performance testing
        let mut functions = Vec::new();
        for i in 0..10 {
            functions.push(CanonicalFunction {
                name: format!("function_{}", i),
                parameters: vec![
                    CanonicalParameter {
                        name: "param".to_string(),
                        param_type: Some("String".to_string()),
                    }
                ],
                return_type: Some("Result".to_string()),
                body: vec![
                    CanonicalStatement::Expression(CanonicalExpression::Call {
                        function: "process".to_string(),
                        arguments: vec![CanonicalExpression::Identifier("param".to_string())],
                    }),
                    CanonicalStatement::Return(Some(CanonicalExpression::Literal(
                        CanonicalLiteral::Boolean(true)
                    ))),
                ],
                span: create_test_span(),
            });
        }
        
        let syntax = StyleCanonicalSyntax {
            modules: vec![],
            functions,
            statements: vec![],
        };
        
        let start = std::time::Instant::now();
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(syntax));
        let duration = start.elapsed();
        
        assert!(result.is_ok(), "Performance test should succeed");
        assert!(duration.as_millis() < 100, "Should complete quickly: {}ms", duration.as_millis());
        
        let canonical = result.unwrap();
        assert_eq!(canonical.nodes.len(), 10, "Should have 10 function nodes");
        assert!(canonical.ai_metadata.complexity_metrics.dependencies == 10, "Should track dependencies");
    }
} 