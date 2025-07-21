//! Main normalization coordinator for canonical form conversion.
//!
//! This module implements the core normalization coordinator that delegates to
//! style-specific normalizers while managing the overall normalization process,
//! metadata preservation, and AI-comprehensible metadata generation.

use crate::{
    detection::SyntaxStyle,
    normalization::{
        traits::{StyleNormalizer, AIMetadata, ComplexityMetrics, SemanticRelationship, RelationshipType},
        c_like::CLikeNormalizer,
        python_like::PythonLikeNormalizer,
        rust_like::RustLikeNormalizer,
        canonical::CanonicalNormalizer,
        CanonicalForm, MetadataPreserver
    },
    styles::{
        c_like::CLikeSyntax,
        python_like::PythonLikeSyntax,
        rust_like::RustLikeSyntax,
        canonical::CanonicalSyntax,
    },
};
use prism_ast::{AstNode, Stmt};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use thiserror::Error;
use prism_common::span::Span;

/// Enum representing parsed syntax from different styles
#[derive(Debug)]
pub enum ParsedSyntax {
    /// C-like syntax (C/C++/Java/JavaScript style)
    CLike(CLikeSyntax),
    
    /// Python-like syntax (Python/CoffeeScript style)
    PythonLike(PythonLikeSyntax),
    
    /// Rust-like syntax (Rust/Go style)
    RustLike(RustLikeSyntax),
    
    /// Canonical Prism syntax (AST nodes)
    Canonical(Vec<AstNode<Stmt>>),
}

/// Main normalization coordinator that delegates to style-specific normalizers.
/// 
/// The Normalizer coordinates the normalization process by:
/// 1. Determining the appropriate style-specific normalizer
/// 2. Delegating the normalization work to that normalizer
/// 3. Managing metadata preservation and AI enhancement
/// 4. Providing a unified interface for all syntax styles
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing exclusively on
/// "normalization coordination and orchestration". It delegates actual
/// normalization work to specialized normalizers while managing the
/// overall process and cross-cutting concerns.
#[derive(Debug)]
pub struct Normalizer {
    /// C-like syntax normalizer
    c_like_normalizer: CLikeNormalizer,
    
    /// Python-like syntax normalizer
    python_like_normalizer: PythonLikeNormalizer,
    
    /// Rust-like syntax normalizer
    rust_like_normalizer: RustLikeNormalizer,
    
    /// Canonical syntax normalizer
    canonical_normalizer: CanonicalNormalizer,
    
    /// Metadata preservation engine
    metadata_preserver: MetadataPreserver,
    
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
    pub source_style: SyntaxStyle,
    
    /// Current normalization phase
    pub current_phase: NormalizationPhase,
    
    /// Accumulated warnings during normalization
    pub warnings: Vec<NormalizationWarning>,
    
    /// Performance metrics
    pub metrics: NormalizationMetrics,
    
    /// Current scope depth for tracking nested structures
    pub scope_depth: usize,
    
    /// Symbol table for tracking identifiers
    pub symbols: HashMap<String, SymbolInfo>,
}

impl NormalizationContext {
    /// Create a new normalization context
    pub fn new(source_style: SyntaxStyle) -> Self {
        Self {
            source_style,
            current_phase: NormalizationPhase::StructureConversion,
            warnings: Vec::new(),
            metrics: NormalizationMetrics::default(),
            scope_depth: 0,
            symbols: HashMap::new(),
        }
    }
    
    /// Start a new normalization phase
    pub fn start_phase(&mut self, phase_name: &str) {
        match phase_name {
            "python_normalization" => self.current_phase = NormalizationPhase::StructureConversion,
            "python_validation" => self.current_phase = NormalizationPhase::ValidationAndOptimization,
            "ai_metadata_generation" => self.current_phase = NormalizationPhase::AIMetadataGeneration,
            "semantic_analysis" => self.current_phase = NormalizationPhase::SemanticAnalysis,
            _ => {
                // Default to structure conversion for unknown phases
                self.current_phase = NormalizationPhase::StructureConversion;
            }
        }
    }
    
    /// End the current normalization phase
    pub fn end_phase(&mut self, _phase_name: &str) {
        // Phase ended - could add timing or other cleanup here
    }
    
    /// Add a warning to the context
    pub fn add_warning(&mut self, warning: NormalizationWarning) {
        self.warnings.push(warning);
    }
    
    /// Increment the scope depth
    pub fn enter_scope(&mut self) {
        self.scope_depth += 1;
    }
    
    /// Decrement the scope depth
    pub fn exit_scope(&mut self) {
        if self.scope_depth > 0 {
            self.scope_depth -= 1;
        }
    }
    
    /// Track a symbol in the current scope
    pub fn track_symbol(&mut self, name: String, symbol_info: SymbolInfo) {
        self.symbols.insert(name, symbol_info);
    }
    
    /// Get a symbol from the symbol table
    pub fn get_symbol(&self, name: &str) -> Option<&SymbolInfo> {
        self.symbols.get(name)
    }
    
    /// Get a mutable reference to a symbol
    pub fn get_symbol_mut(&mut self, name: &str) -> Option<&mut SymbolInfo> {
        self.symbols.get_mut(name)
    }
}

/// Phases of the normalization process
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NormalizationPhase {
    /// Initial structure conversion
    StructureConversion,
    
    /// Semantic analysis and enrichment
    SemanticAnalysis,
    
    /// AI metadata generation
    AIMetadataGeneration,
    
    /// Final validation and optimization
    ValidationAndOptimization,
}

/// Warnings generated during normalization
#[derive(Debug, Clone)]
pub struct NormalizationWarning {
    /// Warning message
    pub message: String,
    
    /// Source location if available
    pub span: Option<Span>,
    
    /// Warning severity
    pub severity: WarningSeverity,
    
    /// Suggested fix if available
    pub suggestion: Option<String>,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningSeverity {
    /// Informational warning
    Info,
    
    /// Warning about potential issues
    Warning,
    
    /// Error that should be addressed
    Error,
}

/// Performance metrics for normalization
#[derive(Debug, Clone, Default)]
pub struct NormalizationMetrics {
    /// Time spent on structure conversion
    pub structure_conversion_time: std::time::Duration,
    
    /// Time spent on semantic analysis
    pub semantic_analysis_time: std::time::Duration,
    
    /// Time spent on AI metadata generation
    pub ai_metadata_time: std::time::Duration,
    
    /// Number of nodes processed
    pub nodes_processed: usize,
    
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// Symbol information for tracking during normalization
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    
    /// Symbol type if known
    pub symbol_type: Option<String>,
    
    /// Scope where symbol is defined
    pub scope_depth: usize,
    
    /// Whether symbol is mutable
    pub is_mutable: bool,
    
    /// Usage count
    pub usage_count: usize,
}

/// Errors that can occur during normalization
#[derive(Debug, Error)]
pub enum NormalizationError {
    #[error("Unsupported syntax construct: {construct} at {span:?}")]
    UnsupportedConstruct { construct: String, span: Span },
    
    #[error("Semantic validation failed: {message}")]
    ValidationFailed { message: String },
    
    #[error("AI metadata generation failed: {reason}")]
    AIMetadataFailed { reason: String },
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
}

/// Metadata preservation engine
#[derive(Debug, Default)]
pub struct MetadataPreserver {
    /// Original source mappings
    source_mappings: HashMap<String, Span>,
    
    /// Preserved comments
    comments: Vec<PreservedComment>,
    
    /// Formatting information
    formatting_info: FormattingInfo,
}

/// Preserved comment information
#[derive(Debug, Clone)]
pub struct PreservedComment {
    /// Comment text
    pub text: String,
    
    /// Comment location
    pub span: Span,
    
    /// Comment type (line, block, doc)
    pub comment_type: CommentType,
}

/// Types of comments
#[derive(Debug, Clone, PartialEq)]
pub enum CommentType {
    Line,
    Block,
    Documentation,
}

/// Formatting information to preserve
#[derive(Debug, Clone, Default)]
pub struct FormattingInfo {
    /// Indentation style
    pub indentation: IndentationInfo,
    
    /// Brace placement style
    pub brace_style: BraceStyleInfo,
    
    /// Line endings
    pub line_endings: LineEndingStyle,
}

/// Indentation information
#[derive(Debug, Clone)]
pub struct IndentationInfo {
    /// Use spaces or tabs
    pub style: IndentStyle,
    
    /// Number of spaces per indent level
    pub size: usize,
}

/// Indentation styles
#[derive(Debug, Clone, PartialEq)]
pub enum IndentStyle {
    Spaces,
    Tabs,
    Mixed,
}

/// Brace style information
#[derive(Debug, Clone, PartialEq)]
pub enum BraceStyleInfo {
    SameLine,
    NextLine,
    Mixed,
}

impl Default for BraceStyleInfo {
    fn default() -> Self {
        BraceStyleInfo::SameLine
    }
}

/// Line ending styles
#[derive(Debug, Clone, PartialEq)]
pub enum LineEndingStyle {
    Unix,    // \n
    Windows, // \r\n
    Mac,     // \r
    Mixed,
}

impl Default for LineEndingStyle {
    fn default() -> Self {
        LineEndingStyle::Unix
    }
}

/// Semantic validator for normalization
#[derive(Debug, Default)]
pub struct SemanticValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    
    /// Rule description
    pub description: String,
    
    /// Rule severity
    pub severity: WarningSeverity,
}

/// AI metadata enhancer
#[derive(Debug, Default)]
pub struct AIMetadataEnhancer {
    /// AI context generators
    context_generators: Vec<AIContextGenerator>,
}

/// AI context generator
#[derive(Debug, Clone)]
pub struct AIContextGenerator {
    /// Generator name
    pub name: String,
    
    /// Generator description
    pub description: String,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            preserve_formatting: true,
            generate_ai_metadata: true,
            validation_level: ValidationLevel::Full,
            custom_rules: HashMap::new(),
        }
    }
}

impl Default for IndentationInfo {
    fn default() -> Self {
        Self {
            style: IndentStyle::Spaces,
            size: 4,
        }
    }
}

impl Normalizer {
    /// Create a new normalizer with default configuration
    pub fn new() -> Self {
        Self::with_config(NormalizationConfig::default())
    }
    
    /// Create a new normalizer with custom configuration
    pub fn with_config(config: NormalizationConfig) -> Self {
        Self {
            c_like_normalizer: CLikeNormalizer::new(),
            python_like_normalizer: PythonLikeNormalizer::new(),
            rust_like_normalizer: RustLikeNormalizer::new(),
            canonical_normalizer: CanonicalNormalizer::new(),
            metadata_preserver: MetadataPreserver::default(),
            config,
        }
    }
    
    /// Normalize syntax from any style to canonical form
    pub fn normalize(&mut self, syntax: ParsedSyntax) -> Result<CanonicalForm, NormalizationError> {
        match syntax {
            ParsedSyntax::CLike(c_like_syntax) => self.normalize_c_like(&c_like_syntax),
            ParsedSyntax::PythonLike(python_like_syntax) => self.normalize_python_like(&python_like_syntax),
            ParsedSyntax::RustLike(rust_like_syntax) => self.normalize_rust_like(&rust_like_syntax),
            ParsedSyntax::Canonical(canonical_ast) => self.normalize_canonical(&canonical_ast),
        }
    }
    
    /// Normalize C-like syntax to canonical form
    pub fn normalize_c_like(&mut self, syntax: &CLikeSyntax) -> Result<CanonicalForm, NormalizationError> {
        self.normalize_with_style_normalizer(
            SyntaxStyle::CLike,
            |normalizer, context| normalizer.c_like_normalizer.normalize(syntax, context)
        )
    }
    
    /// Normalize Python-like syntax to canonical form
    pub fn normalize_python_like(&mut self, syntax: &PythonLikeSyntax) -> Result<CanonicalForm, NormalizationError> {
        self.normalize_with_style_normalizer(
            SyntaxStyle::PythonLike,
            |normalizer, context| normalizer.python_like_normalizer.normalize(syntax, context)
        )
    }
    
    /// Normalize Rust-like syntax to canonical form
    pub fn normalize_rust_like(&mut self, syntax: &RustLikeSyntax) -> Result<CanonicalForm, NormalizationError> {
        self.normalize_with_style_normalizer(
            SyntaxStyle::RustLike,
            |normalizer, context| normalizer.rust_like_normalizer.normalize(syntax, context)
        )
    }
    
    /// Normalize canonical AST to canonical form
    pub fn normalize_canonical(&mut self, ast: &[AstNode<Stmt>]) -> Result<CanonicalForm, NormalizationError> {
        self.normalize_with_style_normalizer(
            SyntaxStyle::Canonical,
            |normalizer, context| normalizer.canonical_normalizer.normalize(ast, context)
        )
    }
    
    /// Common normalization workflow using a style-specific normalizer
    fn normalize_with_style_normalizer<F>(&mut self, style: SyntaxStyle, normalize_fn: F) -> Result<CanonicalForm, NormalizationError>
    where
        F: FnOnce(&mut Self, &mut NormalizationContext) -> Result<CanonicalSyntax, NormalizationError>,
    {
        let mut context = NormalizationContext {
            source_style: style,
            current_phase: NormalizationPhase::StructureConversion,
            warnings: Vec::new(),
            metrics: NormalizationMetrics::default(),
            scope_depth: 0,
            symbols: HashMap::new(),
        };
        
        let start_time = std::time::Instant::now();
        
        // Phase 1: Structure conversion (delegated to style-specific normalizer)
        let canonical_syntax = normalize_fn(self, &mut context)?;
        context.metrics.structure_conversion_time = start_time.elapsed();
        
        // Phase 2: Semantic analysis
        context.current_phase = NormalizationPhase::SemanticAnalysis;
        let semantic_start = std::time::Instant::now();
        self.perform_semantic_analysis(&canonical_syntax, &mut context)?;
        context.metrics.semantic_analysis_time = semantic_start.elapsed();
        
        // Phase 3: AI metadata generation
        let ai_metadata = if self.config.generate_ai_metadata {
            context.current_phase = NormalizationPhase::AIMetadataGeneration;
            let ai_start = std::time::Instant::now();
            let metadata = self.generate_ai_metadata(&canonical_syntax, &mut context)?;
            context.metrics.ai_metadata_time = ai_start.elapsed();
            metadata
        } else {
            AIMetadata::default()
        };
        
        // Phase 4: Final validation
        context.current_phase = NormalizationPhase::ValidationAndOptimization;
        self.validate_canonical_form(&canonical_syntax, &mut context)?;
        
        Ok(CanonicalForm {
            syntax: canonical_syntax,
            metadata: self.create_canonical_metadata(&context),
            ai_metadata,
            warnings: context.warnings,
            metrics: context.metrics,
        })
    }
    
    /// Convert C-like structure to canonical form
    fn convert_c_like_structure(&self, syntax: &CLikeSyntax, context: &mut NormalizationContext) -> Result<CanonicalSyntax, NormalizationError> {
        let mut canonical_modules = Vec::new();
        let mut canonical_functions = Vec::new();
        let mut canonical_statements = Vec::new();
        
        // Convert modules
        for module in &syntax.modules {
            canonical_modules.push(self.convert_c_like_module(module, context)?);
        }
        
        // Convert functions
        for function in &syntax.functions {
            canonical_functions.push(self.convert_c_like_function(function, context)?);
        }
        
        // Convert statements
        for statement in &syntax.statements {
            canonical_statements.push(self.convert_c_like_statement(statement, context)?);
        }
        
        Ok(CanonicalSyntax {
            modules: canonical_modules,
            functions: canonical_functions,
            statements: canonical_statements,
        })
    }
    
    /// Convert C-like module to canonical form
    fn convert_c_like_module(&self, module: &CLikeModule, context: &mut NormalizationContext) -> Result<CanonicalModule, NormalizationError> {
        context.scope_depth += 1;
        
        let mut canonical_items = Vec::new();
        
        for item in &module.body {
            canonical_items.push(self.convert_c_like_item(item, context)?);
        }
        
        context.scope_depth -= 1;
        
        Ok(CanonicalModule {
            name: module.name.clone(),
            items: canonical_items,
            span: module.span,
        })
    }
    
    /// Convert C-like item to canonical form
    fn convert_c_like_item(&self, item: &CLikeItem, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        match item {
            CLikeItem::Function(function) => {
                Ok(CanonicalItem::Function(self.convert_c_like_function(function, context)?))
            }
            CLikeItem::Statement(statement) => {
                Ok(CanonicalItem::Statement(self.convert_c_like_statement(statement, context)?))
            }
            CLikeItem::TypeDeclaration { name, type_def, span } => {
                // Convert type declarations to canonical statements
                Ok(CanonicalItem::Statement(CanonicalStatement::Assignment {
                    name: format!("type_{}", name),
                    value: CanonicalExpression::Identifier(type_def.clone()),
                }))
            }
            CLikeItem::VariableDeclaration { name, var_type, initializer, span } => {
                // Track the symbol
                context.symbols.insert(name.clone(), SymbolInfo {
                    name: name.clone(),
                    symbol_type: var_type.clone(),
                    scope_depth: context.scope_depth,
                    is_mutable: true, // C-like variables are mutable by default
                    usage_count: 0,
                });
                
                let canonical_value = if let Some(init) = initializer {
                    self.convert_c_like_expression(init, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("undefined".to_string()))
                };
                
                Ok(CanonicalItem::Statement(CanonicalStatement::Assignment {
                    name: name.clone(),
                    value: canonical_value,
                }))
            }
        }
    }
    
    /// Convert C-like function to canonical form
    fn convert_c_like_function(&self, function: &CLikeFunction, context: &mut NormalizationContext) -> Result<CanonicalFunction, NormalizationError> {
        context.scope_depth += 1;
        
        // Convert parameters
        let mut canonical_parameters = Vec::new();
        for param in &function.parameters {
            canonical_parameters.push(CanonicalParameter {
                name: param.name.clone(),
                param_type: param.param_type.clone(),
            });
            
            // Track parameter symbols
            context.symbols.insert(param.name.clone(), SymbolInfo {
                name: param.name.clone(),
                symbol_type: param.param_type.clone(),
                scope_depth: context.scope_depth,
                is_mutable: false, // Parameters are immutable by default
                usage_count: 0,
            });
        }
        
        // Convert body statements
        let mut canonical_body = Vec::new();
        for statement in &function.body {
            canonical_body.push(self.convert_c_like_statement(statement, context)?);
        }
        
        context.scope_depth -= 1;
        
        Ok(CanonicalFunction {
            name: function.name.clone(),
            parameters: canonical_parameters,
            return_type: function.return_type.clone(),
            body: canonical_body,
            span: function.span,
        })
    }
    
    /// Convert C-like statement to canonical form
    fn convert_c_like_statement(&self, statement: &CLikeStatement, context: &mut NormalizationContext) -> Result<CanonicalStatement, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match statement {
            CLikeStatement::Expression(expr) => {
                Ok(CanonicalStatement::Expression(self.convert_c_like_expression(expr, context)?))
            }
            CLikeStatement::Return(expr_opt) => {
                let canonical_expr = if let Some(expr) = expr_opt {
                    Some(self.convert_c_like_expression(expr, context)?)
                } else {
                    None
                };
                Ok(CanonicalStatement::Return(canonical_expr))
            }
            CLikeStatement::Assignment { name, value } => {
                // Update symbol usage
                if let Some(symbol) = context.symbols.get_mut(name) {
                    symbol.usage_count += 1;
                }
                
                Ok(CanonicalStatement::Assignment {
                    name: name.clone(),
                    value: self.convert_c_like_expression(value, context)?,
                })
            }
            CLikeStatement::If { condition, then_block, else_block } => {
                // For now, convert to a simple conditional expression
                // In a full implementation, this would be more sophisticated
                let condition_expr = self.convert_c_like_expression(condition, context)?;
                
                // Convert blocks to expressions (simplified)
                let then_expr = if then_block.len() == 1 {
                    if let CLikeStatement::Expression(expr) = &then_block[0] {
                        self.convert_c_like_expression(expr, context)?
                    } else {
                        CanonicalExpression::Literal(CanonicalLiteral::String("then_block".to_string()))
                    }
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("then_block".to_string()))
                };
                
                let else_expr = if let Some(else_stmts) = else_block {
                    if else_stmts.len() == 1 {
                        if let CLikeStatement::Expression(expr) = &else_stmts[0] {
                            self.convert_c_like_expression(expr, context)?
                        } else {
                            CanonicalExpression::Literal(CanonicalLiteral::String("else_block".to_string()))
                        }
                    } else {
                        CanonicalExpression::Literal(CanonicalLiteral::String("else_block".to_string()))
                    }
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("no_else".to_string()))
                };
                
                // Create a ternary-like expression
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "conditional".to_string(),
                    arguments: vec![condition_expr, then_expr, else_expr],
                }))
            }
            CLikeStatement::While { condition, body } => {
                let condition_expr = self.convert_c_like_expression(condition, context)?;
                
                // Convert while to a function call (simplified)
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "while_loop".to_string(),
                    arguments: vec![
                        condition_expr,
                        CanonicalExpression::Literal(CanonicalLiteral::String("loop_body".to_string())),
                    ],
                }))
            }
            CLikeStatement::For { init, condition, increment, body } => {
                // Convert for loop to a function call (simplified)
                let mut args = Vec::new();
                
                if let Some(_init_stmt) = init {
                    args.push(CanonicalExpression::Literal(CanonicalLiteral::String("init".to_string())));
                }
                
                if let Some(cond_expr) = condition {
                    args.push(self.convert_c_like_expression(cond_expr, context)?);
                }
                
                if let Some(inc_expr) = increment {
                    args.push(self.convert_c_like_expression(inc_expr, context)?);
                }
                
                args.push(CanonicalExpression::Literal(CanonicalLiteral::String("for_body".to_string())));
                
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "for_loop".to_string(),
                    arguments: args,
                }))
            }
            CLikeStatement::DoWhile { body, condition } => {
                let condition_expr = self.convert_c_like_expression(condition, context)?;
                
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "do_while_loop".to_string(),
                    arguments: vec![
                        CanonicalExpression::Literal(CanonicalLiteral::String("loop_body".to_string())),
                        condition_expr,
                    ],
                }))
            }
            CLikeStatement::Switch { expression, cases, default_case } => {
                let switch_expr = self.convert_c_like_expression(expression, context)?;
                
                // Convert switch to a match-like expression
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "switch_statement".to_string(),
                    arguments: vec![
                        switch_expr,
                        CanonicalExpression::Literal(CanonicalLiteral::String("switch_cases".to_string())),
                    ],
                }))
            }
            CLikeStatement::Break(label) => {
                let label_expr = if let Some(lbl) = label {
                    CanonicalExpression::Literal(CanonicalLiteral::String(lbl.clone()))
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
                };
                
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "break_statement".to_string(),
                    arguments: vec![label_expr],
                }))
            }
            CLikeStatement::Continue(label) => {
                let label_expr = if let Some(lbl) = label {
                    CanonicalExpression::Literal(CanonicalLiteral::String(lbl.clone()))
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
                };
                
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "continue_statement".to_string(),
                    arguments: vec![label_expr],
                }))
            }
            CLikeStatement::Block(statements) => {
                // Convert block to a sequence expression
                let mut canonical_exprs = Vec::new();
                for stmt in statements {
                    if let CanonicalStatement::Expression(expr) = self.convert_c_like_statement(stmt, context)? {
                        canonical_exprs.push(expr);
                    }
                }
                
                if canonical_exprs.len() == 1 {
                    Ok(CanonicalStatement::Expression(canonical_exprs.into_iter().next().unwrap()))
                } else {
                    Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                        function: "block".to_string(),
                        arguments: canonical_exprs,
                    }))
                }
            }
            CLikeStatement::Try { body, catch_blocks, finally_block } => {
                // Convert try-catch to a function call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "try_catch".to_string(),
                    arguments: vec![
                        CanonicalExpression::Literal(CanonicalLiteral::String("try_body".to_string())),
                        CanonicalExpression::Literal(CanonicalLiteral::String("catch_blocks".to_string())),
                    ],
                }))
            }
            CLikeStatement::Throw(expr) => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "throw".to_string(),
                    arguments: vec![self.convert_c_like_expression(expr, context)?],
                }))
            }
            CLikeStatement::VariableDeclaration { name, var_type, initializer } => {
                // Track the symbol
                context.symbols.insert(name.clone(), SymbolInfo {
                    name: name.clone(),
                    symbol_type: var_type.clone(),
                    scope_depth: context.scope_depth,
                    is_mutable: true,
                    usage_count: 0,
                });
                
                let canonical_value = if let Some(init) = initializer {
                    self.convert_c_like_expression(init, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("undefined".to_string()))
                };
                
                Ok(CanonicalStatement::Assignment {
                    name: name.clone(),
                    value: canonical_value,
                })
            }
            CLikeStatement::Empty => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Literal(
                    CanonicalLiteral::String("empty_statement".to_string())
                )))
            }
        }
    }
    
    /// Convert C-like expression to canonical form
    fn convert_c_like_expression(&self, expression: &CLikeExpression, context: &mut NormalizationContext) -> Result<CanonicalExpression, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match expression {
            CLikeExpression::Identifier(name) => {
                // Update symbol usage
                if let Some(symbol) = context.symbols.get_mut(name) {
                    symbol.usage_count += 1;
                }
                
                Ok(CanonicalExpression::Identifier(name.clone()))
            }
            CLikeExpression::Literal(literal) => {
                Ok(CanonicalExpression::Literal(self.convert_c_like_literal(literal)?))
            }
            CLikeExpression::Call { function, arguments } => {
                let function_name = match function.as_ref() {
                    CLikeExpression::Identifier(name) => name.clone(),
                    _ => "complex_function".to_string(), // Simplified for complex function expressions
                };
                
                let mut canonical_args = Vec::new();
                for arg in arguments {
                    canonical_args.push(self.convert_c_like_expression(arg, context)?);
                }
                
                Ok(CanonicalExpression::Call {
                    function: function_name,
                    arguments: canonical_args,
                })
            }
            CLikeExpression::Binary { left, operator, right } => {
                let left_expr = self.convert_c_like_expression(left, context)?;
                let right_expr = self.convert_c_like_expression(right, context)?;
                
                let op_string = match operator {
                    BinaryOperator::Add => "add",
                    BinaryOperator::Subtract => "subtract",
                    BinaryOperator::Multiply => "multiply",
                    BinaryOperator::Divide => "divide",
                    BinaryOperator::Modulo => "modulo",
                    BinaryOperator::Equal => "equal",
                    BinaryOperator::NotEqual => "not_equal",
                    BinaryOperator::Less => "less",
                    BinaryOperator::LessEqual => "less_equal",
                    BinaryOperator::Greater => "greater",
                    BinaryOperator::GreaterEqual => "greater_equal",
                    BinaryOperator::LogicalAnd => "logical_and",
                    BinaryOperator::LogicalOr => "logical_or",
                    BinaryOperator::BitwiseAnd => "bitwise_and",
                    BinaryOperator::BitwiseOr => "bitwise_or",
                    BinaryOperator::BitwiseXor => "bitwise_xor",
                    BinaryOperator::LeftShift => "left_shift",
                    BinaryOperator::RightShift => "right_shift",
                    BinaryOperator::Comma => "comma",
                };
                
                Ok(CanonicalExpression::Binary {
                    left: Box::new(left_expr),
                    operator: op_string.to_string(),
                    right: Box::new(right_expr),
                })
            }
            CLikeExpression::Unary { operator, operand } => {
                let operand_expr = self.convert_c_like_expression(operand, context)?;
                
                let op_string = match operator {
                    UnaryOperator::Plus => "unary_plus",
                    UnaryOperator::Minus => "unary_minus",
                    UnaryOperator::LogicalNot => "logical_not",
                    UnaryOperator::BitwiseNot => "bitwise_not",
                    UnaryOperator::AddressOf => "address_of",
                    UnaryOperator::Dereference => "dereference",
                };
                
                Ok(CanonicalExpression::Call {
                    function: op_string.to_string(),
                    arguments: vec![operand_expr],
                })
            }
            CLikeExpression::Ternary { condition, true_expr, false_expr } => {
                let cond_expr = self.convert_c_like_expression(condition, context)?;
                let true_canonical = self.convert_c_like_expression(true_expr, context)?;
                let false_canonical = self.convert_c_like_expression(false_expr, context)?;
                
                Ok(CanonicalExpression::Call {
                    function: "ternary".to_string(),
                    arguments: vec![cond_expr, true_canonical, false_canonical],
                })
            }
            CLikeExpression::Assignment { left, operator, right } => {
                let right_expr = self.convert_c_like_expression(right, context)?;
                
                // For simplified canonical form, treat all assignments as basic assignment
                if let CLikeExpression::Identifier(name) = left.as_ref() {
                    // Update symbol usage
                    if let Some(symbol) = context.symbols.get_mut(name) {
                        symbol.usage_count += 1;
                    }
                    
                    Ok(CanonicalExpression::Call {
                        function: "assign".to_string(),
                        arguments: vec![
                            CanonicalExpression::Identifier(name.clone()),
                            right_expr,
                        ],
                    })
                } else {
                    Ok(CanonicalExpression::Call {
                        function: "complex_assign".to_string(),
                        arguments: vec![
                            CanonicalExpression::Literal(CanonicalLiteral::String("complex_target".to_string())),
                            right_expr,
                        ],
                    })
                }
            }
            CLikeExpression::MemberAccess { object, member, safe_navigation } => {
                let object_expr = self.convert_c_like_expression(object, context)?;
                let access_type = if *safe_navigation { "safe_member_access" } else { "member_access" };
                
                Ok(CanonicalExpression::Call {
                    function: access_type.to_string(),
                    arguments: vec![
                        object_expr,
                        CanonicalExpression::Literal(CanonicalLiteral::String(member.clone())),
                    ],
                })
            }
            CLikeExpression::IndexAccess { object, index } => {
                let object_expr = self.convert_c_like_expression(object, context)?;
                let index_expr = self.convert_c_like_expression(index, context)?;
                
                Ok(CanonicalExpression::Call {
                    function: "index_access".to_string(),
                    arguments: vec![object_expr, index_expr],
                })
            }
            CLikeExpression::ArrayLiteral(elements) => {
                let mut canonical_elements = Vec::new();
                for element in elements {
                    canonical_elements.push(self.convert_c_like_expression(element, context)?);
                }
                
                Ok(CanonicalExpression::Call {
                    function: "array".to_string(),
                    arguments: canonical_elements,
                })
            }
            CLikeExpression::ObjectLiteral(fields) => {
                let mut canonical_args = Vec::new();
                for field in fields {
                    canonical_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(field.key.clone())));
                    canonical_args.push(self.convert_c_like_expression(&field.value, context)?);
                }
                
                Ok(CanonicalExpression::Call {
                    function: "object".to_string(),
                    arguments: canonical_args,
                })
            }
            CLikeExpression::Lambda { parameters, body } => {
                let mut param_names = Vec::new();
                for param in parameters {
                    param_names.push(CanonicalExpression::Literal(CanonicalLiteral::String(param.name.clone())));
                }
                
                let body_expr = self.convert_c_like_expression(body, context)?;
                
                let mut args = param_names;
                args.push(body_expr);
                
                Ok(CanonicalExpression::Call {
                    function: "lambda".to_string(),
                    arguments: args,
                })
            }
            CLikeExpression::Cast { target_type, expression } => {
                let expr = self.convert_c_like_expression(expression, context)?;
                
                Ok(CanonicalExpression::Call {
                    function: "cast".to_string(),
                    arguments: vec![
                        CanonicalExpression::Literal(CanonicalLiteral::String(target_type.clone())),
                        expr,
                    ],
                })
            }
            CLikeExpression::Parenthesized(expr) => {
                // Parentheses are just grouping, so we can directly convert the inner expression
                self.convert_c_like_expression(expr, context)
            }
            CLikeExpression::PostfixIncrement(expr) => {
                let operand_expr = self.convert_c_like_expression(expr, context)?;
                Ok(CanonicalExpression::Call {
                    function: "postfix_increment".to_string(),
                    arguments: vec![operand_expr],
                })
            }
            CLikeExpression::PostfixDecrement(expr) => {
                let operand_expr = self.convert_c_like_expression(expr, context)?;
                Ok(CanonicalExpression::Call {
                    function: "postfix_decrement".to_string(),
                    arguments: vec![operand_expr],
                })
            }
            CLikeExpression::PrefixIncrement(expr) => {
                let operand_expr = self.convert_c_like_expression(expr, context)?;
                Ok(CanonicalExpression::Call {
                    function: "prefix_increment".to_string(),
                    arguments: vec![operand_expr],
                })
            }
            CLikeExpression::PrefixDecrement(expr) => {
                let operand_expr = self.convert_c_like_expression(expr, context)?;
                Ok(CanonicalExpression::Call {
                    function: "prefix_decrement".to_string(),
                    arguments: vec![operand_expr],
                })
            }
        }
    }
    
    /// Convert C-like literal to canonical form
    fn convert_c_like_literal(&self, literal: &CLikeLiteral) -> Result<CanonicalLiteral, NormalizationError> {
        match literal {
            CLikeLiteral::String(s) => Ok(CanonicalLiteral::String(s.clone())),
            CLikeLiteral::Integer(i) => Ok(CanonicalLiteral::Integer(*i)),
            CLikeLiteral::Float(f) => Ok(CanonicalLiteral::Float(*f)),
            CLikeLiteral::Boolean(b) => Ok(CanonicalLiteral::Boolean(*b)),
            CLikeLiteral::Null => Ok(CanonicalLiteral::String("null".to_string())),
            CLikeLiteral::Character(c) => Ok(CanonicalLiteral::String(c.to_string())),
        }
    }
    
    /// Perform semantic analysis on the canonical form
    fn perform_semantic_analysis(&self, syntax: &CanonicalSyntax, context: &mut NormalizationContext) -> Result<(), NormalizationError> {
        if self.config.validation_level == ValidationLevel::None {
            return Ok(());
        }
        
        // Basic semantic checks
        self.check_symbol_usage(context)?;
        self.check_function_calls(syntax, context)?;
        self.check_type_consistency(syntax, context)?;
        
        Ok(())
    }
    
    /// Check symbol usage patterns
    fn check_symbol_usage(&self, context: &mut NormalizationContext) -> Result<(), NormalizationError> {
        for (name, symbol) in &context.symbols {
            if symbol.usage_count == 0 {
                context.warnings.push(NormalizationWarning {
                    message: format!("Unused symbol: {}", name),
                    span: None,
                    severity: WarningSeverity::Warning,
                    suggestion: Some(format!("Consider removing unused symbol '{}'", name)),
                });
            }
        }
        
        Ok(())
    }
    
    /// Check function calls for validity
    fn check_function_calls(&self, syntax: &CanonicalSyntax, context: &mut NormalizationContext) -> Result<(), NormalizationError> {
        // Basic function call validation
        for function in &syntax.functions {
            if function.name.is_empty() {
                context.warnings.push(NormalizationWarning {
                    message: "Function has empty name".to_string(),
                    span: Some(function.span),
                    severity: WarningSeverity::Error,
                    suggestion: Some("Provide a meaningful function name".to_string()),
                });
            }
        }
        
        Ok(())
    }
    
    /// Check type consistency
    fn check_type_consistency(&self, syntax: &CanonicalSyntax, context: &mut NormalizationContext) -> Result<(), NormalizationError> {
        // Basic type consistency checks
        for function in &syntax.functions {
            for param in &function.parameters {
                if param.name.is_empty() {
                    context.warnings.push(NormalizationWarning {
                        message: "Parameter has empty name".to_string(),
                        span: Some(function.span),
                        severity: WarningSeverity::Warning,
                        suggestion: Some("Provide meaningful parameter names".to_string()),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate AI metadata for the canonical form
    fn generate_ai_metadata(&self, syntax: &CanonicalSyntax, context: &mut NormalizationContext) -> Result<AIMetadata, NormalizationError> {
        // Generate AI-comprehensible metadata
        context.warnings.push(NormalizationWarning {
            message: format!("Generated AI metadata for {} functions, {} modules", 
                           syntax.functions.len(), syntax.modules.len()),
            span: None,
            severity: WarningSeverity::Info,
            suggestion: None,
        });
        
        // Create basic AI metadata
        let metadata = AIMetadata {
            business_context: Some(format!("Code with {} functions and {} modules", 
                                         syntax.functions.len(), syntax.modules.len())),
            domain_concepts: vec![
                "functions".to_string(),
                "modules".to_string(),
                "statements".to_string(),
            ],
            architectural_patterns: vec![
                "modular_design".to_string(),
                "functional_decomposition".to_string(),
            ],
                         complexity_metrics: ComplexityMetrics {
                 cyclomatic_complexity: syntax.functions.len(),
                 cognitive_complexity: syntax.statements.len(),
                 nesting_depth: 1,
                 dependency_count: syntax.modules.len(),
             },
                         semantic_relationships: vec![
                 SemanticRelationship {
                     source: "module".to_string(),
                     target: "function".to_string(),
                     relationship_type: RelationshipType::Composition,
                     strength: 0.9,
                 }
             ],
        };
        
        Ok(metadata)
    }
    
    /// Validate the final canonical form
    fn validate_canonical_form(&self, syntax: &CanonicalSyntax, context: &mut NormalizationContext) -> Result<(), NormalizationError> {
        if self.config.validation_level == ValidationLevel::None {
            return Ok(());
        }
        
        // Final validation checks
        if syntax.modules.is_empty() && syntax.functions.is_empty() && syntax.statements.is_empty() {
            context.warnings.push(NormalizationWarning {
                message: "Empty canonical form generated".to_string(),
                span: None,
                severity: WarningSeverity::Warning,
                suggestion: Some("Verify that the input contained valid constructs".to_string()),
            });
        }
        
        Ok(())
    }
    
    /// Create canonical metadata from context
    fn create_canonical_metadata(&self, context: &NormalizationContext) -> CanonicalMetadata {
        CanonicalMetadata {
            source_style: context.source_style.clone(),
            normalization_version: "1.0.0".to_string(),
            symbol_count: context.symbols.len(),
            warning_count: context.warnings.len(),
            processing_time: context.metrics.structure_conversion_time + 
                           context.metrics.semantic_analysis_time + 
                           context.metrics.ai_metadata_time,
        }
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// The final canonical form output
#[derive(Debug, Clone)]
pub struct CanonicalForm {
    /// The normalized canonical syntax
    pub syntax: CanonicalSyntax,
    
    /// Metadata about the normalization process
    pub metadata: CanonicalMetadata,
    
    /// Warnings generated during normalization
    pub warnings: Vec<NormalizationWarning>,
    
    /// Performance metrics
    pub metrics: NormalizationMetrics,
    
    /// AI metadata for code understanding
    pub ai_metadata: AIMetadata,
}

/// Metadata about the canonical form
#[derive(Debug, Clone)]
pub struct CanonicalMetadata {
    /// Original source syntax style
    pub source_style: SyntaxStyle,
    
    /// Normalization engine version
    pub normalization_version: String,
    
    /// Number of symbols processed
    pub symbol_count: usize,
    
    /// Number of warnings generated
    pub warning_count: usize,
    
    /// Total processing time
    pub processing_time: std::time::Duration,
} 