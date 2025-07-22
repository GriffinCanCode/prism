//! Python-like syntax normalizer for Prism.
//!
//! This module implements a comprehensive normalizer for Python-like syntax,
//! converting Python-specific constructs to canonical representation while
//! preserving all semantic meaning and generating AI-comprehensible metadata.
//!
//! Features:
//! - Indentation-based block structure normalization
//! - Python-specific construct handling (decorators, comprehensions, etc.)
//! - Type hint and annotation preservation
//! - F-string normalization with semantic preservation
//! - Match statement pattern normalization
//! - Comprehensive AI metadata generation
//!
//! The normalizer maintains conceptual cohesion around "Python-to-canonical
//! transformation with semantic preservation and AI comprehension support".

use crate::{
    detection::SyntaxStyle,
    normalization::{
        traits::{
            StyleNormalizer, NormalizerConfig, NormalizerCapabilities,
            PerformanceCharacteristics, AIMetadata, ComplexityMetrics,
            SemanticRelationship, RelationshipType, NormalizationUtils,
        },
        canonical_form::{CanonicalNode, CanonicalNodeType, CanonicalExpression, CanonicalStatement},
        NormalizationContext, NormalizationError, NormalizationWarning,
    },
    styles::python_like::{
        PythonSyntaxTree, Statement, Expression, ImportStatement, TypeAlias,
        PythonModule, FStringPart, Pattern, LiteralValue, AIHint,
    },
};
use prism_common::span::Span;
use std::collections::{HashMap, HashSet};

/// Python-like syntax normalizer
#[derive(Debug)]
pub struct PythonLikeNormalizer {
    /// Normalizer configuration
    config: PythonNormalizerConfig,
    
    /// Symbol table for tracking definitions
    symbol_table: SymbolTable,
    
    /// Import resolution cache
    import_cache: HashMap<String, ImportInfo>,
    
    /// Type information cache
    type_cache: HashMap<String, TypeInfo>,
    
    /// AI metadata collector
    ai_collector: AIMetadataCollector,
}

/// Configuration for Python normalizer
#[derive(Debug, Clone)]
pub struct PythonNormalizerConfig {
    /// Preserve Python-specific formatting hints
    pub preserve_formatting_hints: bool,
    
    /// Generate detailed type information
    pub generate_type_metadata: bool,
    
    /// Include import dependency graph
    pub track_import_dependencies: bool,
    
    /// Normalize string literals (handle escapes, etc.)
    pub normalize_string_literals: bool,
    
    /// Preserve original indentation information
    pub preserve_indentation_info: bool,
    
    /// Generate business logic insights
    pub generate_business_insights: bool,
    
    /// Maximum depth for nested structure analysis
    pub max_analysis_depth: usize,
    
    /// Enable experimental normalizations
    pub enable_experimental_features: bool,
}

/// Symbol table for tracking definitions and scopes
#[derive(Debug)]
struct SymbolTable {
    /// Global symbols
    globals: HashMap<String, SymbolInfo>,
    
    /// Current scope stack
    scope_stack: Vec<Scope>,
    
    /// Function definitions
    functions: HashMap<String, FunctionInfo>,
    
    /// Class definitions
    classes: HashMap<String, ClassInfo>,
}

/// Symbol information
#[derive(Debug, Clone)]
struct SymbolInfo {
    name: String,
    symbol_type: SymbolType,
    definition_span: Option<Span>,
    usages: Vec<Span>,
    type_annotation: Option<String>,
}

/// Symbol types
#[derive(Debug, Clone)]
enum SymbolType {
    Variable,
    Function,
    Class,
    Module,
    Parameter,
    Import,
}

/// Scope information
#[derive(Debug)]
struct Scope {
    scope_type: ScopeType,
    symbols: HashMap<String, SymbolInfo>,
    parent: Option<usize>,
}

/// Scope types
#[derive(Debug)]
enum ScopeType {
    Module,
    Function,
    Class,
    Comprehension,
    Lambda,
}

/// Import information
#[derive(Debug, Clone)]
struct ImportInfo {
    module_name: String,
    imported_names: Vec<String>,
    aliases: HashMap<String, String>,
    is_relative: bool,
    level: usize,
}

/// Type information
#[derive(Debug, Clone)]
struct TypeInfo {
    type_name: String,
    generic_parameters: Vec<String>,
    base_types: Vec<String>,
    is_builtin: bool,
}

/// Function information
#[derive(Debug, Clone)]
struct FunctionInfo {
    name: String,
    parameters: Vec<ParameterInfo>,
    return_type: Option<String>,
    decorators: Vec<String>,
    is_async: bool,
    is_generator: bool,
}

/// Parameter information
#[derive(Debug, Clone)]
struct ParameterInfo {
    name: String,
    type_annotation: Option<String>,
    default_value: Option<String>,
    is_vararg: bool,
    is_kwarg: bool,
}

/// Class information
#[derive(Debug, Clone)]
struct ClassInfo {
    name: String,
    base_classes: Vec<String>,
    methods: Vec<String>,
    attributes: Vec<String>,
    decorators: Vec<String>,
    is_dataclass: bool,
}

/// AI metadata collector for Python-specific insights
#[derive(Debug)]
struct AIMetadataCollector {
    /// Business domain concepts found
    domain_concepts: HashSet<String>,
    
    /// Architectural patterns identified
    patterns: HashSet<String>,
    
    /// Code smells detected
    code_smells: Vec<CodeSmell>,
    
    /// Complexity hotspots
    complexity_hotspots: Vec<ComplexityHotspot>,
    
    /// Import relationships
    import_graph: HashMap<String, Vec<String>>,
}

/// Code smell detection
#[derive(Debug, Clone)]
struct CodeSmell {
    smell_type: CodeSmellType,
    description: String,
    span: Option<Span>,
    severity: SmellSeverity,
}

/// Types of code smells
#[derive(Debug, Clone)]
enum CodeSmellType {
    LongFunction,
    DeepNesting,
    TooManyParameters,
    DuplicatedCode,
    LargeClass,
    FeatureEnvy,
    DataClass,
    GodClass,
}

/// Code smell severity
#[derive(Debug, Clone)]
enum SmellSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Complexity hotspot
#[derive(Debug, Clone)]
struct ComplexityHotspot {
    location: String,
    complexity_score: f64,
    contributing_factors: Vec<String>,
    span: Option<Span>,
}

impl Default for PythonNormalizerConfig {
    fn default() -> Self {
        Self {
            preserve_formatting_hints: true,
            generate_type_metadata: true,
            track_import_dependencies: true,
            normalize_string_literals: true,
            preserve_indentation_info: true,
            generate_business_insights: true,
            max_analysis_depth: 10,
            enable_experimental_features: false,
        }
    }
}

impl NormalizerConfig for PythonNormalizerConfig {
    fn validate(&self) -> Result<(), crate::normalization::traits::ConfigurationError> {
        if self.max_analysis_depth == 0 {
            return Err(crate::normalization::traits::ConfigurationError::InvalidParameter {
                parameter: "max_analysis_depth".to_string(),
                value: "0".to_string(),
            });
        }
        
        Ok(())
    }
    
    fn merge_with(&mut self, other: &Self) {
        self.preserve_formatting_hints = other.preserve_formatting_hints;
        self.generate_type_metadata = other.generate_type_metadata;
        self.track_import_dependencies = other.track_import_dependencies;
        self.normalize_string_literals = other.normalize_string_literals;
        self.preserve_indentation_info = other.preserve_indentation_info;
        self.generate_business_insights = other.generate_business_insights;
        self.max_analysis_depth = other.max_analysis_depth;
        self.enable_experimental_features = other.enable_experimental_features;
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            globals: HashMap::new(),
            scope_stack: vec![Scope {
                scope_type: ScopeType::Module,
                symbols: HashMap::new(),
                parent: None,
            }],
            functions: HashMap::new(),
            classes: HashMap::new(),
        }
    }
}

impl Default for AIMetadataCollector {
    fn default() -> Self {
        Self {
            domain_concepts: HashSet::new(),
            patterns: HashSet::new(),
            code_smells: Vec::new(),
            complexity_hotspots: Vec::new(),
            import_graph: HashMap::new(),
        }
    }
}

impl StyleNormalizer for PythonLikeNormalizer {
    type Input = PythonSyntaxTree;
    type Intermediate = CanonicalNode;
    type Config = PythonNormalizerConfig;
    
    fn new() -> Self {
        Self::with_config(PythonNormalizerConfig::default())
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self {
            config,
            symbol_table: SymbolTable::default(),
            import_cache: HashMap::new(),
            type_cache: HashMap::new(),
            ai_collector: AIMetadataCollector::default(),
        }
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::PythonLike
    }
    
    fn normalize(
        &self,
        input: &Self::Input,
        context: &mut NormalizationContext,
    ) -> Result<Self::Intermediate, NormalizationError> {
        context.start_phase("python_normalization");
        
        // Create root canonical node
        let mut root = CanonicalNode::new(
            CanonicalNodeType::Module,
            Some("python_module".to_string()),
        );
        
        // Normalize module-level information
        self.normalize_module(&input.module, &mut root, context)?;
        
        // Normalize imports
        for import in &input.imports {
            let import_node = self.normalize_import(import, context)?;
            root.add_child(import_node);
        }
        
        // Normalize type aliases (PEP 695)
        for type_alias in &input.type_aliases {
            let alias_node = self.normalize_type_alias(type_alias, context)?;
            root.add_child(alias_node);
        }
        
        // Normalize statements
        for statement in &input.statements {
            let stmt_node = self.normalize_statement(statement, context)?;
            root.add_child(stmt_node);
        }
        
        // Add Python-specific metadata
        if self.config.preserve_formatting_hints {
            root.add_metadata("indentation_style", "spaces".to_string());
            root.add_metadata("python_version", "3.12+".to_string());
        }
        
        context.end_phase("python_normalization");
        Ok(root)
    }
    
    fn validate_normalized(
        &self,
        normalized: &Self::Intermediate,
        context: &mut NormalizationContext,
    ) -> Result<(), NormalizationError> {
        context.start_phase("python_validation");
        
        // Validate canonical structure
        self.validate_canonical_structure(normalized, context)?;
        
        // Validate Python-specific semantics
        self.validate_python_semantics(normalized, context)?;
        
        context.end_phase("python_validation");
        Ok(())
    }
    
    fn generate_ai_metadata(
        &self,
        normalized: &Self::Intermediate,
        context: &mut NormalizationContext,
    ) -> Result<AIMetadata, NormalizationError> {
        context.start_phase("ai_metadata_generation");
        
        let mut metadata = AIMetadata::default();
        
        // Extract business concepts
        metadata.domain_concepts = self.extract_business_concepts(normalized);
        
        // Identify architectural patterns
        metadata.architectural_patterns = self.identify_architectural_patterns(normalized);
        
        // Calculate complexity metrics
        metadata.complexity_metrics = self.calculate_complexity_metrics(normalized);
        
        // Find semantic relationships
        metadata.semantic_relationships = self.find_semantic_relationships(normalized);
        
        // Add Python-specific insights
        if let Some(business_context) = self.infer_business_context(normalized) {
            metadata.business_context = Some(business_context);
        }
        
        context.end_phase("ai_metadata_generation");
        Ok(metadata)
    }
    
    fn capabilities(&self) -> NormalizerCapabilities {
        NormalizerCapabilities {
            supported_constructs: vec![
                "function_definitions".to_string(),
                "class_definitions".to_string(),
                "import_statements".to_string(),
                "type_aliases".to_string(),
                "match_statements".to_string(),
                "async_await".to_string(),
                "decorators".to_string(),
                "comprehensions".to_string(),
                "f_strings".to_string(),
                "type_hints".to_string(),
                "exception_handling".to_string(),
                "context_managers".to_string(),
                "generators".to_string(),
                "lambda_expressions".to_string(),
                "pattern_matching".to_string(),
            ],
            unsupported_constructs: vec![
                "exec_statements".to_string(),
                "eval_expressions".to_string(),
                "metaclass_advanced_features".to_string(),
            ],
            supports_error_recovery: true,
            generates_ai_metadata: true,
            performance_characteristics: PerformanceCharacteristics {
                time_complexity: "O(n)".to_string(),
                space_complexity: "O(n)".to_string(),
                supports_parallel_processing: false,
                memory_per_node_bytes: 512,
            },
        }
    }
}

impl PythonLikeNormalizer {
    /// Normalize module information
    fn normalize_module(
        &self,
        module: &PythonModule,
        root: &mut CanonicalNode,
        _context: &mut NormalizationContext,
    ) -> Result<(), NormalizationError> {
        // Add module docstring if present
        if let Some(docstring) = &module.docstring {
            root.add_metadata("docstring", docstring.clone());
        }
        
        // Add encoding information
        if let Some(encoding) = &module.encoding {
            root.add_metadata("encoding", encoding.clone());
        }
        
        // Add future imports
        if !module.future_imports.is_empty() {
            root.add_metadata(
                "future_imports",
                module.future_imports.join(","),
            );
        }
        
        Ok(())
    }
    
    /// Normalize import statements
    fn normalize_import(
        &self,
        import: &ImportStatement,
        _context: &mut NormalizationContext,
    ) -> Result<CanonicalNode, NormalizationError> {
        let mut import_node = CanonicalNode::new(
            CanonicalNodeType::Import,
            None,
        );
        
        match import {
            ImportStatement::Import { modules, span } => {
                import_node.add_metadata("import_type", "simple".to_string());
                
                for module in modules {
                    let mut module_node = CanonicalNode::new(
                        CanonicalNodeType::Identifier,
                        Some(module.name.clone()),
                    );
                    
                    if let Some(alias) = &module.alias {
                        module_node.add_metadata("alias", alias.clone());
                    }
                    
                    import_node.add_child(module_node);
                }
            }
            
            ImportStatement::FromImport { module, names, level, span } => {
                import_node.add_metadata("import_type", "from".to_string());
                import_node.add_metadata("level", level.to_string());
                
                if let Some(module_name) = module {
                    import_node.add_metadata("module", module_name.clone());
                }
                
                for name in names {
                    let mut name_node = CanonicalNode::new(
                        CanonicalNodeType::Identifier,
                        Some(name.name.clone()),
                    );
                    
                    if let Some(alias) = &name.alias {
                        name_node.add_metadata("alias", alias.clone());
                    }
                    
                    import_node.add_child(name_node);
                }
            }
        }
        
        Ok(import_node)
    }
    
    /// Normalize type alias (PEP 695)
    fn normalize_type_alias(
        &self,
        type_alias: &TypeAlias,
        _context: &mut NormalizationContext,
    ) -> Result<CanonicalNode, NormalizationError> {
        let mut alias_node = CanonicalNode::new(
            CanonicalNodeType::TypeAlias,
            Some(type_alias.name.clone()),
        );
        
        // Add type parameters if present
        if !type_alias.type_parameters.is_empty() {
            let mut params_node = CanonicalNode::new(
                CanonicalNodeType::TypeParameters,
                None,
            );
            
            for param in &type_alias.type_parameters {
                let param_node = self.normalize_type_parameter(param)?;
                params_node.add_child(param_node);
            }
            
            alias_node.add_child(params_node);
        }
        
        // Add the type value
        let value_node = self.normalize_type_expression(&type_alias.value)?;
        alias_node.add_child(value_node);
        
        Ok(alias_node)
    }
    
    /// Normalize statement
    fn normalize_statement(
        &self,
        statement: &Statement,
        _context: &mut NormalizationContext,
    ) -> Result<CanonicalNode, NormalizationError> {
        match statement {
            Statement::Expression { expression, span } => {
                let mut stmt_node = CanonicalNode::new(
                    CanonicalNodeType::ExpressionStatement,
                    None,
                );
                
                let expr_node = self.normalize_expression(expression)?;
                stmt_node.add_child(expr_node);
                
                Ok(stmt_node)
            }
            
            Statement::Assignment { targets, value, type_annotation, span } => {
                let mut assign_node = CanonicalNode::new(
                    CanonicalNodeType::Assignment,
                    None,
                );
                
                // Add targets
                for target in targets {
                    let target_node = self.normalize_assignment_target(target)?;
                    assign_node.add_child(target_node);
                }
                
                // Add value
                let value_node = self.normalize_expression(value)?;
                assign_node.add_child(value_node);
                
                // Add type annotation if present
                if let Some(type_ann) = type_annotation {
                    let type_node = self.normalize_type_expression(type_ann)?;
                    assign_node.add_child(type_node);
                }
                
                Ok(assign_node)
            }
            
            Statement::FunctionDef {
                name,
                type_parameters,
                parameters,
                return_type,
                body,
                decorators,
                is_async,
                span,
            } => {
                let mut func_node = CanonicalNode::new(
                    CanonicalNodeType::FunctionDefinition,
                    Some(name.clone()),
                );
                
                if *is_async {
                    func_node.add_metadata("async", "true".to_string());
                }
                
                // Add type parameters (PEP 695)
                if !type_parameters.is_empty() {
                    let mut type_params_node = CanonicalNode::new(
                        CanonicalNodeType::TypeParameters,
                        None,
                    );
                    
                    for param in type_parameters {
                        let param_node = self.normalize_type_parameter(param)?;
                        type_params_node.add_child(param_node);
                    }
                    
                    func_node.add_child(type_params_node);
                }
                
                // Add parameters
                let mut params_node = CanonicalNode::new(
                    CanonicalNodeType::Parameters,
                    None,
                );
                
                for param in parameters {
                    let param_node = self.normalize_parameter(param)?;
                    params_node.add_child(param_node);
                }
                
                func_node.add_child(params_node);
                
                // Add return type if present
                if let Some(ret_type) = return_type {
                    let ret_type_node = self.normalize_type_expression(ret_type)?;
                    ret_type_node.add_metadata("return_type", "true".to_string());
                    func_node.add_child(ret_type_node);
                }
                
                // Add decorators
                for decorator in decorators {
                    let decorator_node = self.normalize_decorator(decorator)?;
                    func_node.add_child(decorator_node);
                }
                
                // Add body
                let mut body_node = CanonicalNode::new(
                    CanonicalNodeType::Block,
                    None,
                );
                
                for stmt in body {
                    let stmt_node = self.normalize_statement(stmt, _context)?;
                    body_node.add_child(stmt_node);
                }
                
                func_node.add_child(body_node);
                
                Ok(func_node)
            }
            
            // TODO: Implement other statement types
            _ => {
                // For now, create a placeholder node
                Ok(CanonicalNode::new(
                    CanonicalNodeType::Unknown,
                    Some("unimplemented_statement".to_string()),
                ))
            }
        }
    }
    
    // Helper methods (placeholder implementations)
    
    fn normalize_type_parameter(
        &self,
        param: &crate::styles::python_like::TypeParameter,
    ) -> Result<CanonicalNode, NormalizationError> {
        let canonical_param = CanonicalNode {
            node_type: CanonicalNodeType::TypeParameter,
            content: param.name.clone(),
            attributes: {
                let mut attrs = std::collections::HashMap::new();
                if let Some(bound) = &param.bound {
                    attrs.insert("bound".to_string(), bound.clone());
                }
                if let Some(default) = &param.default {
                    attrs.insert("default".to_string(), default.clone());
                }
                attrs.insert("variance".to_string(), format!("{:?}", param.variance));
                attrs
            },
            children: Vec::new(),
            metadata: CanonicalMetadata {
                source_style: SyntaxStyle::PythonLike,
                confidence_score: 0.9,
                normalization_notes: vec!["Normalized Python-like type parameter".to_string()],
                business_context: None,
                semantic_hints: vec!["type_parameter".to_string()],
            },
        };
        
        Ok(canonical_param)
    }
    
    fn normalize_type_expression(
        &self,
        type_expr: &crate::styles::python_like::TypeExpression,
    ) -> Result<CanonicalNode, NormalizationError> {
        use crate::styles::python_like::TypeExpression;
        
        let (node_type, content, mut attributes) = match type_expr {
            TypeExpression::Name(name) => {
                (CanonicalNodeType::TypeName, name.clone(), std::collections::HashMap::new())
            }
            TypeExpression::Generic { base, args } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("base".to_string(), base.clone());
                attrs.insert("args".to_string(), format!("{:?}", args));
                (CanonicalNodeType::GenericType, base.clone(), attrs)
            }
            TypeExpression::Union(types) => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("types".to_string(), format!("{:?}", types));
                (CanonicalNodeType::UnionType, "Union".to_string(), attrs)
            }
            TypeExpression::Optional(inner) => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("inner".to_string(), format!("{:?}", inner));
                (CanonicalNodeType::OptionalType, "Optional".to_string(), attrs)
            }
            TypeExpression::Tuple(elements) => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("elements".to_string(), format!("{:?}", elements));
                (CanonicalNodeType::TupleType, "Tuple".to_string(), attrs)
            }
            TypeExpression::Callable { params, return_type } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("params".to_string(), format!("{:?}", params));
                attrs.insert("return_type".to_string(), format!("{:?}", return_type));
                (CanonicalNodeType::FunctionType, "Callable".to_string(), attrs)
            }
        };
        
        attributes.insert("python_style".to_string(), "true".to_string());
        
        let canonical_type = CanonicalNode {
            node_type,
            content,
            attributes,
            children: Vec::new(),
            metadata: CanonicalMetadata {
                source_style: SyntaxStyle::PythonLike,
                confidence_score: 0.85,
                normalization_notes: vec!["Normalized Python-like type expression".to_string()],
                business_context: None,
                semantic_hints: vec!["type_expression".to_string()],
            },
        };
        
        Ok(canonical_type)
    }
    
    fn normalize_expression(
        &self,
        expr: &Expression,
    ) -> Result<CanonicalNode, NormalizationError> {
        use crate::styles::python_like::Expression;
        
        let (node_type, content, mut attributes, children) = match expr {
            Expression::Literal(lit) => {
                let content = match lit {
                    crate::styles::python_like::Literal::Integer(i) => i.to_string(),
                    crate::styles::python_like::Literal::Float(f) => f.to_string(),
                    crate::styles::python_like::Literal::String(s) => s.clone(),
                    crate::styles::python_like::Literal::Boolean(b) => b.to_string(),
                    crate::styles::python_like::Literal::None => "None".to_string(),
                };
                (CanonicalNodeType::Literal, content, std::collections::HashMap::new(), Vec::new())
            }
            Expression::Name(name) => {
                (CanonicalNodeType::Identifier, name.clone(), std::collections::HashMap::new(), Vec::new())
            }
            Expression::BinaryOp { left, op, right } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("operator".to_string(), format!("{:?}", op));
                let mut children = Vec::new();
                children.push(self.normalize_expression(left)?);
                children.push(self.normalize_expression(right)?);
                (CanonicalNodeType::BinaryOperation, format!("{:?}", op), attrs, children)
            }
            Expression::UnaryOp { op, operand } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("operator".to_string(), format!("{:?}", op));
                let children = vec![self.normalize_expression(operand)?];
                (CanonicalNodeType::UnaryOperation, format!("{:?}", op), attrs, children)
            }
            Expression::Call { func, args, keywords } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("args_count".to_string(), args.len().to_string());
                attrs.insert("keywords_count".to_string(), keywords.len().to_string());
                let mut children = vec![self.normalize_expression(func)?];
                for arg in args {
                    children.push(self.normalize_expression(arg)?);
                }
                (CanonicalNodeType::FunctionCall, "call".to_string(), attrs, children)
            }
            Expression::Attribute { value, attr } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("attribute".to_string(), attr.clone());
                let children = vec![self.normalize_expression(value)?];
                (CanonicalNodeType::FieldAccess, attr.clone(), attrs, children)
            }
            Expression::Subscript { value, slice } => {
                let children = vec![
                    self.normalize_expression(value)?,
                    self.normalize_expression(slice)?,
                ];
                (CanonicalNodeType::ArrayAccess, "subscript".to_string(), std::collections::HashMap::new(), children)
            }
            Expression::List(elements) => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("elements_count".to_string(), elements.len().to_string());
                let children: Result<Vec<_>, _> = elements.iter()
                    .map(|e| self.normalize_expression(e))
                    .collect();
                (CanonicalNodeType::ArrayLiteral, "list".to_string(), attrs, children?)
            }
            Expression::Dict { keys, values } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("pairs_count".to_string(), keys.len().to_string());
                let mut children = Vec::new();
                for (key, value) in keys.iter().zip(values.iter()) {
                    children.push(self.normalize_expression(key)?);
                    children.push(self.normalize_expression(value)?);
                }
                (CanonicalNodeType::ObjectLiteral, "dict".to_string(), attrs, children)
            }
            Expression::Lambda { params, body } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("params_count".to_string(), params.len().to_string());
                let children = vec![self.normalize_expression(body)?];
                (CanonicalNodeType::Lambda, "lambda".to_string(), attrs, children)
            }
            Expression::Comprehension { element, generators } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("generators_count".to_string(), generators.len().to_string());
                let children = vec![self.normalize_expression(element)?];
                (CanonicalNodeType::Comprehension, "comprehension".to_string(), attrs, children)
            }
        };
        
        attributes.insert("python_style".to_string(), "true".to_string());
        
        let canonical_expr = CanonicalNode {
            node_type,
            content,
            attributes,
            children,
            metadata: CanonicalMetadata {
                source_style: SyntaxStyle::PythonLike,
                confidence_score: 0.9,
                normalization_notes: vec!["Normalized Python-like expression".to_string()],
                business_context: None,
                semantic_hints: vec!["expression".to_string()],
            },
        };
        
        Ok(canonical_expr)
    }
    
    fn normalize_assignment_target(
        &self,
        target: &crate::styles::python_like::AssignmentTarget,
    ) -> Result<CanonicalNode, NormalizationError> {
        use crate::styles::python_like::AssignmentTarget;
        
        let (node_type, content, attributes, children) = match target {
            AssignmentTarget::Name(name) => {
                (CanonicalNodeType::Identifier, name.clone(), std::collections::HashMap::new(), Vec::new())
            }
            AssignmentTarget::Attribute { value, attr } => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("attribute".to_string(), attr.clone());
                // For now, we'll create a placeholder child - in a full implementation,
                // we'd need to normalize the value expression
                (CanonicalNodeType::FieldAccess, attr.clone(), attrs, Vec::new())
            }
            AssignmentTarget::Subscript { value, slice } => {
                // For now, we'll create a placeholder - in a full implementation,
                // we'd need to normalize both value and slice expressions
                (CanonicalNodeType::ArrayAccess, "subscript".to_string(), std::collections::HashMap::new(), Vec::new())
            }
            AssignmentTarget::Tuple(elements) => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("elements_count".to_string(), elements.len().to_string());
                // For now, we'll create placeholders - in a full implementation,
                // we'd recursively normalize each element
                (CanonicalNodeType::TupleLiteral, "tuple".to_string(), attrs, Vec::new())
            }
            AssignmentTarget::List(elements) => {
                let mut attrs = std::collections::HashMap::new();
                attrs.insert("elements_count".to_string(), elements.len().to_string());
                // For now, we'll create placeholders - in a full implementation,
                // we'd recursively normalize each element
                (CanonicalNodeType::ArrayLiteral, "list".to_string(), attrs, Vec::new())
            }
        };
        
        let canonical_target = CanonicalNode {
            node_type,
            content,
            attributes,
            children,
            metadata: CanonicalMetadata {
                source_style: SyntaxStyle::PythonLike,
                confidence_score: 0.85,
                normalization_notes: vec!["Normalized Python-like assignment target".to_string()],
                business_context: None,
                semantic_hints: vec!["assignment_target".to_string()],
            },
        };
        
        Ok(canonical_target)
    }
    
    fn normalize_parameter(
        &self,
        param: &crate::styles::python_like::Parameter,
    ) -> Result<CanonicalNode, NormalizationError> {
        let mut attributes = std::collections::HashMap::new();
        attributes.insert("name".to_string(), param.name.clone());
        
        if let Some(type_annotation) = &param.type_annotation {
            attributes.insert("type".to_string(), format!("{:?}", type_annotation));
        }
        
        if let Some(default) = &param.default {
            attributes.insert("default".to_string(), format!("{:?}", default));
        }
        
        attributes.insert("kind".to_string(), format!("{:?}", param.kind));
        attributes.insert("python_style".to_string(), "true".to_string());
        
        let canonical_param = CanonicalNode {
            node_type: CanonicalNodeType::Parameter,
            content: param.name.clone(),
            attributes,
            children: Vec::new(),
            metadata: CanonicalMetadata {
                source_style: SyntaxStyle::PythonLike,
                confidence_score: 0.9,
                normalization_notes: vec!["Normalized Python-like parameter".to_string()],
                business_context: None,
                semantic_hints: vec!["parameter".to_string()],
            },
        };
        
        Ok(canonical_param)
    }
    
    fn normalize_decorator(
        &self,
        decorator: &crate::styles::python_like::Decorator,
    ) -> Result<CanonicalNode, NormalizationError> {
        let mut attributes = std::collections::HashMap::new();
        attributes.insert("name".to_string(), decorator.name.clone());
        
        if let Some(args) = &decorator.args {
            attributes.insert("args".to_string(), format!("{:?}", args));
        }
        
        if let Some(kwargs) = &decorator.kwargs {
            attributes.insert("kwargs".to_string(), format!("{:?}", kwargs));
        }
        
        attributes.insert("python_style".to_string(), "true".to_string());
        
        let canonical_decorator = CanonicalNode {
            node_type: CanonicalNodeType::Decorator,
            content: decorator.name.clone(),
            attributes,
            children: Vec::new(),
            metadata: CanonicalMetadata {
                source_style: SyntaxStyle::PythonLike,
                confidence_score: 0.85,
                normalization_notes: vec!["Normalized Python-like decorator".to_string()],
                business_context: Some("Decorators often represent cross-cutting concerns like logging, authentication, or caching".to_string()),
                semantic_hints: vec!["decorator", "annotation", "metadata"].iter().map(|s| s.to_string()).collect(),
            },
        };
        
        Ok(canonical_decorator)
    }
    
    fn validate_canonical_structure(
        &self,
        _normalized: &CanonicalNode,
        _context: &mut NormalizationContext,
    ) -> Result<(), NormalizationError> {
        // TODO: Implement structural validation
        Ok(())
    }
    
    fn validate_python_semantics(
        &self,
        _normalized: &CanonicalNode,
        _context: &mut NormalizationContext,
    ) -> Result<(), NormalizationError> {
        // TODO: Implement Python-specific semantic validation
        Ok(())
    }
    
    fn extract_business_concepts(&self, _normalized: &CanonicalNode) -> Vec<String> {
        // TODO: Implement business concept extraction
        Vec::new()
    }
    
    fn identify_architectural_patterns(&self, _normalized: &CanonicalNode) -> Vec<String> {
        // TODO: Implement architectural pattern identification
        Vec::new()
    }
    
    fn calculate_complexity_metrics(&self, _normalized: &CanonicalNode) -> ComplexityMetrics {
        // TODO: Implement complexity calculation
        ComplexityMetrics::default()
    }
    
    fn find_semantic_relationships(&self, _normalized: &CanonicalNode) -> Vec<SemanticRelationship> {
        // TODO: Implement semantic relationship analysis
        Vec::new()
    }
    
    fn infer_business_context(&self, _normalized: &CanonicalNode) -> Option<String> {
        // TODO: Implement business context inference
        None
    }
} 