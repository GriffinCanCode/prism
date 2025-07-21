//! Core Semantic Analysis Engine
//!
//! This module embodies the single concept of "Semantic Analysis Orchestration".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: coordinating and orchestrating semantic analysis across all program elements.
//!
//! **Conceptual Responsibility**: Orchestrate comprehensive semantic analysis
//! **What it does**: program analysis, symbol resolution, semantic coordination
//! **What it doesn't do**: type inference, validation, pattern recognition (delegates to specialized modules)

use crate::{SemanticResult, SemanticError, SemanticConfig};
use prism_ast::{Program, AstNode, Item, Stmt, Expr, Type};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Business rule extracted from semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    /// Rule name
    pub rule_name: String,
    /// Rule type/category
    pub rule_type: String,
    /// Confidence in this rule (0.0 to 1.0)
    pub confidence: f64,
    /// Rule description
    pub description: String,
    /// Evidence supporting this rule
    pub evidence: Vec<String>,
    /// Location where rule was detected
    pub location: Option<Span>,
}

/// Semantic relationship between code elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelationship {
    /// Source element
    pub source: String,
    /// Target element
    pub target: String,
    /// Type of relationship
    pub relationship_type: String,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Evidence supporting this relationship
    pub evidence: Vec<String>,
    /// Location where relationship was detected
    pub location: Option<Span>,
}

/// Core semantic analyzer that orchestrates all semantic analysis
#[derive(Debug)]
pub struct SemanticAnalyzer {
    /// Configuration
    config: AnalysisConfig,
    /// Symbol table
    symbols: HashMap<Symbol, SymbolInfo>,
    /// Type information
    types: HashMap<NodeId, TypeInfo>,
    /// Current analysis context
    context: AnalysisContext,
}

/// Configuration for semantic analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable deep analysis
    pub enable_deep_analysis: bool,
    /// Enable cross-module analysis
    pub enable_cross_module: bool,
    /// Maximum analysis depth
    pub max_depth: usize,
    /// Enable symbol resolution
    pub enable_symbol_resolution: bool,
}

/// Result of semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Symbol information
    pub symbols: HashMap<Symbol, SymbolInfo>,
    /// Type information
    pub types: HashMap<NodeId, TypeInfo>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Symbol information with semantic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol identifier
    pub id: Symbol,
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: SymbolType,
    /// Source location
    pub location: Span,
    /// Visibility
    pub visibility: Visibility,
    /// Semantic annotations
    pub semantic_annotations: Vec<String>,
    /// Business context
    pub business_context: Option<String>,
    /// AI hints for understanding
    pub ai_hints: Vec<String>,
}

/// Type information with semantic meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type identifier
    pub type_id: NodeId,
    /// Type kind
    pub type_kind: TypeKind,
    /// Source location
    pub location: Span,
    /// Semantic meaning
    pub semantic_meaning: Option<String>,
    /// Business domain
    pub domain: Option<String>,
    /// AI description
    pub ai_description: Option<String>,
}

/// Symbol type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolType {
    /// Variable
    Variable,
    /// Function
    Function,
    /// Type definition
    Type,
    /// Module
    Module,
    /// Constant
    Constant,
    /// Parameter
    Parameter,
}

/// Type kind classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeKind {
    /// Primitive type
    Primitive(String),
    /// Semantic type with business meaning
    Semantic(String),
    /// Composite type
    Composite(String),
    /// Function type
    Function,
    /// Generic type
    Generic,
    /// Effect type
    Effect,
}

/// Visibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    /// Public
    Public,
    /// Private
    Private,
    /// Internal
    Internal,
}

/// Analysis context
#[derive(Debug, Clone)]
struct AnalysisContext {
    /// Current module
    current_module: Option<String>,
    /// Current function
    current_function: Option<String>,
    /// Current type
    current_type: Option<String>,
    /// Nesting depth
    depth: usize,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: String,
    /// Analysis duration
    pub duration_ms: u64,
    /// Symbols analyzed
    pub symbols_analyzed: usize,
    /// Types analyzed
    pub types_analyzed: usize,
    /// Warnings generated
    pub warnings: Vec<String>,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new(config: &SemanticConfig) -> SemanticResult<Self> {
        let analysis_config = AnalysisConfig {
            enable_deep_analysis: config.enable_ai_metadata,
            enable_cross_module: true,
            max_depth: config.max_analysis_depth,
            enable_symbol_resolution: true,
        };

        Ok(Self {
            config: analysis_config,
            symbols: HashMap::new(),
            types: HashMap::new(),
            context: AnalysisContext {
                current_module: None,
                current_function: None,
                current_type: None,
                depth: 0,
            },
        })
    }

    /// Analyze a complete program
    pub fn analyze_program(&mut self, program: &Program) -> SemanticResult<AnalysisResult> {
        let start_time = std::time::Instant::now();
        
        // Reset state for new analysis
        self.reset_analysis();

        // Analyze all items in the program
        for item in &program.items {
            self.analyze_item(item)?;
        }

        let duration = start_time.elapsed();
        let metadata = AnalysisMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_millis() as u64,
            symbols_analyzed: self.symbols.len(),
            types_analyzed: self.types.len(),
            warnings: Vec::new(), // Would be populated with actual warnings
        };

        Ok(AnalysisResult {
            symbols: self.symbols.clone(),
            types: self.types.clone(),
            metadata,
        })
    }

    /// Reset analysis state
    fn reset_analysis(&mut self) {
        self.symbols.clear();
        self.types.clear();
        self.context = AnalysisContext {
            current_module: None,
            current_function: None,
            current_type: None,
            depth: 0,
        };
    }

    /// Analyze a single item
    fn analyze_item(&mut self, item: &AstNode<Item>) -> SemanticResult<()> {
        if self.context.depth >= self.config.max_depth {
            return Err(SemanticError::ValidationError {
                location: item.span,
                message: "Maximum analysis depth exceeded".to_string(),
            });
        }

        self.context.depth += 1;

        match &item.kind {
            Item::Function(func_decl) => {
                self.analyze_function(item.id, func_decl, item.span)?;
            }
            Item::Type(type_decl) => {
                self.analyze_type_declaration(item.id, type_decl, item.span)?;
            }
            Item::Module(module_decl) => {
                self.analyze_module(item.id, module_decl, item.span)?;
            }
            Item::Variable(var_decl) => {
                self.analyze_variable(item.id, var_decl, item.span)?;
            }
            Item::Const(const_decl) => {
                self.analyze_constant(item.id, const_decl, item.span)?;
            }
            Item::Import(import_decl) => {
                self.analyze_import(item.id, import_decl, item.span)?;
            }
            Item::Export(export_decl) => {
                self.analyze_export(item.id, export_decl, item.span)?;
            }
            Item::Statement(stmt) => {
                // Handle statement items (statements at module level)
                // Create a temporary AstNode wrapper for the statement
                let stmt_node = AstNode {
                    kind: stmt.clone(),
                    span: item.span,
                    id: item.id,
                    metadata: Default::default(),
                };
                self.analyze_statement(&stmt_node)?;
            }
        }

        self.context.depth -= 1;
        Ok(())
    }

    /// Analyze a function declaration
    fn analyze_function(&mut self, node_id: NodeId, func_decl: &prism_ast::FunctionDecl, span: Span) -> SemanticResult<()> {
        let old_function = self.context.current_function.clone();
        self.context.current_function = Some(func_decl.name.to_string());

        // Create symbol info for function
        let symbol_info = SymbolInfo {
            id: func_decl.name,
            name: func_decl.name.to_string(),
            symbol_type: SymbolType::Function,
            location: span,
            visibility: self.determine_visibility(&func_decl.visibility),
            semantic_annotations: Vec::new(), // Would extract from attributes
            business_context: self.infer_business_context(&func_decl.name.to_string()),
            ai_hints: self.generate_function_ai_hints(&func_decl.name.to_string()),
        };

        self.symbols.insert(func_decl.name, symbol_info);

        // Create type info for function
        let type_info = TypeInfo {
            type_id: node_id,
            type_kind: TypeKind::Function,
            location: span,
            semantic_meaning: self.infer_function_semantic_meaning(&func_decl.name.to_string()),
            domain: self.infer_domain(&func_decl.name.to_string()),
            ai_description: Some(format!("Function: {}", func_decl.name)),
        };

        self.types.insert(node_id, type_info);

        // Analyze function body if present
        if let Some(body) = &func_decl.body {
            if let Stmt::Block(block) = &body.kind {
                for stmt in &block.statements {
                    self.analyze_statement(stmt)?;
                }
            } else {
                // Single statement
                self.analyze_statement(body)?;
            }
        }

        self.context.current_function = old_function;
        Ok(())
    }

    /// Analyze a type declaration
    fn analyze_type_declaration(&mut self, node_id: NodeId, type_decl: &prism_ast::TypeDecl, span: Span) -> SemanticResult<()> {
        let old_type = self.context.current_type.clone();
        self.context.current_type = Some(type_decl.name.to_string());

        // Create symbol info for type
        let symbol_info = SymbolInfo {
            id: type_decl.name,
            name: type_decl.name.to_string(),
            symbol_type: SymbolType::Type,
            location: span,
            visibility: self.determine_visibility(&type_decl.visibility),
            semantic_annotations: Vec::new(), // Would extract from attributes
            business_context: self.infer_business_context(&type_decl.name.to_string()),
            ai_hints: self.generate_type_ai_hints(&type_decl.name.to_string()),
        };

        self.symbols.insert(type_decl.name, symbol_info);

        // Create type info
        let type_kind = match &type_decl.kind {
            prism_ast::TypeKind::Semantic(_) => TypeKind::Semantic(type_decl.name.to_string()),
            prism_ast::TypeKind::Struct(_) => TypeKind::Composite("struct".to_string()),
            prism_ast::TypeKind::Enum(_) => TypeKind::Composite("enum".to_string()),
            prism_ast::TypeKind::Alias(_) => TypeKind::Primitive("alias".to_string()),
            prism_ast::TypeKind::Trait(_) => TypeKind::Composite("trait".to_string()),
        };

        let type_info = TypeInfo {
            type_id: node_id,
            type_kind,
            location: span,
            semantic_meaning: self.infer_type_semantic_meaning(&type_decl.name.to_string()),
            domain: self.infer_domain(&type_decl.name.to_string()),
            ai_description: Some(format!("Type: {}", type_decl.name)),
        };

        self.types.insert(node_id, type_info);

        self.context.current_type = old_type;
        Ok(())
    }

    /// Analyze a section declaration
    fn analyze_section(&mut self, section: &AstNode<prism_ast::SectionDecl>) -> SemanticResult<()> {
        // For now, just analyze the statements in the section
        // In a full implementation, we'd track section-specific context
        for stmt in &section.kind.items {
            self.analyze_statement(stmt)?;
        }
        Ok(())
    }

    /// Analyze a module declaration
    fn analyze_module(&mut self, node_id: NodeId, module_decl: &prism_ast::ModuleDecl, span: Span) -> SemanticResult<()> {
        let old_module = self.context.current_module.clone();
        self.context.current_module = Some(module_decl.name.to_string());

        // Create symbol info for module
        let symbol_info = SymbolInfo {
            id: module_decl.name,
            name: module_decl.name.to_string(),
            symbol_type: SymbolType::Module,
            location: span,
            visibility: Visibility::Public, // Modules are typically public
            semantic_annotations: Vec::new(),
            business_context: self.infer_business_context(&module_decl.name.to_string()),
            ai_hints: self.generate_module_ai_hints(&module_decl.name.to_string()),
        };

        self.symbols.insert(module_decl.name, symbol_info);

        // Analyze module items
        for section in &module_decl.sections {
            self.analyze_section(section)?;
        }

        self.context.current_module = old_module;
        Ok(())
    }

    /// Analyze a variable declaration
    fn analyze_variable(&mut self, _node_id: NodeId, var_decl: &prism_ast::VariableDecl, span: Span) -> SemanticResult<()> {
        let symbol_info = SymbolInfo {
            id: var_decl.name,
            name: var_decl.name.to_string(),
            symbol_type: SymbolType::Variable,
            location: span,
            visibility: self.determine_visibility(&var_decl.visibility),
            semantic_annotations: Vec::new(),
            business_context: self.infer_business_context(&var_decl.name.to_string()),
            ai_hints: self.generate_variable_ai_hints(&var_decl.name.to_string()),
        };

        self.symbols.insert(var_decl.name, symbol_info);
        Ok(())
    }

    /// Analyze a constant declaration
    fn analyze_constant(&mut self, node_id: NodeId, const_decl: &prism_ast::ConstDecl, span: Span) -> SemanticResult<()> {
        let symbol_info = SymbolInfo {
            id: const_decl.name,
            name: const_decl.name.to_string(),
            symbol_type: SymbolType::Constant,
            location: span,
            visibility: self.determine_visibility(&const_decl.visibility),
            semantic_annotations: Vec::new(),
            business_context: self.infer_business_context(&const_decl.name.to_string()),
            ai_hints: self.generate_constant_ai_hints(&const_decl.name.to_string()),
        };

        self.symbols.insert(const_decl.name, symbol_info);
        Ok(())
    }

    /// Analyze an import declaration
    fn analyze_import(&mut self, _node_id: NodeId, _import_decl: &prism_ast::ImportDecl, _span: Span) -> SemanticResult<()> {
        // Import analysis would go here
        Ok(())
    }

    /// Analyze an export declaration
    fn analyze_export(&mut self, _node_id: NodeId, _export_decl: &prism_ast::ExportDecl, _span: Span) -> SemanticResult<()> {
        // Export analysis would go here
        Ok(())
    }

    /// Analyze a statement
    fn analyze_statement(&mut self, stmt: &AstNode<Stmt>) -> SemanticResult<()> {
        match &stmt.kind {
            Stmt::Expression(expr_stmt) => {
                self.analyze_expression(&expr_stmt.expression)?;
            }
            Stmt::Variable(var_decl) => {
                // Handle statement-level variable declarations
                let symbol_info = SymbolInfo {
                    id: var_decl.name,
                    name: var_decl.name.to_string(),
                    symbol_type: SymbolType::Variable,
                    location: stmt.span,
                    visibility: Visibility::Private, // Statement variables are private
                    semantic_annotations: Vec::new(),
                    business_context: self.infer_business_context(&var_decl.name.to_string()),
                    ai_hints: self.generate_variable_ai_hints(&var_decl.name.to_string()),
                };
                self.symbols.insert(var_decl.name, symbol_info);
            }
            Stmt::Return(return_stmt) => {
                if let Some(expr) = &return_stmt.value {
                    self.analyze_expression(expr)?;
                }
            }
            Stmt::If(if_stmt) => {
                self.analyze_expression(&if_stmt.condition)?;
                if let Stmt::Block(block) = &if_stmt.then_branch.kind {
                    for stmt in &block.statements {
                        self.analyze_statement(stmt)?;
                    }
                } else {
                    self.analyze_statement(&if_stmt.then_branch)?;
                }
                if let Some(else_body) = &if_stmt.else_branch {
                    if let Stmt::Block(block) = &else_body.kind {
                        for stmt in &block.statements {
                            self.analyze_statement(stmt)?;
                        }
                    } else {
                        self.analyze_statement(else_body)?;
                    }
                }
            }
            Stmt::While(while_stmt) => {
                self.analyze_expression(&while_stmt.condition)?;
                if let Stmt::Block(block) = &while_stmt.body.kind {
                    for stmt in &block.statements {
                        self.analyze_statement(stmt)?;
                    }
                } else {
                    self.analyze_statement(&while_stmt.body)?;
                }
            }
            Stmt::For(for_stmt) => {
                // Analyze the iterable expression
                self.analyze_expression(&for_stmt.iterable)?;
                
                // Analyze the body
                if let Stmt::Block(block) = &for_stmt.body.kind {
                    for stmt in &block.statements {
                        self.analyze_statement(stmt)?;
                    }
                } else {
                    self.analyze_statement(&for_stmt.body)?;
                }
            }
            Stmt::Block(block_stmt) => {
                for stmt in &block_stmt.statements {
                    self.analyze_statement(stmt)?;
                }
            }
            // For now, handle other statement types with a catch-all
            // In a full implementation, each would have specific analysis
            _ => {
                // TODO: Implement analysis for other statement types
                // (Function, Type, Module, Section, Import, Export, Match, Try, etc.)
            }
        }
        Ok(())
    }

    /// Analyze an expression
    fn analyze_expression(&mut self, expr: &AstNode<Expr>) -> SemanticResult<()> {
        match &expr.kind {
            Expr::Literal(_) => {
                // Literal expressions don't need deep analysis
            }
            Expr::Variable(var_expr) => {
                // Track identifier usage
                // This would update usage statistics
            }
            Expr::Binary(binary_expr) => {
                self.analyze_expression(&binary_expr.left)?;
                self.analyze_expression(&binary_expr.right)?;
            }
            Expr::Unary(unary_expr) => {
                self.analyze_expression(&unary_expr.operand)?;
            }
            Expr::Call(call_expr) => {
                self.analyze_expression(&call_expr.callee)?;
                for arg in &call_expr.arguments {
                    self.analyze_expression(arg)?;
                }
            }
            Expr::Member(member_expr) => {
                self.analyze_expression(&member_expr.object)?;
            }
            Expr::Index(index_expr) => {
                self.analyze_expression(&index_expr.object)?;
                self.analyze_expression(&index_expr.index)?;
            }
            Expr::Array(array_expr) => {
                for element in &array_expr.elements {
                    self.analyze_expression(element)?;
                }
            }
            Expr::Object(object_expr) => {
                for field in &object_expr.fields {
                    self.analyze_expression(&field.value)?;
                }
            }
            // For now, handle other expression types with a catch-all
            // In a full implementation, each would have specific analysis
            _ => {
                // TODO: Implement analysis for other expression types
                // (Lambda, Match, If, While, For, Try, Range, TypeAssertion, etc.)
            }
        }
        Ok(())
    }

    // Helper methods for semantic inference

    fn determine_visibility(&self, visibility: &prism_ast::Visibility) -> Visibility {
        match visibility {
            prism_ast::Visibility::Public => Visibility::Public,
            prism_ast::Visibility::Private => Visibility::Private,
            prism_ast::Visibility::Internal => Visibility::Internal,
        }
    }


    fn infer_business_context(&self, name: &str) -> Option<String> {
        // Simple business context inference based on naming patterns
        let lower_name = name.to_lowercase();
        if lower_name.contains("user") {
            Some("User Management".to_string())
        } else if lower_name.contains("payment") || lower_name.contains("money") {
            Some("Financial Operations".to_string())
        } else if lower_name.contains("auth") || lower_name.contains("login") {
            Some("Authentication & Security".to_string())
        } else if lower_name.contains("data") || lower_name.contains("store") {
            Some("Data Management".to_string())
        } else {
            None
        }
    }

    fn infer_domain(&self, name: &str) -> Option<String> {
        // Domain inference based on naming patterns
        let lower_name = name.to_lowercase();
        if lower_name.contains("web") || lower_name.contains("http") {
            Some("Web Services".to_string())
        } else if lower_name.contains("db") || lower_name.contains("database") {
            Some("Database".to_string())
        } else if lower_name.contains("ui") || lower_name.contains("interface") {
            Some("User Interface".to_string())
        } else {
            Some("Business Logic".to_string())
        }
    }

    fn infer_function_semantic_meaning(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        if lower_name.starts_with("get") {
            Some("Data retrieval operation".to_string())
        } else if lower_name.starts_with("set") || lower_name.starts_with("update") {
            Some("Data modification operation".to_string())
        } else if lower_name.starts_with("create") || lower_name.starts_with("new") {
            Some("Resource creation operation".to_string())
        } else if lower_name.starts_with("delete") || lower_name.starts_with("remove") {
            Some("Resource deletion operation".to_string())
        } else if lower_name.starts_with("validate") || lower_name.starts_with("check") {
            Some("Validation operation".to_string())
        } else {
            Some("Business logic operation".to_string())
        }
    }

    fn infer_type_semantic_meaning(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        if lower_name.ends_with("id") {
            Some("Unique identifier type".to_string())
        } else if lower_name.contains("config") {
            Some("Configuration data type".to_string())
        } else if lower_name.contains("result") || lower_name.contains("response") {
            Some("Operation result type".to_string())
        } else if lower_name.contains("request") {
            Some("Operation input type".to_string())
        } else {
            Some("Business entity type".to_string())
        }
    }

    fn generate_function_ai_hints(&self, name: &str) -> Vec<String> {
        let mut hints = Vec::new();
        let lower_name = name.to_lowercase();
        
        if lower_name.starts_with("get") {
            hints.push("This function retrieves data and should be side-effect free".to_string());
        }
        if lower_name.contains("async") || lower_name.contains("await") {
            hints.push("This function performs asynchronous operations".to_string());
        }
        if lower_name.contains("validate") {
            hints.push("This function performs data validation and may return errors".to_string());
        }
        
        hints
    }

    fn generate_type_ai_hints(&self, name: &str) -> Vec<String> {
        let mut hints = Vec::new();
        let lower_name = name.to_lowercase();
        
        if lower_name.ends_with("id") {
            hints.push("This type represents a unique identifier and should be treated as opaque".to_string());
        }
        if lower_name.contains("money") || lower_name.contains("price") {
            hints.push("This type represents monetary values and requires careful handling".to_string());
        }
        if lower_name.contains("email") {
            hints.push("This type represents email addresses and should be validated".to_string());
        }
        
        hints
    }

    fn generate_module_ai_hints(&self, name: &str) -> Vec<String> {
        let mut hints = Vec::new();
        hints.push(format!("Module '{}' provides cohesive functionality", name));
        hints.push("This module follows Prism's Conceptual Cohesion principle".to_string());
        hints
    }

    fn generate_variable_ai_hints(&self, name: &str) -> Vec<String> {
        let mut hints = Vec::new();
        let lower_name = name.to_lowercase();
        
        if lower_name.contains("temp") || lower_name.contains("tmp") {
            hints.push("This appears to be a temporary variable".to_string());
        }
        if lower_name.contains("config") {
            hints.push("This variable holds configuration data".to_string());
        }
        
        hints
    }

    fn generate_constant_ai_hints(&self, name: &str) -> Vec<String> {
        let mut hints = Vec::new();
        hints.push("This is a compile-time constant value".to_string());
        if name.chars().all(|c| c.is_uppercase() || c == '_') {
            hints.push("Follows standard constant naming convention".to_string());
        }
        hints
    }

    /// Extract business rules from analysis results
    pub fn extract_business_rules(&self, analysis: &AnalysisResult) -> Vec<BusinessRule> {
        let mut rules = Vec::new();

        // Extract business rules from symbols
        for (symbol, symbol_info) in &analysis.symbols {
            // Business context rules
            if let Some(ref business_context) = symbol_info.business_context {
                rules.push(BusinessRule {
                    rule_name: format!("{} business rule", symbol_info.name),
                    rule_type: "business_context".to_string(),
                    confidence: 0.85,
                    description: format!("Business context: {}", business_context),
                    evidence: vec![format!("Symbol {} has business context", symbol_info.name)],
                    location: Some(symbol_info.location),
                });
            }

            // Validation rules from function names
            if symbol_info.name.contains("validate") || symbol_info.name.contains("check") {
                rules.push(BusinessRule {
                    rule_name: format!("{} validation rule", symbol_info.name),
                    rule_type: "validation".to_string(),
                    confidence: 0.8,
                    description: format!("Function {} performs validation", symbol_info.name),
                    evidence: vec!["Function name suggests validation behavior".to_string()],
                    location: Some(symbol_info.location),
                });
            }

            // Security rules from capability usage
            if !symbol_info.ai_hints.is_empty() {
                let security_hints: Vec<_> = symbol_info.ai_hints.iter()
                    .filter(|hint| hint.contains("security") || hint.contains("permission") || hint.contains("auth"))
                    .collect();
                
                if !security_hints.is_empty() {
                    rules.push(BusinessRule {
                        rule_name: format!("{} security rule", symbol_info.name),
                        rule_type: "security".to_string(),
                        confidence: 0.9,
                        description: format!("Symbol {} has security implications", symbol_info.name),
                        evidence: security_hints.iter().map(|s| s.to_string()).collect(),
                        location: Some(symbol_info.location),
                    });
                }
            }
        }

        // Extract business rules from types
        for (type_id, type_info) in &analysis.types {
            // Configuration rules
            if type_info.domain.as_ref().map_or(false, |d| d.contains("config")) {
                rules.push(BusinessRule {
                    rule_name: "Configuration type rule".to_string(),
                    rule_type: "configuration".to_string(),
                    confidence: 0.75,
                    description: "Type used for configuration management".to_string(),
                    evidence: vec![format!("Domain: {:?}", type_info.domain)],
                    location: Some(type_info.location),
                });
            }

            // Data validation rules from AI descriptions
            if let Some(ref ai_description) = type_info.ai_description {
                if ai_description.contains("constraint") || ai_description.contains("validation") {
                    rules.push(BusinessRule {
                        rule_name: "Data constraint rule".to_string(),
                        rule_type: "data_constraint".to_string(),
                        confidence: 0.8,
                        description: format!("Type has data constraints: {}", ai_description),
                        evidence: vec![ai_description.clone()],
                        location: Some(type_info.location),
                    });
                }
            }
        }

        // Extract module-level business rules
        let symbol_count = analysis.symbols.len();
        let business_context_count = analysis.symbols.values()
            .filter(|s| s.business_context.is_some())
            .count();

        if symbol_count > 0 {
            let business_context_ratio = business_context_count as f64 / symbol_count as f64;
            
            if business_context_ratio > 0.5 {
                rules.push(BusinessRule {
                    rule_name: "Domain-driven design rule".to_string(),
                    rule_type: "architecture".to_string(),
                    confidence: 0.7 + business_context_ratio * 0.3,
                    description: format!("Module follows domain-driven design with {}% business context coverage", 
                                       (business_context_ratio * 100.0) as u32),
                    evidence: vec![
                        format!("{} out of {} symbols have business context", business_context_count, symbol_count)
                    ],
                    location: None,
                });
            }
        }

        // Sort by confidence
        rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        rules
    }

    /// Extract semantic relationships from analysis
    pub fn extract_semantic_relationships(&self, analysis: &AnalysisResult) -> Vec<SemanticRelationship> {
        let mut relationships = Vec::new();

        // Symbol-to-type relationships
        for (symbol, symbol_info) in &analysis.symbols {
            for (type_id, type_info) in &analysis.types {
                // Check if symbol and type are related by location proximity
                if self.are_locations_related(&symbol_info.location, &type_info.location) {
                    relationships.push(SemanticRelationship {
                        source: symbol_info.name.clone(),
                        target: type_info.ai_description.clone().unwrap_or_else(|| "Unknown type".to_string()),
                        relationship_type: "uses_type".to_string(),
                        strength: 0.8,
                        evidence: vec!["Location proximity suggests type usage".to_string()],
                        location: Some(symbol_info.location),
                    });
                }
            }

            // Business context relationships
            if let Some(ref business_context) = symbol_info.business_context {
                relationships.push(SemanticRelationship {
                    source: symbol_info.name.clone(),
                    target: business_context.clone(),
                    relationship_type: "implements_business_rule".to_string(),
                    strength: 0.9,
                    evidence: vec!["Symbol has explicit business context".to_string()],
                    location: Some(symbol_info.location),
                });
            }
        }

        // Module cohesion relationships
        let symbols_with_business_context: Vec<_> = analysis.symbols.values()
            .filter(|s| s.business_context.is_some())
            .collect();

        if symbols_with_business_context.len() >= 2 {
            for i in 0..symbols_with_business_context.len() {
                for j in (i + 1)..symbols_with_business_context.len() {
                    let symbol_a = symbols_with_business_context[i];
                    let symbol_b = symbols_with_business_context[j];
                    
                    relationships.push(SemanticRelationship {
                        source: symbol_a.name.clone(),
                        target: symbol_b.name.clone(),
                        relationship_type: "cohesive_with".to_string(),
                        strength: 0.6,
                        evidence: vec!["Both symbols have business context in same module".to_string()],
                        location: Some(symbol_a.location),
                    });
                }
            }
        }

        relationships
    }

    /// Check if two locations are semantically related
    fn are_locations_related(&self, loc1: &Span, loc2: &Span) -> bool {
        // Simple heuristic: locations are related if they're within 50 lines of each other
        let line_diff = if loc1.start.line > loc2.start.line {
            loc1.start.line - loc2.start.line
        } else {
            loc2.start.line - loc1.start.line
        };
        
        line_diff <= 50
    }
} 