//! PIR Builder - AST to PIR Transformation Pipeline
//!
//! This module implements the transformation from semantic AST to PIR,
//! focusing on business logic organization rather than technical details.

use crate::{
    PIRError, PIRResult,
    semantic::*,
    business::*,
    ai_integration::*,
    quality::*,
};
use prism_ast::{Program, Item, ModuleDecl, FunctionDecl, TypeDecl, ConstDecl, Expr, LiteralValue, BinaryOperator, Stmt, Type, TypeKind};
use prism_common::symbol::Symbol;
use std::collections::HashMap;
use chrono::Utc;
use tracing::{info, warn, debug};

/// PIR Builder configuration
#[derive(Debug, Clone)]
pub struct PIRBuilderConfig {
    /// Enable semantic validation during build
    pub enable_validation: bool,
    /// Enable AI metadata extraction
    pub enable_ai_metadata: bool,
    /// Enable cohesion analysis
    pub enable_cohesion_analysis: bool,
    /// Enable effect graph construction
    pub enable_effect_graph: bool,
    /// Enable business context extraction
    pub enable_business_context: bool,
    /// Maximum build depth for recursive structures
    pub max_build_depth: usize,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
}

impl Default for PIRBuilderConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_ai_metadata: true,
            enable_cohesion_analysis: true,
            enable_effect_graph: true,
            enable_business_context: true,
            max_build_depth: 1000,
            enable_performance_profiling: false,
        }
    }
}

/// PIR Builder - Complete AST to PIR transformation
#[derive(Debug)]
pub struct PIRBuilder {
    /// Configuration for PIR generation
    pub config: PIRBuilderConfig,
    /// Current build depth (for recursion protection)
    current_depth: usize,
}

impl PIRBuilder {
    /// Create a new PIR builder with default configuration
    pub fn new() -> Self {
        Self::with_config(PIRBuilderConfig::default())
    }

    /// Create a new PIR builder with custom configuration
    pub fn with_config(config: PIRBuilderConfig) -> Self {
        Self {
            config,
            current_depth: 0,
        }
    }

    /// Build PIR from an AST program
    pub fn build_from_program(&mut self, program: &Program) -> PIRResult<PrismIR> {
        info!("Starting PIR generation from AST program");

        // Initialize PIR
        let mut pir = PrismIR::new();

        // Extract modules and global items
        let mut modules = Vec::new();
        let mut global_types = Vec::new();
        let mut global_functions = Vec::new();
        let mut global_constants = Vec::new();

        // Process all top-level items
        for item in &program.items {
            match &item.kind {
                Item::Module(module_decl) => {
                    let pir_module = self.build_module(module_decl)?;
                    modules.push(pir_module);
                }
                Item::Function(func_decl) => {
                    let pir_function = self.build_function(func_decl)?;
                    global_functions.push(pir_function);
                }
                Item::Type(type_decl) => {
                    let pir_type = self.build_semantic_type(type_decl)?;
                    global_types.push(pir_type);
                }
                Item::Const(const_decl) => {
                    let pir_constant = self.build_constant(const_decl)?;
                    global_constants.push(pir_constant);
                }
                _ => {
                    debug!("Skipping unsupported item kind");
                }
            }
        }

        // Create a global module if we have global items
        if !global_types.is_empty() || !global_functions.is_empty() || !global_constants.is_empty() {
            let global_module = self.create_global_module(global_types, global_functions, global_constants)?;
            modules.insert(0, global_module);
        }

        pir.modules = modules;

        // Build additional components if enabled
        if self.config.enable_ai_metadata {
            pir.ai_metadata = self.extract_ai_metadata(&pir)?;
        }

        if self.config.enable_cohesion_analysis {
            pir.cohesion_metrics = self.analyze_cohesion(&pir)?;
        }

        if self.config.enable_effect_graph {
            pir.effect_graph = self.build_effect_graph(&pir)?;
        }

        // Update PIR metadata
        pir.metadata = PIRMetadata {
            version: crate::PIRVersion::CURRENT.to_string(),
            created_at: Utc::now().to_rfc3339(),
            source_hash: self.calculate_source_hash(program),
            optimization_level: 0,
            target_platforms: Vec::new(),
        };

        info!("PIR generation completed successfully");
        Ok(pir)
    }

    /// Build a PIR module from an AST module declaration
    fn build_module(&mut self, module_decl: &ModuleDecl) -> PIRResult<PIRModule> {
        debug!("Building PIR module: {}", module_decl.name.to_string());

        let name = module_decl.name.to_string();
        let capability = module_decl.capability.clone().unwrap_or_else(|| "general".to_string());

        // Build sections from module sections
        let mut sections = Vec::new();
        for section_node in &module_decl.sections {
            let section = self.build_section(&section_node.kind)?;
            sections.push(section);
        }

        // Extract business context
        let business_context = if self.config.enable_business_context {
            self.extract_business_context(module_decl)?
        } else {
            BusinessContext::new("default".to_string())
        };

        // Create performance profile
        let performance_profile = if self.config.enable_performance_profiling {
            self.create_performance_profile(&sections)?
        } else {
            PerformanceProfile::default()
        };

        Ok(PIRModule {
            name,
            capability,
            sections,
            dependencies: Vec::new(), // TODO: Extract from module
            business_context,
            domain_rules: Vec::new(), // TODO: Extract domain rules
            effects: Vec::new(),      // TODO: Extract effects
            capabilities: Vec::new(), // TODO: Extract capabilities
            performance_profile,
            cohesion_score: 0.8, // TODO: Calculate actual cohesion
        })
    }

    /// Build a PIR section from an AST section
    fn build_section(&mut self, section: &prism_ast::SectionDecl) -> PIRResult<PIRSection> {
        use prism_ast::SectionKind;
        
        match section.kind {
            SectionKind::Types => {
                let mut types = Vec::new();
                for item_node in &section.items {
                    if let prism_ast::Stmt::Type(type_decl) = &item_node.kind {
                        let pir_type = self.build_semantic_type(type_decl)?;
                        types.push(pir_type);
                    }
                }
                Ok(PIRSection::Types(TypeSection { types }))
            }
            SectionKind::Interface => {
                let mut interfaces = Vec::new();
                // TODO: Build interfaces from items
                Ok(PIRSection::Interface(InterfaceSection { interfaces }))
            }
            _ => {
                // For now, create an empty implementation section
                Ok(PIRSection::Implementation(ImplementationSection { items: Vec::new() }))
            }
        }
    }

    /// Build a semantic type from an AST type declaration
    fn build_semantic_type(&mut self, type_decl: &TypeDecl) -> PIRResult<PIRSemanticType> {
        let name = type_decl.name.to_string();
        
        // Convert the type kind to PIR type info
        let base_type = self.convert_type_kind_to_pir(&type_decl.kind)?;
        
        // Extract domain from type name or use default
        let domain = self.extract_domain_from_name(&name);

        Ok(PIRSemanticType {
            name,
            base_type,
            domain,
            business_rules: Vec::new(), // TODO: Extract from attributes
            validation_predicates: Vec::new(), // TODO: Extract validation
            constraints: Vec::new(), // TODO: Extract constraints
            ai_context: PIRTypeAIContext {
                intent: None,
                examples: Vec::new(),
                common_mistakes: Vec::new(),
                best_practices: Vec::new(),
            },
            security_classification: SecurityClassification::Internal,
        })
    }

    /// Convert AST type kind to PIR type info
    fn convert_type_kind_to_pir(&mut self, type_kind: &prism_ast::TypeKind) -> PIRResult<PIRTypeInfo> {
        use prism_ast::TypeKind;
        
        match type_kind {
            TypeKind::Alias(type_node) => {
                // For aliases, convert the underlying type
                self.convert_ast_type_to_pir(&type_node.kind)
            }
            TypeKind::Struct(struct_type) => {
                let mut fields = Vec::new();
                for field in &struct_type.fields {
                    let pir_field = PIRField {
                        name: field.name.resolve().unwrap_or_else(|| "unknown".to_string()),
                        field_type: self.convert_ast_type_to_pir(&field.field_type.kind)?,
                        visibility: self.convert_visibility(&field.visibility),
                        business_meaning: None,
                        validation_rules: Vec::new(),
                    };
                    fields.push(pir_field);
                }

                Ok(PIRTypeInfo::Composite(PIRCompositeType {
                    kind: PIRCompositeKind::Struct,
                    fields,
                    methods: Vec::new(),
                }))
            }
            TypeKind::Enum(enum_type) => {
                // For now, create a simple enum representation
                Ok(PIRTypeInfo::Composite(PIRCompositeType {
                    kind: PIRCompositeKind::Enum,
                    fields: Vec::new(), // TODO: Convert enum variants to fields
                    methods: Vec::new(),
                }))
            }
            _ => {
                // Default to a simple string type for unsupported kinds
                Ok(PIRTypeInfo::Primitive(PIRPrimitiveType::String))
            }
        }
    }

    /// Convert AST type to PIR type info
    fn convert_ast_type_to_pir(&mut self, ast_type: &prism_ast::Type) -> PIRResult<PIRTypeInfo> {
        use prism_ast::{Type, PrimitiveType};
        
        match ast_type {
            Type::Primitive(prim) => {
                let pir_prim = match prim {
                    PrimitiveType::Boolean => PIRPrimitiveType::Boolean,
                    PrimitiveType::String => PIRPrimitiveType::String,
                    PrimitiveType::Integer(int_type) => {
                        use prism_ast::IntegerType;
                        match int_type {
                            IntegerType::Signed(width) => PIRPrimitiveType::Integer { signed: true, width: *width },
                            IntegerType::Unsigned(width) => PIRPrimitiveType::Integer { signed: false, width: *width },
                            _ => PIRPrimitiveType::Integer { signed: true, width: 32 },
                        }
                    }
                    PrimitiveType::Float(float_type) => {
                        use prism_ast::FloatType;
                        match float_type {
                            FloatType::F32 => PIRPrimitiveType::Float { width: 32 },
                            FloatType::F64 => PIRPrimitiveType::Float { width: 64 },
                            _ => PIRPrimitiveType::Float { width: 64 },
                        }
                    }
                    _ => PIRPrimitiveType::String, // Default fallback
                };
                Ok(PIRTypeInfo::Primitive(pir_prim))
            }
            Type::Named(named) => {
                // For named types, create a reference (simplified)
                Ok(PIRTypeInfo::Primitive(PIRPrimitiveType::String)) // Placeholder
            }
            Type::Tuple(tuple) => {
                Ok(PIRTypeInfo::Composite(PIRCompositeType {
                    kind: PIRCompositeKind::Tuple,
                    fields: Vec::new(), // TODO: Convert tuple elements
                    methods: Vec::new(),
                }))
            }
            _ => {
                // Default fallback
                Ok(PIRTypeInfo::Primitive(PIRPrimitiveType::String))
            }
        }
    }

    /// Build a PIR function from an AST function declaration
    fn build_function(&mut self, func_decl: &FunctionDecl) -> PIRResult<PIRFunction> {
        let name = func_decl.name.resolve().unwrap_or_else(|| "unknown".to_string());
        
        // Build function signature
        let mut parameters = Vec::new();
        for param in &func_decl.parameters {
            let pir_param = PIRParameter {
                name: param.name.resolve().unwrap_or_else(|| "param".to_string()),
                param_type: if let Some(type_ann) = &param.type_annotation {
                    self.convert_ast_type_to_pir(&type_ann.kind)?
                } else {
                    PIRTypeInfo::Primitive(PIRPrimitiveType::String) // Default
                },
                default_value: None, // TODO: Convert default value
                business_meaning: None,
            };
            parameters.push(pir_param);
        }

        let return_type = if let Some(ret_type) = &func_decl.return_type {
            Box::new(self.convert_ast_type_to_pir(&ret_type.kind)?)
        } else {
            Box::new(PIRTypeInfo::Primitive(PIRPrimitiveType::Unit))
        };

        let signature = PIRFunctionType {
            parameters,
            return_type,
            effects: EffectSignature {
                input_effects: Vec::new(),
                output_effects: Vec::new(),
                effect_dependencies: Vec::new(),
            },
            contracts: PIRPerformanceContract {
                preconditions: Vec::new(),
                postconditions: Vec::new(),
                performance_guarantees: Vec::new(),
            },
        };

        // Convert function body
        let body = if let Some(body_stmt) = &func_decl.body {
            self.convert_ast_stmt_to_pir_expr(&body_stmt.kind)?
        } else {
            PIRExpression::Literal(PIRLiteral::Unit)
        };

        Ok(PIRFunction {
            name,
            signature,
            body,
            responsibility: None,
            algorithm: None,
            complexity: None,
            capabilities_required: Vec::new(),
            performance_characteristics: Vec::new(),
            ai_hints: Vec::new(),
        })
    }

    /// Build a PIR constant from an AST constant declaration
    fn build_constant(&mut self, const_decl: &ConstDecl) -> PIRResult<PIRConstant> {
        let name = const_decl.name.resolve().unwrap_or_else(|| "unknown".to_string());
        
        let const_type = if let Some(type_ann) = &const_decl.type_annotation {
            self.convert_ast_type_to_pir(&type_ann.kind)?
        } else {
            PIRTypeInfo::Primitive(PIRPrimitiveType::String)
        };

        let value = self.convert_ast_expr_to_pir(&const_decl.value.kind)?;

        Ok(PIRConstant {
            name,
            const_type,
            value,
            business_meaning: None,
        })
    }

    /// Convert AST expression to PIR expression
    fn convert_ast_expr_to_pir(&mut self, expr: &Expr) -> PIRResult<PIRExpression> {
        use prism_ast::{Expr, LiteralExpr};
        
        match expr {
            Expr::Literal(lit_expr) => {
                let pir_literal = self.convert_literal_value(&lit_expr.value)?;
                Ok(PIRExpression::Literal(pir_literal))
            }
            Expr::Variable(var_expr) => {
                let name = var_expr.name.resolve().unwrap_or_else(|| "unknown".to_string());
                Ok(PIRExpression::Variable(name))
            }
            Expr::Binary(bin_expr) => {
                let left = Box::new(self.convert_ast_expr_to_pir(&bin_expr.left.kind)?);
                let right = Box::new(self.convert_ast_expr_to_pir(&bin_expr.right.kind)?);
                let operator = self.convert_binary_operator(&bin_expr.operator);
                
                Ok(PIRExpression::Binary {
                    left,
                    operator,
                    right,
                })
            }
            Expr::Block(block_expr) => {
                let mut statements = Vec::new();
                for stmt_node in &block_expr.statements {
                    let pir_stmt = self.convert_ast_stmt_to_pir(&stmt_node.kind)?;
                    statements.push(pir_stmt);
                }

                let result = if let Some(result_expr) = &block_expr.final_expr {
                    Some(Box::new(self.convert_ast_expr_to_pir(&result_expr.kind)?))
                } else {
                    None
                };

                Ok(PIRExpression::Block {
                    statements,
                    result,
                })
            }
            _ => {
                // Default fallback for unsupported expressions
                Ok(PIRExpression::Literal(PIRLiteral::Unit))
            }
        }
    }

    /// Convert AST statement to PIR expression (for function bodies)
    fn convert_ast_stmt_to_pir_expr(&mut self, stmt: &prism_ast::Stmt) -> PIRResult<PIRExpression> {
        // For simplicity, convert statements to block expressions
        Ok(PIRExpression::Block {
            statements: vec![self.convert_ast_stmt_to_pir(stmt)?],
            result: None,
        })
    }

    /// Convert AST statement to PIR statement
    fn convert_ast_stmt_to_pir(&mut self, stmt: &prism_ast::Stmt) -> PIRResult<PIRStatement> {
        use prism_ast::Stmt;
        
        match stmt {
            Stmt::Expression(expr_stmt) => {
                let expr = self.convert_ast_expr_to_pir(&expr_stmt.expression.kind)?;
                Ok(PIRStatement::Expression(expr))
            }
            Stmt::Return(ret_stmt) => {
                let value = if let Some(val) = &ret_stmt.value {
                    Some(self.convert_ast_expr_to_pir(&val.kind)?)
                } else {
                    None
                };
                Ok(PIRStatement::Return(value))
            }
            _ => {
                // Default fallback
                Ok(PIRStatement::Expression(PIRExpression::Literal(PIRLiteral::Unit)))
            }
        }
    }

    /// Convert literal value
    fn convert_literal_value(&self, value: &LiteralValue) -> PIRResult<PIRLiteral> {
        match value {
            LiteralValue::Integer(i) => Ok(PIRLiteral::Integer(*i)),
            LiteralValue::Float(f) => Ok(PIRLiteral::Float(*f)),
            LiteralValue::String(s) => Ok(PIRLiteral::String(s.clone())),
            LiteralValue::Boolean(b) => Ok(PIRLiteral::Boolean(*b)),
            LiteralValue::Null => Ok(PIRLiteral::Unit),
            _ => Ok(PIRLiteral::String("unsupported".to_string())),
        }
    }

    /// Convert binary operator
    fn convert_binary_operator(&self, op: &BinaryOperator) -> PIRBinaryOp {
        match op {
            BinaryOperator::Add => PIRBinaryOp::Add,
            BinaryOperator::Subtract => PIRBinaryOp::Subtract,
            BinaryOperator::Multiply => PIRBinaryOp::Multiply,
            BinaryOperator::Divide => PIRBinaryOp::Divide,
            BinaryOperator::Modulo => PIRBinaryOp::Modulo,
            BinaryOperator::Equal => PIRBinaryOp::Equal,
            BinaryOperator::NotEqual => PIRBinaryOp::NotEqual,
            BinaryOperator::Less => PIRBinaryOp::Less,
            BinaryOperator::LessEqual => PIRBinaryOp::LessEqual,
            BinaryOperator::Greater => PIRBinaryOp::Greater,
            BinaryOperator::GreaterEqual => PIRBinaryOp::GreaterEqual,
            BinaryOperator::And => PIRBinaryOp::And,
            BinaryOperator::Or => PIRBinaryOp::Or,
            BinaryOperator::SemanticEqual => PIRBinaryOp::SemanticEqual,
            _ => PIRBinaryOp::Equal, // Default fallback
        }
    }

    /// Convert visibility
    fn convert_visibility(&self, vis: &prism_ast::Visibility) -> PIRVisibility {
        match vis {
            prism_ast::Visibility::Public => PIRVisibility::Public,
            prism_ast::Visibility::Private => PIRVisibility::Private,
            prism_ast::Visibility::Internal => PIRVisibility::Internal,
        }
    }

    /// Create a global module for global items
    fn create_global_module(
        &mut self, 
        types: Vec<PIRSemanticType>, 
        functions: Vec<PIRFunction>,
        constants: Vec<PIRConstant>
    ) -> PIRResult<PIRModule> {
        let mut sections = Vec::new();

        if !types.is_empty() {
            sections.push(PIRSection::Types(TypeSection { types }));
        }

        if !functions.is_empty() {
            sections.push(PIRSection::Functions(FunctionSection { functions }));
        }

        if !constants.is_empty() {
            sections.push(PIRSection::Constants(ConstantSection { constants }));
        }

        Ok(PIRModule {
            name: "global".to_string(),
            capability: "global_definitions".to_string(),
            sections,
            dependencies: Vec::new(),
            business_context: BusinessContext::new("global".to_string()),
            domain_rules: Vec::new(),
            effects: Vec::new(),
            capabilities: Vec::new(),
            performance_profile: PerformanceProfile::default(),
            cohesion_score: 1.0,
        })
    }

    /// Extract business context from module
    fn extract_business_context(&self, module_decl: &ModuleDecl) -> PIRResult<BusinessContext> {
        let domain = module_decl.name.resolve().unwrap_or_else(|| "unknown".to_string());
        Ok(BusinessContext::new(domain))
    }

    /// Create performance profile for sections
    fn create_performance_profile(&self, _sections: &[PIRSection]) -> PIRResult<PerformanceProfile> {
        Ok(PerformanceProfile::default())
    }

    /// Extract AI metadata
    fn extract_ai_metadata(&self, _pir: &PrismIR) -> PIRResult<AIMetadata> {
        Ok(AIMetadata::default())
    }

    /// Analyze cohesion
    fn analyze_cohesion(&self, _pir: &PrismIR) -> PIRResult<CohesionMetrics> {
        Ok(CohesionMetrics {
            overall_score: 0.8,
            module_scores: HashMap::new(),
            coupling_metrics: CouplingMetrics {
                afferent: HashMap::new(),
                efferent: HashMap::new(),
                instability: HashMap::new(),
            },
        })
    }

    /// Build effect graph
    fn build_effect_graph(&self, _pir: &PrismIR) -> PIRResult<EffectGraph> {
        Ok(EffectGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
        })
    }

    /// Extract domain from type name
    fn extract_domain_from_name(&self, name: &str) -> String {
        // Simple heuristic: use the first part of camelCase or the whole name
        if name.contains("User") || name.contains("user") {
            "user_domain".to_string()
        } else if name.contains("Payment") || name.contains("payment") {
            "financial_domain".to_string()
        } else {
            "general_domain".to_string()
        }
    }

    /// Calculate source hash
    fn calculate_source_hash(&self, _program: &Program) -> u64 {
        // Simplified hash calculation
        42 // TODO: Implement proper hashing
    }
}

impl Default for PIRBuilder {
    fn default() -> Self {
        Self::new()
    }
} 