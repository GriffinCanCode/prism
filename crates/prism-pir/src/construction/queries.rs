//! PIR Construction Queries - CompilerQuery Implementations
//!
//! This module implements PIR-specific queries using the established CompilerQuery
//! trait from prism-compiler. Each query represents a specific transformation or
//! analysis step in PIR construction.

use crate::{
    PIRResult, PIRError,
    semantic::*,
    business::*,
    ai_integration::*,
};
use prism_ast::{Program, Item, ModuleDecl, FunctionDecl, TypeDecl, AstNode};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_compiler::query::{CompilerQuery, QueryContext, CacheKey, InvalidationTrigger, QueryId};
use prism_compiler::{CompilerResult, CompilerError};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Input for AST to PIR transformation query
#[derive(Debug, Clone, Hash)]
pub struct ASTToPIRInput {
    pub program: Program,
    pub semantic_context: Option<NodeId>,
    pub optimization_level: u8,
}

/// Output of AST to PIR transformation
#[derive(Debug, Clone)]
pub struct ASTToPIROutput {
    pub pir: PrismIR,
    pub diagnostics: Vec<PIRDiagnostic>,
    pub transformation_metadata: TransformationMetadata,
}

/// PIR diagnostic information
#[derive(Debug, Clone)]
pub struct PIRDiagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<Span>,
    pub suggestion: Option<String>,
}

/// Diagnostic severity level
#[derive(Debug, Clone)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
    Hint,
}

/// Transformation metadata for analysis
#[derive(Debug, Clone)]
pub struct TransformationMetadata {
    pub phases_completed: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub semantic_preservation_score: f64,
    pub business_context_coverage: f64,
}

/// Main AST to PIR transformation query
pub struct ASTToPIRQuery;

#[async_trait]
impl CompilerQuery<ASTToPIRInput, ASTToPIROutput> for ASTToPIRQuery {
    async fn execute(&self, input: ASTToPIRInput, context: QueryContext) -> CompilerResult<ASTToPIROutput> {
        // Create PIR builder with context-aware configuration
        let config = self.create_config_from_context(&context);
        let mut builder = crate::transformation_pipeline::PIRBuilder::with_config(config);

        // Perform transformation with error handling
        let pir = builder.build_from_program(&input.program)
            .map_err(|e| CompilerError::PIRTransformation(format!("PIR construction failed: {}", e)))?;

        // Collect diagnostics from transformation
        let diagnostics = self.collect_diagnostics(&builder, &pir);

        // Generate transformation metadata
        let transformation_metadata = self.generate_metadata(&builder, &pir);

        Ok(ASTToPIROutput {
            pir,
            diagnostics,
            transformation_metadata,
        })
    }

    fn cache_key(&self, input: &ASTToPIRInput) -> CacheKey {
        CacheKey::from_input("ast_to_pir", input)
            .with_semantic_context(&input.semantic_context)
            .with_target_config(&format!("opt_level_{}", input.optimization_level))
    }

    async fn dependencies(&self, input: &ASTToPIRInput, context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        let mut deps = HashSet::new();
        
        // Add dependencies for each module in the program
        for item in &input.program.items {
            if let Item::Module(module) = &item.kind {
                let module_query_id = PIRModuleQuery::query_id_for_module(&module.name);
                deps.insert(module_query_id);
            }
        }

        // Add semantic analysis dependencies if context is provided
        if let Some(semantic_ctx) = input.semantic_context {
            deps.insert(QueryId::new()); // Semantic analysis query ID
        }

        Ok(deps)
    }

    fn invalidate_on(&self, input: &ASTToPIRInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        
        // Invalidate on source file changes
        triggers.insert(InvalidationTrigger::SemanticContextChanged(
            input.semantic_context.unwrap_or(NodeId::new())
        ));
        
        // Invalidate on optimization level changes
        triggers.insert(InvalidationTrigger::OptimizationLevelChanged);
        
        // Invalidate on configuration changes
        triggers.insert(InvalidationTrigger::ConfigChanged);

        triggers
    }

    fn query_type(&self) -> &'static str {
        "ast_to_pir"
    }
}

impl ASTToPIRQuery {
    fn create_config_from_context(&self, context: &QueryContext) -> crate::transformation_pipeline::PIRBuilderConfig {
        // Extract configuration from query context
        crate::transformation_pipeline::PIRBuilderConfig {
            enable_validation: true,
            enable_ai_metadata: true,
            enable_cohesion_analysis: true,
            enable_effect_graph: true,
            enable_business_context: true,
            max_build_depth: 1000,
            enable_performance_profiling: context.profiling_enabled.unwrap_or(false),
        }
    }

    fn collect_diagnostics(&self, builder: &crate::transformation_pipeline::PIRBuilder, pir: &PrismIR) -> Vec<PIRDiagnostic> {
        let mut diagnostics = Vec::new();

        // Check for incomplete transformations
        for module in &pir.modules {
            if module.sections.is_empty() {
                diagnostics.push(PIRDiagnostic {
                    level: DiagnosticLevel::Warning,
                    message: format!("Module '{}' has no sections", module.name),
                    span: None,
                    suggestion: Some("Consider adding type, interface, or implementation sections".to_string()),
                });
            }

            // Check cohesion scores
            if module.cohesion_score < 0.5 {
                diagnostics.push(PIRDiagnostic {
                    level: DiagnosticLevel::Info,
                    message: format!("Module '{}' has low cohesion score: {:.2}", module.name, module.cohesion_score),
                    span: None,
                    suggestion: Some("Consider refactoring to improve conceptual cohesion".to_string()),
                });
            }
        }

        diagnostics
    }

    fn generate_metadata(&self, builder: &crate::transformation_pipeline::PIRBuilder, pir: &PrismIR) -> TransformationMetadata {
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert("modules_processed".to_string(), pir.modules.len() as f64);
        
        // Calculate average cohesion score
        let avg_cohesion = if !pir.modules.is_empty() {
            pir.modules.iter().map(|m| m.cohesion_score).sum::<f64>() / pir.modules.len() as f64
        } else {
            0.0
        };

        TransformationMetadata {
            phases_completed: vec![
                "ast_analysis".to_string(),
                "semantic_extraction".to_string(),
                "business_context".to_string(),
                "pir_construction".to_string(),
            ],
            performance_metrics,
            semantic_preservation_score: 0.95, // TODO: Calculate actual score
            business_context_coverage: avg_cohesion,
        }
    }
}

/// Input for PIR module construction query
#[derive(Debug, Clone, Hash)]
pub struct PIRModuleInput {
    pub module_decl: ModuleDecl,
    pub semantic_context: Option<NodeId>,
    pub dependencies: Vec<String>,
}

/// Output of PIR module construction
#[derive(Debug, Clone)]
pub struct PIRModuleOutput {
    pub module: PIRModule,
    pub extracted_dependencies: Vec<String>,
    pub business_context: BusinessContext,
}

/// PIR module construction query
pub struct PIRModuleQuery;

#[async_trait]
impl CompilerQuery<PIRModuleInput, PIRModuleOutput> for PIRModuleQuery {
    async fn execute(&self, input: PIRModuleInput, context: QueryContext) -> CompilerResult<PIRModuleOutput> {
        let mut builder = crate::transformation_pipeline::PIRBuilder::new();
        
        // Build the module
        let module = builder.build_module(&input.module_decl)
            .map_err(|e| CompilerError::PIRTransformation(format!("Module construction failed: {}", e)))?;

        // Extract business context
        let business_context = builder.extract_business_context(&input.module_decl)
            .map_err(|e| CompilerError::PIRTransformation(format!("Business context extraction failed: {}", e)))?;

        // Analyze dependencies (simplified for now)
        let extracted_dependencies = input.dependencies;

        Ok(PIRModuleOutput {
            module,
            extracted_dependencies,
            business_context,
        })
    }

    fn cache_key(&self, input: &PIRModuleInput) -> CacheKey {
        CacheKey::from_input("pir_module", input)
            .with_semantic_context(&input.semantic_context)
    }

    async fn dependencies(&self, input: &PIRModuleInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        let mut deps = HashSet::new();
        
        // Add dependencies for each section in the module
        for section in &input.module_decl.sections {
            for item in &section.items {
                match &item.kind {
                    prism_ast::Stmt::Function(func) => {
                        let func_query_id = PIRFunctionQuery::query_id_for_function(&func.name);
                        deps.insert(func_query_id);
                    }
                    prism_ast::Stmt::Type(type_decl) => {
                        let type_query_id = PIRTypeQuery::query_id_for_type(&type_decl.name);
                        deps.insert(type_query_id);
                    }
                    _ => {}
                }
            }
        }

        Ok(deps)
    }

    fn invalidate_on(&self, input: &PIRModuleInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        
        if let Some(ctx) = input.semantic_context {
            triggers.insert(InvalidationTrigger::SemanticContextChanged(ctx));
        }
        
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "pir_module"
    }
}

impl PIRModuleQuery {
    pub fn query_id_for_module(module_name: &Symbol) -> QueryId {
        // Generate consistent query ID based on module name
        QueryId::new() // Simplified for now
    }
}

/// Input for PIR function construction query
#[derive(Debug, Clone, Hash)]
pub struct PIRFunctionInput {
    pub function_decl: FunctionDecl,
    pub module_context: Option<String>,
    pub semantic_context: Option<NodeId>,
}

/// Output of PIR function construction
#[derive(Debug, Clone)]
pub struct PIRFunctionOutput {
    pub function: PIRFunction,
    pub extracted_effects: Vec<String>,
    pub performance_characteristics: Vec<String>,
}

/// PIR function construction query
pub struct PIRFunctionQuery;

#[async_trait]
impl CompilerQuery<PIRFunctionInput, PIRFunctionOutput> for PIRFunctionQuery {
    async fn execute(&self, input: PIRFunctionInput, _context: QueryContext) -> CompilerResult<PIRFunctionOutput> {
        let mut builder = crate::transformation_pipeline::PIRBuilder::new();
        
        // Build the function
        let function = builder.build_function(&input.function_decl)
            .map_err(|e| CompilerError::PIRTransformation(format!("Function construction failed: {}", e)))?;

        // Extract effects (simplified)
        let extracted_effects = vec!["io".to_string(), "memory".to_string()];

        // Extract performance characteristics (simplified)
        let performance_characteristics = vec!["linear_time".to_string()];

        Ok(PIRFunctionOutput {
            function,
            extracted_effects,
            performance_characteristics,
        })
    }

    fn cache_key(&self, input: &PIRFunctionInput) -> CacheKey {
        CacheKey::from_input("pir_function", input)
            .with_semantic_context(&input.semantic_context)
    }

    async fn dependencies(&self, _input: &PIRFunctionInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Functions may depend on type queries for their parameters and return types
        Ok(HashSet::new()) // Simplified for now
    }

    fn invalidate_on(&self, input: &PIRFunctionInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        
        if let Some(ctx) = input.semantic_context {
            triggers.insert(InvalidationTrigger::SemanticContextChanged(ctx));
        }
        
        triggers
    }

    fn query_type(&self) -> &'static str {
        "pir_function"
    }
}

impl PIRFunctionQuery {
    pub fn query_id_for_function(function_name: &Symbol) -> QueryId {
        // Generate consistent query ID based on function name
        QueryId::new() // Simplified for now
    }
}

/// Input for PIR type construction query
#[derive(Debug, Clone, Hash)]
pub struct PIRTypeInput {
    pub type_decl: TypeDecl,
    pub domain_context: Option<String>,
    pub semantic_context: Option<NodeId>,
}

/// Output of PIR type construction
#[derive(Debug, Clone)]
pub struct PIRTypeOutput {
    pub semantic_type: PIRSemanticType,
    pub business_rules: Vec<String>,
    pub validation_predicates: Vec<String>,
}

/// PIR type construction query
pub struct PIRTypeQuery;

#[async_trait]
impl CompilerQuery<PIRTypeInput, PIRTypeOutput> for PIRTypeQuery {
    async fn execute(&self, input: PIRTypeInput, _context: QueryContext) -> CompilerResult<PIRTypeOutput> {
        let mut builder = crate::transformation_pipeline::PIRBuilder::new();
        
        // Build the semantic type
        let semantic_type = builder.build_semantic_type(&input.type_decl)
            .map_err(|e| CompilerError::PIRTransformation(format!("Type construction failed: {}", e)))?;

        // Extract business rules (simplified)
        let business_rules = vec!["non_empty".to_string(), "valid_format".to_string()];

        // Extract validation predicates (simplified)
        let validation_predicates = vec!["length_check".to_string(), "format_validation".to_string()];

        Ok(PIRTypeOutput {
            semantic_type,
            business_rules,
            validation_predicates,
        })
    }

    fn cache_key(&self, input: &PIRTypeInput) -> CacheKey {
        CacheKey::from_input("pir_type", input)
            .with_semantic_context(&input.semantic_context)
    }

    async fn dependencies(&self, _input: &PIRTypeInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Types may depend on other type queries for composition
        Ok(HashSet::new()) // Simplified for now
    }

    fn invalidate_on(&self, input: &PIRTypeInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        
        if let Some(ctx) = input.semantic_context {
            triggers.insert(InvalidationTrigger::SemanticContextChanged(ctx));
        }
        
        triggers
    }

    fn query_type(&self) -> &'static str {
        "pir_type"
    }
}

impl PIRTypeQuery {
    pub fn query_id_for_type(type_name: &Symbol) -> QueryId {
        // Generate consistent query ID based on type name
        QueryId::new() // Simplified for now
    }
}

/// PIR validation query for semantic preservation
pub struct PIRValidationQuery;

/// PIR optimization query for performance improvements
pub struct PIROptimizationQuery;

// Additional query implementations would go here...
// These are placeholders for the complete system 