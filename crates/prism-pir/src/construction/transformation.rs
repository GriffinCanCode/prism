//! AST to PIR Transformation Pipeline
//!
//! This module implements the core transformation logic for converting AST nodes
//! into PIR representation while preserving semantic information, business context,
//! and AI metadata.
//!
//! **Conceptual Responsibility**: AST->PIR transformation coordination
//! **What it does**: Transforms AST nodes to PIR, preserves semantic information, extracts business context
//! **What it doesn't do**: AST parsing, PIR validation, semantic analysis (delegates to domain experts)

use crate::{PIRResult, PIRError};
use crate::semantic::{PrismIR, PIRModule, PIRSection, PIRSemanticType, PIRFunction, TypeSection, FunctionSection};
use crate::business::BusinessContext;
use crate::ai_integration::AIMetadata;
use crate::construction::semantic_preservation::{SemanticPreservationValidator, PreservationConfig};
use crate::construction::business_extraction::{BusinessContextExtractor, BusinessExtractionConfig};
use crate::construction::effect_integration::{EffectSystemIntegrator, EffectIntegrationConfig};
use crate::construction::ai_extraction::{AIMetadataExtractor as AIExtractor, AIExtractionConfig};
use prism_common::{PIRConstructor, PIRConstructionConfig, PrismError};
use prism_ast::{Program, AstNode, Item, ModuleDecl, FunctionDecl, TypeDecl};
use std::collections::HashMap;
use std::sync::Arc;

/// AST to PIR transformer implementing the PIRConstructor trait
pub struct ASTToPIRTransformer {
    /// Transformation configuration
    config: PIRConstructionConfig,
    /// Semantic context extractor
    semantic_extractor: SemanticContextExtractor,
    /// Business context extractor  
    business_extractor: BusinessContextExtractor,
    /// AI metadata extractor
    ai_extractor: AIExtractor,
    /// Semantic preservation validator
    preservation_validator: Option<SemanticPreservationValidator>,
    /// Effect system integrator
    effect_integrator: Option<EffectSystemIntegrator>,
}

/// Extracts semantic context from AST nodes
pub struct SemanticContextExtractor {
    /// Enable type inference
    enable_type_inference: bool,
    /// Enable constraint extraction
    enable_constraint_extraction: bool,
}

/// Extracts business context from AST nodes (placeholder - actual implementation in business_extraction.rs)
pub struct BusinessContextExtractorPlaceholder {
    /// Enable capability analysis
    enable_capability_analysis: bool,
    /// Enable cohesion measurement
    enable_cohesion_measurement: bool,
}

/// Extracts AI metadata from AST nodes (placeholder - actual implementation in ai_extraction.rs)
pub struct AIMetadataExtractorPlaceholder {
    /// Enable function context extraction
    enable_function_contexts: bool,
    /// Enable type context extraction
    enable_type_contexts: bool,
}

/// Transformation context for maintaining state during conversion
pub struct TransformationContext {
    /// Current module being processed
    current_module: Option<String>,
    /// Type mappings for cross-references
    type_mappings: HashMap<String, String>,
    /// Function mappings for cross-references
    function_mappings: HashMap<String, String>,
    /// Collected diagnostics
    diagnostics: Vec<TransformationDiagnostic>,
}

/// Diagnostic information from transformation
#[derive(Debug, Clone)]
pub struct TransformationDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    /// Diagnostic message
    pub message: String,
    /// Source location
    pub location: Option<prism_common::span::Span>,
}

/// Diagnostic levels
#[derive(Debug, Clone)]
pub enum DiagnosticLevel {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
}

impl ASTToPIRTransformer {
    /// Create a new AST to PIR transformer
    pub fn new(config: PIRConstructionConfig) -> Self {
        let preservation_validator = if config.enable_validation {
            Some(SemanticPreservationValidator::new(PreservationConfig::default()))
        } else {
            None
        };

        let effect_integrator = if config.enable_effect_graph {
            Some(EffectSystemIntegrator::new(EffectIntegrationConfig::default()))
        } else {
            None
        };

        Self {
            semantic_extractor: SemanticContextExtractor {
                enable_type_inference: config.enable_ai_metadata,
                enable_constraint_extraction: config.enable_validation,
            },
            business_extractor: BusinessContextExtractor::new(BusinessExtractionConfig {
                enable_domain_inference: config.enable_business_context,
                enable_rule_extraction: config.enable_business_context,
                enable_capability_analysis: config.enable_business_context,
                min_domain_confidence: 0.3,
            }),
            ai_extractor: AIExtractor::new(AIExtractionConfig {
                enable_function_contexts: config.enable_ai_metadata,
                enable_type_contexts: config.enable_ai_metadata,
                enable_module_contexts: config.enable_ai_metadata,
                enable_documentation_analysis: config.enable_ai_metadata,
                enable_pattern_recognition: config.enable_ai_metadata,
                enable_learning_hints: config.enable_ai_metadata,
                min_confidence_threshold: 0.5,
            }),
            preservation_validator,
            effect_integrator,
            config,
        }
    }

    /// Transform a program to PIR
    pub fn transform_program(&mut self, program: &Program) -> PIRResult<PrismIR> {
        let mut context = TransformationContext::new();
        let mut pir = PrismIR::new();

        // Extract modules from program
        let modules = self.extract_modules(program, &mut context)?;
        pir.modules = modules;

        // Build type registry
        pir.type_registry = self.build_type_registry(program, &context)?;

        // Build effect graph if enabled
        if self.config.enable_effect_graph {
            if let Some(ref mut integrator) = self.effect_integrator {
                let effect_result = integrator.integrate_effects(program)?;
                pir.effect_graph = effect_result.effect_graph;
                
                // Add effect integration diagnostics to context
                for diagnostic in &effect_result.diagnostics {
                    let diagnostic_level = match diagnostic.level {
                        crate::construction::effect_integration::EffectDiagnosticLevel::Info => DiagnosticLevel::Info,
                        crate::construction::effect_integration::EffectDiagnosticLevel::Warning => DiagnosticLevel::Warning,
                        crate::construction::effect_integration::EffectDiagnosticLevel::Error => DiagnosticLevel::Error,
                    };
                    
                    context.diagnostics.push(TransformationDiagnostic {
                        level: diagnostic_level,
                        message: format!("Effect integration: {}", diagnostic.message),
                        location: diagnostic.location,
                    });
                }
            } else {
                pir.effect_graph = self.build_effect_graph(program, &context)?;
            }
        }

        // Extract business context if enabled
        if self.config.enable_business_context {
            self.extract_business_contexts(&mut pir, program, &context)?;
        }

        // Extract AI metadata if enabled
        if self.config.enable_ai_metadata {
            let ai_result = self.ai_extractor.extract_ai_metadata(program)?;
            pir.ai_metadata = ai_result.metadata;
            
            // Add AI extraction diagnostics to context
            for diagnostic in &ai_result.diagnostics {
                let diagnostic_level = match diagnostic.level {
                    crate::construction::ai_extraction::AIExtractionLevel::Info => DiagnosticLevel::Info,
                    crate::construction::ai_extraction::AIExtractionLevel::Insight => DiagnosticLevel::Info,
                    crate::construction::ai_extraction::AIExtractionLevel::Learning => DiagnosticLevel::Info,
                    crate::construction::ai_extraction::AIExtractionLevel::Improvement => DiagnosticLevel::Warning,
                };
                
                context.diagnostics.push(TransformationDiagnostic {
                    level: diagnostic_level,
                    message: format!("AI extraction: {}", diagnostic.message),
                    location: diagnostic.location,
                });
            }
        }

        // Calculate cohesion metrics if enabled
        if self.config.enable_cohesion_analysis {
            pir.cohesion_metrics = self.calculate_cohesion_metrics(&pir)?;
        }

        // Update metadata
        pir.metadata.source_hash = self.calculate_source_hash(program);
        pir.metadata.created_at = chrono::Utc::now().to_rfc3339();

        // Validate semantic preservation if enabled
        if let Some(ref mut validator) = self.preservation_validator {
            let preservation_result = validator.validate_preservation(program, &pir)?;
            
            // Add preservation findings to context diagnostics
            for finding in &preservation_result.findings {
                let diagnostic_level = match finding.severity {
                    crate::construction::semantic_preservation::PreservationSeverity::Info => DiagnosticLevel::Info,
                    crate::construction::semantic_preservation::PreservationSeverity::Warning => DiagnosticLevel::Warning,
                    crate::construction::semantic_preservation::PreservationSeverity::Error => DiagnosticLevel::Error,
                    crate::construction::semantic_preservation::PreservationSeverity::Critical => DiagnosticLevel::Error,
                };
                
                context.diagnostics.push(TransformationDiagnostic {
                    level: diagnostic_level,
                    message: format!("Semantic preservation: {}", finding.description),
                    location: finding.location,
                });
            }
            
            // If preservation failed critically, return error
            if !preservation_result.success && preservation_result.score < 0.5 {
                return Err(PIRError::SemanticViolation {
                    message: format!("Semantic preservation failed with score: {:.2}", preservation_result.score),
                    location: "transformation".to_string(),
                });
            }
        }

        Ok(pir)
    }

    /// Extract modules from program items
    fn extract_modules(&mut self, program: &Program, context: &mut TransformationContext) -> PIRResult<Vec<PIRModule>> {
        let mut modules = Vec::new();
        let mut global_items = Vec::new();

        // Separate modules from global items
        for item in &program.items {
            match &item.kind {
                Item::Module(module_decl) => {
                    let pir_module = self.transform_module(module_decl, context)?;
                    modules.push(pir_module);
                }
                _ => {
                    global_items.push(item);
                }
            }
        }

        // Create global module for non-module items
        if !global_items.is_empty() {
            let global_module = self.create_global_module(&global_items, context)?;
            modules.push(global_module);
        }

        Ok(modules)
    }

    /// Transform a module declaration to PIR module
    fn transform_module(&mut self, module_decl: &ModuleDecl, context: &mut TransformationContext) -> PIRResult<PIRModule> {
        context.current_module = Some(module_decl.name.clone());

        let capability = self.extract_module_capability(module_decl);
        let sections = self.transform_module_sections(module_decl, context)?;
        let business_context = self.business_extractor.extract_module_business_context(module_decl)?;
        let performance_profile = self.extract_performance_profile(module_decl);
        let cohesion_score = if self.config.enable_cohesion_analysis {
            self.calculate_module_cohesion(module_decl)
        } else {
            0.5 // Default score
        };

        Ok(PIRModule {
            name: module_decl.name.clone(),
            capability,
            sections,
            dependencies: Vec::new(), // TODO: Extract dependencies
            business_context,
            domain_rules: Vec::new(), // TODO: Extract domain rules
            effects: Vec::new(), // TODO: Extract effects
            capabilities: Vec::new(), // TODO: Extract capabilities
            performance_profile,
            cohesion_score,
        })
    }

    /// Transform module sections
    fn transform_module_sections(&mut self, module_decl: &ModuleDecl, context: &mut TransformationContext) -> PIRResult<Vec<PIRSection>> {
        let mut sections = Vec::new();
        let mut types = Vec::new();
        let mut functions = Vec::new();

        // Process module items
        for item in &module_decl.items {
            match &item.kind {
                Item::Type(type_decl) => {
                    let pir_type = self.transform_type_declaration(type_decl, context)?;
                    types.push(pir_type);
                }
                Item::Function(func_decl) => {
                    let pir_function = self.transform_function_declaration(func_decl, context)?;
                    functions.push(pir_function);
                }
                _ => {
                    // Handle other item types
                    context.diagnostics.push(TransformationDiagnostic {
                        level: DiagnosticLevel::Info,
                        message: format!("Unhandled item type in module {}", module_decl.name),
                        location: Some(item.span),
                    });
                }
            }
        }

        // Create sections
        if !types.is_empty() {
            sections.push(PIRSection::Types(TypeSection { types }));
        }
        if !functions.is_empty() {
            sections.push(PIRSection::Functions(FunctionSection { functions }));
        }

        Ok(sections)
    }

    /// Transform type declaration to PIR semantic type
    fn transform_type_declaration(&mut self, type_decl: &TypeDecl, context: &mut TransformationContext) -> PIRResult<PIRSemanticType> {
        let base_type = self.semantic_extractor.extract_base_type_info(type_decl)?;
        let domain = self.extract_type_domain(type_decl);
        let business_rules = self.business_extractor.extract_type_business_rules(type_decl)?;
        let validation_predicates = self.extract_validation_predicates(type_decl);
        let constraints = self.extract_type_constraints(type_decl);
        let ai_context = self.ai_extractor.extract_type_ai_context(type_decl)?;
        let security_classification = self.extract_security_classification(type_decl);

        // Update type mappings
        context.type_mappings.insert(type_decl.name.to_string(), type_decl.name.to_string());

        Ok(PIRSemanticType {
            name: type_decl.name.to_string(),
            base_type,
            domain,
            business_rules,
            validation_predicates,
            constraints,
            ai_context,
            security_classification,
        })
    }

    /// Transform function declaration to PIR function
    fn transform_function_declaration(&mut self, func_decl: &FunctionDecl, context: &mut TransformationContext) -> PIRResult<PIRFunction> {
        let signature = self.semantic_extractor.extract_function_signature(func_decl)?;
        let body = self.transform_function_body(func_decl)?;
        let responsibility = self.extract_function_responsibility(func_decl);
        let algorithm = self.extract_algorithm_description(func_decl);
        let complexity = self.analyze_function_complexity(func_decl);
        let capabilities_required = self.extract_required_capabilities(func_decl);
        let performance_characteristics = self.extract_performance_characteristics(func_decl);
        let ai_hints = self.ai_extractor.extract_function_ai_hints(func_decl)?;

        // Update function mappings
        context.function_mappings.insert(func_decl.name.to_string(), func_decl.name.to_string());

        Ok(PIRFunction {
            name: func_decl.name.to_string(),
            signature,
            body,
            responsibility,
            algorithm,
            complexity,
            capabilities_required,
            performance_characteristics,
            ai_hints,
        })
    }

    /// Create global module for non-module items
    fn create_global_module(&mut self, items: &[&AstNode<Item>], context: &mut TransformationContext) -> PIRResult<PIRModule> {
        context.current_module = Some("global".to_string());

        let mut sections = Vec::new();
        let mut types = Vec::new();
        let mut functions = Vec::new();

        for item in items {
            match &item.kind {
                Item::Type(type_decl) => {
                    let pir_type = self.transform_type_declaration(type_decl, context)?;
                    types.push(pir_type);
                }
                Item::Function(func_decl) => {
                    let pir_function = self.transform_function_declaration(func_decl, context)?;
                    functions.push(pir_function);
                }
                _ => {
                    // Handle other global items
                }
            }
        }

        if !types.is_empty() {
            sections.push(PIRSection::Types(TypeSection { types }));
        }
        if !functions.is_empty() {
            sections.push(PIRSection::Functions(FunctionSection { functions }));
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
            performance_profile: crate::quality::PerformanceProfile::default(),
            cohesion_score: 0.8, // Global modules have decent cohesion
        })
    }

    // Helper methods for extraction (stubs for now)
    fn extract_module_capability(&self, _module_decl: &ModuleDecl) -> String {
        "default_capability".to_string() // TODO: Extract from attributes
    }

    fn extract_type_domain(&self, _type_decl: &TypeDecl) -> String {
        "default_domain".to_string() // TODO: Extract from attributes
    }

    fn extract_function_responsibility(&self, _func_decl: &FunctionDecl) -> Option<String> {
        None // TODO: Extract from documentation/attributes
    }

    fn extract_algorithm_description(&self, _func_decl: &FunctionDecl) -> Option<String> {
        None // TODO: Extract from documentation
    }

    fn analyze_function_complexity(&self, _func_decl: &FunctionDecl) -> Option<crate::semantic::PIRComplexityAnalysis> {
        None // TODO: Implement complexity analysis
    }

    fn extract_required_capabilities(&self, _func_decl: &FunctionDecl) -> Vec<crate::semantic::Capability> {
        Vec::new() // TODO: Extract from effects/attributes
    }

    fn extract_performance_characteristics(&self, _func_decl: &FunctionDecl) -> Vec<String> {
        Vec::new() // TODO: Extract from attributes/analysis
    }

    fn extract_validation_predicates(&self, _type_decl: &TypeDecl) -> Vec<crate::semantic::ValidationPredicate> {
        Vec::new() // TODO: Extract from type constraints
    }

    fn extract_type_constraints(&self, _type_decl: &TypeDecl) -> Vec<crate::semantic::PIRTypeConstraint> {
        Vec::new() // TODO: Extract from AST type constraints
    }

    fn extract_security_classification(&self, _type_decl: &TypeDecl) -> crate::semantic::SecurityClassification {
        crate::semantic::SecurityClassification::Public // TODO: Extract from attributes
    }

    fn transform_function_body(&self, _func_decl: &FunctionDecl) -> PIRResult<crate::semantic::PIRExpression> {
        // TODO: Transform function body to PIR expressions
        Ok(crate::semantic::PIRExpression::Literal(crate::semantic::PIRLiteral::Unit))
    }

    fn extract_performance_profile(&self, _module_decl: &ModuleDecl) -> crate::quality::PerformanceProfile {
        crate::quality::PerformanceProfile::default() // TODO: Analyze performance characteristics
    }

    fn calculate_module_cohesion(&self, _module_decl: &ModuleDecl) -> f64 {
        0.7 // TODO: Implement cohesion calculation
    }

    fn build_type_registry(&self, _program: &Program, _context: &TransformationContext) -> PIRResult<crate::semantic::SemanticTypeRegistry> {
        Ok(crate::semantic::SemanticTypeRegistry {
            types: HashMap::new(),
            relationships: HashMap::new(),
            global_constraints: Vec::new(),
        })
    }

    fn build_effect_graph(&self, _program: &Program, _context: &TransformationContext) -> PIRResult<crate::semantic::EffectGraph> {
        Ok(crate::semantic::EffectGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
        })
    }

    fn extract_business_contexts(&self, _pir: &mut PrismIR, _program: &Program, _context: &TransformationContext) -> PIRResult<()> {
        // TODO: Extract business contexts from modules
        Ok(())
    }

    fn extract_ai_metadata(&self, _program: &Program, _context: &TransformationContext) -> PIRResult<AIMetadata> {
        Ok(AIMetadata::default())
    }

    fn calculate_cohesion_metrics(&self, _pir: &PrismIR) -> PIRResult<crate::semantic::CohesionMetrics> {
        Ok(crate::semantic::CohesionMetrics {
            overall_score: 0.7,
            module_scores: HashMap::new(),
            coupling_metrics: crate::semantic::CouplingMetrics {
                afferent: HashMap::new(),
                efferent: HashMap::new(),
                instability: HashMap::new(),
            },
        })
    }

    fn calculate_source_hash(&self, _program: &Program) -> u64 {
        // TODO: Calculate hash of source program
        0
    }
}

impl PIRConstructor for ASTToPIRTransformer {
    type Program = Program;
    type PIR = PrismIR;
    type Error = PIRError;

    fn construct_pir(&mut self, program: Self::Program) -> Result<Self::PIR, Self::Error> {
        self.transform_program(&program)
    }

    fn construct_pir_with_config(
        &mut self,
        program: Self::Program,
        config: &PIRConstructionConfig,
    ) -> Result<Self::PIR, Self::Error> {
        // Update configuration
        self.config = config.clone();
        self.transform_program(&program)
    }
}

// Implementation for extractors
impl SemanticContextExtractor {
    fn extract_base_type_info(&self, _type_decl: &TypeDecl) -> PIRResult<crate::semantic::PIRTypeInfo> {
        // TODO: Extract actual type information from AST
        Ok(crate::semantic::PIRTypeInfo::Primitive(crate::semantic::PIRPrimitiveType::String))
    }

    fn extract_function_signature(&self, _func_decl: &FunctionDecl) -> PIRResult<crate::semantic::PIRFunctionType> {
        // TODO: Extract function signature from AST
        Ok(crate::semantic::PIRFunctionType {
            parameters: Vec::new(),
            return_type: Box::new(crate::semantic::PIRTypeInfo::Primitive(crate::semantic::PIRPrimitiveType::Unit)),
            effects: crate::semantic::EffectSignature {
                input_effects: Vec::new(),
                output_effects: Vec::new(),
                effect_dependencies: Vec::new(),
            },
            contracts: crate::semantic::PIRPerformanceContract {
                preconditions: Vec::new(),
                postconditions: Vec::new(),
                performance_guarantees: Vec::new(),
            },
        })
    }
}

impl BusinessContextExtractor {
    fn extract_module_business_context(&self, module_decl: &ModuleDecl) -> PIRResult<BusinessContext> {
        let domain = self.extract_business_domain(module_decl);
        Ok(BusinessContext::new(domain))
    }

    fn extract_type_business_rules(&self, _type_decl: &TypeDecl) -> PIRResult<Vec<crate::business::BusinessRule>> {
        // TODO: Extract business rules from type attributes/documentation
        Ok(Vec::new())
    }

    fn extract_business_domain(&self, module_decl: &ModuleDecl) -> String {
        // TODO: Extract domain from module attributes
        format!("{}_domain", module_decl.name)
    }
}

impl AIMetadataExtractor {
    fn extract_type_ai_context(&self, _type_decl: &TypeDecl) -> PIRResult<crate::semantic::PIRTypeAIContext> {
        // TODO: Extract AI context from type documentation/attributes
        Ok(crate::semantic::PIRTypeAIContext {
            intent: None,
            examples: Vec::new(),
            common_mistakes: Vec::new(),
            best_practices: Vec::new(),
        })
    }

    fn extract_function_ai_hints(&self, _func_decl: &FunctionDecl) -> PIRResult<Vec<String>> {
        // TODO: Extract AI hints from function documentation/attributes
        Ok(Vec::new())
    }
}

impl TransformationContext {
    fn new() -> Self {
        Self {
            current_module: None,
            type_mappings: HashMap::new(),
            function_mappings: HashMap::new(),
            diagnostics: Vec::new(),
        }
    }
} 