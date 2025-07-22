//! Core TypeScript Backend Implementation
//!
//! This module provides the main TypeScript backend implementation that integrates
//! all the modular components (types, semantic preservation, runtime integration, 
//! validation, optimization, source maps, ESM generation, branded types, template literals).

use super::{TypeScriptResult, TypeScriptError, TypeScriptBackendConfig, TypeScriptFeatures, TypeScriptTarget};
use super::types::{TypeScriptType, TypeScriptTypeConverter};
use super::semantic_preservation::{SemanticTypePreserver, BusinessRuleGenerator};
use super::runtime_integration::{RuntimeIntegrator, CapabilityManager, EffectTracker};
use super::validation::{TypeScriptValidator, ValidationConfig};
use super::optimization::{TypeScriptOptimizer, OptimizationConfig};
use super::source_maps::{SourceMapGenerator, SourceMapConfig};
use super::esm_generation::{ESMGenerator, ESMConfig};
use super::branded_types::{BrandedTypeGenerator, BrandingConfig};
use super::template_literals::{TemplateLiteralGenerator, TemplateConfig};

use crate::backends::{
    CompilationContext, CompilationTarget, CodeGenBackend, CodeArtifact, 
    CodeGenConfig, CodeGenStats, BackendCapabilities, AIMetadataLevel, AIMetadata,
    PrismIR, PIRModule, PIRFunction, PIRSemanticType, PIRExpression, PIRStatement,
    PIRSection, FunctionSection, ConstantSection, TypeSection,
};
use crate::CodeGenResult;
use async_trait::async_trait;
use prism_ast::Program;
use std::path::PathBuf;
use std::collections::HashMap;
use tracing::{debug, info, span, Level};

/// TypeScript backend with modular architecture
pub struct TypeScriptBackend {
    /// Backend configuration
    config: TypeScriptBackendConfig,
    /// Type converter for PIR to TypeScript type mapping
    type_converter: TypeScriptTypeConverter,
    /// Semantic type preserver
    semantic_preserver: SemanticTypePreserver,
    /// Runtime integrator
    runtime_integrator: RuntimeIntegrator,
    /// Code validator
    validator: TypeScriptValidator,
    /// Code optimizer
    optimizer: TypeScriptOptimizer,
    /// Source map generator
    source_map_generator: SourceMapGenerator,
    /// ESM generator
    esm_generator: ESMGenerator,
    /// Branded type generator
    branded_type_generator: BrandedTypeGenerator,
    /// Template literal generator
    template_literal_generator: TemplateLiteralGenerator,
    /// Business rule generator
    business_rule_generator: BusinessRuleGenerator,
}

impl TypeScriptBackend {
    /// Create new TypeScript backend with configuration
    pub fn new(config: CodeGenConfig) -> Self {
        let ts_config = TypeScriptBackendConfig::from_codegen_config(&config);
        
        // Create type converter
        let type_converter = TypeScriptTypeConverter::new(
            ts_config.typescript_features.clone(),
            ts_config.target,
        );

        // Create semantic preserver
        let semantic_preserver = SemanticTypePreserver::new(
            ts_config.semantic_config.clone(),
            type_converter.clone(),
        );

        // Create runtime integrator
        let runtime_integrator = RuntimeIntegrator::new(ts_config.runtime_config.clone());

        // Create validator
        let validator = TypeScriptValidator::new(ts_config.validation_config.clone());

        // Create optimizer
        let optimizer = TypeScriptOptimizer::new(ts_config.optimization_config.clone());

        // Create source map generator
        let source_map_generator = SourceMapGenerator::new(ts_config.source_map_config.clone());

        // Create ESM generator
        let esm_generator = ESMGenerator::new(ts_config.esm_config.clone());

        // Create branded type generator
        let branded_type_generator = BrandedTypeGenerator::new(ts_config.branding_config.clone());

        // Create template literal generator
        let template_literal_generator = TemplateLiteralGenerator::new(ts_config.template_config.clone());

        // Create business rule generator
        let business_rule_generator = BusinessRuleGenerator::new(ts_config.semantic_config.clone());

        Self {
            config: ts_config,
            type_converter,
            semantic_preserver,
            runtime_integrator,
            validator,
            optimizer,
            source_map_generator,
            esm_generator,
            branded_type_generator,
            template_literal_generator,
            business_rule_generator,
        }
    }

    /// Configure TypeScript target
    pub fn with_target(mut self, target: TypeScriptTarget) -> Self {
        self.config.target = target;
        // Update dependent components
        self.type_converter = TypeScriptTypeConverter::new(
            self.config.typescript_features.clone(),
            target,
        );
        self
    }

    /// Configure TypeScript features
    pub fn with_features(mut self, features: TypeScriptFeatures) -> Self {
        self.config.typescript_features = features.clone();
        // Update dependent components
        self.type_converter = TypeScriptTypeConverter::new(
            features,
            self.config.target,
        );
        self
    }

    /// Generate complete TypeScript module from PIR
    async fn generate_typescript_module(&mut self, pir: &PrismIR, config: &CodeGenConfig) -> TypeScriptResult<String> {
        let mut output = String::new();

        // Generate module header with metadata
        output.push_str(&self.generate_module_header(pir)?);

        // Generate runtime integration
        output.push_str(&self.runtime_integrator.generate_runtime_integration(pir)?);

        // Generate semantic type registry
        if let Some(type_registry) = &pir.semantic_types {
            output.push_str(&self.generate_semantic_type_registry(type_registry)?);
        }

        // Generate business rule system
        let semantic_types: Vec<_> = pir.type_registry.types.values().collect();
        output.push_str(&self.business_rule_generator.generate_business_rule_system(&semantic_types)?);

        // Generate PIR modules
        for module in &pir.modules {
            output.push_str(&self.generate_pir_module(module, config).await?);
        }

        // Generate runtime support functions
        output.push_str(&self.generate_runtime_support_functions());

        // Generate module exports
        output.push_str(&self.generate_module_exports());

        Ok(output)
    }

    /// Generate module header with comprehensive metadata
    fn generate_module_header(&self, pir: &PrismIR) -> TypeScriptResult<String> {
        let target_info = self.config.target.to_string();
        
        Ok(format!(
            r#"/**
 * Generated by Prism Compiler - Modular TypeScript Backend
 * PIR Version: {}
 * Generated at: {}
 * Optimization Level: {}
 * TypeScript Target: {}
 * Features: Branded Types={}, Template Literals={}, ESM={}, Satisfies={}
 * 
 * Semantic Metadata:
 * - Cohesion Score: {:.2}
 * - Module Count: {}
 * - Type Registry: {} types
 * - Effect Registry: {} effects
 *
 * Business Context:
 * - AI Metadata Level: {:?}
 * - Security Classification: Capability-based
 * - Performance Profile: Zero-cost abstractions with runtime safety
 *
 * TypeScript 2025 Features:
 * - Modern ESM imports/exports
 * - Branded types for semantic safety
 * - Template literal types for domain strings
 * - Satisfies operator for type safety
 * - Enhanced control flow analysis
 */

// Modern ESM imports with type-only imports where appropriate
import type {{
    SemanticValue,
    ValidationError,
    BusinessRuleValidator,
    EffectSignature,
    CapabilityRequirement,
    PerformanceContract,
    PrismRuntime,
    ExecutionContext,
}} from '@prism/runtime-types';

import {{
    createPrismRuntime,
    CapabilityManager,
    EffectTracker,
    BusinessRuleEngine,
    PerformanceMonitor,
    createSemanticValue,
    validateBusinessRules,
    ValidationError,
    CapabilityError,
    EffectError,
    RuntimeError,
}} from '@prism/runtime';

// TypeScript 5.x+ utility types for advanced type manipulation
type Prettify<T> = {{
    [K in keyof T]: T[K];
}} & {{}};

type Exact<T, U> = T extends U ? (U extends T ? T : never) : never;

// Template literal types for dynamic string-based types
type DomainPrefix<T extends string> = `${{T}}_`;
type CapabilityName<T extends string> = `capability:${{T}}`;
type EffectName<T extends string> = `effect:${{T}}`;

// Enhanced branded type system with satisfies operator support
declare const __brand: unique symbol;
type Brand<T, B extends string> = T & {{ readonly [__brand]: B }};

// Result type for functional error handling
export type Result<T, E = Error> = 
    | {{ readonly success: true; readonly data: T }} 
    | {{ readonly success: false; readonly error: E }};

export const Result = {{
    ok: <T>(data: T): Result<T, never> => ({{ success: true, data }} as const),
    err: <E>(error: E): Result<never, E> => ({{ success: false, error }} as const),
    isOk: <T, E>(result: Result<T, E>): result is {{ readonly success: true; readonly data: T }} => result.success,
    isErr: <T, E>(result: Result<T, E>): result is {{ readonly success: false; readonly error: E }} => !result.success,
}} as const;

// Cohesion metrics embedded as compile-time constants
export const PRISM_COHESION_SCORE = {} as const;
export const PRISM_MODULE_COUNT = {} as const;
export const PRISM_GENERATION_TIMESTAMP = '{}' as const;

// Module capability registry for runtime validation
export const MODULE_CAPABILITIES = {{
{}
}} as const satisfies Record<string, readonly string[]>;

"#,
            pir.metadata.version,
            pir.metadata.created_at.as_deref().unwrap_or("unknown"),
            self.config.core_config.optimization_level,
            target_info,
            self.config.typescript_features.branded_types,
            self.config.typescript_features.template_literal_types,
            self.config.typescript_features.esm_modules,
            self.config.typescript_features.use_satisfies_operator,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            pir.type_registry.types.len(),
            pir.effect_graph.nodes.len(),
            self.config.core_config.ai_metadata_level,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            chrono::Utc::now().to_rfc3339(),
            pir.modules.iter()
                .map(|m| format!("  '{}': [{}] as const", 
                    m.name, 
                    m.capabilities.iter()
                        .map(|c| format!("'{}'", c.name))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
                .collect::<Vec<_>>()
                .join(",\n")
        ))
    }

    /// Generate semantic type registry
    fn generate_semantic_type_registry(&mut self, registry: &crate::backends::SemanticTypeRegistry) -> TypeScriptResult<String> {
        let mut output = String::new();
        
        output.push_str("\n// === SEMANTIC TYPE REGISTRY ===\n");
        output.push_str("// Enhanced with TypeScript 2025 features and branded types\n\n");

        // Generate each semantic type
        for (type_name, semantic_type) in &registry.types {
            output.push_str(&self.semantic_preserver.generate_semantic_type(semantic_type)?);
        }

        Ok(output)
    }

    /// Generate PIR module as TypeScript code
    async fn generate_pir_module(&mut self, module: &PIRModule, config: &CodeGenConfig) -> TypeScriptResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            r#"
// === MODULE: {} ===
// Enhanced with TypeScript 5.x+ features and modern patterns
// Capability Domain: {}
// Business Context: {}
// Cohesion Score: {:.2}

/**
 * Module namespace with branded types and capability management
 * Uses modern TypeScript namespace patterns for organization
 */
export namespace {} {{
    // Module metadata available at compile-time
    export const METADATA = {{
        name: '{}',
        capability: '{}',
        domain: '{}',
        cohesionScore: {},
        generatedAt: '{}',
    }} as const satisfies ModuleMetadata;
    
    // Template literal types for this module's capabilities
    export type ModuleCapability = CapabilityName<'{}'>;
    export type ModuleEffect = EffectName<'{}'>;
    
"#,
            module.name,
            module.capability,
            module.business_context.domain,
            module.cohesion_score,
            self.to_pascal_case(&module.name),
            module.name,
            module.capability,
            module.business_context.domain,
            module.cohesion_score,
            chrono::Utc::now().to_rfc3339(),
            module.capability.to_lowercase(),
            module.capability.to_lowercase()
        ));

        // Generate sections
        for section in &module.sections {
            match section {
                PIRSection::Types(type_section) => {
                    output.push_str("    // === TYPE DEFINITIONS ===\n");
                    for semantic_type in &type_section.types {
                        output.push_str(&format!(
                            "    export type {} = {};\n",
                            semantic_type.name,
                            semantic_type.name
                        ));
                    }
                    output.push('\n');
                }
                PIRSection::Functions(function_section) => {
                    output.push_str("    // === FUNCTION DEFINITIONS ===\n");
                    for function in &function_section.functions {
                        output.push_str(&self.generate_typescript_function(function, config).await?);
                    }
                }
                PIRSection::Constants(constant_section) => {
                    output.push_str("    // === CONSTANTS ===\n");
                    for constant in &constant_section.constants {
                        output.push_str(&self.generate_typescript_constant(constant).await?);
                    }
                }
                _ => {
                    output.push_str("    // Other sections handled elsewhere\n");
                }
            }
        }

        output.push_str("}\n\n");
        
        Ok(output)
    }

    /// Generate TypeScript function from PIR function
    async fn generate_typescript_function(&self, function: &PIRFunction, _config: &CodeGenConfig) -> TypeScriptResult<String> {
        let params = self.generate_function_parameters(&function.signature.parameters)?;
        let return_type = self.type_converter.convert_pir_type_to_typescript(&function.signature.return_type)
            .map_err(|e| TypeScriptError::TypeConversion { message: e.to_string() })?;
        
        let capabilities = function.capabilities_required.iter()
            .map(|c| format!("'{}'", c.name))
            .collect::<Vec<_>>()
            .join(", ");
        
        let effects = function.signature.effects.effects.iter()
            .map(|e| format!("'{}'", e.name))
            .collect::<Vec<_>>()
            .join(", ");

        Ok(format!(
            r#"    /**
     * Enhanced function: {}
     * Responsibility: {}
     * 
     * @capabilities [{}]
     * @effects [{}]
     * 
     * Uses TypeScript 5.x+ features for enhanced type safety
     */
    export async function {}({}): Promise<Result<{}, FunctionError>> {{
        // Enhanced capability validation
        const requiredCapabilities = [{}] as const satisfies readonly ModuleCapability[];
        await CapabilityManager.validateCapabilities(requiredCapabilities);
        
        // Effect tracking
        const effectTracker = new EffectTracker<[{}]>();
        const requiredEffects = [{}] as const satisfies readonly ModuleEffect[];
        await effectTracker.trackEffects(requiredEffects);
        
        try {{
            // Performance monitoring
            const perfMonitor = new PerformanceMonitor('{}');
            await perfMonitor.start();
            
            // Function implementation
            const result = await (async (): Promise<{}> => {{
                // Function body would be generated here
                throw new Error('Function implementation not yet generated');
            }})();
            
            await perfMonitor.end();
            await effectTracker.complete();
            
            return Result.ok(result);
        }} catch (error) {{
            await effectTracker.abort();
            
            if (error instanceof ValidationError || error instanceof CapabilityError) {{
                return Result.err(new FunctionError(error.message, {{
                    function: '{}',
                    cause: error,
                    context: {{ capabilities: requiredCapabilities, effects: requiredEffects }}
                }}));
            }}
            
            throw error;
        }}
    }}

"#,
            function.name,
            function.responsibility.as_deref().unwrap_or("N/A"),
            capabilities,
            effects,
            function.name,
            params,
            return_type,
            capabilities,
            effects,
            effects,
            function.name,
            return_type,
            function.name
        ))
    }

    /// Generate TypeScript constant from PIR constant
    async fn generate_typescript_constant(&self, constant: &crate::backends::PIRConstant) -> TypeScriptResult<String> {
        let const_type = self.type_converter.convert_pir_type_to_typescript(&constant.const_type)
            .map_err(|e| TypeScriptError::TypeConversion { message: e.to_string() })?;
        let value = self.generate_expression(&constant.value)?;
        
        Ok(format!(
            r#"    /**
     * Constant: {}
     * Business meaning: {}
     * Type: {}
     */
    export const {} = {} as const satisfies {};

"#,
            constant.name,
            constant.business_meaning.as_deref().unwrap_or("N/A"),
            const_type,
            constant.name.to_uppercase(),
            value,
            const_type
        ))
    }

    /// Generate function parameters
    fn generate_function_parameters(&self, parameters: &[crate::backends::PIRParameter]) -> TypeScriptResult<String> {
        Ok(parameters.iter()
            .map(|param| {
                let param_type = self.type_converter.convert_pir_type_to_typescript(&param.param_type)
                    .unwrap_or_else(|_| TypeScriptType::Unknown);
                let default_value = param.default_value.as_ref()
                    .map(|v| format!(" = {}", self.generate_expression(v).unwrap_or_else(|_| "undefined".to_string())))
                    .unwrap_or_default();
                    
                format!("{}: {}{}", param.name, param_type, default_value)
            })
            .collect::<Vec<_>>()
            .join(", "))
    }

    /// Generate expression
    fn generate_expression(&self, expr: &PIRExpression) -> TypeScriptResult<String> {
        match expr {
            PIRExpression::Literal(lit) => {
                match lit {
                    crate::backends::PIRLiteral::Integer(i) => Ok(i.to_string()),
                    crate::backends::PIRLiteral::Float(f) => Ok(f.to_string()),
                    crate::backends::PIRLiteral::Boolean(b) => Ok(b.to_string()),
                    crate::backends::PIRLiteral::String(s) => Ok(format!("'{}'", s.replace('\'', "\\'"))),
                    crate::backends::PIRLiteral::Unit => Ok("undefined"),
                }
            }
            PIRExpression::Variable(name) => Ok(name.clone()),
            _ => Ok("/* Complex expression - implementation needed */".to_string())
        }
    }

    /// Generate runtime support functions
    fn generate_runtime_support_functions(&self) -> String {
        r#"
// === RUNTIME SUPPORT FUNCTIONS ===

/**
 * Base type validation utility
 */
function isValidBaseType(value: unknown, expectedType: string): boolean {
    switch (expectedType) {
        case 'number': return typeof value === 'number' && !Number.isNaN(value);
        case 'string': return typeof value === 'string';
        case 'boolean': return typeof value === 'boolean';
        default: return true; // For complex types, assume valid
    }
}

/**
 * Create semantic value with domain branding
 */
function createSemanticValue<T>(value: T, domain: string): T {
    // Zero-cost abstraction - no runtime overhead
    return value;
}

/**
 * Extract semantic value
 */
function extractSemanticValue<T>(semanticValue: T): T {
    return semanticValue;
}

// Enhanced error types
export class FunctionError extends Error {
    constructor(message: string, public readonly details: {
        function: string;
        cause: Error;
        context: Record<string, unknown>;
    }) {
        super(message);
        this.name = 'FunctionError';
    }
}

// Module metadata type
export interface ModuleMetadata {
    readonly name: string;
    readonly capability: string;
    readonly domain: string;
    readonly cohesionScore: number;
    readonly generatedAt: string;
}

"#.to_string()
    }

    /// Generate module exports
    fn generate_module_exports(&self) -> String {
        r#"
// === MODULE EXPORTS ===

// Export all generated types and functions
// (Individual exports would be generated based on PIR content)

// Runtime information
export const PRISM_TYPESCRIPT_BACKEND_INFO = {
    version: '2.0.0',
    features: {
        brandedTypes: true,
        templateLiterals: true,
        esmModules: true,
        satisfiesOperator: true,
        enhancedControlFlow: true,
        semanticPreservation: true,
        runtimeIntegration: true,
        aiMetadata: true,
    },
    target: 'ES2023',
    generatedAt: new Date().toISOString(),
} as const;

"#.to_string()
    }

    /// Helper method to convert to PascalCase
    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect::<String>()
    }
}

impl TypeScriptBackendConfig {
    /// Create TypeScript backend config from CodeGen config
    pub fn from_codegen_config(config: &CodeGenConfig) -> Self {
        Self {
            core_config: config.clone(),
            typescript_features: TypeScriptFeatures::default(),
            target: TypeScriptTarget::default(),
            semantic_config: super::semantic_preservation::SemanticPreservationConfig::default(),
            runtime_config: super::runtime_integration::RuntimeIntegrationConfig::default(),
            validation_config: super::validation::TypeScriptValidationConfig::default(),
            optimization_config: super::optimization::TypeScriptOptimizationConfig::default(),
            source_map_config: super::source_maps::SourceMapConfig::default(),
            esm_config: super::esm_generation::ESMConfig::default(),
            branding_config: super::branded_types::BrandingConfig::default(),
            template_config: super::template_literals::TemplateConfig::default(),
        }
    }
}

#[async_trait]
impl CodeGenBackend for TypeScriptBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::TypeScript
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let _span = span!(Level::INFO, "typescript_pir_codegen").entered();
        let start_time = std::time::Instant::now();

        info!("Generating TypeScript from PIR with modular architecture");

        // Clone self to make it mutable for generation
        let mut backend = self.clone();
        
        // Generate TypeScript code
        let typescript_content = backend.generate_typescript_module(pir, config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "TypeScript".to_string(),
                message: format!("TypeScript generation failed: {:?}", e),
            })?;

        let generation_time = start_time.elapsed().as_millis() as u64;

        // Apply optimizations
        let optimized_content = if config.optimization_level > 0 {
            backend.optimizer.optimize(&typescript_content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "TypeScript".to_string(),
                    message: format!("TypeScript optimization failed: {:?}", e),
                })?
        } else {
            typescript_content
        };

        // Validate the generated code
        let validation_issues = backend.validator.validate(&optimized_content)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "TypeScript".to_string(),
                message: format!("TypeScript validation failed: {:?}", e),
            })?;

        // Generate source map if enabled
        let source_map = if config.source_maps {
            Some(backend.source_map_generator.generate(&optimized_content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "TypeScript".to_string(),
                    message: format!("Source map generation failed: {:?}", e),
                })?)
        } else {
            None
        };

        // Generate AI metadata
        let ai_metadata = AIMetadata {
            semantic_types: pir.type_registry.types.keys()
                .map(|k| (k.clone(), format!("Semantic type: {}", k)))
                .collect(),
            business_context: pir.modules.iter()
                .map(|m| (m.name.clone(), format!("Module: {}, Capability: {}", m.name, m.capability)))
                .collect(),
            performance_hints: vec![
                "Uses TypeScript 2025 features for optimal performance".to_string(),
                "Zero-cost semantic abstractions".to_string(),
                "Branded types for compile-time safety".to_string(),
            ],
        };

        Ok(CodeArtifact {
            target: CompilationTarget::TypeScript,
            content: optimized_content,
            source_map,
            ai_metadata,
            output_path: PathBuf::from("prism-generated.ts"),
            stats: CodeGenStats {
                lines_generated: typescript_content.lines().count(),
                generation_time,
                optimizations_applied: if config.optimization_level > 0 { 
                    config.optimization_level as usize 
                } else { 0 },
                memory_usage: typescript_content.len(),
            },
        })
    }

    async fn generate_code(
        &self,
        program: &Program,
        context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        // Convert AST to PIR first, then use PIR generation
        let mut pir_builder = crate::backends::PIRConstructionBuilder::new();
        let pir = pir_builder.build_from_program(program)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "TypeScript".to_string(),
                message: format!("PIR construction failed: {:?}", e),
            })?;
        self.generate_code_from_pir(&pir, context, config).await
    }

    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut semantic_preserver = self.semantic_preserver.clone();
        semantic_preserver.generate_semantic_type(semantic_type)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "TypeScript".to_string(),
                message: format!("Semantic type generation failed: {:?}", e),
            })
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        self.generate_typescript_function(function, config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "TypeScript".to_string(),
                message: format!("Function generation failed: {:?}", e),
            })
    }

    async fn generate_validation_logic(
        &self,
        semantic_type: &PIRSemanticType,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        self.generate_semantic_type(semantic_type, config).await
    }

    async fn generate_runtime_support(
        &self,
        pir: &PrismIR,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut runtime_integrator = self.runtime_integrator.clone();
        runtime_integrator.generate_runtime_integration(pir)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "TypeScript".to_string(),
                message: format!("Runtime support generation failed: {:?}", e),
            })
    }

    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        if config.optimization_level > 0 {
            artifact.content = self.optimizer.optimize(&artifact.content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "TypeScript".to_string(),
                    message: format!("Optimization failed: {:?}", e),
                })?;
            artifact.stats.optimizations_applied = config.optimization_level as usize;
        }
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        self.validator.validate(&artifact.content)
            .map_err(|e| crate::CodeGenError::ValidationError {
                target: "TypeScript".to_string(),
                errors: vec![format!("Validation failed: {:?}", e)],
            })
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            source_maps: true,
            debug_info: true,
            incremental: true,
            parallel: true,
            optimization_levels: vec![0, 1, 2, 3],
        }
    }
}

impl Clone for TypeScriptBackend {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            type_converter: self.type_converter.clone(),
            semantic_preserver: SemanticTypePreserver::new(
                self.config.semantic_config.clone(),
                self.type_converter.clone(),
            ),
            runtime_integrator: RuntimeIntegrator::new(self.config.runtime_config.clone()),
            validator: TypeScriptValidator::new(self.config.validation_config.clone()),
            optimizer: TypeScriptOptimizer::new(self.config.optimization_config.clone()),
            source_map_generator: SourceMapGenerator::new(self.config.source_map_config.clone()),
            esm_generator: ESMGenerator::new(self.config.esm_config.clone()),
            branded_type_generator: BrandedTypeGenerator::new(self.config.branding_config.clone()),
            template_literal_generator: TemplateLiteralGenerator::new(self.config.template_config.clone()),
            business_rule_generator: BusinessRuleGenerator::new(self.config.semantic_config.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let config = CodeGenConfig::default();
        let backend = TypeScriptBackend::new(config);
        
        assert_eq!(backend.target(), CompilationTarget::TypeScript);
        assert!(backend.capabilities().source_maps);
        assert!(backend.capabilities().debug_info);
    }

    #[test]
    fn test_configuration() {
        let config = CodeGenConfig::default();
        let backend = TypeScriptBackend::new(config)
            .with_target(TypeScriptTarget::Deno)
            .with_features(TypeScriptFeatures {
                template_literal_types: true,
                branded_types: true,
                ..TypeScriptFeatures::default()
            });
        
        assert_eq!(backend.config.target, TypeScriptTarget::Deno);
        assert!(backend.config.typescript_features.template_literal_types);
        assert!(backend.config.typescript_features.branded_types);
    }
} 