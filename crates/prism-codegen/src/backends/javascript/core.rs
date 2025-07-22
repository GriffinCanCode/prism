//! Core JavaScript Backend Implementation
//!
//! This module provides the main JavaScript backend implementation that integrates
//! all the modular components (types, semantic preservation, runtime integration, 
//! validation, optimization, source maps, ESM generation, performance).

use super::{JavaScriptResult, JavaScriptError, JavaScriptBackendConfig, JavaScriptFeatures, JavaScriptTarget};
use super::types::{JavaScriptType, JavaScriptTypeConverter};
use super::semantic_preservation::{SemanticTypePreserver, BusinessRuleGenerator};
use super::runtime_integration::{RuntimeIntegrator, CapabilityManager, EffectTracker};
use super::validation::{JavaScriptValidator, ValidationConfig};
use super::optimization::{JavaScriptOptimizer, OptimizationConfig};
use super::source_maps::{SourceMapGenerator, SourceMapConfig};
use super::esm_generation::{ESMGenerator, ESMConfig};
use super::performance::{PerformanceOptimizer, PerformanceConfig};

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

/// JavaScript backend with modular architecture
#[derive(Debug, Clone)]
pub struct JavaScriptBackend {
    /// Backend configuration
    config: JavaScriptBackendConfig,
    /// Type converter for PIR to JavaScript type mapping
    type_converter: JavaScriptTypeConverter,
    /// Semantic type preserver
    semantic_preserver: SemanticTypePreserver,
    /// Business rule generator
    business_rule_generator: BusinessRuleGenerator,
    /// Runtime integrator
    runtime_integrator: RuntimeIntegrator,
    /// Capability manager
    capability_manager: CapabilityManager,
    /// Effect tracker
    effect_tracker: EffectTracker,
    /// Code validator
    validator: JavaScriptValidator,
    /// Code optimizer
    optimizer: JavaScriptOptimizer,
    /// Source map generator
    source_map_generator: SourceMapGenerator,
    /// ESM generator
    esm_generator: ESMGenerator,
    /// Performance optimizer
    performance_optimizer: PerformanceOptimizer,
}

impl JavaScriptBackend {
    /// Create a new JavaScript backend with the given configuration
    pub fn new(config: JavaScriptBackendConfig) -> Self {
        Self {
            type_converter: JavaScriptTypeConverter::new(config.type_config.clone()),
            semantic_preserver: SemanticTypePreserver::new(config.semantic_config.clone()),
            business_rule_generator: BusinessRuleGenerator::new(config.semantic_config.clone()),
            runtime_integrator: RuntimeIntegrator::new(config.runtime_config.clone()),
            capability_manager: CapabilityManager::new(config.runtime_config.clone()),
            effect_tracker: EffectTracker::new(config.runtime_config.clone()),
            validator: JavaScriptValidator::new(config.validation_config.clone()),
            optimizer: JavaScriptOptimizer::new(config.optimization_config.clone()),
            source_map_generator: SourceMapGenerator::new(config.source_map_config.clone()),
            esm_generator: ESMGenerator::new(config.esm_config.clone()),
            performance_optimizer: PerformanceOptimizer::new(config.performance_config.clone()),
            config,
        }
    }

    /// Generate JavaScript module from PIR
    async fn generate_javascript_module(&mut self, pir: &PrismIR, config: &CodeGenConfig) -> JavaScriptResult<String> {
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
    fn generate_module_header(&self, pir: &PrismIR) -> JavaScriptResult<String> {
        let target_info = self.config.target.to_string();
        
        Ok(format!(
            r#"/**
 * Generated by Prism Compiler - Modern JavaScript Backend
 * PIR Version: {}
 * Generated at: {}
 * Optimization Level: {}
 * JavaScript Target: {}
 * Features: ESM={}, Async/Await={}, Classes={}, Optional Chaining={}
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
 * - Performance Profile: Optimized for modern JavaScript engines
 *
 * JavaScript 2025 Features:
 * - Modern ESM imports/exports with tree-shaking
 * - Runtime type validation with branded types
 * - Async/await for all effectful operations
 * - WeakMap/WeakRef for memory-efficient caching
 * - Proxy-based semantic type enforcement
 * - Symbol-based private properties
 */

'use strict';

// Modern ESM imports with explicit runtime dependencies
{}

// Performance optimizations for modern JavaScript engines
const PRISM_RUNTIME_CONFIG = Object.freeze({{
    enableTypeValidation: true,
    enableEffectTracking: true,
    enableCapabilityValidation: true,
    enableBusinessRules: true,
    enablePerformanceMonitoring: {},
    target: '{}',
    features: {},
}});

// Cohesion metrics embedded as compile-time constants
export const PRISM_COHESION_SCORE = {};
export const PRISM_MODULE_COUNT = {};
export const PRISM_GENERATION_TIMESTAMP = '{}';

// Module capability registry for runtime validation
export const MODULE_CAPABILITIES = Object.freeze({{
{}
}});

// Effect registry for runtime tracking
export const MODULE_EFFECTS = Object.freeze({{
{}
}});

"#,
            pir.metadata.version,
            chrono::Utc::now().to_rfc3339(),
            2, // Default optimization level
            target_info,
            self.config.features.esm,
            self.config.features.async_await,
            self.config.features.classes,
            self.config.features.optional_chaining,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            pir.type_registry.types.len(),
            pir.effect_graph.nodes.len(),
            AIMetadataLevel::Full, // Default
            self.generate_runtime_imports()?,
            self.config.performance_config.enable_monitoring,
            target_info,
            self.serialize_features()?,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            chrono::Utc::now().to_rfc3339(),
            self.generate_capabilities_object(pir)?,
            self.generate_effects_object(pir)?,
        ))
    }

    /// Generate runtime imports based on configuration
    fn generate_runtime_imports(&self) -> JavaScriptResult<String> {
        let mut imports = String::new();

        if self.config.features.esm {
            imports.push_str("import {\n");
            imports.push_str("  PrismRuntime,\n");
            imports.push_str("  SemanticType,\n");
            imports.push_str("  EffectTracker,\n");
            imports.push_str("  CapabilityManager,\n");
            imports.push_str("  BusinessRuleEngine,\n");
            imports.push_str("  PerformanceMonitor,\n");
            imports.push_str("  ValidationError,\n");
            imports.push_str("  CapabilityError,\n");
            imports.push_str("  EffectError,\n");
            imports.push_str("  RuntimeError,\n");
            imports.push_str("} from '@prism/runtime';\n\n");

            imports.push_str("import {\n");
            imports.push_str("  createSemanticType,\n");
            imports.push_str("  validateBusinessRules,\n");
            imports.push_str("  createProxy,\n");
            imports.push_str("  createBrandedType,\n");
            imports.push_str("} from '@prism/runtime-utils';\n\n");
        } else {
            imports.push_str("const {\n");
            imports.push_str("  PrismRuntime,\n");
            imports.push_str("  SemanticType,\n");
            imports.push_str("  EffectTracker,\n");
            imports.push_str("  CapabilityManager,\n");
            imports.push_str("  BusinessRuleEngine,\n");
            imports.push_str("  PerformanceMonitor,\n");
            imports.push_str("  ValidationError,\n");
            imports.push_str("  CapabilityError,\n");
            imports.push_str("  EffectError,\n");
            imports.push_str("  RuntimeError,\n");
            imports.push_str("} = require('@prism/runtime');\n\n");

            imports.push_str("const {\n");
            imports.push_str("  createSemanticType,\n");
            imports.push_str("  validateBusinessRules,\n");
            imports.push_str("  createProxy,\n");
            imports.push_str("  createBrandedType,\n");
            imports.push_str("} = require('@prism/runtime-utils');\n\n");
        }

        Ok(imports)
    }

    /// Serialize JavaScript features to string
    fn serialize_features(&self) -> JavaScriptResult<String> {
        Ok(format!(
            "{{ esm: {}, asyncAwait: {}, classes: {}, optionalChaining: {}, privateFields: {}, topLevelAwait: {} }}",
            self.config.features.esm,
            self.config.features.async_await,
            self.config.features.classes,
            self.config.features.optional_chaining,
            self.config.features.private_fields,
            self.config.features.top_level_await,
        ))
    }

    /// Generate capabilities object
    fn generate_capabilities_object(&self, pir: &PrismIR) -> JavaScriptResult<String> {
        let capabilities: Vec<String> = pir.modules.iter()
            .flat_map(|m| &m.capabilities)
            .map(|cap| format!("  '{}': ['{}']", cap.name, cap.description))
            .collect();

        Ok(capabilities.join(",\n"))
    }

    /// Generate effects object
    fn generate_effects_object(&self, pir: &PrismIR) -> JavaScriptResult<String> {
        let effects: Vec<String> = pir.effect_graph.nodes.iter()
            .map(|(name, node)| format!(
                "  '{}': {{ type: '{}', capabilities: [{}] }}",
                name,
                node.effect_type,
                node.capabilities.iter()
                    .map(|c| format!("'{}'", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .collect();

        Ok(effects.join(",\n"))
    }

    /// Generate semantic type registry
    fn generate_semantic_type_registry(&self, type_registry: &crate::backends::SemanticTypeRegistry) -> JavaScriptResult<String> {
        let mut output = String::new();

        output.push_str("// === SEMANTIC TYPE REGISTRY ===\n");
        output.push_str("const SEMANTIC_TYPES = new Map();\n\n");

        for (type_name, semantic_type) in &type_registry.types {
            output.push_str(&format!(
                "// Semantic Type: {} - {}\n",
                type_name,
                semantic_type.domain
            ));

            output.push_str(&format!(
                "SEMANTIC_TYPES.set('{}', createSemanticType({{\n",
                type_name
            ));
            output.push_str(&format!("  name: '{}',\n", semantic_type.name));
            output.push_str(&format!("  domain: '{}',\n", semantic_type.domain));
            output.push_str(&format!("  baseType: '{}',\n", self.convert_pir_type_to_js(&semantic_type.base_type)));
            
            // Business rules
            output.push_str("  businessRules: [\n");
            for rule in &semantic_type.business_rules {
                output.push_str(&format!(
                    "    {{ name: '{}', description: '{}', expression: '{}' }},\n",
                    rule.name, rule.description, rule.expression
                ));
            }
            output.push_str("  ],\n");

            // Validation predicates
            output.push_str("  validationPredicates: [\n");
            for predicate in &semantic_type.validation_predicates {
                output.push_str(&format!(
                    "    {{ name: '{}', expression: '{}' }},\n",
                    predicate.name, predicate.expression
                ));
            }
            output.push_str("  ],\n");

            output.push_str("}));\n\n");
        }

        Ok(output)
    }

    /// Convert PIR type to JavaScript representation
    fn convert_pir_type_to_js(&self, pir_type: &crate::backends::PIRTypeInfo) -> String {
        match pir_type {
            crate::backends::PIRTypeInfo::Primitive(prim) => {
                match prim {
                    crate::backends::PIRPrimitiveType::Integer { signed: _, width } => {
                        match width {
                            32 => "number".to_string(),
                            64 if self.config.features.bigint => "bigint".to_string(),
                            64 => "number".to_string(),
                            _ => "number".to_string(),
                        }
                    }
                    crate::backends::PIRPrimitiveType::Float { width: _ } => "number".to_string(),
                    crate::backends::PIRPrimitiveType::Boolean => "boolean".to_string(),
                    crate::backends::PIRPrimitiveType::String => "string".to_string(),
                    crate::backends::PIRPrimitiveType::Unit => "undefined".to_string(),
                }
            }
            crate::backends::PIRTypeInfo::Composite(_) => "object".to_string(),
            crate::backends::PIRTypeInfo::Function(_) => "function".to_string(),
            crate::backends::PIRTypeInfo::Generic(_) => "any".to_string(),
            crate::backends::PIRTypeInfo::Effect(_) => "Effect".to_string(),
        }
    }

    /// Generate PIR module
    async fn generate_pir_module(&mut self, module: &PIRModule, config: &CodeGenConfig) -> JavaScriptResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            r#"
// === MODULE: {} ===
// Enhanced with JavaScript 2025+ features and modern patterns
// Capability Domain: {}
// Business Context: {}
// Cohesion Score: {:.2}

/**
 * Module: {} - {}
 * Responsibility: {}
 * Capabilities: [{}]
 * 
 * Uses modern JavaScript patterns for semantic preservation and runtime safety
 */
"#,
            module.name,
            module.capability,
            module.business_context.domain,
            module.cohesion_score,
            module.name,
            module.business_context.domain,
            module.business_context.description.as_deref().unwrap_or("N/A"),
            module.capabilities.iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));

        // Generate module namespace using modern JavaScript patterns
        if self.config.features.classes {
            output.push_str(&format!(
                "export class {} {{\n",
                self.to_pascal_case(&module.name)
            ));
            output.push_str("  // Module metadata\n");
            output.push_str(&format!("  static MODULE_NAME = '{}';\n", module.name));
            output.push_str(&format!("  static CAPABILITY = '{}';\n", module.capability));
            output.push_str(&format!("  static COHESION_SCORE = {};\n", module.cohesion_score));
            output.push_str(&format!("  static GENERATED_AT = '{}';\n", chrono::Utc::now().to_rfc3339()));
            output.push_str("\n");
        }

        // Generate sections
        for section in &module.sections {
            match section {
                crate::backends::PIRSection::Functions(function_section) => {
                    output.push_str(&self.generate_function_section(function_section, config).await?);
                }
                crate::backends::PIRSection::Constants(constant_section) => {
                    output.push_str(&self.generate_constant_section(constant_section)?);
                }
                crate::backends::PIRSection::Types(type_section) => {
                    output.push_str(&self.generate_type_section(type_section)?);
                }
                _ => {
                    output.push_str(&format!("  // Section type not yet implemented: {:?}\n", section));
                }
            }
        }

        if self.config.features.classes {
            output.push_str("}\n\n");
        }

        Ok(output)
    }

    /// Generate function section
    async fn generate_function_section(&mut self, section: &FunctionSection, config: &CodeGenConfig) -> JavaScriptResult<String> {
        let mut output = String::new();

        output.push_str("  // === FUNCTIONS ===\n");

        for function in &section.functions {
            output.push_str(&self.generate_enhanced_function(function, config).await?);
        }

        Ok(output)
    }

    /// Generate enhanced function with modern JavaScript patterns
    async fn generate_enhanced_function(&mut self, function: &PIRFunction, _config: &CodeGenConfig) -> JavaScriptResult<String> {
        let mut output = String::new();

        // Generate comprehensive JSDoc
        output.push_str("  /**\n");
        output.push_str(&format!("   * {}\n", function.responsibility.as_deref().unwrap_or("Function")));
        output.push_str("   *\n");
        
        // Parameters
        for param in &function.signature.parameters {
            output.push_str(&format!("   * @param {{{}}} {} - Parameter\n", 
                self.convert_pir_type_to_js(&param.param_type), param.name));
        }
        
        // Return type
        let return_type = function.signature.return_type.as_ref()
            .map(|t| self.convert_pir_type_to_js(t))
            .unwrap_or_else(|| "void".to_string());
        output.push_str(&format!("   * @returns {{Promise<{}>}} Function result\n", return_type));
        
        // Capabilities and effects
        if !function.capabilities_required.is_empty() {
            output.push_str("   * @capabilities ");
            output.push_str(&function.capabilities_required.iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", "));
            output.push_str("\n");
        }
        
        if !function.signature.effects.effects.is_empty() {
            output.push_str("   * @effects ");
            output.push_str(&function.signature.effects.effects.iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", "));
            output.push_str("\n");
        }
        
        output.push_str("   */\n");

        // Generate function signature
        let params = function.signature.parameters.iter()
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
            .join(", ");

        if self.config.features.classes {
            output.push_str("  static ");
        }

        if self.config.features.async_await {
            output.push_str("async ");
        }

        output.push_str(&format!("function {}({}) {{\n", function.name, params));

        // Generate capability validation
        if !function.capabilities_required.is_empty() {
            output.push_str("    // Capability validation\n");
            output.push_str("    const capabilityManager = new CapabilityManager(PrismRuntime.getInstance());\n");
            output.push_str("    await capabilityManager.validateCapabilities([\n");
            for capability in &function.capabilities_required {
                output.push_str(&format!("      '{}',\n", capability.name));
            }
            output.push_str("    ]);\n\n");
        }

        // Generate effect tracking
        if !function.signature.effects.effects.is_empty() {
            output.push_str("    // Effect tracking\n");
            output.push_str("    const effectTracker = new EffectTracker();\n");
            output.push_str("    await effectTracker.trackEffects([\n");
            for effect in &function.signature.effects.effects {
                output.push_str(&format!("      '{}',\n", effect.name));
            }
            output.push_str("    ]);\n\n");
        }

        // Performance monitoring
        if self.config.performance_config.enable_monitoring {
            output.push_str("    // Performance monitoring\n");
            output.push_str(&format!("    const perfMonitor = new PerformanceMonitor('{}');\n", function.name));
            output.push_str("    const perfStart = perfMonitor.start();\n\n");
        }

        // Function body
        output.push_str("    try {\n");
        output.push_str("      // Function implementation would be generated here\n");
        output.push_str("      const result = undefined; // Placeholder\n");
        output.push_str("\n");

        // Complete tracking
        if !function.signature.effects.effects.is_empty() {
            output.push_str("      await effectTracker.complete();\n");
        }

        if self.config.performance_config.enable_monitoring {
            output.push_str("      perfMonitor.end(perfStart);\n");
        }

        output.push_str("      return result;\n");
        output.push_str("    } catch (error) {\n");

        // Error handling
        if !function.signature.effects.effects.is_empty() {
            output.push_str("      await effectTracker.abort();\n");
        }

        if self.config.performance_config.enable_monitoring {
            output.push_str("      perfMonitor.recordError(error);\n");
        }

        output.push_str("      throw error;\n");
        output.push_str("    }\n");
        output.push_str("  }\n\n");

        Ok(output)
    }

    /// Generate constant section
    fn generate_constant_section(&self, section: &ConstantSection) -> JavaScriptResult<String> {
        let mut output = String::new();

        output.push_str("  // === CONSTANTS ===\n");

        for constant in &section.constants {
            let value = self.generate_js_constant(&constant.value)?;
            let business_meaning = constant.business_meaning.as_deref().unwrap_or("No description");

            if self.config.features.classes {
                output.push_str(&format!(
                    "  static {} = {}; // {}\n",
                    constant.name.to_uppercase(),
                    value,
                    business_meaning
                ));
            } else {
                output.push_str(&format!(
                    "export const {} = {}; // {}\n",
                    constant.name.to_uppercase(),
                    value,
                    business_meaning
                ));
            }
        }

        output.push_str("\n");
        Ok(output)
    }

    /// Generate type section
    fn generate_type_section(&self, _section: &TypeSection) -> JavaScriptResult<String> {
        let mut output = String::new();
        output.push_str("  // === TYPES ===\n");
        output.push_str("  // Type definitions would be generated here\n\n");
        Ok(output)
    }

    /// Generate JavaScript constant
    fn generate_js_constant(&self, expr: &PIRExpression) -> JavaScriptResult<String> {
        match expr {
            PIRExpression::Literal(lit) => {
                match lit {
                    crate::backends::PIRLiteral::Integer(i) => {
                        if *i > i32::MAX as i64 && self.config.features.bigint {
                            Ok(format!("{}n", i))
                        } else {
                            Ok(i.to_string())
                        }
                    }
                    crate::backends::PIRLiteral::Float(f) => Ok(f.to_string()),
                    crate::backends::PIRLiteral::Boolean(b) => Ok(b.to_string()),
                    crate::backends::PIRLiteral::String(s) => {
                        if self.config.features.template_literals && s.contains("${") {
                            Ok(format!("`{}`", s.replace('`', "\\`")))
                        } else {
                            Ok(format!("'{}'", s.replace('\'', "\\'")))
                        }
                    }
                    crate::backends::PIRLiteral::Unit => Ok("undefined".to_string()),
                }
            }
            _ => Err(JavaScriptError::CodeGeneration {
                message: "Only literal expressions supported for constants".to_string(),
            })
        }
    }

    /// Generate runtime support functions
    fn generate_runtime_support_functions(&self) -> String {
        format!(
            r#"
// === RUNTIME SUPPORT FUNCTIONS ===

/**
 * Initialize the Prism JavaScript runtime
 * @param {{Object}} config - Runtime configuration
 * @returns {{Promise<void>}} Initialization promise
 */
export async function initializePrismRuntime(config = {{}}) {{
  const runtimeConfig = {{
    ...PRISM_RUNTIME_CONFIG,
    ...config,
  }};

  await PrismRuntime.initialize(runtimeConfig);
  
  // Register semantic types
  for (const [name, type] of SEMANTIC_TYPES) {{
    await PrismRuntime.registerSemanticType(name, type);
  }}
  
  console.log(`Prism JavaScript runtime initialized successfully`);
  console.log(`Modules: {{}}, Cohesion Score: {{}}, Features: {{}}`, 
    PRISM_MODULE_COUNT, PRISM_COHESION_SCORE, JSON.stringify(PRISM_RUNTIME_CONFIG.features));
}}

/**
 * Validate business rules for a value
 * @param {{string}} typeName - Semantic type name
 * @param {{any}} value - Value to validate
 * @returns {{Promise<boolean>}} Validation result
 */
export async function validateBusinessRules(typeName, value) {{
  const semanticType = SEMANTIC_TYPES.get(typeName);
  if (!semanticType) {{
    throw new ValidationError(`Unknown semantic type: ${{typeName}}`);
  }}
  
  return await semanticType.validate(value);
}}

/**
 * Create a runtime error with context
 * @param {{string}} message - Error message
 * @param {{Object}} context - Error context
 * @returns {{RuntimeError}} Runtime error
 */
export function createRuntimeError(message, context = {{}}) {{
  return new RuntimeError(message, {{
    ...context,
    timestamp: new Date().toISOString(),
    runtime: 'javascript',
    version: '2.0.0',
  }});
}}

"#
        )
    }

    /// Generate module exports
    fn generate_module_exports(&self) -> String {
        if self.config.features.esm {
            format!(
                r#"
// === MODULE EXPORTS ===
export {{
  PRISM_COHESION_SCORE,
  PRISM_MODULE_COUNT,
  PRISM_GENERATION_TIMESTAMP,
  MODULE_CAPABILITIES,
  MODULE_EFFECTS,
  SEMANTIC_TYPES,
  initializePrismRuntime,
  validateBusinessRules,
  createRuntimeError,
}};

export default {{
  runtime: PrismRuntime,
  cohesionScore: PRISM_COHESION_SCORE,
  moduleCount: PRISM_MODULE_COUNT,
  capabilities: MODULE_CAPABILITIES,
  effects: MODULE_EFFECTS,
  semanticTypes: SEMANTIC_TYPES,
}};
"#
            )
        } else {
            format!(
                r#"
// === MODULE EXPORTS ===
module.exports = {{
  PRISM_COHESION_SCORE,
  PRISM_MODULE_COUNT,
  PRISM_GENERATION_TIMESTAMP,
  MODULE_CAPABILITIES,
  MODULE_EFFECTS,
  SEMANTIC_TYPES,
  initializePrismRuntime,
  validateBusinessRules,
  createRuntimeError,
  
  // Default export
  default: {{
    runtime: PrismRuntime,
    cohesionScore: PRISM_COHESION_SCORE,
    moduleCount: PRISM_MODULE_COUNT,
    capabilities: MODULE_CAPABILITIES,
    effects: MODULE_EFFECTS,
    semanticTypes: SEMANTIC_TYPES,
  }},
}};
"#
            )
        }
    }

    /// Convert string to PascalCase
    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                }
            })
            .collect()
    }
}

#[async_trait]
impl CodeGenBackend for JavaScriptBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::JavaScript
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let _span = span!(Level::INFO, "javascript_pir_codegen").entered();
        let start_time = std::time::Instant::now();

        info!("Generating JavaScript from PIR with modern architecture");

        // Clone self to make it mutable for generation
        let mut backend = self.clone();
        
        // Generate JavaScript code
        let javascript_content = backend.generate_javascript_module(pir, config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "JavaScript".to_string(),
                message: format!("JavaScript generation failed: {:?}", e),
            })?;

        let generation_time = start_time.elapsed().as_millis() as u64;

        // Apply optimizations
        let optimized_content = if config.optimization_level > 0 {
            backend.optimizer.optimize(&javascript_content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "JavaScript".to_string(),
                    message: format!("JavaScript optimization failed: {:?}", e),
                })?
        } else {
            javascript_content
        };

        // Validate the generated code
        let validation_issues = backend.validator.validate(&optimized_content)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "JavaScript".to_string(),
                message: format!("JavaScript validation failed: {:?}", e),
            })?;

        // Generate source map if enabled
        let source_map = if config.source_maps {
            Some(backend.source_map_generator.generate(&optimized_content, "generated.js")
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "JavaScript".to_string(),
                    message: format!("Source map generation failed: {:?}", e),
                })?)
        } else {
            None
        };

        // Generate AI metadata
        let ai_metadata = AIMetadata {
            semantic_types: pir.type_registry.types.keys().cloned()
                .zip(pir.type_registry.types.values().map(|t| t.domain.clone()))
                .collect(),
            business_context: pir.modules.iter()
                .map(|m| (m.name.clone(), m.business_context.domain.clone()))
                .collect(),
            performance_hints: vec![
                "Uses modern JavaScript features for optimal performance".to_string(),
                "Implements capability-based security model".to_string(),
                "Includes comprehensive runtime type validation".to_string(),
            ],
        };

        // Log validation issues as warnings
        for issue in &validation_issues {
            tracing::warn!("JavaScript validation issue: {}", issue);
        }

        Ok(CodeArtifact {
            target: CompilationTarget::JavaScript,
            content: optimized_content.clone(),
            source_map,
            ai_metadata,
            output_path: PathBuf::from("output.js"),
            stats: CodeGenStats {
                lines_generated: optimized_content.lines().count(),
                generation_time,
                optimizations_applied: if config.optimization_level > 0 { 1 } else { 0 },
                memory_usage: optimized_content.len(),
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
        let pir = pir_builder.build_from_program(program)?;
        self.generate_code_from_pir(&pir, context, config).await
    }

    async fn generate_semantic_type(
        &self,
        semantic_type: &PIRSemanticType,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            "/**\n * Semantic Type: {}\n * Domain: {}\n * Security: {:?}\n */\n",
            semantic_type.name, semantic_type.domain, semantic_type.security_classification
        ));
        
        // Generate branded type class using modern JavaScript patterns
        if self.config.features.classes {
            output.push_str(&format!(
                "export class {} extends SemanticType {{\n",
                semantic_type.name
            ));
            
            // Constructor with validation
            output.push_str("  constructor(value) {\n");
            output.push_str("    super({\n");
            output.push_str(&format!("      name: '{}',\n", semantic_type.name));
            output.push_str(&format!("      domain: '{}',\n", semantic_type.domain));
            output.push_str(&format!("      baseType: '{}',\n", self.convert_pir_type_to_js(&semantic_type.base_type)));
            output.push_str("      value\n");
            output.push_str("    });\n");
            output.push_str("    this.validate();\n");
            output.push_str("  }\n\n");
            
            // Validation method
            output.push_str("  validate() {\n");
            for rule in &semantic_type.business_rules {
                output.push_str(&format!(
                    "    // Business rule: {} - {}\n",
                    rule.name, rule.description
                ));
                output.push_str(&format!(
                    "    if (!this.validateBusinessRule('{}', this.value)) {{\n",
                    rule.name
                ));
                output.push_str(&format!(
                    "      throw new ValidationError('Business rule violation: {}');\n",
                    rule.description
                ));
                output.push_str("    }\n");
            }
            
            for predicate in &semantic_type.validation_predicates {
                output.push_str(&format!(
                    "    // Validation predicate: {}\n",
                    predicate.name
                ));
                output.push_str(&format!(
                    "    if (!this.validatePredicate('{}', this.value)) {{\n",
                    predicate.expression
                ));
                output.push_str(&format!(
                    "      throw new ValidationError('Predicate validation failed: {}');\n",
                    predicate.expression
                ));
                output.push_str("    }\n");
            }
            
            output.push_str("    return true;\n");
            output.push_str("  }\n");
            output.push_str("}\n\n");
        }
        
        Ok(output)
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        // Delegate to the enhanced function generator
        let mut backend = self.clone();
        backend.generate_enhanced_function(function, _config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "JavaScript".to_string(),
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
        _pir: &PrismIR,
        _config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        Ok(self.generate_runtime_support_functions())
    }

    async fn optimize(
        &self,
        artifact: &mut CodeArtifact,
        config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        if config.optimization_level > 0 {
            let optimized = self.optimizer.optimize(&artifact.content)
                .map_err(|e| crate::CodeGenError::OptimizationError {
                    target: "JavaScript".to_string(),
                    message: format!("Optimization failed: {:?}", e),
                })?;
            artifact.content = optimized;
            artifact.stats.optimizations_applied += 1;
        }
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        self.validator.validate(&artifact.content)
            .map_err(|e| crate::CodeGenError::ValidationError {
                target: "JavaScript".to_string(),
                message: format!("Validation failed: {:?}", e),
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