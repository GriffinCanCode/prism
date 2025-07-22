//! Core Python Backend Implementation
//!
//! This module provides the main Python backend implementation that integrates
//! all the modular components (types, semantic preservation, runtime integration,
//! validation, optimization, AST generation, dataclass generation, async support, packaging).

use super::{PythonResult, PythonError, PythonBackendConfig, PythonFeatures, PythonTarget};
use super::types::{PythonType, PythonTypeConverter};
use super::semantic_preservation::{SemanticTypePreserver, BusinessRuleGenerator};
use super::runtime_integration::{RuntimeIntegrator, CapabilityManager, EffectTracker};
use super::validation::{PythonValidator, ValidationConfig};
use super::optimization::{PythonOptimizer, OptimizationConfig};
use super::ast_generation::{PythonASTGenerator, ASTConfig};
use super::dataclass_generation::{DataclassGenerator, DataclassConfig};
use super::async_support::{AsyncPatternGenerator, AsyncConfig};
use super::packaging::{PackagingGenerator, PyProjectConfig};

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

/// Python backend with modular architecture
pub struct PythonBackend {
    /// Backend configuration
    config: PythonBackendConfig,
    /// Type converter for PIR to Python type mapping
    type_converter: PythonTypeConverter,
    /// Semantic type preserver
    semantic_preserver: SemanticTypePreserver,
    /// Runtime integrator
    runtime_integrator: RuntimeIntegrator,
    /// Code validator
    validator: PythonValidator,
    /// Code optimizer
    optimizer: PythonOptimizer,
    /// AST generator
    ast_generator: PythonASTGenerator,
    /// Dataclass generator
    dataclass_generator: DataclassGenerator,
    /// Async pattern generator
    async_generator: AsyncPatternGenerator,
    /// Packaging generator
    packaging_generator: PackagingGenerator,
    /// Business rule generator
    business_rule_generator: BusinessRuleGenerator,
}

impl PythonBackend {
    /// Create new Python backend with configuration
    pub fn new(config: CodeGenConfig) -> Self {
        let python_config = PythonBackendConfig::from_codegen_config(&config);
        
        // Create type converter
        let type_converter = PythonTypeConverter::new(python_config.type_config.clone());

        // Create semantic preserver
        let semantic_preserver = SemanticTypePreserver::new(
            python_config.semantic_config.clone(),
            type_converter.clone(),
        );

        // Create runtime integrator
        let runtime_integrator = RuntimeIntegrator::new(python_config.runtime_config.clone());

        // Create validator
        let validator = PythonValidator::new(python_config.validation_config.clone());

        // Create optimizer
        let optimizer = PythonOptimizer::new(python_config.optimization_config.clone());

        // Create AST generator
        let ast_generator = PythonASTGenerator::new(python_config.ast_config.clone());

        // Create dataclass generator
        let dataclass_generator = DataclassGenerator::new(python_config.dataclass_config.clone());

        // Create async pattern generator
        let async_generator = AsyncPatternGenerator::new(python_config.async_config.clone());

        // Create packaging generator
        let packaging_generator = PackagingGenerator::new(python_config.packaging_config.clone());

        // Create business rule generator
        let business_rule_generator = BusinessRuleGenerator::new(python_config.semantic_config.clone());

        Self {
            config: python_config,
            type_converter,
            semantic_preserver,
            runtime_integrator,
            validator,
            optimizer,
            ast_generator,
            dataclass_generator,
            async_generator,
            packaging_generator,
            business_rule_generator,
        }
    }

    /// Configure Python target
    pub fn with_target(mut self, target: PythonTarget) -> Self {
        self.config.target = target;
        // Update dependent components
        self.config.type_config.target_version = target;
        self.type_converter = PythonTypeConverter::new(self.config.type_config.clone());
        self
    }

    /// Configure Python features
    pub fn with_features(mut self, features: PythonFeatures) -> Self {
        self.config.python_features = features.clone();
        self.config.type_config.features = features;
        // Update dependent components
        self.type_converter = PythonTypeConverter::new(self.config.type_config.clone());
        self
    }

    /// Generate complete Python module from PIR
    async fn generate_python_module(&mut self, pir: &PrismIR, config: &CodeGenConfig) -> PythonResult<String> {
        let mut output = String::new();

        // Generate module header with metadata
        output.push_str(&self.generate_module_header(pir)?);

        // Generate imports
        output.push_str(&self.generate_imports(pir)?);

        // Generate runtime integration
        output.push_str(&self.runtime_integrator.generate_runtime_integration(pir)?);

        // Generate semantic type registry
        let semantic_types: Vec<_> = pir.type_registry.types.values().collect();
        output.push_str(&self.generate_semantic_type_registry(&semantic_types)?);

        // Generate business rule system
        output.push_str(&self.business_rule_generator.generate_business_rule_system(&semantic_types)?);

        // Generate PIR modules
        for module in &pir.modules {
            output.push_str(&self.generate_pir_module(module, config).await?);
        }

        // Generate runtime support functions
        output.push_str(&self.generate_runtime_support_functions());

        // Generate main execution block
        output.push_str(&self.generate_main_block());

        Ok(output)
    }

    /// Generate module header with comprehensive metadata
    fn generate_module_header(&self, pir: &PrismIR) -> PythonResult<String> {
        let target_info = self.config.target.to_string();
        
        Ok(format!(
            r#"""
Generated by Prism Compiler - Modular Python Backend

PIR Version: {}
Generated at: {}
Optimization Level: {}
Python Target: {}
Features: Type Hints={}, Dataclasses={}, Pydantic={}, Async={}, Pattern Matching={}

Semantic Metadata:
- Cohesion Score: {:.2}
- Module Count: {}
- Type Registry: {} types
- Effect Registry: {} effects

Business Context:
- AI Metadata Level: {:?}
- Security Classification: Capability-based
- Performance Profile: Dynamic typing with runtime validation

Python 2025 Features:
- Modern type hints with PEP 695 syntax
- Dataclasses and Pydantic models for semantic safety
- Async/await for effect handling
- Structural pattern matching for control flow
- Protocol-based structural typing
"""

# Python version requirement
from __future__ import annotations

import asyncio
import sys
from typing import (
    Any, Dict, List, Optional, Union, Callable, Protocol, TypeVar, Generic,
    Awaitable, Coroutine, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from pathlib import Path

# Pydantic for runtime validation (if available)
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback validation
    class ValidationError(Exception):
        pass

# Type variables for generics
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('prism_runtime')

# Module metadata
PRISM_COHESION_SCORE: float = {:.2}
PRISM_MODULE_COUNT: int = {}
PRISM_GENERATION_TIMESTAMP: str = '{}'
PYTHON_TARGET_VERSION: str = '{}'

"#,
            pir.metadata.version,
            pir.metadata.created_at.as_deref().unwrap_or("unknown"),
            self.config.core_config.optimization_level,
            target_info,
            self.config.python_features.type_hints,
            self.config.python_features.dataclasses,
            self.config.python_features.pydantic_models,
            self.config.python_features.async_await,
            self.config.python_features.pattern_matching,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            pir.type_registry.types.len(),
            pir.effect_graph.nodes.len(),
            self.config.core_config.ai_metadata_level,
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            chrono::Utc::now().to_rfc3339(),
            target_info,
        ))
    }

    /// Generate imports section
    fn generate_imports(&self, pir: &PrismIR) -> PythonResult<String> {
        let mut imports = std::collections::HashSet::new();
        
        // Standard library imports
        imports.insert("import asyncio".to_string());
        imports.insert("import logging".to_string());
        imports.insert("from typing import Any, Dict, List, Optional, Union, Protocol".to_string());
        imports.insert("from dataclasses import dataclass".to_string());
        
        // Prism runtime imports
        imports.insert("# Prism runtime integration".to_string());
        imports.insert("from prism_runtime import (".to_string());
        imports.insert("    PrismRuntime,".to_string());
        imports.insert("    SemanticType,".to_string());
        imports.insert("    EffectTracker,".to_string());
        imports.insert("    CapabilityManager,".to_string());
        imports.insert("    BusinessRuleEngine,".to_string());
        imports.insert("    ValidationError,".to_string());
        imports.insert("    CapabilityError,".to_string());
        imports.insert("    EffectError,".to_string());
        imports.insert(")".to_string());
        
        // Conditional imports based on features
        if self.config.python_features.pydantic_models {
            imports.insert("try:".to_string());
            imports.insert("    from pydantic import BaseModel, Field, validator".to_string());
            imports.insert("except ImportError:".to_string());
            imports.insert("    BaseModel = object  # Fallback".to_string());
        }
        
        let mut import_list: Vec<_> = imports.into_iter().collect();
        import_list.sort();
        
        Ok(format!("{}\n\n", import_list.join("\n")))
    }

    /// Generate semantic type registry
    fn generate_semantic_type_registry(&mut self, semantic_types: &[&PIRSemanticType]) -> PythonResult<String> {
        let mut output = String::new();
        
        output.push_str("# === SEMANTIC TYPE REGISTRY ===\n");
        output.push_str("# Enhanced with Python 3.12+ features and semantic preservation\n\n");

        // Generate each semantic type
        for semantic_type in semantic_types {
            output.push_str(&self.semantic_preserver.generate_semantic_type(semantic_type)?);
        }

        Ok(output)
    }

    /// Generate PIR module as Python code
    async fn generate_pir_module(&mut self, module: &PIRModule, config: &CodeGenConfig) -> PythonResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!(
            r#"
# === MODULE: {} ===
# Enhanced with Python 3.12+ features and modern patterns
# Capability Domain: {}
# Business Context: {}
# Cohesion Score: {:.2}

class {}Module:
    """
    Module namespace with semantic types and capability management.
    Uses modern Python patterns for organization and type safety.
    """
    
    # Module metadata available at runtime
    METADATA = {{
        'name': '{}',
        'capability': '{}',
        'domain': '{}',
        'cohesion_score': {:.2},
        'generated_at': '{}',
    }}
    
    def __init__(self):
        self.runtime = PrismRuntime()
        self.capability_manager = CapabilityManager()
        self.effect_tracker = EffectTracker()
        
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
        ));

        // Generate sections
        for section in &module.sections {
            match section {
                PIRSection::Types(type_section) => {
                    output.push_str("    # === TYPE DEFINITIONS ===\n");
                    for semantic_type in &type_section.types {
                        let dataclass_code = self.dataclass_generator.generate_dataclass(semantic_type)?;
                        // Indent the dataclass code
                        for line in dataclass_code.lines() {
                            output.push_str("    ");
                            output.push_str(line);
                            output.push('\n');
                        }
                    }
                    output.push('\n');
                }
                PIRSection::Functions(function_section) => {
                    output.push_str("    # === FUNCTION DEFINITIONS ===\n");
                    for function in &function_section.functions {
                        output.push_str(&self.generate_python_function(function, config).await?);
                    }
                }
                PIRSection::Constants(constant_section) => {
                    output.push_str("    # === CONSTANTS ===\n");
                    for constant in &constant_section.constants {
                        output.push_str(&self.generate_python_constant(constant).await?);
                    }
                }
                _ => {
                    output.push_str("    # Other sections handled elsewhere\n");
                }
            }
        }

        output.push_str("\n");
        
        Ok(output)
    }

    /// Generate Python function from PIR function
    async fn generate_python_function(&self, function: &PIRFunction, _config: &CodeGenConfig) -> PythonResult<String> {
        let mut output = String::new();
        
        // Generate function documentation
        output.push_str(&format!(
            r#"    async def {}(self"#,
            function.name
        ));
        
        // Generate parameters
        for param in &function.signature.parameters {
            let param_type = self.type_converter.convert_pir_type_to_python(&param.param_type)
                .map_err(|e| PythonError::TypeConversion { message: e.to_string() })?;
            output.push_str(&format!(", {}: {}", param.name, param_type));
        }
        
        // Generate return type
        let return_type = self.type_converter.convert_pir_type_to_python(&function.signature.return_type)
            .map_err(|e| PythonError::TypeConversion { message: e.to_string() })?;
        
        if !matches!(function.signature.return_type.as_ref(), 
            crate::backends::PIRTypeInfo::Primitive(crate::backends::PIRPrimitiveType::Unit)) {
            output.push_str(&format!(") -> {}:\n", return_type));
        } else {
            output.push_str(") -> None:\n");
        }
        
        // Generate docstring
        output.push_str(&format!(
            r#"        """
        Function: {}
        Business Responsibility: {}
        Algorithm: {}
        
        Required Capabilities: [{}]
        Effects: [{}]
        
        Args:
{}
        
        Returns:
            {}: Function result
            
        Raises:
            CapabilityError: If required capabilities are not available
            EffectError: If effects cannot be properly tracked
            ValidationError: If business rules are violated
        """
"#,
            function.name,
            function.responsibility.as_deref().unwrap_or("N/A"),
            function.algorithm.as_deref().unwrap_or("N/A"),
            function.capabilities_required.iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            function.signature.effects.effects.iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            function.signature.parameters.iter()
                .map(|p| format!("            {}: {}", p.name, 
                    self.type_converter.convert_pir_type_to_python(&p.param_type).unwrap_or(PythonType::Any)))
                .collect::<Vec<_>>()
                .join("\n"),
            return_type
        ));
        
        // Generate capability validation
        if !function.capabilities_required.is_empty() {
            output.push_str("        # Capability validation\n");
            output.push_str("        await self.capability_manager.validate_capabilities([\n");
            for capability in &function.capabilities_required {
                output.push_str(&format!("            '{}',\n", capability.name));
            }
            output.push_str("        ])\n\n");
        }
        
        // Generate effect tracking
        if !function.signature.effects.effects.is_empty() {
            output.push_str("        # Effect tracking\n");
            output.push_str("        async with self.effect_tracker.track_effects([\n");
            for effect in &function.signature.effects.effects {
                output.push_str(&format!("            '{}',\n", effect.name));
            }
            output.push_str("        ]):\n");
            output.push_str("            # Function implementation would be generated here\n");
            output.push_str("            result = None  # Placeholder\n");
            output.push_str("            return result\n");
        } else {
            output.push_str("        # Function implementation would be generated here\n");
            output.push_str("        result = None  # Placeholder\n");
            output.push_str("        return result\n");
        }
        
        output.push('\n');
        Ok(output)
    }

    /// Generate Python constant from PIR constant
    async fn generate_python_constant(&self, constant: &crate::backends::PIRConstant) -> PythonResult<String> {
        let const_type = self.type_converter.convert_pir_type_to_python(&constant.const_type)
            .map_err(|e| PythonError::TypeConversion { message: e.to_string() })?;
        let value = self.generate_expression(&constant.value)?;
        
        Ok(format!(
            r#"    # Constant: {}
    # Business meaning: {}
    # Type: {}
    {}: {} = {}

"#,
            constant.name,
            constant.business_meaning.as_deref().unwrap_or("N/A"),
            const_type,
            constant.name.to_uppercase(),
            const_type,
            value
        ))
    }

    /// Generate expression
    fn generate_expression(&self, expr: &PIRExpression) -> PythonResult<String> {
        match expr {
            PIRExpression::Literal(lit) => {
                match lit {
                    crate::backends::PIRLiteral::Integer(i) => Ok(i.to_string()),
                    crate::backends::PIRLiteral::Float(f) => Ok(f.to_string()),
                    crate::backends::PIRLiteral::Boolean(b) => Ok(if *b { "True" } else { "False" }.to_string()),
                    crate::backends::PIRLiteral::String(s) => Ok(format!("'{}'", s.replace('\'', "\\'"))),
                    crate::backends::PIRLiteral::Unit => Ok("None".to_string()),
                }
            }
            _ => Err(PythonError::CodeGeneration {
                message: "Only literal expressions supported for constants".to_string(),
            })
        }
    }

    /// Generate runtime support functions
    fn generate_runtime_support_functions(&self) -> String {
        format!(
            r#"
# === RUNTIME SUPPORT FUNCTIONS ===
# Python-specific runtime integration with comprehensive error handling

class PrismPythonRuntime:
    """
    Python-specific runtime for Prism semantic preservation.
    Integrates with the broader Prism runtime ecosystem.
    """
    
    def __init__(self):
        self.semantic_types: Dict[str, type] = {{}}
        self.business_rules: Dict[str, Callable] = {{}}
        self.effect_registry: Dict[str, Any] = {{}}
        self.capability_registry: Dict[str, Any] = {{}}
        self.logger = logging.getLogger('prism_python_runtime')
        
    def register_semantic_type(self, name: str, type_class: type) -> None:
        """Register a semantic type with the runtime."""
        self.semantic_types[name] = type_class
        self.logger.info(f"Registered semantic type: {{name}}")
        
    def register_business_rule(self, name: str, rule_func: Callable) -> None:
        """Register a business rule validator."""
        self.business_rules[name] = rule_func
        self.logger.info(f"Registered business rule: {{name}}")
        
    def validate_business_rule(self, rule_name: str, value: Any) -> bool:
        """Validate a value against a business rule."""
        if rule_name not in self.business_rules:
            self.logger.warning(f"Business rule not found: {{rule_name}}")
            return True  # Fail open for missing rules
            
        try:
            return self.business_rules[rule_name](value)
        except Exception as e:
            self.logger.error(f"Business rule validation failed: {{rule_name}}, error: {{e}}")
            return False
            
    async def initialize_async_runtime(self) -> None:
        """Initialize async runtime components."""
        self.logger.info("Initializing Prism Python async runtime")
        # Initialize async components here
        
    def get_system_metadata(self) -> Dict[str, Any]:
        """Get comprehensive system metadata for AI consumption."""
        return {{
            'runtime_type': 'python',
            'python_version': sys.version,
            'semantic_types_count': len(self.semantic_types),
            'business_rules_count': len(self.business_rules),
            'cohesion_score': PRISM_COHESION_SCORE,
            'module_count': PRISM_MODULE_COUNT,
            'generation_timestamp': PRISM_GENERATION_TIMESTAMP,
            'target_version': PYTHON_TARGET_VERSION,
        }}

# Global runtime instance
prism_runtime = PrismPythonRuntime()

"#
        )
    }

    /// Generate main execution block
    fn generate_main_block(&self) -> String {
        format!(
            r#"
# === MAIN EXECUTION ===
# Entry point for the generated Prism Python module

async def main() -> None:
    """
    Main execution function for the Prism-generated Python code.
    Initializes runtime and demonstrates module functionality.
    """
    try:
        # Initialize runtime
        await prism_runtime.initialize_async_runtime()
        
        logger.info("Prism Python runtime initialized successfully")
        logger.info(f"Cohesion Score: {{PRISM_COHESION_SCORE:.2f}}")
        logger.info(f"Module Count: {{PRISM_MODULE_COUNT}}")
        
        # Example usage would be generated here
        logger.info("Generated code execution completed successfully")
        
    except Exception as e:
        logger.error(f"Runtime initialization failed: {{e}}")
        raise

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())

"#
        )
    }

    /// Convert string to PascalCase
    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars: Vec<char> = word.chars().collect();
                if !chars.is_empty() {
                    chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
                }
                chars.into_iter().collect::<String>()
            })
            .collect::<String>()
    }
}

#[async_trait]
impl CodeGenBackend for PythonBackend {
    fn target(&self) -> CompilationTarget {
        CompilationTarget::Python
    }

    async fn generate_code_from_pir(
        &self,
        pir: &PrismIR,
        _context: &CompilationContext,
        config: &CodeGenConfig,
    ) -> CodeGenResult<CodeArtifact> {
        let _span = span!(Level::INFO, "python_pir_codegen").entered();
        let start_time = std::time::Instant::now();

        info!("Generating Python from PIR with modular architecture");

        // Clone self to make it mutable for generation
        let mut backend = self.clone();
        
        // Generate Python code
        let python_content = backend.generate_python_module(pir, config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "Python".to_string(),
                message: format!("Python generation failed: {:?}", e),
            })?;

        let generation_time = start_time.elapsed().as_millis() as u64;

        // Apply optimizations
        let optimized_content = if config.optimization_level > 0 {
            backend.optimizer.optimize(&python_content)
                .map_err(|e| crate::CodeGenError::CodeGenerationError {
                    target: "Python".to_string(),
                    message: format!("Python optimization failed: {:?}", e),
                })?
        } else {
            python_content
        };

        // Validate the generated code
        let validation_issues = backend.validator.validate(&optimized_content)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "Python".to_string(),
                message: format!("Python validation failed: {:?}", e),
            })?;

        // Generate AI metadata
        let ai_metadata = AIMetadata {
            semantic_types: pir.type_registry.types.keys()
                .map(|k| (k.clone(), format!("Semantic type: {}", k)))
                .collect(),
            business_context: pir.modules.iter()
                .map(|m| (m.name.clone(), format!("Module: {}, Capability: {}", m.name, m.capability)))
                .collect(),
            performance_hints: vec![
                "Uses Python 3.12+ features for optimal performance".to_string(),
                "Async/await patterns for effect handling".to_string(),
                "Type hints for static analysis and AI comprehension".to_string(),
                "Dataclasses for zero-cost semantic abstractions".to_string(),
            ],
        };

        Ok(CodeArtifact {
            target: CompilationTarget::Python,
            content: optimized_content,
            source_map: None, // Python doesn't typically use source maps
            ai_metadata,
            output_path: PathBuf::from("prism_generated.py"),
            stats: CodeGenStats {
                lines_generated: python_content.lines().count(),
                generation_time,
                optimizations_applied: if config.optimization_level > 0 { 
                    config.optimization_level as usize 
                } else { 0 },
                memory_usage: python_content.len(),
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
                target: "Python".to_string(),
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
                target: "Python".to_string(),
                message: format!("Semantic type generation failed: {:?}", e),
            })
    }

    async fn generate_function_with_effects(
        &self,
        function: &PIRFunction,
        config: &CodeGenConfig,
    ) -> CodeGenResult<String> {
        self.generate_python_function(function, config).await
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "Python".to_string(),
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
        _config: &CodeGenConfig,
    ) -> CodeGenResult<()> {
        // Use our internal optimizer
        let optimized_content = self.optimizer.optimize(&artifact.content)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "Python".to_string(),
                message: format!("Optimization failed: {:?}", e),
            })?;
        
        artifact.content = optimized_content;
        artifact.stats.optimizations_applied += 1;
        
        Ok(())
    }

    async fn validate(&self, artifact: &CodeArtifact) -> CodeGenResult<Vec<String>> {
        let validation_results = self.validator.validate(&artifact.content)
            .map_err(|e| crate::CodeGenError::CodeGenerationError {
                target: "Python".to_string(),
                message: format!("Validation failed: {:?}", e),
            })?;
        
        Ok(validation_results)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            source_maps: false, // Python doesn't typically use source maps
            debug_info: true,
            incremental: true,
            parallel: true,
            optimization_levels: vec![0, 1, 2, 3],
        }
    }
}

impl Clone for PythonBackend {
    fn clone(&self) -> Self {
        Self::new(self.config.core_config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_backend_creation() {
        let config = CodeGenConfig::default();
        let backend = PythonBackend::new(config);
        assert_eq!(backend.target(), CompilationTarget::Python);
    }

    #[test]
    fn test_python_backend_capabilities() {
        let backend = PythonBackend::new(CodeGenConfig::default());
        let caps = backend.capabilities();
        assert!(!caps.source_maps); // Python doesn't use source maps
        assert!(caps.debug_info);
        assert!(caps.incremental);
        assert!(caps.parallel);
        assert_eq!(caps.optimization_levels, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_to_pascal_case() {
        let backend = PythonBackend::new(CodeGenConfig::default());
        assert_eq!(backend.to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(backend.to_pascal_case("user_account"), "UserAccount");
        assert_eq!(backend.to_pascal_case("api"), "Api");
    }
} 