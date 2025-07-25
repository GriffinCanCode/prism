//! Semantic-to-Bytecode Compiler Integration
//!
//! This module handles the compilation of semantic information from prism-semantic
//! into bytecode-compatible formats, ensuring proper preservation without duplicating
//! the semantic analysis logic.

use super::{VMSemanticConfig, SymbolFromStr};
use prism_semantic::{
    SemanticEngine, SemanticResult, SemanticError,
    types::{SemanticType, BusinessRule, TypeConstraint},
    database::SemanticInfo,
    analyzer::{SymbolInfo, TypeInfo},
    type_inference::engine::pir_integration::PIRMetadata,
    validation::ValidationResult,
};
use prism_vm::{
    PrismBytecode,
    bytecode::{
        BytecodeSemanticMetadata, CompiledBusinessRule, CompiledValidationPredicate,
        ValidationMetadata, ValidationConfig, ValidationCost, OptimizationHint,
        OptimizationHintType, SecurityContext, SecurityClassification,
        BusinessRuleCategory, SemanticMetadataBuilder, SemanticInformationRegistry,
        Instruction, PrismOpcode,
    },
};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::sync::Arc;

/// Semantic-to-bytecode compiler that integrates with prism-semantic
#[derive(Debug)]
pub struct SemanticBytecodeCompiler {
    /// Reference to semantic engine
    semantic_engine: Arc<SemanticEngine>,
    /// VM semantic configuration
    config: VMSemanticConfig,
    /// Business rule compiler
    business_rule_compiler: BusinessRuleCompiler,
    /// Validation predicate compiler
    validation_compiler: ValidationPredicateCompiler,
    /// Metadata builder
    metadata_builder: SemanticMetadataBuilder,
}

/// Compiles business rules from semantic analysis into bytecode
#[derive(Debug)]
struct BusinessRuleCompiler {
    /// Compilation cache
    compilation_cache: HashMap<String, CompiledBusinessRule>,
}

/// Compiles validation predicates from semantic analysis into bytecode
#[derive(Debug)]
struct ValidationPredicateCompiler {
    /// Compilation cache
    compilation_cache: HashMap<String, CompiledValidationPredicate>,
}

impl SemanticBytecodeCompiler {
    /// Create a new semantic bytecode compiler
    pub fn new(
        semantic_engine: Arc<SemanticEngine>,
        config: VMSemanticConfig,
    ) -> SemanticResult<Self> {
        Ok(Self {
            semantic_engine,
            config,
            business_rule_compiler: BusinessRuleCompiler::new(),
            validation_compiler: ValidationPredicateCompiler::new(),
            metadata_builder: SemanticMetadataBuilder::new(),
        })
    }

    /// Compile semantic information into bytecode format
    pub fn compile_semantic_info_to_bytecode(
        &mut self,
        mut base_bytecode: PrismBytecode,
        semantic_info: &SemanticInfo,
        pir_metadata: Option<&PIRMetadata>,
    ) -> SemanticResult<(PrismBytecode, SemanticInformationRegistry)> {
        // Initialize semantic registry
        let mut semantic_registry = SemanticInformationRegistry::new();

        // Step 1: Compile business rules from semantic analysis
        let compiled_business_rules = self.compile_business_rules_from_semantic_info(semantic_info)?;
        
        // Step 2: Compile validation predicates
        let compiled_validation_predicates = self.compile_validation_predicates_from_semantic_info(semantic_info)?;

        // Step 3: Build comprehensive semantic metadata
        let semantic_metadata = self.build_semantic_metadata_from_analysis(
            semantic_info,
            pir_metadata,
            &compiled_business_rules,
            &compiled_validation_predicates,
        )?;

        // Step 4: Enhance bytecode with semantic information
        self.enhance_bytecode_with_semantics(
            &mut base_bytecode,
            &semantic_metadata,
            &compiled_business_rules,
            &compiled_validation_predicates,
        )?;

        // Step 5: Populate semantic registry for runtime access
        self.populate_semantic_registry(
            &mut semantic_registry,
            semantic_info,
            &semantic_metadata,
            &compiled_business_rules,
            &compiled_validation_predicates,
        )?;

        Ok((base_bytecode, semantic_registry))
    }

    /// Compile business rules from semantic analysis results
    fn compile_business_rules_from_semantic_info(
        &mut self,
        semantic_info: &SemanticInfo,
    ) -> SemanticResult<Vec<CompiledBusinessRule>> {
        let mut compiled_rules = Vec::new();

        // Extract business rules from symbol information
        for (symbol, symbol_info) in &semantic_info.symbols {
            if let Some(ref business_context) = symbol_info.business_context {
                let rule_id = format!("symbol_business_rule_{}", symbol_info.name);
                
                if let Some(cached_rule) = self.business_rule_compiler.compilation_cache.get(&rule_id) {
                    compiled_rules.push(cached_rule.clone());
                    continue;
                }

                let compiled_rule = self.business_rule_compiler.compile_business_context_rule(
                    &rule_id,
                    &symbol_info.name,
                    business_context,
                    symbol_info.location,
                )?;

                self.business_rule_compiler.compilation_cache.insert(rule_id, compiled_rule.clone());
                compiled_rules.push(compiled_rule);
            }
        }

        // Extract business rules from validation results
        if let Some(ref validation_result) = semantic_info.validation_result {
            for rule_violation in &validation_result.rule_violations {
                let rule_id = format!("validation_business_rule_{}", rule_violation.rule.id);
                
                if let Some(cached_rule) = self.business_rule_compiler.compilation_cache.get(&rule_id) {
                    compiled_rules.push(cached_rule.clone());
                    continue;
                }

                let compiled_rule = self.business_rule_compiler.compile_validation_business_rule(
                    &rule_violation.rule,
                    rule_violation.location,
                )?;

                self.business_rule_compiler.compilation_cache.insert(rule_id, compiled_rule.clone());
                compiled_rules.push(compiled_rule);
            }
        }

        Ok(compiled_rules)
    }

    /// Compile validation predicates from semantic analysis
    fn compile_validation_predicates_from_semantic_info(
        &mut self,
        semantic_info: &SemanticInfo,
    ) -> SemanticResult<Vec<CompiledValidationPredicate>> {
        let mut compiled_predicates = Vec::new();

        // Extract validation predicates from validation results
        if let Some(ref validation_result) = semantic_info.validation_result {
            for constraint_violation in &validation_result.constraint_violations {
                let predicate_id = format!("constraint_validation_{}", constraint_violation.constraint);
                
                if let Some(cached_predicate) = self.validation_compiler.compilation_cache.get(&predicate_id) {
                    compiled_predicates.push(cached_predicate.clone());
                    continue;
                }

                let compiled_predicate = self.validation_compiler.compile_constraint_validation(
                    &predicate_id,
                    &constraint_violation.constraint,
                    &constraint_violation.description,
                    constraint_violation.location,
                )?;

                self.validation_compiler.compilation_cache.insert(predicate_id, compiled_predicate.clone());
                compiled_predicates.push(compiled_predicate);
            }
        }

        // Extract validation predicates from type constraints
        for (node_id, type_info) in &semantic_info.types {
            if let Some(ref semantic_meaning) = type_info.semantic_meaning {
                let predicate_id = format!("type_semantic_validation_{}", node_id);
                
                if let Some(cached_predicate) = self.validation_compiler.compilation_cache.get(&predicate_id) {
                    compiled_predicates.push(cached_predicate.clone());
                    continue;
                }

                let compiled_predicate = self.validation_compiler.compile_semantic_type_validation(
                    &predicate_id,
                    semantic_meaning,
                    type_info.location,
                )?;

                self.validation_compiler.compilation_cache.insert(predicate_id, compiled_predicate.clone());
                compiled_predicates.push(compiled_predicate);
            }
        }

        Ok(compiled_predicates)
    }

    /// Build comprehensive semantic metadata from analysis results
    fn build_semantic_metadata_from_analysis(
        &mut self,
        semantic_info: &SemanticInfo,
        pir_metadata: Option<&PIRMetadata>,
        compiled_business_rules: &[CompiledBusinessRule],
        compiled_validation_predicates: &[CompiledValidationPredicate],
    ) -> SemanticResult<BytecodeSemanticMetadata> {
        self.metadata_builder
            .with_original_semantic_types(self.extract_semantic_types_from_info(semantic_info)?)
            .with_business_rules(compiled_business_rules.to_vec())
            .with_validation_predicates(compiled_validation_predicates.to_vec())
            .with_ai_context(self.extract_ai_context_from_info(semantic_info)?)
            .with_optimization_hints(self.generate_optimization_hints_from_analysis(semantic_info, pir_metadata)?)
            .with_security_context(self.extract_security_context_from_info(semantic_info)?)
            .build()
    }

    /// Extract semantic types from semantic info
    fn extract_semantic_types_from_info(
        &self,
        semantic_info: &SemanticInfo,
    ) -> SemanticResult<HashMap<String, prism_semantic::types::SemanticType>> {
        let mut semantic_types = HashMap::new();

        for (symbol, symbol_info) in &semantic_info.symbols {
            // Convert symbol info to semantic type
            // This is a simplified conversion - in practice, we'd have more sophisticated mapping
            let semantic_type = prism_semantic::types::SemanticType::primitive(
                &symbol_info.name,
                prism_ast::PrimitiveType::String, // Simplified
                symbol_info.location,
            );
            
            semantic_types.insert(symbol_info.name.clone(), semantic_type);
        }

        Ok(semantic_types)
    }

    /// Extract AI context from semantic info
    fn extract_ai_context_from_info(
        &self,
        semantic_info: &SemanticInfo,
    ) -> SemanticResult<HashMap<String, String>> {
        let mut ai_context = HashMap::new();

        // Extract AI hints from symbols
        for symbol_info in semantic_info.symbols.values() {
            if !symbol_info.ai_hints.is_empty() {
                ai_context.insert(
                    format!("symbol_{}", symbol_info.name),
                    symbol_info.ai_hints.join("; "),
                );
            }
        }

        // Extract AI descriptions from types
        for (node_id, type_info) in &semantic_info.types {
            if let Some(ref ai_description) = type_info.ai_description {
                ai_context.insert(
                    format!("type_{}", node_id),
                    ai_description.clone(),
                );
            }
        }

        // Extract AI metadata if available
        if let Some(ref ai_metadata) = semantic_info.ai_metadata {
            ai_context.insert(
                "global_business_context".to_string(),
                ai_metadata.business_context.clone().unwrap_or_default(),
            );
        }

        Ok(ai_context)
    }

    /// Generate optimization hints from semantic analysis
    fn generate_optimization_hints_from_analysis(
        &self,
        semantic_info: &SemanticInfo,
        pir_metadata: Option<&PIRMetadata>,
    ) -> SemanticResult<Vec<OptimizationHint>> {
        let mut hints = Vec::new();

        // Generate hints from symbol analysis
        for symbol_info in semantic_info.symbols.values() {
            if let Some(ref business_context) = symbol_info.business_context {
                if business_context.contains("performance") || business_context.contains("hot") {
                    hints.push(OptimizationHint {
                        hint_type: OptimizationHintType::HotPath,
                        target_function: Some(symbol_info.name.clone()),
                        expected_benefit: "High".to_string(),
                        implementation_cost: "Medium".to_string(),
                        confidence: 0.8,
                        metadata: HashMap::from([
                            ("business_context".to_string(), business_context.clone()),
                        ]),
                    });
                }
            }
        }

        // Generate hints from PIR metadata if available
        if let Some(pir_metadata) = pir_metadata {
            for optimization_hint in &pir_metadata.construction_hints.optimization_opportunities {
                hints.push(OptimizationHint {
                    hint_type: match optimization_hint.optimization_type.as_str() {
                        "inline" => OptimizationHintType::InlineCandidate,
                        "specialize" => OptimizationHintType::SpecializationCandidate,
                        "vectorize" => OptimizationHintType::VectorizationCandidate,
                        _ => OptimizationHintType::General,
                    },
                    target_function: Some(optimization_hint.target.clone()),
                    expected_benefit: optimization_hint.expected_benefit.clone(),
                    implementation_cost: optimization_hint.implementation_cost.clone(),
                    confidence: 0.9, // PIR metadata is high confidence
                    metadata: HashMap::new(),
                });
            }
        }

        Ok(hints)
    }

    /// Extract security context from semantic info
    fn extract_security_context_from_info(
        &self,
        semantic_info: &SemanticInfo,
    ) -> SemanticResult<SecurityContext> {
        // Analyze symbols for security implications
        let mut has_sensitive_data = false;
        let mut requires_validation = false;

        for symbol_info in semantic_info.symbols.values() {
            // Check for security-related AI hints
            for hint in &symbol_info.ai_hints {
                if hint.contains("security") || hint.contains("sensitive") || hint.contains("auth") {
                    has_sensitive_data = true;
                }
                if hint.contains("validation") || hint.contains("verify") {
                    requires_validation = true;
                }
            }

            // Check business context for security implications
            if let Some(ref business_context) = symbol_info.business_context {
                if business_context.contains("Authentication") || business_context.contains("Financial") {
                    has_sensitive_data = true;
                    requires_validation = true;
                }
            }
        }

        let classification = if has_sensitive_data {
            SecurityClassification::Sensitive
        } else {
            SecurityClassification::Public
        };

        Ok(SecurityContext {
            classification,
            requires_validation,
            access_controls: Vec::new(), // Would be populated from actual security analysis
            audit_requirements: Vec::new(), // Would be populated from compliance analysis
        })
    }

    /// Enhance bytecode with semantic information
    fn enhance_bytecode_with_semantics(
        &mut self,
        bytecode: &mut PrismBytecode,
        semantic_metadata: &BytecodeSemanticMetadata,
        compiled_business_rules: &[CompiledBusinessRule],
        compiled_validation_predicates: &[CompiledValidationPredicate],
    ) -> SemanticResult<()> {
        // Ensure semantic registry is initialized
        if bytecode.semantic_registry.is_none() {
            bytecode.semantic_registry = Some(SemanticInformationRegistry::new());
        }

        // Add semantic metadata to bytecode
        bytecode.semantic_metadata = Some(semantic_metadata.clone());

        // Enhance type definitions with semantic information
        for type_def in &mut bytecode.types {
            self.enhance_type_definition_with_semantics(
                type_def,
                semantic_metadata,
                compiled_business_rules,
                compiled_validation_predicates,
            )?;
        }

        Ok(())
    }

    /// Enhance individual type definition with semantic information
    fn enhance_type_definition_with_semantics(
        &self,
        type_def: &mut prism_vm::bytecode::TypeDefinition,
        semantic_metadata: &BytecodeSemanticMetadata,
        compiled_business_rules: &[CompiledBusinessRule],
        compiled_validation_predicates: &[CompiledValidationPredicate],
    ) -> SemanticResult<()> {
        // Add semantic metadata to type definition
        if type_def.semantic_metadata.is_none() {
            type_def.semantic_metadata = Some(semantic_metadata.clone());
        }

        // Add relevant business rules
        for business_rule in compiled_business_rules {
            if business_rule.applies_to_type(&type_def.name) {
                type_def.compiled_business_rules.push(business_rule.clone());
            }
        }

        // Add relevant validation predicates
        for validation_predicate in compiled_validation_predicates {
            if validation_predicate.applies_to_type(&type_def.name) {
                type_def.compiled_validation_predicates.push(validation_predicate.clone());
            }
        }

        Ok(())
    }

    /// Populate semantic registry for runtime access
    fn populate_semantic_registry(
        &mut self,
        registry: &mut SemanticInformationRegistry,
        semantic_info: &SemanticInfo,
        semantic_metadata: &BytecodeSemanticMetadata,
        compiled_business_rules: &[CompiledBusinessRule],
        compiled_validation_predicates: &[CompiledValidationPredicate],
    ) -> SemanticResult<()> {
        // Register semantic metadata
        registry.register_global_metadata(semantic_metadata.clone());

        // Register business rules
        for business_rule in compiled_business_rules {
            registry.register_business_rule(business_rule.clone());
        }

        // Register validation predicates
        for validation_predicate in compiled_validation_predicates {
            registry.register_validation_predicate(validation_predicate.clone());
        }

        // Register type mappings
        for (symbol, symbol_info) in &semantic_info.symbols {
            registry.register_symbol_mapping(
                symbol_info.name.clone(),
                symbol_info.location,
                symbol_info.business_context.clone(),
            );
        }

        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: VMSemanticConfig) -> SemanticResult<()> {
        self.config = new_config;
        Ok(())
    }
}

impl BusinessRuleCompiler {
    fn new() -> Self {
        Self {
            compilation_cache: HashMap::new(),
        }
    }

    fn compile_business_context_rule(
        &self,
        rule_id: &str,
        symbol_name: &str,
        business_context: &str,
        location: Span,
    ) -> SemanticResult<CompiledBusinessRule> {
        // Generate bytecode for business context validation
        let validation_bytecode = vec![
            Instruction::new(PrismOpcode::LoadLocal, vec![0], location),
            Instruction::new(PrismOpcode::LoadConstant, vec![1], location), // Business context check
            Instruction::new(PrismOpcode::Call, vec![2], location), // Call validation function
            Instruction::new(PrismOpcode::ReturnValue, vec![], location),
        ];

        Ok(CompiledBusinessRule {
            rule_id: rule_id.to_string(),
            rule_name: format!("Business context rule for {}", symbol_name),
            description: format!("Validates business context: {}", business_context),
            category: BusinessRuleCategory::BusinessLogic,
            enforcement_level: prism_semantic::types::EnforcementLevel::Runtime,
            validation_bytecode,
            validation_metadata: ValidationMetadata {
                expected_types: vec![symbol_name.to_string()],
                validation_cost: ValidationCost::Low,
                can_cache_result: true,
                side_effects: false,
            },
            error_message: format!("Business context validation failed for {}", symbol_name),
            confidence: 0.8,
        })
    }

    fn compile_validation_business_rule(
        &self,
        business_rule: &prism_semantic::analyzer::BusinessRule,
        location: Span,
    ) -> SemanticResult<CompiledBusinessRule> {
        // Generate bytecode for business rule validation
        let validation_bytecode = vec![
            Instruction::new(PrismOpcode::LoadLocal, vec![0], location),
            Instruction::new(PrismOpcode::LoadConstant, vec![1], location), // Rule predicate
            Instruction::new(PrismOpcode::Call, vec![2], location), // Call validation function
            Instruction::new(PrismOpcode::ReturnValue, vec![], location),
        ];

        Ok(CompiledBusinessRule {
            rule_id: business_rule.rule_name.clone(),
            rule_name: business_rule.rule_name.clone(),
            description: business_rule.description.clone(),
            category: match business_rule.rule_type.as_str() {
                "business_context" => BusinessRuleCategory::BusinessLogic,
                "validation" => BusinessRuleCategory::DataValidation,
                "security" => BusinessRuleCategory::Security,
                _ => BusinessRuleCategory::General,
            },
            enforcement_level: prism_semantic::types::EnforcementLevel::Runtime,
            validation_bytecode,
            validation_metadata: ValidationMetadata {
                expected_types: Vec::new(),
                validation_cost: ValidationCost::Medium,
                can_cache_result: true,
                side_effects: false,
            },
            error_message: format!("Business rule validation failed: {}", business_rule.rule_name),
            confidence: business_rule.confidence,
        })
    }
}

impl ValidationPredicateCompiler {
    fn new() -> Self {
        Self {
            compilation_cache: HashMap::new(),
        }
    }

    fn compile_constraint_validation(
        &self,
        predicate_id: &str,
        constraint: &str,
        description: &str,
        location: Span,
    ) -> SemanticResult<CompiledValidationPredicate> {
        // Generate bytecode for constraint validation
        let validation_bytecode = vec![
            Instruction::new(PrismOpcode::LoadLocal, vec![0], location),
            Instruction::new(PrismOpcode::LoadConstant, vec![1], location), // Constraint check
            Instruction::new(PrismOpcode::Call, vec![2], location), // Call validation function
            Instruction::new(PrismOpcode::ReturnValue, vec![], location),
        ];

        Ok(CompiledValidationPredicate {
            predicate_id: predicate_id.to_string(),
            predicate_name: format!("Constraint validation: {}", constraint),
            description: description.to_string(),
            validation_bytecode,
            validation_config: ValidationConfig {
                strict_mode: true,
                fail_fast: false,
                cache_results: true,
                log_violations: true,
            },
            error_message: format!("Constraint validation failed: {}", constraint),
            confidence: 0.9,
        })
    }

    fn compile_semantic_type_validation(
        &self,
        predicate_id: &str,
        semantic_meaning: &str,
        location: Span,
    ) -> SemanticResult<CompiledValidationPredicate> {
        // Generate bytecode for semantic type validation
        let validation_bytecode = vec![
            Instruction::new(PrismOpcode::LoadLocal, vec![0], location),
            Instruction::new(PrismOpcode::LoadConstant, vec![1], location), // Semantic check
            Instruction::new(PrismOpcode::Call, vec![2], location), // Call validation function
            Instruction::new(PrismOpcode::ReturnValue, vec![], location),
        ];

        Ok(CompiledValidationPredicate {
            predicate_id: predicate_id.to_string(),
            predicate_name: format!("Semantic type validation: {}", semantic_meaning),
            description: format!("Validates semantic meaning: {}", semantic_meaning),
            validation_bytecode,
            validation_config: ValidationConfig {
                strict_mode: false,
                fail_fast: false,
                cache_results: true,
                log_violations: false,
            },
            error_message: format!("Semantic type validation failed: {}", semantic_meaning),
            confidence: 0.7,
        })
    }
}

// Extension trait for CompiledBusinessRule
trait BusinessRuleTypeApplicability {
    fn applies_to_type(&self, type_name: &str) -> bool;
}

impl BusinessRuleTypeApplicability for CompiledBusinessRule {
    fn applies_to_type(&self, type_name: &str) -> bool {
        // Simple heuristic - in practice, this would be more sophisticated
        self.rule_name.contains(type_name) || 
        self.description.contains(type_name) ||
        self.validation_metadata.expected_types.contains(&type_name.to_string())
    }
}

// Extension trait for CompiledValidationPredicate
trait ValidationPredicateTypeApplicability {
    fn applies_to_type(&self, type_name: &str) -> bool;
}

impl ValidationPredicateTypeApplicability for CompiledValidationPredicate {
    fn applies_to_type(&self, type_name: &str) -> bool {
        // Simple heuristic - in practice, this would be more sophisticated
        self.predicate_name.contains(type_name) || 
        self.description.contains(type_name)
    }
} 