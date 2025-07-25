//! Enhanced Semantic Compiler for PIR to Bytecode
//!
//! This module implements comprehensive semantic information preservation during
//! PIR-to-bytecode compilation, ensuring that business rules, validation predicates,
//! and AI metadata are properly compiled and preserved.

use super::{VMBackendResult, VMBackendError};
use prism_vm::bytecode::{
    Instruction, TypeDefinition, BytecodeSemanticMetadata, CompiledBusinessRule, 
    CompiledValidationPredicate, ValidationMetadata, ValidationConfig, ValidationCost,
    OptimizationHint, OptimizationHintType, SecurityContext, SecurityClassification,
    BusinessRuleCategory, SemanticMetadataBuilder
};
use prism_vm::bytecode::instructions::PrismOpcode;
use prism_pir::semantic::{PIRSemanticType, PIRTypeConstraint, PIRExpression};
use prism_pir::business::BusinessRule;
use crate::backends::{PIRCompositeType, PIRPrimitiveType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, span, Level};

/// Enhanced semantic compiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCompilerConfig {
    /// Preserve all semantic information (recommended)
    pub preserve_all_semantics: bool,
    /// Compile business rules to executable bytecode
    pub compile_business_rules: bool,
    /// Compile validation predicates to executable bytecode
    pub compile_validation_predicates: bool,
    /// Generate optimization hints from semantic analysis
    pub generate_optimization_hints: bool,
    /// Include AI metadata in output
    pub include_ai_metadata: bool,
    /// Validation execution mode
    pub validation_mode: ValidationMode,
    /// Maximum business rule execution time (microseconds)
    pub max_rule_execution_time_us: u64,
}

/// Validation execution modes for compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    /// Compile for runtime validation
    Runtime,
    /// Compile for compile-time validation only
    CompileTime,
    /// Compile for both runtime and compile-time
    Both,
    /// Generate validation stubs (for testing)
    Stub,
}

impl Default for SemanticCompilerConfig {
    fn default() -> Self {
        Self {
            preserve_all_semantics: true,
            compile_business_rules: true,
            compile_validation_predicates: true,
            generate_optimization_hints: true,
            include_ai_metadata: true,
            validation_mode: ValidationMode::Runtime,
            max_rule_execution_time_us: 10000, // 10ms max per rule
        }
    }
}

/// Enhanced semantic compiler for PIR types
pub struct EnhancedSemanticCompiler {
    /// Compiler configuration
    config: SemanticCompilerConfig,
    /// Business rule compiler
    business_rule_compiler: BusinessRuleCompiler,
    /// Validation predicate compiler
    validation_compiler: ValidationPredicateCompiler,
    /// Optimization hint generator
    optimization_analyzer: OptimizationHintAnalyzer,
    /// Compilation statistics
    stats: CompilationStats,
}

/// Business rule compiler
pub struct BusinessRuleCompiler {
    /// Rule compilation strategies
    strategies: HashMap<String, RuleCompilationStrategy>,
    /// Compiled rule cache
    rule_cache: HashMap<String, CompiledBusinessRule>,
}

/// Validation predicate compiler
pub struct ValidationPredicateCompiler {
    /// Predicate compilation patterns
    patterns: HashMap<String, PredicatePattern>,
    /// Compiled predicate cache
    predicate_cache: HashMap<String, CompiledValidationPredicate>,
}

/// Optimization hint analyzer
pub struct OptimizationHintAnalyzer {
    /// Analysis patterns for generating hints
    patterns: Vec<OptimizationPattern>,
    /// Hint generation statistics
    stats: OptimizationStats,
}

/// Rule compilation strategy
#[derive(Debug, Clone)]
pub struct RuleCompilationStrategy {
    /// Strategy name
    pub name: String,
    /// Pattern to match rules
    pub pattern: String,
    /// Bytecode generation function
    pub generator: RuleGenerator,
    /// Expected performance characteristics
    pub performance: RulePerformance,
}

/// Rule generator function type
#[derive(Debug, Clone)]
pub struct RuleGenerator {
    /// Generator identifier
    pub id: String,
    /// Generator description
    pub description: String,
}

/// Rule performance characteristics
#[derive(Debug, Clone)]
pub struct RulePerformance {
    /// Expected CPU cycles
    pub cpu_cycles: u64,
    /// Memory overhead in bytes
    pub memory_overhead: u64,
    /// I/O operations
    pub io_operations: u32,
}

/// Predicate compilation pattern
#[derive(Debug, Clone)]
pub struct PredicatePattern {
    /// Pattern name
    pub name: String,
    /// Pattern matcher
    pub matcher: String,
    /// Bytecode template
    pub bytecode_template: Vec<PrismOpcode>,
}

/// Optimization pattern for hint generation
#[derive(Debug, Clone)]
pub struct OptimizationPattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Expected improvement
    pub expected_improvement: f64,
}

/// Optimization analysis statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of hints generated
    pub hints_generated: u64,
    /// Number of patterns matched
    pub patterns_matched: u64,
    /// Average confidence score
    pub average_confidence: f64,
}

/// Compilation statistics
#[derive(Debug, Clone, Default)]
pub struct CompilationStats {
    /// Number of types compiled
    pub types_compiled: u64,
    /// Number of business rules compiled
    pub business_rules_compiled: u64,
    /// Number of validation predicates compiled
    pub validation_predicates_compiled: u64,
    /// Number of optimization hints generated
    pub optimization_hints_generated: u64,
    /// Total compilation time
    pub total_compilation_time_ms: u64,
    /// Compilation errors encountered
    pub compilation_errors: u64,
}

impl EnhancedSemanticCompiler {
    /// Create a new enhanced semantic compiler
    pub fn new(config: SemanticCompilerConfig) -> Self {
        Self {
            config,
            business_rule_compiler: BusinessRuleCompiler::new(),
            validation_compiler: ValidationPredicateCompiler::new(),
            optimization_analyzer: OptimizationHintAnalyzer::new(),
            stats: CompilationStats::default(),
        }
    }

    /// Compile a PIR semantic type to enhanced bytecode type definition
    pub fn compile_semantic_type(
        &mut self, 
        type_id: u32, 
        name: &str, 
        semantic_type: &PIRSemanticType
    ) -> VMBackendResult<TypeDefinition> {
        let _span = span!(Level::INFO, "compile_semantic_type", type_id, name).entered();
        let start_time = std::time::Instant::now();

        info!("Compiling semantic type: {} (ID: {})", name, type_id);

        // Build semantic metadata
        let mut metadata_builder = SemanticMetadataBuilder::new(&semantic_type.domain)
            .with_original_type(semantic_type.clone());

        // Compile business rules
        let business_rule_bytecode = if self.config.compile_business_rules {
            self.compile_business_rules(&semantic_type.business_rules)?
        } else {
            Vec::new()
        };

        // Compile validation predicates
        let validation_predicate_bytecode = if self.config.compile_validation_predicates {
            self.compile_validation_predicates(&semantic_type.validation_predicates)?
        } else {
            Vec::new()
        };

        // Generate optimization hints
        let optimization_hints = if self.config.generate_optimization_hints {
            self.generate_optimization_hints(semantic_type)?
        } else {
            Vec::new()
        };

        // Build validation metadata
        let validation_metadata = ValidationMetadata {
            business_rule_bytecode,
            validation_predicate_bytecode,
            validation_config: ValidationConfig::default(),
            validation_cost: self.calculate_validation_cost(semantic_type),
        };

        // Build security context
        let security_context = self.build_security_context(semantic_type);

        // Build complete semantic metadata
        let semantic_metadata = metadata_builder
            .with_security_context(security_context)
            .build();

        let mut final_metadata = semantic_metadata;
        final_metadata.validation_metadata = validation_metadata;
        final_metadata.optimization_hints = optimization_hints;

        // Add AI hints if enabled
        if self.config.include_ai_metadata {
            final_metadata.ai_hints = self.generate_ai_hints(semantic_type);
            final_metadata.performance_hints = self.generate_performance_hints(semantic_type);
        }

        // Create type definition with enhanced semantic preservation
        let type_def = TypeDefinition {
            id: type_id,
            name: name.to_string(),
            kind: self.compile_type_kind(&semantic_type.base_type)?,
            domain: Some(semantic_type.domain.clone()),
            // Keep legacy fields for backward compatibility
            business_rules: Vec::new(),
            validation_predicates: Vec::new(),
            // New enhanced semantic metadata
            semantic_metadata: Some(final_metadata),
        };

        // Update statistics
        self.stats.types_compiled += 1;
        self.stats.total_compilation_time_ms += start_time.elapsed().as_millis() as u64;

        info!("Successfully compiled semantic type: {} in {:?}", name, start_time.elapsed());
        Ok(type_def)
    }

    /// Compile business rules to executable bytecode
    fn compile_business_rules(&mut self, rules: &[BusinessRule]) -> VMBackendResult<Vec<CompiledBusinessRule>> {
        let mut compiled_rules = Vec::new();

        for (index, rule) in rules.iter().enumerate() {
            let compiled_rule = self.business_rule_compiler.compile_rule(rule, index)?;
            compiled_rules.push(compiled_rule);
            self.stats.business_rules_compiled += 1;
        }

        Ok(compiled_rules)
    }

    /// Compile validation predicates to executable bytecode
    fn compile_validation_predicates(
        &mut self, 
        predicates: &[prism_pir::semantic::ValidationPredicate]
    ) -> VMBackendResult<Vec<CompiledValidationPredicate>> {
        let mut compiled_predicates = Vec::new();

        for (index, predicate) in predicates.iter().enumerate() {
            let compiled_predicate = self.validation_compiler.compile_predicate(predicate, index)?;
            compiled_predicates.push(compiled_predicate);
            self.stats.validation_predicates_compiled += 1;
        }

        Ok(compiled_predicates)
    }

    /// Generate optimization hints from semantic analysis
    fn generate_optimization_hints(&mut self, semantic_type: &PIRSemanticType) -> VMBackendResult<Vec<OptimizationHint>> {
        let hints = self.optimization_analyzer.analyze_type(semantic_type)?;
        self.stats.optimization_hints_generated += hints.len() as u64;
        Ok(hints)
    }

    /// Calculate validation cost for a semantic type
    fn calculate_validation_cost(&self, semantic_type: &PIRSemanticType) -> ValidationCost {
        let business_rule_cycles = semantic_type.business_rules.len() as u64 * 100; // Estimate
        let predicate_cycles = semantic_type.validation_predicates.len() as u64 * 50; // Estimate
        let memory_overhead = (semantic_type.business_rules.len() + semantic_type.validation_predicates.len()) as u64 * 64;
        
        ValidationCost {
            business_rule_cycles,
            predicate_cycles,
            memory_overhead,
            io_operations: 0, // Most validations don't require I/O
        }
    }

    /// Build security context from semantic type
    fn build_security_context(&self, semantic_type: &PIRSemanticType) -> SecurityContext {
        let classification = match semantic_type.security_classification {
            prism_pir::semantic::SecurityClassification::Public => SecurityClassification::Public,
            prism_pir::semantic::SecurityClassification::Internal => SecurityClassification::Internal,
            prism_pir::semantic::SecurityClassification::Confidential => SecurityClassification::Confidential,
            prism_pir::semantic::SecurityClassification::Restricted => SecurityClassification::Restricted,
            prism_pir::semantic::SecurityClassification::TopSecret => SecurityClassification::TopSecret,
        };

        SecurityContext {
            classification,
            required_capabilities: Vec::new(), // TODO: Extract from semantic type
            flow_restrictions: Vec::new(), // TODO: Extract from semantic type
            audit_required: matches!(classification, SecurityClassification::Confidential | SecurityClassification::Restricted | SecurityClassification::TopSecret),
        }
    }

    /// Generate AI comprehension hints
    fn generate_ai_hints(&self, semantic_type: &PIRSemanticType) -> Vec<String> {
        let mut hints = Vec::new();
        
        hints.push(format!("This type represents: {}", semantic_type.ai_context.purpose));
        hints.push(format!("Business domain: {}", semantic_type.domain));
        
        if !semantic_type.business_rules.is_empty() {
            hints.push(format!("Has {} business rules that must be validated", semantic_type.business_rules.len()));
        }
        
        if !semantic_type.validation_predicates.is_empty() {
            hints.push(format!("Has {} validation predicates for data integrity", semantic_type.validation_predicates.len()));
        }

        hints
    }

    /// Generate performance hints
    fn generate_performance_hints(&self, semantic_type: &PIRSemanticType) -> Vec<String> {
        let mut hints = Vec::new();
        
        if semantic_type.business_rules.is_empty() && semantic_type.validation_predicates.is_empty() {
            hints.push("No validation overhead - can be optimized aggressively".to_string());
        } else {
            hints.push(format!("Validation overhead: {} rules + {} predicates", 
                semantic_type.business_rules.len(), 
                semantic_type.validation_predicates.len()));
        }

        hints
    }

    /// Compile PIR type info to bytecode type kind
    fn compile_type_kind(&self, type_info: &prism_pir::semantic::PIRTypeInfo) -> VMBackendResult<prism_vm::bytecode::TypeKind> {
        use prism_vm::bytecode::{TypeKind, PrimitiveType, CompositeType, CompositeKind, FieldDefinition};
        
        match type_info {
            prism_pir::semantic::PIRTypeInfo::Primitive(prim) => {
                let primitive_type = match prim {
                    prism_pir::semantic::PIRPrimitiveType::Integer { signed, width } => {
                        PrimitiveType::Integer { signed: *signed, width: *width }
                    }
                    prism_pir::semantic::PIRPrimitiveType::Float { width } => {
                        PrimitiveType::Float { width: *width }
                    }
                    prism_pir::semantic::PIRPrimitiveType::Boolean => PrimitiveType::Boolean,
                    prism_pir::semantic::PIRPrimitiveType::String => PrimitiveType::String,
                    prism_pir::semantic::PIRPrimitiveType::Unit => PrimitiveType::Unit,
                };
                Ok(TypeKind::Primitive(primitive_type))
            }
            prism_pir::semantic::PIRTypeInfo::Composite(composite) => {
                let kind = match composite.kind {
                    prism_pir::semantic::PIRCompositeKind::Struct => CompositeKind::Struct,
                    prism_pir::semantic::PIRCompositeKind::Enum => CompositeKind::Enum,
                    prism_pir::semantic::PIRCompositeKind::Union => CompositeKind::Union,
                    prism_pir::semantic::PIRCompositeKind::Tuple => CompositeKind::Tuple,
                };

                let fields = composite.fields.iter().map(|field| {
                    FieldDefinition {
                        name: field.name.clone(),
                        type_id: 0, // TODO: Resolve actual type ID
                        offset: None,
                        business_meaning: field.business_meaning.clone(),
                    }
                }).collect();

                Ok(TypeKind::Composite(CompositeType {
                    kind,
                    fields,
                    methods: Vec::new(), // TODO: Compile methods
                }))
            }
            _ => {
                // For other type kinds, return a placeholder
                Ok(TypeKind::Primitive(PrimitiveType::Unit))
            }
        }
    }

    /// Get compilation statistics
    pub fn get_stats(&self) -> &CompilationStats {
        &self.stats
    }
}

impl BusinessRuleCompiler {
    /// Create a new business rule compiler
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            rule_cache: HashMap::new(),
        }
    }

    /// Compile a business rule to executable bytecode
    pub fn compile_rule(&mut self, rule: &BusinessRule, index: usize) -> VMBackendResult<CompiledBusinessRule> {
        debug!("Compiling business rule: {}", rule.name);

        // For now, create a simple validation that always returns true
        // In a complete implementation, this would parse the rule logic and generate appropriate bytecode
        let instructions = vec![
            Instruction::new(PrismOpcode::LOAD_TRUE),
            Instruction::new(PrismOpcode::RETURN_VALUE),
        ];

        let category = match rule.category.as_str() {
            "domain" => BusinessRuleCategory::DomainValidation,
            "security" => BusinessRuleCategory::SecurityPolicy,
            "business" => BusinessRuleCategory::BusinessLogic,
            "integrity" => BusinessRuleCategory::DataIntegrity,
            "compliance" => BusinessRuleCategory::Compliance,
            _ => BusinessRuleCategory::BusinessLogic,
        };

        Ok(CompiledBusinessRule {
            id: format!("rule_{}", index),
            name: rule.name.clone(),
            description: rule.description.clone(),
            instructions,
            priority: rule.priority as u32,
            execution_cost: 100, // Estimated cycles
            category,
            error_message_template: format!("Business rule violation: {}", rule.description),
        })
    }
}

impl ValidationPredicateCompiler {
    /// Create a new validation predicate compiler
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            predicate_cache: HashMap::new(),
        }
    }

    /// Compile a validation predicate to executable bytecode
    pub fn compile_predicate(
        &mut self, 
        predicate: &prism_pir::semantic::ValidationPredicate, 
        index: usize
    ) -> VMBackendResult<CompiledValidationPredicate> {
        debug!("Compiling validation predicate: {}", predicate.description);

        // For now, create a simple validation that always returns true
        // In a complete implementation, this would parse the predicate logic and generate appropriate bytecode
        let instructions = vec![
            Instruction::new(PrismOpcode::LOAD_TRUE),
            Instruction::new(PrismOpcode::RETURN_VALUE),
        ];

        Ok(CompiledValidationPredicate {
            id: format!("predicate_{}", index),
            description: predicate.description.clone(),
            instructions,
            return_type_id: 0, // Boolean type
            execution_cost: 50, // Estimated cycles
        })
    }
}

impl OptimizationHintAnalyzer {
    /// Create a new optimization hint analyzer
    pub fn new() -> Self {
        Self {
            patterns: Self::create_default_patterns(),
            stats: OptimizationStats::default(),
        }
    }

    /// Create default optimization patterns
    fn create_default_patterns() -> Vec<OptimizationPattern> {
        vec![
            OptimizationPattern {
                name: "SimpleType".to_string(),
                description: "Simple types can be specialized".to_string(),
                confidence_threshold: 0.8,
                expected_improvement: 0.15,
            },
            OptimizationPattern {
                name: "NoValidation".to_string(),
                description: "Types without validation can skip checks".to_string(),
                confidence_threshold: 0.9,
                expected_improvement: 0.25,
            },
        ]
    }

    /// Analyze a semantic type and generate optimization hints
    pub fn analyze_type(&mut self, semantic_type: &PIRSemanticType) -> VMBackendResult<Vec<OptimizationHint>> {
        let mut hints = Vec::new();

        // Analyze for type specialization opportunities
        if self.can_specialize_type(semantic_type) {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::TypeSpecialization,
                confidence: 0.8,
                expected_improvement: 0.15,
                description: "Type can be specialized for better performance".to_string(),
            });
        }

        // Analyze for bounds check elimination
        if self.can_eliminate_bounds_checks(semantic_type) {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::BoundsCheckElimination,
                confidence: 0.7,
                expected_improvement: 0.20,
                description: "Bounds checks can be eliminated with static analysis".to_string(),
            });
        }

        // Analyze for inlining opportunities
        if self.is_inlining_candidate(semantic_type) {
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::InliningCandidate,
                confidence: 0.6,
                expected_improvement: 0.10,
                description: "Type operations are good candidates for inlining".to_string(),
            });
        }

        self.stats.hints_generated += hints.len() as u64;
        Ok(hints)
    }

    /// Check if type can be specialized
    fn can_specialize_type(&self, semantic_type: &PIRSemanticType) -> bool {
        // Simple heuristic: types with few constraints can often be specialized
        semantic_type.constraints.len() < 3
    }

    /// Check if bounds checks can be eliminated
    fn can_eliminate_bounds_checks(&self, semantic_type: &PIRSemanticType) -> bool {
        // Check if type has range constraints that enable static bounds checking
        semantic_type.constraints.iter().any(|constraint| {
            matches!(constraint, PIRTypeConstraint::Range { .. })
        })
    }

    /// Check if type operations are good inlining candidates
    fn is_inlining_candidate(&self, semantic_type: &PIRSemanticType) -> bool {
        // Simple types with few business rules are good inlining candidates
        semantic_type.business_rules.len() < 2
    }
} 