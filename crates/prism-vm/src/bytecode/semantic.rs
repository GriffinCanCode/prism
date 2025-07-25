//! Semantic Information Preservation for Bytecode
//!
//! This module implements comprehensive semantic information preservation in bytecode,
//! ensuring that business rules, validation predicates, and AI metadata are maintained
//! throughout compilation and available at runtime.

use crate::{VMResult, PrismVMError, bytecode::{Instruction, TypeDefinition}};
use prism_pir::semantic::{PIRSemanticType, PIRTypeConstraint};
use prism_common::{NodeId, symbol::Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, span, Level};

/// Enhanced semantic metadata for bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodeSemanticMetadata {
    /// Original PIR semantic type information
    pub original_semantic_type: Option<PIRSemanticType>,
    /// Business domain context
    pub business_domain: String,
    /// AI comprehension hints
    pub ai_hints: Vec<String>,
    /// Performance characteristics
    pub performance_hints: Vec<String>,
    /// Security implications
    pub security_context: SecurityContext,
    /// Validation metadata
    pub validation_metadata: ValidationMetadata,
    /// Optimization hints derived from semantic analysis
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Security context for semantic types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Security classification level
    pub classification: SecurityClassification,
    /// Required capabilities for access
    pub required_capabilities: Vec<String>,
    /// Information flow restrictions
    pub flow_restrictions: Vec<FlowRestriction>,
    /// Audit requirements
    pub audit_required: bool,
}

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
    TopSecret,
}

/// Information flow restriction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowRestriction {
    /// Source context
    pub from: String,
    /// Target context
    pub to: String,
    /// Restriction type
    pub restriction_type: RestrictionType,
}

/// Types of flow restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestrictionType {
    NoFlow,
    RequiresApproval,
    RequiresTransformation,
    LogOnly,
}

/// Validation metadata for runtime checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Compiled business rule bytecode
    pub business_rule_bytecode: Vec<CompiledBusinessRule>,
    /// Compiled validation predicate bytecode
    pub validation_predicate_bytecode: Vec<CompiledValidationPredicate>,
    /// Runtime validation configuration
    pub validation_config: ValidationConfig,
    /// Validation performance characteristics
    pub validation_cost: ValidationCost,
}

/// Compiled business rule with execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledBusinessRule {
    /// Rule identifier
    pub id: String,
    /// Rule name for debugging
    pub name: String,
    /// Rule description
    pub description: String,
    /// Compiled bytecode instructions
    pub instructions: Vec<Instruction>,
    /// Rule priority (higher = more important)
    pub priority: u32,
    /// Expected execution cost
    pub execution_cost: u64,
    /// Rule category
    pub category: BusinessRuleCategory,
    /// Error message template
    pub error_message_template: String,
}

/// Business rule categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessRuleCategory {
    /// Domain validation rules
    DomainValidation,
    /// Security policy enforcement
    SecurityPolicy,
    /// Business logic constraints
    BusinessLogic,
    /// Data integrity rules
    DataIntegrity,
    /// Compliance requirements
    Compliance,
}

/// Compiled validation predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledValidationPredicate {
    /// Predicate identifier
    pub id: String,
    /// Predicate description
    pub description: String,
    /// Compiled bytecode instructions
    pub instructions: Vec<Instruction>,
    /// Expected return type (should be boolean)
    pub return_type_id: u32,
    /// Execution cost estimate
    pub execution_cost: u64,
}

/// Runtime validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable runtime business rule validation
    pub enable_business_rules: bool,
    /// Enable runtime predicate validation
    pub enable_predicates: bool,
    /// Validation mode
    pub validation_mode: ValidationMode,
    /// Maximum validation time per operation (microseconds)
    pub max_validation_time_us: u64,
    /// Validation failure handling
    pub failure_handling: ValidationFailureHandling,
}

/// Validation execution modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    /// Always validate (strict mode)
    Always,
    /// Validate in debug builds only
    DebugOnly,
    /// Probabilistic validation (sample-based)
    Probabilistic { sample_rate: f64 },
    /// Validate only on first access
    FirstAccess,
    /// Disabled
    Disabled,
}

/// How to handle validation failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationFailureHandling {
    /// Throw exception immediately
    ThrowException,
    /// Log error and continue
    LogAndContinue,
    /// Return error value
    ReturnError,
    /// Trigger debugger
    TriggerDebugger,
}

/// Validation cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCost {
    /// Estimated CPU cycles for business rules
    pub business_rule_cycles: u64,
    /// Estimated CPU cycles for predicates
    pub predicate_cycles: u64,
    /// Memory overhead in bytes
    pub memory_overhead: u64,
    /// I/O operations required
    pub io_operations: u32,
}

/// Optimization hints derived from semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Hint type
    pub hint_type: OptimizationHintType,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Expected performance improvement
    pub expected_improvement: f64,
    /// Hint description
    pub description: String,
}

/// Types of optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationHintType {
    /// Can be specialized for specific types
    TypeSpecialization,
    /// Can be inlined safely
    InliningCandidate,
    /// Bounds checking can be eliminated
    BoundsCheckElimination,
    /// Null checks can be eliminated
    NullCheckElimination,
    /// Memory allocation can be optimized
    AllocationOptimization,
    /// Loop can be vectorized
    VectorizationCandidate,
    /// Can use SIMD instructions
    SIMDCandidate,
}

/// Semantic information registry for runtime access
#[derive(Debug)]
pub struct SemanticInformationRegistry {
    /// Type ID to semantic metadata mapping
    type_metadata: HashMap<u32, BytecodeSemanticMetadata>,
    /// Function ID to semantic metadata mapping
    function_metadata: HashMap<u32, FunctionSemanticMetadata>,
    /// Global semantic configuration
    config: SemanticRegistryConfig,
}

/// Function-level semantic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSemanticMetadata {
    /// Function business responsibility
    pub responsibility: String,
    /// Algorithm description
    pub algorithm_description: Option<String>,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Side effects
    pub side_effects: Vec<String>,
    /// AI comprehension context
    pub ai_context: FunctionAIContext,
}

/// Performance profile for functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Time complexity
    pub time_complexity: String,
    /// Space complexity
    pub space_complexity: String,
    /// Expected execution time range (microseconds)
    pub execution_time_range: (u64, u64),
    /// Memory usage pattern
    pub memory_pattern: MemoryPattern,
    /// Cache behavior
    pub cache_behavior: CacheBehavior,
}

/// Memory usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPattern {
    Constant,
    Linear,
    Logarithmic,
    Quadratic,
    Exponential,
    Unknown,
}

/// Cache behavior characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheBehavior {
    CacheFriendly,
    CacheNeutral,
    CacheUnfriendly,
    Unknown,
}

/// AI context for functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAIContext {
    /// What this function does (human-readable)
    pub purpose: String,
    /// When to use this function
    pub usage_context: Vec<String>,
    /// Common patterns involving this function
    pub common_patterns: Vec<String>,
    /// Potential pitfalls
    pub pitfalls: Vec<String>,
}

/// Configuration for semantic registry
#[derive(Debug, Clone)]
pub struct SemanticRegistryConfig {
    /// Enable runtime semantic queries
    pub enable_runtime_queries: bool,
    /// Cache semantic information for performance
    pub enable_caching: bool,
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    /// Enable semantic debugging
    pub enable_debugging: bool,
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            classification: SecurityClassification::Internal,
            required_capabilities: Vec::new(),
            flow_restrictions: Vec::new(),
            audit_required: false,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_business_rules: true,
            enable_predicates: true,
            validation_mode: ValidationMode::Always,
            max_validation_time_us: 1000, // 1ms max
            failure_handling: ValidationFailureHandling::ThrowException,
        }
    }
}

impl Default for SemanticRegistryConfig {
    fn default() -> Self {
        Self {
            enable_runtime_queries: true,
            enable_caching: true,
            max_cache_size: 1000,
            enable_debugging: true,
        }
    }
}

impl SemanticInformationRegistry {
    /// Create a new semantic information registry
    pub fn new() -> Self {
        Self::with_config(SemanticRegistryConfig::default())
    }

    /// Create a registry with custom configuration
    pub fn with_config(config: SemanticRegistryConfig) -> Self {
        Self {
            type_metadata: HashMap::new(),
            function_metadata: HashMap::new(),
            config,
        }
    }

    /// Register semantic metadata for a type
    pub fn register_type_metadata(&mut self, type_id: u32, metadata: BytecodeSemanticMetadata) {
        debug!("Registering semantic metadata for type {}", type_id);
        self.type_metadata.insert(type_id, metadata);
    }

    /// Register semantic metadata for a function
    pub fn register_function_metadata(&mut self, function_id: u32, metadata: FunctionSemanticMetadata) {
        debug!("Registering semantic metadata for function {}", function_id);
        self.function_metadata.insert(function_id, metadata);
    }

    /// Get semantic metadata for a type
    pub fn get_type_metadata(&self, type_id: u32) -> Option<&BytecodeSemanticMetadata> {
        self.type_metadata.get(&type_id)
    }

    /// Get semantic metadata for a function
    pub fn get_function_metadata(&self, function_id: u32) -> Option<&FunctionSemanticMetadata> {
        self.function_metadata.get(&function_id)
    }

    /// Query types by business domain
    pub fn query_types_by_domain(&self, domain: &str) -> Vec<(u32, &BytecodeSemanticMetadata)> {
        self.type_metadata
            .iter()
            .filter(|(_, metadata)| metadata.business_domain == domain)
            .map(|(id, metadata)| (*id, metadata))
            .collect()
    }

    /// Query functions by responsibility
    pub fn query_functions_by_responsibility(&self, responsibility: &str) -> Vec<(u32, &FunctionSemanticMetadata)> {
        self.function_metadata
            .iter()
            .filter(|(_, metadata)| metadata.responsibility.contains(responsibility))
            .map(|(id, metadata)| (*id, metadata))
            .collect()
    }

    /// Get all optimization hints for a type
    pub fn get_optimization_hints(&self, type_id: u32) -> Vec<&OptimizationHint> {
        self.type_metadata
            .get(&type_id)
            .map(|metadata| metadata.optimization_hints.iter().collect())
            .unwrap_or_default()
    }

    /// Validate a value against type business rules
    pub fn validate_business_rules(&self, type_id: u32, value: &[u8]) -> VMResult<Vec<ValidationResult>> {
        let _span = span!(Level::DEBUG, "validate_business_rules", type_id).entered();

        let metadata = self.type_metadata.get(&type_id)
            .ok_or_else(|| PrismVMError::SemanticError {
                message: format!("No semantic metadata found for type {}", type_id),
            })?;

        if !metadata.validation_metadata.validation_config.enable_business_rules {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        for rule in &metadata.validation_metadata.business_rule_bytecode {
            // TODO: Execute business rule bytecode against value
            // For now, return a placeholder result
            results.push(ValidationResult {
                rule_id: rule.id.clone(),
                passed: true,
                error_message: None,
                execution_time_us: 0,
            });
        }

        Ok(results)
    }

    /// Export semantic information for AI tools
    pub fn export_ai_metadata(&self) -> AISemanticExport {
        AISemanticExport {
            type_metadata: self.type_metadata.clone(),
            function_metadata: self.function_metadata.clone(),
            export_timestamp: chrono::Utc::now(),
            schema_version: "1.0".to_string(),
        }
    }
}

/// Result of a validation check
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Rule that was checked
    pub rule_id: String,
    /// Whether the validation passed
    pub passed: bool,
    /// Error message if validation failed
    pub error_message: Option<String>,
    /// Time taken to execute validation (microseconds)
    pub execution_time_us: u64,
}

/// Export format for AI tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISemanticExport {
    /// All type metadata
    pub type_metadata: HashMap<u32, BytecodeSemanticMetadata>,
    /// All function metadata
    pub function_metadata: HashMap<u32, FunctionSemanticMetadata>,
    /// When this export was generated
    pub export_timestamp: chrono::DateTime<chrono::Utc>,
    /// Schema version for compatibility
    pub schema_version: String,
}

/// Builder for semantic metadata
pub struct SemanticMetadataBuilder {
    metadata: BytecodeSemanticMetadata,
}

impl SemanticMetadataBuilder {
    /// Start building semantic metadata
    pub fn new(business_domain: impl Into<String>) -> Self {
        Self {
            metadata: BytecodeSemanticMetadata {
                original_semantic_type: None,
                business_domain: business_domain.into(),
                ai_hints: Vec::new(),
                performance_hints: Vec::new(),
                security_context: SecurityContext::default(),
                validation_metadata: ValidationMetadata {
                    business_rule_bytecode: Vec::new(),
                    validation_predicate_bytecode: Vec::new(),
                    validation_config: ValidationConfig::default(),
                    validation_cost: ValidationCost {
                        business_rule_cycles: 0,
                        predicate_cycles: 0,
                        memory_overhead: 0,
                        io_operations: 0,
                    },
                },
                optimization_hints: Vec::new(),
            },
        }
    }

    /// Set the original PIR semantic type
    pub fn with_original_type(mut self, semantic_type: PIRSemanticType) -> Self {
        self.metadata.original_semantic_type = Some(semantic_type);
        self
    }

    /// Add an AI hint
    pub fn with_ai_hint(mut self, hint: impl Into<String>) -> Self {
        self.metadata.ai_hints.push(hint.into());
        self
    }

    /// Add a performance hint
    pub fn with_performance_hint(mut self, hint: impl Into<String>) -> Self {
        self.metadata.performance_hints.push(hint.into());
        self
    }

    /// Set security context
    pub fn with_security_context(mut self, context: SecurityContext) -> Self {
        self.metadata.security_context = context;
        self
    }

    /// Add an optimization hint
    pub fn with_optimization_hint(mut self, hint: OptimizationHint) -> Self {
        self.metadata.optimization_hints.push(hint);
        self
    }

    /// Build the metadata
    pub fn build(self) -> BytecodeSemanticMetadata {
        self.metadata
    }
} 