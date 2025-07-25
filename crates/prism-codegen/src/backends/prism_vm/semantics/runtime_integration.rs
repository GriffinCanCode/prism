//! Runtime Semantic Integration
//!
//! This module provides runtime integration between the VM execution system
//! and semantic information, enabling runtime validation and semantic queries
//! without duplicating logic from prism-semantic.

use super::{VMSemanticConfig, SymbolFromStr};
use prism_semantic::{
    SemanticEngine, SemanticResult, SemanticError,
    types::{SemanticType, BusinessRule, TypeConstraint},
    database::SemanticInfo,
    analyzer::{SymbolInfo, TypeInfo},
    validation::{ValidationResult, ValidationRule},
};
use prism_vm::{
    VMResult, PrismVMError,
    bytecode::{
        BytecodeSemanticMetadata, CompiledBusinessRule, CompiledValidationPredicate,
        ValidationResult as VMValidationResult, SemanticInformationRegistry,
    },
    execution::{ExecutionStack, StackValue, Interpreter},
};
use prism_common::{NodeId, symbol::Symbol};
use std::collections::HashMap;
use tracing::{debug, span, Level};

/// Runtime semantic validator that uses prism-semantic for validation logic
pub struct RuntimeSemanticValidator {
    /// Reference to the semantic engine for validation rules
    semantic_engine: SemanticEngine,
    
    /// Configuration for runtime validation
    config: RuntimeValidationConfig,
    
    /// Cache for validation results
    validation_cache: HashMap<ValidationCacheKey, CachedValidationResult>,
    
    /// Statistics for monitoring
    stats: RuntimeValidationStats,
}

/// Configuration for runtime semantic validation
#[derive(Debug, Clone)]
pub struct RuntimeValidationConfig {
    /// Enable business rule validation at runtime
    pub enable_business_rules: bool,
    
    /// Enable validation predicates at runtime
    pub enable_predicates: bool,
    
    /// Maximum validation time per operation (microseconds)
    pub max_validation_time_us: u64,
    
    /// Cache validation results for performance
    pub enable_caching: bool,
    
    /// Maximum cache size
    pub max_cache_size: usize,
    
    /// Validation failure handling strategy
    pub failure_handling: ValidationFailureHandling,
}

/// How to handle validation failures
#[derive(Debug, Clone)]
pub enum ValidationFailureHandling {
    /// Throw exception immediately
    ThrowException,
    /// Log error and continue execution
    LogAndContinue,
    /// Return error value to caller
    ReturnError,
    /// Trigger debugger if available
    TriggerDebugger,
}

/// Cache key for validation results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ValidationCacheKey {
    type_id: u32,
    value_hash: u64,
    rule_id: String,
}

/// Cached validation result
#[derive(Debug, Clone)]
struct CachedValidationResult {
    result: VMValidationResult,
    timestamp: std::time::Instant,
    hit_count: u64,
}

/// Runtime validation statistics
#[derive(Debug, Default)]
pub struct RuntimeValidationStats {
    /// Total validations performed
    pub total_validations: u64,
    
    /// Cache hits
    pub cache_hits: u64,
    
    /// Cache misses
    pub cache_misses: u64,
    
    /// Total validation time (microseconds)
    pub total_validation_time_us: u64,
    
    /// Failed validations
    pub failed_validations: u64,
    
    /// Average validation time
    pub avg_validation_time_us: f64,
}

/// Runtime semantic query interface
pub struct RuntimeSemanticQuery {
    /// Semantic registry for metadata access
    semantic_registry: SemanticInformationRegistry,
    
    /// Semantic engine for advanced queries
    semantic_engine: SemanticEngine,
    
    /// Query configuration
    config: QueryConfig,
}

/// Configuration for semantic queries
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Enable runtime semantic queries
    pub enable_queries: bool,
    
    /// Maximum query execution time
    pub max_query_time_us: u64,
    
    /// Cache query results
    pub enable_query_caching: bool,
}

impl RuntimeSemanticValidator {
    /// Create a new runtime validator using prism-semantic
    pub fn new(semantic_engine: SemanticEngine, config: RuntimeValidationConfig) -> VMResult<Self> {
        let _span = span!(Level::DEBUG, "runtime_validator_new").entered();
        
        Ok(Self {
            semantic_engine,
            config,
            validation_cache: HashMap::new(),
            stats: RuntimeValidationStats::default(),
        })
    }
    
    /// Validate a value against business rules using semantic information
    pub fn validate_business_rules(
        &mut self,
        type_id: u32,
        value: &StackValue,
        metadata: &BytecodeSemanticMetadata,
    ) -> VMResult<Vec<VMValidationResult>> {
        let _span = span!(Level::DEBUG, "validate_business_rules", type_id = type_id).entered();
        
        if !self.config.enable_business_rules {
            return Ok(vec![]);
        }
        
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        
        // Validate each compiled business rule
        for compiled_rule in &metadata.validation_metadata.business_rule_bytecode {
            let cache_key = ValidationCacheKey {
                type_id,
                value_hash: self.hash_stack_value(value),
                rule_id: compiled_rule.id.clone(),
            };
            
            // Check cache first
            if let Some(cached) = self.check_cache(&cache_key) {
                results.push(cached.result);
                continue;
            }
            
            // Use prism-semantic to get the original validation rule
            let validation_result = match self.get_semantic_validation_rule(&compiled_rule.id) {
                Ok(semantic_rule) => {
                    // Convert stack value to semantic format for validation
                    let semantic_value = self.stack_value_to_semantic(value, type_id)?;
                    
                    // Use prism-semantic's validation engine
                    match self.semantic_engine.validate_rule(&semantic_rule, &semantic_value) {
                        Ok(semantic_result) => self.convert_semantic_validation_result(
                            semantic_result,
                            &compiled_rule.id,
                            start_time.elapsed().as_micros() as u64,
                        ),
                        Err(semantic_error) => VMValidationResult {
                            rule_id: compiled_rule.id.clone(),
                            passed: false,
                            error_message: Some(format!("Semantic validation error: {}", semantic_error)),
                            execution_time_us: start_time.elapsed().as_micros() as u64,
                        },
                    }
                }
                Err(_) => {
                    // Fallback to compiled bytecode execution if semantic rule not available
                    self.execute_compiled_business_rule(compiled_rule, value, type_id)?
                }
            };
            
            // Cache the result
            self.cache_validation_result(cache_key, validation_result.clone());
            results.push(validation_result);
        }
        
        self.update_stats(start_time.elapsed().as_micros() as u64, results.len());
        Ok(results)
    }
    
    /// Validate using predicates
    pub fn validate_predicates(
        &mut self,
        type_id: u32,
        value: &StackValue,
        metadata: &BytecodeSemanticMetadata,
    ) -> VMResult<Vec<VMValidationResult>> {
        let _span = span!(Level::DEBUG, "validate_predicates", type_id = type_id).entered();
        
        if !self.config.enable_predicates {
            return Ok(vec![]);
        }
        
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        
        for compiled_predicate in &metadata.validation_metadata.validation_predicate_bytecode {
            let cache_key = ValidationCacheKey {
                type_id,
                value_hash: self.hash_stack_value(value),
                rule_id: compiled_predicate.id.clone(),
            };
            
            if let Some(cached) = self.check_cache(&cache_key) {
                results.push(cached.result);
                continue;
            }
            
            // Execute the compiled predicate
            let validation_result = self.execute_compiled_predicate(compiled_predicate, value, type_id)?;
            
            self.cache_validation_result(cache_key, validation_result.clone());
            results.push(validation_result);
        }
        
        self.update_stats(start_time.elapsed().as_micros() as u64, results.len());
        Ok(results)
    }
    
    /// Get validation statistics
    pub fn get_stats(&self) -> &RuntimeValidationStats {
        &self.stats
    }
    
    /// Clear validation cache
    pub fn clear_cache(&mut self) {
        self.validation_cache.clear();
    }
    
    // Private helper methods
    
    fn get_semantic_validation_rule(&self, rule_id: &str) -> SemanticResult<ValidationRule> {
        // Use prism-semantic to retrieve the original validation rule
        self.semantic_engine.get_validation_rule(rule_id)
    }
    
    fn stack_value_to_semantic(&self, value: &StackValue, type_id: u32) -> VMResult<prism_semantic::types::Value> {
        // Convert VM stack value to semantic value format
        // This uses prism-semantic's value representation
        match value {
            StackValue::Integer(i) => Ok(prism_semantic::types::Value::Integer(*i)),
            StackValue::Float(f) => Ok(prism_semantic::types::Value::Float(*f)),
            StackValue::Boolean(b) => Ok(prism_semantic::types::Value::Boolean(*b)),
            StackValue::String(s) => Ok(prism_semantic::types::Value::String(s.clone())),
            StackValue::Null => Ok(prism_semantic::types::Value::Null),
            _ => Err(PrismVMError::RuntimeError(format!(
                "Unsupported stack value type for semantic validation: {:?}", value
            ))),
        }
    }
    
    fn convert_semantic_validation_result(
        &self,
        semantic_result: prism_semantic::validation::ValidationResult,
        rule_id: &str,
        execution_time_us: u64,
    ) -> VMValidationResult {
        VMValidationResult {
            rule_id: rule_id.to_string(),
            passed: semantic_result.is_valid(),
            error_message: if semantic_result.is_valid() {
                None
            } else {
                Some(semantic_result.error_message().unwrap_or("Validation failed").to_string())
            },
            execution_time_us,
        }
    }
    
    fn execute_compiled_business_rule(
        &self,
        compiled_rule: &CompiledBusinessRule,
        value: &StackValue,
        type_id: u32,
    ) -> VMResult<VMValidationResult> {
        // Execute the compiled bytecode for the business rule
        // This is a fallback when semantic engine validation is not available
        let start_time = std::time::Instant::now();
        
        // Create a minimal execution context for rule validation
        let mut mini_interpreter = self.create_validation_interpreter()?;
        
        // Push the value onto the stack
        mini_interpreter.push_value(value.clone())?;
        
        // Execute the compiled rule instructions
        let result = mini_interpreter.execute_instructions(&compiled_rule.instructions)?;
        
        let execution_time = start_time.elapsed().as_micros() as u64;
        
        // Convert execution result to validation result
        let passed = match result {
            Some(StackValue::Boolean(b)) => b,
            Some(StackValue::Integer(i)) => i != 0,
            _ => false,
        };
        
        Ok(VMValidationResult {
            rule_id: compiled_rule.id.clone(),
            passed,
            error_message: if passed {
                None
            } else {
                Some(compiled_rule.error_message_template.clone())
            },
            execution_time_us: execution_time,
        })
    }
    
    fn execute_compiled_predicate(
        &self,
        compiled_predicate: &CompiledValidationPredicate,
        value: &StackValue,
        type_id: u32,
    ) -> VMResult<VMValidationResult> {
        let start_time = std::time::Instant::now();
        
        let mut mini_interpreter = self.create_validation_interpreter()?;
        mini_interpreter.push_value(value.clone())?;
        
        let result = mini_interpreter.execute_instructions(&compiled_predicate.instructions)?;
        let execution_time = start_time.elapsed().as_micros() as u64;
        
        let passed = match result {
            Some(StackValue::Boolean(b)) => b,
            _ => false,
        };
        
        Ok(VMValidationResult {
            rule_id: compiled_predicate.id.clone(),
            passed,
            error_message: if passed {
                None
            } else {
                Some(format!("Predicate validation failed: {}", compiled_predicate.description))
            },
            execution_time_us: execution_time,
        })
    }
    
    fn create_validation_interpreter(&self) -> VMResult<ValidationInterpreter> {
        // Create a minimal interpreter for executing validation bytecode
        ValidationInterpreter::new()
    }
    
    fn hash_stack_value(&self, value: &StackValue) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // Simple hash based on value type and content
        match value {
            StackValue::Integer(i) => {
                "int".hash(&mut hasher);
                i.hash(&mut hasher);
            }
            StackValue::Float(f) => {
                "float".hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            StackValue::Boolean(b) => {
                "bool".hash(&mut hasher);
                b.hash(&mut hasher);
            }
            StackValue::String(s) => {
                "string".hash(&mut hasher);
                s.hash(&mut hasher);
            }
            StackValue::Null => {
                "null".hash(&mut hasher);
            }
            _ => {
                "other".hash(&mut hasher);
            }
        }
        hasher.finish()
    }
    
    fn check_cache(&mut self, key: &ValidationCacheKey) -> Option<CachedValidationResult> {
        if !self.config.enable_caching {
            return None;
        }
        
        if let Some(cached) = self.validation_cache.get_mut(key) {
            cached.hit_count += 1;
            self.stats.cache_hits += 1;
            Some(cached.clone())
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }
    
    fn cache_validation_result(&mut self, key: ValidationCacheKey, result: VMValidationResult) {
        if !self.config.enable_caching {
            return;
        }
        
        // Implement LRU eviction if cache is full
        if self.validation_cache.len() >= self.config.max_cache_size {
            self.evict_least_recently_used();
        }
        
        let cached_result = CachedValidationResult {
            result,
            timestamp: std::time::Instant::now(),
            hit_count: 0,
        };
        
        self.validation_cache.insert(key, cached_result);
    }
    
    fn evict_least_recently_used(&mut self) {
        // Simple LRU eviction - remove oldest entry
        if let Some((oldest_key, _)) = self.validation_cache
            .iter()
            .min_by_key(|(_, cached)| cached.timestamp)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.validation_cache.remove(&oldest_key);
        }
    }
    
    fn update_stats(&mut self, execution_time_us: u64, validation_count: usize) {
        self.stats.total_validations += validation_count as u64;
        self.stats.total_validation_time_us += execution_time_us;
        self.stats.avg_validation_time_us = 
            self.stats.total_validation_time_us as f64 / self.stats.total_validations as f64;
    }
}

impl RuntimeSemanticQuery {
    /// Create a new runtime query interface
    pub fn new(
        semantic_registry: SemanticInformationRegistry,
        semantic_engine: SemanticEngine,
        config: QueryConfig,
    ) -> Self {
        Self {
            semantic_registry,
            semantic_engine,
            config,
        }
    }
    
    /// Query semantic information by type
    pub fn query_type_semantics(&self, type_id: u32) -> VMResult<Option<BytecodeSemanticMetadata>> {
        if !self.config.enable_queries {
            return Ok(None);
        }
        
        Ok(self.semantic_registry.get_type_metadata(type_id).cloned())
    }
    
    /// Query business rules for a type
    pub fn query_business_rules(&self, type_id: u32) -> VMResult<Vec<CompiledBusinessRule>> {
        if let Some(metadata) = self.semantic_registry.get_type_metadata(type_id) {
            Ok(metadata.validation_metadata.business_rule_bytecode.clone())
        } else {
            Ok(vec![])
        }
    }
    
    /// Query types by business domain
    pub fn query_types_by_domain(&self, domain: &str) -> VMResult<Vec<(u32, BytecodeSemanticMetadata)>> {
        Ok(self.semantic_registry.query_types_by_domain(domain)
            .into_iter()
            .map(|(id, metadata)| (id, metadata.clone()))
            .collect())
    }
}

/// Minimal interpreter for executing validation bytecode
struct ValidationInterpreter {
    stack: Vec<StackValue>,
}

impl ValidationInterpreter {
    fn new() -> VMResult<Self> {
        Ok(Self {
            stack: Vec::new(),
        })
    }
    
    fn push_value(&mut self, value: StackValue) -> VMResult<()> {
        self.stack.push(value);
        Ok(())
    }
    
    fn execute_instructions(&mut self, instructions: &[prism_vm::bytecode::Instruction]) -> VMResult<Option<StackValue>> {
        // Execute a simplified set of instructions for validation
        for instruction in instructions {
            match &instruction.opcode {
                prism_vm::bytecode::PrismOpcode::LOAD_TRUE => {
                    self.stack.push(StackValue::Boolean(true));
                }
                prism_vm::bytecode::PrismOpcode::LOAD_FALSE => {
                    self.stack.push(StackValue::Boolean(false));
                }
                prism_vm::bytecode::PrismOpcode::RETURN_VALUE => {
                    return Ok(self.stack.pop());
                }
                prism_vm::bytecode::PrismOpcode::RETURN => {
                    return Ok(None);
                }
                // Add more instruction implementations as needed for validation
                _ => {
                    return Err(PrismVMError::RuntimeError(format!(
                        "Unsupported instruction in validation context: {:?}",
                        instruction.opcode
                    )));
                }
            }
        }
        
        Ok(self.stack.pop())
    }
}

impl Default for RuntimeValidationConfig {
    fn default() -> Self {
        Self {
            enable_business_rules: true,
            enable_predicates: true,
            max_validation_time_us: 1000, // 1ms default
            enable_caching: true,
            max_cache_size: 1000,
            failure_handling: ValidationFailureHandling::LogAndContinue,
        }
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            enable_queries: true,
            max_query_time_us: 500, // 0.5ms default
            enable_query_caching: true,
        }
    }
} 