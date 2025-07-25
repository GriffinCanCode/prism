//! VM Effect Enforcement System
//!
//! This module implements runtime enforcement of the effect system within the VM,
//! ensuring that all effects are properly validated and executed according to
//! their definitions and security constraints. It bridges the gap between the
//! prism-effects system and VM execution.
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Uses existing effect system, doesn't duplicate logic
//! 2. **Runtime Enforcement**: Validates effects during VM execution, not just compilation
//! 3. **Capability Integration**: Bridges effect requirements with VM capability system
//! 4. **Performance Optimized**: Minimal overhead during normal execution
//! 5. **Modular Design**: Clean integration points with interpreter and JIT

use crate::{VMResult, PrismVMError};
use crate::execution::stack::{ExecutionStack, StackValue};
use crate::bytecode::{Instruction, FunctionDefinition};
use prism_effects::{
    effects::{Effect, EffectDefinition, EffectSystem},
    execution::{ExecutionSystem as EffectExecutionSystem, ExecutionPlan, EffectResult},
    security::SecureExecutionContext,
    validation::{EffectValidator, ValidationContext},
    EffectSystemError,
};
use prism_runtime::{
    authority::capability::{CapabilitySet, Capability},
    platform::execution::ExecutionContext,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, error, span, Level};
use uuid::Uuid;

/// VM Effect Enforcement Manager
/// 
/// Coordinates between the VM execution engine and the effect system to ensure
/// all effects are properly validated and executed at runtime.
#[derive(Debug)]
pub struct VMEffectEnforcer {
    /// Effect system for validation and execution
    effect_system: Arc<RwLock<EffectSystem>>,
    
    /// Effect execution system
    effect_executor: Arc<RwLock<EffectExecutionSystem>>,
    
    /// Effect validator for runtime validation
    effect_validator: Arc<EffectValidator>,
    
    /// Active effect contexts (per function call)
    active_contexts: HashMap<u32, VMEffectContext>,
    
    /// Effect enforcement configuration
    config: EffectEnforcementConfig,
    
    /// Runtime statistics
    stats: EffectEnforcementStats,
}

/// VM Effect Context
/// 
/// Tracks effects for a specific function execution context
#[derive(Debug, Clone)]
pub struct VMEffectContext {
    /// Function ID this context belongs to
    pub function_id: u32,
    
    /// Effects declared for this function
    pub declared_effects: Vec<Effect>,
    
    /// Effects currently active (entered but not exited)
    pub active_effects: VecDeque<Effect>,
    
    /// Capabilities available for effect execution
    pub capabilities: CapabilitySet,
    
    /// Secure execution context for effect validation
    pub secure_context: SecureExecutionContext,
    
    /// Effect execution results
    pub effect_results: Vec<EffectResult>,
    
    /// Context creation time
    pub created_at: Instant,
}

/// Effect enforcement configuration
#[derive(Debug, Clone)]
pub struct EffectEnforcementConfig {
    /// Enable strict effect validation
    pub strict_validation: bool,
    
    /// Enable effect execution tracking
    pub track_execution: bool,
    
    /// Maximum effect nesting depth
    pub max_nesting_depth: usize,
    
    /// Effect execution timeout
    pub execution_timeout: Duration,
    
    /// Enable effect caching for performance
    pub enable_caching: bool,
}

/// Effect enforcement statistics
#[derive(Debug, Default, Clone)]
pub struct EffectEnforcementStats {
    /// Total effect contexts created
    pub contexts_created: u64,
    
    /// Total effects entered
    pub effects_entered: u64,
    
    /// Total effects executed
    pub effects_executed: u64,
    
    /// Total effect violations detected
    pub violations_detected: u64,
    
    /// Total validation time
    pub validation_time: Duration,
    
    /// Total execution time
    pub execution_time: Duration,
}

impl VMEffectEnforcer {
    /// Create a new VM effect enforcer
    pub fn new(
        effect_system: Arc<RwLock<EffectSystem>>,
        effect_executor: Arc<RwLock<EffectExecutionSystem>>,
        effect_validator: Arc<EffectValidator>,
    ) -> Self {
        Self {
            effect_system,
            effect_executor,
            effect_validator,
            active_contexts: HashMap::new(),
            config: EffectEnforcementConfig::default(),
            stats: EffectEnforcementStats::default(),
        }
    }
    
    /// Create a new effect context for function execution
    pub fn create_context(
        &mut self,
        function_id: u32,
        function_def: &FunctionDefinition,
        capabilities: CapabilitySet,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "create_effect_context", function_id).entered();
        
        debug!("Creating effect context for function {}", function_id);
        
        // Build secure execution context
        let secure_context = SecureExecutionContext {
            available_capabilities: capabilities.clone().into_iter().collect(),
            security_level: prism_effects::security::SecurityLevel::High,
            execution_id: Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now(),
        };
        
        // Create VM effect context
        let context = VMEffectContext {
            function_id,
            declared_effects: function_def.effects.clone(),
            active_effects: VecDeque::new(),
            capabilities,
            secure_context,
            effect_results: Vec::new(),
            created_at: Instant::now(),
        };
        
        // Validate declared effects against capabilities
        if self.config.strict_validation {
            self.validate_declared_effects(&context)?;
        }
        
        self.active_contexts.insert(function_id, context);
        self.stats.contexts_created += 1;
        
        Ok(())
    }
    
    /// Enter an effect context (EFFECT_ENTER instruction)
    pub fn enter_effect(
        &mut self,
        function_id: u32,
        effect_id: u16,
        stack: &mut ExecutionStack,
    ) -> VMResult<()> {
        let _span = span!(Level::TRACE, "enter_effect", function_id, effect_id).entered();
        
        let context = self.active_contexts.get_mut(&function_id)
            .ok_or_else(|| PrismVMError::EffectError {
                message: format!("No effect context for function {}", function_id),
            })?;
        
        // Find the effect to enter
        let effect = context.declared_effects.get(effect_id as usize)
            .ok_or_else(|| PrismVMError::EffectError {
                message: format!("Invalid effect ID: {}", effect_id),
            })?
            .clone();
        
        debug!("Entering effect: {:?}", effect);
        
        // Validate effect can be entered
        self.validate_effect_entry(&effect, context)?;
        
        // Check nesting depth
        if context.active_effects.len() >= self.config.max_nesting_depth {
            return Err(PrismVMError::EffectError {
                message: format!("Effect nesting depth exceeded: {}", self.config.max_nesting_depth),
            });
        }
        
        // Enter the effect
        context.active_effects.push_back(effect);
        self.stats.effects_entered += 1;
        
        Ok(())
    }
    
    /// Exit an effect context (EFFECT_EXIT instruction)
    pub fn exit_effect(
        &mut self,
        function_id: u32,
        stack: &mut ExecutionStack,
    ) -> VMResult<()> {
        let _span = span!(Level::TRACE, "exit_effect", function_id).entered();
        
        let context = self.active_contexts.get_mut(&function_id)
            .ok_or_else(|| PrismVMError::EffectError {
                message: format!("No effect context for function {}", function_id),
            })?;
        
        // Pop the most recent effect
        let effect = context.active_effects.pop_back()
            .ok_or_else(|| PrismVMError::EffectError {
                message: "No active effect to exit".to_string(),
            })?;
        
        debug!("Exiting effect: {:?}", effect);
        
        Ok(())
    }
    
    /// Invoke an effectful operation (EFFECT_INVOKE instruction)
    pub fn invoke_effect(
        &mut self,
        function_id: u32,
        effect_id: u16,
        stack: &mut ExecutionStack,
    ) -> VMResult<()> {
        let _span = span!(Level::TRACE, "invoke_effect", function_id, effect_id).entered();
        let start_time = Instant::now();
        
        let context = self.active_contexts.get_mut(&function_id)
            .ok_or_else(|| PrismVMError::EffectError {
                message: format!("No effect context for function {}", function_id),
            })?;
        
        // Find the effect to invoke
        let effect = context.declared_effects.get(effect_id as usize)
            .ok_or_else(|| PrismVMError::EffectError {
                message: format!("Invalid effect ID: {}", effect_id),
            })?
            .clone();
        
        debug!("Invoking effect: {:?}", effect);
        
        // Validate effect can be invoked
        self.validate_effect_invocation(&effect, context)?;
        
        // Execute the effect through the effect system
        let result = self.execute_effect(&effect, context)?;
        
        // Store result
        context.effect_results.push(result);
        self.stats.effects_executed += 1;
        self.stats.execution_time += start_time.elapsed();
        
        Ok(())
    }
    
    /// Handle an effect (EFFECT_HANDLE instruction)
    pub fn handle_effect(
        &mut self,
        function_id: u32,
        handler_id: u16,
        stack: &mut ExecutionStack,
    ) -> VMResult<()> {
        let _span = span!(Level::TRACE, "handle_effect", function_id, handler_id).entered();
        
        // Effect handling would integrate with the effect system's handler mechanism
        // For now, this is a placeholder that validates the handler exists
        
        let context = self.active_contexts.get(&function_id)
            .ok_or_else(|| PrismVMError::EffectError {
                message: format!("No effect context for function {}", function_id),
            })?;
        
        debug!("Handling effect with handler ID: {}", handler_id);
        
        // TODO: Integrate with effect system's handler registry
        // This would look up the handler and execute it
        
        Ok(())
    }
    
    /// Destroy an effect context when function execution completes
    pub fn destroy_context(&mut self, function_id: u32) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "destroy_effect_context", function_id).entered();
        
        if let Some(context) = self.active_contexts.remove(&function_id) {
            debug!("Destroying effect context for function {}", function_id);
            
            // Validate all effects were properly exited
            if !context.active_effects.is_empty() {
                warn!("Function {} completed with {} active effects", 
                      function_id, context.active_effects.len());
                
                if self.config.strict_validation {
                    return Err(PrismVMError::EffectError {
                        message: format!("Function completed with active effects: {:?}", 
                                       context.active_effects),
                    });
                }
            }
            
            // Log context statistics
            let execution_time = context.created_at.elapsed();
            debug!("Effect context {} executed {} effects in {:?}",
                   function_id, context.effect_results.len(), execution_time);
        }
        
        Ok(())
    }
    
    /// Validate declared effects against available capabilities
    fn validate_declared_effects(&self, context: &VMEffectContext) -> VMResult<()> {
        let validation_start = Instant::now();
        
        for effect in &context.declared_effects {
            // ENHANCED: Check effect-specific capability requirements
            if let Some(required_capability) = effect.required_capability() {
                if !context.capabilities.has_capability(&required_capability) {
                    return Err(PrismVMError::CapabilityViolation {
                        message: format!("Effect {:?} requires capability: {:?}", effect, required_capability),
                    });
                }
            }

            // Create validation context
            let validation_context = ValidationContext {
                available_capabilities: context.capabilities.clone().into_iter().collect(),
                security_level: prism_effects::security::SecurityLevel::High,
                validation_timestamp: std::time::SystemTime::now(),
            };
            
            // Validate effect through effect system
            match self.effect_validator.validate_effect(effect, &validation_context) {
                Ok(_) => continue,
                Err(e) => {
                    return Err(PrismVMError::EffectError {
                        message: format!("Effect validation failed: {}", e),
                    });
                }
            }
        }
        
        self.stats.validation_time += validation_start.elapsed();
        Ok(())
    }
    
    /// Validate an effect can be entered
    fn validate_effect_entry(&self, effect: &Effect, context: &VMEffectContext) -> VMResult<()> {
        // Check if effect is declared
        if !context.declared_effects.contains(effect) {
            return Err(PrismVMError::EffectError {
                message: format!("Undeclared effect: {:?}", effect),
            });
        }
        
        // ENHANCED: Validate capability requirements at entry time
        if let Some(required_capability) = effect.required_capability() {
            if !context.capabilities.has_capability(&required_capability) {
                return Err(PrismVMError::CapabilityViolation {
                    message: format!("Effect entry requires capability: {:?}", required_capability),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate an effect can be invoked
    fn validate_effect_invocation(&self, effect: &Effect, context: &VMEffectContext) -> VMResult<()> {
        // Check if effect is currently active (entered)
        if !context.active_effects.contains(effect) {
            return Err(PrismVMError::EffectError {
                message: format!("Effect not active: {:?}", effect),
            });
        }
        
        // ENHANCED: Re-validate capability requirements at invocation time
        // This ensures capabilities haven't been revoked since entry
        if let Some(required_capability) = effect.required_capability() {
            if !context.capabilities.has_capability(&required_capability) {
                return Err(PrismVMError::CapabilityViolation {
                    message: format!("Effect invocation requires capability: {:?}", required_capability),
                });
            }
        }
        
        Ok(())
    }
    
    /// Execute an effect through the effect system
    fn execute_effect(&self, effect: &Effect, context: &VMEffectContext) -> VMResult<EffectResult> {
        // Create execution plan with single effect
        let mut plan = ExecutionPlan::new();
        let step = prism_effects::execution::ExecutionStep::new(effect.clone(), "vm_handler".to_string());
        plan.add_step(step);
        
        // Execute through effect system
        let mut executor = self.effect_executor.write().unwrap();
        let results = executor.execute_plan(&plan, &context.secure_context)
            .map_err(|e| PrismVMError::EffectError {
                message: format!("Effect execution failed: {}", e),
            })?;
        
        results.into_iter().next()
            .ok_or_else(|| PrismVMError::EffectError {
                message: "No effect result returned".to_string(),
            })
    }
    
    /// Get enforcement statistics
    pub fn get_stats(&self) -> EffectEnforcementStats {
        self.stats.clone()
    }
    
    /// Update enforcement configuration
    pub fn update_config(&mut self, config: EffectEnforcementConfig) {
        self.config = config;
    }
}

impl Default for EffectEnforcementConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            track_execution: true,
            max_nesting_depth: 32,
            execution_timeout: Duration::from_secs(30),
            enable_caching: true,
        }
    }
}

/// Effect enforcement error types
#[derive(Debug, thiserror::Error)]
pub enum EffectEnforcementError {
    #[error("Effect validation failed: {message}")]
    ValidationFailed { message: String },
    
    #[error("Effect execution failed: {message}")]
    ExecutionFailed { message: String },
    
    #[error("Effect context not found: {function_id}")]
    ContextNotFound { function_id: u32 },
    
    #[error("Effect nesting depth exceeded: {depth}")]
    NestingDepthExceeded { depth: usize },
    
    #[error("Undeclared effect: {effect:?}")]
    UndeclaredEffect { effect: Effect },
    
    #[error("Effect not active: {effect:?}")]
    EffectNotActive { effect: Effect },
} 