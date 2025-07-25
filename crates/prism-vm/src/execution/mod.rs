//! Execution Engine
//!
//! This module implements the execution engine for the Prism VM, including
//! the stack-based interpreter and JIT compiler integration.

pub mod interpreter;
pub mod stack;
pub mod effects;

#[cfg(feature = "jit")]
pub mod jit;

// Re-export main types
pub use interpreter::{Interpreter, InterpreterConfig};
pub use stack::{ExecutionStack, StackFrame, StackValue, AdvancedStackManager};
pub use effects::{VMEffectEnforcer, VMEffectContext, EffectEnforcementConfig, EffectEnforcementStats};

use crate::bytecode::SemanticInformationRegistry;

#[cfg(feature = "jit")]
pub use jit::{JitCompiler, JitConfig};

use crate::{VMResult, PrismVMError, bytecode::PrismBytecode};
use prism_runtime::authority::capability::CapabilitySet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, span, Level};

/// Main Prism VM instance
#[derive(Debug)]
pub struct PrismVM {
    /// Bytecode interpreter
    interpreter: Interpreter,
    
    /// JIT compiler (optional)
    #[cfg(feature = "jit")]
    jit_compiler: Option<JitCompiler>,
    
    /// VM configuration
    config: VMConfig,
    
    /// Loaded bytecode modules
    modules: HashMap<String, Arc<PrismBytecode>>,
    
    /// Global capability set
    capabilities: CapabilitySet,
    
    /// Effect enforcer for runtime validation
    effect_enforcer: Option<Arc<RwLock<VMEffectEnforcer>>>,
}

/// VM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMConfig {
    /// Maximum stack size
    pub max_stack_size: usize,
    /// Enable JIT compilation
    pub enable_jit: bool,
    /// JIT compilation threshold (number of executions before JIT)
    pub jit_threshold: u32,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable debugging
    pub enable_debugging: bool,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: Option<u64>,
    /// Garbage collection configuration
    pub gc_config: GCConfig,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCConfig {
    /// Enable garbage collection
    pub enabled: bool,
    /// Initial heap size in bytes
    pub initial_heap_size: usize,
    /// Maximum heap size in bytes
    pub max_heap_size: usize,
    /// GC trigger threshold (percentage of heap used)
    pub gc_threshold: f64,
}

impl Default for VMConfig {
    fn default() -> Self {
        Self {
            max_stack_size: 1024 * 1024, // 1MB stack
            enable_jit: false, // Disabled by default for now
            jit_threshold: 100,
            enable_profiling: false,
            enable_debugging: true,
            max_execution_time_ms: None,
            gc_config: GCConfig::default(),
        }
    }
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_heap_size: 1024 * 1024, // 1MB
            max_heap_size: 100 * 1024 * 1024, // 100MB
            gc_threshold: 0.75, // Trigger GC at 75% heap usage
        }
    }
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Return value
    pub return_value: Option<StackValue>,
    /// Execution statistics
    pub stats: ExecutionStats,
    /// Whether execution completed successfully
    pub success: bool,
}

/// Execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total execution time in microseconds
    pub execution_time_us: u64,
    /// Number of instructions executed
    pub instructions_executed: u64,
    /// Maximum stack depth reached
    pub max_stack_depth: usize,
    /// Number of function calls
    pub function_calls: u64,
    /// Number of garbage collections triggered
    pub gc_collections: u64,
    /// Memory allocated in bytes
    pub memory_allocated: u64,
}

/// VM error type
pub type VMError = PrismVMError;

impl PrismVM {
    /// Create a new Prism VM with default configuration
    pub fn new() -> VMResult<Self> {
        Self::with_config(VMConfig::default())
    }

    /// Create a new Prism VM with custom configuration
    pub fn with_config(config: VMConfig) -> VMResult<Self> {
        let _span = span!(Level::INFO, "vm_init").entered();
        info!("Initializing Prism VM with config: {:?}", config);

        let interpreter = Interpreter::new(InterpreterConfig {
            max_stack_size: config.max_stack_size,
            enable_profiling: config.enable_profiling,
            enable_debugging: config.enable_debugging,
        })?;

        #[cfg(feature = "jit")]
        let jit_compiler = if config.enable_jit {
            Some(JitCompiler::new(JitConfig {
                threshold: config.jit_threshold,
            })?)
        } else {
            None
        };

        Ok(Self {
            interpreter,
            #[cfg(feature = "jit")]
            jit_compiler,
            config,
            modules: HashMap::new(),
            capabilities: CapabilitySet::new(),
            effect_enforcer: None,
        })
    }

    /// Load a bytecode module with semantic information preservation
    pub fn load_module(&mut self, name: String, mut bytecode: PrismBytecode) -> VMResult<()> {
        let _span = span!(Level::INFO, "load_module", module = %name).entered();
        debug!("Loading module with semantic preservation: {}", name);

        // Validate bytecode
        bytecode.validate()?;

        // Initialize semantic registry from bytecode
        bytecode.initialize_semantic_registry();

        // Store the module
        self.modules.insert(name.clone(), Arc::new(bytecode));

        info!("Successfully loaded module with semantic information: {}", name);
        Ok(())
    }

    /// Get semantic information for a type across all loaded modules
    pub fn get_type_semantic_info(&self, type_id: u32) -> Option<&crate::bytecode::BytecodeSemanticMetadata> {
        for module in self.modules.values() {
            if let Some(metadata) = module.get_type_semantic_metadata(type_id) {
                return Some(metadata);
            }
        }
        None
    }

    /// Query types by business domain across all modules
    pub fn query_types_by_domain(&self, domain: &str) -> Vec<(String, u32, &crate::bytecode::BytecodeSemanticMetadata)> {
        let mut results = Vec::new();
        
        for (module_name, module) in &self.modules {
            if let Some(registry) = &module.semantic_registry {
                let domain_types = registry.query_types_by_domain(domain);
                for (type_id, metadata) in domain_types {
                    results.push((module_name.clone(), type_id, metadata));
                }
            }
        }
        
        results
    }

    /// Execute a function by name
    pub fn execute_function(&mut self, module_name: &str, function_name: &str, args: Vec<StackValue>) -> VMResult<ExecutionResult> {
        let _span = span!(Level::INFO, "execute_function", 
            module = %module_name, 
            function = %function_name
        ).entered();

        // Get the module
        let module = self.modules.get(module_name)
            .ok_or_else(|| PrismVMError::ExecutionError {
                message: format!("Module not found: {}", module_name),
            })?;

        // Get the function
        let function = module.get_function(function_name)
            .ok_or_else(|| PrismVMError::ExecutionError {
                message: format!("Function not found: {}", function_name),
            })?;

        // Check argument count
        if args.len() != function.param_count as usize {
            return Err(PrismVMError::ExecutionError {
                message: format!("Argument count mismatch: expected {}, got {}", 
                    function.param_count, args.len()),
            });
        }

        // ENHANCED: Comprehensive capability verification at function level
        // This matches compile-time verification completeness
        for required_capability in &function.capabilities {
            if !self.capabilities.has_capability(required_capability) {
                return Err(PrismVMError::CapabilityViolation {
                    message: format!("Function {}::{} requires capability: {:?}", 
                        module_name, function_name, required_capability),
                });
            }
        }

        // ENHANCED: Verify effect-related capabilities for declared effects
        for effect in &function.effects {
            if let Some(required_capability) = effect.required_capability() {
                if !self.capabilities.has_capability(&required_capability) {
                    return Err(PrismVMError::CapabilityViolation {
                        message: format!("Function {}::{} effect {:?} requires capability: {:?}", 
                            module_name, function_name, effect, required_capability),
                    });
                }
            }
        }

        // ENHANCED: Pre-validate instruction-level capabilities to catch issues early
        // This provides better error messages and prevents partial execution
        for instruction in &function.instructions {
            for required_cap in &instruction.required_capabilities {
                if !self.capabilities.has_capability(required_cap) {
                    return Err(PrismVMError::CapabilityViolation {
                        message: format!("Function {}::{} contains instruction requiring capability: {:?}", 
                            module_name, function_name, required_cap),
                    });
                }
            }
        }

        info!("Executing function: {}::{}", module_name, function_name);

        // Execute using interpreter
        let result = self.interpreter.execute_function(module.as_ref(), function, args)?;

        debug!("Function execution completed: {:?}", result.stats);
        Ok(result)
    }

    /// Execute the main function of a module
    pub fn execute_main(&mut self, module_name: &str) -> VMResult<ExecutionResult> {
        self.execute_function(module_name, "main", vec![])
    }

    /// Set up effect enforcement with the provided effect system components
    pub fn setup_effect_enforcement(
        &mut self,
        effect_system: Arc<RwLock<prism_effects::effects::EffectSystem>>,
        effect_executor: Arc<RwLock<prism_effects::execution::ExecutionSystem>>,
        effect_validator: Arc<prism_effects::validation::EffectValidator>,
    ) -> VMResult<()> {
        let _span = span!(Level::INFO, "setup_effect_enforcement").entered();
        info!("Setting up VM effect enforcement");
        
        // Create the effect enforcer
        let enforcer = Arc::new(RwLock::new(VMEffectEnforcer::new(
            effect_system,
            effect_executor,
            effect_validator,
        )));
        
        // Set it in the VM
        self.effect_enforcer = Some(enforcer.clone());
        
        // Set it in the interpreter
        self.interpreter.set_effect_enforcer(enforcer.clone());
        
        // Set it in the JIT compiler if available
        #[cfg(feature = "jit")]
        if let Some(ref mut jit_compiler) = self.jit_compiler {
            jit_compiler.set_effect_enforcer(enforcer.clone());
        }
        
        info!("Effect enforcement setup completed");
        Ok(())
    }
    
    /// Get effect enforcement statistics
    pub fn get_effect_stats(&self) -> Option<EffectEnforcementStats> {
        self.effect_enforcer.as_ref().map(|enforcer| {
            let enforcer = enforcer.read().unwrap();
            let mut stats = enforcer.get_stats();
            
            // Add JIT effect statistics if available
            #[cfg(feature = "jit")]
            if let Some(ref jit_compiler) = self.jit_compiler {
                if let Some(jit_effect_stats) = jit_compiler.get_effect_stats() {
                    // Merge JIT statistics with interpreter statistics
                    stats.contexts_created += jit_effect_stats.contexts_created;
                    stats.effects_entered += jit_effect_stats.effects_entered;
                    stats.effects_executed += jit_effect_stats.effects_executed;
                    stats.violations_detected += jit_effect_stats.violations_detected;
                    stats.validation_time += jit_effect_stats.validation_time;
                    stats.execution_time += jit_effect_stats.execution_time;
                }
            }
            
            stats
        })
    }
    
    /// Update effect enforcement configuration
    pub fn update_effect_config(&mut self, config: EffectEnforcementConfig) -> VMResult<()> {
        if let Some(ref effect_enforcer) = self.effect_enforcer {
            let mut enforcer = effect_enforcer.write().unwrap();
            enforcer.update_config(config);
            Ok(())
        } else {
            Err(PrismVMError::EffectError {
                message: "Effect enforcer not initialized".to_string(),
            })
        }
    }

    /// Get VM statistics
    pub fn get_stats(&self) -> VMStats {
        VMStats {
            modules_loaded: self.modules.len(),
            interpreter_stats: self.interpreter.get_stats(),
            #[cfg(feature = "jit")]
            jit_stats: self.jit_compiler.as_ref().map(|jit| jit.get_stats()),
            effect_stats: self.get_effect_stats(),
        }
    }

    /// Set capabilities for the VM
    pub fn set_capabilities(&mut self, capabilities: CapabilitySet) {
        self.capabilities = capabilities.clone();
        self.interpreter.set_capabilities(capabilities);
    }

    /// Get current capabilities
    pub fn get_capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    /// Shutdown the VM and cleanup resources
    pub fn shutdown(self) -> VMResult<()> {
        let _span = span!(Level::INFO, "vm_shutdown").entered();
        info!("Shutting down Prism VM");

        // Shutdown interpreter
        self.interpreter.shutdown()?;

        #[cfg(feature = "jit")]
        if let Some(jit) = self.jit_compiler {
            jit.shutdown()?;
        }

        info!("VM shutdown complete");
        Ok(())
    }
}

impl Default for PrismVM {
    fn default() -> Self {
        Self::new().expect("Failed to create default VM")
    }
}

/// VM statistics
#[derive(Debug, Clone)]
pub struct VMStats {
    /// Number of modules loaded
    pub modules_loaded: usize,
    /// Interpreter statistics
    pub interpreter_stats: interpreter::InterpreterStats,
    /// JIT compiler statistics (if enabled)
    #[cfg(feature = "jit")]
    pub jit_stats: Option<jit::JitStats>,
    /// Effect enforcement statistics
    pub effect_stats: Option<EffectEnforcementStats>,
} 