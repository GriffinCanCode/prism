//! Target-Specific Execution Adapters
//!
//! This module provides adapters for executing code on different target platforms
//! (TypeScript, WebAssembly, Native, PrismVM) while maintaining capability-based
//! security and effect tracking.

use super::context::{ExecutionContext, ExecutionTarget, TypeScriptConfig, WebAssemblyConfig, NativeConfig, PrismVMConfig};
use super::errors::ExecutionError;
use crate::{authority::capability, Executable, RuntimeError};
use std::collections::HashMap;

/// Trait for target-specific execution adapters
pub trait TargetAdapter: Send + Sync {
    /// Execute code on this target
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError>;
    
    /// Get adapter name
    fn name(&self) -> &'static str;
    
    /// Check if adapter is available
    fn is_available(&self) -> bool { true }
    
    /// Get supported features for this adapter
    fn supported_features(&self) -> Vec<String> {
        vec!["basic_execution".to_string()]
    }
    
    /// Initialize the adapter with configuration
    fn initialize(&mut self, _config: &ExecutionContext) -> Result<(), ExecutionError> {
        Ok(())
    }
    
    /// Cleanup after execution
    fn cleanup(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }
}

/// Concrete implementation of target adapters
#[derive(Debug)]
pub enum TargetAdapterImpl {
    TypeScript(TypeScriptAdapter),
    WebAssembly(WebAssemblyAdapter),
    Native(NativeAdapter),
    PrismVM(PrismVMAdapter),
}

impl TargetAdapterImpl {
    /// Create a new adapter for the specified target
    pub fn new(target: ExecutionTarget) -> Result<Self, ExecutionError> {
        match target {
            ExecutionTarget::TypeScript => Ok(Self::TypeScript(TypeScriptAdapter::new()?)),
            ExecutionTarget::WebAssembly => Ok(Self::WebAssembly(WebAssemblyAdapter::new()?)),
            ExecutionTarget::Native => Ok(Self::Native(NativeAdapter::new()?)),
            ExecutionTarget::PrismVM => Ok(Self::PrismVM(PrismVMAdapter::new()?)),
        }
    }
    
    /// Execute code on this target adapter
    pub fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.execute(code, capabilities, context),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.execute(code, capabilities, context),
            TargetAdapterImpl::Native(adapter) => adapter.execute(code, capabilities, context),
            TargetAdapterImpl::PrismVM(adapter) => adapter.execute(code, capabilities, context),
        }
    }
    
    /// Get adapter name
    pub fn name(&self) -> &'static str {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.name(),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.name(),
            TargetAdapterImpl::Native(adapter) => adapter.name(),
            TargetAdapterImpl::PrismVM(adapter) => adapter.name(),
        }
    }
    
    /// Check if adapter is available
    pub fn is_available(&self) -> bool {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.is_available(),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.is_available(),
            TargetAdapterImpl::Native(adapter) => adapter.is_available(),
            TargetAdapterImpl::PrismVM(adapter) => adapter.is_available(),
        }
    }
    
    /// Get supported features
    pub fn supported_features(&self) -> Vec<String> {
        match self {
            TargetAdapterImpl::TypeScript(adapter) => adapter.supported_features(),
            TargetAdapterImpl::WebAssembly(adapter) => adapter.supported_features(),
            TargetAdapterImpl::Native(adapter) => adapter.supported_features(),
            TargetAdapterImpl::PrismVM(adapter) => adapter.supported_features(),
        }
    }
}

/// TypeScript execution adapter
#[derive(Debug)]
pub struct TypeScriptAdapter {
    /// Adapter configuration
    config: TypeScriptConfig,
}

impl TypeScriptAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: TypeScriptConfig::default(),
        })
    }
    
    pub fn with_config(config: TypeScriptConfig) -> Result<Self, ExecutionError> {
        Ok(Self { config })
    }
}

impl TargetAdapter for TypeScriptAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Transpile code to JavaScript using the integrated codegen backend
        // 2. Set up Node.js/V8 execution environment
        // 3. Execute with capability restrictions
        // 4. Return result
        
        // For now, delegate to the code's execute method
        code.execute(capabilities, context)
            .map_err(|e| match e {
                RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("TypeScript execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "TypeScript"
    }
    
    fn supported_features(&self) -> Vec<String> {
        vec![
            "basic_execution".to_string(),
            "type_checking".to_string(),
            "source_maps".to_string(),
            "npm_integration".to_string(),
        ]
    }
}

/// WebAssembly execution adapter
#[derive(Debug)]
pub struct WebAssemblyAdapter {
    /// WASM runtime configuration
    config: WebAssemblyConfig,
}

impl WebAssemblyAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: WebAssemblyConfig::default(),
        })
    }
    
    pub fn with_config(config: WebAssemblyConfig) -> Result<Self, ExecutionError> {
        Ok(Self { config })
    }
}

impl TargetAdapter for WebAssemblyAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Compile code to WebAssembly using the integrated codegen backend
        // 2. Set up WASM runtime (wasmtime, wasmer, etc.)
        // 3. Execute with capability restrictions
        // 4. Return result
        
        // For now, delegate to the code's execute method
        code.execute(capabilities, context)
            .map_err(|e| match e {
                RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("WebAssembly execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "WebAssembly"
    }
    
    fn supported_features(&self) -> Vec<String> {
        vec![
            "basic_execution".to_string(),
            "memory_safety".to_string(),
            "portable_execution".to_string(),
            "simd_support".to_string(),
        ]
    }
}

/// Native execution adapter
#[derive(Debug)]
pub struct NativeAdapter {
    /// Native execution configuration
    config: NativeConfig,
}

impl NativeAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: NativeConfig::default(),
        })
    }
    
    pub fn with_config(config: NativeConfig) -> Result<Self, ExecutionError> {
        Ok(Self { config })
    }
}

impl TargetAdapter for NativeAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Compile code to native binary using the integrated LLVM backend
        // 2. Set up native execution environment
        // 3. Execute with capability restrictions
        // 4. Return result
        
        // For now, delegate to the code's execute method
        code.execute(capabilities, context)
            .map_err(|e| match e {
                RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("Native execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "Native"
    }
    
    fn supported_features(&self) -> Vec<String> {
        vec![
            "basic_execution".to_string(),
            "high_performance".to_string(),
            "debug_info".to_string(),
            "optimization".to_string(),
        ]
    }
}

/// Prism VM execution adapter
#[derive(Debug)]
pub struct PrismVMAdapter {
    /// VM configuration
    config: PrismVMConfig,
}

impl PrismVMAdapter {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            config: PrismVMConfig::default(),
        })
    }

    pub fn with_config(config: PrismVMConfig) -> Result<Self, ExecutionError> {
        Ok(Self { config })
    }
}

impl TargetAdapter for PrismVMAdapter {
    fn execute<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> Result<T, ExecutionError> {
        // In a real implementation, this would:
        // 1. Load Prism bytecode from the code object using the integrated VM backend
        // 2. Create a Prism VM instance with appropriate configuration
        // 3. Set capabilities on the VM
        // 4. Execute the bytecode
        // 5. Return the result
        
        // For now, delegate to the code's execute method with VM-specific error handling
        code.execute(capabilities, context)
            .map_err(|e| match e {
                RuntimeError::Authority(cap_err) => ExecutionError::Capability(cap_err),
                RuntimeError::Resource(resource_err) => ExecutionError::Resource(resource_err),
                _ => ExecutionError::Generic { 
                    message: format!("Prism VM execution failed: {}", e) 
                },
            })
    }
    
    fn name(&self) -> &'static str {
        "PrismVM"
    }
    
    fn is_available(&self) -> bool {
        // Check if Prism VM is available
        // For now, always return true since we have the VM implementation
        true
    }
    
    fn supported_features(&self) -> Vec<String> {
        vec![
            "basic_execution".to_string(),
            "bytecode_execution".to_string(),
            "capability_integration".to_string(),
            "effect_tracking".to_string(),
            "debugging_support".to_string(),
        ]
    }
}

/// Registry for managing target adapters
#[derive(Debug)]
pub struct AdapterRegistry {
    /// Registered adapters by target
    adapters: HashMap<ExecutionTarget, TargetAdapterImpl>,
}

impl AdapterRegistry {
    /// Create a new adapter registry
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }
    
    /// Register all default adapters
    pub fn register_defaults(&mut self) -> Result<(), ExecutionError> {
        self.register_adapter(ExecutionTarget::TypeScript, TargetAdapterImpl::new(ExecutionTarget::TypeScript)?)?;
        self.register_adapter(ExecutionTarget::WebAssembly, TargetAdapterImpl::new(ExecutionTarget::WebAssembly)?)?;
        self.register_adapter(ExecutionTarget::Native, TargetAdapterImpl::new(ExecutionTarget::Native)?)?;
        self.register_adapter(ExecutionTarget::PrismVM, TargetAdapterImpl::new(ExecutionTarget::PrismVM)?)?;
        Ok(())
    }
    
    /// Register an adapter for a target
    pub fn register_adapter(&mut self, target: ExecutionTarget, adapter: TargetAdapterImpl) -> Result<(), ExecutionError> {
        self.adapters.insert(target, adapter);
        Ok(())
    }
    
    /// Get an adapter for a target
    pub fn get_adapter(&self, target: ExecutionTarget) -> Result<&TargetAdapterImpl, ExecutionError> {
        self.adapters.get(&target).ok_or(ExecutionError::UnsupportedTarget { target })
    }
    
    /// Get all available targets
    pub fn available_targets(&self) -> Vec<ExecutionTarget> {
        self.adapters.keys().copied().collect()
    }
    
    /// Check if a target is supported
    pub fn is_target_supported(&self, target: ExecutionTarget) -> bool {
        self.adapters.contains_key(&target)
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register_defaults().expect("Failed to register default adapters");
        registry
    }
} 