//! Backend Integration
//!
//! This module provides integration between the prism-codegen backends and
//! the runtime execution adapters, ensuring no duplication of logic while
//! enabling seamless code generation and execution.

use super::context::{ExecutionTarget, ExecutionContext};
use super::errors::{ExecutionError, ExecutionResult};
use super::manager::{GeneratedCode, GeneratedContent, GenerationMetadata};
use crate::{authority::capability, Executable};
use std::collections::HashMap;
use std::sync::Arc;

/// Integration bridge for codegen backends
#[derive(Debug)]
pub struct BackendIntegrationBridge {
    /// Integration handlers by target
    integrations: HashMap<ExecutionTarget, BackendIntegrationImpl>,
}

impl BackendIntegrationBridge {
    /// Create a new integration bridge
    pub fn new() -> Self {
        let mut integrations: HashMap<ExecutionTarget, BackendIntegrationImpl> = HashMap::new();
        
        // Register all backend integrations
        integrations.insert(ExecutionTarget::TypeScript, BackendIntegrationImpl::TypeScript(TypeScriptIntegration::new()));
        integrations.insert(ExecutionTarget::WebAssembly, BackendIntegrationImpl::WebAssembly(WebAssemblyIntegration::new()));
        integrations.insert(ExecutionTarget::Native, BackendIntegrationImpl::Native(NativeIntegration::new()));
        integrations.insert(ExecutionTarget::PrismVM, BackendIntegrationImpl::PrismVM(PrismVMIntegration::new()));
        
        // Note: JavaScript and Python targets would be handled by TypeScript integration
        // as they share similar execution environments. If separate targets are needed,
        // they can be added here with their own integration implementations.
        
        Self { integrations }
    }
    
    /// Execute generated code using the appropriate backend integration
    pub async fn execute_generated_code<T>(
        &self,
        target: ExecutionTarget,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T>
    where
        T: Send + 'static,
    {
        let integration = self.integrations.get(&target)
            .ok_or_else(|| ExecutionError::UnsupportedTarget { target })?;
            
        integration.execute_generated_code(generated_code, capabilities, context).await
    }
    
    /// Check if a target has backend integration
    pub fn has_integration(&self, target: ExecutionTarget) -> bool {
        self.integrations.contains_key(&target)
    }
    
    /// Get integration capabilities for a target
    pub fn get_integration_capabilities(&self, target: ExecutionTarget) -> ExecutionResult<IntegrationCapabilities> {
        let integration = self.integrations.get(&target)
            .ok_or_else(|| ExecutionError::UnsupportedTarget { target })?;
            
        Ok(integration.capabilities())
    }
}

/// Backend integration implementation enum
#[derive(Debug, Clone)]
pub enum BackendIntegrationImpl {
    TypeScript(TypeScriptIntegration),
    WebAssembly(WebAssemblyIntegration),
    Native(NativeIntegration),
    PrismVM(PrismVMIntegration),
}

impl BackendIntegrationImpl {
    /// Execute generated code
    pub async fn execute_generated_code<T>(
        &self,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T>
    where
        T: Send + 'static,
    {
        match self {
            Self::TypeScript(integration) => integration.execute_generated_code(generated_code, capabilities, context).await,
            Self::WebAssembly(integration) => integration.execute_generated_code(generated_code, capabilities, context).await,
            Self::Native(integration) => integration.execute_generated_code(generated_code, capabilities, context).await,
            Self::PrismVM(integration) => integration.execute_generated_code(generated_code, capabilities, context).await,
        }
    }
    
    /// Get integration capabilities
    pub fn capabilities(&self) -> IntegrationCapabilities {
        match self {
            Self::TypeScript(integration) => integration.capabilities(),
            Self::WebAssembly(integration) => integration.capabilities(),
            Self::Native(integration) => integration.capabilities(),
            Self::PrismVM(integration) => integration.capabilities(),
        }
    }
    
    /// Get backend name
    pub fn backend_name(&self) -> &str {
        match self {
            Self::TypeScript(integration) => integration.backend_name(),
            Self::WebAssembly(integration) => integration.backend_name(),
            Self::Native(integration) => integration.backend_name(),
            Self::PrismVM(integration) => integration.backend_name(),
        }
    }
    
    /// Check if this integration is available
    pub fn is_available(&self) -> bool {
        match self {
            Self::TypeScript(integration) => integration.is_available(),
            Self::WebAssembly(integration) => integration.is_available(),
            Self::Native(integration) => integration.is_available(),
            Self::PrismVM(integration) => integration.is_available(),
        }
    }
}

/// Trait for individual backend integrations  
pub trait BackendIntegration: Send + Sync {
    /// Execute generated code
    async fn execute_generated_code<T>(
        &self,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T>
    where
        T: Send + 'static;
    
    /// Get integration capabilities
    fn capabilities(&self) -> IntegrationCapabilities;
    
    /// Get backend name
    fn backend_name(&self) -> &str;
    
    /// Check if this integration is available
    fn is_available(&self) -> bool { true }
}

/// Capabilities of a backend integration
#[derive(Debug, Clone)]
pub struct IntegrationCapabilities {
    /// Supported content types
    pub supported_content_types: Vec<ContentType>,
    /// Supports source maps
    pub supports_source_maps: bool,
    /// Supports debugging
    pub supports_debugging: bool,
    /// Supports hot reload
    pub supports_hot_reload: bool,
    /// Performance characteristics
    pub performance_tier: PerformanceTier,
}

/// Types of generated content
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    SourceCode,
    Bytecode,
    NativeBinary,
}

/// Performance tiers for different backends
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceTier {
    /// Interpreted execution (slower startup, good for development)
    Interpreted,
    /// JIT compiled (medium startup, good runtime performance)
    JIT,
    /// AOT compiled (fast startup, excellent runtime performance)
    AOT,
}

/// TypeScript backend integration
#[derive(Debug, Clone)]
pub struct TypeScriptIntegration {
    /// Integration configuration
    config: TypeScriptIntegrationConfig,
}

impl TypeScriptIntegration {
    pub fn new() -> Self {
        Self {
            config: TypeScriptIntegrationConfig::default(),
        }
    }
}

impl BackendIntegration for TypeScriptIntegration {
    async fn execute_generated_code<T>(
        &self,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T> {
        // Verify we have TypeScript source code
        let source_code = match &generated_code.content {
            GeneratedContent::SourceCode(code) => code,
            _ => return Err(ExecutionError::Generic {
                message: "TypeScript integration expects source code".to_string(),
            }),
        };
        
        // In a real implementation, this would:
        // 1. Set up a Node.js/V8 runtime environment
        // 2. Load the TypeScript runtime if needed
        // 3. Execute the generated TypeScript/JavaScript code
        // 4. Apply capability restrictions during execution
        // 5. Return the result
        
        // For now, return a placeholder error
        Err(ExecutionError::Generic {
            message: "TypeScript execution integration not yet fully implemented".to_string(),
        })
    }
    
    fn capabilities(&self) -> IntegrationCapabilities {
        IntegrationCapabilities {
            supported_content_types: vec![ContentType::SourceCode],
            supports_source_maps: true,
            supports_debugging: true,
            supports_hot_reload: true,
            performance_tier: PerformanceTier::JIT,
        }
    }
    
    fn backend_name(&self) -> &str {
        "TypeScriptBackend"
    }
}

/// WebAssembly backend integration
#[derive(Debug, Clone)]
pub struct WebAssemblyIntegration {
    /// Integration configuration
    config: WebAssemblyIntegrationConfig,
}

impl WebAssemblyIntegration {
    pub fn new() -> Self {
        Self {
            config: WebAssemblyIntegrationConfig::default(),
        }
    }
}

impl BackendIntegration for WebAssemblyIntegration {
    async fn execute_generated_code<T>(
        &self,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T> {
        // Verify we have WebAssembly bytecode
        let bytecode = match &generated_code.content {
            GeneratedContent::Bytecode(code) => code,
            _ => return Err(ExecutionError::Generic {
                message: "WebAssembly integration expects bytecode".to_string(),
            }),
        };
        
        // In a real implementation, this would:
        // 1. Set up a WebAssembly runtime (wasmtime, wasmer, etc.)
        // 2. Load the WebAssembly module from bytecode
        // 3. Set up WASI imports if needed
        // 4. Execute with capability restrictions
        // 5. Return the result
        
        // For now, return a placeholder error
        Err(ExecutionError::Generic {
            message: "WebAssembly execution integration not yet fully implemented".to_string(),
        })
    }
    
    fn capabilities(&self) -> IntegrationCapabilities {
        IntegrationCapabilities {
            supported_content_types: vec![ContentType::Bytecode],
            supports_source_maps: true,
            supports_debugging: false, // WASM debugging is limited
            supports_hot_reload: false,
            performance_tier: PerformanceTier::AOT,
        }
    }
    
    fn backend_name(&self) -> &str {
        "WebAssemblyBackend"
    }
}

/// Native LLVM backend integration
#[derive(Debug, Clone)]
pub struct NativeIntegration {
    /// Integration configuration
    config: NativeIntegrationConfig,
}

impl NativeIntegration {
    pub fn new() -> Self {
        Self {
            config: NativeIntegrationConfig::default(),
        }
    }
}

impl BackendIntegration for NativeIntegration {
    async fn execute_generated_code<T>(
        &self,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T> {
        // Verify we have native binary
        let binary = match &generated_code.content {
            GeneratedContent::NativeBinary(code) => code,
            _ => return Err(ExecutionError::Generic {
                message: "Native integration expects native binary".to_string(),
            }),
        };
        
        // In a real implementation, this would:
        // 1. Write the binary to a temporary executable file
        // 2. Set appropriate permissions
        // 3. Execute the binary in a sandboxed environment
        // 4. Apply capability restrictions via system-level mechanisms
        // 5. Return the result
        
        // For now, return a placeholder error
        Err(ExecutionError::Generic {
            message: "Native execution integration not yet fully implemented".to_string(),
        })
    }
    
    fn capabilities(&self) -> IntegrationCapabilities {
        IntegrationCapabilities {
            supported_content_types: vec![ContentType::NativeBinary],
            supports_source_maps: false,
            supports_debugging: true,
            supports_hot_reload: false,
            performance_tier: PerformanceTier::AOT,
        }
    }
    
    fn backend_name(&self) -> &str {
        "LLVMBackend"
    }
}

/// Prism VM backend integration
#[derive(Debug, Clone)]
pub struct PrismVMIntegration {
    /// Integration configuration
    config: PrismVMIntegrationConfig,
}

impl PrismVMIntegration {
    pub fn new() -> Self {
        Self {
            config: PrismVMIntegrationConfig::default(),
        }
    }
}

impl BackendIntegration for PrismVMIntegration {
    async fn execute_generated_code<T>(
        &self,
        generated_code: &GeneratedCode,
        capabilities: &capability::CapabilitySet,
        context: &ExecutionContext,
    ) -> ExecutionResult<T> {
        // Verify we have Prism bytecode
        let bytecode = match &generated_code.content {
            GeneratedContent::Bytecode(code) => code,
            _ => return Err(ExecutionError::Generic {
                message: "Prism VM integration expects bytecode".to_string(),
            }),
        };
        
        // In a real implementation, this would:
        // 1. Create a Prism VM instance
        // 2. Load the bytecode into the VM
        // 3. Set up capability restrictions in the VM
        // 4. Execute the bytecode
        // 5. Return the result
        
        // For now, return a placeholder error
        Err(ExecutionError::Generic {
            message: "Prism VM execution integration not yet fully implemented".to_string(),
        })
    }
    
    fn capabilities(&self) -> IntegrationCapabilities {
        IntegrationCapabilities {
            supported_content_types: vec![ContentType::Bytecode],
            supports_source_maps: true,
            supports_debugging: true,
            supports_hot_reload: true,
            performance_tier: PerformanceTier::JIT,
        }
    }
    
    fn backend_name(&self) -> &str {
        "PrismVMBackend"
    }
}

// Configuration types for each integration

#[derive(Debug, Clone)]
pub struct TypeScriptIntegrationConfig {
    pub node_runtime_path: Option<String>,
    pub enable_typescript_checking: bool,
    pub target_version: String,
}

impl Default for TypeScriptIntegrationConfig {
    fn default() -> Self {
        Self {
            node_runtime_path: None, // Use system Node.js
            enable_typescript_checking: true,
            target_version: "ES2020".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WebAssemblyIntegrationConfig {
    pub runtime_engine: WasmEngine,
    pub enable_wasi: bool,
    pub memory_limit_mb: usize,
}

#[derive(Debug, Clone)]
pub enum WasmEngine {
    Wasmtime,
    Wasmer,
    V8,
}

impl Default for WebAssemblyIntegrationConfig {
    fn default() -> Self {
        Self {
            runtime_engine: WasmEngine::Wasmtime,
            enable_wasi: true,
            memory_limit_mb: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NativeIntegrationConfig {
    pub sandbox_enabled: bool,
    pub execution_timeout_secs: u64,
    pub temp_dir: Option<String>,
}

impl Default for NativeIntegrationConfig {
    fn default() -> Self {
        Self {
            sandbox_enabled: true,
            execution_timeout_secs: 30,
            temp_dir: None, // Use system temp dir
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrismVMIntegrationConfig {
    pub enable_jit: bool,
    pub stack_size_mb: usize,
    pub enable_debugging: bool,
}

impl Default for PrismVMIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_jit: false, // Conservative default
            stack_size_mb: 1,
            enable_debugging: true,
        }
    }
} 