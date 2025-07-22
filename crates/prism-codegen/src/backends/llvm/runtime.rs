//! LLVM Runtime Integration
//!
//! This module handles integration with the Prism runtime system,
//! including capability validation, effect tracking, and security enforcement.

use super::{LLVMResult, LLVMError};
use super::types::{LLVMTargetArch, LLVMOptimizationLevel};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// LLVM runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMRuntimeConfig {
    /// Target architecture
    pub target_arch: LLVMTargetArch,
    /// Enable capability validation
    pub enable_capability_validation: bool,
    /// Enable effect tracking
    pub enable_effect_tracking: bool,
    /// Enable security enforcement
    pub enable_security_enforcement: bool,
    /// Enable memory safety checks
    pub enable_memory_safety: bool,
    /// Enable concurrency safety
    pub enable_concurrency_safety: bool,
    /// Runtime library path
    pub runtime_library_path: Option<String>,
    /// Custom runtime functions
    pub custom_runtime_functions: Vec<RuntimeFunction>,
    /// Memory allocator type
    pub allocator_type: AllocatorType,
    /// Garbage collection configuration
    pub gc_config: Option<GCConfig>,
}

impl Default for LLVMRuntimeConfig {
    fn default() -> Self {
        Self {
            target_arch: LLVMTargetArch::default(),
            enable_capability_validation: true,
            enable_effect_tracking: true,
            enable_security_enforcement: true,
            enable_memory_safety: true,
            enable_concurrency_safety: true,
            runtime_library_path: None,
            custom_runtime_functions: Vec::new(),
            allocator_type: AllocatorType::System,
            gc_config: None,
        }
    }
}

/// Runtime function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeFunction {
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: String,
    /// Function attributes
    pub attributes: Vec<String>,
    /// Whether function is always available
    pub always_available: bool,
    /// Security level required
    pub security_level: SecurityLevel,
}

/// Memory allocator types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocatorType {
    /// System allocator (malloc/free)
    System,
    /// Custom allocator
    Custom(String),
    /// Garbage collected
    GarbageCollected,
    /// Pool allocator
    Pool,
    /// Stack allocator
    Stack,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCConfig {
    /// GC algorithm type
    pub algorithm: GCAlgorithm,
    /// Initial heap size
    pub initial_heap_size: usize,
    /// Maximum heap size
    pub max_heap_size: usize,
    /// GC trigger threshold
    pub gc_threshold: f64,
    /// Enable incremental GC
    pub incremental: bool,
}

/// Garbage collection algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GCAlgorithm {
    /// Mark and sweep
    MarkAndSweep,
    /// Copying collector
    Copying,
    /// Generational
    Generational,
    /// Reference counting
    ReferenceCounting,
}

/// Security levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// No security requirements
    None,
    /// Basic security
    Basic,
    /// Enhanced security
    Enhanced,
    /// Critical security
    Critical,
}

/// LLVM runtime integration manager
pub struct LLVMRuntime {
    /// Runtime configuration
    config: LLVMRuntimeConfig,
    /// Runtime function declarations
    runtime_functions: HashMap<String, RuntimeFunctionDecl>,
    /// Capability validation functions
    capability_validators: HashMap<String, String>,
    /// Effect tracking functions
    effect_trackers: HashMap<String, String>,
    /// Security enforcement functions
    security_enforcers: HashMap<String, String>,
}

/// Runtime function declaration
#[derive(Debug, Clone)]
pub struct RuntimeFunctionDecl {
    /// Function name in LLVM IR
    pub llvm_name: String,
    /// Return type
    pub return_type: String,
    /// Parameter types
    pub param_types: Vec<String>,
    /// Function attributes
    pub attributes: Vec<String>,
    /// Whether function is external
    pub is_external: bool,
    /// Security level required
    pub security_level: SecurityLevel,
}

/// Runtime integration result
#[derive(Debug, Clone)]
pub struct RuntimeIntegrationResult {
    /// Generated runtime declarations
    pub declarations: Vec<String>,
    /// Generated runtime calls
    pub calls: Vec<String>,
    /// Required runtime libraries
    pub libraries: Vec<String>,
    /// Initialization code
    pub initialization: Vec<String>,
    /// Cleanup code
    pub cleanup: Vec<String>,
}

impl LLVMRuntime {
    /// Create new runtime integration manager
    pub fn new(config: LLVMRuntimeConfig) -> Self {
        let mut runtime = Self {
            config,
            runtime_functions: HashMap::new(),
            capability_validators: HashMap::new(),
            effect_trackers: HashMap::new(),
            security_enforcers: HashMap::new(),
        };

        runtime.initialize_builtin_functions();
        runtime
    }

    /// Initialize built-in runtime functions
    fn initialize_builtin_functions(&mut self) {
        // Capability validation functions
        if self.config.enable_capability_validation {
            self.add_capability_validator("validate_read_capability", 
                "i1 @prism_validate_read_capability(i8*, i64)");
            self.add_capability_validator("validate_write_capability", 
                "i1 @prism_validate_write_capability(i8*, i64)");
            self.add_capability_validator("validate_execute_capability", 
                "i1 @prism_validate_execute_capability(i8*)");
            self.add_capability_validator("validate_network_capability", 
                "i1 @prism_validate_network_capability(i8*, i16)");
            self.add_capability_validator("validate_file_capability", 
                "i1 @prism_validate_file_capability(i8*, i32)");
        }

        // Effect tracking functions
        if self.config.enable_effect_tracking {
            self.add_effect_tracker("begin_effect_tracking", 
                "void @prism_begin_effect_tracking(i8*)");
            self.add_effect_tracker("end_effect_tracking", 
                "void @prism_end_effect_tracking(i8*)");
            self.add_effect_tracker("track_io_effect", 
                "void @prism_track_io_effect(i8*, i64)");
            self.add_effect_tracker("track_network_effect", 
                "void @prism_track_network_effect(i8*, i64)");
            self.add_effect_tracker("track_computation_effect", 
                "void @prism_track_computation_effect(i64)");
        }

        // Security enforcement functions
        if self.config.enable_security_enforcement {
            self.add_security_enforcer("check_bounds", 
                "void @prism_check_bounds(i8*, i64, i64)");
            self.add_security_enforcer("validate_pointer", 
                "i1 @prism_validate_pointer(i8*)");
            self.add_security_enforcer("secure_zero", 
                "void @prism_secure_zero(i8*, i64)");
            self.add_security_enforcer("audit_access", 
                "void @prism_audit_access(i8*, i32)");
        }

        // Memory safety functions
        if self.config.enable_memory_safety {
            self.add_runtime_function("safe_malloc", RuntimeFunctionDecl {
                llvm_name: "prism_safe_malloc".to_string(),
                return_type: "i8*".to_string(),
                param_types: vec!["i64".to_string()],
                attributes: vec!["nounwind".to_string(), "malloc".to_string()],
                is_external: true,
                security_level: SecurityLevel::Enhanced,
            });

            self.add_runtime_function("safe_free", RuntimeFunctionDecl {
                llvm_name: "prism_safe_free".to_string(),
                return_type: "void".to_string(),
                param_types: vec!["i8*".to_string()],
                attributes: vec!["nounwind".to_string()],
                is_external: true,
                security_level: SecurityLevel::Enhanced,
            });

            self.add_runtime_function("safe_realloc", RuntimeFunctionDecl {
                llvm_name: "prism_safe_realloc".to_string(),
                return_type: "i8*".to_string(),
                param_types: vec!["i8*".to_string(), "i64".to_string()],
                attributes: vec!["nounwind".to_string()],
                is_external: true,
                security_level: SecurityLevel::Enhanced,
            });
        }

        // Concurrency safety functions
        if self.config.enable_concurrency_safety {
            self.add_runtime_function("acquire_lock", RuntimeFunctionDecl {
                llvm_name: "prism_acquire_lock".to_string(),
                return_type: "void".to_string(),
                param_types: vec!["i8*".to_string()],
                attributes: vec!["nounwind".to_string()],
                is_external: true,
                security_level: SecurityLevel::Basic,
            });

            self.add_runtime_function("release_lock", RuntimeFunctionDecl {
                llvm_name: "prism_release_lock".to_string(),
                return_type: "void".to_string(),
                param_types: vec!["i8*".to_string()],
                attributes: vec!["nounwind".to_string()],
                is_external: true,
                security_level: SecurityLevel::Basic,
            });

            self.add_runtime_function("atomic_compare_exchange", RuntimeFunctionDecl {
                llvm_name: "prism_atomic_cmpxchg".to_string(),
                return_type: "i1".to_string(),
                param_types: vec!["i8*".to_string(), "i64".to_string(), "i64".to_string()],
                attributes: vec!["nounwind".to_string()],
                is_external: true,
                security_level: SecurityLevel::Basic,
            });
        }

        // Add custom runtime functions
        for custom_func in &self.config.custom_runtime_functions.clone() {
            self.add_runtime_function(&custom_func.name, RuntimeFunctionDecl {
                llvm_name: custom_func.name.clone(),
                return_type: self.extract_return_type(&custom_func.signature),
                param_types: self.extract_param_types(&custom_func.signature),
                attributes: custom_func.attributes.clone(),
                is_external: true,
                security_level: custom_func.security_level.clone(),
            });
        }
    }

    /// Add capability validator
    fn add_capability_validator(&mut self, name: &str, signature: &str) {
        self.capability_validators.insert(name.to_string(), signature.to_string());
    }

    /// Add effect tracker
    fn add_effect_tracker(&mut self, name: &str, signature: &str) {
        self.effect_trackers.insert(name.to_string(), signature.to_string());
    }

    /// Add security enforcer
    fn add_security_enforcer(&mut self, name: &str, signature: &str) {
        self.security_enforcers.insert(name.to_string(), signature.to_string());
    }

    /// Add runtime function
    fn add_runtime_function(&mut self, name: &str, decl: RuntimeFunctionDecl) {
        self.runtime_functions.insert(name.to_string(), decl);
    }

    /// Generate runtime integration code
    pub fn generate_runtime_integration(&self, optimization_level: LLVMOptimizationLevel) -> LLVMResult<RuntimeIntegrationResult> {
        let mut result = RuntimeIntegrationResult {
            declarations: Vec::new(),
            calls: Vec::new(),
            libraries: Vec::new(),
            initialization: Vec::new(),
            cleanup: Vec::new(),
        };

        // Generate function declarations
        for (name, decl) in &self.runtime_functions {
            let declaration = self.generate_function_declaration(decl)?;
            result.declarations.push(declaration);
        }

        // Generate capability validator declarations
        for (_, signature) in &self.capability_validators {
            result.declarations.push(format!("declare {}", signature));
        }

        // Generate effect tracker declarations
        for (_, signature) in &self.effect_trackers {
            result.declarations.push(format!("declare {}", signature));
        }

        // Generate security enforcer declarations
        for (_, signature) in &self.security_enforcers {
            result.declarations.push(format!("declare {}", signature));
        }

        // Generate initialization code
        if self.config.enable_capability_validation || 
           self.config.enable_effect_tracking || 
           self.config.enable_security_enforcement {
            result.initialization.push(self.generate_runtime_initialization()?);
        }

        // Generate cleanup code
        result.cleanup.push(self.generate_runtime_cleanup()?);

        // Add required libraries
        result.libraries.push("prism_runtime".to_string());
        if let Some(ref lib_path) = self.config.runtime_library_path {
            result.libraries.push(lib_path.clone());
        }

        // Add allocator-specific libraries
        match self.config.allocator_type {
            AllocatorType::Custom(ref allocator) => {
                result.libraries.push(allocator.clone());
            }
            AllocatorType::GarbageCollected => {
                result.libraries.push("prism_gc".to_string());
            }
            _ => {}
        }

        Ok(result)
    }

    /// Generate function declaration
    fn generate_function_declaration(&self, decl: &RuntimeFunctionDecl) -> LLVMResult<String> {
        let attributes = if decl.attributes.is_empty() {
            String::new()
        } else {
            format!(" {}", decl.attributes.join(" "))
        };

        let params = decl.param_types.join(", ");
        
        Ok(format!(
            "declare{} {} @{}({})",
            attributes,
            decl.return_type,
            decl.llvm_name,
            params
        ))
    }

    /// Generate runtime initialization code
    fn generate_runtime_initialization(&self) -> LLVMResult<String> {
        let mut init_code = Vec::new();

        init_code.push("define void @prism_runtime_init() {".to_string());
        init_code.push("entry:".to_string());

        if self.config.enable_capability_validation {
            init_code.push("  call void @prism_init_capability_system()".to_string());
        }

        if self.config.enable_effect_tracking {
            init_code.push("  call void @prism_init_effect_tracking()".to_string());
        }

        if self.config.enable_security_enforcement {
            init_code.push("  call void @prism_init_security_system()".to_string());
        }

        if let Some(ref gc_config) = self.config.gc_config {
            init_code.push(format!(
                "  call void @prism_init_gc(i64 {}, i64 {})",
                gc_config.initial_heap_size,
                gc_config.max_heap_size
            ));
        }

        init_code.push("  ret void".to_string());
        init_code.push("}".to_string());

        Ok(init_code.join("\n"))
    }

    /// Generate runtime cleanup code
    fn generate_runtime_cleanup(&self) -> LLVMResult<String> {
        let mut cleanup_code = Vec::new();

        cleanup_code.push("define void @prism_runtime_cleanup() {".to_string());
        cleanup_code.push("entry:".to_string());

        if self.config.enable_effect_tracking {
            cleanup_code.push("  call void @prism_finalize_effect_tracking()".to_string());
        }

        if self.config.enable_capability_validation {
            cleanup_code.push("  call void @prism_finalize_capability_system()".to_string());
        }

        if self.config.gc_config.is_some() {
            cleanup_code.push("  call void @prism_finalize_gc()".to_string());
        }

        cleanup_code.push("  ret void".to_string());
        cleanup_code.push("}".to_string());

        Ok(cleanup_code.join("\n"))
    }

    /// Generate capability validation call
    pub fn generate_capability_validation(&self, capability_type: &str, args: &[String]) -> LLVMResult<String> {
        if !self.config.enable_capability_validation {
            return Ok(String::new());
        }

        let validator_name = match capability_type {
            "read" => "validate_read_capability",
            "write" => "validate_write_capability",
            "execute" => "validate_execute_capability",
            "network" => "validate_network_capability",
            "file" => "validate_file_capability",
            _ => return Err(LLVMError::InvalidCapabilityType(capability_type.to_string())),
        };

        if let Some(signature) = self.capability_validators.get(validator_name) {
            let args_str = args.join(", ");
            Ok(format!("call {}", signature.replace("@", &format!("@{}({})", 
                signature.split('@').nth(1).unwrap().split('(').next().unwrap(), args_str))))
        } else {
            Err(LLVMError::MissingRuntimeFunction(validator_name.to_string()))
        }
    }

    /// Generate effect tracking call
    pub fn generate_effect_tracking(&self, effect_type: &str, args: &[String]) -> LLVMResult<String> {
        if !self.config.enable_effect_tracking {
            return Ok(String::new());
        }

        let tracker_name = match effect_type {
            "begin" => "begin_effect_tracking",
            "end" => "end_effect_tracking",
            "io" => "track_io_effect",
            "network" => "track_network_effect",
            "computation" => "track_computation_effect",
            _ => return Err(LLVMError::InvalidEffectType(effect_type.to_string())),
        };

        if let Some(signature) = self.effect_trackers.get(tracker_name) {
            let args_str = args.join(", ");
            Ok(format!("call {}", signature.replace("@", &format!("@{}({})", 
                signature.split('@').nth(1).unwrap().split('(').next().unwrap(), args_str))))
        } else {
            Err(LLVMError::MissingRuntimeFunction(tracker_name.to_string()))
        }
    }

    /// Generate security enforcement call
    pub fn generate_security_enforcement(&self, check_type: &str, args: &[String]) -> LLVMResult<String> {
        if !self.config.enable_security_enforcement {
            return Ok(String::new());
        }

        let enforcer_name = match check_type {
            "bounds" => "check_bounds",
            "pointer" => "validate_pointer",
            "zero" => "secure_zero",
            "audit" => "audit_access",
            _ => return Err(LLVMError::InvalidSecurityCheck(check_type.to_string())),
        };

        if let Some(signature) = self.security_enforcers.get(enforcer_name) {
            let args_str = args.join(", ");
            Ok(format!("call {}", signature.replace("@", &format!("@{}({})", 
                signature.split('@').nth(1).unwrap().split('(').next().unwrap(), args_str))))
        } else {
            Err(LLVMError::MissingRuntimeFunction(enforcer_name.to_string()))
        }
    }

    /// Generate memory allocation call
    pub fn generate_memory_allocation(&self, size: &str, alignment: Option<&str>) -> LLVMResult<String> {
        match self.config.allocator_type {
            AllocatorType::System => {
                Ok(format!("call i8* @malloc(i64 {})", size))
            }
            AllocatorType::Custom(ref allocator) => {
                Ok(format!("call i8* @{}(i64 {})", allocator, size))
            }
            AllocatorType::GarbageCollected => {
                Ok(format!("call i8* @prism_gc_alloc(i64 {})", size))
            }
            AllocatorType::Pool => {
                Ok(format!("call i8* @prism_pool_alloc(i64 {})", size))
            }
            AllocatorType::Stack => {
                if let Some(align) = alignment {
                    Ok(format!("alloca i8, i64 {}, align {}", size, align))
                } else {
                    Ok(format!("alloca i8, i64 {}", size))
                }
            }
        }
    }

    /// Generate memory deallocation call
    pub fn generate_memory_deallocation(&self, ptr: &str) -> LLVMResult<String> {
        match self.config.allocator_type {
            AllocatorType::System => {
                Ok(format!("call void @free(i8* {})", ptr))
            }
            AllocatorType::Custom(ref allocator) => {
                Ok(format!("call void @{}_free(i8* {})", allocator, ptr))
            }
            AllocatorType::GarbageCollected => {
                // No explicit deallocation needed for GC
                Ok(String::new())
            }
            AllocatorType::Pool => {
                Ok(format!("call void @prism_pool_free(i8* {})", ptr))
            }
            AllocatorType::Stack => {
                // Stack allocation doesn't need explicit deallocation
                Ok(String::new())
            }
        }
    }

    /// Helper methods
    fn extract_return_type(&self, signature: &str) -> String {
        if let Some(space_pos) = signature.find(' ') {
            signature[..space_pos].to_string()
        } else {
            "void".to_string()
        }
    }

    fn extract_param_types(&self, signature: &str) -> Vec<String> {
        if let Some(paren_start) = signature.find('(') {
            if let Some(paren_end) = signature.find(')') {
                let params = &signature[paren_start + 1..paren_end];
                if params.is_empty() {
                    Vec::new()
                } else {
                    params.split(',').map(|p| p.trim().to_string()).collect()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Get runtime configuration
    pub fn get_config(&self) -> &LLVMRuntimeConfig {
        &self.config
    }

    /// Get runtime functions
    pub fn get_runtime_functions(&self) -> &HashMap<String, RuntimeFunctionDecl> {
        &self.runtime_functions
    }

    /// Check if function is available
    pub fn is_function_available(&self, name: &str) -> bool {
        self.runtime_functions.contains_key(name) ||
        self.capability_validators.contains_key(name) ||
        self.effect_trackers.contains_key(name) ||
        self.security_enforcers.contains_key(name)
    }

    /// Get function security level
    pub fn get_function_security_level(&self, name: &str) -> Option<SecurityLevel> {
        self.runtime_functions.get(name).map(|f| f.security_level.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        assert!(runtime.config.enable_capability_validation);
        assert!(runtime.config.enable_effect_tracking);
        assert!(runtime.config.enable_security_enforcement);
    }

    #[test]
    fn test_runtime_integration_generation() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        let result = runtime.generate_runtime_integration(LLVMOptimizationLevel::Aggressive).unwrap();
        
        assert!(!result.declarations.is_empty());
        assert!(!result.initialization.is_empty());
        assert!(!result.cleanup.is_empty());
        assert!(!result.libraries.is_empty());
    }

    #[test]
    fn test_capability_validation_generation() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        let args = vec!["i8* %ptr".to_string(), "i64 1024".to_string()];
        let call = runtime.generate_capability_validation("read", &args).unwrap();
        
        assert!(call.contains("prism_validate_read_capability"));
    }

    #[test]
    fn test_effect_tracking_generation() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        let args = vec!["i8* %effect_name".to_string()];
        let call = runtime.generate_effect_tracking("begin", &args).unwrap();
        
        assert!(call.contains("prism_begin_effect_tracking"));
    }

    #[test]
    fn test_memory_allocation_generation() {
        let config = LLVMRuntimeConfig {
            allocator_type: AllocatorType::System,
            ..Default::default()
        };
        let runtime = LLVMRuntime::new(config);
        
        let call = runtime.generate_memory_allocation("1024", None).unwrap();
        assert!(call.contains("malloc"));
        
        let dealloc = runtime.generate_memory_deallocation("i8* %ptr").unwrap();
        assert!(dealloc.contains("free"));
    }

    #[test]
    fn test_gc_allocation() {
        let config = LLVMRuntimeConfig {
            allocator_type: AllocatorType::GarbageCollected,
            gc_config: Some(GCConfig {
                algorithm: GCAlgorithm::MarkAndSweep,
                initial_heap_size: 1024 * 1024,
                max_heap_size: 64 * 1024 * 1024,
                gc_threshold: 0.8,
                incremental: true,
            }),
            ..Default::default()
        };
        let runtime = LLVMRuntime::new(config);
        
        let call = runtime.generate_memory_allocation("1024", None).unwrap();
        assert!(call.contains("prism_gc_alloc"));
        
        // GC doesn't need explicit deallocation
        let dealloc = runtime.generate_memory_deallocation("i8* %ptr").unwrap();
        assert!(dealloc.is_empty());
    }

    #[test]
    fn test_security_enforcement() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        let args = vec!["i8* %ptr".to_string(), "i64 0".to_string(), "i64 1024".to_string()];
        let call = runtime.generate_security_enforcement("bounds", &args).unwrap();
        
        assert!(call.contains("prism_check_bounds"));
    }

    #[test]
    fn test_function_availability() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        assert!(runtime.is_function_available("safe_malloc"));
        assert!(runtime.is_function_available("validate_read_capability"));
        assert!(!runtime.is_function_available("nonexistent_function"));
    }

    #[test]
    fn test_custom_runtime_functions() {
        let custom_func = RuntimeFunction {
            name: "custom_function".to_string(),
            signature: "i32 custom_function(i32, i32)".to_string(),
            attributes: vec!["nounwind".to_string()],
            always_available: true,
            security_level: SecurityLevel::Basic,
        };
        
        let config = LLVMRuntimeConfig {
            custom_runtime_functions: vec![custom_func],
            ..Default::default()
        };
        
        let runtime = LLVMRuntime::new(config);
        assert!(runtime.is_function_available("custom_function"));
        
        let security_level = runtime.get_function_security_level("custom_function");
        assert_eq!(security_level, Some(SecurityLevel::Basic));
    }
} 