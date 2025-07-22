//! WebAssembly Runtime Integration
//!
//! This module handles integration with the Prism runtime system for WebAssembly targets,
//! including capability management, effect tracking, and runtime support functions.

use super::{WasmResult, WasmError};
use super::types::{WasmRuntimeTarget, WasmFeatures};
use crate::backends::{Effect, Capability, PIRSemanticType};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// WebAssembly runtime integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRuntimeConfig {
    /// Target runtime environment
    pub target: WasmRuntimeTarget,
    /// Enabled WebAssembly features
    pub features: WasmFeatures,
    /// Enable capability validation
    pub enable_capability_validation: bool,
    /// Enable effect tracking
    pub enable_effect_tracking: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable debug support
    pub enable_debug_support: bool,
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        Self {
            target: WasmRuntimeTarget::default(),
            features: WasmFeatures::default(),
            enable_capability_validation: true,
            enable_effect_tracking: true,
            enable_performance_monitoring: true,
            enable_debug_support: true,
        }
    }
}

/// WebAssembly runtime integration system
pub struct WasmRuntimeIntegration {
    /// Runtime configuration
    config: WasmRuntimeConfig,
    /// Registered capabilities
    capabilities: HashMap<String, WasmCapability>,
    /// Registered effects
    effects: HashMap<String, WasmEffect>,
    /// Runtime function imports
    runtime_imports: Vec<WasmRuntimeImport>,
}

/// WebAssembly capability wrapper
#[derive(Debug, Clone)]
pub struct WasmCapability {
    /// Capability name
    pub name: String,
    /// Required WASM imports
    pub required_imports: Vec<String>,
    /// Validation function name
    pub validation_function: String,
    /// Description for debugging
    pub description: String,
}

/// WebAssembly effect wrapper
#[derive(Debug, Clone)]
pub struct WasmEffect {
    /// Effect name
    pub name: String,
    /// Effect type
    pub effect_type: String,
    /// Required WASM imports for tracking
    pub tracking_imports: Vec<String>,
    /// Begin tracking function
    pub begin_function: String,
    /// End tracking function
    pub end_function: String,
}

/// WebAssembly runtime import definition
#[derive(Debug, Clone)]
pub struct WasmRuntimeImport {
    /// Import module name
    pub module: String,
    /// Import function name
    pub name: String,
    /// Function signature
    pub signature: String,
    /// Description
    pub description: String,
    /// Required for which features
    pub required_for: Vec<String>,
}

impl WasmRuntimeIntegration {
    /// Create new runtime integration with configuration
    pub fn new(config: WasmRuntimeConfig) -> Self {
        let mut integration = Self {
            config,
            capabilities: HashMap::new(),
            effects: HashMap::new(),
            runtime_imports: Vec::new(),
        };

        // Initialize standard runtime imports
        integration.initialize_standard_imports();
        integration
    }

    /// Initialize standard Prism runtime imports
    fn initialize_standard_imports(&mut self) {
        // Core capability system imports
        self.add_runtime_import(WasmRuntimeImport {
            module: "prism_runtime".to_string(),
            name: "validate_capability".to_string(),
            signature: "(param i32 i32) (result i32)".to_string(),
            description: "Validate a capability by name".to_string(),
            required_for: vec!["capability_validation".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "prism_runtime".to_string(),
            name: "track_effect".to_string(),
            signature: "(param i32 i32 i32) (result i64)".to_string(),
            description: "Begin tracking an effect".to_string(),
            required_for: vec!["effect_tracking".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "prism_runtime".to_string(),
            name: "end_effect_tracking".to_string(),
            signature: "(param i64)".to_string(),
            description: "End effect tracking".to_string(),
            required_for: vec!["effect_tracking".to_string()],
        });

        // Business rule validation
        self.add_runtime_import(WasmRuntimeImport {
            module: "prism_runtime".to_string(),
            name: "validate_business_rule".to_string(),
            signature: "(param i32 i32 i32) (result i32)".to_string(),
            description: "Validate a business rule".to_string(),
            required_for: vec!["business_rule_validation".to_string()],
        });

        // Performance monitoring
        self.add_runtime_import(WasmRuntimeImport {
            module: "prism_runtime".to_string(),
            name: "performance_start".to_string(),
            signature: "(param i32 i32) (result i64)".to_string(),
            description: "Start performance monitoring".to_string(),
            required_for: vec!["performance_monitoring".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "prism_runtime".to_string(),
            name: "performance_end".to_string(),
            signature: "(param i64)".to_string(),
            description: "End performance monitoring".to_string(),
            required_for: vec!["performance_monitoring".to_string()],
        });

        // Target-specific imports
        match self.config.target {
            WasmRuntimeTarget::WASI => {
                self.add_wasi_imports();
            }
            WasmRuntimeTarget::Browser => {
                self.add_browser_imports();
            }
            WasmRuntimeTarget::NodeJS => {
                self.add_nodejs_imports();
            }
            _ => {
                // Generic imports for other targets
                self.add_generic_imports();
            }
        }
    }

    /// Add WASI-specific imports
    fn add_wasi_imports(&mut self) {
        self.add_runtime_import(WasmRuntimeImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "proc_exit".to_string(),
            signature: "(param i32)".to_string(),
            description: "Exit the process".to_string(),
            required_for: vec!["wasi".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "fd_write".to_string(),
            signature: "(param i32 i32 i32 i32) (result i32)".to_string(),
            description: "Write to file descriptor".to_string(),
            required_for: vec!["wasi".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "wasi_snapshot_preview1".to_string(),
            name: "clock_time_get".to_string(),
            signature: "(param i32 i64 i32) (result i32)".to_string(),
            description: "Get current time".to_string(),
            required_for: vec!["wasi".to_string()],
        });
    }

    /// Add browser-specific imports
    fn add_browser_imports(&mut self) {
        self.add_runtime_import(WasmRuntimeImport {
            module: "env".to_string(),
            name: "console_log".to_string(),
            signature: "(param i32 i32)".to_string(),
            description: "Log to browser console".to_string(),
            required_for: vec!["browser".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "env".to_string(),
            name: "performance_now".to_string(),
            signature: "(result f64)".to_string(),
            description: "Get high-resolution timestamp".to_string(),
            required_for: vec!["browser".to_string()],
        });
    }

    /// Add Node.js-specific imports
    fn add_nodejs_imports(&mut self) {
        self.add_runtime_import(WasmRuntimeImport {
            module: "env".to_string(),
            name: "node_log".to_string(),
            signature: "(param i32 i32)".to_string(),
            description: "Log via Node.js console".to_string(),
            required_for: vec!["nodejs".to_string()],
        });

        self.add_runtime_import(WasmRuntimeImport {
            module: "env".to_string(),
            name: "node_exit".to_string(),
            signature: "(param i32)".to_string(),
            description: "Exit Node.js process".to_string(),
            required_for: vec!["nodejs".to_string()],
        });
    }

    /// Add generic imports for other targets
    fn add_generic_imports(&mut self) {
        self.add_runtime_import(WasmRuntimeImport {
            module: "env".to_string(),
            name: "debug_log".to_string(),
            signature: "(param i32 i32)".to_string(),
            description: "Generic debug logging".to_string(),
            required_for: vec!["debug".to_string()],
        });
    }

    /// Add a runtime import
    pub fn add_runtime_import(&mut self, import: WasmRuntimeImport) {
        self.runtime_imports.push(import);
    }

    /// Register a capability for WASM integration
    pub fn register_capability(&mut self, capability: &Capability) -> WasmResult<()> {
        let wasm_capability = WasmCapability {
            name: capability.name.clone(),
            required_imports: vec!["validate_capability".to_string()],
            validation_function: format!("validate_cap_{}", capability.name.to_lowercase()),
            description: format!("Capability: {}", capability.name),
        };

        self.capabilities.insert(capability.name.clone(), wasm_capability);
        Ok(())
    }

    /// Register an effect for WASM integration
    pub fn register_effect(&mut self, effect: &Effect) -> WasmResult<()> {
        let wasm_effect = WasmEffect {
            name: effect.name.clone(),
            effect_type: effect.effect_type.clone(),
            tracking_imports: vec!["track_effect".to_string(), "end_effect_tracking".to_string()],
            begin_function: format!("begin_effect_{}", effect.name.to_lowercase()),
            end_function: format!("end_effect_{}", effect.name.to_lowercase()),
        };

        self.effects.insert(effect.name.clone(), wasm_effect);
        Ok(())
    }

    /// Generate WASM import declarations
    pub fn generate_import_declarations(&self) -> String {
        let mut output = String::new();
        
        output.push_str("  ;; === PRISM RUNTIME IMPORTS ===\n");
        output.push_str("  ;; Runtime integration for capability, effect, and performance management\n\n");

        // Group imports by module
        let mut imports_by_module: HashMap<String, Vec<&WasmRuntimeImport>> = HashMap::new();
        for import in &self.runtime_imports {
            imports_by_module.entry(import.module.clone()).or_default().push(import);
        }

        // Generate imports grouped by module
        for (module, imports) in imports_by_module {
            output.push_str(&format!("  ;; {} imports\n", module));
            for import in imports {
                output.push_str(&format!(
                    "  (import \"{}\" \"{}\" (func ${} {}))\n",
                    import.module, import.name, import.name, import.signature
                ));
            }
            output.push('\n');
        }

        output
    }

    /// Generate capability validation functions
    pub fn generate_capability_functions(&self) -> String {
        let mut output = String::new();
        
        if self.config.enable_capability_validation {
            output.push_str("  ;; === CAPABILITY VALIDATION FUNCTIONS ===\n");
            
            for (_, capability) in &self.capabilities {
                output.push_str(&format!(
                    "  ;; Validation function for capability: {}\n",
                    capability.name
                ));
                output.push_str(&format!(
                    "  (func ${} (param $context i32) (result i32)\n",
                    capability.validation_function
                ));
                output.push_str("    ;; Load capability name\n");
                let capability_offset = self.get_capability_name_offset(&capability.name);
                output.push_str(&format!(
                    "    i32.const {} ;; capability name offset\n",
                    capability_offset
                ));
                output.push_str(&format!(
                    "    i32.const {} ;; capability name length\n",
                    capability.name.len()
                ));
                output.push_str("    call $validate_capability\n");
                output.push_str("  )\n\n");
            }
        }

        output
    }

    /// Generate effect tracking functions
    pub fn generate_effect_functions(&self) -> String {
        let mut output = String::new();
        
        if self.config.enable_effect_tracking {
            output.push_str("  ;; === EFFECT TRACKING FUNCTIONS ===\n");
            
            for (_, effect) in &self.effects {
                // Begin effect tracking function
                output.push_str(&format!(
                    "  ;; Begin tracking for effect: {}\n",
                    effect.name
                ));
                output.push_str(&format!(
                    "  (func ${} (result i64)\n",
                    effect.begin_function
                ));
                output.push_str("    ;; Load effect name and type\n");
                output.push_str("    i32.const 0 ;; TODO: effect name offset\n");
                output.push_str(&format!("    i32.const {} ;; effect name length\n", effect.name.len()));
                output.push_str("    i32.const 0 ;; TODO: effect type\n");
                output.push_str("    call $track_effect\n");
                output.push_str("  )\n\n");

                // End effect tracking function
                output.push_str(&format!(
                    "  ;; End tracking for effect: {}\n",
                    effect.name
                ));
                output.push_str(&format!(
                    "  (func ${} (param $handle i64)\n",
                    effect.end_function
                ));
                output.push_str("    local.get $handle\n");
                output.push_str("    call $end_effect_tracking\n");
                output.push_str("  )\n\n");
            }
        }

        output
    }

    /// Generate runtime initialization function
    pub fn generate_runtime_initialization(&self) -> String {
        let mut output = String::new();
        
        output.push_str("  ;; === RUNTIME INITIALIZATION ===\n");
        output.push_str("  (func $prism_runtime_init\n");
        output.push_str("    (local $init_result i32)\n");
        output.push_str("    \n");
        
        if self.config.enable_capability_validation {
            output.push_str("    ;; Initialize capability system\n");
            output.push_str(&format!(
                "    i32.const {} ;; number of capabilities\n",
                self.capabilities.len()
            ));
            output.push_str("    ;; TODO: call capability system init\n");
        }
        
        if self.config.enable_effect_tracking {
            output.push_str("    ;; Initialize effect tracking\n");
            output.push_str(&format!(
                "    i32.const {} ;; number of effects\n",
                self.effects.len()
            ));
            output.push_str("    ;; TODO: call effect system init\n");
        }
        
        if self.config.enable_performance_monitoring {
            output.push_str("    ;; Initialize performance monitoring\n");
            output.push_str("    ;; TODO: call performance system init\n");
        }
        
        output.push_str("    ;; Runtime initialization complete\n");
        output.push_str("  )\n\n");
        
        output
    }

    /// Generate semantic type validation integration
    pub fn generate_semantic_type_integration(&self, semantic_types: &[PIRSemanticType]) -> String {
        let mut output = String::new();
        
        output.push_str("  ;; === SEMANTIC TYPE INTEGRATION ===\n");
        output.push_str("  ;; Runtime validation for semantic types with business rules\n\n");
        
        for semantic_type in semantic_types {
            output.push_str(&format!(
                "  ;; Semantic type validation: {} (Domain: {})\n",
                semantic_type.name, semantic_type.domain
            ));
            
            // Generate validation function that integrates with runtime
            output.push_str(&format!(
                "  (func $validate_semantic_{} (param $value i32) (result i32)\n",
                semantic_type.name.to_lowercase()
            ));
            output.push_str("    (local $validation_result i32)\n");
            output.push_str("    \n");
            
            // Validate business rules
            for (i, rule) in semantic_type.business_rules.iter().enumerate() {
                output.push_str(&format!(
                    "    ;; Business rule {}: {}\n",
                    i, rule.name
                ));
                output.push_str("    local.get $value\n");
                output.push_str("    i32.const 0 ;; TODO: rule name offset\n");
                output.push_str(&format!("    i32.const {} ;; rule name length\n", rule.name.len()));
                output.push_str("    call $validate_business_rule\n");
                output.push_str("    local.set $validation_result\n");
                output.push_str("    \n");
                output.push_str("    ;; Check validation result\n");
                output.push_str("    local.get $validation_result\n");
                output.push_str("    i32.const 0\n");
                output.push_str("    i32.eq\n");
                output.push_str("    if\n");
                output.push_str("      ;; Business rule validation failed\n");
                output.push_str("      i32.const 0\n");
                output.push_str("      return\n");
                output.push_str("    end\n");
                output.push_str("    \n");
            }
            
            output.push_str("    ;; All validations passed\n");
            output.push_str("    i32.const 1\n");
            output.push_str("  )\n\n");
        }
        
        output
    }

    /// Get runtime configuration
    pub fn get_config(&self) -> &WasmRuntimeConfig {
        &self.config
    }

    /// Get registered capabilities
    pub fn get_capabilities(&self) -> &HashMap<String, WasmCapability> {
        &self.capabilities
    }

    /// Get registered effects
    pub fn get_effects(&self) -> &HashMap<String, WasmEffect> {
        &self.effects
    }
}

/// WebAssembly capability system integration
pub struct WasmCapabilitySystem {
    /// Runtime integration
    runtime: WasmRuntimeIntegration,
}

impl WasmCapabilitySystem {
    /// Create new capability system
    pub fn new(config: WasmRuntimeConfig) -> Self {
        Self {
            runtime: WasmRuntimeIntegration::new(config),
        }
    }

    /// Generate capability checking code for a function
    pub fn generate_capability_check(&self, required_capabilities: &[Capability]) -> String {
        let mut output = String::new();
        
        if !required_capabilities.is_empty() {
            output.push_str("    ;; === CAPABILITY VALIDATION ===\n");
            
            for capability in required_capabilities {
                if let Some(wasm_cap) = self.runtime.capabilities.get(&capability.name) {
                    output.push_str(&format!(
                        "    ;; Check capability: {}\n",
                        capability.name
                    ));
                    output.push_str("    i32.const 0 ;; execution context\n");
                    output.push_str(&format!("    call ${}\n", wasm_cap.validation_function));
                    output.push_str("    i32.const 0\n");
                    output.push_str("    i32.eq\n");
                    output.push_str("    if\n");
                    output.push_str("      ;; Capability check failed\n");
                    output.push_str("      unreachable\n");
                    output.push_str("    end\n");
                    output.push_str("    \n");
                }
            }
        }
        
        output
    }

    /// Generate effect tracking code
    pub fn generate_effect_tracking(&self, effects: &[Effect]) -> (String, String) {
        let mut begin_code = String::new();
        let mut end_code = String::new();
        
        if !effects.is_empty() {
            begin_code.push_str("    ;; === BEGIN EFFECT TRACKING ===\n");
            end_code.push_str("    ;; === END EFFECT TRACKING ===\n");
            
            for effect in effects {
                if let Some(wasm_effect) = self.runtime.effects.get(&effect.name) {
                    begin_code.push_str(&format!(
                        "    call ${} ;; Begin tracking {}\n",
                        wasm_effect.begin_function, effect.name
                    ));
                    begin_code.push_str("    local.set $effect_handle\n");
                    
                    end_code.push_str("    local.get $effect_handle\n");
                    end_code.push_str(&format!(
                        "    call ${} ;; End tracking {}\n",
                        wasm_effect.end_function, effect.name
                    ));
                }
            }
        }
        
        (begin_code, end_code)
    }
}

impl Default for WasmRuntimeIntegration {
    fn default() -> Self {
        Self::new(WasmRuntimeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_integration_creation() {
        let integration = WasmRuntimeIntegration::default();
        
        // Should have standard imports
        assert!(!integration.runtime_imports.is_empty());
        
        // Should contain core runtime imports
        let import_names: Vec<_> = integration.runtime_imports.iter()
            .map(|i| i.name.as_str())
            .collect();
        assert!(import_names.contains(&"validate_capability"));
        assert!(import_names.contains(&"track_effect"));
    }

    #[test]
    fn test_capability_registration() {
        let mut integration = WasmRuntimeIntegration::default();
        
        let capability = Capability {
            name: "test_capability".to_string(),
            // Add other required fields based on actual Capability struct
        };
        
        let result = integration.register_capability(&capability);
        assert!(result.is_ok());
        assert!(integration.capabilities.contains_key("test_capability"));
    }

    #[test]
    fn test_import_generation() {
        let integration = WasmRuntimeIntegration::default();
        let imports = integration.generate_import_declarations();
        
        assert!(imports.contains("prism_runtime"));
        assert!(imports.contains("validate_capability"));
        assert!(imports.contains("track_effect"));
    }

    #[test]
    fn test_wasi_target_imports() {
        let config = WasmRuntimeConfig {
            target: WasmRuntimeTarget::WASI,
            ..WasmRuntimeConfig::default()
        };
        let integration = WasmRuntimeIntegration::new(config);
        
        let imports = integration.generate_import_declarations();
        assert!(imports.contains("wasi_snapshot_preview1"));
        assert!(imports.contains("proc_exit"));
    }
} 