//! WebAssembly Code Validation
//!
//! This module provides comprehensive validation for generated WebAssembly code,
//! ensuring correctness, security, and compliance with WebAssembly standards.

use super::{WasmResult, WasmError};
use super::types::{WasmRuntimeTarget, WasmFeatures};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// WebAssembly validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmValidatorConfig {
    /// Target runtime for validation
    pub target: WasmRuntimeTarget,
    /// Enabled WebAssembly features
    pub features: WasmFeatures,
    /// Validate module structure
    pub validate_structure: bool,
    /// Validate type usage
    pub validate_types: bool,
    /// Validate imports and exports
    pub validate_imports_exports: bool,
    /// Validate memory usage
    pub validate_memory: bool,
    /// Validate function signatures
    pub validate_function_signatures: bool,
    /// Validate control flow
    pub validate_control_flow: bool,
    /// Validate Prism-specific semantics
    pub validate_prism_semantics: bool,
    /// Maximum allowed stack depth
    pub max_stack_depth: u32,
    /// Maximum allowed locals per function
    pub max_locals_per_function: u32,
}

impl Default for WasmValidatorConfig {
    fn default() -> Self {
        Self {
            target: WasmRuntimeTarget::default(),
            features: WasmFeatures::default(),
            validate_structure: true,
            validate_types: true,
            validate_imports_exports: true,
            validate_memory: true,
            validate_function_signatures: true,
            validate_control_flow: true,
            validate_prism_semantics: true,
            max_stack_depth: 1024,
            max_locals_per_function: 256,
        }
    }
}

/// WebAssembly code validator
pub struct WasmValidator {
    /// Validation configuration
    config: WasmValidatorConfig,
    /// Validation results
    results: ValidationResults,
}

/// Validation results and diagnostics
#[derive(Debug, Clone, Default)]
pub struct ValidationResults {
    /// Validation errors (must be fixed)
    pub errors: Vec<ValidationError>,
    /// Validation warnings (should be addressed)
    pub warnings: Vec<ValidationWarning>,
    /// Validation information (for optimization hints)
    pub info: Vec<ValidationInfo>,
    /// Overall validation success
    pub is_valid: bool,
    /// Validation time in milliseconds
    pub validation_time_ms: u64,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Line number where error occurred
    pub line: Option<usize>,
    /// Error category
    pub category: ValidationCategory,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Line number where warning occurred
    pub line: Option<usize>,
    /// Warning category
    pub category: ValidationCategory,
}

/// Validation information
#[derive(Debug, Clone)]
pub struct ValidationInfo {
    /// Info message
    pub message: String,
    /// Line number
    pub line: Option<usize>,
    /// Info category
    pub category: ValidationCategory,
}

/// Validation categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationCategory {
    /// Module structure issues
    Structure,
    /// Type system issues
    Types,
    /// Import/export issues
    ImportsExports,
    /// Memory management issues
    Memory,
    /// Function signature issues
    Functions,
    /// Control flow issues
    ControlFlow,
    /// Prism semantic issues
    PrismSemantics,
    /// Performance issues
    Performance,
    /// Security issues
    Security,
}

impl WasmValidator {
    /// Create new validator with configuration
    pub fn new(config: WasmValidatorConfig) -> Self {
        Self {
            config,
            results: ValidationResults::default(),
        }
    }

    /// Validate WebAssembly code
    pub fn validate(&mut self, wasm_code: &str) -> WasmResult<&ValidationResults> {
        let start_time = std::time::Instant::now();
        
        // Reset results
        self.results = ValidationResults::default();

        // Parse code into lines for analysis
        let lines: Vec<&str> = wasm_code.lines().collect();

        // Run validation passes
        if self.config.validate_structure {
            self.validate_module_structure(&lines)?;
        }

        if self.config.validate_types {
            self.validate_type_usage(&lines)?;
        }

        if self.config.validate_imports_exports {
            self.validate_imports_and_exports(&lines)?;
        }

        if self.config.validate_memory {
            self.validate_memory_usage(&lines)?;
        }

        if self.config.validate_function_signatures {
            self.validate_function_signatures(&lines)?;
        }

        if self.config.validate_control_flow {
            self.validate_control_flow(&lines)?;
        }

        if self.config.validate_prism_semantics {
            self.validate_prism_semantics(&lines)?;
        }

        // Additional target-specific validation
        self.validate_target_specific(&lines)?;

        // Determine overall validation result
        self.results.is_valid = self.results.errors.is_empty();
        self.results.validation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(&self.results)
    }

    /// Validate module structure
    fn validate_module_structure(&mut self, lines: &[&str]) -> WasmResult<()> {
        let mut module_found = false;
        let mut module_closed = false;
        let mut paren_depth = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check for module declaration
            if trimmed.starts_with("(module") {
                if module_found {
                    self.add_error("MULTIPLE_MODULES", 
                        "Multiple module declarations found", 
                        Some(line_num + 1), 
                        ValidationCategory::Structure);
                }
                module_found = true;
            }

            // Track parentheses balance
            paren_depth += trimmed.chars().filter(|&c| c == '(').count() as i32;
            paren_depth -= trimmed.chars().filter(|&c| c == ')').count() as i32;

            // Check for module closing
            if trimmed == ")" && paren_depth == 0 && module_found {
                module_closed = true;
            }
        }

        // Validate overall structure
        if !module_found {
            self.add_error("NO_MODULE", 
                "No module declaration found", 
                None, 
                ValidationCategory::Structure);
        }

        if module_found && !module_closed {
            self.add_error("UNCLOSED_MODULE", 
                "Module declaration not properly closed", 
                None, 
                ValidationCategory::Structure);
        }

        if paren_depth != 0 {
            self.add_error("UNBALANCED_PARENS", 
                &format!("Unbalanced parentheses (depth: {})", paren_depth), 
                None, 
                ValidationCategory::Structure);
        }

        Ok(())
    }

    /// Validate type usage
    fn validate_type_usage(&mut self, lines: &[&str]) -> WasmResult<()> {
        let valid_types = ["i32", "i64", "f32", "f64", "externref", "funcref", "v128"];
        let mut declared_functions = HashSet::new();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check function declarations
            if trimmed.starts_with("(func ") {
                if let Some(func_name) = self.extract_function_name(trimmed) {
                    if declared_functions.contains(&func_name) {
                        self.add_error("DUPLICATE_FUNCTION", 
                            &format!("Duplicate function declaration: {}", func_name), 
                            Some(line_num + 1), 
                            ValidationCategory::Functions);
                    }
                    declared_functions.insert(func_name);
                }
            }

            // Validate type references
            for word in trimmed.split_whitespace() {
                if word.starts_with("(param ") || word.starts_with("(result ") {
                    continue; // These are handled separately
                }
                
                // Check for invalid type usage
                if word.ends_with(".const") || word.ends_with(".add") || word.ends_with(".sub") {
                    let type_part = word.split('.').next().unwrap_or("");
                    if !type_part.is_empty() && !valid_types.contains(&type_part) {
                        self.add_error("INVALID_TYPE", 
                            &format!("Invalid type used: {}", type_part), 
                            Some(line_num + 1), 
                            ValidationCategory::Types);
                    }
                }
            }

            // Check for SIMD usage without feature enabled
            if !self.config.features.simd && trimmed.contains("v128") {
                self.add_error("SIMD_NOT_ENABLED", 
                    "SIMD instructions used but SIMD feature not enabled", 
                    Some(line_num + 1), 
                    ValidationCategory::Types);
            }

            // Check for reference types without feature enabled
            if !self.config.features.reference_types && 
               (trimmed.contains("externref") || trimmed.contains("funcref")) {
                self.add_error("REFERENCE_TYPES_NOT_ENABLED", 
                    "Reference types used but reference types feature not enabled", 
                    Some(line_num + 1), 
                    ValidationCategory::Types);
            }
        }

        Ok(())
    }

    /// Validate imports and exports
    fn validate_imports_and_exports(&mut self, lines: &[&str]) -> WasmResult<()> {
        let mut imports = HashMap::new();
        let mut exports = HashSet::new();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check import declarations
            if trimmed.starts_with("(import ") {
                if let Some((module, name)) = self.extract_import_info(trimmed) {
                    imports.insert(format!("{}::{}", module, name), line_num + 1);
                    
                    // Validate target-specific imports
                    self.validate_target_import(&module, &name, line_num + 1)?;
                }
            }

            // Check export declarations
            if trimmed.contains("(export ") {
                if let Some(export_name) = self.extract_export_name(trimmed) {
                    if exports.contains(&export_name) {
                        self.add_error("DUPLICATE_EXPORT", 
                            &format!("Duplicate export: {}", export_name), 
                            Some(line_num + 1), 
                            ValidationCategory::ImportsExports);
                    }
                    exports.insert(export_name);
                }
            }
        }

        // Check for required exports based on target
        match self.config.target {
            WasmRuntimeTarget::WASI => {
                if !exports.contains("_start") {
                    self.add_warning("MISSING_WASI_START", 
                        "WASI target should export _start function", 
                        None, 
                        ValidationCategory::ImportsExports);
                }
            }
            WasmRuntimeTarget::Browser => {
                if !exports.contains("memory") {
                    self.add_warning("MISSING_MEMORY_EXPORT", 
                        "Browser target should typically export memory", 
                        None, 
                        ValidationCategory::ImportsExports);
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Validate memory usage
    fn validate_memory_usage(&mut self, lines: &[&str]) -> WasmResult<()> {
        let mut memory_declared = false;
        let mut memory_exports = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check memory declaration
            if trimmed.starts_with("(memory ") {
                if memory_declared {
                    self.add_error("MULTIPLE_MEMORY", 
                        "Multiple memory declarations found", 
                        Some(line_num + 1), 
                        ValidationCategory::Memory);
                }
                memory_declared = true;
                
                // Validate memory limits
                if let Some((initial, max)) = self.extract_memory_limits(trimmed) {
                    if initial > 65536 {
                        self.add_error("MEMORY_INITIAL_TOO_LARGE", 
                            &format!("Initial memory too large: {} pages", initial), 
                            Some(line_num + 1), 
                            ValidationCategory::Memory);
                    }
                    
                    if let Some(max_pages) = max {
                        if max_pages > 65536 {
                            self.add_error("MEMORY_MAX_TOO_LARGE", 
                                &format!("Maximum memory too large: {} pages", max_pages), 
                                Some(line_num + 1), 
                                ValidationCategory::Memory);
                        }
                        
                        if max_pages < initial {
                            self.add_error("MEMORY_MAX_LESS_THAN_INITIAL", 
                                "Maximum memory less than initial memory", 
                                Some(line_num + 1), 
                                ValidationCategory::Memory);
                        }
                    }
                }
            }

            // Count memory exports
            if trimmed.contains("(export \"memory\"") {
                memory_exports += 1;
            }

            // Check for bulk memory operations without feature enabled
            if !self.config.features.bulk_memory && 
               (trimmed.contains("memory.copy") || trimmed.contains("memory.fill")) {
                self.add_error("BULK_MEMORY_NOT_ENABLED", 
                    "Bulk memory operations used but bulk memory feature not enabled", 
                    Some(line_num + 1), 
                    ValidationCategory::Memory);
            }
        }

        if memory_exports > 1 {
            self.add_error("MULTIPLE_MEMORY_EXPORTS", 
                "Multiple memory exports found", 
                None, 
                ValidationCategory::Memory);
        }

        Ok(())
    }

    /// Validate function signatures
    fn validate_function_signatures(&mut self, lines: &[&str]) -> WasmResult<()> {
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("(func ") {
                let mut local_count = 0;
                
                // Count locals in this function
                for (inner_line_num, inner_line) in lines[line_num..].iter().enumerate() {
                    let inner_trimmed = inner_line.trim();
                    
                    if inner_trimmed.starts_with("(local ") {
                        local_count += 1;
                    }
                    
                    // Stop at end of function
                    if inner_trimmed == ")" && inner_line_num > 0 {
                        break;
                    }
                }
                
                if local_count > self.config.max_locals_per_function {
                    self.add_warning("TOO_MANY_LOCALS", 
                        &format!("Function has {} locals, consider reducing for better performance", local_count), 
                        Some(line_num + 1), 
                        ValidationCategory::Performance);
                }

                // Check for multi-value returns without feature enabled
                if !self.config.features.multi_value && self.has_multi_value_return(trimmed) {
                    self.add_error("MULTI_VALUE_NOT_ENABLED", 
                        "Multi-value returns used but multi-value feature not enabled", 
                        Some(line_num + 1), 
                        ValidationCategory::Functions);
                }
            }
        }

        Ok(())
    }

    /// Validate control flow
    fn validate_control_flow(&mut self, lines: &[&str]) -> WasmResult<()> {
        let mut block_stack = Vec::new();
        let mut max_depth = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Track block structure
            if trimmed.starts_with("block") || trimmed.starts_with("loop") || trimmed.starts_with("if") {
                block_stack.push((trimmed.to_string(), line_num + 1));
                max_depth = max_depth.max(block_stack.len());
            }

            if trimmed == "end" {
                if block_stack.is_empty() {
                    self.add_error("UNMATCHED_END", 
                        "Unmatched 'end' instruction", 
                        Some(line_num + 1), 
                        ValidationCategory::ControlFlow);
                } else {
                    block_stack.pop();
                }
            }

            // Check for tail calls without feature enabled
            if !self.config.features.tail_calls && trimmed.contains("return_call") {
                self.add_error("TAIL_CALLS_NOT_ENABLED", 
                    "Tail calls used but tail calls feature not enabled", 
                    Some(line_num + 1), 
                    ValidationCategory::ControlFlow);
            }
        }

        if !block_stack.is_empty() {
            self.add_error("UNCLOSED_BLOCKS", 
                &format!("Unclosed control blocks: {}", block_stack.len()), 
                None, 
                ValidationCategory::ControlFlow);
        }

        if max_depth as u32 > self.config.max_stack_depth {
            self.add_warning("DEEP_NESTING", 
                &format!("Deep control flow nesting: {} levels", max_depth), 
                None, 
                ValidationCategory::Performance);
        }

        Ok(())
    }

    /// Validate Prism-specific semantics
    fn validate_prism_semantics(&mut self, lines: &[&str]) -> WasmResult<()> {
        let mut has_prism_runtime_init = false;
        let mut has_capability_validation = false;
        let mut has_effect_tracking = false;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check for Prism runtime initialization
            if trimmed.contains("prism_runtime_init") {
                has_prism_runtime_init = true;
            }

            // Check for capability validation
            if trimmed.contains("validate_capability") || trimmed.contains("prism_validate_capability") {
                has_capability_validation = true;
            }

            // Check for effect tracking
            if trimmed.contains("track_effect") || trimmed.contains("prism_track_effect") {
                has_effect_tracking = true;
            }

            // Check for semantic metadata preservation
            if trimmed.starts_with(";;") && (
                trimmed.contains("Business") ||
                trimmed.contains("Semantic") ||
                trimmed.contains("Domain:")
            ) {
                self.add_info(&format!("Semantic metadata preserved: {}", 
                    trimmed.trim_start_matches(";;").trim()), 
                    Some(line_num + 1), 
                    ValidationCategory::PrismSemantics);
            }

            // Validate business rule integration
            if trimmed.contains("validate_business_rule") {
                self.add_info("Business rule validation found", 
                    Some(line_num + 1), 
                    ValidationCategory::PrismSemantics);
            }
        }

        // Check for required Prism integrations
        if !has_prism_runtime_init {
            self.add_warning("MISSING_RUNTIME_INIT", 
                "Prism runtime initialization not found", 
                None, 
                ValidationCategory::PrismSemantics);
        }

        if !has_capability_validation {
            self.add_warning("MISSING_CAPABILITY_VALIDATION", 
                "No capability validation found", 
                None, 
                ValidationCategory::Security);
        }

        if !has_effect_tracking {
            self.add_warning("MISSING_EFFECT_TRACKING", 
                "No effect tracking found", 
                None, 
                ValidationCategory::PrismSemantics);
        }

        Ok(())
    }

    /// Validate target-specific requirements
    fn validate_target_specific(&mut self, lines: &[&str]) -> WasmResult<()> {
        match self.config.target {
            WasmRuntimeTarget::WASI => self.validate_wasi_compliance(lines),
            WasmRuntimeTarget::Browser => self.validate_browser_compliance(lines),
            WasmRuntimeTarget::NodeJS => self.validate_nodejs_compliance(lines),
            _ => Ok(()),
        }
    }

    /// Validate WASI compliance
    fn validate_wasi_compliance(&mut self, lines: &[&str]) -> WasmResult<()> {
        let mut has_wasi_imports = false;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.contains("wasi_snapshot_preview1") {
                has_wasi_imports = true;
            }

            // Check for non-WASI imports that might not work
            if trimmed.contains("(import \"env\"") {
                self.add_warning("ENV_IMPORT_IN_WASI", 
                    "Environment imports may not be available in WASI", 
                    Some(line_num + 1), 
                    ValidationCategory::ImportsExports);
            }
        }

        if !has_wasi_imports {
            self.add_warning("NO_WASI_IMPORTS", 
                "WASI target but no WASI imports found", 
                None, 
                ValidationCategory::ImportsExports);
        }

        Ok(())
    }

    /// Validate browser compliance
    fn validate_browser_compliance(&mut self, _lines: &[&str]) -> WasmResult<()> {
        // Browser-specific validations would go here
        Ok(())
    }

    /// Validate Node.js compliance
    fn validate_nodejs_compliance(&mut self, _lines: &[&str]) -> WasmResult<()> {
        // Node.js-specific validations would go here
        Ok(())
    }

    /// Validate target-specific import
    fn validate_target_import(&mut self, module: &str, name: &str, line_num: usize) -> WasmResult<()> {
        match self.config.target {
            WasmRuntimeTarget::WASI => {
                if module != "wasi_snapshot_preview1" && module != "prism_runtime" {
                    self.add_warning("NON_WASI_IMPORT", 
                        &format!("Non-WASI import may not be available: {}::{}", module, name), 
                        Some(line_num), 
                        ValidationCategory::ImportsExports);
                }
            }
            _ => {}
        }

        Ok(())
    }

    // Helper methods for parsing

    fn extract_function_name(&self, line: &str) -> Option<String> {
        // Extract function name from "(func $name ..." or "(func (export "name") ..."
        if let Some(start) = line.find("$") {
            if let Some(end) = line[start..].find(" ") {
                return Some(line[start..start + end].to_string());
            }
        }
        None
    }

    fn extract_import_info(&self, line: &str) -> Option<(String, String)> {
        // Parse "(import "module" "name" ...)"
        let parts: Vec<&str> = line.split('"').collect();
        if parts.len() >= 4 {
            Some((parts[1].to_string(), parts[3].to_string()))
        } else {
            None
        }
    }

    fn extract_export_name(&self, line: &str) -> Option<String> {
        // Parse "(export "name" ...)"
        if let Some(start) = line.find("\"") {
            if let Some(end) = line[start + 1..].find("\"") {
                return Some(line[start + 1..start + 1 + end].to_string());
            }
        }
        None
    }

    fn extract_memory_limits(&self, line: &str) -> Option<(u32, Option<u32>)> {
        // Parse "(memory initial)" or "(memory initial max)"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(initial) = parts[1].parse::<u32>() {
                let max = if parts.len() >= 3 {
                    parts[2].strip_suffix(")").unwrap_or(parts[2]).parse().ok()
                } else {
                    None
                };
                return Some((initial, max));
            }
        }
        None
    }

    fn has_multi_value_return(&self, line: &str) -> bool {
        // Check if function signature has multiple return values
        line.matches("(result").count() > 1
    }

    // Helper methods for adding validation results

    fn add_error(&mut self, code: &str, message: &str, line: Option<usize>, category: ValidationCategory) {
        self.results.errors.push(ValidationError {
            code: code.to_string(),
            message: message.to_string(),
            line,
            category,
        });
    }

    fn add_warning(&mut self, code: &str, message: &str, line: Option<usize>, category: ValidationCategory) {
        self.results.warnings.push(ValidationWarning {
            code: code.to_string(),
            message: message.to_string(),
            line,
            category,
        });
    }

    fn add_info(&mut self, message: &str, line: Option<usize>, category: ValidationCategory) {
        self.results.info.push(ValidationInfo {
            message: message.to_string(),
            line,
            category,
        });
    }

    /// Get validation results
    pub fn get_results(&self) -> &ValidationResults {
        &self.results
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("WebAssembly Validation Report\n");
        report.push_str("=============================\n\n");
        
        report.push_str(&format!("Overall Result: {}\n", 
            if self.results.is_valid { "VALID" } else { "INVALID" }));
        report.push_str(&format!("Validation Time: {}ms\n\n", self.results.validation_time_ms));
        
        if !self.results.errors.is_empty() {
            report.push_str(&format!("ERRORS ({})\n", self.results.errors.len()));
            report.push_str("--------\n");
            for error in &self.results.errors {
                report.push_str(&format!("[{}] {}: {}\n", 
                    error.code, 
                    error.line.map_or("".to_string(), |l| format!("Line {}", l)),
                    error.message));
            }
            report.push('\n');
        }
        
        if !self.results.warnings.is_empty() {
            report.push_str(&format!("WARNINGS ({})\n", self.results.warnings.len()));
            report.push_str("----------\n");
            for warning in &self.results.warnings {
                report.push_str(&format!("[{}] {}: {}\n", 
                    warning.code,
                    warning.line.map_or("".to_string(), |l| format!("Line {}", l)),
                    warning.message));
            }
            report.push('\n');
        }
        
        if !self.results.info.is_empty() {
            report.push_str(&format!("INFORMATION ({})\n", self.results.info.len()));
            report.push_str("-------------\n");
            for info in &self.results.info {
                report.push_str(&format!("{}: {}\n", 
                    info.line.map_or("".to_string(), |l| format!("Line {}", l)),
                    info.message));
            }
        }
        
        report
    }
}

impl Default for WasmValidator {
    fn default() -> Self {
        Self::new(WasmValidatorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = WasmValidator::default();
        assert!(validator.config.validate_structure);
        assert!(validator.config.validate_prism_semantics);
    }

    #[test]
    fn test_module_structure_validation() {
        let mut validator = WasmValidator::default();
        
        let valid_code = "(module\n  (func $test)\n)";
        let results = validator.validate(valid_code).unwrap();
        assert!(results.is_valid);
        
        let invalid_code = "(module\n  (func $test)"; // Missing closing
        let results = validator.validate(invalid_code).unwrap();
        assert!(!results.is_valid);
        assert!(!results.errors.is_empty());
    }

    #[test]
    fn test_import_validation() {
        let mut validator = WasmValidator::default();
        
        let code_with_imports = r#"
(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (import "prism_runtime" "validate_capability" (func $validate_cap (param i32 i32) (result i32)))
)
"#;
        
        let results = validator.validate(code_with_imports).unwrap();
        assert!(results.is_valid);
    }

    #[test]
    fn test_prism_semantics_validation() {
        let mut validator = WasmValidator::default();
        
        let code_with_prism = r#"
(module
  ;; Business rule validation
  (func $validate_business_rule)
  ;; Semantic Type: UserId (Domain: User Management)
  (func $prism_runtime_init)
  (func $prism_validate_capability)
  (func $prism_track_effect)
)
"#;
        
        let results = validator.validate(code_with_prism).unwrap();
        assert!(results.is_valid);
        assert!(results.info.iter().any(|i| i.message.contains("Semantic metadata")));
    }

    #[test]
    fn test_feature_validation() {
        let mut validator = WasmValidator::new(WasmValidatorConfig {
            features: WasmFeatures {
                simd: false,
                reference_types: false,
                ..WasmFeatures::default()
            },
            ..WasmValidatorConfig::default()
        });
        
        let code_with_simd = r#"
(module
  (func $test
    v128.const i32x4 0 1 2 3
  )
)
"#;
        
        let results = validator.validate(code_with_simd).unwrap();
        assert!(!results.is_valid);
        assert!(results.errors.iter().any(|e| e.code == "SIMD_NOT_ENABLED"));
    }
} 