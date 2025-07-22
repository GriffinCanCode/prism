//! LLVM IR Validation
//!
//! This module provides comprehensive validation for generated LLVM IR,
//! ensuring correctness, security, and compliance with LLVM standards.

use super::{LLVMResult, LLVMError};
use super::types::{LLVMTargetArch, LLVMOptimizationLevel};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// LLVM validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMValidatorConfig {
    /// Target architecture for validation
    pub target_arch: LLVMTargetArch,
    /// Optimization level (affects validation strictness)
    pub optimization_level: LLVMOptimizationLevel,
    /// Validate module structure
    pub validate_structure: bool,
    /// Validate type usage
    pub validate_types: bool,
    /// Validate function signatures
    pub validate_function_signatures: bool,
    /// Validate memory operations
    pub validate_memory_operations: bool,
    /// Validate control flow
    pub validate_control_flow: bool,
    /// Validate Prism-specific semantics
    pub validate_prism_semantics: bool,
    /// Validate debug information
    pub validate_debug_info: bool,
    /// Maximum allowed function size (instructions)
    pub max_function_size: u32,
    /// Maximum allowed basic block size
    pub max_basic_block_size: u32,
    /// Enable performance warnings
    pub performance_warnings: bool,
    /// Enable security checks
    pub security_checks: bool,
}

impl Default for LLVMValidatorConfig {
    fn default() -> Self {
        Self {
            target_arch: LLVMTargetArch::default(),
            optimization_level: LLVMOptimizationLevel::Aggressive,
            validate_structure: true,
            validate_types: true,
            validate_function_signatures: true,
            validate_memory_operations: true,
            validate_control_flow: true,
            validate_prism_semantics: true,
            validate_debug_info: true,
            max_function_size: 10000,
            max_basic_block_size: 1000,
            performance_warnings: true,
            security_checks: true,
        }
    }
}

/// LLVM IR validator
pub struct LLVMValidator {
    /// Validation configuration
    config: LLVMValidatorConfig,
    /// Validation results
    results: ValidationResults,
    /// Symbol table for validation
    symbol_table: HashMap<String, SymbolInfo>,
    /// Type table for validation
    type_table: HashMap<String, String>,
}

/// Validation results and diagnostics
#[derive(Debug, Clone, Default)]
pub struct ValidationResults {
    /// Validation errors (must be fixed)
    pub errors: Vec<ValidationError>,
    /// Validation warnings (should be addressed)
    pub warnings: Vec<ValidationWarning>,
    /// Performance suggestions
    pub suggestions: Vec<ValidationSuggestion>,
    /// Security issues
    pub security_issues: Vec<SecurityIssue>,
    /// Overall validation success
    pub is_valid: bool,
    /// Validation time in milliseconds
    pub validation_time_ms: u64,
    /// Statistics about the validated code
    pub stats: ValidationStats,
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
    /// Column number
    pub column: Option<usize>,
    /// Error severity
    pub severity: ValidationSeverity,
    /// Suggested fix
    pub suggestion: Option<String>,
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
    /// Suggested improvement
    pub suggestion: Option<String>,
}

/// Performance or style suggestion
#[derive(Debug, Clone)]
pub struct ValidationSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Suggestion message
    pub message: String,
    /// Line number
    pub line: Option<usize>,
    /// Expected performance impact
    pub impact: PerformanceImpact,
    /// Suggested action
    pub action: String,
}

/// Security issue
#[derive(Debug, Clone)]
pub struct SecurityIssue {
    /// Security issue type
    pub issue_type: SecurityIssueType,
    /// Issue description
    pub message: String,
    /// Line number
    pub line: Option<usize>,
    /// Severity level
    pub severity: SecuritySeverity,
    /// Mitigation suggestion
    pub mitigation: String,
}

/// Validation statistics
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    /// Total lines of IR
    pub total_lines: usize,
    /// Number of functions
    pub function_count: usize,
    /// Number of basic blocks
    pub basic_block_count: usize,
    /// Number of instructions
    pub instruction_count: usize,
    /// Number of global variables
    pub global_count: usize,
    /// Number of type definitions
    pub type_count: usize,
    /// Average function size
    pub avg_function_size: f64,
    /// Largest function size
    pub max_function_size: usize,
    /// Complexity score
    pub complexity_score: f64,
}

/// Symbol information for validation
#[derive(Debug, Clone)]
struct SymbolInfo {
    /// Symbol type
    symbol_type: SymbolType,
    /// Defined location
    defined_at: Option<usize>,
    /// Used locations
    used_at: Vec<usize>,
    /// Symbol attributes
    attributes: Vec<String>,
}

/// Symbol types
#[derive(Debug, Clone, PartialEq, Eq)]
enum SymbolType {
    Function,
    GlobalVariable,
    LocalVariable,
    BasicBlock,
    Type,
}

/// Validation categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationCategory {
    /// Module structure issues
    Structure,
    /// Type system issues
    Types,
    /// Function definition issues
    Functions,
    /// Memory management issues
    Memory,
    /// Control flow issues
    ControlFlow,
    /// Prism semantic issues
    PrismSemantics,
    /// Performance issues
    Performance,
    /// Security issues
    Security,
    /// Debug information issues
    DebugInfo,
}

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Suggestion types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionType {
    Performance,
    Style,
    Maintainability,
    Optimization,
}

/// Performance impact levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceImpact {
    Low,
    Medium,
    High,
    Critical,
}

/// Security issue types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityIssueType {
    BufferOverflow,
    IntegerOverflow,
    UnvalidatedInput,
    MemoryLeak,
    UseAfterFree,
    DoubleFree,
    UnauthorizedAccess,
}

/// Security severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl LLVMValidator {
    /// Create new validator with configuration
    pub fn new(config: LLVMValidatorConfig) -> Self {
        Self {
            config,
            results: ValidationResults::default(),
            symbol_table: HashMap::new(),
            type_table: HashMap::new(),
        }
    }

    /// Validate LLVM IR code
    pub fn validate(&mut self, llvm_ir: &str) -> LLVMResult<&ValidationResults> {
        let start_time = std::time::Instant::now();
        
        // Reset results
        self.results = ValidationResults::default();
        self.symbol_table.clear();
        self.type_table.clear();

        // Parse IR into lines for analysis
        let lines: Vec<&str> = llvm_ir.lines().collect();
        self.results.stats.total_lines = lines.len();

        // Run validation passes
        if self.config.validate_structure {
            self.validate_module_structure(&lines)?;
        }

        if self.config.validate_types {
            self.validate_type_usage(&lines)?;
        }

        if self.config.validate_function_signatures {
            self.validate_function_signatures(&lines)?;
        }

        if self.config.validate_memory_operations {
            self.validate_memory_operations(&lines)?;
        }

        if self.config.validate_control_flow {
            self.validate_control_flow(&lines)?;
        }

        if self.config.validate_prism_semantics {
            self.validate_prism_semantics(&lines)?;
        }

        if self.config.validate_debug_info {
            self.validate_debug_info(&lines)?;
        }

        if self.config.performance_warnings {
            self.analyze_performance(&lines)?;
        }

        if self.config.security_checks {
            self.analyze_security(&lines)?;
        }

        // Generate statistics
        self.generate_statistics(&lines);

        // Determine overall validation result
        self.results.is_valid = self.results.errors.is_empty();
        self.results.validation_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(&self.results)
    }

    /// Validate module structure
    fn validate_module_structure(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut has_target_triple = false;
        let mut has_data_layout = false;
        let mut function_count = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("target triple") {
                has_target_triple = true;
            } else if trimmed.starts_with("target datalayout") {
                has_data_layout = true;
            } else if trimmed.starts_with("define ") {
                function_count += 1;
            }

            // Check for malformed lines
            if !trimmed.is_empty() && !trimmed.starts_with(";") {
                if trimmed.contains("<<INVALID>>") || trimmed.contains("<<ERROR>>") {
                    self.results.errors.push(ValidationError {
                        code: "E001".to_string(),
                        message: "Malformed LLVM IR detected".to_string(),
                        line: Some(line_num + 1),
                        column: None,
                        severity: ValidationSeverity::Error,
                        suggestion: Some("Check IR generation for syntax errors".to_string()),
                        category: ValidationCategory::Structure,
                    });
                }
            }
        }

        if !has_target_triple {
            self.results.warnings.push(ValidationWarning {
                code: "W001".to_string(),
                message: "Missing target triple specification".to_string(),
                line: None,
                category: ValidationCategory::Structure,
                suggestion: Some("Add 'target triple = \"...\"' to specify target architecture".to_string()),
            });
        }

        if !has_data_layout {
            self.results.warnings.push(ValidationWarning {
                code: "W002".to_string(),
                message: "Missing data layout specification".to_string(),
                line: None,
                category: ValidationCategory::Structure,
                suggestion: Some("Add 'target datalayout = \"...\"' for proper type layout".to_string()),
            });
        }

        if function_count == 0 {
            self.results.warnings.push(ValidationWarning {
                code: "W003".to_string(),
                message: "No functions defined - empty module".to_string(),
                line: None,
                category: ValidationCategory::Structure,
                suggestion: Some("Add at least one function definition".to_string()),
            });
        }

        self.results.stats.function_count = function_count;

        Ok(())
    }

    /// Validate type usage
    fn validate_type_usage(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut defined_types = HashSet::new();
        let mut used_types = HashSet::new();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Track type definitions
            if trimmed.starts_with("%struct.") || trimmed.starts_with("%union.") {
                if let Some(type_name) = self.extract_type_name(trimmed) {
                    defined_types.insert(type_name.clone());
                    self.type_table.insert(type_name, trimmed.to_string());
                }
            }

            // Track type usage
            for type_name in self.extract_used_types(trimmed) {
                used_types.insert(type_name);
            }

            // Check for invalid type usage
            if trimmed.contains("i0") || trimmed.contains("i1024") {
                self.results.errors.push(ValidationError {
                    code: "E101".to_string(),
                    message: "Invalid integer type width".to_string(),
                    line: Some(line_num + 1),
                    column: None,
                    severity: ValidationSeverity::Error,
                    suggestion: Some("Use valid integer widths (i1, i8, i16, i32, i64, i128)".to_string()),
                    category: ValidationCategory::Types,
                });
            }
        }

        // Check for undefined types
        for used_type in &used_types {
            if used_type.starts_with("%") && !defined_types.contains(used_type) {
                self.results.errors.push(ValidationError {
                    code: "E102".to_string(),
                    message: format!("Undefined type: {}", used_type),
                    line: None,
                    column: None,
                    severity: ValidationSeverity::Error,
                    suggestion: Some(format!("Define type {} before using it", used_type)),
                    category: ValidationCategory::Types,
                });
            }
        }

        self.results.stats.type_count = defined_types.len();

        Ok(())
    }

    /// Validate function signatures
    fn validate_function_signatures(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut current_function = None;
        let mut function_size = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("define ") {
                // Validate function definition
                if !self.is_valid_function_signature(trimmed) {
                    self.results.errors.push(ValidationError {
                        code: "E201".to_string(),
                        message: "Invalid function signature".to_string(),
                        line: Some(line_num + 1),
                        column: None,
                        severity: ValidationSeverity::Error,
                        suggestion: Some("Check function return type and parameter types".to_string()),
                        category: ValidationCategory::Functions,
                    });
                }

                current_function = self.extract_function_name(trimmed);
                function_size = 0;
            } else if trimmed == "}" {
                // End of function
                if let Some(func_name) = &current_function {
                    if function_size > self.config.max_function_size as usize {
                        self.results.warnings.push(ValidationWarning {
                            code: "W201".to_string(),
                            message: format!("Function {} is too large ({} instructions)", func_name, function_size),
                            line: Some(line_num + 1),
                            category: ValidationCategory::Performance,
                            suggestion: Some("Consider breaking large functions into smaller ones".to_string()),
                        });
                    }

                    self.results.stats.max_function_size = self.results.stats.max_function_size.max(function_size);
                }
                current_function = None;
            } else if current_function.is_some() && !trimmed.is_empty() && !trimmed.starts_with(";") {
                function_size += 1;
            }
        }

        // Calculate average function size
        if self.results.stats.function_count > 0 {
            self.results.stats.avg_function_size = 
                self.results.stats.instruction_count as f64 / self.results.stats.function_count as f64;
        }

        Ok(())
    }

    /// Validate memory operations
    fn validate_memory_operations(&mut self, lines: &[&str]) -> LLVMResult<()> {
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check for unsafe memory operations
            if trimmed.contains("getelementptr") && !trimmed.contains("inbounds") {
                self.results.warnings.push(ValidationWarning {
                    code: "W301".to_string(),
                    message: "Unsafe getelementptr without bounds checking".to_string(),
                    line: Some(line_num + 1),
                    category: ValidationCategory::Memory,
                    suggestion: Some("Use 'getelementptr inbounds' for better optimization and safety".to_string()),
                });
            }

            // Check for potential memory leaks
            if trimmed.contains("call") && (trimmed.contains("malloc") || trimmed.contains("calloc")) {
                self.results.suggestions.push(ValidationSuggestion {
                    suggestion_type: SuggestionType::Performance,
                    message: "Memory allocation detected - ensure corresponding free".to_string(),
                    line: Some(line_num + 1),
                    impact: PerformanceImpact::Medium,
                    action: "Add corresponding free() call or use RAII pattern".to_string(),
                });
            }

            // Check for alignment issues
            if trimmed.contains("load") || trimmed.contains("store") {
                if !trimmed.contains("align") {
                    self.results.suggestions.push(ValidationSuggestion {
                        suggestion_type: SuggestionType::Performance,
                        message: "Memory operation without explicit alignment".to_string(),
                        line: Some(line_num + 1),
                        impact: PerformanceImpact::Low,
                        action: "Add explicit alignment for better performance".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate control flow
    fn validate_control_flow(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut basic_block_count = 0;
        let mut unreachable_count = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Count basic blocks
            if trimmed.ends_with(":") && !trimmed.starts_with(";") {
                basic_block_count += 1;
            }

            // Check for unreachable code
            if trimmed == "unreachable" {
                unreachable_count += 1;
                self.results.warnings.push(ValidationWarning {
                    code: "W401".to_string(),
                    message: "Unreachable code detected".to_string(),
                    line: Some(line_num + 1),
                    category: ValidationCategory::ControlFlow,
                    suggestion: Some("Remove unreachable code or fix control flow".to_string()),
                });
            }

            // Check for infinite loops
            if trimmed.starts_with("br label %") {
                let target = trimmed.split_whitespace().nth(2);
                if let Some(target_label) = target {
                    // Simple heuristic: if we branch to the same label frequently, it might be infinite
                    if trimmed.contains(target_label) {
                        self.results.suggestions.push(ValidationSuggestion {
                            suggestion_type: SuggestionType::Performance,
                            message: "Potential infinite loop detected".to_string(),
                            line: Some(line_num + 1),
                            impact: PerformanceImpact::High,
                            action: "Ensure loop has proper termination condition".to_string(),
                        });
                    }
                }
            }
        }

        self.results.stats.basic_block_count = basic_block_count;

        Ok(())
    }

    /// Validate Prism-specific semantics
    fn validate_prism_semantics(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut has_capability_validation = false;
        let mut has_effect_tracking = false;
        let mut has_semantic_types = false;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check for Prism runtime integration
            if trimmed.contains("prism_validate_capability") {
                has_capability_validation = true;
            }

            if trimmed.contains("prism_track_effect") || trimmed.contains("prism_begin_effect_tracking") {
                has_effect_tracking = true;
            }

            if trimmed.contains("%struct.") && (
                trimmed.contains("semantic") || 
                trimmed.contains("business") || 
                trimmed.contains("domain")
            ) {
                has_semantic_types = true;
            }

            // Check for TODO comments
            if trimmed.contains("TODO:") || trimmed.contains("FIXME:") {
                self.results.warnings.push(ValidationWarning {
                    code: "W501".to_string(),
                    message: "Incomplete implementation detected".to_string(),
                    line: Some(line_num + 1),
                    category: ValidationCategory::PrismSemantics,
                    suggestion: Some("Complete the implementation or remove TODO/FIXME comments".to_string()),
                });
            }
        }

        if !has_capability_validation && self.config.security_checks {
            self.results.warnings.push(ValidationWarning {
                code: "W502".to_string(),
                message: "No capability validation found - potential security risk".to_string(),
                line: None,
                category: ValidationCategory::Security,
                suggestion: Some("Add capability validation for security-sensitive functions".to_string()),
            });
        }

        if !has_effect_tracking {
            self.results.suggestions.push(ValidationSuggestion {
                suggestion_type: SuggestionType::Maintainability,
                message: "No effect tracking detected".to_string(),
                line: None,
                impact: PerformanceImpact::Low,
                action: "Consider adding effect tracking for better debugging and analysis".to_string(),
            });
        }

        Ok(())
    }

    /// Validate debug information
    fn validate_debug_info(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut has_debug_info = false;

        for line in lines {
            let trimmed = line.trim();

            if trimmed.contains("!dbg") || trimmed.contains("DILocation") || trimmed.contains("DISubprogram") {
                has_debug_info = true;
                break;
            }
        }

        if !has_debug_info && self.config.optimization_level == LLVMOptimizationLevel::None {
            self.results.suggestions.push(ValidationSuggestion {
                suggestion_type: SuggestionType::Maintainability,
                message: "No debug information found in unoptimized build".to_string(),
                line: None,
                impact: PerformanceImpact::Low,
                action: "Add debug information for better debugging experience".to_string(),
            });
        }

        Ok(())
    }

    /// Analyze performance characteristics
    fn analyze_performance(&mut self, lines: &[&str]) -> LLVMResult<()> {
        let mut instruction_count = 0;
        let mut call_count = 0;
        let mut branch_count = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if !trimmed.is_empty() && !trimmed.starts_with(";") && !trimmed.starts_with("target") {
                instruction_count += 1;
            }

            if trimmed.contains("call ") {
                call_count += 1;
            }

            if trimmed.starts_with("br ") {
                branch_count += 1;
            }

            // Check for performance anti-patterns
            if trimmed.contains("call") && trimmed.contains("strlen") {
                self.results.suggestions.push(ValidationSuggestion {
                    suggestion_type: SuggestionType::Performance,
                    message: "Frequent strlen calls can be expensive".to_string(),
                    line: Some(line_num + 1),
                    impact: PerformanceImpact::Medium,
                    action: "Cache string length or use length-aware string operations".to_string(),
                });
            }

            if trimmed.contains("alloca") && trimmed.contains("[") {
                self.results.suggestions.push(ValidationSuggestion {
                    suggestion_type: SuggestionType::Performance,
                    message: "Large stack allocation detected".to_string(),
                    line: Some(line_num + 1),
                    impact: PerformanceImpact::Medium,
                    action: "Consider heap allocation for large arrays".to_string(),
                });
            }
        }

        self.results.stats.instruction_count = instruction_count;

        // Calculate complexity score
        let complexity_factors = [
            (call_count as f64, 2.0),
            (branch_count as f64, 1.5),
            (self.results.stats.basic_block_count as f64, 1.2),
        ];

        self.results.stats.complexity_score = complexity_factors
            .iter()
            .map(|(count, weight)| count * weight)
            .sum::<f64>() / instruction_count.max(1) as f64;

        Ok(())
    }

    /// Analyze security characteristics
    fn analyze_security(&mut self, lines: &[&str]) -> LLVMResult<()> {
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Check for buffer overflow risks
            if trimmed.contains("getelementptr") && !trimmed.contains("inbounds") {
                self.results.security_issues.push(SecurityIssue {
                    issue_type: SecurityIssueType::BufferOverflow,
                    message: "Potential buffer overflow in pointer arithmetic".to_string(),
                    line: Some(line_num + 1),
                    severity: SecuritySeverity::Medium,
                    mitigation: "Use 'getelementptr inbounds' or add bounds checking".to_string(),
                });
            }

            // Check for integer overflow
            if (trimmed.contains("add") || trimmed.contains("mul")) && !trimmed.contains("nsw") && !trimmed.contains("nuw") {
                self.results.security_issues.push(SecurityIssue {
                    issue_type: SecurityIssueType::IntegerOverflow,
                    message: "Potential integer overflow in arithmetic operation".to_string(),
                    line: Some(line_num + 1),
                    severity: SecuritySeverity::Low,
                    mitigation: "Add 'nsw' or 'nuw' flags or use overflow-checking intrinsics".to_string(),
                });
            }

            // Check for memory safety issues
            if trimmed.contains("free") {
                self.results.security_issues.push(SecurityIssue {
                    issue_type: SecurityIssueType::UseAfterFree,
                    message: "Manual memory management detected - ensure no use-after-free".to_string(),
                    line: Some(line_num + 1),
                    severity: SecuritySeverity::Medium,
                    mitigation: "Use RAII patterns or ensure pointers are nulled after free".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Generate validation statistics
    fn generate_statistics(&mut self, lines: &[&str]) {
        let mut global_count = 0;

        for line in lines {
            let trimmed = line.trim();
            
            if trimmed.starts_with("@") && !trimmed.contains("(") {
                global_count += 1;
            }
        }

        self.results.stats.global_count = global_count;
    }

    /// Helper methods
    fn extract_type_name(&self, line: &str) -> Option<String> {
        if let Some(eq_pos) = line.find('=') {
            let before_eq = &line[..eq_pos].trim();
            if before_eq.starts_with('%') {
                return Some(before_eq.to_string());
            }
        }
        None
    }

    fn extract_used_types(&self, line: &str) -> Vec<String> {
        let mut types = Vec::new();
        let words: Vec<&str> = line.split_whitespace().collect();
        
        for word in words {
            if word.starts_with('%') && word.contains('.') {
                types.push(word.to_string());
            }
        }
        
        types
    }

    fn is_valid_function_signature(&self, line: &str) -> bool {
        line.contains("(") && line.contains(")") && (line.contains("i32") || line.contains("void"))
    }

    fn extract_function_name(&self, line: &str) -> Option<String> {
        if let Some(at_pos) = line.find('@') {
            if let Some(paren_pos) = line[at_pos..].find('(') {
                let name = &line[at_pos + 1..at_pos + paren_pos];
                return Some(name.to_string());
            }
        }
        None
    }

    /// Get validation results
    pub fn get_results(&self) -> &ValidationResults {
        &self.results
    }

    /// Get validation configuration
    pub fn get_config(&self) -> &LLVMValidatorConfig {
        &self.config
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        format!(
            r#"LLVM IR Validation Report
=========================

Overall Status: {}
Validation Time: {}ms

Statistics:
- Total Lines: {}
- Functions: {}
- Basic Blocks: {}
- Instructions: {}
- Global Variables: {}
- Type Definitions: {}
- Average Function Size: {:.1}
- Largest Function: {} instructions
- Complexity Score: {:.2}

Issues Found:
- Errors: {}
- Warnings: {}
- Performance Suggestions: {}
- Security Issues: {}

Errors:
{}

Warnings:
{}

Performance Suggestions:
{}

Security Issues:
{}
"#,
            if self.results.is_valid { "VALID" } else { "INVALID" },
            self.results.validation_time_ms,
            self.results.stats.total_lines,
            self.results.stats.function_count,
            self.results.stats.basic_block_count,
            self.results.stats.instruction_count,
            self.results.stats.global_count,
            self.results.stats.type_count,
            self.results.stats.avg_function_size,
            self.results.stats.max_function_size,
            self.results.stats.complexity_score,
            self.results.errors.len(),
            self.results.warnings.len(),
            self.results.suggestions.len(),
            self.results.security_issues.len(),
            self.results.errors.iter()
                .map(|e| format!("  [{}] Line {}: {}", e.code, e.line.unwrap_or(0), e.message))
                .collect::<Vec<_>>()
                .join("\n"),
            self.results.warnings.iter()
                .map(|w| format!("  [{}] Line {}: {}", w.code, w.line.unwrap_or(0), w.message))
                .collect::<Vec<_>>()
                .join("\n"),
            self.results.suggestions.iter()
                .map(|s| format!("  [{:?}] Line {}: {} ({})", s.impact, s.line.unwrap_or(0), s.message, s.action))
                .collect::<Vec<_>>()
                .join("\n"),
            self.results.security_issues.iter()
                .map(|s| format!("  [{:?}] Line {}: {} ({})", s.severity, s.line.unwrap_or(0), s.message, s.mitigation))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

impl Clone for LLVMValidator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            results: ValidationResults::default(), // Don't clone results
            symbol_table: HashMap::new(),
            type_table: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = LLVMValidator::new(LLVMValidatorConfig::default());
        assert!(validator.config.validate_structure);
        assert!(validator.config.validate_types);
        assert!(validator.config.performance_warnings);
    }

    #[test]
    fn test_basic_validation() {
        let mut validator = LLVMValidator::new(LLVMValidatorConfig::default());
        
        let ir = r#"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @main() {
entry:
  ret i32 0
}
"#;
        
        let result = validator.validate(ir).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert_eq!(result.stats.function_count, 1);
    }

    #[test]
    fn test_error_detection() {
        let mut validator = LLVMValidator::new(LLVMValidatorConfig::default());
        
        let invalid_ir = r#"
define i0 @invalid() {
  ret i0 0
}
"#;
        
        let result = validator.validate(invalid_ir).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
        assert!(result.errors.iter().any(|e| e.code == "E101"));
    }

    #[test]
    fn test_warning_detection() {
        let mut validator = LLVMValidator::new(LLVMValidatorConfig::default());
        
        let ir_without_triple = r#"
define i32 @test() {
  ret i32 0
}
"#;
        
        let result = validator.validate(ir_without_triple).unwrap();
        assert!(result.is_valid); // No errors, just warnings
        assert!(!result.warnings.is_empty());
        assert!(result.warnings.iter().any(|w| w.code == "W001"));
    }

    #[test]
    fn test_security_analysis() {
        let mut validator = LLVMValidator::new(LLVMValidatorConfig::default());
        
        let unsafe_ir = r#"
define i32 @unsafe() {
  %ptr = getelementptr i32, i32* %base, i32 %offset
  %val = load i32, i32* %ptr
  ret i32 %val
}
"#;
        
        let result = validator.validate(unsafe_ir).unwrap();
        assert!(!result.security_issues.is_empty());
        assert!(result.security_issues.iter().any(|s| s.issue_type == SecurityIssueType::BufferOverflow));
    }

    #[test]
    fn test_performance_analysis() {
        let mut validator = LLVMValidator::new(LLVMValidatorConfig::default());
        
        let performance_ir = r#"
define i32 @performance_test() {
  %len = call i32 @strlen(i8* %str)
  %arr = alloca [1000 x i32]
  ret i32 0
}
"#;
        
        let result = validator.validate(performance_ir).unwrap();
        assert!(!result.suggestions.is_empty());
        assert!(result.suggestions.iter().any(|s| s.suggestion_type == SuggestionType::Performance));
    }

    #[test]
    fn test_statistics_generation() {
        let mut validator = LLVMValidator::new(LLVMValidatorConfig::default());
        
        let ir = r#"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test1() {
entry:
  %x = add i32 1, 2
  %y = mul i32 %x, 3
  ret i32 %y
}

define void @test2() {
entry:
  ret void
}
"#;
        
        let result = validator.validate(ir).unwrap();
        assert_eq!(result.stats.function_count, 2);
        assert!(result.stats.instruction_count > 0);
        assert!(result.stats.avg_function_size > 0.0);
    }
} 