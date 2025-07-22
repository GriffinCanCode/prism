//! Python Optimization Module
//!
//! This module handles optimization of generated Python code using various
//! techniques including code simplification, performance hints, and modern
//! Python optimization patterns.

use super::{PythonResult, PythonError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use regex::Regex;

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable performance hints
    pub enable_performance_hints: bool,
    /// Enable code simplification
    pub enable_simplification: bool,
    /// Enable import optimization
    pub enable_import_optimization: bool,
    /// Enable constant folding
    pub enable_constant_folding: bool,
    /// Enable dead code elimination
    pub enable_dead_code_elimination: bool,
    /// Enable async optimization
    pub enable_async_optimization: bool,
    /// Enable type hint optimization
    pub enable_type_hint_optimization: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Target Python version for optimizations
    pub target_version: String,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_performance_hints: true,
            enable_simplification: true,
            enable_import_optimization: true,
            enable_constant_folding: true,
            enable_dead_code_elimination: false, // Conservative default
            enable_async_optimization: true,
            enable_type_hint_optimization: true,
            optimization_level: 2,
            target_version: "3.12".to_string(),
        }
    }
}

/// Python optimizer with comprehensive optimization strategies
pub struct PythonOptimizer {
    config: OptimizationConfig,
    optimization_cache: HashMap<String, String>,
    performance_hints: Vec<PerformanceHint>,
}

impl PythonOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            optimization_cache: HashMap::new(),
            performance_hints: Vec::new(),
        }
    }

    /// Optimize Python code comprehensively
    pub fn optimize(&self, code: &str) -> PythonResult<String> {
        let mut optimized_code = code.to_string();
        
        // Apply optimizations based on level
        if self.config.optimization_level >= 1 {
            optimized_code = self.optimize_imports(&optimized_code)?;
            optimized_code = self.simplify_code(&optimized_code)?;
        }
        
        if self.config.optimization_level >= 2 {
            optimized_code = self.optimize_type_hints(&optimized_code)?;
            optimized_code = self.optimize_async_patterns(&optimized_code)?;
            optimized_code = self.fold_constants(&optimized_code)?;
        }
        
        if self.config.optimization_level >= 3 {
            optimized_code = self.eliminate_dead_code(&optimized_code)?;
            optimized_code = self.optimize_performance_critical_sections(&optimized_code)?;
        }
        
        // Add performance hints as comments
        if self.config.enable_performance_hints {
            optimized_code = self.add_performance_hints(&optimized_code)?;
        }
        
        Ok(optimized_code)
    }

    /// Optimize import statements
    fn optimize_imports(&self, code: &str) -> PythonResult<String> {
        if !self.config.enable_import_optimization {
            return Ok(code.to_string());
        }

        let lines: Vec<&str> = code.lines().collect();
        let mut optimized_lines = Vec::new();
        let mut imports = Vec::new();
        let mut from_imports: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_import_section = true;
        
        for line in lines {
            let trimmed = line.trim();
            
            if trimmed.is_empty() || trimmed.starts_with('#') {
                if in_import_section {
                    optimized_lines.push(line.to_string());
                } else {
                    optimized_lines.push(line.to_string());
                }
                continue;
            }
            
            if trimmed.starts_with("import ") && in_import_section {
                let import_name = trimmed.strip_prefix("import ").unwrap().trim();
                if !imports.contains(&import_name.to_string()) {
                    imports.push(import_name.to_string());
                }
            } else if trimmed.starts_with("from ") && trimmed.contains(" import ") && in_import_section {
                if let Some(captures) = Regex::new(r"from\s+(\S+)\s+import\s+(.+)")
                    .unwrap()
                    .captures(trimmed) 
                {
                    let module = captures.get(1).unwrap().as_str();
                    let imports_str = captures.get(2).unwrap().as_str();
                    
                    let module_imports = from_imports.entry(module.to_string()).or_insert_with(Vec::new);
                    for import in imports_str.split(',') {
                        let import_name = import.trim().to_string();
                        if !module_imports.contains(&import_name) {
                            module_imports.push(import_name);
                        }
                    }
                }
            } else {
                // Not an import line, end of import section
                if in_import_section {
                    // Add optimized imports
                    self.add_optimized_imports(&mut optimized_lines, &imports, &from_imports);
                    optimized_lines.push("".to_string()); // Add blank line after imports
                    in_import_section = false;
                }
                optimized_lines.push(line.to_string());
            }
        }
        
        // If we ended while still in import section
        if in_import_section {
            self.add_optimized_imports(&mut optimized_lines, &imports, &from_imports);
        }
        
        Ok(optimized_lines.join("\n"))
    }

    /// Add optimized imports to the output
    fn add_optimized_imports(
        &self,
        output: &mut Vec<String>,
        imports: &[String],
        from_imports: &HashMap<String, Vec<String>>,
    ) {
        // Sort and add regular imports
        let mut sorted_imports = imports.to_vec();
        sorted_imports.sort();
        for import in sorted_imports {
            output.push(format!("import {}", import));
        }
        
        if !imports.is_empty() && !from_imports.is_empty() {
            output.push("".to_string()); // Blank line between import types
        }
        
        // Sort and add from imports
        let mut sorted_modules: Vec<_> = from_imports.keys().collect();
        sorted_modules.sort();
        
        for module in sorted_modules {
            let mut module_imports = from_imports[module].clone();
            module_imports.sort();
            
            if module_imports.len() <= 3 {
                output.push(format!("from {} import {}", module, module_imports.join(", ")));
            } else {
                // Multi-line import for readability
                output.push(format!("from {} import (", module));
                for (i, import) in module_imports.iter().enumerate() {
                    if i == module_imports.len() - 1 {
                        output.push(format!("    {}", import));
                    } else {
                        output.push(format!("    {},", import));
                    }
                }
                output.push(")".to_string());
            }
        }
    }

    /// Simplify code patterns
    fn simplify_code(&self, code: &str) -> PythonResult<String> {
        if !self.config.enable_simplification {
            return Ok(code.to_string());
        }

        let mut simplified = code.to_string();
        
        // Simplify boolean expressions
        simplified = Regex::new(r"if\s+(.+)\s+==\s+True:")
            .unwrap()
            .replace_all(&simplified, "if $1:")
            .to_string();
        
        simplified = Regex::new(r"if\s+(.+)\s+==\s+False:")
            .unwrap()
            .replace_all(&simplified, "if not $1:")
            .to_string();
        
        // Simplify string formatting
        simplified = self.optimize_string_formatting(&simplified)?;
        
        // Simplify list/dict comprehensions where possible
        simplified = self.optimize_comprehensions(&simplified)?;
        
        // Remove unnecessary parentheses
        simplified = self.remove_unnecessary_parentheses(&simplified)?;
        
        Ok(simplified)
    }

    /// Optimize string formatting
    fn optimize_string_formatting(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Convert old-style % formatting to f-strings (Python 3.6+)
        if self.config.target_version >= "3.6".to_string() {
            // Simple pattern: "text %s" % variable -> f"text {variable}"
            optimized = Regex::new(r#""([^"]*%s[^"]*)"\s*%\s*(\w+)"#)
                .unwrap()
                .replace_all(&optimized, |caps: &regex::Captures| {
                    let template = caps.get(1).unwrap().as_str().replace("%s", "{}");
                    let var = caps.get(2).unwrap().as_str();
                    format!(r#"f"{}""#, template.replace("{}", &format!("{{{}}}", var)))
                })
                .to_string();
        }
        
        Ok(optimized)
    }

    /// Optimize comprehensions
    fn optimize_comprehensions(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Convert simple loops to comprehensions
        // This is a basic example - real implementation would be more sophisticated
        optimized = Regex::new(r"result = \[\]\nfor (\w+) in (.+):\n    result\.append\((.+)\)")
            .unwrap()
            .replace_all(&optimized, "result = [$3 for $1 in $2]")
            .to_string();
        
        Ok(optimized)
    }

    /// Remove unnecessary parentheses
    fn remove_unnecessary_parentheses(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Remove parentheses around single return values
        optimized = Regex::new(r"return \(([^,()]+)\)")
            .unwrap()
            .replace_all(&optimized, "return $1")
            .to_string();
        
        Ok(optimized)
    }

    /// Optimize type hints
    fn optimize_type_hints(&self, code: &str) -> PythonResult<String> {
        if !self.config.enable_type_hint_optimization {
            return Ok(code.to_string());
        }

        let mut optimized = code.to_string();
        
        // Use modern union syntax (Python 3.10+)
        if self.config.target_version >= "3.10".to_string() {
            optimized = Regex::new(r"Union\[([^]]+)\]")
                .unwrap()
                .replace_all(&optimized, |caps: &regex::Captures| {
                    let types = caps.get(1).unwrap().as_str();
                    types.replace(", ", " | ")
                })
                .to_string();
            
            optimized = Regex::new(r"Optional\[([^]]+)\]")
                .unwrap()
                .replace_all(&optimized, "$1 | None")
                .to_string();
        }
        
        // Simplify generic type hints where possible
        optimized = self.simplify_generic_types(&optimized)?;
        
        Ok(optimized)
    }

    /// Simplify generic type hints
    fn simplify_generic_types(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Use built-in generics (Python 3.9+)
        if self.config.target_version >= "3.9".to_string() {
            optimized = optimized.replace("List[", "list[");
            optimized = optimized.replace("Dict[", "dict[");
            optimized = optimized.replace("Set[", "set[");
            optimized = optimized.replace("Tuple[", "tuple[");
        }
        
        Ok(optimized)
    }

    /// Optimize async patterns
    fn optimize_async_patterns(&self, code: &str) -> PythonResult<String> {
        if !self.config.enable_async_optimization {
            return Ok(code.to_string());
        }

        let mut optimized = code.to_string();
        
        // Convert synchronous patterns to async where beneficial
        optimized = self.optimize_async_context_managers(&optimized)?;
        optimized = self.optimize_async_comprehensions(&optimized)?;
        
        Ok(optimized)
    }

    /// Optimize async context managers
    fn optimize_async_context_managers(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Convert with statements to async with for known async context managers
        optimized = Regex::new(r"with\s+(EffectTracker\(\)\.track_effects\([^)]+\))\s+as\s+(\w+):")
            .unwrap()
            .replace_all(&optimized, "async with $1 as $2:")
            .to_string();
        
        Ok(optimized)
    }

    /// Optimize async comprehensions
    fn optimize_async_comprehensions(&self, code: &str) -> PythonResult<String> {
        // This would implement async comprehension optimizations
        // For now, return as-is
        Ok(code.to_string())
    }

    /// Fold constants
    fn fold_constants(&self, code: &str) -> PythonResult<String> {
        if !self.config.enable_constant_folding {
            return Ok(code.to_string());
        }

        let mut optimized = code.to_string();
        
        // Simple constant folding examples
        optimized = Regex::new(r"(\d+)\s*\+\s*(\d+)")
            .unwrap()
            .replace_all(&optimized, |caps: &regex::Captures| {
                let a: i32 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
                let b: i32 = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
                (a + b).to_string()
            })
            .to_string();
        
        optimized = Regex::new(r"(\d+)\s*\*\s*(\d+)")
            .unwrap()
            .replace_all(&optimized, |caps: &regex::Captures| {
                let a: i32 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
                let b: i32 = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
                (a * b).to_string()
            })
            .to_string();
        
        Ok(optimized)
    }

    /// Eliminate dead code
    fn eliminate_dead_code(&self, code: &str) -> PythonResult<String> {
        if !self.config.enable_dead_code_elimination {
            return Ok(code.to_string());
        }

        let lines: Vec<&str> = code.lines().collect();
        let mut optimized_lines = Vec::new();
        let mut in_unreachable_code = false;
        
        for line in lines {
            let trimmed = line.trim();
            
            // Reset unreachable flag for new function/class definitions
            if trimmed.starts_with("def ") || trimmed.starts_with("class ") || trimmed.starts_with("async def ") {
                in_unreachable_code = false;
            }
            
            // Mark code after return/raise as unreachable
            if trimmed.starts_with("return ") || trimmed.starts_with("raise ") {
                optimized_lines.push(line.to_string());
                in_unreachable_code = true;
                continue;
            }
            
            // Skip unreachable code (but keep comments and blank lines)
            if in_unreachable_code && !trimmed.is_empty() && !trimmed.starts_with('#') {
                continue;
            }
            
            // Reset unreachable flag for new blocks
            if !line.starts_with(' ') && !line.starts_with('\t') && !trimmed.is_empty() {
                in_unreachable_code = false;
            }
            
            optimized_lines.push(line.to_string());
        }
        
        Ok(optimized_lines.join("\n"))
    }

    /// Optimize performance-critical sections
    fn optimize_performance_critical_sections(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Add __slots__ to classes where beneficial
        optimized = self.add_slots_optimization(&optimized)?;
        
        // Optimize frequent operations
        optimized = self.optimize_frequent_operations(&optimized)?;
        
        Ok(optimized)
    }

    /// Add __slots__ optimization to classes
    fn add_slots_optimization(&self, code: &str) -> PythonResult<String> {
        // This would analyze classes and add __slots__ where beneficial
        // For now, return as-is to avoid breaking existing code
        Ok(code.to_string())
    }

    /// Optimize frequent operations
    fn optimize_frequent_operations(&self, code: &str) -> PythonResult<String> {
        let mut optimized = code.to_string();
        
        // Use more efficient methods for common operations
        optimized = optimized.replace(".append(", ".append(");  // No-op for now
        
        Ok(optimized)
    }

    /// Add performance hints as comments
    fn add_performance_hints(&self, code: &str) -> PythonResult<String> {
        let mut lines: Vec<String> = code.lines().map(|s| s.to_string()).collect();
        let mut hint_index = 0;
        
        // Add hints at strategic locations
        for i in 0..lines.len() {
            let line = &lines[i];
            
            // Add hints for loops
            if line.trim().starts_with("for ") && line.contains(" in ") {
                if let Some(hint) = self.get_loop_performance_hint(line) {
                    lines.insert(i, format!("    # PERF: {}", hint));
                    hint_index += 1;
                }
            }
            
            // Add hints for function definitions
            if line.trim().starts_with("def ") || line.trim().starts_with("async def ") {
                if let Some(hint) = self.get_function_performance_hint(line) {
                    lines.insert(i, format!("    # PERF: {}", hint));
                    hint_index += 1;
                }
            }
        }
        
        Ok(lines.join("\n"))
    }

    /// Get performance hint for loops
    fn get_loop_performance_hint(&self, line: &str) -> Option<String> {
        if line.contains("range(len(") {
            Some("Consider using enumerate() instead of range(len())".to_string())
        } else if line.contains("in dict.keys()") {
            Some("Iterating over dict.keys() is redundant, use 'for key in dict:'".to_string())
        } else {
            None
        }
    }

    /// Get performance hint for functions
    fn get_function_performance_hint(&self, line: &str) -> Option<String> {
        if line.contains("def ") && !line.contains("->") {
            Some("Consider adding type hints for better performance and clarity".to_string())
        } else {
            None
        }
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self, original: &str, optimized: &str) -> OptimizationStats {
        OptimizationStats {
            original_lines: original.lines().count(),
            optimized_lines: optimized.lines().count(),
            lines_saved: original.lines().count().saturating_sub(optimized.lines().count()),
            original_size: original.len(),
            optimized_size: optimized.len(),
            size_reduction: original.len().saturating_sub(optimized.len()),
            optimizations_applied: self.count_optimizations_applied(original, optimized),
            performance_hints_added: self.performance_hints.len(),
        }
    }

    /// Count optimizations applied
    fn count_optimizations_applied(&self, original: &str, optimized: &str) -> usize {
        let mut count = 0;
        
        // Count import optimizations
        if original.matches("import ").count() != optimized.matches("import ").count() {
            count += 1;
        }
        
        // Count type hint optimizations
        if original.contains("Union[") && !optimized.contains("Union[") {
            count += 1;
        }
        
        // Count boolean simplifications
        if original.contains("== True") && !optimized.contains("== True") {
            count += 1;
        }
        
        count
    }
}

/// Performance hint for code optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHint {
    pub category: String,
    pub message: String,
    pub line_number: Option<usize>,
    pub severity: HintSeverity,
}

/// Severity level for performance hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HintSeverity {
    Info,
    Warning,
    Critical,
}

/// Optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub original_lines: usize,
    pub optimized_lines: usize,
    pub lines_saved: usize,
    pub original_size: usize,
    pub optimized_size: usize,
    pub size_reduction: usize,
    pub optimizations_applied: usize,
    pub performance_hints_added: usize,
}

/// Performance hints collection
pub struct PerformanceHints {
    hints: Vec<PerformanceHint>,
}

impl PerformanceHints {
    pub fn new() -> Self {
        Self {
            hints: Vec::new(),
        }
    }

    pub fn add_hint(&mut self, category: String, message: String, line_number: Option<usize>, severity: HintSeverity) {
        self.hints.push(PerformanceHint {
            category,
            message,
            line_number,
            severity,
        });
    }

    pub fn get_hints(&self) -> &[PerformanceHint] {
        &self.hints
    }

    pub fn get_hints_by_severity(&self, severity: HintSeverity) -> Vec<&PerformanceHint> {
        self.hints.iter()
            .filter(|hint| matches!(hint.severity, severity))
            .collect()
    }
} 