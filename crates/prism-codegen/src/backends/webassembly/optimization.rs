//! WebAssembly Code Optimization
//!
//! This module provides optimization passes for WebAssembly code generation,
//! following modern WASM optimization practices while preserving semantic information.

use super::{WasmResult, WasmError};
use super::types::WasmOptimizationLevel;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// WebAssembly optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmOptimizerConfig {
    /// Optimization level
    pub level: WasmOptimizationLevel,
    /// Preserve debug information
    pub preserve_debug_info: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable instruction combining
    pub instruction_combining: bool,
    /// Enable local variable optimization
    pub local_optimization: bool,
    /// Enable control flow optimization
    pub control_flow_optimization: bool,
    /// Preserve semantic metadata
    pub preserve_semantic_metadata: bool,
}

impl Default for WasmOptimizerConfig {
    fn default() -> Self {
        Self {
            level: WasmOptimizationLevel::Speed,
            preserve_debug_info: true,
            dead_code_elimination: true,
            constant_folding: true,
            instruction_combining: true,
            local_optimization: true,
            control_flow_optimization: true,
            preserve_semantic_metadata: true,
        }
    }
}

impl From<WasmOptimizationLevel> for WasmOptimizerConfig {
    fn from(level: WasmOptimizationLevel) -> Self {
        match level {
            WasmOptimizationLevel::None => Self {
                level,
                preserve_debug_info: true,
                dead_code_elimination: false,
                constant_folding: false,
                instruction_combining: false,
                local_optimization: false,
                control_flow_optimization: false,
                preserve_semantic_metadata: true,
            },
            WasmOptimizationLevel::Size => Self {
                level,
                preserve_debug_info: false,
                dead_code_elimination: true,
                constant_folding: true,
                instruction_combining: true,
                local_optimization: true,
                control_flow_optimization: false,
                preserve_semantic_metadata: false,
            },
            WasmOptimizationLevel::Speed => Self {
                level,
                preserve_debug_info: true,
                dead_code_elimination: true,
                constant_folding: true,
                instruction_combining: true,
                local_optimization: true,
                control_flow_optimization: true,
                preserve_semantic_metadata: true,
            },
            WasmOptimizationLevel::Maximum => Self {
                level,
                preserve_debug_info: false,
                dead_code_elimination: true,
                constant_folding: true,
                instruction_combining: true,
                local_optimization: true,
                control_flow_optimization: true,
                preserve_semantic_metadata: false,
            },
        }
    }
}

/// WebAssembly code optimizer
pub struct WasmOptimizer {
    /// Optimization configuration
    config: WasmOptimizerConfig,
    /// Optimization statistics
    stats: OptimizationStats,
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of instructions before optimization
    pub instructions_before: usize,
    /// Number of instructions after optimization
    pub instructions_after: usize,
    /// Number of locals before optimization
    pub locals_before: usize,
    /// Number of locals after optimization
    pub locals_after: usize,
    /// Number of dead code eliminations
    pub dead_code_eliminations: usize,
    /// Number of constant foldings
    pub constant_foldings: usize,
    /// Number of instruction combinations
    pub instruction_combinations: usize,
    /// Optimization time in milliseconds
    pub optimization_time_ms: u64,
}

impl OptimizationStats {
    /// Calculate instruction reduction percentage
    pub fn instruction_reduction_percent(&self) -> f64 {
        if self.instructions_before == 0 {
            0.0
        } else {
            ((self.instructions_before - self.instructions_after) as f64 / self.instructions_before as f64) * 100.0
        }
    }

    /// Calculate local reduction percentage
    pub fn local_reduction_percent(&self) -> f64 {
        if self.locals_before == 0 {
            0.0
        } else {
            ((self.locals_before - self.locals_after) as f64 / self.locals_before as f64) * 100.0
        }
    }
}

impl WasmOptimizer {
    /// Create new optimizer with configuration
    pub fn new(config: WasmOptimizerConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats::default(),
        }
    }

    /// Create optimizer from optimization level
    pub fn from_level(level: WasmOptimizationLevel) -> Self {
        Self::new(WasmOptimizerConfig::from(level))
    }

    /// Optimize WebAssembly code
    pub fn optimize(&mut self, wasm_code: &str) -> WasmResult<String> {
        let start_time = std::time::Instant::now();
        
        // Reset statistics
        self.stats = OptimizationStats::default();
        self.stats.instructions_before = self.count_instructions(wasm_code);
        self.stats.locals_before = self.count_locals(wasm_code);

        let mut optimized_code = wasm_code.to_string();

        // Apply optimization passes based on configuration
        if self.config.dead_code_elimination {
            optimized_code = self.eliminate_dead_code(&optimized_code)?;
        }

        if self.config.constant_folding {
            optimized_code = self.fold_constants(&optimized_code)?;
        }

        if self.config.instruction_combining {
            optimized_code = self.combine_instructions(&optimized_code)?;
        }

        if self.config.local_optimization {
            optimized_code = self.optimize_locals(&optimized_code)?;
        }

        if self.config.control_flow_optimization {
            optimized_code = self.optimize_control_flow(&optimized_code)?;
        }

        // Remove debug information if not preserving it
        if !self.config.preserve_debug_info {
            optimized_code = self.remove_debug_info(&optimized_code)?;
        }

        // Remove semantic metadata if not preserving it
        if !self.config.preserve_semantic_metadata {
            optimized_code = self.remove_semantic_metadata(&optimized_code)?;
        }

        // Update final statistics
        self.stats.instructions_after = self.count_instructions(&optimized_code);
        self.stats.locals_after = self.count_locals(&optimized_code);
        self.stats.optimization_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(optimized_code)
    }

    /// Eliminate dead code (unused functions, unreachable code)
    fn eliminate_dead_code(&mut self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();
        let mut in_unreachable_block = false;
        let mut eliminations = 0;

        for line in code.lines() {
            let trimmed = line.trim();
            
            // Skip obvious dead code patterns
            if trimmed.starts_with(";; TODO:") || 
               trimmed.starts_with(";; Complex expression") ||
               trimmed.contains("unreachable ;; Non-exhaustive match") {
                eliminations += 1;
                continue;
            }

            // Track unreachable blocks
            if trimmed == "unreachable" {
                in_unreachable_block = true;
                optimized.push_str(line);
                optimized.push('\n');
                continue;
            }

            // Skip code after unreachable until function/block end
            if in_unreachable_block {
                if trimmed.starts_with("end") || trimmed.starts_with(")") {
                    in_unreachable_block = false;
                    optimized.push_str(line);
                    optimized.push('\n');
                } else {
                    eliminations += 1;
                }
                continue;
            }

            optimized.push_str(line);
            optimized.push('\n');
        }

        self.stats.dead_code_eliminations = eliminations;
        Ok(optimized)
    }

    /// Fold constants (evaluate constant expressions at compile time)
    fn fold_constants(&mut self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();
        let mut foldings = 0;
        let lines: Vec<&str> = code.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Look for simple constant folding opportunities
            if i + 2 < lines.len() {
                let next1 = lines[i + 1].trim();
                let next2 = lines[i + 2].trim();

                // Pattern: i32.const X, i32.const Y, i32.add -> i32.const (X+Y)
                if trimmed.starts_with("i32.const ") && 
                   next1.starts_with("i32.const ") && 
                   next2 == "i32.add" {
                    
                    if let (Some(x), Some(y)) = (
                        self.extract_i32_const(trimmed),
                        self.extract_i32_const(next1)
                    ) {
                        // Fold the addition
                        let result = x.wrapping_add(y);
                        let indent = line.len() - trimmed.len();
                        optimized.push_str(&format!("{}i32.const {} ;; folded: {} + {}\n", 
                            " ".repeat(indent), result, x, y));
                        foldings += 1;
                        i += 3; // Skip the three lines we just folded
                        continue;
                    }
                }

                // Pattern: i32.const X, i32.const Y, i32.sub -> i32.const (X-Y)
                if trimmed.starts_with("i32.const ") && 
                   next1.starts_with("i32.const ") && 
                   next2 == "i32.sub" {
                    
                    if let (Some(x), Some(y)) = (
                        self.extract_i32_const(trimmed),
                        self.extract_i32_const(next1)
                    ) {
                        let result = x.wrapping_sub(y);
                        let indent = line.len() - trimmed.len();
                        optimized.push_str(&format!("{}i32.const {} ;; folded: {} - {}\n", 
                            " ".repeat(indent), result, x, y));
                        foldings += 1;
                        i += 3;
                        continue;
                    }
                }

                // Pattern: i32.const 0, X, i32.add -> X (additive identity)
                if trimmed == "i32.const 0" && next2 == "i32.add" {
                    // Just keep the middle instruction
                    optimized.push_str(&format!("{} ;; optimized: 0 + X = X\n", lines[i + 1]));
                    foldings += 1;
                    i += 3;
                    continue;
                }

                // Pattern: i32.const 1, X, i32.mul -> X (multiplicative identity)
                if trimmed == "i32.const 1" && next2 == "i32.mul" {
                    optimized.push_str(&format!("{} ;; optimized: 1 * X = X\n", lines[i + 1]));
                    foldings += 1;
                    i += 3;
                    continue;
                }
            }

            optimized.push_str(line);
            optimized.push('\n');
            i += 1;
        }

        self.stats.constant_foldings = foldings;
        Ok(optimized)
    }

    /// Combine instructions (peephole optimizations)
    fn combine_instructions(&mut self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();
        let mut combinations = 0;
        let lines: Vec<&str> = code.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Look for instruction combination opportunities
            if i + 1 < lines.len() {
                let next = lines[i + 1].trim();

                // Pattern: local.set $x, local.get $x -> local.tee $x
                if trimmed.starts_with("local.set ") && 
                   next.starts_with("local.get ") {
                    
                    let set_var = trimmed.strip_prefix("local.set ").unwrap_or("");
                    let get_var = next.strip_prefix("local.get ").unwrap_or("");
                    
                    if set_var == get_var {
                        let indent = line.len() - trimmed.len();
                        optimized.push_str(&format!("{}local.tee {} ;; combined set+get\n", 
                            " ".repeat(indent), set_var));
                        combinations += 1;
                        i += 2;
                        continue;
                    }
                }

                // Pattern: drop, drop -> (remove both, they're redundant)
                if trimmed == "drop" && next == "drop" {
                    // Skip both drops if they're consecutive
                    combinations += 1;
                    i += 2;
                    continue;
                }
            }

            optimized.push_str(line);
            optimized.push('\n');
            i += 1;
        }

        self.stats.instruction_combinations = combinations;
        Ok(optimized)
    }

    /// Optimize local variables (remove unused, reorder)
    fn optimize_locals(&mut self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();
        let mut used_locals = HashMap::new();

        // First pass: identify used locals
        for line in code.lines() {
            let trimmed = line.trim();
            
            if let Some(local) = self.extract_local_reference(trimmed) {
                *used_locals.entry(local).or_insert(0) += 1;
            }
        }

        // Second pass: remove unused local declarations
        for line in code.lines() {
            let trimmed = line.trim();
            
            // Check if this is a local declaration
            if trimmed.starts_with("(local ") {
                if let Some(local_name) = self.extract_local_declaration(trimmed) {
                    if used_locals.get(&local_name).unwrap_or(&0) > &0 {
                        optimized.push_str(line);
                        optimized.push('\n');
                    }
                    // Skip unused local declarations
                    continue;
                }
            }
            
            optimized.push_str(line);
            optimized.push('\n');
        }

        Ok(optimized)
    }

    /// Optimize control flow (remove redundant branches, simplify conditions)
    fn optimize_control_flow(&mut self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();
        let lines: Vec<&str> = code.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            // Look for control flow optimization opportunities
            if i + 2 < lines.len() {
                let next1 = lines[i + 1].trim();
                let next2 = lines[i + 2].trim();

                // Pattern: if, end (empty if block) -> remove
                if trimmed.starts_with("if") && next1.trim().is_empty() && next2 == "end" {
                    // Skip empty if block, but keep the condition evaluation
                    if trimmed.contains("(result") {
                        // Need to provide a default value for result type
                        optimized.push_str("    drop ;; removed empty if block\n");
                        optimized.push_str("    i32.const 0 ;; default value\n");
                    } else {
                        optimized.push_str("    drop ;; removed empty if block\n");
                    }
                    i += 3;
                    continue;
                }
            }

            optimized.push_str(line);
            optimized.push('\n');
            i += 1;
        }

        Ok(optimized)
    }

    /// Remove debug information
    fn remove_debug_info(&self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();

        for line in code.lines() {
            let trimmed = line.trim_start();
            
            // Remove debug comments but keep important structural comments
            if trimmed.starts_with(";;") {
                if trimmed.contains("Generated by") || 
                   trimmed.contains("===") ||
                   trimmed.contains("Module:") ||
                   trimmed.contains("Function:") {
                    optimized.push_str(line);
                    optimized.push('\n');
                }
                // Skip other debug comments
                continue;
            }
            
            optimized.push_str(line);
            optimized.push('\n');
        }

        Ok(optimized)
    }

    /// Remove semantic metadata
    fn remove_semantic_metadata(&self, code: &str) -> WasmResult<String> {
        let mut optimized = String::new();

        for line in code.lines() {
            let trimmed = line.trim_start();
            
            // Remove semantic metadata comments
            if trimmed.starts_with(";;") && (
                trimmed.contains("Business") ||
                trimmed.contains("Semantic") ||
                trimmed.contains("Effect") ||
                trimmed.contains("Capability") ||
                trimmed.contains("Domain:") ||
                trimmed.contains("Security:") ||
                trimmed.contains("Cohesion")
            ) {
                continue;
            }
            
            optimized.push_str(line);
            optimized.push('\n');
        }

        Ok(optimized)
    }

    /// Extract i32 constant value from instruction
    fn extract_i32_const(&self, instruction: &str) -> Option<i32> {
        instruction.strip_prefix("i32.const ")
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
    }

    /// Extract local variable reference from instruction
    fn extract_local_reference(&self, instruction: &str) -> Option<String> {
        if instruction.starts_with("local.get ") || 
           instruction.starts_with("local.set ") ||
           instruction.starts_with("local.tee ") {
            instruction.split_whitespace()
                .nth(1)
                .map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Extract local variable name from declaration
    fn extract_local_declaration(&self, declaration: &str) -> Option<String> {
        // Parse "(local $name type)" format
        if declaration.starts_with("(local ") {
            let parts: Vec<&str> = declaration.split_whitespace().collect();
            if parts.len() >= 2 {
                Some(parts[1].to_string())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Count instructions in WASM code
    fn count_instructions(&self, code: &str) -> usize {
        code.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with(";;") && !line.starts_with("(") && !line.starts_with(")"))
            .filter(|line| !line.starts_with("end") && !line.starts_with("else"))
            .count()
    }

    /// Count local variables in WASM code
    fn count_locals(&self, code: &str) -> usize {
        code.lines()
            .map(|line| line.trim())
            .filter(|line| line.starts_with("(local "))
            .count()
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Get optimization configuration
    pub fn get_config(&self) -> &WasmOptimizerConfig {
        &self.config
    }

    /// Generate optimization report
    pub fn generate_report(&self) -> String {
        format!(
            r#"WebAssembly Optimization Report
================================

Configuration:
- Level: {:?}
- Preserve Debug Info: {}
- Preserve Semantic Metadata: {}

Results:
- Instructions: {} -> {} ({:.1}% reduction)
- Locals: {} -> {} ({:.1}% reduction)
- Dead Code Eliminations: {}
- Constant Foldings: {}
- Instruction Combinations: {}
- Optimization Time: {}ms

Performance Impact:
- Code Size Reduction: {:.1}%
- Estimated Runtime Improvement: {:.1}%
"#,
            self.config.level,
            self.config.preserve_debug_info,
            self.config.preserve_semantic_metadata,
            self.stats.instructions_before,
            self.stats.instructions_after,
            self.stats.instruction_reduction_percent(),
            self.stats.locals_before,
            self.stats.locals_after,
            self.stats.local_reduction_percent(),
            self.stats.dead_code_eliminations,
            self.stats.constant_foldings,
            self.stats.instruction_combinations,
            self.stats.optimization_time_ms,
            self.stats.instruction_reduction_percent(),
            self.stats.instruction_reduction_percent() * 0.3 // Rough estimate
        )
    }
}

impl Default for WasmOptimizer {
    fn default() -> Self {
        Self::new(WasmOptimizerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = WasmOptimizer::default();
        assert_eq!(optimizer.config.level, WasmOptimizationLevel::Speed);
    }

    #[test]
    fn test_constant_folding() {
        let mut optimizer = WasmOptimizer::from_level(WasmOptimizationLevel::Speed);
        
        let code = r#"
    i32.const 5
    i32.const 3
    i32.add
"#;
        
        let result = optimizer.fold_constants(code).unwrap();
        assert!(result.contains("i32.const 8"));
        assert_eq!(optimizer.stats.constant_foldings, 1);
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut optimizer = WasmOptimizer::from_level(WasmOptimizationLevel::Speed);
        
        let code = r#"
    ;; TODO: implement this
    i32.const 42
    unreachable
    i32.const 1
    i32.add
"#;
        
        let result = optimizer.eliminate_dead_code(code).unwrap();
        assert!(!result.contains("TODO"));
        assert!(optimizer.stats.dead_code_eliminations > 0);
    }

    #[test]
    fn test_instruction_combining() {
        let mut optimizer = WasmOptimizer::from_level(WasmOptimizationLevel::Speed);
        
        let code = r#"
    local.set $temp
    local.get $temp
"#;
        
        let result = optimizer.combine_instructions(code).unwrap();
        assert!(result.contains("local.tee"));
        assert_eq!(optimizer.stats.instruction_combinations, 1);
    }

    #[test]
    fn test_optimization_levels() {
        let none_config = WasmOptimizerConfig::from(WasmOptimizationLevel::None);
        assert!(!none_config.dead_code_elimination);
        assert!(none_config.preserve_debug_info);

        let max_config = WasmOptimizerConfig::from(WasmOptimizationLevel::Maximum);
        assert!(max_config.dead_code_elimination);
        assert!(!max_config.preserve_debug_info);
    }

    #[test]
    fn test_i32_const_extraction() {
        let optimizer = WasmOptimizer::default();
        
        assert_eq!(optimizer.extract_i32_const("i32.const 42"), Some(42));
        assert_eq!(optimizer.extract_i32_const("i32.const -10"), Some(-10));
        assert_eq!(optimizer.extract_i32_const("i32.const 42 ;; comment"), Some(42));
        assert_eq!(optimizer.extract_i32_const("f32.const 42.0"), None);
    }

    #[test]
    fn test_stats_calculation() {
        let mut stats = OptimizationStats::default();
        stats.instructions_before = 100;
        stats.instructions_after = 80;
        stats.locals_before = 10;
        stats.locals_after = 8;

        assert_eq!(stats.instruction_reduction_percent(), 20.0);
        assert_eq!(stats.local_reduction_percent(), 20.0);
    }
} 