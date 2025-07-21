//! Dependency Cohesion Analysis
//!
//! This module focuses on analyzing dependency cohesion by examining
//! module dependencies, coupling patterns, and dependency health.

use crate::CohesionResult;
use prism_ast::{AstNode, Item, ModuleDecl};

/// Specialized analyzer for dependency cohesion
#[derive(Debug)]
pub struct DependencyCohesionAnalyzer {
    /// Thresholds for dependency analysis
    dependency_thresholds: DependencyThresholds,
}

impl DependencyCohesionAnalyzer {
    /// Create a new dependency cohesion analyzer
    pub fn new() -> Self {
        Self {
            dependency_thresholds: DependencyThresholds::default(),
        }
    }
    
    /// Analyze dependency cohesion across multiple modules
    pub fn analyze_program(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> CohesionResult<f64> {
        let mut total_score = 0.0;
        let mut module_count = 0;
        
        for (_, module_decl) in modules {
            let dependency_score = self.calculate_dependency_score(module_decl);
            total_score += dependency_score;
            module_count += 1;
        }
        
        Ok(if module_count > 0 { total_score / module_count as f64 } else { 0.0 })
    }
    
    /// Analyze dependency cohesion within a single module
    pub fn analyze_module(&self, _module_item: &AstNode<Item>, module_decl: &ModuleDecl) -> CohesionResult<f64> {
        Ok(self.calculate_dependency_score(module_decl))
    }
    
    /// Calculate dependency score for a module
    fn calculate_dependency_score(&self, module_decl: &ModuleDecl) -> f64 {
        let dependency_count = module_decl.dependencies.len();
        
        // Base score calculation using logarithmic penalty
        let base_score = match dependency_count {
            0 => 100.0, // Perfect cohesion with no external dependencies
            1..=3 => 95.0, // Excellent - minimal dependencies
            4..=6 => 85.0, // Good - reasonable dependencies
            7..=10 => 70.0, // Acceptable - moderate dependencies
            11..=15 => 55.0, // Concerning - many dependencies
            _ => 40.0 - (dependency_count as f64 * 1.5).min(30.0) // Poor - too many dependencies
        };
        
        // Analyze dependency quality
        let quality_modifier = self.analyze_dependency_quality(module_decl);
        
        // Analyze dependency patterns
        let pattern_modifier = self.analyze_dependency_patterns(module_decl);
        
        // Combine scores
        (base_score + quality_modifier + pattern_modifier).clamp(0.0, 100.0)
    }
    
    /// Analyze the quality of dependencies
    fn analyze_dependency_quality(&self, module_decl: &ModuleDecl) -> f64 {
        if module_decl.dependencies.is_empty() {
            return 0.0; // No dependencies to analyze
        }
        
        let mut quality_score = 0.0;
        let mut well_named_deps = 0;
        let mut standard_deps = 0;
        
        for dependency in &module_decl.dependencies {
            let dep_name = dependency.name.to_string().to_lowercase();
            
            // Check for well-named dependencies (meaningful names)
            if self.is_well_named_dependency(&dep_name) {
                well_named_deps += 1;
            }
            
            // Check for standard/common dependencies
            if self.is_standard_dependency(&dep_name) {
                standard_deps += 1;
            }
        }
        
        let total_deps = module_decl.dependencies.len();
        
        // Calculate quality bonuses
        let naming_ratio = well_named_deps as f64 / total_deps as f64;
        if naming_ratio >= 0.8 {
            quality_score += 5.0; // Most dependencies are well-named
        } else if naming_ratio >= 0.6 {
            quality_score += 3.0; // Good naming
        }
        
        let standard_ratio = standard_deps as f64 / total_deps as f64;
        if standard_ratio >= 0.5 {
            quality_score += 3.0; // Many standard dependencies
        }
        
        // Penalty for too many non-standard dependencies
        let non_standard_ratio = 1.0 - standard_ratio;
        if non_standard_ratio >= 0.7 {
            quality_score -= 5.0; // Too many non-standard dependencies
        }
        
        quality_score
    }
    
    /// Analyze dependency patterns for cohesion insights
    fn analyze_dependency_patterns(&self, module_decl: &ModuleDecl) -> f64 {
        if module_decl.dependencies.is_empty() {
            return 0.0;
        }
        
        let mut pattern_score = 0.0;
        
        // Group dependencies by category
        let mut categories = std::collections::HashMap::new();
        for dependency in &module_decl.dependencies {
            let category = self.categorize_dependency(&dependency.name.to_string());
            *categories.entry(category).or_insert(0) += 1;
        }
        
        // Analyze category distribution
        let category_count = categories.len();
        let total_deps = module_decl.dependencies.len();
        
        // Prefer focused dependency usage (fewer categories)
        match category_count {
            1 => pattern_score += 8.0, // Very focused
            2 => pattern_score += 6.0, // Well focused
            3 => pattern_score += 4.0, // Reasonably focused
            4 => pattern_score += 2.0, // Moderately focused
            _ => pattern_score -= 2.0, // Unfocused
        }
        
        // Check for balanced category usage
        let max_category_size = categories.values().max().unwrap_or(&0);
        let dominance_ratio = *max_category_size as f64 / total_deps as f64;
        
        if dominance_ratio >= 0.8 {
            pattern_score += 3.0; // Strong focus on one category
        } else if dominance_ratio >= 0.6 {
            pattern_score += 2.0; // Good focus
        }
        
        // Check for common dependency patterns
        pattern_score += self.detect_dependency_patterns(&categories);
        
        pattern_score
    }
    
    /// Check if a dependency has a meaningful name
    fn is_well_named_dependency(&self, dep_name: &str) -> bool {
        // Heuristics for well-named dependencies
        dep_name.len() >= 3 && // Not too short
        !dep_name.chars().all(|c| c.is_uppercase()) && // Not all caps
        dep_name.chars().any(|c| c.is_alphabetic()) && // Contains letters
        !dep_name.starts_with("tmp") && // Not temporary
        !dep_name.starts_with("temp") &&
        !dep_name.contains("123") // Not test/placeholder names
    }
    
    /// Check if a dependency is a standard/common library
    fn is_standard_dependency(&self, dep_name: &str) -> bool {
        // Common standard dependencies
        let standard_deps = [
            "std", "core", "alloc", // Rust standard
            "serde", "tokio", "async", // Common Rust crates
            "log", "env", "config", "clap", // Utilities
            "http", "url", "uuid", "chrono", // Common types
            "prism", // Our own framework
        ];
        
        standard_deps.iter().any(|&std_dep| dep_name.contains(std_dep))
    }
    
    /// Categorize a dependency by its apparent purpose
    fn categorize_dependency(&self, dep_name: &str) -> DependencyCategory {
        let name_lower = dep_name.to_lowercase();
        
        if name_lower.contains("prism") {
            DependencyCategory::Framework
        } else if name_lower.contains("serde") || name_lower.contains("json") || name_lower.contains("serialize") {
            DependencyCategory::Serialization
        } else if name_lower.contains("async") || name_lower.contains("tokio") || name_lower.contains("future") {
            DependencyCategory::Async
        } else if name_lower.contains("http") || name_lower.contains("web") || name_lower.contains("client") {
            DependencyCategory::Network
        } else if name_lower.contains("log") || name_lower.contains("trace") || name_lower.contains("debug") {
            DependencyCategory::Logging
        } else if name_lower.contains("test") || name_lower.contains("mock") || name_lower.contains("assert") {
            DependencyCategory::Testing
        } else if name_lower.contains("config") || name_lower.contains("env") || name_lower.contains("setting") {
            DependencyCategory::Configuration
        } else if name_lower.contains("error") || name_lower.contains("result") || name_lower.contains("thiserror") {
            DependencyCategory::ErrorHandling
        } else if name_lower.contains("std") || name_lower.contains("core") || name_lower.contains("alloc") {
            DependencyCategory::Standard
        } else {
            DependencyCategory::Other
        }
    }
    
    /// Detect common dependency patterns
    fn detect_dependency_patterns(&self, categories: &std::collections::HashMap<DependencyCategory, usize>) -> f64 {
        let mut pattern_score = 0.0;
        
        // Good patterns
        if categories.contains_key(&DependencyCategory::Framework) && 
           categories.contains_key(&DependencyCategory::Standard) {
            pattern_score += 2.0; // Framework + standard is good
        }
        
        if categories.contains_key(&DependencyCategory::ErrorHandling) {
            pattern_score += 1.0; // Error handling is good practice
        }
        
        if categories.contains_key(&DependencyCategory::Logging) {
            pattern_score += 1.0; // Logging is good practice
        }
        
        // Concerning patterns
        if categories.get(&DependencyCategory::Other).unwrap_or(&0) > &3 {
            pattern_score -= 2.0; // Too many unknown dependencies
        }
        
        if categories.contains_key(&DependencyCategory::Testing) && categories.len() == 1 {
            pattern_score -= 1.0; // Only testing dependencies might indicate incomplete module
        }
        
        pattern_score
    }
}

/// Thresholds for dependency analysis
#[derive(Debug)]
struct DependencyThresholds {
    /// Maximum dependencies before penalty
    max_dependencies: usize,
    /// Ideal dependency count
    ideal_dependencies: usize,
    /// Minimum dependencies for bonus
    min_dependencies_bonus: usize,
}

impl Default for DependencyThresholds {
    fn default() -> Self {
        Self {
            max_dependencies: 10,
            ideal_dependencies: 5,
            min_dependencies_bonus: 2,
        }
    }
}

/// Categories of dependencies for analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DependencyCategory {
    /// Framework dependencies (like prism-*)
    Framework,
    /// Standard library dependencies
    Standard,
    /// Serialization dependencies
    Serialization,
    /// Async/concurrency dependencies
    Async,
    /// Network/HTTP dependencies
    Network,
    /// Logging dependencies
    Logging,
    /// Testing dependencies
    Testing,
    /// Configuration dependencies
    Configuration,
    /// Error handling dependencies
    ErrorHandling,
    /// Other/unknown dependencies
    Other,
} 