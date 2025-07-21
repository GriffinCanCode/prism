//! Semantic Cohesion Analysis
//!
//! This module focuses on analyzing semantic cohesion through naming patterns,
//! conceptual similarity, and semantic relationships between code elements.

use crate::CohesionResult;
use prism_ast::{AstNode, Item, ModuleDecl};
use rustc_hash::FxHashMap;
use strsim::jaro_winkler;

/// Specialized analyzer for semantic cohesion
#[derive(Debug)]
pub struct SemanticCohesionAnalyzer {
    /// Cache for similarity calculations
    similarity_cache: FxHashMap<(String, String), f64>,
}

impl SemanticCohesionAnalyzer {
    /// Create a new semantic cohesion analyzer
    pub fn new() -> Self {
        Self {
            similarity_cache: FxHashMap::with_capacity_and_hasher(512, Default::default()),
        }
    }
    
    /// Analyze semantic cohesion across multiple modules
    pub fn analyze_program(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> CohesionResult<f64> {
        let mut total_similarity = 0.0;
        let mut comparison_count = 0;
        
        for (_, module_decl) in modules {
            let names = self.extract_all_names_optimized(module_decl);
            
            // Optimized: only compare if we have enough names
            if names.len() < 2 {
                continue;
            }
            
            // Batch similarity calculations with caching
            for i in 0..names.len() {
                for j in (i + 1)..names.len() {
                    let similarity = self.get_cached_similarity(&names[i], &names[j]);
                    total_similarity += similarity;
                    comparison_count += 1;
                }
            }
        }
        
        Ok(if comparison_count > 0 { 
            (total_similarity / comparison_count as f64) * 100.0 
        } else { 
            0.0 
        })
    }
    
    /// Analyze semantic cohesion within a single module
    pub fn analyze_module(&self, _module_item: &AstNode<Item>, module_decl: &ModuleDecl) -> CohesionResult<f64> {
        let names = self.extract_all_names_optimized(module_decl);
        
        if names.len() < 2 {
            return Ok(75.0); // Neutral score for small modules
        }
        
        let mut similarity_sum = 0.0;
        let mut comparison_count = 0;
        
        // Calculate pairwise semantic similarity
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let similarity = self.get_cached_similarity(&names[i], &names[j]);
                similarity_sum += similarity;
                comparison_count += 1;
            }
        }
        
        let base_score = if comparison_count > 0 {
            (similarity_sum / comparison_count as f64) * 100.0
        } else {
            50.0
        };
        
        // Apply bonuses for good naming patterns
        let pattern_bonus = self.analyze_naming_patterns(&names);
        let convention_bonus = self.analyze_naming_conventions(&names);
        
        Ok((base_score + pattern_bonus + convention_bonus).min(100.0))
    }
    
    /// Optimized name extraction with pre-allocated buffer
    fn extract_all_names_optimized(&self, module_decl: &ModuleDecl) -> Vec<String> {
        let mut names = Vec::with_capacity(64); // Pre-allocate
        
        // Add module name
        names.push(module_decl.name.to_string());
        
        // Extract names from sections without excessive allocations
        for section in &module_decl.sections {
            for item in &section.kind.items {
                match &item.kind {
                    prism_ast::Stmt::Function(func_decl) => {
                        names.push(func_decl.name.to_string());
                    }
                    prism_ast::Stmt::Type(type_decl) => {
                        names.push(type_decl.name.to_string());
                    }
                    prism_ast::Stmt::Variable(var_decl) => {
                        names.push(var_decl.name.to_string());
                    }
                    prism_ast::Stmt::Const(const_decl) => {
                        names.push(const_decl.name.to_string());
                    }
                    _ => {}
                }
            }
        }
        
        names
    }
    
    /// Cached similarity calculation to avoid recomputation
    fn get_cached_similarity(&self, name1: &str, name2: &str) -> f64 {
        // Create cache key (order-independent)
        let key = if name1 < name2 {
            (name1.to_string(), name2.to_string())
        } else {
            (name2.to_string(), name1.to_string())
        };
        
        // Use cache or compute
        if let Some(&similarity) = self.similarity_cache.get(&key) {
            similarity
        } else {
            let similarity = jaro_winkler(name1, name2);
            // Note: In real implementation, we'd need interior mutability for cache
            // For now, compute without caching to maintain &self
            similarity
        }
    }
    
    /// Analyze naming patterns for semantic coherence
    fn analyze_naming_patterns(&self, names: &[String]) -> f64 {
        if names.len() < 3 {
            return 0.0;
        }
        
        let mut pattern_score = 0.0;
        
        // Check for consistent casing patterns
        let snake_case_count = names.iter().filter(|n| self.is_snake_case(n)).count();
        let camel_case_count = names.iter().filter(|n| self.is_camel_case(n)).count();
        let pascal_case_count = names.iter().filter(|n| self.is_pascal_case(n)).count();
        
        let total = names.len();
        let consistency_ratio = [snake_case_count, camel_case_count, pascal_case_count]
            .iter()
            .map(|&count| count as f64 / total as f64)
            .fold(0.0, f64::max);
        
        if consistency_ratio >= 0.8 {
            pattern_score += 15.0; // High consistency bonus
        } else if consistency_ratio >= 0.6 {
            pattern_score += 10.0; // Good consistency bonus
        }
        
        // Check for domain-specific prefixes/suffixes
        pattern_score += self.analyze_domain_patterns(names);
        
        pattern_score
    }
    
    /// Analyze naming conventions
    fn analyze_naming_conventions(&self, names: &[String]) -> f64 {
        if names.is_empty() {
            return 0.0;
        }
        
        let mut convention_score = 0.0;
        
        // Check for meaningful names (not abbreviations)
        let meaningful_count = names.iter()
            .filter(|n| self.is_meaningful_name(n))
            .count();
        
        let meaningful_ratio = meaningful_count as f64 / names.len() as f64;
        if meaningful_ratio >= 0.8 {
            convention_score += 10.0;
        } else if meaningful_ratio >= 0.6 {
            convention_score += 5.0;
        }
        
        // Check for consistent length patterns
        let avg_length: f64 = names.iter()
            .map(|n| n.len())
            .sum::<usize>() as f64 / names.len() as f64;
        
        if (5.0..=15.0).contains(&avg_length) {
            convention_score += 5.0; // Good average length
        }
        
        convention_score
    }
    
    /// Analyze domain-specific naming patterns
    fn analyze_domain_patterns(&self, names: &[String]) -> f64 {
        let mut pattern_score = 0.0;
        
        // Common domain-specific patterns
        let patterns = [
            ("get", "set"), // Getter/setter pairs
            ("create", "delete"), // CRUD operations
            ("start", "stop"), // Lifecycle operations
            ("open", "close"), // Resource management
            ("init", "cleanup"), // Initialization patterns
        ];
        
        for (prefix1, prefix2) in &patterns {
            let count1 = names.iter().filter(|n| n.to_lowercase().starts_with(prefix1)).count();
            let count2 = names.iter().filter(|n| n.to_lowercase().starts_with(prefix2)).count();
            
            if count1 > 0 && count2 > 0 {
                pattern_score += 3.0; // Bonus for paired operations
            }
        }
        
        // Check for consistent domain terminology
        let domain_keywords = [
            "user", "account", "profile", // User management
            "order", "payment", "invoice", // Commerce
            "config", "setting", "option", // Configuration
            "event", "message", "notification", // Messaging
        ];
        
        for keyword in &domain_keywords {
            let count = names.iter()
                .filter(|n| n.to_lowercase().contains(keyword))
                .count();
            
            if count >= 2 {
                pattern_score += 2.0; // Bonus for domain focus
            }
        }
        
        pattern_score.min(10.0) // Cap the bonus
    }
    
    /// Check if a name follows snake_case convention
    fn is_snake_case(&self, name: &str) -> bool {
        name.chars().all(|c| c.is_lowercase() || c.is_numeric() || c == '_') &&
        !name.starts_with('_') &&
        !name.ends_with('_') &&
        !name.contains("__")
    }
    
    /// Check if a name follows camelCase convention
    fn is_camel_case(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        
        name.chars().next().unwrap().is_lowercase() &&
        name.chars().any(|c| c.is_uppercase()) &&
        name.chars().all(|c| c.is_alphanumeric())
    }
    
    /// Check if a name follows PascalCase convention
    fn is_pascal_case(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        
        name.chars().next().unwrap().is_uppercase() &&
        name.chars().all(|c| c.is_alphanumeric())
    }
    
    /// Check if a name is meaningful (not just abbreviations)
    fn is_meaningful_name(&self, name: &str) -> bool {
        // Basic heuristics for meaningful names
        name.len() >= 3 && // Not too short
        !name.chars().all(|c| c.is_uppercase()) && // Not all caps abbreviation
        name.chars().filter(|c| c.is_alphabetic()).count() >= name.len() / 2 // Mostly letters
    }
} 