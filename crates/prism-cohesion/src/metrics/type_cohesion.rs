//! Type Cohesion Analysis
//!
//! This module focuses exclusively on analyzing type cohesion within modules.
//! It examines type relationships, naming patterns, structural organization,
//! and semantic coherence of type definitions.

use crate::CohesionResult;
use prism_ast::{AstNode, Item, ModuleDecl};
use rustc_hash::{FxHashMap, FxHashSet};
use strsim::jaro_winkler;

/// Specialized analyzer for type cohesion
#[derive(Debug)]
pub struct TypeCohesionAnalyzer {
    /// Cache for type information
    type_cache: FxHashMap<String, TypeInfo>,
}

impl TypeCohesionAnalyzer {
    /// Create a new type cohesion analyzer
    pub fn new() -> Self {
        Self {
            type_cache: FxHashMap::with_capacity_and_hasher(256, Default::default()),
        }
    }
    
    /// Analyze type cohesion across multiple modules
    pub fn analyze_program(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> CohesionResult<f64> {
        let mut total_cohesion = 0.0;
        let mut module_count = 0;
        
        for (_, module_decl) in modules {
            let module_cohesion = self.analyze_module_type_cohesion(module_decl)?;
            total_cohesion += module_cohesion;
            module_count += 1;
        }
        
        Ok(if module_count > 0 { total_cohesion / module_count as f64 } else { 0.0 })
    }
    
    /// Analyze type cohesion within a single module
    pub fn analyze_module(&self, _module_item: &AstNode<Item>, module_decl: &ModuleDecl) -> CohesionResult<f64> {
        self.analyze_module_type_cohesion(module_decl)
    }
    
    /// Analyze type cohesion within a single module (core implementation)
    fn analyze_module_type_cohesion(&self, module_decl: &ModuleDecl) -> CohesionResult<f64> {
        let type_section = module_decl.sections.iter()
            .find(|s| matches!(s.kind.kind, prism_ast::SectionKind::Types));
        
        let Some(type_section) = type_section else {
            // No types section = moderate cohesion (50.0)
            return Ok(50.0);
        };
        
        // Fast path for small type sections
        if type_section.kind.items.len() <= 3 {
            return Ok(90.0);
        }
        
        // Extract types and analyze relationships
        let types = self.extract_types_from_section(type_section);
        
        if types.is_empty() {
            return Ok(50.0);
        }
        
        // Calculate different aspects of type cohesion
        let naming_cohesion = self.calculate_type_naming_cohesion(&types);
        let structural_cohesion = self.calculate_structural_cohesion(&types);
        let semantic_cohesion = self.calculate_type_semantic_cohesion(&types);
        let relationship_cohesion = self.calculate_type_relationship_cohesion(&types);
        
        // Weighted combination of cohesion aspects
        let overall_cohesion = naming_cohesion * 0.25 +
            structural_cohesion * 0.35 +
            semantic_cohesion * 0.25 +
            relationship_cohesion * 0.15;
        
        Ok(overall_cohesion)
    }
    
    /// Extract type information from a types section
    fn extract_types_from_section(&self, section: &AstNode<prism_ast::SectionDecl>) -> Vec<TypeInfo> {
        let mut types = Vec::with_capacity(section.kind.items.len());
        
        for item in &section.kind.items {
            if let prism_ast::Stmt::Type(type_decl) = &item.kind {
                let type_info = TypeInfo {
                    name: type_decl.name.to_string(),
                    kind: self.classify_type_kind(&type_decl.kind),
                    complexity: self.estimate_type_complexity(&type_decl.kind),
                    relationships: self.extract_type_relationships(&type_decl.kind),
                    semantic_domain: self.infer_semantic_domain(&type_decl.name.to_string()),
                    is_public: matches!(type_decl.visibility, prism_ast::Visibility::Public),
                };
                types.push(type_info);
            }
        }
        
        types
    }
    
    /// Calculate cohesion based on type naming patterns
    fn calculate_type_naming_cohesion(&self, types: &[TypeInfo]) -> f64 {
        if types.len() < 2 {
            return 100.0;
        }
        
        let mut similarity_sum = 0.0;
        let mut comparison_count = 0;
        
        // Analyze naming patterns
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                let similarity = jaro_winkler(&types[i].name, &types[j].name);
                
                // Boost similarity if types are in the same semantic domain
                let domain_boost = if types[i].semantic_domain == types[j].semantic_domain {
                    0.2
                } else {
                    0.0
                };
                
                similarity_sum += (similarity + domain_boost).min(1.0);
                comparison_count += 1;
            }
        }
        
        let base_score = if comparison_count > 0 {
            (similarity_sum / comparison_count as f64) * 100.0
        } else {
            50.0
        };
        
        // Apply naming convention bonuses
        let convention_bonus = self.analyze_naming_conventions(types);
        
        (base_score + convention_bonus).min(100.0)
    }
    
    /// Calculate structural cohesion based on type organization
    fn calculate_structural_cohesion(&self, types: &[TypeInfo]) -> f64 {
        let mut score = 70.0; // Base structural score
        
        // Analyze type complexity distribution
        let complexity_variance = self.calculate_complexity_variance(types);
        if complexity_variance < 2.0 {
            score += 15.0; // Bonus for consistent complexity
        } else if complexity_variance > 5.0 {
            score -= 20.0; // Penalty for high variance
        }
        
        // Analyze type kind distribution
        let kind_distribution = self.analyze_type_kind_distribution(types);
        score += self.score_type_distribution(&kind_distribution);
        
        // Check for balanced public/private mix
        let public_ratio = types.iter().filter(|t| t.is_public).count() as f64 / types.len() as f64;
        if (0.2..=0.8).contains(&public_ratio) {
            score += 10.0; // Good public/private balance
        }
        
        score.clamp(0.0, 100.0)
    }
    
    /// Calculate semantic cohesion based on business domains
    fn calculate_type_semantic_cohesion(&self, types: &[TypeInfo]) -> f64 {
        if types.len() < 2 {
            return 90.0;
        }
        
        // Group types by semantic domain
        let mut domain_groups: FxHashMap<String, usize> = FxHashMap::default();
        for type_info in types {
            *domain_groups.entry(type_info.semantic_domain.clone()).or_insert(0) += 1;
        }
        
        // Calculate domain concentration (higher = more cohesive)
        let total_types = types.len();
        let domain_count = domain_groups.len();
        
        // Ideal case: 1-2 domains for most types
        let domain_score = match domain_count {
            1 => 100.0, // Perfect cohesion - single domain
            2 => 85.0,  // Good cohesion - two related domains
            3 => 70.0,  // Acceptable cohesion
            4 => 55.0,  // Moderate cohesion
            _ => 40.0,  // Low cohesion - too many domains
        };
        
        // Check for dominant domain (80%+ of types)
        let max_domain_size = domain_groups.values().max().unwrap_or(&0);
        let dominance_ratio = *max_domain_size as f64 / total_types as f64;
        
        let dominance_bonus = if dominance_ratio >= 0.8 {
            15.0
        } else if dominance_ratio >= 0.6 {
            10.0
        } else if dominance_ratio >= 0.4 {
            5.0
        } else {
            0.0
        };
        
        (domain_score as f64 + dominance_bonus as f64).min(100.0_f64)
    }
    
    /// Calculate relationship cohesion based on type interdependencies
    fn calculate_type_relationship_cohesion(&self, types: &[TypeInfo]) -> f64 {
        if types.len() < 2 {
            return 90.0;
        }
        
        let mut internal_references = 0;
        let mut external_references = 0;
        let type_names: FxHashSet<String> = types.iter().map(|t| t.name.clone()).collect();
        
        // Count internal vs external type references
        for type_info in types {
            for relationship in &type_info.relationships {
                if type_names.contains(&relationship.target_type) {
                    internal_references += 1;
                } else {
                    external_references += 1;
                }
            }
        }
        
        let total_references = internal_references + external_references;
        if total_references == 0 {
            return 70.0; // Neutral score for isolated types
        }
        
        // High internal reference ratio = high cohesion
        let internal_ratio = internal_references as f64 / total_references as f64;
        let base_score = internal_ratio * 80.0 + 20.0; // 20-100 range
        
        // Bonus for reasonable number of relationships
        let relationship_density = total_references as f64 / types.len() as f64;
        let density_bonus = if (1.0..=3.0).contains(&relationship_density) {
            10.0 // Good relationship density
        } else {
            0.0
        };
        
        (base_score + density_bonus).min(100.0)
    }
    
    // HELPER METHODS
    
    /// Analyze naming conventions across types
    fn analyze_naming_conventions(&self, types: &[TypeInfo]) -> f64 {
        let mut convention_score = 0.0;
        
        // Check for consistent casing
        let pascal_case_count = types.iter()
            .filter(|t| self.is_pascal_case(&t.name))
            .count();
        
        if pascal_case_count == types.len() {
            convention_score += 15.0; // All types follow PascalCase
        } else if pascal_case_count >= types.len() * 3 / 4 {
            convention_score += 10.0; // Most types follow convention
        }
        
        // Check for domain prefixes/suffixes
        let common_prefixes = self.find_common_prefixes(types);
        let common_suffixes = self.find_common_suffixes(types);
        
        if !common_prefixes.is_empty() || !common_suffixes.is_empty() {
            convention_score += 10.0; // Consistent naming patterns
        }
        
        convention_score
    }
    
    /// Calculate variance in type complexity
    fn calculate_complexity_variance(&self, types: &[TypeInfo]) -> f64 {
        if types.len() < 2 {
            return 0.0;
        }
        
        let mean_complexity: f64 = types.iter()
            .map(|t| t.complexity)
            .sum::<f64>() / types.len() as f64;
        
        let variance: f64 = types.iter()
            .map(|t| (t.complexity - mean_complexity).powi(2))
            .sum::<f64>() / types.len() as f64;
        
        variance.sqrt()
    }
    
    /// Analyze distribution of type kinds
    fn analyze_type_kind_distribution(&self, types: &[TypeInfo]) -> FxHashMap<TypeKind, usize> {
        let mut distribution = FxHashMap::default();
        
        for type_info in types {
            *distribution.entry(type_info.kind.clone()).or_insert(0) += 1;
        }
        
        distribution
    }
    
    /// Score type distribution for structural cohesion
    fn score_type_distribution(&self, distribution: &FxHashMap<TypeKind, usize>) -> f64 {
        let total_types: usize = distribution.values().sum();
        if total_types == 0 {
            return 0.0;
        }
        
        // Prefer modules with focused type usage
        match distribution.len() {
            1 => 20.0, // Single type kind - very focused
            2 => 15.0, // Two type kinds - good focus
            3 => 10.0, // Three type kinds - acceptable
            4 => 5.0,  // Four type kinds - moderate
            _ => 0.0,  // Too many type kinds - unfocused
        }
    }
    
    /// Check if a name follows PascalCase convention
    fn is_pascal_case(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        
        name.chars().next().unwrap().is_uppercase() &&
        name.chars().all(|c| c.is_alphanumeric() || c == '_') &&
        !name.contains("__") // No double underscores
    }
    
    /// Find common prefixes in type names
    fn find_common_prefixes(&self, types: &[TypeInfo]) -> Vec<String> {
        if types.len() < 2 {
            return Vec::new();
        }
        
        let mut prefixes = Vec::new();
        
        // Simple prefix detection (first 3-5 characters)
        for prefix_len in 3..=5 {
            let mut prefix_counts: FxHashMap<String, usize> = FxHashMap::default();
            
            for type_info in types {
                if type_info.name.len() >= prefix_len {
                    let prefix = type_info.name[..prefix_len].to_string();
                    *prefix_counts.entry(prefix).or_insert(0) += 1;
                }
            }
            
            // If >50% of types share a prefix, it's significant
            for (prefix, count) in prefix_counts {
                if count >= types.len() / 2 {
                    prefixes.push(prefix);
                }
            }
        }
        
        prefixes
    }
    
    /// Find common suffixes in type names
    fn find_common_suffixes(&self, types: &[TypeInfo]) -> Vec<String> {
        if types.len() < 2 {
            return Vec::new();
        }
        
        let mut suffixes = Vec::new();
        
        // Common suffixes to look for
        let common_suffixes = ["Type", "Data", "Info", "Config", "Event", "Request", "Response"];
        
        for suffix in &common_suffixes {
            let count = types.iter()
                .filter(|t| t.name.ends_with(suffix))
                .count();
            
            if count >= types.len() / 2 {
                suffixes.push(suffix.to_string());
            }
        }
        
        suffixes
    }
    
    /// Classify the kind of a type declaration
    fn classify_type_kind(&self, type_kind: &prism_ast::TypeKind) -> TypeKind {
        match type_kind {
            prism_ast::TypeKind::Struct(_) => TypeKind::Struct,
            prism_ast::TypeKind::Enum(_) => TypeKind::Enum,
            prism_ast::TypeKind::Trait(_) => TypeKind::Trait,
            prism_ast::TypeKind::Alias(_) => TypeKind::Alias,
            prism_ast::TypeKind::Semantic(_) => TypeKind::Semantic,
        }
    }
    
    /// Estimate the complexity of a type declaration
    fn estimate_type_complexity(&self, type_kind: &prism_ast::TypeKind) -> f64 {
        match type_kind {
            prism_ast::TypeKind::Alias(_) => 1.0,
            prism_ast::TypeKind::Enum(enum_decl) => {
                1.5 + (enum_decl.variants.len() as f64 * 0.3)
            }
            prism_ast::TypeKind::Struct(struct_decl) => {
                2.0 + (struct_decl.fields.len() as f64 * 0.4)
            }
            prism_ast::TypeKind::Trait(trait_decl) => {
                2.5 + (trait_decl.methods.len() as f64 * 0.5)
            }
            prism_ast::TypeKind::Semantic(_) => 3.0, // Semantic types are inherently complex
        }
    }
    
    /// Extract type relationships from a type declaration
    fn extract_type_relationships(&self, type_kind: &prism_ast::TypeKind) -> Vec<TypeRelationship> {
        let mut relationships = Vec::new();
        
        match type_kind {
            prism_ast::TypeKind::Struct(struct_decl) => {
                for field in &struct_decl.fields {
                    if let Some(target_type) = self.extract_type_name_from_ast(&field.field_type) {
                        relationships.push(TypeRelationship {
                            relationship_type: RelationshipType::Contains,
                            target_type,
                            strength: 0.9,
                        });
                    }
                }
            }
            prism_ast::TypeKind::Enum(enum_decl) => {
                for variant in &enum_decl.variants {
                    for field_type in &variant.fields {
                        if let Some(target_type) = self.extract_type_name_from_ast(field_type) {
                            relationships.push(TypeRelationship {
                                relationship_type: RelationshipType::Contains,
                                target_type,
                                strength: 0.8,
                            });
                        }
                    }
                }
            }
            prism_ast::TypeKind::Trait(trait_decl) => {
                for method in &trait_decl.methods {
                    if let Some(target_type) = self.extract_type_name_from_ast(&method.method_type.return_type) {
                        relationships.push(TypeRelationship {
                            relationship_type: RelationshipType::Returns,
                            target_type,
                            strength: 0.7,
                        });
                    }
                }
            }
            prism_ast::TypeKind::Alias(alias_type) => {
                if let Some(target_type) = self.extract_type_name_from_ast(alias_type) {
                    relationships.push(TypeRelationship {
                        relationship_type: RelationshipType::AliasOf,
                        target_type,
                        strength: 1.0,
                    });
                }
            }
            prism_ast::TypeKind::Semantic(_) => {
                // Semantic types would need deeper analysis
                // For now, assume they're self-contained
            }
        }
        
        relationships
    }
    
    /// Extract type name from AST type node (simplified)
    fn extract_type_name_from_ast(&self, type_node: &AstNode<prism_ast::Type>) -> Option<String> {
        match &type_node.kind {
            prism_ast::Type::Named(named) => Some(named.name.to_string()),
            prism_ast::Type::Primitive(prim) => Some(format!("{:?}", prim)),
            _ => None, // For complex types, would need more sophisticated extraction
        }
    }
    
    /// Infer semantic domain from type name
    fn infer_semantic_domain(&self, type_name: &str) -> String {
        let name_lower = type_name.to_lowercase();
        
        // Domain classification based on naming patterns
        if name_lower.contains("user") || name_lower.contains("account") || name_lower.contains("profile") {
            "user_management".to_string()
        } else if name_lower.contains("order") || name_lower.contains("payment") || name_lower.contains("transaction") {
            "commerce".to_string()
        } else if name_lower.contains("config") || name_lower.contains("setting") || name_lower.contains("option") {
            "configuration".to_string()
        } else if name_lower.contains("event") || name_lower.contains("message") || name_lower.contains("notification") {
            "messaging".to_string()
        } else if name_lower.contains("data") || name_lower.contains("record") || name_lower.contains("entity") {
            "data_model".to_string()
        } else if name_lower.contains("error") || name_lower.contains("exception") || name_lower.contains("result") {
            "error_handling".to_string()
        } else {
            "general".to_string()
        }
    }
}

/// Information about a type for cohesion analysis
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type classification
    pub kind: TypeKind,
    /// Estimated complexity score
    pub complexity: f64,
    /// Relationships to other types
    pub relationships: Vec<TypeRelationship>,
    /// Inferred semantic domain
    pub semantic_domain: String,
    /// Whether the type is public
    pub is_public: bool,
}

/// Classification of type kinds
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeKind {
    /// Struct type
    Struct,
    /// Enum type
    Enum,
    /// Trait type
    Trait,
    /// Type alias
    Alias,
    /// Semantic type
    Semantic,
}

/// Relationship between types
#[derive(Debug, Clone)]
pub struct TypeRelationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Target type name
    pub target_type: String,
    /// Strength of relationship (0.0 to 1.0)
    pub strength: f64,
}

/// Types of relationships between types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationshipType {
    /// Type contains another type (composition)
    Contains,
    /// Type is an alias of another type
    AliasOf,
    /// Type returns another type
    Returns,
    /// Type inherits from another type
    InheritsFrom,
    /// Type implements another type
    Implements,
} 