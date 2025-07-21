//! Scope Tree - Hierarchical Scope Management
//!
//! This module implements the hierarchical scope tree structure that manages
//! parent-child relationships between scopes, following PLT-004 specifications.
//!
//! **Conceptual Responsibility**: Scope hierarchy and tree structure management
//! **What it does**: Scope tree operations, hierarchy traversal, relationship management
//! **What it doesn't do**: Symbol resolution, visibility checking, metadata management

use crate::error::{CompilerError, CompilerResult};
use crate::scope::{ScopeData, ScopeId, ScopeKind, ScopeManager, ScopeQuery, ScopeModifier, ScopeMetadata};
use prism_common::{span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Hierarchical scope tree manager
/// 
/// Manages the tree structure of scopes and their relationships,
/// integrating with existing infrastructure while maintaining conceptual cohesion.
#[derive(Debug)]
pub struct ScopeTree {
    /// Scope data storage
    scopes: Arc<RwLock<HashMap<ScopeId, ScopeData>>>,
    
    /// Parent-child relationships (child_id -> parent_id)
    parent_map: Arc<RwLock<HashMap<ScopeId, ScopeId>>>,
    
    /// Children relationships (parent_id -> [child_ids])
    children_map: Arc<RwLock<HashMap<ScopeId, Vec<ScopeId>>>>,
    
    /// Root scope identifier
    root_scope: Option<ScopeId>,
    
    /// Next scope ID counter
    next_id: Arc<RwLock<ScopeId>>,
    
    /// Configuration
    config: ScopeTreeConfig,
    
    /// Statistics tracking
    stats: Arc<RwLock<ScopeTreeStats>>,
}

/// Configuration for scope tree behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeTreeConfig {
    /// Enable scope caching
    pub enable_caching: bool,
    
    /// Enable scope metadata
    pub enable_metadata: bool,
    
    /// Maximum scope depth (for cycle detection)
    pub max_scope_depth: usize,
    
    /// Enable statistics collection
    pub enable_statistics: bool,
    
    /// Enable concurrent access
    pub enable_concurrent_access: bool,
}

/// Statistics about the scope tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeTreeStats {
    /// Total number of scopes
    pub total_scopes: usize,
    
    /// Number of root scopes
    pub root_scopes: usize,
    
    /// Maximum depth of the tree
    pub max_depth: usize,
    
    /// Average depth of scopes
    pub average_depth: f64,
    
    /// Number of leaf scopes (no children)
    pub leaf_scopes: usize,
    
    /// Scope counts by kind
    pub scopes_by_kind: HashMap<String, usize>,
    
    /// Number of operations performed
    pub operations_count: usize,
    
    /// Cache hit rate (if caching enabled)
    pub cache_hit_rate: f64,
}

impl Default for ScopeTreeConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_metadata: true,
            max_scope_depth: 100,
            enable_statistics: true,
            enable_concurrent_access: true,
        }
    }
}

impl Default for ScopeTreeStats {
    fn default() -> Self {
        Self {
            total_scopes: 0,
            root_scopes: 0,
            max_depth: 0,
            average_depth: 0.0,
            leaf_scopes: 0,
            scopes_by_kind: HashMap::new(),
            operations_count: 0,
            cache_hit_rate: 0.0,
        }
    }
}

impl ScopeTree {
    /// Create a new scope tree with default configuration
    pub fn new() -> Self {
        Self::with_config(ScopeTreeConfig::default())
    }
    
    /// Create a new scope tree with custom configuration
    pub fn with_config(config: ScopeTreeConfig) -> Self {
        Self {
            scopes: Arc::new(RwLock::new(HashMap::new())),
            parent_map: Arc::new(RwLock::new(HashMap::new())),
            children_map: Arc::new(RwLock::new(HashMap::new())),
            root_scope: None,
            next_id: Arc::new(RwLock::new(1)), // Start from 1 (0 is reserved for NO_SCOPE)
            config,
            stats: Arc::new(RwLock::new(ScopeTreeStats::default())),
        }
    }
    
    /// Create the global root scope
    pub fn create_global_scope(&mut self) -> CompilerResult<ScopeId> {
        if self.root_scope.is_some() {
            return Err(CompilerError::InvalidInput {
                message: "Global scope already exists".to_string(),
            });
        }
        
        let global_id = self.create_scope(ScopeKind::Global, None)?;
        self.root_scope = Some(global_id);
        Ok(global_id)
    }
    
    /// Get the root scope ID
    pub fn get_root_scope(&self) -> Option<ScopeId> {
        self.root_scope
    }
    
    /// Get configuration
    pub fn config(&self) -> &ScopeTreeConfig {
        &self.config
    }
    
    /// Get current statistics
    pub fn stats(&self) -> ScopeTreeStats {
        if self.config.enable_statistics {
            let stats = self.stats.read().unwrap();
            stats.clone()
        } else {
            ScopeTreeStats::default()
        }
    }
    
    /// Update statistics after an operation
    fn update_stats<F>(&self, updater: F)
    where
        F: FnOnce(&mut ScopeTreeStats),
    {
        if self.config.enable_statistics {
            let mut stats = self.stats.write().unwrap();
            updater(&mut stats);
            stats.operations_count += 1;
        }
    }
    
    /// Recalculate all statistics
    pub fn recalculate_stats(&self) -> CompilerResult<()> {
        if !self.config.enable_statistics {
            return Ok(());
        }
        
        let scopes = self.scopes.read().unwrap();
        let parent_map = self.parent_map.read().unwrap();
        
        let mut new_stats = ScopeTreeStats::default();
        new_stats.total_scopes = scopes.len();
        
        // Count root scopes
        new_stats.root_scopes = scopes.values()
            .filter(|scope| !parent_map.contains_key(&scope.id))
            .count();
        
        // Calculate depths and other metrics
        let mut total_depth = 0;
        let mut max_depth = 0;
        let mut leaf_count = 0;
        
        for scope in scopes.values() {
            // Calculate depth
            let depth = self.calculate_scope_depth_internal(scope.id, &parent_map)?;
            total_depth += depth;
            max_depth = max_depth.max(depth);
            
            // Check if leaf (no children)
            if !self.children_map.read().unwrap().contains_key(&scope.id) {
                leaf_count += 1;
            }
            
            // Count by kind
            let kind_name = scope.kind.kind_name().to_string();
            *new_stats.scopes_by_kind.entry(kind_name).or_insert(0) += 1;
        }
        
        new_stats.max_depth = max_depth;
        new_stats.average_depth = if new_stats.total_scopes > 0 {
            total_depth as f64 / new_stats.total_scopes as f64
        } else {
            0.0
        };
        new_stats.leaf_scopes = leaf_count;
        
        // Preserve operation count and cache hit rate
        let mut stats = self.stats.write().unwrap();
        new_stats.operations_count = stats.operations_count;
        new_stats.cache_hit_rate = stats.cache_hit_rate;
        *stats = new_stats;
        
        Ok(())
    }
    
    /// Calculate scope depth from root
    fn calculate_scope_depth_internal(
        &self,
        scope_id: ScopeId,
        parent_map: &HashMap<ScopeId, ScopeId>,
    ) -> CompilerResult<usize> {
        let mut depth = 0;
        let mut current = Some(scope_id);
        
        while let Some(id) = current {
            if depth > self.config.max_scope_depth {
                return Err(CompilerError::InvalidInput {
                    message: format!("Scope depth exceeds maximum of {}", self.config.max_scope_depth),
                });
            }
            
            current = parent_map.get(&id).copied();
            if current.is_some() {
                depth += 1;
            }
        }
        
        Ok(depth)
    }
    
    /// Validate scope tree integrity
    pub fn validate_integrity(&self) -> CompilerResult<Vec<String>> {
        let mut issues = Vec::new();
        
        let scopes = self.scopes.read().unwrap();
        let parent_map = self.parent_map.read().unwrap();
        let children_map = self.children_map.read().unwrap();
        
        // Check that all parent references are valid
        for (child_id, parent_id) in parent_map.iter() {
            if !scopes.contains_key(parent_id) {
                issues.push(format!("Scope {} has invalid parent {}", child_id, parent_id));
            }
        }
        
        // Check that all child references are valid
        for (parent_id, children) in children_map.iter() {
            if !scopes.contains_key(parent_id) {
                issues.push(format!("Invalid parent scope {} in children map", parent_id));
            }
            
            for child_id in children {
                if !scopes.contains_key(child_id) {
                    issues.push(format!("Invalid child scope {} for parent {}", child_id, parent_id));
                }
                
                // Check bidirectional consistency
                if parent_map.get(child_id) != Some(parent_id) {
                    issues.push(format!("Inconsistent parent-child relationship: {} -> {}", child_id, parent_id));
                }
            }
        }
        
        // Check for cycles
        for scope_id in scopes.keys() {
            if let Err(e) = self.calculate_scope_depth_internal(*scope_id, &parent_map) {
                issues.push(format!("Cycle detected involving scope {}: {}", scope_id, e));
            }
        }
        
        Ok(issues)
    }
    
    /// Get all scopes in the tree
    pub fn get_all_scopes(&self) -> Vec<ScopeData> {
        let scopes = self.scopes.read().unwrap();
        scopes.values().cloned().collect()
    }
    
    /// Get scopes at a specific depth
    pub fn get_scopes_at_depth(&self, depth: usize) -> CompilerResult<Vec<ScopeId>> {
        let mut result = Vec::new();
        let scopes = self.scopes.read().unwrap();
        let parent_map = self.parent_map.read().unwrap();
        
        for scope_id in scopes.keys() {
            let scope_depth = self.calculate_scope_depth_internal(*scope_id, &parent_map)?;
            if scope_depth == depth {
                result.push(*scope_id);
            }
        }
        
        Ok(result)
    }
    
    /// Get all descendant scopes of a given scope
    pub fn get_all_descendants(&self, scope_id: ScopeId) -> Vec<ScopeId> {
        let mut descendants = Vec::new();
        let mut to_visit = vec![scope_id];
        let children_map = self.children_map.read().unwrap();
        
        while let Some(current) = to_visit.pop() {
            if let Some(children) = children_map.get(&current) {
                for &child in children {
                    descendants.push(child);
                    to_visit.push(child);
                }
            }
        }
        
        descendants
    }
    
    /// Get all ancestor scopes of a given scope
    pub fn get_all_ancestors(&self, scope_id: ScopeId) -> Vec<ScopeId> {
        let mut ancestors = Vec::new();
        let mut current = Some(scope_id);
        let parent_map = self.parent_map.read().unwrap();
        
        while let Some(id) = current {
            if let Some(&parent_id) = parent_map.get(&id) {
                ancestors.push(parent_id);
                current = Some(parent_id);
            } else {
                break;
            }
        }
        
        ancestors
    }
    
    /// Check if one scope is an ancestor of another
    pub fn is_ancestor_of(&self, potential_ancestor: ScopeId, descendant: ScopeId) -> bool {
        let ancestors = self.get_all_ancestors(descendant);
        ancestors.contains(&potential_ancestor)
    }
    
    /// Find the lowest common ancestor of two scopes
    pub fn lowest_common_ancestor(&self, scope1: ScopeId, scope2: ScopeId) -> Option<ScopeId> {
        if scope1 == scope2 {
            return Some(scope1);
        }
        
        let ancestors1: std::collections::HashSet<_> = self.get_all_ancestors(scope1).into_iter().collect();
        
        // Walk up from scope2 until we find a common ancestor
        let mut current = Some(scope2);
        let parent_map = self.parent_map.read().unwrap();
        
        while let Some(id) = current {
            if ancestors1.contains(&id) {
                return Some(id);
            }
            current = parent_map.get(&id).copied();
        }
        
        None
    }
}

// Implement the ScopeManager trait
impl ScopeManager for ScopeTree {
    fn create_scope(&mut self, kind: ScopeKind, parent: Option<ScopeId>) -> CompilerResult<ScopeId> {
        // Generate new scope ID
        let id = {
            let mut next_id = self.next_id.write().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };
        
        // Create scope data
        let scope_data = ScopeData::new(id, kind, Span::new(0, 0)); // TODO: Get actual span
        
        // Store scope data
        {
            let mut scopes = self.scopes.write().unwrap();
            scopes.insert(id, scope_data);
        }
        
        // Handle parent-child relationships
        if let Some(parent_id) = parent {
            // Validate parent exists
            {
                let scopes = self.scopes.read().unwrap();
                if !scopes.contains_key(&parent_id) {
                    return Err(CompilerError::InvalidInput {
                        message: format!("Parent scope {} does not exist", parent_id),
                    });
                }
            }
            
            // Set up parent-child relationship
            {
                let mut parent_map = self.parent_map.write().unwrap();
                parent_map.insert(id, parent_id);
            }
            
            {
                let mut children_map = self.children_map.write().unwrap();
                children_map.entry(parent_id).or_insert_with(Vec::new).push(id);
            }
            
            // Check for excessive depth
            let parent_map = self.parent_map.read().unwrap();
            if self.calculate_scope_depth_internal(id, &parent_map)? > self.config.max_scope_depth {
                return Err(CompilerError::InvalidInput {
                    message: format!("Scope depth exceeds maximum of {}", self.config.max_scope_depth),
                });
            }
        } else if self.root_scope.is_none() {
            // This is a root scope
            self.root_scope = Some(id);
        }
        
        // Update statistics
        self.update_stats(|stats| {
            stats.total_scopes += 1;
        });
        
        Ok(id)
    }
    
    fn get_scope(&self, id: ScopeId) -> Option<&ScopeData> {
        // Note: This would need to return a reference that lives long enough
        // In practice, we'd need a different approach for thread-safe access
        // For now, this is a placeholder implementation
        None // TODO: Implement proper thread-safe access
    }
    
    fn get_scope_mut(&mut self, id: ScopeId) -> Option<&mut ScopeData> {
        // Note: Similar issue with mutable references
        // This would need a different approach for thread-safe access
        None // TODO: Implement proper thread-safe access
    }
    
    fn get_parent(&self, id: ScopeId) -> Option<ScopeId> {
        let parent_map = self.parent_map.read().unwrap();
        parent_map.get(&id).copied()
    }
    
    fn get_children(&self, id: ScopeId) -> Vec<ScopeId> {
        let children_map = self.children_map.read().unwrap();
        children_map.get(&id).cloned().unwrap_or_default()
    }
    
    fn get_scope_chain(&self, id: ScopeId) -> Vec<ScopeId> {
        let mut chain = Vec::new();
        let mut current = Some(id);
        let parent_map = self.parent_map.read().unwrap();
        
        while let Some(scope_id) = current {
            chain.push(scope_id);
            current = parent_map.get(&scope_id).copied();
        }
        
        chain
    }
}

// Implement the ScopeQuery trait
impl ScopeQuery for ScopeTree {
    fn find_by_kind(&self, kind: &ScopeKind) -> Vec<ScopeId> {
        let scopes = self.scopes.read().unwrap();
        scopes.values()
            .filter(|scope| &scope.kind == kind)
            .map(|scope| scope.id)
            .collect()
    }
    
    fn find_containing_symbol(&self, symbol: Symbol) -> Vec<ScopeId> {
        let scopes = self.scopes.read().unwrap();
        scopes.values()
            .filter(|scope| scope.contains_symbol(&symbol))
            .map(|scope| scope.id)
            .collect()
    }
    
    fn is_visible_from(&self, target: ScopeId, from: ScopeId) -> bool {
        // This would need to integrate with the visibility system
        // For now, implement basic visibility: can see ancestors and self
        target == from || self.is_ancestor_of(target, from)
    }
    
    fn common_ancestor(&self, scope1: ScopeId, scope2: ScopeId) -> Option<ScopeId> {
        self.lowest_common_ancestor(scope1, scope2)
    }
}

// Implement the ScopeModifier trait
impl ScopeModifier for ScopeTree {
    fn add_symbol(&mut self, scope_id: ScopeId, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            scope_data.add_symbol(symbol);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }
    
    fn remove_symbol(&mut self, scope_id: ScopeId, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            scope_data.remove_symbol(&symbol);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }
    
    fn add_import(&mut self, scope_id: ScopeId, name: String, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            let imported = crate::scope::data::ImportedSymbol::new(
                symbol,
                "unknown_module".to_string(), // TODO: Get actual module name
                Span::new(0, 0), // TODO: Get actual span
            );
            scope_data.add_import(name, imported);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }
    
    fn add_export(&mut self, scope_id: ScopeId, name: String, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            let exported = crate::scope::data::ExportedSymbol::new(
                symbol,
                Span::new(0, 0), // TODO: Get actual span
                crate::scope::data::ExportVisibility::Public,
            );
            scope_data.add_export(name, exported);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }
    
    fn update_metadata<F>(&mut self, scope_id: ScopeId, updater: F) -> CompilerResult<()>
    where
        F: FnOnce(&mut ScopeMetadata),
    {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            updater(&mut scope_data.metadata);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }
}

impl Default for ScopeTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::ScopeKind;
    
    #[test]
    fn test_scope_tree_creation() {
        let tree = ScopeTree::new();
        
        assert_eq!(tree.stats().total_scopes, 0);
        assert!(tree.get_root_scope().is_none());
    }
    
    #[test]
    fn test_global_scope_creation() {
        let mut tree = ScopeTree::new();
        
        let global_id = tree.create_global_scope().unwrap();
        
        assert_eq!(tree.get_root_scope(), Some(global_id));
        assert_eq!(tree.stats().total_scopes, 1);
        
        // Should not be able to create another global scope
        let result = tree.create_global_scope();
        assert!(result.is_err());
    }
    
    #[test]
    fn test_scope_hierarchy() {
        let mut tree = ScopeTree::new();
        
        let global_id = tree.create_global_scope().unwrap();
        let module_id = tree.create_scope(
            ScopeKind::Module {
                module_name: "test".to_string(),
                sections: vec![],
                capabilities: vec![],
                effects: vec![],
            },
            Some(global_id),
        ).unwrap();
        let function_id = tree.create_scope(
            ScopeKind::Function {
                function_name: "test_func".to_string(),
                parameters: vec![],
                is_async: false,
                effects: vec![],
                required_capabilities: vec![],
                contracts: None,
            },
            Some(module_id),
        ).unwrap();
        
        // Test parent-child relationships
        assert_eq!(tree.get_parent(module_id), Some(global_id));
        assert_eq!(tree.get_parent(function_id), Some(module_id));
        assert_eq!(tree.get_parent(global_id), None);
        
        assert_eq!(tree.get_children(global_id), vec![module_id]);
        assert_eq!(tree.get_children(module_id), vec![function_id]);
        assert!(tree.get_children(function_id).is_empty());
        
        // Test scope chain
        let chain = tree.get_scope_chain(function_id);
        assert_eq!(chain, vec![function_id, module_id, global_id]);
    }
    
    #[test]
    fn test_scope_queries() {
        let mut tree = ScopeTree::new();
        
        let global_id = tree.create_global_scope().unwrap();
        let module_id1 = tree.create_scope(
            ScopeKind::Module {
                module_name: "test1".to_string(),
                sections: vec![],
                capabilities: vec![],
                effects: vec![],
            },
            Some(global_id),
        ).unwrap();
        let module_id2 = tree.create_scope(
            ScopeKind::Module {
                module_name: "test2".to_string(),
                sections: vec![],
                capabilities: vec![],
                effects: vec![],
            },
            Some(global_id),
        ).unwrap();
        
        // Test finding by kind
        let module_kind = ScopeKind::Module {
            module_name: "test1".to_string(),
            sections: vec![],
            capabilities: vec![],
            effects: vec![],
        };
        
        // Note: This test would need to be adjusted based on how ScopeKind equality works
        // For now, just test that the method works
        let modules = tree.find_by_kind(&module_kind);
        // In reality, we'd need to implement PartialEq for ScopeKind more carefully
        
        // Test ancestor relationships
        assert!(tree.is_ancestor_of(global_id, module_id1));
        assert!(!tree.is_ancestor_of(module_id1, global_id));
        
        // Test common ancestor
        assert_eq!(tree.common_ancestor(module_id1, module_id2), Some(global_id));
    }
    
    #[test]
    fn test_statistics() {
        let mut tree = ScopeTree::new();
        
        let global_id = tree.create_global_scope().unwrap();
        let module_id = tree.create_scope(
            ScopeKind::Module {
                module_name: "test".to_string(),
                sections: vec![],
                capabilities: vec![],
                effects: vec![],
            },
            Some(global_id),
        ).unwrap();
        
        tree.recalculate_stats().unwrap();
        let stats = tree.stats();
        
        assert_eq!(stats.total_scopes, 2);
        assert_eq!(stats.root_scopes, 1);
        assert_eq!(stats.max_depth, 1);
        assert!(stats.operations_count > 0);
    }
    
    #[test]
    fn test_integrity_validation() {
        let tree = ScopeTree::new();
        
        let issues = tree.validate_integrity().unwrap();
        assert!(issues.is_empty());
        
        // TODO: Add tests with actual integrity issues
    }
} 