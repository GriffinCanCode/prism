//! Scope Management Subsystem
//!
//! This subsystem implements hierarchical scope management for the Prism compiler,
//! following PLT-004 specifications. It provides:
//!
//! - **Hierarchical scope trees** with parent-child relationships
//! - **Semantic scope classification** (Module, Function, Block, etc.)
//! - **Symbol containment** (references to symbols, not storage)
//! - **Import/export tracking** for module system integration
//! - **AI-comprehensible metadata** for external tool integration
//! - **Effect and capability boundaries** for security integration
//!
//! ## Conceptual Cohesion
//!
//! This subsystem embodies the single concept of "Scope Hierarchy Management".
//! It does NOT handle:
//! - Symbol storage (delegated to symbol_table)
//! - Symbol resolution (delegated to resolution)
//! - Type inference (delegated to semantic analysis)
//! - Runtime scopes (delegated to runtime system)
//!
//! ## Architecture
//!
//! ```
//! scope/
//! ├── mod.rs           # Public API and re-exports
//! ├── tree.rs          # ScopeTree - hierarchical structure
//! ├── data.rs          # ScopeData - scope information
//! ├── kinds.rs         # ScopeKind - scope classification
//! ├── visibility.rs    # Visibility rules and access control
//! ├── metadata.rs      # AI metadata and semantic context
//! ├── boundaries.rs    # Effect and capability boundaries
//! └── builder.rs       # ScopeBuilder - construction utilities
//! ```

// Core scope management
pub mod tree;
pub mod data;
pub mod kinds;
pub mod visibility;
pub mod metadata;
pub mod boundaries;
pub mod builder;

// Re-export main types for convenience
pub use tree::{ScopeTree, ScopeTreeConfig, ScopeTreeStats};
pub use data::{ScopeData, ScopeId};
pub use kinds::{ScopeKind, SectionType, BlockType, ControlFlowType};
pub use visibility::{ScopeVisibility, VisibilityRule, AccessLevel};
pub use metadata::{ScopeMetadata, AIScopeContext, ScopeDocumentation};
pub use boundaries::{EffectBoundary, CapabilityBoundary, SecurityBoundary};
pub use builder::{ScopeBuilder, ScopeBuilderConfig};

// Type aliases for external use
pub type ScopeRef = std::sync::Arc<ScopeData>;
pub type ScopeWeakRef = std::sync::Weak<ScopeData>;

/// Main scope management interface
pub trait ScopeManager {
    /// Create a new scope with the given kind and parent
    fn create_scope(&mut self, kind: ScopeKind, parent: Option<ScopeId>) -> crate::error::CompilerResult<ScopeId>;
    
    /// Get scope data by ID
    fn get_scope(&self, id: ScopeId) -> Option<&ScopeData>;
    
    /// Get mutable scope data by ID
    fn get_scope_mut(&mut self, id: ScopeId) -> Option<&mut ScopeData>;
    
    /// Get the parent scope of a given scope
    fn get_parent(&self, id: ScopeId) -> Option<ScopeId>;
    
    /// Get the children scopes of a given scope
    fn get_children(&self, id: ScopeId) -> Vec<ScopeId>;
    
    /// Get the scope chain from a scope to the root
    fn get_scope_chain(&self, id: ScopeId) -> Vec<ScopeId>;
}

/// Scope query interface for advanced operations
pub trait ScopeQuery {
    /// Find scopes by kind
    fn find_by_kind(&self, kind: &ScopeKind) -> Vec<ScopeId>;
    
    /// Find scopes containing a specific symbol
    fn find_containing_symbol(&self, symbol: prism_common::symbol::Symbol) -> Vec<ScopeId>;
    
    /// Check if a scope is visible from another scope
    fn is_visible_from(&self, target: ScopeId, from: ScopeId) -> bool;
    
    /// Get the common ancestor of two scopes
    fn common_ancestor(&self, scope1: ScopeId, scope2: ScopeId) -> Option<ScopeId>;
}

/// Scope modification interface
pub trait ScopeModifier {
    /// Add a symbol reference to a scope
    fn add_symbol(&mut self, scope_id: ScopeId, symbol: prism_common::symbol::Symbol) -> crate::error::CompilerResult<()>;
    
    /// Remove a symbol reference from a scope
    fn remove_symbol(&mut self, scope_id: ScopeId, symbol: prism_common::symbol::Symbol) -> crate::error::CompilerResult<()>;
    
    /// Add an import to a scope
    fn add_import(&mut self, scope_id: ScopeId, name: String, symbol: prism_common::symbol::Symbol) -> crate::error::CompilerResult<()>;
    
    /// Add an export from a scope
    fn add_export(&mut self, scope_id: ScopeId, name: String, symbol: prism_common::symbol::Symbol) -> crate::error::CompilerResult<()>;
    
    /// Update scope metadata
    fn update_metadata<F>(&mut self, scope_id: ScopeId, updater: F) -> crate::error::CompilerResult<()>
    where
        F: FnOnce(&mut ScopeMetadata);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CompilerResult;
    
    #[test]
    fn test_scope_subsystem_exports() {
        // Ensure all main types are properly exported
        let _tree_config = ScopeTreeConfig::default();
        let _scope_kind = ScopeKind::Global;
        let _visibility = ScopeVisibility::Public;
        let _metadata = ScopeMetadata::default();
    }
} 

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::scope::{ScopeTree, ScopeKind, ScopeBuilder, ScopeBuilderConfig};
    use prism_common::span::Span;
    
    #[test]
    fn test_scope_subsystem_integration() {
        // Test that all the main components work together
        let mut tree = ScopeTree::new();
        
        // Create global scope
        let global_id = tree.create_global_scope().unwrap();
        assert_eq!(tree.get_root_scope(), Some(global_id));
        
        // Create module scope
        let module_kind = ScopeKind::Module {
            module_name: "TestModule".to_string(),
            sections: vec!["interface".to_string()],
            capabilities: vec!["read".to_string()],
            effects: vec![],
        };
        
        let module_id = tree.create_scope(module_kind, Some(global_id)).unwrap();
        
        // Test hierarchy
        assert_eq!(tree.get_parent(module_id), Some(global_id));
        assert_eq!(tree.get_children(global_id), vec![module_id]);
        
        // Test scope chain
        let chain = tree.get_scope_chain(module_id);
        assert_eq!(chain, vec![module_id, global_id]);
        
        // Test statistics
        let stats = tree.stats();
        assert_eq!(stats.total_scopes, 2);
        assert_eq!(stats.root_scopes, 1);
        
        // Test builder
        let builder = ScopeBuilder::new();
        assert!(builder.config().enable_ai_metadata);
    }
    
    #[test]
    fn test_scope_metadata_integration() {
        use crate::scope::{ScopeMetadata, AIScopeContext, ScopeDocumentation};
        
        // Test metadata creation
        let mut metadata = ScopeMetadata::new(Some("Test scope responsibility".to_string()));
        
        // Add AI context
        let ai_context = AIScopeContext::new("Test scope for user management".to_string());
        metadata = metadata.with_ai_context(ai_context);
        
        // Add documentation
        let doc = ScopeDocumentation::new("Test scope documentation".to_string());
        metadata = metadata.with_documentation(doc);
        
        // Test AI summary
        let summary = metadata.ai_summary();
        assert!(summary.contains("Responsibility"));
        assert!(summary.contains("Purpose"));
    }
    
    #[test]
    fn test_scope_visibility_integration() {
        use crate::scope::{ScopeVisibility, AccessLevel, VisibilityContext, AccessResult};
        
        // Test visibility rules
        let public_vis = ScopeVisibility::Public;
        let private_vis = ScopeVisibility::Private;
        
        let context = VisibilityContext::SymbolAccess;
        
        // Test access results
        match public_vis.allows_access_from(1, &context) {
            AccessResult::Allowed(level) => assert_eq!(level, AccessLevel::Read),
            _ => panic!("Public visibility should allow access"),
        }
        
        match private_vis.allows_access_from(1, &context) {
            AccessResult::Denied(_) => (), // Expected
            _ => panic!("Private visibility should deny access"),
        }
        
        // Test visibility comparison
        assert!(private_vis.is_more_restrictive_than(&public_vis));
        assert!(!public_vis.is_more_restrictive_than(&private_vis));
    }
    
    #[test]
    fn test_scope_boundaries_integration() {
        use crate::scope::{EffectBoundary, EffectBoundaryType, CapabilityBoundary, CapabilityBoundaryType};
        
        // Test effect boundaries
        let mut effect_boundary = EffectBoundary::new(
            "test_effect_boundary".to_string(),
            EffectBoundaryType::Restrictive,
        );
        
        effect_boundary.allow_effect("IO".to_string());
        effect_boundary.prohibit_effect("Network".to_string());
        
        assert!(effect_boundary.is_effect_allowed("IO"));
        assert!(!effect_boundary.is_effect_allowed("Network"));
        assert!(!effect_boundary.is_effect_allowed("Unknown"));
        
        // Test capability boundaries
        let mut capability_boundary = CapabilityBoundary::new(
            "test_capability_boundary".to_string(),
            CapabilityBoundaryType::Grant,
        );
        
        capability_boundary.grant_capability("FileRead".to_string());
        
        assert!(capability_boundary.is_capability_available("FileRead"));
        assert!(!capability_boundary.is_capability_available("NetworkAccess"));
    }
} 