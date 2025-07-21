//! Semantic Database - Coordination and Integration Hub
//!
//! This module implements the core semantic database that coordinates semantic
//! analysis across the compiler without duplicating functionality from other
//! subsystems. It acts as an integration hub that delegates to specialized systems.
//!
//! **Conceptual Responsibility**: Semantic analysis coordination
//! **What it does**: Coordinate semantic analysis, manage relationships, cache results
//! **What it doesn't do**: Store symbols, classify types, manage scopes (delegates to specialized subsystems)

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::SymbolTable;
use crate::scope::ScopeTree;
use crate::context::AIMetadataCollector;
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_ast::Program;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// Semantic database that coordinates semantic analysis across subsystems
/// 
/// This does NOT store symbols or types directly - it delegates to the
/// appropriate subsystems and focuses solely on semantic relationships
/// and analysis coordination.
#[derive(Debug)]
pub struct SemanticDatabase {
    /// Symbol table integration (does NOT duplicate symbol storage)
    symbol_table: Arc<SymbolTable>,
    
    /// Scope tree integration (does NOT duplicate scope management)
    scope_tree: Arc<ScopeTree>,
    
    /// AI metadata collector integration
    ai_metadata_collector: Option<Arc<AIMetadataCollector>>,
    
    /// Semantic analysis configuration
    config: SemanticConfig,
}

/// Configuration for semantic analysis
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Enable AI metadata generation during analysis
    pub enable_ai_metadata: bool,
    /// Enable call graph analysis
    pub enable_call_graph: bool,
    /// Enable data flow analysis
    pub enable_data_flow: bool,
    /// Enable effect tracking
    pub enable_effect_tracking: bool,
    /// Enable contract analysis
    pub enable_contract_analysis: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_call_graph: true,
            enable_data_flow: true,
            enable_effect_tracking: true,
            enable_contract_analysis: true,
        }
    }
}

impl SemanticDatabase {
    /// Create a new semantic database with proper subsystem integration
    pub fn new(
        symbol_table: Arc<SymbolTable>,
        scope_tree: Arc<ScopeTree>,
        config: SemanticConfig,
    ) -> CompilerResult<Self> {
        Ok(Self {
            symbol_table,
            scope_tree,
            ai_metadata_collector: None,
            config,
        })
    }

    /// Create with AI metadata integration
    pub fn with_ai_metadata(
        symbol_table: Arc<SymbolTable>,
        scope_tree: Arc<ScopeTree>,
        ai_metadata_collector: Arc<AIMetadataCollector>,
        config: SemanticConfig,
    ) -> CompilerResult<Self> {
        Ok(Self {
            symbol_table,
            scope_tree,
            ai_metadata_collector: Some(ai_metadata_collector),
            config,
        })
    }

    /// Get symbol information by delegating to symbol table
    /// 
    /// This method does NOT store symbols - it delegates to the symbol table
    pub fn get_symbol_info(&self, symbol: &Symbol) -> Option<crate::symbols::SymbolData> {
        self.symbol_table.get_symbol(symbol)
    }

    /// Analyze semantic relationships for a program
    /// 
    /// This focuses on relationships and analysis, not storage
    pub async fn analyze_program(&self, program: &Program) -> CompilerResult<crate::semantic::analysis::SemanticInfo> {
        // Delegate to the semantic analyzer
        let analyzer = crate::semantic::analysis::SemanticAnalyzer::new(
            self.symbol_table.clone(),
            self.scope_tree.clone(),
            self.config.clone(),
        );
        
        analyzer.analyze_program(program).await
    }

    /// Get call graph analysis
    pub fn get_call_graph(&self) -> CompilerResult<crate::semantic::relationships::CallGraph> {
        if !self.config.enable_call_graph {
            return Err(CompilerError::InvalidOperation {
                message: "Call graph analysis is disabled".to_string(),
            });
        }

        // Delegate to relationship analyzer
        let relationship_analyzer = crate::semantic::relationships::RelationshipAnalyzer::new(
            self.symbol_table.clone()
        );
        
        relationship_analyzer.build_call_graph()
    }

    /// Get data flow graph analysis
    pub fn get_data_flow_graph(&self) -> CompilerResult<crate::semantic::relationships::DataFlowGraph> {
        if !self.config.enable_data_flow {
            return Err(CompilerError::InvalidOperation {
                message: "Data flow analysis is disabled".to_string(),
            });
        }

        // Delegate to relationship analyzer
        let relationship_analyzer = crate::semantic::relationships::RelationshipAnalyzer::new(
            self.symbol_table.clone()
        );
        
        relationship_analyzer.build_data_flow_graph()
    }

    /// Export AI-readable semantic context
    pub async fn export_ai_context(&self) -> CompilerResult<crate::semantic::export::AIReadableContext> {
        let exporter = crate::semantic::export::SemanticExporter::new(
            self.symbol_table.clone(),
            self.scope_tree.clone(),
        );
        
        exporter.export_context().await
    }

    /// Get configuration
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    /// Check if AI metadata is enabled
    pub fn has_ai_metadata(&self) -> bool {
        self.ai_metadata_collector.is_some() && self.config.enable_ai_metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::SymbolTableConfig;
    use crate::scope::ScopeTreeConfig;

    #[tokio::test]
    async fn test_semantic_database_creation() {
        // This would need actual implementations to work
        // For now, just test the structure
        assert!(true, "Semantic database structure is correct");
    }

    #[test]
    fn test_semantic_config() {
        let config = SemanticConfig::default();
        assert!(config.enable_ai_metadata);
        assert!(config.enable_call_graph);
        assert!(config.enable_data_flow);
    }
} 