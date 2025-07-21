//! Semantic Analysis Engine
//!
//! This module implements the core semantic analysis algorithms without
//! duplicating functionality from other subsystems. It focuses purely on
//! analyzing semantic relationships and generating analysis results.
//!
//! **Conceptual Responsibility**: Semantic analysis algorithms
//! **What it does**: Analyze semantic relationships, generate analysis results, validate consistency
//! **What it doesn't do**: Store symbols, manage scopes, classify types (uses other subsystems)

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolTable, SymbolData};
use crate::scope::ScopeTree;
use crate::semantic::SemanticConfig;
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_ast::{Program, AstNode, Item};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Semantic analyzer that performs analysis without duplicating storage
#[derive(Debug)]
pub struct SemanticAnalyzer {
    /// Symbol table integration (does NOT store symbols)
    symbol_table: Arc<SymbolTable>,
    
    /// Scope tree integration (does NOT manage scopes)
    scope_tree: Arc<ScopeTree>,
    
    /// Analysis configuration
    config: SemanticConfig,
}

/// Result of semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
    /// Semantic information extracted
    pub semantic_info: SemanticInfo,
    /// Analysis warnings and notes
    pub warnings: Vec<AnalysisWarning>,
}

/// Semantic information for a program (cleaned up - no duplication)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInfo {
    /// References to symbols (not storage - just references)
    pub symbol_references: HashMap<Symbol, SymbolReference>,
    /// Effect signatures found during analysis
    pub effects: HashMap<NodeId, crate::semantic::effects::EffectSignature>,
    /// Contract specifications found during analysis
    pub contracts: HashMap<NodeId, crate::semantic::contracts::ContractSpecification>,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// Reference to a symbol (not storage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolReference {
    /// Symbol being referenced
    pub symbol: Symbol,
    /// Node ID where this reference was found
    pub node_id: NodeId,
    /// Location of the reference
    pub location: Span,
    /// Type of reference (definition, usage, etc.)
    pub reference_type: ReferenceType,
    /// Semantic context of the reference
    pub semantic_context: String,
}

/// Type of symbol reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    /// Symbol definition
    Definition,
    /// Symbol usage/call
    Usage,
    /// Symbol assignment
    Assignment,
    /// Symbol import
    Import,
    /// Symbol export
    Export,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: String,
    /// Analysis duration in milliseconds
    pub duration_ms: u64,
    /// Number of symbols analyzed
    pub symbols_analyzed: usize,
    /// Number of nodes processed
    pub nodes_processed: usize,
    /// Analysis warnings
    pub warnings: Vec<String>,
}

/// Analysis warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisWarning {
    /// Warning message
    pub message: String,
    /// Location where warning occurred
    pub location: Option<Span>,
    /// Warning severity
    pub severity: WarningSeverity,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Warning severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningSeverity {
    /// Low severity warning
    Low,
    /// Medium severity warning
    Medium,
    /// High severity warning
    High,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new(
        symbol_table: Arc<SymbolTable>,
        scope_tree: Arc<ScopeTree>,
        config: SemanticConfig,
    ) -> Self {
        Self {
            symbol_table,
            scope_tree,
            config,
        }
    }

    /// Analyze a program and generate semantic information
    /// 
    /// This method analyzes relationships and generates analysis results
    /// without duplicating symbol or scope storage
    pub async fn analyze_program(&self, program: &Program) -> CompilerResult<SemanticInfo> {
        let start_time = std::time::Instant::now();
        
        let mut symbol_references = HashMap::new();
        let mut effects = HashMap::new();
        let mut contracts = HashMap::new();
        let mut nodes_processed = 0;

        // Analyze each item in the program
        for item in &program.items {
            self.analyze_item(item, &mut symbol_references, &mut effects, &mut contracts)?;
            nodes_processed += 1;
        }

        Ok(SemanticInfo {
            symbol_references,
            effects,
            contracts,
            analyzed_at: Utc::now(),
        })
    }

    /// Analyze a single program item
    fn analyze_item(
        &self,
        item: &Item,
        symbol_references: &mut HashMap<Symbol, SymbolReference>,
        effects: &mut HashMap<NodeId, crate::semantic::effects::EffectSignature>,
        contracts: &mut HashMap<NodeId, crate::semantic::contracts::ContractSpecification>,
    ) -> CompilerResult<()> {
        match &item.kind {
            prism_ast::Item::Function(func_decl) => {
                self.analyze_function(func_decl, symbol_references, effects, contracts)?;
            }
            prism_ast::Item::Module(module_decl) => {
                self.analyze_module(module_decl, symbol_references)?;
            }
            prism_ast::Item::Type(type_decl) => {
                self.analyze_type_declaration(type_decl, symbol_references)?;
            }
            _ => {
                // Handle other item types as needed
            }
        }
        Ok(())
    }

    /// Analyze a function declaration
    fn analyze_function(
        &self,
        _func_decl: &prism_ast::FunctionDecl,
        _symbol_references: &mut HashMap<Symbol, SymbolReference>,
        _effects: &mut HashMap<NodeId, crate::semantic::effects::EffectSignature>,
        _contracts: &mut HashMap<NodeId, crate::semantic::contracts::ContractSpecification>,
    ) -> CompilerResult<()> {
        // Implementation would analyze function semantics
        // This is a placeholder - actual implementation would:
        // 1. Extract function signature information from symbol table
        // 2. Analyze function body for semantic relationships
        // 3. Generate effect signatures if enabled
        // 4. Extract contract specifications if present
        Ok(())
    }

    /// Analyze a module declaration
    fn analyze_module(
        &self,
        _module_decl: &prism_ast::ModuleDecl,
        _symbol_references: &mut HashMap<Symbol, SymbolReference>,
    ) -> CompilerResult<()> {
        // Implementation would analyze module semantics
        // This is a placeholder
        Ok(())
    }

    /// Analyze a type declaration
    fn analyze_type_declaration(
        &self,
        _type_decl: &prism_ast::TypeDecl,
        _symbol_references: &mut HashMap<Symbol, SymbolReference>,
    ) -> CompilerResult<()> {
        // Implementation would analyze type semantics
        // This is a placeholder
        Ok(())
    }

    /// Validate semantic consistency
    pub fn validate_consistency(&self, semantic_info: &SemanticInfo) -> CompilerResult<Vec<AnalysisWarning>> {
        let mut warnings = Vec::new();

        // Validate that all referenced symbols exist in symbol table
        for (symbol, reference) in &semantic_info.symbol_references {
            if self.symbol_table.get_symbol(symbol).is_none() {
                warnings.push(AnalysisWarning {
                    message: format!("Referenced symbol '{}' not found in symbol table", symbol),
                    location: Some(reference.location),
                    severity: WarningSeverity::High,
                    suggested_fix: Some("Check symbol spelling and scope".to_string()),
                });
            }
        }

        Ok(warnings)
    }

    /// Get analysis statistics
    pub fn get_analysis_stats(&self) -> AnalysisStats {
        AnalysisStats {
            total_symbols: self.symbol_table.stats().total_symbols,
            total_scopes: self.scope_tree.stats().total_scopes,
            analysis_features_enabled: self.get_enabled_features(),
        }
    }

    /// Get list of enabled analysis features
    fn get_enabled_features(&self) -> Vec<String> {
        let mut features = Vec::new();
        
        if self.config.enable_ai_metadata {
            features.push("AI Metadata".to_string());
        }
        if self.config.enable_call_graph {
            features.push("Call Graph".to_string());
        }
        if self.config.enable_data_flow {
            features.push("Data Flow".to_string());
        }
        if self.config.enable_effect_tracking {
            features.push("Effect Tracking".to_string());
        }
        if self.config.enable_contract_analysis {
            features.push("Contract Analysis".to_string());
        }
        
        features
    }
}

/// Analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStats {
    /// Total symbols available for analysis
    pub total_symbols: usize,
    /// Total scopes available for analysis
    pub total_scopes: usize,
    /// List of enabled analysis features
    pub analysis_features_enabled: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_result_creation() {
        let metadata = AnalysisMetadata {
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            duration_ms: 100,
            symbols_analyzed: 10,
            nodes_processed: 50,
            warnings: Vec::new(),
        };

        let semantic_info = SemanticInfo {
            symbol_references: HashMap::new(),
            effects: HashMap::new(),
            contracts: HashMap::new(),
            analyzed_at: Utc::now(),
        };

        let result = AnalysisResult {
            metadata,
            semantic_info,
            warnings: Vec::new(),
        };

        assert_eq!(result.metadata.symbols_analyzed, 10);
        assert_eq!(result.metadata.nodes_processed, 50);
    }

    #[test]
    fn test_warning_severity() {
        let warning = AnalysisWarning {
            message: "Test warning".to_string(),
            location: None,
            severity: WarningSeverity::Medium,
            suggested_fix: None,
        };

        match warning.severity {
            WarningSeverity::Medium => assert!(true),
            _ => assert!(false, "Expected medium severity"),
        }
    }
} 