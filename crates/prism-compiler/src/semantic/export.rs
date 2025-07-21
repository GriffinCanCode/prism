//! Semantic Export - AI-Readable Context Generation
//!
//! This module implements export functionality for generating AI-readable
//! semantic context from the analysis results. It focuses on presenting
//! semantic information in a format optimized for AI consumption.
//!
//! **Conceptual Responsibility**: AI-readable semantic export
//! **What it does**: Export semantic context, generate AI metadata, format analysis results
//! **What it doesn't do**: Perform analysis, store data, manage subsystems

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::SymbolTable;
use crate::scope::ScopeTree;
use crate::semantic::{
    analysis::SemanticInfo,
    relationships::{CallGraph, DataFlowGraph, TypeRelationships},
    effects::EffectSignature,
    contracts::ContractSpecification,
};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// AI-readable semantic context for external tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIReadableContext {
    /// Context metadata
    pub metadata: ContextMetadata,
    /// Semantic analysis summary
    pub semantic_summary: SemanticSummary,
    /// Symbol information for AI comprehension
    pub symbol_context: SymbolContext,
    /// Relationship graphs
    pub relationships: RelationshipContext,
    /// Effect and contract information
    pub behavioral_contracts: BehavioralContext,
    /// Business and domain context
    pub business_context: BusinessContext,
    /// Performance and optimization hints
    pub performance_context: PerformanceContext,
}

/// Context metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Export timestamp
    pub exported_at: DateTime<Utc>,
    /// Semantic analysis version
    pub analysis_version: String,
    /// Context format version
    pub format_version: String,
    /// Export configuration used
    pub export_config: ExportConfig,
}

/// Semantic analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSummary {
    /// Total symbols analyzed
    pub total_symbols: usize,
    /// Total relationships discovered
    pub total_relationships: usize,
    /// Analysis completeness (0.0 to 1.0)
    pub completeness_score: f64,
    /// Analysis confidence (0.0 to 1.0)
    pub confidence_score: f64,
    /// Key insights discovered
    pub key_insights: Vec<String>,
}

/// Symbol context for AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolContext {
    /// Function symbols with AI context
    pub functions: HashMap<Symbol, FunctionContext>,
    /// Type symbols with AI context
    pub types: HashMap<Symbol, TypeContext>,
    /// Module symbols with AI context
    pub modules: HashMap<Symbol, ModuleContext>,
    /// Symbol usage patterns
    pub usage_patterns: Vec<UsagePattern>,
}

/// Function context for AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionContext {
    /// Function symbol
    pub symbol: Symbol,
    /// Function purpose and description
    pub purpose: String,
    /// Algorithm description
    pub algorithm: Option<String>,
    /// Complexity analysis
    pub complexity: Option<ComplexityInfo>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common patterns and idioms
    pub patterns: Vec<String>,
    /// Performance characteristics
    pub performance: Vec<String>,
}

/// Type context for AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeContext {
    /// Type symbol
    pub symbol: Symbol,
    /// Type purpose and meaning
    pub purpose: String,
    /// Business domain information
    pub domain: Option<String>,
    /// Usage patterns
    pub usage_patterns: Vec<String>,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// Validation rules
    pub validation_rules: Vec<String>,
}

/// Module context for AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleContext {
    /// Module symbol
    pub symbol: Symbol,
    /// Module responsibility
    pub responsibility: String,
    /// Capabilities provided
    pub capabilities: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Architectural role
    pub architectural_role: String,
}

/// Usage pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Symbols involved in pattern
    pub symbols: Vec<Symbol>,
    /// Pattern frequency
    pub frequency: u32,
    /// Pattern benefits
    pub benefits: Vec<String>,
}

/// Relationship context for AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipContext {
    /// Call relationship summary
    pub call_relationships: CallRelationshipSummary,
    /// Data flow summary
    pub data_flow_summary: DataFlowSummary,
    /// Type relationship summary
    pub type_relationships: TypeRelationshipSummary,
    /// Architectural patterns identified
    pub architectural_patterns: Vec<ArchitecturalPattern>,
}

/// Call relationship summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRelationshipSummary {
    /// Total function calls
    pub total_calls: usize,
    /// Most called functions
    pub hot_functions: Vec<Symbol>,
    /// Call depth statistics
    pub call_depth_stats: DepthStats,
    /// Circular dependencies
    pub circular_dependencies: Vec<Vec<Symbol>>,
}

/// Data flow summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowSummary {
    /// Total data dependencies
    pub total_dependencies: usize,
    /// Data flow complexity
    pub complexity_score: f64,
    /// Critical data paths
    pub critical_paths: Vec<DataPath>,
    /// Data bottlenecks
    pub bottlenecks: Vec<Symbol>,
}

/// Type relationship summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationshipSummary {
    /// Inheritance depth statistics
    pub inheritance_depth: DepthStats,
    /// Type usage frequency
    pub usage_frequency: HashMap<Symbol, u32>,
    /// Type conversion patterns
    pub conversion_patterns: Vec<String>,
}

/// Depth statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthStats {
    /// Maximum depth
    pub max_depth: usize,
    /// Average depth
    pub avg_depth: f64,
    /// Depth distribution
    pub depth_distribution: HashMap<usize, usize>,
}

/// Data path information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPath {
    /// Path identifier
    pub id: String,
    /// Symbols in the path
    pub symbols: Vec<Symbol>,
    /// Path complexity
    pub complexity: f64,
    /// Path description
    pub description: String,
}

/// Architectural pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalPattern {
    /// Pattern name
    pub name: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Symbols involved
    pub symbols: Vec<Symbol>,
    /// Pattern confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Pattern benefits
    pub benefits: Vec<String>,
}

/// Type of architectural pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Creational patterns
    Creational,
    /// Structural patterns
    Structural,
    /// Behavioral patterns
    Behavioral,
    /// Architectural patterns
    Architectural,
    /// Domain-specific patterns
    DomainSpecific(String),
}

/// Behavioral context (effects and contracts)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralContext {
    /// Effect summaries
    pub effects: Vec<EffectSummary>,
    /// Contract summaries
    pub contracts: Vec<ContractSummary>,
    /// Behavioral patterns
    pub patterns: Vec<BehavioralPattern>,
}

/// Effect summary for AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSummary {
    /// Function symbol
    pub symbol: Symbol,
    /// Effect categories
    pub effect_categories: Vec<String>,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Effect description
    pub description: String,
}

/// Contract summary for AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSummary {
    /// Function symbol
    pub symbol: Symbol,
    /// Contract type
    pub contract_type: String,
    /// Contract description
    pub description: String,
    /// Critical conditions
    pub critical_conditions: Vec<String>,
}

/// Behavioral pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Symbols exhibiting pattern
    pub symbols: Vec<Symbol>,
    /// Pattern implications
    pub implications: Vec<String>,
}

/// Business context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Business domains identified
    pub domains: Vec<BusinessDomain>,
    /// Business rules
    pub rules: Vec<BusinessRule>,
    /// Stakeholder information
    pub stakeholders: Vec<Stakeholder>,
    /// Compliance requirements
    pub compliance: Vec<ComplianceRequirement>,
}

/// Business domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessDomain {
    /// Domain name
    pub name: String,
    /// Domain description
    pub description: String,
    /// Related symbols
    pub symbols: Vec<Symbol>,
    /// Domain expertise required
    pub expertise_level: ExpertiseLevel,
}

/// Business rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Affected symbols
    pub symbols: Vec<Symbol>,
    /// Rule criticality
    pub criticality: RuleCriticality,
}

/// Stakeholder information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stakeholder {
    /// Stakeholder name/role
    pub name: String,
    /// Areas of interest
    pub interests: Vec<String>,
    /// Related symbols
    pub symbols: Vec<Symbol>,
}

/// Compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    /// Requirement identifier
    pub id: String,
    /// Requirement description
    pub description: String,
    /// Compliance standard
    pub standard: String,
    /// Affected symbols
    pub symbols: Vec<Symbol>,
}

/// Performance context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContext {
    /// Performance hotspots
    pub hotspots: Vec<PerformanceHotspot>,
    /// Optimization opportunities
    pub optimizations: Vec<OptimizationOpportunity>,
    /// Performance patterns
    pub patterns: Vec<PerformancePattern>,
    /// Resource usage analysis
    pub resource_usage: ResourceUsageAnalysis,
}

/// Performance hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHotspot {
    /// Symbol causing hotspot
    pub symbol: Symbol,
    /// Hotspot type
    pub hotspot_type: HotspotType,
    /// Severity (0.0 to 1.0)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Suggested improvements
    pub improvements: Vec<String>,
}

/// Type of performance hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotspotType {
    /// CPU intensive
    CPU,
    /// Memory intensive
    Memory,
    /// I/O intensive
    IO,
    /// Network intensive
    Network,
    /// Algorithmic complexity
    Algorithmic,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: OptimizationType,
    /// Affected symbols
    pub symbols: Vec<Symbol>,
    /// Potential impact (0.0 to 1.0)
    pub impact: f64,
    /// Implementation difficulty (0.0 to 1.0)
    pub difficulty: f64,
    /// Description
    pub description: String,
}

/// Type of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Algorithmic optimization
    Algorithmic,
    /// Data structure optimization
    DataStructure,
    /// Caching opportunity
    Caching,
    /// Parallelization opportunity
    Parallelization,
    /// Memory optimization
    Memory,
}

/// Performance pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Symbols exhibiting pattern
    pub symbols: Vec<Symbol>,
    /// Performance impact
    pub impact: PerformanceImpact,
}

/// Performance impact level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceImpact {
    /// Positive impact
    Positive(f64),
    /// Negative impact
    Negative(f64),
    /// Neutral impact
    Neutral,
}

/// Resource usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageAnalysis {
    /// Memory usage patterns
    pub memory_patterns: Vec<String>,
    /// CPU usage patterns
    pub cpu_patterns: Vec<String>,
    /// I/O patterns
    pub io_patterns: Vec<String>,
    /// Resource bottlenecks
    pub bottlenecks: Vec<String>,
}

/// Complexity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityInfo {
    /// Time complexity
    pub time_complexity: String,
    /// Space complexity
    pub space_complexity: String,
    /// Cyclomatic complexity
    pub cyclomatic_complexity: Option<u32>,
    /// Cognitive complexity
    pub cognitive_complexity: Option<u32>,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Include detailed symbol information
    pub include_detailed_symbols: bool,
    /// Include relationship graphs
    pub include_relationships: bool,
    /// Include behavioral context
    pub include_behavioral: bool,
    /// Include business context
    pub include_business: bool,
    /// Include performance context
    pub include_performance: bool,
    /// Export format
    pub format: ExportFormat,
}

/// Export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// YAML format
    YAML,
    /// XML format
    XML,
    /// Custom format
    Custom(String),
}

/// Expertise level required
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    /// Beginner level
    Beginner,
    /// Intermediate level
    Intermediate,
    /// Advanced level
    Advanced,
    /// Expert level
    Expert,
}

/// Rule criticality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCriticality {
    /// Low criticality
    Low,
    /// Medium criticality
    Medium,
    /// High criticality
    High,
    /// Critical
    Critical,
}

/// Semantic exporter for generating AI-readable context
#[derive(Debug)]
pub struct SemanticExporter {
    /// Symbol table integration
    symbol_table: Arc<SymbolTable>,
    /// Scope tree integration
    scope_tree: Arc<ScopeTree>,
}

impl SemanticExporter {
    /// Create a new semantic exporter
    pub fn new(symbol_table: Arc<SymbolTable>, scope_tree: Arc<ScopeTree>) -> Self {
        Self {
            symbol_table,
            scope_tree,
        }
    }

    /// Export comprehensive AI-readable context
    pub async fn export_context(&self) -> CompilerResult<AIReadableContext> {
        let metadata = self.build_metadata().await;
        let semantic_summary = self.build_semantic_summary().await?;
        let symbol_context = self.build_symbol_context().await?;
        let relationships = self.build_relationship_context().await?;
        let behavioral_contracts = self.build_behavioral_context().await?;
        let business_context = self.build_business_context().await?;
        let performance_context = self.build_performance_context().await?;

        Ok(AIReadableContext {
            metadata,
            semantic_summary,
            symbol_context,
            relationships,
            behavioral_contracts,
            business_context,
            performance_context,
        })
    }

    /// Build context metadata
    async fn build_metadata(&self) -> ContextMetadata {
        ContextMetadata {
            exported_at: Utc::now(),
            analysis_version: env!("CARGO_PKG_VERSION").to_string(),
            format_version: "1.0".to_string(),
            export_config: ExportConfig {
                include_detailed_symbols: true,
                include_relationships: true,
                include_behavioral: true,
                include_business: true,
                include_performance: true,
                format: ExportFormat::JSON,
            },
        }
    }

    /// Build semantic summary
    async fn build_semantic_summary(&self) -> CompilerResult<SemanticSummary> {
        let stats = self.symbol_table.stats();
        
        Ok(SemanticSummary {
            total_symbols: stats.total_symbols,
            total_relationships: 0, // Would be calculated from actual analysis
            completeness_score: 0.85, // Would be calculated based on analysis coverage
            confidence_score: 0.90, // Would be calculated based on analysis confidence
            key_insights: vec![
                "Well-structured module organization".to_string(),
                "Good separation of concerns".to_string(),
                "Consistent naming conventions".to_string(),
            ],
        })
    }

    /// Build symbol context
    async fn build_symbol_context(&self) -> CompilerResult<SymbolContext> {
        // This would build comprehensive symbol context from symbol table
        Ok(SymbolContext {
            functions: HashMap::new(),
            types: HashMap::new(),
            modules: HashMap::new(),
            usage_patterns: Vec::new(),
        })
    }

    /// Build relationship context
    async fn build_relationship_context(&self) -> CompilerResult<RelationshipContext> {
        // This would analyze relationships and build context
        Ok(RelationshipContext {
            call_relationships: CallRelationshipSummary {
                total_calls: 0,
                hot_functions: Vec::new(),
                call_depth_stats: DepthStats {
                    max_depth: 0,
                    avg_depth: 0.0,
                    depth_distribution: HashMap::new(),
                },
                circular_dependencies: Vec::new(),
            },
            data_flow_summary: DataFlowSummary {
                total_dependencies: 0,
                complexity_score: 0.0,
                critical_paths: Vec::new(),
                bottlenecks: Vec::new(),
            },
            type_relationships: TypeRelationshipSummary {
                inheritance_depth: DepthStats {
                    max_depth: 0,
                    avg_depth: 0.0,
                    depth_distribution: HashMap::new(),
                },
                usage_frequency: HashMap::new(),
                conversion_patterns: Vec::new(),
            },
            architectural_patterns: Vec::new(),
        })
    }

    /// Build behavioral context
    async fn build_behavioral_context(&self) -> CompilerResult<BehavioralContext> {
        Ok(BehavioralContext {
            effects: Vec::new(),
            contracts: Vec::new(),
            patterns: Vec::new(),
        })
    }

    /// Build business context
    async fn build_business_context(&self) -> CompilerResult<BusinessContext> {
        Ok(BusinessContext {
            domains: Vec::new(),
            rules: Vec::new(),
            stakeholders: Vec::new(),
            compliance: Vec::new(),
        })
    }

    /// Build performance context
    async fn build_performance_context(&self) -> CompilerResult<PerformanceContext> {
        Ok(PerformanceContext {
            hotspots: Vec::new(),
            optimizations: Vec::new(),
            patterns: Vec::new(),
            resource_usage: ResourceUsageAnalysis {
                memory_patterns: Vec::new(),
                cpu_patterns: Vec::new(),
                io_patterns: Vec::new(),
                bottlenecks: Vec::new(),
            },
        })
    }
}

/// Semantic export result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticExport {
    /// Exported context
    pub context: AIReadableContext,
    /// Export statistics
    pub statistics: ExportStatistics,
}

/// Export statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStatistics {
    /// Export duration in milliseconds
    pub export_duration_ms: u64,
    /// Total data size in bytes
    pub total_size_bytes: usize,
    /// Number of symbols exported
    pub symbols_exported: usize,
    /// Number of relationships exported
    pub relationships_exported: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_detailed_symbols: true,
            include_relationships: true,
            include_behavioral: true,
            include_business: true,
            include_performance: true,
            format: ExportFormat::JSON,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_config_creation() {
        let config = ExportConfig::default();
        assert!(config.include_detailed_symbols);
        assert!(config.include_relationships);
        assert!(config.include_behavioral);
        assert!(matches!(config.format, ExportFormat::JSON));
    }

    #[test]
    fn test_semantic_summary_creation() {
        let summary = SemanticSummary {
            total_symbols: 100,
            total_relationships: 50,
            completeness_score: 0.85,
            confidence_score: 0.90,
            key_insights: vec!["Good structure".to_string()],
        };

        assert_eq!(summary.total_symbols, 100);
        assert_eq!(summary.total_relationships, 50);
        assert_eq!(summary.key_insights.len(), 1);
    }
} 