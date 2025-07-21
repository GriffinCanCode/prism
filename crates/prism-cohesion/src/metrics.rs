//! Core Cohesion Metrics Calculation
//!
//! This module implements the PLD-002 cohesion scoring algorithm by leveraging
//! existing graph analysis from prism-compiler and semantic analysis from prism-semantic.
//! 
//! **Conceptual Responsibility**: Calculate cohesion metrics using multiple dimensions
//! **Integration Strategy**: Compose with existing graph analysis, don't duplicate

use crate::{CohesionResult, MetricWeights};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use serde::{Serialize, Deserialize};
use rustc_hash::FxHashMap;
use std::time::Instant;

// Import specialized analyzers
mod type_cohesion;
mod semantic_cohesion;
mod business_cohesion;
mod dependency_cohesion;

use type_cohesion::TypeCohesionAnalyzer;
use semantic_cohesion::SemanticCohesionAnalyzer;
use business_cohesion::BusinessCohesionAnalyzer;
use dependency_cohesion::DependencyCohesionAnalyzer;

/// Complete cohesion metrics for a program or module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionMetrics {
    /// Overall cohesion score (0-100)
    pub overall_score: f64,
    
    /// Individual metric scores
    pub type_cohesion: f64,
    pub data_flow_cohesion: f64,
    pub semantic_cohesion: f64,
    pub business_cohesion: f64,
    pub dependency_cohesion: f64,
    
    /// Detailed analysis
    pub analysis: CohesionAnalysis,
    
    /// Metric calculation metadata
    pub metadata: MetricsMetadata,
}

/// Detailed cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysis {
    /// Factors contributing positively to cohesion
    pub strengths: Vec<String>,
    
    /// Areas needing improvement
    pub weaknesses: Vec<String>,
    
    /// Specific improvement suggestions
    pub suggestions: Vec<String>,
    
    /// Detected architectural patterns
    pub patterns: Vec<String>,
    
    /// Conceptual boundaries identified
    pub boundaries: Vec<String>,
}

/// Metadata about metric calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsMetadata {
    /// Calculation timestamp
    pub calculated_at: String,
    
    /// Analysis depth used
    pub analysis_depth: String,
    
    /// Number of elements analyzed
    pub elements_analyzed: usize,
    
    /// Confidence in results (0-1)
    pub confidence: f64,
    
    /// Performance metrics
    pub calculation_time_ms: u64,
}

impl CohesionMetrics {
    /// Create empty metrics for edge cases
    pub fn empty() -> Self {
        Self {
            overall_score: 0.0,
            type_cohesion: 0.0,
            data_flow_cohesion: 0.0,
            semantic_cohesion: 0.0,
            business_cohesion: 0.0,
            dependency_cohesion: 0.0,
            analysis: CohesionAnalysis {
                strengths: Vec::new(),
                weaknesses: vec!["No analyzable content found".to_string()],
                suggestions: vec!["Add modules with meaningful content".to_string()],
                patterns: Vec::new(),
                boundaries: Vec::new(),
            },
            metadata: MetricsMetadata {
                calculated_at: chrono::Utc::now().to_rfc3339(),
                analysis_depth: "Empty".to_string(),
                elements_analyzed: 0,
                confidence: 0.0,
                calculation_time_ms: 0,
            },
        }
    }
}

/// Metrics calculator that leverages existing analysis systems
#[derive(Debug)]
pub struct MetricsCalculator {
    /// Metric weights for scoring
    weights: MetricWeights,
    
    /// Specialized analyzers
    type_analyzer: TypeCohesionAnalyzer,
    semantic_analyzer: SemanticCohesionAnalyzer,
    business_analyzer: BusinessCohesionAnalyzer,
    dependency_analyzer: DependencyCohesionAnalyzer,
    
    /// String similarity cache (pre-sized for better performance)
    similarity_cache: FxHashMap<(String, String), f64>,
    
    /// Calculation statistics
    stats: CalculationStats,
}

/// Calculation statistics
#[derive(Debug, Default)]
struct CalculationStats {
    /// Number of metrics calculated
    metrics_calculated: usize,
    
    /// Cache hits
    cache_hits: usize,
    
    /// Total calculation time
    total_time_ms: u64,
}

impl MetricsCalculator {
    /// Create a new metrics calculator
    pub fn new(weights: MetricWeights) -> Self {
        Self {
            weights,
            type_analyzer: TypeCohesionAnalyzer::new(),
            semantic_analyzer: SemanticCohesionAnalyzer::new(),
            business_analyzer: BusinessCohesionAnalyzer::new(),
            dependency_analyzer: DependencyCohesionAnalyzer::new(),
            similarity_cache: FxHashMap::with_capacity_and_hasher(512, Default::default()),
            stats: CalculationStats::default(),
        }
    }
    
    /// Calculate cohesion metrics for a complete program (OPTIMIZED)
    pub fn calculate_program_metrics(&self, program: &Program) -> CohesionResult<CohesionMetrics> {
        let start_time = Instant::now();
        
        // Extract modules for analysis (optimized with capacity hint)
        let mut modules = Vec::with_capacity(program.items.len());
        for item in &program.items {
            if let Item::Module(module_decl) = &item.kind {
                modules.push((item, module_decl));
            }
        }
        
        if modules.is_empty() {
            return Ok(CohesionMetrics::empty());
        }
        
        // Calculate metrics using specialized analyzers
        let type_cohesion = self.type_analyzer.analyze_program(&modules)?;
        let semantic_cohesion = self.semantic_analyzer.analyze_program(&modules)?;
        let business_cohesion = self.business_analyzer.analyze_program(&modules)?;
        let dependency_cohesion = self.dependency_analyzer.analyze_program(&modules)?;
        
        // Use optimized heuristic for data flow (as discussed - not worth the complexity)
        let data_flow_cohesion = self.calculate_data_flow_cohesion_fast(&modules);
        
        // Calculate weighted overall score per PLD-002
        let overall_score = type_cohesion * self.weights.type_cohesion +
            data_flow_cohesion * self.weights.data_flow_cohesion +
            semantic_cohesion * self.weights.semantic_cohesion +
            business_cohesion * self.weights.business_cohesion +
            dependency_cohesion * self.weights.dependency_cohesion;
        
        // Generate analysis insights (optimized)
        let analysis = self.generate_analysis_fast(&type_cohesion, &data_flow_cohesion, 
                                                  &semantic_cohesion, &business_cohesion, 
                                                  &dependency_cohesion);
        
        let calculation_time = start_time.elapsed().as_millis() as u64;
        
        Ok(CohesionMetrics {
            overall_score,
            type_cohesion,
            data_flow_cohesion,
            semantic_cohesion,
            business_cohesion,
            dependency_cohesion,
            analysis,
            metadata: MetricsMetadata {
                calculated_at: chrono::Utc::now().to_rfc3339(),
                analysis_depth: "Program".to_string(),
                elements_analyzed: modules.len(),
                confidence: self.calculate_confidence_fast(&modules),
                calculation_time_ms: calculation_time,
            },
        })
    }
    
    /// Calculate cohesion metrics for a single module
    pub fn calculate_module_metrics(
        &self, 
        module_item: &AstNode<Item>, 
        module_decl: &ModuleDecl
    ) -> CohesionResult<CohesionMetrics> {
        let start_time = Instant::now();
        
        // Analyze module using specialized analyzers
        let type_cohesion = self.type_analyzer.analyze_module(module_item, module_decl)?;
        let semantic_cohesion = self.semantic_analyzer.analyze_module(module_item, module_decl)?;
        let business_cohesion = self.business_analyzer.analyze_module(module_item, module_decl)?;
        let dependency_cohesion = self.dependency_analyzer.analyze_module(module_item, module_decl)?;
        let data_flow_cohesion = self.calculate_module_data_flow_cohesion(module_decl);
        
        let overall_score = type_cohesion * self.weights.type_cohesion +
            data_flow_cohesion * self.weights.data_flow_cohesion +
            semantic_cohesion * self.weights.semantic_cohesion +
            business_cohesion * self.weights.business_cohesion +
            dependency_cohesion * self.weights.dependency_cohesion;
        
        let analysis = self.generate_module_analysis(module_decl, &type_cohesion, 
                                                   &data_flow_cohesion, &semantic_cohesion, 
                                                   &business_cohesion, &dependency_cohesion);
        
        let calculation_time = start_time.elapsed().as_millis() as u64;
        
        Ok(CohesionMetrics {
            overall_score,
            type_cohesion,
            data_flow_cohesion,
            semantic_cohesion,
            business_cohesion,
            dependency_cohesion,
            analysis,
            metadata: MetricsMetadata {
                calculated_at: chrono::Utc::now().to_rfc3339(),
                analysis_depth: "Module".to_string(),
                elements_analyzed: module_decl.sections.len(),
                confidence: self.calculate_module_confidence(module_decl),
                calculation_time_ms: calculation_time,
            },
        })
    }
    
    // PRIVATE HELPER METHODS
    
    /// Optimized data flow calculation (fast heuristic)
    fn calculate_data_flow_cohesion_fast(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> f64 {
        // Fast heuristic: well-organized sections = good data flow
        let mut total_score = 0.0;
        let mut module_count = 0;
        
        for (_, module_decl) in modules {
            let section_count = module_decl.sections.len();
            let has_interface = module_decl.sections.iter()
                .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Interface));
            
            let score = match (section_count, has_interface) {
                (0, _) => 30.0,
                (1, _) => 60.0,
                (2..=4, true) => 85.0,
                (2..=4, false) => 75.0,
                (5..=7, true) => 80.0,
                (5..=7, false) => 65.0,
                (_, true) => 70.0,
                (_, false) => 55.0,
            };
            
            total_score += score;
            module_count += 1;
        }
        
        if module_count > 0 { total_score / module_count as f64 } else { 80.0 }
    }
    
    /// Calculate data flow cohesion for a single module
    fn calculate_module_data_flow_cohesion(&self, module_decl: &ModuleDecl) -> f64 {
        let section_count = module_decl.sections.len();
        let has_interface = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Interface));
        
        match (section_count, has_interface) {
            (0, _) => 30.0,
            (1, _) => 60.0,
            (2..=4, true) => 85.0,
            (2..=4, false) => 75.0,
            (5..=7, true) => 80.0,
            (5..=7, false) => 65.0,
            (_, true) => 70.0,
            (_, false) => 55.0,
        }
    }
    
    /// Fast confidence calculation
    fn calculate_confidence_fast(&self, modules: &[(&AstNode<Item>, &ModuleDecl)]) -> f64 {
        // Base confidence on module count and organization
        let module_count = modules.len();
        let organized_modules = modules.iter()
            .filter(|(_, m)| !m.sections.is_empty())
            .count();
        
        let base_confidence = 0.7;
        let organization_bonus = (organized_modules as f64 / module_count as f64) * 0.2;
        let size_bonus = if module_count > 1 { 0.1 } else { 0.0 };
        
        (base_confidence + organization_bonus + size_bonus).min(1.0)
    }
    
    /// Calculate confidence for a single module
    fn calculate_module_confidence(&self, module_decl: &ModuleDecl) -> f64 {
        let mut confidence = 0.7; // Base confidence
        
        if module_decl.description.is_some() { confidence += 0.1; }
        if module_decl.capability.is_some() { confidence += 0.1; }
        if !module_decl.sections.is_empty() { confidence += 0.1; }
        
        confidence.min(1.0)
    }
    
    /// Fast analysis generation
    fn generate_analysis_fast(&self, type_cohesion: &f64, _data_flow_cohesion: &f64, 
                             semantic_cohesion: &f64, business_cohesion: &f64, 
                             _dependency_cohesion: &f64) -> CohesionAnalysis {
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();
        let mut suggestions = Vec::new();
        
        // Fast analysis based on thresholds
        if *type_cohesion >= 80.0 {
            strengths.push("Well-organized type structure".to_string());
        } else if *type_cohesion < 60.0 {
            weaknesses.push("Type organization could be improved".to_string());
            suggestions.push("Consider reorganizing types by domain".to_string());
        }
        
        if *semantic_cohesion >= 80.0 {
            strengths.push("Consistent naming conventions".to_string());
        } else if *semantic_cohesion < 60.0 {
            weaknesses.push("Inconsistent naming patterns detected".to_string());
            suggestions.push("Standardize naming conventions".to_string());
        }
        
        if *business_cohesion >= 80.0 {
            strengths.push("Clear business focus".to_string());
        } else if *business_cohesion < 60.0 {
            weaknesses.push("Unclear business responsibilities".to_string());
            suggestions.push("Define clear capability boundaries".to_string());
        }
        
        CohesionAnalysis {
            strengths,
            weaknesses,
            suggestions,
            patterns: vec!["Standard module organization".to_string()],
            boundaries: Vec::new(),
        }
    }
    
    /// Generate analysis for a single module
    fn generate_module_analysis(&self, _module_decl: &ModuleDecl, type_cohesion: &f64, 
                               _data_flow_cohesion: &f64, semantic_cohesion: &f64, 
                               business_cohesion: &f64, _dependency_cohesion: &f64) -> CohesionAnalysis {
        // Similar to generate_analysis_fast but module-specific
        self.generate_analysis_fast(type_cohesion, _data_flow_cohesion, semantic_cohesion, business_cohesion, _dependency_cohesion)
    }
} 