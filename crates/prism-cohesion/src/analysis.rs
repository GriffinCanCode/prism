//! Cohesion Analysis Orchestration
//!
//! This module provides high-level coordination of cohesion analysis,
//! integrating with existing semantic analysis systems and managing
//! the overall analysis workflow.
//!
//! **Conceptual Responsibility**: Orchestrate cohesion analysis workflow
//! **Integration Strategy**: Coordinate with semantic analysis, don't duplicate

use crate::{CohesionResult, CohesionError, CohesionMetrics, MetricsCalculator, AnalysisDepth};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::{span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use rustc_hash::FxHashMap;

/// Configuration for cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Analysis depth and comprehensiveness
    pub depth: AnalysisDepth,
    
    /// Enable semantic integration
    pub enable_semantic_integration: bool,
    
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
    
    /// Enable AI context generation
    pub enable_ai_context: bool,
    
    /// Enable real-time analysis
    pub enable_real_time: bool,
    
    /// Maximum analysis time (milliseconds)
    pub max_analysis_time_ms: u64,
    
    /// Minimum confidence threshold
    pub min_confidence: f64,
}

/// Context for cohesion analysis
#[derive(Debug, Clone)]
pub struct AnalysisContext {
    /// Source identifier
    pub source_id: prism_common::SourceId,
    
    /// Analysis start time
    pub start_time: Instant,
    
    /// Current analysis phase
    pub current_phase: AnalysisPhase,
    
    /// Accumulated warnings
    pub warnings: Vec<AnalysisWarning>,
    
    /// Analysis statistics
    pub stats: AnalysisStats,
}

/// Analysis phases
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisPhase {
    /// Initialization phase
    Initialization,
    /// Module extraction
    ModuleExtraction,
    /// Metrics calculation
    MetricsCalculation,
    /// Pattern detection
    PatternDetection,
    /// Boundary analysis
    BoundaryAnalysis,
    /// Violation detection
    ViolationDetection,
    /// Suggestion generation
    SuggestionGeneration,
    /// AI insight generation
    AIInsightGeneration,
    /// Result compilation
    ResultCompilation,
    /// Completed
    Completed,
}

/// Analysis warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisWarning {
    /// Warning message
    pub message: String,
    
    /// Warning category
    pub category: WarningCategory,
    
    /// Location if relevant
    pub location: Option<Span>,
    
    /// Analysis phase where warning occurred
    pub phase: AnalysisPhase,
}

/// Warning categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningCategory {
    /// Performance warning
    Performance,
    /// Data quality warning
    DataQuality,
    /// Configuration warning
    Configuration,
    /// Integration warning
    Integration,
    /// Analysis limitation
    Limitation,
}

/// Analysis statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisStats {
    /// Total modules analyzed
    pub modules_analyzed: usize,
    
    /// Total sections analyzed
    pub sections_analyzed: usize,
    
    /// Total items analyzed
    pub items_analyzed: usize,
    
    /// Cache hits
    pub cache_hits: usize,
    
    /// Time spent in each phase (milliseconds)
    pub phase_times: HashMap<AnalysisPhase, u64>,
    
    /// Memory usage peak (bytes)
    pub peak_memory_usage: Option<u64>,
}

/// Complete analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Overall analysis success
    pub success: bool,
    
    /// Overall cohesion metrics
    pub metrics: CohesionMetrics,
    
    /// Individual module analyses
    pub module_analyses: Vec<ModuleAnalysis>,
    
    /// Analysis context and metadata
    pub context: AnalysisResultContext,
    
    /// Warnings encountered during analysis
    pub warnings: Vec<AnalysisWarning>,
}

/// Individual module analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleAnalysis {
    /// Module name
    pub module_name: String,
    
    /// Module cohesion metrics
    pub metrics: CohesionMetrics,
    
    /// Section-level analyses
    pub section_analyses: Vec<SectionAnalysis>,
    
    /// Module-specific patterns
    pub patterns: Vec<String>,
    
    /// Module analysis confidence
    pub confidence: f64,
}

/// Section-level analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionAnalysis {
    /// Section name/type
    pub section_name: String,
    
    /// Section cohesion score
    pub cohesion_score: f64,
    
    /// Items in this section
    pub item_count: usize,
    
    /// Section-specific insights
    pub insights: Vec<String>,
    
    /// Section metrics
    pub metrics: HashMap<String, f64>,
}

/// Analysis result context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResultContext {
    /// Analysis configuration used
    pub config: AnalysisConfig,
    
    /// Analysis statistics
    pub stats: AnalysisStats,
    
    /// Total analysis duration
    pub total_duration: Duration,
    
    /// Analysis timestamp
    pub timestamp: String,
    
    /// Analysis version/hash
    pub analysis_version: String,
}

/// Main cohesion analyzer
#[derive(Debug)]
pub struct CohesionAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    
    /// Metrics calculator
    metrics_calculator: MetricsCalculator,
    
    /// Analysis cache
    cache: AnalysisCache,
    
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
}

/// Analysis cache for performance
#[derive(Debug)]
struct AnalysisCache {
    /// Cached module analyses with LRU eviction
    module_cache: FxHashMap<String, ModuleAnalysis>,
    
    /// Cached metric calculations
    metrics_cache: FxHashMap<String, CohesionMetrics>,
    
    /// Cache access order for LRU eviction
    access_order: Vec<String>,
    
    /// Maximum cache size
    max_cache_size: usize,
    
    /// Cache statistics
    hits: usize,
    misses: usize,
}

impl AnalysisCache {
    /// Create new analysis cache with specified capacity
    fn new() -> Self {
        Self {
            module_cache: FxHashMap::with_capacity_and_hasher(128, Default::default()),
            metrics_cache: FxHashMap::with_capacity_and_hasher(128, Default::default()),
            access_order: Vec::with_capacity(256),
            max_cache_size: 100, // Reasonable default
            hits: 0,
            misses: 0,
        }
    }
    
    /// Get cached metrics with LRU tracking
    fn get_metrics(&mut self, key: &str) -> Option<&CohesionMetrics> {
        if let Some(metrics) = self.metrics_cache.get(key) {
            self.hits += 1;
            // Update LRU order
            if let Some(pos) = self.access_order.iter().position(|x| x == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.to_string());
            Some(metrics)
        } else {
            self.misses += 1;
            None
        }
    }
    
    /// Cache metrics with LRU eviction
    fn cache_metrics(&mut self, key: String, metrics: CohesionMetrics) {
        // Evict if at capacity
        if self.metrics_cache.len() >= self.max_cache_size {
            if let Some(oldest_key) = self.access_order.first().cloned() {
                self.metrics_cache.remove(&oldest_key);
                self.access_order.remove(0);
            }
        }
        
        self.metrics_cache.insert(key.clone(), metrics);
        self.access_order.push(key);
    }
    
    /// Get cached module analysis with LRU tracking
    fn get_module_analysis(&mut self, key: &str) -> Option<&ModuleAnalysis> {
        if let Some(analysis) = self.module_cache.get(key) {
            self.hits += 1;
            // Update LRU order (simplified for performance)
            Some(analysis)
        } else {
            self.misses += 1;
            None
        }
    }
    
    /// Cache module analysis with size limits
    fn cache_module_analysis(&mut self, key: String, analysis: ModuleAnalysis) {
        if self.module_cache.len() >= self.max_cache_size {
            // Simple eviction - remove random entry for performance
            if let Some(oldest_key) = self.module_cache.keys().next().cloned() {
                self.module_cache.remove(&oldest_key);
            }
        }
        
        self.module_cache.insert(key, analysis);
    }
    
    /// Get cache statistics
    fn get_hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Performance monitoring
#[derive(Debug)]
struct PerformanceMonitor {
    /// Phase timing
    phase_timings: HashMap<AnalysisPhase, Vec<Duration>>,
    
    /// Memory usage tracking
    memory_snapshots: Vec<u64>,
    
    /// Performance warnings threshold
    warning_threshold_ms: u64,
}

impl CohesionAnalyzer {
    /// Create new cohesion analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        let metrics_calculator = MetricsCalculator::new(crate::MetricWeights::default());
        
        Self {
            config,
            metrics_calculator,
            cache: AnalysisCache::new(),
            performance_monitor: PerformanceMonitor::new(),
        }
    }
    
    /// Create analyzer with custom metrics calculator
    pub fn with_metrics_calculator(config: AnalysisConfig, metrics_calculator: MetricsCalculator) -> Self {
        Self {
            config,
            metrics_calculator,
            cache: AnalysisCache::new(),
            performance_monitor: PerformanceMonitor::new(),
        }
    }
    
    /// Analyze a complete program (OPTIMIZED)
    pub fn analyze_program(&mut self, program: &Program) -> CohesionResult<AnalysisResult> {
        let mut context = AnalysisContext::new(program.source_id);
        
        // Early validation with fast path
        if program.items.is_empty() {
            return Err(CohesionError::InsufficientData {
                data_type: "Program has no items".to_string(),
            });
        }
        
        // Phase 1: Initialization (optimized)
        context.enter_phase(AnalysisPhase::Initialization);
        
        // Phase 2: Module extraction (optimized)
        context.enter_phase(AnalysisPhase::ModuleExtraction);
        let modules = self.extract_modules_fast(program, &mut context)?;
        
        if modules.is_empty() {
            context.add_warning(AnalysisWarning {
                message: "No modules found for analysis".to_string(),
                category: WarningCategory::DataQuality,
                location: None,
                phase: AnalysisPhase::ModuleExtraction,
            });
            
            // Early return for empty programs
            return Ok(AnalysisResult {
                success: false,
                metrics: CohesionMetrics::empty(),
                module_analyses: Vec::new(),
                context: AnalysisResultContext {
                    config: self.config.clone(),
                    stats: context.stats,
                    total_duration: context.start_time.elapsed(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    analysis_version: env!("CARGO_PKG_VERSION").to_string(),
                },
                warnings: context.warnings,
            });
        }
        
        // Phase 3: Metrics calculation (optimized)
        context.enter_phase(AnalysisPhase::MetricsCalculation);
        let overall_metrics = self.calculate_program_metrics_fast(program, &mut context)?;
        
        // Early exit for quick analysis
        if self.config.depth == AnalysisDepth::Quick {
            return Ok(AnalysisResult {
                success: true,
                metrics: overall_metrics,
                module_analyses: Vec::new(), // Skip detailed module analysis for quick mode
                context: AnalysisResultContext {
                    config: self.config.clone(),
                    stats: context.stats,
                    total_duration: context.start_time.elapsed(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    analysis_version: env!("CARGO_PKG_VERSION").to_string(),
                },
                warnings: context.warnings,
            });
        }
        
        // Phase 4: Individual module analysis (optimized for batch processing)
        let mut module_analyses = Vec::with_capacity(modules.len());
        for (module_item, module_decl) in modules {
            // Check timeout
            if context.start_time.elapsed().as_millis() > self.config.max_analysis_time_ms.into() {
                context.add_warning(AnalysisWarning {
                    message: "Analysis timeout reached, stopping early".to_string(),
                    category: WarningCategory::Performance,
                    location: None,
                    phase: AnalysisPhase::MetricsCalculation,
                });
                break;
            }
            
            let module_analysis = self.analyze_individual_module_fast(module_item, module_decl, &mut context)?;
            module_analyses.push(module_analysis);
        }
        
        // Phase 5: Pattern detection (if enabled and not quick)
        if self.config.enable_pattern_detection && self.config.depth != AnalysisDepth::Quick {
            context.enter_phase(AnalysisPhase::PatternDetection);
            self.detect_patterns(&module_analyses, &mut context)?;
        }
        
        // Phase 6: Result compilation
        context.enter_phase(AnalysisPhase::ResultCompilation);
        let analysis_result = self.compile_results(overall_metrics, module_analyses, context)?;
        
        Ok(analysis_result)
    }
    
    /// Analyze a single module
    pub fn analyze_module(&mut self, module_item: &AstNode<Item>, module_decl: &ModuleDecl) -> CohesionResult<ModuleAnalysis> {
        let mut context = AnalysisContext::new(prism_common::SourceId::new(0)); // TODO: Get actual source ID
        
        self.analyze_individual_module(module_item, module_decl, &mut context)
    }
    
    /// Validate analysis requirements
    fn validate_analysis_requirements(&self, program: &Program, context: &mut AnalysisContext) -> CohesionResult<()> {
        // Check if program has sufficient content for analysis
        if program.items.is_empty() {
            return Err(CohesionError::InsufficientData {
                data_type: "Program has no items".to_string(),
            });
        }
        
        // Check for timeout configuration
        if self.config.max_analysis_time_ms == 0 {
            context.add_warning(AnalysisWarning {
                message: "No analysis timeout configured".to_string(),
                category: WarningCategory::Configuration,
                location: None,
                phase: AnalysisPhase::Initialization,
            });
        }
        
        Ok(())
    }
    
    /// Extract modules from program
    fn extract_modules<'a>(&self, program: &'a Program, context: &mut AnalysisContext) -> CohesionResult<Vec<(&'a AstNode<Item>, &'a ModuleDecl)>> {
        let modules: Vec<_> = program.items.iter()
            .filter_map(|item| {
                if let Item::Module(module_decl) = &item.kind {
                    Some((item, module_decl))
                } else {
                    None
                }
            })
            .collect();
        
        context.stats.modules_analyzed = modules.len();
        
        Ok(modules)
    }
    
    /// Calculate overall program metrics
    fn calculate_program_metrics(&mut self, program: &Program, context: &mut AnalysisContext) -> CohesionResult<CohesionMetrics> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_program_cache_key(program);
        if let Some(cached_metrics) = self.cache.get_metrics(&cache_key) {
            context.stats.cache_hits += 1;
            return Ok(cached_metrics.clone());
        }
        
        // Calculate metrics
        let metrics = self.metrics_calculator.calculate_program_metrics(program)?;
        
        // Cache results
        self.cache.cache_metrics(cache_key, metrics.clone());
        
        let calculation_time = start_time.elapsed();
        context.stats.phase_times.insert(AnalysisPhase::MetricsCalculation, calculation_time.as_millis() as u64);
        
        // Performance monitoring
        if calculation_time.as_millis() > self.performance_monitor.warning_threshold_ms.into() {
            context.add_warning(AnalysisWarning {
                message: format!("Metrics calculation took {}ms (longer than {}ms threshold)", 
                               calculation_time.as_millis(), self.performance_monitor.warning_threshold_ms),
                category: WarningCategory::Performance,
                location: None,
                phase: AnalysisPhase::MetricsCalculation,
            });
        }
        
        Ok(metrics)
    }
    
    /// Analyze individual module
    fn analyze_individual_module(&mut self, module_item: &AstNode<Item>, module_decl: &ModuleDecl, context: &mut AnalysisContext) -> CohesionResult<ModuleAnalysis> {
        let start_time = Instant::now();
        
        // Check cache
        let cache_key = self.generate_module_cache_key(module_decl);
        if let Some(cached_analysis) = self.cache.get_module_analysis(&cache_key) {
            context.stats.cache_hits += 1;
            return Ok(cached_analysis.clone());
        }
        
        // Calculate module metrics
        let module_metrics = self.metrics_calculator.calculate_module_metrics(module_item, module_decl)?;
        
        // Analyze sections
        let mut section_analyses = Vec::new();
        for section in &module_decl.sections {
            let section_analysis = self.analyze_section(section, context)?;
            section_analyses.push(section_analysis);
        }
        
        // Detect module-specific patterns
        let patterns = self.detect_module_patterns(module_decl, &section_analyses)?;
        
        // Calculate confidence
        let confidence = self.calculate_module_confidence(module_decl, &section_analyses);
        
        let module_analysis = ModuleAnalysis {
            module_name: module_decl.name.to_string(),
            metrics: module_metrics,
            section_analyses,
            patterns,
            confidence,
        };
        
        // Cache results
        self.cache.cache_module_analysis(cache_key, module_analysis.clone());
        
        let analysis_time = start_time.elapsed();
        context.stats.sections_analyzed += module_decl.sections.len();
        
        Ok(module_analysis)
    }
    
    /// Analyze a section within a module
    fn analyze_section(&self, section: &AstNode<prism_ast::SectionDecl>, _context: &mut AnalysisContext) -> CohesionResult<SectionAnalysis> {
        let section_name = format!("{:?}", section.kind.kind);
        let item_count = section.kind.items.len();
        
        // Calculate section-specific cohesion score
        let cohesion_score = self.calculate_section_cohesion(&section.kind)?;
        
        // Generate section-specific insights
        let insights = self.generate_section_insights(&section.kind, item_count);
        
        // Calculate section metrics
        let mut metrics = HashMap::new();
        metrics.insert("item_count".to_string(), item_count as f64);
        metrics.insert("naming_consistency".to_string(), self.calculate_section_naming_consistency(&section.kind)?);
        metrics.insert("organization_score".to_string(), self.calculate_section_organization(&section.kind)?);
        
        let section_analysis = SectionAnalysis {
            section_name: section_name.clone(),
            cohesion_score,
            item_count,
            insights,
            metrics,
        };
        
        Ok(section_analysis)
    }
    
    /// Calculate cohesion score for a specific section
    fn calculate_section_cohesion(&self, section: &prism_ast::SectionDecl) -> CohesionResult<f64> {
        let item_count = section.items.len();
        
        if item_count == 0 {
            return Ok(100.0); // Empty section is perfectly cohesive
        }
        
        // Base score depends on section type appropriateness
        let base_score = match section.kind {
            prism_ast::SectionKind::Types => 85.0,
            prism_ast::SectionKind::Interface => 90.0,
            prism_ast::SectionKind::Internal => 80.0,
            prism_ast::SectionKind::Config => 95.0,
            prism_ast::SectionKind::Events => 85.0,
            _ => 75.0,
        };
        
        // Adjust based on item count (too many or too few items reduce cohesion)
        let size_penalty = if item_count > 20 {
            (item_count - 20) as f64 * 2.0 // Penalty for very large sections
        } else if item_count < 3 && matches!(section.kind, prism_ast::SectionKind::Types | prism_ast::SectionKind::Interface) {
            10.0 // Small penalty for very small important sections
        } else {
            0.0
        };
        
        Ok((base_score - size_penalty).max(0.0).min(100.0))
    }
    
    /// Generate insights for a section
    fn generate_section_insights(&self, section: &prism_ast::SectionDecl, item_count: usize) -> Vec<String> {
        let mut insights = Vec::new();
        
        match section.kind {
            prism_ast::SectionKind::Types => {
                if item_count > 15 {
                    insights.push("Large number of types - consider splitting into sub-modules".to_string());
                } else if item_count > 0 {
                    insights.push("Good type organization with appropriate section separation".to_string());
                }
            },
            prism_ast::SectionKind::Interface => {
                if item_count > 0 {
                    insights.push("Clear public interface definition".to_string());
                } else {
                    insights.push("Empty interface section - consider adding public API definitions".to_string());
                }
            },
            prism_ast::SectionKind::Internal => {
                if item_count > 20 {
                    insights.push("Large internal section - consider extracting helper modules".to_string());
                } else if item_count > 0 {
                    insights.push("Well-organized internal implementation".to_string());
                }
            },
            _ => {
                insights.push(format!("Section contains {} items", item_count));
            }
        }
        
        insights
    }
    
    /// Calculate naming consistency within a section
    fn calculate_section_naming_consistency(&self, section: &prism_ast::SectionDecl) -> CohesionResult<f64> {
        let names: Vec<String> = section.items.iter()
            .filter_map(|item| {
                match &item.kind {
                    prism_ast::Stmt::Function(func_decl) => Some(func_decl.name.to_string()),
                    prism_ast::Stmt::Type(type_decl) => Some(type_decl.name.to_string()),
                    prism_ast::Stmt::Variable(var_decl) => Some(var_decl.name.to_string()),
                    prism_ast::Stmt::Const(const_decl) => Some(const_decl.name.to_string()),
                    _ => None,
                }
            })
            .collect();
        
        if names.len() < 2 {
            return Ok(100.0); // Single item or no items = perfect consistency
        }
        
        // Calculate pairwise name similarity
        let mut total_similarity = 0.0;
        let mut comparison_count = 0;
        
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let similarity = strsim::jaro_winkler(&names[i], &names[j]);
                total_similarity += similarity;
                comparison_count += 1;
            }
        }
        
        let average_similarity = if comparison_count > 0 {
            total_similarity / comparison_count as f64
        } else {
            1.0
        };
        
        Ok(average_similarity * 100.0)
    }
    
    /// Calculate organization score for a section
    fn calculate_section_organization(&self, section: &prism_ast::SectionDecl) -> CohesionResult<f64> {
        let item_count = section.items.len();
        
        // Base organization score
        let mut score = 80.0;
        
        // Check if items are appropriate for section type
        let mut appropriate_items = 0;
        for item in &section.items {
            let is_appropriate = match (&section.kind, &item.kind) {
                (prism_ast::SectionKind::Types, prism_ast::Stmt::Type(_)) => true,
                (prism_ast::SectionKind::Interface, prism_ast::Stmt::Function(_)) => true,
                (prism_ast::SectionKind::Config, prism_ast::Stmt::Const(_)) => true,
                (prism_ast::SectionKind::Internal, _) => true, // Internal can contain various items
                _ => false,
            };
            
            if is_appropriate {
                appropriate_items += 1;
            }
        }
        
        if item_count > 0 {
            let appropriateness_ratio = appropriate_items as f64 / item_count as f64;
            score *= appropriateness_ratio;
        }
        
        Ok(score.max(0.0).min(100.0))
    }
    
    /// Detect patterns across modules
    fn detect_patterns(&self, _module_analyses: &[ModuleAnalysis], _context: &mut AnalysisContext) -> CohesionResult<Vec<String>> {
        // TODO: Implement pattern detection
        // This would identify common architectural patterns, anti-patterns, etc.
        Ok(vec!["Standard module organization pattern".to_string()])
    }
    
    /// Detect patterns within a single module
    fn detect_module_patterns(&self, module_decl: &ModuleDecl, _section_analyses: &[SectionAnalysis]) -> CohesionResult<Vec<String>> {
        let mut patterns = Vec::new();
        
        // Check for standard module patterns
        if module_decl.capability.is_some() {
            patterns.push("Capability-driven design".to_string());
        }
        
        if !module_decl.sections.is_empty() {
            patterns.push("Section-based organization".to_string());
        }
        
        if module_decl.ai_context.is_some() {
            patterns.push("AI-documented module".to_string());
        }
        
        Ok(patterns)
    }
    
    /// Calculate confidence in module analysis
    fn calculate_module_confidence(&self, module_decl: &ModuleDecl, section_analyses: &[SectionAnalysis]) -> f64 {
        let mut confidence = 0.7; // Base confidence
        
        // Boost confidence for well-documented modules
        if module_decl.description.is_some() {
            confidence += 0.1;
        }
        
        if module_decl.capability.is_some() {
            confidence += 0.1;
        }
        
        // Boost confidence for organized modules
        if !module_decl.sections.is_empty() {
            confidence += 0.1;
        }
        
        // Consider section analysis quality
        if !section_analyses.is_empty() {
            let avg_section_confidence = section_analyses.len() as f64 / 10.0; // Simple heuristic
            confidence += avg_section_confidence.min(0.1);
        }
        
        confidence.min(1.0)
    }
    
    /// Compile final analysis results
    fn compile_results(&self, overall_metrics: CohesionMetrics, module_analyses: Vec<ModuleAnalysis>, context: AnalysisContext) -> CohesionResult<AnalysisResult> {
        let total_duration = context.start_time.elapsed();
        
        let result_context = AnalysisResultContext {
            config: self.config.clone(),
            stats: context.stats,
            total_duration,
            timestamp: chrono::Utc::now().to_rfc3339(),
            analysis_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        Ok(AnalysisResult {
            success: true,
            metrics: overall_metrics,
            module_analyses,
            context: result_context,
            warnings: context.warnings,
        })
    }
    
    /// Generate cache key for program
    fn generate_program_cache_key(&self, program: &Program) -> String {
        format!("program_{}_{}", program.source_id.0, program.items.len())
    }
    
    /// Generate cache key for module
    fn generate_module_cache_key(&self, module_decl: &ModuleDecl) -> String {
        format!("module_{}_{}", module_decl.name, module_decl.sections.len())
    }

    /// Fast module extraction with pre-allocation
    fn extract_modules_fast<'a>(&self, program: &'a Program, context: &mut AnalysisContext) -> CohesionResult<Vec<(&'a AstNode<Item>, &'a ModuleDecl)>> {
        let mut modules = Vec::with_capacity(program.items.len()); // Pre-allocate
        
        for item in &program.items {
            if let Item::Module(module_decl) = &item.kind {
                modules.push((item, module_decl));
            }
        }
        
        context.stats.modules_analyzed = modules.len();
        Ok(modules)
    }
    
    /// Fast program metrics calculation with caching
    fn calculate_program_metrics_fast(&mut self, program: &Program, context: &mut AnalysisContext) -> CohesionResult<CohesionMetrics> {
        let start_time = Instant::now();
        
        // Generate fast cache key
        let cache_key = format!("prog_{}_{}", program.source_id.0, program.items.len());
        
        // Check cache first
        if let Some(cached_metrics) = self.cache.get_metrics(&cache_key) {
            context.stats.cache_hits += 1;
            return Ok(cached_metrics.clone());
        }
        
        // Calculate metrics using optimized calculator
        let metrics = self.metrics_calculator.calculate_program_metrics(program)?;
        
        // Cache results
        self.cache.cache_metrics(cache_key, metrics.clone());
        
        let calculation_time = start_time.elapsed();
        context.stats.phase_times.insert(AnalysisPhase::MetricsCalculation, calculation_time.as_millis() as u64);
        
        // Performance monitoring
        if calculation_time.as_millis() > self.performance_monitor.warning_threshold_ms.into() {
            context.add_warning(AnalysisWarning {
                message: format!("Metrics calculation took {}ms (longer than {}ms threshold)", 
                               calculation_time.as_millis(), self.performance_monitor.warning_threshold_ms),
                category: WarningCategory::Performance,
                location: None,
                phase: AnalysisPhase::MetricsCalculation,
            });
        }
        
        Ok(metrics)
    }
    
    /// Fast individual module analysis
    fn analyze_individual_module_fast(&mut self, module_item: &AstNode<Item>, module_decl: &ModuleDecl, context: &mut AnalysisContext) -> CohesionResult<ModuleAnalysis> {
        let start_time = Instant::now();
        
        // Fast cache key
        let cache_key = format!("mod_{}_{}", module_decl.name, module_decl.sections.len());
        
        // Check cache
        if let Some(cached_analysis) = self.cache.get_module_analysis(&cache_key) {
            context.stats.cache_hits += 1;
            return Ok(cached_analysis.clone());
        }
        
        // Calculate module metrics (optimized)
        let module_metrics = self.metrics_calculator.calculate_module_metrics(module_item, module_decl)?;
        
        // Analyze sections (optimized)
        let mut section_analyses = Vec::with_capacity(module_decl.sections.len());
        for section in &module_decl.sections {
            let section_analysis = self.analyze_section_fast(section, context)?;
            section_analyses.push(section_analysis);
        }
        
        // Fast pattern detection
        let patterns = self.detect_module_patterns_fast(module_decl, &section_analyses);
        
        // Fast confidence calculation
        let confidence = self.calculate_module_confidence_fast(module_decl, &section_analyses);
        
        let module_analysis = ModuleAnalysis {
            module_name: module_decl.name.to_string(),
            metrics: module_metrics,
            section_analyses,
            patterns,
            confidence,
        };
        
        // Cache results
        self.cache.cache_module_analysis(cache_key, module_analysis.clone());
        
        context.stats.sections_analyzed += module_decl.sections.len();
        
        Ok(module_analysis)
    }
    
    /// Fast section analysis
    fn analyze_section_fast(&self, section: &AstNode<prism_ast::SectionDecl>, _context: &mut AnalysisContext) -> CohesionResult<SectionAnalysis> {
        let section_name = format!("{:?}", section.kind.kind);
        let item_count = section.kind.items.len();
        
        // Fast cohesion calculation using lookup table
        let cohesion_score = match section.kind.kind {
            prism_ast::SectionKind::Types => {
                if item_count <= 10 { 90.0 } else { 80.0 }
            },
            prism_ast::SectionKind::Interface => {
                if item_count > 0 { 95.0 } else { 50.0 }
            },
            prism_ast::SectionKind::Internal => {
                if item_count <= 15 { 85.0 } else { 70.0 }
            },
            prism_ast::SectionKind::Config => 95.0,
            prism_ast::SectionKind::Events => 85.0,
            _ => 75.0,
        };
        
        // Fast insights generation
        let insights = match section.kind.kind {
            prism_ast::SectionKind::Types if item_count > 15 => {
                vec!["Consider splitting large type section".to_string()]
            },
            prism_ast::SectionKind::Interface if item_count == 0 => {
                vec!["Empty interface section - consider adding public API".to_string()]
            },
            _ => vec![format!("Section contains {} items", item_count)],
        };
        
        // Fast metrics calculation
        let mut metrics = FxHashMap::with_capacity_and_hasher(4, Default::default());
        metrics.insert("item_count".to_string(), item_count as f64);
        metrics.insert("section_score".to_string(), cohesion_score);
        
        Ok(SectionAnalysis {
            section_name,
            cohesion_score,
            item_count,
            insights,
            metrics,
        })
    }
    
    /// Fast module pattern detection
    fn detect_module_patterns_fast(&self, module_decl: &ModuleDecl, _section_analyses: &[SectionAnalysis]) -> Vec<String> {
        let mut patterns = Vec::with_capacity(4);
        
        if module_decl.capability.is_some() {
            patterns.push("Capability-driven design".to_string());
        }
        
        if !module_decl.sections.is_empty() {
            patterns.push("Section-based organization".to_string());
        }
        
        if module_decl.ai_context.is_some() {
            patterns.push("AI-documented module".to_string());
        }
        
        // Fast section analysis
        let has_types = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Types));
        let has_interface = module_decl.sections.iter()
            .any(|s| matches!(s.kind.kind, prism_ast::SectionKind::Interface));
        
        if has_types && has_interface {
            patterns.push("Well-structured API module".to_string());
        }
        
        patterns
    }
    
    /// Fast confidence calculation
    fn calculate_module_confidence_fast(&self, module_decl: &ModuleDecl, section_analyses: &[SectionAnalysis]) -> f64 {
        let mut confidence: f64 = 0.6; // Base confidence
        
        // Quick bonuses
        if module_decl.description.is_some() { confidence += 0.1; }
        if module_decl.capability.is_some() { confidence += 0.15; }
        if !module_decl.sections.is_empty() { confidence += 0.1; }
        if section_analyses.len() > 2 { confidence += 0.05; }
        
        confidence.min(1.0)
    }
}

impl AnalysisConfig {
    /// Create configuration from cohesion config
    pub fn from_cohesion_config(cohesion_config: &crate::CohesionConfig) -> Self {
        Self {
            depth: cohesion_config.analysis_depth,
            enable_semantic_integration: true,
            enable_pattern_detection: cohesion_config.analysis_depth != AnalysisDepth::Quick,
            enable_ai_context: cohesion_config.enable_ai_insights,
            enable_real_time: false, // Default to batch analysis
            max_analysis_time_ms: 10000, // 10 second default timeout
            min_confidence: 0.5,
        }
    }
    
    /// Create quick analysis configuration
    pub fn quick() -> Self {
        Self {
            depth: AnalysisDepth::Quick,
            enable_semantic_integration: false,
            enable_pattern_detection: false,
            enable_ai_context: false,
            enable_real_time: false,
            max_analysis_time_ms: 1000,
            min_confidence: 0.3,
        }
    }
    
    /// Create comprehensive analysis configuration
    pub fn comprehensive() -> Self {
        Self {
            depth: AnalysisDepth::Comprehensive,
            enable_semantic_integration: true,
            enable_pattern_detection: true,
            enable_ai_context: true,
            enable_real_time: false,
            max_analysis_time_ms: 30000,
            min_confidence: 0.8,
        }
    }
}

impl AnalysisContext {
    /// Create new analysis context
    pub fn new(source_id: prism_common::SourceId) -> Self {
        Self {
            source_id,
            start_time: Instant::now(),
            current_phase: AnalysisPhase::Initialization,
            warnings: Vec::new(),
            stats: AnalysisStats::default(),
        }
    }
    
    /// Enter a new analysis phase
    pub fn enter_phase(&mut self, phase: AnalysisPhase) {
        let now = Instant::now();
        let phase_duration = now.duration_since(self.start_time).as_millis() as u64;
        
        // Record time for previous phase
        if let Some(previous_duration) = self.stats.phase_times.get(&self.current_phase) {
            self.stats.phase_times.insert(self.current_phase.clone(), phase_duration - previous_duration);
        } else {
            self.stats.phase_times.insert(self.current_phase.clone(), phase_duration);
        }
        
        self.current_phase = phase;
    }
    
    /// Add warning to context
    pub fn add_warning(&mut self, warning: AnalysisWarning) {
        self.warnings.push(warning);
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    fn new() -> Self {
        Self {
            phase_timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            warning_threshold_ms: 5000, // 5 second warning threshold
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            depth: AnalysisDepth::Standard,
            enable_semantic_integration: true,
            enable_pattern_detection: true,
            enable_ai_context: true,
            enable_real_time: false,
            max_analysis_time_ms: 10000,
            min_confidence: 0.6,
        }
    }
} 