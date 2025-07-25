//! Analysis Pipeline Coordinator
//!
//! This module provides the central coordinator for running analysis passes in the correct
//! dependency order, managing shared state, and collecting results.

use super::shared::{Analysis, AnalysisKind, AnalysisError, AnalysisMetadata, OptimizationOpportunity};
use crate::{VMResult, PrismVMError, bytecode::FunctionDefinition};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use std::sync::Arc;

/// Configuration for the analysis pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enable specific analysis passes
    pub enabled_analyses: HashSet<AnalysisKind>,
    /// Global analysis timeout
    pub analysis_timeout: Duration,
    /// Maximum analysis iterations for iterative analyses
    pub max_iterations: usize,
    /// Enable parallel analysis execution where possible
    pub enable_parallel: bool,
    /// Optimization detection configuration
    pub optimization_config: OptimizationConfig,
}

/// Configuration for optimization detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable optimization opportunity detection
    pub enable_detection: bool,
    /// Minimum benefit threshold for reporting opportunities
    pub min_benefit_threshold: f64,
    /// Maximum cost threshold for considering optimizations
    pub max_cost_threshold: f64,
    /// Enable speculative optimizations
    pub enable_speculative: bool,
    /// Aggressiveness level (0.0 to 1.0)
    pub aggressiveness: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut enabled_analyses = HashSet::new();
        enabled_analyses.insert(AnalysisKind::ControlFlow);
        enabled_analyses.insert(AnalysisKind::DataFlow);
        enabled_analyses.insert(AnalysisKind::Loop);
        enabled_analyses.insert(AnalysisKind::Type);
        enabled_analyses.insert(AnalysisKind::Effect);
        enabled_analyses.insert(AnalysisKind::Hotness);
        enabled_analyses.insert(AnalysisKind::Capability);
        enabled_analyses.insert(AnalysisKind::CapabilityAwareInlining);

        Self {
            enabled_analyses,
            analysis_timeout: Duration::from_millis(500),
            max_iterations: 100,
            enable_parallel: true,
            optimization_config: OptimizationConfig {
                enable_detection: true,
                min_benefit_threshold: 0.05,
                max_cost_threshold: 0.2,
                enable_speculative: true,
                aggressiveness: 0.5,
            },
        }
    }
}

/// Analysis execution context containing shared state
#[derive(Debug, Clone)]
pub struct AnalysisContext {
    /// Function being analyzed
    pub function: Arc<FunctionDefinition>,
    /// Analysis results by kind
    pub results: HashMap<AnalysisKind, Box<dyn std::any::Any + Send + Sync>>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
    /// Detected optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

impl AnalysisContext {
    /// Create a new analysis context for a function
    pub fn new(function: FunctionDefinition) -> Self {
        Self {
            function: Arc::new(function),
            results: HashMap::new(),
            metadata: AnalysisMetadata::default(),
            optimization_opportunities: Vec::new(),
        }
    }

    /// Get analysis result of a specific type
    pub fn get_result<T: 'static + Clone>(&self, kind: AnalysisKind) -> Option<T> {
        self.results.get(&kind)
            .and_then(|any| any.downcast_ref::<T>())
            .cloned()
    }

    /// Store analysis result
    pub fn store_result<T: 'static + Clone + Send + Sync>(&mut self, kind: AnalysisKind, result: T) {
        self.results.insert(kind, Box::new(result));
    }

    /// Check if analysis result is available
    pub fn has_result(&self, kind: AnalysisKind) -> bool {
        self.results.contains_key(&kind)
    }

    /// Add optimization opportunities
    pub fn add_opportunities(&mut self, mut opportunities: Vec<OptimizationOpportunity>) {
        self.optimization_opportunities.append(&mut opportunities);
    }
}

/// Analysis pipeline coordinator
pub struct AnalysisPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Registered analyzers
    analyzers: HashMap<AnalysisKind, Box<dyn AnalyzerWrapper>>,
    /// Dependency graph
    dependency_graph: DependencyGraph,
    /// Optimization detector
    optimization_detector: OptimizationDetector,
}

/// Wrapper trait for type-erased analyzers
trait AnalyzerWrapper: Send + Sync {
    fn analyze(&mut self, context: &mut AnalysisContext) -> VMResult<()>;
    fn dependencies(&self) -> Vec<AnalysisKind>;
    fn analysis_kind(&self) -> AnalysisKind;
}

/// Generic wrapper for analyzers implementing the Analysis trait
struct GenericAnalyzerWrapper<A: Analysis> {
    analyzer: A,
}

impl<A: Analysis> GenericAnalyzerWrapper<A> {
    fn new(analyzer: A) -> Self {
        Self { analyzer }
    }
}

impl<A: Analysis + 'static> AnalyzerWrapper for GenericAnalyzerWrapper<A>
where
    A::Result: 'static + Clone + Send + Sync,
    A::Dependencies: for<'a> From<&'a AnalysisContext>,
{
    fn analyze(&mut self, context: &mut AnalysisContext) -> VMResult<()> {
        let start_time = Instant::now();
        
        // Build dependencies from context
        let deps = A::Dependencies::from(context);
        
        // Validate dependencies
        A::validate_dependencies(&deps)?;
        
        // Run analysis
        let result = self.analyzer.analyze(&context.function, deps)?;
        
        // Store result
        context.store_result(A::analysis_kind(), result);
        
        // Update metadata
        let analysis_time = start_time.elapsed();
        context.metadata.analysis_time += analysis_time;
        context.metadata.passes_run.push(A::analysis_kind().to_string());
        
        Ok(())
    }

    fn dependencies(&self) -> Vec<AnalysisKind> {
        A::dependencies()
    }

    fn analysis_kind(&self) -> AnalysisKind {
        A::analysis_kind()
    }
}

/// Dependency graph for analysis ordering
#[derive(Debug, Clone)]
struct DependencyGraph {
    /// Adjacency list representation
    graph: HashMap<AnalysisKind, Vec<AnalysisKind>>,
    /// Reverse dependencies (dependents)
    reverse_graph: HashMap<AnalysisKind, Vec<AnalysisKind>>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    fn new() -> Self {
        Self {
            graph: HashMap::new(),
            reverse_graph: HashMap::new(),
        }
    }

    /// Add a dependency edge
    fn add_dependency(&mut self, dependent: AnalysisKind, dependency: AnalysisKind) {
        self.graph.entry(dependent).or_default().push(dependency);
        self.reverse_graph.entry(dependency).or_default().push(dependent);
    }

    /// Compute topological ordering of analyses
    fn topological_sort(&self, enabled: &HashSet<AnalysisKind>) -> Result<Vec<AnalysisKind>, AnalysisError> {
        let mut in_degree: HashMap<AnalysisKind, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Initialize in-degrees for enabled analyses
        for &kind in enabled {
            let deps = self.graph.get(&kind).map(|v| v.len()).unwrap_or(0);
            let filtered_deps = self.graph.get(&kind)
                .map(|deps| deps.iter().filter(|&&dep| enabled.contains(&dep)).count())
                .unwrap_or(0);
            
            in_degree.insert(kind, filtered_deps);
            if filtered_deps == 0 {
                queue.push_back(kind);
            }
        }

        // Kahn's algorithm
        while let Some(current) = queue.pop_front() {
            result.push(current);

            if let Some(dependents) = self.reverse_graph.get(&current) {
                for &dependent in dependents {
                    if !enabled.contains(&dependent) {
                        continue;
                    }

                    if let Some(degree) = in_degree.get_mut(&dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != enabled.len() {
            let remaining: Vec<AnalysisKind> = enabled.iter()
                .filter(|&&kind| !result.contains(&kind))
                .copied()
                .collect();
            return Err(AnalysisError::DependencyCycle(remaining));
        }

        Ok(result)
    }
}

/// Optimization opportunity detector
pub struct OptimizationDetector {
    /// Detection configuration
    config: OptimizationConfig,
}

impl OptimizationDetector {
    /// Create a new optimization detector
    fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Detect optimization opportunities from analysis results
    pub fn detect_opportunities(&mut self, context: &AnalysisContext) -> VMResult<Vec<OptimizationOpportunity>> {
        if !self.config.enable_detection {
            return Ok(Vec::new());
        }

        let mut opportunities = Vec::new();
        let mut opportunity_id = 0;

        // Detect opportunities from each analysis type
        opportunities.extend(self.detect_control_flow_opportunities(context, &mut opportunity_id)?);
        opportunities.extend(self.detect_data_flow_opportunities(context, &mut opportunity_id)?);
        opportunities.extend(self.detect_loop_opportunities(context, &mut opportunity_id)?);
        opportunities.extend(self.detect_type_opportunities(context, &mut opportunity_id)?);
        opportunities.extend(self.detect_effect_opportunities(context, &mut opportunity_id)?);
        opportunities.extend(self.detect_hotness_opportunities(context, &mut opportunity_id)?);
        opportunities.extend(self.detect_capability_opportunities(context, &mut opportunity_id)?);

        // Filter and rank opportunities
        self.filter_and_rank_opportunities(&mut opportunities);

        Ok(opportunities)
    }

    /// Filter and rank opportunities based on profitability
    fn filter_and_rank_opportunities(&self, opportunities: &mut Vec<OptimizationOpportunity>) {
        // Filter based on thresholds
        opportunities.retain(|opp| {
            opp.estimated_benefit >= self.config.min_benefit_threshold &&
            opp.implementation_cost <= self.config.max_cost_threshold &&
            opp.is_profitable()
        });

        // Sort by risk-adjusted benefit (descending)
        opportunities.sort_by(|a, b| {
            b.risk_adjusted_benefit()
                .partial_cmp(&a.risk_adjusted_benefit())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Opportunity detection methods for each analysis type
    fn detect_control_flow_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect CFG-based opportunities like unreachable code elimination
        Ok(Vec::new())
    }

    fn detect_data_flow_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect data flow opportunities like dead code elimination
        Ok(Vec::new())
    }

    fn detect_loop_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect loop optimization opportunities
        Ok(Vec::new())
    }

    fn detect_type_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect type specialization opportunities
        Ok(Vec::new())
    }

    fn detect_effect_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect effect-based optimization opportunities
        Ok(Vec::new())
    }

    fn detect_hotness_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect hotness-based optimization opportunities
        Ok(Vec::new())
    }

    fn detect_capability_opportunities(&self, _context: &AnalysisContext, _id: &mut u32) -> VMResult<Vec<OptimizationOpportunity>> {
        // Implementation would detect capability-aware optimization opportunities
        Ok(Vec::new())
    }
}

impl AnalysisPipeline {
    /// Create a new analysis pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            optimization_detector: OptimizationDetector::new(config.optimization_config.clone()),
            config,
            analyzers: HashMap::new(),
            dependency_graph: DependencyGraph::new(),
        }
    }

    /// Register an analyzer with the pipeline
    pub fn register_analyzer<A: Analysis + 'static>(&mut self, analyzer: A) -> VMResult<()>
    where
        A::Result: 'static + Clone + Send + Sync,
        A::Dependencies: for<'a> From<&'a AnalysisContext>,
    {
        let kind = A::analysis_kind();
        let dependencies = A::dependencies();

        // Add dependencies to graph
        for dep in &dependencies {
            self.dependency_graph.add_dependency(kind, *dep);
        }

        // Store analyzer
        let wrapper = GenericAnalyzerWrapper::new(analyzer);
        self.analyzers.insert(kind, Box::new(wrapper));

        Ok(())
    }

    /// Run the analysis pipeline on a function
    pub fn analyze_function(&mut self, function: FunctionDefinition) -> VMResult<AnalysisContext> {
        let start_time = Instant::now();
        let mut context = AnalysisContext::new(function);

        // Compute execution order
        let execution_order = self.dependency_graph
            .topological_sort(&self.config.enabled_analyses)
            .map_err(|e| PrismVMError::AnalysisError(e.to_string()))?;

        // Execute analyses in dependency order
        for kind in execution_order {
            if let Some(analyzer) = self.analyzers.get_mut(&kind) {
                // Check timeout
                if start_time.elapsed() > self.config.analysis_timeout {
                    context.metadata.warnings.push(format!("Analysis timeout reached, skipping {}", kind));
                    break;
                }

                // Run analysis
                match analyzer.analyze(&mut context) {
                    Ok(()) => {
                        // Analysis completed successfully
                    }
                    Err(e) => {
                        context.metadata.warnings.push(format!("Analysis {} failed: {}", kind, e));
                        // Continue with other analyses
                    }
                }
            }
        }

        // Detect optimization opportunities
        match self.optimization_detector.detect_opportunities(&context) {
            Ok(opportunities) => {
                context.optimization_opportunities = opportunities;
            }
            Err(e) => {
                context.metadata.warnings.push(format!("Optimization detection failed: {}", e));
            }
        }

        // Finalize metadata
        context.metadata.analysis_time = start_time.elapsed();
        context.metadata.confidence = self.calculate_confidence(&context);

        Ok(context)
    }

    /// Calculate overall analysis confidence
    fn calculate_confidence(&self, context: &AnalysisContext) -> f64 {
        let total_analyses = self.config.enabled_analyses.len();
        let completed_analyses = context.results.len();
        
        let base_confidence = completed_analyses as f64 / total_analyses.max(1) as f64;
        
        // Adjust for warnings
        let warning_penalty = context.metadata.warnings.len() as f64 * 0.1;
        
        (base_confidence - warning_penalty).max(0.0).min(1.0)
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> PipelineStatistics {
        PipelineStatistics {
            registered_analyzers: self.analyzers.len(),
            enabled_analyses: self.config.enabled_analyses.clone(),
            dependency_edges: self.dependency_graph.graph.values()
                .map(|deps| deps.len())
                .sum(),
        }
    }
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    /// Number of registered analyzers
    pub registered_analyzers: usize,
    /// Set of enabled analyses
    pub enabled_analyses: HashSet<AnalysisKind>,
    /// Total number of dependency edges
    pub dependency_edges: usize,
}

/// Trait for converting analysis context to dependency types
/// This will be implemented by each analysis's dependency type
pub trait FromAnalysisContext {
    fn from_context(context: &AnalysisContext) -> Self;
}

// We'll implement this for common dependency patterns
impl FromAnalysisContext for () {
    fn from_context(_context: &AnalysisContext) -> Self {
        ()
    }
}

// Example implementation for analyses that need CFG
impl<T> From<&AnalysisContext> for Option<T>
where
    T: 'static + Clone,
{
    fn from(context: &AnalysisContext) -> Self {
        // This is a generic implementation - specific analyses would implement their own
        None
    }
} 