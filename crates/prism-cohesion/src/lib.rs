//! Conceptual cohesion analysis and metrics for Prism language.
//!
//! This crate implements PLD-002: Smart Module System & Conceptual Cohesion,
//! providing comprehensive analysis of how well code components relate to each
//! other conceptually. It measures various types of cohesion and provides
//! actionable suggestions for improving code organization.
//!
//! ## Features
//!
//! - **Conceptual Cohesion Metrics**: Multi-dimensional cohesion analysis
//! - **Boundary Detection**: Automatic detection of conceptual boundaries
//! - **Responsibility Analysis**: Analysis of single responsibility principle
//! - **AI-Driven Insights**: AI-comprehensible analysis and suggestions
//! - **Real-time Analysis**: Incremental cohesion tracking during development
//! - **Violation Detection**: Detection and reporting of cohesion violations
//! - **Restructuring Integration**: Automatic module restructuring based on metrics
//!
//! ## Architecture
//!
//! The cohesion analysis system follows conceptual cohesion principles with
//! specialized modules for each type of analysis:
//!
//! - `metrics/` - Core cohesion metric calculations
//! - `analysis/` - High-level cohesion analysis orchestration
//! - `boundaries/` - Conceptual boundary detection and validation
//! - `violations/` - Cohesion violation detection and reporting
//! - `suggestions/` - Improvement suggestion generation
//! - `ai_insights/` - AI-comprehensible analysis and metadata
//! - `restructuring/` - Module restructuring integration and execution
//!
//! ## Cohesion Types
//!
//! The system analyzes multiple dimensions of cohesion:
//!
//! - **Type Cohesion**: How well types relate to each other
//! - **Data Flow Cohesion**: How smoothly data flows between functions
//! - **Semantic Cohesion**: How similar names and concepts are
//! - **Business Cohesion**: How focused the business capability is
//! - **Dependency Cohesion**: How focused external dependencies are
//!
//! ## Examples
//!
//! ```rust
//! use prism_cohesion::{CohesionAnalyzer, AnalysisConfig};
//! use prism_ast::Program;
//!
//! let analyzer = CohesionAnalyzer::new(AnalysisConfig::comprehensive());
//! let analysis_result = analyzer.analyze_program(&program)?;
//!
//! println!("Overall cohesion score: {:.1}", analysis_result.overall_score);
//! for suggestion in analysis_result.improvement_suggestions {
//!     println!("Suggestion: {}", suggestion.description);
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Re-export main types for public API
pub use analysis::{CohesionAnalyzer, AnalysisConfig, AnalysisResult};
pub use metrics::{MetricsCalculator, CohesionMetrics, CohesionAnalysis, MetricsMetadata};
pub use boundaries::{BoundaryDetector, ConceptualBoundary, BoundaryType};
pub use violations::{ViolationDetector, CohesionViolation, ViolationType, ViolationSeverity};
pub use suggestions::{SuggestionEngine, CohesionSuggestion, SuggestionType, EffortLevel};
pub use ai_insights::{AIInsightGenerator, CohesionAIInsights};
pub use confidence::{ConfidenceCalculator, ConfidenceBreakdown, ConfidenceLevel};

// Re-export restructuring types for convenience
pub use restructuring::{
    CohesionRestructuringSystem, RestructuringSystemConfig, RestructuringAnalysis,
    SafetyAnalysis, RestructuringRisk, RiskCategory, ImpactEstimate
};

// Main modules - each with a single responsibility
pub mod analysis;
pub mod metrics;
pub mod boundaries;
pub mod violations;
pub mod suggestions;
pub mod ai_insights;
pub mod confidence;
pub mod restructuring;  // NEW: Module restructuring integration

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during cohesion analysis
#[derive(Debug, thiserror::Error)]
pub enum CohesionError {
    /// Analysis error
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    
    /// Metric calculation error
    #[error("Metric calculation error: {0}")]
    MetricError(String),
    
    /// Boundary detection error
    #[error("Boundary detection error: {0}")]
    BoundaryError(String),
    
    /// Violation detection error
    #[error("Violation detection error: {0}")]
    ViolationError(String),
    
    /// Suggestion generation error
    #[error("Suggestion generation error: {0}")]
    SuggestionError(String),
    
    /// AI integration error
    #[error("AI integration error: {0}")]
    AIError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Insufficient data error
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    /// Invalid configuration error
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Safety violation error
    #[error("Safety violation: {0}")]
    SafetyViolation(String),
    
    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    /// Integration error
    #[error("Integration error: {0}")]
    IntegrationError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type for cohesion analysis
pub type CohesionResult<T> = Result<T, CohesionError>;

/// Configuration for cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionConfig {
    /// Analysis depth and comprehensiveness
    pub analysis_depth: AnalysisDepth,

    /// Minimum acceptable cohesion score (0-100)
    pub minimum_acceptable_score: f64,

    /// Enable AI-driven insights
    pub enable_ai_insights: bool,

    /// Enable violation detection
    pub enable_violation_detection: bool,

    /// Enable improvement suggestions
    pub enable_suggestions: bool,

    /// Weights for different cohesion metrics
    pub metric_weights: MetricWeights,

    /// Thresholds for violation detection
    pub violation_thresholds: ViolationThresholds,

    /// Custom analysis rules
    pub custom_rules: Vec<String>,
    
    /// Performance optimization level
    pub optimization_level: OptimizationLevel,
}

/// Analysis depth levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Quick analysis (essential metrics only)
    Quick,
    /// Standard analysis (most metrics)
    Standard,
    /// Comprehensive analysis (all metrics and insights)
    Comprehensive,
    /// Deep analysis (includes advanced AI insights)
    Deep,
}

/// Weights for different cohesion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    /// Weight for type cohesion (default: 0.25)
    pub type_cohesion: f64,
    /// Weight for data flow cohesion (default: 0.25)
    pub data_flow_cohesion: f64,
    /// Weight for semantic cohesion (default: 0.30)
    pub semantic_cohesion: f64,
    /// Weight for business cohesion (default: 0.15)
    pub business_cohesion: f64,
    /// Weight for dependency cohesion (default: 0.05)
    pub dependency_cohesion: f64,
}

/// Thresholds for violation detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationThresholds {
    /// Minimum score before error violation (default: 40.0)
    pub error_threshold: f64,
    /// Minimum score before warning violation (default: 60.0)
    pub warning_threshold: f64,
    /// Minimum score before info violation (default: 80.0)
    pub info_threshold: f64,
}

/// Performance optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Maximum speed, reduced accuracy (good for real-time IDE analysis)
    Speed,
    /// Balanced speed and accuracy (default)
    Balanced,
    /// Maximum accuracy, slower analysis (good for CI/CD)
    Accuracy,
}

/// Complete cohesion analysis system
#[derive(Debug)]
pub struct CohesionSystem {
    /// Configuration
    config: CohesionConfig,

    /// Core analyzer
    analyzer: CohesionAnalyzer,

    /// Metrics calculator
    metrics_calculator: MetricsCalculator,

    /// Boundary detector
    boundary_detector: BoundaryDetector,

    /// Violation detector
    violation_detector: ViolationDetector,

    /// Suggestion engine
    suggestion_engine: SuggestionEngine,

    /// AI insight generator
    ai_insight_generator: AIInsightGenerator,
}

impl CohesionSystem {
    /// Create a new cohesion system with default configuration
    pub fn new() -> Self {
        let config = CohesionConfig::default();
        Self::with_config(config)
    }

    /// Create a new cohesion system with custom configuration
    pub fn with_config(config: CohesionConfig) -> Self {
        Self {
            analyzer: CohesionAnalyzer::new(AnalysisConfig::from_cohesion_config(&config)),
            metrics_calculator: MetricsCalculator::new(config.metric_weights.clone()),
            boundary_detector: BoundaryDetector::new(),
            violation_detector: ViolationDetector::new(config.violation_thresholds.clone()),
            suggestion_engine: SuggestionEngine::new(),
            ai_insight_generator: AIInsightGenerator::new(),
            config,
        }
    }

    /// Analyze cohesion for a complete program
    pub fn analyze_program(&mut self, program: &prism_ast::Program) -> CohesionResult<ProgramCohesionAnalysis> {
        // Step 1: Core cohesion analysis
        let analysis_result = self.analyzer.analyze_program(program)?;

        // Step 2: Calculate detailed metrics
        let detailed_metrics = self.metrics_calculator.calculate_program_metrics(program)?;

        // Step 3: Detect conceptual boundaries
        let boundaries = if self.config.analysis_depth != AnalysisDepth::Quick {
            self.boundary_detector.detect_program_boundaries(program, &analysis_result)?
        } else {
            Vec::new()
        };

        // Step 4: Detect violations
        let violations = if self.config.enable_violation_detection {
            self.violation_detector.detect_program_violations(program, &detailed_metrics)?
        } else {
            Vec::new()
        };

        // Step 5: Generate improvement suggestions
        let suggestions = if self.config.enable_suggestions {
            self.suggestion_engine.generate_program_suggestions(program, &detailed_metrics, &violations)?
        } else {
            Vec::new()
        };

        // Step 6: Generate AI insights
        let ai_insights = if self.config.enable_ai_insights {
            Some(self.ai_insight_generator.generate_program_insights(
                program, 
                &detailed_metrics, 
                &boundaries, 
                &violations
            )?)
        } else {
            None
        };

        Ok(ProgramCohesionAnalysis {
            overall_score: detailed_metrics.overall_score,
            module_analyses: analysis_result.module_analyses.into_iter().map(|ma| ModuleCohesionAnalysis {
                module_name: ma.module_name,
                overall_score: ma.metrics.overall_score,
                metrics: ma.metrics,
                conceptual_boundaries: Vec::new(), // TODO: Extract from boundaries
                violations: Vec::new(), // TODO: Extract module-specific violations
                improvement_suggestions: Vec::new(), // TODO: Extract module-specific suggestions
                section_analyses: ma.section_analyses.into_iter().map(|sa| SectionCohesionAnalysis {
                    section_name: sa.section_name,
                    cohesion_score: sa.cohesion_score,
                    item_count: sa.item_count,
                    section_metrics: sa.metrics,
                }).collect(),
            }).collect(),
            detailed_metrics,
            conceptual_boundaries: boundaries,
            violations,
            improvement_suggestions: suggestions,
            ai_insights,
            analysis_metadata: AnalysisMetadata {
                analysis_depth: self.config.analysis_depth,
                total_items_analyzed: program.items.len(),
                analysis_duration: std::time::Duration::from_millis(0), // TODO: Track actual duration
                config_hash: self.calculate_config_hash(),
            },
        })
    }

    /// Analyze cohesion for a single module
    pub fn analyze_module(&mut self, module: &prism_ast::AstNode<prism_ast::Item>) -> CohesionResult<ModuleCohesionAnalysis> {
        // Extract module if the item is a module
        if let prism_ast::Item::Module(module_decl) = &module.kind {
            let module_result = self.analyzer.analyze_module(module, &module_decl)?;
            let module_metrics = self.metrics_calculator.calculate_module_metrics(module, &module_decl)?;

            let boundaries = if self.config.analysis_depth != AnalysisDepth::Quick {
                self.boundary_detector.detect_module_boundaries(module, &module_decl)?
            } else {
                Vec::new()
            };

            let violations = if self.config.enable_violation_detection {
                self.violation_detector.detect_module_violations(module, &module_metrics)?
            } else {
                Vec::new()
            };

            let suggestions = if self.config.enable_suggestions {
                self.suggestion_engine.generate_module_suggestions(module, &module_metrics, &violations)?
            } else {
                Vec::new()
            };

            Ok(ModuleCohesionAnalysis {
                module_name: module_decl.name.to_string(),
                overall_score: module_metrics.overall_score,
                metrics: module_metrics,
                conceptual_boundaries: boundaries,
                violations,
                improvement_suggestions: suggestions,
                section_analyses: module_result.section_analyses.into_iter().map(|sa| SectionCohesionAnalysis {
                    section_name: sa.section_name,
                    cohesion_score: sa.cohesion_score,
                    item_count: sa.item_count,
                    section_metrics: sa.metrics,
                }).collect(),
            })
        } else {
            Err(CohesionError::InvalidConfiguration {
                reason: "Item is not a module".to_string(),
            })
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &CohesionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: CohesionConfig) {
        self.config = config.clone();
        self.analyzer = CohesionAnalyzer::new(AnalysisConfig::from_cohesion_config(&config));
        self.metrics_calculator = MetricsCalculator::new(config.metric_weights.clone());
        self.violation_detector = ViolationDetector::new(config.violation_thresholds.clone());
    }

    /// Calculate configuration hash for caching
    fn calculate_config_hash(&self) -> u64 {
        // TODO: Implement proper config hashing
        0
    }
}

/// Result of program-level cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramCohesionAnalysis {
    /// Overall program cohesion score (0-100)
    pub overall_score: f64,

    /// Individual module analyses
    pub module_analyses: Vec<ModuleCohesionAnalysis>,

    /// Detailed metrics breakdown
    pub detailed_metrics: CohesionMetrics,

    /// Detected conceptual boundaries
    pub conceptual_boundaries: Vec<ConceptualBoundary>,

    /// Detected violations
    pub violations: Vec<CohesionViolation>,

    /// Improvement suggestions
    pub improvement_suggestions: Vec<CohesionSuggestion>,

    /// AI-generated insights (if enabled)
    pub ai_insights: Option<CohesionAIInsights>,

    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
}

/// Result of module-level cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleCohesionAnalysis {
    /// Module name
    pub module_name: String,

    /// Overall module cohesion score (0-100)
    pub overall_score: f64,

    /// Detailed metrics for the module
    pub metrics: CohesionMetrics,

    /// Conceptual boundaries within the module
    pub conceptual_boundaries: Vec<ConceptualBoundary>,

    /// Violations found in the module
    pub violations: Vec<CohesionViolation>,

    /// Improvement suggestions for the module
    pub improvement_suggestions: Vec<CohesionSuggestion>,

    /// Section-level analyses
    pub section_analyses: Vec<SectionCohesionAnalysis>,
}

/// Result of section-level cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionCohesionAnalysis {
    /// Section name/type
    pub section_name: String,

    /// Section cohesion score (0-100)
    pub cohesion_score: f64,

    /// Items in this section
    pub item_count: usize,

    /// Section-specific metrics
    pub section_metrics: HashMap<String, f64>,
}

/// Metadata about the analysis process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Depth of analysis performed
    pub analysis_depth: AnalysisDepth,

    /// Total number of items analyzed
    pub total_items_analyzed: usize,

    /// Duration of the analysis
    pub analysis_duration: std::time::Duration,

    /// Hash of configuration used
    pub config_hash: u64,
}

impl Default for CohesionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CohesionConfig {
    fn default() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Standard,
            minimum_acceptable_score: 70.0,
            enable_ai_insights: true,
            enable_violation_detection: true,
            enable_suggestions: true,
            metric_weights: MetricWeights::default(),
            violation_thresholds: ViolationThresholds::default(),
            custom_rules: Vec::new(),
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            type_cohesion: 0.25,
            data_flow_cohesion: 0.25,
            semantic_cohesion: 0.30,
            business_cohesion: 0.15,
            dependency_cohesion: 0.05,
        }
    }
}

impl Default for ViolationThresholds {
    fn default() -> Self {
        Self {
            error_threshold: 40.0,
            warning_threshold: 60.0,
            info_threshold: 80.0,
        }
    }
}

impl CohesionConfig {
    /// Create configuration for quick analysis
    pub fn quick() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Quick,
            enable_ai_insights: false,
            enable_violation_detection: false,
            enable_suggestions: false,
            optimization_level: OptimizationLevel::Speed,
            ..Default::default()
        }
    }

    /// Create configuration for comprehensive analysis
    pub fn comprehensive() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Comprehensive,
            enable_ai_insights: true,
            enable_violation_detection: true,
            enable_suggestions: true,
            optimization_level: OptimizationLevel::Balanced,
            ..Default::default()
        }
    }

    /// Create configuration for deep analysis with AI insights
    pub fn deep() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Deep,
            enable_ai_insights: true,
            enable_violation_detection: true,
            enable_suggestions: true,
            minimum_acceptable_score: 80.0,
            optimization_level: OptimizationLevel::Accuracy,
            ..Default::default()
        }
    }
    
    /// Create configuration optimized for real-time IDE analysis
    pub fn real_time() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Quick,
            enable_ai_insights: false,
            enable_violation_detection: true, // Keep violations for real-time feedback
            enable_suggestions: false, // Skip expensive suggestion generation
            optimization_level: OptimizationLevel::Speed,
            minimum_acceptable_score: 60.0, // Lower threshold for real-time
            ..Default::default()
        }
    }
    
    /// Create configuration optimized for CI/CD builds
    pub fn ci_cd() -> Self {
        Self {
            analysis_depth: AnalysisDepth::Comprehensive,
            enable_ai_insights: true,
            enable_violation_detection: true,
            enable_suggestions: true,
            optimization_level: OptimizationLevel::Accuracy,
            minimum_acceptable_score: 80.0, // Higher threshold for builds
            ..Default::default()
        }
    }
}

/// Analyze cohesion for a program using default configuration
pub fn analyze_program_cohesion(program: &prism_ast::Program) -> CohesionResult<ProgramCohesionAnalysis> {
    let mut system = CohesionSystem::new();
    system.analyze_program(program)
}

/// Calculate basic cohesion metrics for a program
pub fn calculate_cohesion_metrics(program: &prism_ast::Program) -> CohesionResult<CohesionMetrics> {
    let calculator = MetricsCalculator::new(MetricWeights::default());
    calculator.calculate_program_metrics(program)
}

/// Check if a program meets minimum cohesion standards
pub fn meets_cohesion_standards(program: &prism_ast::Program, minimum_score: f64) -> CohesionResult<bool> {
    let metrics = calculate_cohesion_metrics(program)?;
    Ok(metrics.overall_score >= minimum_score)
} 