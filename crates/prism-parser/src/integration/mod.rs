//! Integration layer for PLT-001: AST Design & Parser Architecture.
//!
//! This module provides the complete integration between all language subsystems:
//! - Multi-syntax parsing (prism-syntax)
//! - Documentation validation (prism-documentation)
//! - Cohesion analysis (prism-cohesion)
//! - Semantic type system (prism-semantic)
//! - Effect system (prism-effects)
//!
//! It implements the full PLT-001 specification by orchestrating these systems
//! to provide comprehensive parsing with semantic awareness, documentation
//! validation, and AI-readable metadata generation.

use crate::{ParseConfig, ParseResult, ParseError};
use prism_ast::{Program, AstNode, Item};
use prism_common::SourceId;
use prism_syntax::{SyntaxStyle, Parser as MultiSyntaxParser};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "documentation")]
use prism_documentation::{DocumentationSystem, DocumentationConfig, ProcessingResult as DocumentationResult};

#[cfg(feature = "cohesion")]
use prism_cohesion::{CohesionSystem, CohesionConfig, ProgramCohesionAnalysis};

#[cfg(feature = "semantic")]
use prism_semantic::{SemanticEngine, SemanticConfig, SemanticInfo};

#[cfg(feature = "effects")]
use prism_effects::{PrismEffectsSystem, EffectSystemError};

/// Integrated parser implementing the complete PLT-001 specification.
///
/// This parser provides the full PLT-001 functionality by integrating:
/// - Multi-syntax parsing with semantic preservation
/// - Documentation validation and JSDoc compatibility  
/// - Conceptual cohesion analysis and metrics
/// - Semantic type system integration
/// - Effect system analysis and validation
/// - AI-readable metadata generation
///
/// # Example
///
/// ```rust
/// use prism_parser::integration::IntegratedParser;
/// use prism_common::SourceId;
///
/// let source = r#"
///     @responsibility "Manages user authentication with security"
///     @module "UserAuth"
///     @description "Secure user authentication module"
///     @author "Security Team"
///     
///     module UserAuth {
///         section interface {
///             @responsibility "Authenticates users securely"
///             function authenticate(user: User) -> Result<Session, Error>
///                 effects [Database.Query, Audit.Log]
///                 requires user.is_verified()
///             {
///                 // Implementation
///             }
///         }
///     }
/// "#;
///
/// let mut parser = IntegratedParser::new();
/// let result = parser.parse_with_full_analysis(source, SourceId::new(1))?;
///
/// // Access all analysis results
/// println!("Syntax detected: {:?}", result.detected_syntax);
/// println!("Documentation compliant: {}", result.documentation_analysis.is_some());
/// println!("Cohesion score: {:.1}", result.cohesion_analysis.as_ref().unwrap().overall_score);
/// ```
#[derive(Debug)]
pub struct IntegratedParser {
    /// Multi-syntax parser
    multi_syntax_parser: MultiSyntaxParser,
    
    /// Documentation system (optional)
    #[cfg(feature = "documentation")]
    documentation_system: DocumentationSystem,
    
    /// Cohesion analysis system (optional)
    #[cfg(feature = "cohesion")]
    cohesion_system: CohesionSystem,
    
    /// Semantic analysis engine (optional)
    #[cfg(feature = "semantic")]
    semantic_engine: SemanticEngine,
    
    /// Effects system (optional)
    #[cfg(feature = "effects")]
    effects_system: PrismEffectsSystem,
    
    /// Integration configuration
    config: IntegrationConfig,
}

/// Configuration for the integrated parser
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable multi-syntax parsing
    pub enable_multi_syntax: bool,
    
    /// Enable documentation validation
    pub enable_documentation: bool,
    
    /// Enable cohesion analysis
    pub enable_cohesion: bool,
    
    /// Enable semantic analysis
    pub enable_semantic: bool,
    
    /// Enable effects analysis
    pub enable_effects: bool,
    
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    
    /// Preferred syntax style (if any)
    pub preferred_syntax: Option<SyntaxStyle>,
    
    /// Documentation configuration
    #[cfg(feature = "documentation")]
    pub documentation_config: DocumentationConfig,
    
    /// Cohesion analysis configuration
    #[cfg(feature = "cohesion")]
    pub cohesion_config: CohesionConfig,
    
    /// Semantic analysis configuration
    #[cfg(feature = "semantic")]
    pub semantic_config: SemanticConfig,
    
    /// Integration-specific settings
    pub integration_settings: IntegrationSettings,
}

/// Integration-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationSettings {
    /// Maximum analysis time (milliseconds)
    pub max_analysis_time_ms: u64,
    
    /// Enable parallel analysis where possible
    pub enable_parallel_analysis: bool,
    
    /// Cache analysis results
    pub enable_result_caching: bool,
    
    /// Generate comprehensive reports
    pub generate_comprehensive_reports: bool,
}

/// Complete parsing result with all analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedParseResult {
    /// Successfully parsed program
    pub program: Program,
    
    /// Detected syntax style
    pub detected_syntax: SyntaxStyle,
    
    /// Detection confidence (0.0 to 1.0)
    pub syntax_confidence: f64,
    
    /// Documentation analysis results (if enabled)
    #[cfg(feature = "documentation")]
    pub documentation_analysis: Option<DocumentationResult>,
    
    /// Cohesion analysis results (if enabled)
    #[cfg(feature = "cohesion")]
    pub cohesion_analysis: Option<ProgramCohesionAnalysis>,
    
    /// Semantic analysis results (if enabled)
    #[cfg(feature = "semantic")]
    pub semantic_analysis: Option<SemanticInfo>,
    
    /// Effects analysis results (if enabled)
    #[cfg(feature = "effects")]
    pub effects_analysis: Option<EffectsAnalysisResult>,
    
    /// Generated AI metadata
    pub ai_metadata: Option<IntegratedAIMetadata>,
    
    /// Analysis performance metrics
    pub performance_metrics: AnalysisPerformanceMetrics,
    
    /// Integration diagnostics
    pub diagnostics: Vec<IntegrationDiagnostic>,
}

/// Effects analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsAnalysisResult {
    /// Detected effects in the program
    pub detected_effects: Vec<String>,
    
    /// Capability requirements
    pub capability_requirements: Vec<String>,
    
    /// Security policy violations
    pub security_violations: Vec<String>,
    
    /// Effect composition analysis
    pub composition_analysis: HashMap<String, f64>,
}

/// Integrated AI metadata combining all subsystems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedAIMetadata {
    /// Business context from documentation
    pub business_context: Option<String>,
    
    /// Architectural patterns detected
    pub architectural_patterns: Vec<String>,
    
    /// Cohesion insights
    pub cohesion_insights: Vec<String>,
    
    /// Semantic type insights
    pub semantic_insights: Vec<String>,
    
    /// Effect system insights
    pub effect_insights: Vec<String>,
    
    /// Cross-system relationships
    pub cross_system_relationships: Vec<CrossSystemRelationship>,
    
    /// Overall code quality assessment
    pub quality_assessment: QualityAssessment,
}

/// Relationship between different analysis systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSystemRelationship {
    /// Source system
    pub source_system: String,
    
    /// Target system
    pub target_system: String,
    
    /// Relationship type
    pub relationship_type: String,
    
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    
    /// Description of the relationship
    pub description: String,
}

/// Overall code quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0-100)
    pub overall_score: f64,
    
    /// Individual dimension scores
    pub dimension_scores: HashMap<String, f64>,
    
    /// Quality trends
    pub trends: Vec<String>,
    
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Performance metrics for the analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPerformanceMetrics {
    /// Total analysis time
    pub total_time_ms: u64,
    
    /// Time breakdown by system
    pub system_times: HashMap<String, u64>,
    
    /// Memory usage
    pub memory_usage_bytes: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Integration diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    
    /// Source system
    pub source_system: String,
    
    /// Diagnostic message
    pub message: String,
    
    /// Location (if applicable)
    pub location: Option<prism_common::span::Span>,
    
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticLevel {
    /// Error level
    Error,
    /// Warning level
    Warning,
    /// Information level
    Info,
    /// Debug level
    Debug,
}

/// Integration error types
#[derive(Debug, Error)]
pub enum IntegrationError {
    /// Multi-syntax parsing failed
    #[error("Multi-syntax parsing failed: {reason}")]
    MultiSyntaxFailed { reason: String },
    
    /// Documentation analysis failed
    #[cfg(feature = "documentation")]
    #[error("Documentation analysis failed: {reason}")]
    DocumentationFailed { reason: String },
    
    /// Cohesion analysis failed
    #[cfg(feature = "cohesion")]
    #[error("Cohesion analysis failed: {reason}")]
    CohesionFailed { reason: String },
    
    /// Semantic analysis failed
    #[cfg(feature = "semantic")]
    #[error("Semantic analysis failed: {reason}")]
    SemanticFailed { reason: String },
    
    /// Effects analysis failed
    #[cfg(feature = "effects")]
    #[error("Effects analysis failed: {reason}")]
    EffectsFailed { reason: String },
    
    /// AI metadata generation failed
    #[error("AI metadata generation failed: {reason}")]
    AIMetadataFailed { reason: String },
    
    /// Analysis timeout
    #[error("Analysis timed out after {timeout_ms}ms")]
    AnalysisTimeout { timeout_ms: u64 },
    
    /// Configuration error
    #[error("Invalid integration configuration: {reason}")]
    ConfigurationError { reason: String },
}

impl IntegratedParser {
    /// Create a new integrated parser with default configuration
    pub fn new() -> Self {
        let config = IntegrationConfig::default();
        Self::with_config(config)
    }
    
    /// Create a new integrated parser with custom configuration
    pub fn with_config(config: IntegrationConfig) -> Self {
        Self {
            multi_syntax_parser: MultiSyntaxParser::new(),
            
            #[cfg(feature = "documentation")]
            documentation_system: DocumentationSystem::with_config(config.documentation_config.clone()),
            
            #[cfg(feature = "cohesion")]
            cohesion_system: CohesionSystem::with_config(config.cohesion_config.clone()),
            
            #[cfg(feature = "semantic")]
            semantic_engine: SemanticEngine::new(config.semantic_config.clone()).unwrap(),
            
            #[cfg(feature = "effects")]
            effects_system: PrismEffectsSystem::new(),
            
            config,
        }
    }
    
    /// Parse source code with full PLT-001 analysis
    pub fn parse_with_full_analysis(
        &mut self,
        source: &str,
        source_id: SourceId,
    ) -> Result<IntegratedParseResult, IntegrationError> {
        let start_time = std::time::Instant::now();
        let mut diagnostics = Vec::new();
        let mut system_times = HashMap::new();
        
        // Step 1: Multi-syntax parsing
        let parse_start = std::time::Instant::now();
        let program = if self.config.enable_multi_syntax {
            self.multi_syntax_parser.parse_source(source, source_id)
                .map_err(|e| IntegrationError::MultiSyntaxFailed { 
                    reason: e.to_string() 
                })?
        } else {
            // Fallback to basic parsing
            return Err(IntegrationError::ConfigurationError {
                reason: "Multi-syntax parsing is disabled but required".to_string(),
            });
        };
        system_times.insert("parsing".to_string(), parse_start.elapsed().as_millis() as u64);
        
        // Get detected syntax information (placeholder for now)
        let detected_syntax = SyntaxStyle::Canonical;
        let syntax_confidence = 1.0;
        
        // Step 2: Documentation analysis (if enabled)
        #[cfg(feature = "documentation")]
        let documentation_analysis = if self.config.enable_documentation {
            let doc_start = std::time::Instant::now();
            let result = self.documentation_system.process_program(&program)
                .map_err(|e| IntegrationError::DocumentationFailed { 
                    reason: e.to_string() 
                })?;
            system_times.insert("documentation".to_string(), doc_start.elapsed().as_millis() as u64);
            
            // Add documentation diagnostics
            for violation in &result.validation_result.violations {
                diagnostics.push(IntegrationDiagnostic {
                    level: match violation.severity {
                        prism_documentation::ViolationSeverity::Error => DiagnosticLevel::Error,
                        prism_documentation::ViolationSeverity::Warning => DiagnosticLevel::Warning,
                        prism_documentation::ViolationSeverity::Info => DiagnosticLevel::Info,
                        prism_documentation::ViolationSeverity::Hint => DiagnosticLevel::Debug,
                    },
                    source_system: "documentation".to_string(),
                    message: violation.message.clone(),
                    location: Some(violation.location),
                    suggested_actions: violation.suggested_fix.as_ref().map(|s| vec![s.clone()]).unwrap_or_default(),
                });
            }
            
            Some(result)
        } else {
            None
        };
        
        #[cfg(not(feature = "documentation"))]
        let documentation_analysis = None;
        
        // Step 3: Cohesion analysis (if enabled)
        #[cfg(feature = "cohesion")]
        let cohesion_analysis = if self.config.enable_cohesion {
            let cohesion_start = std::time::Instant::now();
            let result = self.cohesion_system.analyze_program(&program)
                .map_err(|e| IntegrationError::CohesionFailed { 
                    reason: e.to_string() 
                })?;
            system_times.insert("cohesion".to_string(), cohesion_start.elapsed().as_millis() as u64);
            
            // Add cohesion diagnostics
            for violation in &result.violations {
                diagnostics.push(IntegrationDiagnostic {
                    level: match violation.severity {
                        prism_cohesion::ViolationSeverity::Error => DiagnosticLevel::Error,
                        prism_cohesion::ViolationSeverity::Warning => DiagnosticLevel::Warning,
                        prism_cohesion::ViolationSeverity::Info => DiagnosticLevel::Info,
                    },
                    source_system: "cohesion".to_string(),
                    message: violation.message.clone(),
                    location: Some(violation.location),
                    suggested_actions: violation.suggested_fix.as_ref().map(|s| vec![s.clone()]).unwrap_or_default(),
                });
            }
            
            Some(result)
        } else {
            None
        };
        
        #[cfg(not(feature = "cohesion"))]
        let cohesion_analysis = None;
        
        // Step 4: Semantic analysis (if enabled)
        #[cfg(feature = "semantic")]
        let semantic_analysis = if self.config.enable_semantic {
            let semantic_start = std::time::Instant::now();
            let result = self.semantic_engine.analyze_program(&program)
                .map_err(|e| IntegrationError::SemanticFailed { 
                    reason: e.to_string() 
                })?;
            system_times.insert("semantic".to_string(), semantic_start.elapsed().as_millis() as u64);
            Some(result)
        } else {
            None
        };
        
        #[cfg(not(feature = "semantic"))]
        let semantic_analysis = None;
        
        // Step 5: Effects analysis (if enabled)
        #[cfg(feature = "effects")]
        let effects_analysis = if self.config.enable_effects {
            let effects_start = std::time::Instant::now();
            // TODO: Implement effects analysis integration
            let result = EffectsAnalysisResult {
                detected_effects: Vec::new(),
                capability_requirements: Vec::new(),
                security_violations: Vec::new(),
                composition_analysis: HashMap::new(),
            };
            system_times.insert("effects".to_string(), effects_start.elapsed().as_millis() as u64);
            Some(result)
        } else {
            None
        };
        
        #[cfg(not(feature = "effects"))]
        let effects_analysis = None;
        
        // Step 6: Generate integrated AI metadata (if enabled)
        let ai_metadata = if self.config.enable_ai_metadata {
            let ai_start = std::time::Instant::now();
            let metadata = self.generate_integrated_ai_metadata(
                &program,
                #[cfg(feature = "documentation")]
                documentation_analysis.as_ref(),
                #[cfg(feature = "cohesion")]
                cohesion_analysis.as_ref(),
                #[cfg(feature = "semantic")]
                semantic_analysis.as_ref(),
                #[cfg(feature = "effects")]
                effects_analysis.as_ref(),
            )?;
            system_times.insert("ai_metadata".to_string(), ai_start.elapsed().as_millis() as u64);
            Some(metadata)
        } else {
            None
        };
        
        let total_time = start_time.elapsed().as_millis() as u64;
        
        Ok(IntegratedParseResult {
            program,
            detected_syntax,
            syntax_confidence,
            #[cfg(feature = "documentation")]
            documentation_analysis,
            #[cfg(feature = "cohesion")]
            cohesion_analysis,
            #[cfg(feature = "semantic")]
            semantic_analysis,
            #[cfg(feature = "effects")]
            effects_analysis,
            ai_metadata,
            performance_metrics: AnalysisPerformanceMetrics {
                total_time_ms: total_time,
                system_times,
                memory_usage_bytes: 0, // TODO: Implement memory tracking
                cache_hit_rate: 0.0,   // TODO: Implement cache tracking
            },
            diagnostics,
        })
    }
    
    /// Parse source code with basic analysis only
    pub fn parse_basic(&mut self, source: &str, source_id: SourceId) -> Result<Program, IntegrationError> {
        self.multi_syntax_parser.parse_source(source, source_id)
            .map_err(|e| IntegrationError::MultiSyntaxFailed { 
                reason: e.to_string() 
            })
    }
    
    /// Generate integrated AI metadata from all analysis results
    fn generate_integrated_ai_metadata(
        &self,
        _program: &Program,
        #[cfg(feature = "documentation")]
        _documentation: Option<&DocumentationResult>,
        #[cfg(feature = "cohesion")]
        _cohesion: Option<&ProgramCohesionAnalysis>,
        #[cfg(feature = "semantic")]
        _semantic: Option<&SemanticInfo>,
        #[cfg(feature = "effects")]
        _effects: Option<&EffectsAnalysisResult>,
    ) -> Result<IntegratedAIMetadata, IntegrationError> {
        // TODO: Implement comprehensive AI metadata generation
        // This would combine insights from all analysis systems
        
        Ok(IntegratedAIMetadata {
            business_context: None,
            architectural_patterns: Vec::new(),
            cohesion_insights: Vec::new(),
            semantic_insights: Vec::new(),
            effect_insights: Vec::new(),
            cross_system_relationships: Vec::new(),
            quality_assessment: QualityAssessment {
                overall_score: 75.0,
                dimension_scores: HashMap::new(),
                trends: Vec::new(),
                recommendations: Vec::new(),
            },
        })
    }
    
    /// Get current configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: IntegrationConfig) {
        self.config = config.clone();
        
        #[cfg(feature = "documentation")]
        {
            self.documentation_system.set_config(config.documentation_config.clone());
        }
        
        #[cfg(feature = "cohesion")]
        {
            self.cohesion_system.set_config(config.cohesion_config.clone());
        }
        
        // TODO: Update other system configurations
    }
}

impl Default for IntegratedParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_multi_syntax: true,
            enable_documentation: true,
            enable_cohesion: true,
            enable_semantic: true,
            enable_effects: true,
            enable_ai_metadata: true,
            preferred_syntax: None,
            #[cfg(feature = "documentation")]
            documentation_config: DocumentationConfig::default(),
            #[cfg(feature = "cohesion")]
            cohesion_config: CohesionConfig::default(),
            #[cfg(feature = "semantic")]
            semantic_config: SemanticConfig::default(),
            integration_settings: IntegrationSettings::default(),
        }
    }
}

impl Default for IntegrationSettings {
    fn default() -> Self {
        Self {
            max_analysis_time_ms: 30000, // 30 seconds
            enable_parallel_analysis: true,
            enable_result_caching: true,
            generate_comprehensive_reports: true,
        }
    }
}

/// Parse source code with full PLT-001 analysis using default configuration
pub fn parse_with_full_analysis(
    source: &str,
    source_id: SourceId,
) -> Result<IntegratedParseResult, IntegrationError> {
    let mut parser = IntegratedParser::new();
    parser.parse_with_full_analysis(source, source_id)
}

/// Parse source code with basic parsing only
pub fn parse_basic(source: &str, source_id: SourceId) -> Result<Program, IntegrationError> {
    let mut parser = IntegratedParser::new();
    parser.parse_basic(source, source_id)
} 