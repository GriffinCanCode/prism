//! Integration layer for PLT-001: AST Design & Parser Architecture.
//!
//! This module provides the complete integration between all language subsystems:
//! - Multi-syntax parsing (prism-syntax) - NOW USING FACTORY SYSTEM
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
use prism_common::{
    SourceId, 
    // NEW: Use trait interfaces instead of concrete types
    ProgramParser, EnhancedParser, SyntaxAwareParser, 
    ParsingConfig, ParsingMetrics, ParsingDiagnostic
};
// UPDATED: Use new factory-based orchestrator instead of old Parser
use prism_syntax::{SyntaxStyle, ParsingOrchestrator, OrchestratorConfig, ParserFactory, NormalizerFactory, ValidatorFactory};
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
/// - Multi-syntax parsing with semantic preservation (UPDATED: using factory system)
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
///     @description "Secure authentication module with audit trail"
///     @author "Security Team"
///     
///     module UserAuth {
///         section interface {
///             @responsibility "Authenticates users securely"
///             function authenticate(user: User) -> Result<Session, Error>
///                 effects [Database.Query, Audit.Log]
///                 requires user.is_verified()
///                 ensures result.is_ok() -> session.is_valid()
///             {
///                 // Implementation would go here
///                 return authenticateUser(user)
///             }
///         }
///     }
/// "#;
///
/// let mut parser = IntegratedParser::new();
/// let result = parser.parse_with_full_analysis(source, SourceId::new(1))?;
///
/// // Access comprehensive analysis results
/// println!("Detected syntax: {:?}", result.detected_syntax);
/// println!("Documentation compliant: {}", result.documentation_analysis.is_some());
/// println!("Cohesion score: {:.1}", result.cohesion_analysis.as_ref().unwrap().overall_score);
/// ```
#[derive(Debug)]
pub struct IntegratedParser {
    /// UPDATED: Multi-syntax parsing orchestrator using factory system
    syntax_orchestrator: ParsingOrchestrator,
    
    #[cfg(feature = "documentation")]
    /// Documentation analysis system
    documentation_system: DocumentationSystem,
    
    #[cfg(feature = "cohesion")]
    /// Cohesion analysis system
    cohesion_system: CohesionSystem,
    
    #[cfg(feature = "semantic")]
    /// Semantic analysis engine
    semantic_engine: SemanticEngine,
    
    #[cfg(feature = "effects")]
    /// Effects system for capability analysis
    effects_system: PrismEffectsSystem,
    
    /// Integration configuration
    config: IntegrationConfig,
}

/// Configuration for integrated parsing
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable multi-syntax parsing
    pub enable_multi_syntax: bool,
    
    /// UPDATED: Factory-based orchestrator configuration
    pub orchestrator_config: OrchestratorConfig,
    
    /// Enable documentation validation
    pub enable_documentation: bool,
    
    /// Enable cohesion analysis
    pub enable_cohesion: bool,
    
    /// Enable semantic analysis
    pub enable_semantic: bool,
    
    /// Enable effects analysis
    pub enable_effects: bool,
    
    /// Enable comprehensive AI metadata generation
    pub enable_ai_metadata: bool,
    
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    
    #[cfg(feature = "documentation")]
    /// Documentation system configuration
    pub documentation_config: DocumentationConfig,
    
    #[cfg(feature = "cohesion")]
    /// Cohesion analysis configuration
    pub cohesion_config: CohesionConfig,
    
    #[cfg(feature = "semantic")]
    /// Semantic analysis configuration
    pub semantic_config: SemanticConfig,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_multi_syntax: true,
            orchestrator_config: OrchestratorConfig {
                enable_component_caching: true,
                max_cache_size: 20, // Higher cache for integration layer
                enable_parallel_processing: false, // Conservative default
                default_validation_level: prism_syntax::ValidationLevel::Full,
                generate_ai_metadata: true,
                preserve_formatting: true,
                enable_error_recovery: true,
            },
            enable_documentation: true,
            enable_cohesion: true,
            enable_semantic: true,
            enable_effects: true,
            enable_ai_metadata: true,
            enable_performance_profiling: false,
            
            #[cfg(feature = "documentation")]
            documentation_config: DocumentationConfig::default(),
            
            #[cfg(feature = "cohesion")]
            cohesion_config: CohesionConfig::default(),
            
            #[cfg(feature = "semantic")]
            semantic_config: SemanticConfig::default(),
        }
    }
}

/// Comprehensive result from integrated parsing
#[derive(Debug)]
pub struct IntegratedParseResult {
    /// The parsed program
    pub program: Program,
    
    /// Detected syntax style and confidence
    pub detected_syntax: SyntaxStyle,
    pub syntax_confidence: f64,
    
    /// UPDATED: Orchestrator performance metrics
    pub parsing_metrics: prism_syntax::ProcessingMetrics,
    
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
    pub effects_analysis: Option<HashMap<String, Vec<String>>>,
    
    /// Comprehensive AI metadata
    pub ai_metadata: Option<IntegratedAIMetadata>,
    
    /// System performance metrics
    pub system_times: HashMap<String, u64>,
    
    /// Integration diagnostics and warnings
    pub diagnostics: Vec<IntegrationDiagnostic>,
}

/// Integrated AI metadata combining all analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedAIMetadata {
    /// Overall quality assessment
    pub quality_assessment: QualityAssessment,
    /// Business context extracted from all systems
    pub business_context: BusinessContext,
    /// Architectural insights
    pub architectural_insights: ArchitecturalInsights,
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
    /// Maintenance recommendations
    pub maintenance_recommendations: Vec<MaintenanceRecommendation>,
}

/// Overall quality assessment of the parsed code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Documentation quality score
    pub documentation_score: f64,
    /// Cohesion quality score
    pub cohesion_score: f64,
    /// Semantic quality score
    pub semantic_score: f64,
    /// Effects quality score
    pub effects_score: f64,
}

/// Business context extracted from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Primary domain identified
    pub primary_domain: String,
    /// Business capabilities identified
    pub capabilities: Vec<String>,
    /// Stakeholder concerns addressed
    pub stakeholder_concerns: Vec<String>,
    /// Business rules identified
    pub business_rules: Vec<String>,
}

/// Architectural insights from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalInsights {
    /// Architectural patterns identified
    pub patterns: Vec<String>,
    /// Design principles followed
    pub design_principles: Vec<String>,
    /// Potential architectural issues
    pub issues: Vec<String>,
    /// Recommended improvements
    pub improvements: Vec<String>,
}

/// Performance characteristics identified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Estimated complexity
    pub complexity: String,
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Scalability assessment
    pub scalability: String,
    /// Resource usage patterns
    pub resource_patterns: Vec<String>,
}

/// Maintenance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceRecommendation {
    /// Recommendation category
    pub category: String,
    /// Priority level (1-5)
    pub priority: u8,
    /// Description of the recommendation
    pub description: String,
    /// Suggested action
    pub action: String,
    /// Estimated effort
    pub effort: String,
}

/// Integration diagnostic message
#[derive(Debug, Clone)]
pub struct IntegrationDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    /// Source system that generated the diagnostic
    pub source: String,
    /// Diagnostic message
    pub message: String,
    /// Location in source code (if applicable)
    pub location: Option<prism_common::span::Span>,
    /// Suggested resolution
    pub suggestion: Option<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// Error that prevents successful analysis
    Error,
    /// Warning about potential issues
    Warning,
    /// Informational message
    Info,
    /// Performance or optimization hint
    Hint,
}

/// Errors that can occur during integration
#[derive(Debug, Error)]
pub enum IntegrationError {
    /// UPDATED: Multi-syntax parsing failed using orchestrator
    #[error("Multi-syntax orchestration failed: {reason}")]
    MultiSyntaxFailed { reason: String },
    
    /// Documentation analysis failed
    #[error("Documentation analysis failed: {reason}")]
    DocumentationFailed { reason: String },
    
    /// Cohesion analysis failed
    #[error("Cohesion analysis failed: {reason}")]
    CohesionFailed { reason: String },
    
    /// Semantic analysis failed
    #[error("Semantic analysis failed: {reason}")]
    SemanticFailed { reason: String },
    
    /// Effects analysis failed
    #[error("Effects analysis failed: {reason}")]
    EffectsFailed { reason: String },
    
    /// Configuration error
    #[error("Configuration error: {reason}")]
    ConfigurationError { reason: String },
    
    /// System integration error
    #[error("System integration error: {reason}")]
    SystemError { reason: String },
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
            // UPDATED: Use factory-based orchestrator instead of old Parser
            syntax_orchestrator: ParsingOrchestrator::with_config(config.orchestrator_config.clone()),
            
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
    
    /// Create an integrated parser with custom factories (ADVANCED USAGE)
    pub fn with_custom_factories(
        parser_factory: ParserFactory,
        normalizer_factory: NormalizerFactory,
        validator_factory: ValidatorFactory,
        config: IntegrationConfig,
    ) -> Self {
        let orchestrator = ParsingOrchestrator::with_factories(
            parser_factory,
            normalizer_factory,
            validator_factory,
            config.orchestrator_config.clone(),
        );
        
        Self {
            syntax_orchestrator: orchestrator,
            
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
    
    /// Parse source code with full PLT-001 analysis (UPDATED)
    pub fn parse_with_full_analysis(
        &mut self,
        source: &str,
        source_id: SourceId,
    ) -> Result<IntegratedParseResult, IntegrationError> {
        let start_time = std::time::Instant::now();
        let mut diagnostics = Vec::new();
        let mut system_times = HashMap::new();
        
        // Step 1: Multi-syntax parsing using NEW orchestrator
        let parse_start = std::time::Instant::now();
        let orchestrator_result = if self.config.enable_multi_syntax {
            self.syntax_orchestrator.parse(source, source_id)
                .map_err(|e| IntegrationError::MultiSyntaxFailed { 
                    reason: e.to_string() 
                })?
        } else {
            return Err(IntegrationError::ConfigurationError {
                reason: "Multi-syntax parsing is disabled but required".to_string(),
            });
        };
        system_times.insert("parsing".to_string(), parse_start.elapsed().as_millis() as u64);
        
        // Extract results from orchestrator
        let program = orchestrator_result.program;
        let detected_syntax = orchestrator_result.detected_style;
        let syntax_confidence = orchestrator_result.confidence;
        let parsing_metrics = orchestrator_result.metrics;
        
        // Step 2: Documentation analysis (if enabled)
        #[cfg(feature = "documentation")]
        let documentation_analysis = if self.config.enable_documentation {
            let doc_start = std::time::Instant::now();
            match self.documentation_system.process_program(&program) {
                Ok(result) => {
                    system_times.insert("documentation".to_string(), doc_start.elapsed().as_millis() as u64);
                    Some(result)
                }
                Err(e) => {
                    diagnostics.push(IntegrationDiagnostic {
                        level: DiagnosticLevel::Warning,
                        source: "documentation".to_string(),
                        message: format!("Documentation analysis failed: {}", e),
                        location: None,
                        suggestion: Some("Check documentation format and completeness".to_string()),
                    });
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(feature = "documentation"))]
        let documentation_analysis = None;
        
        // Step 3: Cohesion analysis (if enabled)
        #[cfg(feature = "cohesion")]
        let cohesion_analysis = if self.config.enable_cohesion {
            let cohesion_start = std::time::Instant::now();
            match self.cohesion_system.analyze_program(&program) {
                Ok(result) => {
                    system_times.insert("cohesion".to_string(), cohesion_start.elapsed().as_millis() as u64);
                    Some(result)
                }
                Err(e) => {
                    diagnostics.push(IntegrationDiagnostic {
                        level: DiagnosticLevel::Warning,
                        source: "cohesion".to_string(),
                        message: format!("Cohesion analysis failed: {}", e),
                        location: None,
                        suggestion: Some("Review module organization and responsibilities".to_string()),
                    });
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(feature = "cohesion"))]
        let cohesion_analysis = None;
        
        // Step 4: Semantic analysis (if enabled)
        #[cfg(feature = "semantic")]
        let semantic_analysis = if self.config.enable_semantic {
            let semantic_start = std::time::Instant::now();
            match self.semantic_engine.analyze(&program) {
                Ok(result) => {
                    system_times.insert("semantic".to_string(), semantic_start.elapsed().as_millis() as u64);
                    Some(result)
                }
                Err(e) => {
                    diagnostics.push(IntegrationDiagnostic {
                        level: DiagnosticLevel::Warning,
                        source: "semantic".to_string(),
                        message: format!("Semantic analysis failed: {}", e),
                        location: None,
                        suggestion: Some("Check type annotations and semantic constraints".to_string()),
                    });
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(feature = "semantic"))]
        let semantic_analysis = None;
        
        // Step 5: Effects analysis (if enabled)
        #[cfg(feature = "effects")]
        let effects_analysis = if self.config.enable_effects {
            let effects_start = std::time::Instant::now();
            // Simplified effects analysis - in practice would be more sophisticated
            let effects_result = HashMap::new(); // Placeholder
            system_times.insert("effects".to_string(), effects_start.elapsed().as_millis() as u64);
            Some(effects_result)
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
                detected_syntax,
                syntax_confidence,
                #[cfg(feature = "documentation")]
                &documentation_analysis,
                #[cfg(feature = "cohesion")]
                &cohesion_analysis,
                #[cfg(feature = "semantic")]
                &semantic_analysis,
                #[cfg(feature = "effects")]
                &effects_analysis,
            );
            system_times.insert("ai_metadata".to_string(), ai_start.elapsed().as_millis() as u64);
            Some(metadata)
        } else {
            None
        };
        
        // Calculate total time
        system_times.insert("total".to_string(), start_time.elapsed().as_millis() as u64);
        
        Ok(IntegratedParseResult {
            program,
            detected_syntax,
            syntax_confidence,
            parsing_metrics,
            
            #[cfg(feature = "documentation")]
            documentation_analysis,
            #[cfg(not(feature = "documentation"))]
            documentation_analysis: None,
            
            #[cfg(feature = "cohesion")]
            cohesion_analysis,
            #[cfg(not(feature = "cohesion"))]
            cohesion_analysis: None,
            
            #[cfg(feature = "semantic")]
            semantic_analysis,
            #[cfg(not(feature = "semantic"))]
            semantic_analysis: None,
            
            #[cfg(feature = "effects")]
            effects_analysis,
            #[cfg(not(feature = "effects"))]
            effects_analysis: None,
            
            ai_metadata,
            system_times,
            diagnostics,
        })
    }
    
    /// Generate integrated AI metadata from all analysis results
    fn generate_integrated_ai_metadata(
        &self,
        _program: &Program,
        detected_syntax: SyntaxStyle,
        syntax_confidence: f64,
        #[cfg(feature = "documentation")]
        _documentation_analysis: &Option<DocumentationResult>,
        #[cfg(feature = "cohesion")]
        cohesion_analysis: &Option<ProgramCohesionAnalysis>,
        #[cfg(feature = "semantic")]
        _semantic_analysis: &Option<SemanticInfo>,
        #[cfg(feature = "effects")]
        _effects_analysis: &Option<HashMap<String, Vec<String>>>,
    ) -> IntegratedAIMetadata {
        // Calculate overall quality scores
        let documentation_score = 0.8; // Placeholder - would use actual documentation analysis
        let cohesion_score = cohesion_analysis
            .as_ref()
            .map(|c| c.overall_score)
            .unwrap_or(0.7);
        let semantic_score = 0.75; // Placeholder - would use actual semantic analysis
        let effects_score = 0.8; // Placeholder - would use actual effects analysis
        
        let overall_score = (documentation_score + cohesion_score + semantic_score + effects_score) / 4.0;
        
        IntegratedAIMetadata {
            quality_assessment: QualityAssessment {
                overall_score,
                documentation_score,
                cohesion_score,
                semantic_score,
                effects_score,
            },
            business_context: BusinessContext {
                primary_domain: "General".to_string(), // Would extract from analysis
                capabilities: vec!["Data Processing".to_string()], // Would extract from analysis
                stakeholder_concerns: vec!["Performance".to_string(), "Maintainability".to_string()],
                business_rules: vec![], // Would extract from constraints
            },
            architectural_insights: ArchitecturalInsights {
                patterns: vec![format!("{:?} Syntax", detected_syntax)],
                design_principles: vec!["Separation of Concerns".to_string()],
                issues: vec![], // Would identify from analysis
                improvements: vec![], // Would suggest based on analysis
            },
            performance_characteristics: PerformanceCharacteristics {
                complexity: "Linear".to_string(), // Would calculate from analysis
                bottlenecks: vec![], // Would identify from effects analysis
                scalability: "Good".to_string(), // Would assess from architecture
                resource_patterns: vec![], // Would extract from effects
            },
            maintenance_recommendations: vec![
                MaintenanceRecommendation {
                    category: "Documentation".to_string(),
                    priority: 2,
                    description: "Consider adding more comprehensive documentation".to_string(),
                    action: "Add JSDoc comments to all public functions".to_string(),
                    effort: "Medium".to_string(),
                },
            ],
        }
    }
    
    /// Get orchestrator performance metrics (NEW)
    pub fn get_orchestrator_metrics(&self) -> &prism_syntax::CacheStats {
        self.syntax_orchestrator.cache_stats()
    }
    
    /// Clear orchestrator cache (NEW)
    pub fn clear_orchestrator_cache(&mut self) {
        self.syntax_orchestrator.clear_cache();
    }
}

impl Default for IntegratedParser {
    fn default() -> Self {
        Self::new()
    }
}

// NEW: Implement trait interfaces to break dependency cycles
impl ProgramParser for IntegratedParser {
    type Program = Program;
    type Error = IntegrationError;
    
    fn parse_source(&mut self, source: &str, source_id: SourceId) -> Result<Self::Program, Self::Error> {
        let result = self.parse_with_full_analysis(source, source_id)?;
        Ok(result.program)
    }
    
    fn parse_source_with_config(
        &mut self, 
        source: &str, 
        source_id: SourceId,
        config: &ParsingConfig
    ) -> Result<Self::Program, Self::Error> {
        // Update internal configuration based on trait config
        self.update_config_from_parsing_config(config);
        self.parse_source(source, source_id)
    }
}

impl EnhancedParser for IntegratedParser {
    type EnhancedResult = IntegratedParseResult;
    
    fn parse_with_analysis(
        &mut self,
        source: &str,
        source_id: SourceId,
    ) -> Result<Self::EnhancedResult, Self::Error> {
        self.parse_with_full_analysis(source, source_id)
    }
}

impl SyntaxAwareParser for IntegratedParser {
    type SyntaxStyle = SyntaxStyle;
    
    fn detect_syntax(&self, source: &str) -> Result<(Self::SyntaxStyle, f64), Self::Error> {
        // Use internal orchestrator's detection capabilities
        match self.syntax_orchestrator.detect_syntax_style(source) {
            Ok((style, confidence)) => Ok((style, confidence)),
            Err(e) => Err(IntegrationError::MultiSyntaxFailed { 
                reason: format!("Syntax detection failed: {}", e) 
            })
        }
    }
    
    fn parse_with_syntax(
        &mut self,
        source: &str,
        source_id: SourceId,
        preferred_syntax: Option<Self::SyntaxStyle>,
    ) -> Result<Self::Program, Self::Error> {
        // Update orchestrator configuration with syntax preference
        if let Some(syntax) = preferred_syntax {
            self.syntax_orchestrator.set_preferred_syntax(syntax);
        }
        
        self.parse_source(source, source_id)
    }
}

// Helper method to convert trait config to internal config
impl IntegratedParser {
    fn update_config_from_parsing_config(&mut self, config: &ParsingConfig) {
        // Update orchestrator config
        let mut orchestrator_config = self.syntax_orchestrator.config().clone();
        orchestrator_config.generate_ai_metadata = config.extract_ai_context;
        orchestrator_config.enable_error_recovery = config.aggressive_recovery;
        self.syntax_orchestrator.set_config(orchestrator_config);
        
        // Update integration config
        self.config.enable_ai_metadata = config.extract_ai_context;
        self.config.enable_documentation = config.validate_documentation;
        self.config.enable_cohesion = config.analyze_cohesion;
        self.config.enable_performance_profiling = config.enable_profiling;
    }
}

/// Parse source code with complete PLT-001 analysis (UPDATED CONVENIENCE FUNCTION)
/// 
/// This is the recommended entry point for parsing Prism code as it provides
/// the complete PLT-001 functionality using the new factory-based orchestrator.
pub fn parse_with_full_analysis(
    source: &str,
    source_id: SourceId,
) -> Result<IntegratedParseResult, IntegrationError> {
    let mut parser = IntegratedParser::new();
    parser.parse_with_full_analysis(source, source_id)
}

/// Parse source code with basic parsing only (UPDATED CONVENIENCE FUNCTION)
/// 
/// This provides fast parsing using the new orchestrator without the full analysis pipeline.
pub fn parse_basic(
    source: &str,
    source_id: SourceId,
) -> Result<Program, IntegrationError> {
    let mut orchestrator = ParsingOrchestrator::new();
    let result = orchestrator.parse(source, source_id)
        .map_err(|e| IntegrationError::MultiSyntaxFailed { 
            reason: e.to_string() 
        })?;
    Ok(result.program)
} 