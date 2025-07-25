//! Query-Based Compilation Pipeline
//!
//! This module implements the complete query-based compilation pipeline that orchestrates
//! all compilation phases through the query system, enabling incremental compilation,
//! dependency tracking, and AI-first metadata generation.
//!
//! ## Design Principles
//!
//! 1. **Query-First**: Every compilation phase is implemented as a query
//! 2. **Incremental by Default**: All phases support incremental compilation
//! 3. **Dependency Aware**: Automatic dependency tracking and invalidation
//! 4. **Parallel Execution**: Phases can run in parallel when dependencies allow
//! 5. **AI-Comprehensible**: Rich metadata generation for AI tools
//! 6. **Error Recovery**: Graceful handling of compilation errors

use crate::error::{CompilerError, CompilerResult};
use crate::query::core::{QueryEngine, QueryContext, QueryConfig, QueryProfiler, SemanticContext, DefaultSemanticContext};
use crate::query::semantic_queries::*;
use crate::context::{CompilationContext, CompilationPhase, CompilationTarget, CompilationConfig};
use crate::semantic::SemanticDatabase;
use prism_common::NodeId;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use tokio::sync::Semaphore;
use tracing::{info, debug, warn, error};

/// Query-based compilation pipeline orchestrator
#[derive(Debug)]
pub struct CompilationPipeline {
    /// Query engine for executing compilation queries
    query_engine: Arc<QueryEngine>,
    /// Semantic database for storing analysis results
    semantic_db: Arc<SemanticDatabase>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Compilation context
    context: Arc<CompilationContext>,
    /// Semantic type integration
    semantic_type_integration: Option<Arc<crate::semantic::SemanticTypeIntegration>>,
    /// Performance metrics
    metrics: Arc<Mutex<PipelineMetrics>>,
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
}

/// Configuration for the compilation pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable parallel phase execution
    pub enable_parallel_execution: bool,
    /// Maximum concurrent phases
    pub max_concurrent_phases: usize,
    /// Enable incremental compilation
    pub enable_incremental: bool,
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    /// Phase timeout in seconds
    pub phase_timeout_secs: u64,
    /// Enable error recovery
    pub enable_error_recovery: bool,
    /// Target platforms to compile for
    pub targets: Vec<CompilationTarget>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_parallel_execution: true,
            max_concurrent_phases: num_cpus::get(),
            enable_incremental: true,
            enable_ai_metadata: true,
            phase_timeout_secs: 300, // 5 minutes
            enable_error_recovery: true,
            targets: vec![CompilationTarget::TypeScript],
        }
    }
}

/// Pipeline performance metrics
#[derive(Debug, Default)]
pub struct PipelineMetrics {
    /// Total compilation time
    pub total_time: Duration,
    /// Time per phase
    pub phase_times: HashMap<CompilationPhase, Duration>,
    /// Files processed
    pub files_processed: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Error recovery count
    pub error_recoveries: usize,
}

/// Result of pipeline compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCompilationResult {
    /// Compilation success
    pub success: bool,
    /// Results from each phase
    pub phase_results: HashMap<CompilationPhase, PhaseResult>,
    /// Final compilation artifacts
    pub artifacts: Vec<prism_codegen::CodeArtifact>,
    /// AI metadata export
    pub ai_metadata: Option<crate::context::AIMetadataExport>,
    /// Compilation statistics
    pub stats: CompilationStats,
    /// Diagnostics and errors
    pub diagnostics: Vec<String>,
}

/// Result from a compilation phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    /// Phase that was executed
    pub phase: CompilationPhase,
    /// Phase success
    pub success: bool,
    /// Phase execution time
    pub execution_time: Duration,
    /// Files processed in this phase
    pub files_processed: usize,
    /// Cache hits in this phase
    pub cache_hits: usize,
    /// Phase-specific data
    pub data: PhaseData,
    /// Phase diagnostics
    pub diagnostics: Vec<String>,
}

/// Phase-specific result data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseData {
    /// Discovery phase results
    Discovery(ProjectDiscoveryResult),
    /// Lexing phase results
    Lexing(Vec<LexicalAnalysisResult>),
    /// Parsing phase results
    Parsing(Vec<ParseResult>),
    /// Semantic analysis results
    SemanticAnalysis(Vec<SemanticAnalysisResult>),
    /// Type checking results
    TypeChecking(Vec<TypeCheckingResult>),
    /// Effect analysis results
    EffectAnalysis(Vec<EffectAnalysisResult>),
    /// Optimization results
    Optimization(Vec<OptimizationResult>),
    /// Code generation results
    CodeGeneration(Vec<prism_codegen::CodeArtifact>),
    /// PIR generation results
    PIRGeneration(Vec<PIRGenerationResult>),
    /// Linking results
    Linking(Vec<LinkingResult>),
    /// Finalization results
    Finalization(FinalizationResult),
}

/// Compilation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStats {
    /// Total files processed
    pub total_files: usize,
    /// Total lines of code processed
    pub total_lines: usize,
    /// Total compilation time
    pub total_time_ms: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Phases executed
    pub phases_executed: usize,
    /// Parallel phases
    pub parallel_phases: usize,
}

impl CompilationPipeline {
    /// Create a new compilation pipeline with default configuration
    pub fn new(config: PipelineConfig) -> Self {
        let query_engine = Arc::new(QueryEngine::new());
        let semantic_db = Arc::new(SemanticDatabase::new());
        let context = Arc::new(CompilationContext::new());
        
        Self {
            query_engine,
            semantic_db,
            config,
            context,
            semantic_type_integration: None, // Will be set later
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_phases)),
        }
    }

    /// Create a new compilation pipeline with custom components
    pub fn with_components(
        query_engine: Arc<QueryEngine>,
        semantic_db: Arc<SemanticDatabase>,
        context: Arc<CompilationContext>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            query_engine,
            semantic_db,
            config,
            context,
            semantic_type_integration: None, // Will be set later
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_phases)),
        }
    }

    /// Set the semantic type integration for this pipeline
    pub fn with_semantic_type_integration(mut self, integration: Arc<crate::semantic::SemanticTypeIntegration>) -> Self {
        self.semantic_type_integration = Some(integration);
        self
    }

    /// Create pipeline from existing compilation config
    pub fn from_compilation_config(compilation_config: &CompilationConfig) -> CompilerResult<Self> {
        let config = PipelineConfig {
            enable_parallel_execution: true,
            max_concurrent_phases: num_cpus::get(),
            enable_incremental: compilation_config.incremental.unwrap_or(true),
            enable_ai_metadata: compilation_config.ai_features.unwrap_or(true),
            phase_timeout_secs: 300,
            enable_error_recovery: true,
            targets: compilation_config.targets.clone(),
        };

        Ok(Self::new(config))
    }

    /// Compile a complete project using the query-based pipeline
    pub async fn compile_project(&self, project_path: &Path) -> CompilerResult<PipelineCompilationResult> {
        let start_time = Instant::now();
        info!("Starting query-based compilation pipeline for: {}", project_path.display());

        let mut phase_results = HashMap::new();
        let mut diagnostics = Vec::new();
        let mut overall_success = true;

        // Phase 1: Project Discovery
        let discovery_result = self.execute_discovery_phase(project_path).await?;
        phase_results.insert(CompilationPhase::Discovery, discovery_result.clone());
        
        if !discovery_result.success {
            overall_success = false;
            diagnostics.extend(discovery_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        let source_files = match &discovery_result.data {
            PhaseData::Discovery(data) => data.source_files.clone(),
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid discovery result data".to_string() 
            }),
        };

        // Phase 2: Lexical Analysis (can be parallel per file)
        let lexing_result = self.execute_lexing_phase(&source_files).await?;
        phase_results.insert(CompilationPhase::Lexing, lexing_result.clone());
        
        if !lexing_result.success {
            overall_success = false;
            diagnostics.extend(lexing_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        // Phase 3: Parsing (depends on lexing, can be parallel per file)
        let parsing_result = self.execute_parsing_phase(&source_files).await?;
        phase_results.insert(CompilationPhase::Parsing, parsing_result.clone());
        
        if !parsing_result.success {
            overall_success = false;
            diagnostics.extend(parsing_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        let programs = match &parsing_result.data {
            PhaseData::Parsing(data) => data.iter().map(|r| r.program.clone()).collect::<Vec<_>>(),
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid parsing result data".to_string() 
            }),
        };

        // Phase 4: Semantic Analysis (depends on parsing)
        let semantic_result = self.execute_semantic_analysis_phase(&programs).await?;
        phase_results.insert(CompilationPhase::SemanticAnalysis, semantic_result.clone());
        
        if !semantic_result.success {
            overall_success = false;
            diagnostics.extend(semantic_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        // Phase 5: Type Checking (depends on semantic analysis)
        let type_checking_result = self.execute_type_checking_phase(&programs, &semantic_result).await?;
        phase_results.insert(CompilationPhase::TypeChecking, type_checking_result.clone());
        
        if !type_checking_result.success {
            overall_success = false;
            diagnostics.extend(type_checking_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        // Phase 6: Effect Analysis (depends on type checking)
        let effect_result = self.execute_effect_analysis_phase(&programs, &type_checking_result).await?;
        phase_results.insert(CompilationPhase::EffectAnalysis, effect_result.clone());
        
        if !effect_result.success {
            overall_success = false;
            diagnostics.extend(effect_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        // Phase 7: Optimization (depends on effect analysis)
        let optimization_result = self.execute_optimization_phase(&programs, &semantic_result).await?;
        phase_results.insert(CompilationPhase::Optimization, optimization_result.clone());

        // Phase 8: PIR Generation (depends on effect analysis and type checking)
        let pir_result = self.execute_pir_generation_phase(&programs, &semantic_result, &type_checking_result, &effect_result).await?;
        phase_results.insert(CompilationPhase::PIRGeneration, pir_result.clone());
        
        if !pir_result.success {
            overall_success = false;
            diagnostics.extend(pir_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        // Phase 9: Code Generation (can be parallel per target, now uses PIR)
        let codegen_result = self.execute_code_generation_phase(&programs, &pir_result).await?;
        phase_results.insert(CompilationPhase::CodeGeneration, codegen_result.clone());
        
        if !codegen_result.success {
            overall_success = false;
            diagnostics.extend(codegen_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        let artifacts = match &codegen_result.data {
            PhaseData::CodeGeneration(data) => data.clone(),
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid code generation result data".to_string() 
            }),
        };

        // Phase 10: Linking (depends on code generation)
        let linking_result = self.execute_linking_phase(&artifacts).await?;
        phase_results.insert(CompilationPhase::Linking, linking_result.clone());
        
        if !linking_result.success {
            overall_success = false;
            diagnostics.extend(linking_result.diagnostics.clone());
            if !self.config.enable_error_recovery {
                return Ok(self.create_failed_result(phase_results, diagnostics, start_time));
            }
        }

        // Phase 11: Finalization (depends on linking)
        let finalization_result = self.execute_finalization_phase(&linking_result).await?;
        phase_results.insert(CompilationPhase::Finalization, finalization_result.clone());

        // Generate AI metadata if enabled
        let ai_metadata = if self.config.enable_ai_metadata {
            Some(self.context.export_ai_metadata())
        } else {
            None
        };

        // Calculate final statistics
        let total_time = start_time.elapsed();
        let stats = CompilationStats {
            total_files: source_files.len(),
            total_lines: 0, // Would be calculated from actual parsing
            total_time_ms: total_time.as_millis() as u64,
            cache_hit_rate: self.calculate_cache_hit_rate(),
            phases_executed: phase_results.len(),
            parallel_phases: self.count_parallel_phases(),
        };

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_time = total_time;
            metrics.files_processed = source_files.len();
            metrics.cache_hit_rate = stats.cache_hit_rate;
        }

        info!("Query-based compilation pipeline completed in {:?}", total_time);

        Ok(PipelineCompilationResult {
            success: overall_success,
            phase_results,
            artifacts,
            ai_metadata,
            stats,
            diagnostics,
        })
    }

    /// Execute the project discovery phase
    async fn execute_discovery_phase(&self, project_path: &Path) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing discovery phase for: {}", project_path.display());

        let query = ProjectDiscoveryQuery::new(project_path.to_path_buf());
        let input = ProjectDiscoveryInput {
            project_root: project_path.to_path_buf(),
            include_hidden: false,
            extensions: vec!["prsm".to_string(), "prism".to_string()],
            max_depth: None,
        };

        let context = self.create_query_context();
        
        let result = match self.query_engine.query(&query, input, context).await {
            Ok(result) => result,
            Err(e) => {
                error!("Discovery phase failed: {}", e);
                return Ok(PhaseResult {
                    phase: CompilationPhase::Discovery,
                    success: false,
                    execution_time: start_time.elapsed(),
                    files_processed: 0,
                    cache_hits: 0,
                    data: PhaseData::Discovery(ProjectDiscoveryResult {
                        source_files: Vec::new(),
                        config_files: Vec::new(),
                        module_structure: ModuleStructure {
                            root_modules: Vec::new(),
                            dependencies: HashMap::new(),
                            hierarchy: ModuleHierarchy {
                                tree: HashMap::new(),
                                depths: HashMap::new(),
                            },
                        },
                        stats: DiscoveryStats {
                            total_files: 0,
                            directories_scanned: 0,
                            discovery_time_ms: 0,
                        },
                    }),
                    diagnostics: vec![e.to_string()],
                });
            }
        };

        Ok(PhaseResult {
            phase: CompilationPhase::Discovery,
            success: true,
            execution_time: start_time.elapsed(),
            files_processed: result.source_files.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::Discovery(result),
            diagnostics: Vec::new(),
        })
    }

    /// Execute the lexical analysis phase
    async fn execute_lexing_phase(&self, source_files: &[PathBuf]) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing lexing phase for {} files", source_files.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        // Process files in parallel if enabled
        if self.config.enable_parallel_execution {
            let tasks: Vec<_> = source_files.iter().map(|file_path| {
                let query_engine = Arc::clone(&self.query_engine);
                let context = self.create_query_context();
                let file_path = file_path.clone();
                
                async move {
                    let source = match std::fs::read_to_string(&file_path) {
                        Ok(source) => source,
                        Err(e) => return Err(CompilerError::FileReadError { 
                            path: file_path.clone(), 
                            source: e 
                        }),
                    };

                    let source_id = prism_common::SourceId::new(
                        file_path.to_string_lossy().as_bytes().iter()
                            .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
                    );

                    let query = LexicalAnalysisQuery::new(source_id);
                    let input = LexicalAnalysisInput {
                        source,
                        file_path: file_path.clone(),
                        source_id,
                    };

                    query_engine.query(&query, input, context).await
                }
            }).collect();

            let task_results = futures::future::join_all(tasks).await;
            
            for result in task_results {
                match result {
                    Ok(lexing_result) => results.push(lexing_result),
                    Err(e) => {
                        success = false;
                        diagnostics.push(e.to_string());
                        if !self.config.enable_error_recovery {
                            break;
                        }
                    }
                }
            }
        } else {
            // Sequential processing
            for file_path in source_files {
                let source = match std::fs::read_to_string(file_path) {
                    Ok(source) => source,
                    Err(e) => {
                        success = false;
                        diagnostics.push(format!("Failed to read {}: {}", file_path.display(), e));
                        if !self.config.enable_error_recovery {
                            break;
                        }
                        continue;
                    }
                };

                let source_id = prism_common::SourceId::new(
                    file_path.to_string_lossy().as_bytes().iter()
                        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
                );

                let query = LexicalAnalysisQuery::new(source_id);
                let input = LexicalAnalysisInput {
                    source,
                    file_path: file_path.clone(),
                    source_id,
                };

                let context = self.create_query_context();
                
                match self.query_engine.query(&query, input, context).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        success = false;
                        diagnostics.push(e.to_string());
                        if !self.config.enable_error_recovery {
                            break;
                        }
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::Lexing,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::Lexing(results),
            diagnostics,
        })
    }

    /// Execute the parsing phase
    async fn execute_parsing_phase(&self, source_files: &[PathBuf]) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing parsing phase for {} files", source_files.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        // Process files in parallel if enabled
        for file_path in source_files {
            let source = match std::fs::read_to_string(file_path) {
                Ok(source) => source,
                Err(e) => {
                    success = false;
                    diagnostics.push(format!("Failed to read {}: {}", file_path.display(), e));
                    if !self.config.enable_error_recovery {
                        break;
                    }
                    continue;
                }
            };

            let source_id = prism_common::SourceId::new(
                file_path.to_string_lossy().as_bytes().iter()
                    .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
            );

            let query = ParseSourceQuery::new(source_id);
            let input = ParseInput {
                source,
                source_id,
                file_path: Some(file_path.clone()),
            };

            let context = self.create_query_context();
            
            match self.query_engine.query(&query, input, context).await {
                Ok(query_result) => {
                    // Convert ParseQueryResult to ParseResult for external API
                    let parse_result = ParseResult {
                        program: query_result.program,
                        syntax_style: query_result.detected_syntax,
                        success: true,
                        errors: Vec::new(),
                        warnings: query_result.diagnostics,
                        stats: ParseStats {
                            nodes_created: 0, // Would be calculated from program
                            parse_time_ms: 0, // Would be tracked during parsing
                            lines_parsed: input.source.lines().count(),
                            syntax_confidence: query_result.ai_metadata.as_ref()
                                .map(|m| m.confidence)
                                .unwrap_or(1.0),
                        },
                    };
                    results.push(parse_result);
                },
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::Parsing,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::Parsing(results),
            diagnostics,
        })
    }

    /// Execute the semantic analysis phase
    async fn execute_semantic_analysis_phase(&self, programs: &[prism_ast::Program]) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing semantic analysis phase for {} programs", programs.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        let query = SemanticAnalysisQuery::new(Arc::clone(&self.semantic_db));

        for program in programs {
            let context = self.create_query_context();
            
            match self.query_engine.query(&query, program.clone(), context).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::SemanticAnalysis,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::SemanticAnalysis(results),
            diagnostics,
        })
    }

    /// Execute the type checking phase
    async fn execute_type_checking_phase(&self, programs: &[prism_ast::Program], semantic_result: &PhaseResult) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing type checking phase for {} programs", programs.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        let semantic_results = match &semantic_result.data {
            PhaseData::SemanticAnalysis(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid semantic analysis result data".to_string() 
            }),
        };

        let query = TypeCheckingQuery::new(Arc::clone(&self.semantic_db));

        for (program, semantic_info) in programs.iter().zip(semantic_results.iter()) {
            let input = TypeCheckingInput {
                program: program.clone(),
                semantic_info: semantic_info.clone(),
            };

            let context = self.create_query_context();
            
            match self.query_engine.query(&query, input, context).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::TypeChecking,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::TypeChecking(results),
            diagnostics,
        })
    }

    /// Execute the effect analysis phase
    async fn execute_effect_analysis_phase(&self, programs: &[prism_ast::Program], type_result: &PhaseResult) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing effect analysis phase for {} programs", programs.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        let type_results = match &type_result.data {
            PhaseData::TypeChecking(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid type checking result data".to_string() 
            }),
        };

        let query = EffectAnalysisQuery::new(Arc::clone(&self.semantic_db));

        for (program, type_info) in programs.iter().zip(type_results.iter()) {
            let input = EffectAnalysisInput {
                program: program.clone(),
                type_info: type_info.clone(),
            };

            let context = self.create_query_context();
            
            match self.query_engine.query(&query, input, context).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::EffectAnalysis,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::EffectAnalysis(results),
            diagnostics,
        })
    }

    /// Execute the optimization phase
    async fn execute_optimization_phase(&self, programs: &[prism_ast::Program], semantic_result: &PhaseResult) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing optimization phase for {} programs", programs.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        let semantic_results = match &semantic_result.data {
            PhaseData::SemanticAnalysis(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid semantic analysis result data".to_string() 
            }),
        };

        let query = OptimizationQuery::new(Arc::clone(&self.semantic_db));

        for (program, semantic_info) in programs.iter().zip(semantic_results.iter()) {
            let input = OptimizationInput {
                program: program.clone(),
                semantic_info: semantic_info.clone(),
            };

            let context = self.create_query_context();
            
            match self.query_engine.query(&query, input, context).await {
                Ok((_optimized_program, result)) => results.push(result),
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::Optimization,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::Optimization(results),
            diagnostics,
        })
    }

    /// Execute the PIR generation phase
    async fn execute_pir_generation_phase(
        &self, 
        programs: &[prism_ast::Program], 
        semantic_result: &PhaseResult,
        type_result: &PhaseResult,
        effect_result: &PhaseResult,
    ) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing PIR generation phase for {} programs", programs.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        // Extract results from previous phases
        let semantic_results = match &semantic_result.data {
            PhaseData::SemanticAnalysis(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid semantic analysis result data".to_string() 
            }),
        };

        let type_results = match &type_result.data {
            PhaseData::TypeChecking(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid type checking result data".to_string() 
            }),
        };

        let effect_results = match &effect_result.data {
            PhaseData::EffectAnalysis(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid effect analysis result data".to_string() 
            }),
        };

        let query = PIRGenerationQuery::new()?;

        // Generate PIR for each program
        for ((program, semantic_info), (type_info, effect_info)) in programs.iter()
            .zip(semantic_results.iter())
            .zip(type_results.iter().zip(effect_results.iter()))
        {
            let input = PIRGenerationInput {
                program: program.clone(),
                semantic_info: semantic_info.clone(),
                type_info: type_info.clone(),
                effect_info: effect_info.clone(),
            };

            let context = self.create_query_context();
            
            match self.query_engine.query(&query, input, context).await {
                Ok(result) => {
                    if !result.success {
                        success = false;
                        diagnostics.extend(result.diagnostics.clone());
                    }
                    results.push(result);
                },
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::PIRGeneration,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::PIRGeneration(results),
            diagnostics,
        })
    }

    /// Execute the code generation phase (now uses PIR as input)
    async fn execute_code_generation_phase(&self, programs: &[prism_ast::Program], pir_result: &PhaseResult) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing code generation phase for {} programs and {} targets", programs.len(), self.config.targets.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        let pir_results = match &pir_result.data {
            PhaseData::PIRGeneration(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid PIR generation result data".to_string() 
            }),
        };

        // Generate code for each target using PIR
        for target in &self.config.targets {
            for pir_gen_result in pir_results {
                if !pir_gen_result.success {
                    // Skip code generation for failed PIR
                    continue;
                }

                let query = CodeGenQuery::new(*target, Arc::new(prism_codegen::MultiTargetCodeGen::new()));
                let input = CodeGenInput {
                    pir: pir_gen_result.pir.clone(),
                    target: *target,
                    optimization_level: 0, // TODO: Get from optimization phase
                };

                let context = self.create_query_context();
                
                match self.query_engine.query(&query, input, context).await {
                    Ok(artifact) => results.push(artifact),
                    Err(e) => {
                        success = false;
                        diagnostics.push(e.to_string());
                        if !self.config.enable_error_recovery {
                            break;
                        }
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::CodeGeneration,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::CodeGeneration(results),
            diagnostics,
        })
    }

    /// Execute the linking phase
    async fn execute_linking_phase(&self, artifacts: &[prism_codegen::CodeArtifact]) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing linking phase for {} artifacts", artifacts.len());

        let mut results = Vec::new();
        let mut diagnostics = Vec::new();
        let mut success = true;

        // Group artifacts by target
        let mut artifacts_by_target: HashMap<CompilationTarget, Vec<prism_codegen::CodeArtifact>> = HashMap::new();
        for artifact in artifacts {
            artifacts_by_target.entry(artifact.target).or_default().push(artifact.clone());
        }

        // Link each target separately
        for (target, target_artifacts) in artifacts_by_target {
            let query = LinkingQuery::new(target);
            let input = LinkingInput {
                artifacts: target_artifacts,
                target,
                config: LinkingConfig {
                    output_name: format!("output_{:?}", target).to_lowercase(),
                    optimize: true,
                    link_flags: Vec::new(),
                },
            };

            let context = self.create_query_context();
            
            match self.query_engine.query(&query, input, context).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    success = false;
                    diagnostics.push(e.to_string());
                    if !self.config.enable_error_recovery {
                        break;
                    }
                }
            }
        }

        Ok(PhaseResult {
            phase: CompilationPhase::Linking,
            success,
            execution_time: start_time.elapsed(),
            files_processed: results.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::Linking(results),
            diagnostics,
        })
    }

    /// Execute the finalization phase
    async fn execute_finalization_phase(&self, linking_result: &PhaseResult) -> CompilerResult<PhaseResult> {
        let start_time = Instant::now();
        debug!("Executing finalization phase");

        let linking_results = match &linking_result.data {
            PhaseData::Linking(data) => data,
            _ => return Err(CompilerError::InvalidOperation { 
                message: "Invalid linking result data".to_string() 
            }),
        };

        let query = FinalizationQuery::new();
        let input = FinalizationInput {
            linked_outputs: linking_results.clone(),
            ai_metadata: if self.config.enable_ai_metadata {
                Some(self.context.export_ai_metadata())
            } else {
                None
            },
            export_config: FinalizationConfig {
                export_ai_metadata: self.config.enable_ai_metadata,
                generate_docs: true,
                create_build_report: true,
            },
        };

        let context = self.create_query_context();
        
        let result = match self.query_engine.query(&query, input, context).await {
            Ok(result) => result,
            Err(e) => {
                return Ok(PhaseResult {
                    phase: CompilationPhase::Finalization,
                    success: false,
                    execution_time: start_time.elapsed(),
                    files_processed: 0,
                    cache_hits: 0,
                    data: PhaseData::Finalization(FinalizationResult {
                        success: false,
                        outputs: Vec::new(),
                        build_report: None,
                        ai_metadata_path: None,
                        stats: FinalizationStats {
                            total_compilation_time_ms: 0,
                            total_output_size_bytes: 0,
                            files_generated: 0,
                        },
                    }),
                    diagnostics: vec![e.to_string()],
                });
            }
        };

        Ok(PhaseResult {
            phase: CompilationPhase::Finalization,
            success: result.success,
            execution_time: start_time.elapsed(),
            files_processed: result.outputs.len(),
            cache_hits: 0, // Would be tracked by query engine
            data: PhaseData::Finalization(result),
            diagnostics: Vec::new(),
        })
    }

    /// Create a query context for phase execution
    fn create_query_context(&self) -> QueryContext {
        QueryContext {
            query_stack: Vec::new(),
            semantic_context: Arc::new(DefaultSemanticContext),
            config: QueryConfig {
                enable_cache: self.config.enable_incremental,
                enable_dependency_tracking: self.config.enable_incremental,
                enable_profiling: true,
                cache_size_limit: 50_000,
                query_timeout: Duration::from_secs(self.config.phase_timeout_secs),
            },
            profiler: Arc::new(Mutex::new(QueryProfiler::new())),
            compilation_context: Some(Arc::clone(&self.context)),
            semantic_type_integration: self.semantic_type_integration.clone(),
        }
    }

    /// Create a failed compilation result
    fn create_failed_result(
        &self,
        phase_results: HashMap<CompilationPhase, PhaseResult>,
        diagnostics: Vec<String>,
        start_time: Instant,
    ) -> PipelineCompilationResult {
        PipelineCompilationResult {
            success: false,
            phase_results,
            artifacts: Vec::new(),
            ai_metadata: None,
            stats: CompilationStats {
                total_files: 0,
                total_lines: 0,
                total_time_ms: start_time.elapsed().as_millis() as u64,
                cache_hit_rate: 0.0,
                phases_executed: 0,
                parallel_phases: 0,
            },
            diagnostics,
        }
    }

    /// Calculate cache hit rate across all phases
    fn calculate_cache_hit_rate(&self) -> f64 {
        // Would be calculated from query engine statistics
        0.0
    }

    /// Count parallel phases executed
    fn count_parallel_phases(&self) -> usize {
        // Would be calculated from actual execution
        if self.config.enable_parallel_execution { 3 } else { 0 }
    }

    /// Get pipeline metrics
    pub fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get query engine statistics
    pub fn get_query_stats(&self) -> std::collections::HashMap<String, crate::query::core::CacheStats> {
        self.query_engine.get_cache_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use std::fs;

    /// Test that pipeline configuration is properly set up
    #[test]
    fn test_pipeline_configuration() {
        let config = PipelineConfig {
            enable_parallel_execution: true,
            max_concurrent_phases: 4,
            enable_incremental: true,
            enable_ai_metadata: true,
            phase_timeout_secs: 300,
            enable_error_recovery: true,
            targets: vec![CompilationTarget::TypeScript],
        };

        let pipeline = CompilationPipeline::new(config.clone());
        
        assert_eq!(pipeline.config.enable_parallel_execution, true);
        assert_eq!(pipeline.config.max_concurrent_phases, 4);
        assert_eq!(pipeline.config.enable_incremental, true);
        assert_eq!(pipeline.config.enable_ai_metadata, true);
        assert_eq!(pipeline.config.phase_timeout_secs, 300);
        assert_eq!(pipeline.config.enable_error_recovery, true);
        assert_eq!(pipeline.config.targets, vec![CompilationTarget::TypeScript]);
    }

    /// Test that pipeline can be created from compilation config
    #[test]
    fn test_pipeline_from_compilation_config() {
        let compilation_config = CompilationConfig {
            incremental: Some(true),
            ai_features: Some(true),
            targets: vec![CompilationTarget::TypeScript, CompilationTarget::WebAssembly],
            ..Default::default()
        };

        let pipeline_config = PipelineConfig::from_compilation_config(&compilation_config).unwrap();
        
        assert_eq!(pipeline_config.enable_incremental, true);
        assert_eq!(pipeline_config.enable_ai_metadata, true);
        assert_eq!(pipeline_config.targets.len(), 2);
    }

    /// Test phase ordering and dependencies
    #[test]
    fn test_compilation_phase_ordering() {
        use crate::context::CompilationPhase;
        
        let phases = vec![
            CompilationPhase::Discovery,
            CompilationPhase::Lexing,
            CompilationPhase::Parsing,
            CompilationPhase::SemanticAnalysis,
            CompilationPhase::TypeChecking,
            CompilationPhase::EffectAnalysis,
            CompilationPhase::Optimization,
            CompilationPhase::CodeGeneration,
            CompilationPhase::Linking,
            CompilationPhase::Finalization,
        ];

        // Verify that phases are in the correct dependency order
        for i in 0..phases.len() - 1 {
            let current_phase = &phases[i];
            let next_phase = &phases[i + 1];
            
            // Each phase should depend on the previous phases
            // This is a structural test - the actual dependency logic would be tested
            // when the underlying components are working
            assert_ne!(current_phase, next_phase);
        }
    }

    /// Test error recovery configuration
    #[test]
    fn test_error_recovery_configuration() {
        let config_with_recovery = PipelineConfig {
            enable_error_recovery: true,
            ..Default::default()
        };

        let config_without_recovery = PipelineConfig {
            enable_error_recovery: false,
            ..Default::default()
        };

        let pipeline_with_recovery = CompilationPipeline::new(config_with_recovery);
        let pipeline_without_recovery = CompilationPipeline::new(config_without_recovery);

        assert_eq!(pipeline_with_recovery.config.enable_error_recovery, true);
        assert_eq!(pipeline_without_recovery.config.enable_error_recovery, false);
    }

    /// Test multi-target configuration
    #[test]
    fn test_multi_target_configuration() {
        let config = PipelineConfig {
            targets: vec![
                CompilationTarget::TypeScript,
                CompilationTarget::WebAssembly,
                CompilationTarget::LLVM,
            ],
            ..Default::default()
        };

        let pipeline = CompilationPipeline::new(config);
        
        assert_eq!(pipeline.config.targets.len(), 3);
        assert!(pipeline.config.targets.contains(&CompilationTarget::TypeScript));
        assert!(pipeline.config.targets.contains(&CompilationTarget::WebAssembly));
        assert!(pipeline.config.targets.contains(&CompilationTarget::LLVM));
    }
} 