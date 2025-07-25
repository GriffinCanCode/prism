//! Semantic Analysis Queries
//!
//! This module provides specialized queries for semantic analysis operations,
//! integrating with the compiler's query system to provide incremental semantic analysis.

use crate::error::{CompilerError, CompilerResult};
use crate::query::core::{CompilerQuery, QueryContext, CacheKey, QueryId, InvalidationTrigger};
use crate::semantic::SemanticDatabase;
use crate::context::{CompilationPhase, CompilationTarget};
use prism_ast::Program;
use prism_common::{NodeId, SourceId};
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Query for performing semantic analysis on a parsed program
#[derive(Debug, Clone)]
pub struct SemanticAnalysisQuery {
    /// Reference to the semantic database
    semantic_db: Arc<SemanticDatabase>,
}

/// Result of semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysisResult {
    /// Whether analysis was successful
    pub success: bool,
    /// Semantic metadata extracted
    pub semantic_metadata: SemanticMetadata,
    /// Diagnostics from analysis
    pub diagnostics: Vec<String>,
    /// AI-readable context
    pub ai_context: Option<AISemanticContext>,
}

/// Semantic metadata from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMetadata {
    /// Symbol table information
    pub symbols: Vec<SymbolInfo>,
    /// Type information
    pub types: Vec<TypeInfo>,
    /// Effect information
    pub effects: Vec<EffectInfo>,
    /// Module information
    pub modules: Vec<ModuleInfo>,
}

/// Symbol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: String,
    /// Location in source
    pub location: Option<NodeId>,
    /// Type information
    pub type_info: Option<String>,
}

/// Type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type constraints
    pub constraints: Vec<String>,
    /// Semantic metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Effect information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectInfo {
    /// Effect type
    pub effect_type: String,
    /// Capabilities required
    pub capabilities: Vec<String>,
    /// Side effects
    pub side_effects: Vec<String>,
}

/// Module information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    /// Module capabilities
    pub capabilities: Vec<String>,
    /// Business context
    pub business_context: String,
}

/// AI-readable semantic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISemanticContext {
    /// Overall quality score
    pub quality_score: f64,
    /// Semantic patterns detected
    pub patterns: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

impl SemanticAnalysisQuery {
    /// Create a new semantic analysis query
    pub fn new(semantic_db: Arc<SemanticDatabase>) -> Self {
        Self { semantic_db }
    }

    /// Extract symbols from the program (delegates to symbols subsystem)
    fn extract_symbols_from_program(&self, program: &Program) -> CompilerResult<Vec<SymbolInfo>> {
        let mut symbols = Vec::new();
        
        // Extract symbols from program items
        for item in &program.items {
            match item.data() {
                prism_ast::Item::Function(func) => {
                    symbols.push(SymbolInfo {
                        name: func.name.to_string(),
                        kind: "function".to_string(),
                        location: Some(item.span()),
                        type_info: func.return_type.as_ref().map(|t| format!("{:?}", t.data())),
                    });
                }
                prism_ast::Item::Struct(struct_def) => {
                    symbols.push(SymbolInfo {
                        name: struct_def.name.clone(),
                        kind: "struct".to_string(),
                        location: Some(item.span()),
                        type_info: Some("composite".to_string()),
                    });
                }
                prism_ast::Item::TypeAlias(type_alias) => {
                    symbols.push(SymbolInfo {
                        name: type_alias.name.clone(),
                        kind: "type_alias".to_string(),
                        location: Some(item.span()),
                        type_info: Some(format!("{:?}", type_alias.type_expression.data())),
                    });
                }
                _ => {
                    // Handle other item types as needed
                }
            }
        }
        
        Ok(symbols)
    }
    
    /// Convert semantic types to metadata format
    fn convert_semantic_types_to_metadata(
        &self,
        integration_result: &crate::semantic::SemanticTypeIntegrationResult,
    ) -> CompilerResult<Vec<TypeInfo>> {
        let mut types = Vec::new();
        
        for (node_id, semantic_type) in &integration_result.semantic_types {
            match semantic_type {
                prism_semantic::SemanticType::Complex { name, base_type, constraints, .. } => {
                    types.push(TypeInfo {
                        name: name.clone(),
                        kind: format!("{:?}", base_type),
                        constraints: constraints.iter().map(|c| c.name.clone()).collect(),
                        location: None, // TODO: Extract from semantic type
                    });
                }
                prism_semantic::SemanticType::Primitive(prim) => {
                    types.push(TypeInfo {
                        name: format!("{:?}", prim),
                        kind: "primitive".to_string(),
                        constraints: Vec::new(),
                        location: None,
                    });
                }
                _ => {
                    // Handle other semantic type variants
                    types.push(TypeInfo {
                        name: "unknown".to_string(),
                        kind: "unknown".to_string(),
                        constraints: Vec::new(),
                        location: None,
                    });
                }
            }
        }
        
        Ok(types)
    }
    
    /// Extract effects from the program (placeholder implementation)
    fn extract_effects_from_program(&self, _program: &Program) -> CompilerResult<Vec<EffectInfo>> {
        // This would integrate with the effects subsystem
        // For now, return empty vec
        Ok(Vec::new())
    }
    
    /// Extract modules from the program
    fn extract_modules_from_program(&self, program: &Program) -> CompilerResult<Vec<ModuleInfo>> {
        // Extract basic module information
        // This would typically integrate with the module registry
        Ok(vec![
            ModuleInfo {
                name: "main".to_string(), // Default module name
                path: None,
                items: program.items.len(),
                dependencies: Vec::new(), // TODO: Extract dependencies
            }
        ])
    }
    
    /// Generate AI context from semantic analysis results
    fn generate_ai_context(
        &self,
        program: &Program,
        integration_result: &crate::semantic::SemanticTypeIntegrationResult,
    ) -> CompilerResult<AISemanticContext> {
        let quality_score = if integration_result.metadata.failed_conversions == 0 {
            0.9 // High quality if no conversion failures
        } else {
            0.6 // Lower quality if there were issues
        };
        
        let patterns = vec![
            "semantic_types_present".to_string(),
            "structured_code".to_string(),
        ];
        
        let mut recommendations = Vec::new();
        
        if integration_result.metadata.ast_types_processed == 0 {
            recommendations.push("Consider adding semantic type annotations for better analysis".to_string());
        }
        
        if !integration_result.warnings.is_empty() {
            recommendations.push("Review semantic type warnings for potential improvements".to_string());
        }
        
        Ok(AISemanticContext {
            quality_score,
            patterns,
            recommendations,
        })
    }
}

#[async_trait]
impl CompilerQuery<Program, SemanticAnalysisResult> for SemanticAnalysisQuery {
    async fn execute(&self, input: Program, context: QueryContext) -> CompilerResult<SemanticAnalysisResult> {
        use tracing::{info, debug, warn};
        
        info!("Executing semantic analysis with integrated semantic type system");
        
        // 1. **Get Semantic Type Integration from Context**
        let semantic_type_integration = context.get_semantic_type_integration()
            .ok_or_else(|| CompilerError::MissingDependency {
                dependency: "semantic_type_integration".to_string(),
                context: "semantic analysis query".to_string(),
            })?;
        
        // Get the real compilation context from the query context
        let compilation_context = context.get_compilation_context()
            .ok_or_else(|| CompilerError::MissingDependency {
                dependency: "compilation_context".to_string(),
                context: "semantic analysis query".to_string(),
            })?;
        
        // Integrate semantic types from the program
        let semantic_integration_result = semantic_type_integration
            .integrate_program_types(&input, &compilation_context)
            .await?;
        
        debug!(
            "Semantic type integration completed: {} types processed, {} successful conversions",
            semantic_integration_result.metadata.ast_types_processed,
            semantic_integration_result.metadata.successful_conversions
        );
        
        // 2. **Symbol Extraction**: Extract symbols using the existing symbol table system
        // (This delegates to the symbols subsystem, following SoC principles)
        let symbols = self.extract_symbols_from_program(&input)?;
        
        // 3. **Type Information**: Convert semantic types to query result format
        let types = self.convert_semantic_types_to_metadata(&semantic_integration_result)?;
        
        // 4. **Effect Analysis**: Basic effect extraction (placeholder for now)
        let effects = self.extract_effects_from_program(&input)?;
        
        // 5. **Module Analysis**: Extract module information
        let modules = self.extract_modules_from_program(&input)?;
        
        let semantic_metadata = SemanticMetadata {
            symbols,
            types,
            effects,
            modules,
        };

        // 6. **AI Context Generation**: Generate AI-comprehensible context
        let ai_context = self.generate_ai_context(&input, &semantic_integration_result)?;
        
        // 7. **Collect Diagnostics**: Include any warnings from semantic type integration
        let mut diagnostics = Vec::new();
        for warning in semantic_integration_result.warnings {
            diagnostics.push(warning);
        }

        Ok(SemanticAnalysisResult {
            success: semantic_integration_result.metadata.failed_conversions == 0,
            semantic_metadata,
            diagnostics,
            ai_context: Some(ai_context),
        })
    }

    fn cache_key(&self, input: &Program) -> CacheKey {
        CacheKey::from_input("semantic_analysis", input)
    }

    async fn dependencies(&self, _input: &Program, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Semantic analysis depends on parsing being complete
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &Program) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers.insert(InvalidationTrigger::OptimizationLevelChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "semantic_analysis"
    }
}

/// Query for parsing source code into an AST
#[derive(Debug, Clone)]
pub struct ParseSourceQuery {
    /// Source ID for tracking
    source_id: SourceId,
}

/// Input for parsing query
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ParseInput {
    /// Source code to parse
    pub source: String,
    /// Source ID
    pub source_id: SourceId,
    /// File path for context
    pub file_path: Option<std::path::PathBuf>,
}

/// Result of parsing query (compiler-internal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseQueryResult {
    /// Parsed program
    pub program: Program,
    /// Syntax style detected
    pub detected_syntax: String,
    /// Parsing diagnostics
    pub diagnostics: Vec<String>,
    /// AI metadata from parsing
    pub ai_metadata: Option<ParsingAIMetadata>,
}

/// AI metadata from parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingAIMetadata {
    /// Parsing confidence score
    pub confidence: f64,
    /// Syntax patterns detected
    pub syntax_patterns: Vec<String>,
    /// Code quality indicators
    pub quality_indicators: Vec<String>,
}

impl ParseSourceQuery {
    /// Create a new parse source query
    pub fn new(source_id: SourceId) -> Self {
        Self { source_id }
    }
}

#[async_trait]
impl CompilerQuery<ParseInput, ParseQueryResult> for ParseSourceQuery {
    async fn execute(&self, input: ParseInput, _context: QueryContext) -> CompilerResult<ParseQueryResult> {
        use prism_syntax::{parse_with_orchestrator, detect_syntax_style};
        
        // Step 1: Detect syntax style
        let detection_result = detect_syntax_style(&input.source);
        
        // Step 2: Parse using prism-syntax orchestrator
        let program = parse_with_orchestrator(&input.source, input.source_id)
            .map_err(|e| CompilerError::ParseError {
                message: e.to_string(),
                location: prism_common::span::Span::dummy(),
                suggestions: vec!["Check syntax for errors".to_string()],
            })?;

        // Step 3: Generate AI metadata
        let ai_metadata = ParsingAIMetadata {
            confidence: detection_result.confidence,
            syntax_patterns: detection_result.evidence.into_iter().map(|e| e.to_string()).collect(),
            quality_indicators: vec!["well_structured".to_string()],
        };

        Ok(ParseQueryResult {
            program,
            detected_syntax: format!("{:?}", detection_result.detected_style),
            diagnostics: detection_result.warnings,
            ai_metadata: Some(ai_metadata),
        })
    }

    fn cache_key(&self, input: &ParseInput) -> CacheKey {
        CacheKey::from_input("parse_source", input)
    }

    async fn dependencies(&self, _input: &ParseInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Parsing has no dependencies - it's the first step
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, input: &ParseInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        if let Some(file_path) = &input.file_path {
            triggers.insert(InvalidationTrigger::FileChanged(file_path.clone()));
        }
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "parse_source"
    }
}

/// Query for optimization and transformations
#[derive(Debug, Clone)]
pub struct OptimizationQuery {
    /// Transformation configuration
    config: Option<crate::context::TransformationConfig>,
}

/// Optimization input
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OptimizationInput {
    /// Program to optimize
    pub program: Program,
    /// Semantic information
    pub semantic_info: SemanticAnalysisResult,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimized program
    pub optimized_program: Program,
    /// Transformations applied
    pub transformations_applied: Vec<String>,
    /// Performance improvements
    pub performance_improvements: Vec<String>,
}

impl OptimizationQuery {
    /// Create a new optimization query
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Create with configuration
    pub fn with_config(config: crate::context::TransformationConfig) -> Self {
        Self { config: Some(config) }
    }
}

#[async_trait]
impl CompilerQuery<OptimizationInput, (Program, OptimizationResult)> for OptimizationQuery {
    async fn execute(&self, input: OptimizationInput, _context: QueryContext) -> CompilerResult<(Program, OptimizationResult)> {
        // For now, return the program unchanged with placeholder optimization info
        // In a full implementation, this would apply actual transformations
        
        let optimization_result = OptimizationResult {
            optimized_program: input.program.clone(),
            transformations_applied: vec!["placeholder_optimization".to_string()],
            performance_improvements: vec!["Maintained original performance".to_string()],
        };

        Ok((input.program, optimization_result))
    }

    fn cache_key(&self, input: &OptimizationInput) -> CacheKey {
        CacheKey::from_input("optimization", input)
            .with_target_config(&format!("opt_level_{}", 
                self.config.as_ref()
                    .map(|c| if c.enable_optimizations { "high" } else { "low" })
                    .unwrap_or("default")))
    }

    async fn dependencies(&self, _input: &OptimizationInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Optimization depends on semantic analysis
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &OptimizationInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::OptimizationLevelChanged);
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "optimization"
    }
}

/// Query for code generation
#[derive(Debug, Clone)]
pub struct CodeGenQuery {
    /// Target compilation target
    target: crate::context::CompilationTarget,
    /// Code generator reference
    codegen: Arc<prism_codegen::MultiTargetCodeGen>,
}

/// Code generation input (now uses PIR as stable interface)
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CodeGenInput {
    /// PIR to generate code from (stable semantic bridge)
    pub pir: prism_pir::semantic::PrismIR,
    /// Target platform for code generation
    pub target: crate::context::CompilationTarget,
    /// Optimization level to apply
    pub optimization_level: u8,
}

impl CodeGenQuery {
    /// Create a new code generation query
    pub fn new(target: crate::context::CompilationTarget, codegen: Arc<prism_codegen::MultiTargetCodeGen>) -> Self {
        Self { target, codegen }
    }
}

#[async_trait]
impl CompilerQuery<CodeGenInput, prism_codegen::CodeArtifact> for CodeGenQuery {
    async fn execute(&self, input: CodeGenInput, _context: QueryContext) -> CompilerResult<prism_codegen::CodeArtifact> {
        // For now, create a placeholder artifact
        // In a full implementation, this would use the actual code generator
        
        use prism_codegen::{CodeArtifact, CodeGenStats};
        use std::path::PathBuf;

        Ok(CodeArtifact {
            target: self.target,
            content: "// Generated code placeholder".to_string(),
            source_map: None,
            ai_metadata: None,
            output_path: PathBuf::from("generated.ts"), // Default for TypeScript
            stats: CodeGenStats {
                lines_generated: 1,
                generation_time: 0,
                optimizations_applied: 0,
                memory_usage: 100,
            },
        })
    }

    fn cache_key(&self, input: &CodeGenInput) -> CacheKey {
        CacheKey::from_input("code_generation", input)
            .with_target_config(&format!("{:?}", self.target))
    }

    async fn dependencies(&self, _input: &CodeGenInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Code generation depends on semantic analysis and optimization
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &CodeGenInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::OptimizationLevelChanged);
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "code_generation"
    }
} 

// ===== NEW COMPILATION PHASE QUERIES =====

/// Query for project discovery phase
#[derive(Debug, Clone)]
pub struct ProjectDiscoveryQuery {
    /// Project root path
    project_root: PathBuf,
}

/// Input for project discovery
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ProjectDiscoveryInput {
    /// Project root directory
    pub project_root: PathBuf,
    /// Include hidden files
    pub include_hidden: bool,
    /// File extensions to discover
    pub extensions: Vec<String>,
    /// Maximum directory depth
    pub max_depth: Option<u32>,
}

/// Result of project discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDiscoveryResult {
    /// Discovered source files
    pub source_files: Vec<PathBuf>,
    /// Project configuration files found
    pub config_files: Vec<PathBuf>,
    /// Module structure discovered
    pub module_structure: ModuleStructure,
    /// Discovery statistics
    pub stats: DiscoveryStats,
}

/// Module structure discovered during project discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleStructure {
    /// Root modules
    pub root_modules: Vec<String>,
    /// Module dependencies
    pub dependencies: std::collections::HashMap<String, Vec<String>>,
    /// Module hierarchy
    pub hierarchy: ModuleHierarchy,
}

/// Module hierarchy information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleHierarchy {
    /// Module tree structure
    pub tree: std::collections::HashMap<String, Vec<String>>,
    /// Depth of each module
    pub depths: std::collections::HashMap<String, u32>,
}

/// Discovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStats {
    /// Total files discovered
    pub total_files: usize,
    /// Total directories scanned
    pub directories_scanned: usize,
    /// Discovery time in milliseconds
    pub discovery_time_ms: u64,
}

impl ProjectDiscoveryQuery {
    /// Create a new project discovery query
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

#[async_trait]
impl CompilerQuery<ProjectDiscoveryInput, ProjectDiscoveryResult> for ProjectDiscoveryQuery {
    async fn execute(&self, input: ProjectDiscoveryInput, _context: QueryContext) -> CompilerResult<ProjectDiscoveryResult> {
        use walkdir::WalkDir;
        use std::time::Instant;
        
        let start_time = Instant::now();
        let mut source_files = Vec::new();
        let mut config_files = Vec::new();
        let mut directories_scanned = 0;

        // Default extensions if none provided
        let extensions = if input.extensions.is_empty() {
            vec!["prsm".to_string(), "prism".to_string()]
        } else {
            input.extensions
        };

        // Walk the directory tree
        let walker = if let Some(max_depth) = input.max_depth {
            WalkDir::new(&input.project_root).max_depth(max_depth as usize)
        } else {
            WalkDir::new(&input.project_root)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            
            if path.is_dir() {
                directories_scanned += 1;
                continue;
            }

            // Skip hidden files unless requested
            if !input.include_hidden {
                if let Some(file_name) = path.file_name() {
                    if file_name.to_string_lossy().starts_with('.') {
                        continue;
                    }
                }
            }

            // Check for source files
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if extensions.contains(&ext_str) {
                    source_files.push(path.to_path_buf());
                } else if ext_str == "toml" || ext_str == "json" {
                    config_files.push(path.to_path_buf());
                }
            }
        }

        // Analyze module structure (simplified for now)
        let module_structure = ModuleStructure {
            root_modules: vec!["main".to_string()], // Placeholder
            dependencies: std::collections::HashMap::new(),
            hierarchy: ModuleHierarchy {
                tree: std::collections::HashMap::new(),
                depths: std::collections::HashMap::new(),
            },
        };

        let stats = DiscoveryStats {
            total_files: source_files.len() + config_files.len(),
            directories_scanned,
            discovery_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(ProjectDiscoveryResult {
            source_files,
            config_files,
            module_structure,
            stats,
        })
    }

    fn cache_key(&self, input: &ProjectDiscoveryInput) -> CacheKey {
        CacheKey::from_input("project_discovery", input)
    }

    async fn dependencies(&self, _input: &ProjectDiscoveryInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Discovery has no dependencies - it's the first step
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, input: &ProjectDiscoveryInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        // Invalidate when project structure changes
        triggers.insert(InvalidationTrigger::FileChanged(input.project_root.clone()));
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "project_discovery"
    }
}

/// Query for lexical analysis phase
#[derive(Debug, Clone)]
pub struct LexicalAnalysisQuery {
    /// Source ID for tracking
    source_id: SourceId,
}

/// Input for lexical analysis
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LexicalAnalysisInput {
    /// Source code to tokenize
    pub source: String,
    /// Source file path
    pub file_path: PathBuf,
    /// Source ID
    pub source_id: SourceId,
}

/// Result of lexical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalAnalysisResult {
    /// Generated tokens
    pub tokens: Vec<TokenInfo>,
    /// Lexical errors found
    pub errors: Vec<String>,
    /// Lexical warnings
    pub warnings: Vec<String>,
    /// Lexing statistics
    pub stats: LexingStats,
}

/// Token information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// Token type
    pub token_type: String,
    /// Token value
    pub value: String,
    /// Location in source
    pub span: prism_common::span::Span,
}

/// Lexing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexingStats {
    /// Total tokens generated
    pub total_tokens: usize,
    /// Lexing time in milliseconds
    pub lexing_time_ms: u64,
    /// Lines processed
    pub lines_processed: usize,
}

impl LexicalAnalysisQuery {
    /// Create a new lexical analysis query
    pub fn new(source_id: SourceId) -> Self {
        Self { source_id }
    }
}

#[async_trait]
impl CompilerQuery<LexicalAnalysisInput, LexicalAnalysisResult> for LexicalAnalysisQuery {
    async fn execute(&self, input: LexicalAnalysisInput, _context: QueryContext) -> CompilerResult<LexicalAnalysisResult> {
        use std::time::Instant;
        
        let start_time = Instant::now();
        
        // For now, create a placeholder implementation
        // In a real implementation, this would use prism-lexer
        let tokens = vec![
            TokenInfo {
                token_type: "MODULE".to_string(),
                value: "module".to_string(),
                span: prism_common::span::Span::dummy(),
            }
        ];

        let stats = LexingStats {
            total_tokens: tokens.len(),
            lexing_time_ms: start_time.elapsed().as_millis() as u64,
            lines_processed: input.source.lines().count(),
        };

        Ok(LexicalAnalysisResult {
            tokens,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats,
        })
    }

    fn cache_key(&self, input: &LexicalAnalysisInput) -> CacheKey {
        CacheKey::from_input("lexical_analysis", input)
    }

    async fn dependencies(&self, _input: &LexicalAnalysisInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Lexical analysis has no dependencies
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, input: &LexicalAnalysisInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::FileChanged(input.file_path.clone()));
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "lexical_analysis"
    }
}

/// Query for type checking phase
#[derive(Debug, Clone)]
pub struct TypeCheckingQuery {
    /// Reference to the semantic database
    semantic_db: Arc<SemanticDatabase>,
}

/// Input for type checking
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypeCheckingInput {
    /// Program to type check
    pub program: Program,
    /// Semantic analysis result
    pub semantic_info: SemanticAnalysisResult,
}

/// Result of type checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCheckingResult {
    /// Type checking success
    pub success: bool,
    /// Type information extracted
    pub type_info: TypeCheckingInfo,
    /// Type errors found
    pub errors: Vec<String>,
    /// Type warnings
    pub warnings: Vec<String>,
    /// Type checking statistics
    pub stats: TypeCheckingStats,
}

/// Type checking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCheckingInfo {
    /// Types inferred
    pub inferred_types: std::collections::HashMap<NodeId, String>,
    /// Type constraints
    pub constraints: Vec<TypeConstraint>,
    /// Generic instantiations
    pub generic_instantiations: Vec<GenericInstantiation>,
}

/// Type constraint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// Constraint description
    pub description: String,
    /// Location where constraint applies
    pub location: NodeId,
}

/// Generic instantiation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericInstantiation {
    /// Generic type name
    pub generic_name: String,
    /// Concrete type
    pub concrete_type: String,
    /// Instantiation location
    pub location: NodeId,
}

/// Type checking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCheckingStats {
    /// Types checked
    pub types_checked: usize,
    /// Constraints solved
    pub constraints_solved: usize,
    /// Type checking time in milliseconds
    pub type_checking_time_ms: u64,
}

impl TypeCheckingQuery {
    /// Create a new type checking query
    pub fn new(semantic_db: Arc<SemanticDatabase>) -> Self {
        Self { semantic_db }
    }
}

#[async_trait]
impl CompilerQuery<TypeCheckingInput, TypeCheckingResult> for TypeCheckingQuery {
    async fn execute(&self, input: TypeCheckingInput, _context: QueryContext) -> CompilerResult<TypeCheckingResult> {
        use std::time::Instant;
        
        let start_time = Instant::now();
        
        // Placeholder implementation - would integrate with prism-semantic
        let type_info = TypeCheckingInfo {
            inferred_types: std::collections::HashMap::new(),
            constraints: Vec::new(),
            generic_instantiations: Vec::new(),
        };

        let stats = TypeCheckingStats {
            types_checked: 0,
            constraints_solved: 0,
            type_checking_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(TypeCheckingResult {
            success: true,
            type_info,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats,
        })
    }

    fn cache_key(&self, input: &TypeCheckingInput) -> CacheKey {
        CacheKey::from_input("type_checking", input)
    }

    async fn dependencies(&self, _input: &TypeCheckingInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Type checking depends on semantic analysis
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &TypeCheckingInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::SemanticContextChanged(NodeId(0)));
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "type_checking"
    }
}

/// Query for effect analysis phase
#[derive(Debug, Clone)]
pub struct EffectAnalysisQuery {
    /// Reference to the semantic database
    semantic_db: Arc<SemanticDatabase>,
}

/// Input for effect analysis
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EffectAnalysisInput {
    /// Program to analyze
    pub program: Program,
    /// Type checking result
    pub type_info: TypeCheckingResult,
}

/// Result of effect analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectAnalysisResult {
    /// Effect analysis success
    pub success: bool,
    /// Effect information extracted
    pub effect_info: EffectAnalysisInfo,
    /// Effect errors found
    pub errors: Vec<String>,
    /// Effect warnings
    pub warnings: Vec<String>,
    /// Effect analysis statistics
    pub stats: EffectAnalysisStats,
}

/// Effect analysis information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectAnalysisInfo {
    /// Effects identified
    pub effects: Vec<EffectInfo>,
    /// Capability requirements
    pub capabilities: Vec<CapabilityInfo>,
    /// Effect composition
    pub compositions: Vec<EffectComposition>,
}

/// Capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityInfo {
    /// Capability name
    pub name: String,
    /// Required by functions
    pub required_by: Vec<String>,
    /// Security level
    pub security_level: String,
}

/// Effect composition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectComposition {
    /// Composed effects
    pub effects: Vec<String>,
    /// Result effect
    pub result: String,
}

/// Effect analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectAnalysisStats {
    /// Effects analyzed
    pub effects_analyzed: usize,
    /// Capabilities checked
    pub capabilities_checked: usize,
    /// Effect analysis time in milliseconds
    pub effect_analysis_time_ms: u64,
}

impl EffectAnalysisQuery {
    /// Create a new effect analysis query
    pub fn new(semantic_db: Arc<SemanticDatabase>) -> Self {
        Self { semantic_db }
    }
}

#[async_trait]
impl CompilerQuery<EffectAnalysisInput, EffectAnalysisResult> for EffectAnalysisQuery {
    async fn execute(&self, input: EffectAnalysisInput, _context: QueryContext) -> CompilerResult<EffectAnalysisResult> {
        use std::time::Instant;
        
        let start_time = Instant::now();
        
        // Placeholder implementation - would integrate with prism-effects
        let effect_info = EffectAnalysisInfo {
            effects: Vec::new(),
            capabilities: Vec::new(),
            compositions: Vec::new(),
        };

        let stats = EffectAnalysisStats {
            effects_analyzed: 0,
            capabilities_checked: 0,
            effect_analysis_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(EffectAnalysisResult {
            success: true,
            effect_info,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats,
        })
    }

    fn cache_key(&self, input: &EffectAnalysisInput) -> CacheKey {
        CacheKey::from_input("effect_analysis", input)
    }

    async fn dependencies(&self, _input: &EffectAnalysisInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Effect analysis depends on type checking
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &EffectAnalysisInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::SemanticContextChanged(NodeId(0)));
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "effect_analysis"
    }
}

/// Query for linking phase
#[derive(Debug, Clone)]
pub struct LinkingQuery {
    /// Target for linking
    target: CompilationTarget,
}

/// Input for linking
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LinkingInput {
    /// Code artifacts to link
    pub artifacts: Vec<prism_codegen::CodeArtifact>,
    /// Target platform
    pub target: CompilationTarget,
    /// Linking configuration
    pub config: LinkingConfig,
}

/// Linking configuration
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LinkingConfig {
    /// Output file name
    pub output_name: String,
    /// Enable optimization
    pub optimize: bool,
    /// Additional link flags
    pub link_flags: Vec<String>,
}

/// Result of linking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingResult {
    /// Linking success
    pub success: bool,
    /// Output file path
    pub output_path: PathBuf,
    /// Linking statistics
    pub stats: LinkingStats,
    /// Linking diagnostics
    pub diagnostics: Vec<String>,
}

/// Linking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingStats {
    /// Artifacts linked
    pub artifacts_linked: usize,
    /// Output size in bytes
    pub output_size_bytes: u64,
    /// Linking time in milliseconds
    pub linking_time_ms: u64,
}

impl LinkingQuery {
    /// Create a new linking query
    pub fn new(target: CompilationTarget) -> Self {
        Self { target }
    }
}

#[async_trait]
impl CompilerQuery<LinkingInput, LinkingResult> for LinkingQuery {
    async fn execute(&self, input: LinkingInput, _context: QueryContext) -> CompilerResult<LinkingResult> {
        use std::time::Instant;
        
        let start_time = Instant::now();
        
        // Placeholder implementation - would integrate with target-specific linkers
        let output_path = PathBuf::from(format!("{}.out", input.config.output_name));
        
        let stats = LinkingStats {
            artifacts_linked: input.artifacts.len(),
            output_size_bytes: 1024, // Placeholder
            linking_time_ms: start_time.elapsed().as_millis() as u64,
        };

        Ok(LinkingResult {
            success: true,
            output_path,
            stats,
            diagnostics: Vec::new(),
        })
    }

    fn cache_key(&self, input: &LinkingInput) -> CacheKey {
        CacheKey::from_input("linking", input)
            .with_target_config(&format!("{:?}", self.target))
    }

    async fn dependencies(&self, _input: &LinkingInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Linking depends on code generation
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &LinkingInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers.insert(InvalidationTrigger::OptimizationLevelChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "linking"
    }
}

/// Query for finalization phase
#[derive(Debug, Clone)]
pub struct FinalizationQuery;

/// Input for finalization
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FinalizationInput {
    /// Linked outputs
    pub linked_outputs: Vec<LinkingResult>,
    /// AI metadata to export
    pub ai_metadata: Option<crate::context::AIMetadataExport>,
    /// Export configuration
    pub export_config: FinalizationConfig,
}

/// Finalization configuration
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FinalizationConfig {
    /// Export AI metadata
    pub export_ai_metadata: bool,
    /// Generate documentation
    pub generate_docs: bool,
    /// Create build report
    pub create_build_report: bool,
}

/// Result of finalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizationResult {
    /// Finalization success
    pub success: bool,
    /// Final outputs
    pub outputs: Vec<PathBuf>,
    /// Build report path
    pub build_report: Option<PathBuf>,
    /// AI metadata export path
    pub ai_metadata_path: Option<PathBuf>,
    /// Finalization statistics
    pub stats: FinalizationStats,
}

/// Finalization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizationStats {
    /// Total compilation time
    pub total_compilation_time_ms: u64,
    /// Final output size
    pub total_output_size_bytes: u64,
    /// Files generated
    pub files_generated: usize,
}

impl FinalizationQuery {
    /// Create a new finalization query
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CompilerQuery<FinalizationInput, FinalizationResult> for FinalizationQuery {
    async fn execute(&self, input: FinalizationInput, _context: QueryContext) -> CompilerResult<FinalizationResult> {
        let mut outputs = Vec::new();
        let mut total_size = 0u64;
        
        // Collect all output paths
        for linking_result in &input.linked_outputs {
            outputs.push(linking_result.output_path.clone());
            total_size += linking_result.stats.output_size_bytes;
        }

        // Export AI metadata if requested
        let ai_metadata_path = if input.export_config.export_ai_metadata && input.ai_metadata.is_some() {
            let path = PathBuf::from("ai_metadata.json");
            // Would write AI metadata to file here
            Some(path)
        } else {
            None
        };

        // Generate build report if requested
        let build_report = if input.export_config.create_build_report {
            let path = PathBuf::from("build_report.json");
            // Would generate comprehensive build report here
            Some(path)
        } else {
            None
        };

        let stats = FinalizationStats {
            total_compilation_time_ms: 0, // Would be calculated from context
            total_output_size_bytes: total_size,
            files_generated: outputs.len(),
        };

        Ok(FinalizationResult {
            success: true,
            outputs,
            build_report,
            ai_metadata_path,
            stats,
        })
    }

    fn cache_key(&self, input: &FinalizationInput) -> CacheKey {
        CacheKey::from_input("finalization", input)
    }

    async fn dependencies(&self, _input: &FinalizationInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Finalization depends on linking
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &FinalizationInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "finalization"
    }
} 

// ===== PIR GENERATION QUERY =====

/// Query for PIR generation phase
#[derive(Debug, Clone)]
pub struct PIRGenerationQuery {
    /// PIR construction builder
    pir_builder: Arc<prism_pir::construction::PIRConstructionBuilder>,
}

/// Input for PIR generation
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PIRGenerationInput {
    /// Program to generate PIR from
    pub program: Program,
    /// Semantic analysis results
    pub semantic_info: SemanticAnalysisResult,
    /// Type checking results
    pub type_info: TypeCheckingResult,
    /// Effect analysis results
    pub effect_info: EffectAnalysisResult,
}

/// Result of PIR generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRGenerationResult {
    /// Generated PIR
    pub pir: prism_pir::semantic::PrismIR,
    /// PIR generation success
    pub success: bool,
    /// PIR generation diagnostics
    pub diagnostics: Vec<String>,
    /// PIR generation statistics
    pub stats: PIRGenerationStats,
    /// PIR validation results
    pub validation_results: Vec<String>,
}

/// PIR generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRGenerationStats {
    /// Number of modules generated
    pub modules_generated: usize,
    /// Number of functions generated
    pub functions_generated: usize,
    /// Number of types preserved
    pub types_preserved: usize,
    /// PIR generation time in milliseconds
    pub generation_time_ms: u64,
    /// Semantic preservation score (0.0 to 1.0)
    pub semantic_preservation_score: f64,
}

impl PIRGenerationQuery {
    /// Create a new PIR generation query
    pub fn new() -> CompilerResult<Self> {
        let pir_builder = Arc::new(
            prism_pir::construction::PIRConstructionBuilder::new()
                .map_err(|e| CompilerError::InvalidOperation { 
                    message: format!("Failed to create PIR builder: {}", e) 
                })?
        );
        
        Ok(Self { pir_builder })
    }
    
    /// Create a PIR generation query with custom builder
    pub fn with_builder(pir_builder: Arc<prism_pir::construction::PIRConstructionBuilder>) -> Self {
        Self { pir_builder }
    }
}

#[async_trait]
impl CompilerQuery<PIRGenerationInput, PIRGenerationResult> for PIRGenerationQuery {
    async fn execute(&self, input: PIRGenerationInput, _context: QueryContext) -> CompilerResult<PIRGenerationResult> {
        use tracing::{info, debug, warn};
        use std::time::Instant;
        
        let start_time = Instant::now();
        info!("Executing PIR generation from semantic analysis results");
        
        let mut diagnostics = Vec::new();
        let mut success = true;
        
        // Generate PIR from the program using the construction builder
        let pir_result = {
            // Create a mutable builder for this generation
            let mut builder = prism_pir::construction::PIRConstructionBuilder::new()
                .map_err(|e| CompilerError::InvalidOperation { 
                    message: format!("Failed to create PIR builder: {}", e) 
                })?;
            
            // Build PIR from the program
            builder.build_from_program(&input.program)
                .map_err(|e| CompilerError::InvalidOperation { 
                    message: format!("PIR generation failed: {}", e) 
                })
        };
        
        let pir = match pir_result {
            Ok(pir) => pir,
            Err(e) => {
                success = false;
                diagnostics.push(e.to_string());
                
                // Return a minimal PIR for error recovery
                prism_pir::semantic::PrismIR::new()
            }
        };
        
        // Calculate statistics
        let modules_generated = pir.modules.len();
        let functions_generated = pir.modules.iter()
            .map(|module| module.sections.iter()
                .filter_map(|section| match section {
                    prism_pir::semantic::PIRSection::Functions(funcs) => Some(funcs.functions.len()),
                    _ => None,
                }).sum::<usize>())
            .sum();
        let types_preserved = pir.type_registry.types.len();
        
        let generation_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Calculate semantic preservation score based on successful transformations
        let semantic_preservation_score = if success {
            // Base score on successful generation and presence of semantic information
            let base_score = 0.8;
            let semantic_bonus = if input.semantic_info.success { 0.1 } else { 0.0 };
            let type_bonus = if input.type_info.success { 0.1 } else { 0.0 };
            (base_score + semantic_bonus + type_bonus).min(1.0)
        } else {
            0.0
        };
        
        let stats = PIRGenerationStats {
            modules_generated,
            functions_generated,
            types_preserved,
            generation_time_ms,
            semantic_preservation_score,
        };
        
        // Perform basic validation
        let validation_results = if success {
            self.validate_pir_quality(&pir, &input)
        } else {
            vec!["PIR generation failed - skipping validation".to_string()]
        };
        
        debug!(
            "PIR generation completed: {} modules, {} functions, {} types preserved",
            modules_generated, functions_generated, types_preserved
        );
        
        Ok(PIRGenerationResult {
            pir,
            success,
            diagnostics,
            stats,
            validation_results,
        })
    }

    fn cache_key(&self, input: &PIRGenerationInput) -> CacheKey {
        CacheKey::from_input("pir_generation", input)
    }

    async fn dependencies(&self, _input: &PIRGenerationInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // PIR generation depends on semantic analysis, type checking, and effect analysis
        let mut deps = HashSet::new();
        // In a full implementation, these would be proper QueryIds
        // deps.insert(QueryId::semantic_analysis());
        // deps.insert(QueryId::type_checking());
        // deps.insert(QueryId::effect_analysis());
        Ok(deps)
    }

    fn invalidate_on(&self, input: &PIRGenerationInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        // Invalidate when semantic analysis results change
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "pir_generation"
    }
}

impl PIRGenerationQuery {
    /// Validate PIR quality and semantic preservation
    fn validate_pir_quality(&self, pir: &prism_pir::semantic::PrismIR, input: &PIRGenerationInput) -> Vec<String> {
        let mut validation_results = Vec::new();
        
        // Check if all program modules were preserved
        let program_modules = input.program.items.iter()
            .filter(|item| matches!(item.kind, prism_ast::Item::Module(_)))
            .count();
        
        if pir.modules.len() < program_modules {
            validation_results.push(format!(
                "Module preservation issue: {} modules in program, {} in PIR",
                program_modules, pir.modules.len()
            ));
        }
        
        // Check semantic type preservation
        if pir.type_registry.types.is_empty() && !input.semantic_info.semantic_metadata.types.is_empty() {
            validation_results.push("Type information may not have been fully preserved in PIR".to_string());
        }
        
        // Check effect preservation
        if pir.effect_graph.effects.is_empty() && !input.effect_info.effects.is_empty() {
            validation_results.push("Effect information may not have been fully preserved in PIR".to_string());
        }
        
        // Validate business context preservation
        for module in &pir.modules {
            if module.business_context.capability.is_empty() {
                validation_results.push(format!(
                    "Module '{}' missing business context information",
                    module.name
                ));
            }
        }
        
        if validation_results.is_empty() {
            validation_results.push("PIR quality validation passed".to_string());
        }
        
        validation_results
    }
}

// ===== RESULT TYPES =====

/// Result of parsing phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    /// The parsed program
    pub program: Program,
    /// Detected syntax style
    pub syntax_style: String,
    /// Parsing success
    pub success: bool,
    /// Parse errors
    pub errors: Vec<String>,
    /// Parse warnings
    pub warnings: Vec<String>,
    /// Parsing statistics
    pub stats: ParseStats,
}

/// Parsing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseStats {
    /// Number of AST nodes created
    pub nodes_created: usize,
    /// Parsing time in milliseconds
    pub parse_time_ms: u64,
    /// Lines parsed
    pub lines_parsed: usize,
    /// Syntax detection confidence
    pub syntax_confidence: f64,
}

/// Result of type checking phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCheckingResult {
    /// Type checking success
    pub success: bool,
    /// Type information discovered
    pub type_info: Vec<TypeInfo>,
    /// Type errors found
    pub type_errors: Vec<String>,
    /// Type warnings
    pub type_warnings: Vec<String>,
    /// Type checking statistics
    pub stats: TypeCheckingStats,
}

/// Type checking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCheckingStats {
    /// Types resolved
    pub types_resolved: usize,
    /// Type constraints checked
    pub constraints_checked: usize,
    /// Type checking time in milliseconds
    pub type_check_time_ms: u64,
}

/// Result of effect analysis phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectAnalysisResult {
    /// Effect analysis success
    pub success: bool,
    /// Effects discovered
    pub effects: Vec<EffectInfo>,
    /// Capability requirements
    pub capabilities: Vec<String>,
    /// Effect violations found
    pub violations: Vec<String>,
    /// Effect analysis statistics
    pub stats: EffectAnalysisStats,
}

/// Effect analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectAnalysisStats {
    /// Effects analyzed
    pub effects_analyzed: usize,
    /// Capabilities validated
    pub capabilities_validated: usize,
    /// Effect analysis time in milliseconds
    pub effect_analysis_time_ms: u64,
}

/// Result of optimization phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimization success
    pub success: bool,
    /// Optimizations applied
    pub optimizations: Vec<OptimizationInfo>,
    /// Performance improvements
    pub improvements: Vec<String>,
    /// Optimization statistics
    pub stats: OptimizationStats,
}

/// Optimization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationInfo {
    /// Optimization type
    pub optimization_type: String,
    /// Description of what was optimized
    pub description: String,
    /// Performance impact
    pub impact: String,
}

/// Optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Optimization time in milliseconds
    pub optimization_time_ms: u64,
    /// Estimated performance improvement
    pub performance_improvement_percent: f64,
}

/// Result of linking phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingResult {
    /// Linking success
    pub success: bool,
    /// Linked artifacts
    pub linked_artifacts: Vec<LinkedArtifact>,
    /// Linking errors
    pub errors: Vec<String>,
    /// Linking statistics
    pub stats: LinkingStats,
}

/// Linked artifact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedArtifact {
    /// Artifact name
    pub name: String,
    /// Target platform
    pub target: String,
    /// Output path
    pub output_path: PathBuf,
    /// Size in bytes
    pub size_bytes: u64,
}

/// Linking statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingStats {
    /// Artifacts linked
    pub artifacts_linked: usize,
    /// Total output size in bytes
    pub total_size_bytes: u64,
    /// Linking time in milliseconds
    pub linking_time_ms: u64,
} 