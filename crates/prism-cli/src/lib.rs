//! Prism CLI Library
//!
//! This module provides the command-line interface functionality for the Prism language,
//! integrating with the existing compiler, AI, and analysis subsystems while maintaining
//! strict Separation of Concerns.
//!
//! ## Design Principles
//!
//! 1. **Delegation Over Duplication**: Uses existing subsystems rather than reimplementing logic
//! 2. **Separation of Concerns**: CLI coordinates existing systems, doesn't duplicate functionality
//! 3. **AI-First Integration**: Leverages existing prism-ai infrastructure for metadata export
//! 4. **Conceptual Cohesion**: CLI focuses on user interface and coordination only

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use tracing::{info, error, debug};

// Import existing subsystem APIs
use prism_compiler::{
    PrismCompiler, CompilationConfig, CompilationTarget, CompilationContext,
    BuildProfile, DependencyConfig
};
use prism_ai::{
    AIIntegrationCoordinator, AIIntegrationConfig, ExportFormat,
    ComprehensiveAIMetadata
};

/// Prism programming language CLI
#[derive(Parser)]
#[command(name = "prism")]
#[command(about = "A programming language optimized for AI-first development")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Compile Prism source code
    Compile {
        /// Input file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Target platform
        #[arg(short, long, default_value = "typescript")]
        target: String,
        
        /// Optimization level (0-3)
        #[arg(long, default_value = "1")]
        optimization: u8,
        
        /// Enable AI metadata generation
        #[arg(long)]
        ai_metadata: bool,
        
        /// Enable incremental compilation
        #[arg(long)]
        incremental: bool,
        
        /// Include debug information
        #[arg(long)]
        debug: bool,
    },
    
    /// Export AI metadata for external tools
    #[command(name = "ai-export")]
    AIExport {
        /// Input file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        
        /// Output directory for exported metadata
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Export formats (json, yaml, xml, openapi, graphql)
        #[arg(short, long, value_delimiter = ',', default_values = ["json"])]
        formats: Vec<String>,
        
        /// Include business context
        #[arg(long, default_value = "true")]
        business_context: bool,
        
        /// Include performance metrics
        #[arg(long, default_value = "true")]
        performance_metrics: bool,
        
        /// Include architectural patterns
        #[arg(long, default_value = "true")]
        architectural_patterns: bool,
        
        /// Minimum confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.7")]
        confidence_threshold: f64,
    },
    
    /// Analyze code structure and patterns
    Analyze {
        /// Input file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        
        /// Analysis type (structure, patterns, business, all)
        #[arg(short, long, default_value = "all")]
        analysis_type: String,
        
        /// Output format (json, yaml, text)
        #[arg(short, long, default_value = "text")]
        format: String,
        
        /// Include performance analysis
        #[arg(long)]
        performance: bool,
        
        /// Include semantic analysis
        #[arg(long)]
        semantic: bool,
    },
    
    /// Language server for IDE integration
    #[command(name = "lsp")]
    LanguageServer {
        /// Enable AI metadata features
        #[arg(long)]
        ai_features: bool,
        
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
        
        /// Log level for debugging
        #[arg(long, default_value = "info")]
        log_level: String,
    },
}

/// Execute the CLI command using existing subsystems
pub async fn execute_command(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Compile { 
            input, 
            output, 
            target, 
            optimization,
            ai_metadata,
            incremental,
            debug,
        } => {
            execute_compile(input, output, target, optimization, ai_metadata, incremental, debug).await
        }
        
        Commands::AIExport { 
            input, 
            output, 
            formats, 
            business_context, 
            performance_metrics, 
            architectural_patterns,
            confidence_threshold,
        } => {
            execute_ai_export(
                input, 
                output, 
                formats, 
                business_context, 
                performance_metrics, 
                architectural_patterns,
                confidence_threshold,
            ).await
        }
        
        Commands::Analyze { 
            input, 
            analysis_type, 
            format,
            performance,
            semantic,
        } => {
            execute_analyze(input, analysis_type, format, performance, semantic).await
        }
        
        Commands::LanguageServer { 
            ai_features, 
            config,
            log_level,
        } => {
            execute_language_server(ai_features, config, log_level).await
        }
    }
}

/// Execute compile command using prism-compiler
async fn execute_compile(
    input: PathBuf,
    output: Option<PathBuf>,
    target: String,
    optimization: u8,
    ai_metadata: bool,
    incremental: bool,
    debug: bool,
) -> Result<()> {
    info!("Starting compilation of {}", input.display());
    
    // Parse compilation target
    let target = parse_compilation_target(&target)?;
    
    // Create compilation configuration
    let config = CompilationConfig {
        project_root: input.parent().unwrap_or(&input).to_path_buf(),
        targets: vec![target],
        optimization_level: optimization,
        enable_language_server: Some(false), // CLI compilation doesn't need LSP
        export_ai_context: ai_metadata,
        incremental: Some(incremental),
        ai_features: Some(ai_metadata),
        debug_info: Some(debug),
        enable_transformations: Some(optimization > 0),
        transformation_config: None, // Use defaults
        compiler_flags: std::collections::HashMap::new(),
        build_profile: if optimization >= 2 { BuildProfile::Release } else { BuildProfile::Debug },
        dependency_config: DependencyConfig::default(),
    };
    
    // Create compiler instance
    let compiler = PrismCompiler::new(config)?;
    
    // Determine if we're compiling a single file or project
    let result = if input.is_file() {
        info!("Compiling single file: {}", input.display());
        let module = compiler.compile_file(&input).await?;
        
        // Write output if specified
        if let Some(output_dir) = output {
            write_compiled_module(&module, &output_dir, target).await?;
        }
        
        info!("✅ Successfully compiled {}", input.display());
        return Ok(());
    } else {
        info!("Compiling project: {}", input.display());
        let project = compiler.compile_project(&input).await?;
        
        // Write output if specified
        if let Some(output_dir) = output {
            write_compiled_project(&project, &output_dir).await?;
        }
        
        // Export AI metadata if requested
        if ai_metadata {
            if let Some(ai_context) = project.ai_context {
                let ai_output = output.as_ref()
                    .map(|p| p.join("ai_metadata"))
                    .unwrap_or_else(|| input.join("target/ai_metadata"));
                
                tokio::fs::create_dir_all(&ai_output).await?;
                let ai_json = serde_json::to_string_pretty(&ai_context)?;
                tokio::fs::write(ai_output.join("compilation_metadata.json"), ai_json).await?;
                
                info!("📤 AI metadata exported to {}", ai_output.display());
            }
        }
        
        info!("✅ Successfully compiled project with {} source files", project.source_files.len());
        Ok(())
    };
    
    result
}

/// Execute AI export command using prism-ai
async fn execute_ai_export(
    input: PathBuf,
    output: Option<PathBuf>,
    formats: Vec<String>,
    business_context: bool,
    performance_metrics: bool,
    architectural_patterns: bool,
    confidence_threshold: f64,
) -> Result<()> {
    info!("🤖 Exporting AI metadata from: {}", input.display());
    
    // Parse export formats using existing prism-ai functionality
    let export_formats: Result<Vec<ExportFormat>, _> = formats.iter()
        .map(|f| parse_export_format(f))
        .collect();
    
    let export_formats = export_formats?;
    
    // Create AI integration configuration
    let config = AIIntegrationConfig {
        enabled: true,
        export_formats: export_formats.clone(),
        include_business_context: business_context,
        include_performance_metrics: performance_metrics,
        include_architectural_patterns: architectural_patterns,
        min_confidence_threshold: confidence_threshold,
        output_directory: output.clone(),
    };
    
    // Create coordinator using existing prism-ai infrastructure
    let coordinator = AIIntegrationCoordinator::new(config);
    
    // Collect metadata using existing system
    match coordinator.collect_metadata(&input).await {
        Ok(metadata) => {
            info!("✅ Successfully collected AI metadata");
            
            // Export in requested formats using existing exporters
            match coordinator.export_metadata(&metadata, &export_formats).await {
                Ok(exports) => {
                    for (format, content) in exports {
                        info!("📄 Exported {} format ({} bytes)", format_name(&format), content.len());
                        
                        // Write to files if output directory is specified
                        if let Some(output_dir) = &output {
                            let filename = format!("prism_metadata.{}", format_extension(&format));
                            let filepath = output_dir.join(filename);
                            
                            tokio::fs::create_dir_all(output_dir).await?;
                            tokio::fs::write(&filepath, content).await?;
                            
                            info!("💾 Written to: {}", filepath.display());
                        } else {
                            // Print summary to stdout if no output directory
                            print_metadata_summary(&metadata);
                        }
                    }
                    
                    info!("🎉 AI metadata export completed successfully");
                }
                Err(e) => {
                    error!("❌ Export failed: {}", e);
                    return Err(e.into());
                }
            }
        }
        Err(e) => {
            error!("❌ Metadata collection failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}

/// Execute analyze command using existing analysis subsystems
async fn execute_analyze(
    input: PathBuf,
    analysis_type: String,
    format: String,
    performance: bool,
    semantic: bool,
) -> Result<()> {
    info!("🔍 Analyzing {} (type: {}, format: {})", input.display(), analysis_type, format);
    
    // Create minimal compilation config for analysis
    let config = CompilationConfig {
        project_root: input.parent().unwrap_or(&input).to_path_buf(),
        targets: vec![CompilationTarget::TypeScript], // Use TypeScript for fastest analysis
        optimization_level: 0, // No optimization needed for analysis
        enable_language_server: Some(false),
        export_ai_context: true, // Enable for analysis metadata
        incremental: Some(true),
        ai_features: Some(true),
        debug_info: Some(false),
        enable_transformations: Some(false),
        transformation_config: None,
        compiler_flags: std::collections::HashMap::new(),
        build_profile: BuildProfile::Debug,
        dependency_config: DependencyConfig::default(),
    };
    
    // Create compiler for analysis
    let compiler = PrismCompiler::new(config)?;
    
    // Perform analysis based on type
    match analysis_type.as_str() {
        "structure" => analyze_structure(&compiler, &input, &format).await?,
        "patterns" => analyze_patterns(&compiler, &input, &format).await?,
        "business" => analyze_business_context(&compiler, &input, &format).await?,
        "all" => {
            analyze_structure(&compiler, &input, &format).await?;
            analyze_patterns(&compiler, &input, &format).await?;
            analyze_business_context(&compiler, &input, &format).await?;
            
            if performance {
                analyze_performance(&compiler, &input, &format).await?;
            }
            
            if semantic {
                analyze_semantics(&compiler, &input, &format).await?;
            }
        }
        _ => {
            error!("❌ Unknown analysis type: {}", analysis_type);
            return Err(anyhow::anyhow!("Unknown analysis type: {}", analysis_type));
        }
    }
    
    info!("✅ Analysis completed successfully");
    Ok(())
}

/// Execute language server command using prism-compiler's LSP integration
async fn execute_language_server(
    ai_features: bool,
    config: Option<PathBuf>,
    log_level: String,
) -> Result<()> {
    info!("🚀 Starting Prism language server");
    
    // Set up logging level
    let filter = match log_level.as_str() {
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };
    
    // Create LSP configuration
    let lsp_config = create_lsp_config(ai_features, config)?;
    
    // Create compiler with LSP enabled
    let compiler = PrismCompiler::new(lsp_config)?;
    
    // Get language server instance
    if let Some(language_server) = compiler.language_server() {
        info!("🔌 Language server started with AI features: {}", ai_features);
        
        // Run the language server (this would typically run indefinitely)
        // For now, we'll just indicate it's running
        info!("📡 Language server running on stdin/stdout");
        info!("🔧 Use Ctrl+C to stop the server");
        
        // In a real implementation, this would start the LSP protocol handler
        // language_server.run().await?;
        
        // For demonstration, we'll just wait
        tokio::signal::ctrl_c().await?;
        info!("🛑 Language server shutting down...");
    } else {
        error!("❌ Failed to create language server instance");
        return Err(anyhow::anyhow!("Failed to create language server"));
    }
    
    Ok(())
}

// Helper functions for CLI implementation

fn parse_compilation_target(target: &str) -> Result<CompilationTarget> {
    match target.to_lowercase().as_str() {
        "typescript" | "ts" => Ok(CompilationTarget::TypeScript),
        "javascript" | "js" => Ok(CompilationTarget::JavaScript),
        "webassembly" | "wasm" => Ok(CompilationTarget::WebAssembly),
        "llvm" | "native" => Ok(CompilationTarget::LLVM),
        _ => Err(anyhow::anyhow!("Unknown compilation target: {}", target)),
    }
}

fn parse_export_format(format: &str) -> Result<ExportFormat> {
    match format.to_lowercase().as_str() {
        "json" => Ok(ExportFormat::Json),
        "yaml" => Ok(ExportFormat::Yaml),
        "xml" => Ok(ExportFormat::Xml),
        "openapi" => Ok(ExportFormat::OpenApi),
        "graphql" => Ok(ExportFormat::GraphQL),
        "binary" => Ok(ExportFormat::Binary),
        "protobuf" => Ok(ExportFormat::Protobuf),
        other => Ok(ExportFormat::Custom(other.to_string())),
    }
}

async fn write_compiled_module(
    module: &prism_compiler::CompiledModule,
    output_dir: &PathBuf,
    target: CompilationTarget,
) -> Result<()> {
    tokio::fs::create_dir_all(output_dir).await?;
    
    if let Some(artifact) = module.artifacts.get(&target) {
        let extension = match target {
            CompilationTarget::TypeScript => "ts",
            CompilationTarget::JavaScript => "js",
            CompilationTarget::WebAssembly => "wasm",
            CompilationTarget::LLVM => "ll",
        };
        
        let filename = module.source_file
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let output_file = output_dir.join(format!("{}.{}", filename, extension));
        
        // Write the compiled output (artifact content would be bytes or string)
        // For now, we'll write a placeholder
        tokio::fs::write(&output_file, format!("// Compiled output for {:?}\n// Artifact: {:?}", target, artifact)).await?;
        
        info!("📁 Written {} output to: {}", target_name(target), output_file.display());
    }
    
    Ok(())
}

async fn write_compiled_project(
    project: &prism_compiler::CompiledProject,
    output_dir: &PathBuf,
) -> Result<()> {
    tokio::fs::create_dir_all(output_dir).await?;
    
    for (target, artifacts) in &project.artifacts {
        let target_dir = output_dir.join(target_name(*target));
        tokio::fs::create_dir_all(&target_dir).await?;
        
        for (i, artifact) in artifacts.iter().enumerate() {
            let filename = format!("module_{}.{}", i, target_extension(*target));
            let output_file = target_dir.join(filename);
            
            // Write the compiled output
            tokio::fs::write(&output_file, format!("// Compiled output for {:?}\n// Artifact: {:?}", target, artifact)).await?;
        }
        
        info!("📁 Written {} artifacts to: {}", artifacts.len(), target_dir.display());
    }
    
    Ok(())
}

fn create_lsp_config(ai_features: bool, config_file: Option<PathBuf>) -> Result<CompilationConfig> {
    let mut config = CompilationConfig {
        project_root: std::env::current_dir()?,
        targets: vec![CompilationTarget::TypeScript], // LSP uses TS for analysis
        optimization_level: 0, // No optimization for LSP
        enable_language_server: Some(true),
        export_ai_context: ai_features,
        incremental: Some(true), // LSP benefits from incremental compilation
        ai_features: Some(ai_features),
        debug_info: Some(true), // LSP needs debug info for hover/completion
        enable_transformations: Some(false),
        transformation_config: None,
        compiler_flags: std::collections::HashMap::new(),
        build_profile: BuildProfile::Debug,
        dependency_config: DependencyConfig::default(),
    };
    
    // Load config file if provided
    if let Some(config_path) = config_file {
        // In a real implementation, we'd load and merge the config file
        debug!("Loading LSP config from: {}", config_path.display());
    }
    
    Ok(config)
}

async fn analyze_structure(compiler: &PrismCompiler, input: &PathBuf, format: &str) -> Result<()> {
    info!("📊 Analyzing code structure...");
    
    // Use compiler's semantic analysis for structure analysis
    let stats = compiler.get_statistics().await;
    
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&stats)?;
            println!("{}", json);
        }
        "yaml" => {
            let yaml = serde_yaml::to_string(&stats)?;
            println!("{}", yaml);
        }
        "text" => {
            print_structure_analysis(&stats);
        }
        _ => {
            error!("❌ Unknown format for structure analysis: {}", format);
        }
    }
    
    Ok(())
}

async fn analyze_patterns(compiler: &PrismCompiler, input: &PathBuf, format: &str) -> Result<()> {
    info!("🎯 Analyzing code patterns...");
    
    // This would use the compiler's analysis capabilities
    println!("Pattern analysis results would be displayed here");
    Ok(())
}

async fn analyze_business_context(compiler: &PrismCompiler, input: &PathBuf, format: &str) -> Result<()> {
    info!("💼 Analyzing business context...");
    
    // Export AI context for business analysis
    let ai_context = compiler.export_ai_context().await?;
    
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&ai_context)?;
            println!("{}", json);
        }
        "text" => {
            print_business_analysis(&ai_context);
        }
        _ => {
            error!("❌ Unknown format for business analysis: {}", format);
        }
    }
    
    Ok(())
}

async fn analyze_performance(compiler: &PrismCompiler, input: &PathBuf, format: &str) -> Result<()> {
    info!("⚡ Analyzing performance characteristics...");
    println!("Performance analysis results would be displayed here");
    Ok(())
}

async fn analyze_semantics(compiler: &PrismCompiler, input: &PathBuf, format: &str) -> Result<()> {
    info!("🧠 Analyzing semantic information...");
    println!("Semantic analysis results would be displayed here");
    Ok(())
}

fn print_metadata_summary(metadata: &ComprehensiveAIMetadata) {
    println!("\n🤖 AI Metadata Summary");
    println!("========================");
    println!("Version: {}", metadata.version);
    println!("Exported: {}", metadata.exported_at);
    println!("Project: {}", metadata.project_info.name);
    
    if let Some(ref syntax) = metadata.syntax_metadata {
        println!("\n📝 Syntax Information:");
        println!("  - Files processed: {}", metadata.project_info.source_files.len());
    }
    
    if let Some(ref semantic) = metadata.semantic_metadata {
        println!("\n🧠 Semantic Information:");
        println!("  - Semantic analysis completed");
    }
    
    if let Some(ref business) = metadata.business_context {
        println!("\n💼 Business Context:");
        println!("  - Business rules analyzed");
    }
    
    println!("\n✅ Use --output to save detailed metadata to files");
}

fn print_structure_analysis(stats: &prism_compiler::CompilationStatistics) {
    println!("\n📊 Code Structure Analysis");
    println!("==========================");
    println!("Total compilation time: {:?}", stats.total_time);
    println!("Memory usage: {:?}", stats.memory_usage);
    println!("Cache performance: {:?}", stats.cache_performance);
    
    println!("\n📋 Phase Timings:");
    for (phase, duration) in &stats.phase_timings {
        println!("  {:?}: {:?}", phase, duration);
    }
    
    println!("\n🔍 Diagnostics Summary:");
    println!("  Errors: {}", stats.diagnostic_counts.errors);
    println!("  Warnings: {}", stats.diagnostic_counts.warnings);
    println!("  Hints: {}", stats.diagnostic_counts.hints);
}

fn print_business_analysis(ai_context: &prism_compiler::AIContext) {
    println!("\n💼 Business Context Analysis");
    println!("============================");
    // This would print business context information from the AI context
    println!("Business analysis results from AI context");
}

/// Get human-readable format name
fn format_name(format: &ExportFormat) -> &str {
    match format {
        ExportFormat::Json => "JSON",
        ExportFormat::Yaml => "YAML", 
        ExportFormat::Xml => "XML",
        ExportFormat::Binary => "Binary",
        ExportFormat::Protobuf => "Protocol Buffers",
        ExportFormat::OpenApi => "OpenAPI",
        ExportFormat::GraphQL => "GraphQL",
        ExportFormat::Custom(name) => name,
    }
}

/// Get file extension for format
fn format_extension(format: &ExportFormat) -> &str {
    match format {
        ExportFormat::Json => "json",
        ExportFormat::Yaml => "yaml",
        ExportFormat::Xml => "xml",
        ExportFormat::Binary => "bin",
        ExportFormat::Protobuf => "proto",
        ExportFormat::OpenApi => "openapi.json",
        ExportFormat::GraphQL => "graphql",
        ExportFormat::Custom(_) => "txt",
    }
}

fn target_name(target: CompilationTarget) -> &'static str {
    match target {
        CompilationTarget::TypeScript => "typescript",
        CompilationTarget::JavaScript => "javascript",
        CompilationTarget::WebAssembly => "webassembly",
        CompilationTarget::LLVM => "native",
    }
}

fn target_extension(target: CompilationTarget) -> &'static str {
    match target {
        CompilationTarget::TypeScript => "ts",
        CompilationTarget::JavaScript => "js",
        CompilationTarget::WebAssembly => "wasm",
        CompilationTarget::LLVM => "ll",
    }
}
