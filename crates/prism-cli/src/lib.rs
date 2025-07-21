//! Prism CLI Library
//!
//! This module provides the command-line interface functionality for the Prism language,
//! including AI metadata export capabilities.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;

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
        #[arg(short, long, default_value = "native")]
        target: String,
        
        /// Enable AI metadata generation
        #[arg(long)]
        ai_metadata: bool,
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
    },
}

/// Execute the CLI command
pub async fn execute_command(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Compile { 
            input, 
            output, 
            target, 
            ai_metadata 
        } => {
            execute_compile(input, output, target, ai_metadata).await
        }
        
        Commands::AIExport { 
            input, 
            output, 
            formats, 
            business_context, 
            performance_metrics, 
            architectural_patterns 
        } => {
            execute_ai_export(
                input, 
                output, 
                formats, 
                business_context, 
                performance_metrics, 
                architectural_patterns
            ).await
        }
        
        Commands::Analyze { 
            input, 
            analysis_type, 
            format 
        } => {
            execute_analyze(input, analysis_type, format).await
        }
        
        Commands::LanguageServer { 
            ai_features, 
            config 
        } => {
            execute_language_server(ai_features, config).await
        }
    }
}

/// Execute compile command
async fn execute_compile(
    input: PathBuf,
    output: Option<PathBuf>,
    target: String,
    ai_metadata: bool,
) -> Result<()> {
    println!("Compiling {} for target: {}", input.display(), target);
    
    if ai_metadata {
        println!("AI metadata generation enabled");
    }
    
    // This would integrate with the actual compiler
    println!("Compilation would happen here");
    
    Ok(())
}

/// Execute AI export command
async fn execute_ai_export(
    input: PathBuf,
    output: Option<PathBuf>,
    formats: Vec<String>,
    business_context: bool,
    performance_metrics: bool,
    architectural_patterns: bool,
) -> Result<()> {
    use prism_ai::{AIIntegrationCoordinator, AIIntegrationConfig, ExportFormat};
    
    println!("Exporting AI metadata from: {}", input.display());
    
    // Parse export formats
    let export_formats: Result<Vec<ExportFormat>, _> = formats.iter()
        .map(|f| match f.to_lowercase().as_str() {
            "json" => Ok(ExportFormat::Json),
            "yaml" => Ok(ExportFormat::Yaml),
            "xml" => Ok(ExportFormat::Xml),
            "openapi" => Ok(ExportFormat::OpenApi),
            "graphql" => Ok(ExportFormat::GraphQL),
            _ => Err(anyhow::anyhow!("Unknown export format: {}", f)),
        })
        .collect();
    
    let export_formats = export_formats?;
    
    // Create AI integration configuration
    let config = AIIntegrationConfig {
        enabled: true,
        export_formats: export_formats.clone(),
        include_business_context: business_context,
        include_performance_metrics: performance_metrics,
        include_architectural_patterns: architectural_patterns,
        min_confidence_threshold: 0.7,
        output_directory: output,
    };
    
    // Create coordinator and collect metadata
    let coordinator = AIIntegrationCoordinator::new(config);
    
    match coordinator.collect_metadata(&input).await {
        Ok(metadata) => {
            println!("Successfully collected metadata");
            
            // Export in requested formats
            match coordinator.export_metadata(&metadata, &export_formats).await {
                Ok(exports) => {
                    for (format, content) in exports {
                        println!("Exported {} format ({} bytes)", format_name(&format), content.len());
                        
                                                 // Write to files if output directory is specified
                         if let Some(output_dir) = &output {
                            let filename = format!("prism_metadata.{}", format_extension(&format));
                            let filepath = output_dir.join(filename);
                            
                            tokio::fs::create_dir_all(output_dir).await?;
                            tokio::fs::write(&filepath, content).await?;
                            
                            println!("Written to: {}", filepath.display());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Export failed: {}", e);
                    return Err(e.into());
                }
            }
        }
        Err(e) => {
            eprintln!("Metadata collection failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}

/// Execute analyze command
async fn execute_analyze(
    input: PathBuf,
    analysis_type: String,
    format: String,
) -> Result<()> {
    println!("Analyzing {} (type: {}, format: {})", input.display(), analysis_type, format);
    
    // This would integrate with the actual analyzer
    println!("Analysis would happen here");
    
    Ok(())
}

/// Execute language server command
async fn execute_language_server(
    ai_features: bool,
    config: Option<PathBuf>,
) -> Result<()> {
    println!("Starting Prism language server");
    
    if ai_features {
        println!("AI features enabled");
    }
    
    if let Some(config_path) = config {
        println!("Using config: {}", config_path.display());
    }
    
    // This would start the actual language server
    println!("Language server would start here");
    
    Ok(())
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
