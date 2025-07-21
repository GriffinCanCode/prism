//! Prism programming language CLI
//!
//! This is the main entry point for the Prism compiler and tools.

use clap::Parser;
use prism_cli::{Cli, execute_command};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Parse command line arguments
    let cli = Cli::parse();
    
    // Execute the command
    execute_command(cli).await
} 