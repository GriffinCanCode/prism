//! End-to-End Integration Test for Prism Compiler Pipeline
//!
//! This test attempts to compile a simple Prism program through the complete pipeline:
//! Source Code → Lexer → Parser → Compiler → PIR → TypeScript Codegen → Output
//!
//! This will help us identify exactly what's broken in the integration.

use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;
use anyhow::Result;

use prism_cli::{Cli, Commands, execute_command};

/// Simple test program in Prism syntax
const TEST_PROGRAM: &str = r#"
module TestModule {
    fn hello_world() -> String {
        "Hello, World!"
    }
    
    fn add_numbers(a: Number, b: Number) -> Number {
        a + b
    }
}
"#;

#[tokio::test]
async fn test_complete_pipeline_end_to_end() -> Result<()> {
    println!("🚀 Starting end-to-end pipeline integration test...");
    
    // Create temporary directory for test project
    let temp_dir = TempDir::new()?;
    let project_path = temp_dir.path();
    let source_file = project_path.join("test.prism");
    
    // Write test program to file
    fs::write(&source_file, TEST_PROGRAM)?;
    println!("✅ Created test file: {}", source_file.display());
    
    // Test CLI compilation command
    println!("\n📋 Testing CLI Compilation Command");
    let result = test_cli_compilation(&source_file).await;
    match result {
        Ok(_) => println!("✅ CLI compilation succeeded"),
        Err(e) => {
            println!("❌ CLI compilation failed: {}", e);
            print_error_details(&e);
        }
    }
    
    println!("\n🎯 End-to-end integration test completed");
    Ok(())
}

/// Test CLI compilation command
async fn test_cli_compilation(source_file: &PathBuf) -> Result<()> {
    let output_dir = source_file.parent().unwrap().join("output");
    
    let cli = Cli {
        command: Commands::Compile {
            input: source_file.clone(),
            output: Some(output_dir.clone()),
            target: "typescript".to_string(),
            optimization: 1,
            ai_metadata: true,
            incremental: false,
            debug: true,
        }
    };
    
    execute_command(cli).await?;
    
    // Check if output was generated
    if output_dir.exists() {
        println!("✅ Output directory created: {}", output_dir.display());
        
        // Look for generated TypeScript files
        for entry in fs::read_dir(&output_dir)? {
            let entry = entry?;
            println!("📄 Generated file: {}", entry.file_name().to_string_lossy());
        }
    } else {
        println!("❌ No output directory created");
    }
    
    Ok(())
}

/// Helper function to print detailed error information
fn print_error_details(error: &anyhow::Error) {
    println!("❌ Error: {}", error);
    
    let mut source = error.source();
    let mut level = 1;
    while let Some(err) = source {
        println!("  {}Caused by: {}", "  ".repeat(level), err);
        source = err.source();
        level += 1;
    }
} 