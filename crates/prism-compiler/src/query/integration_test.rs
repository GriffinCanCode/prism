//! Integration Tests for Query-Based Compilation Pipeline
//!
//! This module contains comprehensive tests that verify the complete integration
//! of the query-based compilation pipeline with incremental compilation, file watching,
//! and all compilation phases.

use crate::error::{CompilerError, CompilerResult};
use crate::query::pipeline::{CompilationPipeline, PipelineConfig};
use crate::query::incremental::{IncrementalCompiler, IncrementalConfig};
use crate::context::{CompilationConfig, CompilationTarget};
use crate::PrismCompiler;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

/// Test the complete query-based compilation pipeline
#[tokio::test]
async fn test_complete_pipeline_integration() -> CompilerResult<()> {
    // Create a temporary project
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();
    
    // Create test files
    create_test_project(project_path)?;
    
    // Create pipeline configuration
    let pipeline_config = PipelineConfig {
        enable_parallel_execution: true,
        max_concurrent_phases: 2,
        enable_incremental: true,
        enable_ai_metadata: true,
        phase_timeout_secs: 30,
        enable_error_recovery: true,
        targets: vec![CompilationTarget::TypeScript],
    };
    
    // Create and test pipeline
    let pipeline = CompilationPipeline::new(pipeline_config);
    let result = pipeline.compile_project(project_path).await?;
    
    // Verify compilation result
    assert!(result.success, "Pipeline compilation should succeed");
    assert!(!result.phase_results.is_empty(), "Should have phase results");
    assert!(result.stats.total_files > 0, "Should have processed files");
    
    // Verify AI metadata was generated
    if result.ai_metadata.is_some() {
        println!("✅ AI metadata generated successfully");
    }
    
    println!("✅ Pipeline integration test passed");
    Ok(())
}

/// Test incremental compilation with file watching
#[tokio::test]
async fn test_incremental_compilation_integration() -> CompilerResult<()> {
    // Create a temporary project
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();
    
    // Create test files
    create_test_project(project_path)?;
    
    // Create pipeline and incremental compiler
    let pipeline_config = PipelineConfig::default();
    let pipeline = Arc::new(CompilationPipeline::new(pipeline_config));
    
    let incremental_config = IncrementalConfig {
        enable_file_watching: true,
        debounce_ms: 50, // Shorter for testing
        enable_semantic_detection: true,
        max_watched_files: 1000,
        enable_dependency_invalidation: true,
        auto_recompile: false,
    };
    
    let incremental_compiler = IncrementalCompiler::new(pipeline, incremental_config)?;
    
    // Start watching
    incremental_compiler.start_watching(project_path).await?;
    
    // Initial compilation
    let initial_result = incremental_compiler.compile_incremental(project_path).await?;
    assert!(initial_result.result.success, "Initial compilation should succeed");
    
    // Modify a file
    let main_file = project_path.join("main.prsm");
    fs::write(&main_file, "module Main { fn updated() -> String { \"updated\" } }")?;
    
    // Wait for file system events to be processed
    sleep(Duration::from_millis(200)).await;
    
    // Incremental compilation
    let incremental_result = incremental_compiler.compile_incremental(project_path).await?;
    assert!(incremental_result.result.success, "Incremental compilation should succeed");
    assert!(incremental_result.incremental_info.changes_detected > 0, "Should detect changes");
    
    // Stop watching
    incremental_compiler.stop_watching().await?;
    
    println!("✅ Incremental compilation integration test passed");
    Ok(())
}

/// Test full PrismCompiler integration with new pipeline
#[tokio::test]
async fn test_prism_compiler_integration() -> CompilerResult<()> {
    // Create a temporary project
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();
    
    // Create test files
    create_test_project(project_path)?;
    
    // Create compilation configuration
    let config = CompilationConfig {
        project_root: project_path.to_path_buf(),
        targets: vec![CompilationTarget::TypeScript],
        incremental: Some(true),
        ai_features: Some(true),
        enable_language_server: Some(false),
        export_ai_metadata: true,
        enable_transformations: Some(true),
        optimization_level: Some(2),
        parallel_compilation: Some(true),
    };
    
    // Create compiler
    let compiler = PrismCompiler::new(config)?;
    
    // Test backward-compatible compilation
    let result = compiler.compile_project(project_path).await?;
    assert!(result.success, "Compilation should succeed");
    
    // Test new pipeline method
    let pipeline_result = compiler.compile_project_with_pipeline(project_path).await?;
    assert!(pipeline_result.success, "Pipeline compilation should succeed");
    
    // Test incremental compilation
    if compiler.is_incremental_enabled() {
        let incremental_result = compiler.compile_project_incremental(project_path).await?;
        assert!(incremental_result.result.success, "Incremental compilation should succeed");
        
        // Test file watching
        compiler.start_watching(project_path).await?;
        
        let main_file = project_path.join("main.prsm");
        assert!(compiler.is_watching(&main_file).await, "Should be watching main file");
        
        compiler.stop_watching().await?;
    }
    
    // Test statistics
    let query_stats = compiler.get_query_stats();
    assert!(!query_stats.is_empty(), "Should have query statistics");
    
    let pipeline_metrics = compiler.get_pipeline_metrics();
    assert!(pipeline_metrics.files_processed > 0, "Should have processed files");
    
    println!("✅ PrismCompiler integration test passed");
    Ok(())
}

/// Test error recovery in the pipeline
#[tokio::test]
async fn test_pipeline_error_recovery() -> CompilerResult<()> {
    // Create a temporary project with syntax errors
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();
    
    // Create test files with intentional errors
    fs::write(project_path.join("main.prsm"), "module Main { invalid syntax }")?;
    fs::write(project_path.join("lib.prsm"), "module Lib { fn valid() -> String { \"ok\" } }")?;
    
    // Create pipeline with error recovery enabled
    let pipeline_config = PipelineConfig {
        enable_error_recovery: true,
        enable_parallel_execution: false, // Disable for deterministic testing
        ..Default::default()
    };
    
    let pipeline = CompilationPipeline::new(pipeline_config);
    let result = pipeline.compile_project(project_path).await?;
    
    // Should complete with errors but not crash
    assert!(!result.success, "Should report failure due to syntax errors");
    assert!(!result.diagnostics.is_empty(), "Should have diagnostic messages");
    assert!(!result.phase_results.is_empty(), "Should have attempted all phases");
    
    println!("✅ Pipeline error recovery test passed");
    Ok(())
}

/// Test parallel phase execution
#[tokio::test]
async fn test_parallel_phase_execution() -> CompilerResult<()> {
    // Create a larger test project
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();
    
    // Create multiple test files
    for i in 0..5 {
        fs::write(
            project_path.join(format!("module_{}.prsm", i)),
            format!("module Module{} {{ fn function_{}() -> String {{ \"module_{}\" }} }}", i, i, i)
        )?;
    }
    
    // Test with parallel execution enabled
    let parallel_config = PipelineConfig {
        enable_parallel_execution: true,
        max_concurrent_phases: 4,
        ..Default::default()
    };
    
    let parallel_pipeline = CompilationPipeline::new(parallel_config);
    let parallel_start = std::time::Instant::now();
    let parallel_result = parallel_pipeline.compile_project(project_path).await?;
    let parallel_time = parallel_start.elapsed();
    
    // Test with sequential execution
    let sequential_config = PipelineConfig {
        enable_parallel_execution: false,
        max_concurrent_phases: 1,
        ..Default::default()
    };
    
    let sequential_pipeline = CompilationPipeline::new(sequential_config);
    let sequential_start = std::time::Instant::now();
    let sequential_result = sequential_pipeline.compile_project(project_path).await?;
    let sequential_time = sequential_start.elapsed();
    
    // Both should succeed
    assert!(parallel_result.success, "Parallel compilation should succeed");
    assert!(sequential_result.success, "Sequential compilation should succeed");
    
    // Parallel should generally be faster (though this isn't guaranteed in tests)
    println!("Parallel time: {:?}, Sequential time: {:?}", parallel_time, sequential_time);
    
    println!("✅ Parallel phase execution test passed");
    Ok(())
}

/// Test query caching and invalidation
#[tokio::test]
async fn test_query_caching_integration() -> CompilerResult<()> {
    // Create a temporary project
    let temp_dir = TempDir::new().unwrap();
    let project_path = temp_dir.path();
    
    // Create test files
    create_test_project(project_path)?;
    
    // Create pipeline with caching enabled
    let pipeline_config = PipelineConfig {
        enable_incremental: true,
        ..Default::default()
    };
    
    let pipeline = Arc::new(CompilationPipeline::new(pipeline_config));
    
    let incremental_config = IncrementalConfig {
        enable_file_watching: false, // Disable for controlled testing
        enable_semantic_detection: true,
        ..Default::default()
    };
    
    let incremental_compiler = IncrementalCompiler::new(pipeline, incremental_config)?;
    
    // First compilation (cache miss)
    let first_result = incremental_compiler.compile_incremental(project_path).await?;
    assert!(first_result.result.success, "First compilation should succeed");
    
    // Second compilation without changes (should use cache)
    let second_result = incremental_compiler.compile_incremental(project_path).await?;
    assert!(second_result.result.success, "Second compilation should succeed");
    
    // The second compilation should be faster due to caching
    // (In a real implementation, we'd verify cache hit statistics)
    
    println!("✅ Query caching integration test passed");
    Ok(())
}

/// Helper function to create a test project
fn create_test_project(project_path: &Path) -> CompilerResult<()> {
    // Create main module
    fs::write(
        project_path.join("main.prsm"),
        r#"
module Main {
    import Lib;
    
    fn main() -> String {
        Lib.hello()
    }
    
    fn calculate(x: Number, y: Number) -> Number {
        x + y
    }
}
"#,
    ).map_err(|e| CompilerError::FileWriteError { 
        path: project_path.join("main.prsm"), 
        source: e 
    })?;
    
    // Create library module
    fs::write(
        project_path.join("lib.prsm"),
        r#"
module Lib {
    fn hello() -> String {
        "Hello from Lib!"
    }
    
    fn process_data(data: Array<String>) -> Array<String> {
        data.map(s => s.toUpperCase())
    }
}
"#,
    ).map_err(|e| CompilerError::FileWriteError { 
        path: project_path.join("lib.prsm"), 
        source: e 
    })?;
    
    // Create configuration file
    fs::write(
        project_path.join("prism.toml"),
        r#"
[project]
name = "test-project"
version = "0.1.0"

[compilation]
targets = ["typescript"]
incremental = true
ai_features = true

[dependencies]
# No external dependencies for test
"#,
    ).map_err(|e| CompilerError::FileWriteError { 
        path: project_path.join("prism.toml"), 
        source: e 
    })?;
    
    Ok(())
}

/// Test runner for all integration tests
#[tokio::test]
async fn run_all_integration_tests() -> CompilerResult<()> {
    println!("🚀 Running all query-based compilation pipeline integration tests...\n");
    
    test_complete_pipeline_integration().await?;
    test_incremental_compilation_integration().await?;
    test_prism_compiler_integration().await?;
    test_pipeline_error_recovery().await?;
    test_parallel_phase_execution().await?;
    test_query_caching_integration().await?;
    
    println!("\n🎉 All integration tests passed successfully!");
    Ok(())
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Benchmark the compilation pipeline performance
    #[tokio::test]
    async fn benchmark_pipeline_performance() -> CompilerResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let project_path = temp_dir.path();
        
        // Create a larger project for benchmarking
        for i in 0..20 {
            fs::write(
                project_path.join(format!("module_{:02}.prsm", i)),
                format!(
                    r#"
module Module{} {{
    import Module{};
    
    fn function_{}() -> String {{
        "result from module {}"
    }}
    
    fn compute_{}(x: Number) -> Number {{
        x * {} + {}
    }}
    
    fn process_array_{}(items: Array<String>) -> Array<String> {{
        items.filter(s => s.length > {}).map(s => s + "_processed")
    }}
}}
"#,
                    i,
                    (i + 1) % 20,
                    i, i,
                    i, i, i,
                    i, i % 10
                )
            )?;
        }
        
        let pipeline_config = PipelineConfig {
            enable_parallel_execution: true,
            max_concurrent_phases: num_cpus::get(),
            enable_incremental: true,
            enable_ai_metadata: true,
            ..Default::default()
        };
        
        let pipeline = CompilationPipeline::new(pipeline_config);
        
        // Warm up
        let _ = pipeline.compile_project(project_path).await?;
        
        // Benchmark
        let start = Instant::now();
        let result = pipeline.compile_project(project_path).await?;
        let duration = start.elapsed();
        
        assert!(result.success, "Benchmark compilation should succeed");
        
        println!("📊 Benchmark Results:");
        println!("   - Files processed: {}", result.stats.total_files);
        println!("   - Total time: {:?}", duration);
        println!("   - Time per file: {:?}", duration / result.stats.total_files as u32);
        println!("   - Cache hit rate: {:.2}%", result.stats.cache_hit_rate * 100.0);
        println!("   - Parallel phases: {}", result.stats.parallel_phases);
        
        Ok(())
    }
} 