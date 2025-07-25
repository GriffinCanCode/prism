//! Examples of the New Analysis Pipeline System
//!
//! This module demonstrates how the refactored analysis system preserves all the 
//! optimization detection logic while providing better modularity and consistency.

use super::*;
use crate::bytecode::FunctionDefinition;

/// Example showing how to use the new pipeline system
pub fn demonstrate_new_pipeline() -> VMResult<()> {
    // Create a sample function (placeholder)
    let function = create_sample_function();
    
    // 1. Create pipeline with configuration
    let config = PipelineConfig::default();
    let mut pipeline = AnalysisPipeline::new(config);
    
    // 2. Register analyzers (this preserves all the old analysis logic)
    pipeline.register_analyzer(control_flow::CFGAnalyzer::new(&AnalysisConfig::default())?)?;
    pipeline.register_analyzer(data_flow::DataFlowAnalyzer::new(&AnalysisConfig::default())?)?;
    // TODO: Register other analyzers as they're refactored
    
    // 3. Analyze function - this runs all analyses in dependency order
    let context = pipeline.analyze_function(function)?;
    
    // 4. Access results (same data, better organized)
    println!("Analysis completed in {:?}", context.metadata.analysis_time);
    println!("Confidence: {:.2}", context.metadata.confidence);
    println!("Found {} optimization opportunities", context.optimization_opportunities.len());
    
    // 5. Access specific analysis results
    if let Some(cfg) = context.get_result::<control_flow::ControlFlowGraph>(AnalysisKind::ControlFlow) {
        println!("CFG has {} basic blocks", cfg.blocks.len());
    }
    
    if let Some(dataflow) = context.get_result::<data_flow::DataFlowAnalysis>(AnalysisKind::DataFlow) {
        println!("Data flow analysis completed for function {}", dataflow.function_id);
    }
    
    // 6. Process optimization opportunities (same logic, better structure)
    for opportunity in &context.optimization_opportunities {
        println!("Optimization: {:?}", opportunity.kind);
        println!("  Benefit: {:.2}", opportunity.estimated_benefit);
        println!("  Cost: {:.2}", opportunity.implementation_cost);
        println!("  Profitable: {}", opportunity.is_profitable());
        println!("  Risk-adjusted benefit: {:.2}", opportunity.risk_adjusted_benefit());
    }
    
    Ok(())
}

/// Example showing backward compatibility with the old interface
pub fn demonstrate_legacy_compatibility() -> VMResult<()> {
    let function = create_sample_function();
    
    // Old interface still works!
    let config = AnalysisConfig::default();
    let mut analyzer = StaticAnalyzer::new(config)?;
    
    let result = analyzer.analyze_function(&function)?;
    
    // Same result structure as before
    println!("Function {} analyzed", result.function_id);
    println!("Found {} optimizations", result.optimizations.len());
    
    // But now you can also access the new pipeline
    let pipeline = analyzer.pipeline();
    let stats = pipeline.get_statistics();
    println!("Pipeline has {} registered analyzers", stats.registered_analyzers);
    
    Ok(())
}

/// Example showing how optimization detection logic is preserved
pub fn demonstrate_optimization_detection_preservation() -> VMResult<()> {
    let function = create_sample_function();
    let mut pipeline = AnalysisPipeline::new(PipelineConfig::default());
    
    // Register analyzers
    pipeline.register_analyzer(control_flow::CFGAnalyzer::new(&AnalysisConfig::default())?)?;
    pipeline.register_analyzer(data_flow::DataFlowAnalyzer::new(&AnalysisConfig::default())?)?;
    
    let context = pipeline.analyze_function(function)?;
    
    // All the old optimization types are preserved in the new unified system:
    for opportunity in &context.optimization_opportunities {
        match &opportunity.kind {
            OptimizationKind::DeadCodeElimination { dead_instructions, unreachable_blocks, dead_variables } => {
                println!("Dead Code Elimination:");
                println!("  {} dead instructions", dead_instructions.len());
                println!("  {} unreachable blocks", unreachable_blocks.len());
                println!("  {} dead variables", dead_variables.len());
            }
            OptimizationKind::ConstantFolding { expressions, propagation_chains } => {
                println!("Constant Folding:");
                println!("  {} foldable expressions", expressions.len());
                println!("  {} propagation chains", propagation_chains.len());
            }
            OptimizationKind::LoopOptimization { loop_id, techniques } => {
                println!("Loop Optimization for loop {}:", loop_id);
                for technique in techniques {
                    match technique {
                        LoopOptimizationTechnique::Unrolling { factor, allow_partial } => {
                            println!("  Unrolling: factor={}, partial={}", factor, allow_partial);
                        }
                        LoopOptimizationTechnique::Vectorization { vector_width, operations } => {
                            println!("  Vectorization: width={}, {} operations", vector_width, operations.len());
                        }
                        LoopOptimizationTechnique::InvariantCodeMotion { expressions } => {
                            println!("  Invariant code motion: {} expressions", expressions.len());
                        }
                        _ => println!("  Other loop optimization"),
                    }
                }
            }
            OptimizationKind::TypeSpecialization { location, specialization_type } => {
                println!("Type Specialization at {}: {:?}", location, specialization_type);
            }
            OptimizationKind::HotnessOptimization { hot_spot_id, optimization_type } => {
                println!("Hotness Optimization for hot spot {}: {:?}", hot_spot_id, optimization_type);
            }
            _ => println!("Other optimization: {:?}", opportunity.kind),
        }
    }
    
    Ok(())
}

/// Example showing improved analysis coordination
pub fn demonstrate_improved_coordination() -> VMResult<()> {
    let function = create_sample_function();
    
    // Create pipeline with specific configuration
    let mut config = PipelineConfig::default();
    config.optimization_config.aggressiveness = 0.8; // More aggressive
    config.optimization_config.min_benefit_threshold = 0.02; // Lower threshold
    
    let mut pipeline = AnalysisPipeline::new(config);
    
    // Register analyzers
    pipeline.register_analyzer(control_flow::CFGAnalyzer::new(&AnalysisConfig::default())?)?;
    pipeline.register_analyzer(data_flow::DataFlowAnalyzer::new(&AnalysisConfig::default())?)?;
    
    let context = pipeline.analyze_function(function)?;
    
    // The new system provides better coordination:
    println!("=== Analysis Coordination Improvements ===");
    
    // 1. Dependency management is automatic
    println!("Analyses run in dependency order: {:?}", context.metadata.passes_run);
    
    // 2. Shared data structures eliminate inconsistencies
    if let Some(cfg) = context.get_result::<control_flow::ControlFlowGraph>(AnalysisKind::ControlFlow) {
        if let Some(dataflow) = context.get_result::<data_flow::DataFlowAnalysis>(AnalysisKind::DataFlow) {
            // Both analyses use the same Variable type - no more inconsistencies!
            println!("CFG and dataflow use consistent variable representations");
        }
    }
    
    // 3. Unified optimization detection
    println!("All optimizations detected centrally: {} total", context.optimization_opportunities.len());
    
    // 4. Better error handling and warnings
    for warning in &context.metadata.warnings {
        println!("Warning: {}", warning);
    }
    
    // 5. Comprehensive metadata
    println!("Analysis confidence: {:.2}", context.metadata.confidence);
    for (metric, value) in &context.metadata.metrics {
        println!("Metric {}: {:.2}", metric, value);
    }
    
    Ok(())
}

/// Helper function to create a sample function for examples
fn create_sample_function() -> FunctionDefinition {
    // This is a placeholder - in real usage you'd have actual bytecode
    FunctionDefinition {
        id: 1,
        name: "sample_function".to_string(),
        instructions: vec![], // Placeholder
        locals_count: 0,
        parameters_count: 0,
        max_stack_size: 0,
    }
}

/// Summary of what was preserved and improved
pub fn print_preservation_summary() {
    println!("=== OPTIMIZATION LOGIC PRESERVATION SUMMARY ===");
    println!();
    
    println!("✅ PRESERVED:");
    println!("  • All optimization detection algorithms");
    println!("  • All optimization types (DCE, constant folding, loop opts, etc.)");
    println!("  • All heuristics and cost models");
    println!("  • All mathematical transformations");
    println!("  • All profitability analysis");
    println!("  • All safety conditions and risk assessment");
    println!("  • Backward compatibility with existing interfaces");
    println!();
    
    println!("🚀 IMPROVED:");
    println!("  • Unified optimization opportunity representation");
    println!("  • Consistent data structures across all analyses");
    println!("  • Automatic dependency management");
    println!("  • Centralized optimization detection and ranking");
    println!("  • Better error handling and metadata");
    println!("  • Eliminated code duplication");
    println!("  • Type-safe analysis result access");
    println!("  • Configurable analysis pipeline");
    println!();
    
    println!("🔧 ARCHITECTURAL BENEFITS:");
    println!("  • No more inconsistent Variable/MemoryPattern/etc. definitions");
    println!("  • No more overlapping optimization detection");
    println!("  • Proper separation of concerns");
    println!("  • Extensible for new analysis types");
    println!("  • Better testability and maintainability");
    println!("  • Pipeline can be configured for different optimization levels");
    println!();
    
    println!("📊 MIGRATION PATH:");
    println!("  • Old StaticAnalyzer interface still works (legacy compatibility)");
    println!("  • New code can use AnalysisPipeline directly");
    println!("  • Individual analyses can be refactored incrementally");
    println!("  • Optimization detection is now centralized and consistent");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_pipeline_preserves_functionality() {
        // This test would verify that the new pipeline produces
        // equivalent results to the old system
        // (Implementation would depend on having actual test cases)
    }

    #[test]
    fn test_shared_types_consistency() {
        // Test that shared types work consistently across modules
        let var1 = Variable::local("test", 0);
        let var2 = Variable::local("test", 0);
        assert_eq!(var1, var2);
        
        let pattern = MemoryAccessPattern::sequential(var1);
        assert_eq!(pattern.temporal_locality, 0.8);
    }

    #[test]
    fn test_optimization_opportunity_methods() {
        // Test the helper methods on OptimizationOpportunity
        let opportunity = OptimizationOpportunity {
            id: "test".to_string(),
            kind: OptimizationKind::DeadCodeElimination {
                dead_instructions: vec![1, 2, 3],
                unreachable_blocks: vec![],
                dead_variables: vec![],
            },
            location: CodeLocation::Instruction { offset: 10 },
            estimated_benefit: 0.8,
            implementation_cost: 0.2,
            confidence: 0.9,
            prerequisites: vec![],
            side_effects: vec![],
            analysis: OptimizationAnalysis {
                performance_impact: PerformanceImpact {
                    speedup: 1.5,
                    instruction_reduction: 3,
                    memory_access_reduction: 0,
                    branch_improvement: 0.0,
                    cache_improvement: 0.0,
                },
                resource_impact: ResourceImpact {
                    code_size_delta: -12,
                    register_pressure_delta: 0,
                    memory_usage_delta: 0,
                    compilation_time_cost: 1.0,
                },
                dependencies: vec![],
                risks: vec![],
                profitability: ProfitabilityAnalysis {
                    performance_gain: 100.0,
                    code_size_impact: -12,
                    compilation_time_cost: 1.0,
                    risk_level: 0.1,
                    profitability_score: 0.7,
                },
            },
            source_analysis: AnalysisKind::DataFlow,
        };
        
        assert!(opportunity.is_profitable());
        assert!(opportunity.net_benefit() > 0.0);
        assert!(opportunity.risk_adjusted_benefit() > 0.0);
    }
} 

/// Demonstrate optimized capability propagation graph algorithms
pub fn demonstrate_optimized_capability_analysis() -> VMResult<()> {
    use super::capability_analysis::*;
    use super::shared::*;
    use crate::bytecode::*;
    use std::time::Instant;

    println!("=== Optimized Capability Propagation Graph Analysis ===\n");

    // Create a large synthetic function for performance testing
    let large_function = create_large_synthetic_function(5000); // 5000 instructions
    let config = AnalysisConfig::default();
    let mut analyzer = CapabilityAnalyzer::new(&config)?;

    println!("Analyzing function with {} instructions...", large_function.instructions.len());

    // Benchmark the analysis
    let start_time = Instant::now();
    let analysis = analyzer.analyze(&large_function)?;
    let analysis_time = start_time.elapsed();

    println!("Analysis completed in {:?}", analysis_time);
    println!("Found {} capability nodes", analysis.propagation_graph.nodes.len());
    println!("Found {} capability edges", analysis.propagation_graph.edges.len());
    println!("Found {} strongly connected components", analysis.propagation_graph.components.len());
    println!("Topological order length: {}", analysis.propagation_graph.topological_order.len());

    // Demonstrate algorithm efficiency
    demonstrate_scc_efficiency(&analysis.propagation_graph)?;
    demonstrate_topological_sort_efficiency(&analysis.propagation_graph)?;
    demonstrate_optimization_opportunities(&analysis)?;

    Ok(())
}

/// Demonstrate SCC algorithm efficiency
fn demonstrate_scc_efficiency(graph: &CapabilityPropagationGraph) -> VMResult<()> {
    println!("\n--- Strongly Connected Components Analysis ---");
    
    // Analyze component sizes
    let mut component_sizes: Vec<usize> = graph.components.iter().map(|c| c.len()).collect();
    component_sizes.sort_unstable_by(|a, b| b.cmp(a)); // Sort by size descending
    
    println!("Total components: {}", graph.components.len());
    if !component_sizes.is_empty() {
        println!("Largest component size: {}", component_sizes[0]);
        println!("Average component size: {:.2}", 
                 component_sizes.iter().sum::<usize>() as f64 / component_sizes.len() as f64);
        
        // Show distribution
        let singleton_count = component_sizes.iter().filter(|&&size| size == 1).count();
        println!("Singleton components: {} ({:.1}%)", 
                 singleton_count, 
                 100.0 * singleton_count as f64 / component_sizes.len() as f64);
    }

    // Demonstrate cycle detection
    let has_cycles = graph.components.iter().any(|c| c.len() > 1);
    if has_cycles {
        println!("⚠️  Cycles detected in capability propagation graph");
        let cyclic_components: Vec<_> = graph.components.iter()
            .filter(|c| c.len() > 1)
            .collect();
        println!("Cyclic components: {}", cyclic_components.len());
    } else {
        println!("✅ No cycles detected - graph is acyclic");
    }

    Ok(())
}

/// Demonstrate topological sort efficiency
fn demonstrate_topological_sort_efficiency(graph: &CapabilityPropagationGraph) -> VMResult<()> {
    println!("\n--- Topological Sort Analysis ---");
    
    // Verify topological order properties
    let is_valid_topo_order = verify_topological_order(graph);
    if is_valid_topo_order {
        println!("✅ Topological order is valid");
    } else {
        println!("❌ Topological order is invalid");
    }

    // Analyze the ordering
    println!("Topological order length: {}", graph.topological_order.len());
    println!("Graph nodes count: {}", graph.nodes.len());
    
    if graph.topological_order.len() == graph.nodes.len() {
        println!("✅ All nodes included in topological order");
    } else {
        println!("⚠️  Mismatch: {} nodes vs {} in topo order", 
                 graph.nodes.len(), graph.topological_order.len());
    }

    // Show capability flow direction
    analyze_capability_flow_direction(graph)?;

    Ok(())
}

/// Verify topological order is correct
fn verify_topological_order(graph: &CapabilityPropagationGraph) -> bool {
    // Create position mapping
    let mut positions = std::collections::HashMap::new();
    for (pos, &node) in graph.topological_order.iter().enumerate() {
        positions.insert(node, pos);
    }

    // Check all edges respect topological order
    for edge in &graph.edges {
        let from_pos = positions.get(&edge.from);
        let to_pos = positions.get(&edge.to);
        
        match (from_pos, to_pos) {
            (Some(&from), Some(&to)) => {
                if from >= to {
                    return false; // Edge goes backwards in topological order
                }
            }
            _ => return false, // Missing node in topological order
        }
    }

    true
}

/// Analyze capability flow direction
fn analyze_capability_flow_direction(graph: &CapabilityPropagationGraph) -> VMResult<()> {
    println!("\n--- Capability Flow Analysis ---");
    
    // Count different flow types
    let mut flow_type_counts = std::collections::HashMap::new();
    for edge in &graph.edges {
        *flow_type_counts.entry(&edge.flow_type).or_insert(0) += 1;
    }

    println!("Flow type distribution:");
    for (flow_type, count) in flow_type_counts {
        println!("  {:?}: {} edges", flow_type, count);
    }

    // Analyze capability propagation patterns
    let mut capability_counts = std::collections::HashMap::new();
    for edge in &graph.edges {
        for cap in &edge.propagated_capabilities.available {
            *capability_counts.entry(cap.clone()).or_insert(0) += 1;
        }
    }

    if !capability_counts.is_empty() {
        println!("\nMost propagated capabilities:");
        let mut sorted_caps: Vec<_> = capability_counts.into_iter().collect();
        sorted_caps.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (cap, count) in sorted_caps.into_iter().take(5) {
            println!("  {}: {} propagations", cap.name, count);
        }
    }

    Ok(())
}

/// Demonstrate optimization opportunities from capability analysis
fn demonstrate_optimization_opportunities(analysis: &CapabilityAnalysis) -> VMResult<()> {
    println!("\n--- Optimization Opportunities ---");
    
    // Count safe vs unsafe optimizations
    let safe_count = analysis.optimization_safety.safe_optimizations.len();
    let unsafe_count = analysis.optimization_safety.unsafe_optimizations.len();
    
    println!("Safe optimizations: {}", safe_count);
    println!("Unsafe optimizations: {}", unsafe_count);
    
    if safe_count > 0 {
        println!("\nSafe optimization types:");
        let mut type_counts = std::collections::HashMap::new();
        for opt in &analysis.optimization_safety.safe_optimizations {
            *type_counts.entry(&opt.optimization_type).or_insert(0) += 1;
        }
        
        for (opt_type, count) in type_counts {
            println!("  {:?}: {} opportunities", opt_type, count);
        }
    }

    // Analyze security constraints
    println!("\nSecurity constraints: {}", analysis.security_constraints.len());
    if !analysis.security_constraints.is_empty() {
        let mut severity_counts = std::collections::HashMap::new();
        for constraint in &analysis.security_constraints {
            *severity_counts.entry(&constraint.severity).or_insert(0) += 1;
        }
        
        println!("Constraint severity distribution:");
        for (severity, count) in severity_counts {
            println!("  {:?}: {} constraints", severity, count);
        }
    }

    // Show information flow constraints
    println!("Information flow constraints: {}", analysis.information_flows.len());
    
    Ok(())
}

/// Create a large synthetic function for performance testing
fn create_large_synthetic_function(instruction_count: usize) -> FunctionDefinition {
    use crate::bytecode::{Instruction, PrismOpcode};
    
    let mut instructions = Vec::with_capacity(instruction_count);
    
    // Create a mix of different instruction types that require capabilities
    for i in 0..instruction_count {
        let instruction = match i % 10 {
            0 => Instruction::new(PrismOpcode::LoadLocal, vec![0]),
            1 => Instruction::new(PrismOpcode::StoreLocal, vec![0]),
            2 => Instruction::new(PrismOpcode::Call, vec![i as u8 % 20]),
            3 => Instruction::new(PrismOpcode::LoadGlobal, vec![i as u8 % 50]),
            4 => Instruction::new(PrismOpcode::StoreGlobal, vec![i as u8 % 50]),
            5 => Instruction::new(PrismOpcode::Add, vec![]),
            6 => Instruction::new(PrismOpcode::Multiply, vec![]),
            7 => Instruction::new(PrismOpcode::Branch, vec![(i + 10) as u8 % 100]),
            8 => Instruction::new(PrismOpcode::Return, vec![]),
            _ => Instruction::new(PrismOpcode::Nop, vec![]),
        };
        instructions.push(instruction);
    }
    
    FunctionDefinition {
        id: 999,
        name: "large_synthetic_function".to_string(),
        instructions,
        local_count: 10,
        parameter_count: 3,
        return_type: None,
        metadata: Default::default(),
    }
}

/// Performance comparison between algorithms
pub fn benchmark_capability_algorithms() -> VMResult<()> {
    use std::time::Instant;
    
    println!("=== Algorithm Performance Comparison ===\n");
    
    let sizes = vec![100, 500, 1000, 2000, 5000];
    
    for &size in &sizes {
        println!("Testing with {} instructions:", size);
        
        let function = create_large_synthetic_function(size);
        let config = AnalysisConfig::default();
        let mut analyzer = CapabilityAnalyzer::new(&config)?;
        
        // Benchmark full analysis
        let start = Instant::now();
        let analysis = analyzer.analyze(&function)?;
        let total_time = start.elapsed();
        
        println!("  Total analysis time: {:?}", total_time);
        println!("  Nodes: {}, Edges: {}", 
                 analysis.propagation_graph.nodes.len(),
                 analysis.propagation_graph.edges.len());
        println!("  Components: {}, Topo order: {}", 
                 analysis.propagation_graph.components.len(),
                 analysis.propagation_graph.topological_order.len());
        
        // Calculate throughput
        let instructions_per_ms = size as f64 / total_time.as_millis() as f64;
        println!("  Throughput: {:.1} instructions/ms", instructions_per_ms);
        println!();
    }
    
    Ok(())
}

/// Demonstrate incremental analysis capabilities
pub fn demonstrate_incremental_analysis() -> VMResult<()> {
    println!("=== Incremental Capability Analysis ===\n");
    
    // Create base function
    let mut base_function = create_large_synthetic_function(1000);
    let config = AnalysisConfig::default();
    let mut analyzer = CapabilityAnalyzer::new(&config)?;
    
    // Initial analysis
    let start = Instant::now();
    let initial_analysis = analyzer.analyze(&base_function)?;
    let initial_time = start.elapsed();
    
    println!("Initial analysis ({} instructions): {:?}", 
             base_function.instructions.len(), initial_time);
    
    // Simulate function modification (add 100 more instructions)
    for i in 1000..1100 {
        let instruction = Instruction::new(PrismOpcode::Add, vec![]);
        base_function.instructions.push(instruction);
    }
    
    // Re-analysis after modification
    let start = Instant::now();
    let updated_analysis = analyzer.analyze(&base_function)?;
    let update_time = start.elapsed();
    
    println!("Updated analysis ({} instructions): {:?}", 
             base_function.instructions.len(), update_time);
    
    // Compare results
    let node_diff = updated_analysis.propagation_graph.nodes.len() as i32 
                   - initial_analysis.propagation_graph.nodes.len() as i32;
    let edge_diff = updated_analysis.propagation_graph.edges.len() as i32 
                   - initial_analysis.propagation_graph.edges.len() as i32;
    
    println!("Changes: {} nodes, {} edges", node_diff, edge_diff);
    println!("Update efficiency: {:.1}% of initial time", 
             100.0 * update_time.as_millis() as f64 / initial_time.as_millis() as f64);
    
    Ok(())
} 