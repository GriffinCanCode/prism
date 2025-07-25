//! Security Constraint Optimization Demo
//!
//! This example demonstrates the security constraint optimization system,
//! showing how constraint validation is optimized for hot code paths while
//! maintaining security guarantees.

use prism_vm::{
    VMResult, PrismVMError,
    bytecode::{FunctionDefinition, Instruction, PrismBytecode},
    execution::jit::{
        JitCompiler, JitConfig,
        security_constraint_optimizer::{
            SecurityConstraintOptimizer, ConstraintOptimizerConfig, ConstraintType,
            HotPathConfig, CacheConfig, BatchValidatorConfig, SpeculationConfig,
        },
        analysis::{
            AnalysisConfig, StaticAnalyzer,
            hotness::{HotnessAnalyzer, HotnessAnalysis},
            capability_analysis::CapabilityAnalyzer,
            control_flow::CFGAnalyzer,
        },
        capability_guards::{CapabilityGuardGenerator, GuardGeneratorConfig, ExecutionContext},
    },
};
use prism_runtime::authority::capability::{
    CapabilitySet, Capability, Authority, ConstraintSet, ComponentId,
    FileSystemAuthority, FileOperation, TimeConstraint, RateLimit, ResourceLimit,
};
use prism_effects::security::{SecurityLevel, InformationFlow};
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use tracing::{info, debug, warn, Level};

fn main() -> VMResult<()> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    info!("Starting Security Constraint Optimization Demo");

    // Create test functions with different constraint patterns
    let hot_validation_function = create_hot_validation_function()?;
    let mixed_constraint_function = create_mixed_constraint_function()?;
    let batch_suitable_function = create_batch_suitable_function()?;

    // Demo 1: Basic constraint optimization
    demo_basic_constraint_optimization(&hot_validation_function)?;
    
    // Demo 2: Hot path specialization
    demo_hot_path_specialization(&hot_validation_function)?;
    
    // Demo 3: Constraint result caching
    demo_constraint_caching(&mixed_constraint_function)?;
    
    // Demo 4: Batch constraint validation
    demo_batch_validation(&batch_suitable_function)?;
    
    // Demo 5: Speculative constraint validation
    demo_speculative_validation(&hot_validation_function)?;
    
    // Demo 6: Hardware-accelerated validation
    demo_hardware_acceleration(&mixed_constraint_function)?;
    
    // Demo 7: Adaptive optimization
    demo_adaptive_optimization(&hot_validation_function)?;
    
    // Demo 8: Performance comparison
    demo_performance_comparison(&hot_validation_function)?;

    info!("Security Constraint Optimization Demo completed successfully");
    Ok(())
}

/// Demo 1: Basic constraint optimization
fn demo_basic_constraint_optimization(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 1: Basic Constraint Optimization ===");

    // Create constraint optimizer with default configuration
    let config = ConstraintOptimizerConfig::default();
    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Create analysis infrastructure
    let analysis_config = AnalysisConfig::default();
    let mut hotness_analyzer = HotnessAnalyzer::new(&analysis_config)?;
    let mut capability_analyzer = CapabilityAnalyzer::new(&analysis_config)?;
    let mut guard_generator = CapabilityGuardGenerator::new(GuardGeneratorConfig::default())?;

    // Perform analysis
    let hotness_analysis = hotness_analyzer.analyze(function)?;
    let capability_analysis = capability_analyzer.analyze(function)?;
    
    // Create control flow graph
    let mut cfg_analyzer = CFGAnalyzer::new(&analysis_config)?;
    let cfg = cfg_analyzer.analyze(function)?;
    
    // Generate capability guards
    let guard_analysis = guard_generator.analyze_and_generate_guards(
        function,
        &cfg,
        &capability_analysis,
    )?;

    // Optimize constraints for hot paths
    let optimized_validation = optimizer.optimize_for_hot_paths(
        function,
        &hotness_analysis,
        &guard_analysis,
    )?;

    info!("Basic optimization completed:");
    info!("  Hot paths identified: {}", optimized_validation.validation_plan.hot_paths.len());
    info!("  Specialized validators: {}", optimized_validation.validation_plan.specialized_validators.len());
    info!("  Estimated improvement: {:.2}%", optimized_validation.estimated_performance_improvement * 100.0);

    // Demonstrate constraint types identified
    let mut constraint_type_counts = HashMap::new();
    for hot_path in &optimized_validation.validation_plan.hot_paths {
        for constraint_check in &hot_path.constraint_sequence {
            *constraint_type_counts.entry(&constraint_check.constraint_type).or_insert(0) += 1;
        }
    }

    info!("Constraint types in hot paths:");
    for (constraint_type, count) in &constraint_type_counts {
        debug!("  {:?}: {} occurrences", constraint_type, count);
    }

    Ok(())
}

/// Demo 2: Hot path specialization
fn demo_hot_path_specialization(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 2: Hot Path Specialization ===");

    // Create configuration optimized for hot path specialization
    let config = ConstraintOptimizerConfig {
        enable_hot_path_specialization: true,
        hot_path_threshold: 0.05, // Lower threshold to catch more hot paths
        max_specialized_validators: 50,
        enable_hardware_acceleration: true,
        ..ConstraintOptimizerConfig::default()
    };

    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Simulate hot path execution data
    let simulated_hotness = create_simulated_hotness_analysis(function)?;
    let simulated_capability_analysis = create_simulated_capability_analysis(function)?;
    let simulated_guard_analysis = create_simulated_guard_analysis(function)?;

    // Generate optimized validation with specialization focus
    let optimized_validation = optimizer.optimize_for_hot_paths(
        function,
        &simulated_hotness,
        &simulated_guard_analysis,
    )?;

    info!("Hot path specialization results:");
    for (i, hot_path) in optimized_validation.validation_plan.hot_paths.iter().enumerate() {
        info!("  Hot path {}: frequency={:.3}, checks={}, improvement={:.2}%",
              i,
              hot_path.execution_frequency,
              hot_path.constraint_sequence.len(),
              hot_path.performance_improvement.unwrap_or(0.0) * 100.0);

        // Show optimization opportunities
        for opportunity in &hot_path.optimization_opportunities {
            debug!("    Optimization: {:?}", opportunity);
        }
    }

    // Show specialized validators
    info!("Specialized validators generated: {}", 
          optimized_validation.validation_plan.specialized_validators.len());
    
    for (validator_id, validator) in &optimized_validation.validation_plan.specialized_validators {
        debug!("  Validator {}: avg_time={:?}, throughput={:.0} ops/sec",
               validator_id,
               validator.performance.avg_validation_time,
               validator.performance.throughput);
    }

    Ok(())
}

/// Demo 3: Constraint result caching
fn demo_constraint_caching(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 3: Constraint Result Caching ===");

    // Create configuration optimized for caching
    let config = ConstraintOptimizerConfig {
        enable_caching: true,
        cache_size_limit: 5000,
        cache_ttl: Duration::from_secs(600), // 10 minutes
        enable_adaptive_optimization: true,
        ..ConstraintOptimizerConfig::default()
    };

    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Create test capabilities with various constraints
    let mut capabilities = CapabilitySet::new();
    
    // Add file system capability with time constraints
    let file_capability = Capability::new(
        Authority::FileSystem(FileSystemAuthority {
            operations: [FileOperation::Read, FileOperation::Write].into_iter().collect(),
            allowed_paths: vec!["/tmp/*".to_string(), "/var/log/*".to_string()],
        }),
        ConstraintSet {
            time_constraints: vec![
                TimeConstraint::ValidUntil(SystemTime::now() + Duration::from_secs(3600)),
                TimeConstraint::TimeOfDay { start_hour: 9, end_hour: 17 }, // Business hours
            ],
            rate_limits: vec![
                RateLimit::PerSecond(100),
                RateLimit::PerMinute(5000),
            ],
            resource_limits: vec![
                ResourceLimit::Memory(1024 * 1024), // 1MB
                ResourceLimit::CpuTime(Duration::from_millis(100)),
            ],
            context_constraints: Vec::new(),
        },
        Duration::from_secs(3600),
        ComponentId::new(1),
    );
    capabilities.add(file_capability);

    // Simulate multiple validation requests to demonstrate caching
    let execution_context = ExecutionContext {
        function_id: function.id,
        bytecode_offset: 0,
        execution_id: 12345,
        capabilities: capabilities.clone(),
        context_data: HashMap::new(),
    };

    // Create mock hotness and guard analysis
    let hotness_analysis = create_simulated_hotness_analysis(function)?;
    let guard_analysis = create_simulated_guard_analysis(function)?;

    // Generate optimized validation
    let optimized_validation = optimizer.optimize_for_hot_paths(
        function,
        &hotness_analysis,
        &guard_analysis,
    )?;

    info!("Caching optimization results:");
    info!("  Cache configuration: size_limit={}, ttl={:?}",
          5000, Duration::from_secs(600));
    info!("  Partitioning strategy: {:?}", 
          optimized_validation.validation_plan.cache_optimization.partitioning);
    info!("  Prefetching strategy: {:?}", 
          optimized_validation.validation_plan.cache_optimization.prefetching_strategy);

    // Show TTL by constraint type
    info!("  TTL by constraint type:");
    for (constraint_type, ttl) in &optimized_validation.validation_plan.cache_optimization.ttl_by_type {
        debug!("    {:?}: {:?}", constraint_type, ttl);
    }

    // Simulate cache performance
    info!("Simulating cache performance...");
    for i in 0..10 {
        debug!("  Validation round {}: simulating cache hits/misses", i + 1);
        // In a real implementation, this would execute actual validations
        // and show cache hit rates improving over time
    }

    Ok(())
}

/// Demo 4: Batch constraint validation
fn demo_batch_validation(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 4: Batch Constraint Validation ===");

    // Create configuration optimized for batch processing
    let config = ConstraintOptimizerConfig {
        enable_batch_validation: true,
        max_batch_size: 64,
        enable_hardware_acceleration: true, // For SIMD operations
        ..ConstraintOptimizerConfig::default()
    };

    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Create analysis with multiple similar constraints suitable for batching
    let hotness_analysis = create_batch_suitable_hotness_analysis(function)?;
    let guard_analysis = create_batch_suitable_guard_analysis(function)?;

    // Generate optimized validation
    let optimized_validation = optimizer.optimize_for_hot_paths(
        function,
        &hotness_analysis,
        &guard_analysis,
    )?;

    info!("Batch validation optimization results:");
    info!("  Batch configuration: max_size={}, strategy={:?}",
          optimized_validation.validation_plan.batch_configuration.batch_sizes.values().max().unwrap_or(&0),
          optimized_validation.validation_plan.batch_configuration.strategy);

    // Show optimal batch sizes by constraint type
    info!("  Optimal batch sizes by constraint type:");
    for (constraint_type, batch_size) in &optimized_validation.validation_plan.batch_configuration.batch_sizes {
        debug!("    {:?}: {} constraints per batch", constraint_type, batch_size);
    }

    // Show timeout configuration
    info!("  Batch timeouts:");
    for (constraint_type, timeout) in &optimized_validation.validation_plan.batch_configuration.timeouts {
        debug!("    {:?}: {:?}", constraint_type, timeout);
    }

    // Simulate batch processing performance
    info!("Simulating batch processing performance:");
    let constraint_types = vec![
        ConstraintType::Time { subtype: crate::execution::jit::security_constraint_optimizer::TimeConstraintType::ValidUntil },
        ConstraintType::RateLimit { limit_type: crate::execution::jit::security_constraint_optimizer::RateLimitType::PerSecond },
        ConstraintType::Resource { resource_type: crate::execution::jit::security_constraint_optimizer::ResourceType::Memory },
    ];

    for constraint_type in &constraint_types {
        let batch_size = optimized_validation.validation_plan.batch_configuration.batch_sizes
            .get(constraint_type).cloned().unwrap_or(16);
        
        // Simulate performance improvement from batching
        let individual_time = Duration::from_micros(100); // 100μs per constraint
        let batch_time = Duration::from_micros(batch_size as u64 * 20); // 20μs per constraint in batch
        let improvement = (individual_time.as_nanos() as f64 * batch_size as f64 - batch_time.as_nanos() as f64) 
                         / (individual_time.as_nanos() as f64 * batch_size as f64) * 100.0;
        
        info!("    {:?}: {:.1}% performance improvement with batch size {}",
              constraint_type, improvement, batch_size);
    }

    Ok(())
}

/// Demo 5: Speculative constraint validation
fn demo_speculative_validation(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 5: Speculative Constraint Validation ===");

    // Create configuration optimized for speculation
    let config = ConstraintOptimizerConfig {
        enable_speculative_validation: true,
        speculation_threshold: 0.8, // High confidence threshold
        enable_adaptive_optimization: true,
        ..ConstraintOptimizerConfig::default()
    };

    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Create analysis with predictable constraint patterns
    let hotness_analysis = create_predictable_hotness_analysis(function)?;
    let guard_analysis = create_predictable_guard_analysis(function)?;

    // Generate optimized validation
    let optimized_validation = optimizer.optimize_for_hot_paths(
        function,
        &hotness_analysis,
        &guard_analysis,
    )?;

    info!("Speculative validation optimization results:");
    info!("  Speculation models: {}", 
          optimized_validation.validation_plan.speculation_configuration.models.len());

    // Show speculation models by constraint type
    info!("  Speculation models by constraint type:");
    for (constraint_type, model) in &optimized_validation.validation_plan.speculation_configuration.models {
        debug!("    {:?}: model_type={:?}, accuracy={:.2}%",
               constraint_type, model.model_type, model.accuracy * 100.0);
    }

    // Show confidence thresholds
    info!("  Confidence thresholds:");
    for (constraint_type, threshold) in &optimized_validation.validation_plan.speculation_configuration.confidence_thresholds {
        debug!("    {:?}: {:.2}", constraint_type, threshold);
    }

    // Show fallback strategies
    info!("  Fallback strategies:");
    for (constraint_type, strategy) in &optimized_validation.validation_plan.speculation_configuration.fallback_strategies {
        debug!("    {:?}: {:?}", constraint_type, strategy);
    }

    // Simulate speculation performance
    info!("Simulating speculation performance:");
    let speculation_scenarios = vec![
        ("High confidence scenario", 0.95, true),
        ("Medium confidence scenario", 0.75, false),
        ("Low confidence scenario", 0.45, false),
    ];

    for (scenario_name, confidence, should_speculate) in &speculation_scenarios {
        let action = if *should_speculate { "SPECULATE" } else { "FALLBACK" };
        info!("    {}: confidence={:.2} -> {}", scenario_name, confidence, action);
        
        if *should_speculate {
            // Simulate speculation time savings
            let normal_time = Duration::from_micros(200);
            let speculation_time = Duration::from_micros(50);
            let savings = ((normal_time.as_nanos() - speculation_time.as_nanos()) as f64 
                          / normal_time.as_nanos() as f64) * 100.0;
            debug!("      Time savings: {:.1}%", savings);
        }
    }

    Ok(())
}

/// Demo 6: Hardware-accelerated validation
fn demo_hardware_acceleration(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 6: Hardware-Accelerated Validation ===");

    // Create configuration with hardware acceleration enabled
    let config = ConstraintOptimizerConfig {
        enable_hardware_acceleration: true,
        enable_batch_validation: true, // Often needed for hardware acceleration
        ..ConstraintOptimizerConfig::default()
    };

    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Create analysis suitable for hardware acceleration
    let hotness_analysis = create_simulated_hotness_analysis(function)?;
    let guard_analysis = create_simulated_guard_analysis(function)?;

    // Generate optimized validation
    let optimized_validation = optimizer.optimize_for_hot_paths(
        function,
        &hotness_analysis,
        &guard_analysis,
    )?;

    info!("Hardware acceleration optimization results:");
    info!("  Enabled features: {}", 
          optimized_validation.validation_plan.hardware_configuration.enabled_features.len());

    // Show enabled hardware features
    info!("  Hardware features detected and enabled:");
    for feature in &optimized_validation.validation_plan.hardware_configuration.enabled_features {
        debug!("    {:?}", feature);
    }

    // Show utilization strategies
    info!("  Hardware utilization strategies:");
    for (feature, strategy) in &optimized_validation.validation_plan.hardware_configuration.utilization_strategies {
        debug!("    {:?}: {:?}", feature, strategy);
    }

    // Show activation thresholds
    info!("  Activation thresholds:");
    for (feature, threshold) in &optimized_validation.validation_plan.hardware_configuration.activation_thresholds {
        debug!("    {:?}: {:.2}", feature, threshold);
    }

    // Simulate hardware acceleration performance
    info!("Simulating hardware acceleration performance:");
    let hardware_scenarios = vec![
        ("SIMD batch processing", 4.5),
        ("AES-NI cryptographic operations", 8.2),
        ("Vector instructions", 3.1),
        ("Hardware random number generation", 2.8),
    ];

    for (scenario_name, speedup_factor) in &hardware_scenarios {
        info!("    {}: {:.1}x speedup", scenario_name, speedup_factor);
    }

    Ok(())
}

/// Demo 7: Adaptive optimization
fn demo_adaptive_optimization(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 7: Adaptive Optimization ===");

    // Create configuration with adaptive optimization enabled
    let config = ConstraintOptimizerConfig {
        enable_adaptive_optimization: true,
        metrics_interval: Duration::from_secs(5), // Frequent adaptation
        enable_caching: true,
        enable_batch_validation: true,
        enable_speculative_validation: true,
        ..ConstraintOptimizerConfig::default()
    };

    let mut optimizer = SecurityConstraintOptimizer::new(config)?;

    // Simulate multiple optimization rounds with changing conditions
    for round in 1..=5 {
        info!("Adaptive optimization round {}", round);

        // Create varying analysis conditions
        let hotness_analysis = create_varying_hotness_analysis(function, round)?;
        let guard_analysis = create_varying_guard_analysis(function, round)?;

        // Generate optimized validation
        let optimized_validation = optimizer.optimize_for_hot_paths(
            function,
            &hotness_analysis,
            &guard_analysis,
        )?;

        info!("  Round {} results:", round);
        info!("    Hot paths: {}", optimized_validation.validation_plan.hot_paths.len());
        info!("    Specialized validators: {}", optimized_validation.validation_plan.specialized_validators.len());
        info!("    Estimated improvement: {:.2}%", 
              optimized_validation.estimated_performance_improvement * 100.0);

        // Show adaptation decisions
        let cache_enabled = !optimized_validation.validation_plan.cache_optimization.ttl_by_type.is_empty();
        let batch_enabled = !optimized_validation.validation_plan.batch_configuration.batch_sizes.is_empty();
        let speculation_enabled = !optimized_validation.validation_plan.speculation_configuration.models.is_empty();

        debug!("    Adaptations: cache={}, batch={}, speculation={}", 
               cache_enabled, batch_enabled, speculation_enabled);

        // Simulate changing performance characteristics
        match round {
            1 => info!("    Baseline performance established"),
            2 => info!("    High cache hit rate observed - increasing cache usage"),
            3 => info!("    Batch processing showing good results - expanding batch sizes"),
            4 => info!("    Speculation accuracy improved - lowering confidence thresholds"),
            5 => info!("    Optimal configuration reached - stabilizing"),
            _ => {}
        }
    }

    Ok(())
}

/// Demo 8: Performance comparison
fn demo_performance_comparison(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 8: Performance Comparison ===");

    // Create different optimization configurations for comparison
    let configs = vec![
        ("Baseline (no optimization)", ConstraintOptimizerConfig {
            enable_hot_path_specialization: false,
            enable_caching: false,
            enable_batch_validation: false,
            enable_speculative_validation: false,
            enable_hardware_acceleration: false,
            ..ConstraintOptimizerConfig::default()
        }),
        ("Cache only", ConstraintOptimizerConfig {
            enable_caching: true,
            ..ConstraintOptimizerConfig::default()
        }),
        ("Specialization + Cache", ConstraintOptimizerConfig {
            enable_hot_path_specialization: true,
            enable_caching: true,
            ..ConstraintOptimizerConfig::default()
        }),
        ("Full optimization", ConstraintOptimizerConfig::default()),
    ];

    let hotness_analysis = create_simulated_hotness_analysis(function)?;
    let guard_analysis = create_simulated_guard_analysis(function)?;

    info!("Performance comparison results:");
    info!("  Configuration                | Improvement | Hot Paths | Validators | Cache Hit Rate");
    info!("  ----------------------------|-------------|-----------|------------|---------------");

    for (config_name, config) in &configs {
        let mut optimizer = SecurityConstraintOptimizer::new(config.clone())?;
        
        let optimized_validation = optimizer.optimize_for_hot_paths(
            function,
            &hotness_analysis,
            &guard_analysis,
        )?;

        let improvement = optimized_validation.estimated_performance_improvement * 100.0;
        let hot_paths = optimized_validation.validation_plan.hot_paths.len();
        let validators = optimized_validation.validation_plan.specialized_validators.len();
        let cache_hit_rate = if config.enable_caching { 85.0 } else { 0.0 }; // Simulated

        info!("  {:28} | {:9.1}% | {:9} | {:10} | {:12.1}%",
              config_name, improvement, hot_paths, validators, cache_hit_rate);
    }

    // Show detailed breakdown of optimization techniques
    info!("\nOptimization technique effectiveness:");
    let techniques = vec![
        ("Hot path specialization", 25.0),
        ("Constraint result caching", 40.0),
        ("Batch validation", 15.0),
        ("Speculative validation", 12.0),
        ("Hardware acceleration", 18.0),
    ];

    for (technique, improvement) in &techniques {
        info!("  {}: {:.1}% improvement", technique, improvement);
    }

    Ok(())
}

// Helper functions to create test data

fn create_hot_validation_function() -> VMResult<FunctionDefinition> {
    use crate::bytecode::instructions::PrismOpcode;

    let mut instructions = Vec::new();
    
    // Hot loop with frequent capability checks
    for i in 0..20 {
        instructions.push(Instruction {
            opcode: PrismOpcode::CAP_CHECK(i % 3), // Rotate through different capability checks
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            has_side_effects: false,
        });
        
        instructions.push(Instruction {
            opcode: PrismOpcode::LOAD_CONST(i),
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            has_side_effects: false,
        });
        
        instructions.push(Instruction {
            opcode: PrismOpcode::ADD,
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            has_side_effects: false,
        });
    }
    
    instructions.push(Instruction {
        opcode: PrismOpcode::RETURN_VALUE,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });

    Ok(FunctionDefinition {
        id: 1,
        name: "hot_validation_function".to_string(),
        instructions,
        local_count: 5,
        parameter_count: 1,
        capabilities: Vec::new(),
        declared_effects: Vec::new(),
    })
}

fn create_mixed_constraint_function() -> VMResult<FunctionDefinition> {
    use crate::bytecode::instructions::PrismOpcode;

    let mut instructions = Vec::new();
    
    // Mix of different constraint types
    instructions.push(Instruction {
        opcode: PrismOpcode::CAP_CHECK(0), // Capability presence
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });
    
    instructions.push(Instruction {
        opcode: PrismOpcode::IO_READ(1024), // Resource constraint
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    instructions.push(Instruction {
        opcode: PrismOpcode::CALL(100), // Rate limit constraint
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    instructions.push(Instruction {
        opcode: PrismOpcode::RETURN_VALUE,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });

    Ok(FunctionDefinition {
        id: 2,
        name: "mixed_constraint_function".to_string(),
        instructions,
        local_count: 3,
        parameter_count: 0,
        capabilities: Vec::new(),
        declared_effects: Vec::new(),
    })
}

fn create_batch_suitable_function() -> VMResult<FunctionDefinition> {
    use crate::bytecode::instructions::PrismOpcode;

    let mut instructions = Vec::new();
    
    // Many similar constraint checks suitable for batching
    for i in 0..32 {
        instructions.push(Instruction {
            opcode: PrismOpcode::CAP_CHECK(0), // Same capability check repeatedly
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            has_side_effects: false,
        });
        
        instructions.push(Instruction {
            opcode: PrismOpcode::LOAD_CONST(i),
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            has_side_effects: false,
        });
    }
    
    instructions.push(Instruction {
        opcode: PrismOpcode::RETURN_VALUE,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });

    Ok(FunctionDefinition {
        id: 3,
        name: "batch_suitable_function".to_string(),
        instructions,
        local_count: 2,
        parameter_count: 0,
        capabilities: Vec::new(),
        declared_effects: Vec::new(),
    })
}

// Helper functions to create simulated analysis data
// (These would be replaced with actual analysis results in a real implementation)

fn create_simulated_hotness_analysis(function: &FunctionDefinition) -> VMResult<HotnessAnalysis> {
    use crate::execution::jit::analysis::hotness::*;
    
    Ok(HotnessAnalysis {
        function_id: function.id,
        hot_spots: vec![
            HotSpot {
                location: HotSpotLocation::Instruction { offset: 0 },
                hotness_score: 0.8,
                execution_count: 10000,
                avg_execution_time: Duration::from_micros(50),
                optimization_potential: 0.7,
                hot_spot_type: HotSpotType::Branch { 
                    prediction_accuracy: 0.9, 
                    taken_percentage: 0.8 
                },
                confidence: 0.9,
                related_hot_spots: Vec::new(),
            }
        ],
        cold_regions: Vec::new(),
        profile_data: ProfileData::default(),
        execution_frequency: ExecutionFrequency {
            total_executions: 50000,
            hot_threshold: 0.1,
            cold_threshold: 0.01,
            ..ExecutionFrequency::default()
        },
        optimization_opportunities: Vec::new(),
        compilation_tier_recommendations: Vec::new(),
        performance_characteristics: PerformanceCharacteristics::default(),
    })
}

fn create_simulated_capability_analysis(function: &FunctionDefinition) -> VMResult<crate::execution::jit::analysis::capability_analysis::CapabilityAnalysis> {
    use crate::execution::jit::analysis::capability_analysis::*;
    
    Ok(CapabilityAnalysis {
        function_id: function.id,
        capability_flow: CapabilityFlow {
            flow_graph: FlowGraph { nodes: Vec::new(), edges: Vec::new() },
            entry_capabilities: std::collections::HashMap::new(),
            exit_capabilities: std::collections::HashMap::new(),
            capability_propagation: std::collections::HashMap::new(),
        },
        security_boundaries: Vec::new(),
        capability_requirements: std::collections::HashMap::new(),
        security_implications: SecurityImplications {
            risk_level: SecurityRiskLevel::Medium,
            identified_risks: Vec::new(),
            mitigation_strategies: Vec::new(),
        },
        optimization_opportunities: Vec::new(),
    })
}

fn create_simulated_guard_analysis(function: &FunctionDefinition) -> VMResult<crate::execution::jit::capability_guards::CapabilityGuardAnalysis> {
    use crate::execution::jit::capability_guards::*;
    
    Ok(CapabilityGuardAnalysis {
        function_id: function.id,
        required_guards: vec![
            RequiredGuard {
                guard_id: 1,
                guard_type: GuardType::CapabilityPresence,
                bytecode_location: 0,
                required_capability: Capability::new(
                    prism_runtime::authority::capability::Authority::FileSystem(
                        prism_runtime::authority::capability::FileSystemAuthority {
                            operations: [prism_runtime::authority::capability::FileOperation::Read].into_iter().collect(),
                            allowed_paths: vec!["/tmp/*".to_string()],
                        }
                    ),
                    prism_runtime::authority::capability::ConstraintSet::new(),
                    Duration::from_secs(3600),
                    prism_runtime::authority::capability::ComponentId::new(1),
                ),
                criticality: GuardCriticality::Medium,
                execution_frequency: 0.8,
                guard_params: GuardParameters {
                    timeout: Some(Duration::from_millis(10)),
                    caching_strategy: CachingStrategy::TimeBased { duration: Duration::from_secs(60) },
                    validation_depth: 1,
                    custom_params: std::collections::HashMap::new(),
                },
                failure_strategy: GuardFailureStrategy::LogAndContinue {
                    log_level: LogLevel::Warn,
                    warning: "Guard validation failed".to_string(),
                },
            }
        ],
        guard_placement_plan: GuardPlacementPlan {
            placements: Vec::new(),
            guard_clusters: Vec::new(),
            optimizations: Vec::new(),
            estimated_overhead: 5.0,
        },
        security_risk_assessment: SecurityRiskAssessment {
            overall_risk: SecurityRiskLevel::Medium,
            risk_factors: Vec::new(),
            mitigations: Vec::new(),
            residual_risks: Vec::new(),
            confidence: 0.8,
        },
        performance_impact: PerformanceImpactAnalysis {
            total_overhead_percent: 5.0,
            per_guard_overhead: std::collections::HashMap::new(),
            hot_path_impact: Vec::new(),
            optimization_opportunities: Vec::new(),
            benchmark_results: None,
        },
        deoptimization_strategy: DeoptimizationStrategy {
            triggers: Vec::new(),
            targets: Vec::new(),
            state_transfer: StateTransferStrategy::FullReconstruction,
            recompilation_policy: RecompilationPolicy {
                allow_recompilation: true,
                recompilation_conditions: Vec::new(),
                max_attempts: 3,
                backoff_strategy: BackoffStrategy::Exponential {
                    base: 2.0,
                    max_delay: Duration::from_secs(60),
                },
            },
        },
        guard_dependencies: GuardDependencyGraph {
            nodes: std::collections::HashMap::new(),
            edges: Vec::new(),
            execution_order: Vec::new(),
            circular_dependencies: Vec::new(),
        },
    })
}

// Additional helper functions for different demo scenarios
fn create_batch_suitable_hotness_analysis(function: &FunctionDefinition) -> VMResult<HotnessAnalysis> {
    // Similar to simulated but with patterns suitable for batching
    create_simulated_hotness_analysis(function)
}

fn create_batch_suitable_guard_analysis(function: &FunctionDefinition) -> VMResult<crate::execution::jit::capability_guards::CapabilityGuardAnalysis> {
    // Similar to simulated but with many similar guards
    create_simulated_guard_analysis(function)
}

fn create_predictable_hotness_analysis(function: &FunctionDefinition) -> VMResult<HotnessAnalysis> {
    // Similar to simulated but with predictable patterns
    create_simulated_hotness_analysis(function)
}

fn create_predictable_guard_analysis(function: &FunctionDefinition) -> VMResult<crate::execution::jit::capability_guards::CapabilityGuardAnalysis> {
    // Similar to simulated but with predictable guard patterns
    create_simulated_guard_analysis(function)
}

fn create_varying_hotness_analysis(function: &FunctionDefinition, round: i32) -> VMResult<HotnessAnalysis> {
    // Create analysis that varies by round to simulate changing conditions
    let mut analysis = create_simulated_hotness_analysis(function)?;
    
    // Vary execution frequency based on round
    analysis.execution_frequency.total_executions = 50000 + (round as u64 * 10000);
    analysis.execution_frequency.hot_threshold = 0.1 - (round as f64 * 0.01);
    
    Ok(analysis)
}

fn create_varying_guard_analysis(function: &FunctionDefinition, round: i32) -> VMResult<crate::execution::jit::capability_guards::CapabilityGuardAnalysis> {
    // Create guard analysis that varies by round
    let mut analysis = create_simulated_guard_analysis(function)?;
    
    // Vary guard execution frequency based on round
    for guard in &mut analysis.required_guards {
        guard.execution_frequency = 0.8 + (round as f64 * 0.05);
    }
    
    Ok(analysis)
} 