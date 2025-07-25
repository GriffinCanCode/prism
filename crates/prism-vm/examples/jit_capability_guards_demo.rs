//! JIT Capability Guards Demo
//!
//! This example demonstrates the JIT capability guard system in action,
//! showing how runtime checks are generated and executed to validate
//! capabilities before executing optimized code.

use prism_vm::{
    VMResult, PrismVMError,
    bytecode::{FunctionDefinition, Instruction, PrismBytecode},
    execution::jit::{
        JitCompiler, JitConfig,
        capability_guards::{
            CapabilityGuardGenerator, GuardGeneratorConfig, GuardType,
            GuardPlacementStrategy, ExecutionContext,
        },
        analysis::{
            AnalysisConfig, StaticAnalyzer,
            capability_analysis::CapabilityAnalyzer,
            control_flow::CFGAnalyzer,
        },
    },
};
use prism_runtime::authority::capability::{
    CapabilitySet, Capability, Authority, 
    FileSystemAuthority, FileOperation,
    ConstraintSet, ComponentId,
};
use prism_effects::security::{SecurityLevel, InformationFlow};
use std::time::Duration;
use tracing::{info, debug, warn, Level};

fn main() -> VMResult<()> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    info!("Starting JIT Capability Guards Demo");

    // Create a function that requires file system access
    let file_access_function = create_file_access_function()?;
    
    // Create a function that requires network access
    let network_function = create_network_function()?;
    
    // Create a function with mixed capabilities
    let mixed_capability_function = create_mixed_capability_function()?;

    // Demo 1: Basic capability guard generation
    demo_basic_guard_generation(&file_access_function)?;
    
    // Demo 2: Performance-aware guard placement
    demo_performance_aware_placement(&network_function)?;
    
    // Demo 3: Security-first guard placement
    demo_security_first_placement(&mixed_capability_function)?;
    
    // Demo 4: Adaptive guard placement
    demo_adaptive_placement(&file_access_function)?;
    
    // Demo 5: Guard execution and failure handling
    demo_guard_execution(&file_access_function)?;
    
    // Demo 6: Deoptimization on guard failure
    demo_deoptimization(&network_function)?;
    
    // Demo 7: Guard optimization techniques
    demo_guard_optimization(&mixed_capability_function)?;
    
    // Demo 8: Audit logging and security analysis
    demo_audit_logging(&file_access_function)?;

    info!("JIT Capability Guards Demo completed successfully");
    Ok(())
}

/// Demo 1: Basic capability guard generation
fn demo_basic_guard_generation(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 1: Basic Capability Guard Generation ===");

    // Create guard generator with default configuration
    let config = GuardGeneratorConfig::default();
    let mut guard_generator = CapabilityGuardGenerator::new(config)?;

    // Create analysis infrastructure
    let analysis_config = AnalysisConfig::default();
    let mut cfg_analyzer = CFGAnalyzer::new(&analysis_config)?;
    let mut capability_analyzer = CapabilityAnalyzer::new(&analysis_config)?;

    // Perform analysis
    let cfg = cfg_analyzer.analyze(function)?;
    let capability_analysis = capability_analyzer.analyze(function)?;

    // Generate guard analysis
    let guard_analysis = guard_generator.analyze_and_generate_guards(
        function,
        &cfg,
        &capability_analysis,
    )?;

    info!("Generated {} guards for function '{}'", 
          guard_analysis.required_guards.len(), 
          function.name);

    for (i, guard) in guard_analysis.required_guards.iter().enumerate() {
        debug!("Guard {}: {:?} at location {} (criticality: {:?})",
               i, guard.guard_type, guard.bytecode_location, guard.criticality);
    }

    info!("Estimated performance overhead: {:.2}%", 
          guard_analysis.performance_impact.total_overhead_percent);
    info!("Security risk level: {:?}", 
          guard_analysis.security_risk_assessment.overall_risk);

    Ok(())
}

/// Demo 2: Performance-aware guard placement
fn demo_performance_aware_placement(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 2: Performance-Aware Guard Placement ===");

    // Create configuration optimized for performance
    let config = GuardGeneratorConfig {
        enable_guards: true,
        placement_strategy: GuardPlacementStrategy::Minimal,
        security_level: 0.3, // Lower security for better performance
        enable_optimization: true,
        max_overhead_percent: 5.0, // Strict performance constraint
        enable_deoptimization: true,
        enable_audit_logging: false, // Disable for performance
        guard_timeout: Duration::from_millis(1),
        ..GuardGeneratorConfig::default()
    };

    let mut guard_generator = CapabilityGuardGenerator::new(config)?;

    // Perform analysis with performance focus
    let analysis_config = AnalysisConfig::default();
    let mut cfg_analyzer = CFGAnalyzer::new(&analysis_config)?;
    let mut capability_analyzer = CapabilityAnalyzer::new(&analysis_config)?;

    let cfg = cfg_analyzer.analyze(function)?;
    let capability_analysis = capability_analyzer.analyze(function)?;

    let guard_analysis = guard_generator.analyze_and_generate_guards(
        function,
        &cfg,
        &capability_analysis,
    )?;

    info!("Performance-optimized guards: {}", guard_analysis.required_guards.len());
    info!("Estimated overhead: {:.2}%", 
          guard_analysis.performance_impact.total_overhead_percent);

    // Show guard clustering for efficiency
    for (i, cluster) in guard_analysis.guard_placement_plan.guard_clusters.iter().enumerate() {
        debug!("Guard cluster {}: {} guards with {:?} execution strategy",
               i, cluster.guards.len(), cluster.execution_strategy);
    }

    // Show optimization opportunities
    for optimization in &guard_analysis.performance_impact.optimization_opportunities {
        debug!("Optimization opportunity: {} (improvement: {:.2}%)",
               optimization.description, optimization.expected_improvement * 100.0);
    }

    Ok(())
}

/// Demo 3: Security-first guard placement
fn demo_security_first_placement(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 3: Security-First Guard Placement ===");

    // Create configuration optimized for security
    let config = GuardGeneratorConfig {
        enable_guards: true,
        placement_strategy: GuardPlacementStrategy::Comprehensive,
        security_level: 0.9, // High security
        enable_optimization: false, // Disable optimizations that might reduce security
        max_overhead_percent: 50.0, // Allow high overhead for security
        enable_deoptimization: true,
        enable_audit_logging: true,
        guard_timeout: Duration::from_millis(100), // Allow longer validation times
        ..GuardGeneratorConfig::default()
    };

    let mut guard_generator = CapabilityGuardGenerator::new(config)?;

    // Perform analysis with security focus
    let analysis_config = AnalysisConfig::default();
    let mut cfg_analyzer = CFGAnalyzer::new(&analysis_config)?;
    let mut capability_analyzer = CapabilityAnalyzer::new(&analysis_config)?;

    let cfg = cfg_analyzer.analyze(function)?;
    let capability_analysis = capability_analyzer.analyze(function)?;

    let guard_analysis = guard_generator.analyze_and_generate_guards(
        function,
        &cfg,
        &capability_analysis,
    )?;

    info!("Security-optimized guards: {}", guard_analysis.required_guards.len());
    info!("Security risk level: {:?}", guard_analysis.security_risk_assessment.overall_risk);

    // Show risk factors and mitigations
    for (i, risk_factor) in guard_analysis.security_risk_assessment.risk_factors.iter().enumerate() {
        debug!("Risk factor {}: {} (level: {:?})",
               i, risk_factor.description, risk_factor.risk_level);
    }

    for (i, mitigation) in guard_analysis.security_risk_assessment.mitigations.iter().enumerate() {
        debug!("Mitigation {}: {} (effectiveness: {:.2}%)",
               i, mitigation.description, mitigation.effectiveness * 100.0);
    }

    Ok(())
}

/// Demo 4: Adaptive guard placement
fn demo_adaptive_placement(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 4: Adaptive Guard Placement ===");

    use crate::execution::jit::capability_guards::{AdaptationParams};

    // Create adaptive configuration
    let config = GuardGeneratorConfig {
        enable_guards: true,
        placement_strategy: GuardPlacementStrategy::Adaptive {
            initial_strategy: Box::new(GuardPlacementStrategy::Balanced),
            adaptation_params: AdaptationParams {
                min_samples: 100,
                performance_threshold: 0.1, // 10% performance degradation threshold
                security_threshold: 1, // Any security violation triggers adaptation
                adaptation_interval: Duration::from_secs(60),
            },
        },
        security_level: 0.6,
        enable_optimization: true,
        max_overhead_percent: 15.0,
        enable_deoptimization: true,
        enable_audit_logging: true,
        guard_timeout: Duration::from_millis(10),
        ..GuardGeneratorConfig::default()
    };

    let mut guard_generator = CapabilityGuardGenerator::new(config)?;

    // Simulate multiple compilation rounds with adaptation
    for round in 1..=3 {
        info!("Adaptive compilation round {}", round);

        let analysis_config = AnalysisConfig::default();
        let mut cfg_analyzer = CFGAnalyzer::new(&analysis_config)?;
        let mut capability_analyzer = CapabilityAnalyzer::new(&analysis_config)?;

        let cfg = cfg_analyzer.analyze(function)?;
        let capability_analysis = capability_analyzer.analyze(function)?;

        let guard_analysis = guard_generator.analyze_and_generate_guards(
            function,
            &cfg,
            &capability_analysis,
        )?;

        info!("Round {}: {} guards, {:.2}% overhead",
              round, guard_analysis.required_guards.len(),
              guard_analysis.performance_impact.total_overhead_percent);

        // Simulate runtime feedback that would trigger adaptation
        if round == 2 {
            info!("Simulating performance degradation - triggering adaptation");
            // In a real implementation, this would come from runtime monitoring
        }
    }

    Ok(())
}

/// Demo 5: Guard execution and failure handling
fn demo_guard_execution(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 5: Guard Execution and Failure Handling ===");

    let config = GuardGeneratorConfig::default();
    let guard_generator = CapabilityGuardGenerator::new(config)?;

    // Create test capabilities - one valid, one invalid
    let mut valid_capabilities = CapabilitySet::new();
    let file_capability = Capability::new(
        Authority::FileSystem(FileSystemAuthority {
            operations: [FileOperation::Read].into_iter().collect(),
            allowed_paths: vec!["/tmp/*".to_string()],
        }),
        ConstraintSet::new(),
        Duration::from_secs(3600),
        ComponentId::new(1),
    );
    valid_capabilities.add(file_capability);

    let invalid_capabilities = CapabilitySet::new(); // Empty set

    // Create execution context
    let execution_context = ExecutionContext {
        function_id: function.id,
        bytecode_offset: 0,
        execution_id: 12345,
        capabilities: valid_capabilities.clone(),
        context_data: std::collections::HashMap::new(),
    };

    // Create a test guard
    use crate::execution::jit::capability_guards::{
        RequiredGuard, GuardCriticality, GuardParameters, GuardFailureStrategy,
        CachingStrategy, FailureAction, LogLevel,
    };

    let test_guard = RequiredGuard {
        guard_id: 1,
        guard_type: GuardType::CapabilityPresence,
        bytecode_location: 0,
        required_capability: file_capability.clone(),
        criticality: GuardCriticality::High,
        execution_frequency: 0.8,
        guard_params: GuardParameters {
            timeout: Some(Duration::from_millis(10)),
            caching_strategy: CachingStrategy::TimeBased { 
                duration: Duration::from_secs(60) 
            },
            validation_depth: 1,
            custom_params: std::collections::HashMap::new(),
        },
        failure_strategy: GuardFailureStrategy::LogAndContinue {
            log_level: LogLevel::Warn,
            warning: "Capability validation failed".to_string(),
        },
    };

    // Test successful guard execution
    info!("Testing successful guard execution...");
    let result = guard_generator.execute_guard(
        &test_guard,
        &valid_capabilities,
        &execution_context,
    )?;

    info!("Guard execution result: passed={}, time={:?}, cached={}",
          result.passed, result.execution_time, result.was_cached);

    // Test failed guard execution
    info!("Testing failed guard execution...");
    let result = guard_generator.execute_guard(
        &test_guard,
        &invalid_capabilities,
        &execution_context,
    )?;

    info!("Guard execution result: passed={}, time={:?}",
          result.passed, result.execution_time);

    if !result.passed {
        let failure_response = guard_generator.handle_guard_failure(
            &test_guard,
            "Required capability not present",
            &execution_context,
        )?;

        match failure_response {
            crate::execution::jit::capability_guards::GuardFailureResponse::Continue { warning } => {
                warn!("Guard failure handled: {}", warning);
            }
            _ => {
                info!("Guard failure response: {:?}", failure_response);
            }
        }
    }

    Ok(())
}

/// Demo 6: Deoptimization on guard failure
fn demo_deoptimization(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 6: Deoptimization on Guard Failure ===");

    // Create configuration with deoptimization enabled
    let config = GuardGeneratorConfig {
        enable_deoptimization: true,
        ..GuardGeneratorConfig::default()
    };

    let guard_generator = CapabilityGuardGenerator::new(config)?;

    // Create a guard that will trigger deoptimization on failure
    use crate::execution::jit::capability_guards::{
        RequiredGuard, GuardCriticality, GuardParameters, GuardFailureStrategy,
        CachingStrategy,
    };

    let deopt_guard = RequiredGuard {
        guard_id: 2,
        guard_type: GuardType::SecurityBoundaryCheck,
        bytecode_location: 10,
        required_capability: Capability::new(
            Authority::FileSystem(FileSystemAuthority {
                operations: [FileOperation::Write].into_iter().collect(),
                allowed_paths: vec!["/secure/*".to_string()],
            }),
            ConstraintSet::new(),
            Duration::from_secs(3600),
            ComponentId::new(1),
        ),
        criticality: GuardCriticality::Critical,
        execution_frequency: 0.1, // Rarely executed, but critical
        guard_params: GuardParameters {
            timeout: Some(Duration::from_millis(5)),
            caching_strategy: CachingStrategy::None, // Always check for security
            validation_depth: 2,
            custom_params: std::collections::HashMap::new(),
        },
        failure_strategy: GuardFailureStrategy::Deoptimize {
            reason: "Security boundary violation detected".to_string(),
            allow_recompilation: false, // Don't recompile due to security concern
        },
    };

    // Create execution context
    let execution_context = ExecutionContext {
        function_id: function.id,
        bytecode_offset: 10,
        execution_id: 67890,
        capabilities: CapabilitySet::new(), // No capabilities - will fail
        context_data: std::collections::HashMap::new(),
    };

    info!("Testing deoptimization trigger...");

    // Execute guard (will fail)
    let result = guard_generator.execute_guard(
        &deopt_guard,
        &CapabilitySet::new(),
        &execution_context,
    )?;

    if !result.passed {
        let failure_response = guard_generator.handle_guard_failure(
            &deopt_guard,
            "Security boundary check failed",
            &execution_context,
        )?;

        match failure_response {
            crate::execution::jit::capability_guards::GuardFailureResponse::Deoptimized { reason, bytecode_offset } => {
                info!("Deoptimization triggered: {} at bytecode offset {}", reason, bytecode_offset);
                info!("Execution will continue in interpreter mode");
            }
            _ => {
                warn!("Expected deoptimization but got: {:?}", failure_response);
            }
        }
    }

    Ok(())
}

/// Demo 7: Guard optimization techniques
fn demo_guard_optimization(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 7: Guard Optimization Techniques ===");

    // Create configuration with optimization enabled
    let config = GuardGeneratorConfig {
        enable_optimization: true,
        placement_strategy: GuardPlacementStrategy::Balanced,
        ..GuardGeneratorConfig::default()
    };

    let mut guard_generator = CapabilityGuardGenerator::new(config)?;

    // Perform analysis to see optimization opportunities
    let analysis_config = AnalysisConfig::default();
    let mut cfg_analyzer = CFGAnalyzer::new(&analysis_config)?;
    let mut capability_analyzer = CapabilityAnalyzer::new(&analysis_config)?;

    let cfg = cfg_analyzer.analyze(function)?;
    let capability_analysis = capability_analyzer.analyze(function)?;

    let guard_analysis = guard_generator.analyze_and_generate_guards(
        function,
        &cfg,
        &capability_analysis,
    )?;

    info!("Guard optimization analysis:");
    info!("Total guards before optimization: {}", guard_analysis.required_guards.len());

    // Show optimization techniques applied
    for (i, optimization) in guard_analysis.guard_placement_plan.optimizations.iter().enumerate() {
        match optimization {
            crate::execution::jit::capability_guards::GuardOptimization::RedundancyElimination { eliminated_guards, reason } => {
                info!("Optimization {}: Eliminated {} redundant guards - {}", 
                      i, eliminated_guards.len(), reason);
            }
            crate::execution::jit::capability_guards::GuardOptimization::LoopHoisting { original_location, hoisted_location } => {
                info!("Optimization {}: Hoisted guard from location {} to {}", 
                      i, original_location, hoisted_location);
            }
            crate::execution::jit::capability_guards::GuardOptimization::GuardCombination { combined_guards, new_guard } => {
                info!("Optimization {}: Combined {} guards into one (ID: {})", 
                      i, combined_guards.len(), new_guard.guard_id);
            }
            crate::execution::jit::capability_guards::GuardOptimization::Specialization { common_case, fallback, condition } => {
                info!("Optimization {}: Specialized guard for common case - {}", 
                      i, condition);
            }
            crate::execution::jit::capability_guards::GuardOptimization::ResultCaching { cache_key, invalidation: _ } => {
                info!("Optimization {}: Added result caching with key: {}", 
                      i, cache_key);
            }
        }
    }

    info!("Final estimated overhead: {:.2}%", 
          guard_analysis.guard_placement_plan.estimated_overhead);

    Ok(())
}

/// Demo 8: Audit logging and security analysis
fn demo_audit_logging(function: &FunctionDefinition) -> VMResult<()> {
    info!("=== Demo 8: Audit Logging and Security Analysis ===");

    // Create configuration with comprehensive audit logging
    let config = GuardGeneratorConfig {
        enable_audit_logging: true,
        security_level: 0.8,
        ..GuardGeneratorConfig::default()
    };

    let guard_generator = CapabilityGuardGenerator::new(config)?;

    // Create test scenario with various guard executions
    let mut capabilities = CapabilitySet::new();
    capabilities.add(Capability::new(
        Authority::FileSystem(FileSystemAuthority {
            operations: [FileOperation::Read, FileOperation::Write].into_iter().collect(),
            allowed_paths: vec!["/tmp/*".to_string()],
        }),
        ConstraintSet::new(),
        Duration::from_secs(3600),
        ComponentId::new(1),
    ));

    let execution_context = ExecutionContext {
        function_id: function.id,
        bytecode_offset: 0,
        execution_id: 11111,
        capabilities: capabilities.clone(),
        context_data: std::collections::HashMap::new(),
    };

    // Execute various types of guards to generate audit log entries
    let guard_types = vec![
        GuardType::CapabilityPresence,
        GuardType::AuthorityLevelCheck,
        GuardType::ConstraintValidation,
        GuardType::EffectPermissionCheck,
    ];

    for (i, guard_type) in guard_types.iter().enumerate() {
        use crate::execution::jit::capability_guards::{
            RequiredGuard, GuardCriticality, GuardParameters, GuardFailureStrategy,
            CachingStrategy, LogLevel,
        };

        let test_guard = RequiredGuard {
            guard_id: i as u32 + 10,
            guard_type: guard_type.clone(),
            bytecode_location: i as u32,
            required_capability: capabilities.active_capabilities()[0].clone(),
            criticality: GuardCriticality::Medium,
            execution_frequency: 0.5,
            guard_params: GuardParameters {
                timeout: Some(Duration::from_millis(10)),
                caching_strategy: CachingStrategy::TimeBased { 
                    duration: Duration::from_secs(30) 
                },
                validation_depth: 1,
                custom_params: std::collections::HashMap::new(),
            },
            failure_strategy: GuardFailureStrategy::LogAndContinue {
                log_level: LogLevel::Info,
                warning: format!("Guard {} executed", i),
            },
        };

        let result = guard_generator.execute_guard(
            &test_guard,
            &capabilities,
            &execution_context,
        )?;

        info!("Guard {} ({:?}): passed={}, time={:?}",
              i, guard_type, result.passed, result.execution_time);
    }

    info!("Audit logging demo completed - check logs for detailed security audit trail");

    Ok(())
}

// Helper functions to create test functions

fn create_file_access_function() -> VMResult<FunctionDefinition> {
    use crate::bytecode::instructions::PrismOpcode;

    let mut instructions = Vec::new();
    
    // Load file path
    instructions.push(Instruction {
        opcode: PrismOpcode::LOAD_CONST(0), // "/tmp/test.txt"
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });
    
    // Open file for reading
    instructions.push(Instruction {
        opcode: PrismOpcode::IO_OPEN(0), // Open mode: read
        required_capabilities: vec![
            // This would be populated with actual capability requirements
        ],
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Read from file
    instructions.push(Instruction {
        opcode: PrismOpcode::IO_READ(1024), // Read 1024 bytes
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Close file
    instructions.push(Instruction {
        opcode: PrismOpcode::IO_CLOSE(0),
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Return result
    instructions.push(Instruction {
        opcode: PrismOpcode::RETURN_VALUE,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });

    Ok(FunctionDefinition {
        id: 1,
        name: "file_access_function".to_string(),
        instructions,
        local_count: 2,
        parameter_count: 0,
        capabilities: Vec::new(),
        declared_effects: Vec::new(),
    })
}

fn create_network_function() -> VMResult<FunctionDefinition> {
    use crate::bytecode::instructions::PrismOpcode;

    let mut instructions = Vec::new();
    
    // Load URL
    instructions.push(Instruction {
        opcode: PrismOpcode::LOAD_CONST(0), // "https://api.example.com/data"
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });
    
    // Make HTTP request
    instructions.push(Instruction {
        opcode: PrismOpcode::CALL(100), // HTTP client function
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Process response
    instructions.push(Instruction {
        opcode: PrismOpcode::CALL(101), // Response processor
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });
    
    // Return result
    instructions.push(Instruction {
        opcode: PrismOpcode::RETURN_VALUE,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });

    Ok(FunctionDefinition {
        id: 2,
        name: "network_function".to_string(),
        instructions,
        local_count: 3,
        parameter_count: 0,
        capabilities: Vec::new(),
        declared_effects: Vec::new(),
    })
}

fn create_mixed_capability_function() -> VMResult<FunctionDefinition> {
    use crate::bytecode::instructions::PrismOpcode;

    let mut instructions = Vec::new();
    
    // File operation
    instructions.push(Instruction {
        opcode: PrismOpcode::IO_READ(512),
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Network operation
    instructions.push(Instruction {
        opcode: PrismOpcode::CALL(200), // Network call
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Database operation
    instructions.push(Instruction {
        opcode: PrismOpcode::CALL(201), // Database query
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: true,
    });
    
    // Capability check
    instructions.push(Instruction {
        opcode: PrismOpcode::CAP_CHECK(0),
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });
    
    // Return result
    instructions.push(Instruction {
        opcode: PrismOpcode::RETURN_VALUE,
        required_capabilities: Vec::new(),
        effects: Vec::new(),
        has_side_effects: false,
    });

    Ok(FunctionDefinition {
        id: 3,
        name: "mixed_capability_function".to_string(),
        instructions,
        local_count: 4,
        parameter_count: 1,
        capabilities: Vec::new(),
        declared_effects: Vec::new(),
    })
} 