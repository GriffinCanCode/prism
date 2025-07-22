//! Comprehensive Tests for Modular LLVM Backend
//!
//! This module provides thorough testing of all LLVM backend components,
//! ensuring proper integration and functionality.

#[cfg(test)]
mod tests {
    use super::*;
    use super::core::*;
    use super::types::*;
    use super::instructions::*;
    use super::optimization::*;
    use super::validation::*;
    use super::runtime::*;
    use super::debug_info::*;
    use super::target_machine::*;

    /// Create a test LLVM backend configuration
    fn create_test_config() -> LLVMBackendConfig {
        LLVMBackendConfig {
            target_config: LLVMTargetConfig {
                target_arch: LLVMTargetArch::X86_64,
                target_cpu: "x86-64".to_string(),
                target_features: vec!["sse4.2".to_string()],
                optimization_level: LLVMOptimizationLevel::Aggressive,
                code_model: CodeModel::Default,
                relocation_model: RelocationModel::PIC,
                enable_fast_isel: true,
                enable_global_isel: false,
                disable_tail_calls: false,
                enable_unsafe_fp_math: false,
            },
            type_config: LLVMTypeConfig::default(),
            optimizer_config: LLVMOptimizerConfig {
                level: LLVMOptimizationLevel::Aggressive,
                enable_inlining: true,
                enable_vectorization: true,
                enable_loop_optimization: true,
                enable_dead_code_elimination: true,
                enable_constant_folding: true,
                enable_instruction_combining: true,
                preserve_debug_info: true,
                preserve_semantic_metadata: true,
                max_inline_depth: 5,
                vectorization_threshold: 4,
                loop_unroll_threshold: 150,
            },
            validator_config: LLVMValidatorConfig::default(),
            runtime_config: LLVMRuntimeConfig::default(),
            debug_config: LLVMDebugConfig::default(),
            instruction_config: LLVMInstructionConfig::default(),
            output_path: None,
            emit_assembly: false,
            emit_llvm_ir: true,
            verbose: false,
        }
    }

    /// Create a simple test PIR module
    fn create_test_pir_module() -> PIRModule {
        PIRModule {
            name: "test_module".to_string(),
            types: vec![
                PIRTypeDefinition {
                    name: "TestStruct".to_string(),
                    definition: PIRTypeInfo::Composite(PIRCompositeType {
                        kind: PIRCompositeKind::Struct,
                        fields: vec![
                            ("field1".to_string(), PIRTypeInfo::Primitive(PIRPrimitiveType::I32)),
                            ("field2".to_string(), PIRTypeInfo::Primitive(PIRPrimitiveType::I64)),
                        ],
                        alignment: None,
                        packed: false,
                    }),
                }
            ],
            globals: vec![
                PIRGlobalVariable {
                    name: "test_global".to_string(),
                    var_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                    linkage: PIRLinkage::Public,
                    is_constant: true,
                    is_thread_local: false,
                    initializer: Some(PIRConstantExpression::Integer(42)),
                }
            ],
            function_declarations: vec![
                PIRFunctionDeclaration {
                    name: "external_function".to_string(),
                    return_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                    parameters: vec![
                        PIRParameter {
                            name: "param1".to_string(),
                            param_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                        }
                    ],
                }
            ],
            function_definitions: vec![
                PIRFunctionDefinition {
                    name: "test_function".to_string(),
                    return_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                    parameters: vec![
                        PIRParameter {
                            name: "x".to_string(),
                            param_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                        },
                        PIRParameter {
                            name: "y".to_string(),
                            param_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                        }
                    ],
                    body: vec![
                        PIRStatement::VariableDecl {
                            name: "result".to_string(),
                            var_type: "i32".to_string(),
                            initializer: Some(PIRExpression::BinaryOp {
                                op: BinaryOperator::Add,
                                left: Box::new(PIRExpression::Variable("x".to_string())),
                                right: Box::new(PIRExpression::Variable("y".to_string())),
                            }),
                            is_mutable: false,
                        },
                        PIRStatement::Return(Some(PIRExpression::Variable("result".to_string()))),
                    ],
                    location: Some(SourceLocation {
                        file: "test.prism".to_string(),
                        line: 10,
                        column: 1,
                        scope: None,
                    }),
                }
            ],
        }
    }

    #[test]
    fn test_backend_creation() {
        let config = create_test_config();
        let result = LLVMBackend::new(config);
        assert!(result.is_ok());

        let backend = result.unwrap();
        assert_eq!(backend.get_config().target_config.target_arch, LLVMTargetArch::X86_64);
        assert_eq!(backend.get_config().optimizer_config.level, LLVMOptimizationLevel::Aggressive);
    }

    #[test]
    fn test_type_system() {
        let config = LLVMTypeConfig::default();
        let type_system = LLVMTypeSystem::new(config);

        // Test primitive type conversion
        let i32_type = PIRTypeInfo::Primitive(PIRPrimitiveType::I32);
        let llvm_type = type_system.convert_pir_type(&i32_type).unwrap();
        assert_eq!(llvm_type, LLVMType::Integer(32));

        // Test composite type conversion
        let struct_type = PIRTypeInfo::Composite(PIRCompositeType {
            kind: PIRCompositeKind::Struct,
            fields: vec![
                ("x".to_string(), PIRTypeInfo::Primitive(PIRPrimitiveType::I32)),
                ("y".to_string(), PIRTypeInfo::Primitive(PIRPrimitiveType::F64)),
            ],
            alignment: None,
            packed: false,
        });
        let llvm_struct = type_system.convert_pir_type(&struct_type).unwrap();
        assert!(matches!(llvm_struct, LLVMType::Struct { .. }));
    }

    #[test]
    fn test_instruction_generation() {
        let config = LLVMInstructionConfig::default();
        let runtime_config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(runtime_config);
        let mut generator = LLVMInstructionGenerator::new(config, runtime, None);

        // Test literal generation
        let int_lit = PIRLiteral::Integer(42);
        let result = generator.generate_literal(&int_lit).unwrap();
        assert_eq!(result, "i32 42");

        // Test binary operation generation
        let left = PIRExpression::Literal(PIRLiteral::Integer(10));
        let right = PIRExpression::Literal(PIRLiteral::Integer(20));
        let add_expr = PIRExpression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(left),
            right: Box::new(right),
        };

        let result = generator.generate_expression(&add_expr, None).unwrap();
        assert!(result.starts_with('%'));

        let instructions = generator.get_instructions();
        assert!(!instructions.is_empty());
        assert!(instructions.iter().any(|i| i.contains("add")));
    }

    #[test]
    fn test_optimization() {
        let config = LLVMOptimizerConfig {
            level: LLVMOptimizationLevel::Aggressive,
            enable_constant_folding: true,
            enable_dead_code_elimination: true,
            enable_instruction_combining: true,
            ..Default::default()
        };
        
        let mut optimizer = LLVMOptimizer::new(config);
        
        let test_ir = r#"
define i32 @test() {
entry:
  %1 = add i32 1, 2
  %2 = add i32 %1, 0
  ret i32 %2
}
"#;
        
        let result = optimizer.optimize(test_ir).unwrap();
        let optimization_results = optimizer.get_optimization_results();
        
        assert!(optimization_results.optimizations_applied.len() > 0);
        assert!(optimization_results.optimization_time_ms > 0);
    }

    #[test]
    fn test_validation() {
        let config = LLVMValidatorConfig::default();
        let mut validator = LLVMValidator::new(config);
        
        // Test valid IR
        let valid_ir = r#"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @main() {
entry:
  ret i32 0
}
"#;
        
        let result = validator.validate(valid_ir).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert_eq!(result.stats.function_count, 1);

        // Test invalid IR
        let invalid_ir = r#"
define i0 @invalid() {
  ret i0 0
}
"#;
        
        let result = validator.validate(invalid_ir).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_runtime_integration() {
        let config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(config);
        
        // Test runtime integration generation
        let result = runtime.generate_runtime_integration(LLVMOptimizationLevel::Aggressive).unwrap();
        
        assert!(!result.declarations.is_empty());
        assert!(!result.initialization.is_empty());
        assert!(!result.libraries.is_empty());

        // Test capability validation generation
        let args = vec!["i8* %ptr".to_string(), "i64 1024".to_string()];
        let call = runtime.generate_capability_validation("read", &args).unwrap();
        assert!(call.contains("prism_validate_read_capability"));

        // Test memory allocation generation
        let alloc_call = runtime.generate_memory_allocation("1024", None).unwrap();
        assert!(alloc_call.contains("malloc") || alloc_call.contains("prism_"));
    }

    #[test]
    fn test_debug_info_generation() {
        let config = LLVMDebugConfig {
            enable_debug_info: true,
            source_file: Some("test.prism".to_string()),
            source_directory: Some("/tmp".to_string()),
            ..Default::default()
        };
        
        let mut debug_info = LLVMDebugInfo::new(config);
        let metadata = debug_info.initialize().unwrap();
        
        assert!(!metadata.is_empty());
        assert!(debug_info.compile_unit.is_some());

        // Test function metadata creation
        let function = DebugFunction {
            name: "test_function".to_string(),
            mangled_name: None,
            return_type: "i32".to_string(),
            parameter_types: vec!["i32".to_string(), "i32".to_string()],
            location: SourceLocation {
                file: "test.prism".to_string(),
                line: 10,
                column: 1,
                scope: None,
            },
            scope: 0,
            is_local: false,
            is_definition: true,
        };
        
        let func_metadata = debug_info.create_function_metadata(&function).unwrap();
        assert_eq!(func_metadata.metadata_type, DebugMetadataType::Function);
        assert!(func_metadata.content.contains("test_function"));
    }

    #[test]
    fn test_target_machine() {
        let config = LLVMTargetConfig {
            target_arch: LLVMTargetArch::X86_64,
            target_cpu: "x86-64".to_string(),
            target_features: vec!["sse4.2".to_string()],
            optimization_level: LLVMOptimizationLevel::Aggressive,
            code_model: CodeModel::Default,
            relocation_model: RelocationModel::PIC,
            enable_fast_isel: true,
            enable_global_isel: false,
            disable_tail_calls: false,
            enable_unsafe_fp_math: false,
        };
        
        let result = LLVMTargetMachine::new(config);
        assert!(result.is_ok());
        
        let target_machine = result.unwrap();
        assert!(target_machine.supports_architecture(&LLVMTargetArch::X86_64));
        assert!(!target_machine.supports_architecture(&LLVMTargetArch::WebAssembly));
        
        assert_eq!(target_machine.get_target_triple(), "x86_64-unknown-linux-gnu");
    }

    #[test]
    fn test_complete_code_generation() {
        let config = create_test_config();
        let mut backend = LLVMBackend::new(config).unwrap();
        let pir_module = create_test_pir_module();
        
        let result = backend.generate_code(&pir_module);
        assert!(result.is_ok());
        
        let code_result = result.unwrap();
        assert!(!code_result.llvm_ir.is_empty());
        assert!(code_result.validation.is_valid);
        assert_eq!(code_result.stats.functions_generated, 1);
        
        // Verify the generated IR contains expected elements
        assert!(code_result.llvm_ir.contains("target triple"));
        assert!(code_result.llvm_ir.contains("target datalayout"));
        assert!(code_result.llvm_ir.contains("define i32 @test_function"));
        assert!(code_result.llvm_ir.contains("@test_global"));
        assert!(code_result.llvm_ir.contains("declare i32 @external_function"));
    }

    #[test]
    fn test_semantic_type_generation() {
        let config = create_test_config();
        let backend = LLVMBackend::new(config).unwrap();
        
        let semantic_type = super::super::PIRSemanticType {
            name: "UserId".to_string(),
            domain: "User Management".to_string(),
            base_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
            business_rules: vec![
                super::super::BusinessRule {
                    name: "positive_id".to_string(),
                    description: "User ID must be positive".to_string(),
                    expression: "value > 0".to_string(),
                    enforcement_level: super::super::EnforcementLevel::Error,
                }
            ],
            validation_predicates: vec![
                super::super::ValidationPredicate {
                    name: "range_check".to_string(),
                    expression: "value >= 1 && value <= 1000000".to_string(),
                }
            ],
            performance_constraints: Vec::new(),
            security_classification: super::super::SecurityClassification::Internal,
            ai_context: None,
        };
        
        let validation_function = backend.generate_validation_function(&semantic_type).unwrap();
        assert!(validation_function.contains("define i1 @validate_UserId"));
        assert!(validation_function.contains("prism_validate_business_rule"));
    }

    #[test]
    fn test_error_handling() {
        let config = create_test_config();
        let backend = LLVMBackend::new(config).unwrap();
        
        // Test with invalid PIR module
        let invalid_module = PIRModule {
            name: "invalid".to_string(),
            types: vec![
                PIRTypeDefinition {
                    name: "InvalidType".to_string(),
                    definition: PIRTypeInfo::Primitive(PIRPrimitiveType::I32), // This should cause issues
                }
            ],
            globals: Vec::new(),
            function_declarations: Vec::new(),
            function_definitions: vec![
                PIRFunctionDefinition {
                    name: "invalid_function".to_string(),
                    return_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                    parameters: Vec::new(),
                    body: vec![
                        // Missing return statement should be handled gracefully
                    ],
                    location: None,
                }
            ],
        };
        
        let result = backend.generate_code(&invalid_module);
        // Should still succeed but with warnings or default behavior
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimization_levels() {
        for &level in &[
            LLVMOptimizationLevel::None,
            LLVMOptimizationLevel::Basic,
            LLVMOptimizationLevel::Aggressive,
            LLVMOptimizationLevel::Maximum,
        ] {
            let mut config = create_test_config();
            config.optimizer_config.level = level;
            
            let mut backend = LLVMBackend::new(config).unwrap();
            let pir_module = create_test_pir_module();
            
            let result = backend.generate_code(&pir_module);
            assert!(result.is_ok(), "Failed for optimization level {:?}", level);
            
            let code_result = result.unwrap();
            assert!(!code_result.llvm_ir.is_empty());
            
            // Higher optimization levels should generally result in more optimizations
            if level != LLVMOptimizationLevel::None {
                assert!(code_result.stats.optimizations_applied > 0);
            }
        }
    }

    #[test]
    fn test_target_architectures() {
        for &arch in &[
            LLVMTargetArch::X86_64,
            LLVMTargetArch::AArch64,
            LLVMTargetArch::RISCV64,
        ] {
            let mut config = create_test_config();
            config.target_config.target_arch = arch;
            
            let backend_result = LLVMBackend::new(config);
            assert!(backend_result.is_ok(), "Failed to create backend for {:?}", arch);
            
            let mut backend = backend_result.unwrap();
            let pir_module = create_test_pir_module();
            
            let result = backend.generate_code(&pir_module);
            assert!(result.is_ok(), "Failed code generation for {:?}", arch);
        }
    }

    #[test]
    fn test_concurrent_code_generation() {
        use std::sync::Arc;
        use std::thread;
        
        let config = Arc::new(create_test_config());
        let pir_module = Arc::new(create_test_pir_module());
        
        let handles: Vec<_> = (0..4).map(|i| {
            let config = Arc::clone(&config);
            let pir_module = Arc::clone(&pir_module);
            
            thread::spawn(move || {
                let mut backend = LLVMBackend::new((*config).clone()).unwrap();
                let result = backend.generate_code(&*pir_module);
                assert!(result.is_ok(), "Thread {} failed", i);
                result.unwrap()
            })
        }).collect();
        
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // All results should be valid and similar
        assert_eq!(results.len(), 4);
        for result in &results {
            assert!(result.validation.is_valid);
            assert!(!result.llvm_ir.is_empty());
        }
    }

    #[test]
    fn test_memory_usage() {
        let config = create_test_config();
        let mut backend = LLVMBackend::new(config).unwrap();
        
        // Create a larger PIR module to test memory usage
        let mut large_module = create_test_pir_module();
        
        // Add many functions to test memory scaling
        for i in 0..100 {
            large_module.function_definitions.push(PIRFunctionDefinition {
                name: format!("function_{}", i),
                return_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                parameters: vec![
                    PIRParameter {
                        name: "param".to_string(),
                        param_type: PIRTypeInfo::Primitive(PIRPrimitiveType::I32),
                    }
                ],
                body: vec![
                    PIRStatement::Return(Some(PIRExpression::Variable("param".to_string()))),
                ],
                location: None,
            });
        }
        
        let result = backend.generate_code(&large_module);
        assert!(result.is_ok());
        
        let code_result = result.unwrap();
        assert_eq!(code_result.stats.functions_generated, 101); // 100 + original test function
        assert!(code_result.stats.memory_usage > 0);
    }
} 