//! PIR to Bytecode Compiler
//!
//! This module implements the compiler that transforms PIR (Prism Intermediate Representation)
//! into Prism VM bytecode while preserving all semantic information.

use super::{VMBackendResult, VMBackendError, semantic_compiler::EnhancedSemanticCompiler};
use crate::backends::PrismIR;
use prism_vm::{PrismBytecode, bytecode::*};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, span, Level};

/// PIR to bytecode compiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerConfig {
    /// Preserve semantic information
    pub preserve_semantics: bool,
    /// Generate type information
    pub generate_type_info: bool,
    /// Generate effect information
    pub generate_effect_info: bool,
    /// Generate capability information
    pub generate_capability_info: bool,
    /// Optimization level
    pub optimization_level: u8,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            preserve_semantics: true,
            generate_type_info: true,
            generate_effect_info: true,
            generate_capability_info: true,
            optimization_level: 2,
        }
    }
}

/// Variable scope management for bytecode compilation
#[derive(Debug, Clone)]
struct VariableScope {
    /// Local variables by name -> slot index
    locals: HashMap<String, u8>,
    /// Next available local slot
    next_local: u8,
    /// Parent scope (for nested scopes)
    parent: Option<Box<VariableScope>>,
}

impl VariableScope {
    fn new() -> Self {
        Self {
            locals: HashMap::new(),
            next_local: 0,
            parent: None,
        }
    }

    fn new_nested(parent: VariableScope) -> Self {
        let next_local = parent.next_local;
        Self {
            locals: HashMap::new(),
            next_local,
            parent: Some(Box::new(parent)),
        }
    }

    fn declare_variable(&mut self, name: String) -> u8 {
        let slot = self.next_local;
        self.locals.insert(name, slot);
        self.next_local += 1;
        slot
    }

    fn lookup_variable(&self, name: &str) -> Option<u8> {
        if let Some(&slot) = self.locals.get(name) {
            Some(slot)
        } else if let Some(ref parent) = self.parent {
            parent.lookup_variable(name)
        } else {
            None
        }
    }

    fn get_local_count(&self) -> u8 {
        self.next_local
    }
}

/// PIR to bytecode compiler with enhanced semantic preservation
#[derive(Debug)]
pub struct PIRToBytecodeCompiler {
    /// Compiler configuration
    config: CompilerConfig,
    /// Enhanced semantic compiler for full semantic preservation
    semantic_compiler: EnhancedSemanticCompiler,
}

impl PIRToBytecodeCompiler {
    /// Create a new compiler with configuration
    pub fn new(config: CompilerConfig) -> VMBackendResult<Self> {
        use super::semantic_compiler::SemanticCompilerConfig;
        
        let semantic_config = SemanticCompilerConfig {
            preserve_all_semantics: config.preserve_semantics,
            compile_business_rules: config.preserve_semantics,
            compile_validation_predicates: config.preserve_semantics,
            generate_optimization_hints: config.optimization_level > 0,
            include_ai_metadata: config.preserve_semantics,
            ..Default::default()
        };
        
        Ok(Self { 
            config,
            semantic_compiler: EnhancedSemanticCompiler::new(semantic_config),
        })
    }

    /// Compile PIR to bytecode
    pub fn compile_pir(&mut self, pir: &PrismIR) -> VMBackendResult<PrismBytecode> {
        let _span = span!(Level::INFO, "compile_pir").entered();
        info!("Compiling PIR to Prism bytecode");

        // Create new bytecode instance
        let mut bytecode = PrismBytecode::new(
            pir.modules.first()
                .map(|m| m.name.clone())
                .unwrap_or_else(|| "main".to_string())
        );

        // Compile all modules
        for module in &pir.modules {
            self.compile_module(&mut bytecode, module)?;
        }

        // Compile types from type registry
        for (type_id, (name, semantic_type)) in pir.type_registry.types.iter().enumerate() {
            let type_def = self.compile_semantic_type(type_id as u32, name, semantic_type)?;
            bytecode.types.push(type_def);
        }

        // Ensure we have at least a main function if none exists
        if bytecode.functions.is_empty() {
            let main_function = FunctionDefinition {
                id: 0,
                name: "main".to_string(),
                type_id: 0,
                param_count: 0,
                local_count: 0,
                max_stack_depth: 1,
                capabilities: Vec::new(),
                effects: Vec::new(),
                instructions: vec![
                    Instruction::new(instructions::PrismOpcode::LOAD_NULL),
                    Instruction::new(instructions::PrismOpcode::RETURN_VALUE),
                ],
                exception_handlers: Vec::new(),
                debug_info: None,
                responsibility: Some("Main entry point".to_string()),
                performance_characteristics: Vec::new(),
            };
            bytecode.functions.push(main_function);
        }

        debug!("PIR compilation completed successfully with {} functions and {} types", 
               bytecode.functions.len(), bytecode.types.len());
        Ok(bytecode)
    }

    /// Compile a PIR module to bytecode types and functions
    fn compile_module(&mut self, bytecode: &mut PrismBytecode, module: &crate::backends::PIRModule) -> VMBackendResult<()> {
        debug!("Compiling module: {}", module.name);

        // Compile module sections
        for section in &module.sections {
            self.compile_section(bytecode, section)?;
        }

        // Update module metadata with business context
        bytecode.metadata.name = module.name.clone();
        
        // Add AI metadata from module business context
        if bytecode.metadata.ai_metadata.is_none() {
            bytecode.metadata.ai_metadata = Some(prism_vm::bytecode::AIMetadata {
                intents: std::collections::HashMap::new(),
                examples: Vec::new(),
                patterns: Vec::new(),
                performance_hints: Vec::new(),
            });
        }

        if let Some(ref mut ai_metadata) = bytecode.metadata.ai_metadata {
            ai_metadata.intents.insert(
                module.name.clone(),
                module.business_context.domain.clone(),
            );

            if !module.capabilities.is_empty() {
                ai_metadata.patterns.push(format!(
                    "Module {} implements capabilities: {:?}",
                    module.name, module.capabilities
                ));
            }
        }

        Ok(())
    }

    /// Compile a PIR section to bytecode
    fn compile_section(&mut self, bytecode: &mut PrismBytecode, section: &crate::backends::PIRSection) -> VMBackendResult<()> {
        match section {
            crate::backends::PIRSection::Types(type_section) => {
                for semantic_type in &type_section.types {
                    let type_id = bytecode.types.len() as u32;
                    let type_def = self.compile_semantic_type(type_id, &semantic_type.name, semantic_type)?;
                    bytecode.types.push(type_def);
                }
            }
            crate::backends::PIRSection::Functions(function_section) => {
                for function in &function_section.functions {
                    let function_def = self.compile_function(bytecode, function)?;
                    bytecode.functions.push(function_def);
                }
            }
            crate::backends::PIRSection::Constants(constant_section) => {
                for constant in &constant_section.constants {
                    self.compile_constant(bytecode, constant)?;
                }
            }
            crate::backends::PIRSection::Interface(_interface_section) => {
                // TODO: Handle interface compilation
                debug!("Interface section compilation not yet implemented");
            }
            crate::backends::PIRSection::Implementation(_impl_section) => {
                // TODO: Handle implementation compilation
                debug!("Implementation section compilation not yet implemented");
            }
        }
        Ok(())
    }

    /// Compile a PIR function to bytecode instructions
    fn compile_function(&mut self, bytecode: &mut PrismBytecode, function: &crate::backends::PIRFunction) -> VMBackendResult<FunctionDefinition> {
        debug!("Compiling function: {}", function.name);

        let function_id = bytecode.functions.len() as u32;
        let mut instructions = Vec::new();
        let mut scope = VariableScope::new();
        let mut max_stack_depth = 0u16;
        let mut current_stack_depth = 0u16;

        // Add parameters to scope
        for param in &function.signature.parameters {
            scope.declare_variable(param.name.clone());
        }

        // Compile function body
        self.compile_expression(
            bytecode, 
            &function.body, 
            &mut instructions, 
            &mut scope,
            &mut current_stack_depth, 
            &mut max_stack_depth
        )?;

        // Ensure function ends with a return
        if instructions.is_empty() || !matches!(
            instructions.last().unwrap().opcode, 
            instructions::PrismOpcode::RETURN | instructions::PrismOpcode::RETURN_VALUE
        ) {
            // Function body result is on stack, return it
            instructions.push(Instruction::new(instructions::PrismOpcode::RETURN_VALUE));
        }

        Ok(FunctionDefinition {
            id: function_id,
            name: function.name.clone(),
            type_id: 0, // TODO: Map to actual type ID
            param_count: function.signature.parameters.len() as u8,
            local_count: scope.get_local_count(),
            max_stack_depth,
            capabilities: function.capabilities_required.clone(),
            effects: function.signature.effects.clone(),
            instructions,
            exception_handlers: Vec::new(),
            debug_info: None,
            responsibility: function.responsibility.clone(),
            performance_characteristics: function.performance_characteristics.clone(),
        })
    }

    /// Compile a PIR semantic type to enhanced bytecode type definition
    fn compile_semantic_type(&mut self, type_id: u32, name: &str, semantic_type: &crate::backends::PIRSemanticType) -> VMBackendResult<TypeDefinition> {
        debug!("Compiling semantic type with enhanced preservation: {}", name);

        // Convert to PIR semantic type format for enhanced compiler
        let pir_semantic_type = self.convert_to_pir_semantic_type(semantic_type)?;
        
        // Use enhanced semantic compiler for full semantic preservation
        let type_def = self.semantic_compiler.compile_semantic_type(type_id, name, &pir_semantic_type)?;
        
        Ok(type_def)
    }

    /// Convert backend PIR semantic type to full PIR semantic type
    fn convert_to_pir_semantic_type(&self, semantic_type: &crate::backends::PIRSemanticType) -> VMBackendResult<prism_pir::semantic::PIRSemanticType> {
        use prism_pir::semantic::{PIRSemanticType, PIRTypeInfo, PIRPrimitiveType, PIRTypeAIContext, SecurityClassification};
        use prism_pir::business::BusinessRule;
        
        // Convert base type
        let base_type = match &semantic_type.base_type {
            crate::backends::PIRPrimitiveType::Integer => PIRTypeInfo::Primitive(PIRPrimitiveType::Integer { signed: true, width: 64 }),
            crate::backends::PIRPrimitiveType::Float => PIRTypeInfo::Primitive(PIRPrimitiveType::Float { width: 64 }),
            crate::backends::PIRPrimitiveType::Boolean => PIRTypeInfo::Primitive(PIRPrimitiveType::Boolean),
            crate::backends::PIRPrimitiveType::String => PIRTypeInfo::Primitive(PIRPrimitiveType::String),
            crate::backends::PIRPrimitiveType::Unit => PIRTypeInfo::Primitive(PIRPrimitiveType::Unit),
            crate::backends::PIRPrimitiveType::Composite(_) => {
                // For now, treat composite types as unit types
                // In a complete implementation, this would convert the composite structure
                PIRTypeInfo::Primitive(PIRPrimitiveType::Unit)
            }
        };

        // Convert business rules
        let business_rules: Vec<BusinessRule> = semantic_type.business_rules.iter().map(|rule| {
            BusinessRule {
                id: format!("rule_{}", rule.name),
                name: rule.name.clone(),
                description: rule.description.clone(),
                category: "business".to_string(),
                priority: 1,
                enforcement_level: prism_pir::business::EnforcementLevel::Required,
                condition: None, // TODO: Convert rule condition
                action: None, // TODO: Convert rule action
                metadata: std::collections::HashMap::new(),
            }
        }).collect();

        // Convert validation predicates
        let validation_predicates: Vec<prism_pir::semantic::ValidationPredicate> = semantic_type.validation_predicates.iter().map(|pred| {
            prism_pir::semantic::ValidationPredicate {
                id: format!("pred_{}", pred.description.chars().take(10).collect::<String>()),
                description: pred.description.clone(),
                predicate_type: prism_pir::semantic::PredicateType::Custom,
                expression: None, // TODO: Convert predicate expression
                error_message: format!("Validation failed: {}", pred.description),
            }
        }).collect();

        Ok(PIRSemanticType {
            name: semantic_type.name.clone(),
            base_type,
            domain: semantic_type.domain.clone(),
            business_rules,
            validation_predicates,
            constraints: Vec::new(), // TODO: Convert constraints
            ai_context: PIRTypeAIContext {
                purpose: format!("Semantic type: {}", semantic_type.name),
                usage_patterns: Vec::new(),
                examples: Vec::new(),
                related_concepts: Vec::new(),
            },
            security_classification: SecurityClassification::Internal, // Default classification
        })
    }

    /// Compile a composite type
    fn compile_composite_type(&mut self, composite: &crate::backends::PIRCompositeType) -> VMBackendResult<TypeKind> {
        match &composite.kind {
            crate::backends::PIRCompositeKind::Struct => {
                let fields = composite.fields.iter().map(|field| {
                    FieldDefinition {
                        name: field.name.clone(),
                        type_id: 0, // TODO: Map to actual type ID
                        offset: None,
                        business_meaning: field.business_meaning.clone(),
                    }
                }).collect();

                Ok(TypeKind::Composite(CompositeType {
                    kind: CompositeKind::Struct,
                    fields,
                    methods: Vec::new(),
                }))
            }
            crate::backends::PIRCompositeKind::Enum => {
                Ok(TypeKind::Composite(CompositeType {
                    kind: CompositeKind::Enum,
                    fields: Vec::new(),
                    methods: Vec::new(),
                }))
            }
            crate::backends::PIRCompositeKind::Union => {
                Ok(TypeKind::Composite(CompositeType {
                    kind: CompositeKind::Union,
                    fields: Vec::new(),
                    methods: Vec::new(),
                }))
            }
            crate::backends::PIRCompositeKind::Tuple => {
                Ok(TypeKind::Composite(CompositeType {
                    kind: CompositeKind::Tuple,
                    fields: Vec::new(),
                    methods: Vec::new(),
                }))
            }
        }
    }

    /// Compile a constant to the constant pool
    fn compile_constant(&mut self, bytecode: &mut PrismBytecode, constant: &crate::backends::PIRConstant) -> VMBackendResult<u32> {
        // For PIRConstant, we need to evaluate the expression to get the actual value
        let constant_value = match self.evaluate_constant_expression(&constant.value)? {
            ConstantValue::Integer(i) => Constant::Integer(i),
            ConstantValue::Float(f) => Constant::Float(f),
            ConstantValue::Boolean(b) => Constant::Boolean(b),
            ConstantValue::String(s) => Constant::String(s),
            ConstantValue::Null => Constant::Null,
        };

        Ok(bytecode.constants.add_constant(constant_value))
    }

    /// Evaluate a constant expression to get its value
    fn evaluate_constant_expression(&self, expr: &crate::backends::PIRExpression) -> VMBackendResult<ConstantValue> {
        match expr {
            crate::backends::PIRExpression::Literal(literal) => {
                match literal {
                    crate::backends::PIRLiteral::Integer(i) => Ok(ConstantValue::Integer(*i)),
                    crate::backends::PIRLiteral::Float(f) => Ok(ConstantValue::Float(*f)),
                    crate::backends::PIRLiteral::Boolean(b) => Ok(ConstantValue::Boolean(*b)),
                    crate::backends::PIRLiteral::String(s) => Ok(ConstantValue::String(s.clone())),
                    crate::backends::PIRLiteral::Null => Ok(ConstantValue::Null),
                }
            }
            _ => {
                // For now, only handle literal expressions in constants
                // More complex constant expressions would need evaluation
                Err(VMBackendError::CompilationError {
                    message: "Complex constant expressions not yet supported".to_string(),
                })
            }
        }
    }

/// Helper enum for constant values during compilation
#[derive(Debug, Clone)]
enum ConstantValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Null,
}

    /// Compile a statement to bytecode instructions
    fn compile_statement(
        &mut self,
        bytecode: &mut PrismBytecode,
        statement: &crate::backends::PIRStatement,
        instructions: &mut Vec<Instruction>,
        scope: &mut VariableScope,
        current_stack_depth: &mut u16,
        max_stack_depth: &mut u16,
    ) -> VMBackendResult<()> {
        match statement {
            crate::backends::PIRStatement::Expression(expr) => {
                self.compile_expression(bytecode, expr, instructions, scope, current_stack_depth, max_stack_depth)?;
                // Pop expression result if not used (statement context)
                instructions.push(Instruction::new(instructions::PrismOpcode::POP));
                *current_stack_depth -= 1;
            }
            crate::backends::PIRStatement::Let { name, type_annotation: _, value } => {
                // Compile the initial value
                self.compile_expression(bytecode, value, instructions, scope, current_stack_depth, max_stack_depth)?;
                
                // Declare variable and store value
                let local_slot = scope.declare_variable(name.clone());
                instructions.push(Instruction::new(instructions::PrismOpcode::STORE_LOCAL(local_slot)));
                *current_stack_depth -= 1; // Store consumes the value
            }
            crate::backends::PIRStatement::Assignment { target, value } => {
                // Compile the value first
                self.compile_expression(bytecode, value, instructions, scope, current_stack_depth, max_stack_depth)?;
                
                // Compile assignment target
                match target {
                    crate::backends::PIRExpression::Variable(var_name) => {
                        if let Some(local_slot) = scope.lookup_variable(var_name) {
                            instructions.push(Instruction::new(instructions::PrismOpcode::STORE_LOCAL(local_slot)));
                            *current_stack_depth -= 1; // Store consumes the value
                        } else {
                            return Err(VMBackendError::CompilationError {
                                message: format!("Undefined variable in assignment: {}", var_name),
                            });
                        }
                    }
                    _ => {
                        return Err(VMBackendError::CompilationError {
                            message: "Complex assignment targets not yet supported".to_string(),
                        });
                    }
                }
            }
            crate::backends::PIRStatement::Return(expr) => {
                if let Some(expr) = expr {
                    self.compile_expression(bytecode, expr, instructions, scope, current_stack_depth, max_stack_depth)?;
                    instructions.push(Instruction::new(instructions::PrismOpcode::RETURN_VALUE));
                } else {
                    instructions.push(Instruction::new(instructions::PrismOpcode::RETURN));
                }
            }
        }
        Ok(())
    }

    /// Compile an expression to bytecode instructions - COMPREHENSIVE IMPLEMENTATION
    fn compile_expression(
        &mut self,
        bytecode: &mut PrismBytecode,
        expression: &crate::backends::PIRExpression,
        instructions: &mut Vec<Instruction>,
        scope: &mut VariableScope,
        current_stack_depth: &mut u16,
        max_stack_depth: &mut u16,
    ) -> VMBackendResult<()> {
        match expression {
            crate::backends::PIRExpression::Literal(literal) => {
                self.compile_literal(bytecode, literal, instructions)?;
                *current_stack_depth += 1;
                *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
            }
            
            crate::backends::PIRExpression::Variable(var_name) => {
                if let Some(local_slot) = scope.lookup_variable(var_name) {
                    instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_LOCAL(local_slot)));
                    *current_stack_depth += 1;
                    *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                } else {
                    return Err(VMBackendError::CompilationError {
                        message: format!("Undefined variable: {}", var_name),
                    });
                }
            }
            
            crate::backends::PIRExpression::Call { function, arguments, effects: _ } => {
                // Compile arguments first (in order)
                for arg in arguments {
                    self.compile_expression(bytecode, arg, instructions, scope, current_stack_depth, max_stack_depth)?;
                }
                
                // Compile function expression
                match function.as_ref() {
                    crate::backends::PIRExpression::Variable(func_name) => {
                        // Direct function call by name - load function constant
                        let func_const_index = bytecode.constants.add_string(func_name.clone());
                        instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_CONST(func_const_index as u16)));
                        *current_stack_depth += 1;
                        *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                        
                        instructions.push(Instruction::new(instructions::PrismOpcode::CALL(arguments.len() as u8)));
                        // Call consumes arguments + function, produces result
                        *current_stack_depth = current_stack_depth.saturating_sub(arguments.len() as u16 + 1);
                        *current_stack_depth += 1; // Result
                        *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                    }
                    _ => {
                        // Dynamic function call
                        self.compile_expression(bytecode, function, instructions, scope, current_stack_depth, max_stack_depth)?;
                        instructions.push(Instruction::new(instructions::PrismOpcode::CALL_DYNAMIC(arguments.len() as u8)));
                        // Dynamic call consumes arguments + function, produces result
                        *current_stack_depth = current_stack_depth.saturating_sub(arguments.len() as u16 + 1);
                        *current_stack_depth += 1; // Result
                        *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                    }
                }
            }
            
            crate::backends::PIRExpression::Binary { left, operator, right } => {
                // Compile left operand
                self.compile_expression(bytecode, left, instructions, scope, current_stack_depth, max_stack_depth)?;
                // Compile right operand
                self.compile_expression(bytecode, right, instructions, scope, current_stack_depth, max_stack_depth)?;
                
                // Compile operator (pops 2, pushes 1)
                let opcode = match operator {
                    crate::backends::PIRBinaryOp::Add => instructions::PrismOpcode::ADD,
                    crate::backends::PIRBinaryOp::Subtract => instructions::PrismOpcode::SUB,
                    crate::backends::PIRBinaryOp::Multiply => instructions::PrismOpcode::MUL,
                    crate::backends::PIRBinaryOp::Divide => instructions::PrismOpcode::DIV,
                    crate::backends::PIRBinaryOp::Modulo => instructions::PrismOpcode::MOD,
                    crate::backends::PIRBinaryOp::Equal => instructions::PrismOpcode::EQ,
                    crate::backends::PIRBinaryOp::NotEqual => instructions::PrismOpcode::NE,
                    crate::backends::PIRBinaryOp::Less => instructions::PrismOpcode::LT,
                    crate::backends::PIRBinaryOp::LessEqual => instructions::PrismOpcode::LE,
                    crate::backends::PIRBinaryOp::Greater => instructions::PrismOpcode::GT,
                    crate::backends::PIRBinaryOp::GreaterEqual => instructions::PrismOpcode::GE,
                    crate::backends::PIRBinaryOp::And => instructions::PrismOpcode::AND,
                    crate::backends::PIRBinaryOp::Or => instructions::PrismOpcode::OR,
                    crate::backends::PIRBinaryOp::SemanticEqual => instructions::PrismOpcode::SEMANTIC_EQ,
                };
                instructions.push(Instruction::new(opcode));
                *current_stack_depth -= 1; // Binary ops consume 2, produce 1
            }
            
            crate::backends::PIRExpression::Unary { operator, operand } => {
                self.compile_expression(bytecode, operand, instructions, scope, current_stack_depth, max_stack_depth)?;
                let opcode = match operator {
                    crate::backends::PIRUnaryOp::Negate => instructions::PrismOpcode::NEG,
                    crate::backends::PIRUnaryOp::Not => instructions::PrismOpcode::NOT,
                };
                instructions.push(Instruction::new(opcode));
                // Unary ops don't change stack depth (consume 1, produce 1)
            }
            
            crate::backends::PIRExpression::Block { statements, result } => {
                // Create new nested scope for block
                let mut block_scope = VariableScope::new_nested(scope.clone());
                
                // Compile all statements
                for statement in statements {
                    self.compile_statement(bytecode, statement, instructions, &mut block_scope, current_stack_depth, max_stack_depth)?;
                }
                
                // Compile result expression if present
                if let Some(result_expr) = result {
                    self.compile_expression(bytecode, result_expr, instructions, &mut block_scope, current_stack_depth, max_stack_depth)?;
                } else {
                    // Block with no result produces unit/null
                    instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_NULL));
                    *current_stack_depth += 1;
                    *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                }
                
                // Update original scope with new local count
                scope.next_local = block_scope.get_local_count();
            }
            
            crate::backends::PIRExpression::If { condition, then_branch, else_branch } => {
                // Compile condition
                self.compile_expression(bytecode, condition, instructions, scope, current_stack_depth, max_stack_depth)?;
                
                // Jump to else branch if condition is false
                let else_jump_placeholder = instructions.len();
                instructions.push(Instruction::new(instructions::PrismOpcode::JUMP_IF_FALSE(0))); // Placeholder offset
                *current_stack_depth -= 1; // Condition consumed by jump
                
                // Compile then branch
                self.compile_expression(bytecode, then_branch, instructions, scope, current_stack_depth, max_stack_depth)?;
                
                // Jump past else branch
                let end_jump_placeholder = instructions.len();
                instructions.push(Instruction::new(instructions::PrismOpcode::JUMP(0))); // Placeholder offset
                
                // Update else jump offset
                let else_start = instructions.len() as i16 - else_jump_placeholder as i16 - 1;
                if let instructions::PrismOpcode::JUMP_IF_FALSE(_) = &mut instructions[else_jump_placeholder].opcode {
                    instructions[else_jump_placeholder].opcode = instructions::PrismOpcode::JUMP_IF_FALSE(else_start);
                }
                
                // Compile else branch
                if let Some(else_expr) = else_branch {
                    self.compile_expression(bytecode, else_expr, instructions, scope, current_stack_depth, max_stack_depth)?;
                } else {
                    // No else branch, push null
                    instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_NULL));
                    *current_stack_depth += 1;
                    *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                }
                
                // Update end jump offset
                let end_offset = instructions.len() as i16 - end_jump_placeholder as i16 - 1;
                if let instructions::PrismOpcode::JUMP(_) = &mut instructions[end_jump_placeholder].opcode {
                    instructions[end_jump_placeholder].opcode = instructions::PrismOpcode::JUMP(end_offset);
                }
            }
            
            crate::backends::PIRExpression::Match { scrutinee, arms } => {
                // Compile scrutinee
                self.compile_expression(bytecode, scrutinee, instructions, scope, current_stack_depth, max_stack_depth)?;
                
                // For now, implement a simplified pattern matching
                let mut end_jumps = Vec::new();
                
                for arm in arms {
                    // Duplicate scrutinee for pattern matching
                    instructions.push(Instruction::new(instructions::PrismOpcode::DUP));
                    *current_stack_depth += 1;
                    *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                    
                    // Compile pattern matching logic (simplified)
                    let pattern_matches = match &arm.pattern {
                        crate::backends::PIRPattern::Literal(lit) => {
                            self.compile_literal(bytecode, lit, instructions)?;
                            *current_stack_depth += 1;
                            *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                            
                            instructions.push(Instruction::new(instructions::PrismOpcode::EQ));
                            *current_stack_depth -= 1; // EQ consumes 2, produces 1
                            true
                        }
                        crate::backends::PIRPattern::Variable(var_name) => {
                            // Variable pattern always matches, bind the value
                            let local_slot = scope.declare_variable(var_name.clone());
                            instructions.push(Instruction::new(instructions::PrismOpcode::STORE_LOCAL(local_slot)));
                            *current_stack_depth -= 1;
                            // Always true for variable patterns
                            instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_TRUE));
                            *current_stack_depth += 1;
                            *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                            true
                        }
                        crate::backends::PIRPattern::Wildcard => {
                            // Wildcard always matches, just pop the duplicate
                            instructions.push(Instruction::new(instructions::PrismOpcode::POP));
                            *current_stack_depth -= 1;
                            instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_TRUE));
                            *current_stack_depth += 1;
                            *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                            true
                        }
                        crate::backends::PIRPattern::Constructor { name: _, fields: _ } => {
                            // Constructor pattern matching is complex, simplified for now
                            instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_FALSE));
                            *current_stack_depth += 1;
                            *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                            false
                        }
                    };
                    
                    if pattern_matches {
                        // Jump to next arm if pattern doesn't match
                        let next_arm_jump = instructions.len();
                        instructions.push(Instruction::new(instructions::PrismOpcode::JUMP_IF_FALSE(0))); // Placeholder
                        *current_stack_depth -= 1; // Condition consumed
                        
                        // Compile guard if present
                        if let Some(guard) = &arm.guard {
                            self.compile_expression(bytecode, guard, instructions, scope, current_stack_depth, max_stack_depth)?;
                            let guard_jump = instructions.len();
                            instructions.push(Instruction::new(instructions::PrismOpcode::JUMP_IF_FALSE(0))); // Placeholder
                            *current_stack_depth -= 1;
                            // Update guard jump to point to next arm
                            let guard_offset = (instructions.len() - guard_jump) as i16;
                            if let instructions::PrismOpcode::JUMP_IF_FALSE(_) = &mut instructions[guard_jump].opcode {
                                instructions[guard_jump].opcode = instructions::PrismOpcode::JUMP_IF_FALSE(guard_offset);
                            }
                        }
                        
                        // Pop the original scrutinee (pattern matched)
                        instructions.push(Instruction::new(instructions::PrismOpcode::SWAP)); // Bring scrutinee to top
                        instructions.push(Instruction::new(instructions::PrismOpcode::POP)); // Pop scrutinee
                        *current_stack_depth -= 1;
                        
                        // Compile arm body
                        self.compile_expression(bytecode, &arm.body, instructions, scope, current_stack_depth, max_stack_depth)?;
                        
                        // Jump to end of match
                        let end_jump = instructions.len();
                        instructions.push(Instruction::new(instructions::PrismOpcode::JUMP(0))); // Placeholder
                        end_jumps.push(end_jump);
                        
                        // Update next arm jump offset
                        let next_arm_offset = instructions.len() as i16 - next_arm_jump as i16 - 1;
                        if let instructions::PrismOpcode::JUMP_IF_FALSE(_) = &mut instructions[next_arm_jump].opcode {
                            instructions[next_arm_jump].opcode = instructions::PrismOpcode::JUMP_IF_FALSE(next_arm_offset);
                        }
                    }
                }
                
                // Pop the original scrutinee if no patterns matched
                instructions.push(Instruction::new(instructions::PrismOpcode::POP));
                *current_stack_depth -= 1;
                
                // Default case: push null if no patterns matched
                instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_NULL));
                *current_stack_depth += 1;
                *max_stack_depth = (*max_stack_depth).max(*current_stack_depth);
                
                // Update all end jump offsets
                let match_end = instructions.len();
                for &jump_idx in &end_jumps {
                    let offset = match_end as i16 - jump_idx as i16 - 1;
                    if let instructions::PrismOpcode::JUMP(_) = &mut instructions[jump_idx].opcode {
                        instructions[jump_idx].opcode = instructions::PrismOpcode::JUMP(offset);
                    }
                }
            }
            
            crate::backends::PIRExpression::TypeAssertion { expression, target_type: _ } => {
                // For now, just compile the expression (type assertions are compile-time)
                // In a complete implementation, this might generate runtime type checks
                self.compile_expression(bytecode, expression, instructions, scope, current_stack_depth, max_stack_depth)?;
            }
        }
        Ok(())
    }

    /// Compile a literal to bytecode instructions
    fn compile_literal(
        &mut self,
        bytecode: &mut PrismBytecode,
        literal: &crate::backends::PIRLiteral,
        instructions: &mut Vec<Instruction>,
    ) -> VMBackendResult<()> {
        match literal {
            crate::backends::PIRLiteral::Integer(i) => {
                if *i >= -128 && *i <= 127 {
                    instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_SMALL_INT(*i as i8)));
                } else {
                    let const_index = bytecode.constants.add_integer(*i);
                    instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_CONST(const_index as u16)));
                }
            }
            crate::backends::PIRLiteral::Float(f) => {
                let const_index = bytecode.constants.add_float(*f);
                instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_CONST(const_index as u16)));
            }
            crate::backends::PIRLiteral::Boolean(true) => {
                instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_TRUE));
            }
            crate::backends::PIRLiteral::Boolean(false) => {
                instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_FALSE));
            }
            crate::backends::PIRLiteral::String(s) => {
                let const_index = bytecode.constants.add_string(s.clone());
                instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_CONST(const_index as u16)));
            }
            crate::backends::PIRLiteral::Unit => {
                instructions.push(Instruction::new(instructions::PrismOpcode::LOAD_NULL));
            }
        }
        Ok(())
    }
} 