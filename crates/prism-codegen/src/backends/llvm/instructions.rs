//! LLVM Instruction Generation
//!
//! This module handles LLVM IR instruction generation from PIR expressions,
//! including arithmetic, control flow, memory operations, and function calls.

use super::{LLVMResult, LLVMError};
use super::types::{LLVMType, LLVMOptimizationLevel};
use super::runtime::LLVMRuntime;
use super::debug_info::{LLVMDebugInfo, SourceLocation, DebugVariable};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// LLVM instruction generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMInstructionConfig {
    /// Optimization level
    pub optimization_level: LLVMOptimizationLevel,
    /// Enable bounds checking
    pub enable_bounds_checking: bool,
    /// Enable overflow checking
    pub enable_overflow_checking: bool,
    /// Enable null pointer checking
    pub enable_null_checking: bool,
    /// Enable debug information
    pub enable_debug_info: bool,
    /// Enable fast math operations
    pub enable_fast_math: bool,
    /// Enable vectorization hints
    pub enable_vectorization: bool,
    /// Use native calling convention
    pub use_native_calling_convention: bool,
    /// Maximum inline depth
    pub max_inline_depth: u32,
}

impl Default for LLVMInstructionConfig {
    fn default() -> Self {
        Self {
            optimization_level: LLVMOptimizationLevel::Aggressive,
            enable_bounds_checking: true,
            enable_overflow_checking: true,
            enable_null_checking: true,
            enable_debug_info: true,
            enable_fast_math: false,
            enable_vectorization: true,
            use_native_calling_convention: true,
            max_inline_depth: 5,
        }
    }
}

/// LLVM instruction generator
pub struct LLVMInstructionGenerator {
    /// Generator configuration
    config: LLVMInstructionConfig,
    /// Runtime integration
    runtime: LLVMRuntime,
    /// Debug information generator
    debug_info: Option<LLVMDebugInfo>,
    /// Temporary variable counter
    temp_counter: u32,
    /// Basic block counter
    block_counter: u32,
    /// Current function context
    current_function: Option<String>,
    /// Variable registry
    variables: HashMap<String, VariableInfo>,
    /// Type registry
    types: HashMap<String, LLVMType>,
    /// Generated instructions buffer
    instructions: Vec<String>,
}

/// Variable information for instruction generation
#[derive(Debug, Clone)]
struct VariableInfo {
    /// LLVM register name
    llvm_name: String,
    /// Variable type
    var_type: LLVMType,
    /// Whether variable is mutable
    is_mutable: bool,
    /// Source location
    location: Option<SourceLocation>,
    /// Whether variable is parameter
    is_parameter: bool,
    /// Scope depth
    scope_depth: u32,
}

/// PIR expression types (simplified for this example)
#[derive(Debug, Clone)]
pub enum PIRExpression {
    /// Literal value
    Literal(PIRLiteral),
    /// Variable reference
    Variable(String),
    /// Binary operation
    BinaryOp {
        op: BinaryOperator,
        left: Box<PIRExpression>,
        right: Box<PIRExpression>,
    },
    /// Unary operation
    UnaryOp {
        op: UnaryOperator,
        operand: Box<PIRExpression>,
    },
    /// Function call
    FunctionCall {
        function: String,
        args: Vec<PIRExpression>,
    },
    /// Array access
    ArrayAccess {
        array: Box<PIRExpression>,
        index: Box<PIRExpression>,
    },
    /// Field access
    FieldAccess {
        object: Box<PIRExpression>,
        field: String,
    },
    /// Assignment
    Assignment {
        target: Box<PIRExpression>,
        value: Box<PIRExpression>,
    },
    /// Conditional expression
    Conditional {
        condition: Box<PIRExpression>,
        then_expr: Box<PIRExpression>,
        else_expr: Box<PIRExpression>,
    },
}

/// PIR literal types
#[derive(Debug, Clone)]
pub enum PIRLiteral {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Null,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Xor,
    Shl, Shr,
    BitAnd, BitOr, BitXor,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOperator {
    Neg, Not, BitNot,
    Deref, AddressOf,
}

/// PIR statement types
#[derive(Debug, Clone)]
pub enum PIRStatement {
    /// Expression statement
    Expression(PIRExpression),
    /// Variable declaration
    VariableDecl {
        name: String,
        var_type: String,
        initializer: Option<PIRExpression>,
        is_mutable: bool,
    },
    /// If statement
    If {
        condition: PIRExpression,
        then_block: Vec<PIRStatement>,
        else_block: Option<Vec<PIRStatement>>,
    },
    /// While loop
    While {
        condition: PIRExpression,
        body: Vec<PIRStatement>,
    },
    /// For loop
    For {
        init: Option<Box<PIRStatement>>,
        condition: Option<PIRExpression>,
        update: Option<PIRExpression>,
        body: Vec<PIRStatement>,
    },
    /// Return statement
    Return(Option<PIRExpression>),
    /// Break statement
    Break,
    /// Continue statement
    Continue,
    /// Block statement
    Block(Vec<PIRStatement>),
}

impl LLVMInstructionGenerator {
    /// Create new instruction generator
    pub fn new(
        config: LLVMInstructionConfig,
        runtime: LLVMRuntime,
        debug_info: Option<LLVMDebugInfo>,
    ) -> Self {
        Self {
            config,
            runtime,
            debug_info,
            temp_counter: 0,
            block_counter: 0,
            current_function: None,
            variables: HashMap::new(),
            types: HashMap::new(),
            instructions: Vec::new(),
        }
    }

    /// Generate instructions for a PIR expression
    pub fn generate_expression(&mut self, expr: &PIRExpression, location: Option<&SourceLocation>) -> LLVMResult<String> {
        match expr {
            PIRExpression::Literal(lit) => self.generate_literal(lit),
            PIRExpression::Variable(name) => self.generate_variable_access(name),
            PIRExpression::BinaryOp { op, left, right } => {
                self.generate_binary_operation(op, left, right, location)
            }
            PIRExpression::UnaryOp { op, operand } => {
                self.generate_unary_operation(op, operand, location)
            }
            PIRExpression::FunctionCall { function, args } => {
                self.generate_function_call(function, args, location)
            }
            PIRExpression::ArrayAccess { array, index } => {
                self.generate_array_access(array, index, location)
            }
            PIRExpression::FieldAccess { object, field } => {
                self.generate_field_access(object, field, location)
            }
            PIRExpression::Assignment { target, value } => {
                self.generate_assignment(target, value, location)
            }
            PIRExpression::Conditional { condition, then_expr, else_expr } => {
                self.generate_conditional(condition, then_expr, else_expr, location)
            }
        }
    }

    /// Generate instructions for a PIR statement
    pub fn generate_statement(&mut self, stmt: &PIRStatement, location: Option<&SourceLocation>) -> LLVMResult<()> {
        match stmt {
            PIRStatement::Expression(expr) => {
                self.generate_expression(expr, location)?;
            }
            PIRStatement::VariableDecl { name, var_type, initializer, is_mutable } => {
                self.generate_variable_declaration(name, var_type, initializer.as_ref(), *is_mutable, location)?;
            }
            PIRStatement::If { condition, then_block, else_block } => {
                self.generate_if_statement(condition, then_block, else_block.as_ref(), location)?;
            }
            PIRStatement::While { condition, body } => {
                self.generate_while_loop(condition, body, location)?;
            }
            PIRStatement::For { init, condition, update, body } => {
                self.generate_for_loop(init.as_deref(), condition.as_ref(), update.as_ref(), body, location)?;
            }
            PIRStatement::Return(expr) => {
                self.generate_return_statement(expr.as_ref(), location)?;
            }
            PIRStatement::Break => {
                self.generate_break_statement(location)?;
            }
            PIRStatement::Continue => {
                self.generate_continue_statement(location)?;
            }
            PIRStatement::Block(statements) => {
                self.generate_block(statements, location)?;
            }
        }
        Ok(())
    }

    /// Generate literal value
    fn generate_literal(&mut self, lit: &PIRLiteral) -> LLVMResult<String> {
        match lit {
            PIRLiteral::Integer(val) => Ok(format!("i32 {}", val)),
            PIRLiteral::Float(val) => Ok(format!("double {}", val)),
            PIRLiteral::Boolean(val) => Ok(format!("i1 {}", if *val { 1 } else { 0 })),
            PIRLiteral::String(val) => {
                let global_name = self.generate_string_constant(val)?;
                Ok(format!("i8* {}", global_name))
            }
            PIRLiteral::Null => Ok("i8* null".to_string()),
        }
    }

    /// Generate variable access
    fn generate_variable_access(&mut self, name: &str) -> LLVMResult<String> {
        if let Some(var_info) = self.variables.get(name) {
            let temp_name = self.next_temp();
            let load_instr = format!(
                "  {} = load {}, {}* {}",
                temp_name,
                var_info.var_type.to_llvm_string(),
                var_info.var_type.to_llvm_string(),
                var_info.llvm_name
            );
            
            self.instructions.push(load_instr);
            Ok(temp_name)
        } else {
            Err(LLVMError::UndefinedVariable(name.to_string()))
        }
    }

    /// Generate binary operation
    fn generate_binary_operation(
        &mut self,
        op: &BinaryOperator,
        left: &PIRExpression,
        right: &PIRExpression,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        let left_val = self.generate_expression(left, location)?;
        let right_val = self.generate_expression(right, location)?;
        
        let result_temp = self.next_temp();
        let op_str = self.binary_op_to_llvm(op, &left_val, &right_val)?;
        
        let instruction = format!("  {} = {}", result_temp, op_str);
        
        // Add debug location if available
        let final_instruction = if let Some(loc) = location {
            if let Some(ref debug_info) = self.debug_info {
                debug_info.add_debug_location(&instruction, loc)
            } else {
                instruction
            }
        } else {
            instruction
        };
        
        self.instructions.push(final_instruction);
        Ok(result_temp)
    }

    /// Generate unary operation
    fn generate_unary_operation(
        &mut self,
        op: &UnaryOperator,
        operand: &PIRExpression,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        let operand_val = self.generate_expression(operand, location)?;
        let result_temp = self.next_temp();
        
        let op_str = match op {
            UnaryOperator::Neg => format!("sub i32 0, {}", self.extract_value(&operand_val)),
            UnaryOperator::Not => format!("xor i1 {}, true", self.extract_value(&operand_val)),
            UnaryOperator::BitNot => format!("xor i32 {}, -1", self.extract_value(&operand_val)),
            UnaryOperator::Deref => {
                // Load from pointer
                format!("load i32, i32* {}", self.extract_value(&operand_val))
            }
            UnaryOperator::AddressOf => {
                // This would need more complex handling in a real implementation
                return Err(LLVMError::UnsupportedOperation("AddressOf".to_string()));
            }
        };
        
        let instruction = format!("  {} = {}", result_temp, op_str);
        self.instructions.push(instruction);
        Ok(result_temp)
    }

    /// Generate function call
    fn generate_function_call(
        &mut self,
        function: &str,
        args: &[PIRExpression],
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        // Generate arguments
        let mut arg_values = Vec::new();
        for arg in args {
            let arg_val = self.generate_expression(arg, location)?;
            arg_values.push(arg_val);
        }

        // Generate capability validation if needed
        if self.runtime.is_function_available(function) {
            if let Some(security_level) = self.runtime.get_function_security_level(function) {
                // Add capability validation based on security level
                let validation_args = vec![format!("i8* getelementptr inbounds ([{} x i8], [{}  x i8]* @.str.{}, i32 0, i32 0)", function.len() + 1, function.len() + 1, function.len())];
                let validation_call = self.runtime.generate_capability_validation("execute", &validation_args)?;
                if !validation_call.is_empty() {
                    self.instructions.push(format!("  {}", validation_call));
                }
            }
        }

        let result_temp = self.next_temp();
        let args_str = arg_values.join(", ");
        
        // Determine return type (simplified)
        let return_type = "i32"; // This would be determined from function signature
        
        let call_instruction = format!(
            "  {} = call {} @{}({})",
            result_temp,
            return_type,
            function,
            args_str
        );
        
        self.instructions.push(call_instruction);
        Ok(result_temp)
    }

    /// Generate array access
    fn generate_array_access(
        &mut self,
        array: &PIRExpression,
        index: &PIRExpression,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        let array_val = self.generate_expression(array, location)?;
        let index_val = self.generate_expression(index, location)?;
        
        // Generate bounds checking if enabled
        if self.config.enable_bounds_checking {
            let bounds_check = self.runtime.generate_security_enforcement(
                "bounds",
                &[
                    self.extract_value(&array_val).to_string(),
                    self.extract_value(&index_val).to_string(),
                    "1024".to_string(), // Array size - would be determined dynamically
                ]
            )?;
            if !bounds_check.is_empty() {
                self.instructions.push(format!("  {}", bounds_check));
            }
        }
        
        let ptr_temp = self.next_temp();
        let gep_instruction = format!(
            "  {} = getelementptr inbounds i32, i32* {}, i32 {}",
            ptr_temp,
            self.extract_value(&array_val),
            self.extract_value(&index_val)
        );
        self.instructions.push(gep_instruction);
        
        let result_temp = self.next_temp();
        let load_instruction = format!(
            "  {} = load i32, i32* {}",
            result_temp,
            ptr_temp
        );
        self.instructions.push(load_instruction);
        
        Ok(result_temp)
    }

    /// Generate field access
    fn generate_field_access(
        &mut self,
        object: &PIRExpression,
        field: &str,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        let object_val = self.generate_expression(object, location)?;
        
        // This is simplified - would need struct type information
        let field_index = 0; // Would be determined from struct definition
        
        let ptr_temp = self.next_temp();
        let gep_instruction = format!(
            "  {} = getelementptr inbounds %struct.Object, %struct.Object* {}, i32 0, i32 {}",
            ptr_temp,
            self.extract_value(&object_val),
            field_index
        );
        self.instructions.push(gep_instruction);
        
        let result_temp = self.next_temp();
        let load_instruction = format!(
            "  {} = load i32, i32* {}",
            result_temp,
            ptr_temp
        );
        self.instructions.push(load_instruction);
        
        Ok(result_temp)
    }

    /// Generate assignment
    fn generate_assignment(
        &mut self,
        target: &PIRExpression,
        value: &PIRExpression,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        let value_val = self.generate_expression(value, location)?;
        
        match target {
            PIRExpression::Variable(name) => {
                if let Some(var_info) = self.variables.get(name) {
                    if !var_info.is_mutable {
                        return Err(LLVMError::ImmutableAssignment(name.clone()));
                    }
                    
                    let store_instruction = format!(
                        "  store {} {}, {}* {}",
                        var_info.var_type.to_llvm_string(),
                        self.extract_value(&value_val),
                        var_info.var_type.to_llvm_string(),
                        var_info.llvm_name
                    );
                    self.instructions.push(store_instruction);
                    Ok(value_val)
                } else {
                    Err(LLVMError::UndefinedVariable(name.clone()))
                }
            }
            PIRExpression::ArrayAccess { array, index } => {
                let array_val = self.generate_expression(array, location)?;
                let index_val = self.generate_expression(index, location)?;
                
                let ptr_temp = self.next_temp();
                let gep_instruction = format!(
                    "  {} = getelementptr inbounds i32, i32* {}, i32 {}",
                    ptr_temp,
                    self.extract_value(&array_val),
                    self.extract_value(&index_val)
                );
                self.instructions.push(gep_instruction);
                
                let store_instruction = format!(
                    "  store i32 {}, i32* {}",
                    self.extract_value(&value_val),
                    ptr_temp
                );
                self.instructions.push(store_instruction);
                Ok(value_val)
            }
            _ => Err(LLVMError::InvalidAssignmentTarget),
        }
    }

    /// Generate conditional expression
    fn generate_conditional(
        &mut self,
        condition: &PIRExpression,
        then_expr: &PIRExpression,
        else_expr: &PIRExpression,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<String> {
        let cond_val = self.generate_expression(condition, location)?;
        
        let then_block = self.next_block_label("cond.then");
        let else_block = self.next_block_label("cond.else");
        let merge_block = self.next_block_label("cond.end");
        
        // Branch instruction
        let branch_instruction = format!(
            "  br i1 {}, label %{}, label %{}",
            self.extract_value(&cond_val),
            then_block,
            else_block
        );
        self.instructions.push(branch_instruction);
        
        // Then block
        self.instructions.push(format!("{}:", then_block));
        let then_val = self.generate_expression(then_expr, location)?;
        self.instructions.push(format!("  br label %{}", merge_block));
        
        // Else block
        self.instructions.push(format!("{}:", else_block));
        let else_val = self.generate_expression(else_expr, location)?;
        self.instructions.push(format!("  br label %{}", merge_block));
        
        // Merge block
        self.instructions.push(format!("{}:", merge_block));
        let result_temp = self.next_temp();
        let phi_instruction = format!(
            "  {} = phi i32 [ {}, %{} ], [ {}, %{} ]",
            result_temp,
            self.extract_value(&then_val),
            then_block,
            self.extract_value(&else_val),
            else_block
        );
        self.instructions.push(phi_instruction);
        
        Ok(result_temp)
    }

    /// Generate variable declaration
    fn generate_variable_declaration(
        &mut self,
        name: &str,
        var_type: &str,
        initializer: Option<&PIRExpression>,
        is_mutable: bool,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<()> {
        let llvm_type = self.convert_type_name(var_type)?;
        let alloca_name = format!("%{}", name);
        
        // Generate alloca instruction
        let alloca_instruction = format!(
            "  {} = alloca {}, align {}",
            alloca_name,
            llvm_type.to_llvm_string(),
            llvm_type.alignment()
        );
        self.instructions.push(alloca_instruction);
        
        // Store variable information
        let var_info = VariableInfo {
            llvm_name: alloca_name.clone(),
            var_type: llvm_type.clone(),
            is_mutable,
            location: location.cloned(),
            is_parameter: false,
            scope_depth: 0, // Would be tracked properly in a full implementation
        };
        self.variables.insert(name.to_string(), var_info.clone());
        
        // Generate debug information
        if let Some(ref debug_info) = self.debug_info {
            if let Some(loc) = location {
                let debug_var = DebugVariable {
                    name: name.to_string(),
                    var_type: llvm_type.to_llvm_string(),
                    location: loc.clone(),
                    llvm_value: alloca_name.clone(),
                    scope: 0, // Would be tracked properly
                    is_parameter: false,
                    parameter_index: None,
                };
                
                if let Ok(debug_metadata) = debug_info.create_variable_metadata(&debug_var) {
                    let debug_declare = debug_info.generate_debug_declare(&debug_var, debug_metadata.id);
                    if !debug_declare.is_empty() {
                        self.instructions.push(format!("  {}", debug_declare));
                    }
                }
            }
        }
        
        // Handle initializer
        if let Some(init_expr) = initializer {
            let init_val = self.generate_expression(init_expr, location)?;
            let store_instruction = format!(
                "  store {} {}, {}* {}",
                llvm_type.to_llvm_string(),
                self.extract_value(&init_val),
                llvm_type.to_llvm_string(),
                alloca_name
            );
            self.instructions.push(store_instruction);
        }
        
        Ok(())
    }

    /// Generate if statement
    fn generate_if_statement(
        &mut self,
        condition: &PIRExpression,
        then_block: &[PIRStatement],
        else_block: Option<&Vec<PIRStatement>>,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<()> {
        let cond_val = self.generate_expression(condition, location)?;
        
        let then_label = self.next_block_label("if.then");
        let else_label = self.next_block_label("if.else");
        let end_label = self.next_block_label("if.end");
        
        // Branch to then or else
        let branch_target = if else_block.is_some() { &else_label } else { &end_label };
        let branch_instruction = format!(
            "  br i1 {}, label %{}, label %{}",
            self.extract_value(&cond_val),
            then_label,
            branch_target
        );
        self.instructions.push(branch_instruction);
        
        // Then block
        self.instructions.push(format!("{}:", then_label));
        for stmt in then_block {
            self.generate_statement(stmt, location)?;
        }
        self.instructions.push(format!("  br label %{}", end_label));
        
        // Else block (if present)
        if let Some(else_stmts) = else_block {
            self.instructions.push(format!("{}:", else_label));
            for stmt in else_stmts {
                self.generate_statement(stmt, location)?;
            }
            self.instructions.push(format!("  br label %{}", end_label));
        }
        
        // End block
        self.instructions.push(format!("{}:", end_label));
        
        Ok(())
    }

    /// Generate while loop
    fn generate_while_loop(
        &mut self,
        condition: &PIRExpression,
        body: &[PIRStatement],
        location: Option<&SourceLocation>,
    ) -> LLVMResult<()> {
        let loop_header = self.next_block_label("while.cond");
        let loop_body = self.next_block_label("while.body");
        let loop_end = self.next_block_label("while.end");
        
        // Jump to condition check
        self.instructions.push(format!("  br label %{}", loop_header));
        
        // Loop condition
        self.instructions.push(format!("{}:", loop_header));
        let cond_val = self.generate_expression(condition, location)?;
        let branch_instruction = format!(
            "  br i1 {}, label %{}, label %{}",
            self.extract_value(&cond_val),
            loop_body,
            loop_end
        );
        self.instructions.push(branch_instruction);
        
        // Loop body
        self.instructions.push(format!("{}:", loop_body));
        for stmt in body {
            self.generate_statement(stmt, location)?;
        }
        self.instructions.push(format!("  br label %{}", loop_header));
        
        // Loop end
        self.instructions.push(format!("{}:", loop_end));
        
        Ok(())
    }

    /// Generate for loop
    fn generate_for_loop(
        &mut self,
        init: Option<&PIRStatement>,
        condition: Option<&PIRExpression>,
        update: Option<&PIRExpression>,
        body: &[PIRStatement],
        location: Option<&SourceLocation>,
    ) -> LLVMResult<()> {
        // Initialize
        if let Some(init_stmt) = init {
            self.generate_statement(init_stmt, location)?;
        }
        
        let loop_header = self.next_block_label("for.cond");
        let loop_body = self.next_block_label("for.body");
        let loop_update = self.next_block_label("for.inc");
        let loop_end = self.next_block_label("for.end");
        
        // Jump to condition check
        self.instructions.push(format!("  br label %{}", loop_header));
        
        // Loop condition
        self.instructions.push(format!("{}:", loop_header));
        if let Some(cond) = condition {
            let cond_val = self.generate_expression(cond, location)?;
            let branch_instruction = format!(
                "  br i1 {}, label %{}, label %{}",
                self.extract_value(&cond_val),
                loop_body,
                loop_end
            );
            self.instructions.push(branch_instruction);
        } else {
            // Infinite loop
            self.instructions.push(format!("  br label %{}", loop_body));
        }
        
        // Loop body
        self.instructions.push(format!("{}:", loop_body));
        for stmt in body {
            self.generate_statement(stmt, location)?;
        }
        self.instructions.push(format!("  br label %{}", loop_update));
        
        // Loop update
        self.instructions.push(format!("{}:", loop_update));
        if let Some(update_expr) = update {
            self.generate_expression(update_expr, location)?;
        }
        self.instructions.push(format!("  br label %{}", loop_header));
        
        // Loop end
        self.instructions.push(format!("{}:", loop_end));
        
        Ok(())
    }

    /// Generate return statement
    fn generate_return_statement(
        &mut self,
        expr: Option<&PIRExpression>,
        location: Option<&SourceLocation>,
    ) -> LLVMResult<()> {
        if let Some(return_expr) = expr {
            let return_val = self.generate_expression(return_expr, location)?;
            let return_instruction = format!(
                "  ret i32 {}",
                self.extract_value(&return_val)
            );
            self.instructions.push(return_instruction);
        } else {
            self.instructions.push("  ret void".to_string());
        }
        Ok(())
    }

    /// Generate break statement
    fn generate_break_statement(&mut self, _location: Option<&SourceLocation>) -> LLVMResult<()> {
        // Would need loop context tracking in a full implementation
        self.instructions.push("  br label %loop.end".to_string());
        Ok(())
    }

    /// Generate continue statement
    fn generate_continue_statement(&mut self, _location: Option<&SourceLocation>) -> LLVMResult<()> {
        // Would need loop context tracking in a full implementation
        self.instructions.push("  br label %loop.continue".to_string());
        Ok(())
    }

    /// Generate block statement
    fn generate_block(
        &mut self,
        statements: &[PIRStatement],
        location: Option<&SourceLocation>,
    ) -> LLVMResult<()> {
        for stmt in statements {
            self.generate_statement(stmt, location)?;
        }
        Ok(())
    }

    /// Helper methods
    fn next_temp(&mut self) -> String {
        let temp = format!("%{}", self.temp_counter);
        self.temp_counter += 1;
        temp
    }

    fn next_block_label(&mut self, prefix: &str) -> String {
        let label = format!("{}.{}", prefix, self.block_counter);
        self.block_counter += 1;
        label
    }

    fn extract_value(&self, value: &str) -> &str {
        // Extract the actual value from "type value" format
        if let Some(space_pos) = value.find(' ') {
            &value[space_pos + 1..]
        } else {
            value
        }
    }

    fn convert_type_name(&self, type_name: &str) -> LLVMResult<LLVMType> {
        match type_name {
            "i32" => Ok(LLVMType::Integer(32)),
            "i64" => Ok(LLVMType::Integer(64)),
            "f32" => Ok(LLVMType::Float(32)),
            "f64" => Ok(LLVMType::Float(64)),
            "bool" => Ok(LLVMType::Integer(1)),
            _ => Err(LLVMError::UnsupportedType(type_name.to_string())),
        }
    }

    fn binary_op_to_llvm(&self, op: &BinaryOperator, left: &str, right: &str) -> LLVMResult<String> {
        let left_val = self.extract_value(left);
        let right_val = self.extract_value(right);
        
        let op_str = match op {
            BinaryOperator::Add => {
                if self.config.enable_overflow_checking {
                    format!("add nsw i32 {}, {}", left_val, right_val)
                } else {
                    format!("add i32 {}, {}", left_val, right_val)
                }
            }
            BinaryOperator::Sub => {
                if self.config.enable_overflow_checking {
                    format!("sub nsw i32 {}, {}", left_val, right_val)
                } else {
                    format!("sub i32 {}, {}", left_val, right_val)
                }
            }
            BinaryOperator::Mul => {
                if self.config.enable_overflow_checking {
                    format!("mul nsw i32 {}, {}", left_val, right_val)
                } else {
                    format!("mul i32 {}, {}", left_val, right_val)
                }
            }
            BinaryOperator::Div => format!("sdiv i32 {}, {}", left_val, right_val),
            BinaryOperator::Mod => format!("srem i32 {}, {}", left_val, right_val),
            BinaryOperator::Eq => format!("icmp eq i32 {}, {}", left_val, right_val),
            BinaryOperator::Ne => format!("icmp ne i32 {}, {}", left_val, right_val),
            BinaryOperator::Lt => format!("icmp slt i32 {}, {}", left_val, right_val),
            BinaryOperator::Le => format!("icmp sle i32 {}, {}", left_val, right_val),
            BinaryOperator::Gt => format!("icmp sgt i32 {}, {}", left_val, right_val),
            BinaryOperator::Ge => format!("icmp sge i32 {}, {}", left_val, right_val),
            BinaryOperator::And => format!("and i1 {}, {}", left_val, right_val),
            BinaryOperator::Or => format!("or i1 {}, {}", left_val, right_val),
            BinaryOperator::Xor => format!("xor i1 {}, {}", left_val, right_val),
            BinaryOperator::BitAnd => format!("and i32 {}, {}", left_val, right_val),
            BinaryOperator::BitOr => format!("or i32 {}, {}", left_val, right_val),
            BinaryOperator::BitXor => format!("xor i32 {}, {}", left_val, right_val),
            BinaryOperator::Shl => format!("shl i32 {}, {}", left_val, right_val),
            BinaryOperator::Shr => format!("ashr i32 {}, {}", left_val, right_val),
        };
        
        Ok(op_str)
    }

    fn generate_string_constant(&mut self, value: &str) -> LLVMResult<String> {
        let global_name = format!("@.str.{}", value.len());
        // This would be added to the global constants section
        // For now, just return the reference
        Ok(format!("getelementptr inbounds ([{} x i8], [{} x i8]* {}, i32 0, i32 0)", 
                   value.len() + 1, value.len() + 1, global_name))
    }

    /// Get generated instructions
    pub fn get_instructions(&self) -> &[String] {
        &self.instructions
    }

    /// Clear instructions buffer
    pub fn clear_instructions(&mut self) {
        self.instructions.clear();
    }

    /// Get current function context
    pub fn get_current_function(&self) -> Option<&String> {
        self.current_function.as_ref()
    }

    /// Set current function context
    pub fn set_current_function(&mut self, function: Option<String>) {
        self.current_function = function;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::runtime::LLVMRuntimeConfig;

    fn create_test_generator() -> LLVMInstructionGenerator {
        let config = LLVMInstructionConfig::default();
        let runtime_config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(runtime_config);
        
        LLVMInstructionGenerator::new(config, runtime, None)
    }

    #[test]
    fn test_literal_generation() {
        let mut generator = create_test_generator();
        
        let int_lit = PIRLiteral::Integer(42);
        let result = generator.generate_literal(&int_lit).unwrap();
        assert_eq!(result, "i32 42");
        
        let bool_lit = PIRLiteral::Boolean(true);
        let result = generator.generate_literal(&bool_lit).unwrap();
        assert_eq!(result, "i1 1");
        
        let float_lit = PIRLiteral::Float(3.14);
        let result = generator.generate_literal(&float_lit).unwrap();
        assert_eq!(result, "double 3.14");
    }

    #[test]
    fn test_binary_operation_generation() {
        let mut generator = create_test_generator();
        
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
    fn test_variable_declaration() {
        let mut generator = create_test_generator();
        
        let init_expr = PIRExpression::Literal(PIRLiteral::Integer(42));
        generator.generate_variable_declaration(
            "test_var",
            "i32",
            Some(&init_expr),
            true,
            None
        ).unwrap();
        
        let instructions = generator.get_instructions();
        assert!(instructions.iter().any(|i| i.contains("alloca")));
        assert!(instructions.iter().any(|i| i.contains("store")));
    }

    #[test]
    fn test_function_call_generation() {
        let mut generator = create_test_generator();
        
        let args = vec![
            PIRExpression::Literal(PIRLiteral::Integer(10)),
            PIRExpression::Literal(PIRLiteral::Integer(20)),
        ];
        
        let call_expr = PIRExpression::FunctionCall {
            function: "test_function".to_string(),
            args,
        };
        
        let result = generator.generate_expression(&call_expr, None).unwrap();
        assert!(result.starts_with('%'));
        
        let instructions = generator.get_instructions();
        assert!(instructions.iter().any(|i| i.contains("call")));
        assert!(instructions.iter().any(|i| i.contains("test_function")));
    }

    #[test]
    fn test_if_statement_generation() {
        let mut generator = create_test_generator();
        
        let condition = PIRExpression::Literal(PIRLiteral::Boolean(true));
        let then_stmt = vec![PIRStatement::Return(Some(PIRExpression::Literal(PIRLiteral::Integer(1))))];
        let else_stmt = vec![PIRStatement::Return(Some(PIRExpression::Literal(PIRLiteral::Integer(0))))];
        
        generator.generate_if_statement(&condition, &then_stmt, Some(&else_stmt), None).unwrap();
        
        let instructions = generator.get_instructions();
        assert!(instructions.iter().any(|i| i.contains("br i1")));
        assert!(instructions.iter().any(|i| i.contains("if.then")));
        assert!(instructions.iter().any(|i| i.contains("if.else")));
        assert!(instructions.iter().any(|i| i.contains("if.end")));
    }

    #[test]
    fn test_while_loop_generation() {
        let mut generator = create_test_generator();
        
        let condition = PIRExpression::Literal(PIRLiteral::Boolean(true));
        let body = vec![PIRStatement::Break];
        
        generator.generate_while_loop(&condition, &body, None).unwrap();
        
        let instructions = generator.get_instructions();
        assert!(instructions.iter().any(|i| i.contains("while.cond")));
        assert!(instructions.iter().any(|i| i.contains("while.body")));
        assert!(instructions.iter().any(|i| i.contains("while.end")));
    }

    #[test]
    fn test_overflow_checking() {
        let config = LLVMInstructionConfig {
            enable_overflow_checking: true,
            ..Default::default()
        };
        let runtime_config = LLVMRuntimeConfig::default();
        let runtime = LLVMRuntime::new(runtime_config);
        let mut generator = LLVMInstructionGenerator::new(config, runtime, None);
        
        let left = PIRExpression::Literal(PIRLiteral::Integer(10));
        let right = PIRExpression::Literal(PIRLiteral::Integer(20));
        let add_expr = PIRExpression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(left),
            right: Box::new(right),
        };
        
        generator.generate_expression(&add_expr, None).unwrap();
        
        let instructions = generator.get_instructions();
        assert!(instructions.iter().any(|i| i.contains("add nsw")));
    }
} 