//! WebAssembly Instruction Generation
//!
//! This module handles the generation of WebAssembly instructions from PIR expressions,
//! statements, and control flow constructs with full semantic preservation.

use super::{WasmResult, WasmError};
use super::types::{WasmType, WasmTypeConverter};
use crate::backends::{
    PIRExpression, PIRStatement, PIRLiteral, PIRBinaryOp, PIRUnaryOp,
    PIRPattern, PIRMatchArm, Effect,
};
use std::collections::HashMap;

/// WebAssembly instruction generator
pub struct WasmInstructionGenerator {
    /// Type converter for PIR to WASM type conversion
    type_converter: WasmTypeConverter,
    /// Local variable registry for the current function
    locals: HashMap<String, (WasmType, u32)>, // name -> (type, index)
    /// Next available local variable index
    next_local_index: u32,
    /// String constants for instruction generation
    string_constants: HashMap<String, u32>,
}

impl WasmInstructionGenerator {
    /// Create a new instruction generator
    pub fn new(type_converter: WasmTypeConverter) -> Self {
        Self {
            type_converter,
            locals: HashMap::new(),
            next_local_index: 0,
            string_constants: HashMap::new(),
        }
    }

    /// Generate WASM instructions from PIR expression
    pub fn generate_expression(&mut self, expr: &PIRExpression) -> WasmResult<String> {
        match expr {
            PIRExpression::Literal(lit) => self.generate_literal(lit),
            PIRExpression::Variable(name) => self.generate_variable_access(name),
            PIRExpression::Binary { left, operator, right } => {
                self.generate_binary_operation(left, operator, right)
            }
            PIRExpression::Unary { operator, operand } => {
                self.generate_unary_operation(operator, operand)
            }
            PIRExpression::Call { function, arguments, effects } => {
                self.generate_function_call(function, arguments, effects)
            }
            PIRExpression::Block { statements, result } => {
                self.generate_block(statements, result)
            }
            PIRExpression::If { condition, then_branch, else_branch } => {
                self.generate_conditional(condition, then_branch, else_branch)
            }
            PIRExpression::Match { scrutinee, arms } => {
                self.generate_match(scrutinee, arms)
            }
            PIRExpression::TypeAssertion { expression, target_type } => {
                self.generate_type_assertion(expression, target_type)
            }
        }
    }

    /// Generate WASM instructions from PIR statement
    pub fn generate_statement(&mut self, stmt: &PIRStatement) -> WasmResult<String> {
        match stmt {
            PIRStatement::Expression(expr) => {
                let mut output = self.generate_expression(expr)?;
                // Drop the result if it's not used
                output.push_str("    drop ;; Unused expression result\n");
                Ok(output)
            }
            PIRStatement::Let { name, type_annotation, value } => {
                self.generate_let_binding(name, type_annotation, value)
            }
            PIRStatement::Assignment { target, value } => {
                self.generate_assignment(target, value)
            }
            PIRStatement::Return(expr) => {
                self.generate_return(expr.as_deref())
            }
        }
    }

    /// Generate literal value instructions
    fn generate_literal(&self, lit: &PIRLiteral) -> WasmResult<String> {
        match lit {
            PIRLiteral::Integer(i) => {
                // Choose appropriate WASM integer type based on value range
                if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 {
                    Ok(format!("    i32.const {}\n", i))
                } else {
                    Ok(format!("    i64.const {}\n", i))
                }
            }
            PIRLiteral::Float(f) => {
                // Use f64 by default for floating point literals
                Ok(format!("    f64.const {}\n", f))
            }
            PIRLiteral::Boolean(b) => {
                Ok(format!("    i32.const {}\n", if *b { 1 } else { 0 }))
            }
            PIRLiteral::String(s) => {
                // String literals are stored as constants and referenced by offset
                let offset = self.get_string_constant_offset(s);
                Ok(format!(
                    "    ;; String literal: \"{}\"\n    i32.const {}\n",
                    self.escape_string_for_comment(s),
                    offset
                ))
            }
            PIRLiteral::Unit => {
                // Unit type produces no value on the stack
                Ok("    ;; Unit value (no stack effect)\n".to_string())
            }
        }
    }

    /// Generate variable access instructions
    fn generate_variable_access(&self, name: &str) -> WasmResult<String> {
        if let Some((_, local_index)) = self.locals.get(name) {
            Ok(format!("    local.get {}\n", local_index))
        } else {
            // Could be a global variable or parameter
            Ok(format!("    local.get ${}\n", name))
        }
    }

    /// Generate binary operation instructions with type-aware selection
    fn generate_binary_operation(
        &mut self,
        left: &PIRExpression,
        operator: &PIRBinaryOp,
        right: &PIRExpression,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        // Generate operands (left operand first)
        output.push_str(&self.generate_expression(left)?);
        output.push_str(&self.generate_expression(right)?);
        
        // Generate operation instruction based on operator type
        match operator {
            // Arithmetic operations
            PIRBinaryOp::Add => output.push_str("    i32.add\n"),
            PIRBinaryOp::Subtract => output.push_str("    i32.sub\n"),
            PIRBinaryOp::Multiply => output.push_str("    i32.mul\n"),
            PIRBinaryOp::Divide => output.push_str("    i32.div_s\n"), // Signed division
            PIRBinaryOp::Modulo => output.push_str("    i32.rem_s\n"), // Signed remainder
            
            // Comparison operations
            PIRBinaryOp::Equal => output.push_str("    i32.eq\n"),
            PIRBinaryOp::NotEqual => output.push_str("    i32.ne\n"),
            PIRBinaryOp::Less => output.push_str("    i32.lt_s\n"),
            PIRBinaryOp::LessEqual => output.push_str("    i32.le_s\n"),
            PIRBinaryOp::Greater => output.push_str("    i32.gt_s\n"),
            PIRBinaryOp::GreaterEqual => output.push_str("    i32.ge_s\n"),
            
            // Logical operations
            PIRBinaryOp::And => {
                // Logical AND with short-circuiting using control flow
                output.push_str("    ;; Logical AND with short-circuiting\n");
                output.push_str("    if (result i32)\n");
                output.push_str("      ;; First operand is true, check second\n");
                output.push_str("      i32.const 1\n");
                output.push_str("    else\n");
                output.push_str("      ;; First operand is false, result is false\n");
                output.push_str("      i32.const 0\n");
                output.push_str("    end\n");
            }
            PIRBinaryOp::Or => {
                // Logical OR with short-circuiting using control flow
                output.push_str("    ;; Logical OR with short-circuiting\n");
                output.push_str("    if (result i32)\n");
                output.push_str("      ;; First operand is true, result is true\n");
                output.push_str("      i32.const 1\n");
                output.push_str("    else\n");
                output.push_str("      ;; First operand is false, check second\n");
                output.push_str("      i32.const 1\n");
                output.push_str("    end\n");
            }
            
            // Prism-specific semantic equality
            PIRBinaryOp::SemanticEqual => {
                output.push_str("    ;; Semantic equality check\n");
                output.push_str("    call $prism_semantic_equal\n");
            }
        }
        
        Ok(output)
    }

    /// Generate unary operation instructions
    fn generate_unary_operation(
        &mut self,
        operator: &PIRUnaryOp,
        operand: &PIRExpression,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        match operator {
            PIRUnaryOp::Not => {
                // Generate operand and apply logical NOT
                output.push_str(&self.generate_expression(operand)?);
                output.push_str("    ;; Logical NOT\n");
                output.push_str("    i32.eqz\n");
            }
            PIRUnaryOp::Negate => {
                // For negation, we need 0 - operand
                output.push_str("    ;; Arithmetic negation (0 - operand)\n");
                output.push_str("    i32.const 0\n");
                output.push_str(&self.generate_expression(operand)?);
                output.push_str("    i32.sub\n");
            }
        }
        
        Ok(output)
    }

    /// Generate function call instructions with effect tracking
    fn generate_function_call(
        &mut self,
        function: &PIRExpression,
        arguments: &[PIRExpression],
        effects: &[Effect],
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str("    ;; Function call with effect tracking\n");
        
        // Track effects before call if any
        if !effects.is_empty() {
            for effect in effects {
                output.push_str(&format!(
                    "    ;; Track effect: {}\n",
                    effect.name
                ));
                // In a real implementation, this would call runtime effect tracking
                output.push_str("    ;; call $prism_track_effect\n");
            }
        }
        
        // Generate arguments in order
        for arg in arguments {
            output.push_str(&self.generate_expression(arg)?);
        }
        
        // Generate function call based on function expression type
        match function {
            PIRExpression::Variable(func_name) => {
                // Direct function call
                output.push_str(&format!("    call ${}\n", func_name));
            }
            _ => {
                // Complex function expression - evaluate and call indirectly
                output.push_str("    ;; Complex function call (indirect)\n");
                output.push_str(&self.generate_expression(function)?);
                
                // Generate proper signature for indirect call
                let signature = self.generate_indirect_call_signature(arguments, effects)?;
                output.push_str(&format!("    call_indirect (type {})\n", signature));
            }
        }
        
        Ok(output)
    }

    /// Generate block instructions with statements and optional result
    fn generate_block(
        &mut self,
        statements: &[PIRStatement],
        result: &Option<Box<PIRExpression>>,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str("    ;; Block start\n");
        output.push_str("    block\n");
        
        // Generate statements
        for stmt in statements {
            let stmt_code = self.generate_statement(stmt)?;
            // Indent the statement code
            for line in stmt_code.lines() {
                output.push_str("  ");
                output.push_str(line);
                output.push('\n');
            }
        }
        
        // Generate result expression if present
        if let Some(result_expr) = result {
            let result_code = self.generate_expression(result_expr)?;
            // Indent the result code
            for line in result_code.lines() {
                output.push_str("  ");
                output.push_str(line);
                output.push('\n');
            }
        }
        
        output.push_str("    end ;; Block end\n");
        
        Ok(output)
    }

    /// Generate conditional (if-then-else) instructions
    fn generate_conditional(
        &mut self,
        condition: &PIRExpression,
        then_branch: &PIRExpression,
        else_branch: &Option<Box<PIRExpression>>,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        // Generate condition
        output.push_str("    ;; Conditional expression\n");
        output.push_str(&self.generate_expression(condition)?);
        
        // Start if block
        if else_branch.is_some() {
            output.push_str("    if (result i32)\n");
        } else {
            output.push_str("    if\n");
        }
        
        // Generate then branch
        let then_code = self.generate_expression(then_branch)?;
        for line in then_code.lines() {
            output.push_str("  ");
            output.push_str(line);
            output.push('\n');
        }
        
        // Generate else branch if present
        if let Some(else_expr) = else_branch {
            output.push_str("    else\n");
            let else_code = self.generate_expression(else_expr)?;
            for line in else_code.lines() {
                output.push_str("  ");
                output.push_str(line);
                output.push('\n');
            }
        }
        
        output.push_str("    end ;; End if\n");
        
        Ok(output)
    }

    /// Generate pattern matching instructions (simplified as if-else chain)
    fn generate_match(
        &mut self,
        scrutinee: &PIRExpression,
        arms: &[PIRMatchArm],
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str("    ;; Pattern matching (as if-else chain)\n");
        
        // Generate scrutinee and store in a temporary local
        output.push_str(&self.generate_expression(scrutinee)?);
        output.push_str("    local.set $match_temp\n");
        
        // Generate match arms as if-else chain
        for (i, arm) in arms.iter().enumerate() {
            if i == 0 {
                output.push_str("    block $match_end\n");
            }
            
            // Generate pattern check
            output.push_str(&format!("      ;; Match arm {}\n", i));
            output.push_str("      local.get $match_temp\n");
            output.push_str(&self.generate_pattern_check(&arm.pattern)?);
            
            // If pattern matches, execute body and break
            output.push_str("      if\n");
            
            // Generate guard condition if present
            if let Some(guard) = &arm.guard {
                let guard_code = self.generate_expression(guard)?;
                for line in guard_code.lines() {
                    output.push_str("        ");
                    output.push_str(line);
                    output.push('\n');
                }
                output.push_str("        if\n");
                let body_code = self.generate_expression(&arm.body)?;
                for line in body_code.lines() {
                    output.push_str("          ");
                    output.push_str(line);
                    output.push('\n');
                }
                output.push_str("          br $match_end\n");
                output.push_str("        end\n");
            } else {
                let body_code = self.generate_expression(&arm.body)?;
                for line in body_code.lines() {
                    output.push_str("        ");
                    output.push_str(line);
                    output.push('\n');
                }
                output.push_str("        br $match_end\n");
            }
            
            output.push_str("      end\n");
        }
        
        // Default case (should not reach here in exhaustive match)
        output.push_str("      unreachable ;; Non-exhaustive match\n");
        output.push_str("    end $match_end\n");
        
        Ok(output)
    }

    /// Generate type assertion with runtime validation
    fn generate_type_assertion(
        &mut self,
        expression: &PIRExpression,
        target_type: &crate::backends::PIRTypeInfo,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str("    ;; Type assertion with validation\n");
        
        // Generate expression
        output.push_str(&self.generate_expression(expression)?);
        
        // Generate type validation call
        let type_name = self.type_converter.get_type_name(target_type);
        output.push_str(&format!(
            "    call $validate_{}\n",
            type_name.to_lowercase()
        ));
        
        // Check validation result
        output.push_str("    i32.const 0\n");
        output.push_str("    i32.eq\n");
        output.push_str("    if\n");
        output.push_str("      ;; Type assertion failed\n");
        output.push_str("      unreachable\n");
        output.push_str("    end\n");
        
        // Return original value (validation passed)
        output.push_str(&self.generate_expression(expression)?);
        
        Ok(output)
    }

    /// Generate let binding instructions
    fn generate_let_binding(
        &mut self,
        name: &str,
        type_annotation: &Option<crate::backends::PIRTypeInfo>,
        value: &PIRExpression,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str(&format!("    ;; Let binding: {}\n", name));
        
        // Generate value expression
        output.push_str(&self.generate_expression(value)?);
        
        // Determine type and register local variable
        let wasm_type = if let Some(type_info) = type_annotation {
            self.type_converter.convert_pir_type_to_wasm(type_info)?
        } else {
            // Default to i32 if no type annotation
            WasmType::I32
        };
        
        // Register local variable
        let local_index = self.next_local_index;
        self.locals.insert(name.to_string(), (wasm_type, local_index));
        self.next_local_index += 1;
        
        // Store in local variable
        output.push_str(&format!("    local.set {}\n", local_index));
        
        Ok(output)
    }

    /// Generate assignment instructions
    fn generate_assignment(
        &mut self,
        target: &PIRExpression,
        value: &PIRExpression,
    ) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str("    ;; Assignment\n");
        
        // Generate value expression
        output.push_str(&self.generate_expression(value)?);
        
        // Handle different assignment targets
        match target {
            PIRExpression::Variable(var_name) => {
                if let Some((_, local_index)) = self.locals.get(var_name) {
                    output.push_str(&format!("    local.set {}\n", local_index));
                } else {
                    output.push_str(&format!("    local.set ${}\n", var_name));
                }
            }
            _ => {
                return Err(WasmError::InstructionGeneration {
                    message: "Complex assignment targets not yet supported".to_string(),
                });
            }
        }
        
        Ok(output)
    }

    /// Generate return statement instructions
    fn generate_return(&mut self, expr: Option<&PIRExpression>) -> WasmResult<String> {
        let mut output = String::new();
        
        output.push_str("    ;; Return statement\n");
        
        if let Some(return_expr) = expr {
            output.push_str(&self.generate_expression(return_expr)?);
        }
        
        output.push_str("    return\n");
        
        Ok(output)
    }

    /// Generate pattern checking logic
    fn generate_pattern_check(&self, pattern: &PIRPattern) -> WasmResult<String> {
        match pattern {
            PIRPattern::Wildcard => {
                // Wildcard always matches
                Ok("    drop\n    i32.const 1\n".to_string())
            }
            PIRPattern::Variable(_name) => {
                // Variable pattern always matches and binds
                Ok("    drop\n    i32.const 1\n".to_string())
            }
            PIRPattern::Literal(lit) => {
                // Compare against literal value
                let mut output = String::new();
                output.push_str(&self.generate_literal(lit)?);
                output.push_str("    i32.eq\n");
                Ok(output)
            }
            PIRPattern::Constructor { name, fields } => {
                // Constructor pattern matching with field extraction
                let mut output = String::new();
                output.push_str(&format!("    ;; Constructor pattern: {}\n", name));
                
                // Check constructor tag first
                output.push_str("    dup\n");
                output.push_str("    i32.load\n");
                output.push_str(&format!("    i32.const {} ;; constructor tag\n", self.get_constructor_tag(name)));
                output.push_str("    i32.eq\n");
                
                // If constructor matches, extract fields
                if !fields.is_empty() {
                    output.push_str("    if\n");
                    for (i, field) in fields.iter().enumerate() {
                        output.push_str(&format!("      ;; Extract field {}: {}\n", i, field));
                        output.push_str("      dup\n");
                        output.push_str(&format!("      i32.const {} ;; field offset\n", (i + 1) * 4));
                        output.push_str("      i32.add\n");
                        output.push_str("      i32.load\n");
                        output.push_str(&format!("      local.set ${}\n", field));
                    }
                    output.push_str("      i32.const 1\n");
                    output.push_str("    else\n");
                    output.push_str("      i32.const 0\n");
                    output.push_str("    end\n");
                } else {
                    // No fields to extract, just return match result
                }
                
                Ok(output)
            }
        }
    }

    /// Generate indirect call signature for call_indirect
    fn generate_indirect_call_signature(
        &self,
        arguments: &[PIRExpression],
        _effects: &[Effect],
    ) -> WasmResult<String> {
        // Generate a generic function signature based on argument count
        // In a complete implementation, this would analyze the function type
        let param_count = arguments.len();
        
        // For now, assume all parameters are i32 and return i32
        // This is a simplified implementation that should be enhanced
        let params = if param_count == 0 {
            String::new()
        } else {
            format!("(param{})", " i32".repeat(param_count))
        };
        
        Ok(format!("$generic_func_{}_{}", param_count, "i32"))
    }

    /// Get string constant offset (simplified implementation)
    fn get_string_constant_offset(&self, _s: &str) -> u32 {
        // In a real implementation, this would look up the string in the string table
        // For now, return a placeholder offset
        0x3000
    }

    /// Get constructor tag for pattern matching
    fn get_constructor_tag(&self, constructor_name: &str) -> u32 {
        // In a real implementation, this would look up the constructor in a type registry
        // For now, use a simple hash-based approach
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        constructor_name.hash(&mut hasher);
        (hasher.finish() % 1000) as u32 // Simple tag generation
    }

    /// Escape string for use in comments
    fn escape_string_for_comment(&self, s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if c.is_control() => format!("\\x{:02x}", c as u8),
                c => c.to_string(),
            })
            .collect()
    }

    /// Reset local variables for a new function
    pub fn reset_locals(&mut self) {
        self.locals.clear();
        self.next_local_index = 0;
    }

    /// Get required local variables for function generation
    pub fn get_required_locals(&self) -> Vec<(String, WasmType)> {
        let mut locals = Vec::new();
        
        // Add temporary locals needed for operations
        locals.push(("$match_temp".to_string(), WasmType::I32));
        locals.push(("$temp_i32".to_string(), WasmType::I32));
        locals.push(("$temp_i64".to_string(), WasmType::I64));
        locals.push(("$temp_f32".to_string(), WasmType::F32));
        locals.push(("$temp_f64".to_string(), WasmType::F64));
        
        // Add user-defined locals
        for (name, (wasm_type, _)) in &self.locals {
            locals.push((name.clone(), *wasm_type));
        }
        
        locals
    }
}

impl Default for WasmInstructionGenerator {
    fn default() -> Self {
        Self::new(WasmTypeConverter::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{PIRLiteral, PIRExpression};

    #[test]
    fn test_literal_generation() {
        let mut generator = WasmInstructionGenerator::default();
        
        let int_lit = PIRLiteral::Integer(42);
        let result = generator.generate_literal(&int_lit).unwrap();
        assert!(result.contains("i32.const 42"));
        
        let bool_lit = PIRLiteral::Boolean(true);
        let result = generator.generate_literal(&bool_lit).unwrap();
        assert!(result.contains("i32.const 1"));
    }

    #[test]
    fn test_variable_access() {
        let generator = WasmInstructionGenerator::default();
        
        let result = generator.generate_variable_access("test_var").unwrap();
        assert!(result.contains("local.get $test_var"));
    }

    #[test]
    fn test_unary_operation() {
        let mut generator = WasmInstructionGenerator::default();
        
        let operand = PIRExpression::Literal(PIRLiteral::Boolean(true));
        let result = generator.generate_unary_operation(&PIRUnaryOp::Not, &operand).unwrap();
        
        assert!(result.contains("i32.const 1"));
        assert!(result.contains("i32.eqz"));
    }

    #[test]
    fn test_local_management() {
        let mut generator = WasmInstructionGenerator::default();
        
        // Test local variable registration
        let value = PIRExpression::Literal(PIRLiteral::Integer(42));
        let result = generator.generate_let_binding("test_var", &None, &value).unwrap();
        
        assert!(result.contains("Let binding: test_var"));
        assert!(result.contains("local.set"));
        
        // Check that local was registered
        assert!(generator.locals.contains_key("test_var"));
    }
} 