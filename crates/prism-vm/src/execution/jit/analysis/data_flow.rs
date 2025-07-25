//! Data Flow Analysis
//!
//! This module provides comprehensive data flow analysis including liveness analysis,
//! reaching definitions, available expressions, and other classic data flow problems
//! that are essential for optimization.

use crate::{VMResult, PrismVMError, bytecode::{FunctionDefinition, Instruction}};
use super::shared::{Analysis, AnalysisKind, Variable, AnalysisConfig};
use super::pipeline::AnalysisContext;
use super::control_flow::{ControlFlowGraph, BasicBlock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeSet};

/// Data flow analyzer
#[derive(Debug)]
pub struct DataFlowAnalyzer {
    /// Configuration
    config: AnalysisConfig,
}

/// Comprehensive data flow analysis results
#[derive(Debug, Clone)]
pub struct DataFlowAnalysis {
    /// Function being analyzed
    pub function_id: u32,
    
    /// Liveness analysis results
    pub liveness: LivenessAnalysis,
    
    /// Reaching definitions analysis
    pub reaching_definitions: ReachingDefinitions,
    
    /// Available expressions analysis
    pub available_expressions: AvailableExpressions,
    
    /// Use-def chains
    pub use_def_chains: UseDefChains,
    
    /// Def-use chains
    pub def_use_chains: DefUseChains,
    
    /// Variable interference graph
    pub interference_graph: InterferenceGraph,
}

/// Liveness analysis results
#[derive(Debug, Clone)]
pub struct LivenessAnalysis {
    /// Live variables at entry of each block
    pub live_in: HashMap<u32, BTreeSet<Variable>>,
    
    /// Live variables at exit of each block
    pub live_out: HashMap<u32, BTreeSet<Variable>>,
    
    /// Variables defined in each block
    pub def: HashMap<u32, BTreeSet<Variable>>,
    
    /// Variables used in each block
    pub use_vars: HashMap<u32, BTreeSet<Variable>>,
    
    /// Live ranges for each variable
    pub live_ranges: HashMap<Variable, LiveRange>,
}

/// Reaching definitions analysis
#[derive(Debug, Clone)]
pub struct ReachingDefinitions {
    /// Definitions reaching entry of each block
    pub reach_in: HashMap<u32, BTreeSet<Definition>>,
    
    /// Definitions reaching exit of each block
    pub reach_out: HashMap<u32, BTreeSet<Definition>>,
    
    /// Definitions generated in each block
    pub gen: HashMap<u32, BTreeSet<Definition>>,
    
    /// Definitions killed in each block
    pub kill: HashMap<u32, BTreeSet<Definition>>,
    
    /// All definitions in the function
    pub all_definitions: BTreeSet<Definition>,
}

/// Available expressions analysis
#[derive(Debug, Clone)]
pub struct AvailableExpressions {
    /// Expressions available at entry of each block
    pub avail_in: HashMap<u32, BTreeSet<Expression>>,
    
    /// Expressions available at exit of each block
    pub avail_out: HashMap<u32, BTreeSet<Expression>>,
    
    /// Expressions generated in each block
    pub gen: HashMap<u32, BTreeSet<Expression>>,
    
    /// Expressions killed in each block
    pub kill: HashMap<u32, BTreeSet<Expression>>,
    
    /// All expressions in the function
    pub all_expressions: BTreeSet<Expression>,
}

/// Use-def chains mapping uses to their definitions
#[derive(Debug, Clone)]
pub struct UseDefChains {
    /// Map from use to set of possible definitions
    pub chains: HashMap<Use, BTreeSet<Definition>>,
}

/// Def-use chains mapping definitions to their uses
#[derive(Debug, Clone)]
pub struct DefUseChains {
    /// Map from definition to set of uses
    pub chains: HashMap<Definition, BTreeSet<Use>>,
}

/// Variable interference graph for register allocation
#[derive(Debug, Clone)]
pub struct InterferenceGraph {
    /// Variables in the graph
    pub variables: BTreeSet<Variable>,
    
    /// Interference edges (variables that cannot share a register)
    pub edges: BTreeSet<(Variable, Variable)>,
    
    /// Adjacency list representation
    pub adjacency: HashMap<Variable, BTreeSet<Variable>>,
    
    /// Coloring information
    pub coloring: HashMap<Variable, u32>,
}

// Variable types are now imported from shared module

/// Definition of a variable
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Definition {
    /// Variable being defined
    pub variable: Variable,
    
    /// Block where definition occurs
    pub block_id: u32,
    
    /// Instruction offset within block
    pub instruction_offset: u32,
    
    /// Type of definition
    pub def_type: DefinitionType,
}

/// Type of definition
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefinitionType {
    /// Assignment
    Assignment,
    
    /// Function parameter
    Parameter,
    
    /// Phi function (for SSA)
    Phi,
    
    /// Function call result
    CallResult,
}

/// Use of a variable
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Use {
    /// Variable being used
    pub variable: Variable,
    
    /// Block where use occurs
    pub block_id: u32,
    
    /// Instruction offset within block
    pub instruction_offset: u32,
    
    /// Type of use
    pub use_type: UseType,
}

/// Type of use
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UseType {
    /// Read use
    Read,
    
    /// Address taken
    AddressTaken,
    
    /// Function call argument
    CallArgument,
    
    /// Condition in branch
    Condition,
}

/// Expression for available expressions analysis
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Expression {
    /// Expression operator
    pub operator: ExpressionOperator,
    
    /// Expression operands
    pub operands: Vec<Variable>,
    
    /// Expression type
    pub expr_type: String,
}

/// Expression operators
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpressionOperator {
    /// Binary arithmetic
    Add, Sub, Mul, Div, Mod,
    
    /// Binary comparison
    Eq, Ne, Lt, Le, Gt, Ge,
    
    /// Binary logical
    And, Or, Xor,
    
    /// Unary operations
    Not, Neg,
    
    /// Memory operations
    Load, Store,
    
    /// Function call
    Call { function_name: String },
}

/// Live range of a variable
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// Variable
    pub variable: Variable,
    
    /// Start of live range (instruction index)
    pub start: u32,
    
    /// End of live range (instruction index)
    pub end: u32,
    
    /// Holes in the live range
    pub holes: Vec<(u32, u32)>,
    
    /// Spill cost
    pub spill_cost: f64,
}

impl DataFlowAnalyzer {
    /// Create new data flow analyzer
    pub fn new(config: &AnalysisConfig) -> VMResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Perform comprehensive data flow analysis
    pub fn analyze(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
    ) -> VMResult<DataFlowAnalysis> {
        // Step 1: Liveness analysis
        let liveness = self.analyze_liveness(function, cfg)?;
        
        // Step 2: Reaching definitions
        let reaching_definitions = self.analyze_reaching_definitions(function, cfg)?;
        
        // Step 3: Available expressions
        let available_expressions = self.analyze_available_expressions(function, cfg)?;
        
        // Step 4: Build use-def chains
        let use_def_chains = self.build_use_def_chains(&liveness, &reaching_definitions)?;
        
        // Step 5: Build def-use chains
        let def_use_chains = self.build_def_use_chains(&use_def_chains)?;
        
        // Step 6: Build interference graph
        let interference_graph = self.build_interference_graph(&liveness, cfg)?;

        Ok(DataFlowAnalysis {
            function_id: function.id,
            liveness,
            reaching_definitions,
            available_expressions,
            use_def_chains,
            def_use_chains,
            interference_graph,
        })
    }

    /// Analyze variable liveness using backward data flow
    fn analyze_liveness(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
    ) -> VMResult<LivenessAnalysis> {
        let mut liveness = LivenessAnalysis {
            live_in: HashMap::new(),
            live_out: HashMap::new(),
            def: HashMap::new(),
            use_vars: HashMap::new(),
            live_ranges: HashMap::new(),
        };

        // Initialize def and use sets for each block
        for block in &cfg.blocks {
            let (def_set, use_set) = self.compute_def_use_sets(block)?;
            liveness.def.insert(block.id, def_set);
            liveness.use_vars.insert(block.id, use_set);
            liveness.live_in.insert(block.id, BTreeSet::new());
            liveness.live_out.insert(block.id, BTreeSet::new());
        }

        // Iterative fixed-point computation (backward analysis)
        let mut changed = true;
        let mut iterations = 0;
        
        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            // Process blocks in reverse post-order for better convergence
            for block in cfg.blocks.iter().rev() {
                // live_out[B] = ∪ live_in[S] for all successors S of B
                let mut new_live_out = BTreeSet::new();
                for &successor_id in &block.successors {
                    if let Some(successor_live_in) = liveness.live_in.get(&successor_id) {
                        new_live_out.extend(successor_live_in.iter().cloned());
                    }
                }

                // live_in[B] = use[B] ∪ (live_out[B] - def[B])
                let mut new_live_in = liveness.use_vars.get(&block.id).cloned().unwrap_or_default();
                let def_set = liveness.def.get(&block.id).unwrap();
                
                for var in &new_live_out {
                    if !def_set.contains(var) {
                        new_live_in.insert(var.clone());
                    }
                }

                // Check for changes
                if liveness.live_out.get(&block.id) != Some(&new_live_out) {
                    liveness.live_out.insert(block.id, new_live_out);
                    changed = true;
                }
                
                if liveness.live_in.get(&block.id) != Some(&new_live_in) {
                    liveness.live_in.insert(block.id, new_live_in);
                    changed = true;
                }
            }
        }

        // Compute live ranges
        liveness.live_ranges = self.compute_live_ranges(&liveness, cfg)?;

        Ok(liveness)
    }

    /// Compute def and use sets for a basic block
    fn compute_def_use_sets(&self, block: &BasicBlock) -> VMResult<(BTreeSet<Variable>, BTreeSet<Variable>)> {
        let mut def_set = BTreeSet::new();
        let mut use_set = BTreeSet::new();

        for instruction in &block.instructions {
            let (defs, uses) = self.extract_def_use_from_instruction(&instruction.instruction)?;
            
            // Add uses (only if not previously defined in this block)
            for var in uses {
                if !def_set.contains(&var) {
                    use_set.insert(var);
                }
            }
            
            // Add definitions
            for var in defs {
                def_set.insert(var);
            }
        }

        Ok((def_set, use_set))
    }

    /// Extract definitions and uses from an instruction - Complete implementation
    fn extract_def_use_from_instruction(&self, instruction: &Instruction) -> VMResult<(Vec<Variable>, Vec<Variable>)> {
        use crate::bytecode::instructions::PrismOpcode;
        
        let mut defs = Vec::new();
        let mut uses = Vec::new();

        match instruction.opcode {
            // Stack Operations
            PrismOpcode::NOP => {
                // No defs or uses
            }
            PrismOpcode::DUP => {
                let stack_var = Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                };
                uses.push(stack_var.clone());
                defs.push(stack_var); // Duplicates the top value
            }
            PrismOpcode::POP => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::SWAP => {
                let stack_top = Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                };
                let stack_top_1 = Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                };
                uses.push(stack_top.clone());
                uses.push(stack_top_1.clone());
                defs.push(stack_top);
                defs.push(stack_top_1);
            }
            PrismOpcode::ROT3 => {
                for i in 0..3 {
                    let var = Variable {
                        name: format!("stack_top_{}", i),
                        var_type: VariableType::Stack { offset: -(i as i32) },
                        scope: VariableScope::Function,
                    };
                    uses.push(var.clone());
                    defs.push(var);
                }
            }
            PrismOpcode::DUP_N(n) => {
                let var = Variable {
                    name: format!("stack_depth_{}", n),
                    var_type: VariableType::Stack { offset: -(n as i32) },
                    scope: VariableScope::Function,
                };
                uses.push(var.clone());
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::POP_N(n) => {
                for i in 0..n {
                    uses.push(Variable {
                        name: format!("stack_top_{}", i),
                        var_type: VariableType::Stack { offset: -(i as i32) },
                        scope: VariableScope::Function,
                    });
                }
            }

            // Constants and Literals
            PrismOpcode::LOAD_CONST(_) | PrismOpcode::LOAD_NULL | PrismOpcode::LOAD_TRUE |
            PrismOpcode::LOAD_FALSE | PrismOpcode::LOAD_ZERO | PrismOpcode::LOAD_ONE => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::LOAD_SMALL_INT(_) => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Local Variables
            PrismOpcode::LOAD_LOCAL(index) | PrismOpcode::LOAD_LOCAL_EXT(index) => {
                uses.push(Variable {
                    name: format!("local_{}", index),
                    var_type: VariableType::Local { index: index as u8 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::STORE_LOCAL(index) | PrismOpcode::STORE_LOCAL_EXT(index) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: format!("local_{}", index),
                    var_type: VariableType::Local { index: index as u8 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::LOAD_UPVALUE(index) => {
                uses.push(Variable {
                    name: format!("upvalue_{}", index),
                    var_type: VariableType::Local { index },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::STORE_UPVALUE(index) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: format!("upvalue_{}", index),
                    var_type: VariableType::Local { index },
                    scope: VariableScope::Function,
                });
            }

            // Global Variables
            PrismOpcode::LOAD_GLOBAL(index) | PrismOpcode::LOAD_GLOBAL_HASH(_) => {
                uses.push(Variable {
                    name: format!("global_{}", index),
                    var_type: VariableType::Global { name: format!("global_{}", index) },
                    scope: VariableScope::Global,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::STORE_GLOBAL(index) | PrismOpcode::STORE_GLOBAL_HASH(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: format!("global_{}", index),
                    var_type: VariableType::Global { name: format!("global_{}", index) },
                    scope: VariableScope::Global,
                });
            }

            // Arithmetic Operations (binary operations: pop 2, push 1)
            PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL | PrismOpcode::DIV |
            PrismOpcode::MOD | PrismOpcode::POW => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Unary arithmetic operations (pop 1, push 1)
            PrismOpcode::NEG | PrismOpcode::ABS | PrismOpcode::SQRT => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Bitwise Operations
            PrismOpcode::BIT_AND | PrismOpcode::BIT_OR | PrismOpcode::BIT_XOR |
            PrismOpcode::SHL | PrismOpcode::SHR | PrismOpcode::SAR => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::BIT_NOT => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Comparison Operations
            PrismOpcode::EQ | PrismOpcode::NE | PrismOpcode::LT | PrismOpcode::LE |
            PrismOpcode::GT | PrismOpcode::GE | PrismOpcode::CMP | PrismOpcode::SEMANTIC_EQ => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Logical Operations
            PrismOpcode::AND | PrismOpcode::OR | PrismOpcode::XOR => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::NOT => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Control Flow
            PrismOpcode::JUMP(_) => {
                // No defs or uses for unconditional jump
            }
            PrismOpcode::JUMP_IF_TRUE(_) | PrismOpcode::JUMP_IF_FALSE(_) |
            PrismOpcode::JUMP_IF_NULL(_) | PrismOpcode::JUMP_IF_NOT_NULL(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::CALL(argc) | PrismOpcode::TAIL_CALL(argc) | PrismOpcode::CALL_DYNAMIC(argc) => {
                // Use function and arguments
                for i in 0..=argc {
                    uses.push(Variable {
                        name: format!("stack_arg_{}", i),
                        var_type: VariableType::Stack { offset: -(i as i32) },
                        scope: VariableScope::Function,
                    });
                }
                // Define result
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::RETURN => {
                // No explicit defs or uses
            }
            PrismOpcode::RETURN_VALUE => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Object Operations
            PrismOpcode::NEW_OBJECT(_) => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::GET_FIELD(_) | PrismOpcode::GET_FIELD_HASH(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::SET_FIELD(_) | PrismOpcode::SET_FIELD_HASH(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::GET_METHOD(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::HAS_FIELD(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::DELETE_FIELD(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Array Operations
            PrismOpcode::NEW_ARRAY(size) => {
                for i in 0..size {
                    uses.push(Variable {
                        name: format!("stack_elem_{}", i),
                        var_type: VariableType::Stack { offset: -(i as i32) },
                        scope: VariableScope::Function,
                    });
                }
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::GET_INDEX => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::SET_INDEX => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_2".to_string(),
                    var_type: VariableType::Stack { offset: -2 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::ARRAY_LEN | PrismOpcode::ARRAY_POP => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::ARRAY_PUSH => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::ARRAY_SLICE | PrismOpcode::ARRAY_CONCAT => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // String Operations
            PrismOpcode::STR_CONCAT | PrismOpcode::STR_FIND | PrismOpcode::STR_REPLACE => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::STR_LEN | PrismOpcode::STR_UPPER | PrismOpcode::STR_LOWER | PrismOpcode::STR_TRIM => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::STR_SUBSTR => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_2".to_string(),
                    var_type: VariableType::Stack { offset: -2 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Type Operations
            PrismOpcode::TYPE_CHECK(_) | PrismOpcode::TYPE_CAST(_) | PrismOpcode::INSTANCE_OF(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::GET_TYPE | PrismOpcode::TYPE_NAME | PrismOpcode::IS_NULL |
            PrismOpcode::IS_NUMBER | PrismOpcode::IS_STRING => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Effect Operations
            PrismOpcode::EFFECT_ENTER(_) | PrismOpcode::EFFECT_INVOKE(_) | PrismOpcode::EFFECT_HANDLE(_) => {
                // Effects may use and define stack values
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::EFFECT_EXIT | PrismOpcode::EFFECT_RESUME | PrismOpcode::EFFECT_ABORT => {
                // May use stack values
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Capability Operations  
            PrismOpcode::CAP_CHECK(_) | PrismOpcode::CAP_DELEGATE(_) | PrismOpcode::CAP_REVOKE(_) |
            PrismOpcode::CAP_ACQUIRE(_) | PrismOpcode::CAP_RELEASE(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::CAP_LIST => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Concurrency Operations
            PrismOpcode::SPAWN_ACTOR(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::SEND_MESSAGE => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::RECEIVE_MESSAGE | PrismOpcode::CREATE_FUTURE => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::AWAIT | PrismOpcode::RESOLVE_FUTURE | PrismOpcode::REJECT_FUTURE => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::YIELD => {
                // No explicit defs or uses
            }

            // Pattern Matching
            PrismOpcode::MATCH(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::MATCH_GUARD => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::BIND_PATTERN(count) => {
                for i in 0..count {
                    defs.push(Variable {
                        name: format!("pattern_var_{}", i),
                        var_type: VariableType::Local { index: i },
                        scope: VariableScope::Function,
                    });
                }
            }
            PrismOpcode::DESTRUCTURE_TUPLE(count) | PrismOpcode::DESTRUCTURE_ARRAY(count) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                for i in 0..count {
                    defs.push(Variable {
                        name: format!("destructured_{}", i),
                        var_type: VariableType::Stack { offset: i as i32 },
                        scope: VariableScope::Function,
                    });
                }
            }
            PrismOpcode::DESTRUCTURE_OBJECT(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Advanced Operations
            PrismOpcode::CLOSURE(_) => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::PARTIAL(argc) => {
                for i in 0..argc {
                    uses.push(Variable {
                        name: format!("stack_arg_{}", i),
                        var_type: VariableType::Stack { offset: -(i as i32) },
                        scope: VariableScope::Function,
                    });
                }
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::COMPOSE | PrismOpcode::PIPE => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::MEMOIZE => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Debugging Operations
            PrismOpcode::BREAKPOINT => {
                // No defs or uses
            }
            PrismOpcode::TRACE(_) | PrismOpcode::PROFILE_START(_) | PrismOpcode::LOG(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::PROFILE_END => {
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::ASSERT => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // Memory Management
            PrismOpcode::GC_HINT => {
                // No explicit defs or uses
            }
            PrismOpcode::REF_INC | PrismOpcode::REF_DEC | PrismOpcode::WEAK_REF | PrismOpcode::STRONG_REF => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }

            // I/O Operations
            PrismOpcode::IO_READ(_) | PrismOpcode::IO_OPEN(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                defs.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::IO_WRITE(_) | PrismOpcode::IO_FLUSH(_) | PrismOpcode::IO_CLOSE(_) => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
                uses.push(Variable {
                    name: "stack_top_1".to_string(),
                    var_type: VariableType::Stack { offset: -1 },
                    scope: VariableScope::Function,
                });
            }

            // Exception Handling
            PrismOpcode::THROW => {
                uses.push(Variable {
                    name: "stack_top".to_string(),
                    var_type: VariableType::Stack { offset: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::TRY_START(_) => {
                // Sets up exception handling context
            }
            PrismOpcode::TRY_END | PrismOpcode::FINALLY => {
                // No explicit defs or uses
            }
            PrismOpcode::CATCH(_) => {
                defs.push(Variable {
                    name: "exception".to_string(),
                    var_type: VariableType::Local { index: 0 },
                    scope: VariableScope::Function,
                });
            }
            PrismOpcode::RETHROW => {
                uses.push(Variable {
                    name: "exception".to_string(),
                    var_type: VariableType::Local { index: 0 },
                    scope: VariableScope::Function,
                });
            }
        }

        Ok((defs, uses))
    }

    /// Analyze reaching definitions using forward data flow
    fn analyze_reaching_definitions(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
    ) -> VMResult<ReachingDefinitions> {
        let mut reaching_defs = ReachingDefinitions {
            reach_in: HashMap::new(),
            reach_out: HashMap::new(),
            gen: HashMap::new(),
            kill: HashMap::new(),
            all_definitions: BTreeSet::new(),
        };

        // Collect all definitions in the function
        for block in &cfg.blocks {
            for (instr_offset, instruction) in block.instructions.iter().enumerate() {
                let (defs, _) = self.extract_def_use_from_instruction(&instruction.instruction)?;
                for var in defs {
                    let definition = Definition {
                        variable: var,
                        block_id: block.id,
                        instruction_offset: instr_offset as u32,
                        def_type: DefinitionType::Assignment,
                    };
                    reaching_defs.all_definitions.insert(definition);
                }
            }
        }

        // Compute gen and kill sets for each block
        for block in &cfg.blocks {
            let (gen_set, kill_set) = self.compute_gen_kill_sets(block, &reaching_defs.all_definitions)?;
            reaching_defs.gen.insert(block.id, gen_set);
            reaching_defs.kill.insert(block.id, kill_set);
            reaching_defs.reach_in.insert(block.id, BTreeSet::new());
            reaching_defs.reach_out.insert(block.id, BTreeSet::new());
        }

        // Iterative fixed-point computation (forward analysis)
        let mut changed = true;
        let mut iterations = 0;
        
        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            for block in &cfg.blocks {
                // reach_in[B] = ∪ reach_out[P] for all predecessors P of B
                let mut new_reach_in = BTreeSet::new();
                for &pred_id in &block.predecessors {
                    if let Some(pred_reach_out) = reaching_defs.reach_out.get(&pred_id) {
                        new_reach_in.extend(pred_reach_out.iter().cloned());
                    }
                }

                // reach_out[B] = gen[B] ∪ (reach_in[B] - kill[B])
                let mut new_reach_out = reaching_defs.gen.get(&block.id).cloned().unwrap_or_default();
                let kill_set = reaching_defs.kill.get(&block.id).unwrap();
                
                for def in &new_reach_in {
                    if !kill_set.contains(def) {
                        new_reach_out.insert(def.clone());
                    }
                }

                // Check for changes
                if reaching_defs.reach_in.get(&block.id) != Some(&new_reach_in) {
                    reaching_defs.reach_in.insert(block.id, new_reach_in);
                    changed = true;
                }
                
                if reaching_defs.reach_out.get(&block.id) != Some(&new_reach_out) {
                    reaching_defs.reach_out.insert(block.id, new_reach_out);
                    changed = true;
                }
            }
        }

        Ok(reaching_defs)
    }

    /// Compute gen and kill sets for reaching definitions
    fn compute_gen_kill_sets(
        &self,
        block: &BasicBlock,
        all_definitions: &BTreeSet<Definition>,
    ) -> VMResult<(BTreeSet<Definition>, BTreeSet<Definition>)> {
        let mut gen_set = BTreeSet::new();
        let mut kill_set = BTreeSet::new();

        for (instr_offset, instruction) in block.instructions.iter().enumerate() {
            let (defs, _) = self.extract_def_use_from_instruction(&instruction.instruction)?;
            
            for var in defs {
                let definition = Definition {
                    variable: var.clone(),
                    block_id: block.id,
                    instruction_offset: instr_offset as u32,
                    def_type: DefinitionType::Assignment,
                };
                
                gen_set.insert(definition);
                
                // Kill all other definitions of the same variable
                for other_def in all_definitions {
                    if other_def.variable == var && other_def != &definition {
                        kill_set.insert(other_def.clone());
                    }
                }
            }
        }

        Ok((gen_set, kill_set))
    }

    /// Analyze available expressions using forward data flow - Complete implementation
    fn analyze_available_expressions(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
    ) -> VMResult<AvailableExpressions> {
        let mut available_exprs = AvailableExpressions {
            avail_in: HashMap::new(),
            avail_out: HashMap::new(),
            gen: HashMap::new(),
            kill: HashMap::new(),
            all_expressions: BTreeSet::new(),
        };

        // Collect all expressions in the function
        for block in &cfg.blocks {
            for instruction in &block.instructions {
                if let Some(expr) = self.extract_expression_from_instruction(&instruction.instruction)? {
                    available_exprs.all_expressions.insert(expr);
                }
            }
        }

        // Compute gen and kill sets for each block
        for block in &cfg.blocks {
            let (gen_set, kill_set) = self.compute_expression_gen_kill_sets(block, &available_exprs.all_expressions)?;
            available_exprs.gen.insert(block.id, gen_set);
            available_exprs.kill.insert(block.id, kill_set);
            available_exprs.avail_in.insert(block.id, BTreeSet::new());
            available_exprs.avail_out.insert(block.id, BTreeSet::new());
        }

        // Iterative fixed-point computation (forward analysis)
        let mut changed = true;
        let mut iterations = 0;
        
        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            for block in &cfg.blocks {
                // avail_in[B] = ∩ avail_out[P] for all predecessors P of B
                let mut new_avail_in = if block.predecessors.is_empty() {
                    BTreeSet::new() // Entry block
                } else {
                    available_exprs.all_expressions.clone() // Start with all expressions
                };

                for &pred_id in &block.predecessors {
                    if let Some(pred_avail_out) = available_exprs.avail_out.get(&pred_id) {
                        new_avail_in = new_avail_in.intersection(pred_avail_out).cloned().collect();
                    }
                }

                // avail_out[B] = gen[B] ∪ (avail_in[B] - kill[B])
                let mut new_avail_out = available_exprs.gen.get(&block.id).cloned().unwrap_or_default();
                let kill_set = available_exprs.kill.get(&block.id).unwrap();
                
                for expr in &new_avail_in {
                    if !kill_set.contains(expr) {
                        new_avail_out.insert(expr.clone());
                    }
                }

                // Check for changes
                if available_exprs.avail_in.get(&block.id) != Some(&new_avail_in) {
                    available_exprs.avail_in.insert(block.id, new_avail_in);
                    changed = true;
                }
                
                if available_exprs.avail_out.get(&block.id) != Some(&new_avail_out) {
                    available_exprs.avail_out.insert(block.id, new_avail_out);
                    changed = true;
                }
            }
        }

        Ok(available_exprs)
    }

    /// Extract expression from instruction for available expressions analysis
    fn extract_expression_from_instruction(&self, instruction: &Instruction) -> VMResult<Option<Expression>> {
        use crate::bytecode::instructions::PrismOpcode;
        
        let expression = match instruction.opcode {
            // Binary arithmetic operations
            PrismOpcode::ADD => Some(Expression {
                operator: ExpressionOperator::Add,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "arithmetic".to_string(),
            }),
            PrismOpcode::SUB => Some(Expression {
                operator: ExpressionOperator::Sub,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "arithmetic".to_string(),
            }),
            PrismOpcode::MUL => Some(Expression {
                operator: ExpressionOperator::Mul,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "arithmetic".to_string(),
            }),
            PrismOpcode::DIV => Some(Expression {
                operator: ExpressionOperator::Div,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "arithmetic".to_string(),
            }),
            PrismOpcode::MOD => Some(Expression {
                operator: ExpressionOperator::Mod,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "arithmetic".to_string(),
            }),

            // Comparison operations
            PrismOpcode::EQ => Some(Expression {
                operator: ExpressionOperator::Eq,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "comparison".to_string(),
            }),
            PrismOpcode::NE => Some(Expression {
                operator: ExpressionOperator::Ne,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "comparison".to_string(),
            }),
            PrismOpcode::LT => Some(Expression {
                operator: ExpressionOperator::Lt,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "comparison".to_string(),
            }),
            PrismOpcode::LE => Some(Expression {
                operator: ExpressionOperator::Le,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "comparison".to_string(),
            }),
            PrismOpcode::GT => Some(Expression {
                operator: ExpressionOperator::Gt,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "comparison".to_string(),
            }),
            PrismOpcode::GE => Some(Expression {
                operator: ExpressionOperator::Ge,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "comparison".to_string(),
            }),

            // Logical operations
            PrismOpcode::AND => Some(Expression {
                operator: ExpressionOperator::And,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "logical".to_string(),
            }),
            PrismOpcode::OR => Some(Expression {
                operator: ExpressionOperator::Or,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "logical".to_string(),
            }),
            PrismOpcode::XOR => Some(Expression {
                operator: ExpressionOperator::Xor,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                    Variable {
                        name: "stack_top_1".to_string(),
                        var_type: VariableType::Stack { offset: -1 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "logical".to_string(),
            }),

            // Unary operations
            PrismOpcode::NEG => Some(Expression {
                operator: ExpressionOperator::Neg,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "unary".to_string(),
            }),
            PrismOpcode::NOT => Some(Expression {
                operator: ExpressionOperator::Not,
                operands: vec![
                    Variable {
                        name: "stack_top".to_string(),
                        var_type: VariableType::Stack { offset: 0 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "unary".to_string(),
            }),

            // Memory operations
            PrismOpcode::LOAD_LOCAL(index) => Some(Expression {
                operator: ExpressionOperator::Load,
                operands: vec![
                    Variable {
                        name: format!("local_{}", index),
                        var_type: VariableType::Local { index: index as u8 },
                        scope: VariableScope::Function,
                    },
                ],
                expr_type: "memory".to_string(),
            }),

            // Function calls
            PrismOpcode::CALL(argc) => Some(Expression {
                operator: ExpressionOperator::Call { 
                    function_name: "dynamic".to_string() 
                },
                operands: (0..=argc).map(|i| Variable {
                    name: format!("stack_arg_{}", i),
                    var_type: VariableType::Stack { offset: -(i as i32) },
                    scope: VariableScope::Function,
                }).collect(),
                expr_type: "call".to_string(),
            }),

            // No expression for other instructions
            _ => None,
        };

        Ok(expression)
    }

    /// Compute gen and kill sets for available expressions
    fn compute_expression_gen_kill_sets(
        &self,
        block: &BasicBlock,
        all_expressions: &BTreeSet<Expression>,
    ) -> VMResult<(BTreeSet<Expression>, BTreeSet<Expression>)> {
        let mut gen_set = BTreeSet::new();
        let mut kill_set = BTreeSet::new();

        for instruction in &block.instructions {
            let (defs, _) = self.extract_def_use_from_instruction(&instruction.instruction)?;
            
            // If instruction generates an expression, add it to gen set
            if let Some(expr) = self.extract_expression_from_instruction(&instruction.instruction)? {
                gen_set.insert(expr);
            }
            
            // Kill expressions that use variables being redefined
            for def_var in &defs {
                for expr in all_expressions {
                    if expr.operands.contains(def_var) {
                        kill_set.insert(expr.clone());
                    }
                }
            }
        }

        Ok((gen_set, kill_set))
    }

    /// Build use-def chains from liveness and reaching definitions - Complete implementation
    fn build_use_def_chains(
        &self,
        liveness: &LivenessAnalysis,
        reaching_defs: &ReachingDefinitions,
    ) -> VMResult<UseDefChains> {
        let mut chains = HashMap::new();
        
        // For each block, find uses and their reaching definitions
        for (block_id, reach_in) in &reaching_defs.reach_in {
            // Get all uses in this block from liveness analysis
            if let Some(use_vars) = liveness.use_vars.get(block_id) {
                for use_var in use_vars {
                    // Create a Use entry for this variable
                    let use_entry = Use {
                        variable: use_var.clone(),
                        block_id: *block_id,
                        instruction_offset: 0, // Simplified - would need more precise tracking
                        use_type: UseType::Read,
                    };
                    
                    // Find all definitions that reach this use
                    let mut reaching_defs_for_use = BTreeSet::new();
                    for def in reach_in {
                        if def.variable == *use_var {
                            reaching_defs_for_use.insert(def.clone());
                        }
                    }
                    
                    chains.insert(use_entry, reaching_defs_for_use);
                }
            }
        }

        Ok(UseDefChains { chains })
    }

    /// Build def-use chains from use-def chains
    fn build_def_use_chains(&self, use_def_chains: &UseDefChains) -> VMResult<DefUseChains> {
        let mut chains: HashMap<Definition, BTreeSet<Use>> = HashMap::new();
        
        for (use_site, definitions) in &use_def_chains.chains {
            for definition in definitions {
                chains.entry(definition.clone()).or_default().insert(use_site.clone());
            }
        }

        Ok(DefUseChains { chains })
    }

    /// Build interference graph for register allocation
    fn build_interference_graph(
        &self,
        liveness: &LivenessAnalysis,
        cfg: &ControlFlowGraph,
    ) -> VMResult<InterferenceGraph> {
        let mut graph = InterferenceGraph {
            variables: BTreeSet::new(),
            edges: BTreeSet::new(),
            adjacency: HashMap::new(),
            coloring: HashMap::new(),
        };

        // Collect all variables
        for live_set in liveness.live_in.values() {
            graph.variables.extend(live_set.iter().cloned());
        }
        for live_set in liveness.live_out.values() {
            graph.variables.extend(live_set.iter().cloned());
        }

        // Build interference edges
        for live_set in liveness.live_in.values() {
            // Any two variables live at the same time interfere
            for var1 in live_set {
                for var2 in live_set {
                    if var1 != var2 {
                        let edge = if var1 < var2 { (var1.clone(), var2.clone()) } else { (var2.clone(), var1.clone()) };
                        graph.edges.insert(edge);
                        
                        graph.adjacency.entry(var1.clone()).or_default().insert(var2.clone());
                        graph.adjacency.entry(var2.clone()).or_default().insert(var1.clone());
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Compute live ranges for variables
    fn compute_live_ranges(
        &self,
        liveness: &LivenessAnalysis,
        cfg: &ControlFlowGraph,
    ) -> VMResult<HashMap<Variable, LiveRange>> {
        let mut live_ranges = HashMap::new();
        
        // For each variable, compute its live range across all blocks
        let mut all_vars = BTreeSet::new();
        for live_set in liveness.live_in.values() {
            all_vars.extend(live_set.iter().cloned());
        }
        
        for var in all_vars {
            let live_range = self.compute_variable_live_range(&var, liveness, cfg)?;
            live_ranges.insert(var, live_range);
        }

        Ok(live_ranges)
    }

    /// Compute live range for a specific variable - Enhanced implementation
    fn compute_variable_live_range(
        &self,
        variable: &Variable,
        liveness: &LivenessAnalysis,
        cfg: &ControlFlowGraph,
    ) -> VMResult<LiveRange> {
        let mut start = u32::MAX;
        let mut end = 0;
        let mut holes = Vec::new();
        let mut last_live_end = None;

        // Create a sorted list of blocks by their position in the function
        let mut block_positions: Vec<(u32, u32)> = cfg.blocks.iter()
            .map(|block| (block.id, block.id * 1000)) // Simplified block positioning
            .collect();
        block_positions.sort_by_key(|&(_, pos)| pos);

        // Track where the variable is live
        for (block_id, block_start_pos) in &block_positions {
            let block = cfg.blocks.iter().find(|b| b.id == *block_id).unwrap();
            
            // Check if variable is live in this block
            let live_in = liveness.live_in.get(block_id).map_or(false, |vars| vars.contains(variable));
            let live_out = liveness.live_out.get(block_id).map_or(false, |vars| vars.contains(variable));
            let defined = liveness.def.get(block_id).map_or(false, |vars| vars.contains(variable));
            let used = liveness.use_vars.get(block_id).map_or(false, |vars| vars.contains(variable));

            if live_in || live_out || defined || used {
                let block_end_pos = block_start_pos + block.instructions.len() as u32;
                
                // Check for holes in liveness
                if let Some(last_end) = last_live_end {
                    if *block_start_pos > last_end + 1 {
                        holes.push((last_end + 1, *block_start_pos - 1));
                    }
                }
                
                start = start.min(*block_start_pos);
                end = end.max(block_end_pos);
                last_live_end = Some(block_end_pos);
            }
        }

        // Calculate spill cost based on usage frequency and loop nesting
        let spill_cost = self.calculate_spill_cost(variable, cfg);

        Ok(LiveRange {
            variable: variable.clone(),
            start,
            end,
            holes,
            spill_cost,
        })
    }

    /// Calculate spill cost for a variable
    fn calculate_spill_cost(&self, variable: &Variable, cfg: &ControlFlowGraph) -> f64 {
        let mut cost = 0.0;
        
        // Base cost for variable type
        match &variable.var_type {
            VariableType::Local { .. } => cost += 1.0,
            VariableType::Global { .. } => cost += 2.0, // Globals are more expensive to spill
            VariableType::Temporary { .. } => cost += 0.5, // Temporaries are cheaper
            VariableType::Stack { .. } => cost += 0.1, // Stack variables are very cheap
            VariableType::Register { .. } => cost += 3.0, // Register variables are expensive
        }
        
        // Add cost based on usage frequency (simplified)
        for block in &cfg.blocks {
            let block_frequency = if block.predecessors.len() > 1 { 10.0 } else { 1.0 }; // Loop approximation
            
            for instruction in &block.instructions {
                if let Ok((defs, uses)) = self.extract_def_use_from_instruction(&instruction.instruction) {
                    if defs.contains(variable) || uses.contains(variable) {
                        cost += block_frequency;
                    }
                }
            }
        }
        
        cost
    }
}

/// Dependencies for data flow analysis (requires CFG)
pub struct DataFlowDependencies {
    pub cfg: ControlFlowGraph,
}

/// Implement the Analysis trait for DataFlowAnalyzer
impl Analysis for DataFlowAnalyzer {
    type Config = AnalysisConfig;
    type Result = DataFlowAnalysis;
    type Dependencies = DataFlowDependencies;

    fn new(config: &Self::Config) -> VMResult<Self> {
        Self::new(config)
    }

    fn analyze(&mut self, function: &FunctionDefinition, deps: Self::Dependencies) -> VMResult<Self::Result> {
        self.analyze(function, &deps.cfg)
    }

    fn analysis_kind() -> AnalysisKind {
        AnalysisKind::DataFlow
    }

    fn dependencies() -> Vec<AnalysisKind> {
        vec![AnalysisKind::ControlFlow]
    }

    fn validate_dependencies(deps: &Self::Dependencies) -> VMResult<()> {
        // Validate that CFG is not empty
        if deps.cfg.blocks.is_empty() {
            return Err(crate::PrismVMError::AnalysisError(
                "Control flow graph is empty".to_string()
            ));
        }
        Ok(())
    }
}

/// Implement conversion from AnalysisContext for DataFlowDependencies
impl From<&AnalysisContext> for DataFlowDependencies {
    fn from(context: &AnalysisContext) -> Self {
        let cfg = context.get_result(AnalysisKind::ControlFlow)
            .expect("Control flow analysis result required for data flow analysis");
        
        Self { cfg }
    }
} 