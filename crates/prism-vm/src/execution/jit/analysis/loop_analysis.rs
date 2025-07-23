//! Loop Analysis
//!
//! This module provides comprehensive loop analysis including loop detection,
//! nesting analysis, induction variable analysis, and loop optimization
//! opportunity identification.

use crate::{VMResult, PrismVMError, bytecode::FunctionDefinition};
use super::{AnalysisConfig, control_flow::{ControlFlowGraph, BasicBlock}};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};

/// Loop analyzer
#[derive(Debug)]
pub struct LoopAnalyzer {
    /// Configuration
    config: AnalysisConfig,
    /// Cached dominance relationships for efficiency
    dominance_cache: HashMap<(u32, u32), bool>,
}

/// Comprehensive loop analysis results
#[derive(Debug, Clone)]
pub struct LoopAnalysis {
    /// Function being analyzed
    pub function_id: u32,
    
    /// Detected loops
    pub loops: Vec<LoopInfo>,
    
    /// Loop nesting forest
    pub loop_forest: Vec<LoopNest>,
    
    /// Block to loop mapping
    pub block_to_loop: HashMap<u32, u32>,
    
    /// Induction variables
    pub induction_variables: HashMap<u32, Vec<InductionVariable>>,
    
    /// Loop optimization opportunities
    pub optimization_opportunities: Vec<LoopOptimizationOpportunity>,
}

/// Information about a detected loop
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Loop ID
    pub id: u32,
    
    /// Loop header (entry point)
    pub header: u32,
    
    /// Loop blocks
    pub blocks: BTreeSet<u32>,
    
    /// Loop exits
    pub exits: Vec<LoopExit>,
    
    /// Loop backedges
    pub backedges: Vec<LoopBackedge>,
    
    /// Loop depth (nesting level)
    pub depth: u32,
    
    /// Parent loop (if nested)
    pub parent: Option<u32>,
    
    /// Child loops
    pub children: Vec<u32>,
    
    /// Loop characteristics
    pub characteristics: LoopCharacteristics,
    
    /// Loop metadata
    pub metadata: LoopMetadata,
}

/// Loop nesting structure
#[derive(Debug, Clone)]
pub struct LoopNest {
    /// Root loop ID
    pub root_loop: u32,
    
    /// Nested structure
    pub nested_loops: Vec<LoopNest>,
    
    /// Nesting depth
    pub depth: u32,
}

/// Loop exit information
#[derive(Debug, Clone)]
pub struct LoopExit {
    /// Exit block (inside loop)
    pub exit_block: u32,
    
    /// Target block (outside loop)
    pub target_block: u32,
    
    /// Exit condition
    pub condition: Option<ExitCondition>,
    
    /// Exit probability
    pub probability: f64,
}

/// Loop backedge information
#[derive(Debug, Clone)]
pub struct LoopBackedge {
    /// Source block (tail of loop)
    pub source: u32,
    
    /// Target block (header of loop)
    pub target: u32,
    
    /// Backedge condition
    pub condition: Option<BackedgeCondition>,
    
    /// Execution frequency
    pub frequency: f64,
}

/// Exit condition for a loop
#[derive(Debug, Clone)]
pub enum ExitCondition {
    /// Conditional branch
    Conditional {
        /// Variable being tested
        variable: String,
        /// Comparison operator
        operator: ComparisonOperator,
        /// Comparison value
        value: i64,
    },
    /// Unconditional exit
    Unconditional,
    /// Exception-based exit
    Exception,
}

/// Backedge condition
#[derive(Debug, Clone)]
pub enum BackedgeCondition {
    /// Conditional backedge
    Conditional {
        /// Variable being tested
        variable: String,
        /// Comparison operator
        operator: ComparisonOperator,
        /// Comparison value
        value: i64,
    },
    /// Unconditional backedge
    Unconditional,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}

/// Loop characteristics
#[derive(Debug, Clone, Default)]
pub struct LoopCharacteristics {
    /// Whether loop is reducible
    pub is_reducible: bool,
    
    /// Whether loop is natural
    pub is_natural: bool,
    
    /// Whether loop has single entry
    pub single_entry: bool,
    
    /// Whether loop has single exit
    pub single_exit: bool,
    
    /// Whether loop is innermost
    pub is_innermost: bool,
    
    /// Whether loop is hot (frequently executed)
    pub is_hot: bool,
    
    /// Loop trip count (if analyzable)
    pub trip_count: TripCount,
    
    /// Loop invariants
    pub invariants: Vec<LoopInvariant>,
}

/// Trip count information
#[derive(Debug, Clone)]
pub enum TripCount {
    /// Constant trip count
    Constant(u64),
    
    /// Variable trip count with bounds
    Variable { min: Option<u64>, max: Option<u64> },
    
    /// Unknown trip count
    Unknown,
}

impl Default for TripCount {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Loop invariant expression
#[derive(Debug, Clone)]
pub struct LoopInvariant {
    /// Invariant expression
    pub expression: String,
    
    /// Variables involved
    pub variables: Vec<String>,
    
    /// Whether it's profitable to hoist
    pub profitable_to_hoist: bool,
    
    /// Hoisting cost estimate
    pub hoisting_cost: f64,
}

/// Loop metadata
#[derive(Debug, Clone, Default)]
pub struct LoopMetadata {
    /// Estimated execution frequency
    pub execution_frequency: f64,
    
    /// Average iteration count
    pub avg_iterations: f64,
    
    /// Loop body size in instructions
    pub body_size: usize,
    
    /// Memory access patterns
    pub memory_patterns: Vec<MemoryAccessPattern>,
    
    /// Performance characteristics
    pub performance_notes: Vec<String>,
}

/// Memory access pattern in loop
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Base address variable
    pub base: String,
    
    /// Access stride
    pub stride: i64,
    
    /// Access pattern type
    pub pattern_type: AccessPatternType,
    
    /// Whether pattern is regular
    pub is_regular: bool,
}

/// Types of memory access patterns
#[derive(Debug, Clone)]
pub enum AccessPatternType {
    /// Sequential access
    Sequential,
    
    /// Strided access
    Strided { stride: i64 },
    
    /// Random access
    Random,
    
    /// Indirect access
    Indirect,
}

/// Induction variable analysis
#[derive(Debug, Clone)]
pub struct InductionVariable {
    /// Variable name
    pub variable: String,
    
    /// Loop ID
    pub loop_id: u32,
    
    /// Induction variable type
    pub iv_type: InductionVariableType,
    
    /// Initial value
    pub initial_value: Option<i64>,
    
    /// Step value
    pub step: i64,
    
    /// Final value (if analyzable)
    pub final_value: Option<i64>,
    
    /// Whether IV is used for loop control
    pub is_loop_control: bool,
    
    /// Dependent variables
    pub dependent_variables: Vec<DependentVariable>,
}

/// Types of induction variables
#[derive(Debug, Clone)]
pub enum InductionVariableType {
    /// Basic induction variable (i = i + c)
    Basic,
    
    /// Derived induction variable (j = a * i + b)
    Derived {
        /// Base induction variable
        base: String,
        /// Multiplier
        multiplier: i64,
        /// Offset
        offset: i64,
    },
}

/// Variable dependent on an induction variable
#[derive(Debug, Clone)]
pub struct DependentVariable {
    /// Variable name
    pub variable: String,
    
    /// Dependency relationship
    pub relationship: DependencyRelationship,
}

/// Dependency relationship
#[derive(Debug, Clone)]
pub enum DependencyRelationship {
    /// Linear dependency (var = a * iv + b)
    Linear { multiplier: i64, offset: i64 },
    
    /// Polynomial dependency
    Polynomial { coefficients: Vec<i64> },
    
    /// Complex dependency
    Complex,
}

/// Loop optimization opportunities
#[derive(Debug, Clone)]
pub struct LoopOptimizationOpportunity {
    /// Loop ID
    pub loop_id: u32,
    
    /// Optimization type
    pub optimization_type: LoopOptimizationType,
    
    /// Estimated benefit
    pub estimated_benefit: f64,
    
    /// Implementation cost
    pub implementation_cost: f64,
    
    /// Prerequisites
    pub prerequisites: Vec<String>,
    
    /// Potential issues
    pub potential_issues: Vec<String>,
}

/// Types of loop optimizations
#[derive(Debug, Clone)]
pub enum LoopOptimizationType {
    /// Loop invariant code motion
    InvariantCodeMotion {
        /// Expressions to hoist
        expressions: Vec<String>,
    },
    
    /// Loop unrolling
    Unrolling {
        /// Unroll factor
        factor: u32,
        /// Whether partial unrolling is acceptable
        allow_partial: bool,
    },
    
    /// Loop vectorization
    Vectorization {
        /// Vector width
        vector_width: u32,
        /// Vectorizable operations
        operations: Vec<String>,
    },
    
    /// Loop fusion
    Fusion {
        /// Target loop to fuse with
        target_loop: u32,
    },
    
    /// Loop distribution/fission
    Distribution {
        /// Split points
        split_points: Vec<u32>,
    },
    
    /// Strength reduction
    StrengthReduction {
        /// Operations to reduce
        operations: Vec<String>,
    },
    
    /// Loop interchange
    Interchange {
        /// Loops to interchange
        loops: Vec<u32>,
    },
    
    /// Loop tiling/blocking
    Tiling {
        /// Tile sizes
        tile_sizes: Vec<u32>,
    },
}

impl LoopAnalyzer {
    /// Create new loop analyzer
    pub fn new(config: &AnalysisConfig) -> VMResult<Self> {
        Ok(Self {
            config: config.clone(),
            dominance_cache: HashMap::new(),
        })
    }

    /// Perform comprehensive loop analysis
    pub fn analyze(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
    ) -> VMResult<LoopAnalysis> {
        // Clear cache for new analysis
        self.dominance_cache.clear();
        
        // Step 1: Detect natural loops
        let loops = self.detect_natural_loops(cfg)?;
        
        // Step 2: Analyze loop nesting
        let loop_forest = self.build_loop_forest(&loops)?;
        
        // Step 3: Build block to loop mapping
        let block_to_loop = self.build_block_to_loop_mapping(&loops);
        
        // Step 4: Analyze induction variables
        let induction_variables = self.analyze_induction_variables(function, cfg, &loops)?;
        
        // Step 5: Identify optimization opportunities
        let optimization_opportunities = self.identify_optimization_opportunities(
            function, cfg, &loops, &induction_variables
        )?;

        Ok(LoopAnalysis {
            function_id: function.id,
            loops,
            loop_forest,
            block_to_loop,
            induction_variables,
            optimization_opportunities,
        })
    }

    /// Detect natural loops using dominance information
    fn detect_natural_loops(&mut self, cfg: &ControlFlowGraph) -> VMResult<Vec<LoopInfo>> {
        let mut loops = Vec::new();
        let mut loop_id = 0;

        // Find back edges (edges where target dominates source)
        for edge in &cfg.edges {
            if self.dominates(&cfg.dominance, edge.to, edge.from) {
                // This is a back edge, indicating a natural loop
                let header = edge.to;
                let tail = edge.from;
                
                // Find all blocks in the loop
                let loop_blocks = self.find_loop_blocks(cfg, header, tail)?;
                
                // Find loop exits
                let exits = self.find_loop_exits(cfg, &loop_blocks)?;
                
                // Find all backedges for this loop
                let backedges = self.find_loop_backedges(cfg, header, &loop_blocks)?;
                
                // Analyze loop characteristics
                let characteristics = self.analyze_loop_characteristics(cfg, &loop_blocks, header)?;
                
                // Compute loop metadata
                let metadata = self.compute_loop_metadata(cfg, &loop_blocks)?;

                let loop_info = LoopInfo {
                    id: loop_id,
                    header,
                    blocks: loop_blocks,
                    exits,
                    backedges,
                    depth: 1, // Will be updated in nesting analysis
                    parent: None, // Will be updated in nesting analysis
                    children: Vec::new(), // Will be updated in nesting analysis
                    characteristics,
                    metadata,
                };

                loops.push(loop_info);
                loop_id += 1;
            }
        }

        // Compute loop nesting relationships
        self.compute_loop_nesting(&mut loops)?;

        Ok(loops)
    }

    /// Find all blocks belonging to a natural loop
    fn find_loop_blocks(&self, cfg: &ControlFlowGraph, header: u32, tail: u32) -> VMResult<BTreeSet<u32>> {
        let mut loop_blocks = BTreeSet::new();
        let mut worklist = VecDeque::new();
        
        loop_blocks.insert(header);
        
        if header != tail {
            loop_blocks.insert(tail);
            worklist.push_back(tail);
        }

        // Backward traversal from tail to header
        while let Some(current) = worklist.pop_front() {
            // Find predecessors of current block
            for edge in &cfg.edges {
                if edge.to == current && loop_blocks.insert(edge.from) {
                    worklist.push_back(edge.from);
                }
            }
        }

        Ok(loop_blocks)
    }

    /// Find loop exits
    fn find_loop_exits(&self, cfg: &ControlFlowGraph, loop_blocks: &BTreeSet<u32>) -> VMResult<Vec<LoopExit>> {
        let mut exits = Vec::new();

        for &block_id in loop_blocks {
            for edge in &cfg.edges {
                if edge.from == block_id && !loop_blocks.contains(&edge.to) {
                    // This is an exit edge
                    let exit = LoopExit {
                        exit_block: block_id,
                        target_block: edge.to,
                        condition: self.analyze_exit_condition(cfg, edge)?,
                        probability: edge.probability,
                    };
                    exits.push(exit);
                }
            }
        }

        Ok(exits)
    }

    /// Find loop backedges
    fn find_loop_backedges(&self, cfg: &ControlFlowGraph, header: u32, loop_blocks: &BTreeSet<u32>) -> VMResult<Vec<LoopBackedge>> {
        let mut backedges = Vec::new();

        for edge in &cfg.edges {
            if edge.to == header && loop_blocks.contains(&edge.from) {
                let backedge = LoopBackedge {
                    source: edge.from,
                    target: edge.to,
                    condition: self.analyze_backedge_condition(cfg, edge)?,
                    frequency: edge.frequency,
                };
                backedges.push(backedge);
            }
        }

        Ok(backedges)
    }

    /// Analyze loop characteristics
    fn analyze_loop_characteristics(
        &self,
        cfg: &ControlFlowGraph,
        loop_blocks: &BTreeSet<u32>,
        header: u32,
    ) -> VMResult<LoopCharacteristics> {
        let mut characteristics = LoopCharacteristics::default();

        // Check if loop is natural (single entry point)
        let mut entry_count = 0;
        for &block_id in loop_blocks {
            for edge in &cfg.edges {
                if edge.to == block_id && !loop_blocks.contains(&edge.from) {
                    entry_count += 1;
                }
            }
        }
        characteristics.single_entry = entry_count <= 1;
        characteristics.is_natural = characteristics.single_entry;

        // Check if loop has single exit
        let mut exit_blocks = HashSet::new();
        for &block_id in loop_blocks {
            for edge in &cfg.edges {
                if edge.from == block_id && !loop_blocks.contains(&edge.to) {
                    exit_blocks.insert(block_id);
                    break; // Count blocks with exits, not individual exit edges
                }
            }
        }
        characteristics.single_exit = exit_blocks.len() == 1;

        // Loop is reducible if it's natural (simplified check)
        characteristics.is_reducible = characteristics.is_natural;

        // Check if loop is innermost (no nested loops)
        characteristics.is_innermost = self.is_innermost_loop(cfg, loop_blocks)?;

        // Determine if loop is hot based on execution frequency
        let total_frequency: f64 = loop_blocks.iter()
            .filter_map(|&block_id| cfg.blocks.iter().find(|b| b.id == block_id))
            .map(|block| block.metadata.execution_frequency)
            .sum();
        characteristics.is_hot = total_frequency > self.config.hot_threshold.unwrap_or(1000.0);

        // Analyze trip count
        characteristics.trip_count = self.analyze_trip_count(cfg, loop_blocks, header)?;

        // Find loop invariants
        characteristics.invariants = self.find_loop_invariants(cfg, loop_blocks)?;

        Ok(characteristics)
    }

    /// Check if a loop is innermost (contains no other loops)
    fn is_innermost_loop(&self, cfg: &ControlFlowGraph, loop_blocks: &BTreeSet<u32>) -> VMResult<bool> {
        // Look for backedges within the loop that would indicate nested loops
        for edge in &cfg.edges {
            if loop_blocks.contains(&edge.from) && loop_blocks.contains(&edge.to) {
                // Check if this is a backedge for a different loop
                if self.dominates(&cfg.dominance, edge.to, edge.from) {
                    // This is a backedge within our loop, indicating a nested loop
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Compute loop metadata
    fn compute_loop_metadata(&self, cfg: &ControlFlowGraph, loop_blocks: &BTreeSet<u32>) -> VMResult<LoopMetadata> {
        let mut metadata = LoopMetadata::default();

        // Compute body size
        for &block_id in loop_blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                metadata.body_size += block.instructions.len();
            }
        }

        // Estimate execution frequency
        metadata.execution_frequency = loop_blocks.iter()
            .filter_map(|&block_id| cfg.blocks.iter().find(|b| b.id == block_id))
            .map(|block| block.metadata.execution_frequency)
            .fold(0.0, f64::max);

        // Estimate average iterations based on backedge frequencies
        let backedge_frequency: f64 = cfg.edges.iter()
            .filter(|edge| loop_blocks.contains(&edge.from) && loop_blocks.contains(&edge.to))
            .map(|edge| edge.frequency)
            .sum();
        
        if metadata.execution_frequency > 0.0 {
            metadata.avg_iterations = backedge_frequency / metadata.execution_frequency;
        }

        // Analyze memory access patterns
        metadata.memory_patterns = self.analyze_memory_patterns(cfg, loop_blocks)?;

        // Add performance notes
        if metadata.body_size > 100 {
            metadata.performance_notes.push("Large loop body - consider loop distribution".to_string());
        }
        if metadata.avg_iterations > 1000.0 {
            metadata.performance_notes.push("High iteration count - good candidate for optimization".to_string());
        }

        Ok(metadata)
    }

    /// Analyze induction variables in loops
    fn analyze_induction_variables(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
        loops: &[LoopInfo],
    ) -> VMResult<HashMap<u32, Vec<InductionVariable>>> {
        let mut induction_variables = HashMap::new();

        for loop_info in loops {
            let ivs = self.find_induction_variables_in_loop(function, cfg, loop_info)?;
            induction_variables.insert(loop_info.id, ivs);
        }

        Ok(induction_variables)
    }

    /// Find induction variables in a specific loop
    fn find_induction_variables_in_loop(
        &self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
        loop_info: &LoopInfo,
    ) -> VMResult<Vec<InductionVariable>> {
        let mut induction_variables = Vec::new();
        let mut variable_updates = HashMap::new();
        let mut variable_uses = HashMap::new();

        // Analyze all instructions in loop blocks
        for &block_id in &loop_info.blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                for (offset, instruction) in block.instructions.iter().enumerate() {
                    self.analyze_instruction_for_induction_variables(
                        &instruction.instruction,
                        block_id,
                        offset as u32,
                        &mut variable_updates,
                        &mut variable_uses,
                    )?;
                }
            }
        }

        // Identify basic induction variables (variables that are incremented by a constant)
        for (var_name, updates) in &variable_updates {
            if let Some(step) = self.is_basic_induction_variable(var_name, updates, &variable_uses) {
                let iv = InductionVariable {
                    variable: var_name.clone(),
                    loop_id: loop_info.id,
                    iv_type: InductionVariableType::Basic,
                    initial_value: self.find_initial_value(var_name, loop_info, cfg),
                    step,
                    final_value: None, // Would require more sophisticated analysis
                    is_loop_control: self.is_loop_control_variable(var_name, loop_info, cfg)?,
                    dependent_variables: self.find_dependent_variables(var_name, &variable_updates, &variable_uses),
                };
                induction_variables.push(iv);
            }
        }

        // Identify derived induction variables
        for basic_iv in &induction_variables.clone() {
            for (var_name, updates) in &variable_updates {
                if var_name != &basic_iv.variable {
                    if let Some((multiplier, offset)) = self.is_derived_induction_variable(
                        var_name, &basic_iv.variable, updates, &variable_uses
                    ) {
                        let derived_iv = InductionVariable {
                            variable: var_name.clone(),
                            loop_id: loop_info.id,
                            iv_type: InductionVariableType::Derived {
                                base: basic_iv.variable.clone(),
                                multiplier,
                                offset,
                            },
                            initial_value: None,
                            step: basic_iv.step * multiplier,
                            final_value: None,
                            is_loop_control: false,
                            dependent_variables: Vec::new(),
                        };
                        induction_variables.push(derived_iv);
                    }
                }
            }
        }

        Ok(induction_variables)
    }

    /// Analyze an instruction for induction variable patterns
    fn analyze_instruction_for_induction_variables(
        &self,
        instruction: &crate::bytecode::Instruction,
        block_id: u32,
        offset: u32,
        variable_updates: &mut HashMap<String, Vec<VariableUpdate>>,
        variable_uses: &mut HashMap<String, Vec<VariableUse>>,
    ) -> VMResult<()> {
        use crate::bytecode::instructions::PrismOpcode;

        match instruction.opcode {
            // Local variable stores (potential induction variable updates)
            PrismOpcode::STORE_LOCAL(slot) => {
                let var_name = format!("local_{}", slot);
                variable_updates.entry(var_name).or_insert_with(Vec::new).push(VariableUpdate {
                    block_id,
                    offset,
                    update_type: UpdateType::Store,
                });
            }
            PrismOpcode::STORE_LOCAL_EXT(slot) => {
                let var_name = format!("local_{}", slot);
                variable_updates.entry(var_name).or_insert_with(Vec::new).push(VariableUpdate {
                    block_id,
                    offset,
                    update_type: UpdateType::Store,
                });
            }
            
            // Local variable loads (potential induction variable uses)
            PrismOpcode::LOAD_LOCAL(slot) => {
                let var_name = format!("local_{}", slot);
                variable_uses.entry(var_name).or_insert_with(Vec::new).push(VariableUse {
                    block_id,
                    offset,
                    use_type: UseType::Load,
                });
            }
            PrismOpcode::LOAD_LOCAL_EXT(slot) => {
                let var_name = format!("local_{}", slot);
                variable_uses.entry(var_name).or_insert_with(Vec::new).push(VariableUse {
                    block_id,
                    offset,
                    use_type: UseType::Load,
                });
            }
            
            // Arithmetic operations that might be part of induction variable updates
            PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL => {
                // These could be part of induction variable computations
                // More sophisticated analysis would track data flow
            }
            
            _ => {} // Other instructions don't directly affect induction variable analysis
        }

        Ok(())
    }

    /// Check if a variable is a basic induction variable
    fn is_basic_induction_variable(
        &self,
        var_name: &str,
        updates: &[VariableUpdate],
        uses: &HashMap<String, Vec<VariableUse>>,
    ) -> Option<i64> {
        // Simplified heuristic: if variable is updated in a consistent pattern
        // and used in comparisons, it might be an induction variable
        if updates.len() >= 1 && uses.get(var_name).map_or(0, |u| u.len()) >= 1 {
            // Assume increment by 1 for now - real implementation would analyze the actual increment
            Some(1)
        } else {
            None
        }
    }

    /// Check if a variable is a derived induction variable
    fn is_derived_induction_variable(
        &self,
        var_name: &str,
        base_var: &str,
        updates: &[VariableUpdate],
        uses: &HashMap<String, Vec<VariableUse>>,
    ) -> Option<(i64, i64)> {
        // Simplified: assume linear relationship if variable is updated and base is used
        if !updates.is_empty() && uses.get(base_var).is_some() {
            // Assume j = 2*i + 0 pattern for simplicity
            Some((2, 0))
        } else {
            None
        }
    }

    /// Find initial value of an induction variable
    fn find_initial_value(&self, var_name: &str, loop_info: &LoopInfo, cfg: &ControlFlowGraph) -> Option<i64> {
        // Look for initialization before loop entry
        // This would require more sophisticated data flow analysis in practice
        Some(0) // Simplified assumption
    }

    /// Check if a variable is used for loop control
    fn is_loop_control_variable(&self, var_name: &str, loop_info: &LoopInfo, cfg: &ControlFlowGraph) -> VMResult<bool> {
        // Check if variable is used in conditional branches at loop exits
        for exit in &loop_info.exits {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == exit.exit_block) {
                // Look for conditional branches that might use this variable
                for instruction in &block.instructions {
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::JUMP_IF_TRUE(_) |
                        crate::bytecode::instructions::PrismOpcode::JUMP_IF_FALSE(_) => {
                            // Simplified: assume any conditional in exit block uses loop control variable
                            return Ok(true);
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(false)
    }

    /// Find variables dependent on an induction variable
    fn find_dependent_variables(
        &self,
        base_var: &str,
        variable_updates: &HashMap<String, Vec<VariableUpdate>>,
        variable_uses: &HashMap<String, Vec<VariableUse>>,
    ) -> Vec<DependentVariable> {
        let mut dependent_vars = Vec::new();
        
        // Look for variables that are computed from the base induction variable
        for (var_name, _updates) in variable_updates {
            if var_name != base_var && variable_uses.get(base_var).is_some() {
                dependent_vars.push(DependentVariable {
                    variable: var_name.clone(),
                    relationship: DependencyRelationship::Linear { multiplier: 1, offset: 0 },
                });
            }
        }
        
        dependent_vars
    }

    /// Identify loop optimization opportunities
    fn identify_optimization_opportunities(
        &mut self,
        function: &FunctionDefinition,
        cfg: &ControlFlowGraph,
        loops: &[LoopInfo],
        induction_variables: &HashMap<u32, Vec<InductionVariable>>,
    ) -> VMResult<Vec<LoopOptimizationOpportunity>> {
        let mut opportunities = Vec::new();

        for loop_info in loops {
            // Check for invariant code motion opportunities
            if !loop_info.characteristics.invariants.is_empty() {
                let benefit = self.estimate_invariant_motion_benefit(&loop_info.characteristics.invariants);
                opportunities.push(LoopOptimizationOpportunity {
                    loop_id: loop_info.id,
                    optimization_type: LoopOptimizationType::InvariantCodeMotion {
                        expressions: loop_info.characteristics.invariants.iter()
                            .filter(|inv| inv.profitable_to_hoist)
                            .map(|inv| inv.expression.clone())
                            .collect(),
                    },
                    estimated_benefit: benefit,
                    implementation_cost: 0.1,
                    prerequisites: vec!["no_side_effects".to_string()],
                    potential_issues: vec!["register_pressure".to_string()],
                });
            }

            // Check for unrolling opportunities
            if let TripCount::Constant(count) = loop_info.characteristics.trip_count {
                if count <= 16 && loop_info.metadata.body_size <= 20 {
                    let factor = self.calculate_optimal_unroll_factor(count, loop_info.metadata.body_size);
                    opportunities.push(LoopOptimizationOpportunity {
                        loop_id: loop_info.id,
                        optimization_type: LoopOptimizationType::Unrolling {
                            factor,
                            allow_partial: true,
                        },
                        estimated_benefit: self.estimate_unrolling_benefit(factor, count),
                        implementation_cost: 0.05,
                        prerequisites: vec!["constant_trip_count".to_string()],
                        potential_issues: vec!["code_size_increase".to_string()],
                    });
                }
            }

            // Check for vectorization opportunities
            if self.can_vectorize_loop(loop_info, induction_variables.get(&loop_info.id))? {
                let vector_width = self.determine_optimal_vector_width(loop_info)?;
                opportunities.push(LoopOptimizationOpportunity {
                    loop_id: loop_info.id,
                    optimization_type: LoopOptimizationType::Vectorization {
                        vector_width,
                        operations: self.identify_vectorizable_operations(loop_info, cfg)?,
                    },
                    estimated_benefit: 0.3 * (vector_width as f64 / 4.0), // Scale with vector width
                    implementation_cost: 0.2,
                    prerequisites: vec!["no_dependencies".to_string(), "simd_support".to_string()],
                    potential_issues: vec!["alignment_requirements".to_string()],
                });
            }

            // Check for strength reduction opportunities
            let strength_reduction_ops = self.identify_strength_reduction_opportunities(loop_info, cfg)?;
            if !strength_reduction_ops.is_empty() {
                opportunities.push(LoopOptimizationOpportunity {
                    loop_id: loop_info.id,
                    optimization_type: LoopOptimizationType::StrengthReduction {
                        operations: strength_reduction_ops,
                    },
                    estimated_benefit: 0.15,
                    implementation_cost: 0.08,
                    prerequisites: vec!["induction_variables".to_string()],
                    potential_issues: vec!["increased_register_pressure".to_string()],
                });
            }

            // Check for loop fusion opportunities
            for other_loop in loops {
                if other_loop.id != loop_info.id && self.can_fuse_loops(loop_info, other_loop, cfg)? {
                    opportunities.push(LoopOptimizationOpportunity {
                        loop_id: loop_info.id,
                        optimization_type: LoopOptimizationType::Fusion {
                            target_loop: other_loop.id,
                        },
                        estimated_benefit: 0.1,
                        implementation_cost: 0.15,
                        prerequisites: vec!["compatible_iteration_spaces".to_string()],
                        potential_issues: vec!["increased_register_pressure".to_string()],
                    });
                }
            }
        }

        Ok(opportunities)
    }

    // Helper methods for optimization opportunity analysis

    fn estimate_invariant_motion_benefit(&self, invariants: &[LoopInvariant]) -> f64 {
        invariants.iter()
            .filter(|inv| inv.profitable_to_hoist)
            .map(|inv| 1.0 / (1.0 + inv.hoisting_cost))
            .sum::<f64>() * 0.1 // Scale factor
    }

    fn calculate_optimal_unroll_factor(&self, trip_count: u64, body_size: usize) -> u32 {
        let max_factor = 8; // Maximum unroll factor
        let size_limit = 100; // Maximum body size after unrolling
        
        let factor_by_count = (trip_count as u32).min(max_factor);
        let factor_by_size = (size_limit / body_size.max(1)) as u32;
        
        factor_by_count.min(factor_by_size).max(2)
    }

    fn estimate_unrolling_benefit(&self, factor: u32, trip_count: u64) -> f64 {
        let overhead_reduction = (factor - 1) as f64 / trip_count as f64;
        overhead_reduction * 0.1 // Loop overhead is typically ~10% of execution time
    }

    fn can_vectorize_loop(&self, loop_info: &LoopInfo, induction_variables: Option<&Vec<InductionVariable>>) -> VMResult<bool> {
        // Check basic requirements for vectorization
        if !loop_info.characteristics.is_innermost {
            return Ok(false);
        }

        if !loop_info.characteristics.single_entry || !loop_info.characteristics.single_exit {
            return Ok(false);
        }

        // Check for regular induction variables
        if let Some(ivs) = induction_variables {
            let has_regular_iv = ivs.iter().any(|iv| {
                matches!(iv.iv_type, InductionVariableType::Basic) && iv.step == 1
            });
            if !has_regular_iv {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }

        // Check loop body size (vectorization is less beneficial for very large loops)
        if loop_info.metadata.body_size > 50 {
            return Ok(false);
        }

        Ok(true)
    }

    fn determine_optimal_vector_width(&self, loop_info: &LoopInfo) -> VMResult<u32> {
        // Determine optimal vector width based on target architecture and loop characteristics
        let base_width = 4; // SSE width
        let avx_width = 8; // AVX width
        
        // Use wider vectors for loops with more iterations
        if loop_info.metadata.avg_iterations > 100.0 {
            Ok(avx_width)
        } else {
            Ok(base_width)
        }
    }

    fn identify_vectorizable_operations(&self, loop_info: &LoopInfo, cfg: &ControlFlowGraph) -> VMResult<Vec<String>> {
        let mut operations = Vec::new();
        
        // Look for arithmetic operations in loop blocks
        for &block_id in &loop_info.blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                for instruction in &block.instructions {
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::ADD => operations.push("add".to_string()),
                        crate::bytecode::instructions::PrismOpcode::SUB => operations.push("sub".to_string()),
                        crate::bytecode::instructions::PrismOpcode::MUL => operations.push("mul".to_string()),
                        _ => {}
                    }
                }
            }
        }
        
        operations.sort();
        operations.dedup();
        Ok(operations)
    }

    fn identify_strength_reduction_opportunities(&self, loop_info: &LoopInfo, cfg: &ControlFlowGraph) -> VMResult<Vec<String>> {
        let mut operations = Vec::new();
        
        // Look for expensive operations that can be strength-reduced
        for &block_id in &loop_info.blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                for instruction in &block.instructions {
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::MUL => {
                            operations.push("multiply_to_add".to_string());
                        }
                        crate::bytecode::instructions::PrismOpcode::DIV => {
                            operations.push("divide_to_shift".to_string());
                        }
                        crate::bytecode::instructions::PrismOpcode::POW => {
                            operations.push("power_to_multiply".to_string());
                        }
                        _ => {}
                    }
                }
            }
        }
        
        operations.sort();
        operations.dedup();
        Ok(operations)
    }

    fn can_fuse_loops(&self, loop1: &LoopInfo, loop2: &LoopInfo, cfg: &ControlFlowGraph) -> VMResult<bool> {
        // Check if two loops can be fused
        
        // Must be at same nesting level
        if loop1.depth != loop2.depth {
            return Ok(false);
        }
        
        // Must have compatible iteration spaces (simplified check)
        if loop1.metadata.avg_iterations != loop2.metadata.avg_iterations {
            return Ok(false);
        }
        
        // Check for data dependencies between loops
        // This would require more sophisticated analysis in practice
        Ok(true)
    }

    // Core helper methods

    fn dominates(&mut self, dominance: &super::control_flow::DominanceInfo, a: u32, b: u32) -> bool {
        // Check cache first
        if let Some(&result) = self.dominance_cache.get(&(a, b)) {
            return result;
        }

        let result = if a == b {
            true
        } else {
            let mut current = b;
            loop {
                if let Some(&idom) = dominance.immediate_dominators.get(&current) {
                    if idom == current {
                        break false; // Reached root without finding a
                    }
                    if idom == a {
                        break true;
                    }
                    current = idom;
                } else {
                    break false;
                }
            }
        };

        // Cache the result
        self.dominance_cache.insert((a, b), result);
        result
    }

    fn build_loop_forest(&self, loops: &[LoopInfo]) -> VMResult<Vec<LoopNest>> {
        let mut forest = Vec::new();
        
        // Find root loops (loops with no parent)
        for loop_info in loops {
            if loop_info.parent.is_none() {
                let nest = self.build_loop_nest_recursive(loop_info, loops);
                forest.push(nest);
            }
        }
        
        Ok(forest)
    }

    fn build_loop_nest_recursive(&self, current_loop: &LoopInfo, all_loops: &[LoopInfo]) -> LoopNest {
        let mut nested_loops = Vec::new();
        
        // Find direct children
        for child_id in &current_loop.children {
            if let Some(child_loop) = all_loops.iter().find(|l| l.id == *child_id) {
                let child_nest = self.build_loop_nest_recursive(child_loop, all_loops);
                nested_loops.push(child_nest);
            }
        }
        
        LoopNest {
            root_loop: current_loop.id,
            nested_loops,
            depth: current_loop.depth,
        }
    }

    fn build_block_to_loop_mapping(&self, loops: &[LoopInfo]) -> HashMap<u32, u32> {
        let mut mapping = HashMap::new();
        
        // Map each block to the innermost loop containing it
        for loop_info in loops {
            for &block_id in &loop_info.blocks {
                // If block is already mapped, check if current loop is deeper (more nested)
                if let Some(&existing_loop_id) = mapping.get(&block_id) {
                    if let Some(existing_loop) = loops.iter().find(|l| l.id == existing_loop_id) {
                        if loop_info.depth > existing_loop.depth {
                            mapping.insert(block_id, loop_info.id);
                        }
                    }
                } else {
                    mapping.insert(block_id, loop_info.id);
                }
            }
        }
        
        mapping
    }

    fn compute_loop_nesting(&self, loops: &mut [LoopInfo]) -> VMResult<()> {
        // Sort loops by the number of blocks (smaller loops are typically more nested)
        let mut loop_indices: Vec<usize> = (0..loops.len()).collect();
        loop_indices.sort_by_key(|&i| loops[i].blocks.len());

        // Compute parent-child relationships
        for i in 0..loop_indices.len() {
            let current_idx = loop_indices[i];
            let current_blocks = loops[current_idx].blocks.clone();
            
            // Find the smallest loop that contains this loop (immediate parent)
            let mut immediate_parent = None;
            let mut min_parent_size = usize::MAX;
            
            for j in (i + 1)..loop_indices.len() {
                let candidate_idx = loop_indices[j];
                let candidate_blocks = &loops[candidate_idx].blocks;
                
                // Check if candidate contains current loop
                if current_blocks.is_subset(candidate_blocks) && candidate_blocks.len() < min_parent_size {
                    immediate_parent = Some(candidate_idx);
                    min_parent_size = candidate_blocks.len();
                }
            }
            
            // Set parent-child relationships
            if let Some(parent_idx) = immediate_parent {
                loops[current_idx].parent = Some(loops[parent_idx].id);
                loops[parent_idx].children.push(loops[current_idx].id);
                loops[current_idx].depth = loops[parent_idx].depth + 1;
            }
        }

        // Update innermost flag
        for loop_info in loops.iter_mut() {
            loop_info.characteristics.is_innermost = loop_info.children.is_empty();
        }

        Ok(())
    }

    fn analyze_exit_condition(&self, cfg: &ControlFlowGraph, edge: &super::control_flow::CFGEdge) -> VMResult<Option<ExitCondition>> {
        // Analyze the condition for loop exit based on edge type
        match edge.edge_type {
            super::control_flow::CFGEdgeType::ConditionalTrue | 
            super::control_flow::CFGEdgeType::ConditionalFalse => {
                // Look at the source block to determine the condition
                if let Some(block) = cfg.blocks.iter().find(|b| b.id == edge.from) {
                    // Look for comparison instructions
                    for instruction in block.instructions.iter().rev() {
                        match instruction.instruction.opcode {
                            crate::bytecode::instructions::PrismOpcode::LT => {
                                return Ok(Some(ExitCondition::Conditional {
                                    variable: "unknown".to_string(), // Would need data flow analysis
                                    operator: ComparisonOperator::LessThan,
                                    value: 0, // Would need constant propagation
                                }));
                            }
                            crate::bytecode::instructions::PrismOpcode::GT => {
                                return Ok(Some(ExitCondition::Conditional {
                                    variable: "unknown".to_string(),
                                    operator: ComparisonOperator::GreaterThan,
                                    value: 0,
                                }));
                            }
                            crate::bytecode::instructions::PrismOpcode::EQ => {
                                return Ok(Some(ExitCondition::Conditional {
                                    variable: "unknown".to_string(),
                                    operator: ComparisonOperator::Equal,
                                    value: 0,
                                }));
                            }
                            _ => {}
                        }
                    }
                }
                Ok(Some(ExitCondition::Conditional {
                    variable: "unknown".to_string(),
                    operator: ComparisonOperator::LessThan,
                    value: 0,
                }))
            }
            super::control_flow::CFGEdgeType::Unconditional => {
                Ok(Some(ExitCondition::Unconditional))
            }
            super::control_flow::CFGEdgeType::Exception => {
                Ok(Some(ExitCondition::Exception))
            }
            _ => Ok(None)
        }
    }

    fn analyze_backedge_condition(&self, cfg: &ControlFlowGraph, edge: &super::control_flow::CFGEdge) -> VMResult<Option<BackedgeCondition>> {
        // Similar to exit condition analysis but for backedges
        match edge.edge_type {
            super::control_flow::CFGEdgeType::ConditionalTrue | 
            super::control_flow::CFGEdgeType::ConditionalFalse |
            super::control_flow::CFGEdgeType::LoopBackedge => {
                Ok(Some(BackedgeCondition::Conditional {
                    variable: "i".to_string(), // Simplified
                    operator: ComparisonOperator::LessThan,
                    value: 10,
                }))
            }
            super::control_flow::CFGEdgeType::Unconditional => {
                Ok(Some(BackedgeCondition::Unconditional))
            }
            _ => Ok(None)
        }
    }

    fn analyze_trip_count(&self, cfg: &ControlFlowGraph, loop_blocks: &BTreeSet<u32>, header: u32) -> VMResult<TripCount> {
        // Analyze loop bounds to determine trip count
        
        // Look for induction variable patterns and bounds
        for &block_id in loop_blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                // Look for patterns like: i < N, i <= N, etc.
                let mut has_comparison = false;
                let mut constant_bound = None;
                
                for instruction in &block.instructions {
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::LT |
                        crate::bytecode::instructions::PrismOpcode::LE |
                        crate::bytecode::instructions::PrismOpcode::GT |
                        crate::bytecode::instructions::PrismOpcode::GE => {
                            has_comparison = true;
                        }
                        crate::bytecode::instructions::PrismOpcode::LOAD_CONST(index) => {
                            // Try to get constant value for bounds analysis
                            if let Some(constant) = cfg.function_id.try_into().ok()
                                .and_then(|_| Some(index as i64)) {
                                constant_bound = Some(constant as u64);
                            }
                        }
                        crate::bytecode::instructions::PrismOpcode::LOAD_SMALL_INT(value) => {
                            if value >= 0 {
                                constant_bound = Some(value as u64);
                            }
                        }
                        _ => {}
                    }
                }
                
                // If we found a comparison with a constant, estimate trip count
                if has_comparison && constant_bound.is_some() {
                    let bound = constant_bound.unwrap();
                    if bound <= 1000 { // Reasonable constant bound
                        return Ok(TripCount::Constant(bound));
                    } else {
                        return Ok(TripCount::Variable { min: Some(1), max: Some(bound) });
                    }
                }
            }
        }
        
        // Check if we can infer bounds from loop structure
        let avg_iterations = cfg.blocks.iter()
            .filter(|b| loop_blocks.contains(&b.id))
            .map(|b| b.metadata.execution_frequency)
            .fold(0.0, f64::max);
            
        if avg_iterations > 0.0 && avg_iterations < 1000.0 {
            Ok(TripCount::Variable { 
                min: Some(1), 
                max: Some(avg_iterations as u64) 
            })
        } else {
            Ok(TripCount::Unknown)
        }
    }

    fn find_loop_invariants(&self, cfg: &ControlFlowGraph, loop_blocks: &BTreeSet<u32>) -> VMResult<Vec<LoopInvariant>> {
        let mut invariants = Vec::new();
        let mut defined_in_loop = HashSet::new();
        let mut used_in_loop = HashSet::new();
        
        // First pass: identify variables defined and used in the loop
        for &block_id in loop_blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                for instruction in &block.instructions {
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::STORE_LOCAL(slot) |
                        crate::bytecode::instructions::PrismOpcode::STORE_LOCAL_EXT(slot) => {
                            defined_in_loop.insert(format!("local_{}", slot));
                        }
                        crate::bytecode::instructions::PrismOpcode::LOAD_LOCAL(slot) |
                        crate::bytecode::instructions::PrismOpcode::LOAD_LOCAL_EXT(slot) => {
                            used_in_loop.insert(format!("local_{}", slot));
                        }
                        crate::bytecode::instructions::PrismOpcode::LOAD_CONST(_) => {
                            // Constants are always invariant
                            invariants.push(LoopInvariant {
                                expression: "constant".to_string(),
                                variables: Vec::new(),
                                profitable_to_hoist: true,
                                hoisting_cost: 0.1,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        
        // Second pass: identify loop invariant expressions
        for &block_id in loop_blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                for instruction in &block.instructions {
                    // Look for computations that only use loop-invariant values
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::ADD |
                        crate::bytecode::instructions::PrismOpcode::SUB |
                        crate::bytecode::instructions::PrismOpcode::MUL => {
                            // Check if operands are loop invariant
                            // This would require data flow analysis to be precise
                            invariants.push(LoopInvariant {
                                expression: format!("arithmetic_{:?}", instruction.instruction.opcode),
                                variables: vec!["unknown".to_string()],
                                profitable_to_hoist: self.is_profitable_to_hoist(&instruction.instruction),
                                hoisting_cost: self.estimate_hoisting_cost(&instruction.instruction),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        
        // Remove duplicates and filter unprofitable invariants
        invariants.sort_by(|a, b| a.expression.cmp(&b.expression));
        invariants.dedup_by(|a, b| a.expression == b.expression);
        invariants.retain(|inv| inv.profitable_to_hoist);
        
        Ok(invariants)
    }

    fn is_profitable_to_hoist(&self, instruction: &crate::bytecode::Instruction) -> bool {
        // Determine if hoisting this instruction would be profitable
        match instruction.opcode {
            crate::bytecode::instructions::PrismOpcode::ADD |
            crate::bytecode::instructions::PrismOpcode::SUB => true, // Cheap operations are always profitable
            crate::bytecode::instructions::PrismOpcode::MUL => true, // Multiplication is moderately expensive
            crate::bytecode::instructions::PrismOpcode::DIV |
            crate::bytecode::instructions::PrismOpcode::POW |
            crate::bytecode::instructions::PrismOpcode::SQRT => true, // Expensive operations are very profitable
            _ => false,
        }
    }

    fn estimate_hoisting_cost(&self, instruction: &crate::bytecode::Instruction) -> f64 {
        // Estimate the cost of hoisting this instruction
        match instruction.opcode {
            crate::bytecode::instructions::PrismOpcode::ADD |
            crate::bytecode::instructions::PrismOpcode::SUB => 0.1,
            crate::bytecode::instructions::PrismOpcode::MUL => 0.2,
            crate::bytecode::instructions::PrismOpcode::DIV => 0.3,
            crate::bytecode::instructions::PrismOpcode::POW |
            crate::bytecode::instructions::PrismOpcode::SQRT => 0.4,
            _ => 0.5,
        }
    }

    fn analyze_memory_patterns(&self, cfg: &ControlFlowGraph, loop_blocks: &BTreeSet<u32>) -> VMResult<Vec<MemoryAccessPattern>> {
        let mut patterns = Vec::new();
        
        // Analyze memory access instructions in the loop
        for &block_id in loop_blocks {
            if let Some(block) = cfg.blocks.iter().find(|b| b.id == block_id) {
                for instruction in &block.instructions {
                    match instruction.instruction.opcode {
                        crate::bytecode::instructions::PrismOpcode::GET_INDEX => {
                            // Array access - likely sequential or strided
                            patterns.push(MemoryAccessPattern {
                                base: "array".to_string(),
                                stride: 1, // Assume unit stride for simplicity
                                pattern_type: AccessPatternType::Sequential,
                                is_regular: true,
                            });
                        }
                        crate::bytecode::instructions::PrismOpcode::SET_INDEX => {
                            // Array store - similar to load
                            patterns.push(MemoryAccessPattern {
                                base: "array".to_string(),
                                stride: 1,
                                pattern_type: AccessPatternType::Sequential,
                                is_regular: true,
                            });
                        }
                        crate::bytecode::instructions::PrismOpcode::GET_FIELD(_) |
                        crate::bytecode::instructions::PrismOpcode::SET_FIELD(_) => {
                            // Object field access - potentially irregular
                            patterns.push(MemoryAccessPattern {
                                base: "object".to_string(),
                                stride: 0, // Field access doesn't have regular stride
                                pattern_type: AccessPatternType::Random,
                                is_regular: false,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        
        // Analyze patterns for regularity and stride
        self.analyze_pattern_regularity(&mut patterns);
        
        Ok(patterns)
    }

    fn analyze_pattern_regularity(&self, patterns: &mut [MemoryAccessPattern]) {
        // Group patterns by base and analyze for regularity
        let mut base_groups: HashMap<String, Vec<&mut MemoryAccessPattern>> = HashMap::new();
        
        for pattern in patterns.iter_mut() {
            base_groups.entry(pattern.base.clone()).or_insert_with(Vec::new).push(pattern);
        }
        
        for (_base, group) in base_groups {
            if group.len() > 1 {
                // Multiple accesses to same base - check for stride pattern
                let first_stride = group[0].stride;
                let all_same_stride = group.iter().all(|p| p.stride == first_stride);
                
                if all_same_stride && first_stride != 0 {
                    for pattern in group {
                        pattern.pattern_type = AccessPatternType::Strided { stride: first_stride };
                        pattern.is_regular = true;
                    }
                }
            }
        }
    }
}

// Helper types for induction variable analysis

#[derive(Debug, Clone)]
struct VariableUpdate {
    block_id: u32,
    offset: u32,
    update_type: UpdateType,
}

#[derive(Debug, Clone)]
enum UpdateType {
    Store,
    Increment,
    Decrement,
}

#[derive(Debug, Clone)]
struct VariableUse {
    block_id: u32,
    offset: u32,
    use_type: UseType,
}

#[derive(Debug, Clone)]
enum UseType {
    Load,
    Comparison,
    Arithmetic,
} 