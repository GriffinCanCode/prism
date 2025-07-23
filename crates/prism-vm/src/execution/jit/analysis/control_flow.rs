//! Control Flow Graph Analysis
//!
//! This module provides comprehensive control flow analysis including CFG construction,
//! dominance analysis, and post-dominance analysis. It forms the foundation for many
//! other optimization analyses.

use crate::{VMResult, PrismVMError, bytecode::{FunctionDefinition, Instruction}};
use super::AnalysisConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Control flow graph analyzer
#[derive(Debug)]
pub struct CFGAnalyzer {
    /// Configuration
    config: AnalysisConfig,
}

/// Control flow graph representation
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Function being analyzed
    pub function_id: u32,
    
    /// Basic blocks
    pub blocks: Vec<BasicBlock>,
    
    /// Edges between blocks
    pub edges: Vec<CFGEdge>,
    
    /// Entry block ID
    pub entry_block: u32,
    
    /// Exit blocks
    pub exit_blocks: Vec<u32>,
    
    /// Dominance information
    pub dominance: DominanceInfo,
    
    /// Post-dominance information
    pub post_dominance: PostDominanceInfo,
    
    /// Loop information
    pub loop_info: LoopInfo,
}

/// Basic block in control flow graph
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block ID
    pub id: u32,
    
    /// Instructions in this block
    pub instructions: Vec<BlockInstruction>,
    
    /// Predecessors
    pub predecessors: Vec<u32>,
    
    /// Successors
    pub successors: Vec<u32>,
    
    /// Block metadata
    pub metadata: BlockMetadata,
}

/// Instruction within a basic block
#[derive(Debug, Clone)]
pub struct BlockInstruction {
    /// Original bytecode offset
    pub bytecode_offset: u32,
    
    /// Instruction opcode and operands
    pub instruction: Instruction,
    
    /// Whether this instruction can throw
    pub can_throw: bool,
    
    /// Whether this instruction has side effects
    pub has_side_effects: bool,
}

/// Basic block metadata
#[derive(Debug, Clone, Default)]
pub struct BlockMetadata {
    /// Block execution frequency (from profiling)
    pub execution_frequency: f64,
    
    /// Whether block is a loop header
    pub is_loop_header: bool,
    
    /// Whether block is a loop exit
    pub is_loop_exit: bool,
    
    /// Loop depth
    pub loop_depth: u32,
    
    /// Block size in bytes
    pub size_bytes: usize,
}

/// Control flow edge
#[derive(Debug, Clone)]
pub struct CFGEdge {
    /// Source block
    pub from: u32,
    
    /// Target block
    pub to: u32,
    
    /// Edge type
    pub edge_type: CFGEdgeType,
    
    /// Edge probability (0.0 to 1.0)
    pub probability: f64,
    
    /// Edge execution frequency
    pub frequency: f64,
}

/// Types of control flow edges
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CFGEdgeType {
    /// Unconditional edge (fall-through or jump)
    Unconditional,
    
    /// True branch of conditional
    ConditionalTrue,
    
    /// False branch of conditional
    ConditionalFalse,
    
    /// Exception edge
    Exception,
    
    /// Switch case edge
    SwitchCase { case_value: i64 },
    
    /// Switch default edge
    SwitchDefault,
    
    /// Loop backedge
    LoopBackedge,
}

/// Dominance information
#[derive(Debug, Clone, Default)]
pub struct DominanceInfo {
    /// Immediate dominators
    pub immediate_dominators: HashMap<u32, u32>,
    
    /// Dominance frontier
    pub dominance_frontier: HashMap<u32, Vec<u32>>,
    
    /// Dominance tree children
    pub dom_tree_children: HashMap<u32, Vec<u32>>,
    
    /// Dominance tree depth
    pub dom_tree_depth: HashMap<u32, u32>,
}

/// Post-dominance information
#[derive(Debug, Clone, Default)]
pub struct PostDominanceInfo {
    /// Immediate post-dominators
    pub immediate_post_dominators: HashMap<u32, u32>,
    
    /// Post-dominance frontier
    pub post_dominance_frontier: HashMap<u32, Vec<u32>>,
    
    /// Post-dominance tree children
    pub post_dom_tree_children: HashMap<u32, Vec<u32>>,
}

/// Loop information
#[derive(Debug, Clone, Default)]
pub struct LoopInfo {
    /// Natural loops
    pub natural_loops: Vec<NaturalLoop>,
    
    /// Loop nesting forest
    pub loop_forest: Vec<LoopNest>,
    
    /// Block to loop mapping
    pub block_to_loop: HashMap<u32, u32>,
}

/// Natural loop
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Loop ID
    pub id: u32,
    
    /// Loop header block
    pub header: u32,
    
    /// Loop blocks
    pub blocks: HashSet<u32>,
    
    /// Loop exits
    pub exits: Vec<u32>,
    
    /// Loop depth
    pub depth: u32,
    
    /// Parent loop (if nested)
    pub parent: Option<u32>,
}

/// Loop nesting structure
#[derive(Debug, Clone)]
pub struct LoopNest {
    /// Loop ID
    pub loop_id: u32,
    
    /// Nested loops
    pub nested_loops: Vec<LoopNest>,
    
    /// Loop metadata
    pub metadata: LoopMetadata,
}

/// Loop metadata
#[derive(Debug, Clone, Default)]
pub struct LoopMetadata {
    /// Estimated iteration count
    pub iteration_count: Option<u64>,
    
    /// Loop execution frequency
    pub frequency: f64,
    
    /// Whether loop is hot
    pub is_hot: bool,
    
    /// Loop optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

impl CFGAnalyzer {
    /// Create new CFG analyzer
    pub fn new(config: &AnalysisConfig) -> VMResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Analyze function and build control flow graph
    pub fn analyze(&mut self, function: &FunctionDefinition) -> VMResult<ControlFlowGraph> {
        // Step 1: Build basic blocks
        let blocks = self.build_basic_blocks(function)?;
        
        // Step 2: Build edges
        let edges = self.build_edges(function, &blocks)?;
        
        // Step 3: Identify entry and exit blocks
        let entry_block = 0; // First block is entry
        let exit_blocks = self.find_exit_blocks(&blocks, &edges);
        
        // Step 4: Compute dominance information
        let dominance = self.compute_dominance(&blocks, &edges, entry_block)?;
        
        // Step 5: Compute post-dominance information
        let post_dominance = self.compute_post_dominance(&blocks, &edges, &exit_blocks)?;
        
        // Step 6: Detect loops
        let loop_info = self.detect_loops(&blocks, &edges, &dominance)?;
        
        Ok(ControlFlowGraph {
            function_id: function.id,
            blocks,
            edges,
            entry_block,
            exit_blocks,
            dominance,
            post_dominance,
            loop_info,
        })
    }

    /// Build basic blocks from function instructions
    fn build_basic_blocks(&self, function: &FunctionDefinition) -> VMResult<Vec<BasicBlock>> {
        let mut blocks = Vec::new();
        let mut current_block = BasicBlock {
            id: 0,
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
            metadata: BlockMetadata::default(),
        };

        let mut block_id = 0;
        let mut leaders = HashSet::new();
        
        // Identify block leaders
        leaders.insert(0); // First instruction is always a leader
        
        for (i, instruction) in function.instructions.iter().enumerate() {
            // Instructions after branches are leaders
            if self.is_branch_instruction(instruction) && i + 1 < function.instructions.len() {
                leaders.insert(i + 1);
            }
            
            // Branch targets are leaders
            if let Some(target) = self.get_branch_target(instruction) {
                leaders.insert(target as usize);
            }
        }

        // Build blocks
        for (i, instruction) in function.instructions.iter().enumerate() {
            if leaders.contains(&i) && !current_block.instructions.is_empty() {
                // Start new block
                blocks.push(current_block);
                block_id += 1;
                current_block = BasicBlock {
                    id: block_id,
                    instructions: Vec::new(),
                    predecessors: Vec::new(),
                    successors: Vec::new(),
                    metadata: BlockMetadata::default(),
                };
            }

            current_block.instructions.push(BlockInstruction {
                bytecode_offset: i as u32,
                instruction: instruction.clone(),
                can_throw: self.can_instruction_throw(instruction),
                has_side_effects: self.has_side_effects(instruction),
            });

            // Update block metadata
            current_block.metadata.size_bytes += std::mem::size_of_val(instruction);
        }

        // Add final block
        if !current_block.instructions.is_empty() {
            blocks.push(current_block);
        }

        Ok(blocks)
    }

    /// Build edges between basic blocks
    fn build_edges(&self, function: &FunctionDefinition, blocks: &[BasicBlock]) -> VMResult<Vec<CFGEdge>> {
        let mut edges = Vec::new();
        
        for (i, block) in blocks.iter().enumerate() {
            if let Some(last_instruction) = block.instructions.last() {
                // Calculate the next instruction offset for fall-through cases
                let next_instruction_offset = last_instruction.bytecode_offset + 1;
                
                match self.get_control_flow_type(&last_instruction.instruction, next_instruction_offset) {
                    ControlFlowType::FallThrough => {
                        if i + 1 < blocks.len() {
                            edges.push(CFGEdge {
                                from: block.id,
                                to: blocks[i + 1].id,
                                edge_type: CFGEdgeType::Unconditional,
                                probability: 1.0,
                                frequency: 0.0, // Will be filled by profiling
                            });
                        }
                    }
                    ControlFlowType::UnconditionalJump { target } => {
                        if let Some(target_block) = self.find_block_by_offset(blocks, target) {
                            edges.push(CFGEdge {
                                from: block.id,
                                to: target_block.id,
                                edge_type: CFGEdgeType::Unconditional,
                                probability: 1.0,
                                frequency: 0.0,
                            });
                        }
                    }
                    ControlFlowType::ConditionalBranch { true_target, false_target } => {
                        if let Some(true_block) = self.find_block_by_offset(blocks, true_target) {
                            edges.push(CFGEdge {
                                from: block.id,
                                to: true_block.id,
                                edge_type: CFGEdgeType::ConditionalTrue,
                                probability: 0.5, // Default, will be updated by profiling
                                frequency: 0.0,
                            });
                        }
                        if let Some(false_block) = self.find_block_by_offset(blocks, false_target) {
                            edges.push(CFGEdge {
                                from: block.id,
                                to: false_block.id,
                                edge_type: CFGEdgeType::ConditionalFalse,
                                probability: 0.5,
                                frequency: 0.0,
                            });
                        }
                    }
                    ControlFlowType::Return => {
                        // No outgoing edges for return
                    }
                    ControlFlowType::Throw => {
                        // Exception edges would be added based on exception handling info
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Compute dominance information using Lengauer-Tarjan algorithm
    fn compute_dominance(
        &self,
        blocks: &[BasicBlock],
        edges: &[CFGEdge],
        entry_block: u32,
    ) -> VMResult<DominanceInfo> {
        let mut dominance = DominanceInfo::default();
        
        // Build adjacency list
        let mut successors: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut predecessors: HashMap<u32, Vec<u32>> = HashMap::new();
        
        for edge in edges {
            successors.entry(edge.from).or_default().push(edge.to);
            predecessors.entry(edge.to).or_default().push(edge.from);
        }

        // Compute immediate dominators using iterative algorithm
        let mut idom = HashMap::new();
        idom.insert(entry_block, entry_block);

        let mut changed = true;
        while changed {
            changed = false;
            
            for block in blocks {
                if block.id == entry_block {
                    continue;
                }
                
                let preds = predecessors.get(&block.id).unwrap_or(&vec![]);
                if preds.is_empty() {
                    continue;
                }

                let mut new_idom = preds[0];
                for &pred in &preds[1..] {
                    if idom.contains_key(&pred) {
                        new_idom = self.intersect(&idom, new_idom, pred);
                    }
                }

                if idom.get(&block.id) != Some(&new_idom) {
                    idom.insert(block.id, new_idom);
                    changed = true;
                }
            }
        }

        dominance.immediate_dominators = idom;
        
        // Compute dominance tree children
        for (&node, &idom_node) in &dominance.immediate_dominators {
            if node != idom_node {
                dominance.dom_tree_children.entry(idom_node).or_default().push(node);
            }
        }

        // Compute dominance frontier
        dominance.dominance_frontier = self.compute_dominance_frontier(blocks, edges, &dominance.immediate_dominators);

        Ok(dominance)
    }

    /// Find intersection of two nodes in dominance tree
    fn intersect(&self, idom: &HashMap<u32, u32>, mut finger1: u32, mut finger2: u32) -> u32 {
        while finger1 != finger2 {
            while finger1 > finger2 {
                finger1 = *idom.get(&finger1).unwrap_or(&finger1);
            }
            while finger2 > finger1 {
                finger2 = *idom.get(&finger2).unwrap_or(&finger2);
            }
        }
        finger1
    }

    /// Compute dominance frontier
    fn compute_dominance_frontier(
        &self,
        blocks: &[BasicBlock],
        edges: &[CFGEdge],
        idom: &HashMap<u32, u32>,
    ) -> HashMap<u32, Vec<u32>> {
        let mut df = HashMap::new();
        
        for block in blocks {
            df.insert(block.id, Vec::new());
        }

        for edge in edges {
            let x = edge.from;
            let y = edge.to;
            
            let mut runner = x;
            while runner != *idom.get(&y).unwrap_or(&y) {
                df.entry(runner).or_default().push(y);
                runner = *idom.get(&runner).unwrap_or(&runner);
            }
        }

        df
    }

    /// Compute post-dominance information using reversed CFG
    fn compute_post_dominance(
        &self,
        blocks: &[BasicBlock],
        edges: &[CFGEdge],
        exit_blocks: &[u32],
    ) -> VMResult<PostDominanceInfo> {
        let mut post_dominance = PostDominanceInfo::default();
        
        // Create virtual exit node if multiple exit blocks exist
        let virtual_exit = if exit_blocks.len() > 1 {
            Some(blocks.len() as u32) // Use next available ID
        } else {
            None
        };
        
        // Build reversed adjacency lists
        let mut predecessors: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut successors: HashMap<u32, Vec<u32>> = HashMap::new();
        
        // Reverse all edges
        for edge in edges {
            predecessors.entry(edge.from).or_default().push(edge.to);
            successors.entry(edge.to).or_default().push(edge.from);
        }
        
        // Add edges from virtual exit to all real exit blocks
        if let Some(virtual_exit_id) = virtual_exit {
            for &exit_block in exit_blocks {
                successors.entry(exit_block).or_default().push(virtual_exit_id);
                predecessors.entry(virtual_exit_id).or_default().push(exit_block);
            }
        }
        
        // Compute immediate post-dominators using iterative algorithm on reversed CFG
        let mut ipdom = HashMap::new();
        let start_node = virtual_exit.unwrap_or(exit_blocks[0]);
        ipdom.insert(start_node, start_node);

        let mut changed = true;
        while changed {
            changed = false;
            
            // Process all blocks (including virtual exit if it exists)
            let mut all_blocks: Vec<u32> = blocks.iter().map(|b| b.id).collect();
            if let Some(virtual_exit_id) = virtual_exit {
                all_blocks.push(virtual_exit_id);
            }
            
            for &block_id in &all_blocks {
                if block_id == start_node {
                    continue;
                }
                
                let preds = predecessors.get(&block_id).unwrap_or(&vec![]);
                if preds.is_empty() {
                    continue;
                }

                let mut new_ipdom = preds[0];
                for &pred in &preds[1..] {
                    if ipdom.contains_key(&pred) {
                        new_ipdom = self.intersect(&ipdom, new_ipdom, pred);
                    }
                }

                if ipdom.get(&block_id) != Some(&new_ipdom) {
                    ipdom.insert(block_id, new_ipdom);
                    changed = true;
                }
            }
        }

        // Remove virtual exit from results if it was used
        if virtual_exit.is_some() {
            ipdom.remove(&virtual_exit.unwrap());
        }

        post_dominance.immediate_post_dominators = ipdom;
        
        // Compute post-dominance tree children
        for (&node, &ipdom_node) in &post_dominance.immediate_post_dominators {
            if node != ipdom_node {
                post_dominance.post_dom_tree_children.entry(ipdom_node).or_default().push(node);
            }
        }

        // Compute post-dominance frontier
        post_dominance.post_dominance_frontier = self.compute_post_dominance_frontier(
            blocks, edges, &post_dominance.immediate_post_dominators
        );

        Ok(post_dominance)
    }

    /// Compute post-dominance frontier
    fn compute_post_dominance_frontier(
        &self,
        blocks: &[BasicBlock],
        edges: &[CFGEdge],
        ipdom: &HashMap<u32, u32>,
    ) -> HashMap<u32, Vec<u32>> {
        let mut pdf = HashMap::new();
        
        for block in blocks {
            pdf.insert(block.id, Vec::new());
        }

        // For post-dominance frontier, we work with reversed edges
        for edge in edges {
            let x = edge.to;   // Reversed
            let y = edge.from; // Reversed
            
            let mut runner = x;
            while runner != *ipdom.get(&y).unwrap_or(&y) {
                pdf.entry(runner).or_default().push(y);
                runner = *ipdom.get(&runner).unwrap_or(&runner);
            }
        }

        pdf
    }

    /// Detect natural loops using dominance information
    fn detect_loops(
        &self,
        blocks: &[BasicBlock],
        edges: &[CFGEdge],
        dominance: &DominanceInfo,
    ) -> VMResult<LoopInfo> {
        let mut loop_info = LoopInfo::default();
        let mut loop_id = 0;

        // Find back edges (edges where target dominates source)
        for edge in edges {
            if self.dominates(dominance, edge.to, edge.from) {
                // This is a back edge, so we have a natural loop
                let header = edge.to;
                let mut loop_blocks = HashSet::new();
                
                // Find all blocks in the loop using DFS from the back edge source
                let mut worklist = VecDeque::new();
                worklist.push_back(edge.from);
                loop_blocks.insert(header);
                
                while let Some(block) = worklist.pop_front() {
                    if loop_blocks.insert(block) {
                        // Add predecessors to worklist
                        for pred_edge in edges {
                            if pred_edge.to == block && !loop_blocks.contains(&pred_edge.from) {
                                worklist.push_back(pred_edge.from);
                            }
                        }
                    }
                }

                // Find loop exits
                let mut exits = Vec::new();
                for &loop_block in &loop_blocks {
                    for edge in edges {
                        if edge.from == loop_block && !loop_blocks.contains(&edge.to) {
                            exits.push(edge.to);
                        }
                    }
                }

                let natural_loop = NaturalLoop {
                    id: loop_id,
                    header,
                    blocks: loop_blocks,
                    exits,
                    depth: 1, // Will be computed later
                    parent: None, // Will be computed later
                };

                loop_info.natural_loops.push(natural_loop);
                loop_id += 1;
            }
        }

        // Compute loop nesting
        self.compute_loop_nesting(&mut loop_info);

        Ok(loop_info)
    }

    /// Check if block a dominates block b
    fn dominates(&self, dominance: &DominanceInfo, a: u32, b: u32) -> bool {
        if a == b {
            return true;
        }
        
        let mut current = b;
        while let Some(&idom) = dominance.immediate_dominators.get(&current) {
            if idom == current {
                break; // Reached root
            }
            if idom == a {
                return true;
            }
            current = idom;
        }
        false
    }

    /// Compute loop nesting relationships
    fn compute_loop_nesting(&self, loop_info: &mut LoopInfo) {
        // Sort loops by size (smaller loops are more deeply nested)
        let mut loops_by_size: Vec<(usize, usize)> = loop_info.natural_loops
            .iter()
            .enumerate()
            .map(|(i, loop_)| (i, loop_.blocks.len()))
            .collect();
        loops_by_size.sort_by_key(|(_, size)| *size);

        // Compute parent-child relationships
        for (i, (loop_i_idx, _)) in loops_by_size.iter().enumerate() {
            let loop_i = &loop_info.natural_loops[*loop_i_idx];
            
            // Find the smallest loop that contains this loop (immediate parent)
            let mut immediate_parent = None;
            let mut min_parent_size = usize::MAX;
            
            for (j, (loop_j_idx, loop_j_size)) in loops_by_size.iter().enumerate() {
                if i == j {
                    continue;
                }
                
                let loop_j = &loop_info.natural_loops[*loop_j_idx];
                
                // Check if loop_j contains loop_i (j is a parent of i)
                if *loop_j_size > loop_i.blocks.len() && 
                   loop_i.blocks.is_subset(&loop_j.blocks) {
                    if *loop_j_size < min_parent_size {
                        immediate_parent = Some(*loop_j_idx as u32);
                        min_parent_size = *loop_j_size;
                    }
                }
            }
            
            // Update parent information
            if let Some(parent_id) = immediate_parent {
                loop_info.natural_loops[*loop_i_idx].parent = Some(parent_id);
                
                // Calculate depth based on parent
                let parent_depth = loop_info.natural_loops
                    .iter()
                    .find(|l| l.id == parent_id)
                    .map(|l| l.depth)
                    .unwrap_or(0);
                loop_info.natural_loops[*loop_i_idx].depth = parent_depth + 1;
            }
        }

        // Build block to loop mapping
        for (loop_idx, natural_loop) in loop_info.natural_loops.iter().enumerate() {
            for &block_id in &natural_loop.blocks {
                // Map block to the innermost loop containing it
                if let Some(&existing_loop_id) = loop_info.block_to_loop.get(&block_id) {
                    let existing_loop = &loop_info.natural_loops[existing_loop_id as usize];
                    if natural_loop.depth > existing_loop.depth {
                        loop_info.block_to_loop.insert(block_id, loop_idx as u32);
                    }
                } else {
                    loop_info.block_to_loop.insert(block_id, loop_idx as u32);
                }
            }
        }

        // Build loop forest (top-level structure)
        for natural_loop in &loop_info.natural_loops {
            if natural_loop.parent.is_none() {
                let loop_nest = self.build_loop_nest(natural_loop, &loop_info.natural_loops);
                loop_info.loop_forest.push(loop_nest);
            }
        }
    }

    /// Build loop nesting structure recursively
    fn build_loop_nest(&self, current_loop: &NaturalLoop, all_loops: &[NaturalLoop]) -> LoopNest {
        let mut nested_loops = Vec::new();
        
        // Find direct children of current loop
        for child_loop in all_loops {
            if child_loop.parent == Some(current_loop.id) {
                let child_nest = self.build_loop_nest(child_loop, all_loops);
                nested_loops.push(child_nest);
            }
        }
        
        LoopNest {
            loop_id: current_loop.id,
            nested_loops,
            metadata: LoopMetadata {
                iteration_count: None, // Would be filled by profiling
                frequency: 0.0,        // Would be filled by profiling
                is_hot: false,         // Would be determined by profiling
                optimization_opportunities: self.identify_loop_optimizations(current_loop),
            },
        }
    }

    /// Identify optimization opportunities for a loop
    fn identify_loop_optimizations(&self, loop_: &NaturalLoop) -> Vec<String> {
        let mut opportunities = Vec::new();
        
        // Basic optimization opportunities based on loop structure
        if loop_.exits.len() == 1 {
            opportunities.push("Single exit loop - candidate for loop unrolling".to_string());
        }
        
        if loop_.blocks.len() <= 3 {
            opportunities.push("Small loop body - candidate for loop unrolling".to_string());
        }
        
        if loop_.depth == 1 {
            opportunities.push("Top-level loop - candidate for vectorization".to_string());
        }
        
        opportunities
    }

    // Helper methods
    fn is_branch_instruction(&self, instruction: &Instruction) -> bool {
        use crate::bytecode::instructions::PrismOpcode;
        matches!(instruction.opcode, 
            PrismOpcode::JUMP(_) | 
            PrismOpcode::JUMP_IF_TRUE(_) | 
            PrismOpcode::JUMP_IF_FALSE(_) |
            PrismOpcode::JUMP_IF_NULL(_) |
            PrismOpcode::JUMP_IF_NOT_NULL(_) |
            PrismOpcode::RETURN |
            PrismOpcode::RETURN_VALUE |
            PrismOpcode::THROW |
            PrismOpcode::TAIL_CALL(_)
        )
    }

    fn get_branch_target(&self, instruction: &Instruction) -> Option<u32> {
        use crate::bytecode::instructions::PrismOpcode;
        match instruction.opcode {
            PrismOpcode::JUMP(target) => Some(target as u32),
            PrismOpcode::JUMP_IF_TRUE(target) => Some(target as u32),
            PrismOpcode::JUMP_IF_FALSE(target) => Some(target as u32),
            PrismOpcode::JUMP_IF_NULL(target) => Some(target as u32),
            PrismOpcode::JUMP_IF_NOT_NULL(target) => Some(target as u32),
            _ => None,
        }
    }

    fn can_instruction_throw(&self, instruction: &Instruction) -> bool {
        use crate::bytecode::instructions::PrismOpcode;
        matches!(instruction.opcode,
            PrismOpcode::DIV |           // Division by zero
            PrismOpcode::MOD |           // Modulo by zero
            PrismOpcode::GET_INDEX |     // Array bounds check
            PrismOpcode::SET_INDEX |     // Array bounds check
            PrismOpcode::GET_FIELD(_) |  // Null pointer access
            PrismOpcode::SET_FIELD(_) |  // Null pointer access
            PrismOpcode::GET_FIELD_HASH(_) | // Null pointer access
            PrismOpcode::SET_FIELD_HASH(_) | // Null pointer access
            PrismOpcode::CALL(_) |       // Function call can throw
            PrismOpcode::CALL_DYNAMIC(_) | // Dynamic call can throw
            PrismOpcode::TAIL_CALL(_) |  // Tail call can throw
            PrismOpcode::THROW |         // Explicit throw
            PrismOpcode::TYPE_CAST(_) |  // Type casting can fail
            PrismOpcode::EFFECT_INVOKE(_) | // Effect operations can throw
            PrismOpcode::CAP_CHECK(_) |  // Capability violations
            PrismOpcode::IO_READ(_) |    // I/O operations can fail
            PrismOpcode::IO_WRITE(_) |   // I/O operations can fail
            PrismOpcode::IO_OPEN(_)      // File operations can fail
        )
    }

    fn has_side_effects(&self, instruction: &Instruction) -> bool {
        use crate::bytecode::instructions::PrismOpcode;
        matches!(instruction.opcode,
            // Store operations
            PrismOpcode::STORE_LOCAL(_) |
            PrismOpcode::STORE_LOCAL_EXT(_) |
            PrismOpcode::STORE_UPVALUE(_) |
            PrismOpcode::STORE_GLOBAL(_) |
            PrismOpcode::STORE_GLOBAL_HASH(_) |
            PrismOpcode::SET_FIELD(_) |
            PrismOpcode::SET_FIELD_HASH(_) |
            PrismOpcode::SET_INDEX |
            PrismOpcode::DELETE_FIELD(_) |
            
            // Function calls
            PrismOpcode::CALL(_) |
            PrismOpcode::CALL_DYNAMIC(_) |
            PrismOpcode::TAIL_CALL(_) |
            
            // Array mutations
            PrismOpcode::ARRAY_PUSH |
            PrismOpcode::ARRAY_POP |
            
            // Effect operations
            PrismOpcode::EFFECT_ENTER(_) |
            PrismOpcode::EFFECT_EXIT |
            PrismOpcode::EFFECT_INVOKE(_) |
            PrismOpcode::EFFECT_HANDLE(_) |
            PrismOpcode::EFFECT_RESUME |
            PrismOpcode::EFFECT_ABORT |
            
            // Capability operations
            PrismOpcode::CAP_DELEGATE(_) |
            PrismOpcode::CAP_REVOKE(_) |
            PrismOpcode::CAP_ACQUIRE(_) |
            PrismOpcode::CAP_RELEASE(_) |
            
            // Concurrency operations
            PrismOpcode::SPAWN_ACTOR(_) |
            PrismOpcode::SEND_MESSAGE |
            PrismOpcode::RESOLVE_FUTURE |
            PrismOpcode::REJECT_FUTURE |
            
            // I/O operations
            PrismOpcode::IO_READ(_) |
            PrismOpcode::IO_WRITE(_) |
            PrismOpcode::IO_FLUSH(_) |
            PrismOpcode::IO_CLOSE(_) |
            PrismOpcode::IO_OPEN(_) |
            
            // Memory management
            PrismOpcode::GC_HINT |
            PrismOpcode::REF_INC |
            PrismOpcode::REF_DEC |
            
            // Exception handling
            PrismOpcode::THROW |
            PrismOpcode::RETHROW |
            
            // Debugging operations
            PrismOpcode::BREAKPOINT |
            PrismOpcode::TRACE(_) |
            PrismOpcode::PROFILE_START(_) |
            PrismOpcode::PROFILE_END |
            PrismOpcode::LOG(_)
        )
    }

    fn get_control_flow_type(&self, instruction: &Instruction, next_offset: u32) -> ControlFlowType {
        use crate::bytecode::instructions::PrismOpcode;
        match instruction.opcode {
            PrismOpcode::JUMP(target) => ControlFlowType::UnconditionalJump { target: target as u32 },
            PrismOpcode::JUMP_IF_TRUE(target) => ControlFlowType::ConditionalBranch {
                true_target: target as u32,
                false_target: next_offset,
            },
            PrismOpcode::JUMP_IF_FALSE(target) => ControlFlowType::ConditionalBranch {
                true_target: next_offset,
                false_target: target as u32,
            },
            PrismOpcode::JUMP_IF_NULL(target) => ControlFlowType::ConditionalBranch {
                true_target: target as u32,
                false_target: next_offset,
            },
            PrismOpcode::JUMP_IF_NOT_NULL(target) => ControlFlowType::ConditionalBranch {
                true_target: target as u32,
                false_target: next_offset,
            },
            PrismOpcode::RETURN | PrismOpcode::RETURN_VALUE => ControlFlowType::Return,
            PrismOpcode::THROW | PrismOpcode::RETHROW => ControlFlowType::Throw,
            _ => ControlFlowType::FallThrough,
        }
    }

    fn find_block_by_offset(&self, blocks: &[BasicBlock], offset: u32) -> Option<&BasicBlock> {
        blocks.iter().find(|block| {
            block.instructions.iter().any(|instr| instr.bytecode_offset == offset)
        })
    }

    fn find_exit_blocks(&self, blocks: &[BasicBlock], edges: &[CFGEdge]) -> Vec<u32> {
        let mut has_outgoing = HashSet::new();
        for edge in edges {
            has_outgoing.insert(edge.from);
        }
        
        blocks.iter()
            .filter(|block| !has_outgoing.contains(&block.id))
            .map(|block| block.id)
            .collect()
    }
}

/// Control flow type for instructions
#[derive(Debug, Clone)]
enum ControlFlowType {
    /// Falls through to next instruction
    FallThrough,
    /// Unconditional jump
    UnconditionalJump { target: u32 },
    /// Conditional branch
    ConditionalBranch { true_target: u32, false_target: u32 },
    /// Return from function
    Return,
    /// Throw exception
    Throw,
} 