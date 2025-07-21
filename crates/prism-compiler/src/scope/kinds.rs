//! Scope Kind Classification
//!
//! This module defines the semantic classification of scopes, including
//! module scopes, function scopes, block scopes, and other scope types
//! as specified in PLT-004.
//!
//! **Conceptual Responsibility**: Scope semantic classification
//! **What it does**: Define scope types, semantic meaning, scope categorization
//! **What it doesn't do**: Scope data storage, hierarchy management, visibility rules

use serde::{Serialize, Deserialize};

/// Semantic classification of scopes
/// 
/// Each scope kind carries semantic meaning about what type of code
/// construct it represents and what rules apply within it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScopeKind {
    /// Global/root scope containing all top-level declarations
    Global,
    
    /// Module scope with PLD-002 Smart Module integration
    Module {
        /// Name of the module
        module_name: String,
        /// Module sections (config, types, interface, etc.)
        sections: Vec<String>,
        /// Capabilities this module provides or requires
        capabilities: Vec<String>,
        /// Module-level effects
        effects: Vec<String>,
    },
    
    /// Section within a module (PLD-002 integration)
    Section {
        /// Type of section (config, types, interface, etc.)
        section_type: SectionType,
        /// Name of the parent module
        parent_module: String,
        /// Section-specific capabilities
        capabilities: Vec<String>,
    },
    
    /// Function scope with effects and contracts (PLD-003 integration)
    Function {
        /// Name of the function
        function_name: String,
        /// Function parameters (simplified representation)
        parameters: Vec<String>,
        /// Whether this is an async function
        is_async: bool,
        /// Effects this function may produce
        effects: Vec<String>,
        /// Capabilities this function requires
        required_capabilities: Vec<String>,
        /// Function contracts (preconditions, postconditions)
        contracts: Option<String>,
    },
    
    /// Block scope (within functions, control structures)
    Block {
        /// Type of block
        block_type: BlockType,
    },
    
    /// Type definition scope
    Type {
        /// Name of the type being defined
        type_name: String,
        /// Category of type (struct, enum, trait, etc.)
        type_category: String,
        /// Generic parameters
        generic_parameters: Vec<String>,
    },
    
    /// Control flow scope (if, while, for, match)
    ControlFlow {
        /// Type of control flow construct
        control_type: ControlFlowType,
        /// Condition or pattern (if applicable)
        condition: Option<String>,
    },
    
    /// Lambda/closure scope
    Lambda {
        /// Variables captured from outer scope
        captures: Vec<String>,
        /// Effects the lambda may produce
        effects: Vec<String>,
        /// Whether this is an async lambda
        is_async: bool,
    },
}

/// Module section types following PLD-002 Smart Module System
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SectionType {
    /// Configuration constants and settings
    Config,
    
    /// Type definitions (structs, enums, traits)
    Types,
    
    /// Error type definitions and handling
    Errors,
    
    /// Private implementation details
    Internal,
    
    /// Public API definitions
    Interface,
    
    /// Event definitions and handlers
    Events,
    
    /// Module lifecycle hooks (init, cleanup, etc.)
    Lifecycle,
    
    /// Test cases and validation
    Tests,
    
    /// Usage examples and documentation
    Examples,
    
    /// Performance-critical code sections
    Performance,
    
    /// Custom section type
    Custom(String),
}

/// Block type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    /// Regular block scope
    Regular,
    
    /// Unsafe block requiring justification (PLD-003 integration)
    Unsafe {
        /// Justification for unsafe code
        justification: String,
        /// Required safety capabilities
        required_capabilities: Vec<String>,
    },
    
    /// Effect block for grouping effects (PLD-003 integration)
    Effect {
        /// Effects produced by this block
        effects: Vec<String>,
        /// Effect composition rules
        composition_rules: Vec<String>,
    },
    
    /// Performance-critical block requiring capabilities
    Performance {
        /// Required capabilities for optimization
        capabilities: Vec<String>,
        /// Performance characteristics
        characteristics: Vec<String>,
    },
    
    /// Capability-checked block
    Capability {
        /// Required capabilities to enter this block
        required_capabilities: Vec<String>,
        /// Capability checking mode
        checking_mode: CapabilityCheckingMode,
    },
}

/// Control flow type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlFlowType {
    /// If statement/expression
    If,
    
    /// While loop
    While,
    
    /// For loop
    For,
    
    /// Match expression/pattern matching
    Match,
    
    /// Try/catch error handling
    Try,
    
    /// Async block
    Async,
    
    /// Loop block (general loop construct)
    Loop,
    
    /// Defer block (cleanup code)
    Defer,
}

/// Capability checking modes for capability blocks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityCheckingMode {
    /// Check capabilities at block entry
    EntryOnly,
    
    /// Check capabilities on every operation
    PerOperation,
    
    /// Inherit capability checking from parent scope
    Inherited,
    
    /// Custom capability checking rules
    Custom(String),
}

impl ScopeKind {
    /// Check if this scope kind requires special handling
    pub fn requires_special_handling(&self) -> bool {
        matches!(
            self,
            ScopeKind::Module { .. } |
            ScopeKind::Function { .. } |
            ScopeKind::Block { block_type: BlockType::Unsafe { .. } } |
            ScopeKind::Block { block_type: BlockType::Effect { .. } }
        )
    }
    
    /// Check if this scope can contain other scopes
    pub fn can_contain_scopes(&self) -> bool {
        !matches!(self, ScopeKind::Lambda { .. })
    }
    
    /// Get the capabilities required by this scope
    pub fn required_capabilities(&self) -> Vec<String> {
        match self {
            ScopeKind::Module { capabilities, .. } => capabilities.clone(),
            ScopeKind::Section { capabilities, .. } => capabilities.clone(),
            ScopeKind::Function { required_capabilities, .. } => required_capabilities.clone(),
            ScopeKind::Block { block_type: BlockType::Unsafe { required_capabilities, .. } } => required_capabilities.clone(),
            ScopeKind::Block { block_type: BlockType::Performance { capabilities, .. } } => capabilities.clone(),
            ScopeKind::Block { block_type: BlockType::Capability { required_capabilities, .. } } => required_capabilities.clone(),
            _ => Vec::new(),
        }
    }
    
    /// Get the effects produced by this scope
    pub fn declared_effects(&self) -> Vec<String> {
        match self {
            ScopeKind::Module { effects, .. } => effects.clone(),
            ScopeKind::Function { effects, .. } => effects.clone(),
            ScopeKind::Block { block_type: BlockType::Effect { effects, .. } } => effects.clone(),
            ScopeKind::Lambda { effects, .. } => effects.clone(),
            _ => Vec::new(),
        }
    }
    
    /// Check if this scope is async
    pub fn is_async(&self) -> bool {
        match self {
            ScopeKind::Function { is_async, .. } => *is_async,
            ScopeKind::Lambda { is_async, .. } => *is_async,
            ScopeKind::ControlFlow { control_type: ControlFlowType::Async, .. } => true,
            _ => false,
        }
    }
    
    /// Get a human-readable name for this scope kind
    pub fn kind_name(&self) -> &'static str {
        match self {
            ScopeKind::Global => "global",
            ScopeKind::Module { .. } => "module",
            ScopeKind::Section { .. } => "section",
            ScopeKind::Function { .. } => "function",
            ScopeKind::Block { .. } => "block",
            ScopeKind::Type { .. } => "type",
            ScopeKind::ControlFlow { .. } => "control_flow",
            ScopeKind::Lambda { .. } => "lambda",
        }
    }
    
    /// Get detailed description of this scope
    pub fn detailed_description(&self) -> String {
        match self {
            ScopeKind::Global => "Global scope".to_string(),
            ScopeKind::Module { module_name, sections, .. } => {
                format!("Module '{}' with {} sections", module_name, sections.len())
            }
            ScopeKind::Section { section_type, parent_module, .. } => {
                format!("Section {:?} in module '{}'", section_type, parent_module)
            }
            ScopeKind::Function { function_name, parameters, is_async, .. } => {
                let async_str = if *is_async { "async " } else { "" };
                format!("{}Function '{}' with {} parameters", async_str, function_name, parameters.len())
            }
            ScopeKind::Block { block_type } => {
                format!("Block ({:?})", block_type)
            }
            ScopeKind::Type { type_name, type_category, .. } => {
                format!("Type '{}' ({})", type_name, type_category)
            }
            ScopeKind::ControlFlow { control_type, .. } => {
                format!("Control flow ({:?})", control_type)
            }
            ScopeKind::Lambda { captures, is_async, .. } => {
                let async_str = if *is_async { "async " } else { "" };
                format!("{}Lambda capturing {} variables", async_str, captures.len())
            }
        }
    }
}

impl SectionType {
    /// Get all standard section types
    pub fn standard_types() -> Vec<SectionType> {
        vec![
            SectionType::Config,
            SectionType::Types,
            SectionType::Errors,
            SectionType::Internal,
            SectionType::Interface,
            SectionType::Events,
            SectionType::Lifecycle,
            SectionType::Tests,
            SectionType::Examples,
            SectionType::Performance,
        ]
    }
    
    /// Check if this is a public section (part of module's public API)
    pub fn is_public(&self) -> bool {
        matches!(
            self,
            SectionType::Interface |
            SectionType::Types |
            SectionType::Events |
            SectionType::Examples
        )
    }
    
    /// Check if this is an internal section (implementation details)
    pub fn is_internal(&self) -> bool {
        matches!(
            self,
            SectionType::Internal |
            SectionType::Config |
            SectionType::Performance
        )
    }
    
    /// Get the typical ordering priority for this section type
    pub fn ordering_priority(&self) -> u8 {
        match self {
            SectionType::Config => 1,
            SectionType::Types => 2,
            SectionType::Errors => 3,
            SectionType::Interface => 4,
            SectionType::Events => 5,
            SectionType::Internal => 6,
            SectionType::Lifecycle => 7,
            SectionType::Performance => 8,
            SectionType::Tests => 9,
            SectionType::Examples => 10,
            SectionType::Custom(_) => 99,
        }
    }
}

impl BlockType {
    /// Check if this block type requires special capabilities
    pub fn requires_capabilities(&self) -> bool {
        matches!(
            self,
            BlockType::Unsafe { .. } |
            BlockType::Performance { .. } |
            BlockType::Capability { .. }
        )
    }
    
    /// Check if this block type affects the effect system
    pub fn affects_effects(&self) -> bool {
        matches!(self, BlockType::Effect { .. })
    }
    
    /// Get the security level required for this block type
    pub fn security_level(&self) -> SecurityLevel {
        match self {
            BlockType::Regular => SecurityLevel::Normal,
            BlockType::Unsafe { .. } => SecurityLevel::Elevated,
            BlockType::Effect { .. } => SecurityLevel::Normal,
            BlockType::Performance { .. } => SecurityLevel::Elevated,
            BlockType::Capability { .. } => SecurityLevel::Controlled,
        }
    }
}

/// Security levels for different scope types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Normal security level
    Normal,
    /// Elevated security level (requires justification)
    Elevated,
    /// Controlled security level (strict capability checking)
    Controlled,
    /// Critical security level (maximum restrictions)
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scope_kind_capabilities() {
        let module = ScopeKind::Module {
            module_name: "test".to_string(),
            sections: vec!["interface".to_string()],
            capabilities: vec!["read".to_string(), "write".to_string()],
            effects: vec![],
        };
        
        let caps = module.required_capabilities();
        assert_eq!(caps.len(), 2);
        assert!(caps.contains(&"read".to_string()));
        assert!(caps.contains(&"write".to_string()));
    }
    
    #[test]
    fn test_section_type_properties() {
        assert!(SectionType::Interface.is_public());
        assert!(!SectionType::Internal.is_public());
        assert!(SectionType::Internal.is_internal());
        assert!(!SectionType::Interface.is_internal());
        
        assert_eq!(SectionType::Config.ordering_priority(), 1);
        assert_eq!(SectionType::Examples.ordering_priority(), 10);
    }
    
    #[test]
    fn test_block_type_security() {
        let regular = BlockType::Regular;
        let unsafe_block = BlockType::Unsafe {
            justification: "Performance critical".to_string(),
            required_capabilities: vec!["unsafe".to_string()],
        };
        
        assert!(!regular.requires_capabilities());
        assert!(unsafe_block.requires_capabilities());
        
        assert_eq!(regular.security_level(), SecurityLevel::Normal);
        assert_eq!(unsafe_block.security_level(), SecurityLevel::Elevated);
    }
    
    #[test]
    fn test_scope_kind_descriptions() {
        let global = ScopeKind::Global;
        assert_eq!(global.kind_name(), "global");
        assert_eq!(global.detailed_description(), "Global scope");
        
        let function = ScopeKind::Function {
            function_name: "test".to_string(),
            parameters: vec!["a".to_string(), "b".to_string()],
            is_async: true,
            effects: vec![],
            required_capabilities: vec![],
            contracts: None,
        };
        
        assert_eq!(function.kind_name(), "function");
        assert!(function.is_async());
        assert!(function.detailed_description().contains("async"));
    }
} 