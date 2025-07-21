//! Symbol Builder for Convenient Construction
//!
//! This module provides a fluent builder API for constructing symbols with
//! comprehensive metadata. It simplifies the process of creating well-formed
//! symbols while ensuring all required information is provided.
//!
//! ## Conceptual Responsibility
//! 
//! This module handles ONE thing: "Symbol Construction and Building"
//! - Fluent API for symbol creation
//! - Validation of symbol data during construction
//! - Default value provision for optional fields
//! - Integration with metadata generation
//! 
//! It does NOT handle:
//! - Symbol storage (delegated to table.rs)
//! - Symbol resolution (delegated to resolution subsystem)
//! - Symbol metadata analysis (delegated to metadata.rs)

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::data::{SymbolData, SymbolVisibility, SymbolEffect, CompilationContext, CompilationPhase};
use crate::symbols::kinds::SymbolKind;
use crate::symbols::metadata::{SymbolMetadata, AISymbolContext, SymbolDocumentation};
use prism_common::{symbol::Symbol, span::Span, NodeId};

/// Builder for constructing symbols with fluent API
#[derive(Debug, Clone)]
pub struct SymbolBuilder {
    /// Symbol name (required)
    name: Option<String>,
    
    /// Symbol kind (required)
    kind: Option<SymbolKind>,
    
    /// Source location (required)
    location: Option<Span>,
    
    /// Symbol visibility (optional, defaults to Private)
    visibility: SymbolVisibility,
    
    /// Associated AST node
    ast_node: Option<NodeId>,
    
    /// Semantic type reference
    semantic_type: Option<String>,
    
    /// Effects for this symbol
    effects: Vec<SymbolEffect>,
    
    /// Required capabilities
    required_capabilities: Vec<String>,
    
    /// Symbol metadata
    metadata: SymbolMetadata,
    
    /// Compilation context
    compilation_context: CompilationContext,
    
    /// Builder configuration
    config: SymbolBuilderConfig,
}

/// Configuration for symbol builder behavior
#[derive(Debug, Clone)]
pub struct SymbolBuilderConfig {
    /// Automatically generate AI context if not provided
    pub auto_generate_ai_context: bool,
    
    /// Validate symbol data during construction
    pub validate_during_build: bool,
    
    /// Generate default documentation if not provided
    pub generate_default_docs: bool,
    
    /// Automatically infer semantic type from kind
    pub auto_infer_semantic_type: bool,
    
    /// Enable strict validation of required fields
    pub strict_validation: bool,
}

impl Default for SymbolBuilderConfig {
    fn default() -> Self {
        Self {
            auto_generate_ai_context: true,
            validate_during_build: true,
            generate_default_docs: false,
            auto_infer_semantic_type: true,
            strict_validation: true,
        }
    }
}

impl Default for SymbolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolBuilder {
    /// Create a new symbol builder
    pub fn new() -> Self {
        Self {
            name: None,
            kind: None,
            location: None,
            visibility: SymbolVisibility::Private,
            ast_node: None,
            semantic_type: None,
            effects: Vec::new(),
            required_capabilities: Vec::new(),
            metadata: SymbolMetadata::default(),
            compilation_context: CompilationContext {
                creation_phase: CompilationPhase::Parsing,
                ..Default::default()
            },
            config: SymbolBuilderConfig::default(),
        }
    }
    
    /// Create a new symbol builder with custom configuration
    pub fn with_config(config: SymbolBuilderConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }
    
    /// Set the symbol name (required)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Set the symbol kind (required)
    pub fn with_kind(mut self, kind: SymbolKind) -> Self {
        self.kind = Some(kind);
        self
    }
    
    /// Set the source location (required)
    pub fn with_location(mut self, location: Span) -> Self {
        self.location = Some(location);
        self
    }
    
    /// Set the symbol visibility
    pub fn with_visibility(mut self, visibility: SymbolVisibility) -> Self {
        self.visibility = visibility;
        self
    }
    
    /// Make the symbol public
    pub fn public(mut self) -> Self {
        self.visibility = SymbolVisibility::Public;
        self
    }
    
    /// Make the symbol private
    pub fn private(mut self) -> Self {
        self.visibility = SymbolVisibility::Private;
        self
    }
    
    /// Make the symbol internal
    pub fn internal(mut self) -> Self {
        self.visibility = SymbolVisibility::Internal;
        self
    }
    
    /// Set the associated AST node
    pub fn with_ast_node(mut self, ast_node: NodeId) -> Self {
        self.ast_node = Some(ast_node);
        self
    }
    
    /// Set the semantic type reference
    pub fn with_semantic_type(mut self, semantic_type: impl Into<String>) -> Self {
        self.semantic_type = Some(semantic_type.into());
        self
    }
    
    /// Add an effect to the symbol
    pub fn with_effect(mut self, effect: SymbolEffect) -> Self {
        self.effects.push(effect);
        self
    }
    
    /// Add multiple effects to the symbol
    pub fn with_effects(mut self, effects: Vec<SymbolEffect>) -> Self {
        self.effects.extend(effects);
        self
    }
    
    /// Add a required capability
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.required_capabilities.push(capability.into());
        self
    }
    
    /// Add multiple required capabilities
    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.required_capabilities.extend(capabilities);
        self
    }
    
    /// Set the symbol metadata
    pub fn with_metadata(mut self, metadata: SymbolMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// Set AI context for the symbol
    pub fn with_ai_context(mut self, ai_context: AISymbolContext) -> Self {
        self.metadata.ai_context = Some(ai_context);
        self
    }
    
    /// Set documentation for the symbol
    pub fn with_documentation(mut self, documentation: SymbolDocumentation) -> Self {
        self.metadata.documentation = Some(documentation);
        self
    }
    
    /// Set business responsibility for the symbol
    pub fn with_responsibility(mut self, responsibility: impl Into<String>) -> Self {
        let responsibility_str = responsibility.into();
        
        // Create or update business context
        if let Some(ref mut business_context) = self.metadata.business_context {
            business_context.domain = responsibility_str;
        } else {
            self.metadata.business_context = Some(
                crate::symbols::metadata::BusinessContext {
                    domain: responsibility_str,
                    business_rules: Vec::new(),
                    stakeholders: Vec::new(),
                    compliance_requirements: Vec::new(),
                    business_impact: None,
                    resource_info: None,
                }
            );
        }
        
        self
    }
    
    /// Set compilation context
    pub fn with_compilation_context(mut self, compilation_context: CompilationContext) -> Self {
        self.compilation_context = compilation_context;
        self
    }
    
    /// Set compilation phase
    pub fn with_compilation_phase(mut self, phase: CompilationPhase) -> Self {
        self.compilation_context.creation_phase = phase;
        self
    }
    
    /// Set source file information
    pub fn with_source_file(mut self, source_file: impl Into<String>) -> Self {
        self.compilation_context.source_file = Some(source_file.into());
        self
    }
    
    /// Build the symbol data
    pub fn build(self) -> CompilerResult<SymbolData> {
        // Validate required fields
        let name = self.name.ok_or_else(|| CompilerError::InvalidInput {
            message: "Symbol name is required".to_string(),
        })?;
        
        let kind = self.kind.ok_or_else(|| CompilerError::InvalidInput {
            message: "Symbol kind is required".to_string(),
        })?;
        
        let location = self.location.ok_or_else(|| CompilerError::InvalidInput {
            message: "Symbol location is required".to_string(),
        })?;
        
        // Intern the symbol name
        let symbol = Symbol::intern(&name);
        
        // Auto-generate AI context if enabled and not provided
        let mut metadata = self.metadata;
        if self.config.auto_generate_ai_context && metadata.ai_context.is_none() {
            metadata.ai_context = Some(self.generate_default_ai_context(&name, &kind));
        }
        
        // Auto-generate documentation if enabled and not provided
        if self.config.generate_default_docs && metadata.documentation.is_none() {
            metadata.documentation = Some(self.generate_default_documentation(&name, &kind));
        }
        
        // Auto-infer semantic type if enabled and not provided
        let semantic_type = if self.config.auto_infer_semantic_type && self.semantic_type.is_none() {
            Some(self.infer_semantic_type(&kind))
        } else {
            self.semantic_type
        };
        
        // Update metadata confidence and timestamp
        metadata.last_updated = Some(std::time::SystemTime::now());
        metadata.calculate_confidence();
        
        // Create the symbol data
        let mut symbol_data = SymbolData::new(symbol, name, kind, location);
        symbol_data.visibility = self.visibility;
        symbol_data.ast_node = self.ast_node;
        symbol_data.semantic_type = semantic_type;
        symbol_data.effects = self.effects;
        symbol_data.required_capabilities = self.required_capabilities;
        symbol_data.metadata = metadata;
        symbol_data.compilation_context = self.compilation_context;
        
        // Validate the constructed symbol if enabled
        if self.config.validate_during_build {
            self.validate_symbol_data(&symbol_data)?;
        }
        
        Ok(symbol_data)
    }
    
    /// Generate default AI context based on symbol name and kind
    fn generate_default_ai_context(&self, name: &str, kind: &SymbolKind) -> AISymbolContext {
        let purpose = match kind {
            SymbolKind::Function { .. } => format!("Function '{}' performs a specific operation", name),
            SymbolKind::Type { .. } => format!("Type '{}' represents a data structure or concept", name),
            SymbolKind::Variable { .. } => format!("Variable '{}' stores a value", name),
            SymbolKind::Constant { .. } => format!("Constant '{}' represents a fixed value", name),
            SymbolKind::Module { .. } => format!("Module '{}' organizes related functionality", name),
            _ => format!("Symbol '{}' serves a specific purpose in the codebase", name),
        };
        
        AISymbolContext::new(purpose)
    }
    
    /// Generate default documentation based on symbol name and kind
    fn generate_default_documentation(&self, name: &str, kind: &SymbolKind) -> SymbolDocumentation {
        let summary = match kind {
            SymbolKind::Function { .. } => format!("Function {}", name),
            SymbolKind::Type { .. } => format!("Type definition for {}", name),
            SymbolKind::Variable { .. } => format!("Variable {}", name),
            SymbolKind::Constant { .. } => format!("Constant {}", name),
            SymbolKind::Module { .. } => format!("Module {}", name),
            _ => format!("Symbol {}", name),
        };
        
        SymbolDocumentation::new(summary)
    }
    
    /// Infer semantic type from symbol kind
    fn infer_semantic_type(&self, kind: &SymbolKind) -> String {
        match kind {
            SymbolKind::Function { .. } => "function_type".to_string(),
            SymbolKind::Type { type_category, .. } => {
                format!("type_{}", type_category.ai_category())
            },
            SymbolKind::Variable { .. } => "variable_type".to_string(),
            SymbolKind::Constant { .. } => "constant_type".to_string(),
            SymbolKind::Module { .. } => "module_type".to_string(),
            SymbolKind::Parameter { .. } => "parameter_type".to_string(),
            SymbolKind::Import { .. } => "import_type".to_string(),
            SymbolKind::Export { .. } => "export_type".to_string(),
            SymbolKind::Capability { .. } => "capability_type".to_string(),
            SymbolKind::Effect { .. } => "effect_type".to_string(),
            SymbolKind::Section { .. } => "section_type".to_string(),
        }
    }
    
    /// Validate the constructed symbol data
    fn validate_symbol_data(&self, symbol_data: &SymbolData) -> CompilerResult<()> {
        if self.config.strict_validation {
            // Validate name is not empty
            if symbol_data.name.is_empty() {
                return Err(CompilerError::InvalidInput {
                    message: "Symbol name cannot be empty".to_string(),
                });
            }
            
            // Validate location is not dummy (in strict mode)
            if symbol_data.location == Span::dummy() {
                return Err(CompilerError::InvalidInput {
                    message: "Symbol location cannot be dummy in strict mode".to_string(),
                });
            }
            
            // Validate effects have names
            for effect in &symbol_data.effects {
                if effect.name.is_empty() {
                    return Err(CompilerError::InvalidInput {
                        message: "Effect names cannot be empty".to_string(),
                    });
                }
            }
            
            // Validate capabilities are not empty
            for capability in &symbol_data.required_capabilities {
                if capability.is_empty() {
                    return Err(CompilerError::InvalidInput {
                        message: "Capability names cannot be empty".to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the current builder configuration
    pub fn config(&self) -> &SymbolBuilderConfig {
        &self.config
    }
}

// Helper trait for type categories to provide AI-readable categories
trait AICategory {
    fn ai_category(&self) -> &'static str;
}

impl AICategory for crate::symbols::kinds::TypeCategory {
    fn ai_category(&self) -> &'static str {
        match self {
            crate::symbols::kinds::TypeCategory::Primitive { .. } => "primitive",
            crate::symbols::kinds::TypeCategory::Semantic { .. } => "semantic",
            crate::symbols::kinds::TypeCategory::Composite { .. } => "composite",
            crate::symbols::kinds::TypeCategory::Function { .. } => "function",
            crate::symbols::kinds::TypeCategory::Generic { .. } => "generic",
            crate::symbols::kinds::TypeCategory::Effect { .. } => "effect",
        }
    }
}

/// Convenience functions for common symbol construction patterns
impl SymbolBuilder {
    /// Create a function symbol
    pub fn function(name: impl Into<String>, location: Span) -> Self {
        Self::new()
            .with_name(name)
            .with_kind(SymbolKind::Function {
                signature: crate::symbols::kinds::FunctionSignature::default(),
                function_type: crate::symbols::kinds::FunctionType::Regular,
                contracts: None,
                performance_info: None,
            })
            .with_location(location)
    }
    
    /// Create a variable symbol
    pub fn variable(name: impl Into<String>, location: Span, is_mutable: bool) -> Self {
        Self::new()
            .with_name(name)
            .with_kind(SymbolKind::Variable {
                variable_info: crate::symbols::kinds::VariableInfo {
                    is_mutable,
                    ..Default::default()
                },
                initialization: None,
                usage_patterns: Vec::new(),
            })
            .with_location(location)
    }
    
    /// Create a type symbol
    pub fn type_symbol(name: impl Into<String>, location: Span, type_category: crate::symbols::kinds::TypeCategory) -> Self {
        Self::new()
            .with_name(name)
            .with_kind(SymbolKind::Type {
                type_category,
                constraints: Vec::new(),
                semantic_info: None,
                business_rules: Vec::new(),
            })
            .with_location(location)
    }
    
    /// Create a module symbol
    pub fn module(name: impl Into<String>, location: Span) -> Self {
        Self::new()
            .with_name(name)
            .with_kind(SymbolKind::Module {
                sections: Vec::new(),
                capabilities: Vec::new(),
                effects: Vec::new(),
                cohesion_info: None,
            })
            .with_location(location)
    }
    
    /// Create a constant symbol
    pub fn constant(name: impl Into<String>, location: Span) -> Self {
        Self::new()
            .with_name(name)
            .with_kind(SymbolKind::Constant {
                constant_info: crate::symbols::kinds::ConstantInfo {
                    value_type: None,
                    is_compile_time: true,
                },
                compile_time_value: None,
            })
            .with_location(location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::kinds::{TypeCategory, PrimitiveType};
    
    #[test]
    fn test_basic_symbol_construction() {
        let symbol_data = SymbolBuilder::new()
            .with_name("test_symbol")
            .with_kind(SymbolKind::Variable {
                variable_info: crate::symbols::kinds::VariableInfo::default(),
                initialization: None,
                usage_patterns: Vec::new(),
            })
            .with_location(Span::dummy())
            .build()
            .unwrap();
        
        assert_eq!(symbol_data.name, "test_symbol");
        assert!(matches!(symbol_data.kind, SymbolKind::Variable { .. }));
    }
    
    #[test]
    fn test_convenience_functions() {
        let function_symbol = SymbolBuilder::function("test_func", Span::dummy())
            .public()
            .build()
            .unwrap();
        
        assert_eq!(function_symbol.name, "test_func");
        assert!(matches!(function_symbol.kind, SymbolKind::Function { .. }));
        assert_eq!(function_symbol.visibility, SymbolVisibility::Public);
        
        let variable_symbol = SymbolBuilder::variable("test_var", Span::dummy(), true)
            .private()
            .build()
            .unwrap();
        
        assert_eq!(variable_symbol.name, "test_var");
        assert!(matches!(variable_symbol.kind, SymbolKind::Variable { .. }));
        assert_eq!(variable_symbol.visibility, SymbolVisibility::Private);
    }
    
    #[test]
    fn test_validation() {
        // Missing name should fail
        let result = SymbolBuilder::new()
            .with_kind(SymbolKind::Variable {
                variable_info: crate::symbols::kinds::VariableInfo::default(),
                initialization: None,
                usage_patterns: Vec::new(),
            })
            .with_location(Span::dummy())
            .build();
        
        assert!(result.is_err());
        
        // Missing kind should fail
        let result = SymbolBuilder::new()
            .with_name("test")
            .with_location(Span::dummy())
            .build();
        
        assert!(result.is_err());
    }
} 