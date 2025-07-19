//! Abstract Syntax Tree definitions for the Prism programming language
//!
//! This crate provides AST node definitions with rich semantic metadata designed
//! for AI-first development, semantic type analysis, and robust compiler infrastructure.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod arena;
pub mod expr;
pub mod metadata;
pub mod node;
pub mod pattern;
pub mod stmt;
pub mod types;
pub mod visitor;
pub mod type_inference;
pub mod transformations;

// Re-export main types
pub use arena::AstArena;
pub use expr::*;
pub use metadata::*;
pub use node::*;
pub use pattern::*;
pub use stmt::*;
pub use types::*;
pub use visitor::*;
pub use type_inference::*;
pub use transformations::*;

use prism_common::SourceId;
use std::collections::HashMap;

/// The root AST node representing a complete Prism program
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Program {
    /// Top-level statements and declarations
    pub items: Vec<AstNode<Item>>,
    /// Source file this program was parsed from
    pub source_id: SourceId,
    /// AI-readable metadata about the program
    pub metadata: ProgramMetadata,
}

/// Top-level items in a Prism program
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Item {
    /// Module declaration
    Module(ModuleDecl),
    /// Function declaration
    Function(FunctionDecl),
    /// Type declaration
    Type(TypeDecl),
    /// Import statement
    Import(ImportDecl),
    /// Export statement
    Export(ExportDecl),
    /// Constant declaration
    Const(ConstDecl),
    /// Variable declaration
    Variable(VariableDecl),
    /// Statement (for recovery)
    Statement(Stmt),
}

/// AI-readable metadata about a program
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProgramMetadata {
    /// The main capability this program provides
    pub primary_capability: Option<String>,
    /// All capabilities referenced in this program
    pub capabilities: Vec<String>,
    /// Dependencies identified in this program
    pub dependencies: Vec<String>,
    /// Security implications of this program
    pub security_implications: Vec<String>,
    /// Performance characteristics
    pub performance_notes: Vec<String>,
    /// AI-generated insights about the program
    pub ai_insights: Vec<String>,
}

impl Program {
    /// Create a new program
    pub fn new(items: Vec<AstNode<Item>>, source_id: SourceId) -> Self {
        Self {
            items,
            source_id,
            metadata: ProgramMetadata::default(),
        }
    }

    /// Add AI-readable metadata to this program
    pub fn with_metadata(mut self, metadata: ProgramMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get all modules in this program
    pub fn modules(&self) -> impl Iterator<Item = &ModuleDecl> {
        self.items.iter().filter_map(|item| match &item.kind {
            Item::Module(module) => Some(module),
            _ => None,
        })
    }

    /// Get all functions in this program
    pub fn functions(&self) -> impl Iterator<Item = &FunctionDecl> {
        self.items.iter().filter_map(|item| match &item.kind {
            Item::Function(func) => Some(func),
            _ => None,
        })
    }

    /// Get all type declarations in this program
    pub fn types(&self) -> impl Iterator<Item = &TypeDecl> {
        self.items.iter().filter_map(|item| match &item.kind {
            Item::Type(ty) => Some(ty),
            _ => None,
        })
    }

    /// Collect all AI contexts from this program
    pub fn collect_ai_contexts(&self) -> Vec<&AiContext> {
        let mut contexts = Vec::new();
        for item in &self.items {
            if let Some(ai_context) = &item.metadata.ai_context {
                contexts.push(ai_context);
            }
        }
        contexts
    }

    /// Get a summary of semantic types used in this program
    pub fn semantic_type_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        
        // Count semantic types across all items
        for item in &self.items {
            match &item.kind {
                Item::Type(type_decl) => {
                    if let TypeKind::Semantic(_) = &type_decl.kind {
                        *summary.entry("semantic_types".to_string()).or_insert(0) += 1;
                    }
                }
                Item::Function(func) => {
                    if func.return_type.is_some() {
                        *summary.entry("typed_functions".to_string()).or_insert(0) += 1;
                    }
                }
                _ => {}
            }
        }
        
        summary
    }
}
