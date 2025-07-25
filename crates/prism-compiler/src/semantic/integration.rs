//! Semantic Type Integration Bridge
//!
//! This module provides the integration layer between AST semantic types and the
//! semantic analysis system. It respects Separation of Concerns by:
//!
//! 1. **Pure Integration**: Only converts between representations, no business logic
//! 2. **Delegation**: Uses existing subsystems instead of duplicating functionality
//! 3. **Clear Boundaries**: AST → Semantic → PIR conversion chain
//!
//! **Conceptual Responsibility**: Type system integration
//! **What it does**: Convert AST semantic types to semantic analysis types
//! **What it doesn't do**: Define types, validate types, store types (delegates)

use crate::error::{CompilerError, CompilerResult};
use crate::context::CompilationContext;
use prism_ast::{Program, AstNode, Type, Item};
use prism_semantic::{SemanticType, SemanticTypeSystem, SemanticResult};
use prism_common::{NodeId, span::Span};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug, warn};

/// Semantic type integration coordinator
/// 
/// This coordinates the conversion of AST semantic types to semantic analysis types
/// without duplicating functionality from other subsystems.
#[derive(Debug)]
pub struct SemanticTypeIntegration {
    /// Reference to semantic type system (doesn't own it)
    semantic_system: Arc<SemanticTypeSystem>,
}

/// Result of semantic type integration
#[derive(Debug, Clone)]
pub struct SemanticTypeIntegrationResult {
    /// Converted semantic types indexed by node ID
    pub semantic_types: HashMap<NodeId, SemanticType>,
    /// Integration metadata
    pub metadata: IntegrationMetadata,
    /// Any warnings during conversion
    pub warnings: Vec<String>,
}

/// Metadata about the integration process
#[derive(Debug, Clone)]
pub struct IntegrationMetadata {
    /// Number of AST semantic types processed
    pub ast_types_processed: usize,
    /// Number of successful conversions
    pub successful_conversions: usize,
    /// Number of failed conversions
    pub failed_conversions: usize,
    /// Integration duration
    pub integration_duration: std::time::Duration,
}

impl SemanticTypeIntegration {
    /// Create new semantic type integration
    pub fn new(semantic_system: Arc<SemanticTypeSystem>) -> Self {
        Self { semantic_system }
    }

    /// Integrate semantic types from an entire program
    /// 
    /// This is the main entry point for semantic type integration
    pub async fn integrate_program_types(
        &self,
        program: &Program,
        context: &CompilationContext,
    ) -> CompilerResult<SemanticTypeIntegrationResult> {
        let start_time = std::time::Instant::now();
        info!("Starting semantic type integration for program");

        let mut semantic_types = HashMap::new();
        let mut warnings = Vec::new();
        let mut ast_types_processed = 0;
        let mut successful_conversions = 0;
        let mut failed_conversions = 0;

        // Process all items in the program
        for item in &program.items {
            match self.integrate_item_types(item, context).await {
                Ok(item_result) => {
                    // Merge results
                    for (node_id, semantic_type) in item_result.semantic_types {
                        semantic_types.insert(node_id, semantic_type);
                    }
                    warnings.extend(item_result.warnings);
                    ast_types_processed += item_result.metadata.ast_types_processed;
                    successful_conversions += item_result.metadata.successful_conversions;
                    failed_conversions += item_result.metadata.failed_conversions;
                }
                Err(error) => {
                    warn!("Failed to integrate item types: {}", error);
                    failed_conversions += 1;
                    warnings.push(format!("Failed to integrate item: {}", error));
                }
            }
        }

        let integration_duration = start_time.elapsed();
        info!(
            "Semantic type integration completed: {} AST types processed, {} successful, {} failed",
            ast_types_processed, successful_conversions, failed_conversions
        );

        Ok(SemanticTypeIntegrationResult {
            semantic_types,
            metadata: IntegrationMetadata {
                ast_types_processed,
                successful_conversions,
                failed_conversions,
                integration_duration,
            },
            warnings,
        })
    }

    /// Integrate semantic types from a single AST item
    async fn integrate_item_types(
        &self,
        item: &AstNode<Item>,
        context: &CompilationContext,
    ) -> CompilerResult<SemanticTypeIntegrationResult> {
        let mut semantic_types = HashMap::new();
        let mut warnings = Vec::new();
        let mut ast_types_processed = 0;
        let mut successful_conversions = 0;
        let mut failed_conversions = 0;

        // Extract semantic types from the item
        match self.extract_semantic_types_from_item(item) {
            Ok(extracted_types) => {
                for (ast_type, location) in extracted_types {
                    ast_types_processed += 1;
                    
                    // Convert AST semantic type to semantic analysis type
                    match self.convert_ast_semantic_type(&ast_type, location) {
                        Ok(semantic_type) => {
                            semantic_types.insert(item.node_id(), semantic_type);
                            successful_conversions += 1;
                        }
                        Err(error) => {
                            warn!("Failed to convert semantic type: {}", error);
                            failed_conversions += 1;
                            warnings.push(format!("Conversion failed: {}", error));
                        }
                    }
                }
            }
            Err(error) => {
                warn!("Failed to extract semantic types from item: {}", error);
                failed_conversions += 1;
                warnings.push(format!("Extraction failed: {}", error));
            }
        }

        Ok(SemanticTypeIntegrationResult {
            semantic_types,
            metadata: IntegrationMetadata {
                ast_types_processed,
                successful_conversions,
                failed_conversions,
                integration_duration: std::time::Duration::from_millis(0),
            },
            warnings,
        })
    }

    /// Extract semantic types from an AST item
    /// 
    /// This traverses the AST to find all semantic type annotations
    fn extract_semantic_types_from_item(
        &self,
        item: &AstNode<Item>,
    ) -> CompilerResult<Vec<(prism_ast::SemanticType, Span)>> {
        let mut semantic_types = Vec::new();

        // Navigate the AST to find semantic types
        match item.data() {
            Item::Function(func) => {
                // Check function return type for semantic annotations
                if let Some(return_type) = &func.return_type {
                    if let Type::Semantic(semantic_type) = return_type.data() {
                        semantic_types.push((semantic_type.clone(), return_type.span()));
                    }
                }

                // Check parameter types for semantic annotations
                for param in &func.parameters {
                    if let Some(param_type) = &param.type_annotation {
                        if let Type::Semantic(semantic_type) = param_type.data() {
                            semantic_types.push((semantic_type.clone(), param_type.span()));
                        }
                    }
                }
            }
            Item::TypeAlias(type_alias) => {
                // Check if the aliased type is semantic
                if let Type::Semantic(semantic_type) = type_alias.type_expression.data() {
                    semantic_types.push((semantic_type.clone(), type_alias.type_expression.span()));
                }
            }
            Item::Struct(struct_def) => {
                // Check struct field types for semantic annotations
                for field in &struct_def.fields {
                    if let Type::Semantic(semantic_type) = field.field_type.data() {
                        semantic_types.push((semantic_type.clone(), field.field_type.span()));
                    }
                }
            }
            _ => {
                // For other item types, we don't extract semantic types for now
                debug!("Item type {:?} not yet supported for semantic type extraction", item.data());
            }
        }

        Ok(semantic_types)
    }

    /// Convert AST semantic type to semantic analysis type
    /// 
    /// This is a pure conversion function that delegates to the semantic type system
    fn convert_ast_semantic_type(
        &self,
        ast_semantic_type: &prism_ast::SemanticType,
        location: Span,
    ) -> CompilerResult<SemanticType> {
        // Use the semantic type system's conversion function
        SemanticType::from_ast_semantic_type(ast_semantic_type, location)
            .map_err(|semantic_error| {
                CompilerError::SemanticError {
                    message: format!("Failed to convert AST semantic type: {}", semantic_error),
                    location,
                }
            })
    }
}

/// Integration utilities
impl SemanticTypeIntegration {
    /// Check if an AST type node contains semantic type information
    pub fn has_semantic_types(type_node: &AstNode<Type>) -> bool {
        matches!(type_node.data(), Type::Semantic(_))
    }

    /// Extract all semantic types from a type node recursively
    pub fn extract_all_semantic_types(type_node: &AstNode<Type>) -> Vec<(prism_ast::SemanticType, Span)> {
        let mut semantic_types = Vec::new();

        match type_node.data() {
            Type::Semantic(semantic_type) => {
                semantic_types.push((semantic_type.clone(), type_node.span()));
            }
            Type::Function(func_type) => {
                // Check function parameter types
                for param in &func_type.parameters {
                    semantic_types.extend(Self::extract_all_semantic_types(&param.type_annotation));
                }
                // Check return type
                semantic_types.extend(Self::extract_all_semantic_types(&func_type.return_type));
            }
            Type::Tuple(tuple_type) => {
                // Check tuple element types
                for element_type in &tuple_type.element_types {
                    semantic_types.extend(Self::extract_all_semantic_types(element_type));
                }
            }
            Type::Array(array_type) => {
                // Check array element type
                semantic_types.extend(Self::extract_all_semantic_types(&array_type.element_type));
            }
            _ => {
                // For other types, no semantic types to extract
            }
        }

        semantic_types
    }
} 