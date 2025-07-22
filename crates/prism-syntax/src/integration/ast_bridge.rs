//! Bridge to prism-ast for AST conversion.
//!
//! This module provides conversion between prism-syntax canonical forms
//! and prism-ast AST structures, preserving semantic information and
//! maintaining proper interfacing between the syntax and AST layers.

use crate::normalization::{CanonicalForm, CanonicalNode, CanonicalExpression, CanonicalStatement, CanonicalType};
use prism_ast::{Program, AstNode, Item, Expr, Stmt, Type, AstArena, ProgramMetadata};
use prism_common::{NodeId, Span, SourceId};
use thiserror::Error;
use std::collections::HashMap;

/// Configuration for AST conversion
#[derive(Debug, Clone)]
pub struct AstBridgeConfig {
    /// Preserve source locations
    pub preserve_locations: bool,
    /// Generate node IDs
    pub generate_node_ids: bool,
    /// Include metadata
    pub include_metadata: bool,
    /// Source ID for the conversion
    pub source_id: SourceId,
}

impl Default for AstBridgeConfig {
    fn default() -> Self {
        Self {
            preserve_locations: true,
            generate_node_ids: true,
            include_metadata: true,
            source_id: SourceId::new(0),
        }
    }
}

/// Bridge for AST conversion
#[derive(Debug)]
pub struct AstBridge {
    /// Configuration
    config: AstBridgeConfig,
    /// Node ID counter
    node_id_counter: u32,
    /// AST arena for memory management
    arena: AstArena,
}

/// Result of AST conversion
#[derive(Debug)]
pub struct AstConversionResult {
    /// Converted AST program
    pub program: Program,
    /// Conversion statistics
    pub stats: ConversionStats,
    /// Warnings generated during conversion
    pub warnings: Vec<ConversionWarning>,
}

/// Conversion statistics
#[derive(Debug, Default)]
pub struct ConversionStats {
    /// Number of items converted
    pub items_converted: usize,
    /// Number of expressions converted
    pub expressions_converted: usize,
    /// Number of statements converted
    pub statements_converted: usize,
    /// Number of types converted
    pub types_converted: usize,
    /// Conversion time in milliseconds
    pub conversion_time_ms: u64,
}

/// Conversion warning
#[derive(Debug)]
pub struct ConversionWarning {
    /// Warning message
    pub message: String,
    /// Location where warning occurred
    pub location: Option<Span>,
}

/// AST integration errors
#[derive(Debug, Error)]
pub enum AstIntegrationError {
    /// Conversion failed
    #[error("AST conversion failed: {reason}")]
    ConversionFailed { reason: String },
}

impl AstBridge {
    /// Create new AST bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(AstBridgeConfig::default())
    }
    
    /// Create new AST bridge with custom configuration
    pub fn with_config(config: AstBridgeConfig) -> Self {
        Self {
            config,
            node_id_counter: 1, // Start at 1, reserve 0 for special cases
            arena: AstArena::new(),
        }
    }
    
    /// Convert canonical form to AST
    pub fn to_ast(&mut self, canonical: &CanonicalForm) -> Result<AstConversionResult, AstIntegrationError> {
        let start_time = std::time::Instant::now();
        let mut stats = ConversionStats::default();
        let mut warnings = Vec::new();
        
        // Convert the canonical form to AST items
        let mut items = Vec::new();
        
        for node in &canonical.nodes {
            match self.convert_canonical_node(node, &mut stats, &mut warnings) {
                Ok(item) => items.push(item),
                Err(e) => {
                    warnings.push(ConversionWarning {
                        message: format!("Failed to convert node: {}", e),
                        location: None, // TODO: Extract location from node
                    });
                }
            }
        }
        
        // Create program metadata
        let metadata = if self.config.include_metadata {
            ProgramMetadata {
                source_id: self.config.source_id,
                module_path: canonical.metadata.get("module_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                language_version: canonical.metadata.get("language_version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("prism-1.0")
                    .to_string(),
            }
        } else {
            ProgramMetadata {
                source_id: self.config.source_id,
                module_path: "unknown".to_string(),
                language_version: "prism-1.0".to_string(),
            }
        };
        
        // Create the AST program
        let program = Program {
            items,
            metadata,
        };
        
        stats.conversion_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(AstConversionResult {
            program,
            stats,
            warnings,
        })
    }
    
    /// Convert a canonical node to an AST item
    fn convert_canonical_node(
        &mut self, 
        node: &CanonicalNode, 
        stats: &mut ConversionStats,
        warnings: &mut Vec<ConversionWarning>
    ) -> Result<Item, AstIntegrationError> {
        match &node.structure {
            crate::normalization::CanonicalStructure::Function { 
                name, 
                parameters, 
                return_type, 
                body 
            } => {
                stats.items_converted += 1;
                
                // Convert parameters
                let ast_parameters = parameters.iter()
                    .map(|param| self.convert_parameter(param))
                    .collect::<Result<Vec<_>, _>>()?;
                
                // Convert return type
                let ast_return_type = return_type.as_ref()
                    .map(|rt| self.convert_canonical_type(rt))
                    .transpose()?;
                
                // Convert body statements
                let ast_body = body.iter()
                    .map(|stmt| self.convert_canonical_statement(stmt, stats))
                    .collect::<Result<Vec<_>, _>>()?;
                
                Ok(Item::Function {
                    id: self.next_node_id(),
                    span: self.create_span(),
                    name: name.clone(),
                    parameters: ast_parameters,
                    return_type: ast_return_type,
                    body: ast_body,
                })
            }
            
            crate::normalization::CanonicalStructure::Type { name, definition } => {
                stats.items_converted += 1;
                
                let ast_type = self.convert_canonical_type(definition)?;
                
                Ok(Item::TypeAlias {
                    id: self.next_node_id(),
                    span: self.create_span(),
                    name: name.clone(),
                    type_def: ast_type,
                })
            }
            
            crate::normalization::CanonicalStructure::Constant { name, value, type_annotation } => {
                stats.items_converted += 1;
                
                let ast_value = self.convert_canonical_expression(value, stats)?;
                let ast_type = type_annotation.as_ref()
                    .map(|ta| self.convert_canonical_type(ta))
                    .transpose()?;
                
                Ok(Item::Constant {
                    id: self.next_node_id(),
                    span: self.create_span(),
                    name: name.clone(),
                    value: ast_value,
                    type_annotation: ast_type,
                })
            }
            
            _ => {
                warnings.push(ConversionWarning {
                    message: format!("Unsupported canonical structure: {:?}", node.structure),
                    location: None,
                });
                
                // Create a placeholder item for unsupported structures
                Ok(Item::Constant {
                    id: self.next_node_id(),
                    span: self.create_span(),
                    name: "unsupported_item".to_string(),
                    value: Expr::Literal {
                        id: self.next_node_id(),
                        span: self.create_span(),
                        value: prism_ast::Literal::String("unsupported".to_string()),
                    },
                    type_annotation: None,
                })
            }
        }
    }
    
    /// Convert a canonical parameter
    fn convert_parameter(&mut self, param: &str) -> Result<prism_ast::Parameter, AstIntegrationError> {
        // Simple parameter parsing - in a real implementation, this would be more sophisticated
        let parts: Vec<&str> = param.split(':').collect();
        let name = parts[0].trim().to_string();
        let type_annotation = if parts.len() > 1 {
            Some(Type::Named {
                id: self.next_node_id(),
                span: self.create_span(),
                name: parts[1].trim().to_string(),
            })
        } else {
            None
        };
        
        Ok(prism_ast::Parameter {
            id: self.next_node_id(),
            span: self.create_span(),
            name,
            type_annotation,
        })
    }
    
    /// Convert a canonical type to AST type
    fn convert_canonical_type(&mut self, canonical_type: &CanonicalType) -> Result<Type, AstIntegrationError> {
        
        Ok(match canonical_type {
            CanonicalType::Primitive(name) => Type::Named {
                id: self.next_node_id(),
                span: self.create_span(),
                name: name.clone(),
            },
            CanonicalType::Generic { base, parameters } => Type::Generic {
                id: self.next_node_id(),
                span: self.create_span(),
                base: Box::new(Type::Named {
                    id: self.next_node_id(),
                    span: self.create_span(),
                    name: base.clone(),
                }),
                parameters: parameters.iter()
                    .map(|p| self.convert_canonical_type(p))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            CanonicalType::Function { parameters, return_type } => Type::Function {
                id: self.next_node_id(),
                span: self.create_span(),
                parameters: parameters.iter()
                    .map(|p| self.convert_canonical_type(p))
                    .collect::<Result<Vec<_>, _>>()?,
                return_type: Box::new(self.convert_canonical_type(return_type)?),
            },
        })
    }
    
    /// Convert a canonical statement to AST statement
    fn convert_canonical_statement(&mut self, stmt: &CanonicalStatement, stats: &mut ConversionStats) -> Result<Stmt, AstIntegrationError> {
        stats.statements_converted += 1;
        
        Ok(match stmt {
            CanonicalStatement::Expression(expr) => Stmt::Expression {
                id: self.next_node_id(),
                span: self.create_span(),
                expression: self.convert_canonical_expression(expr, stats)?,
            },
            CanonicalStatement::Assignment { target, value } => Stmt::Assignment {
                id: self.next_node_id(),
                span: self.create_span(),
                target: target.clone(),
                value: self.convert_canonical_expression(value, stats)?,
            },
            CanonicalStatement::Return(expr) => Stmt::Return {
                id: self.next_node_id(),
                span: self.create_span(),
                value: expr.as_ref()
                    .map(|e| self.convert_canonical_expression(e, stats))
                    .transpose()?,
            },
        })
    }
    
    /// Convert a canonical expression to AST expression
    fn convert_canonical_expression(&mut self, expr: &CanonicalExpression, stats: &mut ConversionStats) -> Result<Expr, AstIntegrationError> {
        stats.expressions_converted += 1;
        
        Ok(match expr {
            CanonicalExpression::Literal(value) => Expr::Literal {
                id: self.next_node_id(),
                span: self.create_span(),
                value: prism_ast::Literal::String(value.clone()), // Simplified - real implementation would parse different literal types
            },
            CanonicalExpression::Identifier(name) => Expr::Identifier {
                id: self.next_node_id(),
                span: self.create_span(),
                name: name.clone(),
            },
            CanonicalExpression::FunctionCall { function, arguments } => Expr::Call {
                id: self.next_node_id(),
                span: self.create_span(),
                function: Box::new(self.convert_canonical_expression(function, stats)?),
                arguments: arguments.iter()
                    .map(|arg| self.convert_canonical_expression(arg, stats))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            CanonicalExpression::BinaryOperation { left, operator, right } => Expr::Binary {
                id: self.next_node_id(),
                span: self.create_span(),
                left: Box::new(self.convert_canonical_expression(left, stats)?),
                operator: self.convert_binary_operator(operator)?,
                right: Box::new(self.convert_canonical_expression(right, stats)?),
            },
        })
    }
    
    /// Convert a binary operator
    fn convert_binary_operator(&self, op: &str) -> Result<prism_ast::BinaryOperator, AstIntegrationError> {
        Ok(match op {
            "+" => prism_ast::BinaryOperator::Add,
            "-" => prism_ast::BinaryOperator::Subtract,
            "*" => prism_ast::BinaryOperator::Multiply,
            "/" => prism_ast::BinaryOperator::Divide,
            "==" => prism_ast::BinaryOperator::Equal,
            "!=" => prism_ast::BinaryOperator::NotEqual,
            "<" => prism_ast::BinaryOperator::LessThan,
            ">" => prism_ast::BinaryOperator::GreaterThan,
            "<=" => prism_ast::BinaryOperator::LessThanOrEqual,
            ">=" => prism_ast::BinaryOperator::GreaterThanOrEqual,
            "&&" => prism_ast::BinaryOperator::LogicalAnd,
            "||" => prism_ast::BinaryOperator::LogicalOr,
            _ => return Err(AstIntegrationError::ConversionFailed {
                reason: format!("Unsupported binary operator: {}", op),
            }),
        })
    }
    
    /// Generate next node ID
    fn next_node_id(&mut self) -> NodeId {
        if self.config.generate_node_ids {
            let id = NodeId::new(self.node_id_counter);
            self.node_id_counter += 1;
            id
        } else {
            NodeId::new(0) // Use dummy ID if not generating
        }
    }
    
    /// Create a span (placeholder implementation)
    fn create_span(&self) -> Span {
        if self.config.preserve_locations {
            Span::dummy() // TODO: Use actual span information from canonical form
        } else {
            Span::dummy()
        }
    }
}

impl Default for AstBridge {
    fn default() -> Self {
        Self::new()
    }
} 