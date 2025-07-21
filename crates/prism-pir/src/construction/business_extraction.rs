//! Business Context Extraction - AST to PIR Business Intelligence
//!
//! This module implements business context extraction from AST nodes, analyzing
//! domain concepts, business rules, and organizational patterns.

use crate::{PIRResult, PIRError};
use crate::business::{BusinessContext, BusinessRule};
use prism_ast::{Program, AstNode, Item, ModuleDecl, FunctionDecl, TypeDecl};
use std::collections::{HashMap, HashSet};

/// Business context extractor for AST analysis
pub struct BusinessContextExtractor {
    /// Configuration for business extraction
    config: BusinessExtractionConfig,
}

/// Configuration for business context extraction
#[derive(Debug, Clone)]
pub struct BusinessExtractionConfig {
    /// Enable domain inference from naming patterns
    pub enable_domain_inference: bool,
    /// Enable business rule extraction from documentation
    pub enable_rule_extraction: bool,
    /// Enable capability analysis from function signatures
    pub enable_capability_analysis: bool,
    /// Minimum confidence threshold for domain classification
    pub min_domain_confidence: f64,
}

impl BusinessContextExtractor {
    /// Create a new business context extractor
    pub fn new(config: BusinessExtractionConfig) -> Self {
        Self { config }
    }

    /// Extract business context for a module
    pub fn extract_module_business_context(
        &mut self, 
        module_decl: &ModuleDecl
    ) -> PIRResult<BusinessContext> {
        let mut context = BusinessContext::new(module_decl.name.clone());

        // Extract domain from module name and content
        if self.config.enable_domain_inference {
            context.domain = self.extract_domain_from_module(module_decl)?;
        }

        // Extract business rules from module items
        if self.config.enable_rule_extraction {
            context.business_rules = self.extract_business_rules_from_module(module_decl)?;
        }

        // Extract capabilities from module functions
        if self.config.enable_capability_analysis {
            context.capabilities = self.extract_module_capabilities(module_decl)?;
        }

        Ok(context)
    }

    /// Extract domain from module analysis
    fn extract_domain_from_module(&self, module_decl: &ModuleDecl) -> PIRResult<String> {
        let module_name = module_decl.name.to_lowercase();
        
        // Simple domain inference based on naming patterns
        if module_name.contains("auth") || module_name.contains("user") {
            Ok("user_management".to_string())
        } else if module_name.contains("payment") || module_name.contains("billing") {
            Ok("financial".to_string())
        } else if module_name.contains("data") || module_name.contains("process") {
            Ok("data_processing".to_string())
        } else {
            Ok(format!("{}_domain", module_name))
        }
    }

    /// Extract business rules from module
    fn extract_business_rules_from_module(&self, module_decl: &ModuleDecl) -> PIRResult<Vec<BusinessRule>> {
        let mut rules = Vec::new();

        // Extract rules from function names and patterns
        for item in &module_decl.items {
            if let Item::Function(func_decl) = &item.kind {
                let function_rules = self.extract_rules_from_function(func_decl)?;
                rules.extend(function_rules);
            }
        }

        Ok(rules)
    }

    /// Extract business rules from function
    fn extract_rules_from_function(&self, func_decl: &FunctionDecl) -> PIRResult<Vec<BusinessRule>> {
        let mut rules = Vec::new();
        let func_name = func_decl.name.to_lowercase();
        
        // Create rules based on function name patterns
        if func_name.starts_with("validate") || func_name.contains("check") {
            rules.push(BusinessRule {
                id: format!("validation_rule_{}", func_decl.name),
                description: format!("Validation rule enforced by function {}", func_decl.name),
                rule_type: crate::business::BusinessRuleType::Validation,
                conditions: Vec::new(),
                actions: Vec::new(),
                priority: 1.0,
                domain: "validation".to_string(),
            });
        }

        Ok(rules)
    }

    /// Extract module capabilities
    fn extract_module_capabilities(&self, module_decl: &ModuleDecl) -> PIRResult<Vec<String>> {
        let mut capabilities = HashSet::new();

        // Analyze module functions for capability patterns
        for item in &module_decl.items {
            if let Item::Function(func_decl) = &item.kind {
                let function_capabilities = self.analyze_function_capabilities(func_decl)?;
                capabilities.extend(function_capabilities);
            }
        }

        // Add module-level capability
        let module_capability = self.infer_module_capability(module_decl)?;
        capabilities.insert(module_capability);

        Ok(capabilities.into_iter().collect())
    }

    /// Analyze function capabilities
    fn analyze_function_capabilities(&self, func_decl: &FunctionDecl) -> PIRResult<HashSet<String>> {
        let mut capabilities = HashSet::new();
        let func_name = func_decl.name.to_lowercase();

        // Pattern-based capability detection
        if func_name.starts_with("create") || func_name.starts_with("new") {
            capabilities.insert("creation".to_string());
        } else if func_name.starts_with("update") || func_name.starts_with("modify") {
            capabilities.insert("modification".to_string());
        } else if func_name.starts_with("delete") || func_name.starts_with("remove") {
            capabilities.insert("deletion".to_string());
        } else if func_name.starts_with("get") || func_name.starts_with("find") {
            capabilities.insert("retrieval".to_string());
        } else if func_name.starts_with("validate") || func_name.starts_with("check") {
            capabilities.insert("validation".to_string());
        } else {
            capabilities.insert("processing".to_string());
        }

        Ok(capabilities)
    }

    /// Infer module capability from overall structure
    fn infer_module_capability(&self, module_decl: &ModuleDecl) -> PIRResult<String> {
        let module_name = module_decl.name.to_lowercase();

        if module_name.contains("service") {
            Ok("service_provision".to_string())
        } else if module_name.contains("model") || module_name.contains("entity") {
            Ok("data_modeling".to_string())
        } else if module_name.contains("util") || module_name.contains("helper") {
            Ok("utility_functions".to_string())
        } else {
            Ok(format!("{}_capability", module_name))
        }
    }
}

impl Default for BusinessExtractionConfig {
    fn default() -> Self {
        Self {
            enable_domain_inference: true,
            enable_rule_extraction: true,
            enable_capability_analysis: true,
            min_domain_confidence: 0.3,
        }
    }
} 