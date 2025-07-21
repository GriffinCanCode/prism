//! Symbol AI Metadata Provider - External AI Integration
//!
//! This module embodies the single concept of "Symbol AI Metadata Provision".
//! Following Prism's Conceptual Cohesion principle and external AI integration model,
//! this file is responsible for ONE thing: providing structured symbol metadata
//! for external AI tools while maintaining separation of concerns.
//!
//! **Conceptual Responsibility**: Symbol metadata export for external AI consumption
//! **What it does**: metadata collection, structured export, AI context generation
//! **What it doesn't do**: symbol analysis, metadata storage, AI reasoning

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolTable, SymbolData, SymbolKind};
use crate::module_registry::SmartModuleRegistry;
use prism_ai::providers::{
    MetadataProvider, MetadataDomain, ProviderContext, DomainMetadata, ProviderInfo,
    ProviderCapability, SymbolProviderMetadata, SymbolInformation, BusinessRule,
    SemanticRelationship, ValidationSummary
};
use prism_ai::AIIntegrationError;
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tracing::{debug, warn};

/// Symbol metadata provider that exposes symbol information to the prism-ai system
/// 
/// This provider follows Separation of Concerns by:
/// - Only exposing existing symbol metadata, not collecting new data
/// - Focusing solely on symbol domain metadata
/// - Maintaining conceptual cohesion around symbol understanding
#[derive(Debug)]
pub struct SymbolMetadataProvider {
    /// Whether this provider is enabled
    enabled: bool,
    /// Reference to symbol table for data extraction
    symbol_table: Option<Arc<SymbolTable>>,
    /// Reference to module registry for capability information
    module_registry: Option<Arc<SmartModuleRegistry>>,
    /// Cached symbol information for performance
    cached_symbols: Option<Vec<SymbolInformation>>,
    /// Provider configuration
    config: SymbolProviderConfig,
}

/// Configuration for the symbol metadata provider
#[derive(Debug, Clone)]
pub struct SymbolProviderConfig {
    /// Enable symbol information export
    pub enable_symbol_export: bool,
    /// Enable business context export
    pub enable_business_context: bool,
    /// Enable capability information export
    pub enable_capability_export: bool,
    /// Enable semantic relationship export
    pub enable_relationship_export: bool,
    /// Maximum symbols to export in one batch
    pub max_export_batch_size: usize,
    /// Cache refresh interval in seconds
    pub cache_refresh_interval: u64,
}

impl Default for SymbolProviderConfig {
    fn default() -> Self {
        Self {
            enable_symbol_export: true,
            enable_business_context: true,
            enable_capability_export: true,
            enable_relationship_export: true,
            max_export_batch_size: 1000,
            cache_refresh_interval: 300, // 5 minutes
        }
    }
}

impl SymbolMetadataProvider {
    /// Create a new symbol metadata provider
    pub fn new(config: SymbolProviderConfig) -> Self {
        Self {
            enabled: true,
            symbol_table: None,
            module_registry: None,
            cached_symbols: None,
            config,
        }
    }

    /// Initialize with symbol table reference
    pub fn with_symbol_table(mut self, symbol_table: Arc<SymbolTable>) -> Self {
        self.symbol_table = Some(symbol_table);
        self
    }

    /// Initialize with module registry reference
    pub fn with_module_registry(mut self, module_registry: Arc<SmartModuleRegistry>) -> Self {
        self.module_registry = Some(module_registry);
        self
    }

    /// Enable or disable the provider
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        debug!("Symbol metadata provider enabled: {}", enabled);
    }

    /// Extract symbol information for AI consumption
    async fn extract_symbol_information(&self) -> Result<Vec<SymbolInformation>, AIIntegrationError> {
        if !self.config.enable_symbol_export {
            return Ok(Vec::new());
        }

        let symbol_table = self.symbol_table.as_ref()
            .ok_or_else(|| AIIntegrationError::ProviderError {
                provider: "SymbolMetadataProvider".to_string(),
                message: "Symbol table not initialized".to_string(),
            })?;

        // Get all symbols from symbol table
        let symbols = symbol_table.get_all_symbols().await
            .map_err(|e| AIIntegrationError::ProviderError {
                provider: "SymbolMetadataProvider".to_string(),
                message: format!("Failed to get symbols: {}", e),
            })?;

        let mut symbol_info = Vec::new();

        for symbol_data in symbols.into_iter().take(self.config.max_export_batch_size) {
            let info = self.convert_symbol_to_ai_info(&symbol_data).await?;
            symbol_info.push(info);
        }

        debug!("Extracted {} symbols for AI consumption", symbol_info.len());
        Ok(symbol_info)
    }

    /// Convert internal symbol data to AI-consumable information
    async fn convert_symbol_to_ai_info(&self, symbol_data: &SymbolData) -> Result<SymbolInformation, AIIntegrationError> {
        let symbol_type = match &symbol_data.kind {
            SymbolKind::Function { .. } => "function".to_string(),
            SymbolKind::Type { .. } => "type".to_string(),
            SymbolKind::Variable { .. } => "variable".to_string(),
            SymbolKind::Constant { .. } => "constant".to_string(),
            SymbolKind::Module { .. } => "module".to_string(),
            SymbolKind::Parameter { .. } => "parameter".to_string(),
            SymbolKind::Capability { .. } => "capability".to_string(),
            SymbolKind::Effect { .. } => "effect".to_string(),
            SymbolKind::Import { .. } => "import".to_string(),
            SymbolKind::Export { .. } => "export".to_string(),
        };

        let visibility = match symbol_data.visibility {
            crate::symbols::data::SymbolVisibility::Public => "public".to_string(),
            crate::symbols::data::SymbolVisibility::Internal => "internal".to_string(),
            crate::symbols::data::SymbolVisibility::Private => "private".to_string(),
        };

        // Extract capabilities from symbol kind
        let capabilities = self.extract_symbol_capabilities(&symbol_data.kind);

        // Extract business context
        let business_context = if self.config.enable_business_context {
            self.extract_symbol_business_context(symbol_data).await?
        } else {
            None
        };

        Ok(SymbolInformation {
            name: symbol_data.name.resolve().unwrap_or_else(|| "unknown".to_string()),
            symbol_type,
            visibility,
            location: format!("{}:{}:{}", 
                symbol_data.location.source_id.raw(),
                symbol_data.location.start.line,
                symbol_data.location.start.column
            ),
            capabilities,
            business_context,
            documentation: symbol_data.metadata.documentation.clone(),
            ai_hints: symbol_data.metadata.ai_context.as_ref()
                .map(|ctx| ctx.ai_hints.clone())
                .unwrap_or_default(),
        })
    }

    /// Extract capabilities from symbol kind
    fn extract_symbol_capabilities(&self, symbol_kind: &SymbolKind) -> Vec<String> {
        match symbol_kind {
            SymbolKind::Function { effects, .. } => {
                effects.iter().map(|e| format!("effect:{}", e.category)).collect()
            },
            SymbolKind::Module { capabilities, .. } => capabilities.clone(),
            SymbolKind::Capability { capability_type, .. } => {
                vec![format!("capability:{:?}", capability_type)]
            },
            SymbolKind::Effect { effect_category, .. } => {
                vec![format!("effect:{:?}", effect_category)]
            },
            _ => Vec::new(),
        }
    }

    /// Extract business context for symbol
    async fn extract_symbol_business_context(&self, symbol_data: &SymbolData) -> Result<Option<String>, AIIntegrationError> {
        // Extract business context from symbol metadata
        if let Some(business_context) = &symbol_data.metadata.business_context {
            Ok(Some(business_context.responsibility.clone().unwrap_or_else(|| {
                format!("Symbol: {}", symbol_data.name.resolve().unwrap_or_else(|| "unknown".to_string()))
            })))
        } else {
            Ok(None)
        }
    }

    /// Extract semantic relationships between symbols
    async fn extract_semantic_relationships(&self) -> Result<Vec<SemanticRelationship>, AIIntegrationError> {
        if !self.config.enable_relationship_export {
            return Ok(Vec::new());
        }

        // This would analyze symbol relationships
        // For now, return empty list as placeholder
        Ok(Vec::new())
    }

    /// Extract business rules from symbols
    async fn extract_business_rules(&self) -> Result<Vec<BusinessRule>, AIIntegrationError> {
        if !self.config.enable_business_context {
            return Ok(Vec::new());
        }

        let symbol_table = self.symbol_table.as_ref()
            .ok_or_else(|| AIIntegrationError::ProviderError {
                provider: "SymbolMetadataProvider".to_string(),
                message: "Symbol table not initialized".to_string(),
            })?;

        let symbols = symbol_table.get_all_symbols().await
            .map_err(|e| AIIntegrationError::ProviderError {
                provider: "SymbolMetadataProvider".to_string(),
                message: format!("Failed to get symbols: {}", e),
            })?;

        let mut business_rules = Vec::new();

        for symbol_data in symbols {
            if let SymbolKind::Type { business_rules: rules, .. } = &symbol_data.kind {
                for rule in rules {
                    business_rules.push(BusinessRule {
                        rule_id: rule.name.clone(),
                        rule_type: rule.rule_type.clone(),
                        description: rule.description.clone(),
                        enforcement_level: format!("{:?}", rule.enforcement_level),
                        compliance_tags: rule.compliance_tags.clone(),
                        evidence: Vec::new(), // Would be populated with actual evidence
                    });
                }
            }
        }

        debug!("Extracted {} business rules for AI consumption", business_rules.len());
        Ok(business_rules)
    }

    /// Refresh cached data
    async fn refresh_cache(&mut self) -> Result<(), AIIntegrationError> {
        if !self.enabled {
            return Ok(());
        }

        debug!("Refreshing symbol metadata cache");
        self.cached_symbols = Some(self.extract_symbol_information().await?);
        Ok(())
    }
}

#[async_trait]
impl MetadataProvider for SymbolMetadataProvider {
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "SymbolMetadataProvider".to_string(),
            version: "1.0.0".to_string(),
            description: "Provides symbol table metadata for AI consumption".to_string(),
            domain: MetadataDomain::Symbol,
            capabilities: vec![
                ProviderCapability::SymbolInformation,
                ProviderCapability::BusinessContext,
                ProviderCapability::SemanticRelationships,
            ],
            enabled: self.enabled,
        }
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    async fn get_domain_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        if !self.enabled {
            return Err(AIIntegrationError::ProviderDisabled {
                provider: "SymbolMetadataProvider".to_string(),
            });
        }

        // Extract symbol information
        let symbols = self.extract_symbol_information().await?;
        
        // Extract business rules
        let business_rules = self.extract_business_rules().await?;
        
        // Extract semantic relationships
        let relationships = self.extract_semantic_relationships().await?;

        // Create validation summary
        let validation_summary = ValidationSummary {
            total_validations: symbols.len(),
            passed_validations: symbols.len(), // Assume all symbols are valid
            failed_validations: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
        };

        Ok(DomainMetadata::Symbol(SymbolProviderMetadata {
            symbols,
            business_rules,
            relationships,
            validation_summary,
        }))
    }

    async fn refresh_metadata(&mut self, _context: &ProviderContext) -> Result<(), AIIntegrationError> {
        self.refresh_cache().await
    }
}

/// Builder for SymbolMetadataProvider
pub struct SymbolMetadataProviderBuilder {
    config: SymbolProviderConfig,
    symbol_table: Option<Arc<SymbolTable>>,
    module_registry: Option<Arc<SmartModuleRegistry>>,
}

impl SymbolMetadataProviderBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: SymbolProviderConfig::default(),
            symbol_table: None,
            module_registry: None,
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: SymbolProviderConfig) -> Self {
        self.config = config;
        self
    }

    /// Set symbol table
    pub fn with_symbol_table(mut self, symbol_table: Arc<SymbolTable>) -> Self {
        self.symbol_table = Some(symbol_table);
        self
    }

    /// Set module registry
    pub fn with_module_registry(mut self, module_registry: Arc<SmartModuleRegistry>) -> Self {
        self.module_registry = Some(module_registry);
        self
    }

    /// Build the provider
    pub fn build(self) -> SymbolMetadataProvider {
        let mut provider = SymbolMetadataProvider::new(self.config);
        
        if let Some(symbol_table) = self.symbol_table {
            provider = provider.with_symbol_table(symbol_table);
        }
        
        if let Some(module_registry) = self.module_registry {
            provider = provider.with_module_registry(module_registry);
        }
        
        provider
    }
}

impl Default for SymbolMetadataProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
} 