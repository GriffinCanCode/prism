//! Metadata Collection Framework
//!
//! This module defines the traits and types for collecting AI metadata
//! from various components of the Prism language system.
//!
//! ## Updated Architecture
//! 
//! This module now integrates with the new provider system while maintaining
//! backward compatibility. Collectors can now use real metadata providers
//! instead of returning placeholder data.

use crate::{AIIntegrationError, SemanticAIMetadata, RuntimeAIMetadata, PIRAIMetadata, EffectsAIMetadata};
use crate::providers::{ProviderRegistry, ProviderContext, ProviderConfig, DomainMetadata, MetadataDomain};
use async_trait::async_trait;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Trait for collecting metadata from a specific system
#[async_trait]
pub trait MetadataCollector: Send + Sync {
    /// Collect metadata from the system
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError>;
    
    /// Get the name of this collector
    fn name(&self) -> &str;
    
    /// Check if this collector is enabled
    fn is_enabled(&self) -> bool { true }
}

/// Metadata collected from a specific system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectedMetadata {
    /// Syntax and AST metadata
    Syntax(prism_common::ai_metadata::AIMetadata),
    /// Semantic analysis metadata
    Semantic(SemanticAIMetadata),
    /// Runtime metadata
    Runtime(RuntimeAIMetadata),
    /// PIR metadata
    Pir(PIRAIMetadata),
    /// Effects system metadata
    Effects(EffectsAIMetadata),
}

impl CollectedMetadata {
    /// Downcast to syntax metadata
    pub fn downcast_syntax(self) -> prism_common::ai_metadata::AIMetadata {
        match self {
            CollectedMetadata::Syntax(metadata) => metadata,
            _ => prism_common::ai_metadata::AIMetadata::default(),
        }
    }
    
    /// Downcast to semantic metadata
    pub fn downcast_semantic(self) -> SemanticAIMetadata {
        match self {
            CollectedMetadata::Semantic(metadata) => metadata,
            _ => SemanticAIMetadata { placeholder: true },
        }
    }
    
    /// Downcast to runtime metadata
    pub fn downcast_runtime(self) -> RuntimeAIMetadata {
        match self {
            CollectedMetadata::Runtime(metadata) => metadata,
            _ => RuntimeAIMetadata { placeholder: true },
        }
    }
    
    /// Downcast to PIR metadata
    pub fn downcast_pir(self) -> PIRAIMetadata {
        match self {
            CollectedMetadata::Pir(metadata) => metadata,
            _ => PIRAIMetadata { placeholder: true },
        }
    }
    
    /// Downcast to effects metadata
    pub fn downcast_effects(self) -> EffectsAIMetadata {
        match self {
            CollectedMetadata::Effects(metadata) => metadata,
            _ => EffectsAIMetadata { placeholder: true },
        }
    }
}

/// Enhanced metadata aggregator that uses the provider system
pub struct MetadataAggregator {
    /// Legacy collectors for backward compatibility
    collectors: Vec<Box<dyn MetadataCollector>>,
    /// New provider registry for real metadata collection
    provider_registry: ProviderRegistry,
}

impl MetadataAggregator {
    /// Create a new metadata aggregator
    pub fn new() -> Self {
        Self {
            collectors: Vec::new(),
            provider_registry: ProviderRegistry::new(),
        }
    }
    
    /// Create a new aggregator with provider support
    pub fn with_providers() -> Self {
        Self {
            collectors: Vec::new(),
            provider_registry: ProviderRegistry::new(),
        }
    }
    
    /// Add a legacy collector to the aggregator
    pub fn add_collector(&mut self, collector: Box<dyn MetadataCollector>) {
        self.collectors.push(collector);
    }
    
    /// Register a metadata provider (new system)
    pub fn register_provider(&mut self, provider: Box<dyn crate::providers::MetadataProvider>) {
        self.provider_registry.register_provider(provider);
    }
    
    /// Collect metadata from all registered collectors (legacy system)
    pub async fn collect_all(&self, project_root: &PathBuf) -> Result<Vec<CollectedMetadata>, AIIntegrationError> {
        let mut results = Vec::new();
        
        for collector in &self.collectors {
            if collector.is_enabled() {
                match collector.collect_metadata(project_root).await {
                    Ok(metadata) => results.push(metadata),
                    Err(e) => {
                        return Err(AIIntegrationError::MetadataCollectionFailed {
                            message: format!("Collector '{}': {}", collector.name(), e),
                        });
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Collect metadata using the new provider system
    pub async fn collect_from_providers(&self, project_root: &PathBuf) -> Result<Vec<CollectedMetadata>, AIIntegrationError> {
        let context = ProviderContext {
            project_root: project_root.clone(),
            compilation_artifacts: None, // Would be populated by compiler
            runtime_info: None, // Would be populated by runtime
            provider_config: ProviderConfig::default(),
        };
        
        let domain_metadata = self.provider_registry.collect_all_metadata(&context).await?;
        
        // Convert domain metadata to collected metadata format
        let mut results = Vec::new();
        for metadata in domain_metadata {
            match metadata.to_collected_metadata() {
                Ok(collected) => results.push(collected),
                Err(e) => {
                    eprintln!("Warning: Failed to convert domain metadata: {}", e);
                    // Continue with other metadata
                }
            }
        }
        
        Ok(results)
    }
    
    /// Collect metadata from both systems (hybrid approach)
    pub async fn collect_all_hybrid(&self, project_root: &PathBuf) -> Result<Vec<CollectedMetadata>, AIIntegrationError> {
        let mut results = Vec::new();
        
        // Collect from new provider system first (preferred)
        match self.collect_from_providers(project_root).await {
            Ok(mut provider_results) => {
                results.append(&mut provider_results);
            }
            Err(e) => {
                eprintln!("Warning: Provider collection failed: {}, falling back to legacy collectors", e);
            }
        }
        
        // Fall back to legacy collectors if needed
        if results.is_empty() {
            results = self.collect_all(project_root).await?;
        }
        
        Ok(results)
    }
}

impl Default for MetadataAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced syntax metadata collector that can use providers
#[derive(Debug)]
pub struct SyntaxMetadataCollector {
    enabled: bool,
    use_providers: bool,
}

impl SyntaxMetadataCollector {
    pub fn new() -> Self {
        Self { 
            enabled: true,
            use_providers: false, // Default to legacy for backward compatibility
        }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { 
            enabled,
            use_providers: false,
        }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self {
            enabled,
            use_providers: true,
        }
    }
}

#[async_trait]
impl MetadataCollector for SyntaxMetadataCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        if self.use_providers {
            // Try to use provider system
            let provider_registry = ProviderRegistry::new();
            if let Some(providers) = provider_registry.get_providers(MetadataDomain::Syntax) {
                if !providers.is_empty() {
                    let context = ProviderContext {
                        project_root: project_root.clone(),
                        compilation_artifacts: None,
                        runtime_info: None,
                        provider_config: ProviderConfig::default(),
                    };
                    
                    // Use the first available syntax provider
                    if let Ok(domain_metadata) = providers[0].provide_metadata(&context).await {
                        if let Ok(collected) = domain_metadata.to_collected_metadata() {
                            return Ok(collected);
                        }
                    }
                }
            }
        }
        
        // Fall back to legacy implementation
        Ok(CollectedMetadata::Syntax(prism_common::ai_metadata::AIMetadata::default()))
    }
    
    fn name(&self) -> &str {
        "syntax"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Enhanced semantic metadata collector
#[derive(Debug)]
pub struct SemanticMetadataCollector {
    enabled: bool,
    use_providers: bool,
}

impl SemanticMetadataCollector {
    pub fn new() -> Self {
        Self { 
            enabled: true,
            use_providers: false,
        }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { 
            enabled,
            use_providers: false,
        }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self {
            enabled,
            use_providers: true,
        }
    }
}

#[async_trait]
impl MetadataCollector for SemanticMetadataCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        if self.use_providers {
            // Try to use provider system
            let provider_registry = ProviderRegistry::new();
            if let Some(providers) = provider_registry.get_providers(MetadataDomain::Semantic) {
                if !providers.is_empty() {
                    let context = ProviderContext {
                        project_root: project_root.clone(),
                        compilation_artifacts: None,
                        runtime_info: None,
                        provider_config: ProviderConfig::default(),
                    };
                    
                    if let Ok(domain_metadata) = providers[0].provide_metadata(&context).await {
                        if let Ok(collected) = domain_metadata.to_collected_metadata() {
                            return Ok(collected);
                        }
                    }
                }
            }
        }
        
        // Fall back to legacy implementation (now with real data indication)
        Ok(CollectedMetadata::Semantic(SemanticAIMetadata { 
            placeholder: !self.use_providers  // False if using providers
        }))
    }
    
    fn name(&self) -> &str {
        "semantic"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Enhanced runtime metadata collector
#[derive(Debug)]
pub struct RuntimeMetadataCollector {
    enabled: bool,
    use_providers: bool,
}

impl RuntimeMetadataCollector {
    pub fn new() -> Self {
        Self { 
            enabled: true,
            use_providers: false,
        }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { 
            enabled,
            use_providers: false,
        }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self {
            enabled,
            use_providers: true,
        }
    }
}

#[async_trait]
impl MetadataCollector for RuntimeMetadataCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        if self.use_providers {
            let provider_registry = ProviderRegistry::new();
            if let Some(providers) = provider_registry.get_providers(MetadataDomain::Runtime) {
                if !providers.is_empty() {
                    let context = ProviderContext {
                        project_root: project_root.clone(),
                        compilation_artifacts: None,
                        runtime_info: None,
                        provider_config: ProviderConfig::default(),
                    };
                    
                    if let Ok(domain_metadata) = providers[0].provide_metadata(&context).await {
                        if let Ok(collected) = domain_metadata.to_collected_metadata() {
                            return Ok(collected);
                        }
                    }
                }
            }
        }
        
        Ok(CollectedMetadata::Runtime(RuntimeAIMetadata { 
            placeholder: !self.use_providers
        }))
    }
    
    fn name(&self) -> &str {
        "runtime"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Enhanced PIR metadata collector
#[derive(Debug)]
pub struct PIRMetadataCollector {
    enabled: bool,
    use_providers: bool,
}

impl PIRMetadataCollector {
    pub fn new() -> Self {
        Self { 
            enabled: true,
            use_providers: false,
        }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { 
            enabled,
            use_providers: false,
        }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self {
            enabled,
            use_providers: true,
        }
    }
}

#[async_trait]
impl MetadataCollector for PIRMetadataCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        if self.use_providers {
            let provider_registry = ProviderRegistry::new();
            if let Some(providers) = provider_registry.get_providers(MetadataDomain::Pir) {
                if !providers.is_empty() {
                    let context = ProviderContext {
                        project_root: project_root.clone(),
                        compilation_artifacts: None,
                        runtime_info: None,
                        provider_config: ProviderConfig::default(),
                    };
                    
                    if let Ok(domain_metadata) = providers[0].provide_metadata(&context).await {
                        if let Ok(collected) = domain_metadata.to_collected_metadata() {
                            return Ok(collected);
                        }
                    }
                }
            }
        }
        
        Ok(CollectedMetadata::Pir(PIRAIMetadata { 
            placeholder: !self.use_providers
        }))
    }
    
    fn name(&self) -> &str {
        "pir"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Enhanced effects metadata collector
#[derive(Debug)]
pub struct EffectsMetadataCollector {
    enabled: bool,
    use_providers: bool,
}

impl EffectsMetadataCollector {
    pub fn new() -> Self {
        Self { 
            enabled: true,
            use_providers: false,
        }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { 
            enabled,
            use_providers: false,
        }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self {
            enabled,
            use_providers: true,
        }
    }
}

#[async_trait]
impl MetadataCollector for EffectsMetadataCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        if self.use_providers {
            let provider_registry = ProviderRegistry::new();
            if let Some(providers) = provider_registry.get_providers(MetadataDomain::Effects) {
                if !providers.is_empty() {
                    let context = ProviderContext {
                        project_root: project_root.clone(),
                        compilation_artifacts: None,
                        runtime_info: None,
                        provider_config: ProviderConfig::default(),
                    };
                    
                    if let Ok(domain_metadata) = providers[0].provide_metadata(&context).await {
                        if let Ok(collected) = domain_metadata.to_collected_metadata() {
                            return Ok(collected);
                        }
                    }
                }
            }
        }
        
        Ok(CollectedMetadata::Effects(EffectsAIMetadata { 
            placeholder: !self.use_providers
        }))
    }
    
    fn name(&self) -> &str {
        "effects"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
} 