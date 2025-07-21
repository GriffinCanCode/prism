//! Context Extraction
//!
//! This module provides utilities for extracting contextual information
//! from Prism programs that can be used by external AI tools.

use crate::AIIntegrationError;
use async_trait::async_trait;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Trait for extracting context from various sources
#[async_trait]
pub trait ContextExtractor: Send + Sync {
    /// Extract context from a project
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError>;
    
    /// Get the name of this context extractor
    fn name(&self) -> &str;
    
    /// Get the priority of this extractor (higher numbers run first)
    fn priority(&self) -> u32 { 0 }
}

/// Context extracted from a project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContext {
    /// Source of the context
    pub source: String,
    /// Context type
    pub context_type: ContextType,
    /// Extracted data
    pub data: ContextData,
    /// Confidence in the extraction (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of context that can be extracted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextType {
    /// Project structure and organization
    ProjectStructure,
    /// Code patterns and idioms
    CodePatterns,
    /// Business domain information
    BusinessDomain,
    /// Architectural patterns
    ArchitecturalPatterns,
    /// Dependencies and relationships
    Dependencies,
    /// Documentation and comments
    Documentation,
    /// Configuration and settings
    Configuration,
}

/// Context data extracted from various sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextData {
    /// Textual description
    Text(String),
    /// Structured data
    Structured(serde_json::Value),
    /// List of items
    List(Vec<String>),
    /// Key-value pairs
    KeyValue(std::collections::HashMap<String, String>),
}

/// Project structure context extractor
#[derive(Debug)]
pub struct ProjectStructureExtractor;

impl ProjectStructureExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for ProjectStructureExtractor {
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        // This would analyze the project structure and extract relevant information
        // For now, return basic information
        Ok(ExtractedContext {
            source: "project_structure".to_string(),
            context_type: ContextType::ProjectStructure,
            data: ContextData::Text(format!("Project root: {}", project_root.display())),
            confidence: 1.0,
        })
    }
    
    fn name(&self) -> &str {
        "project_structure"
    }
    
    fn priority(&self) -> u32 {
        100 // High priority - project structure is fundamental
    }
}

/// Code patterns context extractor
#[derive(Debug)]
pub struct CodePatternsExtractor;

impl CodePatternsExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for CodePatternsExtractor {
    async fn extract_context(&self, _project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        // This would analyze code files to identify common patterns
        // For now, return placeholder information
        Ok(ExtractedContext {
            source: "code_patterns".to_string(),
            context_type: ContextType::CodePatterns,
            data: ContextData::List(vec![
                "Module pattern".to_string(),
                "Function composition".to_string(),
                "Error handling".to_string(),
            ]),
            confidence: 0.8,
        })
    }
    
    fn name(&self) -> &str {
        "code_patterns"
    }
    
    fn priority(&self) -> u32 {
        50
    }
}

/// Business domain context extractor
#[derive(Debug)]
pub struct BusinessDomainExtractor;

impl BusinessDomainExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for BusinessDomainExtractor {
    async fn extract_context(&self, _project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        // This would analyze documentation, comments, and naming to identify business domain
        // For now, return placeholder information
        Ok(ExtractedContext {
            source: "business_domain".to_string(),
            context_type: ContextType::BusinessDomain,
            data: ContextData::KeyValue({
                let mut map = std::collections::HashMap::new();
                map.insert("domain".to_string(), "Software Development".to_string());
                map.insert("subdomain".to_string(), "Language Processing".to_string());
                map
            }),
            confidence: 0.6,
        })
    }
    
    fn name(&self) -> &str {
        "business_domain"
    }
    
    fn priority(&self) -> u32 {
        30
    }
}

/// Dependencies context extractor
#[derive(Debug)]
pub struct DependenciesExtractor;

impl DependenciesExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for DependenciesExtractor {
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        // This would analyze Cargo.toml, package.json, etc. to extract dependencies
        // For now, return basic information
        let cargo_toml = project_root.join("Cargo.toml");
        let has_cargo = cargo_toml.exists();
        
        Ok(ExtractedContext {
            source: "dependencies".to_string(),
            context_type: ContextType::Dependencies,
            data: ContextData::KeyValue({
                let mut map = std::collections::HashMap::new();
                map.insert("has_cargo_toml".to_string(), has_cargo.to_string());
                if has_cargo {
                    map.insert("build_system".to_string(), "Cargo".to_string());
                    map.insert("language".to_string(), "Rust".to_string());
                }
                map
            }),
            confidence: if has_cargo { 0.9 } else { 0.5 },
        })
    }
    
    fn name(&self) -> &str {
        "dependencies"
    }
    
    fn priority(&self) -> u32 {
        80
    }
}

/// Context aggregator that combines multiple extractors
pub struct ContextAggregator {
    extractors: Vec<Box<dyn ContextExtractor>>,
}

impl ContextAggregator {
    /// Create a new context aggregator
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
        }
    }
    
    /// Add a context extractor
    pub fn add_extractor(&mut self, extractor: Box<dyn ContextExtractor>) {
        self.extractors.push(extractor);
        // Sort by priority (highest first)
        self.extractors.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }
    
    /// Extract context using all registered extractors
    pub async fn extract_all_context(&self, project_root: &PathBuf) -> Result<Vec<ExtractedContext>, AIIntegrationError> {
        let mut contexts = Vec::new();
        
        for extractor in &self.extractors {
            match extractor.extract_context(project_root).await {
                Ok(context) => contexts.push(context),
                Err(e) => {
                    // Log error but continue with other extractors
                    eprintln!("Warning: Context extractor '{}' failed: {}", extractor.name(), e);
                }
            }
        }
        
        Ok(contexts)
    }
    
    /// Create a default context aggregator with common extractors
    pub fn with_default_extractors() -> Self {
        let mut aggregator = Self::new();
        aggregator.add_extractor(Box::new(ProjectStructureExtractor::new()));
        aggregator.add_extractor(Box::new(CodePatternsExtractor::new()));
        aggregator.add_extractor(Box::new(BusinessDomainExtractor::new()));
        aggregator.add_extractor(Box::new(DependenciesExtractor::new()));
        aggregator
    }
}

impl Default for ContextAggregator {
    fn default() -> Self {
        Self::with_default_extractors()
    }
}

impl Default for ProjectStructureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CodePatternsExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BusinessDomainExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DependenciesExtractor {
    fn default() -> Self {
        Self::new()
    }
} 