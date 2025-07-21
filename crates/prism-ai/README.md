# Prism AI Integration

A comprehensive AI metadata collection and export system for the Prism programming language that enables external AI tools to understand and work with Prism code through structured metadata.

## Overview

The `prism-ai` crate provides the central coordination point for AI metadata export and integration functionality across the Prism language ecosystem. It orchestrates metadata collection from multiple crates and provides unified export interfaces for external AI tools.

## Design Principles

1. **Separation of Concerns**: AI functionality is separated from core language processing
2. **Modular Integration**: Each crate can contribute metadata independently
3. **Unified Export**: Single interface for all AI metadata export needs
4. **External Focus**: Designed for external AI tool consumption, not internal AI execution
5. **Performance Aware**: Minimal overhead when AI features are disabled

## Architecture

### New Metadata Provider System

The new architecture uses a **Metadata Provider System** that follows strict Separation of Concerns:

```rust
// Each crate implements a MetadataProvider
#[async_trait]
pub trait MetadataProvider: Send + Sync {
    fn domain(&self) -> MetadataDomain;
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError>;
    fn provider_info(&self) -> ProviderInfo;
}
```

### Key Components

- **`providers/`** - New metadata provider system with standardized interfaces
- **`metadata/`** - Enhanced metadata collection framework with provider integration
- **`export/`** - Multi-format export system (JSON, YAML, XML, OpenAPI, GraphQL)
- **`integration/`** - External AI tool integration utilities
- **`context/`** - Context extraction for business intelligence

## Usage

### Basic Usage

```rust
use prism_ai::{AIIntegrationCoordinator, AIIntegrationConfig, ExportFormat};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create configuration
    let config = AIIntegrationConfig {
        enabled: true,
        export_formats: vec![ExportFormat::Json],
        include_business_context: true,
        include_performance_metrics: true,
        include_architectural_patterns: true,
        min_confidence_threshold: 0.7,
        output_directory: Some(PathBuf::from("./ai_metadata")),
    };
    
    // 2. Create coordinator
    let mut coordinator = AIIntegrationCoordinator::new(config);
    
    // 3. Register providers from different crates
    coordinator.register_provider(Box::new(MyCustomProvider::new()));
    
    // 4. Collect metadata
    let project_root = PathBuf::from(".");
    let metadata = coordinator.collect_metadata(&project_root).await?;
    
    // 5. Export in multiple formats
    let exports = coordinator.export_metadata(&metadata, &[ExportFormat::Json]).await?;
    
    println!("Collected metadata from {} sources", metadata.relationships.len());
    Ok(())
}
```

### Implementing a Metadata Provider

Each Prism crate should implement a metadata provider to expose its AI metadata:

```rust
use prism_ai::providers::*;
use async_trait::async_trait;

#[derive(Debug)]
pub struct MyDomainProvider {
    enabled: bool,
}

impl MyDomainProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for MyDomainProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Semantic  // or Pir, Effects, Runtime, etc.
    }
    
    fn name(&self) -> &str {
        "my-domain-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        // Extract metadata from your existing systems
        let metadata = extract_my_domain_metadata(context)?;
        Ok(DomainMetadata::Semantic(metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "My Domain Provider".to_string(),
            version: "1.0.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                ProviderCapability::RealTime,
                ProviderCapability::BusinessContext,
            ],
            dependencies: vec![],
        }
    }
}
```

### Integration with Individual Crates

#### PIR Crate Integration

```rust
// In prism-pir/src/ai_integration/mod.rs
use prism_ai::providers::*;

pub struct PIRMetadataProvider {
    pir_system: Arc<PrismIR>,
}

impl PIRMetadataProvider {
    pub fn new(pir_system: Arc<PrismIR>) -> Self {
        Self { pir_system }
    }
}

#[async_trait]
impl MetadataProvider for PIRMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Pir
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        let metadata = PIRProviderMetadata {
            structure_info: self.extract_structure_info(),
            business_context: self.extract_business_context(),
            optimization_info: self.extract_optimization_info(),
            consistency_data: self.extract_consistency_data(),
        };
        
        Ok(DomainMetadata::Pir(metadata))
    }
    
    // ... implementation details
}
```

## Metadata Domains

The system organizes metadata by domain for clear separation of concerns:

- **`Syntax`** - Parsing, syntax tree metrics, language detection
- **`Semantic`** - Type information, business rules, semantic relationships
- **`Pir`** - PIR structure, business context, optimization data
- **`Effects`** - Effect definitions, capabilities, security analysis
- **`Runtime`** - Execution statistics, performance profiles, resource usage
- **`Documentation`** - Coverage metrics, quality analysis, extracted context
- **`Compiler`** - Compilation statistics, query metrics, coordination info

## Export Formats

Comprehensive export support for different AI tool requirements:

- **JSON** - General AI consumption, web APIs
- **YAML** - Human-readable configuration, documentation
- **XML** - Enterprise integration, structured data exchange
- **OpenAPI** - API documentation, service integration
- **GraphQL** - Query-based access, flexible data retrieval
- **Binary** - Performance-critical scenarios (MessagePack/Protocol Buffers)

## Backward Compatibility

The system maintains full backward compatibility with existing collectors:

```rust
// Legacy collectors still work
let mut coordinator = AIIntegrationCoordinator::new(config);

// Old way (still supported)
coordinator.register_collector("syntax".to_string(), Box::new(SyntaxMetadataCollector::new()));

// New way (preferred)
coordinator.register_provider(Box::new(SyntaxProvider::new()));

// Hybrid collection automatically uses providers first, falls back to collectors
let metadata = coordinator.collect_metadata(&project_root).await?;
```

## Performance Considerations

- **Lazy Loading**: Metadata is collected only when requested
- **Caching**: Provider results can be cached for repeated access
- **Optional Providers**: Providers can be disabled for performance
- **Incremental Updates**: Supports incremental metadata collection
- **Minimal Overhead**: Zero cost when AI features are disabled

## External AI Tool Integration

Built-in integrations for popular AI development tools:

```rust
use prism_ai::integration::*;

// Pre-configured integrations
let vscode_integration = AIToolIntegration::vscode_ai();
let copilot_integration = AIToolIntegration::github_copilot();
let lsp_integration = AIToolIntegration::language_server();

let mut integration_manager = IntegrationManager::new();
integration_manager.register_integration(vscode_integration);
integration_manager.register_integration(copilot_integration);

// Export for all registered tools
let results = integration_manager.export_for_all_integrations(&metadata).await?;
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- **`complete_metadata_collection.rs`** - Full system demonstration
- **`custom_provider.rs`** - Implementing custom providers
- **`export_formats.rs`** - Multi-format export examples
- **`integration_tools.rs`** - External tool integration

## Testing

Run the comprehensive test suite:

```bash
# Unit tests
cargo test

# Integration tests
cargo test --features integration

# Example execution
cargo run --example complete_metadata_collection
```

## Contributing

When adding new metadata providers:

1. **Follow SoC**: Each provider handles only one domain
2. **Maintain Cohesion**: Focus on exposing existing metadata, not collecting new data
3. **Use Existing Structures**: Leverage crate's existing metadata types
4. **Document Capabilities**: Clearly specify provider capabilities
5. **Add Tests**: Include comprehensive test coverage

## Architecture Benefits

✅ **Separation of Concerns**: Each provider focuses on one domain  
✅ **Conceptual Cohesion**: Providers expose existing metadata only  
✅ **Modularity**: Plug-and-play architecture with optional providers  
✅ **No Duplication**: Leverages existing metadata structures  
✅ **Backward Compatibility**: Legacy collectors continue to work  
✅ **Performance Aware**: Minimal overhead, optional collection  
✅ **Extensible**: Easy to add new providers and export formats  
✅ **AI-First**: Designed specifically for external AI tool consumption  

## License

This project is licensed under the MIT OR Apache-2.0 license. 