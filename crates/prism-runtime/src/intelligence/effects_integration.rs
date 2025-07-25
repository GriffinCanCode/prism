//! Effects System Integration for Intelligence Metadata Collection
//!
//! This module provides integration between the prism-effects system and the
//! intelligence metadata collection system. It follows the principle of
//! separation of concerns by only handling the metadata extraction from
//! effect executions, not the effect execution itself.
//!
//! ## Design Principles
//!
//! 1. **Metadata Only**: Only extracts and structures metadata, doesn't execute effects
//! 2. **Existing Integration**: Uses existing effect system infrastructure
//! 3. **AI-Comprehensible**: Generates structured data for AI analysis
//! 4. **Performance Optimized**: Minimal overhead during effect execution
//! 5. **Business Context**: Preserves business meaning of effects

use crate::intelligence::metadata::{
    AIMetadataCollector, EffectExecutionMetadata, BusinessContext, 
    EffectPerformanceCharacteristics, EffectSemantics, AIMetadataError
};
use crate::platform::execution::ExecutionContext;
use crate::resources::effects::{Effect as RuntimeEffect, EffectResult, ResourceMeasurement};
use prism_effects::{Effect as PrismEffect, EffectResult as PrismEffectResult, EffectSystem};
use std::time::{Duration, SystemTime};
use std::sync::Arc;

/// Effects intelligence collector that bridges prism-effects with metadata collection
#[derive(Debug)]
pub struct EffectsIntelligenceCollector {
    /// AI metadata collector for storing effect metadata
    metadata_collector: Arc<AIMetadataCollector>,
    
    /// Effect system for accessing effect definitions
    effect_system: Arc<EffectSystem>,
}

impl EffectsIntelligenceCollector {
    /// Create a new effects intelligence collector
    pub fn new(
        metadata_collector: Arc<AIMetadataCollector>,
        effect_system: Arc<EffectSystem>,
    ) -> Self {
        Self {
            metadata_collector,
            effect_system,
        }
    }

    /// Record effect execution from prism-effects system
    pub fn record_prism_effect_execution(
        &self,
        effect: &PrismEffect,
        result: &PrismEffectResult,
        context: &ExecutionContext,
    ) -> Result<(), AIMetadataError> {
        // Convert prism-effects data to intelligence metadata format
        let effect_metadata = self.convert_prism_effect_to_metadata(effect, result, context)?;
        
        // Store in metadata collection system
        self.store_effect_metadata(effect_metadata)
    }

    /// Record effect execution from runtime effects system
    pub fn record_runtime_effect_execution(
        &self,
        effect: &RuntimeEffect,
        result: &EffectResult,
        context: &ExecutionContext,
    ) -> Result<(), AIMetadataError> {
        // Convert runtime effect data to intelligence metadata format
        let effect_metadata = self.convert_runtime_effect_to_metadata(effect, result, context)?;
        
        // Store in metadata collection system
        self.store_effect_metadata(effect_metadata)
    }

    /// Convert prism-effects Effect and EffectResult to intelligence metadata
    fn convert_prism_effect_to_metadata(
        &self,
        effect: &PrismEffect,
        result: &PrismEffectResult,
        context: &ExecutionContext,
    ) -> Result<EffectExecutionMetadata, AIMetadataError> {
        Ok(EffectExecutionMetadata {
            effect_name: self.extract_effect_name(effect),
            execution_id: context.execution_id,
            timestamp: SystemTime::now(),
            duration: self.extract_duration_from_prism_result(result),
            success: self.extract_success_from_prism_result(result),
            resource_consumption: self.convert_prism_resource_consumption(result),
            business_context: self.analyze_effect_business_context(effect, context),
            performance_characteristics: self.analyze_prism_effect_performance(result),
            semantic_meaning: self.extract_prism_effect_semantics(effect, context),
        })
    }

    /// Convert runtime Effect and EffectResult to intelligence metadata
    fn convert_runtime_effect_to_metadata(
        &self,
        effect: &RuntimeEffect,
        result: &EffectResult,
        context: &ExecutionContext,
    ) -> Result<EffectExecutionMetadata, AIMetadataError> {
        Ok(EffectExecutionMetadata {
            effect_name: effect.name().to_string(),
            execution_id: context.execution_id,
            timestamp: SystemTime::now(),
            duration: result.duration,
            success: result.success,
            resource_consumption: result.resources_consumed.clone(),
            business_context: self.analyze_runtime_effect_business_context(effect, context),
            performance_characteristics: self.analyze_runtime_effect_performance(result),
            semantic_meaning: self.extract_runtime_effect_semantics(effect, context),
        })
    }

    /// Store effect metadata in the metadata collection system
    fn store_effect_metadata(&self, metadata: EffectExecutionMetadata) -> Result<(), AIMetadataError> {
        // Use the existing metadata collector to store the effect metadata
        // This delegates to the existing infrastructure
        let runtime_effect = self.convert_metadata_to_runtime_effect(&metadata);
        let runtime_result = self.convert_metadata_to_runtime_result(&metadata);
        
        // Get execution context from metadata
        let context = self.create_execution_context_from_metadata(&metadata)?;
        
        self.metadata_collector.record_effect_execution(&runtime_effect, &runtime_result, &context)
    }

    /// Extract effect name from prism-effects Effect
    fn extract_effect_name(&self, effect: &PrismEffect) -> String {
        // Access the effect definition name from the prism-effects system
        effect.definition.clone()
    }

    /// Extract duration from prism-effects EffectResult
    fn extract_duration_from_prism_result(&self, result: &PrismEffectResult) -> Duration {
        // Prism effects results might have timing information
        // For now, return a default duration - this would be enhanced
        // with actual timing data from the prism-effects system
        Duration::from_millis(100) // Placeholder
    }

    /// Extract success status from prism-effects EffectResult
    fn extract_success_from_prism_result(&self, result: &PrismEffectResult) -> bool {
        result.success
    }

    /// Convert prism-effects resource consumption to runtime format
    fn convert_prism_resource_consumption(&self, result: &PrismEffectResult) -> ResourceMeasurement {
        // Convert prism-effects resource data to runtime ResourceMeasurement
        // This would extract actual resource consumption from prism-effects
        ResourceMeasurement::default() // Placeholder
    }

    /// Analyze business context for prism-effects Effect
    fn analyze_effect_business_context(
        &self,
        effect: &PrismEffect,
        context: &ExecutionContext,
    ) -> BusinessContext {
        BusinessContext {
            domain: self.infer_domain_from_effect(effect),
            subdomain: self.infer_subdomain_from_effect(effect),
            business_capability: self.infer_capability_from_effect(effect),
            stakeholders: self.identify_stakeholders_from_effect(effect),
            business_rules: self.extract_business_rules_from_effect(effect),
        }
    }

    /// Analyze business context for runtime Effect
    fn analyze_runtime_effect_business_context(
        &self,
        effect: &RuntimeEffect,
        context: &ExecutionContext,
    ) -> BusinessContext {
        BusinessContext {
            domain: self.infer_domain_from_runtime_effect(effect),
            subdomain: self.infer_subdomain_from_runtime_effect(effect),
            business_capability: self.infer_capability_from_runtime_effect(effect),
            stakeholders: self.identify_stakeholders_from_runtime_effect(effect),
            business_rules: self.extract_business_rules_from_runtime_effect(effect),
        }
    }

    /// Analyze performance characteristics for prism-effects EffectResult
    fn analyze_prism_effect_performance(&self, result: &PrismEffectResult) -> EffectPerformanceCharacteristics {
        EffectPerformanceCharacteristics {
            average_duration: Duration::from_millis(100), // Would extract from result
            resource_efficiency: 0.8, // Would calculate from resource usage
            success_rate: if result.success { 1.0 } else { 0.0 },
            bottlenecks: vec![], // Would analyze from performance data
        }
    }

    /// Analyze performance characteristics for runtime EffectResult
    fn analyze_runtime_effect_performance(&self, result: &EffectResult) -> EffectPerformanceCharacteristics {
        EffectPerformanceCharacteristics {
            average_duration: result.duration,
            resource_efficiency: self.calculate_efficiency_from_resources(&result.resources_consumed),
            success_rate: if result.success { 1.0 } else { 0.0 },
            bottlenecks: self.identify_bottlenecks_from_resources(&result.resources_consumed),
        }
    }

    /// Extract semantic meaning from prism-effects Effect
    fn extract_prism_effect_semantics(
        &self,
        effect: &PrismEffect,
        context: &ExecutionContext,
    ) -> EffectSemantics {
        EffectSemantics {
            business_purpose: self.infer_business_purpose_from_effect(effect),
            domain_concepts: self.extract_domain_concepts_from_effect(effect),
            side_effects: self.identify_side_effects_from_effect(effect),
            invariants: self.extract_invariants_from_effect(effect),
        }
    }

    /// Extract semantic meaning from runtime Effect
    fn extract_runtime_effect_semantics(
        &self,
        effect: &RuntimeEffect,
        context: &ExecutionContext,
    ) -> EffectSemantics {
        EffectSemantics {
            business_purpose: self.infer_business_purpose_from_runtime_effect(effect),
            domain_concepts: self.extract_domain_concepts_from_runtime_effect(effect),
            side_effects: self.identify_side_effects_from_runtime_effect(effect),
            invariants: self.extract_invariants_from_runtime_effect(effect),
        }
    }

    // Helper methods for business context analysis
    fn infer_domain_from_effect(&self, effect: &PrismEffect) -> String {
        // Analyze effect definition to infer business domain
        match effect.definition.as_str() {
            name if name.contains("Database") => "Data Management".to_string(),
            name if name.contains("Network") => "Communication".to_string(),
            name if name.contains("File") => "Document Management".to_string(),
            name if name.contains("User") => "User Management".to_string(),
            _ => "General Processing".to_string(),
        }
    }

    fn infer_domain_from_runtime_effect(&self, effect: &RuntimeEffect) -> String {
        match effect {
            RuntimeEffect::IO { .. } => "Input/Output Operations".to_string(),
            RuntimeEffect::Memory { .. } => "Memory Management".to_string(),
            RuntimeEffect::Computation { .. } => "Computational Processing".to_string(),
            RuntimeEffect::SystemCall { .. } => "System Integration".to_string(),
            RuntimeEffect::Custom { name, .. } => format!("Custom: {}", name),
        }
    }

    fn infer_subdomain_from_effect(&self, effect: &PrismEffect) -> String {
        format!("{} Operations", effect.definition)
    }

    fn infer_subdomain_from_runtime_effect(&self, effect: &RuntimeEffect) -> String {
        format!("{} Operations", effect.name())
    }

    fn infer_capability_from_effect(&self, effect: &PrismEffect) -> String {
        format!("{} Processing", effect.definition)
    }

    fn infer_capability_from_runtime_effect(&self, effect: &RuntimeEffect) -> String {
        format!("{} Processing", effect.name())
    }

    fn identify_stakeholders_from_effect(&self, effect: &PrismEffect) -> Vec<String> {
        // Infer stakeholders based on effect type
        vec!["System Users".to_string(), "Administrators".to_string()]
    }

    fn identify_stakeholders_from_runtime_effect(&self, effect: &RuntimeEffect) -> Vec<String> {
        match effect {
            RuntimeEffect::IO { .. } => vec!["Data Consumers".to_string(), "System Operators".to_string()],
            RuntimeEffect::Memory { .. } => vec!["Performance Engineers".to_string(), "System Administrators".to_string()],
            RuntimeEffect::Computation { .. } => vec!["Business Users".to_string(), "Data Analysts".to_string()],
            RuntimeEffect::SystemCall { .. } => vec!["System Administrators".to_string(), "Security Teams".to_string()],
            RuntimeEffect::Custom { .. } => vec!["Domain Experts".to_string(), "Business Users".to_string()],
        }
    }

    fn extract_business_rules_from_effect(&self, effect: &PrismEffect) -> Vec<String> {
        // Extract business rules from effect definition
        vec![format!("{} must be authorized", effect.definition)]
    }

    fn extract_business_rules_from_runtime_effect(&self, effect: &RuntimeEffect) -> Vec<String> {
        match effect {
            RuntimeEffect::IO { .. } => vec!["I/O operations must be within quota limits".to_string()],
            RuntimeEffect::Memory { .. } => vec!["Memory allocations must not exceed system limits".to_string()],
            RuntimeEffect::Computation { .. } => vec!["Computations must complete within timeout".to_string()],
            RuntimeEffect::SystemCall { .. } => vec!["System calls require appropriate capabilities".to_string()],
            RuntimeEffect::Custom { .. } => vec!["Custom effects must follow domain-specific rules".to_string()],
        }
    }

    // Helper methods for semantic analysis
    fn infer_business_purpose_from_effect(&self, effect: &PrismEffect) -> String {
        format!("Execute {} for business operations", effect.definition)
    }

    fn infer_business_purpose_from_runtime_effect(&self, effect: &RuntimeEffect) -> String {
        format!("Execute {} for system operations", effect.name())
    }

    fn extract_domain_concepts_from_effect(&self, effect: &PrismEffect) -> Vec<String> {
        vec![effect.definition.clone(), "Business Process".to_string()]
    }

    fn extract_domain_concepts_from_runtime_effect(&self, effect: &RuntimeEffect) -> Vec<String> {
        vec![effect.name().to_string(), "System Resource".to_string()]
    }

    fn identify_side_effects_from_effect(&self, effect: &PrismEffect) -> Vec<String> {
        vec![format!("{} side effects", effect.definition)]
    }

    fn identify_side_effects_from_runtime_effect(&self, effect: &RuntimeEffect) -> Vec<String> {
        match effect {
            RuntimeEffect::IO { .. } => vec!["File system changes".to_string(), "Network state changes".to_string()],
            RuntimeEffect::Memory { .. } => vec!["Memory allocation changes".to_string()],
            RuntimeEffect::Computation { .. } => vec!["CPU state changes".to_string()],
            RuntimeEffect::SystemCall { .. } => vec!["System state changes".to_string()],
            RuntimeEffect::Custom { .. } => vec!["Custom state changes".to_string()],
        }
    }

    fn extract_invariants_from_effect(&self, effect: &PrismEffect) -> Vec<String> {
        vec![format!("{} maintains system consistency", effect.definition)]
    }

    fn extract_invariants_from_runtime_effect(&self, effect: &RuntimeEffect) -> Vec<String> {
        vec!["Resource limits are respected".to_string(), "System consistency is maintained".to_string()]
    }

    // Helper methods for performance analysis
    fn calculate_efficiency_from_resources(&self, resources: &ResourceMeasurement) -> f64 {
        // Calculate efficiency based on resource consumption
        let total_cost = resources.total_cost();
        if total_cost > 0.0 {
            1.0 / (1.0 + total_cost / 1000.0) // Normalize to 0-1 scale
        } else {
            1.0 // Perfect efficiency if no resources consumed
        }
    }

    fn identify_bottlenecks_from_resources(&self, resources: &ResourceMeasurement) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        
        if resources.cpu_time_ns > 1_000_000_000 { // > 1 second
            bottlenecks.push("CPU intensive".to_string());
        }
        if resources.memory_bytes > 100_000_000 { // > 100MB
            bottlenecks.push("Memory intensive".to_string());
        }
        if resources.network_bytes > 10_000_000 { // > 10MB
            bottlenecks.push("Network intensive".to_string());
        }
        if resources.disk_bytes > 50_000_000 { // > 50MB
            bottlenecks.push("Disk I/O intensive".to_string());
        }
        
        bottlenecks
    }

    // Helper methods for metadata conversion
    fn convert_metadata_to_runtime_effect(&self, metadata: &EffectExecutionMetadata) -> RuntimeEffect {
        // Convert metadata back to runtime effect for storage
        RuntimeEffect::Custom {
            name: metadata.effect_name.clone(),
            metadata: std::collections::HashMap::new(),
        }
    }

    fn convert_metadata_to_runtime_result(&self, metadata: &EffectExecutionMetadata) -> EffectResult {
        EffectResult {
            effect_id: crate::resources::effects::EffectId::new(),
            success: metadata.success,
            error: if metadata.success { None } else { Some("Effect execution failed".to_string()) },
            resources_consumed: metadata.resource_consumption.clone(),
            duration: metadata.duration,
        }
    }

    fn create_execution_context_from_metadata(&self, metadata: &EffectExecutionMetadata) -> Result<ExecutionContext, AIMetadataError> {
        // Create a minimal execution context from metadata
        // This is a simplified version - in practice, you'd want to preserve more context
        let context = ExecutionContext::new(
            crate::platform::execution::ExecutionTarget::Native, // Default target
            crate::authority::capability::ComponentId::new(1), // Default component
            crate::authority::capability::CapabilitySet::new(), // Empty capabilities
        );
        
        Ok(context)
    }
} 