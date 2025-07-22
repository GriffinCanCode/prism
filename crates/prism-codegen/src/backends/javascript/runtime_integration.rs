//! Runtime Integration for JavaScript Backend
//!
//! This module handles integration with prism-runtime infrastructure,
//! including capability management and effect tracking.

use super::{JavaScriptResult, JavaScriptError};
use crate::backends::{PrismIR, PIRModule, PIRFunction, Capability, Effect};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Runtime integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Enable capability management
    pub enable_capabilities: bool,
    /// Enable effect tracking
    pub enable_effects: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable business rule integration
    pub enable_business_rules: bool,
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    /// Runtime validation level
    pub validation_level: ValidationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    None,
    Basic,
    Full,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enable_capabilities: true,
            enable_effects: true,
            enable_performance_monitoring: true,
            enable_business_rules: true,
            enable_ai_metadata: true,
            validation_level: ValidationLevel::Full,
        }
    }
}

/// Runtime integrator for JavaScript backend
pub struct RuntimeIntegrator {
    config: RuntimeConfig,
    registered_capabilities: HashMap<String, String>,
    registered_effects: HashMap<String, String>,
}

impl RuntimeIntegrator {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            registered_capabilities: HashMap::new(),
            registered_effects: HashMap::new(),
        }
    }

    /// Generate comprehensive runtime integration code
    pub fn generate_runtime_integration(&mut self, pir: &PrismIR) -> JavaScriptResult<String> {
        let mut output = String::new();
        
        // Generate runtime imports and initialization
        output.push_str(&self.generate_runtime_imports());
        
        // Generate runtime initialization
        output.push_str(&self.generate_runtime_initialization(pir));
        
        // Generate capability management system
        if self.config.enable_capabilities {
            output.push_str(&self.generate_capability_management(pir));
        }
        
        // Generate effect tracking system
        if self.config.enable_effects {
            output.push_str(&self.generate_effect_tracking(pir));
        }
        
        // Generate performance monitoring
        if self.config.enable_performance_monitoring {
            output.push_str(&self.generate_performance_monitoring());
        }
        
        // Generate business rule integration
        if self.config.enable_business_rules {
            output.push_str(&self.generate_business_rule_integration());
        }
        
        // Generate AI metadata system
        if self.config.enable_ai_metadata {
            output.push_str(&self.generate_ai_metadata_system(pir));
        }
        
        Ok(output)
    }

    /// Generate runtime imports and type definitions
    fn generate_runtime_imports(&self) -> String {
        r#"// === PRISM RUNTIME INTEGRATION ===

// Runtime imports - these would come from @prism/runtime package
const PrismRuntime = globalThis.PrismRuntime || (() => {
    throw new Error('Prism runtime not loaded. Please include @prism/runtime before using generated code.');
})();

// Runtime state management
let runtimeInstance = null;
let isRuntimeInitialized = false;

// Runtime type definitions for JavaScript
class RuntimeError extends Error {
    constructor(message, context = {}) {
        super(message);
        this.name = 'RuntimeError';
        this.context = context;
        this.timestamp = new Date().toISOString();
    }
}

class CapabilityError extends RuntimeError {
    constructor(message, requiredCapability, context = {}) {
        super(message, context);
        this.name = 'CapabilityError';
        this.requiredCapability = requiredCapability;
    }
}

class EffectError extends RuntimeError {
    constructor(message, effect, context = {}) {
        super(message, context);
        this.name = 'EffectError';
        this.effect = effect;
    }
}

class ValidationError extends RuntimeError {
    constructor(message, validationType, context = {}) {
        super(message, context);
        this.name = 'ValidationError';
        this.validationType = validationType;
    }
}

"#.to_string()
    }

    /// Generate runtime initialization
    fn generate_runtime_initialization(&self, pir: &PrismIR) -> String {
        let capabilities = self.extract_capabilities_from_pir(pir);
        let effects = self.extract_effects_from_pir(pir);
        
        format!(
            r#"/**
 * Initialize the Prism JavaScript runtime
 * Must be called before using any generated functions with capabilities or effects
 * 
 * @param {{Object}} config - Optional runtime configuration
 * @returns {{Promise<Object>}} Promise that resolves when runtime is fully initialized
 */
export async function initializePrismRuntime(config = {{}}) {{
    if (runtimeInstance && isRuntimeInitialized) {{
        console.warn('Prism runtime already initialized');
        return runtimeInstance;
    }}
    
    try {{
        const startTime = Date.now();
        
        // Create runtime instance with enhanced configuration
        const runtimeConfig = {{
            enableResourceTracking: true,
            enableCapabilityValidation: true,
            enableBusinessRules: true,
            enableEffectTracking: true,
            enableAIMetadata: true,
            enablePerformanceMonitoring: {},
            resourceLimits: {{
                maxMemoryMB: 512,
                maxExecutionTimeMs: 30000,
                maxConcurrentOperations: 100
            }},
            ...config,
        }};
        
        // Initialize runtime instance
        runtimeInstance = await PrismRuntime.create(runtimeConfig);
        
        // Register all capabilities from PIR
        await registerCapabilities(runtimeInstance, [
{}
        ]);
        
        // Register all effects from PIR
        await registerEffects(runtimeInstance, [
{}
        ]);
        
        // Initialize business rule engine
        if (runtimeConfig.enableBusinessRules) {{
            await initializeBusinessRuleEngine(runtimeInstance);
        }}
        
        // Initialize AI metadata provider
        if (runtimeConfig.enableAIMetadata) {{
            await initializeAIMetadataProvider(runtimeInstance);
        }}
        
        const initializationTime = Date.now() - startTime;
        isRuntimeInitialized = true;
        
        console.log(`Prism JavaScript runtime initialized successfully in ${{initializationTime}}ms`);
        console.log(`Modules: {}, Cohesion Score: {:.2}`, {}, {});
        console.log(`Capabilities: {}, Effects: {}`, {}, {});
        
        return runtimeInstance;
    }} catch (error) {{
        console.error('Failed to initialize Prism runtime:', error);
        throw new RuntimeError('Runtime initialization failed', {{ cause: error }});
    }}
}}

/**
 * Get the current runtime instance
 * Throws if runtime hasn't been initialized
 * 
 * @returns {{Object}} Current runtime instance
 */
export function getRuntimeInstance() {{
    if (!runtimeInstance || !isRuntimeInitialized) {{
        throw new RuntimeError('Prism runtime not initialized. Call initializePrismRuntime() first.');
    }}
    return runtimeInstance;
}}

/**
 * Check if runtime is initialized
 * @returns {{boolean}} True if runtime is ready for use
 */
export function isRuntimeReady() {{
    return runtimeInstance !== null && isRuntimeInitialized;
}}

"#,
            self.config.enable_performance_monitoring,
            capabilities.iter()
                .map(|(name, desc)| format!("            {{ name: '{}', description: '{}' }}", name, desc))
                .collect::<Vec<_>>()
                .join(",\n"),
            effects.iter()
                .map(|(name, desc)| format!("            {{ name: '{}', description: '{}' }}", name, desc))
                .collect::<Vec<_>>()
                .join(",\n"),
            pir.modules.len(),
            pir.cohesion_metrics.overall_score,
            capabilities.len(),
            effects.len()
        )
    }

    /// Generate capability management system
    fn generate_capability_management(&self, pir: &PrismIR) -> String {
        let capabilities = self.extract_capabilities_from_pir(pir);
        
        let mut output = String::new();
        
        output.push_str(r#"
// === CAPABILITY MANAGEMENT SYSTEM ===

/**
 * Register capabilities with the runtime
 * @param {Object} runtime - Runtime instance
 * @param {Array} capabilities - Array of capability definitions
 */
async function registerCapabilities(runtime, capabilities) {
    for (const capability of capabilities) {
        await runtime.registerCapability(capability.name, {
            description: capability.description,
            validator: createCapabilityValidator(capability.name),
            metadata: {
                generatedBy: 'prism-javascript-backend',
                timestamp: new Date().toISOString()
            }
        });
    }
}

/**
 * Create a capability validator function
 * @param {string} capabilityName - Name of the capability
 * @returns {Function} Validator function
 */
function createCapabilityValidator(capabilityName) {
    return async function validateCapability(context) {
        // Default validation - can be overridden for specific capabilities
        return {
            valid: true,
            reason: `Capability ${capabilityName} is available`,
            metadata: {
                checkedAt: new Date().toISOString(),
                context: context
            }
        };
    };
}

/**
 * Validate that required capabilities are available
 * @param {Array<string>} requiredCapabilities - List of required capability names
 * @param {Object} context - Execution context
 * @returns {Promise<void>} Throws if capabilities are not available
 */
export async function validateCapabilities(requiredCapabilities, context = {}) {
    const runtime = getRuntimeInstance();
    
    for (const capabilityName of requiredCapabilities) {
        const result = await runtime.validateCapability(capabilityName, context);
        if (!result.valid) {
            throw new CapabilityError(
                `Required capability '${capabilityName}' is not available: ${result.reason}`,
                capabilityName,
                { context, result }
            );
        }
    }
}

"#);

        // Generate specific capability validators
        for (capability_name, capability_desc) in &capabilities {
            output.push_str(&format!(
                r#"/**
 * Validate {} capability
 * {}
 * @param {{Object}} context - Execution context
 * @returns {{Promise<boolean>}} True if capability is available
 */
export async function validate{}Capability(context = {{}}) {{
    try {{
        await validateCapabilities(['{}'], context);
        return true;
    }} catch (error) {{
        console.warn(`{} capability validation failed:`, error.message);
        return false;
    }}
}}

"#,
                capability_name,
                capability_desc,
                capability_name.replace(" ", "").replace("-", ""),
                capability_name,
                capability_name
            ));
        }

        output
    }

    /// Generate effect tracking system
    fn generate_effect_tracking(&self, pir: &PrismIR) -> String {
        let effects = self.extract_effects_from_pir(pir);
        
        let mut output = String::new();
        
        output.push_str(r#"
// === EFFECT TRACKING SYSTEM ===

/**
 * Register effects with the runtime
 * @param {Object} runtime - Runtime instance
 * @param {Array} effects - Array of effect definitions
 */
async function registerEffects(runtime, effects) {
    for (const effect of effects) {
        await runtime.registerEffect(effect.name, {
            description: effect.description,
            handler: createEffectHandler(effect.name),
            metadata: {
                generatedBy: 'prism-javascript-backend',
                timestamp: new Date().toISOString()
            }
        });
    }
}

/**
 * Create an effect handler function
 * @param {string} effectName - Name of the effect
 * @returns {Function} Effect handler function
 */
function createEffectHandler(effectName) {
    return async function handleEffect(effectData, context) {
        // Default effect handling - can be overridden for specific effects
        return {
            success: true,
            result: effectData,
            metadata: {
                effect: effectName,
                handledAt: new Date().toISOString(),
                context: context
            }
        };
    };
}

/**
 * Track effect execution
 * @param {Array<string>} effects - List of effect names
 * @param {Function} operation - Operation to execute with effect tracking
 * @param {Object} context - Execution context
 * @returns {Promise<any>} Result of the operation
 */
export async function trackEffects(effects, operation, context = {}) {
    const runtime = getRuntimeInstance();
    const tracker = await runtime.createEffectTracker(effects, context);
    
    try {
        await tracker.start();
        const result = await operation();
        await tracker.complete(result);
        return result;
    } catch (error) {
        await tracker.abort(error);
        throw new EffectError(`Effect tracking failed for [${effects.join(', ')}]`, effects, {
            context,
            error: error.message
        });
    }
}

"#);

        // Generate specific effect trackers
        for (effect_name, effect_desc) in &effects {
            output.push_str(&format!(
                r#"/**
 * Track {} effect
 * {}
 * @param {{Function}} operation - Operation to execute with effect tracking
 * @param {{Object}} context - Execution context
 * @returns {{Promise<any>}} Result of the operation
 */
export async function track{}Effect(operation, context = {{}}) {{
    return trackEffects(['{}'], operation, context);
}}

"#,
                effect_name,
                effect_desc,
                effect_name.replace(" ", "").replace("-", ""),
                effect_name
            ));
        }

        output
    }

    /// Generate performance monitoring
    fn generate_performance_monitoring(&self) -> String {
        r#"
// === PERFORMANCE MONITORING ===

/**
 * Performance monitor for JavaScript runtime
 */
class PerformanceMonitor {
    constructor() {
        this.metrics = new Map();
        this.startTimes = new Map();
    }

    /**
     * Start monitoring an operation
     * @param {string} operationName - Name of the operation
     * @returns {string} Monitoring session ID
     */
    start(operationName) {
        const sessionId = `${operationName}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.startTimes.set(sessionId, {
            name: operationName,
            startTime: performance.now(),
            startMemory: this.getMemoryUsage()
        });
        return sessionId;
    }

    /**
     * End monitoring an operation
     * @param {string} sessionId - Monitoring session ID
     * @returns {Object} Performance metrics
     */
    end(sessionId) {
        const startData = this.startTimes.get(sessionId);
        if (!startData) {
            throw new Error(`No monitoring session found for ID: ${sessionId}`);
        }

        const endTime = performance.now();
        const endMemory = this.getMemoryUsage();
        
        const metrics = {
            operation: startData.name,
            duration: endTime - startData.startTime,
            memoryDelta: endMemory - startData.startMemory,
            timestamp: new Date().toISOString()
        };

        this.metrics.set(sessionId, metrics);
        this.startTimes.delete(sessionId);
        
        return metrics;
    }

    /**
     * Get current memory usage
     * @returns {number} Memory usage in bytes
     */
    getMemoryUsage() {
        if (performance.memory) {
            return performance.memory.usedJSHeapSize;
        }
        return 0; // Fallback for environments without memory API
    }

    /**
     * Get all recorded metrics
     * @returns {Array} Array of performance metrics
     */
    getAllMetrics() {
        return Array.from(this.metrics.values());
    }
}

// Global performance monitor instance
export const performanceMonitor = new PerformanceMonitor();

/**
 * Monitor the performance of an async operation
 * @param {string} operationName - Name of the operation
 * @param {Function} operation - Async operation to monitor
 * @returns {Promise<{result: any, metrics: Object}>} Result and performance metrics
 */
export async function monitorPerformance(operationName, operation) {
    const sessionId = performanceMonitor.start(operationName);
    
    try {
        const result = await operation();
        const metrics = performanceMonitor.end(sessionId);
        return { result, metrics };
    } catch (error) {
        performanceMonitor.end(sessionId); // Still record metrics on error
        throw error;
    }
}

"#.to_string()
    }

    /// Generate business rule integration
    fn generate_business_rule_integration(&self) -> String {
        r#"
// === BUSINESS RULE INTEGRATION ===

/**
 * Initialize business rule engine with runtime
 * @param {Object} runtime - Runtime instance
 */
async function initializeBusinessRuleEngine(runtime) {
    await runtime.enableBusinessRuleValidation({
        strictMode: true,
        cacheValidation: true,
        auditTrail: true
    });
}

/**
 * Validate business rules for a value
 * @param {string} typeName - Semantic type name
 * @param {any} value - Value to validate
 * @param {Object} context - Validation context
 * @returns {Promise<{valid: boolean, violations: Array}>} Validation result
 */
export async function validateBusinessRules(typeName, value, context = {}) {
    const runtime = getRuntimeInstance();
    
    try {
        const result = await runtime.validateBusinessRules(typeName, value, context);
        return result;
    } catch (error) {
        throw new ValidationError(`Business rule validation failed for type '${typeName}'`, 'business_rule', {
            typeName,
            value,
            context,
            error: error.message
        });
    }
}

"#.to_string()
    }

    /// Generate AI metadata system
    fn generate_ai_metadata_system(&self, pir: &PrismIR) -> String {
        format!(
            r#"
// === AI METADATA SYSTEM ===

/**
 * Initialize AI metadata provider with runtime
 * @param {{Object}} runtime - Runtime instance
 */
async function initializeAIMetadataProvider(runtime) {{
    await runtime.enableAIMetadata({{
        includeSemanticTypes: true,
        includeBusinessRules: true,
        includePerformanceMetrics: true,
        includeCapabilityInfo: true,
        includeEffectInfo: true
    }});
}}

/**
 * Get comprehensive AI metadata for the generated module
 * @returns {{Object}} AI-readable metadata
 */
export function getAIMetadata() {{
    return {{
        generatedBy: 'prism-javascript-backend',
        generatedAt: new Date().toISOString(),
        pirVersion: '{}',
        cohesionScore: {},
        moduleCount: {},
        capabilities: [
            // Capabilities would be populated from PIR
        ],
        effects: [
            // Effects would be populated from PIR
        ],
        semanticTypes: [
            // Semantic types would be populated from PIR
        ],
        businessRules: [
            // Business rules would be populated from PIR
        ],
        performanceProfile: {{
            target: 'javascript',
            optimizationLevel: 'development',
            memoryModel: 'garbage_collected'
        }}
    }};
}}

/**
 * Export AI metadata to external systems
 * @param {{string}} format - Export format ('json', 'yaml', 'xml')
 * @returns {{string}} Serialized metadata
 */
export function exportAIMetadata(format = 'json') {{
    const metadata = getAIMetadata();
    
    switch (format.toLowerCase()) {{
        case 'json':
            return JSON.stringify(metadata, null, 2);
        case 'yaml':
            // Would need YAML serialization library
            throw new Error('YAML export not yet implemented');
        case 'xml':
            // Would need XML serialization library
            throw new Error('XML export not yet implemented');
        default:
            throw new Error(`Unsupported export format: ${{format}}`);
    }}
}}

"#,
            pir.metadata.version,
            pir.cohesion_metrics.overall_score,
            pir.modules.len()
        )
    }

    /// Extract capabilities from PIR
    fn extract_capabilities_from_pir(&self, pir: &PrismIR) -> Vec<(String, String)> {
        let mut capabilities = Vec::new();
        
        for module in &pir.modules {
            for capability in &module.capabilities {
                capabilities.push((capability.name.clone(), capability.description.clone()));
            }
        }
        
        capabilities.sort();
        capabilities.dedup();
        capabilities
    }

    /// Extract effects from PIR
    fn extract_effects_from_pir(&self, pir: &PrismIR) -> Vec<(String, String)> {
        let mut effects = Vec::new();
        
        for (_effect_name, effect_node) in &pir.effect_graph.nodes {
            effects.push((effect_node.effect_type.clone(), effect_node.effect_type.clone()));
        }
        
        effects.sort();
        effects.dedup();
        effects
    }

    pub fn generate_capability_setup(&self) -> JavaScriptResult<String> {
        Ok("// Capability setup integrated into runtime initialization".to_string())
    }
}

/// Capability manager for JavaScript runtime
pub struct CapabilityManager {
    config: RuntimeConfig,
    capabilities: Vec<String>,
}

impl CapabilityManager {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            capabilities: Vec::new(),
        }
    }

    pub fn add_capability(&mut self, capability: String) {
        self.capabilities.push(capability);
    }

    pub fn generate_capability_checks(&self) -> JavaScriptResult<String> {
        Ok("// Capability checks integrated into runtime system".to_string())
    }
}

/// Effect tracker for JavaScript runtime
pub struct EffectTracker {
    config: RuntimeConfig,
    effects: Vec<String>,
}

impl EffectTracker {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            effects: Vec::new(),
        }
    }

    pub fn track_effect(&mut self, effect: String) {
        self.effects.push(effect);
    }

    pub fn generate_effect_tracking(&self) -> JavaScriptResult<String> {
        Ok("// Effect tracking integrated into runtime system".to_string())
    }
} 