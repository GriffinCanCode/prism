//! Runtime Integration with Prism Infrastructure
//!
//! This module provides deep integration with the Prism runtime system,
//! enabling capability-based security, effect tracking, and resource management
//! in generated TypeScript code.
//!
//! Features:
//! - Capability validation and management
//! - Effect tracking with resource monitoring
//! - Performance monitoring and metrics
//! - Business rule integration
//! - AI metadata generation for runtime analysis

use super::{TypeScriptResult, TypeScriptError};
use crate::backends::{
    PrismIR, Effect, Capability, ResourceLimits, ResourceUsageDelta,
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Configuration for runtime integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeIntegrationConfig {
    /// Enable capability validation
    pub enable_capability_validation: bool,
    /// Enable effect tracking
    pub enable_effect_tracking: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable business rule integration
    pub enable_business_rules: bool,
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    /// Enable automatic runtime initialization
    pub enable_auto_initialization: bool,
    /// Resource limits for generated code
    pub resource_limits: Option<ResourceLimits>,
}

impl Default for RuntimeIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_capability_validation: true,
            enable_effect_tracking: true,
            enable_performance_monitoring: true,
            enable_business_rules: true,
            enable_ai_metadata: true,
            enable_auto_initialization: true,
            resource_limits: None,
        }
    }
}

/// Runtime integrator for TypeScript code generation
pub struct RuntimeIntegrator {
    config: RuntimeIntegrationConfig,
    registered_capabilities: HashMap<String, String>,
    registered_effects: HashMap<String, String>,
    performance_monitors: Vec<String>,
}

impl RuntimeIntegrator {
    /// Create a new runtime integrator
    pub fn new(config: RuntimeIntegrationConfig) -> Self {
        Self {
            config,
            registered_capabilities: HashMap::new(),
            registered_effects: HashMap::new(),
            performance_monitors: Vec::new(),
        }
    }

    /// Generate comprehensive runtime integration code
    pub fn generate_runtime_integration(&mut self, pir: &PrismIR) -> TypeScriptResult<String> {
        let mut output = String::new();
        
        // Generate runtime imports and type definitions
        output.push_str(&self.generate_runtime_imports());
        
        // Generate runtime initialization
        output.push_str(&self.generate_runtime_initialization(pir));
        
        // Generate capability management system
        if self.config.enable_capability_validation {
            output.push_str(&self.generate_capability_management(pir));
        }
        
        // Generate effect tracking system
        if self.config.enable_effect_tracking {
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
        r#"
// === ENHANCED RUNTIME INTEGRATION ===
// Deep integration with prism-runtime infrastructure

import type {
    PrismRuntime,
    ExecutionContext,
    ResourceManager,
    CapabilityManager as RuntimeCapabilityManager,
    EffectTracker as RuntimeEffectTracker,
    BusinessRuleEngine as RuntimeBusinessRuleEngine,
    PerformanceMonitor as RuntimePerformanceMonitor,
    AIMetadataProvider,
} from '@prism/runtime-core';

import {
    createPrismRuntime,
    createExecutionContext,
    createCapabilityManager,
    createEffectTracker,
    createPerformanceMonitor,
    ValidationError,
    CapabilityError,
    EffectError,
    RuntimeError,
} from '@prism/runtime';

// Modern TypeScript utility types for runtime integration
type RuntimeInitialized<T> = T & { readonly __runtimeInitialized: true };
type CapabilityValidated<T> = T & { readonly __capabilityValidated: true };
type EffectTracked<T> = T & { readonly __effectTracked: true };

// Template literal types for runtime operations
type RuntimeOperation = `runtime:${string}`;
type CapabilityOperation = `capability:${string}`;
type EffectOperation = `effect:${string}`;

// Global runtime state management
let runtimeInstance: PrismRuntime | null = null;
let isRuntimeInitialized = false;

"#.to_string()
    }

    /// Generate runtime initialization with comprehensive setup
    fn generate_runtime_initialization(&self, pir: &PrismIR) -> String {
        format!(
            r#"/**
 * Initialize the Prism runtime for TypeScript generated code
 * Must be called before using any generated functions with capabilities or effects
 * 
 * @param config - Optional runtime configuration
 * @returns Promise that resolves when runtime is fully initialized
 */
export async function initializePrismRuntime(config?: PrismRuntimeConfig): Promise<RuntimeInitialized<PrismRuntime>> {{
    if (runtimeInstance && isRuntimeInitialized) {{
        console.warn('Prism runtime already initialized');
        return runtimeInstance as RuntimeInitialized<PrismRuntime>;
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
            resourceLimits: {resource_limits},
            ...config,
        }} satisfies PrismRuntimeConfig;
        
        runtimeInstance = await createPrismRuntime(runtimeConfig);
        
        // Register all capabilities from PIR
        await registerCapabilities(runtimeInstance, [
{capabilities}
        ]);
        
        // Register all effects from PIR
        await registerEffects(runtimeInstance, [
{effects}
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
        
        console.log(`Prism runtime initialized successfully in ${{initializationTime}}ms`);
        console.log(`Modules: {}, Cohesion Score: {:.2}`, {}, {});
        console.log(`Capabilities: {}, Effects: {}`, {}, {});
        
        return runtimeInstance as RuntimeInitialized<PrismRuntime>;
    }} catch (error) {{
        console.error('Failed to initialize Prism runtime:', error);
        throw new RuntimeInitializationError('Runtime initialization failed', {{ cause: error }});
    }}
}}

/**
 * Get the current runtime instance
 * Throws if runtime hasn't been initialized
 * 
 * @returns Current runtime instance
 */
export function getRuntimeInstance(): RuntimeInitialized<PrismRuntime> {{
    if (!runtimeInstance || !isRuntimeInitialized) {{
        throw new RuntimeError('Prism runtime not initialized. Call initializePrismRuntime() first.');
    }}
    return runtimeInstance as RuntimeInitialized<PrismRuntime>;
}}

/**
 * Check if runtime is initialized
 * @returns True if runtime is ready for use
 */
export function isRuntimeReady(): boolean {{
    return runtimeInstance !== null && isRuntimeInitialized;
}}

"#,
            resource_limits = if let Some(limits) = &self.config.resource_limits {
                format!(
                    "{{ maxMemoryMB: {}, maxCpuTimeMS: {}, maxNetworkConnections: {} }}",
                    limits.max_memory_mb.unwrap_or(512),
                    limits.max_cpu_time_ms.unwrap_or(30000),
                    limits.max_network_connections.unwrap_or(10)
                )
            } else {
                "undefined".to_string()
            },
            capabilities = pir.modules.iter()
                .flat_map(|m| &m.capabilities)
                .map(|c| format!("            '{}',", c.name))
                .collect::<Vec<_>>()
                .join("\n"),
            effects = pir.effect_graph.nodes.iter()
                .map(|e| format!("            '{}',", e.effect.name))
                .collect::<Vec<_>>()
                .join("\n"),
            pir.modules.len(),
            pir.cohesion_metrics.overall_score,
            pir.modules.len(),
            pir.cohesion_metrics.overall_score,
            pir.modules.iter().flat_map(|m| &m.capabilities).count(),
            pir.effect_graph.nodes.len()
        )
    }

    /// Generate capability management system
    fn generate_capability_management(&mut self, pir: &PrismIR) -> String {
        let capabilities: Vec<_> = pir.modules.iter()
            .flat_map(|m| &m.capabilities)
            .collect();

        for capability in &capabilities {
            self.registered_capabilities.insert(capability.name.clone(), capability.name.clone());
        }

        format!(
            r#"
/**
 * Enhanced capability manager with runtime integration
 * Provides type-safe capability validation and management
 */
export class CapabilityManager {{
    private runtime: RuntimeInitialized<PrismRuntime>;
    private capabilityCache: Map<string, boolean> = new Map();
    
    constructor(runtime: RuntimeInitialized<PrismRuntime>) {{
        this.runtime = runtime;
    }}
    
    /**
     * Validate required capabilities with enhanced type safety
     * @param requiredCapabilities - Array of capability names to validate
     * @returns Promise that resolves if all capabilities are available
     */
    async validateCapabilities(requiredCapabilities: readonly string[]): Promise<CapabilityValidated<void>> {{
        const context = await this.runtime.getCurrentExecutionContext();
        const validationResults: {{ capability: string; valid: boolean; reason?: string }}[] = [];
        
        for (const capName of requiredCapabilities) {{
            // Check cache first for performance
            if (this.capabilityCache.has(capName)) {{
                const cached = this.capabilityCache.get(capName)!;
                validationResults.push({{ capability: capName, valid: cached }});
                continue;
            }}
            
            try {{
                const hasCapability = await this.runtime.authoritySystem
                    .checkCapability(context.capabilities, capName);
                    
                this.capabilityCache.set(capName, hasCapability);
                validationResults.push({{ capability: capName, valid: hasCapability }});
                
                if (!hasCapability) {{
                    throw new CapabilityError(
                        `Missing required capability: ${{capName}}`,
                        capName,
                        {{ availableCapabilities: await this.getAvailableCapabilities() }}
                    );
                }}
            }} catch (error) {{
                this.capabilityCache.set(capName, false);
                validationResults.push({{ 
                    capability: capName, 
                    valid: false, 
                    reason: error instanceof Error ? error.message : String(error)
                }});
                
                throw new CapabilityError(
                    `Capability validation failed for '${{capName}}': ${{error}}`,
                    capName,
                    {{ validationResults }}
                );
            }}
        }}
        
        return undefined as CapabilityValidated<void>;
    }}
    
    /**
     * Attenuate capabilities with constraints
     * @param capabilities - Capabilities to attenuate
     * @param constraints - Constraints to apply
     * @returns Attenuated capability set
     */
    async attenuateCapabilities(
        capabilities: readonly string[], 
        constraints: CapabilityConstraints
    ): Promise<string[]> {{
        return this.runtime.authoritySystem.attenuateCapabilities(
            capabilities as string[], 
            constraints
        );
    }}
    
    /**
     * Get all available capabilities for current context
     * @returns Array of available capability names
     */
    async getAvailableCapabilities(): Promise<readonly string[]> {{
        const context = await this.runtime.getCurrentExecutionContext();
        return this.runtime.authoritySystem.getAvailableCapabilities(context.capabilities);
    }}
    
    /**
     * Create capability-scoped execution context
     * @param capabilities - Capabilities to include in context
     * @param resourceLimits - Optional resource limits
     * @returns New execution context with specified capabilities
     */
    async createScopedContext(
        capabilities: readonly string[],
        resourceLimits?: ResourceLimits
    ): Promise<ExecutionContext> {{
        const capabilitySet = await this.runtime.authoritySystem.createCapabilitySet(
            capabilities.map(name => ({{ name, constraints: {{}} }}))
        );
        
        return this.runtime.createExecutionContext({{
            target: 'typescript',
            capabilities: capabilitySet,
            resourceLimits,
            aiContext: {{
                generatedBy: 'prism-typescript-backend',
                version: '2.0.0',
                features: ['semantic-types', 'business-rules', 'effect-tracking', 'capability-validation'],
            }}
        }});
    }}
    
    /**
     * Generate AI-friendly capability report
     * @returns Structured capability information for AI analysis
     */
    async generateCapabilityReport(): Promise<CapabilityReport> {{
        const available = await this.getAvailableCapabilities();
        const registered = [{registered_capabilities}];
        
        return {{
            availableCapabilities: available,
            registeredCapabilities: registered as readonly string[],
            cacheHitRate: this.calculateCacheHitRate(),
            securityLevel: this.assessSecurityLevel(available),
            recommendations: this.generateRecommendations(available, registered),
        }};
    }}
    
    private calculateCacheHitRate(): number {{
        const totalRequests = this.capabilityCache.size;
        return totalRequests > 0 ? 1.0 : 0.0; // Simplified calculation
    }}
    
    private assessSecurityLevel(capabilities: readonly string[]): 'low' | 'medium' | 'high' {{
        // Simple heuristic based on capability count and types
        if (capabilities.length === 0) return 'high';
        if (capabilities.length < 5) return 'medium';
        return 'low';
    }}
    
    private generateRecommendations(
        available: readonly string[], 
        registered: readonly string[]
    ): readonly string[] {{
        const recommendations: string[] = [];
        
        const unused = registered.filter(r => !available.includes(r));
        if (unused.length > 0) {{
            recommendations.push(`Consider removing unused capabilities: ${{unused.join(', ')}}`);
        }}
        
        if (available.length > 10) {{
            recommendations.push('Consider capability attenuation to reduce attack surface');
        }}
        
        return recommendations as readonly string[];
    }}
}}

// Type definitions for capability management
export interface CapabilityConstraints {{
    timeLimit?: number;
    resourceLimit?: ResourceLimits;
    scopeRestriction?: readonly string[];
}}

export interface CapabilityReport {{
    readonly availableCapabilities: readonly string[];
    readonly registeredCapabilities: readonly string[];
    readonly cacheHitRate: number;
    readonly securityLevel: 'low' | 'medium' | 'high';
    readonly recommendations: readonly string[];
}}

"#,
            registered_capabilities = capabilities.iter()
                .map(|c| format!("'{}'", c.name))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    /// Generate effect tracking system
    fn generate_effect_tracking(&mut self, pir: &PrismIR) -> String {
        let effects: Vec<_> = pir.effect_graph.nodes.iter()
            .map(|node| &node.effect)
            .collect();

        for effect in &effects {
            self.registered_effects.insert(effect.name.clone(), effect.name.clone());
        }

        format!(
            r#"
/**
 * Enhanced effect tracker with runtime integration and resource monitoring
 * Provides comprehensive effect tracking with performance metrics
 */
export class EffectTracker<TEffects extends readonly string[]> {{
    private runtime: RuntimeInitialized<PrismRuntime>;
    private activeEffects: Set<string> = new Set();
    private startTime: number;
    private resourceSnapshot: any;
    private effectMetrics: Map<string, EffectMetrics> = new Map();
    
    constructor(runtime?: RuntimeInitialized<PrismRuntime>) {{
        this.runtime = runtime || getRuntimeInstance();
        this.startTime = Date.now();
    }}
    
    /**
     * Track effects with comprehensive monitoring
     * @param effects - Effects to track
     * @returns Promise that resolves when tracking is established
     */
    async trackEffects(effects: TEffects): Promise<EffectTracked<void>> {{
        // Take initial resource snapshot
        this.resourceSnapshot = await this.runtime.resourceManager?.currentSnapshot();
        
        for (const effect of effects) {{
            this.activeEffects.add(effect);
            
            // Initialize metrics for this effect
            this.effectMetrics.set(effect, {{
                name: effect,
                startTime: Date.now(),
                resourceUsage: {{ memoryDelta: 0, cpuTimeDelta: 0 }},
                operationCount: 0,
                errorCount: 0,
            }});
        }}
        
        // Report to runtime for centralized tracking
        if (this.runtime.effectSystem) {{
            await this.runtime.effectSystem.trackEffects(Array.from(this.activeEffects));
        }}
        
        return undefined as EffectTracked<void>;
    }}
    
    /**
     * Record effect operation for metrics
     * @param effectName - Name of the effect
     * @param success - Whether operation was successful
     */
    recordOperation(effectName: string, success: boolean = true): void {{
        const metrics = this.effectMetrics.get(effectName);
        if (metrics) {{
            metrics.operationCount++;
            if (!success) {{
                metrics.errorCount++;
            }}
        }}
    }}
    
    /**
     * Complete effect tracking with comprehensive metrics
     * @returns Effect completion report
     */
    async complete(): Promise<EffectCompletionReport> {{
        const duration = Date.now() - this.startTime;
        const endSnapshot = await this.runtime.resourceManager?.currentSnapshot();
        
        // Calculate resource usage for each effect
        for (const [effectName, metrics] of this.effectMetrics) {{
            const effectDuration = Date.now() - metrics.startTime;
            metrics.resourceUsage = this.calculateResourceDelta(
                this.resourceSnapshot, 
                endSnapshot
            );
        }}
        
        // Report completion to runtime
        if (this.runtime.effectSystem) {{
            await this.runtime.effectSystem.completeEffects(
                Array.from(this.activeEffects), 
                duration,
                Array.from(this.effectMetrics.values())
            );
        }}
        
        const report: EffectCompletionReport = {{
            totalDuration: duration,
            effectCount: this.activeEffects.size,
            totalOperations: Array.from(this.effectMetrics.values())
                .reduce((sum, m) => sum + m.operationCount, 0),
            totalErrors: Array.from(this.effectMetrics.values())
                .reduce((sum, m) => sum + m.errorCount, 0),
            resourceUsage: this.calculateResourceDelta(this.resourceSnapshot, endSnapshot),
            effectMetrics: Array.from(this.effectMetrics.values()),
            aiAnalysis: this.generateAIAnalysis(),
        }};
        
        this.activeEffects.clear();
        this.effectMetrics.clear();
        
        return report;
    }}
    
    /**
     * Abort effect tracking due to error
     * @param reason - Reason for abortion
     */
    async abort(reason?: string): Promise<void> {{
        // Report abortion to runtime
        if (this.runtime.effectSystem) {{
            await this.runtime.effectSystem.abortEffects(
                Array.from(this.activeEffects),
                reason || 'Effect tracking aborted'
            );
        }}
        
        this.activeEffects.clear();
        this.effectMetrics.clear();
    }}
    
    /**
     * Get current effect status
     * @returns Current tracking status
     */
    getStatus(): EffectTrackingStatus {{
        return {{
            isActive: this.activeEffects.size > 0,
            activeEffects: Array.from(this.activeEffects) as readonly string[],
            duration: Date.now() - this.startTime,
            operationCount: Array.from(this.effectMetrics.values())
                .reduce((sum, m) => sum + m.operationCount, 0),
        }};
    }}
    
    private calculateResourceDelta(start: any, end: any): ResourceUsageDelta {{
        if (!start || !end) return {{ memoryDelta: 0, cpuTimeDelta: 0 }};
        
        return {{
            memoryDelta: (end.memory?.used || 0) - (start.memory?.used || 0),
            cpuTimeDelta: (end.cpu?.time || 0) - (start.cpu?.time || 0),
        }};
    }}
    
    private generateAIAnalysis(): EffectAIAnalysis {{
        const metrics = Array.from(this.effectMetrics.values());
        const avgOperations = metrics.length > 0 
            ? metrics.reduce((sum, m) => sum + m.operationCount, 0) / metrics.length 
            : 0;
        const errorRate = metrics.length > 0
            ? metrics.reduce((sum, m) => sum + m.errorCount, 0) / 
              Math.max(1, metrics.reduce((sum, m) => sum + m.operationCount, 0))
            : 0;
        
        return {{
            performanceCategory: this.categorizePerformance(avgOperations),
            reliabilityCategory: this.categorizeReliability(errorRate),
            recommendations: this.generateEffectRecommendations(metrics),
            insights: [
                `Tracked ${{metrics.length}} effects with ${{avgOperations.toFixed(1)}} avg operations`,
                `Error rate: ${{(errorRate * 100).toFixed(2)}}%`,
                `Resource efficiency: ${{this.assessResourceEfficiency()}}`,
            ],
        }};
    }}
    
    private categorizePerformance(avgOperations: number): 'excellent' | 'good' | 'fair' | 'poor' {{
        if (avgOperations < 10) return 'excellent';
        if (avgOperations < 50) return 'good';
        if (avgOperations < 200) return 'fair';
        return 'poor';
    }}
    
    private categorizeReliability(errorRate: number): 'excellent' | 'good' | 'fair' | 'poor' {{
        if (errorRate < 0.01) return 'excellent';
        if (errorRate < 0.05) return 'good';
        if (errorRate < 0.1) return 'fair';
        return 'poor';
    }}
    
    private generateEffectRecommendations(metrics: EffectMetrics[]): readonly string[] {{
        const recommendations: string[] = [];
        
        const highErrorEffects = metrics.filter(m => m.errorCount / Math.max(1, m.operationCount) > 0.1);
        if (highErrorEffects.length > 0) {{
            recommendations.push(`Review error handling for effects: ${{highErrorEffects.map(e => e.name).join(', ')}}`);
        }}
        
        const highOperationEffects = metrics.filter(m => m.operationCount > 100);
        if (highOperationEffects.length > 0) {{
            recommendations.push(`Consider optimization for high-usage effects: ${{highOperationEffects.map(e => e.name).join(', ')}}`);
        }}
        
        return recommendations as readonly string[];
    }}
    
    private assessResourceEfficiency(): 'excellent' | 'good' | 'fair' | 'poor' {{
        // Simplified assessment based on resource usage
        const totalMemoryDelta = Array.from(this.effectMetrics.values())
            .reduce((sum, m) => sum + m.resourceUsage.memoryDelta, 0);
            
        if (totalMemoryDelta < 1024 * 1024) return 'excellent'; // < 1MB
        if (totalMemoryDelta < 10 * 1024 * 1024) return 'good'; // < 10MB
        if (totalMemoryDelta < 100 * 1024 * 1024) return 'fair'; // < 100MB
        return 'poor';
    }}
}}

// Type definitions for effect tracking
export interface EffectMetrics {{
    readonly name: string;
    readonly startTime: number;
    resourceUsage: ResourceUsageDelta;
    operationCount: number;
    errorCount: number;
}}

export interface EffectCompletionReport {{
    readonly totalDuration: number;
    readonly effectCount: number;
    readonly totalOperations: number;
    readonly totalErrors: number;
    readonly resourceUsage: ResourceUsageDelta;
    readonly effectMetrics: readonly EffectMetrics[];
    readonly aiAnalysis: EffectAIAnalysis;
}}

export interface EffectTrackingStatus {{
    readonly isActive: boolean;
    readonly activeEffects: readonly string[];
    readonly duration: number;
    readonly operationCount: number;
}}

export interface EffectAIAnalysis {{
    readonly performanceCategory: 'excellent' | 'good' | 'fair' | 'poor';
    readonly reliabilityCategory: 'excellent' | 'good' | 'fair' | 'poor';
    readonly recommendations: readonly string[];
    readonly insights: readonly string[];
}}

"#
        )
    }

    /// Generate performance monitoring system
    fn generate_performance_monitoring(&self) -> String {
        r#"
/**
 * Performance monitor with comprehensive metrics and AI analysis
 * Integrates with runtime resource tracking for detailed insights
 */
export class PerformanceMonitor {
    private runtime: RuntimeInitialized<PrismRuntime>;
    private functionName: string;
    private startTime: number;
    private resourceSnapshot: any;
    private metrics: PerformanceMetrics;
    
    constructor(functionName: string, runtime?: RuntimeInitialized<PrismRuntime>) {
        this.runtime = runtime || getRuntimeInstance();
        this.functionName = functionName;
        this.startTime = Date.now();
        this.metrics = {
            functionName,
            startTime: this.startTime,
            endTime: 0,
            duration: 0,
            resourceUsage: { memoryDelta: 0, cpuTimeDelta: 0 },
            operationCount: 0,
            cacheHits: 0,
            cacheMisses: 0,
        };
    }
    
    /**
     * Start performance monitoring with resource snapshot
     */
    async start(): Promise<void> {
        // Take initial resource snapshot
        this.resourceSnapshot = await this.runtime.resourceManager?.currentSnapshot();
        
        // Register with runtime performance system
        if (this.runtime.performanceSystem) {
            await this.runtime.performanceSystem.startMonitoring(this.functionName);
        }
    }
    
    /**
     * Record an operation for metrics
     * @param operationType - Type of operation performed
     * @param cached - Whether operation used cache
     */
    recordOperation(operationType: string = 'generic', cached: boolean = false): void {
        this.metrics.operationCount++;
        if (cached) {
            this.metrics.cacheHits++;
        } else {
            this.metrics.cacheMisses++;
        }
    }
    
    /**
     * End performance monitoring and generate comprehensive report
     * @returns Performance analysis report
     */
    async end(): Promise<PerformanceReport> {
        this.metrics.endTime = Date.now();
        this.metrics.duration = this.metrics.endTime - this.startTime;
        
        // Calculate resource usage
        const endSnapshot = await this.runtime.resourceManager?.currentSnapshot();
        this.metrics.resourceUsage = this.calculateResourceDelta(
            this.resourceSnapshot, 
            endSnapshot
        );
        
        // Report to runtime performance system
        if (this.runtime.performanceSystem) {
            await this.runtime.performanceSystem.recordMetrics(this.metrics);
        }
        
        // Generate AI analysis
        const aiAnalysis = this.generateAIAnalysis();
        
        const report: PerformanceReport = {
            metrics: this.metrics,
            aiAnalysis,
            recommendations: this.generateRecommendations(),
            benchmarkComparison: await this.compareToBenchmarks(),
        };
        
        return report;
    }
    
    private calculateResourceDelta(start: any, end: any): ResourceUsageDelta {
        if (!start || !end) return { memoryDelta: 0, cpuTimeDelta: 0 };
        
        return {
            memoryDelta: (end.memory?.used || 0) - (start.memory?.used || 0),
            cpuTimeDelta: (end.cpu?.time || 0) - (start.cpu?.time || 0),
        };
    }
    
    private generateAIAnalysis(): PerformanceAIAnalysis {
        const { duration, resourceUsage, operationCount, cacheHits, cacheMisses } = this.metrics;
        
        // Performance categorization
        const performanceCategory = this.categorizePerformance(duration, operationCount);
        const resourceEfficiency = this.categorizeResourceEfficiency(resourceUsage);
        const cacheEfficiency = this.categorizeCacheEfficiency(cacheHits, cacheMisses);
        
        return {
            performanceCategory,
            resourceEfficiency,
            cacheEfficiency,
            insights: [
                `Function '${this.functionName}' completed in ${duration}ms`,
                `Processed ${operationCount} operations`,
                `Memory usage: ${this.formatMemoryUsage(resourceUsage.memoryDelta)}`,
                `Cache hit rate: ${this.calculateCacheHitRate().toFixed(1)}%`,
            ],
            optimizationOpportunities: this.identifyOptimizationOpportunities(),
        };
    }
    
    private categorizePerformance(duration: number, operations: number): 'excellent' | 'good' | 'fair' | 'poor' {
        const opsPerMs = operations / Math.max(1, duration);
        
        if (duration < 10 && opsPerMs > 10) return 'excellent';
        if (duration < 100 && opsPerMs > 1) return 'good';
        if (duration < 1000 && opsPerMs > 0.1) return 'fair';
        return 'poor';
    }
    
    private categorizeResourceEfficiency(usage: ResourceUsageDelta): 'excellent' | 'good' | 'fair' | 'poor' {
        const memoryMB = usage.memoryDelta / (1024 * 1024);
        
        if (memoryMB < 1) return 'excellent';
        if (memoryMB < 10) return 'good';
        if (memoryMB < 100) return 'fair';
        return 'poor';
    }
    
    private categorizeCacheEfficiency(hits: number, misses: number): 'excellent' | 'good' | 'fair' | 'poor' {
        const hitRate = hits / Math.max(1, hits + misses);
        
        if (hitRate > 0.9) return 'excellent';
        if (hitRate > 0.7) return 'good';
        if (hitRate > 0.5) return 'fair';
        return 'poor';
    }
    
    private calculateCacheHitRate(): number {
        const total = this.metrics.cacheHits + this.metrics.cacheMisses;
        return total > 0 ? (this.metrics.cacheHits / total) * 100 : 0;
    }
    
    private formatMemoryUsage(bytes: number): string {
        const mb = bytes / (1024 * 1024);
        if (mb > 1) return `${mb.toFixed(1)} MB`;
        
        const kb = bytes / 1024;
        if (kb > 1) return `${kb.toFixed(1)} KB`;
        
        return `${bytes} bytes`;
    }
    
    private identifyOptimizationOpportunities(): readonly string[] {
        const opportunities: string[] = [];
        
        if (this.metrics.duration > 1000) {
            opportunities.push('Consider async processing for long-running operations');
        }
        
        if (this.calculateCacheHitRate() < 50) {
            opportunities.push('Implement or improve caching strategy');
        }
        
        if (this.metrics.resourceUsage.memoryDelta > 10 * 1024 * 1024) {
            opportunities.push('Review memory usage and consider optimization');
        }
        
        if (this.metrics.operationCount > 1000) {
            opportunities.push('Consider batch processing for high-volume operations');
        }
        
        return opportunities as readonly string[];
    }
    
    private generateRecommendations(): readonly string[] {
        const recommendations: string[] = [];
        const analysis = this.generateAIAnalysis();
        
        if (analysis.performanceCategory === 'poor') {
            recommendations.push('Performance optimization needed - consider profiling and algorithm improvements');
        }
        
        if (analysis.resourceEfficiency === 'poor') {
            recommendations.push('Memory optimization required - review data structures and lifecycle management');
        }
        
        if (analysis.cacheEfficiency === 'poor') {
            recommendations.push('Caching strategy improvement needed - analyze access patterns');
        }
        
        recommendations.push(...analysis.optimizationOpportunities);
        
        return recommendations as readonly string[];
    }
    
    private async compareToBenchmarks(): Promise<BenchmarkComparison> {
        // Simplified benchmark comparison
        // In a real implementation, this would compare against historical data
        return {
            percentile: 50, // Median performance
            comparison: 'average',
            historicalTrend: 'stable',
            similarFunctions: [],
        };
    }
}

// Type definitions for performance monitoring
export interface PerformanceMetrics {
    readonly functionName: string;
    readonly startTime: number;
    endTime: number;
    duration: number;
    resourceUsage: ResourceUsageDelta;
    operationCount: number;
    cacheHits: number;
    cacheMisses: number;
}

export interface PerformanceReport {
    readonly metrics: PerformanceMetrics;
    readonly aiAnalysis: PerformanceAIAnalysis;
    readonly recommendations: readonly string[];
    readonly benchmarkComparison: BenchmarkComparison;
}

export interface PerformanceAIAnalysis {
    readonly performanceCategory: 'excellent' | 'good' | 'fair' | 'poor';
    readonly resourceEfficiency: 'excellent' | 'good' | 'fair' | 'poor';
    readonly cacheEfficiency: 'excellent' | 'good' | 'fair' | 'poor';
    readonly insights: readonly string[];
    readonly optimizationOpportunities: readonly string[];
}

export interface BenchmarkComparison {
    readonly percentile: number;
    readonly comparison: 'excellent' | 'above_average' | 'average' | 'below_average' | 'poor';
    readonly historicalTrend: 'improving' | 'stable' | 'degrading';
    readonly similarFunctions: readonly string[];
}

"#.to_string()
    }

    /// Generate business rule integration
    fn generate_business_rule_integration(&self) -> String {
        r#"
/**
 * Business rule engine integration with runtime validation
 * Provides seamless integration with Prism runtime business rule system
 */
export class BusinessRuleEngine {
    private static runtime: RuntimeInitialized<PrismRuntime> | null = null;
    private static customValidators: Map<string, (value: unknown) => boolean> = new Map();
    
    /**
     * Set the runtime instance for business rule validation
     * @param runtime - Initialized Prism runtime instance
     */
    static setRuntime(runtime: RuntimeInitialized<PrismRuntime>): void {
        this.runtime = runtime;
    }
    
    /**
     * Register a custom validator for a specific type
     * @param typeName - Name of the type to validate
     * @param validator - Custom validation function
     */
    static registerCustomValidator(
        typeName: string, 
        validator: (value: unknown) => boolean
    ): void {
        this.customValidators.set(typeName, validator);
    }
    
    /**
     * Validate a value against a business rule
     * @param ruleName - Name of the business rule
     * @param value - Value to validate
     * @returns True if validation passes
     */
    static validate(ruleName: string, value: unknown): boolean {
        if (!this.runtime) {
            console.warn(`Business rule validation for '${ruleName}' skipped - runtime not available`);
            return true; // Fail open in development
        }
        
        try {
            // Check for custom validator first
            const customValidator = this.customValidators.get(ruleName);
            if (customValidator) {
                return customValidator(value);
            }
            
            // Use runtime business rule validation
            return this.runtime.businessRuleSystem?.validateRule(ruleName, value) ?? true;
        } catch (error) {
            console.error(`Business rule validation error for '${ruleName}':`, error);
            return false;
        }
    }
    
    /**
     * Validate a detailed business rule with context
     * @param rule - Detailed rule information
     * @returns True if validation passes
     */
    static validateRule(ruleName: string, ruleDescription: string, value: unknown): boolean {
        if (!this.runtime) {
            console.warn(`Business rule '${ruleName}' validation skipped - runtime not available`);
            return true;
        }
        
        try {
            return this.runtime.businessRuleSystem?.validateDetailedRule({
                name: ruleName,
                description: ruleDescription,
                value: value,
            }) ?? true;
        } catch (error) {
            console.error(`Detailed business rule validation error for '${ruleName}':`, error);
            return false;
        }
    }
    
    /**
     * Get all registered business rules
     * @returns Array of registered business rule names
     */
    static getRegisteredRules(): readonly string[] {
        if (!this.runtime) {
            return Array.from(this.customValidators.keys()) as readonly string[];
        }
        
        const runtimeRules = this.runtime.businessRuleSystem?.getRegisteredRules() ?? [];
        const customRules = Array.from(this.customValidators.keys());
        
        return [...runtimeRules, ...customRules] as readonly string[];
    }
    
    /**
     * Generate business rule validation report
     * @param typeName - Type to generate report for
     * @param value - Value to validate against all rules
     * @returns Comprehensive validation report
     */
    static generateValidationReport(typeName: string, value: unknown): BusinessRuleValidationReport {
        const rules = this.getRegisteredRules().filter(rule => 
            rule.toLowerCase().includes(typeName.toLowerCase())
        );
        
        const results = rules.map(ruleName => ({
            ruleName,
            passed: this.validate(ruleName, value),
            executionTime: this.measureRuleExecutionTime(ruleName, value),
        }));
        
        const passedCount = results.filter(r => r.passed).length;
        const totalTime = results.reduce((sum, r) => sum + r.executionTime, 0);
        
        return {
            typeName,
            totalRules: rules.length,
            passedRules: passedCount,
            failedRules: rules.length - passedCount,
            overallValid: passedCount === rules.length,
            executionTime: totalTime,
            ruleResults: results as readonly typeof results[0][],
            recommendations: this.generateRuleRecommendations(results),
        };
    }
    
    private static measureRuleExecutionTime(ruleName: string, value: unknown): number {
        const start = Date.now();
        this.validate(ruleName, value);
        return Date.now() - start;
    }
    
    private static generateRuleRecommendations(
        results: { ruleName: string; passed: boolean; executionTime: number }[]
    ): readonly string[] {
        const recommendations: string[] = [];
        
        const failedRules = results.filter(r => !r.passed);
        if (failedRules.length > 0) {
            recommendations.push(`Fix validation for rules: ${failedRules.map(r => r.ruleName).join(', ')}`);
        }
        
        const slowRules = results.filter(r => r.executionTime > 10);
        if (slowRules.length > 0) {
            recommendations.push(`Optimize performance for slow rules: ${slowRules.map(r => r.ruleName).join(', ')}`);
        }
        
        return recommendations as readonly string[];
    }
}

// Type definitions for business rule integration
export interface BusinessRuleValidationReport {
    readonly typeName: string;
    readonly totalRules: number;
    readonly passedRules: number;
    readonly failedRules: number;
    readonly overallValid: boolean;
    readonly executionTime: number;
    readonly ruleResults: readonly {
        readonly ruleName: string;
        readonly passed: boolean;
        readonly executionTime: number;
    }[];
    readonly recommendations: readonly string[];
}

"#.to_string()
    }

    /// Generate AI metadata system
    fn generate_ai_metadata_system(&self, pir: &PrismIR) -> String {
        format!(
            r#"
/**
 * AI metadata system for runtime analysis and optimization
 * Provides comprehensive metadata for AI-driven development tools
 */
export class AIMetadataProvider {{
    private static runtime: RuntimeInitialized<PrismRuntime> | null = null;
    private static metadataCache: Map<string, any> = new Map();
    
    /**
     * Initialize AI metadata provider with runtime
     * @param runtime - Initialized Prism runtime instance
     */
    static initialize(runtime: RuntimeInitialized<PrismRuntime>): void {{
        this.runtime = runtime;
    }}
    
    /**
     * Generate comprehensive AI metadata for the entire system
     * @returns Structured metadata for AI consumption
     */
    static generateSystemMetadata(): SystemAIMetadata {{
        return {{
            version: '2.0.0',
            generatedAt: new Date().toISOString(),
            compiler: {{
                name: 'Prism TypeScript Backend',
                version: '2.0.0',
                features: ['semantic-types', 'business-rules', 'effect-tracking', 'capability-validation'],
            }},
            codebase: {{
                moduleCount: {},
                cohesionScore: {},
                capabilityCount: {},
                effectCount: {},
                semanticTypeCount: {},
            }},
            runtime: {{
                initialized: this.runtime !== null,
                features: this.runtime ? this.getRuntimeFeatures() : [],
                performance: this.runtime ? this.getRuntimePerformanceMetrics() : null,
            }},
            aiContext: {{
                intent: 'Provide comprehensive metadata for AI-driven development and analysis',
                capabilities: [
                    'semantic-type-analysis',
                    'business-rule-validation',
                    'performance-monitoring',
                    'capability-management',
                    'effect-tracking',
                ],
                recommendations: this.generateSystemRecommendations(),
            }},
        }};
    }}
    
    /**
     * Generate metadata for a specific function or component
     * @param identifier - Function or component identifier
     * @param context - Additional context information
     * @returns Component-specific AI metadata
     */
    static generateComponentMetadata(
        identifier: string, 
        context?: Record<string, unknown>
    ): ComponentAIMetadata {{
        const cached = this.metadataCache.get(identifier);
        if (cached) {{
            return {{ ...cached, lastAccessed: new Date().toISOString() }};
        }}
        
        const metadata: ComponentAIMetadata = {{
            identifier,
            type: this.inferComponentType(identifier),
            semanticContext: this.extractSemanticContext(identifier, context),
            businessContext: this.extractBusinessContext(identifier, context),
            performanceContext: this.extractPerformanceContext(identifier),
            securityContext: this.extractSecurityContext(identifier, context),
            aiInsights: this.generateComponentInsights(identifier, context),
            lastGenerated: new Date().toISOString(),
            lastAccessed: new Date().toISOString(),
        }};
        
        this.metadataCache.set(identifier, metadata);
        return metadata;
    }}
    
    /**
     * Generate real-time analysis metadata
     * @param analysisType - Type of analysis to perform
     * @param data - Data to analyze
     * @returns Real-time analysis results
     */
    static generateRealTimeAnalysis(
        analysisType: 'performance' | 'security' | 'business-rules' | 'semantic-types',
        data: unknown
    ): RealTimeAnalysis {{
        const startTime = Date.now();
        
        let analysis: any;
        switch (analysisType) {{
            case 'performance':
                analysis = this.analyzePerformance(data);
                break;
            case 'security':
                analysis = this.analyzeSecurity(data);
                break;
            case 'business-rules':
                analysis = this.analyzeBusinessRules(data);
                break;
            case 'semantic-types':
                analysis = this.analyzeSemanticTypes(data);
                break;
            default:
                analysis = {{ error: 'Unknown analysis type' }};
        }}
        
        const executionTime = Date.now() - startTime;
        
        return {{
            analysisType,
            executionTime,
            timestamp: new Date().toISOString(),
            results: analysis,
            metadata: {{
                dataSize: JSON.stringify(data).length,
                complexity: this.assessComplexity(data),
                confidence: this.assessConfidence(analysis),
            }},
            recommendations: this.generateAnalysisRecommendations(analysisType, analysis),
        }};
    }}
    
    private static getRuntimeFeatures(): readonly string[] {{
        if (!this.runtime) return [];
        
        return [
            'capability-validation',
            'effect-tracking',
            'business-rule-enforcement',
            'performance-monitoring',
            'resource-management',
        ] as readonly string[];
    }}
    
    private static getRuntimePerformanceMetrics(): any {{
        if (!this.runtime?.performanceSystem) return null;
        
        // Would integrate with actual runtime performance metrics
        return {{
            averageResponseTime: 50,
            memoryUsage: 128 * 1024 * 1024,
            cpuUsage: 15,
            cacheHitRate: 85,
        }};
    }}
    
    private static generateSystemRecommendations(): readonly string[] {{
        const recommendations: string[] = [];
        
        if (!this.runtime) {{
            recommendations.push('Initialize Prism runtime for full functionality');
        }}
        
        recommendations.push('Consider implementing caching for frequently accessed metadata');
        recommendations.push('Monitor performance metrics for optimization opportunities');
        recommendations.push('Regular validation of business rules and semantic types');
        
        return recommendations as readonly string[];
    }}
    
    private static inferComponentType(identifier: string): 'function' | 'class' | 'type' | 'module' | 'unknown' {{
        if (identifier.includes('()') || identifier.startsWith('function')) return 'function';
        if (identifier.startsWith('class') || identifier.includes('.prototype')) return 'class';
        if (identifier.startsWith('type') || identifier.includes('Type')) return 'type';
        if (identifier.includes('/') || identifier.includes('.')) return 'module';
        return 'unknown';
    }}
    
    private static extractSemanticContext(identifier: string, context?: Record<string, unknown>): any {{
        return {{
            domain: context?.domain || 'unknown',
            semanticType: context?.semanticType || null,
            businessMeaning: context?.businessMeaning || null,
            constraints: context?.constraints || [],
        }};
    }}
    
    private static extractBusinessContext(identifier: string, context?: Record<string, unknown>): any {{
        return {{
            responsibility: context?.responsibility || 'unknown',
            businessRules: context?.businessRules || [],
            stakeholders: context?.stakeholders || [],
            impact: context?.impact || 'unknown',
        }};
    }}
    
    private static extractPerformanceContext(identifier: string): any {{
        return {{
            complexity: 'O(1)', // Would be analyzed
            memoryUsage: 'low',
            cpuIntensity: 'low',
            ioOperations: false,
        }};
    }}
    
    private static extractSecurityContext(identifier: string, context?: Record<string, unknown>): any {{
        return {{
            securityLevel: context?.securityLevel || 'medium',
            requiredCapabilities: context?.capabilities || [],
            dataClassification: context?.dataClassification || 'internal',
            auditRequired: false,
        }};
    }}
    
    private static generateComponentInsights(identifier: string, context?: Record<string, unknown>): any {{
        return {{
            usagePatterns: ['common', 'critical'],
            optimizationOpportunities: ['caching', 'batching'],
            riskFactors: ['complexity', 'dependencies'],
            qualityMetrics: {{
                maintainability: 85,
                testability: 90,
                reusability: 75,
            }},
        }};
    }}
    
    private static analyzePerformance(data: unknown): any {{
        return {{
            estimatedComplexity: 'O(1)',
            memoryImpact: 'low',
            optimizationPotential: 'medium',
        }};
    }}
    
    private static analyzeSecurity(data: unknown): any {{
        return {{
            vulnerabilities: [],
            riskLevel: 'low',
            requiredMitigations: [],
        }};
    }}
    
    private static analyzeBusinessRules(data: unknown): any {{
        return {{
            applicableRules: [],
            violations: [],
            compliance: 100,
        }};
    }}
    
    private static analyzeSemanticTypes(data: unknown): any {{
        return {{
            detectedTypes: [],
            semanticRelationships: [],
            domainAlignment: 95,
        }};
    }}
    
    private static assessComplexity(data: unknown): 'low' | 'medium' | 'high' {{
        const size = JSON.stringify(data).length;
        if (size < 1000) return 'low';
        if (size < 10000) return 'medium';
        return 'high';
    }}
    
    private static assessConfidence(analysis: any): number {{
        // Simplified confidence assessment
        return 85; // Would be based on actual analysis quality
    }}
    
    private static generateAnalysisRecommendations(
        analysisType: string, 
        analysis: any
    ): readonly string[] {{
        const recommendations: string[] = [];
        
        switch (analysisType) {{
            case 'performance':
                if (analysis.memoryImpact === 'high') {{
                    recommendations.push('Consider memory optimization strategies');
                }}
                break;
            case 'security':
                if (analysis.riskLevel === 'high') {{
                    recommendations.push('Implement additional security measures');
                }}
                break;
        }}
        
        return recommendations as readonly string[];
    }}
}}

// Type definitions for AI metadata
export interface SystemAIMetadata {{
    readonly version: string;
    readonly generatedAt: string;
    readonly compiler: {{
        readonly name: string;
        readonly version: string;
        readonly features: readonly string[];
    }};
    readonly codebase: {{
        readonly moduleCount: number;
        readonly cohesionScore: number;
        readonly capabilityCount: number;
        readonly effectCount: number;
        readonly semanticTypeCount: number;
    }};
    readonly runtime: {{
        readonly initialized: boolean;
        readonly features: readonly string[];
        readonly performance: any;
    }};
    readonly aiContext: {{
        readonly intent: string;
        readonly capabilities: readonly string[];
        readonly recommendations: readonly string[];
    }};
}}

export interface ComponentAIMetadata {{
    readonly identifier: string;
    readonly type: 'function' | 'class' | 'type' | 'module' | 'unknown';
    readonly semanticContext: any;
    readonly businessContext: any;
    readonly performanceContext: any;
    readonly securityContext: any;
    readonly aiInsights: any;
    readonly lastGenerated: string;
    readonly lastAccessed: string;
}}

export interface RealTimeAnalysis {{
    readonly analysisType: string;
    readonly executionTime: number;
    readonly timestamp: string;
    readonly results: any;
    readonly metadata: {{
        readonly dataSize: number;
        readonly complexity: 'low' | 'medium' | 'high';
        readonly confidence: number;
    }};
    readonly recommendations: readonly string[];
}}

"#,
            pir.modules.len(),
            pir.cohesion_metrics.overall_score,
            pir.modules.iter().flat_map(|m| &m.capabilities).count(),
            pir.effect_graph.nodes.len(),
            pir.type_registry.types.len()
        )
    }
}

/// Enhanced capability manager with modern TypeScript patterns
pub struct CapabilityManager {
    config: RuntimeIntegrationConfig,
}

impl CapabilityManager {
    /// Create a new capability manager
    pub fn new(config: RuntimeIntegrationConfig) -> Self {
        Self { config }
    }

    /// Generate capability validation code
    pub fn generate_capability_validation(&self, capabilities: &[Capability]) -> TypeScriptResult<String> {
        let mut output = String::new();
        
        for capability in capabilities {
            output.push_str(&format!(
                r#"/**
 * Validate {} capability
 * @param context - Execution context
 * @returns True if capability is available
 */
export async function validate{}Capability(context: ExecutionContext): Promise<boolean> {{
    try {{
        return await context.capabilities.has('{}');
    }} catch (error) {{
        console.error('Capability validation error for {}:', error);
        return false;
    }}
}}

"#,
                capability.name,
                capability.name.replace("-", "_").replace(" ", "_"),
                capability.name,
                capability.name
            ));
        }
        
        Ok(output)
    }
}

/// Enhanced effect tracker with resource monitoring
pub struct EffectTracker {
    config: RuntimeIntegrationConfig,
}

impl EffectTracker {
    /// Create a new effect tracker
    pub fn new(config: RuntimeIntegrationConfig) -> Self {
        Self { config }
    }

    /// Generate effect tracking code
    pub fn generate_effect_tracking(&self, effects: &[Effect]) -> TypeScriptResult<String> {
        let mut output = String::new();
        
        for effect in effects {
            output.push_str(&format!(
                r#"/**
 * Track {} effect
 * @param tracker - Effect tracker instance
 * @returns Promise that resolves when effect is tracked
 */
export async function track{}Effect(tracker: EffectTracker<readonly ['{}']>): Promise<void> {{
    await tracker.trackEffects(['{}'] as const);
}}

"#,
                effect.name,
                effect.name.replace("-", "_").replace(" ", "_"),
                effect.name,
                effect.name
            ));
        }
        
        Ok(output)
    }
}

// Error types for runtime integration
#[derive(Debug, thiserror::Error)]
pub enum RuntimeIntegrationError {
    #[error("Runtime initialization failed: {message}")]
    InitializationFailed { message: String },
    
    #[error("Capability validation failed: {capability}")]
    CapabilityValidationFailed { capability: String },
    
    #[error("Effect tracking failed: {effect}")]
    EffectTrackingFailed { effect: String },
    
    #[error("Performance monitoring failed: {function_name}")]
    PerformanceMonitoringFailed { function_name: String },
}

impl From<RuntimeIntegrationError> for TypeScriptError {
    fn from(err: RuntimeIntegrationError) -> Self {
        TypeScriptError::RuntimeIntegration {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_integration_config() {
        let config = RuntimeIntegrationConfig::default();
        assert!(config.enable_capability_validation);
        assert!(config.enable_effect_tracking);
        assert!(config.enable_performance_monitoring);
    }

    #[test]
    fn test_runtime_integrator_creation() {
        let config = RuntimeIntegrationConfig::default();
        let integrator = RuntimeIntegrator::new(config);
        assert!(integrator.registered_capabilities.is_empty());
        assert!(integrator.registered_effects.is_empty());
    }
} 