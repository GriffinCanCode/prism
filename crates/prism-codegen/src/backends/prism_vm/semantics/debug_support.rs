//! Semantic Debugging and Introspection Support
//!
//! This module provides debugging capabilities for semantic information in the VM,
//! enabling developers to inspect and understand the semantic aspects of their code
//! at runtime.

use super::{VMSemanticConfig, SymbolFromStr};
use prism_semantic::{
    SemanticEngine, SemanticResult,
    types::{SemanticType, BusinessRule, TypeConstraint},
    database::SemanticInfo,
    analyzer::{SymbolInfo, TypeInfo},
};
use prism_vm::{
    VMResult, PrismVMError,
    bytecode::{
        BytecodeSemanticMetadata, CompiledBusinessRule, CompiledValidationPredicate,
        SemanticInformationRegistry, FunctionDefinition,
    },
    execution::{ExecutionStack, StackValue, Interpreter},
};
use prism_common::{NodeId, symbol::Symbol};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tracing::{debug, span, Level};

/// Semantic debugger for runtime introspection
pub struct SemanticDebugger {
    /// Semantic engine for analysis
    semantic_engine: SemanticEngine,
    
    /// Semantic registry for metadata access
    semantic_registry: SemanticInformationRegistry,
    
    /// Debug configuration
    config: DebugConfig,
    
    /// Active debug sessions
    debug_sessions: HashMap<String, DebugSession>,
    
    /// Breakpoints on semantic events
    semantic_breakpoints: Vec<SemanticBreakpoint>,
    
    /// Debug statistics
    stats: DebugStats,
}

/// Configuration for semantic debugging
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Enable semantic debugging
    pub enable_debugging: bool,
    
    /// Enable semantic breakpoints
    pub enable_breakpoints: bool,
    
    /// Enable semantic tracing
    pub enable_tracing: bool,
    
    /// Maximum debug session duration
    pub max_session_duration_ms: u64,
    
    /// Enable debug logging
    pub enable_logging: bool,
    
    /// Debug output format
    pub output_format: DebugOutputFormat,
}

/// Debug output format options
#[derive(Debug, Clone)]
pub enum DebugOutputFormat {
    /// Human-readable text format
    Text,
    /// JSON format for tooling
    Json,
    /// Structured format for IDEs
    Structured,
}

/// Debug session for tracking semantic debugging state
#[derive(Debug)]
pub struct DebugSession {
    /// Session ID
    pub id: String,
    
    /// Session start time
    pub start_time: std::time::Instant,
    
    /// Current function being debugged
    pub current_function: Option<u32>,
    
    /// Current instruction pointer
    pub current_instruction: Option<u32>,
    
    /// Semantic context at current location
    pub semantic_context: SemanticContext,
    
    /// Debug event history
    pub event_history: Vec<DebugEvent>,
    
    /// Active watches on semantic values
    pub semantic_watches: Vec<SemanticWatch>,
}

/// Semantic context information for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    /// Current type information
    pub current_types: HashMap<String, SemanticType>,
    
    /// Active business rules
    pub active_business_rules: Vec<String>,
    
    /// Validation state
    pub validation_state: ValidationState,
    
    /// Semantic constraints in effect
    pub active_constraints: Vec<TypeConstraint>,
    
    /// AI metadata context
    pub ai_context: HashMap<String, String>,
}

/// Current validation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationState {
    /// Recently validated values
    pub recent_validations: Vec<ValidationEvent>,
    
    /// Failed validations
    pub failed_validations: Vec<ValidationFailure>,
    
    /// Validation performance metrics
    pub validation_metrics: ValidationMetrics,
}

/// Validation event for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEvent {
    /// Timestamp of validation
    pub timestamp: u64,
    
    /// Rule that was validated
    pub rule_id: String,
    
    /// Value that was validated
    pub validated_value: String,
    
    /// Validation result
    pub result: bool,
    
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// Validation failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFailure {
    /// When the failure occurred
    pub timestamp: u64,
    
    /// Rule that failed
    pub rule_id: String,
    
    /// Value that failed validation
    pub failed_value: String,
    
    /// Error message
    pub error_message: String,
    
    /// Stack trace at failure point
    pub stack_trace: Vec<String>,
}

/// Validation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Total validations performed
    pub total_validations: u64,
    
    /// Average validation time
    pub avg_validation_time_us: f64,
    
    /// Maximum validation time
    pub max_validation_time_us: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Debug event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebugEvent {
    /// Semantic breakpoint hit
    BreakpointHit {
        breakpoint_id: String,
        location: DebugLocation,
        context: SemanticContext,
    },
    
    /// Business rule execution
    BusinessRuleExecution {
        rule_id: String,
        input_value: String,
        result: bool,
        execution_time_us: u64,
    },
    
    /// Type inference event
    TypeInference {
        location: DebugLocation,
        inferred_type: String,
        confidence: f64,
    },
    
    /// Semantic validation event
    SemanticValidation {
        type_id: u32,
        validation_result: bool,
        error_message: Option<String>,
    },
    
    /// AI metadata access
    AIMetadataAccess {
        metadata_type: String,
        accessed_data: String,
    },
}

/// Debug location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugLocation {
    /// Function ID
    pub function_id: u32,
    
    /// Instruction offset
    pub instruction_offset: u32,
    
    /// Source location if available
    pub source_location: Option<SourceLocation>,
}

/// Source location for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path
    pub file_path: String,
    
    /// Line number
    pub line: u32,
    
    /// Column number
    pub column: u32,
}

/// Semantic breakpoint configuration
#[derive(Debug, Clone)]
pub struct SemanticBreakpoint {
    /// Breakpoint ID
    pub id: String,
    
    /// Breakpoint type
    pub breakpoint_type: BreakpointType,
    
    /// Condition for triggering breakpoint
    pub condition: BreakpointCondition,
    
    /// Whether breakpoint is enabled
    pub enabled: bool,
    
    /// Hit count
    pub hit_count: u64,
}

/// Types of semantic breakpoints
#[derive(Debug, Clone)]
pub enum BreakpointType {
    /// Break on business rule execution
    BusinessRule { rule_id: String },
    
    /// Break on validation failure
    ValidationFailure { type_id: Option<u32> },
    
    /// Break on type inference
    TypeInference { location: DebugLocation },
    
    /// Break on semantic constraint violation
    ConstraintViolation { constraint_type: String },
    
    /// Break on AI metadata access
    AIMetadataAccess { metadata_type: String },
}

/// Conditions for breakpoint triggering
#[derive(Debug, Clone)]
pub enum BreakpointCondition {
    /// Always trigger
    Always,
    
    /// Trigger on specific value
    ValueEquals { expected_value: String },
    
    /// Trigger on condition expression
    Expression { expression: String },
    
    /// Trigger after N hits
    HitCount { count: u64 },
}

/// Semantic value watch for debugging
#[derive(Debug, Clone)]
pub struct SemanticWatch {
    /// Watch ID
    pub id: String,
    
    /// Expression to watch
    pub expression: String,
    
    /// Current value
    pub current_value: Option<String>,
    
    /// Previous value
    pub previous_value: Option<String>,
    
    /// Whether value has changed
    pub has_changed: bool,
    
    /// Watch type
    pub watch_type: WatchType,
}

/// Types of semantic watches
#[derive(Debug, Clone)]
pub enum WatchType {
    /// Watch a semantic type
    SemanticType { type_id: u32 },
    
    /// Watch business rule state
    BusinessRule { rule_id: String },
    
    /// Watch validation state
    ValidationState { type_id: u32 },
    
    /// Watch AI metadata
    AIMetadata { metadata_key: String },
}

/// Debug statistics
#[derive(Debug, Default)]
pub struct DebugStats {
    /// Total debug sessions created
    pub total_sessions: u64,
    
    /// Active debug sessions
    pub active_sessions: u64,
    
    /// Total breakpoints hit
    pub total_breakpoint_hits: u64,
    
    /// Total debug events recorded
    pub total_debug_events: u64,
    
    /// Total debug time spent
    pub total_debug_time_ms: u64,
}

impl SemanticDebugger {
    /// Create a new semantic debugger
    pub fn new(
        semantic_engine: SemanticEngine,
        semantic_registry: SemanticInformationRegistry,
        config: DebugConfig,
    ) -> Self {
        Self {
            semantic_engine,
            semantic_registry,
            config,
            debug_sessions: HashMap::new(),
            semantic_breakpoints: Vec::new(),
            stats: DebugStats::default(),
        }
    }
    
    /// Start a new debug session
    pub fn start_debug_session(&mut self, session_id: String, function_id: u32) -> VMResult<()> {
        if !self.config.enable_debugging {
            return Ok(());
        }
        
        let _span = span!(Level::DEBUG, "start_debug_session", session_id = %session_id).entered();
        
        let session = DebugSession {
            id: session_id.clone(),
            start_time: std::time::Instant::now(),
            current_function: Some(function_id),
            current_instruction: None,
            semantic_context: self.build_semantic_context(function_id)?,
            event_history: Vec::new(),
            semantic_watches: Vec::new(),
        };
        
        self.debug_sessions.insert(session_id, session);
        self.stats.total_sessions += 1;
        self.stats.active_sessions += 1;
        
        Ok(())
    }
    
    /// End a debug session
    pub fn end_debug_session(&mut self, session_id: &str) -> VMResult<DebugSessionSummary> {
        let _span = span!(Level::DEBUG, "end_debug_session", session_id = session_id).entered();
        
        if let Some(session) = self.debug_sessions.remove(session_id) {
            self.stats.active_sessions -= 1;
            
            let duration = session.start_time.elapsed();
            self.stats.total_debug_time_ms += duration.as_millis() as u64;
            
            Ok(DebugSessionSummary {
                session_id: session.id,
                duration_ms: duration.as_millis() as u64,
                total_events: session.event_history.len(),
                breakpoints_hit: session.event_history.iter()
                    .filter(|e| matches!(e, DebugEvent::BreakpointHit { .. }))
                    .count(),
                validations_performed: session.event_history.iter()
                    .filter(|e| matches!(e, DebugEvent::SemanticValidation { .. }))
                    .count(),
            })
        } else {
            Err(PrismVMError::RuntimeError(format!("Debug session not found: {}", session_id)))
        }
    }
    
    /// Add a semantic breakpoint
    pub fn add_breakpoint(&mut self, breakpoint: SemanticBreakpoint) -> VMResult<()> {
        if !self.config.enable_breakpoints {
            return Ok(());
        }
        
        debug!("Adding semantic breakpoint: {:?}", breakpoint);
        self.semantic_breakpoints.push(breakpoint);
        Ok(())
    }
    
    /// Remove a semantic breakpoint
    pub fn remove_breakpoint(&mut self, breakpoint_id: &str) -> VMResult<bool> {
        let initial_len = self.semantic_breakpoints.len();
        self.semantic_breakpoints.retain(|bp| bp.id != breakpoint_id);
        Ok(self.semantic_breakpoints.len() != initial_len)
    }
    
    /// Add a semantic watch
    pub fn add_watch(&mut self, session_id: &str, watch: SemanticWatch) -> VMResult<()> {
        if let Some(session) = self.debug_sessions.get_mut(session_id) {
            debug!("Adding semantic watch: {:?}", watch);
            session.semantic_watches.push(watch);
            Ok(())
        } else {
            Err(PrismVMError::RuntimeError(format!("Debug session not found: {}", session_id)))
        }
    }
    
    /// Check for breakpoint hits during execution
    pub fn check_breakpoints(
        &mut self,
        session_id: &str,
        location: DebugLocation,
        context: &SemanticContext,
    ) -> VMResult<Option<String>> {
        if !self.config.enable_breakpoints {
            return Ok(None);
        }
        
        for breakpoint in &mut self.semantic_breakpoints {
            if !breakpoint.enabled {
                continue;
            }
            
            if self.should_trigger_breakpoint(breakpoint, &location, context)? {
                breakpoint.hit_count += 1;
                self.stats.total_breakpoint_hits += 1;
                
                // Record the breakpoint hit event
                let event = DebugEvent::BreakpointHit {
                    breakpoint_id: breakpoint.id.clone(),
                    location: location.clone(),
                    context: context.clone(),
                };
                
                if let Some(session) = self.debug_sessions.get_mut(session_id) {
                    session.event_history.push(event);
                }
                
                return Ok(Some(breakpoint.id.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Record a debug event
    pub fn record_event(&mut self, session_id: &str, event: DebugEvent) -> VMResult<()> {
        if let Some(session) = self.debug_sessions.get_mut(session_id) {
            session.event_history.push(event);
            self.stats.total_debug_events += 1;
        }
        Ok(())
    }
    
    /// Get current semantic context for debugging
    pub fn get_semantic_context(&self, session_id: &str) -> VMResult<SemanticContext> {
        if let Some(session) = self.debug_sessions.get(session_id) {
            Ok(session.semantic_context.clone())
        } else {
            Err(PrismVMError::RuntimeError(format!("Debug session not found: {}", session_id)))
        }
    }
    
    /// Export debug information
    pub fn export_debug_info(&self, session_id: &str) -> VMResult<String> {
        if let Some(session) = self.debug_sessions.get(session_id) {
            match self.config.output_format {
                DebugOutputFormat::Json => {
                    serde_json::to_string_pretty(session)
                        .map_err(|e| PrismVMError::RuntimeError(format!("JSON export error: {}", e)))
                }
                DebugOutputFormat::Text => {
                    Ok(self.format_debug_info_as_text(session))
                }
                DebugOutputFormat::Structured => {
                    Ok(self.format_debug_info_as_structured(session))
                }
            }
        } else {
            Err(PrismVMError::RuntimeError(format!("Debug session not found: {}", session_id)))
        }
    }
    
    /// Get debug statistics
    pub fn get_stats(&self) -> &DebugStats {
        &self.stats
    }
    
    // Private helper methods
    
    fn build_semantic_context(&self, function_id: u32) -> VMResult<SemanticContext> {
        // Build semantic context using prism-semantic
        let current_types = self.get_function_types(function_id)?;
        let active_business_rules = self.get_active_business_rules(function_id)?;
        let validation_state = self.get_validation_state(function_id)?;
        let active_constraints = self.get_active_constraints(function_id)?;
        let ai_context = self.get_ai_context(function_id)?;
        
        Ok(SemanticContext {
            current_types,
            active_business_rules,
            validation_state,
            active_constraints,
            ai_context,
        })
    }
    
    fn get_function_types(&self, function_id: u32) -> VMResult<HashMap<String, SemanticType>> {
        // Use prism-semantic to get type information for the function
        let mut types = HashMap::new();
        
        // Query semantic engine for function type information
        if let Ok(function_info) = self.semantic_engine.get_function_info(function_id) {
            for (name, type_info) in function_info.local_types() {
                types.insert(name.clone(), type_info.semantic_type().clone());
            }
        }
        
        Ok(types)
    }
    
    fn get_active_business_rules(&self, function_id: u32) -> VMResult<Vec<String>> {
        // Get active business rules from semantic registry
        let mut rules = Vec::new();
        
        if let Some(function_metadata) = self.semantic_registry.get_function_metadata(function_id) {
            // Extract business rule IDs from function metadata
            // This would need to be implemented based on actual metadata structure
            rules.push(format!("function_{}_rules", function_id));
        }
        
        Ok(rules)
    }
    
    fn get_validation_state(&self, function_id: u32) -> VMResult<ValidationState> {
        // Build validation state from recent validation events
        Ok(ValidationState {
            recent_validations: Vec::new(),
            failed_validations: Vec::new(),
            validation_metrics: ValidationMetrics {
                total_validations: 0,
                avg_validation_time_us: 0.0,
                max_validation_time_us: 0,
                cache_hit_rate: 0.0,
            },
        })
    }
    
    fn get_active_constraints(&self, function_id: u32) -> VMResult<Vec<TypeConstraint>> {
        // Get active type constraints from semantic engine
        if let Ok(constraints) = self.semantic_engine.get_function_constraints(function_id) {
            Ok(constraints)
        } else {
            Ok(Vec::new())
        }
    }
    
    fn get_ai_context(&self, function_id: u32) -> VMResult<HashMap<String, String>> {
        // Get AI metadata context
        let mut context = HashMap::new();
        
        if let Some(function_metadata) = self.semantic_registry.get_function_metadata(function_id) {
            context.insert("purpose".to_string(), function_metadata.ai_context.purpose.clone());
            for (i, usage) in function_metadata.ai_context.usage_context.iter().enumerate() {
                context.insert(format!("usage_{}", i), usage.clone());
            }
        }
        
        Ok(context)
    }
    
    fn should_trigger_breakpoint(
        &self,
        breakpoint: &SemanticBreakpoint,
        location: &DebugLocation,
        context: &SemanticContext,
    ) -> VMResult<bool> {
        // Check if breakpoint condition is met
        match &breakpoint.condition {
            BreakpointCondition::Always => Ok(true),
            BreakpointCondition::ValueEquals { expected_value } => {
                // Check if any value in context matches expected value
                Ok(self.context_contains_value(context, expected_value))
            }
            BreakpointCondition::Expression { expression } => {
                // Evaluate expression against context
                self.evaluate_breakpoint_expression(expression, context)
            }
            BreakpointCondition::HitCount { count } => {
                Ok(breakpoint.hit_count >= *count)
            }
        }
    }
    
    fn context_contains_value(&self, context: &SemanticContext, expected_value: &str) -> bool {
        // Simple value matching in context
        context.ai_context.values().any(|v| v == expected_value) ||
        context.active_business_rules.iter().any(|r| r == expected_value)
    }
    
    fn evaluate_breakpoint_expression(&self, expression: &str, context: &SemanticContext) -> VMResult<bool> {
        // Simple expression evaluation
        // In a real implementation, this would use a proper expression evaluator
        match expression {
            "has_failed_validation" => Ok(!context.validation_state.failed_validations.is_empty()),
            "has_business_rules" => Ok(!context.active_business_rules.is_empty()),
            _ => Ok(false),
        }
    }
    
    fn format_debug_info_as_text(&self, session: &DebugSession) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("Debug Session: {}\n", session.id));
        output.push_str(&format!("Duration: {:?}\n", session.start_time.elapsed()));
        output.push_str(&format!("Events: {}\n", session.event_history.len()));
        output.push_str(&format!("Watches: {}\n", session.semantic_watches.len()));
        
        output.push_str("\nSemantic Context:\n");
        output.push_str(&format!("  Types: {}\n", session.semantic_context.current_types.len()));
        output.push_str(&format!("  Business Rules: {:?}\n", session.semantic_context.active_business_rules));
        
        output.push_str("\nRecent Events:\n");
        for (i, event) in session.event_history.iter().rev().take(10).enumerate() {
            output.push_str(&format!("  {}: {:?}\n", i + 1, event));
        }
        
        output
    }
    
    fn format_debug_info_as_structured(&self, session: &DebugSession) -> String {
        // Format as structured data for IDE integration
        format!(
            "Session: {}\nDuration: {:?}\nEvents: {}\nContext: {:?}",
            session.id,
            session.start_time.elapsed(),
            session.event_history.len(),
            session.semantic_context
        )
    }
}

/// Summary of a completed debug session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSessionSummary {
    /// Session ID
    pub session_id: String,
    
    /// Session duration in milliseconds
    pub duration_ms: u64,
    
    /// Total debug events recorded
    pub total_events: usize,
    
    /// Number of breakpoints hit
    pub breakpoints_hit: usize,
    
    /// Number of validations performed
    pub validations_performed: usize,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_debugging: true,
            enable_breakpoints: true,
            enable_tracing: true,
            max_session_duration_ms: 300_000, // 5 minutes
            enable_logging: true,
            output_format: DebugOutputFormat::Text,
        }
    }
} 