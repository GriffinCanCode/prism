//! Execution Stack
//!
//! This module implements the execution stack for the Prism VM,
//! including stack frames and value management with advanced features
//! like frame pooling, gradual degradation, and predictive overflow handling.

use crate::{VMResult, PrismVMError, bytecode::constants::Constant};
use prism_pir::{Effect, Capability};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::{HashMap, VecDeque};
use std::time::{Instant, Duration};
use tracing::{debug, info, warn, span, Level};

/// Maximum stack size (configurable)
pub const DEFAULT_MAX_STACK_SIZE: usize = 1024 * 1024; // 1MB

/// Stack frame pool for efficient allocation
#[derive(Debug)]
pub struct StackFramePool {
    /// Available frames for reuse
    available_frames: Vec<StackFrame>,
    /// Total allocated frame count
    allocated_count: usize,
    /// Maximum pool size
    max_pool_size: usize,
    /// Pool hit statistics
    pool_hits: u64,
    /// Pool misses
    pool_misses: u64,
}

impl StackFramePool {
    /// Create a new frame pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            available_frames: Vec::with_capacity(max_pool_size),
            allocated_count: 0,
            max_pool_size,
            pool_hits: 0,
            pool_misses: 0,
        }
    }

    /// Get a frame from the pool or create a new one
    pub fn get_frame(&mut self, function_name: String, function_id: u32, return_address: u32) -> StackFrame {
        if let Some(mut frame) = self.available_frames.pop() {
            // Reuse existing frame
            frame.function_name = function_name;
            frame.function_id = function_id;
            frame.return_address = return_address;
            frame.locals.clear();
            frame.upvalues.clear();
            frame.exception_handlers.clear();
            frame.active_effects.clear();
            frame.capabilities.clear();
            self.pool_hits += 1;
            frame
        } else {
            // Create new frame
            self.allocated_count += 1;
            self.pool_misses += 1;
            StackFrame::new(function_name, function_id, return_address)
        }
    }

    /// Return a frame to the pool
    pub fn return_frame(&mut self, frame: StackFrame) {
        if self.available_frames.len() < self.max_pool_size {
            self.available_frames.push(frame);
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> FramePoolStats {
        FramePoolStats {
            pool_size: self.available_frames.len(),
            allocated_count: self.allocated_count,
            hit_rate: if (self.pool_hits + self.pool_misses) > 0 {
                self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
            } else {
                0.0
            },
            total_requests: self.pool_hits + self.pool_misses,
        }
    }
}

/// Frame pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramePoolStats {
    pub pool_size: usize,
    pub allocated_count: usize,
    pub hit_rate: f64,
    pub total_requests: u64,
}

/// Stack operation result with degradation support
#[derive(Debug, Clone)]
pub enum StackOperationResult<T> {
    /// Operation completed successfully
    Success(T),
    /// Operation completed with warning
    SuccessWithWarning(T, StackWarning),
    /// Operation completed in degraded mode
    Degraded(T, DegradationReason),
    /// Operation failed
    Failed(PrismVMError),
}

/// Stack warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StackWarning {
    /// Stack depth approaching limit
    ApproachingDepthLimit { current: usize, limit: usize },
    /// Memory pressure detected
    MemoryPressure { usage_percent: f64 },
    /// Frame pool exhausted
    FramePoolExhausted,
    /// Performance degradation detected
    PerformanceDegradation { latency_ms: f64 },
}

/// Degradation reasons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationReason {
    /// Memory constraints forcing simplified operations
    MemoryConstrained,
    /// Performance constraints requiring optimization
    PerformanceConstrained,
    /// Security constraints limiting functionality
    SecurityConstrained,
    /// Resource exhaustion
    ResourceExhaustion,
}

/// Enhanced execution stack with pooling and degradation support
#[derive(Debug)]
pub struct ExecutionStack {
    /// Stack frames
    frames: Vec<StackFrame>,
    /// Value stack
    values: Vec<StackValue>,
    /// Maximum stack size
    max_size: usize,
    /// Current stack depth
    depth: usize,
    /// Frame pool for efficient allocation
    frame_pool: StackFramePool,
    /// Stack health metrics
    health_metrics: StackHealthMetrics,
    /// Performance history for trend analysis
    performance_history: VecDeque<StackPerformancePoint>,
    /// Last cleanup time
    last_cleanup: Instant,
}

/// Stack health metrics
#[derive(Debug, Clone, Default)]
pub struct StackHealthMetrics {
    /// Total operations performed
    pub total_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Average operation latency
    pub avg_latency_us: f64,
    /// Memory usage trend
    pub memory_trend: f64,
    /// Last health check
    pub last_health_check: Option<Instant>,
}

/// Performance data point
#[derive(Debug, Clone)]
pub struct StackPerformancePoint {
    /// Timestamp
    pub timestamp: Instant,
    /// Stack size at this point
    pub stack_size: usize,
    /// Frame count
    pub frame_count: usize,
    /// Operation latency
    pub latency_us: u64,
    /// Memory usage
    pub memory_usage: usize,
}

/// Execution stack for the VM
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function name
    pub function_name: String,
    /// Function ID
    pub function_id: u32,
    /// Return address (instruction pointer)
    pub return_address: u32,
    /// Base pointer for locals
    pub base_pointer: usize,
    /// Local variables
    pub locals: SmallVec<[StackValue; 8]>,
    /// Upvalues (for closures)
    pub upvalues: SmallVec<[StackValue; 4]>,
    /// Exception handlers active in this frame
    pub exception_handlers: Vec<ExceptionHandler>,
    /// Effects active in this frame
    pub active_effects: Vec<Effect>,
    /// Capabilities available in this frame
    pub capabilities: Vec<Capability>,
}

/// Exception handler information
#[derive(Debug, Clone)]
pub struct ExceptionHandler {
    /// Start instruction offset
    pub start_offset: u32,
    /// End instruction offset
    pub end_offset: u32,
    /// Handler instruction offset
    pub handler_offset: u32,
    /// Exception type (None for catch-all)
    pub exception_type: Option<String>,
}

/// Stack value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StackValue {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Byte array
    Bytes(Vec<u8>),
    /// Array of values
    Array(Vec<StackValue>),
    /// Object with fields
    Object(HashMap<String, StackValue>),
    /// Function reference
    Function {
        /// Function ID
        id: u32,
        /// Captured upvalues (for closures)
        upvalues: Vec<StackValue>,
    },
    /// Type reference
    Type(u32),
    /// Capability token
    Capability(String),
    /// Effect handle
    Effect(String),
}

impl ExecutionStack {
    /// Create a new execution stack
    pub fn new() -> Self {
        Self::with_max_size(DEFAULT_MAX_STACK_SIZE)
    }

    /// Create a new execution stack with maximum size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            frames: Vec::new(),
            values: Vec::new(),
            max_size,
            depth: 0,
            frame_pool: StackFramePool::new(64), // Pool up to 64 frames
            health_metrics: StackHealthMetrics::default(),
            performance_history: VecDeque::with_capacity(1000),
            last_cleanup: Instant::now(),
        }
    }

    /// Push a value onto the stack with degradation support
    pub fn push(&mut self, value: StackValue) -> StackOperationResult<()> {
        let start_time = Instant::now();
        
        // Check for various degradation conditions
        let current_size = self.values.len();
        
        // Hard limit check
        if current_size >= self.max_size {
            return StackOperationResult::Failed(PrismVMError::RuntimeError {
                message: "Stack overflow - maximum size exceeded".to_string(),
            });
        }
        
        // Soft limit warnings
        let usage_ratio = current_size as f64 / self.max_size as f64;
        
        self.values.push(value);
        self.depth = self.depth.max(self.values.len());
        
        // Update health metrics
        self.update_health_metrics(start_time.elapsed());
        
        // Return appropriate result based on usage
        if usage_ratio > 0.95 {
            StackOperationResult::Degraded((), DegradationReason::MemoryConstrained)
        } else if usage_ratio > 0.85 {
            StackOperationResult::SuccessWithWarning((), StackWarning::ApproachingDepthLimit {
                current: current_size + 1,
                limit: self.max_size,
            })
        } else {
            StackOperationResult::Success(())
        }
    }

    /// Pop a value from the stack
    pub fn pop(&mut self) -> StackOperationResult<StackValue> {
        let start_time = Instant::now();
        
        match self.values.pop() {
            Some(value) => {
                self.update_health_metrics(start_time.elapsed());
                StackOperationResult::Success(value)
            }
            None => StackOperationResult::Failed(PrismVMError::RuntimeError {
                message: "Stack underflow".to_string(),
            }),
        }
    }

    /// Peek at the top value without removing it
    pub fn peek(&self) -> VMResult<&StackValue> {
        self.values.last().ok_or_else(|| PrismVMError::RuntimeError {
            message: "Stack is empty".to_string(),
        })
    }

    /// Peek at a value at depth n from the top
    pub fn peek_at(&self, depth: usize) -> VMResult<&StackValue> {
        if depth >= self.values.len() {
            return Err(PrismVMError::RuntimeError {
                message: format!("Invalid stack depth: {}", depth),
            });
        }
        let index = self.values.len() - 1 - depth;
        Ok(&self.values[index])
    }

    /// Duplicate the top value
    pub fn dup(&mut self) -> VMResult<()> {
        let value = self.peek()?.clone();
        self.push(value)
    }

    /// Swap the top two values
    pub fn swap(&mut self) -> VMResult<()> {
        if self.values.len() < 2 {
            return Err(PrismVMError::RuntimeError {
                message: "Not enough values to swap".to_string(),
            });
        }
        let len = self.values.len();
        self.values.swap(len - 1, len - 2);
        Ok(())
    }

    /// Rotate top three values (top -> third)
    pub fn rot3(&mut self) -> VMResult<()> {
        if self.values.len() < 3 {
            return Err(PrismVMError::RuntimeError {
                message: "Not enough values to rotate".to_string(),
            });
        }
        let len = self.values.len();
        // Move top to third position: [a, b, c] -> [c, a, b]
        let top = self.values.remove(len - 1);
        self.values.insert(len - 3, top);
        Ok(())
    }

    /// Push a new stack frame using the frame pool
    pub fn push_frame(&mut self, function_name: String, function_id: u32, return_address: u32) -> StackOperationResult<()> {
        let start_time = Instant::now();
        
        // Check frame depth limits
        let current_frames = self.frames.len();
        let frame_limit = self.max_size / 1024; // Reasonable frame limit
        
        if current_frames >= frame_limit {
            return StackOperationResult::Failed(PrismVMError::RuntimeError {
                message: format!("Frame stack overflow - maximum {} frames exceeded", frame_limit),
            });
        }
        
        // Get frame from pool
        let frame = self.frame_pool.get_frame(function_name, function_id, return_address);
        self.frames.push(frame);
        
        self.update_health_metrics(start_time.elapsed());
        
        // Check for warnings
        if current_frames > frame_limit * 8 / 10 {
            StackOperationResult::SuccessWithWarning((), StackWarning::ApproachingDepthLimit {
                current: current_frames + 1,
                limit: frame_limit,
            })
        } else {
            StackOperationResult::Success(())
        }
    }

    /// Pop the current stack frame and return to pool
    pub fn pop_frame(&mut self) -> StackOperationResult<StackFrame> {
        let start_time = Instant::now();
        
        match self.frames.pop() {
            Some(frame) => {
                // Don't return to pool yet - caller might need the frame data
                self.update_health_metrics(start_time.elapsed());
                StackOperationResult::Success(frame)
            }
            None => StackOperationResult::Failed(PrismVMError::RuntimeError {
                message: "No frame to pop".to_string(),
            }),
        }
    }

    /// Return a frame to the pool after use
    pub fn recycle_frame(&mut self, frame: StackFrame) {
        self.frame_pool.return_frame(frame);
    }

    /// Get the current stack frame
    pub fn current_frame(&self) -> VMResult<&StackFrame> {
        self.frames.last().ok_or_else(|| PrismVMError::RuntimeError {
            message: "No current frame".to_string(),
        })
    }

    /// Get the current stack frame mutably
    pub fn current_frame_mut(&mut self) -> VMResult<&mut StackFrame> {
        self.frames.last_mut().ok_or_else(|| PrismVMError::RuntimeError {
            message: "No current frame".to_string(),
        })
    }

    /// Get local variable from current frame
    pub fn get_local(&self, slot: u8) -> VMResult<&StackValue> {
        let frame = self.frames.last()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for local variable access".to_string(),
            })?;
        
        frame.locals.get(slot as usize)
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: format!("Invalid local variable slot: {}", slot),
            })
    }

    /// Set local variable in current frame
    pub fn set_local(&mut self, slot: u8, value: StackValue) -> VMResult<()> {
        let frame = self.frames.last_mut()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for local variable access".to_string(),
            })?;
        
        // Extend locals if necessary
        while frame.locals.len() <= slot as usize {
            frame.locals.push(StackValue::Null);
        }
        
        frame.locals[slot as usize] = value;
        Ok(())
    }

    /// Get upvalue from current frame
    pub fn get_upvalue(&self, slot: u8) -> VMResult<&StackValue> {
        let frame = self.frames.last()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for upvalue access".to_string(),
            })?;
        
        frame.upvalues.get(slot as usize)
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: format!("Invalid upvalue slot: {}", slot),
            })
    }

    /// Set upvalue in current frame
    pub fn set_upvalue(&mut self, slot: u8, value: StackValue) -> VMResult<()> {
        let frame = self.frames.last_mut()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for upvalue access".to_string(),
            })?;
        
        // Extend upvalues if necessary
        while frame.upvalues.len() <= slot as usize {
            frame.upvalues.push(StackValue::Null);
        }
        
        frame.upvalues[slot as usize] = value;
        Ok(())
    }

    /// Get the current stack size
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Get the maximum stack depth reached
    pub fn max_depth(&self) -> usize {
        self.depth
    }

    /// Get the number of frames
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Clear the stack
    pub fn clear(&mut self) {
        self.values.clear();
        self.frames.clear();
        self.depth = 0;
    }

    /// Check if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get stack statistics
    pub fn statistics(&self) -> StackStatistics {
        StackStatistics {
            current_size: self.values.len(),
            max_depth: self.depth,
            frame_count: self.frames.len(),
            max_size: self.max_size,
        }
    }

    /// Update health metrics
    fn update_health_metrics(&mut self, operation_latency: Duration) {
        let latency_us = operation_latency.as_micros() as f64;
        
        self.health_metrics.total_operations += 1;
        
        // Update average latency using exponential moving average
        let alpha = 0.1;
        if self.health_metrics.total_operations == 1 {
            self.health_metrics.avg_latency_us = latency_us;
        } else {
            self.health_metrics.avg_latency_us = 
                alpha * latency_us + (1.0 - alpha) * self.health_metrics.avg_latency_us;
        }
        
        // Add performance point
        if self.performance_history.len() >= 1000 {
            self.performance_history.pop_front();
        }
        
        self.performance_history.push_back(StackPerformancePoint {
            timestamp: Instant::now(),
            stack_size: self.values.len(),
            frame_count: self.frames.len(),
            latency_us: latency_us as u64,
            memory_usage: self.estimate_memory_usage(),
        });
        
        // Periodic cleanup
        if self.last_cleanup.elapsed() > Duration::from_secs(60) {
            self.perform_maintenance();
            self.last_cleanup = Instant::now();
        }
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        let frame_size = std::mem::size_of::<StackFrame>();
        let value_size = std::mem::size_of::<StackValue>();
        
        self.frames.len() * frame_size + self.values.len() * value_size
    }

    /// Perform periodic maintenance
    fn perform_maintenance(&mut self) {
        // Analyze performance trends
        if let Some(trend) = self.analyze_performance_trend() {
            match trend {
                PerformanceTrend::Degrading => {
                    warn!("Stack performance degrading - consider optimization");
                }
                PerformanceTrend::MemoryPressure => {
                    warn!("Stack memory pressure detected");
                    // Could trigger garbage collection or other optimizations
                }
                _ => {}
            }
        }
        
        // Update memory trend
        if let Some(latest_points) = self.get_recent_performance_points(10) {
            let memory_values: Vec<f64> = latest_points.iter()
                .map(|p| p.memory_usage as f64)
                .collect();
            
            if memory_values.len() >= 2 {
                let trend = self.calculate_trend(&memory_values);
                self.health_metrics.memory_trend = trend;
            }
        }
    }

    /// Analyze performance trends
    fn analyze_performance_trend(&self) -> Option<PerformanceTrend> {
        if self.performance_history.len() < 10 {
            return None;
        }
        
        let recent_points: Vec<_> = self.performance_history.iter().rev().take(10).collect();
        
        // Check latency trend
        let latencies: Vec<f64> = recent_points.iter().map(|p| p.latency_us as f64).collect();
        let latency_trend = self.calculate_trend(&latencies);
        
        // Check memory trend
        let memory_usage: Vec<f64> = recent_points.iter().map(|p| p.memory_usage as f64).collect();
        let memory_trend = self.calculate_trend(&memory_usage);
        
        if latency_trend > 0.1 {
            Some(PerformanceTrend::Degrading)
        } else if memory_trend > 0.2 {
            Some(PerformanceTrend::MemoryPressure)
        } else if latency_trend < -0.1 {
            Some(PerformanceTrend::Improving)
        } else {
            Some(PerformanceTrend::Stable)
        }
    }

    /// Calculate trend using simple linear regression
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
        
        // Calculate slope (trend)
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = n * sum_x2 - sum_x.powi(2);
        
        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Get recent performance points
    fn get_recent_performance_points(&self, count: usize) -> Option<Vec<&StackPerformancePoint>> {
        if self.performance_history.len() < count {
            return None;
        }
        
        Some(self.performance_history.iter().rev().take(count).collect())
    }

    /// Get comprehensive stack health status
    pub fn health_status(&self) -> StackHealthStatus {
        let current_usage = self.values.len() as f64 / self.max_size as f64;
        let frame_usage = self.frames.len() as f64 / (self.max_size / 1024) as f64;
        
        let status = if current_usage > 0.95 || frame_usage > 0.95 {
            HealthLevel::Critical
        } else if current_usage > 0.85 || frame_usage > 0.85 {
            HealthLevel::Warning
        } else if self.health_metrics.avg_latency_us > 1000.0 {
            HealthLevel::Degraded
        } else {
            HealthLevel::Healthy
        };
        
        StackHealthStatus {
            level: status,
            stack_usage_percent: current_usage * 100.0,
            frame_usage_percent: frame_usage * 100.0,
            avg_latency_us: self.health_metrics.avg_latency_us,
            memory_trend: self.health_metrics.memory_trend,
            frame_pool_stats: self.frame_pool.stats(),
            recommendations: self.generate_health_recommendations(),
        }
    }

    /// Generate health recommendations
    fn generate_health_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let usage = self.values.len() as f64 / self.max_size as f64;
        if usage > 0.8 {
            recommendations.push("Consider increasing stack size or optimizing recursive algorithms".to_string());
        }
        
        if self.health_metrics.avg_latency_us > 500.0 {
            recommendations.push("Stack operations are slow - check for memory pressure".to_string());
        }
        
        let pool_stats = self.frame_pool.stats();
        if pool_stats.hit_rate < 0.5 {
            recommendations.push("Frame pool hit rate is low - consider increasing pool size".to_string());
        }
        
        if self.health_metrics.memory_trend > 0.1 {
            recommendations.push("Memory usage is trending upward - monitor for leaks".to_string());
        }
        
        recommendations
    }
}

impl Default for ExecutionStack {
    fn default() -> Self {
        Self::new()
    }
}

impl StackFrame {
    /// Create a new stack frame
    pub fn new(function_name: String, function_id: u32, return_address: u32) -> Self {
        Self {
            function_name,
            function_id,
            return_address,
            base_pointer: 0,
            locals: SmallVec::new(),
            upvalues: SmallVec::new(),
            exception_handlers: Vec::new(),
            active_effects: Vec::new(),
            capabilities: Vec::new(),
        }
    }

    /// Add a local variable
    pub fn add_local(&mut self, value: StackValue) {
        self.locals.push(value);
    }

    /// Add an upvalue
    pub fn add_upvalue(&mut self, value: StackValue) {
        self.upvalues.push(value);
    }

    /// Add an exception handler
    pub fn add_exception_handler(&mut self, handler: ExceptionHandler) {
        self.exception_handlers.push(handler);
    }

    /// Add an active effect
    pub fn add_effect(&mut self, effect: Effect) {
        self.active_effects.push(effect);
    }

    /// Add a capability
    pub fn add_capability(&mut self, capability: Capability) {
        self.capabilities.push(capability);
    }
}

impl StackValue {
    /// Check if this value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            StackValue::Null => false,
            StackValue::Boolean(b) => *b,
            StackValue::Integer(i) => *i != 0,
            StackValue::Float(f) => *f != 0.0 && !f.is_nan(),
            StackValue::String(s) => !s.is_empty(),
            StackValue::Bytes(b) => !b.is_empty(),
            StackValue::Array(a) => !a.is_empty(),
            StackValue::Object(o) => !o.is_empty(),
            StackValue::Function { .. } => true,
            StackValue::Type(_) => true,
            StackValue::Capability(_) => true,
            StackValue::Effect(_) => true,
        }
    }

    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            StackValue::Null => "null",
            StackValue::Boolean(_) => "boolean",
            StackValue::Integer(_) => "integer",
            StackValue::Float(_) => "float",
            StackValue::String(_) => "string",
            StackValue::Bytes(_) => "bytes",
            StackValue::Array(_) => "array",
            StackValue::Object(_) => "object",
            StackValue::Function { .. } => "function",
            StackValue::Type(_) => "type",
            StackValue::Capability(_) => "capability",
            StackValue::Effect(_) => "effect",
        }
    }

    /// Convert from a constant
    pub fn from_constant(constant: &Constant) -> Self {
        match constant {
            Constant::Null => StackValue::Null,
            Constant::Boolean(b) => StackValue::Boolean(*b),
            Constant::Integer(i) => StackValue::Integer(*i),
            Constant::Float(f) => StackValue::Float(*f),
            Constant::String(s) => StackValue::String(s.clone()),
            Constant::Bytes(b) => StackValue::Bytes(b.clone()),
            Constant::Type(id) => StackValue::Type(*id),
            Constant::Function(id) => StackValue::Function {
                id: *id,
                upvalues: Vec::new(),
            },
            Constant::Composite(_) => {
                // For composite constants, we'd need to recursively convert
                // For now, just return null
                StackValue::Null
            }
        }
    }
}

/// Stack statistics
#[derive(Debug, Clone)]
pub struct StackStatistics {
    /// Current stack size
    pub current_size: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Number of frames
    pub frame_count: usize,
    /// Maximum stack size
    pub max_size: usize,
} 

/// Performance trend analysis
#[derive(Debug, Clone)]
enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    MemoryPressure,
}

/// Health levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthLevel {
    Healthy,
    Warning,
    Degraded,
    Critical,
}

/// Comprehensive health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackHealthStatus {
    pub level: HealthLevel,
    pub stack_usage_percent: f64,
    pub frame_usage_percent: f64,
    pub avg_latency_us: f64,
    pub memory_trend: f64,
    pub frame_pool_stats: FramePoolStats,
    pub recommendations: Vec<String>,
} 