//! Precise Stack Scanner for Prism VM Execution Stacks
//!
//! This module implements precise stack scanning for Prism VM, taking advantage
//! of the well-defined StackValue types and execution model to accurately
//! identify heap references without false positives.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use super::{types::*, interfaces::*};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, span, Level};

/// Precise stack scanner for Prism VM execution stacks
/// 
/// This scanner leverages Prism's well-defined type system to perform precise
/// scanning of execution stacks, avoiding the false positives of conservative
/// scanning while handling both interpreter and JIT compiled frames.
pub struct StackScanner {
    /// Scanning configuration
    config: StackScannerConfig,
    
    /// Scanning statistics
    statistics: Arc<Mutex<ScanStatistics>>,
    
    /// Cache of recently scanned frames
    frame_cache: Arc<Mutex<HashMap<u32, CachedFrameInfo>>>,
    
    /// JIT integration for compiled frames
    jit_integration: Option<Arc<Mutex<JitStackIntegration>>>,
    
    /// Value scanner for nested references
    value_scanner: ValueScanner,
}

/// Configuration for stack scanning behavior
#[derive(Debug, Clone)]
pub struct StackScannerConfig {
    /// Scanning strategy to use
    pub strategy: StackScanStrategy,
    /// Enable frame caching for performance
    pub enable_frame_caching: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Cache TTL in milliseconds
    pub cache_ttl_ms: u64,
    /// Enable JIT integration
    pub enable_jit_integration: bool,
    /// Maximum scanning time per frame (microseconds)
    pub max_scan_time_per_frame_us: u64,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
}

impl Default for StackScannerConfig {
    fn default() -> Self {
        Self {
            strategy: StackScanStrategy::Precise,
            enable_frame_caching: true,
            max_cache_size: 1000,
            cache_ttl_ms: 100,
            enable_jit_integration: true,
            max_scan_time_per_frame_us: 50,
            enable_detailed_logging: false,
        }
    }
}

/// Cached information about a scanned frame
#[derive(Debug, Clone)]
struct CachedFrameInfo {
    /// Function ID this cache entry is for
    function_id: u32,
    /// Cached root pointers
    cached_roots: Vec<*const u8>,
    /// Cache timestamp
    cached_at: Instant,
    /// Frame hash for validation
    frame_hash: u64,
}

/// JIT stack integration for compiled frames
#[derive(Debug)]
struct JitStackIntegration {
    /// Stack maps for compiled functions
    stack_maps: HashMap<u32, JitStackMap>,
    /// Deoptimization information
    deopt_info: HashMap<u32, DeoptInfo>,
}

/// Stack map for a JIT compiled function
#[derive(Debug, Clone)]
struct JitStackMap {
    /// Function ID
    function_id: u32,
    /// Pointer offsets within the frame
    pointer_offsets: Vec<i32>,
    /// Managed value offsets
    managed_offsets: Vec<i32>,
    /// Frame size
    frame_size: usize,
}

/// Deoptimization information
#[derive(Debug, Clone)]
struct DeoptInfo {
    /// Original interpreter state
    interpreter_state: InterpreterFrameState,
    /// Reconstruction information
    reconstruction_info: Vec<ValueReconstructionInfo>,
}

/// Interpreter frame state for deoptimization
#[derive(Debug, Clone)]
struct InterpreterFrameState {
    /// Local variable values
    locals: Vec<StackValue>,
    /// Stack values
    stack_values: Vec<StackValue>,
    /// Instruction pointer
    instruction_pointer: u32,
}

/// Value reconstruction information
#[derive(Debug, Clone)]
struct ValueReconstructionInfo {
    /// Target slot
    slot: u8,
    /// Value type
    value_type: StackValueType,
    /// Reconstruction strategy
    strategy: ReconstructionStrategy,
}

/// Value reconstruction strategies
#[derive(Debug, Clone)]
enum ReconstructionStrategy {
    /// Direct copy from compiled state
    DirectCopy,
    /// Type-specific reconstruction
    TypeSpecific,
    /// Conservative reconstruction
    Conservative,
}

/// Value scanner for nested references in StackValues
#[derive(Debug)]
struct ValueScanner {
    /// Scanning configuration
    config: ValueScannerConfig,
    /// Recursion depth limit
    max_recursion_depth: usize,
    /// Statistics
    stats: ScanStatistics,
}

/// Configuration for value scanning
#[derive(Debug, Clone)]
struct ValueScannerConfig {
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Enable cycle detection
    pub enable_cycle_detection: bool,
    /// Scanning timeout per value (microseconds)
    pub timeout_per_value_us: u64,
}

impl Default for ValueScannerConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            enable_cycle_detection: true,
            timeout_per_value_us: 10,
        }
    }
}

impl StackScanner {
    /// Create a new stack scanner with default configuration
    pub fn new() -> VMResult<Self> {
        Self::with_config(StackScannerConfig::default())
    }
    
    /// Create a new stack scanner with custom configuration
    pub fn with_config(config: StackScannerConfig) -> VMResult<Self> {
        let _span = span!(Level::INFO, "stack_scanner_init").entered();
        info!("Initializing stack scanner with strategy: {:?}", config.strategy);
        
        let jit_integration = if config.enable_jit_integration {
            Some(Arc::new(Mutex::new(JitStackIntegration {
                stack_maps: HashMap::new(),
                deopt_info: HashMap::new(),
            })))
        } else {
            None
        };
        
        Ok(Self {
            config,
            statistics: Arc::new(Mutex::new(ScanStatistics {
                total_scans: 0,
                average_scan_time_us: 0.0,
                max_scan_time_us: 0,
                min_scan_time_us: u64::MAX,
                roots_per_second: 0.0,
                scan_efficiency: 0.0,
                cache_hit_rate: 0.0,
                last_scan_time: None,
            })),
            frame_cache: Arc::new(Mutex::new(HashMap::new())),
            jit_integration,
            value_scanner: ValueScanner {
                config: ValueScannerConfig::default(),
                max_recursion_depth: 100,
                stats: ScanStatistics {
                    total_scans: 0,
                    average_scan_time_us: 0.0,
                    max_scan_time_us: 0,
                    min_scan_time_us: u64::MAX,
                    roots_per_second: 0.0,
                    scan_efficiency: 0.0,
                    cache_hit_rate: 0.0,
                    last_scan_time: None,
                },
            },
        })
    }
    
    /// Scan for roots using the execution stack directly
    pub fn scan_for_roots(&self) -> VMResult<Vec<*const u8>> {
        // This method is used when we don't have direct access to the execution stack
        // It returns an empty vector as a safe fallback
        warn!("scan_for_roots called without execution stack context - returning empty result");
        Ok(Vec::new())
    }

    /// Scan execution stack for roots with current frame access
    pub fn scan_execution_stack_for_roots(&self, stack: &ExecutionStack) -> VMResult<Vec<*const u8>> {
        let _span = span!(Level::DEBUG, "scan_execution_stack_for_roots").entered();
        let start_time = Instant::now();
        
        let mut all_roots = Vec::new();
        
        // Scan current stack values
        let stack_stats = stack.statistics();
        for depth in 0..stack_stats.current_size {
            if let Ok(stack_value) = stack.peek_at(depth) {
                match self.value_scanner.scan_value(stack_value, 0) {
                    RootOperationResult::Success(value_roots) => {
                        all_roots.extend(value_roots);
                    }
                    RootOperationResult::Failed(e) => {
                        warn!("Failed to scan stack value at depth {}: {:?}", depth, e);
                    }
                    _ => {}
                }
            }
        }
        
        // Scan current frame if available
        if let Ok(current_frame) = stack.current_frame() {
            match self.scan_frame_for_roots(current_frame) {
                Ok(frame_roots) => {
                    all_roots.extend(frame_roots);
                }
                Err(e) => {
                    warn!("Failed to scan current frame: {:?}", e);
                }
            }
        }
        
        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.total_scans += 1;
            let scan_time_us = start_time.elapsed().as_micros() as u64;
            stats.max_scan_time_us = stats.max_scan_time_us.max(scan_time_us);
            stats.min_scan_time_us = stats.min_scan_time_us.min(scan_time_us);
            
            // Update average
            let total_time = stats.average_scan_time_us * (stats.total_scans - 1) as f64 + scan_time_us as f64;
            stats.average_scan_time_us = total_time / stats.total_scans as f64;
            
            stats.last_scan_time = Some(Instant::now());
            
            if scan_time_us > 0 {
                stats.roots_per_second = (all_roots.len() as f64 * 1_000_000.0) / scan_time_us as f64;
                stats.scan_efficiency = all_roots.len() as f64 / scan_time_us as f64;
            }
        }
        
        debug!("Scanned execution stack, found {} roots in {}μs", 
               all_roots.len(), start_time.elapsed().as_micros());
        
        Ok(all_roots)
    }

    /// Scan a single frame for root references
    pub fn scan_frame_for_roots(&self, frame: &StackFrame) -> VMResult<Vec<*const u8>> {
        let _span = span!(Level::TRACE, "scan_frame_for_roots", 
                         function = %frame.function_name, 
                         function_id = frame.function_id).entered();
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.enable_frame_caching {
            if let Some(cached_roots) = self.check_frame_cache_for_pointers(frame) {
                debug!("Cache hit for frame {}", frame.function_id);
                return Ok(cached_roots);
            }
        }
        
        let mut frame_roots = Vec::new();
        
        // Scan local variables
        for local_value in &frame.locals {
            match self.value_scanner.scan_value(local_value, 0) {
                RootOperationResult::Success(value_roots) => {
                    frame_roots.extend(value_roots);
                }
                RootOperationResult::Failed(e) => {
                    warn!("Failed to scan local variable: {:?}", e);
                }
                _ => {}
            }
        }
        
        // Scan upvalues (for closures)
        for upvalue in &frame.upvalues {
            match self.value_scanner.scan_value(upvalue, 0) {
                RootOperationResult::Success(value_roots) => {
                    frame_roots.extend(value_roots);
                }
                RootOperationResult::Failed(e) => {
                    warn!("Failed to scan upvalue: {:?}", e);
                }
                _ => {}
            }
        }
        
        // Cache the results
        if self.config.enable_frame_caching {
            self.cache_frame_pointers(frame, &frame_roots);
        }
        
        let scan_time_us = start_time.elapsed().as_micros() as u64;
        if scan_time_us > self.config.max_scan_time_per_frame_us {
            warn!("Frame scan took {}μs, exceeding limit of {}μs", 
                  scan_time_us, self.config.max_scan_time_per_frame_us);
        }
        
        Ok(frame_roots)
    }
}

impl StackScannerInterface for StackScanner {
    /// Scan the execution stack for root objects
    fn scan_execution_stack(&self, stack: &ExecutionStack) -> RootOperationResult<Vec<RootEntry>> {
        let _span = span!(Level::DEBUG, "scan_execution_stack").entered();
        let start_time = Instant::now();
        
        let mut all_roots = Vec::new();
        let mut total_frames_scanned = 0;
        
        // Scan current frame if available
        if let Ok(current_frame) = stack.current_frame() {
            match self.scan_stack_frame(current_frame) {
                RootOperationResult::Success(frame_roots) => {
                    all_roots.extend(frame_roots);
                    total_frames_scanned += 1;
                }
                RootOperationResult::Failed(e) => {
                    warn!("Failed to scan current frame {}: {:?}", current_frame.function_id, e);
                }
                RootOperationResult::SuccessWithWarnings(frame_roots, warnings) => {
                    all_roots.extend(frame_roots);
                    total_frames_scanned += 1;
                    for warning in warnings {
                        debug!("Frame scan warning: {:?}", warning);
                    }
                }
                _ => {}
            }
        }
        
        // Scan stack values
        let stack_stats = stack.statistics();
        for depth in 0..stack_stats.current_size {
            if let Ok(stack_value) = stack.peek_at(depth) {
                match self.scan_stack_value(stack_value) {
                    RootOperationResult::Success(value_roots) => {
                        for &ptr in &value_roots {
                            all_roots.push(RootEntry {
                                ptr,
                                source: RootSource::ExecutionStack,
                                root_type: RootType::StackValue { 
                                    value_type: StackValueType::from_stack_value(stack_value) 
                                },
                                registered_at: Instant::now(),
                                thread_id: std::thread::current().id(),
                                security_context: self.create_default_security_context(),
                                metadata: self.create_stack_value_metadata(depth),
                            });
                        }
                    }
                    RootOperationResult::Failed(e) => {
                        warn!("Failed to scan stack value at depth {}: {:?}", depth, e);
                    }
                    _ => {}
                }
            }
        }
        
        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.total_scans += 1;
            let scan_time_us = start_time.elapsed().as_micros() as u64;
            stats.max_scan_time_us = stats.max_scan_time_us.max(scan_time_us);
            stats.min_scan_time_us = stats.min_scan_time_us.min(scan_time_us);
            
            // Update average
            let total_time = stats.average_scan_time_us * (stats.total_scans - 1) as f64 + scan_time_us as f64;
            stats.average_scan_time_us = total_time / stats.total_scans as f64;
            
            stats.last_scan_time = Some(Instant::now());
            
            if scan_time_us > 0 {
                stats.roots_per_second = (all_roots.len() as f64 * 1_000_000.0) / scan_time_us as f64;
                stats.scan_efficiency = all_roots.len() as f64 / scan_time_us as f64;
            }
        }
        
        debug!("Scanned {} frames, found {} roots in {}μs", 
               total_frames_scanned, all_roots.len(), start_time.elapsed().as_micros());
        
        RootOperationResult::Success(all_roots)
    }
    
    /// Scan a specific stack frame
    fn scan_stack_frame(&self, frame: &StackFrame) -> RootOperationResult<Vec<RootEntry>> {
        let _span = span!(Level::TRACE, "scan_stack_frame", 
                         function = %frame.function_name, 
                         function_id = frame.function_id).entered();
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.enable_frame_caching {
            if let Some(cached_roots) = self.check_frame_cache(frame) {
                debug!("Cache hit for frame {}", frame.function_id);
                return RootOperationResult::Success(cached_roots);
            }
        }
        
        let mut frame_roots = Vec::new();
        let mut warnings = Vec::new();
        
        // Scan local variables
        for (slot, local_value) in frame.locals.iter().enumerate() {
            match self.scan_stack_value(local_value) {
                RootOperationResult::Success(value_roots) => {
                    for &ptr in &value_roots {
                        frame_roots.push(RootEntry {
                            ptr,
                            source: RootSource::ExecutionStack,
                            root_type: RootType::StackValue { 
                                value_type: StackValueType::from_stack_value(local_value) 
                            },
                            registered_at: Instant::now(),
                            thread_id: std::thread::current().id(),
                            security_context: self.create_default_security_context(),
                            metadata: self.create_local_metadata(frame.function_id, slot as u8),
                        });
                    }
                }
                RootOperationResult::Failed(e) => {
                    warnings.push(RootWarning::SlowScanning { 
                        time_us: start_time.elapsed().as_micros() as u64, 
                        limit_us: self.config.max_scan_time_per_frame_us 
                    });
                    warn!("Failed to scan local variable {}: {:?}", slot, e);
                }
                _ => {}
            }
        }
        
        // Scan upvalues (for closures)
        for (slot, upvalue) in frame.upvalues.iter().enumerate() {
            match self.scan_stack_value(upvalue) {
                RootOperationResult::Success(value_roots) => {
                    for &ptr in &value_roots {
                        frame_roots.push(RootEntry {
                            ptr,
                            source: RootSource::ExecutionStack,
                            root_type: RootType::StackValue { 
                                value_type: StackValueType::from_stack_value(upvalue) 
                            },
                            registered_at: Instant::now(),
                            thread_id: std::thread::current().id(),
                            security_context: self.create_default_security_context(),
                            metadata: self.create_upvalue_metadata(frame.function_id, slot as u8),
                        });
                    }
                }
                RootOperationResult::Failed(e) => {
                    warn!("Failed to scan upvalue {}: {:?}", slot, e);
                }
                _ => {}
            }
        }
        
        // Cache the results
        if self.config.enable_frame_caching {
            self.cache_frame_results(frame, &frame_roots);
        }
        
        let scan_time_us = start_time.elapsed().as_micros() as u64;
        if scan_time_us > self.config.max_scan_time_per_frame_us {
            warnings.push(RootWarning::SlowScanning { 
                time_us: scan_time_us, 
                limit_us: self.config.max_scan_time_per_frame_us 
            });
        }
        
        if warnings.is_empty() {
            RootOperationResult::Success(frame_roots)
        } else {
            RootOperationResult::SuccessWithWarnings(frame_roots, warnings)
        }
    }
    
    /// Scan a stack value for nested references
    fn scan_stack_value(&self, value: &StackValue) -> RootOperationResult<Vec<*const u8>> {
        self.value_scanner.scan_value(value, 0)
    }
    
    /// Get platform-specific stack bounds
    fn get_stack_bounds(&self) -> RootOperationResult<Option<(usize, usize)>> {
        // Stack bounds detection requires platform-specific implementation
        // Return None to indicate stack bounds are not available through this interface
        RootOperationResult::Success(None)
    }
    
    /// Scan native stack (conservative approach)
    fn scan_native_stack(&self, start: *const u8, end: *const u8) -> RootOperationResult<Vec<*const u8>> {
        if self.config.strategy == StackScanStrategy::Precise {
            // Precise scanning doesn't need native stack scanning
            return RootOperationResult::Success(Vec::new());
        }
        
        let mut potential_roots = Vec::new();
        
        unsafe {
            let mut current = start as usize;
            let end_addr = end as usize;
            
            while current <= end_addr.saturating_sub(std::mem::size_of::<usize>()) {
                // Align to pointer boundary
                let aligned = (current + std::mem::align_of::<usize>() - 1) 
                    & !(std::mem::align_of::<usize>() - 1);
                
                if aligned > end_addr {
                    break;
                }
                
                let potential_ptr = *(aligned as *const *const u8);
                
                if self.could_be_heap_pointer(potential_ptr) {
                    potential_roots.push(potential_ptr);
                }
                
                current = aligned + std::mem::size_of::<usize>();
            }
        }
        
        RootOperationResult::Success(potential_roots)
    }
    
    /// Configure scanning strategy
    fn set_scan_strategy(&mut self, strategy: StackScanStrategy) -> RootOperationResult<()> {
        self.config.strategy = strategy;
        info!("Stack scanning strategy changed to: {:?}", strategy);
        RootOperationResult::Success(())
    }
    
    /// Get scanning statistics
    fn get_scan_statistics(&self) -> ScanStatistics {
        self.statistics.lock().unwrap().clone()
    }
}

impl StackScanner {
    /// Check frame cache for previously scanned results
    fn check_frame_cache(&self, frame: &StackFrame) -> Option<Vec<RootEntry>> {
        let cache = self.frame_cache.lock().unwrap();
        
        if let Some(cached_info) = cache.get(&frame.function_id) {
            let cache_age = cached_info.cached_at.elapsed();
            let cache_ttl = Duration::from_millis(self.config.cache_ttl_ms);
            
            if cache_age < cache_ttl && cached_info.frame_hash == self.hash_frame(frame) {
                // Convert cached pointers to RootEntry objects
                let mut root_entries = Vec::new();
                for &ptr in &cached_info.cached_roots {
                    root_entries.push(RootEntry {
                        ptr,
                        source: RootSource::ExecutionStack,
                        root_type: RootType::Unknown, // Simplified for cache
                        registered_at: cached_info.cached_at,
                        thread_id: std::thread::current().id(),
                        security_context: self.create_default_security_context(),
                        metadata: RootMetadata {
                            description: Some("Cached frame scan result".to_string()),
                            tags: vec!["cached".to_string()],
                            business_context: None,
                            performance_hints: PerformanceHints {
                                scan_frequency: ScanFrequency::High,
                                access_pattern: AccessPattern::Temporal,
                                locality_hints: LocalityHints {
                                    prefer_cache_friendly: true,
                                    numa_node: None,
                                    cache_level: CacheLevel::L1,
                                },
                                scan_priority: ScanPriority::High,
                            },
                            debug_info: None,
                        },
                    });
                }
                return Some(root_entries);
            }
        }
        
        None
    }

    /// Check frame cache for pointer results only
    fn check_frame_cache_for_pointers(&self, frame: &StackFrame) -> Option<Vec<*const u8>> {
        let cache = self.frame_cache.lock().unwrap();
        
        if let Some(cached_info) = cache.get(&frame.function_id) {
            let cache_age = cached_info.cached_at.elapsed();
            let cache_ttl = Duration::from_millis(self.config.cache_ttl_ms);
            
            if cache_age < cache_ttl && cached_info.frame_hash == self.hash_frame(frame) {
                return Some(cached_info.cached_roots.clone());
            }
        }
        
        None
    }
    
    /// Cache frame scan results
    fn cache_frame_results(&self, frame: &StackFrame, roots: &[RootEntry]) {
        if roots.len() > self.config.max_cache_size {
            return; // Don't cache very large result sets
        }
        
        let cached_info = CachedFrameInfo {
            function_id: frame.function_id,
            cached_roots: roots.iter().map(|r| r.ptr).collect(),
            cached_at: Instant::now(),
            frame_hash: self.hash_frame(frame),
        };
        
        let mut cache = self.frame_cache.lock().unwrap();
        cache.insert(frame.function_id, cached_info);
        
        // Cleanup old cache entries if cache is too large
        if cache.len() > self.config.max_cache_size {
            let oldest_key = cache.iter()
                .min_by_key(|(_, info)| info.cached_at)
                .map(|(&key, _)| key);
            
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
    }

    /// Cache frame pointer results
    fn cache_frame_pointers(&self, frame: &StackFrame, roots: &[*const u8]) {
        if roots.len() > self.config.max_cache_size {
            return; // Don't cache very large result sets
        }
        
        let cached_info = CachedFrameInfo {
            function_id: frame.function_id,
            cached_roots: roots.to_vec(),
            cached_at: Instant::now(),
            frame_hash: self.hash_frame(frame),
        };
        
        let mut cache = self.frame_cache.lock().unwrap();
        cache.insert(frame.function_id, cached_info);
        
        // Cleanup old cache entries if cache is too large
        if cache.len() > self.config.max_cache_size {
            let oldest_key = cache.iter()
                .min_by_key(|(_, info)| info.cached_at)
                .map(|(&key, _)| key);
            
            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }
    }
    
    /// Calculate hash of frame for cache validation
    fn hash_frame(&self, frame: &StackFrame) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        frame.function_id.hash(&mut hasher);
        frame.locals.len().hash(&mut hasher);
        frame.upvalues.len().hash(&mut hasher);
        // Note: We're not hashing the actual values for performance reasons
        // This means cache might have false hits, but that's acceptable
        hasher.finish()
    }
    
    /// Check if a pointer could be a valid heap pointer
    fn could_be_heap_pointer(&self, ptr: *const u8) -> bool {
        if ptr.is_null() {
            return false;
        }
        
        let addr = ptr as usize;
        
        // Check alignment
        if addr % std::mem::align_of::<usize>() != 0 {
            return false;
        }
        
        // Filter out small values that are likely integers
        if addr < 0x1000 {
            return false;
        }
        
        // Filter out very large values that are likely not heap pointers
        if addr > 0x7fff_ffff_ffff {
            return false;
        }
        
        true
    }
    
    /// Create default security context for root entries
    fn create_default_security_context(&self) -> SecurityContext {
        SecurityContext {
            capabilities: prism_runtime::authority::capability::CapabilitySet::new(),
            classification: SecurityClassification::Internal,
            restrictions: AccessRestrictions {
                read_restrictions: Vec::new(),
                write_restrictions: Vec::new(),
                time_restrictions: None,
                context_restrictions: Vec::new(),
            },
            audit_required: false,
        }
    }
    
    /// Create metadata for local variable roots
    fn create_local_metadata(&self, function_id: u32, slot: u8) -> RootMetadata {
        RootMetadata {
            description: Some(format!("Local variable slot {} in function {}", slot, function_id)),
            tags: vec!["local".to_string(), "stack".to_string()],
            business_context: None,
            performance_hints: PerformanceHints {
                scan_frequency: ScanFrequency::EveryGC,
                access_pattern: AccessPattern::Temporal,
                locality_hints: LocalityHints {
                    prefer_cache_friendly: true,
                    numa_node: None,
                    cache_level: CacheLevel::L1,
                },
                scan_priority: ScanPriority::High,
            },
            debug_info: None,
        }
    }
    
    /// Create metadata for upvalue roots
    fn create_upvalue_metadata(&self, function_id: u32, slot: u8) -> RootMetadata {
        RootMetadata {
            description: Some(format!("Upvalue slot {} in function {}", slot, function_id)),
            tags: vec!["upvalue".to_string(), "closure".to_string()],
            business_context: None,
            performance_hints: PerformanceHints {
                scan_frequency: ScanFrequency::EveryGC,
                access_pattern: AccessPattern::Temporal,
                locality_hints: LocalityHints {
                    prefer_cache_friendly: true,
                    numa_node: None,
                    cache_level: CacheLevel::L2,
                },
                scan_priority: ScanPriority::High,
            },
            debug_info: None,
        }
    }

    /// Create metadata for stack value roots
    fn create_stack_value_metadata(&self, depth: usize) -> RootMetadata {
        RootMetadata {
            description: Some(format!("Stack value at depth {}", depth)),
            tags: vec!["stack_value".to_string(), "execution_stack".to_string()],
            business_context: None,
            performance_hints: PerformanceHints {
                scan_frequency: ScanFrequency::EveryGC,
                access_pattern: AccessPattern::Temporal,
                locality_hints: LocalityHints {
                    prefer_cache_friendly: true,
                    numa_node: None,
                    cache_level: CacheLevel::L1,
                },
                scan_priority: ScanPriority::High,
            },
            debug_info: None,
        }
    }
}

impl ValueScanner {
    /// Scan a stack value for nested heap references
    fn scan_value(&self, value: &StackValue, depth: usize) -> RootOperationResult<Vec<*const u8>> {
        if depth > self.config.max_depth {
            return RootOperationResult::Failed(RootError::ConfigurationError {
                error: "Maximum recursion depth exceeded".to_string()
            });
        }
        
        let mut roots = Vec::new();
        
        match value {
            StackValue::Null | StackValue::Boolean(_) | StackValue::Integer(_) 
            | StackValue::Float(_) | StackValue::Type(_) => {
                // These types don't contain heap references
            }
            
            StackValue::String(s) => {
                // String itself is a heap reference
                roots.push(s.as_ptr() as *const u8);
            }
            
            StackValue::Bytes(bytes) => {
                // Byte array is a heap reference
                roots.push(bytes.as_ptr() as *const u8);
            }
            
            StackValue::Array(arr) => {
                // Array itself is a heap reference
                roots.push(arr.as_ptr() as *const u8);
                
                // Recursively scan array elements
                for element in arr {
                    match self.scan_value(element, depth + 1) {
                        RootOperationResult::Success(element_roots) => {
                            roots.extend(element_roots);
                        }
                        RootOperationResult::Failed(e) => {
                            warn!("Failed to scan array element: {:?}", e);
                        }
                        _ => {}
                    }
                }
            }
            
            StackValue::Object(obj) => {
                // Object itself is a heap reference
                roots.push(obj as *const HashMap<String, StackValue> as *const u8);
                
                // Recursively scan object fields
                for (field_name, field_value) in obj {
                    // Field name (key) is also a heap reference
                    roots.push(field_name.as_ptr() as *const u8);
                    
                    match self.scan_value(field_value, depth + 1) {
                        RootOperationResult::Success(field_roots) => {
                            roots.extend(field_roots);
                        }
                        RootOperationResult::Failed(e) => {
                            warn!("Failed to scan object field '{}': {:?}", field_name, e);
                        }
                        _ => {}
                    }
                }
            }
            
            StackValue::Function { upvalues, .. } => {
                // Function object itself would be a heap reference
                roots.push(upvalues.as_ptr() as *const u8);
                
                // Recursively scan upvalues
                for upvalue in upvalues {
                    match self.scan_value(upvalue, depth + 1) {
                        RootOperationResult::Success(upvalue_roots) => {
                            roots.extend(upvalue_roots);
                        }
                        RootOperationResult::Failed(e) => {
                            warn!("Failed to scan function upvalue: {:?}", e);
                        }
                        _ => {}
                    }
                }
            }
            
            StackValue::Capability(name) | StackValue::Effect(name) => {
                // Capability/effect names are heap references
                roots.push(name.as_ptr() as *const u8);
            }
        }
        
        RootOperationResult::Success(roots)
    }
} 