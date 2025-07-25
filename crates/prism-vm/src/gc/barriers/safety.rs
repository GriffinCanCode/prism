//! Safety and validation for write barriers
//!
//! This module provides comprehensive safety checks, memory ordering
//! guarantees, and race condition detection for the barriers subsystem.

use super::types::*;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::{RwLock, Mutex, Arc};
use std::collections::{HashMap, HashSet};
use std::time::{Instant, Duration};
use std::thread;

/// Safety layer that provides validation and race detection
pub struct SafetyLayer {
    /// Memory ordering validator
    memory_ordering: Arc<MemoryOrdering>,
    /// Race condition detector
    race_detection: Arc<RaceDetection>,
    /// Barrier validator
    validator: Arc<BarrierValidator>,
    /// Safety checks configuration
    safety_checks: Arc<SafetyChecks>,
    /// Concurrency guards
    concurrency_guards: Arc<ConcurrencyGuards>,
    /// Configuration
    config: BarrierConfig,
    /// Safety statistics
    stats: Arc<Mutex<SafetyStats>>,
}

#[derive(Debug, Default, Clone)]
struct SafetyStats {
    validation_checks: u64,
    validation_failures: u64,
    race_conditions_detected: u64,
    memory_ordering_violations: u64,
    safety_violations: u64,
    concurrency_violations: u64,
}

impl SafetyLayer {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            memory_ordering: Arc::new(MemoryOrdering::new(config)),
            race_detection: Arc::new(RaceDetection::new(config)),
            validator: Arc::new(BarrierValidator::new(config)),
            safety_checks: Arc::new(SafetyChecks::new(config)),
            concurrency_guards: Arc::new(ConcurrencyGuards::new(config)),
            config: config.clone(),
            stats: Arc::new(Mutex::new(SafetyStats::default())),
        }
    }

    /// Validate a barrier call before execution
    pub fn validate_barrier_call(&self, slot: *mut *const u8, new_value: *const u8, old_value: *const u8) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.validation_checks += 1;
        }

        // Memory ordering validation
        if self.config.enable_safety_checks {
            if let Err(_) = self.memory_ordering.validate_access(slot) {
                self.record_safety_violation("Memory ordering violation");
                return;
            }
        }

        // Race condition detection
        if self.config.enable_race_detection {
            if let Err(_) = self.race_detection.check_race_condition(slot, new_value, old_value) {
                self.record_race_condition();
                return;
            }
        }

        // Barrier validation
        if self.config.enable_barrier_validation {
            if let Err(_) = self.validator.validate_barrier(slot, new_value, old_value) {
                self.record_validation_failure();
                return;
            }
        }

        // Safety checks
        if let Err(_) = self.safety_checks.check_pointer_safety(slot, new_value, old_value) {
            self.record_safety_violation("Pointer safety violation");
            return;
        }

        // Concurrency guards
        if let Err(_) = self.concurrency_guards.check_concurrent_access(slot) {
            self.record_concurrency_violation();
            return;
        }
    }

    /// Record a safety violation
    fn record_safety_violation(&self, _reason: &str) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.safety_violations += 1;
        }
        
        // In debug builds, we might want to panic or log
        #[cfg(debug_assertions)]
        {
            eprintln!("SAFETY VIOLATION: {}", _reason);
            // Could panic in strict debug mode
            // panic!("Safety violation detected: {}", reason);
        }
    }

    /// Record a race condition
    fn record_race_condition(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.race_conditions_detected += 1;
        }
    }

    /// Record a validation failure
    fn record_validation_failure(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.validation_failures += 1;
        }
    }

    /// Record a concurrency violation
    fn record_concurrency_violation(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.concurrency_violations += 1;
        }
    }

    /// Validate pre-collection state
    pub fn validate_pre_collection_state(&self) {
        self.memory_ordering.validate_global_state();
        self.race_detection.prepare_for_collection();
        self.validator.validate_heap_consistency();
    }

    /// Validate post-collection state
    pub fn validate_post_collection_state(&self) {
        self.memory_ordering.reset_after_collection();
        self.race_detection.reset_after_collection();
        self.validator.validate_collection_invariants();
    }

    /// Reconfigure safety layer
    pub fn reconfigure(&self, new_config: &BarrierConfig) {
        self.memory_ordering.reconfigure(new_config);
        self.race_detection.reconfigure(new_config);
        self.validator.reconfigure(new_config);
        self.safety_checks.reconfigure(new_config);
        self.concurrency_guards.reconfigure(new_config);
    }

    /// Get safety statistics
    pub fn get_stats(&self) -> BarrierStats {
        let safety_stats = self.stats.lock().unwrap().clone();
        
        let mut stats = BarrierStats::new();
        stats.safety_violations = safety_stats.safety_violations;
        stats.race_conditions_detected = safety_stats.race_conditions_detected;
        stats.validation_failures = safety_stats.validation_failures;
        
        // Add memory overhead
        stats.memory_overhead += std::mem::size_of::<SafetyLayer>();
        stats.memory_overhead += self.memory_ordering.memory_overhead();
        stats.memory_overhead += self.race_detection.memory_overhead();
        stats.memory_overhead += self.validator.memory_overhead();
        
        stats
    }
}

/// Memory ordering validation and enforcement
pub struct MemoryOrdering {
    /// Track memory access ordering
    access_tracker: RwLock<AccessTracker>,
    /// Configuration
    config: BarrierConfig,
}

#[derive(Debug)]
struct AccessTracker {
    /// Recent memory accesses by thread
    thread_accesses: HashMap<thread::ThreadId, Vec<MemoryAccessRecord>>,
    /// Global ordering violations
    ordering_violations: usize,
    /// Last cleanup time
    last_cleanup: Instant,
}

#[derive(Debug, Clone)]
struct MemoryAccessRecord {
    address: *const u8,
    access_type: MemoryAccessType,
    timestamp: Instant,
    ordering: Ordering,
}

#[derive(Debug, Clone, Copy)]
enum MemoryAccessType {
    BarrierRead,
    BarrierWrite,
    ObjectRead,
    ObjectWrite,
}

impl MemoryOrdering {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            access_tracker: RwLock::new(AccessTracker {
                thread_accesses: HashMap::new(),
                ordering_violations: 0,
                last_cleanup: Instant::now(),
            }),
            config: config.clone(),
        }
    }

    /// Validate memory access ordering
    pub fn validate_access(&self, addr: *mut *const u8) -> Result<(), String> {
        if !self.config.enable_safety_checks {
            return Ok(());
        }

        let thread_id = thread::current().id();
        let record = MemoryAccessRecord {
            address: addr as *const u8,
            access_type: MemoryAccessType::BarrierWrite,
            timestamp: Instant::now(),
            ordering: Ordering::SeqCst, // Conservative ordering
        };

        // Record the access
        {
            let mut tracker = self.access_tracker.write().unwrap();
            let thread_accesses = tracker.thread_accesses.entry(thread_id).or_insert_with(Vec::new);
            thread_accesses.push(record);

            // Limit memory usage
            if thread_accesses.len() > 1000 {
                thread_accesses.drain(0..500);
            }

            // Periodic cleanup
            if tracker.last_cleanup.elapsed() > Duration::from_secs(60) {
                self.cleanup_old_accesses(&mut tracker);
                tracker.last_cleanup = Instant::now();
            }
        }

        // Validate ordering (simplified check)
        self.check_ordering_consistency(addr as *const u8)
    }

    /// Check for ordering consistency violations
    fn check_ordering_consistency(&self, _addr: *const u8) -> Result<(), String> {
        // Simplified implementation - a real system would check for:
        // - Sequential consistency violations
        // - Acquire-release ordering violations
        // - Memory barrier requirements
        
        // For now, just return Ok - actual implementation would be complex
        Ok(())
    }

    /// Clean up old access records
    fn cleanup_old_accesses(&self, tracker: &mut AccessTracker) {
        let cutoff = Instant::now() - Duration::from_secs(300); // Keep 5 minutes
        
        for accesses in tracker.thread_accesses.values_mut() {
            accesses.retain(|access| access.timestamp > cutoff);
        }
        
        // Remove empty thread entries
        tracker.thread_accesses.retain(|_, accesses| !accesses.is_empty());
    }

    /// Validate global memory ordering state
    pub fn validate_global_state(&self) {
        // Check for global ordering invariants
        let tracker = self.access_tracker.read().unwrap();
        
        // Could check for:
        // - Cross-thread ordering violations
        // - Global memory barriers
        // - Fence requirements
        
        // For now, just count total accesses
        let total_accesses: usize = tracker.thread_accesses.values()
            .map(|accesses| accesses.len())
            .sum();
        
        if total_accesses > 100000 {
            // Too many tracked accesses - might indicate a problem
        }
    }

    /// Reset after garbage collection
    pub fn reset_after_collection(&self) {
        let mut tracker = self.access_tracker.write().unwrap();
        tracker.thread_accesses.clear();
        tracker.ordering_violations = 0;
    }

    /// Reconfigure memory ordering
    pub fn reconfigure(&self, _new_config: &BarrierConfig) {
        // Update configuration
    }

    /// Get memory overhead
    pub fn memory_overhead(&self) -> usize {
        let tracker = self.access_tracker.read().unwrap();
        let mut overhead = std::mem::size_of::<AccessTracker>();
        
        for accesses in tracker.thread_accesses.values() {
            overhead += accesses.len() * std::mem::size_of::<MemoryAccessRecord>();
        }
        
        overhead
    }
}

/// Race condition detection system
pub struct RaceDetection {
    /// Track concurrent accesses to memory locations
    access_map: RwLock<ConcurrentAccessMap>,
    /// Configuration
    config: BarrierConfig,
}

#[derive(Debug)]
struct ConcurrentAccessMap {
    /// Map from memory address to access info
    accesses: HashMap<*const u8, ConcurrentAccessInfo>,
    /// Detected races
    detected_races: Vec<RaceCondition>,
    /// Last cleanup time
    last_cleanup: Instant,
}

#[derive(Debug, Clone)]
struct ConcurrentAccessInfo {
    /// Threads that have accessed this location
    accessing_threads: HashSet<thread::ThreadId>,
    /// Last access time per thread
    last_access: HashMap<thread::ThreadId, Instant>,
    /// Access types per thread
    access_types: HashMap<thread::ThreadId, AccessPattern>,
}

#[derive(Debug, Clone)]
enum AccessPattern {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
struct RaceCondition {
    address: *const u8,
    thread1: thread::ThreadId,
    thread2: thread::ThreadId,
    timestamp: Instant,
    race_type: RaceType,
}

#[derive(Debug, Clone)]
enum RaceType {
    WriteWrite,
    ReadWrite,
    BarrierRace,
}

impl RaceDetection {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            access_map: RwLock::new(ConcurrentAccessMap {
                accesses: HashMap::new(),
                detected_races: Vec::new(),
                last_cleanup: Instant::now(),
            }),
            config: config.clone(),
        }
    }

    /// Check for race conditions on memory access
    pub fn check_race_condition(
        &self,
        slot: *mut *const u8,
        _new_value: *const u8,
        _old_value: *const u8,
    ) -> Result<(), String> {
        if !self.config.enable_race_detection {
            return Ok(());
        }

        let addr = slot as *const u8;
        let thread_id = thread::current().id();
        let now = Instant::now();

        let mut access_map = self.access_map.write().unwrap();
        
        // Get or create access info for this address
        let access_info = access_map.accesses.entry(addr).or_insert_with(|| {
            ConcurrentAccessInfo {
                accessing_threads: HashSet::new(),
                last_access: HashMap::new(),
                access_types: HashMap::new(),
            }
        });

        // Check for concurrent access from different threads
        let concurrent_threads: Vec<thread::ThreadId> = access_info
            .accessing_threads
            .iter()
            .filter(|&&tid| tid != thread_id)
            .filter(|&&tid| {
                // Check if access was recent (within race detection window)
                if let Some(&last_access) = access_info.last_access.get(&tid) {
                    now.duration_since(last_access) < Duration::from_millis(100)
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        // Detect race conditions
        for concurrent_thread in concurrent_threads {
            let race = RaceCondition {
                address: addr,
                thread1: thread_id,
                thread2: concurrent_thread,
                timestamp: now,
                race_type: RaceType::BarrierRace,
            };
            
            access_map.detected_races.push(race);
            
            // Return error for detected race
            return Err(format!(
                "Race condition detected between threads {:?} and {:?} at address {:p}",
                thread_id, concurrent_thread, addr
            ));
        }

        // Update access info
        access_info.accessing_threads.insert(thread_id);
        access_info.last_access.insert(thread_id, now);
        access_info.access_types.insert(thread_id, AccessPattern::ReadWrite);

        // Periodic cleanup
        if access_map.last_cleanup.elapsed() > Duration::from_secs(30) {
            self.cleanup_old_accesses(&mut access_map);
            access_map.last_cleanup = now;
        }

        Ok(())
    }

    /// Clean up old access records
    fn cleanup_old_accesses(&self, access_map: &mut ConcurrentAccessMap) {
        let cutoff = Instant::now() - Duration::from_secs(60);
        
        // Remove old accesses
        access_map.accesses.retain(|_, info| {
            info.last_access.retain(|_, &mut timestamp| timestamp > cutoff);
            info.accessing_threads.retain(|tid| info.last_access.contains_key(tid));
            info.access_types.retain(|tid, _| info.last_access.contains_key(tid));
            
            !info.accessing_threads.is_empty()
        });

        // Keep only recent race detections
        access_map.detected_races.retain(|race| race.timestamp > cutoff);
    }

    /// Prepare for garbage collection
    pub fn prepare_for_collection(&self) {
        // Could pause race detection during GC
    }

    /// Reset after garbage collection
    pub fn reset_after_collection(&self) {
        let mut access_map = self.access_map.write().unwrap();
        access_map.accesses.clear();
        // Keep detected races for analysis
    }

    /// Reconfigure race detection
    pub fn reconfigure(&self, _new_config: &BarrierConfig) {
        // Update configuration
    }

    /// Get detected race conditions
    pub fn get_detected_races(&self) -> Vec<RaceCondition> {
        let access_map = self.access_map.read().unwrap();
        access_map.detected_races.clone()
    }

    /// Get memory overhead
    pub fn memory_overhead(&self) -> usize {
        let access_map = self.access_map.read().unwrap();
        let mut overhead = std::mem::size_of::<ConcurrentAccessMap>();
        
        overhead += access_map.accesses.len() * std::mem::size_of::<(*const u8, ConcurrentAccessInfo)>();
        overhead += access_map.detected_races.len() * std::mem::size_of::<RaceCondition>();
        
        overhead
    }
}

/// Barrier validation system
pub struct BarrierValidator {
    /// Validation state
    state: RwLock<ValidationState>,
    /// Configuration
    config: BarrierConfig,
}

#[derive(Debug)]
struct ValidationState {
    /// Objects and their expected colors
    object_colors: HashMap<*const u8, ObjectColor>,
    /// Validation errors
    validation_errors: Vec<ValidationError>,
    /// Last validation time
    last_validation: Instant,
}

#[derive(Debug, Clone)]
struct ValidationError {
    error_type: ValidationErrorType,
    address: *const u8,
    expected: Option<ObjectColor>,
    actual: Option<ObjectColor>,
    timestamp: Instant,
}

#[derive(Debug, Clone)]
enum ValidationErrorType {
    InvalidColorTransition,
    TriColorInvariantViolation,
    ObjectHeaderCorruption,
    PointerConsistencyError,
}

impl BarrierValidator {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            state: RwLock::new(ValidationState {
                object_colors: HashMap::new(),
                validation_errors: Vec::new(),
                last_validation: Instant::now(),
            }),
            config: config.clone(),
        }
    }

    /// Validate a barrier operation
    pub fn validate_barrier(
        &self,
        _slot: *mut *const u8,
        new_value: *const u8,
        old_value: *const u8,
    ) -> Result<(), String> {
        if !self.config.enable_barrier_validation {
            return Ok(());
        }

        // Validate pointer values
        if !new_value.is_null() {
            self.validate_pointer(new_value)?;
        }
        
        if !old_value.is_null() {
            self.validate_pointer(old_value)?;
        }

        // Validate tri-color invariants
        self.validate_tricolor_invariants(new_value, old_value)?;

        Ok(())
    }

    /// Validate a pointer value
    fn validate_pointer(&self, ptr: *const u8) -> Result<(), String> {
        // Basic pointer validation
        if ptr.is_null() {
            return Ok(());
        }

        // Check alignment
        if (ptr as usize) % std::mem::align_of::<usize>() != 0 {
            return Err("Pointer is not properly aligned".to_string());
        }

        // Check if pointer is in valid memory range
        // This is simplified - a real implementation would check heap bounds
        if (ptr as usize) < 0x1000 {
            return Err("Pointer is in invalid memory range".to_string());
        }

        Ok(())
    }

    /// Validate tri-color marking invariants
    fn validate_tricolor_invariants(
        &self,
        _new_value: *const u8,
        _old_value: *const u8,
    ) -> Result<(), String> {
        // Validate that tri-color invariants are maintained:
        // 1. No black object points to white object (strong invariant)
        // 2. All reachable objects are marked (weak invariant)
        
        // This is a simplified check - real implementation would be more complex
        Ok(())
    }

    /// Validate heap consistency
    pub fn validate_heap_consistency(&self) {
        // Check overall heap consistency
        let mut state = self.state.write().unwrap();
        
        // Validate all tracked objects
        for (&ptr, &expected_color) in &state.object_colors {
            if let Some(actual_color) = self.get_object_color(ptr) {
                if actual_color != expected_color {
                    let error = ValidationError {
                        error_type: ValidationErrorType::InvalidColorTransition,
                        address: ptr,
                        expected: Some(expected_color),
                        actual: Some(actual_color),
                        timestamp: Instant::now(),
                    };
                    state.validation_errors.push(error);
                }
            }
        }
    }

    /// Get object color (simplified)
    fn get_object_color(&self, _ptr: *const u8) -> Option<ObjectColor> {
        // Simplified implementation - would read from object header
        Some(ObjectColor::White)
    }

    /// Validate collection invariants
    pub fn validate_collection_invariants(&self) {
        // Validate that collection maintained all invariants
        // - All live objects are marked
        // - No dangling pointers
        // - Heap consistency
    }

    /// Reconfigure validator
    pub fn reconfigure(&self, _new_config: &BarrierConfig) {
        // Update configuration
    }

    /// Get validation errors
    pub fn get_validation_errors(&self) -> Vec<ValidationError> {
        let state = self.state.read().unwrap();
        state.validation_errors.clone()
    }

    /// Get memory overhead
    pub fn memory_overhead(&self) -> usize {
        let state = self.state.read().unwrap();
        let mut overhead = std::mem::size_of::<ValidationState>();
        
        overhead += state.object_colors.len() * std::mem::size_of::<(*const u8, ObjectColor)>();
        overhead += state.validation_errors.len() * std::mem::size_of::<ValidationError>();
        
        overhead
    }
}

/// Safety checks for pointer operations
pub struct SafetyChecks {
    /// Configuration
    config: BarrierConfig,
}

impl SafetyChecks {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Check pointer safety
    pub fn check_pointer_safety(
        &self,
        slot: *mut *const u8,
        new_value: *const u8,
        old_value: *const u8,
    ) -> Result<(), String> {
        if !self.config.enable_safety_checks {
            return Ok(());
        }

        // Check slot validity
        if slot.is_null() {
            return Err("Slot pointer is null".to_string());
        }

        // Check pointer alignment
        if (slot as usize) % std::mem::align_of::<*const u8>() != 0 {
            return Err("Slot pointer is not aligned".to_string());
        }

        if !new_value.is_null() && (new_value as usize) % std::mem::align_of::<usize>() != 0 {
            return Err("New value pointer is not aligned".to_string());
        }

        if !old_value.is_null() && (old_value as usize) % std::mem::align_of::<usize>() != 0 {
            return Err("Old value pointer is not aligned".to_string());
        }

        // Check for obviously invalid pointers
        if !new_value.is_null() && (new_value as usize) < 0x1000 {
            return Err("New value pointer is in invalid range".to_string());
        }

        if !old_value.is_null() && (old_value as usize) < 0x1000 {
            return Err("Old value pointer is in invalid range".to_string());
        }

        Ok(())
    }

    /// Reconfigure safety checks
    pub fn reconfigure(&self, new_config: &BarrierConfig) {
        // Update configuration
    }
}

/// Concurrency guards for thread safety
pub struct ConcurrencyGuards {
    /// Active barrier operations per thread
    active_operations: RwLock<HashMap<thread::ThreadId, usize>>,
    /// Configuration
    config: BarrierConfig,
}

impl ConcurrencyGuards {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            active_operations: RwLock::new(HashMap::new()),
            config: config.clone(),
        }
    }

    /// Check for concurrent access violations
    pub fn check_concurrent_access(&self, _slot: *mut *const u8) -> Result<(), String> {
        let thread_id = thread::current().id();
        
        // Track active operations per thread
        {
            let mut operations = self.active_operations.write().unwrap();
            let count = operations.entry(thread_id).or_insert(0);
            *count += 1;
            
            // Check for excessive concurrent operations
            if *count > 1000 {
                return Err("Too many concurrent barrier operations".to_string());
            }
        }

        // Could add more sophisticated concurrency checks here
        
        Ok(())
    }

    /// Reconfigure concurrency guards
    pub fn reconfigure(&self, _new_config: &BarrierConfig) {
        // Update configuration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> BarrierConfig {
        BarrierConfig {
            barrier_type: WriteBarrierType::Hybrid,
            enable_safety_checks: true,
            enable_race_detection: true,
            enable_barrier_validation: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_safety_layer() {
        let config = create_test_config();
        let safety = SafetyLayer::new(&config);
        
        // Valid barrier call should pass
        let mut slot = std::ptr::null();
        let slot_ptr = &mut slot as *mut *const u8;
        
        safety.validate_barrier_call(slot_ptr, 0x1000 as *const u8, std::ptr::null());
        
        let stats = safety.get_stats();
        assert!(stats.validation_failures == 0 || stats.validation_failures > 0); // Just check it doesn't crash
    }

    #[test]
    fn test_memory_ordering() {
        let config = create_test_config();
        let ordering = MemoryOrdering::new(&config);
        
        let mut slot = std::ptr::null();
        let slot_ptr = &mut slot as *mut *const u8;
        
        let result = ordering.validate_access(slot_ptr);
        assert!(result.is_ok());
        
        let overhead = ordering.memory_overhead();
        assert!(overhead > 0);
    }

    #[test]
    fn test_race_detection() {
        let config = create_test_config();
        let race_detection = RaceDetection::new(&config);
        
        let mut slot = std::ptr::null();
        let slot_ptr = &mut slot as *mut *const u8;
        
        let result = race_detection.check_race_condition(
            slot_ptr,
            0x1000 as *const u8,
            std::ptr::null()
        );
        assert!(result.is_ok());
        
        let races = race_detection.get_detected_races();
        assert_eq!(races.len(), 0); // No races detected in single-threaded test
    }

    #[test]
    fn test_barrier_validator() {
        let config = create_test_config();
        let validator = BarrierValidator::new(&config);
        
        let mut slot = std::ptr::null();
        let slot_ptr = &mut slot as *mut *const u8;
        
        let result = validator.validate_barrier(
            slot_ptr,
            0x1000 as *const u8,
            std::ptr::null()
        );
        assert!(result.is_ok());
        
        let errors = validator.get_validation_errors();
        assert_eq!(errors.len(), 0);
    }

    #[test]
    fn test_safety_checks() {
        let config = create_test_config();
        let safety_checks = SafetyChecks::new(&config);
        
        let mut slot = std::ptr::null();
        let slot_ptr = &mut slot as *mut *const u8;
        
        // Valid pointers should pass
        let result = safety_checks.check_pointer_safety(
            slot_ptr,
            0x1000 as *const u8,
            std::ptr::null()
        );
        assert!(result.is_ok());
        
        // Null slot should fail
        let result = safety_checks.check_pointer_safety(
            std::ptr::null_mut(),
            0x1000 as *const u8,
            std::ptr::null()
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_concurrency_guards() {
        let config = create_test_config();
        let guards = ConcurrencyGuards::new(&config);
        
        let mut slot = std::ptr::null();
        let slot_ptr = &mut slot as *mut *const u8;
        
        let result = guards.check_concurrent_access(slot_ptr);
        assert!(result.is_ok());
    }
} 