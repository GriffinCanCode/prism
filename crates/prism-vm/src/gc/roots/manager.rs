//! Central Root Manager for coordinating all root sources
//!
//! This module implements the main RootManager that coordinates between
//! different root sources (stack, global, JIT, etc.) and provides a unified
//! interface for garbage collection.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use super::{
    types::*, interfaces::*,
    stack_scanner::StackScanner,
    global_roots::GlobalRootManager,
    platform_scanner::PlatformStackScanner,
    security::RootSecurityManager,
    analytics::RootAnalytics,
};
use prism_runtime::authority::capability::CapabilitySet;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, span, Level};

/// Central root manager that coordinates all root sources
/// 
/// This is the main entry point for root set management, coordinating
/// between different root sources while maintaining backward compatibility
/// with the existing GC interface.
pub struct RootManager {
    /// Configuration for root management
    config: Arc<RwLock<RootManagerConfig>>,
    
    /// Stack scanner for execution stack roots
    stack_scanner: Arc<Mutex<StackScanner>>,
    
    /// Global root manager for static data
    global_manager: Arc<Mutex<GlobalRootManager>>,
    
    /// Platform-specific stack scanner
    platform_scanner: Arc<Mutex<PlatformStackScanner>>,
    
    /// Security manager for capability-based access
    security_manager: Arc<Mutex<RootSecurityManager>>,
    
    /// Analytics and monitoring
    analytics: Arc<Mutex<RootAnalytics>>,
    
    /// Manually registered roots (for backward compatibility)
    manual_roots: Arc<RwLock<HashSet<*const u8>>>,
    
    /// Root entry metadata
    root_metadata: Arc<RwLock<HashMap<*const u8, RootEntry>>>,
    
    /// Performance statistics
    statistics: Arc<RwLock<RootStatistics>>,
    
    /// Thread-local caches for performance
    thread_caches: Arc<RwLock<HashMap<std::thread::ThreadId, ThreadRootCache>>>,
}

/// Thread-local cache for root operations
#[derive(Debug)]
struct ThreadRootCache {
    /// Cached roots for this thread
    cached_roots: HashSet<*const u8>,
    /// Cache timestamp
    last_updated: Instant,
    /// Cache hit statistics
    hits: u64,
    /// Cache miss statistics
    misses: u64,
}

impl ThreadRootCache {
    fn new() -> Self {
        Self {
            cached_roots: HashSet::new(),
            last_updated: Instant::now(),
            hits: 0,
            misses: 0,
        }
    }
    
    fn is_cache_valid(&self, max_age: Duration) -> bool {
        self.last_updated.elapsed() < max_age
    }
    
    fn invalidate(&mut self) {
        self.cached_roots.clear();
        self.last_updated = Instant::now();
    }
}

impl RootManager {
    /// Create a new root manager with default configuration
    pub fn new() -> VMResult<Self> {
        Self::with_config(RootManagerConfig::default())
    }
    
    /// Create a new root manager with custom configuration
    pub fn with_config(config: RootManagerConfig) -> VMResult<Self> {
        let _span = span!(Level::INFO, "root_manager_init").entered();
        info!("Initializing root manager with config: {:?}", config);
        
        let config = Arc::new(RwLock::new(config));
        
        // Initialize subsystems
        let stack_scanner = Arc::new(Mutex::new(
            StackScanner::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create stack scanner: {}", e),
            })?
        ));
        
        let global_manager = Arc::new(Mutex::new(
            GlobalRootManager::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create global root manager: {}", e),
            })?
        ));
        
        let platform_scanner = Arc::new(Mutex::new(
            PlatformStackScanner::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create platform scanner: {}", e),
            })?
        ));
        
        let security_manager = Arc::new(Mutex::new(
            RootSecurityManager::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create security manager: {}", e),
            })?
        ));
        
        let analytics = Arc::new(Mutex::new(
            RootAnalytics::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create analytics: {}", e),
            })?
        ));
        
        Ok(Self {
            config,
            stack_scanner,
            global_manager,
            platform_scanner,
            security_manager,
            analytics,
            manual_roots: Arc::new(RwLock::new(HashSet::new())),
            root_metadata: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(RootStatistics::default())),
            thread_caches: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Get or create thread-local cache
    fn get_thread_cache(&self) -> ThreadRootCache {
        let thread_id = std::thread::current().id();
        let mut caches = self.thread_caches.write().unwrap();
        
        caches.entry(thread_id)
            .or_insert_with(ThreadRootCache::new)
            .clone()
    }
    
    /// Invalidate thread caches
    fn invalidate_thread_caches(&self) {
        let mut caches = self.thread_caches.write().unwrap();
        for cache in caches.values_mut() {
            cache.invalidate();
        }
    }

    /// Create a security context for a given operation
    fn create_security_context_for_operation(&self, source: &RootSource) -> SecurityContext {
        use prism_runtime::authority::capability::*;
        use std::time::Duration;
        
        let mut capabilities = CapabilitySet::new();
        let component_id = ComponentId::new(1); // VM component ID
        
        match source {
            RootSource::Manual => {
                // Manual roots have comprehensive access for debugging/testing
                let memory_authority = Authority::Memory(MemoryAuthority {
                    max_allocation: usize::MAX,
                    allowed_regions: vec![MemoryRegion::new(
                        0,
                        usize::MAX,
                        [MemoryOperationType::Read, MemoryOperationType::Write, 
                         MemoryOperationType::Execute, MemoryOperationType::Allocate,
                         MemoryOperationType::Deallocate].iter().cloned().collect()
                    )],
                });
                
                let capability = Capability::new(
                    memory_authority,
                    ConstraintSet::new(),
                    Duration::from_secs(3600), // 1 hour
                    component_id,
                );
                capabilities.add(capability);
            }
            
            RootSource::GlobalVariables => {
                // Global variables are read-only memory access
                let memory_authority = Authority::Memory(MemoryAuthority {
                    max_allocation: 0, // No allocation allowed
                    allowed_regions: vec![MemoryRegion::new(
                        0,
                        usize::MAX,
                        [MemoryOperationType::Read].iter().cloned().collect()
                    )],
                });
                
                let capability = Capability::new(
                    memory_authority,
                    ConstraintSet::new(),
                    Duration::from_secs(86400), // 24 hours for globals
                    component_id,
                );
                capabilities.add(capability);
            }
            
            RootSource::CapabilityTokens => {
                // Capability tokens are read-only system access
                let system_authority = Authority::System(SystemAuthority {
                    operations: [SystemOperation::EnvironmentRead].iter().cloned().collect(),
                    allowed_env_vars: vec!["PRISM_*".to_string()],
                });
                
                let capability = Capability::new(
                    system_authority,
                    ConstraintSet::new(),
                    Duration::from_secs(86400), // 24 hours for capability tokens
                    component_id,
                );
                capabilities.add(capability);
            }
            
            RootSource::ExecutionStack => {
                // Stack access requires memory read/write
                let memory_authority = Authority::Memory(MemoryAuthority {
                    max_allocation: 1024 * 1024, // 1MB for stack operations
                    allowed_regions: vec![MemoryRegion::new(
                        0,
                        usize::MAX,
                        [MemoryOperationType::Read, MemoryOperationType::Write].iter().cloned().collect()
                    )],
                });
                
                let capability = Capability::new(
                    memory_authority,
                    ConstraintSet::new(),
                    Duration::from_secs(3600), // 1 hour
                    component_id,
                );
                capabilities.add(capability);
            }
            
            _ => {
                // Default minimal access for unknown sources
                // No capabilities added - will fail validation if not appropriate
            }
        }
        
        SecurityContext {
            capabilities,
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
    
    /// Create a security context for a removal operation
    fn create_security_context_for_removal(&self) -> SecurityContext {
        use prism_runtime::authority::capability::*;
        use std::time::Duration;
        
        let mut capabilities = CapabilitySet::new();
        let component_id = ComponentId::new(1); // VM component ID
        
        // Removal operations require memory management authority
        let memory_authority = Authority::Memory(MemoryAuthority {
            max_allocation: 0, // No allocation for removal
            allowed_regions: vec![MemoryRegion::new(
                0,
                usize::MAX,
                [MemoryOperationType::Read, MemoryOperationType::Deallocate].iter().cloned().collect()
            )],
        });
        
        let capability = Capability::new(
            memory_authority,
            ConstraintSet::new(),
            Duration::from_secs(3600), // 1 hour
            component_id,
        );
        capabilities.add(capability);
        
        SecurityContext {
            capabilities,
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
}

impl RootSetInterface for RootManager {
    /// Add a root object to the set
    fn add_root(&mut self, ptr: *const u8, root_type: RootType, source: RootSource) -> RootOperationResult<()> {
        let _span = span!(Level::DEBUG, "add_root", ptr = ?ptr, source = ?source).entered();
        let start_time = Instant::now();
        
        // Validate pointer
        if ptr.is_null() {
            return RootOperationResult::Failed(RootError::InvalidPointer { 
                ptr: ptr as usize, 
                reason: "Null pointer".to_string() 
            });
        }
        
        // Create security context with appropriate capabilities
        let security_context = self.create_security_context_for_operation(&source);
        
        let operation = RootSecurityOperation::AddRoot { ptr, source };
        if let Err(e) = self.security_manager.lock().unwrap()
            .validate_security_context(&operation, &security_context) {
            return RootOperationResult::SecurityDenied(format!("Security validation failed: {:?}", e));
        }
        
        // Check for duplicates
        {
            let manual_roots = self.manual_roots.read().unwrap();
            if manual_roots.contains(&ptr) {
                return RootOperationResult::Failed(RootError::DuplicateRoot { ptr: ptr as usize });
            }
        }
        
        // Create root entry
        let root_entry = RootEntry {
            ptr,
            source,
            root_type,
            registered_at: Instant::now(),
            thread_id: std::thread::current().id(),
            security_context,
            metadata: RootMetadata {
                description: None,
                tags: Vec::new(),
                business_context: None,
                performance_hints: PerformanceHints {
                    scan_frequency: ScanFrequency::EveryGC,
                    access_pattern: AccessPattern::Unknown,
                    locality_hints: LocalityHints {
                        prefer_cache_friendly: true,
                        numa_node: None,
                        cache_level: CacheLevel::L2,
                    },
                    scan_priority: ScanPriority::Normal,
                },
                debug_info: None,
            },
        };
        
        // Add to appropriate collection
        match source {
            RootSource::Manual => {
                let mut manual_roots = self.manual_roots.write().unwrap();
                manual_roots.insert(ptr);
            }
            RootSource::GlobalVariables => {
                // For global variables, we need to extract name and type information
                // and delegate to the global manager
                if let RootType::GlobalVariable { name, var_type } = &root_type {
                    return match self.global_manager.lock().unwrap()
                        .register_global(ptr, name.clone(), var_type.clone()) {
                        RootOperationResult::Success(_) => RootOperationResult::Success(()),
                        other => other,
                    };
                } else {
                    return RootOperationResult::Failed(RootError::ConfigurationError {
                        error: "Global variable source requires GlobalVariable root type".to_string()
                    });
                }
            }
            RootSource::CapabilityTokens => {
                // For capability tokens, delegate to global manager
                if let RootType::Capability { name } = &root_type {
                    return match self.global_manager.lock().unwrap()
                        .register_capability(ptr, name.clone()) {
                        RootOperationResult::Success(_) => RootOperationResult::Success(()),
                        other => other,
                    };
                } else {
                    return RootOperationResult::Failed(RootError::ConfigurationError {
                        error: "Capability token source requires Capability root type".to_string()
                    });
                }
            }
            _ => {
                // Other sources are handled by their respective managers
                return RootOperationResult::Failed(RootError::ConfigurationError {
                    error: format!("Source {:?} should use specialized manager", source)
                });
            }
        }
        
        // Store metadata
        {
            let mut metadata = self.root_metadata.write().unwrap();
            metadata.insert(ptr, root_entry);
        }
        
        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.total_roots += 1;
            *stats.roots_by_source.entry(source).or_insert(0) += 1;
        }
        
        // Record analytics
        if self.config.read().unwrap().enable_analytics {
            let operation = RootAnalyticsOperation {
                timestamp: start_time,
                operation_type: AnalyticsOperationType::AddRoot,
                duration: start_time.elapsed(),
                roots_affected: 1,
                memory_delta: std::mem::size_of::<RootEntry>() as i64,
                success: true,
                metrics: HashMap::new(),
            };
            
            if let Err(e) = self.analytics.lock().unwrap().record_operation(&operation) {
                warn!("Failed to record analytics: {:?}", e);
            }
        }
        
        // Invalidate caches
        self.invalidate_thread_caches();
        
        debug!("Added root: ptr={:?}, source={:?}", ptr, source);
        RootOperationResult::Success(())
    }
    
    /// Remove a root object from the set
    fn remove_root(&mut self, ptr: *const u8) -> RootOperationResult<()> {
        let _span = span!(Level::DEBUG, "remove_root", ptr = ?ptr).entered();
        let start_time = Instant::now();
        
        // Create security context for removal operation
        let security_context = self.create_security_context_for_removal();
        
        let operation = RootSecurityOperation::RemoveRoot { ptr };
        if let Err(e) = self.security_manager.lock().unwrap()
            .validate_security_context(&operation, &security_context) {
            return RootOperationResult::SecurityDenied(format!("Security validation failed: {:?}", e));
        }
        
        // Find and remove from appropriate collection
        let mut found = false;
        let mut source = RootSource::Manual;
        
        // Check manual roots
        {
            let mut manual_roots = self.manual_roots.write().unwrap();
            if manual_roots.remove(&ptr) {
                found = true;
                source = RootSource::Manual;
            }
        }
        
        if !found {
            return RootOperationResult::Failed(RootError::RootNotFound { ptr: ptr as usize });
        }
        
        // Remove metadata
        {
            let mut metadata = self.root_metadata.write().unwrap();
            metadata.remove(&ptr);
        }
        
        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.total_roots = stats.total_roots.saturating_sub(1);
            if let Some(count) = stats.roots_by_source.get_mut(&source) {
                *count = count.saturating_sub(1);
            }
        }
        
        // Record analytics
        if self.config.read().unwrap().enable_analytics {
            let operation = RootAnalyticsOperation {
                timestamp: start_time,
                operation_type: AnalyticsOperationType::RemoveRoot,
                duration: start_time.elapsed(),
                roots_affected: 1,
                memory_delta: -(std::mem::size_of::<RootEntry>() as i64),
                success: true,
                metrics: HashMap::new(),
            };
            
            if let Err(e) = self.analytics.lock().unwrap().record_operation(&operation) {
                warn!("Failed to record analytics: {:?}", e);
            }
        }
        
        // Invalidate caches
        self.invalidate_thread_caches();
        
        debug!("Removed root: ptr={:?}", ptr);
        RootOperationResult::Success(())
    }
    
    /// Check if a pointer is registered as a root
    fn contains_root(&self, ptr: *const u8) -> bool {
        // Check thread cache first
        let config = self.config.read().unwrap();
        if config.thread_cache_size > 0 {
            let thread_cache = self.get_thread_cache();
            if thread_cache.is_cache_valid(Duration::from_millis(100)) {
                return thread_cache.cached_roots.contains(&ptr);
            }
        }
        
        // Check manual roots
        {
            let manual_roots = self.manual_roots.read().unwrap();
            if manual_roots.contains(&ptr) {
                return true;
            }
        }
        
        // Check global roots
        match self.global_manager.lock().unwrap().get_global_roots() {
            RootOperationResult::Success(global_roots) => {
                if global_roots.contains(&ptr) {
                    return true;
                }
            }
            _ => {} // Continue checking other sources if global check fails
        }
        
        // Check capability roots
        match self.global_manager.lock().unwrap().get_capability_roots() {
            RootOperationResult::Success(capability_roots) => {
                if capability_roots.contains(&ptr) {
                    return true;
                }
            }
            _ => {} // Continue checking other sources
        }
        
        // Check effect roots
        match self.global_manager.lock().unwrap().get_effect_roots() {
            RootOperationResult::Success(effect_roots) => {
                if effect_roots.contains(&ptr) {
                    return true;
                }
            }
            _ => {} // Continue checking other sources
        }
        
        // Check if the pointer exists in root metadata (indicates it was registered)
        {
            let metadata = self.root_metadata.read().unwrap();
            if metadata.contains_key(&ptr) {
                return true;
            }
        }
        
        // If we have a stack scanner available, check if it's a current stack root
        // Note: This is expensive so we do it last
        if let Ok(current_stack) = std::panic::catch_unwind(|| {
            // In a real implementation, we'd need access to the current execution stack
            // For now, we'll skip this check as it requires VM context
            false
        }) {
            return current_stack;
        }
        
        false
    }
    
    /// Get all root objects for garbage collection
    fn get_all_roots(&self) -> RootOperationResult<Vec<*const u8>> {
        let _span = span!(Level::DEBUG, "get_all_roots").entered();
        let start_time = Instant::now();
        
        let mut all_roots = Vec::new();
        
        // Get manual roots
        {
            let manual_roots = self.manual_roots.read().unwrap();
            all_roots.extend(manual_roots.iter().cloned());
        }
        
        // Get stack roots
        match self.stack_scanner.lock().unwrap().scan_for_roots() {
            Ok(stack_roots) => {
                all_roots.extend(stack_roots);
            }
            Err(e) => {
                warn!("Failed to scan stack roots: {:?}", e);
            }
        }
        
        // Get global roots
        match self.global_manager.lock().unwrap().get_global_roots() {
            RootOperationResult::Success(global_roots) => {
                all_roots.extend(global_roots);
            }
            RootOperationResult::Failed(e) => {
                warn!("Failed to get global roots: {:?}", e);
            }
            _ => {}
        }
        
        // Record analytics
        if self.config.read().unwrap().enable_analytics {
            let operation = RootAnalyticsOperation {
                timestamp: start_time,
                operation_type: AnalyticsOperationType::GlobalScan,
                duration: start_time.elapsed(),
                roots_affected: all_roots.len(),
                memory_delta: 0,
                success: true,
                metrics: HashMap::new(),
            };
            
            if let Err(e) = self.analytics.lock().unwrap().record_operation(&operation) {
                warn!("Failed to record analytics: {:?}", e);
            }
        }
        
        debug!("Found {} total roots", all_roots.len());
        RootOperationResult::Success(all_roots)
    }
    
    /// Get roots by source type
    fn get_roots_by_source(&self, source: RootSource) -> RootOperationResult<Vec<*const u8>> {
        match source {
            RootSource::Manual => {
                let manual_roots = self.manual_roots.read().unwrap();
                RootOperationResult::Success(manual_roots.iter().cloned().collect())
            }
            RootSource::GlobalVariables => {
                self.global_manager.lock().unwrap().get_global_roots()
            }
            RootSource::CapabilityTokens => {
                self.global_manager.lock().unwrap().get_capability_roots()
            }
            _ => {
                RootOperationResult::Failed(RootError::ConfigurationError {
                    error: format!("Source {:?} not yet implemented", source)
                })
            }
        }
    }
    
    /// Scan for additional roots (e.g., stack scanning)
    fn scan_for_roots(&mut self) -> RootOperationResult<Vec<*const u8>> {
        let _span = span!(Level::DEBUG, "scan_for_roots").entered();
        let start_time = Instant::now();
        
        let mut discovered_roots = Vec::new();
        
        // Scan execution stack
        match self.stack_scanner.lock().unwrap().scan_for_roots() {
            Ok(stack_roots) => {
                discovered_roots.extend(stack_roots);
            }
            Err(e) => {
                warn!("Stack scanning failed: {:?}", e);
            }
        }
        
        // Scan platform-specific sources
        if self.config.read().unwrap().platform_config.enable_stack_bounds_detection {
            match self.platform_scanner.lock().unwrap().scan_thread_local_storage() {
                RootOperationResult::Success(tls_roots) => {
                    discovered_roots.extend(tls_roots);
                }
                RootOperationResult::Failed(e) => {
                    warn!("TLS scanning failed: {:?}", e);
                }
                _ => {}
            }
        }
        
        // Record analytics
        if self.config.read().unwrap().enable_analytics {
            let operation = RootAnalyticsOperation {
                timestamp: start_time,
                operation_type: AnalyticsOperationType::StackScan,
                duration: start_time.elapsed(),
                roots_affected: discovered_roots.len(),
                memory_delta: 0,
                success: true,
                metrics: HashMap::new(),
            };
            
            if let Err(e) = self.analytics.lock().unwrap().record_operation(&operation) {
                warn!("Failed to record analytics: {:?}", e);
            }
        }
        
        debug!("Discovered {} roots from scanning", discovered_roots.len());
        RootOperationResult::Success(discovered_roots)
    }
    
    /// Clear all roots (used for testing)
    fn clear_all_roots(&mut self) -> RootOperationResult<()> {
        let _span = span!(Level::DEBUG, "clear_all_roots").entered();
        
        // Clear manual roots
        {
            let mut manual_roots = self.manual_roots.write().unwrap();
            manual_roots.clear();
        }
        
        // Clear metadata
        {
            let mut metadata = self.root_metadata.write().unwrap();
            metadata.clear();
        }
        
        // Reset statistics
        {
            let mut stats = self.statistics.write().unwrap();
            *stats = RootStatistics::default();
        }
        
        // Clear global roots
        if let Err(e) = self.global_manager.lock().unwrap().clear_global_roots() {
            warn!("Failed to clear global roots: {:?}", e);
        }
        
        // Invalidate caches
        self.invalidate_thread_caches();
        
        info!("Cleared all roots");
        RootOperationResult::Success(())
    }
    
    /// Get root set statistics
    fn get_statistics(&self) -> RootStatistics {
        self.statistics.read().unwrap().clone()
    }
    
    /// Validate root set integrity
    fn validate_integrity(&self) -> RootOperationResult<()> {
        let _span = span!(Level::DEBUG, "validate_integrity").entered();
        
        if !self.config.read().unwrap().enable_validation {
            return RootOperationResult::Success(());
        }
        
        let mut issues = Vec::new();
        
        // Validate manual roots
        {
            let manual_roots = self.manual_roots.read().unwrap();
            let metadata = self.root_metadata.read().unwrap();
            
            for &ptr in manual_roots.iter() {
                if ptr.is_null() {
                    issues.push("Null pointer in manual roots".to_string());
                }
                
                if !metadata.contains_key(&ptr) {
                    issues.push(format!("Missing metadata for root {:?}", ptr));
                }
            }
        }
        
        // Check for orphaned metadata
        {
            let manual_roots = self.manual_roots.read().unwrap();
            let metadata = self.root_metadata.read().unwrap();
            
            for &ptr in metadata.keys() {
                if !manual_roots.contains(&ptr) {
                    issues.push(format!("Orphaned metadata for root {:?}", ptr));
                }
            }
        }
        
        if !issues.is_empty() {
            return RootOperationResult::Failed(RootError::ConfigurationError {
                error: format!("Integrity validation failed: {}", issues.join(", "))
            });
        }
        
        debug!("Root set integrity validation passed");
        RootOperationResult::Success(())
    }
}

// Backward compatibility methods for existing GC interface
impl RootManager {
    /// Add a root object (legacy interface)
    pub fn add_root(&mut self, ptr: *const u8) {
        let result = RootSetInterface::add_root(
            self, 
            ptr, 
            RootType::Unknown, 
            RootSource::Manual
        );
        
        if let RootOperationResult::Failed(e) = result {
            warn!("Failed to add root: {}", e);
        }
    }
    
    /// Remove a root object (legacy interface)
    pub fn remove_root(&mut self, ptr: *const u8) {
        let result = RootSetInterface::remove_root(self, ptr);
        
        if let RootOperationResult::Failed(e) = result {
            warn!("Failed to remove root: {}", e);
        }
    }
    
    /// Get all root objects (legacy interface)
    pub fn iter(&self) -> impl Iterator<Item = *const u8> + '_ {
        // Collect all roots from different sources
        let mut all_roots = Vec::new();
        
        // Get manual roots
        {
            let manual_roots = self.manual_roots.read().unwrap();
            all_roots.extend(manual_roots.iter().cloned());
        }
        
        // Get global roots
        match self.global_manager.lock().unwrap().get_global_roots() {
            RootOperationResult::Success(global_roots) => {
                all_roots.extend(global_roots);
            }
            _ => {} // Continue with other sources
        }
        
        // Get capability roots
        match self.global_manager.lock().unwrap().get_capability_roots() {
            RootOperationResult::Success(capability_roots) => {
                all_roots.extend(capability_roots);
            }
            _ => {}
        }
        
        // Get effect roots
        match self.global_manager.lock().unwrap().get_effect_roots() {
            RootOperationResult::Success(effect_roots) => {
                all_roots.extend(effect_roots);
            }
            _ => {}
        }
        
        // Return iterator over the collected roots
        all_roots.into_iter()
    }
    
    /// Scan the stack for additional roots (legacy interface)
    pub fn scan_stack(&mut self) {
        if let RootOperationResult::Failed(e) = self.scan_for_roots() {
            warn!("Stack scanning failed: {:?}", e);
        }
    }
    
    /// Register a global variable as a root (legacy interface)
    pub fn add_global(&mut self, ptr: *const u8) {
        if let Err(e) = self.global_manager.lock().unwrap()
            .register_global(ptr, "unknown".to_string(), "unknown".to_string()) {
            warn!("Failed to add global root: {:?}", e);
        }
    }
    
    /// Clear all roots (legacy interface)
    pub fn clear(&mut self) {
        if let RootOperationResult::Failed(e) = self.clear_all_roots() {
            warn!("Failed to clear roots: {:?}", e);
        }
    }
    
    /// Get the number of roots (legacy interface)
    pub fn len(&self) -> usize {
        self.statistics.read().unwrap().total_roots
    }
}

impl Default for RootStatistics {
    fn default() -> Self {
        Self {
            total_roots: 0,
            roots_by_source: HashMap::new(),
            roots_by_type: HashMap::new(),
            scan_stats: ScanStatistics {
                total_scans: 0,
                average_scan_time_us: 0.0,
                max_scan_time_us: 0,
                min_scan_time_us: 0,
                roots_per_second: 0.0,
                scan_efficiency: 0.0,
                cache_hit_rate: 0.0,
                last_scan_time: None,
            },
            memory_stats: RootMemoryStats {
                root_entries_bytes: 0,
                metadata_bytes: 0,
                cache_bytes: 0,
                total_overhead_bytes: 0,
                memory_efficiency: 0.0,
            },
            security_stats: SecurityStats {
                violations_detected: 0,
                capability_checks: 0,
                access_denials: 0,
                audit_events: 0,
                security_overhead_us: 0,
            },
            performance_trends: PerformanceTrends {
                scan_time_trend: TrendDirection::Unknown,
                root_count_trend: TrendDirection::Unknown,
                memory_usage_trend: TrendDirection::Unknown,
                cache_efficiency_trend: TrendDirection::Unknown,
                overall_trend: TrendDirection::Unknown,
            },
        }
    }
} 