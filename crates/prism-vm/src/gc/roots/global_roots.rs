//! Global Root Manager for static data and global variables
//!
//! This module manages global variables, static data, capability tokens,
//! and other roots that persist across the entire VM lifetime.

use crate::{VMResult, PrismVMError};
use super::{types::*, interfaces::*};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// Manager for global roots (static data, global variables, etc.)
pub struct GlobalRootManager {
    /// Global variable roots
    global_variables: Arc<RwLock<HashMap<*const u8, GlobalVariableInfo>>>,
    
    /// Capability token roots
    capability_roots: Arc<RwLock<HashMap<*const u8, CapabilityInfo>>>,
    
    /// Effect handle roots
    effect_roots: Arc<RwLock<HashMap<*const u8, EffectInfo>>>,
    
    /// Static data roots
    static_roots: Arc<RwLock<HashSet<*const u8>>>,
}

#[derive(Debug, Clone)]
struct GlobalVariableInfo {
    name: String,
    var_type: String,
    registered_at: std::time::Instant,
}

#[derive(Debug, Clone)]
struct CapabilityInfo {
    name: String,
    registered_at: std::time::Instant,
}

#[derive(Debug, Clone)]
struct EffectInfo {
    name: String,
    registered_at: std::time::Instant,
}

impl GlobalRootManager {
    pub fn new() -> VMResult<Self> {
        Ok(Self {
            global_variables: Arc::new(RwLock::new(HashMap::new())),
            capability_roots: Arc::new(RwLock::new(HashMap::new())),
            effect_roots: Arc::new(RwLock::new(HashMap::new())),
            static_roots: Arc::new(RwLock::new(HashSet::new())),
        })
    }
}

impl GlobalRootInterface for GlobalRootManager {
    fn register_global(&mut self, ptr: *const u8, name: String, var_type: String) -> RootOperationResult<()> {
        let info = GlobalVariableInfo {
            name: name.clone(),
            var_type,
            registered_at: std::time::Instant::now(),
        };
        
        self.global_variables.write().unwrap().insert(ptr, info);
        debug!("Registered global variable: {} at {:?}", name, ptr);
        RootOperationResult::Success(())
    }
    
    fn unregister_global(&mut self, ptr: *const u8) -> RootOperationResult<()> {
        if self.global_variables.write().unwrap().remove(&ptr).is_some() {
            debug!("Unregistered global variable at {:?}", ptr);
            RootOperationResult::Success(())
        } else {
            RootOperationResult::Failed(RootError::RootNotFound { ptr: ptr as usize })
        }
    }
    
    fn register_capability(&mut self, ptr: *const u8, name: String) -> RootOperationResult<()> {
        let info = CapabilityInfo {
            name: name.clone(),
            registered_at: std::time::Instant::now(),
        };
        
        self.capability_roots.write().unwrap().insert(ptr, info);
        debug!("Registered capability: {} at {:?}", name, ptr);
        RootOperationResult::Success(())
    }
    
    fn register_effect(&mut self, ptr: *const u8, name: String) -> RootOperationResult<()> {
        let info = EffectInfo {
            name: name.clone(),
            registered_at: std::time::Instant::now(),
        };
        
        self.effect_roots.write().unwrap().insert(ptr, info);
        debug!("Registered effect: {} at {:?}", name, ptr);
        RootOperationResult::Success(())
    }
    
    fn get_global_roots(&self) -> RootOperationResult<Vec<*const u8>> {
        let globals = self.global_variables.read().unwrap();
        RootOperationResult::Success(globals.keys().cloned().collect())
    }
    
    fn get_capability_roots(&self) -> RootOperationResult<Vec<*const u8>> {
        let capabilities = self.capability_roots.read().unwrap();
        RootOperationResult::Success(capabilities.keys().cloned().collect())
    }
    
    fn get_effect_roots(&self) -> RootOperationResult<Vec<*const u8>> {
        let effects = self.effect_roots.read().unwrap();
        RootOperationResult::Success(effects.keys().cloned().collect())
    }
    
    fn clear_global_roots(&mut self) -> RootOperationResult<()> {
        self.global_variables.write().unwrap().clear();
        self.capability_roots.write().unwrap().clear();
        self.effect_roots.write().unwrap().clear();
        self.static_roots.write().unwrap().clear();
        info!("Cleared all global roots");
        RootOperationResult::Success(())
    }
} 