//! Handler Registry
//!
//! Registry for managing effect handlers with capability validation

use super::handlers::{EffectHandler, EffectHandlerError};
use crate::security::Capability;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Registry for effect handlers
#[derive(Debug)]
pub struct HandlerRegistry {
    /// Map of effect names to their handlers
    handlers: HashMap<String, Arc<dyn EffectHandler>>,
    /// Capability requirements cache
    capability_cache: HashMap<String, Vec<String>>,
    /// Handler metadata
    metadata: HashMap<String, HandlerMetadata>,
}

impl HandlerRegistry {
    /// Create new handler registry with builtin handlers
    pub fn new() -> Self {
        let mut registry = Self {
            handlers: HashMap::new(),
            capability_cache: HashMap::new(),
            metadata: HashMap::new(),
        };
        registry.register_builtin_handlers();
        registry
    }

    /// Register a new effect handler
    pub fn register_handler(
        &mut self,
        effect_name: String,
        handler: Arc<dyn EffectHandler>,
    ) -> Result<(), EffectHandlerError> {
        // Validate handler capabilities
        let capabilities = handler.required_capabilities();
        self.validate_capabilities(&capabilities)?;

        // Cache capabilities for quick lookup
        self.capability_cache.insert(effect_name.clone(), capabilities);

        // Store metadata
        self.metadata.insert(effect_name.clone(), HandlerMetadata {
            name: handler.name().to_string(),
            registered_at: std::time::SystemTime::now(),
            usage_count: 0,
        });

        // Register the handler
        self.handlers.insert(effect_name, handler);
        Ok(())
    }

    /// Get handler for an effect
    pub fn get_handler(&self, effect_name: &str) -> Option<&Arc<dyn EffectHandler>> {
        self.handlers.get(effect_name)
    }

    /// Get all handlers that can handle a specific effect
    pub fn get_compatible_handlers(&self, effect_name: &str) -> Vec<&Arc<dyn EffectHandler>> {
        self.handlers
            .values()
            .filter(|handler| handler.can_handle(effect_name))
            .collect()
    }

    /// Check if capabilities are available for an effect
    pub fn check_capabilities(&self, effect_name: &str, available_caps: &[Capability]) -> bool {
        if let Some(required) = self.capability_cache.get(effect_name) {
                    let available_names: Vec<String> = available_caps
            .iter()
            .map(|c| c.definition.clone())
            .collect();
            
            required.iter().all(|req| available_names.contains(req))
        } else {
            false
        }
    }

    /// Register all builtin handlers
    fn register_builtin_handlers(&mut self) {
        let builtin = BuiltinHandlers::new();
        
        // Register file system handler
        if let Err(e) = self.register_handler(
            "IO.FileSystem".to_string(),
            Arc::new(builtin.filesystem_handler),
        ) {
            eprintln!("Failed to register filesystem handler: {:?}", e);
        }

        // Register network handler
        if let Err(e) = self.register_handler(
            "IO.Network".to_string(),
            Arc::new(builtin.network_handler),
        ) {
            eprintln!("Failed to register network handler: {:?}", e);
        }

        // Register database handler
        if let Err(e) = self.register_handler(
            "Database".to_string(),
            Arc::new(builtin.database_handler),
        ) {
            eprintln!("Failed to register database handler: {:?}", e);
        }

        // Register cryptography handler
        if let Err(e) = self.register_handler(
            "Cryptography".to_string(),
            Arc::new(builtin.crypto_handler),
        ) {
            eprintln!("Failed to register crypto handler: {:?}", e);
        }
    }

    /// Validate that capabilities are properly formed
    fn validate_capabilities(&self, capabilities: &[String]) -> Result<(), EffectHandlerError> {
        for cap in capabilities {
            if cap.is_empty() {
                return Err(EffectHandlerError::InvalidCapability(
                    "Empty capability name".to_string()
                ));
            }
        }
        Ok(())
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total_handlers: self.handlers.len(),
            total_capabilities: self.capability_cache.values().map(|v| v.len()).sum(),
            builtin_handlers: 4, // filesystem, network, database, crypto
        }
    }
}

impl Default for HandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a registered handler
#[derive(Debug)]
pub struct HandlerMetadata {
    /// Handler name
    pub name: String,
    /// When the handler was registered
    pub registered_at: std::time::SystemTime,
    /// How many times this handler has been used
    pub usage_count: u64,
}

/// Registry statistics
#[derive(Debug)]
pub struct RegistryStats {
    /// Total number of registered handlers
    pub total_handlers: usize,
    /// Total number of capabilities across all handlers
    pub total_capabilities: usize,
    /// Number of builtin handlers
    pub builtin_handlers: usize,
} 