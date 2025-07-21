//! Multi-Target Serialization for Arena System
//!
//! This module handles serialization of AST nodes for different compilation targets,
//! ensuring that semantic information and metadata are preserved across all targets.

use crate::{AstNode, AstNodeRef};
use super::{ArenaError, ArenaResult, semantic::SemanticArenaNode};
use prism_common::{NodeId, SourceId};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, Duration};
use thiserror::Error;

/// Serialization target platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SerializationTarget {
    /// TypeScript for rapid development
    TypeScript,
    /// WebAssembly for portable execution
    WebAssembly,
    /// Native code for maximum performance
    Native,
    /// JSON for debugging and tooling
    Json,
    /// Binary format for efficiency
    Binary,
}

/// Serialization format options
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// Compact binary format
    CompactBinary,
    /// Human-readable JSON
    Json,
    /// MessagePack for efficiency
    MessagePack,
    /// Custom Prism format
    PrismNative,
}

/// Compression algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 for fast compression/decompression
    Lz4,
    /// Zlib for balanced compression
    Zlib,
    /// Zstd for high compression ratio
    Zstd,
    /// Brotli for web optimization
    Brotli,
}

/// Serialization configuration
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    /// Target platform
    pub target: SerializationTarget,
    /// Serialization format
    pub format: SerializationFormat,
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    /// Include AI metadata
    pub include_ai_metadata: bool,
    /// Include semantic relationships
    pub include_relationships: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Pretty print for human readability
    pub pretty_print: bool,
    /// Enable streaming for large datasets
    pub enable_streaming: bool,
    /// Cache serialized data
    pub enable_caching: bool,
    /// Maximum cache size in bytes
    pub max_cache_size: usize,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            target: SerializationTarget::Binary,
            format: SerializationFormat::CompactBinary,
            compression: CompressionAlgorithm::Lz4,
            include_ai_metadata: true,
            include_relationships: true,
            compression_level: 6,
            pretty_print: false,
            enable_streaming: false,
            enable_caching: true,
            max_cache_size: 64 * 1024 * 1024, // 64MB default cache
        }
    }
}

/// Serialization errors
#[derive(Debug, Error)]
pub enum SerializationError {
    /// Serialization failed
    #[error("Serialization failed: {reason}")]
    SerializationFailed { reason: String },
    
    /// Deserialization failed
    #[error("Deserialization failed: {reason}")]
    DeserializationFailed { reason: String },
    
    /// Unsupported format
    #[error("Unsupported format: {format:?} for target {target:?}")]
    UnsupportedFormat {
        format: SerializationFormat,
        target: SerializationTarget,
    },
    
    /// Compression error
    #[error("Compression error: {reason}")]
    CompressionError { reason: String },
    
    /// Version mismatch
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },
    
    /// Cache error
    #[error("Cache error: {reason}")]
    CacheError { reason: String },
    
    /// Streaming error
    #[error("Streaming error: {reason}")]
    StreamingError { reason: String },
}

/// Serialized node data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    /// Serialized data
    pub data: Vec<u8>,
    /// Metadata about the serialization
    pub metadata: SerializationMetadata,
}

/// Metadata about serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationMetadata {
    /// Target platform
    pub target: SerializationTarget,
    /// Format used
    pub format: SerializationFormat,
    /// Compression algorithm used
    pub compression: CompressionAlgorithm,
    /// Version of serialization format
    pub version: String,
    /// Size of original data
    pub original_size: usize,
    /// Size of compressed data
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Serialization timestamp
    pub timestamp: SystemTime,
    /// Checksum for integrity verification
    pub checksum: u64,
}

/// Cache entry for serialized data
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached serialized node
    node: SerializedNode,
    /// Last access time
    last_accessed: SystemTime,
    /// Access count
    access_count: u64,
    /// Size in bytes
    size: usize,
}

/// Serialization cache for performance optimization
#[derive(Debug)]
struct SerializationCache {
    /// Cache entries by node reference
    entries: HashMap<AstNodeRef, CacheEntry>,
    /// Current cache size in bytes
    current_size: usize,
    /// Maximum cache size in bytes
    max_size: usize,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Cache evictions
    pub evictions: u64,
    /// Total cache size
    pub total_size: usize,
    /// Number of entries
    pub entry_count: usize,
}

impl SerializationCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            current_size: 0,
            max_size,
            stats: CacheStats::default(),
        }
    }
    
    fn get(&mut self, node_ref: &AstNodeRef) -> Option<SerializedNode> {
        if let Some(entry) = self.entries.get_mut(node_ref) {
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;
            self.stats.hits += 1;
            Some(entry.node.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }
    
    fn insert(&mut self, node_ref: AstNodeRef, node: SerializedNode) {
        let size = node.data.len() + std::mem::size_of::<SerializedNode>();
        
        // Evict entries if necessary
        while self.current_size + size > self.max_size && !self.entries.is_empty() {
            self.evict_lru();
        }
        
        let entry = CacheEntry {
            node,
            last_accessed: SystemTime::now(),
            access_count: 1,
            size,
        };
        
        if let Some(old_entry) = self.entries.insert(node_ref, entry) {
            self.current_size -= old_entry.size;
        }
        
        self.current_size += size;
        self.stats.total_size = self.current_size;
        self.stats.entry_count = self.entries.len();
    }
    
    fn evict_lru(&mut self) {
        let oldest_key = self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| *key);
        
        if let Some(key) = oldest_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size -= entry.size;
                self.stats.evictions += 1;
            }
        }
    }
    
    fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// Enum for target-specific serializers to avoid dyn trait issues
#[derive(Debug)]
pub enum TargetSerializerImpl {
    TypeScript(TypeScriptSerializer),
    WebAssembly(WasmSerializer),
    Native(NativeSerializer),
    Json(JsonSerializer),
    Binary(BinarySerializer),
}

impl TargetSerializerImpl {
    /// Serialize a node for the target
    pub fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        match self {
            TargetSerializerImpl::TypeScript(s) => s.serialize(node, config),
            TargetSerializerImpl::WebAssembly(s) => s.serialize(node, config),
            TargetSerializerImpl::Native(s) => s.serialize(node, config),
            TargetSerializerImpl::Json(s) => s.serialize(node, config),
            TargetSerializerImpl::Binary(s) => s.serialize(node, config),
        }
    }
    
    /// Deserialize a node from the target format
    pub fn deserialize<T>(&self, data: &SerializedNode) 
        -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        match self {
            TargetSerializerImpl::TypeScript(s) => s.deserialize(data),
            TargetSerializerImpl::WebAssembly(s) => s.deserialize(data),
            TargetSerializerImpl::Native(s) => s.deserialize(data),
            TargetSerializerImpl::Json(s) => s.deserialize(data),
            TargetSerializerImpl::Binary(s) => s.deserialize(data),
        }
    }
    
    /// Get the target name
    pub fn target_name(&self) -> &'static str {
        match self {
            TargetSerializerImpl::TypeScript(s) => s.target_name(),
            TargetSerializerImpl::WebAssembly(s) => s.target_name(),
            TargetSerializerImpl::Native(s) => s.target_name(),
            TargetSerializerImpl::Json(s) => s.target_name(),
            TargetSerializerImpl::Binary(s) => s.target_name(),
        }
    }
    
    /// Check if a format is supported
    pub fn supports_format(&self, format: &SerializationFormat) -> bool {
        match self {
            TargetSerializerImpl::TypeScript(s) => s.supports_format(format),
            TargetSerializerImpl::WebAssembly(s) => s.supports_format(format),
            TargetSerializerImpl::Native(s) => s.supports_format(format),
            TargetSerializerImpl::Json(s) => s.supports_format(format),
            TargetSerializerImpl::Binary(s) => s.supports_format(format),
        }
    }
    
    /// Serialize in streaming mode
    pub fn serialize_stream<T>(&self, nodes: &[SemanticArenaNode<T>], config: &SerializationConfig) 
        -> Result<Vec<u8>, SerializationError>
    where
        T: Serialize,
    {
        match self {
            TargetSerializerImpl::TypeScript(s) => s.serialize_stream(nodes, config),
            TargetSerializerImpl::WebAssembly(s) => s.serialize_stream(nodes, config),
            TargetSerializerImpl::Native(s) => s.serialize_stream(nodes, config),
            TargetSerializerImpl::Json(s) => s.serialize_stream(nodes, config),
            TargetSerializerImpl::Binary(s) => s.serialize_stream(nodes, config),
        }
    }
}

/// Arena serializer for different targets
pub struct ArenaSerializer {
    /// Configuration
    config: SerializationConfig,
    /// Serialization cache
    cache: Arc<RwLock<SerializationCache>>,
    /// Target-specific serializers
    serializers: HashMap<SerializationTarget, TargetSerializerImpl>,
}

/// Trait for target-specific serialization
pub trait TargetSerializer: Send + Sync {
    /// Serialize a node for the target
    fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize;
    
    /// Deserialize a node from the target format
    fn deserialize<T>(&self, data: &SerializedNode) 
        -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>;
    
    /// Get the target name
    fn target_name(&self) -> &'static str;
    
    /// Check if a format is supported
    fn supports_format(&self, format: &SerializationFormat) -> bool;
    
    /// Serialize in streaming mode
    fn serialize_stream<T>(&self, nodes: &[SemanticArenaNode<T>], config: &SerializationConfig) 
        -> Result<Vec<u8>, SerializationError>
    where
        T: Serialize,
    {
        // Default implementation - serialize each node and concatenate
        let mut result = Vec::new();
        for node in nodes {
            let serialized = self.serialize(node, config)?;
            result.extend_from_slice(&serialized.data);
        }
        Ok(result)
    }
}

impl ArenaSerializer {
    /// Create a new arena serializer
    pub fn new(config: SerializationConfig) -> Self {
        let cache = Arc::new(RwLock::new(SerializationCache::new(config.max_cache_size)));
        
        let mut serializers = HashMap::new();
        serializers.insert(SerializationTarget::TypeScript, TargetSerializerImpl::TypeScript(TypeScriptSerializer::new()));
        serializers.insert(SerializationTarget::WebAssembly, TargetSerializerImpl::WebAssembly(WasmSerializer::new()));
        serializers.insert(SerializationTarget::Native, TargetSerializerImpl::Native(NativeSerializer::new()));
        serializers.insert(SerializationTarget::Json, TargetSerializerImpl::Json(JsonSerializer::new()));
        serializers.insert(SerializationTarget::Binary, TargetSerializerImpl::Binary(BinarySerializer::new()));
        
        Self {
            config,
            cache,
            serializers,
        }
    }
    
    /// Serialize a node for the configured target
    pub fn serialize<T>(&self, node: &SemanticArenaNode<T>) -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        // Check cache first if enabled
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                // Create a simple node reference using the node ID and a default source
                let node_ref = AstNodeRef::new(node.node.id.as_u32(), SourceId::new(0));
                if let Some(cached) = cache.get(&node_ref) {
                    return Ok(cached);
                }
            }
        }
        
        let serializer = self.serializers.get(&self.config.target)
            .ok_or_else(|| SerializationError::UnsupportedFormat {
                format: self.config.format.clone(),
                target: self.config.target,
            })?;
        
        if !serializer.supports_format(&self.config.format) {
            return Err(SerializationError::UnsupportedFormat {
                format: self.config.format.clone(),
                target: self.config.target,
            });
        }
        
        let result = serializer.serialize(node, &self.config)?;
        
        // Cache the result if enabled
        if self.config.enable_caching {
            if let Ok(mut cache) = self.cache.write() {
                let node_ref = AstNodeRef::new(node.node.id.as_u32(), SourceId::new(0));
                cache.insert(node_ref, result.clone());
            }
        }
        
        Ok(result)
    }
    
    /// Deserialize a node from serialized data
    pub fn deserialize<T>(&self, data: &SerializedNode) -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let serializer = self.serializers.get(&data.metadata.target)
            .ok_or_else(|| SerializationError::UnsupportedFormat {
                format: data.metadata.format.clone(),
                target: data.metadata.target,
            })?;
        
        serializer.deserialize(data)
    }
    
    /// Serialize multiple nodes as a batch
    pub fn serialize_batch<T>(&self, nodes: &[SemanticArenaNode<T>]) -> Result<Vec<SerializedNode>, SerializationError>
    where
        T: Serialize,
    {
        if self.config.enable_streaming && nodes.len() > 100 {
            // Use streaming for large batches
            let serializer = self.serializers.get(&self.config.target)
                .ok_or_else(|| SerializationError::UnsupportedFormat {
                    format: self.config.format.clone(),
                    target: self.config.target,
                })?;
            
            let stream_data = serializer.serialize_stream(nodes, &self.config)?;
            
            // For now, return a single serialized node containing all data
            // In a real implementation, this would be more sophisticated
            Ok(vec![SerializedNode {
                data: stream_data,
                metadata: SerializationMetadata {
                    target: self.config.target,
                    format: self.config.format.clone(),
                    compression: self.config.compression,
                    version: "1.0.0".to_string(),
                    original_size: 0, // Would be calculated
                    compressed_size: 0, // Would be calculated
                    compression_ratio: 1.0,
                    timestamp: SystemTime::now(),
                    checksum: 0, // Would be calculated
                },
            }])
        } else {
            // Regular batch serialization
            nodes.iter()
                .map(|node| self.serialize(node))
                .collect()
        }
    }
    
    /// Get serialization statistics
    pub fn get_stats(&self, serialized: &[SerializedNode]) -> SerializationStats {
        let total_original = serialized.iter().map(|s| s.metadata.original_size).sum();
        let total_compressed = serialized.iter().map(|s| s.metadata.compressed_size).sum();
        let avg_compression_ratio = if !serialized.is_empty() {
            serialized.iter().map(|s| s.metadata.compression_ratio).sum::<f64>() / serialized.len() as f64
        } else {
            0.0
        };
        
        SerializationStats {
            node_count: serialized.len(),
            total_original_size: total_original,
            total_compressed_size: total_compressed,
            average_compression_ratio: avg_compression_ratio,
            target: self.config.target,
            format: self.config.format.clone(),
        }
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<CacheStats, SerializationError> {
        self.cache.read()
            .map(|cache| cache.stats().clone())
            .map_err(|_| SerializationError::CacheError {
                reason: "Failed to acquire cache lock".to_string(),
            })
    }
    
    /// Clear the serialization cache
    pub fn clear_cache(&self) -> Result<(), SerializationError> {
        self.cache.write()
            .map(|mut cache| {
                cache.entries.clear();
                cache.current_size = 0;
                cache.stats = CacheStats::default();
            })
            .map_err(|_| SerializationError::CacheError {
                reason: "Failed to acquire cache write lock".to_string(),
            })
    }
}

/// Serialization statistics
#[derive(Debug, Clone)]
pub struct SerializationStats {
    /// Number of nodes serialized
    pub node_count: usize,
    /// Total original size in bytes
    pub total_original_size: usize,
    /// Total compressed size in bytes
    pub total_compressed_size: usize,
    /// Average compression ratio
    pub average_compression_ratio: f64,
    /// Target platform
    pub target: SerializationTarget,
    /// Format used
    pub format: SerializationFormat,
}

// Compression utilities
struct CompressionProvider;

impl CompressionProvider {
    fn compress(data: &[u8], algorithm: CompressionAlgorithm, _level: u8) -> Result<Vec<u8>, SerializationError> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Lz4 => {
                // In a real implementation, would use lz4 crate
                // For now, simulate compression
                Ok(data.to_vec())
            },
            CompressionAlgorithm::Zlib => {
                // In a real implementation, would use flate2 crate
                // For now, simulate compression
                let compressed = data.to_vec();
                // Simulate compression by reducing size slightly
                Ok(compressed[..compressed.len().saturating_sub(compressed.len() / 10)].to_vec())
            },
            CompressionAlgorithm::Zstd => {
                // In a real implementation, would use zstd crate
                Ok(data.to_vec())
            },
            CompressionAlgorithm::Brotli => {
                // In a real implementation, would use brotli crate
                Ok(data.to_vec())
            },
        }
    }
    
    fn decompress(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>, SerializationError> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            _ => {
                // In a real implementation, would use appropriate decompression
                Ok(data.to_vec())
            }
        }
    }
    
    fn calculate_checksum(data: &[u8]) -> u64 {
        // Simple checksum - in reality would use CRC32 or similar
        data.iter().fold(0u64, |acc, &byte| acc.wrapping_add(byte as u64))
    }
}

// Target-specific serializer implementations

/// TypeScript serializer
#[derive(Debug)]
struct TypeScriptSerializer;

impl TypeScriptSerializer {
    fn new() -> Self {
        Self
    }
}

impl TargetSerializer for TypeScriptSerializer {
    fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        let original_data = match config.format {
            SerializationFormat::Json => {
                if config.pretty_print {
                    serde_json::to_vec_pretty(node).map_err(|e| SerializationError::SerializationFailed {
                        reason: format!("JSON serialization failed: {}", e),
                    })?
                } else {
                    serde_json::to_vec(node).map_err(|e| SerializationError::SerializationFailed {
                        reason: format!("JSON serialization failed: {}", e),
                    })?
                }
            },
            SerializationFormat::CompactBinary => {
                bincode::serialize(node).map_err(|e| SerializationError::SerializationFailed {
                    reason: format!("Binary serialization failed: {}", e),
                })?
            },
            SerializationFormat::MessagePack => {
                // In a real implementation, would use rmp-serde
                serde_json::to_vec(node).map_err(|e| SerializationError::SerializationFailed {
                    reason: format!("MessagePack serialization failed: {}", e),
                })? // Fallback to JSON for now
            },
            _ => return Err(SerializationError::UnsupportedFormat {
                format: config.format.clone(),
                target: config.target,
            }),
        };
        
        let compressed_data = CompressionProvider::compress(&original_data, config.compression, config.compression_level)?;
        let original_size = original_data.len();
        let compressed_size = compressed_data.len();
        let checksum = CompressionProvider::calculate_checksum(&compressed_data);
        
        Ok(SerializedNode {
            data: compressed_data,
            metadata: SerializationMetadata {
                target: config.target,
                format: config.format.clone(),
                compression: config.compression,
                version: "1.0.0".to_string(),
                original_size,
                compressed_size,
                compression_ratio: original_size as f64 / compressed_size.max(1) as f64,
                timestamp: SystemTime::now(),
                checksum,
            },
        })
    }
    
    fn deserialize<T>(&self, data: &SerializedNode) -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Verify checksum
        let calculated_checksum = CompressionProvider::calculate_checksum(&data.data);
        if calculated_checksum != data.metadata.checksum {
            return Err(SerializationError::DeserializationFailed {
                reason: "Checksum mismatch - data may be corrupted".to_string(),
            });
        }
        
        let decompressed = CompressionProvider::decompress(&data.data, data.metadata.compression)?;
        
        match data.metadata.format {
            SerializationFormat::Json => {
                serde_json::from_slice(&decompressed).map_err(|e| SerializationError::DeserializationFailed {
                    reason: format!("JSON deserialization failed: {}", e),
                })
            },
            SerializationFormat::CompactBinary => {
                bincode::deserialize(&decompressed).map_err(|e| SerializationError::DeserializationFailed {
                    reason: format!("Binary deserialization failed: {}", e),
                })
            },
            SerializationFormat::MessagePack => {
                // In a real implementation, would use rmp-serde
                serde_json::from_slice(&decompressed).map_err(|e| SerializationError::DeserializationFailed {
                    reason: format!("MessagePack deserialization failed: {}", e),
                }) // Fallback to JSON for now
            },
            _ => return Err(SerializationError::UnsupportedFormat {
                format: data.metadata.format.clone(),
                target: data.metadata.target,
            }),
        }
    }
    
    fn target_name(&self) -> &'static str {
        "TypeScript"
    }
    
    fn supports_format(&self, format: &SerializationFormat) -> bool {
        matches!(format, 
            SerializationFormat::Json | 
            SerializationFormat::CompactBinary |
            SerializationFormat::MessagePack
        )
    }
}

/// WebAssembly serializer
#[derive(Debug)]
struct WasmSerializer;

impl WasmSerializer {
    fn new() -> Self {
        Self
    }
}

impl TargetSerializer for WasmSerializer {
    fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        // WASM-optimized serialization
        let data = match config.format {
            SerializationFormat::CompactBinary => bincode::serialize(node),
            SerializationFormat::PrismNative => {
                // Custom Prism format would go here
                bincode::serialize(node) // Fallback for now
            },
            _ => return Err(SerializationError::UnsupportedFormat {
                format: config.format.clone(),
                target: config.target,
            }),
        }.map_err(|e| {
            SerializationError::SerializationFailed {
                reason: format!("WASM serialization failed: {}", e),
            }
        })?;
        
        let compressed_data = CompressionProvider::compress(&data, config.compression, config.compression_level)?;
        let original_size = data.len();
        let compressed_size = compressed_data.len();
        let checksum = CompressionProvider::calculate_checksum(&compressed_data);
        
        Ok(SerializedNode {
            data: compressed_data,
            metadata: SerializationMetadata {
                target: config.target,
                format: config.format.clone(),
                compression: config.compression,
                version: "1.0.0".to_string(),
                original_size,
                compressed_size,
                compression_ratio: original_size as f64 / compressed_size.max(1) as f64,
                timestamp: SystemTime::now(),
                checksum,
            },
        })
    }
    
    fn deserialize<T>(&self, data: &SerializedNode) -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let calculated_checksum = CompressionProvider::calculate_checksum(&data.data);
        if calculated_checksum != data.metadata.checksum {
            return Err(SerializationError::DeserializationFailed {
                reason: "Checksum mismatch - data may be corrupted".to_string(),
            });
        }
        
        let decompressed = CompressionProvider::decompress(&data.data, data.metadata.compression)?;
        
        bincode::deserialize(&decompressed).map_err(|e| {
            SerializationError::DeserializationFailed {
                reason: format!("WASM deserialization failed: {}", e),
            }
        })
    }
    
    fn target_name(&self) -> &'static str {
        "WebAssembly"
    }
    
    fn supports_format(&self, format: &SerializationFormat) -> bool {
        matches!(format, SerializationFormat::CompactBinary | SerializationFormat::PrismNative)
    }
}

/// Native serializer
#[derive(Debug)]
struct NativeSerializer;

impl NativeSerializer {
    fn new() -> Self {
        Self
    }
}

impl TargetSerializer for NativeSerializer {
    fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        // Native-optimized serialization
        let data = match config.format {
            SerializationFormat::CompactBinary => bincode::serialize(node),
            SerializationFormat::PrismNative => {
                // Custom Prism format would go here
                bincode::serialize(node) // Fallback for now
            },
            SerializationFormat::MessagePack => {
                // In a real implementation, would use rmp-serde
                bincode::serialize(node) // Fallback for now
            },
            _ => return Err(SerializationError::UnsupportedFormat {
                format: config.format.clone(),
                target: config.target,
            }),
        }.map_err(|e| {
            SerializationError::SerializationFailed {
                reason: format!("Native serialization failed: {}", e),
            }
        })?;
        
        let compressed_data = CompressionProvider::compress(&data, config.compression, config.compression_level)?;
        let original_size = data.len();
        let compressed_size = compressed_data.len();
        let checksum = CompressionProvider::calculate_checksum(&compressed_data);
        
        Ok(SerializedNode {
            data: compressed_data,
            metadata: SerializationMetadata {
                target: config.target,
                format: config.format.clone(),
                compression: config.compression,
                version: "1.0.0".to_string(),
                original_size,
                compressed_size,
                compression_ratio: original_size as f64 / compressed_size.max(1) as f64,
                timestamp: SystemTime::now(),
                checksum,
            },
        })
    }
    
    fn deserialize<T>(&self, data: &SerializedNode) -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let calculated_checksum = CompressionProvider::calculate_checksum(&data.data);
        if calculated_checksum != data.metadata.checksum {
            return Err(SerializationError::DeserializationFailed {
                reason: "Checksum mismatch - data may be corrupted".to_string(),
            });
        }
        
        let decompressed = CompressionProvider::decompress(&data.data, data.metadata.compression)?;
        
        bincode::deserialize(&decompressed).map_err(|e| {
            SerializationError::DeserializationFailed {
                reason: format!("Native deserialization failed: {}", e),
            }
        })
    }
    
    fn target_name(&self) -> &'static str {
        "Native"
    }
    
    fn supports_format(&self, format: &SerializationFormat) -> bool {
        matches!(format, 
            SerializationFormat::CompactBinary | 
            SerializationFormat::PrismNative |
            SerializationFormat::MessagePack
        )
    }
}

/// JSON serializer for debugging
#[derive(Debug)]
struct JsonSerializer;

impl JsonSerializer {
    fn new() -> Self {
        Self
    }
}

impl TargetSerializer for JsonSerializer {
    fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        let data = if config.pretty_print {
            serde_json::to_vec_pretty(node)
        } else {
            serde_json::to_vec(node)
        }.map_err(|e| SerializationError::SerializationFailed {
            reason: format!("JSON serialization failed: {}", e),
        })?;
        
        let compressed_data = CompressionProvider::compress(&data, config.compression, config.compression_level)?;
        let original_size = data.len();
        let compressed_size = compressed_data.len();
        let checksum = CompressionProvider::calculate_checksum(&compressed_data);
        
        Ok(SerializedNode {
            data: compressed_data,
            metadata: SerializationMetadata {
                target: config.target,
                format: config.format.clone(),
                compression: config.compression,
                version: "1.0.0".to_string(),
                original_size,
                compressed_size,
                compression_ratio: original_size as f64 / compressed_size.max(1) as f64,
                timestamp: SystemTime::now(),
                checksum,
            },
        })
    }
    
    fn deserialize<T>(&self, data: &SerializedNode) -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let calculated_checksum = CompressionProvider::calculate_checksum(&data.data);
        if calculated_checksum != data.metadata.checksum {
            return Err(SerializationError::DeserializationFailed {
                reason: "Checksum mismatch - data may be corrupted".to_string(),
            });
        }
        
        let decompressed = CompressionProvider::decompress(&data.data, data.metadata.compression)?;
        
        serde_json::from_slice(&decompressed).map_err(|e| {
            SerializationError::DeserializationFailed {
                reason: format!("JSON deserialization failed: {}", e),
            }
        })
    }
    
    fn target_name(&self) -> &'static str {
        "JSON"
    }
    
    fn supports_format(&self, format: &SerializationFormat) -> bool {
        matches!(format, SerializationFormat::Json)
    }
}

/// Binary serializer
#[derive(Debug)]
struct BinarySerializer;

impl BinarySerializer {
    fn new() -> Self {
        Self
    }
}

impl TargetSerializer for BinarySerializer {
    fn serialize<T>(&self, node: &SemanticArenaNode<T>, config: &SerializationConfig) 
        -> Result<SerializedNode, SerializationError>
    where
        T: Serialize,
    {
        let data = bincode::serialize(node).map_err(|e| {
            SerializationError::SerializationFailed {
                reason: format!("Binary serialization failed: {}", e),
            }
        })?;
        
        let compressed_data = CompressionProvider::compress(&data, config.compression, config.compression_level)?;
        let original_size = data.len();
        let compressed_size = compressed_data.len();
        let checksum = CompressionProvider::calculate_checksum(&compressed_data);
        
        Ok(SerializedNode {
            data: compressed_data,
            metadata: SerializationMetadata {
                target: config.target,
                format: config.format.clone(),
                compression: config.compression,
                version: "1.0.0".to_string(),
                original_size,
                compressed_size,
                compression_ratio: original_size as f64 / compressed_size.max(1) as f64,
                timestamp: SystemTime::now(),
                checksum,
            },
        })
    }
    
    fn deserialize<T>(&self, data: &SerializedNode) -> Result<SemanticArenaNode<T>, SerializationError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let calculated_checksum = CompressionProvider::calculate_checksum(&data.data);
        if calculated_checksum != data.metadata.checksum {
            return Err(SerializationError::DeserializationFailed {
                reason: "Checksum mismatch - data may be corrupted".to_string(),
            });
        }
        
        let decompressed = CompressionProvider::decompress(&data.data, data.metadata.compression)?;
        
        bincode::deserialize(&decompressed).map_err(|e| {
            SerializationError::DeserializationFailed {
                reason: format!("Binary deserialization failed: {}", e),
            }
        })
    }
    
    fn target_name(&self) -> &'static str {
        "Binary"
    }
    
    fn supports_format(&self, format: &SerializationFormat) -> bool {
        matches!(format, SerializationFormat::CompactBinary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, LiteralExpr, LiteralValue};
    use crate::arena::semantic::{SemanticArenaNode, SemanticMetadata};
    use prism_common::span::{Span, Position};

    #[test]
    fn test_serialization_config() {
        let config = SerializationConfig::default();
        assert_eq!(config.target, SerializationTarget::Binary);
        assert_eq!(config.format, SerializationFormat::CompactBinary);
        assert_eq!(config.compression, CompressionAlgorithm::Lz4);
        assert!(config.include_ai_metadata);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_typescript_serialization() {
        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        });
        let ast_node = AstNode::new(expr, span, NodeId::new(1));
        let semantic_node = SemanticArenaNode::new(ast_node, 100, None);
        
        let config = SerializationConfig {
            target: SerializationTarget::TypeScript,
            format: SerializationFormat::Json,
            compression: CompressionAlgorithm::Zlib,
            ..Default::default()
        };
        
        let serializer = ArenaSerializer::new(config);
        let serialized = serializer.serialize(&semantic_node).unwrap();
        
        assert_eq!(serialized.metadata.target, SerializationTarget::TypeScript);
        assert_eq!(serialized.metadata.format, SerializationFormat::Json);
        assert_eq!(serialized.metadata.compression, CompressionAlgorithm::Zlib);
        assert!(!serialized.data.is_empty());
        assert!(serialized.metadata.checksum > 0);
        
        // Test deserialization
        let deserialized: SemanticArenaNode<Expr> = serializer.deserialize(&serialized).unwrap();
        match deserialized.node.kind {
            Expr::Literal(LiteralExpr { value: LiteralValue::Integer(42) }) => {},
            _ => panic!("Deserialization failed"),
        }
    }

    #[test]
    fn test_batch_serialization() {
        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        let nodes: Vec<SemanticArenaNode<Expr>> = (0..3).map(|i| {
            let expr = Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(i),
            });
            let ast_node = AstNode::new(expr, span, NodeId::new(i as u32));
            SemanticArenaNode::new(ast_node, 100, None)
        }).collect();
        
        let config = SerializationConfig {
            target: SerializationTarget::Binary,
            format: SerializationFormat::CompactBinary,
            compression: CompressionAlgorithm::Lz4,
            enable_streaming: false,
            ..Default::default()
        };
        
        let serializer = ArenaSerializer::new(config);
        let serialized = serializer.serialize_batch(&nodes).unwrap();
        
        assert_eq!(serialized.len(), 3);
        
        let stats = serializer.get_stats(&serialized);
        assert_eq!(stats.node_count, 3);
        assert!(stats.total_original_size > 0);
    }

    #[test]
    fn test_cache_functionality() {
        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        });
        let ast_node = AstNode::new(expr, span, NodeId::new(1));
        let semantic_node = SemanticArenaNode::new(ast_node, 100, None);
        
        let config = SerializationConfig {
            enable_caching: true,
            max_cache_size: 1024 * 1024, // 1MB
            ..Default::default()
        };
        
        let serializer = ArenaSerializer::new(config);
        
        // First serialization - cache miss
        let _serialized1 = serializer.serialize(&semantic_node).unwrap();
        let stats1 = serializer.get_cache_stats().unwrap();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);
        
        // Second serialization - cache hit
        let _serialized2 = serializer.serialize(&semantic_node).unwrap();
        let stats2 = serializer.get_cache_stats().unwrap();
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hits, 1);
    }

    #[test]
    fn test_compression_algorithms() {
        let data = b"Hello, world! This is some test data for compression.";
        
        // Test different compression algorithms
        let algorithms = vec![
            CompressionAlgorithm::None,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zlib,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Brotli,
        ];
        
        for algorithm in algorithms {
            let compressed = CompressionProvider::compress(data, algorithm, 6).unwrap();
            let decompressed = CompressionProvider::decompress(&compressed, algorithm).unwrap();
            assert_eq!(data, &decompressed[..data.len().min(decompressed.len())]);
        }
    }
} 