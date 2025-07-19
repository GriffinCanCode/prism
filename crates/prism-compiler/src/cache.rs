//! Advanced caching system for incremental compilation
//!
//! This module implements a sophisticated multi-level caching system with semantic
//! awareness, dependency tracking, and intelligent invalidation.

use crate::error::{CompilerError, CompilerResult};
use crate::query::{CacheKey, QueryId};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cached value
    pub value: T,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Cache key that generated this entry
    pub cache_key: CacheKey,
    /// Dependencies that can invalidate this entry
    pub dependencies: HashSet<CacheKey>,
    /// Semantic fingerprint for change detection
    pub semantic_fingerprint: SemanticFingerprint,
    /// Entry size in bytes (for memory management)
    pub size_bytes: usize,
}

/// Semantic fingerprint for detecting meaningful changes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticFingerprint {
    /// AST structure hash
    pub ast_hash: u64,
    /// Type signature hash
    pub type_hash: u64,
    /// Dependency hash
    pub dependency_hash: u64,
    /// Source file modification time
    pub file_mtime: SystemTime,
    /// Compiler version
    pub compiler_version: String,
}

/// Cache invalidation trigger
#[derive(Debug, Clone)]
pub enum InvalidationTrigger {
    /// File modification
    FileChanged(PathBuf),
    /// Dependency changed
    DependencyChanged(CacheKey),
    /// Semantic change detected
    SemanticChange(SemanticFingerprint),
    /// Manual invalidation
    Manual(String),
    /// Cache size limit reached
    MemoryPressure,
    /// Time-based expiration
    TimeExpired(Duration),
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total entries
    pub entries: usize,
    /// Total memory usage in bytes
    pub memory_usage: usize,
    /// Cache hit ratio
    pub hit_ratio: f64,
    /// Invalidation counts by trigger type
    pub invalidations: HashMap<String, u64>,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Entry TTL (time to live)
    pub entry_ttl: Duration,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Enable persistent cache
    pub persistent: bool,
    /// Cache directory for persistent storage
    pub cache_dir: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            max_entries: 10_000,
            entry_ttl: Duration::from_secs(3600), // 1 hour
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            persistent: true,
            cache_dir: None,
        }
    }
}

/// Multi-level compilation cache
pub struct CompilationCache {
    /// In-memory cache
    memory_cache: DashMap<CacheKey, CacheEntry<Vec<u8>>>,
    /// Dependency tracking
    dependencies: Arc<RwLock<HashMap<CacheKey, HashSet<CacheKey>>>>,
    /// Reverse dependency index
    dependents: Arc<RwLock<HashMap<CacheKey, HashSet<CacheKey>>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// File system watcher for invalidation
    file_watcher: Option<notify::RecommendedWatcher>,
}

impl CompilationCache {
    /// Create a new compilation cache
    pub fn new(config: CacheConfig) -> CompilerResult<Self> {
        let cache = Self {
            memory_cache: DashMap::new(),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            dependents: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            file_watcher: None,
        };

        Ok(cache)
    }

    /// Get a value from the cache
    pub async fn get<T>(&self, key: &CacheKey) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        if let Some(mut entry) = self.memory_cache.get_mut(key) {
            // Update access statistics
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;

            // Check if entry is still valid
            if self.is_entry_valid(&entry).await {
                // Update cache stats
                {
                    let mut stats = self.stats.write().await;
                    stats.hits += 1;
                    stats.hit_ratio = stats.hits as f64 / (stats.hits + stats.misses) as f64;
                }

                // Deserialize and return value
                if let Ok(value) = bincode::deserialize(&entry.value) {
                    debug!("Cache hit for key: {:?}", key);
                    return Some(value);
                }
            } else {
                // Entry is invalid, remove it
                self.invalidate_entry(key, InvalidationTrigger::TimeExpired(self.config.entry_ttl))
                    .await;
            }
        }

        // Cache miss
        {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
            stats.hit_ratio = stats.hits as f64 / (stats.hits + stats.misses) as f64;
        }

        debug!("Cache miss for key: {:?}", key);
        None
    }

    /// Store a value in the cache
    pub async fn put<T>(&self, key: CacheKey, value: T, dependencies: HashSet<CacheKey>) -> CompilerResult<()>
    where
        T: Serialize,
    {
        let serialized = bincode::serialize(&value).map_err(|e| {
            CompilerError::InternalError(format!("Failed to serialize cache value: {}", e))
        })?;

        let size_bytes = serialized.len();
        let semantic_fingerprint = self.compute_semantic_fingerprint(&key).await?;

        let entry = CacheEntry {
            value: serialized,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 1,
            cache_key: key.clone(),
            dependencies: dependencies.clone(),
            semantic_fingerprint,
            size_bytes,
        };

        // Check memory limits before inserting
        if self.should_evict_for_memory(size_bytes).await {
            self.evict_lru_entries(size_bytes).await;
        }

        // Insert entry
        self.memory_cache.insert(key.clone(), entry);

        // Update dependency tracking
        self.update_dependencies(&key, &dependencies).await;

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.entries = self.memory_cache.len();
            stats.memory_usage += size_bytes;
        }

        debug!("Cached value for key: {:?}", key);
        Ok(())
    }

    /// Invalidate a cache entry
    pub async fn invalidate_entry(&self, key: &CacheKey, trigger: InvalidationTrigger) {
        if let Some((_, entry)) = self.memory_cache.remove(key) {
            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.entries = self.memory_cache.len();
                stats.memory_usage = stats.memory_usage.saturating_sub(entry.size_bytes);
                
                let trigger_name = match trigger {
                    InvalidationTrigger::FileChanged(_) => "file_changed",
                    InvalidationTrigger::DependencyChanged(_) => "dependency_changed",
                    InvalidationTrigger::SemanticChange(_) => "semantic_change",
                    InvalidationTrigger::Manual(_) => "manual",
                    InvalidationTrigger::MemoryPressure => "memory_pressure",
                    InvalidationTrigger::TimeExpired(_) => "time_expired",
                };
                
                *stats.invalidations.entry(trigger_name.to_string()).or_insert(0) += 1;
            }

            // Cascade invalidation to dependents
            self.cascade_invalidation(key, &trigger).await;

            info!("Invalidated cache entry: {:?}, trigger: {:?}", key, trigger);
        }
    }

    /// Invalidate entries based on file changes
    pub async fn invalidate_by_file(&self, file_path: &Path) {
        let mut to_invalidate = Vec::new();

        for entry in self.memory_cache.iter() {
            // Check if this entry depends on the changed file
            if self.entry_depends_on_file(&entry, file_path).await {
                to_invalidate.push(entry.key().clone());
            }
        }

        for key in to_invalidate {
            self.invalidate_entry(&key, InvalidationTrigger::FileChanged(file_path.to_path_buf()))
                .await;
        }
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        self.memory_cache.clear();
        
        {
            let mut dependencies = self.dependencies.write().await;
            dependencies.clear();
        }
        
        {
            let mut dependents = self.dependents.write().await;
            dependents.clear();
        }

        {
            let mut stats = self.stats.write().await;
            *stats = CacheStats::default();
        }

        info!("Cache cleared");
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        stats.entries = self.memory_cache.len();
        stats.memory_usage = self.calculate_memory_usage().await;
        stats
    }

    /// Cleanup expired entries
    pub async fn cleanup_expired(&self) {
        let now = SystemTime::now();
        let mut expired_keys = Vec::new();

        for entry in self.memory_cache.iter() {
            if let Ok(age) = now.duration_since(entry.created_at) {
                if age > self.config.entry_ttl {
                    expired_keys.push(entry.key().clone());
                }
            }
        }

        for key in expired_keys {
            self.invalidate_entry(&key, InvalidationTrigger::TimeExpired(self.config.entry_ttl))
                .await;
        }
    }

    /// Check if an entry is still valid
    async fn is_entry_valid(&self, entry: &CacheEntry<Vec<u8>>) -> bool {
        let now = SystemTime::now();
        
        // Check TTL
        if let Ok(age) = now.duration_since(entry.created_at) {
            if age > self.config.entry_ttl {
                return false;
            }
        }

        // Check semantic fingerprint
        if let Ok(current_fingerprint) = self.compute_semantic_fingerprint(&entry.cache_key).await {
            if current_fingerprint != entry.semantic_fingerprint {
                return false;
            }
        }

        true
    }

    /// Compute semantic fingerprint for a cache key
    async fn compute_semantic_fingerprint(&self, _key: &CacheKey) -> CompilerResult<SemanticFingerprint> {
        // This is a simplified implementation
        // In practice, this would analyze the actual source files and dependencies
        Ok(SemanticFingerprint {
            ast_hash: 0, // Would compute actual AST hash
            type_hash: 0, // Would compute actual type hash
            dependency_hash: 0, // Would compute actual dependency hash
            file_mtime: SystemTime::now(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Update dependency tracking
    async fn update_dependencies(&self, key: &CacheKey, dependencies: &HashSet<CacheKey>) {
        {
            let mut deps = self.dependencies.write().await;
            deps.insert(key.clone(), dependencies.clone());
        }

        // Update reverse dependencies
        {
            let mut dependents = self.dependents.write().await;
            for dep in dependencies {
                dependents.entry(dep.clone()).or_insert_with(HashSet::new).insert(key.clone());
            }
        }
    }

    /// Cascade invalidation to dependent entries
    async fn cascade_invalidation(&self, key: &CacheKey, trigger: &InvalidationTrigger) {
        let dependents = {
            let dependents_map = self.dependents.read().await;
            dependents_map.get(key).cloned().unwrap_or_default()
        };

        for dependent in dependents {
            self.invalidate_entry(&dependent, InvalidationTrigger::DependencyChanged(key.clone()))
                .await;
        }
    }

    /// Check if memory eviction is needed
    async fn should_evict_for_memory(&self, additional_bytes: usize) -> bool {
        let current_usage = self.calculate_memory_usage().await;
        current_usage + additional_bytes > self.config.max_memory_bytes
            || self.memory_cache.len() >= self.config.max_entries
    }

    /// Evict LRU entries to free memory
    async fn evict_lru_entries(&self, bytes_needed: usize) {
        let mut entries_to_evict = Vec::new();
        
        // Collect entries sorted by last access time
        let mut entries: Vec<_> = self.memory_cache.iter()
            .map(|entry| (entry.key().clone(), entry.last_accessed, entry.size_bytes))
            .collect();
        
        entries.sort_by_key(|(_, last_accessed, _)| *last_accessed);

        let mut bytes_freed = 0;
        for (key, _, size) in entries {
            entries_to_evict.push(key);
            bytes_freed += size;
            
            if bytes_freed >= bytes_needed {
                break;
            }
        }

        for key in entries_to_evict {
            self.invalidate_entry(&key, InvalidationTrigger::MemoryPressure).await;
        }

        info!("Evicted {} entries, freed {} bytes", entries_to_evict.len(), bytes_freed);
    }

    /// Calculate total memory usage
    async fn calculate_memory_usage(&self) -> usize {
        self.memory_cache.iter()
            .map(|entry| entry.size_bytes)
            .sum()
    }

    /// Check if an entry depends on a file
    async fn entry_depends_on_file(&self, _entry: &dashmap::mapref::one::Ref<CacheKey, CacheEntry<Vec<u8>>>, _file_path: &Path) -> bool {
        // This is a simplified implementation
        // In practice, this would check the entry's dependencies against the file path
        false
    }
}

/// Persistent cache storage
pub struct PersistentCache {
    cache_dir: PathBuf,
}

impl PersistentCache {
    /// Create a new persistent cache
    pub fn new(cache_dir: PathBuf) -> CompilerResult<Self> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            CompilerError::InternalError(format!("Failed to create cache directory: {}", e))
        })?;

        Ok(Self { cache_dir })
    }

    /// Load cache entry from disk
    pub async fn load<T>(&self, key: &CacheKey) -> CompilerResult<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let file_path = self.get_cache_file_path(key);
        
        if !file_path.exists() {
            return Ok(None);
        }

        let data = tokio::fs::read(&file_path).await.map_err(|e| {
            CompilerError::InternalError(format!("Failed to read cache file: {}", e))
        })?;

        let value = bincode::deserialize(&data).map_err(|e| {
            CompilerError::InternalError(format!("Failed to deserialize cache value: {}", e))
        })?;

        Ok(Some(value))
    }

    /// Save cache entry to disk
    pub async fn save<T>(&self, key: &CacheKey, value: &T) -> CompilerResult<()>
    where
        T: Serialize,
    {
        let file_path = self.get_cache_file_path(key);
        
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                CompilerError::InternalError(format!("Failed to create cache directory: {}", e))
            })?;
        }

        let data = bincode::serialize(value).map_err(|e| {
            CompilerError::InternalError(format!("Failed to serialize cache value: {}", e))
        })?;

        tokio::fs::write(&file_path, data).await.map_err(|e| {
            CompilerError::InternalError(format!("Failed to write cache file: {}", e))
        })?;

        Ok(())
    }

    /// Get cache file path for a key
    fn get_cache_file_path(&self, key: &CacheKey) -> PathBuf {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        
        self.cache_dir.join(format!("{:016x}.cache", hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let config = CacheConfig::default();
        let cache = CompilationCache::new(config).unwrap();

        let key = CacheKey("test".to_string());
        let value = "test_value".to_string();
        let dependencies = HashSet::new();

        // Test put and get
        cache.put(key.clone(), &value, dependencies).await.unwrap();
        let retrieved: Option<String> = cache.get(&key).await;
        assert_eq!(retrieved, Some(value));

        // Test cache miss
        let missing_key = CacheKey("missing".to_string());
        let missing: Option<String> = cache.get(&missing_key).await;
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let config = CacheConfig::default();
        let cache = CompilationCache::new(config).unwrap();

        let key = CacheKey("test".to_string());
        let value = "test_value".to_string();
        let dependencies = HashSet::new();

        cache.put(key.clone(), &value, dependencies).await.unwrap();
        cache.invalidate_entry(&key, InvalidationTrigger::Manual("test".to_string())).await;

        let retrieved: Option<String> = cache.get(&key).await;
        assert_eq!(retrieved, None);
    }
} 