//! Performance Optimization for Arena System
//!
//! This module provides performance optimizations, caching strategies, and memory layout
//! optimizations for the AST arena system.

use crate::{AstNode, AstNodeRef};
use super::{ArenaError, ArenaResult, semantic::{SemanticArenaNode, AccessPatternType}};
use prism_common::{NodeId, SourceId};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, Duration, Instant};

/// Cache strategy for frequently accessed nodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Least Recently Used (LRU) eviction
    LRU { capacity: usize },
    /// Least Frequently Used (LFU) eviction
    LFU { capacity: usize },
    /// Time-based eviction
    TTL { capacity: usize, ttl: Duration },
    /// Adaptive Replacement Cache
    ARC { capacity: usize },
    /// No caching
    None,
}

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::LRU { capacity: 1024 }
    }
}

/// Optimization hints for memory layout and access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// Preferred memory alignment
    pub memory_alignment: usize,
    /// Cache line size optimization
    pub cache_line_size: usize,
    /// Prefetching strategy
    pub prefetch_strategy: PrefetchStrategy,
    /// Compression strategy
    pub compression_strategy: CompressionStrategy,
    /// Access pattern hints
    pub access_patterns: Vec<AccessPatternHint>,
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            memory_alignment: 64, // Cache line aligned
            cache_line_size: 64,
            prefetch_strategy: PrefetchStrategy::Sequential,
            compression_strategy: CompressionStrategy::LZ4,
            access_patterns: Vec::new(),
        }
    }
}

/// Prefetching strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential,
    /// Stride-based prefetching
    Stride { stride: usize },
    /// Adaptive prefetching
    Adaptive,
}

/// Compression strategies for node data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// LZ4 fast compression
    LZ4,
    /// Zstd balanced compression
    Zstd { level: i32 },
    /// Custom Prism compression
    PrismCustom,
}

/// Access pattern hint for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPatternHint {
    /// Pattern type
    pub pattern_type: AccessPatternType,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Expected frequency
    pub frequency: f64,
}

/// Performance metrics for the arena
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average access time
    pub average_access_time: Duration,
    /// Memory efficiency (useful data / total memory)
    pub memory_efficiency: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Fragmentation level (0.0 to 1.0)
    pub fragmentation_level: f64,
    /// GC pressure
    pub gc_pressure: f64,
}

/// Arena cache for frequently accessed nodes
pub struct ArenaCache {
    /// Cache strategy
    strategy: CacheStrategy,
    /// Cached nodes
    cache: HashMap<AstNodeRef, CachedNode>,
    /// Access order for LRU
    access_order: VecDeque<AstNodeRef>,
    /// Access frequency for LFU
    access_frequency: HashMap<AstNodeRef, u64>,
    /// Cache statistics
    stats: CacheStats,
}

/// Cached node with metadata
#[derive(Debug, Clone)]
struct CachedNode {
    /// Serialized node data
    data: Vec<u8>,
    /// Cache timestamp
    cached_at: SystemTime,
    /// Last access time
    last_accessed: SystemTime,
    /// Access count
    access_count: u64,
    /// Memory usage
    memory_usage: usize,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Cache evictions
    pub evictions: u64,
    /// Current cache size
    pub current_size: usize,
    /// Maximum cache size
    pub max_size: usize,
}

impl ArenaCache {
    /// Create a new arena cache
    pub fn new(strategy: CacheStrategy) -> Self {
        let max_size = Self::get_capacity(&strategy);
        Self {
            strategy,
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            access_frequency: HashMap::new(),
            stats: CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                current_size: 0,
                max_size,
            },
        }
    }

    /// Get a node from cache
    pub fn get(&mut self, node_ref: AstNodeRef) -> Option<Vec<u8>> {
        let result = if let Some(cached) = self.cache.get_mut(&node_ref) {
            // Update access metadata
            cached.last_accessed = SystemTime::now();
            cached.access_count += 1;
            
            self.stats.hits += 1;
            Some(cached.data.clone())
        } else {
            self.stats.misses += 1;
            None
        };
        
        // Update strategy-specific metadata after releasing the borrow
        if result.is_some() {
            self.update_access_metadata(node_ref);
        }
        
        result
    }

    /// Put a node in cache
    pub fn put(&mut self, node_ref: AstNodeRef, data: Vec<u8>) {
        let memory_usage = data.len();
        
        // Check if we need to evict
        while self.should_evict(memory_usage) {
            self.evict_one();
        }
        
        let cached_node = CachedNode {
            data,
            cached_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 1,
            memory_usage,
        };
        
        self.cache.insert(node_ref, cached_node);
        self.stats.current_size += memory_usage;
        
        // Update strategy-specific metadata
        self.update_access_metadata(node_ref);
    }

    /// Check if cache should evict to make room
    fn should_evict(&self, new_size: usize) -> bool {
        match &self.strategy {
            CacheStrategy::None => false,
            CacheStrategy::LRU { capacity } |
            CacheStrategy::LFU { capacity } |
            CacheStrategy::TTL { capacity, .. } |
            CacheStrategy::ARC { capacity } => {
                self.stats.current_size + new_size > *capacity
            }
        }
    }

    /// Evict one node based on strategy
    fn evict_one(&mut self) {
        let node_to_evict = match &self.strategy {
            CacheStrategy::None => return,
            CacheStrategy::LRU { .. } => {
                self.access_order.pop_front()
            },
            CacheStrategy::LFU { .. } => {
                self.find_least_frequent()
            },
            CacheStrategy::TTL { ttl, .. } => {
                self.find_expired(*ttl)
            },
            CacheStrategy::ARC { .. } => {
                // Simplified ARC - just use LRU for now
                self.access_order.pop_front()
            },
        };

        if let Some(node_ref) = node_to_evict {
            if let Some(cached) = self.cache.remove(&node_ref) {
                self.stats.current_size -= cached.memory_usage;
                self.stats.evictions += 1;
            }
            self.access_frequency.remove(&node_ref);
        }
    }

    /// Find least frequently used node
    fn find_least_frequent(&self) -> Option<AstNodeRef> {
        self.access_frequency
            .iter()
            .min_by_key(|(_, &freq)| freq)
            .map(|(&node_ref, _)| node_ref)
    }

    /// Find expired node based on TTL
    fn find_expired(&self, ttl: Duration) -> Option<AstNodeRef> {
        let now = SystemTime::now();
        self.cache
            .iter()
            .find(|(_, cached)| {
                now.duration_since(cached.cached_at).unwrap_or(Duration::ZERO) > ttl
            })
            .map(|(&node_ref, _)| node_ref)
    }

    /// Update access metadata for strategy
    fn update_access_metadata(&mut self, node_ref: AstNodeRef) {
        match &self.strategy {
            CacheStrategy::LRU { .. } => {
                // Move to back of access order
                if let Some(pos) = self.access_order.iter().position(|&x| x == node_ref) {
                    self.access_order.remove(pos);
                }
                self.access_order.push_back(node_ref);
            },
            CacheStrategy::LFU { .. } => {
                *self.access_frequency.entry(node_ref).or_insert(0) += 1;
            },
            _ => {}
        }
    }

    /// Get cache capacity
    fn get_capacity(strategy: &CacheStrategy) -> usize {
        match strategy {
            CacheStrategy::None => 0,
            CacheStrategy::LRU { capacity } |
            CacheStrategy::LFU { capacity } |
            CacheStrategy::TTL { capacity, .. } |
            CacheStrategy::ARC { capacity } => *capacity,
        }
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total > 0 {
            self.stats.hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.access_frequency.clear();
        self.stats.current_size = 0;
        self.stats.evictions += self.cache.len() as u64;
    }
}

/// Memory layout optimizer for cache efficiency
pub struct MemoryLayoutOptimizer {
    /// Optimization hints
    hints: OptimizationHints,
    /// Layout statistics
    stats: LayoutStats,
}

/// Layout optimization statistics
#[derive(Debug, Clone)]
pub struct LayoutStats {
    /// Cache line utilization
    pub cache_line_utilization: f64,
    /// Memory alignment efficiency
    pub alignment_efficiency: f64,
    /// Spatial locality score
    pub spatial_locality_score: f64,
    /// Temporal locality score
    pub temporal_locality_score: f64,
}

impl MemoryLayoutOptimizer {
    /// Create a new memory layout optimizer
    pub fn new(hints: OptimizationHints) -> Self {
        Self {
            hints,
            stats: LayoutStats {
                cache_line_utilization: 0.0,
                alignment_efficiency: 0.0,
                spatial_locality_score: 0.0,
                temporal_locality_score: 0.0,
            },
        }
    }

    /// Optimize memory layout for a set of nodes
    pub fn optimize_layout<T>(&mut self, nodes: &[SemanticArenaNode<T>]) -> LayoutOptimization
    where
        T: Serialize,
    {
        let total_size = nodes.iter().map(|n| n.arena_metadata.memory_usage).sum::<usize>();
        let aligned_size = self.calculate_aligned_size(total_size);
        
        // Analyze access patterns
        let access_patterns = self.analyze_access_patterns(nodes);
        
        // Calculate optimization metrics
        let cache_efficiency = self.calculate_cache_efficiency(nodes);
        let memory_savings = self.calculate_memory_savings(total_size, aligned_size);
        
        LayoutOptimization {
            original_size: total_size,
            optimized_size: aligned_size,
            memory_savings,
            cache_efficiency,
            access_patterns,
            recommendations: self.generate_recommendations(nodes),
        }
    }

    /// Calculate aligned size for cache efficiency
    fn calculate_aligned_size(&self, size: usize) -> usize {
        let alignment = self.hints.memory_alignment;
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Analyze access patterns
    fn analyze_access_patterns<T>(&self, nodes: &[SemanticArenaNode<T>]) -> Vec<AccessPatternAnalysis> {
        nodes.iter().map(|node| {
            let frequency = node.calculate_access_frequency();
            let age = node.arena_metadata.age();
            
            AccessPatternAnalysis {
                node_id: node.node.id,
                access_frequency: frequency,
                temporal_locality: self.calculate_temporal_locality(age, frequency),
                spatial_locality: self.calculate_spatial_locality(node),
                cache_friendliness: self.calculate_cache_friendliness(node),
            }
        }).collect()
    }

    /// Calculate temporal locality score
    fn calculate_temporal_locality(&self, age: Duration, frequency: f64) -> f64 {
        if age.as_secs_f64() > 0.0 {
            frequency / age.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate spatial locality score
    fn calculate_spatial_locality<T>(&self, node: &SemanticArenaNode<T>) -> f64 {
        // Simplified spatial locality based on memory usage and relationships
        let base_score = 1.0 - (node.arena_metadata.memory_usage as f64 / 1024.0).min(1.0);
        let relationship_bonus = node.semantic_metadata.relationships.len() as f64 * 0.1;
        (base_score + relationship_bonus).min(1.0)
    }

    /// Calculate cache friendliness
    fn calculate_cache_friendliness<T>(&self, node: &SemanticArenaNode<T>) -> f64 {
        let size_score = if node.arena_metadata.memory_usage <= self.hints.cache_line_size {
            1.0
        } else {
            self.hints.cache_line_size as f64 / node.arena_metadata.memory_usage as f64
        };
        
        let access_score = if node.is_hot_path() { 1.0 } else { 0.5 };
        
        (size_score + access_score) / 2.0
    }

    /// Calculate cache efficiency
    fn calculate_cache_efficiency<T>(&self, nodes: &[SemanticArenaNode<T>]) -> f64 {
        let cache_friendly_nodes = nodes.iter()
            .filter(|node| self.calculate_cache_friendliness(node) > 0.7)
            .count();
        
        if !nodes.is_empty() {
            cache_friendly_nodes as f64 / nodes.len() as f64
        } else {
            0.0
        }
    }

    /// Calculate memory savings from optimization
    fn calculate_memory_savings(&self, original: usize, optimized: usize) -> f64 {
        if original > 0 {
            (original as f64 - optimized as f64) / original as f64
        } else {
            0.0
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations<T>(&self, nodes: &[SemanticArenaNode<T>]) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Check for cache line alignment opportunities
        let misaligned_nodes = nodes.iter()
            .filter(|node| node.arena_metadata.memory_usage % self.hints.cache_line_size != 0)
            .count();
        
        if misaligned_nodes > nodes.len() / 2 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::MemoryAlignment,
                impact: 0.3,
                description: "Align node sizes to cache line boundaries".to_string(),
            });
        }
        
        // Check for compression opportunities
        let large_nodes = nodes.iter()
            .filter(|node| node.arena_metadata.memory_usage > 1024)
            .count();
        
        if large_nodes > 0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::Compression,
                impact: 0.5,
                description: "Compress large nodes to reduce memory usage".to_string(),
            });
        }
        
        recommendations
    }
}

/// Layout optimization result
#[derive(Debug, Clone)]
pub struct LayoutOptimization {
    /// Original memory size
    pub original_size: usize,
    /// Optimized memory size
    pub optimized_size: usize,
    /// Memory savings ratio
    pub memory_savings: f64,
    /// Cache efficiency score
    pub cache_efficiency: f64,
    /// Access pattern analysis
    pub access_patterns: Vec<AccessPatternAnalysis>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Access pattern analysis for a node
#[derive(Debug, Clone)]
pub struct AccessPatternAnalysis {
    /// Node identifier
    pub node_id: NodeId,
    /// Access frequency (accesses per second)
    pub access_frequency: f64,
    /// Temporal locality score
    pub temporal_locality: f64,
    /// Spatial locality score
    pub spatial_locality: f64,
    /// Cache friendliness score
    pub cache_friendliness: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of recommendation
    pub recommendation_type: OptimizationRecommendationType,
    /// Estimated impact (0.0 to 1.0)
    pub impact: f64,
    /// Description of the recommendation
    pub description: String,
}

/// Types of optimization recommendations
#[derive(Debug, Clone)]
pub enum OptimizationRecommendationType {
    /// Memory alignment optimization
    MemoryAlignment,
    /// Compression optimization
    Compression,
    /// Cache layout optimization
    CacheLayout,
    /// Access pattern optimization
    AccessPattern,
    /// Prefetching optimization
    Prefetching,
    /// Garbage collection optimization
    GarbageCollection,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, LiteralExpr, LiteralValue};
    use prism_common::span::{Span, Position};

    #[test]
    fn test_arena_cache_lru() {
        let mut cache = ArenaCache::new(CacheStrategy::LRU { capacity: 100 });
        let source_id = SourceId::new(1);
        
        let node_ref1 = AstNodeRef::new(1, source_id);
        let _node_ref2 = AstNodeRef::new(2, source_id);
        
        // Test cache miss
        assert!(cache.get(node_ref1).is_none());
        assert_eq!(cache.stats().misses, 1);
        
        // Test cache put and hit
        cache.put(node_ref1, vec![1, 2, 3, 4]);
        assert!(cache.get(node_ref1).is_some());
        assert_eq!(cache.stats().hits, 1);
        
        // Test cache size
        assert_eq!(cache.stats().current_size, 4);
    }

    #[test]
    fn test_optimization_hints() {
        let hints = OptimizationHints::default();
        assert_eq!(hints.memory_alignment, 64);
        assert_eq!(hints.cache_line_size, 64);
        assert!(matches!(hints.prefetch_strategy, PrefetchStrategy::Sequential));
    }

    #[test]
    fn test_memory_layout_optimizer() {
        let hints = OptimizationHints::default();
        let mut optimizer = MemoryLayoutOptimizer::new(hints);
        
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
        
        let nodes = vec![semantic_node];
        let optimization = optimizer.optimize_layout(&nodes);
        
        assert_eq!(optimization.original_size, 100);
        assert!(optimization.optimized_size >= 100); // May be aligned
        assert!(!optimization.access_patterns.is_empty());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = ArenaCache::new(CacheStrategy::LRU { capacity: 100 });
        let source_id = SourceId::new(1);
        
        let node_ref = AstNodeRef::new(1, source_id);
        
        // Initial hit rate should be 0
        assert_eq!(cache.hit_rate(), 0.0);
        
        // Add some hits and misses
        cache.put(node_ref, vec![1, 2, 3]);
        cache.get(node_ref); // hit
        cache.get(AstNodeRef::new(2, source_id)); // miss
        
        // Hit rate should be 0.5 (1 hit, 1 miss)
        assert_eq!(cache.hit_rate(), 0.5);
    }
} 