//! Quality Metrics - Performance and Cohesion Analysis
//!
//! This module provides comprehensive quality metrics including performance
//! characteristics, cohesion analysis, and architectural quality indicators.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance profile for code components
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// CPU usage characteristics
    pub cpu_usage: CPUUsageProfile,
    /// Memory usage characteristics
    pub memory_usage: MemoryUsageProfile,
    /// I/O characteristics
    pub io_characteristics: IOProfile,
    /// Network usage characteristics
    pub network_usage: NetworkProfile,
    /// Scalability characteristics
    pub scalability: ScalabilityProfile,
}

/// CPU usage profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUUsageProfile {
    /// Intensity level
    pub intensity: IntensityLevel,
    /// Can be parallelized
    pub parallelizable: bool,
    /// Can be vectorized
    pub vectorizable: bool,
    /// CPU-bound operations
    pub cpu_bound_operations: Vec<String>,
    /// Estimated CPU cycles
    pub estimated_cycles: Option<u64>,
}

/// Memory usage profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageProfile {
    /// Estimated allocation size
    pub estimated_allocation: Option<usize>,
    /// Allocation pattern
    pub allocation_pattern: AllocationPattern,
    /// Memory locality characteristics
    pub locality: MemoryLocality,
    /// Memory-intensive operations
    pub memory_intensive_operations: Vec<String>,
    /// Peak memory usage
    pub peak_usage: Option<usize>,
}

/// I/O profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOProfile {
    /// File system usage
    pub filesystem: bool,
    /// Network usage
    pub network: bool,
    /// Database usage
    pub database: bool,
    /// I/O patterns
    pub io_patterns: Vec<IOPattern>,
    /// Estimated I/O operations
    pub estimated_operations: Option<u64>,
}

/// Network profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfile {
    /// Makes network calls
    pub makes_calls: bool,
    /// Estimated request count
    pub estimated_requests: Option<usize>,
    /// Protocols used
    pub protocols: Vec<String>,
    /// Network patterns
    pub patterns: Vec<NetworkPattern>,
    /// Bandwidth requirements
    pub bandwidth_requirements: Option<String>,
}

/// Scalability profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityProfile {
    /// Horizontal scalability
    pub horizontal_scalability: ScalabilityLevel,
    /// Vertical scalability
    pub vertical_scalability: ScalabilityLevel,
    /// Scaling bottlenecks
    pub bottlenecks: Vec<String>,
    /// Scaling strategies
    pub strategies: Vec<String>,
}

/// Intensity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntensityLevel {
    /// Low intensity
    Low,
    /// Medium intensity
    Medium,
    /// High intensity
    High,
    /// Critical intensity
    Critical,
}

/// Allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPattern {
    /// Stack allocation
    Stack,
    /// Heap allocation
    Heap,
    /// Pool allocation
    Pool,
    /// Arena allocation
    Arena,
}

/// Memory locality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLocality {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Temporal locality
    Temporal,
    /// Spatial locality
    Spatial,
}

/// I/O pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IOPattern {
    /// Sequential I/O
    Sequential,
    /// Random I/O
    Random,
    /// Batch I/O
    Batch,
    /// Streaming I/O
    Streaming,
}

/// Network pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPattern {
    /// Request-response
    RequestResponse,
    /// Streaming
    Streaming,
    /// Publish-subscribe
    PubSub,
    /// Broadcast
    Broadcast,
}

/// Scalability level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalabilityLevel {
    /// Poor scalability
    Poor,
    /// Limited scalability
    Limited,
    /// Good scalability
    Good,
    /// Excellent scalability
    Excellent,
}

/// Transformation history for audit trails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationHistory {
    /// Applied transformations
    pub transformations: Vec<TransformationRecord>,
    /// Transformation metadata
    pub metadata: TransformationMetadata,
}

/// Transformation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRecord {
    /// Transformation ID
    pub id: String,
    /// Transformation name
    pub name: String,
    /// Transformation version
    pub version: String,
    /// Application timestamp
    pub timestamp: String,
    /// Input hash
    pub input_hash: String,
    /// Output hash
    pub output_hash: String,
    /// Configuration used
    pub configuration: HashMap<String, String>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Success flag
    pub success: bool,
    /// Error message (if any)
    pub error: Option<String>,
}

/// Transformation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationMetadata {
    /// Total transformations applied
    pub total_transformations: usize,
    /// Total processing time
    pub total_duration_ms: u64,
    /// First transformation timestamp
    pub first_transformation: Option<String>,
    /// Last transformation timestamp
    pub last_transformation: Option<String>,
    /// Transformation chain integrity
    pub chain_integrity: bool,
}

impl Default for CPUUsageProfile {
    fn default() -> Self {
        Self {
            intensity: IntensityLevel::Medium,
            parallelizable: false,
            vectorizable: false,
            cpu_bound_operations: Vec::new(),
            estimated_cycles: None,
        }
    }
}

impl Default for MemoryUsageProfile {
    fn default() -> Self {
        Self {
            estimated_allocation: None,
            allocation_pattern: AllocationPattern::Heap,
            locality: MemoryLocality::Sequential,
            memory_intensive_operations: Vec::new(),
            peak_usage: None,
        }
    }
}

impl Default for IOProfile {
    fn default() -> Self {
        Self {
            filesystem: false,
            network: false,
            database: false,
            io_patterns: Vec::new(),
            estimated_operations: None,
        }
    }
}

impl Default for NetworkProfile {
    fn default() -> Self {
        Self {
            makes_calls: false,
            estimated_requests: None,
            protocols: Vec::new(),
            patterns: Vec::new(),
            bandwidth_requirements: None,
        }
    }
}

impl Default for ScalabilityProfile {
    fn default() -> Self {
        Self {
            horizontal_scalability: ScalabilityLevel::Good,
            vertical_scalability: ScalabilityLevel::Good,
            bottlenecks: Vec::new(),
            strategies: Vec::new(),
        }
    }
}

impl TransformationHistory {
    /// Add a transformation record
    pub fn add_transformation(&mut self, record: TransformationRecord) {
        self.transformations.push(record);
        
        // Update metadata
        self.metadata.total_transformations = self.transformations.len();
        self.metadata.total_duration_ms = self.transformations.iter().map(|t| t.duration_ms).sum();
        
        if self.metadata.first_transformation.is_none() {
            self.metadata.first_transformation = Some(self.transformations[0].timestamp.clone());
        }
        self.metadata.last_transformation = self.transformations.last().map(|t| t.timestamp.clone());
    }

    /// Verify transformation chain integrity
    pub fn verify_integrity(&mut self) -> bool {
        // Simple integrity check - could be more sophisticated
        let all_successful = self.transformations.iter().all(|t| t.success);
        self.metadata.chain_integrity = all_successful;
        all_successful
    }
} 