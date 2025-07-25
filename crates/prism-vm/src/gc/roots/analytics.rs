//! Analytics and performance monitoring for root set management
//!
//! This module provides comprehensive analytics, performance monitoring,
//! and optimization recommendations for root set operations.

use crate::{VMResult, PrismVMError};
use super::{types::*, interfaces::*};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Analytics and monitoring for root operations
pub struct RootAnalytics {
    /// Operation history for trend analysis
    operation_history: Arc<Mutex<VecDeque<RootAnalyticsOperation>>>,
    
    /// Performance metrics
    metrics: Arc<Mutex<PerformanceMetrics>>,
    
    /// Configuration
    config: AnalyticsConfig,
}

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    total_operations: u64,
    total_time_us: u64,
    operation_counts: HashMap<String, u64>,
    performance_trends: PerformanceTrends,
}

#[derive(Debug, Clone)]
struct AnalyticsConfig {
    max_history_size: usize,
    trend_analysis_window: usize,
    enable_recommendations: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            trend_analysis_window: 1000,
            enable_recommendations: true,
        }
    }
}

impl RootAnalytics {
    pub fn new() -> VMResult<Self> {
        Ok(Self {
            operation_history: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(Mutex::new(PerformanceMetrics {
                total_operations: 0,
                total_time_us: 0,
                operation_counts: HashMap::new(),
                performance_trends: PerformanceTrends {
                    scan_time_trend: TrendDirection::Unknown,
                    root_count_trend: TrendDirection::Unknown,
                    memory_usage_trend: TrendDirection::Unknown,
                    cache_efficiency_trend: TrendDirection::Unknown,
                    overall_trend: TrendDirection::Unknown,
                },
            })),
            config: AnalyticsConfig::default(),
        })
    }
}

impl RootAnalyticsInterface for RootAnalytics {
    fn record_operation(&mut self, operation: &RootAnalyticsOperation) -> RootOperationResult<()> {
        // Record in history
        {
            let mut history = self.operation_history.lock().unwrap();
            history.push_back(operation.clone());
            
            // Limit history size
            while history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_operations += 1;
            metrics.total_time_us += operation.duration.as_micros() as u64;
            
            let op_type = format!("{:?}", operation.operation_type);
            *metrics.operation_counts.entry(op_type).or_insert(0) += 1;
        }
        
        RootOperationResult::Success(())
    }
    
    fn get_performance_trends(&self) -> PerformanceTrends {
        self.metrics.lock().unwrap().performance_trends.clone()
    }
    
    fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        if !self.config.enable_recommendations {
            return Vec::new();
        }
        
        let mut recommendations = Vec::new();
        let metrics = self.metrics.lock().unwrap();
        
        // Analyze performance and generate recommendations
        if metrics.total_operations > 100 {
            let avg_time_us = metrics.total_time_us as f64 / metrics.total_operations as f64;
            
            if avg_time_us > 1000.0 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationType::ReduceScanTime,
                    priority: RecommendationPriority::High,
                    expected_improvement: 0.3,
                    difficulty: ImplementationDifficulty::Moderate,
                    description: "Root scanning is taking longer than expected. Consider enabling caching or optimizing scan order.".to_string(),
                    actions: vec![
                        "Enable frame caching".to_string(),
                        "Optimize scan order by priority".to_string(),
                        "Consider incremental scanning".to_string(),
                    ],
                });
            }
        }
        
        recommendations
    }
    
    fn generate_performance_report(&self) -> PerformanceReport {
        let metrics = self.metrics.lock().unwrap();
        let history = self.operation_history.lock().unwrap();
        
        let performance_score = if metrics.total_operations > 0 {
            let avg_time_us = metrics.total_time_us as f64 / metrics.total_operations as f64;
            // Simple scoring: lower time = higher score
            (1000.0 / (avg_time_us + 1.0) * 100.0).min(100.0)
        } else {
            100.0
        };
        
        let grade = match performance_score {
            90.0..=100.0 => PerformanceGrade::Excellent,
            80.0..=89.9 => PerformanceGrade::Good,
            70.0..=79.9 => PerformanceGrade::Fair,
            50.0..=69.9 => PerformanceGrade::Poor,
            _ => PerformanceGrade::Critical,
        };
        
        PerformanceReport {
            generated_at: std::time::SystemTime::now(),
            period: Duration::from_secs(3600), // 1 hour default
            summary: PerformanceSummary {
                performance_score,
                kpis: HashMap::new(),
                grade,
                description: format!("Root set management performance analysis based on {} operations", metrics.total_operations),
            },
            detailed_stats: RootStatistics {
                total_roots: 0, // Would be filled from actual data
                roots_by_source: HashMap::new(),
                roots_by_type: HashMap::new(),
                scan_stats: ScanStatistics {
                    total_scans: metrics.total_operations,
                    average_scan_time_us: if metrics.total_operations > 0 {
                        metrics.total_time_us as f64 / metrics.total_operations as f64
                    } else {
                        0.0
                    },
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
                performance_trends: metrics.performance_trends.clone(),
            },
            trends: metrics.performance_trends.clone(),
            recommendations: self.get_optimization_recommendations(),
            alerts: Vec::new(),
        }
    }
    
    fn reset_analytics(&mut self) -> RootOperationResult<()> {
        self.operation_history.lock().unwrap().clear();
        
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_operations = 0;
        metrics.total_time_us = 0;
        metrics.operation_counts.clear();
        
        RootOperationResult::Success(())
    }
} 