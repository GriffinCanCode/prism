//! Performance Monitoring and Metrics for JavaScript Backend
//!
//! This module provides performance monitoring, profiling, and metrics
//! collection for generated JavaScript code.

use super::{JavaScriptResult, JavaScriptError};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Enable profiling instrumentation
    pub enable_profiling: bool,
    /// Enable memory usage tracking
    pub enable_memory_tracking: bool,
    /// Performance budget thresholds
    pub budget: PerformanceBudget,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_profiling: false,
            enable_memory_tracking: true,
            budget: PerformanceBudget::default(),
        }
    }
}

/// Performance budget thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBudget {
    /// Maximum bundle size in bytes
    pub max_bundle_size: usize,
    /// Maximum execution time in milliseconds
    pub max_execution_time: u64,
    /// Maximum memory usage in bytes
    pub max_memory_usage: usize,
}

impl Default for PerformanceBudget {
    fn default() -> Self {
        Self {
            max_bundle_size: 1024 * 1024, // 1MB
            max_execution_time: 1000,     // 1 second
            max_memory_usage: 50 * 1024 * 1024, // 50MB
        }
    }
}

/// Performance monitor for JavaScript backend
pub struct PerformanceMonitor {
    config: PerformanceConfig,
    metrics: PerformanceMetrics,
    start_time: Option<Instant>,
}

impl PerformanceMonitor {
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            metrics: PerformanceMetrics::default(),
            start_time: None,
        }
    }

    pub fn start_monitoring(&mut self) {
        if self.config.enabled {
            self.start_time = Some(Instant::now());
        }
    }

    pub fn stop_monitoring(&mut self) -> JavaScriptResult<()> {
        if let Some(start_time) = self.start_time.take() {
            self.metrics.total_time = start_time.elapsed();
        }
        Ok(())
    }

    pub fn record_bundle_size(&mut self, size: usize) {
        self.metrics.bundle_size = size;
    }

    pub fn record_memory_usage(&mut self, usage: usize) {
        self.metrics.memory_usage = usage;
    }

    pub fn generate_performance_report(&self) -> PerformanceReport {
        PerformanceReport {
            metrics: self.metrics.clone(),
            budget_violations: self.check_budget_violations(),
            recommendations: self.generate_recommendations(),
        }
    }

    pub fn generate_instrumentation_code(&self) -> JavaScriptResult<String> {
        if !self.config.enable_profiling {
            return Ok(String::new());
        }

        let mut code = String::new();
        
        code.push_str("// Performance monitoring instrumentation\n");
        code.push_str("const PERF_MONITOR = {\n");
        code.push_str("  startTime: Date.now(),\n");
        code.push_str("  marks: new Map(),\n");
        code.push_str("  measures: new Map(),\n");
        code.push_str("  \n");
        code.push_str("  mark(name) {\n");
        code.push_str("    this.marks.set(name, performance.now());\n");
        code.push_str("  },\n");
        code.push_str("  \n");
        code.push_str("  measure(name, startMark, endMark) {\n");
        code.push_str("    const start = this.marks.get(startMark) || 0;\n");
        code.push_str("    const end = this.marks.get(endMark) || performance.now();\n");
        code.push_str("    this.measures.set(name, end - start);\n");
        code.push_str("  },\n");
        code.push_str("  \n");
        code.push_str("  getReport() {\n");
        code.push_str("    return {\n");
        code.push_str("      marks: Object.fromEntries(this.marks),\n");
        code.push_str("      measures: Object.fromEntries(this.measures),\n");
        code.push_str("      totalTime: Date.now() - this.startTime\n");
        code.push_str("    };\n");
        code.push_str("  }\n");
        code.push_str("};\n\n");

        if self.config.enable_memory_tracking {
            code.push_str("// Memory usage tracking\n");
            code.push_str("const MEMORY_MONITOR = {\n");
            code.push_str("  track() {\n");
            code.push_str("    if (performance.memory) {\n");
            code.push_str("      return {\n");
            code.push_str("        used: performance.memory.usedJSHeapSize,\n");
            code.push_str("        total: performance.memory.totalJSHeapSize,\n");
            code.push_str("        limit: performance.memory.jsHeapSizeLimit\n");
            code.push_str("      };\n");
            code.push_str("    }\n");
            code.push_str("    return null;\n");
            code.push_str("  }\n");
            code.push_str("};\n\n");
        }

        Ok(code)
    }

    fn check_budget_violations(&self) -> Vec<BudgetViolation> {
        let mut violations = Vec::new();

        if self.metrics.bundle_size > self.config.budget.max_bundle_size {
            violations.push(BudgetViolation {
                metric: "bundle_size".to_string(),
                actual: self.metrics.bundle_size as f64,
                threshold: self.config.budget.max_bundle_size as f64,
                severity: ViolationSeverity::High,
            });
        }

        if self.metrics.total_time.as_millis() as u64 > self.config.budget.max_execution_time {
            violations.push(BudgetViolation {
                metric: "execution_time".to_string(),
                actual: self.metrics.total_time.as_millis() as f64,
                threshold: self.config.budget.max_execution_time as f64,
                severity: ViolationSeverity::Medium,
            });
        }

        if self.metrics.memory_usage > self.config.budget.max_memory_usage {
            violations.push(BudgetViolation {
                metric: "memory_usage".to_string(),
                actual: self.metrics.memory_usage as f64,
                threshold: self.config.budget.max_memory_usage as f64,
                severity: ViolationSeverity::High,
            });
        }

        violations
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.metrics.bundle_size > self.config.budget.max_bundle_size {
            recommendations.push("Consider enabling tree shaking to reduce bundle size".to_string());
            recommendations.push("Use dynamic imports for code splitting".to_string());
        }

        if self.metrics.total_time.as_millis() > 500 {
            recommendations.push("Consider optimizing hot paths for better performance".to_string());
        }

        if self.metrics.memory_usage > self.config.budget.max_memory_usage / 2 {
            recommendations.push("Monitor memory usage to prevent leaks".to_string());
        }

        recommendations
    }
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub bundle_size: usize,
    pub memory_usage: usize,
    pub total_time: Duration,
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub metrics: PerformanceMetrics,
    pub budget_violations: Vec<BudgetViolation>,
    pub recommendations: Vec<String>,
}

/// Budget violation
#[derive(Debug, Clone)]
pub struct BudgetViolation {
    pub metric: String,
    pub actual: f64,
    pub threshold: f64,
    pub severity: ViolationSeverity,
}

#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
} 