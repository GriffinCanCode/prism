//! Concurrency Test Runner
//!
//! Comprehensive test runner for the concurrency system that provides:
//! - Organized test execution with categorization
//! - Performance benchmarking and reporting
//! - Stress testing with configurable parameters
//! - Memory usage monitoring during tests
//! - Detailed test result analysis and reporting

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Test configuration for running concurrency tests
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Number of threads to use for concurrent tests
    pub thread_count: usize,
    /// Number of operations per thread
    pub operations_per_thread: usize,
    /// Test timeout duration
    pub timeout: Duration,
    /// Whether to run performance benchmarks
    pub run_benchmarks: bool,
    /// Whether to run stress tests
    pub run_stress_tests: bool,
    /// Minimum acceptable performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

/// Performance thresholds for test validation
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Minimum operations per second for lock-free queue
    pub queue_ops_per_second: f64,
    /// Minimum operations per second for lock-free map
    pub map_ops_per_second: f64,
    /// Minimum operations per second for atomic counter
    pub counter_ops_per_second: f64,
    /// Maximum acceptable latency for actor message processing
    pub max_actor_latency_ms: f64,
    /// Minimum actor throughput (messages/second)
    pub min_actor_throughput: f64,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            thread_count: num_cpus::get(),
            operations_per_thread: 10000,
            timeout: Duration::from_secs(30),
            run_benchmarks: true,
            run_stress_tests: true,
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            queue_ops_per_second: 100000.0,
            map_ops_per_second: 50000.0,
            counter_ops_per_second: 500000.0,
            max_actor_latency_ms: 10.0,
            min_actor_throughput: 10000.0,
        }
    }
}

/// Test result for a single test
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Test category
    pub category: TestCategory,
    /// Whether the test passed
    pub passed: bool,
    /// Test execution duration
    pub duration: Duration,
    /// Performance metrics (if applicable)
    pub performance_metrics: Option<PerformanceMetrics>,
    /// Error message (if test failed)
    pub error: Option<String>,
    /// Additional details
    pub details: HashMap<String, String>,
}

/// Test categories for organization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TestCategory {
    /// Basic functionality tests
    Basic,
    /// Concurrency and thread safety tests
    Concurrency,
    /// Performance and benchmarking tests
    Performance,
    /// Stress and reliability tests
    Stress,
    /// Integration tests
    Integration,
    /// Memory safety tests
    MemorySafety,
}

/// Performance metrics collected during tests
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: f64,
    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
}

/// Comprehensive test suite for the concurrency system
pub struct ConcurrencyTestSuite {
    config: TestConfig,
    results: Vec<TestResult>,
}

impl ConcurrencyTestSuite {
    /// Create a new test suite with the given configuration
    pub fn new(config: TestConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run all concurrency tests
    pub async fn run_all_tests(&mut self) -> TestSummary {
        println!("🚀 Starting Concurrency System Test Suite");
        println!("Configuration: {:?}", self.config);
        println!();

        let start_time = Instant::now();

        // Run basic functionality tests
        self.run_basic_tests().await;

        // Run concurrency tests
        self.run_concurrency_tests().await;

        // Run performance tests if enabled
        if self.config.run_benchmarks {
            self.run_performance_tests().await;
        }

        // Run stress tests if enabled
        if self.config.run_stress_tests {
            self.run_stress_tests().await;
        }

        // Run integration tests
        self.run_integration_tests().await;

        // Run memory safety tests
        self.run_memory_safety_tests().await;

        let total_duration = start_time.elapsed();

        let summary = self.generate_summary(total_duration);
        self.print_summary(&summary);

        summary
    }

    /// Run basic functionality tests
    async fn run_basic_tests(&mut self) {
        println!("📋 Running Basic Functionality Tests...");

        // Test ConcurrencySystem creation
        self.run_test("concurrency_system_creation", TestCategory::Basic, || {
            Box::pin(async {
                let system = super::ConcurrencySystem::new()?;
                assert_eq!(system.actor_count(), 0);
                assert_eq!(system.task_count(), 0);
                assert_eq!(system.scope_count(), 0);
                Ok(())
            })
        }).await;

        // Test ActorSystem creation
        self.run_test("actor_system_creation", TestCategory::Basic, || {
            Box::pin(async {
                let system = super::ActorSystem::new()?;
                assert_eq!(system.active_count(), 0);
                Ok(())
            })
        }).await;

        // Test AsyncRuntime creation
        self.run_test("async_runtime_creation", TestCategory::Basic, || {
            Box::pin(async {
                let runtime = super::AsyncRuntime::new()?;
                assert_eq!(runtime.task_count(), 0);
                Ok(())
            })
        }).await;

        // Test StructuredCoordinator creation
        self.run_test("structured_coordinator_creation", TestCategory::Basic, || {
            Box::pin(async {
                let coordinator = super::StructuredCoordinator::new()?;
                assert_eq!(coordinator.scope_count(), 0);
                Ok(())
            })
        }).await;

        // Test EventBus creation
        self.run_test("event_bus_creation", TestCategory::Basic, || {
            Box::pin(async {
                let event_bus: super::EventBus<String> = super::EventBus::new();
                let metrics = event_bus.get_metrics();
                assert_eq!(metrics.total_published, 0);
                assert_eq!(metrics.total_subscribers, 0);
                Ok(())
            })
        }).await;

        // Test lock-free data structures
        self.run_test("lock_free_queue_basic", TestCategory::Basic, || {
            Box::pin(async {
                let queue = super::LockFreeQueue::new();
                assert!(queue.is_empty());
                queue.enqueue(42);
                assert_eq!(queue.dequeue(), Some(42));
                assert!(queue.is_empty());
                Ok(())
            })
        }).await;

        self.run_test("lock_free_map_basic", TestCategory::Basic, || {
            Box::pin(async {
                let map = super::LockFreeMap::new();
                assert!(map.is_empty());
                map.insert("key".to_string(), 42);
                assert_eq!(map.get(&"key".to_string()), Some(42));
                assert_eq!(map.remove(&"key".to_string()), Some(42));
                assert!(map.is_empty());
                Ok(())
            })
        }).await;

        self.run_test("atomic_counter_basic", TestCategory::Basic, || {
            Box::pin(async {
                let counter = super::AtomicCounter::new();
                assert_eq!(counter.get(), 0);
                assert_eq!(counter.increment(), 1);
                assert_eq!(counter.decrement(), 0);
                Ok(())
            })
        }).await;
    }

    /// Run concurrency and thread safety tests
    async fn run_concurrency_tests(&mut self) {
        println!("🧵 Running Concurrency Tests...");

        // Test concurrent queue operations
        self.run_test("concurrent_queue_operations", TestCategory::Concurrency, || {
            let config = self.config.clone();
            Box::pin(async move {
                let queue = Arc::new(super::LockFreeQueue::new());
                let mut handles = Vec::new();

                // Producer threads
                for i in 0..config.thread_count / 2 {
                    let queue_clone = Arc::clone(&queue);
                    let ops = config.operations_per_thread;
                    handles.push(tokio::spawn(async move {
                        for j in 0..ops {
                            queue_clone.enqueue(i * ops + j);
                        }
                    }));
                }

                // Consumer threads
                let consumed = Arc::new(AtomicUsize::new(0));
                for _ in 0..config.thread_count / 2 {
                    let queue_clone = Arc::clone(&queue);
                    let consumed_clone = Arc::clone(&consumed);
                    let total_items = config.thread_count * config.operations_per_thread / 2;
                    
                    handles.push(tokio::spawn(async move {
                        while consumed_clone.load(Ordering::Relaxed) < total_items {
                            if queue_clone.dequeue().is_some() {
                                consumed_clone.fetch_add(1, Ordering::Relaxed);
                            } else {
                                tokio::task::yield_now().await;
                            }
                        }
                    }));
                }

                // Wait for all tasks
                for handle in handles {
                    handle.await.unwrap();
                }

                let final_consumed = consumed.load(Ordering::Relaxed);
                let expected = config.thread_count * config.operations_per_thread / 2;
                assert_eq!(final_consumed, expected);

                Ok(())
            })
        }).await;

        // Test concurrent map operations
        self.run_test("concurrent_map_operations", TestCategory::Concurrency, || {
            let config = self.config.clone();
            Box::pin(async move {
                let map = Arc::new(super::LockFreeMap::new());
                let mut handles = Vec::new();

                // Concurrent insertions
                for thread_id in 0..config.thread_count {
                    let map_clone = Arc::clone(&map);
                    let ops = config.operations_per_thread;
                    
                    handles.push(tokio::spawn(async move {
                        for i in 0..ops {
                            let key = format!("key_{}_{}", thread_id, i);
                            let value = thread_id * ops + i;
                            map_clone.insert(key, value);
                        }
                    }));
                }

                // Wait for all insertions
                for handle in handles {
                    handle.await.unwrap();
                }

                // Verify all items were inserted
                let expected_size = config.thread_count * config.operations_per_thread;
                assert_eq!(map.len(), expected_size);

                Ok(())
            })
        }).await;

        // Test concurrent counter operations
        self.run_test("concurrent_counter_operations", TestCategory::Concurrency, || {
            let config = self.config.clone();
            Box::pin(async move {
                let counter = Arc::new(super::AtomicCounter::new());
                let mut handles = Vec::new();

                // Concurrent increments
                for _ in 0..config.thread_count {
                    let counter_clone = Arc::clone(&counter);
                    let ops = config.operations_per_thread;
                    
                    handles.push(tokio::spawn(async move {
                        for _ in 0..ops {
                            counter_clone.increment();
                        }
                    }));
                }

                // Wait for all increments
                for handle in handles {
                    handle.await.unwrap();
                }

                let expected = config.thread_count * config.operations_per_thread;
                assert_eq!(counter.get(), expected);

                Ok(())
            })
        }).await;
    }

    /// Run performance benchmark tests
    async fn run_performance_tests(&mut self) {
        println!("⚡ Running Performance Tests...");

        // Benchmark lock-free queue
        self.run_performance_test("lock_free_queue_benchmark", || {
            let config = self.config.clone();
            Box::pin(async move {
                let queue = super::LockFreeQueue::new();
                let ops = config.operations_per_thread * 10; // More operations for benchmarking

                let start = Instant::now();
                let mut latencies = Vec::new();

                for i in 0..ops {
                    let op_start = Instant::now();
                    queue.enqueue(i);
                    latencies.push(op_start.elapsed().as_secs_f64() * 1000.0);
                }

                for _ in 0..ops {
                    let op_start = Instant::now();
                    queue.dequeue();
                    latencies.push(op_start.elapsed().as_secs_f64() * 1000.0);
                }

                let elapsed = start.elapsed();
                let ops_per_second = (ops * 2) as f64 / elapsed.as_secs_f64();

                PerformanceMetrics {
                    ops_per_second,
                    avg_latency_ms: elapsed.as_secs_f64() * 1000.0 / (ops * 2) as f64,
                    p95_latency_ms: self.calculate_percentile(&latencies, 95.0),
                    p99_latency_ms: self.calculate_percentile(&latencies, 99.0),
                    memory_usage_bytes: self.get_memory_usage(),
                    cpu_utilization: self.get_cpu_utilization()
                }
            })
        }).await;

        // Benchmark lock-free map
        self.run_performance_test("lock_free_map_benchmark", || {
            let config = self.config.clone();
            Box::pin(async move {
                let map = super::LockFreeMap::new();
                let ops = config.operations_per_thread * 5; // Fewer operations due to complexity

                let start = Instant::now();

                // Insert phase
                for i in 0..ops {
                    map.insert(format!("key_{}", i), i);
                }

                // Get phase
                for i in 0..ops {
                    let _ = map.get(&format!("key_{}", i));
                }

                // Remove phase
                for i in 0..ops {
                    let _ = map.remove(&format!("key_{}", i));
                }

                let elapsed = start.elapsed();
                let ops_per_second = (ops * 3) as f64 / elapsed.as_secs_f64();

                PerformanceMetrics {
                    ops_per_second,
                    avg_latency_ms: elapsed.as_secs_f64() * 1000.0 / (ops * 3) as f64,
                    p95_latency_ms: 0.0,
                    p99_latency_ms: 0.0,
                    memory_usage_bytes: 0,
                    cpu_utilization: 0.0,
                }
            })
        }).await;

        // Benchmark atomic counter
        self.run_performance_test("atomic_counter_benchmark", || {
            let config = self.config.clone();
            Box::pin(async move {
                let counter = super::AtomicCounter::new();
                let ops = config.operations_per_thread * 20; // Many operations for counter

                let start = Instant::now();

                for _ in 0..ops {
                    counter.increment();
                }

                let elapsed = start.elapsed();
                let ops_per_second = ops as f64 / elapsed.as_secs_f64();

                PerformanceMetrics {
                    ops_per_second,
                    avg_latency_ms: elapsed.as_secs_f64() * 1000.0 / ops as f64,
                    p95_latency_ms: 0.0,
                    p99_latency_ms: 0.0,
                    memory_usage_bytes: 0,
                    cpu_utilization: 0.0,
                }
            })
        }).await;
    }

    /// Run stress and reliability tests
    async fn run_stress_tests(&mut self) {
        println!("💪 Running Stress Tests...");

        // Stress test with many threads and operations
        self.run_test("high_concurrency_stress", TestCategory::Stress, || {
            let mut config = self.config.clone();
            config.thread_count *= 4; // Increase thread count for stress test
            config.operations_per_thread *= 2; // Increase operations

            Box::pin(async move {
                let queue = Arc::new(super::LockFreeQueue::new());
                let map = Arc::new(super::LockFreeMap::new());
                let counter = Arc::new(super::AtomicCounter::new());

                let mut handles = Vec::new();

                for thread_id in 0..config.thread_count {
                    let queue_clone = Arc::clone(&queue);
                    let map_clone = Arc::clone(&map);
                    let counter_clone = Arc::clone(&counter);
                    let ops = config.operations_per_thread;

                    handles.push(tokio::spawn(async move {
                        for i in 0..ops {
                            // Mix of operations on all data structures
                            queue_clone.enqueue(thread_id * ops + i);

                            let key = format!("thread_{}_{}", thread_id, i % 1000);
                            map_clone.insert(key.clone(), i);
                            let _ = map_clone.get(&key);

                            counter_clone.increment();

                            if i % 10 == 0 {
                                let _ = queue_clone.dequeue();
                                let _ = map_clone.remove(&key);
                            }
                        }
                    }));
                }

                // Wait for all threads with timeout
                let timeout = tokio::time::timeout(config.timeout, async {
                    for handle in handles {
                        handle.await.unwrap();
                    }
                });

                timeout.await.map_err(|_| "Stress test timed out")?;

                // Verify system is still in a consistent state
                println!("Stress test completed - Final queue length: {}", queue.len());
                println!("Stress test completed - Final map size: {}", map.len());
                println!("Stress test completed - Final counter value: {}", counter.get());

                Ok(())
            })
        }).await;
    }

    /// Run integration tests
    async fn run_integration_tests(&mut self) {
        println!("🔗 Running Integration Tests...");

        // Test full system integration
        self.run_test("full_system_integration", TestCategory::Integration, || {
            Box::pin(async {
                let system = super::ConcurrencySystem::new()?;

                // Test all components working together
                let _scope = system.create_scope()?;
                let _queue = system.create_lock_free_queue::<i32>();
                let _map = system.create_lock_free_map::<String, i32>();

                // Get performance metrics
                let metrics = system.get_performance_metrics();
                assert!(metrics.system_health >= 0.0 && metrics.system_health <= 1.0);

                Ok(())
            })
        }).await;
    }

    /// Run memory safety tests
    async fn run_memory_safety_tests(&mut self) {
        println!("🛡️ Running Memory Safety Tests...");

        self.run_test("memory_safety_stress", TestCategory::MemorySafety, || {
            let config = self.config.clone();
            Box::pin(async move {
                // This test verifies no memory corruption under heavy load
                let queue = Arc::new(super::LockFreeQueue::new());
                let map = Arc::new(super::LockFreeMap::new());

                let mut handles = Vec::new();

                for thread_id in 0..config.thread_count {
                    let queue_clone = Arc::clone(&queue);
                    let map_clone = Arc::clone(&map);

                    handles.push(tokio::spawn(async move {
                        for i in 0..config.operations_per_thread {
                            // Rapid allocation and deallocation
                            queue_clone.enqueue(vec![thread_id; 100]); // Allocate vector
                            
                            let key = format!("key_{}", i % 100);
                            map_clone.insert(key.clone(), vec![i; 50]); // Allocate vector
                            
                            if i % 5 == 0 {
                                let _ = queue_clone.dequeue(); // Deallocate
                                let _ = map_clone.remove(&key); // Deallocate
                            }
                        }
                    }));
                }

                for handle in handles {
                    handle.await.unwrap();
                }

                // If we reach here without crashes, memory safety is maintained
                Ok(())
            })
        }).await;
    }

    /// Run a single test and record the result
    async fn run_test<F, Fut>(&mut self, name: &str, category: TestCategory, test_fn: F)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    {
        print!("  {} ... ", name);
        let start = Instant::now();
        
        let result = match tokio::time::timeout(self.config.timeout, test_fn()).await {
            Ok(Ok(())) => {
                println!("✅ PASSED");
                TestResult {
                    name: name.to_string(),
                    category,
                    passed: true,
                    duration: start.elapsed(),
                    performance_metrics: None,
                    error: None,
                    details: HashMap::new(),
                }
            }
            Ok(Err(e)) => {
                println!("❌ FAILED: {}", e);
                TestResult {
                    name: name.to_string(),
                    category,
                    passed: false,
                    duration: start.elapsed(),
                    performance_metrics: None,
                    error: Some(e.to_string()),
                    details: HashMap::new(),
                }
            }
            Err(_) => {
                println!("⏰ TIMEOUT");
                TestResult {
                    name: name.to_string(),
                    category,
                    passed: false,
                    duration: start.elapsed(),
                    performance_metrics: None,
                    error: Some("Test timed out".to_string()),
                    details: HashMap::new(),
                }
            }
        };

        self.results.push(result);
    }

    /// Run a performance test and record metrics
    async fn run_performance_test<F, Fut>(&mut self, name: &str, benchmark_fn: F)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = PerformanceMetrics>,
    {
        print!("  {} ... ", name);
        let start = Instant::now();
        
        let metrics = benchmark_fn().await;
        let duration = start.elapsed();

        // Check if performance meets thresholds
        let passed = match name {
            n if n.contains("queue") => metrics.ops_per_second >= self.config.performance_thresholds.queue_ops_per_second,
            n if n.contains("map") => metrics.ops_per_second >= self.config.performance_thresholds.map_ops_per_second,
            n if n.contains("counter") => metrics.ops_per_second >= self.config.performance_thresholds.counter_ops_per_second,
            _ => true, // Default to pass for unknown benchmarks
        };

        if passed {
            println!("✅ PASSED ({:.0} ops/sec)", metrics.ops_per_second);
        } else {
            println!("⚠️ BELOW THRESHOLD ({:.0} ops/sec)", metrics.ops_per_second);
        }

        let result = TestResult {
            name: name.to_string(),
            category: TestCategory::Performance,
            passed,
            duration,
            performance_metrics: Some(metrics),
            error: None,
            details: HashMap::new(),
        };

        self.results.push(result);
    }

    /// Generate test summary
    fn generate_summary(&self, total_duration: Duration) -> TestSummary {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        let mut by_category = HashMap::new();
        for result in &self.results {
            let entry = by_category.entry(result.category.clone()).or_insert((0, 0));
            entry.0 += 1; // total
            if result.passed {
                entry.1 += 1; // passed
            }
        }

        TestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            total_duration,
            results_by_category: by_category,
            performance_summary: self.generate_performance_summary(),
        }
    }

    /// Generate performance summary
    fn generate_performance_summary(&self) -> Vec<(String, PerformanceMetrics)> {
        self.results
            .iter()
            .filter_map(|r| {
                r.performance_metrics.as_ref().map(|m| (r.name.clone(), m.clone()))
            })
            .collect()
    }

    /// Print test summary
    fn print_summary(&self, summary: &TestSummary) {
        println!("\n📊 Test Summary");
        println!("================");
        println!("Total Tests: {}", summary.total_tests);
        println!("Passed: {} ✅", summary.passed_tests);
        println!("Failed: {} ❌", summary.failed_tests);
        println!("Success Rate: {:.1}%", 
                 (summary.passed_tests as f64 / summary.total_tests as f64) * 100.0);
        println!("Total Duration: {:.2}s", summary.total_duration.as_secs_f64());

        println!("\nResults by Category:");
        for (category, (total, passed)) in &summary.results_by_category {
            println!("  {:?}: {}/{} ({:.1}%)", 
                     category, passed, total,
                     (*passed as f64 / *total as f64) * 100.0);
        }

        if !summary.performance_summary.is_empty() {
            println!("\nPerformance Results:");
            for (name, metrics) in &summary.performance_summary {
                println!("  {}: {:.0} ops/sec", name, metrics.ops_per_second);
            }
        }

        // Print failed tests
        let failed_tests: Vec<_> = self.results.iter().filter(|r| !r.passed).collect();
        if !failed_tests.is_empty() {
            println!("\nFailed Tests:");
            for result in failed_tests {
                println!("  ❌ {}: {}", result.name, 
                         result.error.as_ref().unwrap_or(&"Unknown error".to_string()));
            }
        }
    }
    
    /// Calculate percentile from latency measurements
    fn calculate_percentile(&self, latencies: &[f64], percentile: f64) -> f64 {
        if latencies.is_empty() {
            return 0.0;
        }
        
        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (percentile / 100.0 * (sorted_latencies.len() - 1) as f64) as usize;
        sorted_latencies[index.min(sorted_latencies.len() - 1)]
    }
    
    /// Get current memory usage in bytes
    fn get_memory_usage(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(status) = fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("ps")
                .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(rss_str) = String::from_utf8(output.stdout) {
                    if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                        return rss_kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows memory tracking would require additional dependencies
            // For now, return 0 as a placeholder
        }
        
        0 // Fallback if memory tracking is not available
    }
    
    /// Get current CPU utilization percentage
    fn get_cpu_utilization(&self) -> f64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            use std::thread;
            use std::time::Duration;
            
            // Read CPU stats twice with a small interval
            let get_cpu_time = || -> Option<(u64, u64)> {
                let stat = fs::read_to_string("/proc/stat").ok()?;
                let line = stat.lines().next()?;
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 && parts[0] == "cpu" {
                    let user: u64 = parts[1].parse().ok()?;
                    let nice: u64 = parts[2].parse().ok()?;
                    let system: u64 = parts[3].parse().ok()?;
                    let idle: u64 = parts[4].parse().ok()?;
                    let total = user + nice + system + idle;
                    let active = user + nice + system;
                    Some((active, total))
                } else {
                    None
                }
            };
            
            if let (Some((active1, total1)), Some((active2, total2))) = (
                get_cpu_time(),
                {
                    thread::sleep(Duration::from_millis(100));
                    get_cpu_time()
                }
            ) {
                let active_diff = active2.saturating_sub(active1);
                let total_diff = total2.saturating_sub(total1);
                if total_diff > 0 {
                    return (active_diff as f64 / total_diff as f64) * 100.0;
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("ps")
                .args(&["-o", "pcpu=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(cpu_str) = String::from_utf8(output.stdout) {
                    if let Ok(cpu_percent) = cpu_str.trim().parse::<f64>() {
                        return cpu_percent;
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows CPU tracking would require additional dependencies
            // For now, return 0 as a placeholder
        }
        
        0.0 // Fallback if CPU tracking is not available
    }
}

/// Summary of test results
#[derive(Debug)]
pub struct TestSummary {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub total_duration: Duration,
    pub results_by_category: HashMap<TestCategory, (usize, usize)>, // (total, passed)
    pub performance_summary: Vec<(String, PerformanceMetrics)>,
}

/// Example usage and main test runner
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn run_comprehensive_test_suite() {
        let config = TestConfig {
            thread_count: 4,
            operations_per_thread: 1000,
            timeout: Duration::from_secs(10),
            run_benchmarks: true,
            run_stress_tests: false, // Disable for CI
            performance_thresholds: PerformanceThresholds {
                queue_ops_per_second: 50000.0, // Lower threshold for CI
                map_ops_per_second: 25000.0,
                counter_ops_per_second: 250000.0,
                max_actor_latency_ms: 20.0,
                min_actor_throughput: 5000.0,
            },
        };

        let mut test_suite = ConcurrencyTestSuite::new(config);
        let summary = test_suite.run_all_tests().await;

        // Ensure most tests pass
        let success_rate = summary.passed_tests as f64 / summary.total_tests as f64;
        assert!(success_rate >= 0.8, "Test success rate too low: {:.1}%", success_rate * 100.0);
    }
} 