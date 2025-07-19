//! Parallel compilation support
//!
//! This module implements fine-grained parallelism for compilation tasks using a
//! work-stealing scheduler with dependency-aware task execution.

use crate::error::{CompilerError, CompilerResult};
use crate::query::{QueryId, QueryEngine};
use async_trait::async_trait;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, info, span, Level};

/// Maximum number of concurrent compilation tasks
const MAX_CONCURRENT_TASKS: usize = num_cpus::get() * 2;

/// Task priority levels for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Critical path tasks (blocking other work)
    Critical = 0,
    /// High priority tasks (user-facing features)
    High = 1,
    /// Normal priority tasks (background compilation)
    Normal = 2,
    /// Low priority tasks (optimization, cleanup)
    Low = 3,
}

/// A compilation task that can be executed in parallel
#[async_trait]
pub trait ParallelTask: Send + Sync {
    /// Unique identifier for this task
    fn id(&self) -> TaskId;
    
    /// Priority level for scheduling
    fn priority(&self) -> TaskPriority;
    
    /// Task dependencies that must complete first
    fn dependencies(&self) -> Vec<TaskId>;
    
    /// Estimated execution time in milliseconds
    fn estimated_duration(&self) -> u64;
    
    /// Execute the task
    async fn execute(&self, context: &ParallelContext) -> CompilerResult<TaskResult>;
    
    /// Check if task can be executed (all dependencies satisfied)
    fn can_execute(&self, completed: &std::collections::HashSet<TaskId>) -> bool {
        self.dependencies().iter().all(|dep| completed.contains(dep))
    }
}

/// Unique identifier for parallel tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

/// Result of task execution
#[derive(Debug)]
pub struct TaskResult {
    /// Task that produced this result
    pub task_id: TaskId,
    /// Execution time in milliseconds
    pub duration: u64,
    /// Task-specific result data
    pub data: TaskData,
}

/// Task-specific result data
#[derive(Debug)]
pub enum TaskData {
    /// Query execution result
    Query(QueryId),
    /// File compilation result
    FileCompilation(std::path::PathBuf),
    /// Module analysis result
    ModuleAnalysis(String),
    /// Code generation result
    CodeGeneration(Vec<u8>),
}

/// Parallel execution context
#[derive(Debug)]
pub struct ParallelContext {
    /// Query engine for executing compilation queries
    pub query_engine: Arc<QueryEngine>,
    /// Task execution semaphore
    pub semaphore: Arc<Semaphore>,
    /// Profiling data
    pub profiling: Arc<RwLock<ProfilingData>>,
}

/// Profiling data for parallel execution
#[derive(Debug, Default)]
pub struct ProfilingData {
    /// Total tasks executed
    pub tasks_executed: u64,
    /// Total execution time
    pub total_duration: u64,
    /// Task execution times by priority
    pub duration_by_priority: HashMap<TaskPriority, u64>,
    /// Task queue wait times
    pub queue_wait_times: Vec<u64>,
}

/// Work-stealing parallel scheduler
#[derive(Debug)]
pub struct ParallelScheduler {
    /// Task queues by priority
    task_queues: Arc<RwLock<HashMap<TaskPriority, VecDeque<Box<dyn ParallelTask>>>>>,
    /// Completed task IDs
    completed_tasks: Arc<RwLock<std::collections::HashSet<TaskId>>>,
    /// Task results
    task_results: Arc<RwLock<HashMap<TaskId, TaskResult>>>,
    /// Execution context
    context: Arc<ParallelContext>,
    /// Worker threads
    workers: Vec<tokio::task::JoinHandle<()>>,
}

impl ParallelScheduler {
    /// Create a new parallel scheduler
    pub fn new(query_engine: Arc<QueryEngine>) -> Self {
        let context = Arc::new(ParallelContext {
            query_engine,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_TASKS)),
            profiling: Arc::new(RwLock::new(ProfilingData::default())),
        });

        Self {
            task_queues: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(std::collections::HashSet::new())),
            task_results: Arc::new(RwLock::new(HashMap::new())),
            context,
            workers: Vec::new(),
        }
    }

    /// Submit a task for parallel execution
    pub async fn submit_task(&self, task: Box<dyn ParallelTask>) -> CompilerResult<()> {
        let priority = task.priority();
        let mut queues = self.task_queues.write().await;
        
        queues
            .entry(priority)
            .or_insert_with(VecDeque::new)
            .push_back(task);

        debug!("Task submitted with priority {:?}", priority);
        Ok(())
    }

    /// Execute all submitted tasks in parallel
    pub async fn execute_all(&mut self) -> CompilerResult<HashMap<TaskId, TaskResult>> {
        let _span = span!(Level::INFO, "parallel_execution").entered();
        info!("Starting parallel task execution");

        // Start worker threads
        self.start_workers().await;

        // Wait for all tasks to complete
        self.wait_for_completion().await?;

        // Collect results
        let results = self.task_results.read().await.clone();
        
        info!("Parallel execution completed. {} tasks executed", results.len());
        Ok(results)
    }

    /// Start worker threads for task execution
    async fn start_workers(&mut self) {
        let num_workers = num_cpus::get();
        
        for worker_id in 0..num_workers {
            let task_queues = Arc::clone(&self.task_queues);
            let completed_tasks = Arc::clone(&self.completed_tasks);
            let task_results = Arc::clone(&self.task_results);
            let context = Arc::clone(&self.context);

            let worker = tokio::spawn(async move {
                Self::worker_loop(worker_id, task_queues, completed_tasks, task_results, context).await;
            });

            self.workers.push(worker);
        }
    }

    /// Worker thread main loop
    async fn worker_loop(
        worker_id: usize,
        task_queues: Arc<RwLock<HashMap<TaskPriority, VecDeque<Box<dyn ParallelTask>>>>>,
        completed_tasks: Arc<RwLock<std::collections::HashSet<TaskId>>>,
        task_results: Arc<RwLock<HashMap<TaskId, TaskResult>>>,
        context: Arc<ParallelContext>,
    ) {
        debug!("Worker {} started", worker_id);

        loop {
            // Try to steal a task from the highest priority queue
            let task = Self::steal_task(&task_queues, &completed_tasks).await;
            
            match task {
                Some(task) => {
                    // Execute the task
                    if let Ok(result) = Self::execute_task(task, &context).await {
                        // Store result
                        let mut results = task_results.write().await;
                        let mut completed = completed_tasks.write().await;
                        
                        completed.insert(result.task_id);
                        results.insert(result.task_id, result);
                    }
                }
                None => {
                    // No tasks available, check if we should continue
                    if Self::should_continue(&task_queues).await {
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                        continue;
                    } else {
                        break;
                    }
                }
            }
        }

        debug!("Worker {} finished", worker_id);
    }

    /// Steal a task from the highest priority queue
    async fn steal_task(
        task_queues: &Arc<RwLock<HashMap<TaskPriority, VecDeque<Box<dyn ParallelTask>>>>>,
        completed_tasks: &Arc<RwLock<std::collections::HashSet<TaskId>>>,
    ) -> Option<Box<dyn ParallelTask>> {
        let mut queues = task_queues.write().await;
        let completed = completed_tasks.read().await;

        // Try each priority level from highest to lowest
        for priority in [TaskPriority::Critical, TaskPriority::High, TaskPriority::Normal, TaskPriority::Low] {
            if let Some(queue) = queues.get_mut(&priority) {
                // Find a task that can be executed
                for _ in 0..queue.len() {
                    if let Some(task) = queue.pop_front() {
                        if task.can_execute(&completed) {
                            return Some(task);
                        } else {
                            // Put it back at the end
                            queue.push_back(task);
                        }
                    }
                }
            }
        }

        None
    }

    /// Execute a single task
    async fn execute_task(
        task: Box<dyn ParallelTask>,
        context: &Arc<ParallelContext>,
    ) -> CompilerResult<TaskResult> {
        let _permit = context.semaphore.acquire().await.map_err(|_| {
            CompilerError::InternalError("Failed to acquire semaphore".to_string())
        })?;

        let start_time = std::time::Instant::now();
        let task_id = task.id();
        let priority = task.priority();

        debug!("Executing task {:?} with priority {:?}", task_id, priority);

        let result = task.execute(context).await?;
        let duration = start_time.elapsed().as_millis() as u64;

        // Update profiling data
        {
            let mut profiling = context.profiling.write().await;
            profiling.tasks_executed += 1;
            profiling.total_duration += duration;
            *profiling.duration_by_priority.entry(priority).or_insert(0) += duration;
        }

        Ok(TaskResult {
            task_id,
            duration,
            data: result.data,
        })
    }

    /// Check if workers should continue looking for tasks
    async fn should_continue(
        task_queues: &Arc<RwLock<HashMap<TaskPriority, VecDeque<Box<dyn ParallelTask>>>>>,
    ) -> bool {
        let queues = task_queues.read().await;
        queues.values().any(|queue| !queue.is_empty())
    }

    /// Wait for all tasks to complete
    async fn wait_for_completion(&mut self) -> CompilerResult<()> {
        // Wait for all worker threads to finish
        for worker in self.workers.drain(..) {
            worker.await.map_err(|e| {
                CompilerError::InternalError(format!("Worker thread failed: {}", e))
            })?;
        }

        Ok(())
    }

    /// Get profiling statistics
    pub async fn get_profiling_stats(&self) -> ProfilingData {
        self.context.profiling.read().await.clone()
    }
}

/// Utility functions for parallel compilation
pub struct ParallelUtils;

impl ParallelUtils {
    /// Split a large compilation task into smaller parallel tasks
    pub fn split_compilation_task(
        files: Vec<std::path::PathBuf>,
        chunk_size: usize,
    ) -> Vec<Vec<std::path::PathBuf>> {
        files
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Calculate optimal chunk size based on file count and worker count
    pub fn calculate_chunk_size(file_count: usize, worker_count: usize) -> usize {
        std::cmp::max(1, file_count / (worker_count * 4))
    }

    /// Create a dependency graph for task scheduling
    pub fn build_dependency_graph(
        tasks: &[Box<dyn ParallelTask>],
    ) -> HashMap<TaskId, Vec<TaskId>> {
        let mut graph = HashMap::new();
        
        for task in tasks {
            graph.insert(task.id(), task.dependencies());
        }
        
        graph
    }

    /// Validate that task dependencies form a DAG (no cycles)
    pub fn validate_dependencies(
        tasks: &[Box<dyn ParallelTask>],
    ) -> CompilerResult<()> {
        let graph = Self::build_dependency_graph(tasks);
        
        // Simple cycle detection using DFS
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();
        
        for task in tasks {
            if Self::has_cycle(&graph, task.id(), &mut visited, &mut rec_stack) {
                return Err(CompilerError::InternalError(
                    "Cyclic dependency detected in parallel tasks".to_string()
                ));
            }
        }
        
        Ok(())
    }

    /// Helper function for cycle detection
    fn has_cycle(
        graph: &HashMap<TaskId, Vec<TaskId>>,
        task_id: TaskId,
        visited: &mut std::collections::HashSet<TaskId>,
        rec_stack: &mut std::collections::HashSet<TaskId>,
    ) -> bool {
        if rec_stack.contains(&task_id) {
            return true;
        }
        
        if visited.contains(&task_id) {
            return false;
        }
        
        visited.insert(task_id);
        rec_stack.insert(task_id);
        
        if let Some(deps) = graph.get(&task_id) {
            for &dep in deps {
                if Self::has_cycle(graph, dep, visited, rec_stack) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(&task_id);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::QueryEngine;

    struct MockTask {
        id: TaskId,
        priority: TaskPriority,
        dependencies: Vec<TaskId>,
        duration: u64,
    }

    #[async_trait]
    impl ParallelTask for MockTask {
        fn id(&self) -> TaskId {
            self.id
        }

        fn priority(&self) -> TaskPriority {
            self.priority
        }

        fn dependencies(&self) -> Vec<TaskId> {
            self.dependencies.clone()
        }

        fn estimated_duration(&self) -> u64 {
            self.duration
        }

        async fn execute(&self, _context: &ParallelContext) -> CompilerResult<TaskResult> {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.duration)).await;
            
            Ok(TaskResult {
                task_id: self.id,
                duration: self.duration,
                data: TaskData::ModuleAnalysis("test".to_string()),
            })
        }
    }

    #[tokio::test]
    async fn test_parallel_scheduler() {
        let query_engine = Arc::new(QueryEngine::new());
        let mut scheduler = ParallelScheduler::new(query_engine);

        // Submit some test tasks
        let task1 = Box::new(MockTask {
            id: TaskId(1),
            priority: TaskPriority::High,
            dependencies: vec![],
            duration: 10,
        });

        let task2 = Box::new(MockTask {
            id: TaskId(2),
            priority: TaskPriority::Normal,
            dependencies: vec![TaskId(1)],
            duration: 20,
        });

        scheduler.submit_task(task1).await.unwrap();
        scheduler.submit_task(task2).await.unwrap();

        let results = scheduler.execute_all().await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_dependency_validation() {
        let tasks: Vec<Box<dyn ParallelTask>> = vec![
            Box::new(MockTask {
                id: TaskId(1),
                priority: TaskPriority::High,
                dependencies: vec![],
                duration: 10,
            }),
            Box::new(MockTask {
                id: TaskId(2),
                priority: TaskPriority::Normal,
                dependencies: vec![TaskId(1)],
                duration: 20,
            }),
        ];

        assert!(ParallelUtils::validate_dependencies(&tasks).is_ok());
    }
} 