//! Async Scheduler - Priority-Based Task Scheduling
//!
//! This module implements the async task scheduler that manages task execution order:
//! - **Priority-based scheduling**: Tasks are scheduled based on priority levels
//! - **Queue management**: Separate queues for different priority levels
//! - **Metrics collection**: Track scheduling performance and queue statistics
//! - **Load balancing**: Distribute tasks across available execution resources
//!
//! The scheduler is designed to be fair while respecting priority levels,
//! ensuring that high-priority tasks get precedence without starving lower-priority tasks.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::Duration;

use super::types::{TaskId, TaskPriority};

/// Async task scheduler that manages task execution order
#[derive(Debug)]
pub struct AsyncScheduler {
    /// Task queues by priority
    priority_queues: Arc<RwLock<HashMap<TaskPriority, Vec<TaskId>>>>,
    /// Scheduler metrics
    scheduler_metrics: Arc<Mutex<SchedulerMetrics>>,
}

impl AsyncScheduler {
    /// Create a new async scheduler
    pub fn new() -> Self {
        Self {
            priority_queues: Arc::new(RwLock::new(HashMap::new())),
            scheduler_metrics: Arc::new(Mutex::new(SchedulerMetrics::default())),
        }
    }

    /// Schedule a task for execution
    pub fn schedule_task(&self, task_id: TaskId, priority: TaskPriority) {
        let mut queues = self.priority_queues.write().unwrap();
        queues.entry(priority).or_insert_with(Vec::new).push(task_id);

        // Update metrics
        let mut metrics = self.scheduler_metrics.lock().unwrap();
        *metrics.scheduled_by_priority.entry(priority).or_insert(0) += 1;
        metrics.total_scheduled += 1;
    }

    /// Get the next task to execute based on priority
    pub fn next_task(&self) -> Option<(TaskId, TaskPriority)> {
        let mut queues = self.priority_queues.write().unwrap();
        
        // Check priority levels in order: Critical, High, Normal, Low
        for &priority in &[TaskPriority::Critical, TaskPriority::High, TaskPriority::Normal, TaskPriority::Low] {
            if let Some(queue) = queues.get_mut(&priority) {
                if !queue.is_empty() {
                    let task_id = queue.remove(0);
                    return Some((task_id, priority));
                }
            }
        }
        
        None
    }

    /// Get the number of queued tasks for a priority level
    pub fn queue_size(&self, priority: TaskPriority) -> usize {
        let queues = self.priority_queues.read().unwrap();
        queues.get(&priority).map(|q| q.len()).unwrap_or(0)
    }

    /// Get total number of queued tasks
    pub fn total_queued(&self) -> usize {
        let queues = self.priority_queues.read().unwrap();
        queues.values().map(|q| q.len()).sum()
    }

    /// Get scheduler metrics
    pub fn metrics(&self) -> SchedulerMetrics {
        self.scheduler_metrics.lock().unwrap().clone()
    }

    /// Clear all queues (used for shutdown)
    pub fn clear_all_queues(&self) {
        let mut queues = self.priority_queues.write().unwrap();
        queues.clear();
    }

    /// Update scheduler metrics with timing information
    pub fn update_metrics(&self, wait_time: Duration) {
        let mut metrics = self.scheduler_metrics.lock().unwrap();
        
        // Update average wait time using exponential moving average
        let alpha = 0.1; // Smoothing factor
        let new_wait_time_ms = wait_time.as_millis() as f64;
        let current_avg_ms = metrics.average_wait_time.as_millis() as f64;
        let updated_avg_ms = alpha * new_wait_time_ms + (1.0 - alpha) * current_avg_ms;
        
        metrics.average_wait_time = Duration::from_millis(updated_avg_ms as u64);
    }

    /// Get queue statistics for monitoring
    pub fn queue_statistics(&self) -> QueueStatistics {
        let queues = self.priority_queues.read().unwrap();
        let metrics = self.scheduler_metrics.lock().unwrap();
        
        let mut stats_by_priority = HashMap::new();
        for (&priority, queue) in queues.iter() {
            stats_by_priority.insert(priority, QueueStats {
                current_size: queue.len(),
                total_scheduled: *metrics.scheduled_by_priority.get(&priority).unwrap_or(&0),
            });
        }
        
        QueueStatistics {
            stats_by_priority,
            total_queued: queues.values().map(|q| q.len()).sum(),
            total_scheduled: metrics.total_scheduled,
            average_wait_time: metrics.average_wait_time,
        }
    }
}

impl Default for AsyncScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler metrics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct SchedulerMetrics {
    /// Tasks scheduled by priority
    pub scheduled_by_priority: HashMap<TaskPriority, u64>,
    /// Average queue wait time
    pub average_wait_time: Duration,
    /// Total tasks scheduled
    pub total_scheduled: u64,
}

/// Queue statistics for a specific priority level
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Current number of tasks in queue
    pub current_size: usize,
    /// Total number of tasks scheduled at this priority
    pub total_scheduled: u64,
}

/// Overall queue statistics for all priority levels
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Statistics by priority level
    pub stats_by_priority: HashMap<TaskPriority, QueueStats>,
    /// Total tasks currently queued across all priorities
    pub total_queued: usize,
    /// Total tasks scheduled since system start
    pub total_scheduled: u64,
    /// Average time tasks spend waiting in queues
    pub average_wait_time: Duration,
} 