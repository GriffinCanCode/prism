//! Async Support Module
//!
//! This module generates async/await patterns for effects and capabilities,
//! providing comprehensive support for modern Python async programming.

use super::{PythonResult, PythonError};
use crate::backends::{PIRFunction, Effect, Capability, EffectSignature};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Async configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncConfig {
    /// Enable async patterns
    pub enable_async: bool,
    /// Use asyncio
    pub use_asyncio: bool,
    /// Generate async context managers
    pub async_context_managers: bool,
    /// Generate async generators
    pub async_generators: bool,
    /// Generate async comprehensions
    pub async_comprehensions: bool,
    /// Use structured concurrency patterns
    pub structured_concurrency: bool,
    /// Generate timeout handling
    pub timeout_handling: bool,
    /// Generate cancellation support
    pub cancellation_support: bool,
    /// Generate async resource management
    pub async_resource_management: bool,
    /// Use task groups (Python 3.11+)
    pub task_groups: bool,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            enable_async: true,
            use_asyncio: true,
            async_context_managers: true,
            async_generators: true,
            async_comprehensions: true,
            structured_concurrency: true,
            timeout_handling: true,
            cancellation_support: true,
            async_resource_management: true,
            task_groups: true,
        }
    }
}

/// Async pattern generator with comprehensive async support
pub struct AsyncPatternGenerator {
    config: AsyncConfig,
    generated_patterns: HashMap<String, String>,
}

impl AsyncPatternGenerator {
    pub fn new(config: AsyncConfig) -> Self {
        Self {
            config,
            generated_patterns: HashMap::new(),
        }
    }

    /// Generate async function with effect handling
    pub fn generate_async_function(&mut self, function: &PIRFunction) -> PythonResult<String> {
        if !self.config.enable_async {
            return Ok(String::new());
        }

        let mut output = String::new();
        
        // Generate function signature
        output.push_str(&self.generate_async_signature(function)?);
        
        // Generate function body with effect handling
        output.push_str(&self.generate_async_body(function)?);
        
        // Cache the generated pattern
        self.generated_patterns.insert(function.name.clone(), output.clone());
        
        Ok(output)
    }

    /// Generate async function signature
    fn generate_async_signature(&self, function: &PIRFunction) -> PythonResult<String> {
        let mut output = String::new();
        
        // Function documentation
        output.push_str(&format!(
            r#"async def {}("#,
            function.name
        ));
        
        // Parameters
        let params: Vec<String> = function.signature.parameters.iter()
            .map(|p| format!("{}: {}", p.name, "Any")) // Simplified type for now
            .collect();
        
        if !params.is_empty() {
            output.push_str(&params.join(", "));
        }
        
        // Return type
        output.push_str(") -> Awaitable[Any]:\n");
        
        // Docstring with async information
        output.push_str(&format!(
            r#"    """
    Async function: {}
    
    This function handles effects asynchronously and supports:
    - Effect tracking and validation
    - Capability management
    - Timeout handling
    - Cancellation support
    
    Effects: [{}]
    Capabilities: [{}]
    
    Args:
{}
        
    Returns:
        Awaitable result with effect tracking
        
    Raises:
        EffectError: If effect handling fails
        CapabilityError: If capabilities are insufficient
        asyncio.TimeoutError: If operation times out
        asyncio.CancelledError: If operation is cancelled
    """
"#,
            function.name,
            function.signature.effects.effects.iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            function.capabilities_required.iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            function.signature.parameters.iter()
                .map(|p| format!("        {}: Parameter description", p.name))
                .collect::<Vec<_>>()
                .join("\n")
        ));
        
        Ok(output)
    }

    /// Generate async function body
    fn generate_async_body(&self, function: &PIRFunction) -> PythonResult<String> {
        let mut output = String::new();
        
        // Generate effect tracking setup
        if !function.signature.effects.effects.is_empty() {
            output.push_str(&self.generate_effect_tracking_setup(&function.signature.effects)?);
        }
        
        // Generate capability validation
        if !function.capabilities_required.is_empty() {
            output.push_str(&self.generate_async_capability_validation(&function.capabilities_required)?);
        }
        
        // Generate timeout handling
        if self.config.timeout_handling {
            output.push_str(&self.generate_timeout_wrapper(function)?);
        } else {
            output.push_str(&self.generate_basic_async_body(function)?);
        }
        
        Ok(output)
    }

    /// Generate effect tracking setup
    fn generate_effect_tracking_setup(&self, effects: &EffectSignature) -> PythonResult<String> {
        Ok(format!(
            r#"    # Effect tracking setup
    effect_tracker = EffectTracker()
    effect_names = [{}]
    
    try:
        # Initialize effect tracking
        tracking_context = await effect_tracker.track_effects(effect_names)
        
"#,
            effects.effects.iter()
                .map(|e| format!("'{}'", e.name))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    /// Generate async capability validation
    fn generate_async_capability_validation(&self, capabilities: &[Capability]) -> PythonResult<String> {
        Ok(format!(
            r#"        # Async capability validation
        capability_manager = CapabilityManager()
        required_capabilities = [{}]
        
        await capability_manager.validate_capabilities(required_capabilities)
        
"#,
            capabilities.iter()
                .map(|c| format!("'{}'", c.name))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    /// Generate timeout wrapper
    fn generate_timeout_wrapper(&self, function: &PIRFunction) -> PythonResult<String> {
        Ok(format!(
            r#"        # Timeout handling with structured concurrency
        async with asyncio.timeout(30.0):  # Default 30 second timeout
            try:
                # Function implementation with cancellation support
                result = await self._execute_with_cancellation()
                
                # Complete effect tracking
                if 'tracking_context' in locals():
                    await effect_tracker.complete_tracking(tracking_context.context_id())
                
                return result
                
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                logger.info("Function {} was cancelled", '{}')
                if 'tracking_context' in locals():
                    await effect_tracker.abort_tracking(tracking_context.context_id())
                raise
                
            except Exception as e:
                # Handle other exceptions
                logger.error("Function {} failed: {{e}}", '{}')
                if 'tracking_context' in locals():
                    await effect_tracker.abort_tracking(tracking_context.context_id())
                raise
                
    except asyncio.TimeoutError:
        logger.warning("Function {} timed out", '{}')
        if 'tracking_context' in locals():
            await effect_tracker.abort_tracking(tracking_context.context_id())
        raise
"#,
            function.name,
            function.name,
            function.name
        ))
    }

    /// Generate basic async body
    fn generate_basic_async_body(&self, function: &PIRFunction) -> PythonResult<String> {
        Ok(format!(
            r#"        try:
            # Function implementation would be generated here
            result = None  # Placeholder
            
            # Complete effect tracking
            if 'tracking_context' in locals():
                await effect_tracker.complete_tracking(tracking_context.context_id())
            
            return result
            
        except Exception as e:
            # Handle exceptions and cleanup
            logger.error("Function {} failed: {{e}}", '{}')
            if 'tracking_context' in locals():
                await effect_tracker.abort_tracking(tracking_context.context_id())
            raise
    
    async def _execute_with_cancellation(self):
        """Execute function logic with cancellation support."""
        # Check for cancellation periodically
        await asyncio.sleep(0)  # Yield control
        
        # Function implementation would be generated here
        # This is where the actual business logic would go
        return None  # Placeholder
"#,
            function.name
        ))
    }

    /// Generate async context manager
    pub fn generate_async_context_manager(&self, name: &str, effects: &[Effect]) -> PythonResult<String> {
        if !self.config.async_context_managers {
            return Ok(String::new());
        }

        Ok(format!(
            r#"
class {}AsyncContext:
    """
    Async context manager for {} with effect handling.
    
    Provides structured resource management and effect tracking
    using modern Python async context manager patterns.
    """
    
    def __init__(self, effects: List[str] = None):
        self.effects = effects or [{}]
        self.effect_tracker = None
        self.tracking_context = None
        self.start_time = None
    
    async def __aenter__(self) -> '{}AsyncContext':
        """Async context entry with effect setup."""
        self.start_time = asyncio.get_event_loop().time()
        
        # Initialize effect tracking
        self.effect_tracker = EffectTracker()
        self.tracking_context = await self.effect_tracker.track_effects(self.effects)
        
        logger.debug("Entered async context for {}: effects={{}}", self.effects)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context exit with cleanup."""
        try:
            if exc_type is None:
                # Normal completion
                if self.tracking_context:
                    await self.effect_tracker.complete_tracking(self.tracking_context.context_id())
                logger.debug("Completed async context for {}")
            else:
                # Exception occurred
                if self.tracking_context:
                    await self.effect_tracker.abort_tracking(self.tracking_context.context_id())
                logger.error("Async context for {} failed: {{exc_val}}", exc_val)
        finally:
            # Record execution time
            if self.start_time:
                duration = asyncio.get_event_loop().time() - self.start_time
                logger.debug("Async context for {} duration: {{duration:.3f}}s", duration)
        
        # Don't suppress exceptions
        return False
    
    async def wait_for_completion(self, timeout: float = None):
        """Wait for all tracked effects to complete."""
        if self.tracking_context:
            # Implementation would wait for effect completion
            await asyncio.sleep(0)  # Placeholder

# Convenience function to create async context
async def {}_context(effects: List[str] = None):
    """Create async context manager for {}."""
    return {}AsyncContext(effects)
"#,
            name,
            name,
            effects.iter()
                .map(|e| format!("'{}'", e.name))
                .collect::<Vec<_>>()
                .join(", "),
            name,
            name,
            name,
            name,
            name.to_lowercase(),
            name,
            name
        ))
    }

    /// Generate async generator
    pub fn generate_async_generator(&self, name: &str, item_type: &str) -> PythonResult<String> {
        if !self.config.async_generators {
            return Ok(String::new());
        }

        Ok(format!(
            r#"
async def {}_async_generator(source: AsyncIterable[{}]) -> AsyncGenerator[{}, None]:
    """
    Async generator for {} with effect tracking.
    
    Provides async iteration with proper resource management
    and effect handling for each yielded item.
    
    Args:
        source: Async iterable source
        
    Yields:
        Processed {} items
        
    Raises:
        EffectError: If effect handling fails during iteration
    """
    effect_tracker = EffectTracker()
    
    try:
        async with effect_tracker.track_effects(['{}']) as tracking:
            async for item in source:
                # Check for cancellation
                await asyncio.sleep(0)
                
                # Process item with effect tracking
                processed_item = await process_{}_item(item)
                
                # Yield processed item
                yield processed_item
                
    except asyncio.CancelledError:
        logger.info("Async generator for {} was cancelled")
        raise
    except Exception as e:
        logger.error("Async generator for {} failed: {{e}}", e)
        raise

async def process_{}_item(item: {}) -> {}:
    """Process individual {} item asynchronously."""
    # Item processing logic would be generated here
    await asyncio.sleep(0)  # Yield control
    return item  # Placeholder
"#,
            name.to_lowercase(),
            item_type,
            item_type,
            name,
            item_type,
            name.to_lowercase(),
            name.to_lowercase(),
            name,
            name,
            name.to_lowercase(),
            item_type,
            item_type,
            item_type
        ))
    }

    /// Generate async comprehension patterns
    pub fn generate_async_comprehensions(&self) -> PythonResult<String> {
        if !self.config.async_comprehensions {
            return Ok(String::new());
        }

        Ok(r#"
# === ASYNC COMPREHENSION UTILITIES ===
# Modern async comprehension patterns for effect handling

async def async_map(func: Callable[[Any], Awaitable[Any]], iterable: AsyncIterable[Any]) -> List[Any]:
    """Async map function with effect tracking."""
    results = []
    async for item in iterable:
        result = await func(item)
        results.append(result)
    return results

async def async_filter(predicate: Callable[[Any], Awaitable[bool]], iterable: AsyncIterable[Any]) -> List[Any]:
    """Async filter function with effect tracking."""
    results = []
    async for item in iterable:
        if await predicate(item):
            results.append(item)
    return results

async def async_reduce(func: Callable[[Any, Any], Awaitable[Any]], iterable: AsyncIterable[Any], initial=None) -> Any:
    """Async reduce function with effect tracking."""
    accumulator = initial
    first = True
    
    async for item in iterable:
        if first and initial is None:
            accumulator = item
            first = False
        else:
            accumulator = await func(accumulator, item)
            first = False
    
    return accumulator

# Async comprehension decorators
def async_comprehension(func):
    """Decorator to enable async comprehension patterns."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Add effect tracking to comprehensions
        effect_tracker = EffectTracker()
        async with effect_tracker.track_effects(['comprehension']):
            return await func(*args, **kwargs)
    return wrapper
"#.to_string())
    }

    /// Generate structured concurrency patterns
    pub fn generate_structured_concurrency(&self) -> PythonResult<String> {
        if !self.config.structured_concurrency {
            return Ok(String::new());
        }

        let task_group_code = if self.config.task_groups {
            r#"
# Task groups for structured concurrency (Python 3.11+)
async def run_with_task_group(tasks: List[Callable[[], Awaitable[Any]]]) -> List[Any]:
    """Run tasks concurrently using task groups."""
    try:
        import asyncio.taskgroups as tg
        
        async with tg.TaskGroup() as group:
            task_handles = [group.create_task(task()) for task in tasks]
        
        return [task.result() for task in task_handles]
    except ImportError:
        # Fallback for older Python versions
        return await asyncio.gather(*[task() for task in tasks])
"#
        } else {
            ""
        };

        Ok(format!(
            r#"
# === STRUCTURED CONCURRENCY PATTERNS ===
# Modern structured concurrency for effect handling

class ConcurrencyManager:
    """
    Manager for structured concurrency with effect tracking.
    
    Provides safe concurrent execution with proper resource cleanup
    and effect coordination across multiple async operations.
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()
        self.effect_tracker = EffectTracker()
    
    async def run_concurrent(self, tasks: List[Callable[[], Awaitable[Any]]]) -> List[Any]:
        """Run tasks concurrently with effect tracking."""
        async with self.effect_tracker.track_effects(['concurrent_execution']):
            # Use semaphore to limit concurrency
            async def bounded_task(task):
                async with self.semaphore:
                    return await task()
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*[
                bounded_task(task) for task in tasks
            ], return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {{i}} failed: {{result}}")
                    raise result
                processed_results.append(result)
            
            return processed_results
    
    async def run_with_timeout(self, tasks: List[Callable[[], Awaitable[Any]]], timeout: float) -> List[Any]:
        """Run tasks with global timeout."""
        async with asyncio.timeout(timeout):
            return await self.run_concurrent(tasks)
    
    async def cleanup(self):
        """Cleanup all active tasks."""
        if self.active_tasks:
            # Cancel all active tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation to complete
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.active_tasks.clear()

# Global concurrency manager instance
concurrency_manager = ConcurrencyManager()

{}

# Async context manager for concurrent operations
@asynccontextmanager
async def concurrent_context(max_concurrent: int = 10):
    """Async context manager for concurrent operations."""
    manager = ConcurrencyManager(max_concurrent)
    try:
        yield manager
    finally:
        await manager.cleanup()
"#,
            task_group_code
        ))
    }

    /// Generate async resource management
    pub fn generate_async_resource_management(&self) -> PythonResult<String> {
        if !self.config.async_resource_management {
            return Ok(String::new());
        }

        Ok(r#"
# === ASYNC RESOURCE MANAGEMENT ===
# Comprehensive async resource management with effect tracking

class AsyncResourceManager:
    """
    Async resource manager with effect tracking and cleanup.
    
    Manages async resources with proper lifecycle management,
    effect coordination, and graceful cleanup on errors.
    """
    
    def __init__(self):
        self.resources = []
        self.cleanup_tasks = []
        self.effect_tracker = EffectTracker()
    
    async def acquire_resource(self, resource_factory: Callable[[], Awaitable[Any]], cleanup_func: Callable[[Any], Awaitable[None]] = None):
        """Acquire an async resource with cleanup registration."""
        async with self.effect_tracker.track_effects(['resource_acquisition']):
            resource = await resource_factory()
            self.resources.append(resource)
            
            if cleanup_func:
                self.cleanup_tasks.append(lambda: cleanup_func(resource))
            
            return resource
    
    async def release_all(self):
        """Release all managed resources."""
        async with self.effect_tracker.track_effects(['resource_cleanup']):
            # Execute cleanup tasks in reverse order
            for cleanup_task in reversed(self.cleanup_tasks):
                try:
                    await cleanup_task()
                except Exception as e:
                    logger.error(f"Resource cleanup failed: {e}")
            
            self.resources.clear()
            self.cleanup_tasks.clear()
    
    async def __aenter__(self):
        """Async context entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context exit with resource cleanup."""
        await self.release_all()
        return False

# Decorator for automatic resource management
def async_resource_managed(func):
    """Decorator for automatic async resource management."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with AsyncResourceManager() as manager:
            # Inject resource manager into function
            if 'resource_manager' in func.__code__.co_varnames:
                kwargs['resource_manager'] = manager
            return await func(*args, **kwargs)
    return wrapper

# Utility functions for common async resources
async def managed_async_file(file_path: str, mode: str = 'r'):
    """Create managed async file resource."""
    import aiofiles
    file_handle = await aiofiles.open(file_path, mode)
    return file_handle, lambda f: f.close()

async def managed_async_connection(connection_factory: Callable[[], Awaitable[Any]]):
    """Create managed async connection resource."""
    connection = await connection_factory()
    return connection, lambda conn: conn.close()
"#.to_string())
    }

    /// Get generated patterns
    pub fn get_generated_patterns(&self) -> &HashMap<String, String> {
        &self.generated_patterns
    }

    /// Generate complete async support module
    pub fn generate_complete_async_module(&mut self, functions: &[PIRFunction]) -> PythonResult<String> {
        let mut output = String::new();
        
        // Module header
        output.push_str("# === ASYNC SUPPORT MODULE ===\n");
        output.push_str("# Comprehensive async/await patterns for Prism effects and capabilities\n\n");
        
        // Imports
        output.push_str(&self.generate_async_imports());
        
        // Structured concurrency
        output.push_str(&self.generate_structured_concurrency()?);
        
        // Async comprehensions
        output.push_str(&self.generate_async_comprehensions()?);
        
        // Resource management
        output.push_str(&self.generate_async_resource_management()?);
        
        // Function-specific patterns
        for function in functions {
            if !function.signature.effects.effects.is_empty() {
                output.push_str(&self.generate_async_function(function)?);
            }
        }
        
        Ok(output)
    }

    /// Generate async imports
    fn generate_async_imports(&self) -> String {
        r#"import asyncio
import logging
from typing import Any, Awaitable, AsyncGenerator, AsyncIterable, Callable, List, Optional
from contextlib import asynccontextmanager
from functools import wraps
import time

# Prism runtime imports
from prism_runtime import EffectTracker, CapabilityManager, ValidationError, EffectError

logger = logging.getLogger('prism_async')

"#.to_string()
    }
} 