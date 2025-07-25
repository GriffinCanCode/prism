//! Integration tests for the allocators subsystem
//!
//! These tests verify that all allocator components work together correctly
//! and provide the expected performance characteristics.

use super::*;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_allocator_factory() {
    // Test factory methods
    let prism = AllocatorFactory::new_prism_allocator();
    let bump = AllocatorFactory::new_bump_allocator(1024);
    let large = AllocatorFactory::new_large_object_allocator();
    let page = AllocatorFactory::new_page_allocator();
    
    // Verify they implement the Allocator trait
    let _: Box<dyn Allocator> = Box::new(prism);
    let _: Box<dyn Allocator> = Box::new(bump);
    let _: Box<dyn Allocator> = Box::new(large);
    let _: Box<dyn Allocator> = Box::new(page);
}

#[test]
fn test_allocator_manager_integration() {
    let manager = AllocatorManager::new();
    
    // Test different allocation sizes to exercise different allocators
    let small_ptr = manager.allocate(32, 8).expect("Small allocation failed");
    let medium_ptr = manager.allocate(1024, 8).expect("Medium allocation failed");
    let large_ptr = manager.allocate(40000, 8).expect("Large allocation failed");
    
    // Verify pointers are different
    assert_ne!(small_ptr.as_ptr(), medium_ptr.as_ptr());
    assert_ne!(medium_ptr.as_ptr(), large_ptr.as_ptr());
    assert_ne!(small_ptr.as_ptr(), large_ptr.as_ptr());
    
    // Test deallocation
    manager.deallocate(small_ptr, 32);
    manager.deallocate(medium_ptr, 1024);
    manager.deallocate(large_ptr, 40000);
    
    // Verify statistics
    let stats = manager.get_stats();
    assert!(stats.total_requests >= 3);
}

#[test]
fn test_size_class_utils() {
    // Test size class finding
    assert_eq!(SizeClassUtils::find_size_class(1), Some(0)); // Should map to 8
    assert_eq!(SizeClassUtils::find_size_class(8), Some(0));
    assert_eq!(SizeClassUtils::find_size_class(16), Some(1));
    assert_eq!(SizeClassUtils::find_size_class(17), Some(2)); // Should map to 24
    
    // Test size retrieval
    assert_eq!(SizeClassUtils::get_size_for_class(0), Some(8));
    assert_eq!(SizeClassUtils::get_size_for_class(1), Some(16));
    
    // Test large object detection
    assert!(!SizeClassUtils::is_large_object(1000));
    assert!(SizeClassUtils::is_large_object(LARGE_OBJECT_THRESHOLD + 1));
    
    // Test objects per page calculation
    let objects_per_page = SizeClassUtils::objects_per_page(0); // 8-byte objects
    assert_eq!(objects_per_page, PAGE_SIZE / 8);
}

#[test]
fn test_multithreaded_allocation() {
    let manager = Arc::new(AllocatorManager::new());
    let mut handles = vec![];
    
    // Spawn multiple threads doing concurrent allocations
    for thread_id in 0..4 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            let mut allocations = vec![];
            
            // Mix of different allocation sizes
            for i in 0..50 {
                let size = match i % 4 {
                    0 => 32,      // Small
                    1 => 128,     // Medium small
                    2 => 1024,    // Medium
                    3 => 8192,    // Large-ish
                    _ => unreachable!(),
                };
                
                if let Some(ptr) = manager_clone.allocate(size, 8) {
                    // Write to memory to ensure it's valid
                    unsafe {
                        std::ptr::write(ptr.as_ptr(), thread_id as u8);
                        assert_eq!(std::ptr::read(ptr.as_ptr()), thread_id as u8);
                    }
                    allocations.push((ptr, size));
                }
            }
            
            // Deallocate everything
            for (ptr, size) in allocations {
                manager_clone.deallocate(ptr, size);
            }
        });
        handles.push(handle);
    }
    
    // Wait for completion
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Verify final state
    let stats = manager.get_stats();
    assert!(stats.total_requests >= 200); // 4 threads * 50 allocations
}

#[test]
fn test_gc_integration() {
    let manager = AllocatorManager::new();
    
    // Allocate enough to potentially trigger GC
    let mut allocations = vec![];
    for i in 0..1000 {
        let size = 1024; // 1KB each
        if let Some(ptr) = manager.allocate(size, 8) {
            allocations.push((ptr, size));
        }
        
        // Check if GC should be triggered
        if manager.should_trigger_gc() {
            manager.prepare_for_gc();
            break;
        }
    }
    
    // Clean up
    for (ptr, size) in allocations {
        manager.deallocate(ptr, size);
    }
}

#[test]
fn test_allocator_strategy_switching() {
    let manager = AllocatorManager::new();
    
    // Test different strategies
    manager.set_strategy(AllocatorStrategy::SizeClass);
    assert_eq!(manager.get_strategy(), AllocatorStrategy::SizeClass);
    
    manager.set_strategy(AllocatorStrategy::Bump);
    assert_eq!(manager.get_strategy(), AllocatorStrategy::Bump);
    
    manager.set_strategy(AllocatorStrategy::LargeObject);
    assert_eq!(manager.get_strategy(), AllocatorStrategy::LargeObject);
    
    manager.set_strategy(AllocatorStrategy::Adaptive);
    assert_eq!(manager.get_strategy(), AllocatorStrategy::Adaptive);
    
    // Test allocation with different strategies
    for strategy in &[
        AllocatorStrategy::SizeClass,
        AllocatorStrategy::Bump,
        AllocatorStrategy::Adaptive,
    ] {
        manager.set_strategy(*strategy);
        
        // Small allocation
        let ptr = manager.allocate(64, 8).expect("Strategy allocation failed");
        manager.deallocate(ptr, 64);
    }
}

#[test]
fn test_configuration_propagation() {
    let mut config = AllocatorManagerConfig::default();
    config.large_object_threshold = 16384; // 16KB instead of 32KB
    config.global_gc_threshold = 32 * 1024 * 1024; // 32MB
    
    let manager = AllocatorManager::with_config(config.clone());
    let retrieved_config = manager.get_config();
    
    assert_eq!(retrieved_config.large_object_threshold, 16384);
    assert_eq!(retrieved_config.global_gc_threshold, 32 * 1024 * 1024);
}

#[test]
fn test_statistics_collection() {
    let manager = AllocatorManager::new();
    
    // Make various allocations
    let mut ptrs = vec![];
    for size in &[32, 64, 128, 256, 512, 1024] {
        for _ in 0..10 {
            if let Some(ptr) = manager.allocate(*size, 8) {
                ptrs.push((ptr, *size));
            }
        }
    }
    
    let stats = manager.get_stats();
    
    // Verify statistics are reasonable
    assert!(stats.total_requests > 0);
    assert!(stats.prism_stats.allocation_count > 0);
    assert!(stats.prism_stats.live_bytes > 0);
    
    // Clean up
    for (ptr, size) in ptrs {
        manager.deallocate(ptr, size);
    }
    
    // Statistics should reflect deallocations
    let final_stats = manager.get_stats();
    assert_eq!(final_stats.prism_stats.live_bytes, 0);
}

#[test]
fn test_error_conditions() {
    let manager = AllocatorManager::new();
    
    // Test zero-size allocation
    let zero_ptr = manager.allocate(0, 8);
    assert!(zero_ptr.is_some());
    
    // Test very large allocation
    let huge_ptr = manager.allocate(usize::MAX, 8);
    // This might fail or succeed depending on system memory, both are valid
    
    if let Some(ptr) = zero_ptr {
        manager.deallocate(ptr, 0);
    }
    
    if let Some(ptr) = huge_ptr {
        manager.deallocate(ptr, usize::MAX);
    }
}

#[test]
fn test_memory_alignment() {
    let manager = AllocatorManager::new();
    
    // Test various alignment requirements
    for &align in &[1, 2, 4, 8, 16, 32, 64] {
        let ptr = manager.allocate(1024, align);
        if let Some(ptr) = ptr {
            assert_eq!(ptr.as_ptr() as usize % align, 0,
                      "Allocation not aligned to {} bytes", align);
            manager.deallocate(ptr, 1024);
        }
    }
}

#[test]
fn test_allocator_coordination() {
    let manager = AllocatorManager::new();
    
    // Test that different allocation sizes are routed correctly
    let small = manager.allocate(16, 8).unwrap(); // Should go to PrismAllocator
    let large = manager.allocate(50000, 8).unwrap(); // Should go to LargeObjectAllocator
    
    // Verify they're handled by different allocators by checking statistics
    let stats_before = manager.get_stats();
    
    manager.deallocate(small, 16);
    manager.deallocate(large, 50000);
    
    let stats_after = manager.get_stats();
    
    // Both should show activity
    assert!(stats_before.prism_stats.allocation_count > 0);
    assert!(stats_before.large_object_stats.allocation_count > 0);
}

#[test]
fn test_performance_characteristics() {
    let manager = AllocatorManager::new();
    let start_time = std::time::Instant::now();
    
    // Perform many allocations to test performance
    let mut allocations = vec![];
    for _ in 0..1000 {
        if let Some(ptr) = manager.allocate(64, 8) {
            allocations.push(ptr);
        }
    }
    
    let allocation_time = start_time.elapsed();
    
    // Deallocate all
    let dealloc_start = std::time::Instant::now();
    for ptr in allocations {
        manager.deallocate(ptr, 64);
    }
    let deallocation_time = dealloc_start.elapsed();
    
    // Performance should be reasonable (this is a rough check)
    assert!(allocation_time.as_millis() < 100, "Allocations too slow");
    assert!(deallocation_time.as_millis() < 100, "Deallocations too slow");
}

#[test]
fn test_memory_pressure_handling() {
    let manager = AllocatorManager::new();
    
    // Allocate until we hit memory pressure
    let mut allocations = vec![];
    let mut total_allocated = 0;
    
    while total_allocated < 10 * 1024 * 1024 { // 10MB
        if let Some(ptr) = manager.allocate(1024, 8) {
            allocations.push(ptr);
            total_allocated += 1024;
        } else {
            break; // Allocation failed
        }
        
        // Check if GC is triggered
        if manager.should_trigger_gc() {
            manager.prepare_for_gc();
            // In a real scenario, GC would run here
            break;
        }
    }
    
    // Clean up
    for ptr in allocations {
        manager.deallocate(ptr, 1024);
    }
    
    assert!(total_allocated > 0, "Should have allocated some memory");
}

#[test]
fn test_allocator_specific_features() {
    // Test BumpAllocator specific features
    let bump = BumpAllocator::new(1024);
    let ptr1 = bump.allocate(32, 8).unwrap();
    let ptr2 = bump.allocate(32, 8).unwrap();
    
    // Bump allocator should allocate sequentially
    let addr1 = ptr1.as_ptr() as usize;
    let addr2 = ptr2.as_ptr() as usize;
    assert!(addr2 > addr1, "Bump allocator should allocate sequentially");
    
    // Test reset functionality
    let usage_before = bump.memory_usage();
    bump.post_gc_reset();
    let usage_after = bump.memory_usage();
    assert_eq!(usage_after.allocated, 0, "Reset should clear allocated memory");
    
    // Test LargeObjectAllocator specific features
    let large = LargeObjectAllocator::new();
    let large_ptr = large.allocate(100000, 8).unwrap();
    
    let detailed_stats = large.get_detailed_stats();
    assert_eq!(detailed_stats.live_objects, 1);
    assert_eq!(detailed_stats.largest_object_size, 100000);
    
    large.deallocate(large_ptr, 100000);
    
    let final_stats = large.get_detailed_stats();
    assert_eq!(final_stats.live_objects, 0);
}

/// Benchmark test for allocation performance
#[test]
fn benchmark_allocation_performance() {
    let manager = AllocatorManager::new();
    let iterations = 10000;
    
    let start = std::time::Instant::now();
    
    // Allocation benchmark
    let mut ptrs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        if let Some(ptr) = manager.allocate(64, 8) {
            ptrs.push(ptr);
        }
    }
    
    let allocation_time = start.elapsed();
    
    // Deallocation benchmark
    let dealloc_start = std::time::Instant::now();
    for ptr in ptrs {
        manager.deallocate(ptr, 64);
    }
    let deallocation_time = dealloc_start.elapsed();
    
    let total_time = allocation_time + deallocation_time;
    let ops_per_sec = (iterations * 2) as f64 / total_time.as_secs_f64();
    
    println!("Allocation performance: {:.0} ops/sec", ops_per_sec);
    println!("Average allocation time: {:?}", allocation_time / iterations as u32);
    println!("Average deallocation time: {:?}", deallocation_time / iterations as u32);
    
    // Performance should be reasonable
    assert!(ops_per_sec > 100000.0, "Performance too low: {} ops/sec", ops_per_sec);
} 