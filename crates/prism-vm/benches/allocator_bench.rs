use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prism_vm::gc::allocators::{PrismAllocator, PrismAllocatorConfig};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Benchmark single-threaded allocation patterns
fn bench_single_threaded_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_threaded_allocation");
    
    // Test different allocation sizes
    let sizes = vec![8, 32, 128, 512, 1024, 4096, 16384];
    
    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("allocate_deallocate", size),
            &size,
            |b, &size| {
                let allocator = PrismAllocator::new();
                b.iter(|| {
                    let ptr = allocator.allocate(black_box(size), 8).unwrap();
                    allocator.deallocate(black_box(ptr), black_box(size));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark allocation patterns that stress size classes
fn bench_size_class_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_class_patterns");
    
    // Test rapid small allocations (typical for AST nodes)
    group.bench_function("rapid_small_allocs", |b| {
        let allocator = PrismAllocator::new();
        b.iter(|| {
            let mut ptrs = Vec::with_capacity(100);
            for _ in 0..100 {
                let size = black_box(32); // Typical AST node size
                if let Some(ptr) = allocator.allocate(size, 8) {
                    ptrs.push((ptr, size));
                }
            }
            // Deallocate all
            for (ptr, size) in ptrs {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    // Test mixed size allocations
    group.bench_function("mixed_size_allocs", |b| {
        let allocator = PrismAllocator::new();
        let sizes = vec![16, 64, 256, 1024, 4096];
        b.iter(|| {
            let mut ptrs = Vec::with_capacity(50);
            for &size in &sizes {
                for _ in 0..10 {
                    if let Some(ptr) = allocator.allocate(black_box(size), 8) {
                        ptrs.push((ptr, size));
                    }
                }
            }
            // Deallocate all
            for (ptr, size) in ptrs {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    group.finish();
}

/// Benchmark multi-threaded allocation patterns
fn bench_multi_threaded_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_threaded_allocation");
    
    // Test concurrent allocations from multiple threads
    group.bench_function("concurrent_allocs_4_threads", |b| {
        let allocator = Arc::new(PrismAllocator::new());
        b.iter(|| {
            let mut handles = vec![];
            
            for _ in 0..4 {
                let allocator_clone = Arc::clone(&allocator);
                let handle = thread::spawn(move || {
                    let mut ptrs = Vec::with_capacity(25);
                    for _ in 0..25 {
                        let size = black_box(128);
                        if let Some(ptr) = allocator_clone.allocate(size, 8) {
                            ptrs.push((ptr, size));
                        }
                    }
                    // Deallocate all
                    for (ptr, size) in ptrs {
                        allocator_clone.deallocate(ptr, size);
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.finish();
}

/// Benchmark large object allocation
fn bench_large_object_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_object_allocation");
    
    let large_sizes = vec![64 * 1024, 256 * 1024, 1024 * 1024]; // 64KB, 256KB, 1MB
    
    for size in large_sizes {
        group.bench_with_input(
            BenchmarkId::new("large_alloc_dealloc", size),
            &size,
            |b, &size| {
                let allocator = PrismAllocator::new();
                b.iter(|| {
                    let ptr = allocator.allocate(black_box(size), 8).unwrap();
                    allocator.deallocate(black_box(ptr), black_box(size));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark allocator configuration impact
fn bench_allocator_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocator_configs");
    
    // Test with thread caching enabled vs disabled
    group.bench_function("with_thread_cache", |b| {
        let config = PrismAllocatorConfig {
            base: prism_vm::gc::allocators::AllocatorConfig {
                enable_thread_cache: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let allocator = PrismAllocator::with_config(config);
        
        b.iter(|| {
            let mut ptrs = Vec::with_capacity(100);
            for _ in 0..100 {
                let size = black_box(64);
                if let Some(ptr) = allocator.allocate(size, 8) {
                    ptrs.push((ptr, size));
                }
            }
            for (ptr, size) in ptrs {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    group.bench_function("without_thread_cache", |b| {
        let config = PrismAllocatorConfig {
            base: prism_vm::gc::allocators::AllocatorConfig {
                enable_thread_cache: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let allocator = PrismAllocator::with_config(config);
        
        b.iter(|| {
            let mut ptrs = Vec::with_capacity(100);
            for _ in 0..100 {
                let size = black_box(64);
                if let Some(ptr) = allocator.allocate(size, 8) {
                    ptrs.push((ptr, size));
                }
            }
            for (ptr, size) in ptrs {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    group.finish();
}

/// Benchmark memory pressure and GC trigger scenarios
fn bench_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    
    group.bench_function("gc_trigger_detection", |b| {
        let config = PrismAllocatorConfig {
            base: prism_vm::gc::allocators::AllocatorConfig {
                gc_trigger_threshold: 1024 * 1024, // 1MB threshold
                ..Default::default()
            },
            ..Default::default()
        };
        let allocator = PrismAllocator::with_config(config);
        
        b.iter(|| {
            let mut ptrs = Vec::new();
            let mut should_trigger = false;
            
            // Allocate until we should trigger GC
            for _ in 0..1000 {
                let size = black_box(1024);
                if let Some(ptr) = allocator.allocate(size, 8) {
                    ptrs.push((ptr, size));
                    if allocator.should_trigger_gc() {
                        should_trigger = true;
                        break;
                    }
                }
            }
            
            black_box(should_trigger);
            
            // Clean up
            for (ptr, size) in ptrs {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    group.finish();
}

/// Benchmark realistic VM workload patterns
fn bench_vm_workload_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_workload_patterns");
    
    // Simulate AST node allocation pattern
    group.bench_function("ast_node_pattern", |b| {
        let allocator = PrismAllocator::new();
        b.iter(|| {
            let mut nodes = Vec::with_capacity(200);
            
            // Allocate nodes of various sizes (simulating different AST node types)
            for _ in 0..50 {
                // Expression nodes (small)
                if let Some(ptr) = allocator.allocate(black_box(32), 8) {
                    nodes.push((ptr, 32));
                }
            }
            
            for _ in 0..30 {
                // Statement nodes (medium)
                if let Some(ptr) = allocator.allocate(black_box(128), 8) {
                    nodes.push((ptr, 128));
                }
            }
            
            for _ in 0..10 {
                // Function/class nodes (large)
                if let Some(ptr) = allocator.allocate(black_box(512), 8) {
                    nodes.push((ptr, 512));
                }
            }
            
            // Simulate some nodes being freed during compilation
            for _ in 0..20 {
                if let Some((ptr, size)) = nodes.pop() {
                    allocator.deallocate(ptr, size);
                }
            }
            
            // Clean up remaining nodes
            for (ptr, size) in nodes {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    // Simulate bytecode generation pattern
    group.bench_function("bytecode_generation_pattern", |b| {
        let allocator = PrismAllocator::new();
        b.iter(|| {
            let mut buffers = Vec::with_capacity(20);
            
            // Allocate bytecode buffers
            for _ in 0..20 {
                let size = black_box(4096); // 4KB bytecode buffer
                if let Some(ptr) = allocator.allocate(size, 8) {
                    buffers.push((ptr, size));
                }
            }
            
            // Simulate temporary objects during generation
            let mut temps = Vec::with_capacity(100);
            for _ in 0..100 {
                let size = black_box(64); // Temporary objects
                if let Some(ptr) = allocator.allocate(size, 8) {
                    temps.push((ptr, size));
                }
            }
            
            // Clean up temporaries first (they die young)
            for (ptr, size) in temps {
                allocator.deallocate(ptr, size);
            }
            
            // Clean up buffers
            for (ptr, size) in buffers {
                allocator.deallocate(ptr, size);
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_threaded_allocation,
    bench_size_class_patterns,
    bench_multi_threaded_allocation,
    bench_large_object_allocation,
    bench_allocator_configs,
    bench_memory_pressure,
    bench_vm_workload_patterns
);

criterion_main!(benches); 