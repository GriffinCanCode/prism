//! Concrete implementations of object tracers for different data types
//!
//! This module provides specialized tracer implementations for common object
//! types found in the Prism VM, including arrays, structs, closures, and
//! Prism-specific object types.
//!
//! ## Design Principles
//!
//! - **Type Safety**: Each tracer is specialized for specific object layouts
//! - **Performance**: Optimized tracing paths for each object type
//! - **Correctness**: Precise reference tracing without false positives
//! - **Extensibility**: Easy to add new tracer types for custom objects

use super::types::*;
use std::sync::Arc;

/// Tracer for arrays of GC pointers
/// 
/// This tracer handles arrays where each element may contain a GC reference.
/// It's optimized for uniform element sizes and known offsets.
pub struct ArrayTracer {
    /// Size of each array element in bytes
    element_size: usize,
    /// Offset within each element where the GC pointer is located
    element_offset: usize,
    /// Optional validation function for elements
    element_validator: Option<fn(*const u8) -> bool>,
}

impl ArrayTracer {
    /// Create a new array tracer
    /// 
    /// # Parameters
    /// 
    /// - `element_size`: Size of each array element in bytes
    /// - `element_offset`: Offset within each element where the pointer is located
    pub fn new(element_size: usize, element_offset: usize) -> Self {
        Self {
            element_size,
            element_offset,
            element_validator: None,
        }
    }
    
    /// Create an array tracer with element validation
    pub fn with_validator(
        element_size: usize, 
        element_offset: usize,
        validator: fn(*const u8) -> bool
    ) -> Self {
        Self {
            element_size,
            element_offset,
            element_validator: Some(validator),
        }
    }
    
    /// Calculate the number of elements in the array
    fn calculate_element_count(&self, object_size: usize) -> usize {
        if self.element_size == 0 {
            return 0;
        }
        
        // Account for object header
        let header_size = std::mem::size_of::<ObjectHeader>();
        let data_size = object_size.saturating_sub(header_size);
        
        data_size / self.element_size
    }
}

impl ObjectTracer for ArrayTracer {
    unsafe fn trace_references(&self, object_ptr: *const u8, object_size: usize) -> Vec<*const u8> {
        let mut references = Vec::new();
        
        if self.element_size == 0 {
            return references;
        }
        
        let header_size = std::mem::size_of::<ObjectHeader>();
        let data_ptr = object_ptr.add(header_size);
        let num_elements = self.calculate_element_count(object_size);
        
        for i in 0..num_elements {
            let element_ptr = data_ptr.add(i * self.element_size);
            
            // Validate element if validator is provided
            if let Some(validator) = self.element_validator {
                if !validator(element_ptr) {
                    continue;
                }
            }
            
            let field_ptr = element_ptr.add(self.element_offset);
            
            // Ensure we don't read beyond the object bounds
            if field_ptr.add(std::mem::size_of::<*const u8>()) > 
               object_ptr.add(object_size) {
                break;
            }
            
            let gc_ptr = *(field_ptr as *const *const u8);
            
            if !gc_ptr.is_null() {
                references.push(gc_ptr);
            }
        }
        
        references
    }
    
    fn has_references(&self) -> bool {
        true
    }
    
    fn tracer_name(&self) -> &'static str {
        "ArrayTracer"
    }
    
    fn can_trace(&self, _object_ptr: *const u8, object_size: usize) -> bool {
        let header_size = std::mem::size_of::<ObjectHeader>();
        object_size > header_size && self.element_size > 0
    }
    
    fn get_object_size(&self, object_ptr: *const u8) -> Option<usize> {
        unsafe {
            let header = &*(object_ptr as *const ObjectHeader);
            Some(header.size)
        }
    }
}

/// Tracer for structs with known field layouts
/// 
/// This tracer is optimized for objects with a fixed set of GC pointer fields
/// at known offsets within the struct.
pub struct StructTracer {
    /// Offsets of GC pointer fields within the struct
    gc_field_offsets: Vec<usize>,
    /// Optional field validators
    field_validators: Vec<Option<fn(*const u8) -> bool>>,
    /// Name of the struct type (for debugging)
    struct_name: &'static str,
}

impl StructTracer {
    /// Create a new struct tracer with the given field offsets
    pub fn new(gc_field_offsets: Vec<usize>) -> Self {
        let field_validators = vec![None; gc_field_offsets.len()];
        Self {
            gc_field_offsets,
            field_validators,
            struct_name: "UnknownStruct",
        }
    }
    
    /// Create a struct tracer with field validators
    pub fn with_validators(
        gc_field_offsets: Vec<usize>,
        field_validators: Vec<Option<fn(*const u8) -> bool>>
    ) -> Self {
        assert_eq!(gc_field_offsets.len(), field_validators.len());
        Self {
            gc_field_offsets,
            field_validators,
            struct_name: "UnknownStruct",
        }
    }
    
    /// Create a named struct tracer
    pub fn named(gc_field_offsets: Vec<usize>, struct_name: &'static str) -> Self {
        let field_validators = vec![None; gc_field_offsets.len()];
        Self {
            gc_field_offsets,
            field_validators,
            struct_name,
        }
    }
}

impl ObjectTracer for StructTracer {
    unsafe fn trace_references(&self, object_ptr: *const u8, object_size: usize) -> Vec<*const u8> {
        let mut references = Vec::new();
        
        let header_size = std::mem::size_of::<ObjectHeader>();
        let data_ptr = object_ptr.add(header_size);
        
        for (i, &offset) in self.gc_field_offsets.iter().enumerate() {
            let field_ptr = data_ptr.add(offset);
            
            // Bounds check
            if field_ptr.add(std::mem::size_of::<*const u8>()) > 
               object_ptr.add(object_size) {
                continue;
            }
            
            // Validate field if validator is provided
            if let Some(validator) = self.field_validators[i] {
                if !validator(field_ptr) {
                    continue;
                }
            }
            
            let gc_ptr = *(field_ptr as *const *const u8);
            
            if !gc_ptr.is_null() {
                references.push(gc_ptr);
            }
        }
        
        references
    }
    
    fn has_references(&self) -> bool {
        !self.gc_field_offsets.is_empty()
    }
    
    fn tracer_name(&self) -> &'static str {
        self.struct_name
    }
    
    fn can_trace(&self, _object_ptr: *const u8, object_size: usize) -> bool {
        let header_size = std::mem::size_of::<ObjectHeader>();
        object_size > header_size
    }
}

/// Tracer for closures that capture GC references
/// 
/// This tracer handles closure objects that may capture variables containing
/// GC references from their enclosing scope.
pub struct ClosureTracer {
    /// Offsets of captured GC values within the closure
    capture_offsets: Vec<usize>,
    /// Types of captured values (for validation)
    capture_types: Vec<CaptureType>,
    /// Closure signature information (for debugging)
    closure_signature: Option<&'static str>,
}

/// Type of captured value in a closure
#[derive(Debug, Clone, Copy)]
pub enum CaptureType {
    /// Direct GC pointer
    DirectPointer,
    /// Boxed value containing a GC pointer
    BoxedPointer,
    /// Optional GC pointer (may be null)
    OptionalPointer,
    /// Array of GC pointers
    PointerArray { element_count: usize },
}

impl ClosureTracer {
    /// Create a new closure tracer
    pub fn new(capture_offsets: Vec<usize>) -> Self {
        let capture_types = vec![CaptureType::DirectPointer; capture_offsets.len()];
        Self {
            capture_offsets,
            capture_types,
            closure_signature: None,
        }
    }
    
    /// Create a closure tracer with capture type information
    pub fn with_types(
        capture_offsets: Vec<usize>,
        capture_types: Vec<CaptureType>
    ) -> Self {
        assert_eq!(capture_offsets.len(), capture_types.len());
        Self {
            capture_offsets,
            capture_types,
            closure_signature: None,
        }
    }
    
    /// Create a closure tracer with signature information
    pub fn with_signature(
        capture_offsets: Vec<usize>,
        capture_types: Vec<CaptureType>,
        signature: &'static str
    ) -> Self {
        assert_eq!(capture_offsets.len(), capture_types.len());
        Self {
            capture_offsets,
            capture_types,
            closure_signature: Some(signature),
        }
    }
}

impl ObjectTracer for ClosureTracer {
    unsafe fn trace_references(&self, object_ptr: *const u8, object_size: usize) -> Vec<*const u8> {
        let mut references = Vec::new();
        
        let header_size = std::mem::size_of::<ObjectHeader>();
        let data_ptr = object_ptr.add(header_size);
        
        for (i, &offset) in self.capture_offsets.iter().enumerate() {
            let capture_ptr = data_ptr.add(offset);
            
            // Bounds check
            if capture_ptr >= object_ptr.add(object_size) {
                continue;
            }
            
            match self.capture_types[i] {
                CaptureType::DirectPointer => {
                    if capture_ptr.add(std::mem::size_of::<*const u8>()) > 
                       object_ptr.add(object_size) {
                        continue;
                    }
                    
                    let gc_ptr = *(capture_ptr as *const *const u8);
                    if !gc_ptr.is_null() {
                        references.push(gc_ptr);
                    }
                }
                
                CaptureType::BoxedPointer => {
                    // Handle boxed values - the capture contains a pointer to a box
                    // which itself contains the GC pointer
                    if capture_ptr.add(std::mem::size_of::<*const u8>()) > 
                       object_ptr.add(object_size) {
                        continue;
                    }
                    
                    let box_ptr = *(capture_ptr as *const *const u8);
                    if !box_ptr.is_null() {
                        let gc_ptr = *(box_ptr as *const *const u8);
                        if !gc_ptr.is_null() {
                            references.push(gc_ptr);
                        }
                    }
                }
                
                CaptureType::OptionalPointer => {
                    // Handle optional pointers (same as direct but may be null)
                    if capture_ptr.add(std::mem::size_of::<*const u8>()) > 
                       object_ptr.add(object_size) {
                        continue;
                    }
                    
                    let gc_ptr = *(capture_ptr as *const *const u8);
                    if !gc_ptr.is_null() {
                        references.push(gc_ptr);
                    }
                }
                
                CaptureType::PointerArray { element_count } => {
                    // Handle arrays of pointers
                    let array_size = element_count * std::mem::size_of::<*const u8>();
                    if capture_ptr.add(array_size) > object_ptr.add(object_size) {
                        continue;
                    }
                    
                    for j in 0..element_count {
                        let elem_ptr = capture_ptr.add(j * std::mem::size_of::<*const u8>());
                        let gc_ptr = *(elem_ptr as *const *const u8);
                        if !gc_ptr.is_null() {
                            references.push(gc_ptr);
                        }
                    }
                }
            }
        }
        
        references
    }
    
    fn has_references(&self) -> bool {
        !self.capture_offsets.is_empty()
    }
    
    fn tracer_name(&self) -> &'static str {
        "ClosureTracer"
    }
    
    fn can_trace(&self, _object_ptr: *const u8, object_size: usize) -> bool {
        let header_size = std::mem::size_of::<ObjectHeader>();
        object_size > header_size
    }
}

/// Conservative tracer for Prism-specific object types
/// 
/// This tracer uses heuristics to identify potential GC pointers when the
/// exact object layout is unknown. It's safer but potentially less efficient
/// than specialized tracers.
pub struct PrismObjectTracer {
    /// Whether to use conservative scanning
    conservative_mode: bool,
    /// Alignment requirement for pointers
    pointer_alignment: usize,
    /// Address range validation function
    address_validator: Option<fn(*const u8) -> bool>,
}

impl PrismObjectTracer {
    /// Create a new Prism object tracer
    pub fn new() -> Self {
        Self {
            conservative_mode: true,
            pointer_alignment: std::mem::align_of::<*const u8>(),
            address_validator: None,
        }
    }
    
    /// Create a precise tracer (no conservative scanning)
    pub fn precise() -> Self {
        Self {
            conservative_mode: false,
            pointer_alignment: std::mem::align_of::<*const u8>(),
            address_validator: None,
        }
    }
    
    /// Create a tracer with custom address validator
    pub fn with_validator(validator: fn(*const u8) -> bool) -> Self {
        Self {
            conservative_mode: true,
            pointer_alignment: std::mem::align_of::<*const u8>(),
            address_validator: Some(validator),
        }
    }
}

impl Default for PrismObjectTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectTracer for PrismObjectTracer {
    unsafe fn trace_references(&self, object_ptr: *const u8, object_size: usize) -> Vec<*const u8> {
        if !self.conservative_mode {
            // Precise mode: assume no references unless we have specific knowledge
            return Vec::new();
        }
        
        let mut references = Vec::new();
        
        // Skip the object header
        let header_size = std::mem::size_of::<ObjectHeader>();
        if object_size <= header_size {
            return references;
        }
        
        let data_ptr = object_ptr.add(header_size);
        let data_size = object_size - header_size;
        
        // Conservative scanning: look for potential pointers
        let mut current = data_ptr;
        let end = data_ptr.add(data_size);
        
        while current < end {
            // Align to pointer boundary
            let aligned = self.align_pointer(current);
            if aligned.add(std::mem::size_of::<*const u8>()) > end {
                break;
            }
            
            let potential_ptr = *(aligned as *const *const u8);
            
            // Validate the potential pointer
            if self.could_be_gc_pointer(potential_ptr) {
                references.push(potential_ptr);
            }
            
            current = aligned.add(self.pointer_alignment);
        }
        
        references
    }
    
    fn has_references(&self) -> bool {
        self.conservative_mode
    }
    
    fn tracer_name(&self) -> &'static str {
        if self.conservative_mode {
            "PrismConservativeTracer"
        } else {
            "PrismPreciseTracer"
        }
    }
    
    fn can_trace(&self, _object_ptr: *const u8, object_size: usize) -> bool {
        let header_size = std::mem::size_of::<ObjectHeader>();
        object_size > header_size
    }
}

impl PrismObjectTracer {
    /// Align a pointer to the required boundary
    unsafe fn align_pointer(&self, ptr: *const u8) -> *const u8 {
        let addr = ptr as usize;
        let aligned_addr = (addr + self.pointer_alignment - 1) & !(self.pointer_alignment - 1);
        aligned_addr as *const u8
    }
    
    /// Heuristic to determine if a value could be a GC pointer
    fn could_be_gc_pointer(&self, ptr: *const u8) -> bool {
        if ptr.is_null() {
            return false;
        }
        
        // Use custom validator if provided
        if let Some(validator) = self.address_validator {
            return validator(ptr);
        }
        
        let addr = ptr as usize;
        
        // Check alignment
        if addr % self.pointer_alignment != 0 {
            return false;
        }
        
        // Basic address range validation
        // This is a heuristic and would need refinement for production use
        addr >= 0x1000 && addr <= 0x7fff_ffff_ffff
    }
}

/// Factory for creating common tracer types
pub struct TracerFactory;

impl TracerFactory {
    /// Create an array tracer for arrays of direct pointers
    pub fn array_of_pointers(element_size: usize) -> Arc<dyn ObjectTracer> {
        Arc::new(ArrayTracer::new(element_size, 0))
    }
    
    /// Create an array tracer for arrays of objects with pointer fields
    pub fn array_of_objects(element_size: usize, pointer_offset: usize) -> Arc<dyn ObjectTracer> {
        Arc::new(ArrayTracer::new(element_size, pointer_offset))
    }
    
    /// Create a struct tracer for simple structs with pointer fields
    pub fn simple_struct(field_offsets: Vec<usize>) -> Arc<dyn ObjectTracer> {
        Arc::new(StructTracer::new(field_offsets))
    }
    
    /// Create a named struct tracer
    pub fn named_struct(field_offsets: Vec<usize>, name: &'static str) -> Arc<dyn ObjectTracer> {
        Arc::new(StructTracer::named(field_offsets, name))
    }
    
    /// Create a closure tracer for simple closures
    pub fn simple_closure(capture_offsets: Vec<usize>) -> Arc<dyn ObjectTracer> {
        Arc::new(ClosureTracer::new(capture_offsets))
    }
    
    /// Create a conservative tracer for unknown object types
    pub fn conservative() -> Arc<dyn ObjectTracer> {
        Arc::new(PrismObjectTracer::new())
    }
    
    /// Create a precise tracer (no references assumed)
    pub fn precise() -> Arc<dyn ObjectTracer> {
        Arc::new(PrismObjectTracer::precise())
    }
}

/// Utility for building tracers with a fluent interface
pub struct TracerBuilder {
    tracer_type: TracerType,
}

enum TracerType {
    Array {
        element_size: usize,
        element_offset: usize,
        validator: Option<fn(*const u8) -> bool>,
    },
    Struct {
        field_offsets: Vec<usize>,
        validators: Vec<Option<fn(*const u8) -> bool>>,
        name: &'static str,
    },
    Closure {
        capture_offsets: Vec<usize>,
        capture_types: Vec<CaptureType>,
        signature: Option<&'static str>,
    },
    Conservative {
        address_validator: Option<fn(*const u8) -> bool>,
    },
}

impl TracerBuilder {
    /// Start building an array tracer
    pub fn array(element_size: usize) -> Self {
        Self {
            tracer_type: TracerType::Array {
                element_size,
                element_offset: 0,
                validator: None,
            },
        }
    }
    
    /// Start building a struct tracer
    pub fn struct_type(name: &'static str) -> Self {
        Self {
            tracer_type: TracerType::Struct {
                field_offsets: Vec::new(),
                validators: Vec::new(),
                name,
            },
        }
    }
    
    /// Start building a closure tracer
    pub fn closure() -> Self {
        Self {
            tracer_type: TracerType::Closure {
                capture_offsets: Vec::new(),
                capture_types: Vec::new(),
                signature: None,
            },
        }
    }
    
    /// Start building a conservative tracer
    pub fn conservative() -> Self {
        Self {
            tracer_type: TracerType::Conservative {
                address_validator: None,
            },
        }
    }
    
    /// Build the final tracer
    pub fn build(self) -> Arc<dyn ObjectTracer> {
        match self.tracer_type {
            TracerType::Array { element_size, element_offset, validator } => {
                if let Some(validator) = validator {
                    Arc::new(ArrayTracer::with_validator(element_size, element_offset, validator))
                } else {
                    Arc::new(ArrayTracer::new(element_size, element_offset))
                }
            }
            TracerType::Struct { field_offsets, validators, name } => {
                if validators.iter().any(|v| v.is_some()) {
                    Arc::new(StructTracer::with_validators(field_offsets, validators))
                } else {
                    Arc::new(StructTracer::named(field_offsets, name))
                }
            }
            TracerType::Closure { capture_offsets, capture_types, signature } => {
                if let Some(signature) = signature {
                    Arc::new(ClosureTracer::with_signature(capture_offsets, capture_types, signature))
                } else if !capture_types.is_empty() {
                    Arc::new(ClosureTracer::with_types(capture_offsets, capture_types))
                } else {
                    Arc::new(ClosureTracer::new(capture_offsets))
                }
            }
            TracerType::Conservative { address_validator } => {
                if let Some(validator) = address_validator {
                    Arc::new(PrismObjectTracer::with_validator(validator))
                } else {
                    Arc::new(PrismObjectTracer::new())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    
    #[test]
    fn test_array_tracer() {
        let tracer = ArrayTracer::new(8, 0); // Array of 8-byte pointers at offset 0
        
        // Create a mock object with header + 2 pointers
        let mut mock_object = vec![0u8; mem::size_of::<ObjectHeader>() + 16];
        let object_ptr = mock_object.as_ptr();
        
        // Set up object header
        unsafe {
            let header = &mut *(object_ptr as *mut ObjectHeader);
            header.size = mock_object.len();
        }
        
        // Set up array data (2 pointers)
        unsafe {
            let data_ptr = object_ptr.add(mem::size_of::<ObjectHeader>());
            *(data_ptr as *mut *const u8) = 0x1000 as *const u8; // First pointer
            *(data_ptr.add(8) as *mut *const u8) = 0x2000 as *const u8; // Second pointer
        }
        
        unsafe {
            let references = tracer.trace_references(object_ptr, mock_object.len());
            assert_eq!(references.len(), 2);
            assert_eq!(references[0] as usize, 0x1000);
            assert_eq!(references[1] as usize, 0x2000);
        }
    }
    
    #[test]
    fn test_struct_tracer() {
        let tracer = StructTracer::named(vec![0, 16], "TestStruct"); // Two pointers at offsets 0 and 16
        
        // Create a mock object with header + struct data
        let mut mock_object = vec![0u8; mem::size_of::<ObjectHeader>() + 32];
        let object_ptr = mock_object.as_ptr();
        
        // Set up object header
        unsafe {
            let header = &mut *(object_ptr as *mut ObjectHeader);
            header.size = mock_object.len();
        }
        
        // Set up struct data
        unsafe {
            let data_ptr = object_ptr.add(mem::size_of::<ObjectHeader>());
            *(data_ptr as *mut *const u8) = 0x3000 as *const u8; // First field
            *(data_ptr.add(16) as *mut *const u8) = 0x4000 as *const u8; // Second field
        }
        
        unsafe {
            let references = tracer.trace_references(object_ptr, mock_object.len());
            assert_eq!(references.len(), 2);
            assert_eq!(references[0] as usize, 0x3000);
            assert_eq!(references[1] as usize, 0x4000);
        }
    }
    
    #[test]
    fn test_tracer_factory() {
        let array_tracer = TracerFactory::array_of_pointers(8);
        assert_eq!(array_tracer.tracer_name(), "ArrayTracer");
        
        let struct_tracer = TracerFactory::named_struct(vec![0, 8], "TestStruct");
        assert_eq!(struct_tracer.tracer_name(), "TestStruct");
        
        let conservative_tracer = TracerFactory::conservative();
        assert_eq!(conservative_tracer.tracer_name(), "PrismConservativeTracer");
    }
    
    #[test]
    fn test_tracer_builder() {
        let tracer = TracerBuilder::array(8).build();
        assert_eq!(tracer.tracer_name(), "ArrayTracer");
        
        let struct_tracer = TracerBuilder::struct_type("BuilderStruct").build();
        assert_eq!(struct_tracer.tracer_name(), "BuilderStruct");
    }
} 