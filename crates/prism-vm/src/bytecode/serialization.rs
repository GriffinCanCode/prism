//! Bytecode Serialization and Deserialization
//!
//! This module handles the serialization and deserialization of Prism bytecode
//! to and from binary format for storage and transmission.

use super::{PrismBytecode, PRISM_BYTECODE_MAGIC, BYTECODE_FORMAT_VERSION};
use crate::{VMResult, PrismVMError, VMVersion};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Bytecode serializer
pub struct BytecodeSerializer;

/// Bytecode deserializer
pub struct BytecodeDeserializer;

/// Serialization format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationOptions {
    /// Include debug information
    pub include_debug_info: bool,
    /// Compress the bytecode
    pub compress: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Include AI metadata
    pub include_ai_metadata: bool,
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self {
            include_debug_info: true,
            compress: true,
            compression_level: 6,
            include_ai_metadata: true,
        }
    }
}

impl BytecodeSerializer {
    /// Create a new serializer
    pub fn new() -> Self {
        Self
    }

    /// Serialize bytecode to bytes
    pub fn serialize(&self, bytecode: &PrismBytecode, options: &SerializationOptions) -> VMResult<Vec<u8>> {
        // Validate bytecode before serialization
        bytecode.validate()?;

        // Create a copy of bytecode with optional components removed based on options
        let mut bytecode_to_serialize = bytecode.clone();
        
        if !options.include_debug_info {
            bytecode_to_serialize.debug_info = None;
            // Also remove debug info from functions
            for function in &mut bytecode_to_serialize.functions {
                function.debug_info = None;
            }
        }

        if !options.include_ai_metadata {
            bytecode_to_serialize.metadata.ai_metadata = None;
        }

        // Serialize using bincode
        let serialized = bincode::serialize(&bytecode_to_serialize)
            .map_err(|e| PrismVMError::SerializationError { source: e })?;

        // Apply compression if requested
        if options.compress {
            self.compress_data(&serialized, options.compression_level)
        } else {
            Ok(serialized)
        }
    }

    /// Serialize bytecode to a writer
    pub fn serialize_to_writer<W: Write>(
        &self,
        bytecode: &PrismBytecode,
        writer: &mut W,
        options: &SerializationOptions,
    ) -> VMResult<()> {
        let serialized = self.serialize(bytecode, options)?;
        writer.write_all(&serialized)
            .map_err(|e| PrismVMError::IOError { source: e })?;
        Ok(())
    }

    /// Serialize bytecode to a file
    pub fn serialize_to_file<P: AsRef<std::path::Path>>(
        &self,
        bytecode: &PrismBytecode,
        path: P,
        options: &SerializationOptions,
    ) -> VMResult<()> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| PrismVMError::IOError { source: e })?;
        self.serialize_to_writer(bytecode, &mut file, options)
    }

    /// Compress data using a simple compression algorithm
    fn compress_data(&self, data: &[u8], _level: u8) -> VMResult<Vec<u8>> {
        // For now, we'll just return the data as-is
        // In a real implementation, we'd use a compression library like zlib or lz4
        Ok(data.to_vec())
    }
}

impl Default for BytecodeSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl BytecodeDeserializer {
    /// Create a new deserializer
    pub fn new() -> Self {
        Self
    }

    /// Deserialize bytecode from bytes
    pub fn deserialize(&self, data: &[u8]) -> VMResult<PrismBytecode> {
        // Try to decompress first
        let decompressed_data = self.decompress_data(data)?;
        
        // Deserialize using bincode
        let bytecode: PrismBytecode = bincode::deserialize(&decompressed_data)
            .map_err(|e| PrismVMError::SerializationError { source: e })?;

        // Validate the deserialized bytecode
        self.validate_deserialized_bytecode(&bytecode)?;

        Ok(bytecode)
    }

    /// Deserialize bytecode from a reader
    pub fn deserialize_from_reader<R: Read>(&self, reader: &mut R) -> VMResult<PrismBytecode> {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)
            .map_err(|e| PrismVMError::IOError { source: e })?;
        self.deserialize(&buffer)
    }

    /// Deserialize bytecode from a file
    pub fn deserialize_from_file<P: AsRef<std::path::Path>>(&self, path: P) -> VMResult<PrismBytecode> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| PrismVMError::IOError { source: e })?;
        self.deserialize_from_reader(&mut file)
    }

    /// Validate deserialized bytecode
    fn validate_deserialized_bytecode(&self, bytecode: &PrismBytecode) -> VMResult<()> {
        // Check magic number
        if bytecode.magic != PRISM_BYTECODE_MAGIC {
            return Err(PrismVMError::InvalidBytecode {
                message: format!("Invalid magic number: expected 0x{:08X}, got 0x{:08X}", 
                    PRISM_BYTECODE_MAGIC, bytecode.magic),
            });
        }

        // Check format version
        if bytecode.format_version > BYTECODE_FORMAT_VERSION {
            return Err(PrismVMError::InvalidBytecode {
                message: format!("Unsupported bytecode format version: {} (max supported: {})", 
                    bytecode.format_version, BYTECODE_FORMAT_VERSION),
            });
        }

        // Check VM version compatibility
        if !VMVersion::CURRENT.is_compatible_with(&bytecode.vm_version) {
            return Err(PrismVMError::InvalidBytecode {
                message: format!("Incompatible VM version: {} (current: {})", 
                    bytecode.vm_version, VMVersion::CURRENT),
            });
        }

        // Perform full validation
        bytecode.validate()?;

        Ok(())
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> VMResult<Vec<u8>> {
        // For now, we'll just return the data as-is
        // In a real implementation, we'd detect compression and decompress accordingly
        Ok(data.to_vec())
    }

    /// Peek at bytecode metadata without full deserialization
    pub fn peek_metadata(&self, data: &[u8]) -> VMResult<BytecodeMetadata> {
        // This is a simplified implementation that deserializes the whole thing
        // In a real implementation, we'd have a more efficient way to read just the header
        let bytecode = self.deserialize(data)?;
        
        Ok(BytecodeMetadata {
            magic: bytecode.magic,
            format_version: bytecode.format_version,
            vm_version: bytecode.vm_version,
            id: bytecode.id,
            module_name: bytecode.metadata.name,
            module_version: bytecode.metadata.version,
            compiled_at: bytecode.metadata.compiled_at,
            function_count: bytecode.functions.len(),
            type_count: bytecode.types.len(),
            constant_count: bytecode.constants.len(),
            has_debug_info: bytecode.debug_info.is_some(),
            has_ai_metadata: bytecode.metadata.ai_metadata.is_some(),
        })
    }
}

impl Default for BytecodeDeserializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bytecode metadata that can be read without full deserialization
#[derive(Debug, Clone)]
pub struct BytecodeMetadata {
    /// Magic number
    pub magic: u32,
    /// Format version
    pub format_version: u16,
    /// VM version
    pub vm_version: VMVersion,
    /// Bytecode ID
    pub id: uuid::Uuid,
    /// Module name
    pub module_name: String,
    /// Module version
    pub module_version: String,
    /// Compilation timestamp
    pub compiled_at: chrono::DateTime<chrono::Utc>,
    /// Number of functions
    pub function_count: usize,
    /// Number of types
    pub type_count: usize,
    /// Number of constants
    pub constant_count: usize,
    /// Whether debug info is present
    pub has_debug_info: bool,
    /// Whether AI metadata is present
    pub has_ai_metadata: bool,
}

/// Utility functions for bytecode serialization
pub mod utils {
    use super::*;

    /// Serialize bytecode with default options
    pub fn serialize_bytecode(bytecode: &PrismBytecode) -> VMResult<Vec<u8>> {
        let serializer = BytecodeSerializer::new();
        let options = SerializationOptions::default();
        serializer.serialize(bytecode, &options)
    }

    /// Deserialize bytecode
    pub fn deserialize_bytecode(data: &[u8]) -> VMResult<PrismBytecode> {
        let deserializer = BytecodeDeserializer::new();
        deserializer.deserialize(data)
    }

    /// Save bytecode to file with default options
    pub fn save_bytecode_to_file<P: AsRef<std::path::Path>>(
        bytecode: &PrismBytecode,
        path: P,
    ) -> VMResult<()> {
        let serializer = BytecodeSerializer::new();
        let options = SerializationOptions::default();
        serializer.serialize_to_file(bytecode, path, &options)
    }

    /// Load bytecode from file
    pub fn load_bytecode_from_file<P: AsRef<std::path::Path>>(path: P) -> VMResult<PrismBytecode> {
        let deserializer = BytecodeDeserializer::new();
        deserializer.deserialize_from_file(path)
    }

    /// Get bytecode metadata from file without full loading
    pub fn peek_bytecode_metadata<P: AsRef<std::path::Path>>(path: P) -> VMResult<BytecodeMetadata> {
        let data = std::fs::read(path)
            .map_err(|e| PrismVMError::IOError { source: e })?;
        let deserializer = BytecodeDeserializer::new();
        deserializer.peek_metadata(&data)
    }

    /// Check if a file contains valid Prism bytecode
    pub fn is_valid_bytecode_file<P: AsRef<std::path::Path>>(path: P) -> bool {
        match peek_bytecode_metadata(path) {
            Ok(metadata) => metadata.magic == PRISM_BYTECODE_MAGIC,
            Err(_) => false,
        }
    }

    /// Get the size of serialized bytecode
    pub fn get_bytecode_size(bytecode: &PrismBytecode) -> VMResult<usize> {
        let serialized = serialize_bytecode(bytecode)?;
        Ok(serialized.len())
    }

    /// Compare two bytecode files for equality
    pub fn bytecode_files_equal<P1: AsRef<std::path::Path>, P2: AsRef<std::path::Path>>(
        path1: P1,
        path2: P2,
    ) -> VMResult<bool> {
        let bytecode1 = load_bytecode_from_file(path1)?;
        let bytecode2 = load_bytecode_from_file(path2)?;
        
        // Compare IDs (unique identifiers)
        Ok(bytecode1.id == bytecode2.id)
    }
} 