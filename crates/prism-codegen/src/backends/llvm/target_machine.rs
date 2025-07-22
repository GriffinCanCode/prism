//! LLVM Target Machine Configuration
//!
//! This module handles LLVM target machine setup, configuration,
//! and management for different architectures and optimization levels.

use super::{LLVMResult, LLVMError};
use super::types::{LLVMTargetArch, LLVMOptimizationLevel};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// LLVM target machine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMTargetConfig {
    /// Target architecture
    pub target_arch: LLVMTargetArch,
    /// Target CPU (specific processor model)
    pub target_cpu: String,
    /// Target features (CPU extensions, etc.)
    pub target_features: Vec<String>,
    /// Optimization level
    pub optimization_level: LLVMOptimizationLevel,
    /// Code model
    pub code_model: LLVMCodeModel,
    /// Relocation model
    pub relocation_model: LLVMRelocationModel,
    /// Enable position-independent code
    pub position_independent: bool,
    /// Enable fast math optimizations
    pub fast_math: bool,
    /// Enable link-time optimization
    pub lto: bool,
    /// Custom target attributes
    pub custom_attributes: HashMap<String, String>,
}

impl Default for LLVMTargetConfig {
    fn default() -> Self {
        Self {
            target_arch: LLVMTargetArch::default(),
            target_cpu: "generic".to_string(),
            target_features: Vec::new(),
            optimization_level: LLVMOptimizationLevel::Aggressive,
            code_model: LLVMCodeModel::Default,
            relocation_model: LLVMRelocationModel::PIC,
            position_independent: true,
            fast_math: false,
            lto: false,
            custom_attributes: HashMap::new(),
        }
    }
}

/// LLVM code models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLVMCodeModel {
    /// Default code model
    Default,
    /// Small code model (32-bit offsets)
    Small,
    /// Kernel code model
    Kernel,
    /// Medium code model
    Medium,
    /// Large code model
    Large,
}

impl Default for LLVMCodeModel {
    fn default() -> Self {
        Self::Default
    }
}

/// LLVM relocation models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLVMRelocationModel {
    /// Static relocation model
    Static,
    /// Position Independent Code
    PIC,
    /// Dynamic, no position independent code
    DynamicNoPIC,
    /// Read-only position independent code
    ROPI,
    /// Read-write position independent code
    RWPI,
    /// Read-only and read-write position independent code
    ROPI_RWPI,
}

impl Default for LLVMRelocationModel {
    fn default() -> Self {
        Self::PIC
    }
}

/// LLVM target machine manager
pub struct LLVMTargetMachine {
    /// Target configuration
    config: LLVMTargetConfig,
    /// Target triple string
    target_triple: String,
    /// Data layout string
    data_layout: String,
    /// CPU-specific features
    cpu_features: HashMap<String, bool>,
    /// Architecture-specific optimizations
    arch_optimizations: Vec<String>,
}

impl LLVMTargetMachine {
    /// Create a new target machine with the given configuration
    pub fn new(config: LLVMTargetConfig) -> LLVMResult<Self> {
        let target_triple = Self::generate_target_triple(&config)?;
        let data_layout = Self::generate_data_layout(&config)?;
        let cpu_features = Self::detect_cpu_features(&config)?;
        let arch_optimizations = Self::get_arch_optimizations(&config);

        Ok(Self {
            config,
            target_triple,
            data_layout,
            cpu_features,
            arch_optimizations,
        })
    }

    /// Generate target triple string
    fn generate_target_triple(config: &LLVMTargetConfig) -> LLVMResult<String> {
        let base_triple = config.target_arch.to_string();
        
        // Add any custom modifications to the triple based on configuration
        Ok(base_triple)
    }

    /// Generate data layout string for the target
    fn generate_data_layout(config: &LLVMTargetConfig) -> LLVMResult<String> {
        match config.target_arch {
            LLVMTargetArch::X86_64 => {
                Ok("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128".to_string())
            }
            LLVMTargetArch::AArch64 => {
                Ok("e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string())
            }
            LLVMTargetArch::RISCV64 => {
                Ok("e-m:e-p:64:64-i64:64-i128:128-n64-S128".to_string())
            }
            LLVMTargetArch::ARM => {
                Ok("e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string())
            }
            LLVMTargetArch::WebAssembly => {
                Ok("e-m:e-p:32:32-i64:64-n32:64-S128".to_string())
            }
            LLVMTargetArch::PowerPC64 => {
                Ok("e-m:e-i64:64-n32:64-S128-v256:256:256-v512:512:512".to_string())
            }
            LLVMTargetArch::MIPS64 => {
                Ok("e-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".to_string())
            }
        }
    }

    /// Detect available CPU features for the target
    fn detect_cpu_features(config: &LLVMTargetConfig) -> LLVMResult<HashMap<String, bool>> {
        let mut features = HashMap::new();
        
        // Add architecture-specific feature detection
        match config.target_arch {
            LLVMTargetArch::X86_64 => {
                // Common x86-64 features
                features.insert("sse".to_string(), true);
                features.insert("sse2".to_string(), true);
                features.insert("sse3".to_string(), true);
                features.insert("ssse3".to_string(), true);
                features.insert("sse4.1".to_string(), true);
                features.insert("sse4.2".to_string(), true);
                
                // Modern features (may not be available on all CPUs)
                features.insert("avx".to_string(), false);
                features.insert("avx2".to_string(), false);
                features.insert("avx512f".to_string(), false);
                features.insert("popcnt".to_string(), true);
                features.insert("bmi1".to_string(), false);
                features.insert("bmi2".to_string(), false);
            }
            LLVMTargetArch::AArch64 => {
                // ARM64 features
                features.insert("neon".to_string(), true);
                features.insert("fp-armv8".to_string(), true);
                features.insert("crypto".to_string(), false);
                features.insert("crc".to_string(), false);
            }
            LLVMTargetArch::RISCV64 => {
                // RISC-V extensions
                features.insert("m".to_string(), true); // Multiplication
                features.insert("a".to_string(), true); // Atomics
                features.insert("f".to_string(), true); // Single-precision floating-point
                features.insert("d".to_string(), true); // Double-precision floating-point
                features.insert("c".to_string(), true); // Compressed instructions
            }
            _ => {
                // Default features for other architectures
            }
        }
        
        // Apply user-specified features
        for feature in &config.target_features {
            let (enable, feature_name) = if feature.starts_with('+') {
                (true, &feature[1..])
            } else if feature.starts_with('-') {
                (false, &feature[1..])
            } else {
                (true, feature.as_str())
            };
            features.insert(feature_name.to_string(), enable);
        }
        
        Ok(features)
    }

    /// Get architecture-specific optimization passes
    fn get_arch_optimizations(config: &LLVMTargetConfig) -> Vec<String> {
        let mut optimizations = Vec::new();
        
        match config.target_arch {
            LLVMTargetArch::X86_64 => {
                optimizations.push("x86-promote-alloca-to-vector".to_string());
                optimizations.push("x86-domain-reassignment".to_string());
                if config.optimization_level != LLVMOptimizationLevel::None {
                    optimizations.push("x86-optimize-LEAs".to_string());
                    optimizations.push("x86-avoid-SFB".to_string());
                }
            }
            LLVMTargetArch::AArch64 => {
                optimizations.push("aarch64-promote-const".to_string());
                optimizations.push("aarch64-simd-scalar".to_string());
                if config.optimization_level != LLVMOptimizationLevel::None {
                    optimizations.push("aarch64-ccmp".to_string());
                    optimizations.push("aarch64-condopt".to_string());
                }
            }
            LLVMTargetArch::RISCV64 => {
                optimizations.push("riscv-merge-base-offset".to_string());
                if config.optimization_level != LLVMOptimizationLevel::None {
                    optimizations.push("riscv-make-compressible".to_string());
                }
            }
            _ => {
                // Generic optimizations for other architectures
            }
        }
        
        optimizations
    }

    /// Get the target triple string
    pub fn get_target_triple(&self) -> &str {
        &self.target_triple
    }

    /// Get the data layout string
    pub fn get_data_layout(&self) -> &str {
        &self.data_layout
    }

    /// Get the target configuration
    pub fn get_config(&self) -> &LLVMTargetConfig {
        &self.config
    }

    /// Check if a CPU feature is enabled
    pub fn has_feature(&self, feature: &str) -> bool {
        self.cpu_features.get(feature).copied().unwrap_or(false)
    }

    /// Get all enabled CPU features
    pub fn get_enabled_features(&self) -> Vec<String> {
        self.cpu_features.iter()
            .filter_map(|(name, &enabled)| if enabled { Some(name.clone()) } else { None })
            .collect()
    }

    /// Get architecture-specific optimization passes
    pub fn get_optimization_passes(&self) -> &[String] {
        &self.arch_optimizations
    }

    /// Generate LLVM target machine attributes
    pub fn generate_target_attributes(&self) -> HashMap<String, String> {
        let mut attributes = HashMap::new();
        
        // Basic target information
        attributes.insert("target-cpu".to_string(), self.config.target_cpu.clone());
        
        // Feature string
        let feature_string = self.cpu_features.iter()
            .map(|(name, &enabled)| {
                if enabled {
                    format!("+{}", name)
                } else {
                    format!("-{}", name)
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        
        if !feature_string.is_empty() {
            attributes.insert("target-features".to_string(), feature_string);
        }
        
        // Code model
        match self.config.code_model {
            LLVMCodeModel::Small => attributes.insert("code-model".to_string(), "small".to_string()),
            LLVMCodeModel::Kernel => attributes.insert("code-model".to_string(), "kernel".to_string()),
            LLVMCodeModel::Medium => attributes.insert("code-model".to_string(), "medium".to_string()),
            LLVMCodeModel::Large => attributes.insert("code-model".to_string(), "large".to_string()),
            LLVMCodeModel::Default => None,
        };
        
        // Relocation model
        match self.config.relocation_model {
            LLVMRelocationModel::Static => attributes.insert("relocation-model".to_string(), "static".to_string()),
            LLVMRelocationModel::PIC => attributes.insert("relocation-model".to_string(), "pic".to_string()),
            LLVMRelocationModel::DynamicNoPIC => attributes.insert("relocation-model".to_string(), "dynamic-no-pic".to_string()),
            LLVMRelocationModel::ROPI => attributes.insert("relocation-model".to_string(), "ropi".to_string()),
            LLVMRelocationModel::RWPI => attributes.insert("relocation-model".to_string(), "rwpi".to_string()),
            LLVMRelocationModel::ROPI_RWPI => attributes.insert("relocation-model".to_string(), "ropi-rwpi".to_string()),
        };
        
        // Position independent code
        if self.config.position_independent {
            attributes.insert("position-independent".to_string(), "true".to_string());
        }
        
        // Fast math
        if self.config.fast_math {
            attributes.insert("unsafe-fp-math".to_string(), "true".to_string());
            attributes.insert("no-infs-fp-math".to_string(), "true".to_string());
            attributes.insert("no-nans-fp-math".to_string(), "true".to_string());
        }
        
        // Add custom attributes
        attributes.extend(self.config.custom_attributes.clone());
        
        attributes
    }

    /// Update target configuration
    pub fn update_config(&mut self, new_config: LLVMTargetConfig) -> LLVMResult<()> {
        self.config = new_config;
        self.target_triple = Self::generate_target_triple(&self.config)?;
        self.data_layout = Self::generate_data_layout(&self.config)?;
        self.cpu_features = Self::detect_cpu_features(&self.config)?;
        self.arch_optimizations = Self::get_arch_optimizations(&self.config);
        Ok(())
    }

    /// Create a target machine optimized for the current host
    pub fn for_host() -> LLVMResult<Self> {
        let mut config = LLVMTargetConfig::default();
        
        // Detect host architecture
        #[cfg(target_arch = "x86_64")]
        {
            config.target_arch = LLVMTargetArch::X86_64;
            config.target_cpu = "native".to_string();
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            config.target_arch = LLVMTargetArch::AArch64;
            config.target_cpu = "native".to_string();
        }
        
        #[cfg(target_arch = "riscv64")]
        {
            config.target_arch = LLVMTargetArch::RISCV64;
            config.target_cpu = "native".to_string();
        }
        
        Self::new(config)
    }
}

impl Clone for LLVMTargetMachine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            target_triple: self.target_triple.clone(),
            data_layout: self.data_layout.clone(),
            cpu_features: self.cpu_features.clone(),
            arch_optimizations: self.arch_optimizations.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_machine_creation() {
        let config = LLVMTargetConfig::default();
        let target_machine = LLVMTargetMachine::new(config).unwrap();
        
        assert!(!target_machine.get_target_triple().is_empty());
        assert!(!target_machine.get_data_layout().is_empty());
        assert_eq!(target_machine.get_config().target_arch, LLVMTargetArch::X86_64);
    }

    #[test]
    fn test_cpu_features() {
        let mut config = LLVMTargetConfig::default();
        config.target_arch = LLVMTargetArch::X86_64;
        config.target_features = vec!["+avx".to_string(), "-sse".to_string()];
        
        let target_machine = LLVMTargetMachine::new(config).unwrap();
        assert!(target_machine.has_feature("avx"));
        assert!(!target_machine.has_feature("sse"));
    }

    #[test]
    fn test_target_attributes() {
        let config = LLVMTargetConfig {
            target_cpu: "skylake".to_string(),
            fast_math: true,
            position_independent: true,
            ..LLVMTargetConfig::default()
        };
        
        let target_machine = LLVMTargetMachine::new(config).unwrap();
        let attributes = target_machine.generate_target_attributes();
        
        assert_eq!(attributes.get("target-cpu"), Some(&"skylake".to_string()));
        assert_eq!(attributes.get("unsafe-fp-math"), Some(&"true".to_string()));
        assert_eq!(attributes.get("position-independent"), Some(&"true".to_string()));
    }

    #[test]
    fn test_arch_optimizations() {
        let config = LLVMTargetConfig {
            target_arch: LLVMTargetArch::X86_64,
            optimization_level: LLVMOptimizationLevel::Aggressive,
            ..LLVMTargetConfig::default()
        };
        
        let target_machine = LLVMTargetMachine::new(config).unwrap();
        let optimizations = target_machine.get_optimization_passes();
        
        assert!(!optimizations.is_empty());
        assert!(optimizations.iter().any(|opt| opt.contains("x86")));
    }

    #[test]
    fn test_data_layout_generation() {
        let x86_config = LLVMTargetConfig {
            target_arch: LLVMTargetArch::X86_64,
            ..LLVMTargetConfig::default()
        };
        let x86_layout = LLVMTargetMachine::generate_data_layout(&x86_config).unwrap();
        assert!(x86_layout.contains("64:64"));
        
        let arm_config = LLVMTargetConfig {
            target_arch: LLVMTargetArch::AArch64,
            ..LLVMTargetConfig::default()
        };
        let arm_layout = LLVMTargetMachine::generate_data_layout(&arm_config).unwrap();
        assert!(arm_layout.contains("i128:128"));
    }
} 