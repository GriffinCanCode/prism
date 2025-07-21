//! Confidence scoring for syntax style detection.
//!
//! This module implements confidence calculation for syntax detection results,
//! maintaining conceptual cohesion around "confidence assessment and style scoring".

use super::{detector::SyntaxEvidence, SyntaxStyle};
use std::collections::HashMap;

/// Confidence scorer for syntax detection
#[derive(Debug)]
pub struct ConfidenceScorer {
    /// Scoring configuration
    config: ScoringConfig,
}

/// Configuration for confidence scoring
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Base confidence for each style
    pub base_confidence: f64,
    
    /// Evidence weight multipliers
    pub evidence_multipliers: HashMap<String, f64>,
    
    /// Minimum evidence threshold
    pub min_evidence_threshold: usize,
    
    /// Penalty for conflicting evidence
    pub conflict_penalty: f64,
}

/// Confidence level classification
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceLevel {
    /// Very low confidence (0.0 - 0.3)
    VeryLow,
    /// Low confidence (0.3 - 0.5)
    Low,
    /// Medium confidence (0.5 - 0.7)
    Medium,
    /// High confidence (0.7 - 0.9)
    High,
    /// Very high confidence (0.9 - 1.0)
    VeryHigh,
}

/// Detailed confidence information
#[derive(Debug, Clone)]
pub struct DetectionConfidence {
    /// Numeric confidence score
    pub score: f64,
    
    /// Confidence level classification
    pub level: ConfidenceLevel,
    
    /// Breakdown by evidence type
    pub evidence_breakdown: HashMap<String, f64>,
    
    /// Factors affecting confidence
    pub confidence_factors: Vec<ConfidenceFactor>,
}

/// Factor affecting confidence calculation
#[derive(Debug, Clone)]
pub struct ConfidenceFactor {
    /// Name of the factor
    pub name: String,
    
    /// Impact on confidence (positive or negative)
    pub impact: f64,
    
    /// Description of the factor
    pub description: String,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        let mut evidence_multipliers = HashMap::new();
        
        // Keyword evidence is highly reliable
        evidence_multipliers.insert("rust_fn_keyword".to_string(), 1.2);
        evidence_multipliers.insert("canonical_function".to_string(), 1.1);
        
        // Punctuation evidence is moderately reliable
        evidence_multipliers.insert("c_like_braces".to_string(), 1.0);
        evidence_multipliers.insert("python_colons".to_string(), 1.1);
        
        // Structural evidence varies in reliability
        evidence_multipliers.insert("python_indentation".to_string(), 0.8);
        
        Self {
            base_confidence: 0.05, // Low base to require evidence
            evidence_multipliers,
            min_evidence_threshold: 1,
            conflict_penalty: 0.3,
        }
    }
}

impl ConfidenceScorer {
    /// Create a new confidence scorer
    pub fn new() -> Self {
        Self {
            scoring_algorithm: ScoringAlgorithm::Weighted,
            min_confidence_threshold: 0.6,
            max_alternatives: 3,
        }
    }
    
    /// Calculate confidence scores and produce final detection result
    pub fn calculate_confidence(&self, evidence: &[SyntaxEvidence]) -> Result<DetectionResult, DetectionError> {
        if evidence.is_empty() {
            return Ok(DetectionResult {
                detected_style: SyntaxStyle::Canonical, // Default fallback
                confidence: 0.5,
                alternative_styles: Vec::new(),
                detection_metadata: std::collections::HashMap::new(),
            });
        }
        
        // Group evidence by syntax style
        let mut style_scores: std::collections::HashMap<SyntaxStyle, Vec<f64>> = std::collections::HashMap::new();
        
        for evidence_item in evidence {
            style_scores
                .entry(evidence_item.style)
                .or_insert_with(Vec::new)
                .push(evidence_item.confidence);
        }
        
        // Calculate aggregate scores for each style
        let mut final_scores: std::collections::HashMap<SyntaxStyle, f64> = std::collections::HashMap::new();
        
        for (style, scores) in style_scores {
            let aggregate_score = match self.scoring_algorithm {
                ScoringAlgorithm::Weighted => self.calculate_weighted_score(&scores),
                ScoringAlgorithm::Bayesian => self.calculate_bayesian_score(&scores),
                ScoringAlgorithm::MaxConfidence => self.calculate_max_confidence_score(&scores),
            };
            
            final_scores.insert(style, aggregate_score);
        }
        
        // Find the highest scoring style
        let (detected_style, confidence) = final_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&style, &conf)| (style, conf))
            .unwrap_or((SyntaxStyle::Canonical, 0.5));
        
        // Generate alternative styles
        let mut alternatives: Vec<SyntaxStyle> = final_scores
            .iter()
            .filter(|(&style, &conf)| style != detected_style && conf > 0.3)
            .map(|(&style, _)| style)
            .collect();
        
        // Sort alternatives by confidence (descending)
        alternatives.sort_by(|&a, &b| {
            let conf_a = final_scores.get(&a).unwrap_or(&0.0);
            let conf_b = final_scores.get(&b).unwrap_or(&0.0);
            conf_b.partial_cmp(conf_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit number of alternatives
        alternatives.truncate(self.max_alternatives);
        
        // Create detection metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("evidence_count".to_string(), evidence.len().to_string());
        metadata.insert("scoring_algorithm".to_string(), format!("{:?}", self.scoring_algorithm));
        
        for (style, score) in &final_scores {
            metadata.insert(format!("score_{:?}", style).to_lowercase(), format!("{:.3}", score));
        }
        
        Ok(DetectionResult {
            detected_style,
            confidence,
            alternative_styles: alternatives,
            detection_metadata: metadata,
        })
    }
    
    /// Calculate weighted average of confidence scores
    fn calculate_weighted_score(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        
        // Weight higher scores more heavily
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for &score in scores {
            let weight = score * score; // Quadratic weighting favors high confidence
            weighted_sum += score * weight;
            weight_sum += weight;
        }
        
        if weight_sum > 0.0 {
            (weighted_sum / weight_sum).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate Bayesian-style score (simplified)
    fn calculate_bayesian_score(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        
        // Simplified Bayesian approach: combine probabilities
        let mut combined_prob = 0.5; // Prior probability
        
        for &score in scores {
            // Update probability using Bayes' theorem (simplified)
            combined_prob = (combined_prob * score) / 
                (combined_prob * score + (1.0 - combined_prob) * (1.0 - score));
        }
        
        combined_prob.min(1.0)
    }
    
    /// Calculate maximum confidence score
    fn calculate_max_confidence_score(&self, scores: &[f64]) -> f64 {
        scores.iter().fold(0.0, |acc, &x| acc.max(x))
    }
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
} 