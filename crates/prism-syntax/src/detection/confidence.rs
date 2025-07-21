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
            config: ScoringConfig::default(),
        }
    }
    
    /// Score confidence for each syntax style based on evidence
    pub fn score_styles(&self, evidence: &[SyntaxEvidence]) -> HashMap<SyntaxStyle, f64> {
        let mut scores = HashMap::new();
        
        // Initialize base scores
        for style in [SyntaxStyle::CLike, SyntaxStyle::PythonLike, SyntaxStyle::RustLike, SyntaxStyle::Canonical] {
            scores.insert(style, self.config.base_confidence);
        }
        
        if evidence.is_empty() {
            return scores;
        }
        
        // Group evidence by style
        let mut style_evidence: HashMap<SyntaxStyle, Vec<&SyntaxEvidence>> = HashMap::new();
        for ev in evidence {
            style_evidence.entry(ev.style).or_default().push(ev);
        }
        
        // Calculate scores for each style
        for (style, style_ev) in style_evidence {
            let score = self.calculate_style_score(&style_ev);
            scores.insert(style, score);
        }
        
        // Apply conflict penalties
        self.apply_conflict_penalties(&mut scores, evidence);
        
        // Normalize scores to ensure they sum to a reasonable range
        self.normalize_scores(&mut scores);
        
        scores
    }
    
    /// Calculate detailed confidence information
    pub fn calculate_confidence(&self, evidence: &[SyntaxEvidence], style: SyntaxStyle) -> DetectionConfidence {
        let style_evidence: Vec<_> = evidence.iter().filter(|e| e.style == style).collect();
        let score = self.calculate_style_score(&style_evidence);
        let level = Self::classify_confidence(score);
        
        let mut evidence_breakdown = HashMap::new();
        let mut confidence_factors = Vec::new();
        
        // Breakdown by pattern
        for ev in &style_evidence {
            evidence_breakdown.insert(ev.pattern.clone(), ev.weight);
            confidence_factors.push(ConfidenceFactor {
                name: ev.pattern.clone(),
                impact: ev.weight,
                description: ev.description.clone(),
            });
        }
        
        // Add conflict factors
        let conflicting_evidence: Vec<_> = evidence.iter()
            .filter(|e| e.style != style)
            .collect();
            
        if !conflicting_evidence.is_empty() {
            let conflict_weight: f64 = conflicting_evidence.iter().map(|e| e.weight).sum();
            confidence_factors.push(ConfidenceFactor {
                name: "conflicting_evidence".to_string(),
                impact: -conflict_weight * self.config.conflict_penalty,
                description: format!("Conflicting evidence from other styles (weight: {:.2})", conflict_weight),
            });
        }
        
        DetectionConfidence {
            score,
            level,
            evidence_breakdown,
            confidence_factors,
        }
    }
    
    /// Calculate confidence score for a specific style
    fn calculate_style_score(&self, evidence: &[&SyntaxEvidence]) -> f64 {
        if evidence.is_empty() {
            return self.config.base_confidence;
        }
        
        let mut total_weight = 0.0;
        let mut evidence_count = 0;
        
        for ev in evidence {
            let multiplier = self.config.evidence_multipliers
                .get(&ev.pattern)
                .copied()
                .unwrap_or(1.0);
                
            total_weight += ev.weight * multiplier;
            evidence_count += 1;
        }
        
        // Base score plus weighted evidence, with diminishing returns
        let evidence_score = total_weight / (1.0 + total_weight * 0.1);
        let diversity_bonus = if evidence_count > 1 { 0.1 } else { 0.0 };
        
        (self.config.base_confidence + evidence_score + diversity_bonus).min(1.0)
    }
    
    /// Apply penalties for conflicting evidence
    fn apply_conflict_penalties(&self, scores: &mut HashMap<SyntaxStyle, f64>, evidence: &[SyntaxEvidence]) {
        // Calculate total evidence weight for each style
        let mut style_weights: HashMap<SyntaxStyle, f64> = HashMap::new();
        for ev in evidence {
            *style_weights.entry(ev.style).or_default() += ev.weight;
        }
        
        // Apply conflict penalty based on competing evidence
        for (style, score) in scores.iter_mut() {
            let style_weight = style_weights.get(style).copied().unwrap_or(0.0);
            let competing_weight: f64 = style_weights.iter()
                .filter(|(s, _)| *s != style)
                .map(|(_, w)| *w)
                .sum();
                
            if competing_weight > 0.0 {
                let penalty = competing_weight * self.config.conflict_penalty;
                *score = (*score - penalty).max(0.0);
            }
        }
    }
    
    /// Normalize scores to prevent extreme values
    fn normalize_scores(&self, scores: &mut HashMap<SyntaxStyle, f64>) {
        let max_score = scores.values().fold(0.0f64, |a, &b| a.max(b));
        
        // If all scores are very low, boost the highest one
        if max_score < 0.2 {
            if let Some((best_style, _)) = scores.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                let best_style = *best_style;
                scores.insert(best_style, 0.3); // Minimum reasonable confidence
            }
        }
        
        // Ensure scores don't exceed 1.0
        for score in scores.values_mut() {
            *score = score.min(1.0);
        }
    }
    
    /// Classify numeric confidence into levels
    fn classify_confidence(score: f64) -> ConfidenceLevel {
        match score {
            s if s < 0.3 => ConfidenceLevel::VeryLow,
            s if s < 0.5 => ConfidenceLevel::Low,
            s if s < 0.7 => ConfidenceLevel::Medium,
            s if s < 0.9 => ConfidenceLevel::High,
            _ => ConfidenceLevel::VeryHigh,
        }
    }
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self::new()
    }
} 