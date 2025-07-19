//! Business Context - Domain Knowledge Preservation
//!
//! This module captures and preserves business domain knowledge and context
//! information throughout the PIR transformation process.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Business context information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Business domain
    pub domain: String,
    /// Business entities
    pub entities: Vec<BusinessEntity>,
    /// Business relationships
    pub relationships: Vec<BusinessRelationship>,
    /// Business constraints and rules
    pub constraints: Vec<BusinessConstraint>,
}

/// Business entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessEntity {
    /// Entity name
    pub name: String,
    /// Entity description
    pub description: String,
    /// Entity type
    pub entity_type: BusinessEntityType,
    /// Key attributes
    pub attributes: Vec<EntityAttribute>,
    /// Business rules
    pub rules: Vec<String>,
}

/// Business entity type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessEntityType {
    /// Core business object
    CoreObject,
    /// Value object
    ValueObject,
    /// Service
    Service,
    /// Repository
    Repository,
    /// Factory
    Factory,
    /// Event
    Event,
}

/// Entity attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAttribute {
    /// Attribute name
    pub name: String,
    /// Attribute description
    pub description: String,
    /// Data type
    pub data_type: String,
    /// Required flag
    pub required: bool,
    /// Validation rules
    pub validation_rules: Vec<String>,
}

/// Business relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRelationship {
    /// Source entity
    pub source_entity: String,
    /// Target entity
    pub target_entity: String,
    /// Relationship type
    pub relationship_type: BusinessRelationshipType,
    /// Cardinality
    pub cardinality: Cardinality,
    /// Description
    pub description: String,
}

/// Business relationship type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessRelationshipType {
    /// Association
    Association,
    /// Aggregation
    Aggregation,
    /// Composition
    Composition,
    /// Dependency
    Dependency,
    /// Generalization
    Generalization,
}

/// Cardinality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Cardinality {
    /// One to one
    OneToOne,
    /// One to many
    OneToMany,
    /// Many to one
    ManyToOne,
    /// Many to many
    ManyToMany,
}

/// Business constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint description
    pub description: String,
    /// Constraint type
    pub constraint_type: BusinessConstraintType,
    /// Affected entities
    pub affected_entities: Vec<String>,
    /// Enforcement level
    pub enforcement_level: EnforcementLevel,
}

/// Business constraint type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessConstraintType {
    /// Data integrity constraint
    DataIntegrity,
    /// Business rule constraint
    BusinessRule,
    /// Security constraint
    Security,
    /// Performance constraint
    Performance,
    /// Compliance constraint
    Compliance,
}

/// Enforcement level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced
    Mandatory,
    /// Should be enforced
    Recommended,
    /// May be enforced
    Optional,
}

/// Business rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule implementation
    pub implementation: Option<String>,
}

impl BusinessContext {
    /// Create a new empty business context
    pub fn new(domain: String) -> Self {
        Self {
            domain,
            entities: Vec::new(),
            relationships: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a business entity
    pub fn add_entity(&mut self, entity: BusinessEntity) {
        self.entities.push(entity);
    }

    /// Add a business relationship
    pub fn add_relationship(&mut self, relationship: BusinessRelationship) {
        self.relationships.push(relationship);
    }

    /// Add a business constraint
    pub fn add_constraint(&mut self, constraint: BusinessConstraint) {
        self.constraints.push(constraint);
    }

    /// Get entities by type
    pub fn get_entities_by_type(&self, entity_type: &BusinessEntityType) -> Vec<&BusinessEntity> {
        self.entities
            .iter()
            .filter(|entity| std::mem::discriminant(&entity.entity_type) == std::mem::discriminant(entity_type))
            .collect()
    }

    /// Get relationships involving an entity
    pub fn get_relationships_for_entity(&self, entity_name: &str) -> Vec<&BusinessRelationship> {
        self.relationships
            .iter()
            .filter(|rel| rel.source_entity == entity_name || rel.target_entity == entity_name)
            .collect()
    }
} 