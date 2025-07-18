# Prism: Conceptual Overview

Prism is an AI-First programming language designed from first principles for clarity, safety, and unparalleled comprehension by both human developers and artificial intelligence.

## Core Philosophy: Conceptual Cohesion

The fundamental principle of Prism is **"Conceptual Cohesion"**. The language's structure is designed to mirror a developer's mental model, ensuring that code related to a single, clear concept is grouped together. Instead of enforcing arbitrary rules like line counts, the Prism compiler will act as a guide, measuring how well code fits together and suggesting improvements.

This philosophy is built on several key pillars:

* **Clarity Over Brevity**: Prism prioritizes explicitness and descriptiveness over terse syntax.
* **Guidance Over Enforcement**: The compiler assists the developer in writing clear, cohesive code rather than rigidly enforcing arbitrary rules.
* **Self-Documenting Code**: Documentation, examples, and design intent are first-class citizens of the syntax itself.
* **Safety and Predictability**: The language is designed to prevent common errors through a strict, semantic type system and immutability by default.

## The Smart Module System

Prism rejects rigid file structures in favor of **"Smart Modules"**—a system where each file represents a single, cohesive business capability or concept.

### Structure of a Smart Module

A module file is organized into logical sections, which the compiler understands. This keeps related code together while maintaining clear separation of concerns.

* **Header**: An AI-friendly header block that defines the module's purpose and dependencies.
* **Sections**: Explicit blocks for different aspects of the concept, such as `types`, `validations`, and `operations`.

Here's what a Smart Module looks like:

```prism
// File: capabilities/UserManagement.prism
@capability "User Management"
@description "Handles user lifecycle operations like creation and authentication."
@dependencies ["Database", "Cryptography"]

module UserManagement {
    // Module-level documentation and AI hints
    @aiHint "This module is the single source of truth for user data."

    // Grouped type definitions for clarity
    section types {
        Email = String matching emailRegex;
        UserId = PositiveInteger;
        HashedPassword = String length 60;
    }

    // Grouped validation functions
    section validations {
        function isStrongPassword(password: String) returns Boolean {
            // Password strength rules
        }
    }

    // Grouped core business logic operations
    section operations {
        function registerUser(email: Email, password: String) returns Result<UserId> {
            // Main registration logic
        }
    }
}
```

### Compiler-Assisted Organization

Instead of hard limits, the compiler uses **Conceptual Cohesion Metrics** to guide developers.

* **Cohesion Score**: The compiler analyzes data flow, type affinity, and function inter-dependencies to calculate a "cohesion score".
* **Smart Suggestions**: If the score is low, indicating that a module contains multiple unrelated concepts, the compiler will suggest a refactoring, such as splitting the file.

## Key Language Features

### AI-First Design

Prism is designed to be immediately comprehensible to AI systems:

* **Rich Metadata**: Every piece of code can include structured metadata that explains its purpose, constraints, and relationships.
* **Semantic Type System**: Types carry meaning beyond just data structure—they express business rules and constraints.
* **Self-Documenting Syntax**: The language syntax itself conveys intent, reducing the need for separate documentation.

### Safety and Predictability

* **Immutability by Default**: Variables are immutable unless explicitly marked as mutable, preventing many common bugs.
* **Strict Type System**: The type system prevents runtime errors by catching issues at compile time.
* **Contract-Based Programming**: Functions can specify preconditions and postconditions that are verified at compile time.

### Progressive Low-Level Control

Prism provides **graduated access** to low-level operations while maintaining AI comprehension and safety:

* **Capability-Based Access**: Low-level operations require explicit capabilities, preventing accidental unsafe code.
* **Semantic Justification**: All unsafe operations must explain their purpose and alternatives considered.
* **Contextual Containment**: Low-level code stays conceptually cohesive with related high-level functionality.
* **Audit Trail Integration**: Every unsafe operation is tracked and reviewable for security compliance.

### Developer Experience

* **Descriptive Naming**: The language encourages (and tooling supports) ultra-descriptive names that make code self-explanatory.
* **Intelligent Error Messages**: Error messages explain not just what went wrong, but why it happened and how to fix it.
* **AI-Powered Tooling**: The development environment provides AI-assisted code completion, refactoring suggestions, and clarity analysis.

## Compilation Strategy

Prism uses a multi-target compilation approach:

1. **TypeScript Transpilation**: For rapid prototyping and immediate ecosystem access
2. **WebAssembly**: For portable, secure execution across platforms
3. **Native Code**: For maximum performance in production environments

This approach allows developers to start quickly while scaling to high-performance applications as needed.

## The Vision

Prism represents a new paradigm in programming language design—one that prioritizes human and AI comprehension, embraces verbose clarity over cryptic brevity, and uses intelligent tooling to guide developers toward better code organization.

The goal is not just to create another programming language, but to fundamentally rethink how we express computational ideas in a way that is immediately clear to both humans and artificial intelligence.

For detailed implementation plans, see [implementation-plan.md](implementation-plan.md).
