# Prism Style Guide (PSG) Index & Conventions

**Last Updated**: 2025-01-17  
**Status**: Living Document  
**Purpose**: Central index for all Prism Style Guide documents and coding conventions

## Philosophy

The Prism Style Guide embodies our core principle: **"Perfect Legibility for Human and AI Collaboration"**. Every convention is designed to maximize communication between developers, AI systems, and future maintainers while preserving the ability to write high-performance, low-level code when needed.

**Context-Based Approach**: We balance brevity and clarity through context-appropriate naming and documentation strategies, ensuring code is optimized for both human understanding and AI processing efficiency.

## Document Categories

### PSG - Prism Style Guide Documents
Coding conventions, naming standards, and syntactic patterns for writing idiomatic Prism code.

### PLD - Prism Language Design Documents  
High-level language features and design decisions (see [PLD-INDEX](../PLD/PLD-INDEX.md)).

### PLT - Prism Language Technical Documents
Implementation details and technical specifications (see [PLT-INDEX](../PLT/PLT-INDEX.md)).

### PEP - Prism Enhancement Proposals
Community-driven proposals for improvements (see [PEP-INDEX](../PEP/PEP-INDEX.md)).

## Document Status Levels

- **Draft**: Under active development and review
- **Accepted**: Approved and ready for adoption
- **Stable**: Battle-tested and unlikely to change
- **Deprecated**: Superseded by newer conventions

## Style Guide Index

### Core Conventions (PSG-001 to PSG-099)

| ID | Title | Status | Priority | Adoption Phase | Dependencies |
|----|-------|--------|----------|----------------|--------------|
| [PSG-001](./PSG-001.md) | Fundamental Syntax & Formatting | Draft | Core | 1 | None |
| [PSG-002](./PSG-002.md) | Naming Conventions & Identifiers | Draft | Core | 1 | PSG-001 |
| [PSG-003](./PSG-003.md) | PrismDoc Standards | Draft | Core | 1 | PSG-001, PSG-002, PLT-006 |
| PSG-004 | Function Design & Signatures | Planned | Core | 1 | PSG-001, PSG-002 |
| PSG-005 | Type Definitions & Annotations | Planned | Core | 1 | PSG-001, PSG-002, PLD-001 |
| PSG-006 | Error Handling Patterns | Planned | High | 1 | PSG-001, PSG-004 |
| PSG-007 | Module Organization & Structure | Planned | High | 1 | PSG-001, PSG-002 |
| PSG-008 | AI Context & Metadata | Planned | High | 1 | PSG-001, PSG-003 |

### Advanced Patterns (PSG-100 to PSG-199)

| ID | Title | Status | Priority | Adoption Phase | Dependencies |
|----|-------|--------|----------|----------------|--------------|
| PSG-100 | Semantic Type Patterns | Planned | High | 2 | PSG-001 through PSG-005, PLD-001 |
| PSG-101 | Effect System Conventions | Planned | High | 2 | PSG-001, PSG-004, PLD-003 |
| PSG-102 | Capability-Based Design | Planned | High | 2 | PSG-001, PSG-003, PLD-003 |
| PSG-103 | Contract Programming Style | Planned | Medium | 2 | PSG-001, PSG-004, PSG-005 |
| PSG-104 | Concurrency Patterns | Planned | Medium | 3 | PSG-001, PSG-004, PSG-101 |
| PSG-105 | Performance-Critical Code | Planned | Medium | 3 | PSG-001, PSG-004 |

### Domain-Specific Conventions (PSG-200 to PSG-299)

| ID | Title | Status | Priority | Adoption Phase | Dependencies |
|----|-------|--------|----------|----------------|--------------|
| PSG-200 | Web Application Patterns | Planned | Medium | 2 | PSG-001 through PSG-008 |
| PSG-201 | API Design Conventions | Planned | Medium | 2 | PSG-001, PSG-004, PSG-006 |
| PSG-202 | Database Integration Style | Planned | Medium | 2 | PSG-001, PSG-005, PSG-100 |
| PSG-203 | Security-Sensitive Code | Planned | High | 2 | PSG-001, PSG-102, PSG-008 |
| PSG-204 | Financial/Compliance Code | Planned | Medium | 3 | PSG-001, PSG-100, PSG-203 |

### Tooling & Ecosystem (PSG-300 to PSG-399)

| ID | Title | Status | Priority | Adoption Phase | Dependencies |
|----|-------|--------|----------|----------------|--------------|
| PSG-300 | Code Formatting Standards | Planned | High | 1 | PSG-001 |
| PSG-301 | Linting Rules & Configuration | Planned | High | 1 | PSG-001 through PSG-008 |
| PSG-302 | Testing Conventions | Planned | High | 1 | PSG-001, PSG-004, PSG-006 |
| PSG-303 | Build & Deployment Style | Planned | Medium | 2 | PSG-001, PSG-003 |
| PSG-304 | IDE Integration Guidelines | Planned | Medium | 2 | PSG-001, PSG-300 |

## Design Principles

### P1: Context-Appropriate Naming
**Inspired by**: Go's philosophy and modern AI-first development

Naming should be optimized for its context and audience. Local scope favors brevity for cognitive efficiency, while public APIs and business logic favor clarity for comprehension and maintenance.

```prism
// Local scope: Brief and contextual
function processOrder(order: Order) -> Result<(), OrderError> {
    let items = order.items  // Context makes meaning clear
    let total = 0.USD       // Local scope allows brevity
}

// Public API: Clear and descriptive
function calculateMonthlyInterestForSavingsAccount(
    principalAmount: Money<USD>,
    annualInterestRate: Percentage,
    compoundingFrequency: CompoundingPeriod
) -> Money<USD>  // Business logic is self-documenting
```

### P2: Consistency Across Contexts
**Inspired by**: Rust's consistent syntax patterns

Similar concepts should be expressed similarly across all contexts. This reduces cognitive load and improves AI code generation accuracy.

### P3: AI-First Documentation
**Unique to Prism**: Every construct includes AI-readable metadata

All code should include structured metadata that AI systems can parse and understand, enabling better code generation and analysis.

### P4: Semantic Explicitness
**Inspired by**: TypeScript's type annotations, enhanced for domain modeling

Types and constraints should express business meaning, not just data structure. This enables compile-time domain validation and better AI understanding.

### P5: Graceful Complexity
**Inspired by**: Go's "simple things should be simple, complex things should be possible"

Simple code should be trivial to write and understand. Complex code should be clearly marked and well-documented, with performance justifications.

### P6: Universal Accessibility
**Inspired by**: Web accessibility standards, applied to code

Code should be readable by screen readers, understandable by non-native English speakers, and navigable by developers with various disabilities.

## Syntax Philosophy

### Keyword Selection
**Inspired by**: Go's English-like keywords, Python's readability

Prism uses full English words for keywords rather than symbols or abbreviations:
- `function` instead of `fn` or `def`
- `and`/`or` instead of `&&`/`||` 
- `where` for constraints instead of generic syntax

### Punctuation Minimalism
**Inspired by**: Go's minimal punctuation, Python's clean syntax

Prism minimizes punctuation that doesn't add semantic meaning:
- No semicolons required (except for disambiguation)
- Minimal use of parentheses in control flow
- Clean block structure with meaningful indentation

### Semantic Operators
**Unique to Prism**: Operators that express business logic

Prism includes operators that express semantic relationships:
- `===` for semantic equality (not just structural)
- `~=` for type compatibility
- `≈` for conceptual similarity

## Adoption Strategy

### Phase 1: Foundation (Months 1-2)
**Focus**: Establish core conventions that enable basic development

- [ ] PSG-001: Fundamental Syntax & Formatting
- [ ] PSG-002: Naming Conventions & Identifiers  
- [ ] PSG-003: PrismDoc Standards
- [ ] PSG-004: Function Design & Signatures
- [ ] PSG-300: Code Formatting Standards

**Deliverables**:
- Basic style guide for early adopters
- Context-based naming guidelines
- Comprehensive documentation standards
- Formatter configuration
- Linting rules for core conventions

### Phase 2: Semantic Enhancement (Months 3-4)
**Focus**: Add semantic type and AI metadata export conventions

- [ ] PSG-005: Type Definitions & Annotations
- [ ] PSG-006: Error Handling Patterns
- [ ] PSG-007: Module Organization & Structure
- [ ] PSG-008: AI Context & Metadata
- [ ] PSG-100: Semantic Type Patterns

**Deliverables**:
- Complete style guide for semantic types
- Module organization patterns
- AI metadata standards
- Documentation generation tools

### Phase 3: Advanced Patterns (Months 5-6)
**Focus**: Patterns for complex, production-ready code

- [ ] PSG-101: Effect System Conventions
- [ ] PSG-102: Capability-Based Design
- [ ] PSG-103: Contract Programming Style
- [ ] PSG-302: Testing Conventions

**Deliverables**:
- Advanced pattern library
- Security-focused conventions
- Production deployment guidelines

## Language Influences & Acknowledgments

Prism's style guide draws inspiration from the best practices of many languages:

### From Go
- **Clarity over cleverness**: Simple, readable code is preferred
- **Consistent formatting**: One canonical way to format code
- **Descriptive naming**: Names should explain what, not how
- **Minimal punctuation**: Clean syntax without unnecessary symbols

### From Rust
- **Explicit error handling**: All error cases must be handled
- **Memory safety patterns**: Safe patterns by default
- **Trait-based design**: Composable abstractions
- **Comprehensive documentation**: Every public API documented

### From Python
- **Readability counts**: Code is read more than written
- **Explicit is better than implicit**: Clear intent over magic
- **Flat is better than nested**: Avoid deep hierarchies
- **English-like keywords**: Natural language constructs

### From TypeScript
- **Progressive enhancement**: Add types incrementally
- **Structural typing**: Focus on shape, not inheritance
- **Developer experience**: Excellent tooling and error messages
- **Gradual adoption**: Works alongside existing code

### From Haskell
- **Pure functions by default**: Immutability and side-effect tracking
- **Type-driven development**: Types guide implementation
- **Composable abstractions**: Small, combinable pieces
- **Mathematical precision**: Formal reasoning about code

### From C
- **Performance when needed**: Low-level control available
- **Minimal runtime**: Predictable performance characteristics
- **Systems programming**: Direct hardware access when required
- **Explicit resource management**: Clear ownership semantics

## Code Examples

### Basic Module Structure
```prism
@capability "User Authentication"
@description "Handles secure user login and session management"
@dependencies ["Database", "Cryptography"]
@stability Experimental
@version 1.0.0

module UserAuthentication {
    @aiContext "Core authentication module with security-first design"
    
    section types {
        type Email = String where {
            pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            max_length: 254,
            normalized: true
        };
        
        type UserId = UUID tagged "User" where {
            format: "USR-{8}-{4}-{4}-{4}-{12}",
            immutable: true
        };
    }
    
    section interface {
        @description "Authenticate user with email and password"
        @example """
            let result = authenticateUser(
                email: "user@example.com",
                password: "secure_password"
            );
        """
        function authenticateUser(
            email: Email,
            password: PlainPassword
        ) -> Result<AuthenticatedUser, AuthenticationError> {
            // Implementation
        }
    }
}
```

### Semantic Type Definition
```prism
type Money<Currency> = Decimal where {
    precision: 2,
    currency: Currency,
    non_negative: true,
    
    @aiContext {
        purpose: "Represents monetary amounts with currency safety",
        business_rules: [
            "precision must match currency requirements",
            "negative amounts not allowed for balances",
            "currency conversion requires explicit rates"
        ]
    }
};
```

### Function with Contracts
```prism
@description "Transfer funds between accounts with safety guarantees"
@security "Requires authentication and authorization checks"
function transferFunds(
    fromAccount: AccountId,
    toAccount: AccountId,
    amount: Money<USD>
) -> Result<Transaction, TransferError>
where {
    requires: [
        Account.exists(fromAccount),
        Account.exists(toAccount),
        Account.balance(fromAccount) >= amount,
        amount > 0.00.USD
    ],
    ensures: |result| match result {
        Ok(transaction) => {
            Account.balance(fromAccount) == old(Account.balance(fromAccount)) - amount and
            Account.balance(toAccount) == old(Account.balance(toAccount)) + amount
        },
        Err(_) => {
            Account.balance(fromAccount) == old(Account.balance(fromAccount)) and
            Account.balance(toAccount) == old(Account.balance(toAccount))
        }
    }
}
```

## Quick Reference

### File Extensions
- `.prsm` - Prism source files
- `.prsm.md` - Literate Prism files
- `.prsm.test` - Test files
- `.prsm.bench` - Benchmark files

### Naming Conventions
- **Modules**: `PascalCase` (e.g., `UserManagement`)
- **Functions**: `camelCase` - brief in local scope, descriptive in public APIs (e.g., `auth()` locally, `authenticateUser()` publicly)
- **Variables**: `camelCase` - context-appropriate brevity (e.g., `user` locally, `currentUser` in broader scope)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_RETRY_ATTEMPTS`)
- **Types**: `PascalCase` (e.g., `EmailAddress`)
- **Capabilities**: `"Descriptive String"` (e.g., `"User Management"`)
- **Annotations**: Max 80 chars for `@responsibility`, max 100 chars for `@description`

### Common Patterns
```prism
// Error handling
let result = riskyOperation();
match result {
    Ok(value) => processValue(value),
    Err(error) => handleError(error)
}

// Type constraints
type PositiveInteger = Integer where { min_value: 1 };

// AI context
@aiContext {
    purpose: "Brief description of what this does",
    constraints: ["Important limitations or requirements"],
    examples: ["Common usage patterns"]
}
```

## Contributing

### For Core Team
1. Identify style patterns in existing code
2. Document rationale and trade-offs
3. Provide examples and counter-examples
4. Consider AI comprehension impact
5. Submit for team review

### For Community
1. Discuss style questions in forums
2. Propose conventions via PEP process
3. Provide real-world usage examples
4. Consider accessibility implications

## Quick Links

- [PSG Template](./templates/PSG-TEMPLATE.md)
- [Style Guide Checker](./tools/style-checker.md)
- [Formatter Configuration](./tools/formatter.md)
- [Linting Rules](./tools/linter.md)
- [Community Style Forum](https://discuss.prsm-lang.org/c/style)

## Index Maintenance

This index is updated when:
- New PSG documents are created
- Adoption phases change
- Community feedback is incorporated
- Language features evolve

Last automated check: 2025-01-17 00:00:00 UTC

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-07-18 | Team | Initial style guide index with core conventions |

## Review Sign-offs

| Reviewer | Role | Status | Date |
|----------|------|--------|------|
| - | Language Design | Pending | - |
| - | Community | Pending | - |
| - | Accessibility | Pending | - |
| - | AI Integration | Pending | - | 