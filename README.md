# Prism Programming Language

[![Build Status](https://github.com/prism-lang/prism/workflows/CI/badge.svg)](https://github.com/prism-lang/prism/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/prism-lang/prism)
[![Documentation](https://docs.rs/prism/badge.svg)](https://docs.rs/prism)

Prism is an AI-first programming language designed with semantic types, capability-based security, and maximum extensibility. Built for the era where AI and humans collaborate on code, Prism provides unprecedented expressiveness for business logic while maintaining safety and performance.

## 🌟 Key Features

- **Semantic Type System**: Types that express business meaning, not just memory layout
- **AI-First Design**: Every language construct generates structured, AI-legible metadata for external AI tools
- **Capability-Based Security**: Zero-trust execution with fine-grained permission control
- **Smart Module System**: Code organization that follows human mental models
- **Effect System**: Precise tracking of computational effects and resource usage
- **Multi-Target Compilation**: Native (LLVM), TypeScript, and WebAssembly backends

## 🏗️ Project Structure

This project follows a modular workspace architecture with maximum extensibility:

```
prism/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── prism-common/        # Shared utilities and types
│   ├── prism-ast/           # Abstract Syntax Tree definitions
│   ├── prism-lexer/         # Tokenization and lexical analysis
│   ├── prism-parser/        # Parsing with error recovery
│   ├── prism-semantic/      # Type checking and semantic analysis
│   ├── prism-effects/       # Effect system and capabilities
│   ├── prism-codegen/       # Code generation backends
│   ├── prism-ai/            # AI metadata export and context extraction
│   ├── prism-cli/           # Command-line interface
│   └── prism-runtime/       # Runtime support libraries
└── design-docs/             # Language design specifications
    ├── PLD-001.md          # Semantic Type System
    ├── PLD-002.md          # Smart Module System
    ├── PLD-003.md          # Effect System & Capabilities
    └── PLT-001.md          # AST Design & Parser Architecture
```

## 🚀 Quick Start

### Option 1: Docker (Recommended for trying Prism)

```bash
# Pull and run the Prism compiler
docker run --rm -v $(pwd):/workspace ghcr.io/griffincancode/prism/prism-compiler:latest --help

# Start a development environment
docker run --rm -it -v $(pwd):/workspace ghcr.io/griffincancode/prism/prism-dev:latest

# Or use Docker Compose for full development setup
git clone https://github.com/GriffinCanCode/prism.git
cd prism
docker-compose up prism-dev
```

### Option 2: Build from Source

#### Prerequisites
- Rust 1.75.0 or later
- Git
- Docker (optional, for containerized development)

#### Installation
```bash
git clone https://github.com/GriffinCanCode/prism.git
cd prism
cargo build --release

# Build Docker images (optional)
./scripts/build-images.sh
```

### Example Prism Code

```prism
@capability "User Management"
@description "Handles user authentication and profile management"
module UserManagement {
    section types {
        type Email = String where {
            pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            max_length: 254,
            validated: true
        };

        type User = {
            id: UserId,
            email: Email,
            created_at: Timestamp,
            last_login: Option<Timestamp>
        } where {
            invariant: email.is_valid(),
            @aiContext: "Represents a user in the system with validation"
        };
    }

    section interface {
        function authenticate(email: Email, password: String) 
            -> Result<User, AuthError>
            effects [Database.Query, Cryptography.Hash]
            requires email.is_valid()
            ensures |result| match result {
                Ok(user) => user.email == email,
                Err(_) => true
            } {
            // Implementation here
        }
    }
}
```

## 🐳 Docker Usage

Prism provides multiple Docker images for different use cases:

### Available Images

- **`ghcr.io/griffincancode/prism/prism-compiler`**: Optimized compiler for production builds
- **`ghcr.io/griffincancode/prism/prism-dev`**: Full development environment with tools
- **`ghcr.io/griffincancode/prism/prism-lsp`**: Language server for IDE integration
- **`ghcr.io/griffincancode/prism/prism-ci`**: CI/CD optimized environment

### Common Usage Patterns

```bash
# Compile a Prism project
docker run --rm -v $(pwd):/workspace ghcr.io/griffincancode/prism/prism-compiler:latest compile src/

# Interactive development
docker run --rm -it -v $(pwd):/workspace ghcr.io/griffincancode/prism/prism-dev:latest

# Language server for IDE
docker run -d -p 9257:9257 ghcr.io/griffincancode/prism/prism-lsp:latest

# CI/CD pipeline
docker run --rm -v $(pwd):/ci ghcr.io/griffincancode/prism/prism-ci:latest test --all
```

### Development with Docker Compose

```bash
# Start full development environment
docker-compose up prism-dev

# Run specific services
docker-compose up prism-lsp prism-docs

# Clean rebuild
docker-compose build --no-cache
```

### Build Scripts

Prism includes helpful scripts for building and testing:

```bash
# Build all Docker images
./scripts/build-images.sh

# Verify your installation
./scripts/verify-installation.sh

# See all available scripts
ls scripts/ && cat scripts/README.md
```

## 📖 Language Design Documents

Prism's design is documented in a series of technical specifications:

- **[PLD-001](design-docs/PLD-001.md)**: Semantic Type System - Types that carry business meaning
- **[PLD-002](design-docs/PLD-002.md)**: Smart Module System - Code organization by capabilities
- **[PLD-003](design-docs/PLD-003.md)**: Effect System & Capabilities - Security and resource control
- **[PLT-001](design-docs/PLT/PLT-001.md)**: AST Design & Parser Architecture - Implementation details

## 🔧 Development

### Building

```bash
# Build all crates
cargo build

# Build with optimizations
cargo build --release

# Build specific crate
cargo build -p prism-ast
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p prism-parser

# Run benchmarks
cargo bench
```

### Linting

```bash
# Check formatting
cargo fmt --check

# Run clippy
cargo clippy -- -D warnings

# Fix formatting
cargo fmt
```

## 🎯 Design Principles

### 1. AI-First Design
Every language feature is designed with AI comprehension as a primary consideration. Code should be as readable to AI systems as it is to humans.

### 2. Semantic Richness Over Structural Simplicity
Types express what data means, not just how it's stored. Business rules are encoded in the type system for compile-time verification.

### 3. Security by Default
Capability-based security ensures code can only access the resources it explicitly needs. Supply chain attacks are prevented through dependency isolation.

### 4. Progressive Enhancement
Start simple and add complexity only when needed. The language scales from prototypes to enterprise systems.

### 5. Conceptual Cohesion
Code organization follows human mental models rather than arbitrary technical boundaries.

## 🤖 AI Metadata Export

Prism includes first-class AI metadata export features:

- **Semantic Context Extraction**: AI systems can understand code intent through rich type metadata
- **AI-Readable Documentation**: Built-in annotations provide context for AI code analysis
- **Prompt Injection Prevention**: Secure handling of AI-generated code and prompts
- **Structured Metadata Export**: Rich semantic information exported for external AI and development tools

## 🔒 Security Features

### Capability-Based Security
```prism
// Dependencies can only access explicitly granted capabilities
dependency analytics: AnalyticsLib with capabilities {
    network: Network.attenuate({
        allowed_hosts: ["analytics.company.com"],
        protocols: [HTTPS],
        rate_limit: 10.per_minute
    })
    // No file system or database access
}
```

### Effect Tracking
```prism
function process_payment(amount: Money<USD>) -> Result<Receipt, Error>
    effects [Database.Transaction, Network.Send, Cryptography.Sign] {
    // Compiler ensures all effects are handled safely
}
```

## 🎨 Examples

### Semantic Types
```prism
type Money<Currency> = Decimal where {
    precision: 2,
    currency: Currency,
    non_negative: true
};

type AccountId = UUID tagged "Account" where {
    format: "ACC-{8}-{4}-{4}-{4}-{12}",
    checksum: luhn_algorithm
};
```

### Smart Modules
```prism
@capability "Payment Processing"
module PaymentProcessor {
    section config {
        const MAX_AMOUNT = 10000.00.USD;
        const RETRY_ATTEMPTS = 3;
    }

    section interface {
        function charge(card: CreditCard, amount: Money<USD>) 
            -> Result<Transaction, PaymentError>;
    }
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📜 License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🌍 Community

- [Website](https://prism-lang.org)
- [Discord](https://discord.gg/prism-lang)
- [Twitter](https://twitter.com/prism_lang)
- [Reddit](https://reddit.com/r/prism_lang)

## 📚 Resources

- [Language Guide](https://prism-lang.org/guide)
- [API Documentation](https://docs.rs/prism)
- [Examples Repository](https://github.com/prism-lang/examples)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=prism-lang.prsm)

---

**Prism** - Programming for the AI age. Built by humans, designed for collaboration. 