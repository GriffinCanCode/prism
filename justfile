# Justfile for Prism Programming Language Development
# 
# This file contains common development tasks for the Prism project.
# Run `just --list` to see all available commands.

# Default recipe - show available commands
default:
    @just --list

# Build the entire workspace
build:
    cargo build --workspace

# Build with all features enabled
build-all:
    cargo build --workspace --all-features

# Build in release mode
build-release:
    cargo build --workspace --release

# Run all tests
test:
    cargo test --workspace

# Run tests with output
test-verbose:
    cargo test --workspace -- --nocapture

# Run specific crate tests
test-crate crate:
    cargo test --package {{crate}}

# Run clippy on all crates
clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Format all code
fmt:
    cargo fmt --all

# Check formatting without making changes
fmt-check:
    cargo fmt --all -- --check

# Run security audit
audit:
    cargo audit

# Clean build artifacts
clean:
    cargo clean

# Clean and rebuild everything
rebuild: clean build

# Generate documentation
docs:
    cargo doc --workspace --all-features --no-deps --open

# Check that everything compiles
check:
    cargo check --workspace --all-targets --all-features

# Run the CLI
run *args:
    cargo run --bin prism-cli -- {{args}}

# Run the compiler on a file
compile file:
    cargo run --bin prism-cli -- compile {{file}}

# Run the language server
lsp:
    cargo run --bin prism-cli -- lsp

# Development workflow - format, clippy, test
dev: fmt clippy test

# CI workflow - all checks
ci: fmt-check clippy test audit

# Install development tools
install-tools:
    cargo install cargo-audit
    cargo install cargo-watch
    cargo install cargo-expand
    cargo install cargo-bloat

# Watch for changes and run tests
watch:
    cargo watch -x "test --workspace"

# Watch for changes and run clippy
watch-clippy:
    cargo watch -x "clippy --workspace --all-targets --all-features"

# Build the parser
build-parser:
    cd crates/prism-parser && cargo build

# Benchmark the parser
bench:
    cargo bench --package prism-parser

# Profile the parser
profile:
    cargo build --release --package prism-parser
    # Add profiling commands here

# Update dependencies
update:
    cargo update

# Check for outdated dependencies
outdated:
    cargo outdated

# Show dependency tree
deps:
    cargo tree

# Show workspace info
info:
    cargo metadata --format-version 1 | jq '.workspace_members'

# Run example programs
example name:
    cargo run --example {{name}}

# Test the Nix flake
test-nix:
    nix flake check

# Update Nix flake inputs
update-nix:
    nix flake update

# Enter Nix development shell
nix-shell:
    nix develop

# Docker build
docker-build:
    docker build -t prism-lang .

# Docker run
docker-run:
    docker run -it prism-lang

# Generate coverage report
coverage:
    cargo tarpaulin --workspace --out Html --output-dir coverage

# Run mutation tests
mutate:
    cargo mutagen

# Check license compliance
license-check:
    cargo deny check licenses

# Security check
security-check:
    cargo deny check advisories

# Spell check documentation
spell-check:
    cspell "**/*.md" "**/*.rs"

# Run integration tests
integration-test:
    cargo test --test integration_end_to_end

# Performance test
perf-test:
    cargo test --release --package prism-parser -- --ignored

# Memory test
memory-test:
    valgrind --tool=memcheck --leak-check=full cargo test --package prism-parser

# Flamegraph profiling
flamegraph:
    cargo flamegraph --bin prism-cli -- compile examples/hello.prism

# Size analysis
size-analysis:
    cargo bloat --release --crates

# Assembly output
asm function:
    cargo rustc --release --package prism-parser -- --emit asm
    find target -name "*.s" -exec grep -l "{{function}}" {} \;

# LLVM IR output
llvm-ir:
    cargo rustc --release --package prism-parser -- --emit llvm-ir

# Create release
release version:
    git tag -a v{{version}} -m "Release v{{version}}"
    git push origin v{{version}}

# Publish to crates.io (dry run)
publish-dry:
    cargo publish --dry-run --package prism-common
    cargo publish --dry-run --package prism-lexer
    cargo publish --dry-run --package prism-ast
    cargo publish --dry-run --package prism-parser

# Publish to crates.io
publish:
    cargo publish --package prism-common
    sleep 10
    cargo publish --package prism-lexer
    sleep 10
    cargo publish --package prism-ast
    sleep 10
    cargo publish --package prism-parser

# Setup git hooks
setup-hooks:
    cp scripts/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit 