# Multi-stage Dockerfile for Prism Language Distribution
# This creates optimized images for different use cases

# ========================================
# Stage 1: Build Environment
# ========================================
FROM rust:1.75-slim as builder

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy all crate sources
COPY crates/ crates/
COPY design-docs/ design-docs/
COPY docs/ docs/

# Build the compiler in release mode
RUN cargo build --release --workspace

# Build the CLI tool specifically
RUN cargo build --release --bin prism

# ========================================
# Stage 2: Runtime Base Image
# ========================================
FROM debian:bookworm-slim as runtime-base

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create prism user for security
RUN useradd -r -s /bin/false -m -d /home/prism prism

# ========================================
# Stage 3: Prism Compiler Image
# ========================================
FROM runtime-base as prism-compiler

LABEL org.opencontainers.image.title="Prism Language Compiler"
LABEL org.opencontainers.image.description="AI-first programming language compiler with semantic types and capability-based security"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="Prism Language Team"
LABEL org.opencontainers.image.url="https://prism-lang.org"
LABEL org.opencontainers.image.source="https://github.com/GriffinCanCode/prism"

# Copy compiler binaries
COPY --from=builder /build/target/release/prism /usr/local/bin/prism

# Copy additional tools if they exist
# COPY --from=builder /build/target/release/prism-lsp /usr/local/bin/prism-lsp

# Set up working directory
WORKDIR /workspace
RUN chown prism:prism /workspace

# Switch to non-root user
USER prism

# Default command
ENTRYPOINT ["/usr/local/bin/prism"]
CMD ["--help"]

# ========================================
# Stage 4: Prism Development Environment
# ========================================
FROM runtime-base as prism-dev

LABEL org.opencontainers.image.title="Prism Language Development Environment"
LABEL org.opencontainers.image.description="Complete development environment for Prism language with all tools"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy all compiler components
COPY --from=builder /build/target/release/prism /usr/local/bin/prism

# Copy documentation and examples
COPY --from=builder /build/design-docs/ /usr/share/prism/design-docs/
COPY --from=builder /build/docs/ /usr/share/prism/docs/

# Set up development workspace
WORKDIR /workspace
RUN chown prism:prism /workspace

# Create example project structure
RUN mkdir -p /home/prism/.prsm/examples
COPY examples/ /home/prism/.prsm/examples/ || true
RUN chown -R prism:prism /home/prism/.prsm

USER prism

# Default to interactive shell for development
ENTRYPOINT ["/bin/bash"]

# ========================================
# Stage 5: Prism Language Server
# ========================================
FROM runtime-base as prism-lsp

LABEL org.opencontainers.image.title="Prism Language Server"
LABEL org.opencontainers.image.description="Language Server Protocol implementation for Prism language"

# Copy language server binary
COPY --from=builder /build/target/release/prism /usr/local/bin/prism

# Expose LSP port (if using TCP mode)
EXPOSE 9257

# Set up workspace
WORKDIR /workspace
RUN chown prism:prism /workspace

USER prism

# Start language server
ENTRYPOINT ["/usr/local/bin/prism", "lsp"]

# ========================================
# Stage 6: Prism CI/CD Image
# ========================================
FROM runtime-base as prism-ci

LABEL org.opencontainers.image.title="Prism CI/CD Environment"
LABEL org.opencontainers.image.description="Optimized image for continuous integration and deployment"

# Install CI tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy compiler
COPY --from=builder /build/target/release/prism /usr/local/bin/prism

# Copy build scripts
COPY scripts/ /usr/local/bin/prism-scripts/ || true

# Set up CI workspace
WORKDIR /ci
RUN chown prism:prism /ci

USER prism

ENTRYPOINT ["/usr/local/bin/prism"] 