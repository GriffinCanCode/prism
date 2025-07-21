#!/bin/bash
set -euo pipefail

# Prism Language Docker Image Build Script
# Builds all Prism Docker images with proper tagging and optimization

# Configuration
REGISTRY="${REGISTRY:-prism-lang}"
VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo 'dev')}"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build configuration
TARGETS=(
    "prism-compiler:Prism Language Compiler"
    "prism-dev:Prism Development Environment" 
    "prism-lsp:Prism Language Server"
    "prism-ci:Prism CI/CD Environment"
)

# Build function
build_image() {
    local target=$1
    local description=$2
    local image_name="${REGISTRY}/${target}"
    
    log_info "Building ${target} (${description})"
    
    # Build with BuildKit for better caching and multi-stage optimization
    DOCKER_BUILDKIT=1 docker build \
        --target "${target}" \
        --tag "${image_name}:${VERSION}" \
        --tag "${image_name}:latest" \
        --label "org.opencontainers.image.title=${description}" \
        --label "org.opencontainers.image.version=${VERSION}" \
        --label "org.opencontainers.image.created=${BUILD_DATE}" \
        --label "org.opencontainers.image.revision=${GIT_SHA}" \
        --label "org.opencontainers.image.source=https://github.com/GriffinCanCode/prism" \
        --label "org.opencontainers.image.url=https://prism-lang.org" \
        --label "org.opencontainers.image.documentation=https://prism-lang.org/docs" \
        --label "org.opencontainers.image.vendor=Prism Language Team" \
        --label "org.opencontainers.image.licenses=MIT OR Apache-2.0" \
        --cache-from "${image_name}:latest" \
        .
    
    if [ $? -eq 0 ]; then
        log_success "Built ${image_name}:${VERSION}"
    else
        log_error "Failed to build ${target}"
        return 1
    fi
}

# Main build process
main() {
    log_info "Starting Prism Docker image builds"
    log_info "Registry: ${REGISTRY}"
    log_info "Version: ${VERSION}"
    log_info "Build Date: ${BUILD_DATE}"
    log_info "Git SHA: ${GIT_SHA}"
    echo

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "Cargo.toml" ] || [ ! -f "Dockerfile" ]; then
        log_error "Must run from Prism project root (Cargo.toml and Dockerfile must exist)"
        exit 1
    fi

    # Enable BuildKit for better performance
    export DOCKER_BUILDKIT=1

    # Build base image first for better caching
    log_info "Building base runtime image for caching"
    DOCKER_BUILDKIT=1 docker build \
        --target runtime-base \
        --tag "${REGISTRY}/runtime-base:${VERSION}" \
        --cache-from "${REGISTRY}/runtime-base:latest" \
        .

    # Build all target images
    local failed_builds=()
    
    for target_info in "${TARGETS[@]}"; do
        IFS=':' read -r target description <<< "$target_info"
        
        if ! build_image "$target" "$description"; then
            failed_builds+=("$target")
        fi
        echo
    done

    # Report results
    if [ ${#failed_builds[@]} -eq 0 ]; then
        log_success "All images built successfully!"
        
        echo
        log_info "Built images:"
        for target_info in "${TARGETS[@]}"; do
            IFS=':' read -r target description <<< "$target_info"
            echo "  - ${REGISTRY}/${target}:${VERSION}"
        done
        
        echo
        log_info "To run the images:"
        echo "  Compiler:    docker run --rm -v \$(pwd):/workspace ${REGISTRY}/prism-compiler:${VERSION}"
        echo "  Development: docker run --rm -it -v \$(pwd):/workspace ${REGISTRY}/prism-dev:${VERSION}"
        echo "  Language Server: docker run --rm -p 9257:9257 ${REGISTRY}/prism-lsp:${VERSION}"
        echo "  CI/CD:       docker run --rm -v \$(pwd):/ci ${REGISTRY}/prism-ci:${VERSION}"
        
    else
        log_error "Some builds failed:"
        for failed in "${failed_builds[@]}"; do
            echo "  - $failed"
        done
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --registry REGISTRY  Docker registry/namespace (default: prism-lang)"
            echo "  --version VERSION    Image version tag (default: git describe)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main 