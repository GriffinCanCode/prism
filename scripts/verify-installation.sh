#!/bin/bash
set -euo pipefail

# Prism Installation Verification Script
# Tests installation from various package managers

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Test functions
test_command() {
    local cmd="$1"
    local description="$2"
    
    log_info "Testing: $description"
    
    if command -v "$cmd" &> /dev/null; then
        log_success "$cmd is available"
        
        # Test basic functionality
        if "$cmd" --version &> /dev/null; then
            log_success "$cmd --version works"
        elif "$cmd" --help &> /dev/null; then
            log_success "$cmd --help works"
        else
            log_warning "$cmd available but --version/--help failed"
        fi
    else
        log_error "$cmd not found"
        return 1
    fi
}

test_docker_image() {
    local image="$1"
    local description="$2"
    
    log_info "Testing Docker image: $description"
    
    if docker pull "$image" &> /dev/null; then
        log_success "Successfully pulled $image"
        
        if docker run --rm "$image" --help &> /dev/null; then
            log_success "$image runs successfully"
        else
            log_warning "$image pulled but doesn't run properly"
        fi
    else
        log_error "Failed to pull $image"
        return 1
    fi
}

# Main verification
main() {
    log_info "Prism Installation Verification"
    echo
    
    local failed_tests=()
    
    # Test direct binary installation
    log_info "=== Testing Direct Binary Installation ==="
    if ! test_command "prism" "Prism compiler binary"; then
        failed_tests+=("prism-binary")
    fi
    echo
    
    # Test Docker installation
    if command -v docker &> /dev/null; then
        log_info "=== Testing Docker Installation ==="
        
        local images=(
            "ghcr.io/griffincancode/prism/prism-compiler:latest:Prism Compiler"
            "ghcr.io/griffincancode/prism/prism-dev:latest:Prism Development Environment"
            "ghcr.io/griffincancode/prism/prism-lsp:latest:Prism Language Server"
            "ghcr.io/griffincancode/prism/prism-ci:latest:Prism CI/CD Environment"
        )
        
        for image_info in "${images[@]}"; do
            IFS=':' read -r image tag description <<< "$image_info"
            full_image="${image}:${tag}"
            
            if ! test_docker_image "$full_image" "$description"; then
                failed_tests+=("docker-$tag")
            fi
        done
        echo
    else
        log_warning "Docker not available, skipping Docker tests"
        echo
    fi
    
    # Test npm installation
    if command -v npm &> /dev/null; then
        log_info "=== Testing npm Installation ==="
        
        if npm list -g @prism-lang/prism &> /dev/null; then
            log_success "@prism-lang/prism is installed via npm"
            
            if npx @prism-lang/prism --help &> /dev/null; then
                log_success "npm package works correctly"
            else
                log_warning "npm package installed but doesn't work"
                failed_tests+=("npm-execution")
            fi
        else
            log_info "@prism-lang/prism not installed via npm (this is optional)"
        fi
        echo
    else
        log_info "npm not available, skipping npm tests"
        echo
    fi
    
    # Test Homebrew installation (macOS/Linux)
    if command -v brew &> /dev/null; then
        log_info "=== Testing Homebrew Installation ==="
        
        if brew list prism &> /dev/null; then
            log_success "prism is installed via Homebrew"
            
            if brew test prism &> /dev/null; then
                log_success "Homebrew installation passes tests"
            else
                log_warning "Homebrew installation present but tests fail"
                failed_tests+=("homebrew-test")
            fi
        else
            log_info "prism not installed via Homebrew (this is optional)"
        fi
        echo
    else
        log_info "Homebrew not available, skipping Homebrew tests"
        echo
    fi
    
    # Test cargo installation
    if command -v cargo &> /dev/null; then
        log_info "=== Testing Cargo Installation ==="
        
        if cargo install --list | grep -q "prism-cli"; then
            log_success "prism-cli is installed via cargo"
        else
            log_info "prism-cli not installed via cargo (this is optional)"
        fi
        echo
    else
        log_info "Cargo not available, skipping cargo tests"
        echo
    fi
    
    # Summary
    log_info "=== Verification Summary ==="
    
    if [ ${#failed_tests[@]} -eq 0 ]; then
        log_success "All available installation methods are working correctly!"
        echo
        log_info "Installation methods verified:"
        
        if command -v prism &> /dev/null; then
            echo "  ✓ Direct binary installation"
        fi
        
        if command -v docker &> /dev/null; then
            echo "  ✓ Docker images"
        fi
        
        if command -v npm &> /dev/null && npm list -g @prism-lang/prism &> /dev/null; then
            echo "  ✓ npm package"
        fi
        
        if command -v brew &> /dev/null && brew list prism &> /dev/null; then
            echo "  ✓ Homebrew formula"
        fi
        
        if command -v cargo &> /dev/null && cargo install --list | grep -q "prism-cli"; then
            echo "  ✓ Cargo package"
        fi
        
        echo
        log_info "Try running: prism --help"
        
    else
        log_error "Some installation methods failed:"
        for failed in "${failed_tests[@]}"; do
            echo "  ✗ $failed"
        done
        echo
        log_info "This may be expected if you haven't installed Prism via all methods."
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Verifies Prism installation across different package managers."
            echo
            echo "Options:"
            echo "  --help    Show this help message"
            echo
            echo "This script tests:"
            echo "  - Direct binary installation"
            echo "  - Docker images"
            echo "  - npm package (@prism-lang/prism)"
            echo "  - Homebrew formula"
            echo "  - Cargo package (prism-cli)"
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