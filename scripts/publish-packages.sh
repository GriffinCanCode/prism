#!/bin/bash
set -euo pipefail

# Prism Language Package Publishing Script
# Publishes to multiple package managers using Docker-built binaries

# Configuration
VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo 'dev')}"
REGISTRY="${REGISTRY:-prism-lang}"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="${GIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

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

# Create distribution directory
DIST_DIR="dist"
mkdir -p "$DIST_DIR"

# Extract binaries from Docker images for distribution
extract_binaries() {
    log_info "Extracting binaries from Docker images for package distribution"
    
    # Create temporary container to extract binaries
    local platforms=("linux/amd64" "linux/arm64" "darwin/amd64" "darwin/arm64")
    
    for platform in "${platforms[@]}"; do
        local os_arch="${platform//\//-}"
        local extract_dir="${DIST_DIR}/${os_arch}"
        mkdir -p "$extract_dir"
        
        log_info "Extracting binaries for $platform"
        
        # Extract from compiler image (contains the main prism binary)
        local container_id=$(docker create --platform="$platform" "${REGISTRY}/prism-compiler:${VERSION}")
        docker cp "$container_id:/usr/local/bin/prism" "$extract_dir/"
        docker rm "$container_id"
        
        # Make executable and add metadata
        chmod +x "$extract_dir/prism"
        
        # Create archive for distribution
        tar -czf "${DIST_DIR}/prism-${VERSION}-${os_arch}.tar.gz" -C "$extract_dir" prism
        
        log_success "Created ${DIST_DIR}/prism-${VERSION}-${os_arch}.tar.gz"
    done
}

# Publish to crates.io
publish_crates_io() {
    log_info "Publishing to crates.io"
    
    if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
        log_warning "CARGO_REGISTRY_TOKEN not set, skipping crates.io publish"
        return 0
    fi
    
    # Publish in dependency order
    local crates=(
        "prism-common"
        "prism-ast" 
        "prism-lexer"
        "prism-parser"
        "prism-semantic"
        "prism-effects"
        "prism-pir"
        "prism-codegen"
        "prism-syntax"
        "prism-runtime"
        "prism-compiler"
        "prism-ai"
        "prism-cli"
    )
    
    for crate in "${crates[@]}"; do
        if [ -d "crates/$crate" ]; then
            log_info "Publishing $crate to crates.io"
            (cd "crates/$crate" && cargo publish --token "$CARGO_REGISTRY_TOKEN")
            
            # Wait between publishes to avoid rate limiting
            sleep 10
        else
            log_warning "Crate $crate not found, skipping"
        fi
    done
    
    log_success "Published all crates to crates.io"
}

# Generate Homebrew formula
generate_homebrew_formula() {
    log_info "Generating Homebrew formula"
    
    local formula_dir="$DIST_DIR/homebrew"
    mkdir -p "$formula_dir"
    
    # Calculate SHA256 checksums for archives
    local darwin_amd64_sha=$(shasum -a 256 "${DIST_DIR}/prism-${VERSION}-darwin-amd64.tar.gz" | cut -d' ' -f1)
    local darwin_arm64_sha=$(shasum -a 256 "${DIST_DIR}/prism-${VERSION}-darwin-arm64.tar.gz" | cut -d' ' -f1)
    
    cat > "$formula_dir/prism.rb" << EOF
# Homebrew Formula for Prism Programming Language
# Generated automatically by publish-packages.sh

class Prism < Formula
  desc "AI-first programming language with semantic types and capability-based security"
  homepage "https://prism-lang.org"
  license "MIT OR Apache-2.0"
  version "${VERSION}"

  if Hardware::CPU.intel?
    url "https://github.com/GriffinCanCode/prism/releases/download/v${VERSION}/prism-${VERSION}-darwin-amd64.tar.gz"
    sha256 "${darwin_amd64_sha}"
  elsif Hardware::CPU.arm?
    url "https://github.com/GriffinCanCode/prism/releases/download/v${VERSION}/prism-${VERSION}-darwin-arm64.tar.gz"
    sha256 "${darwin_arm64_sha}"
  end

  def install
    bin.install "prism"
    
    # Install shell completions (if available)
    if File.exist?("completions/prism.bash")
      bash_completion.install "completions/prism.bash"
    end
    
    if File.exist?("completions/prism.zsh")
      zsh_completion.install "completions/prism.zsh" => "_prism"
    end
    
    if File.exist?("completions/prism.fish")
      fish_completion.install "completions/prism.fish"
    end
  end

  test do
    # Test basic functionality
    system "#{bin}/prism", "--version"
    system "#{bin}/prism", "--help"
    
    # Test compilation (when implemented)
    # (testpath/"hello.prism").write("// Hello World in Prism\\n")
    # system "#{bin}/prism", "compile", "hello.prism"
  end
end
EOF
    
    log_success "Generated Homebrew formula: $formula_dir/prism.rb"
}

# Generate npm package (for Node.js integration)
generate_npm_package() {
    log_info "Generating npm package"
    
    local npm_dir="$DIST_DIR/npm"
    mkdir -p "$npm_dir/bin"
    
    # Copy binaries for different platforms
    cp "${DIST_DIR}/linux-amd64/prism" "$npm_dir/bin/prism-linux-x64" || log_warning "Linux AMD64 binary not found"
    cp "${DIST_DIR}/linux-arm64/prism" "$npm_dir/bin/prism-linux-arm64" || log_warning "Linux ARM64 binary not found"
    cp "${DIST_DIR}/darwin-amd64/prism" "$npm_dir/bin/prism-darwin-x64" || log_warning "macOS AMD64 binary not found"
    cp "${DIST_DIR}/darwin-arm64/prism" "$npm_dir/bin/prism-darwin-arm64" || log_warning "macOS ARM64 binary not found"
    
    # Generate package.json
    cat > "$npm_dir/package.json" << EOF
{
  "name": "@prism-lang/prism",
  "version": "${VERSION#v}",
  "description": "AI-first programming language with semantic types and capability-based security",
  "keywords": ["compiler", "language", "ai", "semantic-types", "capabilities"],
  "homepage": "https://prism-lang.org",
  "repository": {
    "type": "git",
    "url": "https://github.com/GriffinCanCode/prism.git"
  },
  "license": "MIT OR Apache-2.0",
  "author": "Prism Language Team",
  "bin": {
    "prism": "./bin/prism.js"
  },
  "files": [
    "bin/",
    "README.md"
  ],
  "engines": {
    "node": ">=16"
  },
  "os": ["linux", "darwin"],
  "cpu": ["x64", "arm64"]
}
EOF

    # Generate platform-specific launcher
    cat > "$npm_dir/bin/prism.js" << 'EOF'
#!/usr/bin/env node

const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

// Determine the correct binary based on platform
function getBinaryPath() {
    const platform = os.platform();
    const arch = os.arch();
    
    let binaryName;
    if (platform === 'linux') {
        binaryName = arch === 'arm64' ? 'prism-linux-arm64' : 'prism-linux-x64';
    } else if (platform === 'darwin') {
        binaryName = arch === 'arm64' ? 'prism-darwin-arm64' : 'prism-darwin-x64';
    } else {
        console.error(`Unsupported platform: ${platform}-${arch}`);
        process.exit(1);
    }
    
    return path.join(__dirname, binaryName);
}

// Execute the Prism compiler
const binaryPath = getBinaryPath();
const child = spawn(binaryPath, process.argv.slice(2), { stdio: 'inherit' });

child.on('close', (code) => {
    process.exit(code);
});
EOF

    chmod +x "$npm_dir/bin/prism.js"
    
    # Copy README
    cp README.md "$npm_dir/" || log_warning "README.md not found"
    
    log_success "Generated npm package: $npm_dir"
}

# Generate Debian package
generate_debian_package() {
    log_info "Generating Debian package"
    
    local deb_dir="$DIST_DIR/debian/prism_${VERSION}_amd64"
    mkdir -p "$deb_dir/DEBIAN"
    mkdir -p "$deb_dir/usr/local/bin"
    
    # Copy binary
    cp "${DIST_DIR}/linux-amd64/prism" "$deb_dir/usr/local/bin/"
    
    # Generate control file
    cat > "$deb_dir/DEBIAN/control" << EOF
Package: prism
Version: ${VERSION#v}
Section: devel
Priority: optional
Architecture: amd64
Maintainer: Prism Language Team <team@prism-lang.org>
Description: AI-first programming language compiler
 Prism is an AI-first programming language designed with semantic types,
 capability-based security, and maximum extensibility. Built for the era
 where AI and humans collaborate on code.
Homepage: https://prism-lang.org
EOF
    
    # Build package
    dpkg-deb --build "$deb_dir"
    
    log_success "Generated Debian package: ${deb_dir}.deb"
}

# Generate GitHub Release assets
generate_github_release() {
    log_info "Preparing GitHub Release assets"
    
    local release_dir="$DIST_DIR/github-release"
    mkdir -p "$release_dir"
    
    # Copy all distribution archives
    cp "$DIST_DIR"/*.tar.gz "$release_dir/" || log_warning "No tar.gz files found"
    cp "$DIST_DIR"/*.deb "$release_dir/" || log_warning "No .deb files found"
    
    # Generate release notes
    cat > "$release_dir/RELEASE_NOTES.md" << EOF
# Prism ${VERSION} Release

## Installation Options

### Docker (Recommended)
\`\`\`bash
docker run --rm -v \$(pwd):/workspace ghcr.io/griffincancode/prism/prism-compiler:${VERSION}
\`\`\`

### Homebrew (macOS/Linux)
\`\`\`bash
brew install GriffinCanCode/tap/prism
\`\`\`

### npm (Node.js projects)
\`\`\`bash
npm install -g @prism-lang/prism
\`\`\`

### Direct Download
Download the appropriate binary for your platform from the assets below.

## What's New

- AI-first language design with semantic types
- Capability-based security system
- Multi-syntax parsing support
- Effect system for safe resource management
- Query-based incremental compilation

## Docker Images

- \`ghcr.io/griffincancode/prism/prism-compiler:${VERSION}\`
- \`ghcr.io/griffincancode/prism/prism-dev:${VERSION}\`
- \`ghcr.io/griffincancode/prism/prism-lsp:${VERSION}\`
- \`ghcr.io/griffincancode/prism/prism-ci:${VERSION}\`

Built on: ${BUILD_DATE}
Commit: ${GIT_SHA}
EOF
    
    log_success "Generated GitHub Release assets in $release_dir"
}

# Main publishing pipeline
main() {
    log_info "Starting Prism package publishing pipeline"
    log_info "Version: $VERSION"
    echo
    
    # Check prerequisites
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required for binary extraction"
        exit 1
    fi
    
    # Extract binaries from Docker images
    extract_binaries
    
    # Generate packages for different distribution channels
    generate_homebrew_formula
    generate_npm_package
    generate_debian_package
    generate_github_release
    
    # Publish to registries (if tokens are available)
    if [ "${PUBLISH_CRATES:-false}" = "true" ]; then
        publish_crates_io
    else
        log_info "Skipping crates.io publish (set PUBLISH_CRATES=true to enable)"
    fi
    
    log_success "Package publishing pipeline completed!"
    echo
    log_info "Distribution files created in: $DIST_DIR"
    log_info "Next steps:"
    echo "  1. Upload GitHub Release assets from: $DIST_DIR/github-release/"
    echo "  2. Submit Homebrew formula from: $DIST_DIR/homebrew/prism.rb"
    echo "  3. Publish npm package from: $DIST_DIR/npm/"
    echo "  4. Upload Debian package to repository"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --publish-crates)
            PUBLISH_CRATES="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --version VERSION     Package version (default: git describe)"
            echo "  --publish-crates      Publish to crates.io (requires CARGO_REGISTRY_TOKEN)"
            echo "  --help               Show this help message"
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