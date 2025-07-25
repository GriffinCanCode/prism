{
  description = "Prism Programming Language - AI-first development environment";

  # Inputs are external dependencies for our flake
  inputs = {
    # nixpkgs is the main package repository for Nix
    # Using "nixos-unstable" gives us the latest packages
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # flake-utils helps us write flakes that work on multiple systems (Linux, macOS)
    flake-utils.url = "github:numtide/flake-utils";

    # Rust overlay for latest Rust versions and cross-compilation support
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # Crane for advanced Rust builds (optional, for future use)
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # Outputs define what our flake produces
  outputs = { self, nixpkgs, flake-utils, rust-overlay, crane }:
    # This function creates outputs for each system (x86_64-linux, aarch64-darwin, etc.)
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import nixpkgs with rust overlay for our specific system
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        # Define the Rust toolchain we want
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ 
            "rust-src" 
            "rust-analyzer" 
            "clippy" 
            "rustfmt" 
            "llvm-tools-preview"
          ];
          targets = [
            "x86_64-unknown-linux-gnu"
            "aarch64-unknown-linux-gnu"
            "x86_64-apple-darwin"
            "aarch64-apple-darwin"
            "wasm32-unknown-unknown"
          ];
        };

        # Development tools specific to Prism
        prismDevTools = with pkgs; [
          # Core development tools
          git
          just                    # Command runner (alternative to make)
          ripgrep                 # Fast file search
          fd                      # Fast find alternative
          jq                      # JSON processor
          yq                      # YAML processor
          
          # Documentation and analysis
          graphviz                # For generating diagrams
          plantuml                # For UML diagrams
          pandoc                  # Document conversion
          
          # Performance and debugging
          valgrind                # Memory debugging (Linux only)
          gdb                     # GNU debugger
          lldb                    # LLVM debugger
          hyperfine               # Benchmarking tool
          
          # Language Server and IDE support
          nodejs_20               # For VSCode extensions
          python3                 # For scripts and tooling
          
          # Build and packaging tools
          pkg-config              # For native dependencies
          openssl                 # For TLS support
          zlib                    # Compression library
          
          # LLVM toolchain (for code generation backends)
          llvm_17
          clang_17
          
          # WebAssembly tools
          wasmtime                # WASM runtime
          wasm-pack              # Rust->WASM build tool
          
          # Container and deployment tools
          docker                  # For containerization
          docker-compose          # For multi-container apps
        ];

        # Platform-specific tools
        platformTools = with pkgs; lib.optionals stdenv.isLinux [
          # Linux-specific tools
          strace                  # System call tracer
          ltrace                  # Library call tracer
          perf-tools              # Performance analysis
        ] ++ lib.optionals stdenv.isDarwin [
          # macOS-specific tools
          darwin.apple_sdk.frameworks.Security
          darwin.apple_sdk.frameworks.CoreFoundation
          darwin.apple_sdk.frameworks.SystemConfiguration
        ];

      in
      {
        # Development shell - activated with 'nix develop' or automatically via direnv
        devShells.default = pkgs.mkShell {
          # Packages to include in the shell environment
          buildInputs = [ rustToolchain ] ++ prismDevTools ++ platformTools;

          # Environment variables
          # These are set when the shell is active
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = "1";
          RUST_LOG = "debug";
          
          # Prism-specific environment variables
          PRISM_DEV_MODE = "true";
          PRISM_LOG_LEVEL = "debug";
          PRISM_CACHE_DIR = ".prism-cache";
          
          # Build configuration
          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.zlib.dev}/lib/pkgconfig";
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          
          # LLVM configuration for code generation
          LLVM_SYS_170_PREFIX = "${pkgs.llvm_17.dev}";
          
          # Shell hook runs when entering the shell
          shellHook = ''
            echo "🚀 Prism Development Environment Loaded!"
            echo ""
            echo "📦 Rust $(rustc --version)"
            echo "📦 Cargo $(cargo --version)"
            echo "📦 LLVM ${pkgs.llvm_17.version}"
            echo "📦 Node.js $(node --version)"
            echo ""
            echo "🔧 Available commands:"
            echo "  cargo build          - Build the project"
            echo "  cargo test           - Run tests"
            echo "  cargo clippy         - Run linter"
            echo "  cargo fmt            - Format code"
            echo "  just --list          - Show available just commands"
            echo "  nix flake update     - Update flake inputs"
            echo ""
            echo "🎯 Prism-specific environment:"
            echo "  PRISM_DEV_MODE=${PRISM_DEV_MODE}"
            echo "  PRISM_LOG_LEVEL=${PRISM_LOG_LEVEL}"
            echo "  PRISM_CACHE_DIR=${PRISM_CACHE_DIR}"
            echo ""
            echo "📚 Documentation: https://prism-lang.org"
            echo "🐛 Issues: https://github.com/GriffinCanCode/prism/issues"
            echo ""
            
            # Create cache directory if it doesn't exist
            mkdir -p "$PRISM_CACHE_DIR"
            
            # Set up git hooks if not already done
            if [ ! -f .git/hooks/pre-commit ]; then
              echo "Setting up git hooks..."
              cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
# Pre-commit hook for Prism
set -e

echo "Running pre-commit checks..."

# Format check
if ! cargo fmt --check; then
  echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
  exit 1
fi

# Clippy check
if ! cargo clippy --all-targets --all-features -- -D warnings; then
  echo "❌ Clippy check failed. Fix the warnings above."
  exit 1
fi

# Basic build check
if ! cargo check --all-targets --all-features; then
  echo "❌ Build check failed."
  exit 1
fi

echo "✅ All pre-commit checks passed!"
EOF
              chmod +x .git/hooks/pre-commit
              echo "✅ Git hooks installed!"
            fi
            
            # Check if direnv is available and suggest setup
            if command -v direnv >/dev/null 2>&1; then
              if [ ! -f .envrc ]; then
                echo "💡 Tip: Create .envrc with 'use flake' for automatic environment loading"
              fi
            else
              echo "💡 Tip: Install 'direnv' for automatic environment loading when entering the directory"
            fi
          '';
        };

        # Packages provided by this flake
        packages = {
          # Future: We could add packages here for building Prism itself
          # prism-compiler = ...;
          # prism-cli = ...;
        };

        # Default package
        packages.default = self.packages.${system}.prism-cli or null;

        # Formatter for 'nix fmt'
        formatter = pkgs.nixpkgs-fmt;

        # Development apps
        apps = {
          # Future: Add apps for running Prism tools
          # prism = { ... };
        };
      });
} 