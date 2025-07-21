# Prism Build & Distribution Scripts

This directory contains scripts for building, packaging, and distributing the Prism programming language.

## 📋 Script Overview

| Script | Purpose | Audience | Usage |
|--------|---------|----------|--------|
| `build-images.sh` | Build all Docker images | Contributors, DevOps | `./scripts/build-images.sh` |
| `verify-installation.sh` | Test Prism installations | Users, Support | `./scripts/verify-installation.sh` |
| `publish-packages.sh` | Release to package managers | Maintainers only | `./scripts/publish-packages.sh --version v1.0.0` |

## 🚀 For Users

### Installation Verification
If you're having trouble with your Prism installation, run the verification script:

```bash
./scripts/verify-installation.sh
```

This will test all available installation methods and help diagnose issues.

### Custom Docker Builds
If you need to build custom Docker images (e.g., with modifications):

```bash
./scripts/build-images.sh --registry your-registry --version custom
```

## 🛠️ For Contributors

### Building Development Images
When working on Prism, build fresh Docker images:

```bash
# Build all images
./scripts/build-images.sh

# Build specific registry/version
./scripts/build-images.sh --registry prism-dev --version $(git rev-parse --short HEAD)
```

### Testing Your Changes
After making changes, verify everything still works:

```bash
# Build images with your changes
./scripts/build-images.sh

# Verify the installation works
./scripts/verify-installation.sh
```

## 🔧 For Maintainers

### Release Process
The `publish-packages.sh` script automates the entire release process:

```bash
# Generate packages (dry run)
./scripts/publish-packages.sh --version v1.0.0

# Publish to crates.io (requires CARGO_REGISTRY_TOKEN)
./scripts/publish-packages.sh --version v1.0.0 --publish-crates
```

This script:
- Extracts binaries from Docker images
- Generates Homebrew formula
- Creates npm package
- Builds Debian packages
- Prepares GitHub release assets

### Prerequisites for Publishing

Set these environment variables:

```bash
export CARGO_REGISTRY_TOKEN="your_crates_io_token"
export NPM_TOKEN="your_npm_token"  # Optional
```

## 📖 Integration with GitHub Actions

These scripts are used by our GitHub Actions workflows:

- **`docker-build.yml`**: Uses `build-images.sh` logic
- **`publish-release.yml`**: Uses `publish-packages.sh` logic

The workflows provide the same functionality but run in CI/CD environments.

## 🤝 Contributing

When adding new scripts:

1. Follow the existing naming convention
2. Add comprehensive help text (`--help`)
3. Use the same logging functions (log_info, log_success, etc.)
4. Update this README
5. Make scripts executable (`chmod +x`)

## 🔍 Troubleshooting

### Common Issues

**Docker not found:**
```bash
# Install Docker first
brew install docker  # macOS
# or follow https://docs.docker.com/get-docker/
```

**Permission denied:**
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

**Build failures:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
./scripts/build-images.sh --no-cache
```

### Getting Help

If you encounter issues:

1. Run `./scripts/verify-installation.sh` to diagnose
2. Check the [GitHub Issues](https://github.com/GriffinCanCode/prism/issues)
3. Join our [Discord community](https://discord.gg/prism-lang) 