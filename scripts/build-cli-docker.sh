#!/bin/bash
# Build OpenHands CLI binaries using Docker

set -e

echo "Building OpenHands CLI binaries in Docker..."

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.cli-build -t openhands-cli-builder .

# Create output directory
mkdir -p dist-docker

# Run container and extract artifacts
echo "Extracting binaries..."
docker run --rm -v "$(pwd)/dist-docker:/output" openhands-cli-builder

echo "Build complete! Artifacts are in dist-docker/"
ls -la dist-docker/