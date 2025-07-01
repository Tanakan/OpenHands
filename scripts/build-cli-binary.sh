#!/bin/bash
# Script to build OpenHands CLI binary locally

set -e

echo "Building OpenHands CLI binary..."

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    darwin)
        PLATFORM="darwin"
        ;;
    linux)
        PLATFORM="linux"
        ;;
    mingw*|msys*|cygwin*)
        PLATFORM="windows"
        ;;
    *)
        echo "Unsupported platform: $OS"
        exit 1
        ;;
esac

# Map architecture names
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Platform: $PLATFORM"
echo "Architecture: $ARCH"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install Poetry first."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
poetry install --no-interaction --no-ansi

# Install PyInstaller
echo "Installing PyInstaller..."
poetry add pyinstaller --group dev

# Build the binary
echo "Building binary with PyInstaller..."
poetry run pyinstaller openhands-cli.spec --clean --noconfirm

# Rename the output
if [ "$PLATFORM" = "windows" ]; then
    if [ -f "dist/openhands.exe" ]; then
        mv dist/openhands.exe "dist/openhands-$PLATFORM-$ARCH.exe"
        echo "Binary built successfully: dist/openhands-$PLATFORM-$ARCH.exe"
    fi
else
    if [ -f "dist/openhands" ]; then
        mv dist/openhands "dist/openhands-$PLATFORM-$ARCH"
        chmod +x "dist/openhands-$PLATFORM-$ARCH"
        echo "Binary built successfully: dist/openhands-$PLATFORM-$ARCH"
        
        # Test the binary
        echo "Testing binary..."
        "./dist/openhands-$PLATFORM-$ARCH" --version
    fi
fi

echo "Build complete!"