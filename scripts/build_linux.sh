#!/bin/bash
# Rekarisk — Build Linux package (AppImage-style directory bundle)
# Usage: ./scripts/build_linux.sh

set -e
cd "$(dirname "$0")/.."
echo "🔨 Building Rekarisk for Linux..."

# Clean previous build
rm -rf build/ dist/
mkdir -p dist

# Build with PyInstaller
echo "📦 Running PyInstaller..."
PYTHONPATH=src pyinstaller rekarisk.spec --clean --noconfirm \
    --distpath dist/linux \
    --workpath build/linux

# Create tar.gz archive
cd dist/linux/Rekarisk
echo "📋 Creating archive..."
tar czf ../../Rekarisk-1.0.0-dev-linux-x86_64.tar.gz .
cd ../..

echo "✅ Linux build complete!"
echo "   Directory: dist/linux/Rekarisk/"
echo "   Archive:   dist/Rekarisk-1.0.0-dev-linux-x86_64.tar.gz"
echo ""
echo "To run: ./dist/linux/Rekarisk/Rekarisk"
