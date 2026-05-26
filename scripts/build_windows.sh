#!/bin/bash
# Rekarisk — Build Windows package (cross-compile not supported)
# This script must be run on Windows with Python + PyInstaller installed.
# Usage: scripts\build_windows.bat  (or bash scripts/build_windows.sh on Git Bash)

set -e
cd "$(dirname "$0")/.."
echo "🔨 Building Rekarisk for Windows..."

# Clean previous build
rm -rf build/ dist/
mkdir -p dist

# Build with PyInstaller
echo "📦 Running PyInstaller..."
PYTHONPATH=src pyinstaller rekarisk.spec --clean --noconfirm \
    --distpath dist/windows \
    --workpath build/windows

# Create ZIP archive
cd dist/windows/Rekarisk
echo "📋 Creating ZIP..."
python3 -c "
import zipfile, os
with zipfile.ZipFile('../../Rekarisk-1.0.0-dev-windows-x86_64.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        for f in files:
            fp = os.path.join(root, f)
            zf.write(fp, fp)
print('ZIP created')
"
cd ../..

echo "✅ Windows build complete!"
echo "   Directory: dist/windows/Rekarisk/"
echo "   Archive:   dist/Rekarisk-1.0.0-dev-windows-x86_64.zip"
echo ""
echo "To run: dist\\windows\\Rekarisk\\Rekarisk.exe"
