@echo off
REM Rekarisk — Build Windows package
REM Run this on Windows with Python 3.11+ and PyInstaller installed.
REM Usage: scripts\build_windows.bat

cd /d "%~dp0\.."
echo Building Rekarisk for Windows...

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
mkdir dist

echo Running PyInstaller...
set PYTHONPATH=src
pyinstaller rekarisk.spec --clean --noconfirm --distpath dist\windows --workpath build\windows

echo Creating ZIP...
cd dist\windows\Rekarisk
python -c "import zipfile,os;z=zipfile.ZipFile('../../Rekarisk-1.0.0-dev-windows-x86_64.zip','w',zipfile.ZIP_DEFLATED);[z.write(os.path.join(r,f),os.path.join(r,f)) for r,d,fs in os.walk('.') for f in fs];z.close();print('ZIP created')"
cd ..\..\..

echo.
echo Windows build complete!
echo   Directory: dist\windows\Rekarisk\
echo   Archive:   dist\Rekarisk-1.0.0-dev-windows-x86_64.zip
echo.
echo To run: dist\windows\Rekarisk\Rekarisk.exe
pause
