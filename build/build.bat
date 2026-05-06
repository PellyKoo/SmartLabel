@echo off
REM ============================================================
REM  SmartLabel GUI - One-click build script
REM  Output: build\dist\SmartLabel-1.0.0\
REM  Copy the whole folder to target machine and run SmartLabel.exe
REM ============================================================
setlocal

REM ---------- Resolve Python path ----------
REM  Use SMARTLABEL_PYTHON env var if set, otherwise default to mmcv conda env
if not defined SMARTLABEL_PYTHON (
    set "SMARTLABEL_PYTHON=E:\developer_tools\anaconda3\envs\mmcv\python.exe"
)

if not exist "%SMARTLABEL_PYTHON%" (
    echo [ERROR] Python interpreter not found: %SMARTLABEL_PYTHON%
    echo         Please set SMARTLABEL_PYTHON env var to your python.exe path
    exit /b 1
)

echo === SmartLabel GUI Build ===
echo Python: %SMARTLABEL_PYTHON%

REM Switch to project root (parent of build/)
cd /d "%~dp0.."

REM ---------- Clean old build ----------
echo.
echo [1/3] Cleaning old build artifacts...
if exist "build\dist"  rmdir /s /q "build\dist"
if exist "build\build" rmdir /s /q "build\build"

REM ---------- Run PyInstaller ----------
echo.
echo [2/3] Running PyInstaller (5-15 min, first run is slower)...
"%SMARTLABEL_PYTHON%" -m PyInstaller --noconfirm --clean --distpath build\dist --workpath build\build build\SmartLabel.spec

if errorlevel 1 (
    echo.
    echo [FAILED] PyInstaller build failed
    exit /b 1
)

REM ---------- Post-process: copy extras ----------
echo.
echo [3/3] Copying extras...
set "OUT=build\dist\SmartLabel-1.0.0"
if not exist "%OUT%\models" mkdir "%OUT%\models"
copy /y build\README.md "%OUT%\README.md" > nul

REM ---------- Done ----------
echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Output: %OUT%
echo.
echo  Next steps:
echo    1. Put Qwen2-VL model into %OUT%\models\Qwen2-VL-2B-Instruct\
echo    2. Copy the whole folder to target machine
echo    3. Double-click SmartLabel.exe to launch
echo ============================================================

endlocal
