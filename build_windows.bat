@echo off
REM Builds AlphaSubs.exe (Windows) from the GUI, using PyInstaller.
REM Run this script on a Windows machine with Python 3.10+ installed
REM (from python.org, to make sure Tkinter is included).
REM
REM Usage:
REM   build_windows.bat

setlocal

cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

if not exist .venv\Scripts\activate.bat (
    echo.
    echo ERROR: .venv exists but looks incomplete (missing Scripts\activate.bat^).
    echo This usually means a previous run was interrupted. Delete it and retry:
    echo   rmdir /s /q .venv
    echo   build_windows.bat
    goto :error
)

call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo Building AlphaSubs.exe...
pyinstaller --noconfirm --onefile --windowed --name AlphaSubs ^
    --icon assets\icon.ico ^
    --add-data "assets\icon.png;assets" ^
    --add-data "VERSION;." ^
    --collect-all NDIlib ^
    --collect-all cv2 ^
    --collect-all faster_whisper ^
    --collect-all ctranslate2 ^
    --collect-all onnxruntime ^
    --collect-all _sounddevice_data ^
    --collect-all sacremoses ^
    --collect-all subword_nmt ^
    --collect-all azure.cognitiveservices.speech ^
    --collect-all piper ^
    --collect-all av ^
    src\gui.py
REM The CUDA/cuDNN DLLs (for Whisper on an NVIDIA GPU) are NOT bundled: at
REM ~1 GB they would make the single-file exe very slow to start. To use the
REM GPU with the exe, copy the DLLs from .venv\Lib\site-packages\nvidia\cublas\bin
REM and .venv\Lib\site-packages\nvidia\cudnn\bin into a "cuda" folder next to
REM AlphaSubs.exe (or install the CUDA 12 toolkit + cuDNN 9). Without them,
REM Whisper runs on the CPU.
if errorlevel 1 goto :error

echo.
echo Done. The executable is at dist\AlphaSubs.exe
echo Test it on a clean machine (no Python/venv) before distributing it to operators.
goto :end

:error
echo.
echo Build failed. See the messages above.
exit /b 1

:end
endlocal
