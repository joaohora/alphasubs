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
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :error

echo Building AlphaSubs.exe...
pyinstaller --noconfirm --onefile --windowed --name AlphaSubs ^
    --collect-all NDIlib ^
    --collect-all cv2 ^
    src\gui.py
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
