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
