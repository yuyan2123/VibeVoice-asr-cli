@echo off
chcp 65001 >nul
setlocal enableextensions

REM ============================================================
REM  VibeVoice ASR launcher  (keep this file ASCII-only:
REM  cmd cannot reliably parse non-ASCII bytes in a .bat;
REM  all Chinese output comes from the Python program instead.)
REM
REM  Usage:
REM    double-click -> launch the interactive app. The model loads once
REM    and stays resident; pick actions (file / record / settings) from
REM    the arrow-key menu inside the program.
REM ============================================================

REM Switch to this script's directory (project root)
cd /d "%~dp0"

REM Locate conda: prefer common install locations, then PATH
set "CONDA_BAT="
if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT for %%I in (conda.bat) do if not "%%~$PATH:I"=="" set "CONDA_BAT=%%~$PATH:I"

if not defined CONDA_BAT (
  echo [ERROR] conda not found. Please install Miniconda/Anaconda first.
  goto :end
)

REM Activate the environment
call "%CONDA_BAT%" activate vibevoice-asr
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env "vibevoice-asr".
  echo         Create it first, see project setup instructions.
  goto :end
)

set PYTHONUTF8=1

REM Launch the interactive app (single entry point; all options live in-menu)
python "%~dp0vibevoice_asr.py"
goto :end

:end
echo.
pause
endlocal
