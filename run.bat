@echo off
chcp 65001 >nul
setlocal enableextensions

REM ============================================================
REM  VibeVoice ASR launcher  (keep this file ASCII-only:
REM  cmd cannot reliably parse non-ASCII bytes in a .bat;
REM  all Chinese output comes from the Python program instead.)
REM
REM  First launch with no environment: this script creates the
REM  conda env "vibevoice-asr" from environment.yml automatically.
REM  The model itself is downloaded by the app on first use.
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
  echo [ERROR] conda not found. Please install Miniconda/Anaconda first:
  echo         https://docs.conda.io/projects/miniconda/
  goto :end
)

REM Create the environment on first run if it does not exist yet
call "%CONDA_BAT%" env list | findstr /B /C:"vibevoice-asr" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Conda env "vibevoice-asr" not found.
  echo [INFO] Creating it from environment.yml. This is a one-time step and may
  echo        take several minutes ^(it downloads PyTorch from the cu128 index^).
  call "%CONDA_BAT%" env create -f "%~dp0environment.yml"
  if errorlevel 1 (
    echo [ERROR] Failed to create the conda env. Please review the messages above.
    goto :end
  )
  echo [INFO] Environment created successfully.
)

REM Activate the environment
call "%CONDA_BAT%" activate vibevoice-asr
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env "vibevoice-asr".
  goto :end
)

set PYTHONUTF8=1

REM Launch the interactive app (single entry point; all options live in-menu).
REM On first use the app offers to download the model (~16GB, public on
REM Hugging Face, no account/token needed).
python "%~dp0vibevoice_asr.py"
goto :end

:end
echo.
pause
endlocal
