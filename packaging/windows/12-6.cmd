@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TWELVE_SIX_LOCK_DIR=%~dp0runtime\12-6-lock"
set "TWELVE_SIX_PYTHON=%~dp0runtime\Scripts\python.exe"
if not exist "%TWELVE_SIX_PYTHON%" (
  1>&2 echo error: 12-6 runtime is missing. Expected runtime\Scripts\python.exe
  endlocal & exit /b 10
)
"%TWELVE_SIX_PYTHON%" "%~dp0launcher.py" %*
set "TWELVE_SIX_EXIT=%ERRORLEVEL%"
endlocal & exit /b %TWELVE_SIX_EXIT%
