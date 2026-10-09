@echo off
setlocal
rem One episode with a 64-decision override; the normal default remains 32.
pushd "%~dp0" || exit /b 1
".venv-gui\Scripts\python.exe" -u run_collect.py --brain=vista --episodes 1 --gui --max-decisions 64
set "vista_exit_code=%errorlevel%"
popd
exit /b %vista_exit_code%
