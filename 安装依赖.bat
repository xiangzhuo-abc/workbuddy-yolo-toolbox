@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul
set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo 正在选择受支持的 Python 环境...
  set "PYTHON_LAUNCHER="
  where py >nul 2>&1
  if not errorlevel 1 for %%V in (3.11 3.12 3.13 3.10 3.9 3.14) do (
    if not defined PYTHON_LAUNCHER (
      py -%%V -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1
      if not errorlevel 1 set "PYTHON_LAUNCHER=py -%%V"
    )
  )
  if not defined PYTHON_LAUNCHER (
    where python >nul 2>&1
    if errorlevel 1 (
      echo 未找到 Python 3.9-3.14。请先安装受支持的 Python 版本。
      pause
      exit /b 1
    )
    python -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1
    if errorlevel 1 (
      echo 当前 Python 不在支持范围 Python 3.9-3.14 内。
      pause
      exit /b 1
    )
    set "PYTHON_LAUNCHER=python"
  )
  !PYTHON_LAUNCHER! -m venv "%VENV_DIR%"
)
if not exist "%PYTHON_EXE%" (
  echo 创建包内 Python 环境失败，请确认已安装 Python 3.9-3.14。
  pause
  exit /b 1
)
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)"
if errorlevel 1 (
  echo 包内环境不在 Python 3.9-3.14 支持范围内，请删除 .venv 后重新运行此脚本。
  pause
  exit /b 1
)
"%PYTHON_EXE%" -u tools\install_dependencies.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo 依赖安装失败，退出码: %RC%
pause
exit /b %RC%
