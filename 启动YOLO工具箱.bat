@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo 未找到包内 Python 环境，请先双击 安装依赖.bat。
  pause
  exit /b 1
)
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 14) else 1)"
if errorlevel 1 (
  echo 包内环境不在 Python 3.9-3.14 支持范围内，请删除 .venv 后重新运行 安装依赖.bat。
  pause
  exit /b 1
)
"%PYTHON_EXE%" -u tools\install_dependencies.py --check-only
if errorlevel 1 (
  echo.
  echo 依赖未完整安装，请先双击 安装依赖.bat。
  pause
  exit /b 1
)
"%PYTHON_EXE%" -u tools\yolo_tool_launcher.py
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
