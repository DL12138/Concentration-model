@echo off
cd /d "%~dp0"
echo ============================================
echo   荧光浓度建模 APP 正在启动，请稍候...
echo   浏览器将自动打开，关闭本窗口即停止服务
echo ============================================
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" server.py
) else (
  echo [提示] 未找到项目虚拟环境，尝试使用系统 Python...
  python server.py
)
pause
