@echo off
chcp 65001 > nul
echo ═══════════════════════════════════════════════════════════
echo   Anime Generator Web UI - 启动器
echo   闭环生成控制系统 - 可视化推理界面
echo ═══════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"

echo [1/3] 检查依赖...
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未安装 Node.js
    pause
    exit /b 1
)

echo [2/3] 启动后端 API 服务 (端口 5176)...
start "Anime API" cmd /c "cd /d "%~dp0" ^&^& python api_server.py"

timeout /t 3 >nul

echo [3/3] 启动前端 (端口 5175)...
start "Anime Web" cmd /c "cd /d "%~dp0" ^&^& npm run dev"

echo.
echo ═══════════════════════════════════════════════════════════
echo   启动完成！
echo   前端界面: http://localhost:5175
echo   API 服务: http://localhost:5176
echo ═══════════════════════════════════════════════════════════
echo.
echo   按任意键打开浏览器...
pause > nul

start http://localhost:5175