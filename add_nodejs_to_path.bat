@echo off
echo ========================================
echo 添加Node.js路径到系统环境变量
echo ========================================
echo.

echo 正在以管理员身份运行PowerShell脚本...
echo 请在弹出的UAC对话框中点击"是"

powershell -ExecutionPolicy Bypass -File "add_nodejs_to_path.ps1"

echo.
echo 脚本执行完成！
echo 请重启PowerShell或命令提示符以使更改生效
echo.
pause 