@echo off
echo 设置执行策略...
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"
echo 启动前端开发服务器...
cd frontend
npm start
pause
image.png对于