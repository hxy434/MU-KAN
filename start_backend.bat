@echo off
echo 启动后端服务器...
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
pause
