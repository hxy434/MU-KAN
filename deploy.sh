#!/bin/bash

# 粒子分割项目部署脚本
echo "开始部署粒子分割项目..."

# 更新系统
sudo apt update
sudo apt upgrade -y

# 安装Python和pip
sudo apt install python3 python3-pip python3-venv -y

# 安装Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装Nginx
sudo apt install nginx -y

# 创建项目目录
sudo mkdir -p /var/www/seg-moga
sudo chown $USER:$USER /var/www/seg-moga

# 复制项目文件（假设项目文件已上传到服务器）
# cp -r * /var/www/seg-moga/

# 进入项目目录
cd /var/www/seg-moga

# 安装Python依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn

# 安装前端依赖
cd frontend
npm install
npm run build

# 配置Nginx
sudo tee /etc/nginx/sites-available/seg-moga << EOF
server {
    listen 80;
    server_name yundb.asia www.yundb.asia;

    # 前端静态文件
    location / {
        root /var/www/seg-moga/frontend/build;
        try_files \$uri \$uri/ /index.html;
    }

    # 后端API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# 启用站点
sudo ln -s /etc/nginx/sites-available/seg-moga /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 创建systemd服务文件
sudo tee /etc/systemd/system/seg-moga.service << EOF
[Unit]
Description=Seg-MOGA Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/var/www/seg-moga
Environment=PATH=/var/www/seg-moga/venv/bin
ExecStart=/var/www/seg-moga/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable seg-moga
sudo systemctl start seg-moga

echo "部署完成！"
echo "前端访问地址: http://yundb.asia"
echo "后端API地址: http://yundb.asia/api/"
