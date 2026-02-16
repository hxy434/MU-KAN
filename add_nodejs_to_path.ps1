# 需要以管理员身份运行此脚本
# 永久添加Node.js路径到系统PATH环境变量

Write-Host "正在添加Node.js路径到系统环境变量..." -ForegroundColor Green

# Node.js路径
$nodejsPath = "D:\Node"

# 检查路径是否存在
if (Test-Path $nodejsPath) {
    Write-Host "Node.js路径存在: $nodejsPath" -ForegroundColor Green
} else {
    Write-Host "错误: Node.js路径不存在: $nodejsPath" -ForegroundColor Red
    exit 1
}

# 获取当前系统PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# 检查是否已经存在
if ($currentPath -like "*$nodejsPath*") {
    Write-Host "Node.js路径已经存在于系统PATH中" -ForegroundColor Yellow
} else {
    # 添加新路径到系统PATH
    $newPath = $currentPath + ";" + $nodejsPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    Write-Host "成功添加Node.js路径到系统PATH" -ForegroundColor Green
}

# 验证添加结果
Write-Host "`n验证结果:" -ForegroundColor Cyan
Write-Host "系统PATH中是否包含Node.js路径: $($currentPath -like "*$nodejsPath*")" -ForegroundColor White

Write-Host "`n请重启PowerShell或命令提示符以使更改生效" -ForegroundColor Yellow
Write-Host "重启后，您可以直接使用 'node --version' 和 'npm --version' 命令" -ForegroundColor Yellow

Read-Host "按任意键退出" 