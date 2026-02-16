# 添加Node.js路径到系统PATH
$nodejsPath = "D:\Node"

# 检查路径是否存在
if (Test-Path $nodejsPath) {
    Write-Host "Node.js路径存在: $nodejsPath" -ForegroundColor Green
    
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
} else {
    Write-Host "错误: Node.js路径不存在: $nodejsPath" -ForegroundColor Red
}

Write-Host "请重启PowerShell以使更改生效" -ForegroundColor Yellow 