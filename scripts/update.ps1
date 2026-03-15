# 从 GitHub 拉取最新代码
# 用法: .\scripts\update.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $MyInvocation.MyCommand.Path)
Set-Location ..

$dirty = git diff --quiet; $hasDiff = $LASTEXITCODE -ne 0
$staged = git diff --cached --quiet; $hasStaged = $LASTEXITCODE -ne 0

if ($hasDiff -or $hasStaged) {
    Write-Host "Stashing local changes..."
    git stash
    git pull origin master
    git stash pop
} else {
    git pull origin master
}

Write-Host "Update done."
