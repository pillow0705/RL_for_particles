# 上传代码到 GitHub
# 用法: .\scripts\upload.ps1 [提交信息]

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $MyInvocation.MyCommand.Path)
Set-Location ..

$msg = if ($args[0]) { $args[0] } else { "Update code" }

git add -A
$diff = git diff --cached --quiet; $changed = $LASTEXITCODE -ne 0
if ($changed) {
    git commit -m $msg
} else {
    Write-Host "Nothing to commit."
}

git push origin master
Write-Host "Upload done."
