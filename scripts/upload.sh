#!/bin/bash
# 上传代码到 GitHub
# 用法: bash scripts/upload.sh [提交信息]

set -e
cd "$(dirname "$0")/.."

MSG="${1:-Update code}"

git add -A
if git diff --cached --quiet; then
    echo "Nothing to commit."
else
    git commit -m "$MSG"
fi

git push origin master
echo "Upload done."
