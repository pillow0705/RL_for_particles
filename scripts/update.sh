#!/bin/bash
# 从 GitHub 拉取最新代码
# 用法: bash scripts/update.sh

set -e
cd "$(dirname "$0")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Stashing local changes..."
    git stash
    git pull origin master
    git stash pop
else
    git pull origin master
fi

echo "Update done."
