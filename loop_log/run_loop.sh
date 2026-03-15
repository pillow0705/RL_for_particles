#!/bin/bash
# 持久化 loop 脚本 - 由系统 crontab 每15分钟调用一次
# 无论 Claude Code 会话是否存在都会执行

cd /root/program/RL_for_particles

PROMPT=$(cat loop_log/loop_prompt.md)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "==============================" >> loop_log/cron.log
echo "[$TIMESTAMP] loop triggered" >> loop_log/cron.log

/root/.local/bin/claude \
  --dangerously-skip-permissions \
  -p "$PROMPT" \
  >> loop_log/cron.log 2>&1

echo "[$TIMESTAMP] loop done" >> loop_log/cron.log
