#!/bin/bash
# run.sh — 检查环境、安装依赖、运行训练
# 用法（服务器）: bash run.sh <SECRET_ID> <SECRET_KEY> <LOCAL_IP>
# 用法（本地测试）: bash run.sh --local-test
set -e

LOCAL_TEST=false
if [ "$1" == "--local-test" ]; then
    LOCAL_TEST=true
else
    TENCENT_SECRET_ID=$1
    TENCENT_SECRET_KEY=$2
    LOCAL_IP=$3
fi

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$WORK_DIR/experiments"

echo "=============================="
echo " RL Particle Packing Training"
echo "=============================="
echo " 工作目录: $WORK_DIR"

# ── 1. 检查 Python ────────────────────────────────────────────────
echo "[1/4] 检查 Python..."
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
    echo "  ERROR: 未找到 python3" && exit 1
fi
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PY_VER — $PYTHON"
if $PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"; then
    echo "  版本 OK"
else
    echo "  ERROR: 需要 Python >= 3.9，当前 $PY_VER" && exit 1
fi

# ── 2. 安装缺失依赖 ───────────────────────────────────────────────
echo "[2/4] 检查并安装依赖..."
pip3 install -r "$WORK_DIR/requirements_gpu.txt" -q

echo "  GPU 状态:"
$PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'    CUDA 可用 | {torch.cuda.get_device_name(0)} | 显存 {torch.cuda.get_device_properties(0).total_memory//1024**3}GB')
else:
    print('    无 GPU（仅 CPU 模式）')
"

if $LOCAL_TEST; then
    echo ""
    echo "本地测试通过，依赖检查完毕，未启动训练。"
    exit 0
fi

# ── 3. 训练 ───────────────────────────────────────────────────────
echo "[3/4] 开始训练..."
cd "$WORK_DIR"
python3 train.py
echo "  训练完成"

# ── 4. 传回结果 ───────────────────────────────────────────────────
echo "[4/4] 传回结果到 $LOCAL_IP..."
scp -i /root/.ssh/deploy_key \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=30 \
    -r "$RESULT_DIR/" \
    root@$LOCAL_IP:/root/program/RL_for_particles/
echo "  结果已传回"

# ── 5. 注销实例 ───────────────────────────────────────────────────
echo "注销实例..."
pip3 install tccli -q
tccli configure set secretId "$TENCENT_SECRET_ID" secretKey "$TENCENT_SECRET_KEY" region ap-shanghai
INSTANCE_ID=$(curl -s http://metadata.tencentyun.com/latest/meta-data/instance-id)
echo "  实例 ID: $INSTANCE_ID"
tccli cvm TerminateInstances --InstanceIds "[\"$INSTANCE_ID\"]"
echo "完成，实例正在注销。"
