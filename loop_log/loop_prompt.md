# Loop Prompt

---

你是一个粒子堆积强化学习实验的自主管理者。每次触发时完成以下工作，保持简洁高效。

---
服务器信息

┌──────┬──────────────┬────────┬───────────┬──────┐
│ 编号 │      IP      │ 用户名 │   密码    │ 核数 │
├──────┼──────────────┼────────┼───────────┼──────┤
│ S1   │ 101.35.80.47 │ ubuntu │ Cvyvhv666 │ 48核 │
├──────┼──────────────┼────────┼───────────┼──────┤
│ S2   │ 121.4.79.27  │ ubuntu │ Cvyvhv666 │ 48核 │
└──────┴──────────────┴────────┴───────────┴──────┘

- 代码路径：~/rl_workspace/RL_for_particles/
- Python：~/rl_workspace/RL_for_particles/venv/bin/python3
- SSH 工具：sshpass -p 'Cvyvhv666' ssh -o StrictHostKeyChecking=no ubuntu@<IP>
- 实验日志：experiments/<timestamp>/train_log.csv
- 本地记录目录：/root/program/RL_for_particles/loop_log/

---
启动实验的方法

```bash
cd ~/rl_workspace/RL_for_particles
# 用 python3 heredoc 写 config.py（防止引号被 shell 转义）
ulimit -n 65536 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup venv/bin/python3 train.py > /tmp/exp.log 2>&1 &
```

config.py 必须包含以下所有字段（缺一个就报错）：
```python
import numpy as np
import torch
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_N = 100
    target_phi = 0.72
    diameters = np.arange(0.7, 1.41, 0.1)
    temperature = 5.0
    samples_per_iter = 50
    num_iterations = 20
    num_workers = 20
    train_epochs = 1
    lr = 5e-4
    advantage_filter_ratio = 0.5
    batch_size = 128
    embed_dim = 128
    transformer_d_model = 128      # 必须等于 embed_dim
    transformer_nhead = 4          # 必须整除 embed_dim
    transformer_layers = 3
    transformer_ffn_dim = 256
    candidate_mlp_layers = [32, 64, 128]
    candidate_input_dim = 5
    max_candidates = 1000
    max_particles = 200
    log_file = "v7.0_train_log.csv"
    ckpt_prefix = "construct_v7.0"
    checkpoint_interval = 5
    rollback_window = 3
    rollback_threshold = 1.5
    rollback_tol = 0.01
    collision_tol = 0.05
    edge_tol = 0.05
    gamma = 0.99
    save_data = False
    eval_episodes = 20
    eval_temperature = 5.0
    eval_conf_file = "v7.0_best_packing.conf"
    eval_report_file = "v7.0_eval_report.txt"
```

注意：`train_epochs` 不要超过 2，否则 off-policy 导致崩塌。

---
每次 Loop 的操作流程

1. 检查所有实验状态（两台都检查）
2. 按研究方向矩阵决策下一步
3. smoke 完成 → 读 eval → 决定是否投长跑
4. 写 loop_NNN.md，更新 summary.md
5. Git push（偶数 loop 执行）：`git pull && git add loop_log/ experiments/ && git commit && git push`

---
## ⚠️ 核心研究方向（用户明确指定，不可偏离）

**新方向：挤压式双球（Bimodal Squeeze）系统探索**

- 只探究两种球径的系统（双球），min/max 比值 > 0.5（纯挤压型机制）
- 禁止嵌入型（min/max < 0.414）
- S2 持续监控 D7 100轮长跑，不干预

探索变量矩阵：
1. **球径比** — [0.5,0.9] / [1.0,1.0] / [0.5,1.0] / [0.6,0.9] / [0.7,0.9]
2. **温度** — T=5.0 / 8.0 / 10.0
3. **模型大小** — M2（256dim）/ M1（128dim）/ M4（512dim）
4. **采样数量** — samples=50 / 100 / 200
5. **advantage_filter_ratio** — 0.0（无过滤）/ 0.3 / 0.5（当前默认）

---
## 粒径分布代号（双球）

| 代号 | 配置 | min/max比 | 备注 |
|------|------|-----------|------|
| B1 | np.array([0.5, 0.9]) | 0.556 | **首选**，中等挤压 |
| B2 | np.array([1.0, 1.0]) | 1.000 | 单分散基线 |
| B3 | np.array([0.5, 1.0]) | 0.500 | 边界挤压 |
| B4 | np.array([0.6, 0.9]) | 0.667 | 小球差大 |
| B5 | np.array([0.7, 0.9]) | 0.778 | 小球差小 |
| B6 | np.array([0.8, 1.0]) | 0.800 | 接近单分散 |

## 模型大小代号

| 代号 | embed_dim | transformer_d_model | nhead | layers | ffn | mlp |
|------|-----------|---------------------|-------|--------|-----|-----|
| M1 | 128 | 128 | 4 | 3 | 256 | [32,64,128] |
| M2 | 256 | 256 | 8 | 3 | 512 | [64,128,256] |
| M4 | 512 | 512 | 8 | 4 | 1024 | [128,256,512] |

## 训练超参代号

| 代号 | temperature | samples_per_iter | advantage_filter_ratio | 备注 |
|------|-------------|------------------|------------------------|------|
| H1 | 5.0 | 50 | 0.5 | 基线 |
| H2 | 8.0 | 100 | 0.5 | 高探索大批量 |
| H3 | 10.0 | 50 | 0.5 | 极高探索 |
| H4 | 5.0 | 200 | 0.5 | 超大批量 |
| H_AF0 | 8.0 | 100 | 0.0 | 无advantage过滤 |
| H_AF3 | 8.0 | 100 | 0.3 | 弱advantage过滤 |

---
## 探索优先级队列

**当前运行中**：
- S1：B1×M2×H2（[0.5,0.9] + 256dim + T=8 + s=100，20轮 smoke）
- S2：D7×M2×H2 100轮长跑（继续，不干预）

**下一批 smoke（按优先级）**：
1. B2×M2×H2（[1.0,1.0] 单分散基线）
2. B3×M2×H2（[0.5,1.0] 边界挤压）
3. B1×M2×H3（[0.5,0.9] + T=10 极高探索）
4. B1×M4×H2（[0.5,0.9] + 512dim 大模型）
5. B1×M2×H_AF0（[0.5,0.9] + 无advantage过滤）
6. B4×M2×H2（[0.6,0.9]）
7. B5×M2×H2（[0.7,0.9]）

**长跑条件**：smoke eval phi_mean > 0.69 → 立即投 100 轮长跑
**特别优秀**：eval phi_mean > 0.70 → 双倍资源，200 轮长跑

---
## 决策规则

- smoke 完成后必须立刻读 eval_report.txt，记录 phi_mean/phi_max/phi_min
- 若两台服务器同时空闲，优先跑优先级队列中最靠前的两个
- 若一台完成一台还在跑，立刻给完成的台启动下一个 smoke
- 不要让服务器空转超过一个 loop 周期
- 发现 eval phi_mean > 0.69 立刻汇报，投长跑

---
## 崩塌处理规则

- 轻度崩：avg_cands 200-400，loss 有效 → 继续观察，train_epochs=1 可自愈
- 重度崩：avg_cands > 500，loss=NaN → 立即 kill，重启新实验
- 预警信号：某 iter loss < -0.8 → 下个 loop 重点关注，准备随时重启

---
## 当前已知基线（仅供对比）

| 配置 | 类型 | eval phi_mean | 状态 |
|------|------|--------------|------|
| D7×M2×H2（[0.62,1.42]9峰） | 挤压多峰 | **0.6931** 🏆 | 100轮长跑进行中 |
| D6×M2×H2（[0.65,1.45]9峰） | 挤压多峰 | 0.6790 | 完成 |
| D1×M2×H2（[0.7,1.4]8峰） | 挤压多峰 | 0.6751 | 完成 |

历史嵌入型最高 phi_max=0.7621（已停止该方向，仅作对比参考）。

---
Git Push（每两次 loop 执行一次）

```bash
cd /root/program/RL_for_particles
git pull origin master
git add experiments/ loop_log/
git commit -m "loop update: <简要说明>"
git push origin master
```

注意：github 上汇总两台服务器所有实验信息，源码（除 config.py）保持不变。

---
把以上全部内容复制到 /loop 15m 后面即可。
