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

**唯一方向：[0.7, 1.4] 区间 × 全参数系统探索**

禁止做嵌入型实验（min/max < 0.414，如 [0.4,0.8,1.4]）。
[0.7,1.4] 区间 min/max=0.5，是纯挤压型机制，此前仅做过 20 轮 smoke，完全未充分探索。

---
## 研究参数矩阵

### 维度 A：diameters（粒径分布）

| 代号 | 配置 | 峰数 | 备注 |
|------|------|------|------|
| D1 | np.arange(0.7, 1.41, 0.1) | 8 | **当前基线**，eval=0.6722 |
| D2 | np.arange(0.7, 1.41, 0.05) | 15 | 细密均匀，eval=0.6716 |
| D3 | np.arange(0.7, 1.41, 0.025) | 29 | 超细密，未测 |
| D4 | 偏密小端：[0.7,0.75,0.8,0.85,0.9,1.0,1.1,1.2,1.4] | 9 | 非均匀，未测 |
| D5 | 偏密大端：[0.7,0.9,1.0,1.1,1.15,1.2,1.25,1.3,1.35,1.4] | 10 | 非均匀，未测 |
| D6 | 宽区间延伸：np.arange(0.65, 1.45, 0.1) | 9 | 微扩边界，未测 |

### 维度 B：模型大小

| 代号 | embed_dim | transformer_d_model | nhead | layers | ffn | mlp | 参数量 |
|------|-----------|---------------------|-------|--------|-----|-----|--------|
| M1 | 128 | 128 | 4 | 3 | 256 | [32,64,128] | ~42万 |
| M2 | 256 | 256 | 8 | 3 | 512 | [64,128,256] | ~170万 |
| M3 | 128 | 128 | 4 | 6 | 256 | [32,64,128] | ~65万 |
| M4 | 512 | 512 | 8 | 4 | 1024 | [128,256,512] | ~600万 |

### 维度 C：训练超参

| 代号 | temperature | samples_per_iter | lr | 备注 |
|------|-------------|------------------|----|------|
| H1 | 5.0 | 50 | 5e-4 | **当前基线** |
| H2 | 8.0 | 100 | 5e-4 | 高探索+大批量 |
| H3 | 10.0 | 50 | 5e-4 | 极高探索 |
| H4 | 5.0 | 200 | 5e-4 | 超大批量 |
| H5 | 8.0 | 50 | 1e-3 | 高温+大lr |
| H6 | 5.0 | 100 | 1e-3 | 大批量+大lr |

---
## 探索优先级队列

**当前运行中**：
- S1：D1 × M2 × H1（8峰 + 大模型256，20轮）
- S2：D1 × M1 × H2（8峰 + 高温T=8 + samples=100，20轮）

**下一批 smoke（按优先级）**：
1. D1 × M2 × H2（大模型 + 高温100样本）—— 两个优势结合
2. D1 × M3 × H1（加深layers=6）
3. D3 × M2 × H1（超细密29峰 + 大模型）
4. D1 × M1 × H3（极高温T=10）
5. D4/D5 × M2 × H2（非均匀分布 + 大模型）
6. D1 × M4 × H1（超大模型512dim）

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

| 配置 | 轮数 | eval phi_mean | 状态 |
|------|------|--------------|------|
| D1×M1×H1（8峰标准） | 20 | 0.6722 | 基线，未充分 |
| D2×M1×H1（15峰标准） | 20 | 0.6716 | 基线，未充分 |

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
