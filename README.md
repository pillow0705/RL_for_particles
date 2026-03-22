# RL for Particle Packing

用强化学习构造高密度多分散球形堆积，目标最大化体积分数 φ。

## 当前状态

- 目标粒子数：**N = 2000**，目标 φ = 0.72
- 粒径分布：14 种直径均匀分布在 [0.70, 1.35]
- 已达最高 φ ≈ 0.71（长跑实验）

---

## 架构概览

```
已放置粒子 (N个, 4D特征)          候选放置点 (最多2000个, 5D特征)
        │                                    │
  Linear(4→256)                   MLP: 5→64→128→256
        │                                    │
  前置 [CLS] token                            │
        │                                    │
  Transformer Encoder (4层, 8头)              │
        │                                    │
  取 CLS 输出 (256D)                          │
        │                                    │
        └──────── 点积打分 ────────────────────┘
                      │
              mask 无效候选点
                      │
              softmax(score / T)
                      │
               采样 → 放置粒子
```

**核心思路**：[CLS] token 汇聚当前堆积状态，与候选点嵌入做点积打分。粒子无顺序、无位置编码，坐标本身即空间信息。

---

## 文件结构

```
config.py        超参数（Config 类）
physics.py       numba JIT 物理核心（三球定位、碰撞检测）
env.py           ConstructEnv（增量候选点维护，O(N) 更新）
model.py         PackingPolicy（CandidateEncoder + Transformer + 点积打分）
collector.py     DataCollector（多进程采集）+ VectorizedCollector（备用）
trainer.py       REINFORCE Trainer（advantage 过滤 + 退步回滚）
train.py         主入口：train() / evaluate()
utils.py         实验目录、日志 Tee、保存工具
run.sh           腾讯云 GPU 服务器一键运行脚本
requirements_gpu.txt   GPU 依赖（cu121）
requirements_cpu.txt   CPU 依赖
scripts/
  upload.sh / upload.ps1   推送到 GitHub
  update.sh / update.ps1   从 GitHub 拉取
loop_log/        实验循环日志（每次迭代记录）
experiments/     训练输出（每次运行自动创建子目录）
```

---

## 快速启动

### 本地（CPU）

```bash
pip install -r requirements_cpu.txt
python train.py
```

### 腾讯云 GPU 服务器

```bash
# 首次上传代码
bash scripts/upload.sh

# 服务器上运行（自动安装依赖，训练完传回结果后销毁实例）
bash run.sh <SECRET_ID> <SECRET_KEY> <LOCAL_IP>

# 本地测试依赖是否完整
bash run.sh --local-test
```

---

## 训练算法

1. **第 1 轮**：随机策略采集 50 条轨迹（建立 baseline）
2. **第 2 轮起**：模型策略采样（温度 T=5.0）
3. 每轮结束：
   - 计算 advantage = φ − mean(φ)，**保留 |advantage| 最大的 5%** 步骤用于训练
   - 若 phi_max 退步超过 0.01，**自动回滚模型 + lr×0.5**
   - phi_max 创历史新高时保存 `_best.pth` + `best_packing.conf`

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `target_N` | 2000 | 目标粒子数 |
| `diameters` | arange(0.7, 1.40, 0.05) | 14 种粒径 |
| `max_candidates` | 2000 | 候选点集最大容量 |
| `num_iterations` | 25 | 训练迭代轮数 |
| `samples_per_iter` | 50 | 每轮采集样本数 |
| `num_workers` | 8 | 并行进程数 |
| `advantage_filter_ratio` | 0.95 | 丢弃低信号步骤比例 |
| `temperature` | 5.0 | 采集温度（探索力度） |
| `train_epochs` | 2 | 每轮训练 epoch |
| `rollback_tol` | 0.01 | 退步容忍阈值 |

---

## 输出结构

```
experiments/
  20260322_203029/
    config.json          超参数快照
    run.log              完整训练日志
    train_log.csv        逐轮指标（phi, loss, avg_cands...）
    eval_report.txt      最终评测报告（phi, Z, PDI）
    best_packing.conf    最优堆积坐标（x y z r 格式，末行为 Lx Ly Lz）
    construct_v7.0_best.pth    最优模型权重
    construct_v7.0_final.pth   最终模型权重
```

`best_packing.conf` 格式：
```
x1 y1 z1 r1
x2 y2 z2 r2
...
Lx Ly Lz
```
