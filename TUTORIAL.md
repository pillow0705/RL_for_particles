# 技术文档

本文档详细描述当前代码的架构设计、训练流程和关键工程决策。

---

## 一、任务定义

在周期性边界条件（PBC）的立方箱中，逐步放入多分散球形粒子，目标是最大化最终**体积分数 φ**（堆积率）。每一步，模型从候选位置列表中选出一个放置点。

- 粒子数目标：N = 2000
- 粒径：14 种，直径均匀分布在 [0.70, 1.35]
- 候选点：由三球接触算法实时生成，最多保留 2000 个

---

## 二、物理层（physics.py）

### 候选点生成：三球定位

给定任意三个已放置粒子 A、B、C，新粒子若同时与三者相切，其圆心位置可由解析方程确定（两个解，取位于箱内且不碰撞的一个）。这是生成候选点的核心算法。

### 增量候选点维护（ConstructEnv）

不在每步从零重建候选点集合，而是增量更新：

```
放入粒子 P 后：
  1. 从候选集中删除与 P 碰撞的候选点
  2. 枚举包含 P 的所有三元组 (P, B, C)
  3. 对每个三元组生成新候选点，过滤碰撞后加入候选集
```

维护四个数据结构：
- `_candidate_set`：当前有效候选点（`{idx: (pos, rad, coord)}`）
- `_triplet_set`：生成过的三元组，避免重复计算
- `_cand_to_triplets`：候选点 → 生成它的三元组，用于级联删除
- `_cand_counter`：候选点计数器，生成唯一 idx

**性能**：每步更新复杂度 O(邻居数²)，远低于全量重建的 O(N³)。

### 碰撞检测（相对容差）

```python
gap = dist - (r_i + r_j)
collision = gap < -collision_tol * (r_i + r_j)   # 相对容差
touching   = gap <  edge_tol   * (r_i + r_j)     # 配位关系
```

使用相对容差（乘以半径之和），使容差随粒子大小自适应缩放，避免大粒子被误判碰撞。

---

## 三、模型架构（model.py）

### 全局架构

```
输入
├── 已放置粒子: N × 4  [Δx, Δy, Δz, r/r_max]
│   坐标相对质心，经 PBC wrap，缩放 0.5
└── 候选放置点: C × 5  [x, y, z, r, coord]
    C ≤ 2000，坐标参考系与粒子相同

已放置粒子 → CandidateEncoder(4→256) → [CLS] + tokens → Transformer(4层,8头) → CLS(256D)
候选放置点 → CandidateEncoder(5→64→128→256)                                  → cand_emb(C×256D)

score = CLS · cand_emb^T   →   (C,) 打分
score[无效] = -inf          →   mask
prob = softmax(score / T)   →   采样 / argmax
```

### CandidateEncoder（两用）

同一个 MLP 结构用于编码粒子和候选点（输入维度不同）：
- 粒子特征：4D → Linear(4→256) → ReLU（单层投影）
- 候选点特征：5D → MLP(5→64→128→256) + ReLU（三层渐进）

### Transformer Encoder

- 标准 PyTorch `TransformerEncoderLayer`，`batch_first=True`
- 层数：4，注意力头：8，FFN 维度：512
- 前置可学习 [CLS] token（`nn.Parameter`，shape `(1, 256)`）
- **无位置编码**：粒子坐标本身含空间信息，粒子间无顺序关系

### 点积打分

```python
score[i] = CLS_vec · cand_emb[i]   # 几何点积
```

CLS 是"当前堆积需要什么"的方向向量，候选点嵌入是"这个位置能提供什么"的描述，点积衡量匹配程度。比 MLP 融合解码器更高效、更具几何意义。

### 批量训练（batch_forward）

训练时粒子数各不相同，使用 padding 对齐：
```
padded_tokens: (B, max_N+1, d)   B 个样本填充至最长
pad_mask:      (B, max_N+1)      True = padding，Transformer 忽略
```
一次前向 B 个样本真正并行，GPU 利用率高。

---

## 四、采集层（collector.py）

### DataCollector（当前使用）

使用 `multiprocessing.Pool`（spawn 模式），每个 worker 是独立进程，无 GIL 共享：

```python
ctx = mp.get_context('spawn')
with ctx.Pool(processes=min(cfg.num_workers, n_samples)) as pool:
    for traj in pool.imap_unordered(_worker_collect_episode, args_list):
        ...
```

**为什么不用 ThreadPoolExecutor（VectorizedCollector）**：
env.step 包含大量 Python 层操作（重建 obs、更新状态结构），GIL 导致线程完全串行，N=2000 时 2 线程 ≈ 2× 单线程耗时，无加速。多进程无此限制，8 进程可达约 8× 真实并行。

### 单 episode 耗时参考

N=2000，随机策略，约 2000-2100 步，单进程约 75 秒。8 进程并行时，50 样本约 10-12 分钟。

---

## 五、训练流程（trainer.py / train.py）

### REINFORCE 算法

```
for iteration in range(25):
    1. 采集 50 条轨迹（第1轮随机，后续模型策略）
    2. 每条轨迹所有步的 return = 该轨迹的终局 φ
    3. 归一化 advantage = (φ - mean) / std
    4. 过滤：只保留 |advantage| 最大的 5% 步骤（filter_ratio=0.95）
    5. 批量训练 2 epoch，batch_size=128
       loss = -mean(log π(a|s) × advantage)
       log π 计算用相同温度 T，与采集分布一致
    6. 梯度裁剪 max_norm=1.0
```

### 退步检测与回滚

```python
if phi_max < prev_phi_max - 0.01:
    trainer.rollback()     # 恢复上轮权重
    lr *= 0.5              # 缩小学习率
    # 跳过本轮训练，用恢复的模型重新采集
```

每轮训练完成后 `backup()`，仅在退步时触发回滚。

### 模型保存策略

只在 `phi_max` 创历史新高时保存 `_best.pth` + `best_packing.conf`，避免差模型覆盖好模型。

---

## 六、实验管理

### 目录结构

每次 `train()` 调用自动在 `experiments/` 下以时间戳创建子目录：

```
experiments/20260322_203029/
  config.json       超参数完整快照（JSON）
  run.log           所有 stdout（通过 _Tee 同时输出到终端）
  train_log.csv     每轮：Iteration, PhiMean, PhiMax, PhiMin,
                          AvgSteps, Loss, AdvVar,
                          AvgCandsBefore, AvgFiltered, AvgAdded, AvgCandsAfter
  eval_report.txt   训练结束后评测报告
  best_packing.conf 最优堆积坐标
  construct_v7.0_best.pth
  construct_v7.0_final.pth
```

### train_log.csv 字段说明

| 字段 | 含义 |
|------|------|
| PhiMean/Max/Min | 本轮 50 样本的体积分数统计 |
| AvgSteps | 平均 episode 步数 |
| Loss | 训练损失（退步轮为 nan） |
| AdvVar | 过滤后 advantage 方差（梯度信号强度指标） |
| AvgCandsBefore | 每步候选点数（过滤前）|
| AvgFiltered | 每步被碰撞过滤掉的候选点数 |
| AvgAdded | 每步新增候选点数 |
| AvgCandsAfter | 每步候选点数（过滤后，即实际可选数量）|

---

## 七、GPU 服务器部署

### 环境要求

- Python 3.10+，CUDA 12.x
- 依赖：`torch>=2.1.0`，`numpy>=1.24.0`，`numba>=0.58.0`

### 一键部署

```bash
# 上传代码到服务器（需已配置 SSH 密钥）
bash scripts/upload.sh

# 服务器运行（训练完自动传回结果并销毁实例）
bash run.sh <TENCENT_SECRET_ID> <TENCENT_SECRET_KEY> <LOCAL_IP>
```

### 手动启动训练

```bash
# 后台运行，输出到日志
nohup python -u train.py > /tmp/train_live.log 2>&1 &

# 实时查看进度
tail -f /tmp/train_live.log
```

### CPU/GPU 使用模式

采集和训练**交替**使用资源：

| 阶段 | CPU | GPU |
|------|-----|-----|
| 采集（物理模拟）| ~800%（8进程各100%）| 0% |
| 训练（前向/反向）| 低 | 活跃 |

第 1 轮采集使用随机策略，GPU 全程空闲。第 2 轮起采集时 worker 进程需调用模型推理（CPU 上），GPU 仍空闲；训练时 GPU 活跃。整体 GPU 利用率偏低，这是 RL 采集密集型任务的固有特性。

---

## 八、已知问题与注意事项

1. **numba 无缓存**：`@njit` 未加 `cache=True`，每次 Python 启动都重新 JIT 编译（约 2 秒）。多进程时每个 worker 独立编译，spawn 启动慢约 5-10 秒。

2. **VectorizedCollector 不适用于 N=2000**：线程版因 GIL 在大 N 下完全串行，已切换为 DataCollector（多进程）。VectorizedCollector 保留在 collector.py 供参考，仅适合小 N 或 GPU 推理密集场景。

3. **advantage_filter_ratio=0.95**：只保留最极端 5% 的步骤用于训练，梯度信号极稀疏，AdvVar 可能偏低。若训练不稳定可调小至 0.7~0.8。

4. **温度 T=5.0**：采集时探索性较强，早期阶段合适。随训练深入可考虑逐步降低温度。
