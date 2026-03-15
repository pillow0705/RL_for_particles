# 模型架构升级：从 GNN 到 Transformer

## 为什么要替换 GNN？

原有架构使用图神经网络（GNN）编码已放置的粒子状态，存在几个明显问题：

**1. 训练无法真正并行**
GNN 需要邻接矩阵，而每个训练样本的粒子数不同，邻接矩阵大小也不同。训练时只能用 Python for 循环逐个计算图编码，256 个样本的 batch 实际上是串行的，GPU 利用率极低。

**2. 邻接矩阵维护复杂**
每放一个粒子就要增量更新邻接矩阵，代码复杂且容易出错。

**3. 感受野受限**
GNN 每层只能聚合直接相邻的粒子，需要多层才能感知全局结构。

---

## 新架构设计思路

### 核心想法：粒子作为 Token

把已放置的 N 个粒子看作一个序列，每个粒子是一个 token，特征为：

```
[x/L,  y/L,  z/L,  r/r_max]   ← 归一化坐标 + 归一化半径
```

**不需要位置编码**：NLP 中的位置编码是为了补充 token 的顺序信息，而粒子本身的 xyz 坐标已经包含了完整的空间位置信息，粒子之间也没有顺序关系，加位置编码反而引入错误的偏置。

### 粒子先做 Self-Attention

```
粒子 token (N, 4)
      │
  Linear → (N, d_model)
      │
  [CLS] + N tokens
      │
  Self-Attention × L 层
  ─ 粒子之间互相感知，理解局部密度和整体结构 ─
      │
  CLS token → (embed_dim,)   ← 整个堆积状态的压缩表示
```

Self-Attention 让每个粒子 token 融合所有其他粒子的信息，CLS token 汇聚全局状态，回答"当前堆积是什么样的？"

### 用点积打分，不用解码器

候选点编码后与 CLS 做点积：

```
CLS (embed_dim,)           候选点 (1000, 5)
      │                          │
      │                    Linear → (1000, embed_dim)
      │                          │
      └──────── 点积 ────────────┘
         score[i] = CLS · cand[i]
              │
         (1000,) logits
              │
    masked_fill(-inf) → softmax → 采样
```

**直觉**：CLS 是"当前状态需要什么样的粒子"的方向向量，候选点嵌入是"我是什么样的粒子"的自我描述，点积高说明这个候选点最符合当前需求。这比原来 MLP FusionDecoder 更简洁，也更有几何意义。

---

## 变长输入的处理

粒子数随着堆积过程增长（4 → 200），每个训练样本的粒子数不同。

**推理时**：batch=1，直接输入实际粒子，无需 padding。

**训练时**：将 batch 内所有样本 pad 到相同长度，用 `src_key_padding_mask` 屏蔽 padding 位置：

```python
tokens    (B, max_N, 4)     ← padding 位置填零
pad_mask  (B, max_N)        ← True 表示该位置被忽略

# Self-Attention 自动忽略 padding 粒子
# CLS 输出不受 padding 污染
```

这样图编码终于可以**真正 batch 化**，一次前向处理 256 个样本，GPU 利用率大幅提升。

---

## 同步修复的问题

**采集与训练的 temperature 不一致**

采集时：
```python
probs = softmax(scores / T)   # 用了 temperature
```
训练时原来：
```python
log_probs = log_softmax(scores)   # T 硬编码为 1
```

修复后训练时统一除以 temperature，保证策略梯度计算的 log π(a|s) 与采集分布一致。

---

## 架构对比

| | 原 GNN 架构 | 新 Transformer 架构 |
|---|---|---|
| 粒子编码 | GNN + 邻接矩阵 | Self-Attention，无需邻接矩阵 |
| 训练并行 | for 循环，串行 | 真正 batch，并行 |
| 候选点打分 | MLP FusionDecoder | CLS 点积，直接高效 |
| 代码复杂度 | 高（邻接矩阵维护） | 低 |
| 感受野 | 局部邻居 | 全局 |
