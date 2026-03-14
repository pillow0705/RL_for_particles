"""
RL_Construct_v6.0.py
====================
粒子堆积强化学习 —— 新架构版本

模型架构：
  左侧: 候选点编码器 (MLP)   [5D → embed_dim]
  右侧: 图编码器 (GNN)        [当前所有粒子 → embed_dim]
  融合: 拼接 + MLP            [2*embed_dim → 1 (得分)]

训练算法：
  迭代式 REINFORCE（无 Critic）
  第0轮: 随机策略采集数据
  后续轮: 用当前模型采集数据 → 训练 → 迭代
"""

import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from numba import njit
from scipy.spatial import KDTree

# =====================================================================
# 0. 全局超参数配置
# =====================================================================
class Config:
    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 物理 / 环境参数 ----
    target_N      = 100             # 目标粒子数
    target_phi    = 0.72            # 目标体积分数（用于确定箱子大小）
    diam_min      = 0.7             # 最小直径
    diam_max      = 1.35            # 最大直径（含）
    diam_step     = 0.05            # 直径步长
    diameters     = np.arange(0.7, 1.40, 0.05)  # 所有可选直径
    max_candidates = 1000           # 每步最多候选位置数
    collision_tol  = 0.05           # 碰撞容忍阈值（Numba 内）
    edge_tol       = 0.05           # 图边构建：相切容忍（r_i+r_j+edge_tol 判断）
    max_particles  = 200            # 单局最多放置粒子数（超出则强制结束）

    # ---- 候选点编码器 (左侧 MLP) ----
    candidate_input_dim  = 5                # 输入维度: [x, y, z, r, coord]
    candidate_mlp_layers = [32, 64, 128]    # 逐步升维，末尾 = embed_dim

    # ---- 图编码器 (右侧 GNN) ----
    graph_input_dim   = 4           # 节点输入维度: [x, y, z, r]
    graph_hidden_dim  = 128         # GNN 隐藏层宽度（= embed_dim）
    gnn_layers        = 3           # 消息传递层数

    # ---- 融合解码器 ----
    embed_dim         = 128         # 两侧统一输出维度（须与上面末层一致）
    fusion_layers     = [256, 128]  # 融合 MLP 层宽，最后自动接 → 1

    # ---- 训练超参数 ----
    num_workers       = 2           # 并行采集的进程数
    num_iterations    = 20          # 外层迭代轮数
    samples_per_iter  = 30          # 每轮采集的堆积样本数
    train_epochs      = 10          # 每轮训练的 epoch 数
    batch_size        = 256         # minibatch 大小
    lr                = 3e-4        # 学习率
    gamma             = 0.99        # 折扣因子
    temperature       = 1.0         # 随机采样温度（第0轮）
    baseline_alpha    = 0.05        # 运行 baseline 的指数平滑系数

    # ---- 输出 ----
    log_file      = "v6.0_train_log.csv"
    ckpt_prefix   = "construct_v6.0"
    save_interval = 5               # 每隔几轮保存一次 checkpoint

    # ---- 评测 ----
    eval_episodes    = 10           # 训练结束后贪婪评测的局数
    eval_report_file = "v6.0_eval_report.txt"   # 评测报告文件
    eval_conf_file   = "v6.0_best_packing.conf" # 评测中 phi 最高的堆积


# =====================================================================
# 1. 物理核心（Numba JIT，与原代码保持一致）
# =====================================================================
@njit(inline='always')
def pbc_diff_njit(p1, p2, L):
    """计算 p1 - p2 的周期性边界条件位移向量"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return np.array([
        dx - round(dx / L) * L,
        dy - round(dy / L) * L,
        dz - round(dz / L) * L
    ])


@njit
def solve_three_spheres_njit(p1, r1, p2, r2, p3, r3, r_new):
    """三球定位：求与三个已知球同时相切的新球坐标"""
    dp21 = p2 - p1
    d12  = np.sqrt(np.sum(dp21 ** 2))
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, np.zeros(3), np.zeros(3)

    ex = dp21 / d12
    dp31 = p3 - p1
    i    = np.dot(ex, dp31)
    ey_v = dp31 - i * ex
    d_ey = np.sqrt(np.sum(ey_v ** 2))

    if d_ey < 1e-7:
        return False, np.zeros(3), np.zeros(3)
    ey = ey_v / d_ey

    # 手动叉乘（numba 部分版本对 np.cross 支持有限）
    ez = np.array([
        ex[1] * ey[2] - ex[2] * ey[1],
        ex[2] * ey[0] - ex[0] * ey[2],
        ex[0] * ey[1] - ex[1] * ey[0]
    ])

    x    = (s1 ** 2 - s2 ** 2 + d12 ** 2) / (2 * d12)
    y    = (s1 ** 2 - s3 ** 2 + i ** 2 + d_ey ** 2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1 ** 2 - x ** 2 - y ** 2

    if z_sq < 0:
        return False, np.zeros(3), np.zeros(3)
    z = np.sqrt(z_sq)

    sol1 = p1 + x * ex + y * ey + z * ez
    sol2 = p1 + x * ex + y * ey - z * ez
    return True, sol1, sol2


@njit
def find_candidates_njit(all_pos, all_rad, neighbor_pairs,
                         L, diameters, max_cands, collision_tol):
    """
    KDTree 预筛选后的三球定位。
    neighbor_pairs: (P, 2) int 数组，每行是一对近邻粒子索引 (i, j)
    返回: (M, 5) 数组 [x, y, z, r, coord]
    """
    results  = np.zeros((max_cands, 5))
    count    = 0
    n_pairs  = len(neighbor_pairs)
    n_all    = len(all_pos)

    for d_idx in range(len(diameters)):
        r_new = diameters[d_idx] / 2.0

        for p_idx in range(n_pairs):
            i   = neighbor_pairs[p_idx, 0]
            j   = neighbor_pairs[p_idx, 1]
            p_i = all_pos[i]
            r_i = all_rad[i]
            p_j = p_i + pbc_diff_njit(all_pos[j], p_i, L)
            r_j = all_rad[j]

            # ij 对距离剪枝（KDTree 用的是未考虑 r_new 的阈值，这里精确过滤）
            d_ij = np.sqrt(np.sum((p_j - p_i) ** 2))
            if d_ij > (r_i + r_j + 2 * r_new + collision_tol):
                continue

            for k in range(n_all):
                if k == i or k == j:
                    continue
                p_k = p_i + pbc_diff_njit(all_pos[k], p_i, L)
                r_k = all_rad[k]

                valid, s1, s2 = solve_three_spheres_njit(
                    p_i, r_i, p_j, r_j, p_k, r_k, r_new)
                if not valid:
                    continue

                solutions = np.stack((s1, s2))
                for s_idx in range(2):
                    sol         = solutions[s_idx]
                    collision   = False
                    coordination = 0

                    for m in range(n_all):
                        dm   = pbc_diff_njit(all_pos[m], sol, L)
                        dist = np.sqrt(np.sum(dm ** 2))
                        gap  = dist - (all_rad[m] + r_new)

                        if gap < -collision_tol:
                            collision = True
                            break
                        if gap < collision_tol:
                            coordination += 1

                    if not collision:
                        results[count, 0:3] = sol % L
                        results[count, 3]   = r_new
                        results[count, 4]   = coordination
                        count += 1
                        if count >= max_cands:
                            return results[:count]

    return results[:count]


def get_pbc_center_of_mass(pos_array, L):
    """周期性边界条件下的质心计算"""
    theta   = 2 * np.pi * pos_array / L
    cos_sum = np.mean(np.cos(theta), axis=0)
    sin_sum = np.mean(np.sin(theta), axis=0)
    phi     = np.arctan2(sin_sum, cos_sum)
    phi     = np.where(phi < 0, phi + 2 * np.pi, phi)
    return (phi / (2 * np.pi)) * L


# =====================================================================
# 2. 模型架构
# =====================================================================

# ------------------------------------------------------------------
# 2a. 候选点编码器（左侧 MLP）
# ------------------------------------------------------------------
class CandidateEncoder(nn.Module):
    """
    将单个候选点的 5 维特征逐步升维到 embed_dim。
    层宽由 Config.candidate_mlp_layers 控制。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        in_dim = cfg.candidate_input_dim
        for out_dim in cfg.candidate_mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        # 末层输出维度应等于 embed_dim
        assert in_dim == cfg.embed_dim, (
            f"candidate_mlp_layers 末层 {in_dim} 须等于 embed_dim {cfg.embed_dim}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, n_cands, 5)
        返回: (batch, n_cands, embed_dim)
        """
        return self.net(x)


# ------------------------------------------------------------------
# 2b. 图编码器（右侧 GNN）
# ------------------------------------------------------------------
class GNNLayer(nn.Module):
    """
    单层消息传递：
      聚合：对邻居特征求均值
      更新：Linear(self_feat + agg_feat) + ReLU
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU()
        )

    def forward(self, node_feat, adj):
        """
        node_feat : (N, in_dim)
        adj        : (N, N) 邻接矩阵（float，已归一化或 0/1）
        返回       : (N, out_dim)
        """
        # 归一化聚合（防止度数差异导致数值爆炸）
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.matmul(adj, node_feat) / deg          # (N, in_dim)
        combined = torch.cat([node_feat, agg], dim=-1)    # (N, 2*in_dim)
        return self.update(combined)


class GraphEncoder(nn.Module):
    """
    将当前所有已放置粒子编码为固定维度的全局向量。

    流程：
      1. 节点投影: (N, 4) → (N, hidden)
      2. × gnn_layers 消息传递
      3. 全局池化: MaxPool + MeanPool → (2*hidden,)
      4. 线性投影: (2*hidden,) → (embed_dim,)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        hid = cfg.graph_hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.graph_input_dim, hid),
            nn.ReLU()
        )
        self.gnn_layers = nn.ModuleList(
            [GNNLayer(hid, hid) for _ in range(cfg.gnn_layers)]
        )
        self.output_proj = nn.Linear(hid * 2, cfg.embed_dim)
        self.edge_tol    = cfg.edge_tol

    def build_adjacency(self, pos, rad, L):
        """
        向量化构建邻接矩阵，替代 Python 双重循环。
        pos: (N, 3) numpy  |  rad: (N,) numpy
        返回: (N, N) FloatTensor
        """
        # (N, N, 3) 广播差值，PBC 修正
        diff = pos[:, None, :] - pos[None, :, :]        # (N, N, 3)
        diff -= np.round(diff / L) * L
        dist = np.linalg.norm(diff, axis=-1)            # (N, N)

        # 相切条件：dist <= r_i + r_j + tol
        touch = (dist <= rad[:, None] + rad[None, :] + self.edge_tol).astype(np.float32)
        np.fill_diagonal(touch, 0)                      # 自环置 0
        return torch.from_numpy(touch)

    def forward(self, pos, rad, L, adj_np, device):
        """
        pos    : (N, 3) numpy
        rad    : (N,)   numpy
        adj_np : (N, N) numpy float32，由 ConstructEnv 缓存并增量维护
        返回   : (embed_dim,) tensor
        """
        node_np = np.concatenate(
            [pos / L, (rad / rad.max()).reshape(-1, 1)], axis=1
        ).astype(np.float32)                            # (N, 4)

        node_feat = torch.from_numpy(node_np).to(device)
        adj       = torch.from_numpy(adj_np).to(device) # 直接用缓存，无需重建

        h = self.input_proj(node_feat)                  # (N, hid)
        for layer in self.gnn_layers:
            h = layer(h, adj)

        h_max  = h.max(dim=0)[0]
        h_mean = h.mean(dim=0)
        h_glob = torch.cat([h_max, h_mean], dim=-1)
        return self.output_proj(h_glob)                 # (embed_dim,)


# ------------------------------------------------------------------
# 2c. 融合解码器
# ------------------------------------------------------------------
class FusionDecoder(nn.Module):
    """
    将候选点嵌入与图全局嵌入融合，输出每个候选点的得分。
    层宽由 Config.fusion_layers 控制。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        in_dim = cfg.embed_dim * 2          # 拼接两侧 128 维
        for out_dim in cfg.fusion_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1)) # 最终输出标量得分
        self.net = nn.Sequential(*layers)

    def forward(self, cand_emb, graph_emb):
        """
        cand_emb  : (batch, n_cands, embed_dim)
        graph_emb : (batch, embed_dim)
        返回      : (batch, n_cands) 得分
        """
        n_cands   = cand_emb.size(1)
        g_exp     = graph_emb.unsqueeze(1).expand(-1, n_cands, -1)
        combined  = torch.cat([cand_emb, g_exp], dim=-1)   # (batch, n_cands, 2*embed_dim)
        scores    = self.net(combined).squeeze(-1)           # (batch, n_cands)
        return scores


# ------------------------------------------------------------------
# 2d. 整体策略网络
# ------------------------------------------------------------------
class PackingPolicy(nn.Module):
    """
    完整策略网络：
      CandidateEncoder + GraphEncoder + FusionDecoder
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cand_encoder  = CandidateEncoder(cfg)
        self.graph_encoder = GraphEncoder(cfg)
        self.fusion        = FusionDecoder(cfg)

    def forward(self, cand_feat, graph_pos, graph_rad, L, adj_np, mask):
        """
        单样本推理接口（采集时使用）。
        adj_np : (N, N) numpy，由 ConstructEnv 缓存维护
        """
        device    = cand_feat.device
        cand_emb  = self.cand_encoder(cand_feat)
        graph_emb = self.graph_encoder(graph_pos, graph_rad, L, adj_np, device).unsqueeze(0)
        scores    = self.fusion(cand_emb, graph_emb)
        return scores.masked_fill(mask == 0, float('-inf'))

    def batch_forward(self, obs_batch, mask_batch, samples, device):
        """
        批量推理接口（训练时使用）。
        samples 中每个元素须含 graph_pos / graph_rad / L / adj_np。
        """
        cand_emb = self.cand_encoder(obs_batch)                     # (B, M, embed_dim)

        graph_embs = []
        for s in samples:
            g = self.graph_encoder(s['graph_pos'], s['graph_rad'],
                                   s['L'], s['adj_np'], device)
            graph_embs.append(g)
        graph_emb_batch = torch.stack(graph_embs)                   # (B, embed_dim)

        scores = self.fusion(cand_emb, graph_emb_batch)             # (B, M)
        return scores.masked_fill(mask_batch == 0, float('-inf'))


# =====================================================================
# 3. 构造环境
# =====================================================================
class ConstructEnv:
    """
    粒子堆积构造环境。
    与原代码逻辑一致，去除 PPO 相关部分，
    新增 get_graph_data() 供图编码器使用。
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        avg_vol    = (4.0 / 3.0) * np.pi * (0.5 ** 3)
        total_vol  = cfg.target_N * avg_vol / cfg.target_phi
        self.L     = total_vol ** (1.0 / 3.0)
        self.reset()

    def reset(self):
        """初始化固定四面体底座（4个球）"""
        cfg = self.cfg
        r0, r1, r2, r3 = 0.5, 0.6, 0.4, 0.5
        p0 = np.array([self.L / 2, self.L / 2, self.L / 2])
        p1 = p0 + np.array([r0 + r1, 0, 0])

        d01 = r0 + r1
        d02 = r0 + r2
        d12 = r1 + r2
        x2  = (d02 ** 2 - d12 ** 2 + d01 ** 2) / (2 * d01)
        y2  = np.sqrt(max(0.0, d02 ** 2 - x2 ** 2))
        p2  = p0 + np.array([x2, y2, 0])

        valid, p3, _ = solve_three_spheres_njit(p0, r0, p1, r1, p2, r2, r3)
        if not valid:
            p3 = p0 + np.array([0, 0, r0 + r3])

        self.pos  = [p % self.L for p in [p0, p1, p2, p3]]
        self.rad  = [r0, r1, r2, r3]
        self.n    = 4
        self.current_candidates = []

        # 初始化缓存（全量构建，仅在 reset 时做一次）
        self._rebuild_cache()

        return self._get_obs()

    def _rebuild_cache(self):
        """全量构建 KDTree 和邻接矩阵缓存（仅 reset 时调用）"""
        pos_arr = np.array(self.pos, dtype=np.float64)
        rad_arr = np.array(self.rad, dtype=np.float64)
        self._kdtree = KDTree(pos_arr)
        self._adj_np = self._full_adj(pos_arr, rad_arr)

    def _full_adj(self, pos, rad):
        """全量计算邻接矩阵（numpy 向量化）"""
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= np.round(diff / self.L) * self.L
        dist = np.linalg.norm(diff, axis=-1)
        adj  = (dist <= rad[:, None] + rad[None, :] + self.cfg.edge_tol).astype(np.float32)
        np.fill_diagonal(adj, 0)
        return adj

    def _update_cache_incremental(self, new_pos, new_rad):
        """
        新增一个粒子后增量更新缓存，O(N) 而非 O(N²)。
        在 step() 中粒子加入 self.pos/rad 后立即调用。
        """
        old_pos = np.array(self.pos[:-1], dtype=np.float64)  # (N, 3)
        old_rad = np.array(self.rad[:-1], dtype=np.float64)  # (N,)

        # 新粒子到所有旧粒子的距离（含 PBC）
        diff  = old_pos - new_pos[None, :]
        diff -= np.round(diff / self.L) * self.L
        dists = np.linalg.norm(diff, axis=-1)                # (N,)
        touch = (dists <= old_rad + new_rad + self.cfg.edge_tol).astype(np.float32)

        # 扩展邻接矩阵：在右侧加一列，底部加一行
        N      = len(touch)
        new_adj = np.zeros((N + 1, N + 1), dtype=np.float32)
        new_adj[:N, :N] = self._adj_np
        new_adj[:N, N]  = touch
        new_adj[N, :N]  = touch
        self._adj_np    = new_adj

        # 重建 KDTree（插入操作 KDTree 不支持增量，但重建是 O(N log N) 而非 O(N²)）
        self._kdtree = KDTree(np.array(self.pos, dtype=np.float64))

    # ---- 内部方法 ----
    def _get_candidates(self):
        cfg     = self.cfg
        all_pos = np.array(self.pos, dtype=np.float64)
        all_rad = np.array(self.rad, dtype=np.float64)

        # 使用缓存的 KDTree，无需重建
        max_r_new = cfg.diameters.max() / 2.0
        search_r  = 2 * all_rad.max() + 2 * max_r_new + cfg.collision_tol
        pairs     = np.array(list(self._kdtree.query_pairs(r=search_r)), dtype=np.int64)

        # 若近邻对为空（粒子数极少时），退化为全量
        if len(pairs) == 0:
            n     = len(all_pos)
            pairs = np.array([[i, j] for i in range(n)
                              for j in range(i + 1, n)], dtype=np.int64)

        cands = find_candidates_njit(
            all_pos, all_rad, pairs,
            self.L, cfg.diameters, cfg.max_candidates, cfg.collision_tol
        )
        return cands.tolist()

    def _get_obs(self):
        """
        返回:
          obs  : (max_candidates, 5)  候选点特征（已归一化）
          mask : (max_candidates,)    有效掩码
        """
        cfg   = self.cfg
        cands = self._get_candidates()

        if len(cands) > 0:
            cands_np = np.array(cands)
            # 按配位数降序、半径降序排列
            idx      = np.lexsort((cands_np[:, 3], cands_np[:, 4]))[::-1]
            cands_np = cands_np[idx]
            self.current_candidates = cands_np.tolist()
        else:
            self.current_candidates = []

        obs  = np.zeros((cfg.max_candidates, 5), dtype=np.float32)
        mask = np.zeros(cfg.max_candidates, dtype=np.float32)

        n_valid = min(len(self.current_candidates), cfg.max_candidates)
        if n_valid > 0:
            c = np.array(self.current_candidates[:n_valid])
            # 以质心为原点，做 PBC 修正，缩放坐标
            center    = get_pbc_center_of_mass(np.array(self.pos), self.L)
            c[:, :3] -= center
            c[:, :3]  = c[:, :3] - np.round(c[:, :3] / self.L) * self.L
            c[:, :3] *= 0.5
            obs[:n_valid]  = c
            mask[:n_valid] = 1.0

        return obs, mask

    def get_graph_data(self):
        """返回当前粒子的位置、半径和缓存的邻接矩阵（供图编码器）"""
        return (np.array(self.pos, dtype=np.float64),
                np.array(self.rad, dtype=np.float64),
                self._adj_np.copy())

    def step(self, action_idx):
        """
        执行一步放置。
        返回: (obs, mask), reward, done
        """
        cfg = self.cfg

        # 无候选位置
        if action_idx >= len(self.current_candidates):
            obs, mask = self._get_obs()
            return (obs, mask), 0.0, True

        chosen = self.current_candidates[action_idx]
        pos_new = np.array(chosen[:3])
        r_new   = chosen[3]
        coord   = chosen[4]

        self.pos.append(pos_new)
        self.rad.append(r_new)
        self.n += 1

        # 增量更新缓存（O(N)，不重建整个树/矩阵）
        self._update_cache_incremental(pos_new, r_new)

        done = self.n >= cfg.max_particles
        obs, mask = self._get_obs()

        if np.sum(mask) == 0 and not done:
            done = True

        # 只在终局给奖励：最终堆积率 phi
        reward = self.get_phi() if done else 0.0

        return (obs, mask), reward, done

    def get_phi(self):
        """返回当前体积分数"""
        box_vol   = self.L ** 3
        total_vol = sum((4.0 / 3.0) * np.pi * (r ** 3) for r in self.rad)
        return total_vol / box_vol


# =====================================================================
# 4. 数据采集器
# =====================================================================

def _worker_collect_episode(args):
    """
    子进程中运行单局完整采集。
    worker 内部全程使用 CPU，避免多进程 CUDA 共享问题。
    args: (policy_state_dict, greedy, temperature, seed)
    """
    policy_state_dict, greedy, temperature, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg         = Config()
    cpu_device  = torch.device('cpu')

    # 在子进程内重建模型（CPU）
    if policy_state_dict is not None:
        policy = PackingPolicy(cfg)
        policy.load_state_dict(policy_state_dict)
        policy.eval()
    else:
        policy = None

    env = ConstructEnv(cfg)
    obs, mask = env.reset()
    traj_steps = []
    done = False

    while not done:
        graph_pos, graph_rad, adj_np = env.get_graph_data()
        obs_t  = torch.from_numpy(obs).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if policy is None:
            valid_idx = np.where(mask > 0)[0]
            if len(valid_idx) == 0:
                break
            action_idx = int(np.random.choice(valid_idx))
        else:
            with torch.no_grad():
                scores = policy(obs_t, graph_pos, graph_rad, env.L, adj_np, mask_t)
            scores_np = scores[0].numpy()
            valid_idx = np.where(mask > 0)[0]
            if len(valid_idx) == 0:
                break

            if greedy:
                action_idx = int(np.argmax(scores_np))
            else:
                valid_scores = scores_np[valid_idx]
                valid_scores -= valid_scores.max()
                probs = np.exp(valid_scores / temperature)
                probs /= probs.sum()
                action_idx = int(np.random.choice(valid_idx, p=probs))

        (next_obs, next_mask), reward, done = env.step(action_idx)

        traj_steps.append({
            'obs'       : obs,
            'mask'      : mask,
            'graph_pos' : graph_pos.copy(),
            'graph_rad' : graph_rad.copy(),
            'adj_np'    : adj_np,           # 缓存的邻接矩阵，训练直接用
            'L'         : env.L,
            'action'    : action_idx,
            'reward'    : reward,
        })

        obs, mask = next_obs, next_mask

    phi_final = env.get_phi()
    return {
        'steps'     : traj_steps,
        'phi_final' : phi_final,
        'final_pos' : np.array(env.pos, dtype=np.float64).copy(),
        'final_rad' : np.array(env.rad, dtype=np.float64).copy(),
        'L'         : env.L,
    }


class DataCollector:
    """
    使用给定策略（或随机策略）采集完整堆积轨迹。

    每条轨迹记录：
      steps: list of (obs, mask, graph_pos, graph_rad, action_idx, reward)
      phi_final: 最终堆积率
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def collect(self, policy, n_samples, greedy=False):
        """
        并行采集 n_samples 条轨迹。
        每个子进程独立跑一局完整 episode（CPU 推理），主进程负责汇总。

        policy  : PackingPolicy 或 None（None 则随机策略）
        greedy  : True=贪婪; False=温度采样
        返回    : list of trajectory dict
        """
        cfg = self.cfg

        # 将模型权重移到 CPU 后序列化传给子进程（避免 CUDA 多进程问题）
        if policy is not None:
            state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        else:
            state_dict = None

        # 为每个 episode 分配独立随机种子，保证可复现且互不干扰
        seeds     = np.random.randint(0, 2**31, size=n_samples).tolist()
        args_list = [
            (state_dict, greedy, cfg.temperature, seed)
            for seed in seeds
        ]

        # 并行执行
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(cfg.num_workers, n_samples)) as pool:
            trajectories = pool.map(_worker_collect_episode, args_list)

        # 打印统计
        phis = [t['phi_final'] for t in trajectories]
        print(f"  [采集完成] {n_samples} 条  "
              f"phi: mean={np.mean(phis):.4f}  "
              f"max={np.max(phis):.4f}  "
              f"min={np.min(phis):.4f}")

        return trajectories


# =====================================================================
# 5. 训练器（REINFORCE with running baseline）
# =====================================================================
class Trainer:
    """
    使用 REINFORCE 算法训练策略网络。
    奖励信号：只在终局给 phi_final，所有步共享同一个 return。
    Baseline：运行平均（指数平滑）
    """
    def __init__(self, policy: PackingPolicy, cfg: Config):
        self.policy    = policy
        self.cfg       = cfg
        self.optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
        self.baseline  = 0.0

    def _compute_returns(self, traj):
        """
        终局奖励：每一步的 return 都等于最终 phi。
        含义：这局轨迹里的每个动作，都对最终结果负责。
        """
        phi = traj['phi_final']
        return [phi] * len(traj['steps'])

    def train(self, trajectories):
        """对一批轨迹执行多个 epoch 的 REINFORCE 更新"""
        cfg    = self.cfg
        device = cfg.device
        policy = self.policy

        # 收集所有 (obs, mask, graph_pos, graph_rad, L, action, return)
        all_samples = []
        for traj in trajectories:
            returns = self._compute_returns(traj)
            for step, G in zip(traj['steps'], returns):
                all_samples.append({**step, 'return': G})

        if len(all_samples) == 0:
            return 0.0

        # 更新 baseline（指数平滑）
        all_returns   = np.array([s['return'] for s in all_samples])
        avg_return    = all_returns.mean()
        ret_std       = all_returns.std() + 1e-8   # 用于 advantage 归一化
        self.baseline = ((1 - cfg.baseline_alpha) * self.baseline
                         + cfg.baseline_alpha * avg_return)

        total_loss = 0.0
        n_updates  = 0

        for epoch in range(cfg.train_epochs):
            np.random.shuffle(all_samples)

            for start in range(0, len(all_samples), cfg.batch_size):
                mb = all_samples[start: start + cfg.batch_size]
                if len(mb) == 0:
                    continue

                # 批量组装 obs / mask / action / advantage
                obs_batch  = torch.from_numpy(
                    np.stack([s['obs']  for s in mb])).to(device)   # (B, M, 5)
                mask_batch = torch.from_numpy(
                    np.stack([s['mask'] for s in mb])).to(device)   # (B, M)
                actions    = torch.tensor(
                    [s['action'] for s in mb], dtype=torch.long, device=device)  # (B,)
                advantages = torch.tensor(
                    [(s['return'] - self.baseline) / ret_std for s in mb],
                    dtype=torch.float32, device=device)              # (B,)

                # 批量前向传播（CandidateEncoder + FusionDecoder 全批量，GraphEncoder 逐样本）
                scores    = policy.batch_forward(obs_batch, mask_batch, mb, device)  # (B, M)
                log_probs = torch.nn.functional.log_softmax(scores, dim=-1)          # (B, M)
                log_pa    = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)     # (B,)

                batch_loss = -(log_pa * advantages).mean()
                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()
                n_updates  += 1

        return total_loss / max(n_updates, 1)


# =====================================================================
# 6. 主训练循环
# =====================================================================
def train():
    cfg    = Config()
    device = cfg.device
    print(f"使用设备: {device}")
    print(f"目标粒子数: {cfg.target_N}  目标体积分数: {cfg.target_phi}")

    # 初始化策略网络
    policy    = PackingPolicy(cfg).to(device)
    collector = DataCollector(cfg)
    trainer   = Trainer(policy, cfg)

    # 打开日志
    log_f  = open(cfg.log_file, 'w', newline='')
    writer = csv.writer(log_f)
    writer.writerow(["Iteration", "PhiMean", "PhiMax", "PhiMin",
                     "AvgSteps", "Loss", "Baseline"])

    for iteration in range(cfg.num_iterations):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"迭代 {iteration+1}/{cfg.num_iterations}")

        # ---- 数据采集 ----
        use_policy = None if iteration == 0 else policy
        greedy     = False   # 采集时始终用随机采样以保持探索
        print(f"  采集 {cfg.samples_per_iter} 个样本 "
              f"({'随机策略' if use_policy is None else '模型策略'})...")
        trajs = collector.collect(use_policy, cfg.samples_per_iter,
                                  greedy=greedy)

        # ---- 统计 ----
        phis      = [t['phi_final'] for t in trajs]
        steps_cnt = [len(t['steps']) for t in trajs]
        phi_mean  = np.mean(phis)
        phi_max   = np.max(phis)
        phi_min   = np.min(phis)
        avg_steps = np.mean(steps_cnt)
        print(f"  phi: mean={phi_mean:.4f}  max={phi_max:.4f}  "
              f"min={phi_min:.4f}  avg_steps={avg_steps:.1f}")

        # ---- 训练 ----
        print(f"  训练 {cfg.train_epochs} epoch ...")
        loss = trainer.train(trajs)
        print(f"  loss={loss:.4f}  baseline={trainer.baseline:.4f}  "
              f"耗时={time.time()-t0:.1f}s")

        # ---- 记录 ----
        writer.writerow([iteration + 1, phi_mean, phi_max, phi_min,
                         avg_steps, loss, trainer.baseline])
        log_f.flush()

        # ---- 保存 checkpoint ----
        if (iteration + 1) % cfg.save_interval == 0:
            ckpt = f"{cfg.ckpt_prefix}_iter{iteration+1}.pth"
            torch.save(policy.state_dict(), ckpt)
            print(f"  已保存 checkpoint: {ckpt}")

    log_f.close()
    # 保存最终模型
    final_ckpt = f"{cfg.ckpt_prefix}_final.pth"
    torch.save(policy.state_dict(), final_ckpt)
    print(f"\n训练完成！最终模型已保存至 {final_ckpt}")

    # 自动评测
    evaluate(policy, cfg)


# =====================================================================
# 7. 评测函数（训练结束后自动调用）
# =====================================================================
def evaluate(policy: PackingPolicy, cfg: Config):
    """
    用贪婪策略跑 eval_episodes 局，统计评测指标并输出报告文件。
    同时保存 phi 最高的那局堆积到 .conf 文件。

    报告包含：
      - 粒子数、体积分数 phi、平均配位数 Z、多分散性 PDI
      - 各局明细 + 汇总统计
    """
    device = cfg.device
    policy.eval()
    collector = DataCollector(cfg)

    print(f"\n{'='*60}")
    print(f"开始评测（贪婪策略，共 {cfg.eval_episodes} 局）...")

    trajs = collector.collect(policy, n_samples=cfg.eval_episodes, greedy=True)

    # ---- 对每局重放，拿到完整环境状态以计算详细指标 ----
    records   = []
    best_phi  = -1.0
    best_env  = None

    for traj in trajs:
        # 直接从轨迹中取最终状态，避免重放时随机候选集不一致导致的 bug
        pos_arr = traj['final_pos']
        rad_arr = traj['final_rad']
        L_traj  = traj['L']
        n       = len(pos_arr)

        # 体积分数
        phi = traj['phi_final']

        # 平均配位数 Z：统计每对粒子是否相切
        coord_counts = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos_arr[j] - pos_arr[i]
                diff -= np.round(diff / L_traj) * L_traj
                dist = np.linalg.norm(diff)
                if dist <= rad_arr[i] + rad_arr[j] + cfg.edge_tol:
                    coord_counts[i] += 1
                    coord_counts[j] += 1
        z_mean = coord_counts.mean()

        # 多分散性 PDI
        diams  = rad_arr * 2.0
        pdi    = diams.std() / diams.mean() if diams.mean() > 0 else 0.0

        records.append({
            'n'    : n,
            'phi'  : phi,
            'z'    : z_mean,
            'pdi'  : pdi,
        })

        if phi > best_phi:
            best_phi     = phi
            best_pos_arr = pos_arr
            best_rad_arr = rad_arr
            best_L       = L_traj

    # ---- 汇总统计 ----
    phis = [r['phi'] for r in records]
    ns   = [r['n']   for r in records]
    zs   = [r['z']   for r in records]
    pdis = [r['pdi'] for r in records]

    # ---- 写评测报告 ----
    with open(cfg.eval_report_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"评测报告  模型: {cfg.ckpt_prefix}_final.pth\n")
        f.write(f"评测局数: {cfg.eval_episodes}    目标粒子数: {cfg.target_N}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"{'指标':<20}{'均值':>10}{'最大':>10}{'最小':>10}{'标准差':>10}\n")
        f.write("-" * 60 + "\n")
        for name, vals in [("体积分数 phi", phis),
                            ("粒子数 N",     ns),
                            ("平均配位数 Z", zs),
                            ("多分散性 PDI", pdis)]:
            arr = np.array(vals, dtype=float)
            f.write(f"{name:<20}"
                    f"{arr.mean():>10.4f}"
                    f"{arr.max():>10.4f}"
                    f"{arr.min():>10.4f}"
                    f"{arr.std():>10.4f}\n")

        f.write("\n各局明细:\n")
        f.write(f"{'局':<6}{'N':>6}{'phi':>8}{'Z':>8}{'PDI':>8}\n")
        f.write("-" * 40 + "\n")
        for i, r in enumerate(records):
            f.write(f"{i+1:<6}{r['n']:>6}{r['phi']:>8.4f}"
                    f"{r['z']:>8.2f}{r['pdi']:>8.4f}\n")

    print(f"  评测完成  phi: mean={np.mean(phis):.4f}  "
          f"max={np.max(phis):.4f}  min={np.min(phis):.4f}")
    print(f"  报告已保存: {cfg.eval_report_file}")

    # ---- 保存 phi 最高的堆积 ----
    if best_phi > 0:
        with open(cfg.eval_conf_file, 'w') as f:
            for p, r in zip(best_pos_arr, best_rad_arr):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
            f.write(f"{best_L:.6f} {best_L:.6f} {best_L:.6f}\n")
        print(f"  最优堆积已保存: {cfg.eval_conf_file}  (phi={best_phi:.4f})")

    policy.train()


# =====================================================================
# 8. 生成测试（加载已训练模型，贪婪放置）
# =====================================================================
def generate_packing(model_path, output_file="final_packing_v6.conf"):
    """加载训练好的模型，贪婪放置，输出 .conf 文件"""
    cfg    = Config()
    device = cfg.device

    policy = PackingPolicy(cfg).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy.eval()

    collector = DataCollector(cfg)
    trajs     = collector.collect(policy, n_samples=1, greedy=True)
    traj      = trajs[0]

    # 重放轨迹拿到最终环境状态
    env = ConstructEnv(cfg)
    env.reset()
    for step in traj['steps']:
        env.step(step['action'])

    phi = env.get_phi()
    print(f"生成完成  粒子数={env.n}  phi={phi:.4f}")
    print(f"输出文件: {output_file}")

    with open(output_file, 'w') as f:
        for p, r in zip(env.pos, env.rad):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
        f.write(f"{env.L:.6f} {env.L:.6f} {env.L:.6f}\n")


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()
    # 训练完成后可调用：
    # generate_packing("construct_v6.0_final.pth")
