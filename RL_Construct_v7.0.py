"""
RL_Construct_v7.0.py
====================
粒子堆积强化学习 —— 候选点缓存优化版本

相比 v6.0 的主要改动：
  1. 删除 KDTree，不再每步重建
  2. 维护 _candidate_set 和 _triplet_set 两个集合
  3. 候选点生成从全量重算改为增量维护
  4. 拆出单三元组处理函数 _process_triplet()
"""

import csv
import json
import pathlib
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from itertools import combinations
from numba import njit


# =====================================================================
# 工具：标准输出 + 文件同时写入
# =====================================================================
class _Tee:
    """将 stdout 同时输出到终端和日志文件。"""
    def __init__(self, *files):
        self._files = files

    def write(self, data):
        for f in self._files:
            f.write(data)

    def flush(self):
        for f in self._files:
            f.flush()

# =====================================================================
# 0. 全局超参数配置
# =====================================================================
class Config:
    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 物理 / 环境参数 ----
    target_N      = 50
    target_phi    = 0.72
    diam_min      = 0.7
    diam_max      = 1.35
    diam_step     = 0.05
    diameters     = np.arange(0.7, 1.40, 0.05)
    max_candidates = 1000
    collision_tol  = 0.05
    edge_tol       = 0.05
    max_particles  = 200

    # ---- 候选点编码器 (左侧 MLP) ----
    candidate_input_dim  = 5
    candidate_mlp_layers = [32, 64, 128]

    # ---- 图编码器 (右侧 GNN) ----
    graph_input_dim   = 4
    graph_hidden_dim  = 128
    gnn_layers        = 3

    # ---- 融合解码器 ----
    embed_dim         = 128
    fusion_layers     = [256, 128]

    # ---- 训练超参数 ----
    num_workers       = 8
    num_iterations    = 20
    samples_per_iter  = 20
    train_epochs      = 5
    batch_size        = 256
    lr                = 3e-4
    gamma             = 0.99
    temperature       = 1.0

    # ---- 输出 ----
    log_file      = "v7.0_train_log.csv"
    ckpt_prefix   = "construct_v7.0"
    save_interval = 5

    # ---- 评测 ----
    eval_episodes    = 30
    eval_temperature = 1.0
    eval_report_file = "v7.0_eval_report.txt"
    eval_conf_file   = "v7.0_best_packing.conf"


# =====================================================================
# 1. 物理核心（Numba JIT）
# =====================================================================
@njit(inline='always')
def pbc_diff_njit(p1, p2, L):
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
def check_collision_njit(sol, r_new, all_pos, all_rad, L, collision_tol):
    """
    检查候选点 sol 与所有已有粒子是否碰撞。
    返回: (collision, coordination)
    """
    collision    = False
    coordination = 0
    for m in range(len(all_pos)):
        dm   = pbc_diff_njit(all_pos[m], sol, L)
        dist = np.sqrt(np.sum(dm ** 2))
        gap  = dist - (all_rad[m] + r_new)
        if gap < -collision_tol:
            collision = True
            break
        if gap < collision_tol:
            coordination += 1
    return collision, coordination


@njit
def check_single_collision_njit(sol, r_new, new_pos, new_rad, L, collision_tol):
    """
    只检查候选点 sol 与单个新粒子是否碰撞（增量过滤用）。
    返回: (collision, touching)
    """
    dm   = pbc_diff_njit(new_pos, sol, L)
    dist = np.sqrt(np.sum(dm ** 2))
    gap  = dist - (new_rad + r_new)
    collision = gap < -collision_tol
    touching  = gap < collision_tol
    return collision, touching


def get_pbc_center_of_mass(pos_array, L):
    theta   = 2 * np.pi * pos_array / L
    cos_sum = np.mean(np.cos(theta), axis=0)
    sin_sum = np.mean(np.sin(theta), axis=0)
    phi     = np.arctan2(sin_sum, cos_sum)
    phi     = np.where(phi < 0, phi + 2 * np.pi, phi)
    return (phi / (2 * np.pi)) * L


# =====================================================================
# 2. 模型架构（与 v6.0 完全一致）
# =====================================================================

class CandidateEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        in_dim = cfg.candidate_input_dim
        for out_dim in cfg.candidate_mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        assert in_dim == cfg.embed_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU()
        )

    def forward(self, node_feat, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.matmul(adj, node_feat) / deg
        combined = torch.cat([node_feat, agg], dim=-1)
        return self.update(combined)


class GraphEncoder(nn.Module):
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

    def forward(self, pos, rad, L, adj_np, device):
        node_np = np.concatenate(
            [pos / L, (rad / rad.max()).reshape(-1, 1)], axis=1
        ).astype(np.float32)

        node_feat = torch.from_numpy(node_np).to(device)
        adj       = torch.from_numpy(adj_np).to(device)

        h = self.input_proj(node_feat)
        for layer in self.gnn_layers:
            h = layer(h, adj)

        h_max  = h.max(dim=0)[0]
        h_mean = h.mean(dim=0)
        h_glob = torch.cat([h_max, h_mean], dim=-1)
        return self.output_proj(h_glob)


class FusionDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        in_dim = cfg.embed_dim * 2
        for out_dim in cfg.fusion_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, cand_emb, graph_emb):
        n_cands  = cand_emb.size(1)
        g_exp    = graph_emb.unsqueeze(1).expand(-1, n_cands, -1)
        combined = torch.cat([cand_emb, g_exp], dim=-1)
        scores   = self.net(combined).squeeze(-1)
        return scores


class PackingPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cand_encoder  = CandidateEncoder(cfg)
        self.graph_encoder = GraphEncoder(cfg)
        self.fusion        = FusionDecoder(cfg)

    def forward(self, cand_feat, graph_pos, graph_rad, L, adj_np, mask):
        device    = cand_feat.device
        cand_emb  = self.cand_encoder(cand_feat)
        graph_emb = self.graph_encoder(graph_pos, graph_rad, L, adj_np, device).unsqueeze(0)
        scores    = self.fusion(cand_emb, graph_emb)
        return scores.masked_fill(mask == 0, float('-inf'))

    def batch_forward(self, obs_batch, mask_batch, samples, device):
        cand_emb = self.cand_encoder(obs_batch)

        graph_embs = []
        for s in samples:
            g = self.graph_encoder(s['graph_pos'], s['graph_rad'],
                                   s['L'], s['adj_np'], device)
            graph_embs.append(g)
        graph_emb_batch = torch.stack(graph_embs)

        scores = self.fusion(cand_emb, graph_emb_batch)
        return scores.masked_fill(mask_batch == 0, float('-inf'))


# =====================================================================
# 3. 构造环境（v7.0 核心改动）
# =====================================================================
class ConstructEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        avg_vol   = (4.0 / 3.0) * np.pi * (0.5 ** 3)
        total_vol = cfg.target_N * avg_vol / cfg.target_phi
        self.L    = total_vol ** (1.0 / 3.0)
        self.reset()

    def reset(self):
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

        self.pos = [p % self.L for p in [p0, p1, p2, p3]]
        self.rad = [r0, r1, r2, r3]
        self.n   = 4
        self.current_candidates = []

        # 邻接矩阵缓存（保留，供 GNN 使用）
        self._adj_np = self._full_adj(
            np.array(self.pos, dtype=np.float64),
            np.array(self.rad, dtype=np.float64)
        )

        # ---- v7.0 新增：候选点集合 & 三元组集合 ----
        # _candidate_set: cand_id -> np.array([x,y,z,r,coord])
        # _triplet_set:   (i,j,k) -> set of cand_id
        # _cand_to_triplets: cand_id -> set of (i,j,k)  反向索引，加速删除
        self._candidate_set    = {}
        self._triplet_set      = {}
        self._cand_to_triplets = {}
        self._cand_counter     = 0

        # 全量初始化：枚举初始4个球的所有三元组
        self._init_sets()

        return self._get_obs()

    # ------------------------------------------------------------------
    # 邻接矩阵（保留，供 GNN 使用）
    # ------------------------------------------------------------------
    def _full_adj(self, pos, rad):
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= np.round(diff / self.L) * self.L
        dist = np.linalg.norm(diff, axis=-1)
        adj  = (dist <= rad[:, None] + rad[None, :] + self.cfg.edge_tol).astype(np.float32)
        np.fill_diagonal(adj, 0)
        return adj

    def _update_adj_incremental(self, new_pos, new_rad):
        old_pos = np.array(self.pos[:-1], dtype=np.float64)
        old_rad = np.array(self.rad[:-1], dtype=np.float64)
        diff    = old_pos - new_pos[None, :]
        diff   -= np.round(diff / self.L) * self.L
        dists   = np.linalg.norm(diff, axis=-1)
        touch   = (dists <= old_rad + new_rad + self.cfg.edge_tol).astype(np.float32)
        N       = len(touch)
        new_adj = np.zeros((N + 1, N + 1), dtype=np.float32)
        new_adj[:N, :N] = self._adj_np
        new_adj[:N, N]  = touch
        new_adj[N, :N]  = touch
        self._adj_np    = new_adj

    # ------------------------------------------------------------------
    # v7.0 核心：候选点集合 & 三元组集合的维护
    # ------------------------------------------------------------------
    def _new_cand_id(self):
        cid = self._cand_counter
        self._cand_counter += 1
        return cid

    def _process_triplet(self, i, j, k):
        """
        对三元组 (i,j,k) × 所有直径做三球定位 + 碰撞检测。
        将有效候选点加入 _candidate_set，并更新 _triplet_set。
        """
        cfg     = self.cfg
        all_pos = np.array(self.pos, dtype=np.float64)
        all_rad = np.array(self.rad, dtype=np.float64)
        p_i, r_i = all_pos[i], all_rad[i]
        p_j, r_j = all_pos[j], all_rad[j]
        p_k, r_k = all_pos[k], all_rad[k]

        # PBC 修正：以 p_i 为原点
        p_j = p_i + pbc_diff_njit(p_j, p_i, self.L)
        p_k = p_i + pbc_diff_njit(p_k, p_i, self.L)

        triplet_cands = set()

        for r_new in cfg.diameters / 2.0:
            valid, sol1, sol2 = solve_three_spheres_njit(p_i, r_i, p_j, r_j, p_k, r_k, r_new)
            if not valid:
                continue

            for sol in (sol1, sol2):
                collision, coord = check_collision_njit(
                    sol, r_new, all_pos, all_rad, self.L, cfg.collision_tol)
                if not collision:
                    cid  = self._new_cand_id()
                    feat = np.array([*(sol % self.L), r_new, coord], dtype=np.float32)
                    self._candidate_set[cid]    = feat
                    self._cand_to_triplets[cid] = {(i, j, k)}
                    triplet_cands.add(cid)

        if triplet_cands:
            self._triplet_set[(i, j, k)] = triplet_cands

    def _init_sets(self):
        """初始化：枚举初始 n 个球的所有三元组"""
        for i, j, k in combinations(range(self.n), 3):
            self._process_triplet(i, j, k)

    def _add_new_triplets(self, new_idx):
        """
        新粒子 new_idx 加入后，枚举所有包含它的新三元组并处理。
        新三元组形如 (new_idx, j, k)，j < k < new_idx。
        """
        for j, k in combinations(range(new_idx), 2):
            self._process_triplet(new_idx, j, k)

    def _filter_candidates(self, new_pos, new_rad):
        """
        新粒子加入后，过滤与其碰撞的旧候选点。
        同时更新 _cand_to_triplets 和 _triplet_set。
        """
        new_pos_np = np.array(new_pos, dtype=np.float64)
        to_delete  = []

        for cid, feat in self._candidate_set.items():
            sol   = feat[:3].astype(np.float64)
            r_new = float(feat[3])
            collision, touching = check_single_collision_njit(
                sol, r_new, new_pos_np, new_rad, self.L, self.cfg.collision_tol)

            if collision:
                to_delete.append(cid)
            elif touching:
                # 新粒子与该候选点相切，配位数 +1
                feat[4] += 1

        for cid in to_delete:
            self._remove_candidate(cid)

    def _remove_candidate(self, cid):
        """
        从 _candidate_set 删除候选点，并清理对应三元组的引用。
        若某三元组的候选点全部被删除，则从 _triplet_set 中剔除。
        """
        if cid not in self._candidate_set:
            return

        del self._candidate_set[cid]

        for triplet in self._cand_to_triplets.get(cid, set()):
            if triplet in self._triplet_set:
                self._triplet_set[triplet].discard(cid)
                if not self._triplet_set[triplet]:
                    del self._triplet_set[triplet]

        del self._cand_to_triplets[cid]

    # ------------------------------------------------------------------
    # 环境接口
    # ------------------------------------------------------------------
    def _get_candidates(self):
        """直接从 _candidate_set 读取，不再调用 find_candidates_njit"""
        return list(self._candidate_set.values())

    def get_graph_data(self):
        return (np.array(self.pos, dtype=np.float64),
                np.array(self.rad, dtype=np.float64),
                self._adj_np.copy())

    def _get_obs(self):
        cfg   = self.cfg
        cands = self._get_candidates()

        if len(cands) > 0:
            cands_np = np.array(cands)
            idx      = np.lexsort((cands_np[:, 3], cands_np[:, 4]))[::-1]
            cands_np = cands_np[idx]
            self.current_candidates = cands_np.tolist()
        else:
            self.current_candidates = []

        obs  = np.zeros((cfg.max_candidates, 5), dtype=np.float32)
        mask = np.zeros(cfg.max_candidates, dtype=np.float32)

        n_valid = min(len(self.current_candidates), cfg.max_candidates)
        if n_valid > 0:
            c      = np.array(self.current_candidates[:n_valid])
            center = get_pbc_center_of_mass(np.array(self.pos), self.L)
            c[:, :3] -= center
            c[:, :3]  = c[:, :3] - np.round(c[:, :3] / self.L) * self.L
            c[:, :3] *= 0.5
            obs[:n_valid]  = c
            mask[:n_valid] = 1.0

        return obs, mask

    def step(self, action_idx):
        cfg = self.cfg

        if action_idx >= len(self.current_candidates):
            obs, mask = self._get_obs()
            return (obs, mask), 0.0, True

        chosen  = self.current_candidates[action_idx]
        pos_new = np.array(chosen[:3])
        r_new   = chosen[3]

        # 找到被选中候选点对应的 cand_id 并删除
        chosen_feat = np.array(chosen, dtype=np.float32)
        for cid, feat in list(self._candidate_set.items()):
            if np.allclose(feat, chosen_feat, atol=1e-5):
                self._remove_candidate(cid)
                break

        self.pos.append(pos_new)
        self.rad.append(r_new)
        self.n += 1

        # 增量更新邻接矩阵
        self._update_adj_incremental(pos_new, r_new)

        # ---- v7.0 增量维护 ----
        # 1. 过滤与新粒子碰撞的旧候选点
        self._filter_candidates(pos_new, r_new)
        # 2. 新增包含新粒子的三元组
        self._add_new_triplets(self.n - 1)

        done = self.n >= cfg.max_particles
        obs, mask = self._get_obs()

        if np.sum(mask) == 0 and not done:
            done = True

        reward = self.get_phi() if done else 0.0
        return (obs, mask), reward, done

    def get_phi(self):
        box_vol   = self.L ** 3
        total_vol = sum((4.0 / 3.0) * np.pi * (r ** 3) for r in self.rad)
        return total_vol / box_vol


# =====================================================================
# 4. 数据采集器（与 v6.0 一致）
# =====================================================================
def _worker_collect_episode(args):
    policy_state_dict, greedy, temperature, seed = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg        = Config()
    cpu_device = torch.device('cpu')

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
            'adj_np'    : adj_np,
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
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def collect(self, policy, n_samples, greedy=False):
        cfg = self.cfg

        if policy is not None:
            state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
        else:
            state_dict = None

        seeds     = np.random.randint(0, 2**31, size=n_samples).tolist()
        args_list = [
            (state_dict, greedy, cfg.temperature, seed)
            for seed in seeds
        ]

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(cfg.num_workers, n_samples)) as pool:
            trajectories = pool.map(_worker_collect_episode, args_list)

        phis = [t['phi_final'] for t in trajectories]
        print(f"  [采集完成] {n_samples} 条  "
              f"phi: mean={np.mean(phis):.4f}  "
              f"max={np.max(phis):.4f}  "
              f"min={np.min(phis):.4f}")

        return trajectories


# =====================================================================
# 5. 训练器（与 v6.0 一致）
# =====================================================================
class Trainer:
    def __init__(self, policy: PackingPolicy, cfg: Config):
        self.policy    = policy
        self.cfg       = cfg
        self.optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    def _compute_returns(self, traj):
        phi = traj['phi_final']
        return [phi] * len(traj['steps'])

    def train(self, trajectories):
        cfg    = self.cfg
        device = cfg.device
        policy = self.policy

        all_samples = []
        for traj in trajectories:
            returns = self._compute_returns(traj)
            for step, G in zip(traj['steps'], returns):
                all_samples.append({**step, 'return': G})

        if len(all_samples) == 0:
            return 0.0

        all_returns = np.array([s['return'] for s in all_samples])
        ret_mean    = all_returns.mean()
        ret_std     = all_returns.std() + 1e-8

        total_loss = 0.0
        n_updates  = 0

        for epoch in range(cfg.train_epochs):
            np.random.shuffle(all_samples)

            for start in range(0, len(all_samples), cfg.batch_size):
                mb = all_samples[start: start + cfg.batch_size]
                if len(mb) == 0:
                    continue

                obs_batch  = torch.from_numpy(
                    np.stack([s['obs']  for s in mb])).to(device)
                mask_batch = torch.from_numpy(
                    np.stack([s['mask'] for s in mb])).to(device)
                actions    = torch.tensor(
                    [s['action'] for s in mb], dtype=torch.long, device=device)
                advantages = torch.tensor(
                    [(s['return'] - ret_mean) / ret_std for s in mb],
                    dtype=torch.float32, device=device)

                scores    = policy.batch_forward(obs_batch, mask_batch, mb, device)
                log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                log_pa    = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

                batch_loss = -(log_pa * advantages).mean()
                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()
                n_updates  += 1

        return total_loss / max(n_updates, 1)


# =====================================================================
# 实验目录管理
# =====================================================================
def _create_experiment_dir() -> pathlib.Path:
    """在 experiments/ 下创建下一个可用的 run_NNN 目录。"""
    base = pathlib.Path("experiments")
    base.mkdir(exist_ok=True)
    existing = sorted(base.glob("run_*"))
    next_id  = len(existing) + 1
    exp_dir  = base / f"run_{next_id:03d}"
    exp_dir.mkdir()
    return exp_dir


def _save_config(exp_dir: pathlib.Path):
    """将 Config 所有超参序列化到 config.json。"""
    d = {}
    for k in sorted(vars(Config)):
        if k.startswith('_'):
            continue
        v = getattr(Config, k)
        if callable(v):
            continue
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, torch.device):
            d[k] = str(v)
        else:
            d[k] = v
    with open(exp_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


# =====================================================================
# 6. 主训练循环
# =====================================================================
def train():
    cfg    = Config()
    device = cfg.device

    # ------------------------------------------------------------------
    # 创建实验目录，重定向所有输出路径
    # ------------------------------------------------------------------
    exp_dir = _create_experiment_dir()
    _save_config(exp_dir)

    cfg.log_file         = str(exp_dir / "train_log.csv")
    cfg.ckpt_prefix      = str(exp_dir / "construct_v7.0")
    cfg.eval_report_file = str(exp_dir / "eval_report.txt")
    cfg.eval_conf_file   = str(exp_dir / "best_packing.conf")

    # 同时写入终端和 run.log
    log_txt    = open(exp_dir / "run.log", 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_txt)

    try:
        print(f"实验目录: {exp_dir}")
        print(f"使用设备: {device}")
        print(f"目标粒子数: {cfg.target_N}  目标体积分数: {cfg.target_phi}")

        policy    = PackingPolicy(cfg).to(device)
        collector = DataCollector(cfg)
        trainer   = Trainer(policy, cfg)

        log_f  = open(cfg.log_file, 'w', newline='')
        writer = csv.writer(log_f)
        writer.writerow(["Iteration", "PhiMean", "PhiMax", "PhiMin",
                         "AvgSteps", "Loss"])

        for iteration in range(cfg.num_iterations):
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"迭代 {iteration+1}/{cfg.num_iterations}")

            use_policy = None if iteration == 0 else policy
            greedy     = False
            print(f"  采集 {cfg.samples_per_iter} 个样本 "
                  f"({'随机策略' if use_policy is None else '模型策略'})...")
            trajs = collector.collect(use_policy, cfg.samples_per_iter, greedy=greedy)

            phis      = [t['phi_final'] for t in trajs]
            steps_cnt = [len(t['steps']) for t in trajs]
            phi_mean  = np.mean(phis)
            phi_max   = np.max(phis)
            phi_min   = np.min(phis)
            avg_steps = np.mean(steps_cnt)
            print(f"  phi: mean={phi_mean:.4f}  max={phi_max:.4f}  "
                  f"min={phi_min:.4f}  avg_steps={avg_steps:.1f}")

            print(f"  训练 {cfg.train_epochs} epoch ...")
            loss = trainer.train(trajs)
            print(f"  loss={loss:.4f}  耗时={time.time()-t0:.1f}s")

            writer.writerow([iteration + 1, phi_mean, phi_max, phi_min,
                             avg_steps, loss])
            log_f.flush()

            if (iteration + 1) % cfg.save_interval == 0:
                ckpt = f"{cfg.ckpt_prefix}_iter{iteration+1}.pth"
                torch.save(policy.state_dict(), ckpt)
                print(f"  已保存 checkpoint: {ckpt}")

        log_f.close()
        final_ckpt = f"{cfg.ckpt_prefix}_final.pth"
        torch.save(policy.state_dict(), final_ckpt)
        print(f"\n训练完成！最终模型已保存至 {final_ckpt}")

        evaluate(policy, cfg)

    finally:
        sys.stdout = sys.__stdout__
        log_txt.close()


# =====================================================================
# 7. 评测函数（与 v6.0 一致）
# =====================================================================
def evaluate(policy: PackingPolicy, cfg: Config):
    device = cfg.device
    policy.eval()

    orig_temp       = cfg.temperature
    cfg.temperature = cfg.eval_temperature
    collector = DataCollector(cfg)

    print(f"\n{'='*60}")
    print(f"开始评测（采样策略 T={cfg.eval_temperature}，共 {cfg.eval_episodes} 局）...")

    trajs = collector.collect(policy, n_samples=cfg.eval_episodes, greedy=False)
    cfg.temperature = orig_temp

    records  = []
    best_phi = -1.0

    for traj in trajs:
        pos_arr = traj['final_pos']
        rad_arr = traj['final_rad']
        L_traj  = traj['L']
        n       = len(pos_arr)
        phi     = traj['phi_final']

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

        diams = rad_arr * 2.0
        pdi   = diams.std() / diams.mean() if diams.mean() > 0 else 0.0

        records.append({'n': n, 'phi': phi, 'z': z_mean, 'pdi': pdi})

        if phi > best_phi:
            best_phi     = phi
            best_pos_arr = pos_arr
            best_rad_arr = rad_arr
            best_L       = L_traj

    phis = [r['phi'] for r in records]
    ns   = [r['n']   for r in records]
    zs   = [r['z']   for r in records]
    pdis = [r['pdi'] for r in records]

    with open(cfg.eval_report_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"评测报告  模型: {cfg.ckpt_prefix}_final.pth\n")
        f.write(f"评测局数: {cfg.eval_episodes}    目标粒子数: {cfg.target_N}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'指标':<20}{'均值':>10}{'最大':>10}{'最小':>10}{'标准差':>10}\n")
        f.write("-" * 60 + "\n")
        for name, vals in [("体积分数 phi", phis), ("粒子数 N", ns),
                            ("平均配位数 Z", zs), ("多分散性 PDI", pdis)]:
            arr = np.array(vals, dtype=float)
            f.write(f"{name:<20}{arr.mean():>10.4f}{arr.max():>10.4f}"
                    f"{arr.min():>10.4f}{arr.std():>10.4f}\n")
        f.write("\n各局明细:\n")
        f.write(f"{'局':<6}{'N':>6}{'phi':>8}{'Z':>8}{'PDI':>8}\n")
        f.write("-" * 40 + "\n")
        for i, r in enumerate(records):
            f.write(f"{i+1:<6}{r['n']:>6}{r['phi']:>8.4f}"
                    f"{r['z']:>8.2f}{r['pdi']:>8.4f}\n")

    print(f"  评测完成  phi: mean={np.mean(phis):.4f}  "
          f"max={np.max(phis):.4f}  min={np.min(phis):.4f}")
    print(f"  报告已保存: {cfg.eval_report_file}")

    if best_phi > 0:
        with open(cfg.eval_conf_file, 'w') as f:
            for p, r in zip(best_pos_arr, best_rad_arr):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
            f.write(f"{best_L:.6f} {best_L:.6f} {best_L:.6f}\n")
        print(f"  最优堆积已保存: {cfg.eval_conf_file}  (phi={best_phi:.4f})")

    policy.train()


# =====================================================================
# 8. 生成测试
# =====================================================================
def generate_packing(model_path, output_file="final_packing_v7.conf"):
    cfg    = Config()
    device = cfg.device

    policy = PackingPolicy(cfg).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy.eval()

    collector = DataCollector(cfg)
    trajs     = collector.collect(policy, n_samples=1, greedy=True)
    traj      = trajs[0]

    env = ConstructEnv(cfg)
    env.reset()
    for step in traj['steps']:
        env.step(step['action'])

    phi = env.get_phi()
    print(f"生成完成  粒子数={env.n}  phi={phi:.4f}")

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