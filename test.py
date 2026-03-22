"""
test.py - 使用训练好的模型生成 2000 个粒子
使用 O(N) 增量候选点维护算法
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations

warnings.filterwarnings("ignore", category=UserWarning)

from config import Config
from model import PackingPolicy


# =====================================================================
# 物理辅助函数
# =====================================================================
def pbc_diff(p1, p2, L):
    """PBC 位移（numpy 版本）"""
    d = p1 - p2
    return d - np.round(d / L) * L


def solve_three_spheres(p1, r1, p2, r2, p3, r3, r_new):
    """三球求解（numpy 版本）"""
    dp21 = p2 - p1
    d12 = np.linalg.norm(dp21)
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, None, None

    ex = dp21 / d12
    dp31 = p3 - p1
    i = np.dot(ex, dp31)
    ey_v = dp31 - i * ex
    d_ey = np.linalg.norm(ey_v)

    if d_ey < 1e-7:
        return False, None, None
    ey = ey_v / d_ey
    ez = np.cross(ex, ey)

    x = (s1**2 - s2**2 + d12**2) / (2 * d12)
    y = (s1**2 - s3**2 + i**2 + d_ey**2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1**2 - x**2 - y**2

    if z_sq < 0:
        return False, None, None
    z = np.sqrt(z_sq)

    sol1 = p1 + x * ex + y * ey + z * ez
    sol2 = p1 + x * ex + y * ey - z * ez
    return True, sol1, sol2


def check_single_collision(sol, r_new, new_pos, new_rad, L, tol):
    """检查候选点与单个新粒子是否碰撞（增量过滤用）。返回 (collision, touching)。"""
    r_sum = new_rad + r_new
    gap = np.linalg.norm(pbc_diff(new_pos, sol, L)) - r_sum
    return gap < -tol * r_sum, gap < tol * r_sum


def get_pbc_center_of_mass(pos_array, L):
    """计算 PBC 下的中心质量"""
    theta = 2 * np.pi * pos_array / L
    cos_sum = np.mean(np.cos(theta), axis=0)
    sin_sum = np.mean(np.sin(theta), axis=0)
    phi = np.arctan2(sin_sum, cos_sum)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return (phi / (2 * np.pi)) * L


# =====================================================================
# 环境类 - O(N) 增量维护版本
# =====================================================================
class ConstructEnv2000:
    """
    使用 O(N) 增量候选点维护的环境：
    - 只为接触的粒子对添加三元组
    - 每个粒子配位数 ~8，所以候选生成是 O(N) 而不是 O(N^3)
    """

    def __init__(self, cfg, target_N=2000):
        self.cfg = cfg
        # 根据目标粒子数计算盒子大小
        avg_vol = (4.0 / 3.0) * np.pi * ((cfg.diameters.mean() / 2) ** 3)
        total_vol = target_N * avg_vol / cfg.target_phi
        self.L = total_vol ** (1.0 / 3.0)
        self.reset()

    def reset(self):
        """清空状态，放置初始 4 个粒子"""
        cfg = self.cfg

        # 物理状态
        self.pos = []
        self.rad = []
        self.n = 0

        # 候选集合
        self._candidate_set = {}  # cid → [x,y,z,r,coord]
        self._triplet_set = {}    # (i,j,k) → set of cid
        self._cand_to_triplets = {}  # cid → set of (i,j,k)
        self._cand_counter = 0
        self.current_candidates = []

        # 接触对（用于 O(N) 候选生成）
        self.contact_pairs = set()

        # 放置初始 4 个粒子
        self._init_four_particles()
        # 为初始 4 粒子生成所有三元组候选
        for i, j, k in combinations(range(self.n), 3):
            self._process_triplet(i, j, k)

        self._update_current_candidates()
        return self._get_obs()

    def _new_cand_id(self):
        cid = self._cand_counter
        self._cand_counter += 1
        return cid

    def _init_four_particles(self):
        """随机放置 4 个两两相切的种子粒子"""
        cfg = self.cfg
        # 随机半径
        r0 = np.random.choice(cfg.diameters) / 2
        r1 = np.random.choice(cfg.diameters) / 2
        r2 = np.random.choice(cfg.diameters) / 2
        r3 = np.random.choice(cfg.diameters) / 2

        p0 = np.array([self.L / 2, self.L / 2, self.L / 2])
        p1 = p0 + np.array([r0 + r1, 0, 0])

        d01 = r0 + r1
        d02 = r0 + r2
        d12 = r1 + r2
        x2 = (d02**2 - d12**2 + d01**2) / (2 * d01)
        y2 = np.sqrt(max(0.0, d02**2 - x2**2))
        p2 = p0 + np.array([x2, y2, 0])

        valid, p3, _ = solve_three_spheres(p0, r0, p1, r1, p2, r2, r3)
        if not valid:
            p3 = p0 + np.array([0, 0, r0 + r3])

        for p, r in zip([p0, p1, p2, p3], [r0, r1, r2, r3]):
            self.pos.append(p % self.L)
            self.rad.append(r)
        self.n = 4

        # 初始 4 粒子两两相切
        for i in range(4):
            for j in range(i + 1, 4):
                self.contact_pairs.add((min(i, j), max(i, j)))

    def _check_and_collect_touching(self, sol, r_new, all_pos, all_rad, L, tol):
        """同时做碰撞检测并收集接触粒子索引"""
        touching = []
        for m in range(len(all_pos)):
            r_sum = all_rad[m] + r_new
            gap = np.linalg.norm(pbc_diff(all_pos[m], sol, L)) - r_sum
            if gap < -tol * r_sum:
                return True, []
            if gap < tol * r_sum:
                touching.append(m)
        return False, touching

    def _process_triplet(self, i, j, k):
        """对三元组 (i,j,k) 生成合法候选点"""
        cfg = self.cfg
        all_pos = np.array(self.pos, dtype=np.float64)
        all_rad = np.array(self.rad, dtype=np.float64)
        p_i, r_i = all_pos[i], all_rad[i]
        p_j, r_j = all_pos[j], all_rad[j]
        p_k, r_k = all_pos[k], all_rad[k]

        # PBC 展开：将 j、k 的坐标对齐到以 i 为原点的近邻像
        p_j = p_i + pbc_diff(p_j, p_i, self.L)
        p_k = p_i + pbc_diff(p_k, p_i, self.L)

        triplet_cids = set()

        for r_new in cfg.diameters / 2.0:
            valid, sol1, sol2 = solve_three_spheres(p_i, r_i, p_j, r_j, p_k, r_k, r_new)
            if not valid:
                continue
            for sol in (sol1, sol2):
                collision, touching = self._check_and_collect_touching(
                    sol, r_new, all_pos, all_rad, self.L, cfg.collision_tol)
                if not collision:
                    cid = self._new_cand_id()
                    coord = len(touching)
                    self._candidate_set[cid] = np.array(
                        [*(sol % self.L), r_new, float(coord)], dtype=np.float32)
                    self._cand_to_triplets[cid] = {(i, j, k)}
                    triplet_cids.add(cid)

        if triplet_cids:
            self._triplet_set[(i, j, k)] = triplet_cids

    def _add_new_triplets(self, new_idx):
        """
        【O(N) 关键】只处理新粒子与它已接触的粒子对
        因为每个粒子配位数 ~8，所以循环次数是 O(N) 而不是 O(N^3)
        """
        contacts = [m for m in range(new_idx)
                    if (min(m, new_idx), max(m, new_idx)) in self.contact_pairs]
        for j, k in combinations(contacts, 2):
            self._process_triplet(new_idx, j, k)

    def _filter_candidates(self, new_pos_wrapped, new_rad, new_idx):
        """
        过滤与新粒子碰撞的候选，并更新接触信息
        """
        new_pos_np = np.array(new_pos_wrapped, dtype=np.float64)
        to_delete = []
        for cid, feat in self._candidate_set.items():
            sol = feat[:3].astype(np.float64)
            r_c = float(feat[3])
            collision, touching = check_single_collision(
                sol, r_c, new_pos_np, new_rad, self.L, self.cfg.collision_tol)
            if collision:
                to_delete.append(cid)
            elif touching:
                feat[4] += 1.0  # 更新 coordination
        for cid in to_delete:
            self._remove_candidate(cid)

    def _remove_candidate(self, cid):
        if cid not in self._candidate_set:
            return
        del self._candidate_set[cid]
        for triplet in self._cand_to_triplets.get(cid, set()):
            if triplet in self._triplet_set:
                self._triplet_set[triplet].discard(cid)
                if not self._triplet_set[triplet]:
                    del self._triplet_set[triplet]
        del self._cand_to_triplets[cid]

    def _update_current_candidates(self):
        """构建有序候选列表"""
        if len(self._candidate_set) == 0:
            self.current_candidates = []
            return

        cands_np = np.array(list(self._candidate_set.values()))
        # 排序：配位数高优先，其次半径大
        idx = np.lexsort((cands_np[:, 3], cands_np[:, 4]))[::-1]
        cands_np = cands_np[idx]
        # 限制最大候选数
        cands_np = cands_np[:self.cfg.max_candidates]
        self.current_candidates = cands_np.tolist()

    def _get_obs(self):
        """获取观测"""
        cfg = self.cfg
        self._update_current_candidates()

        obs = np.zeros((cfg.max_candidates, 5), dtype=np.float32)
        mask = np.zeros(cfg.max_candidates, dtype=np.float32)

        n_valid = len(self.current_candidates)
        if n_valid > 0:
            c = np.array(self.current_candidates[:n_valid])
            center = get_pbc_center_of_mass(np.array(self.pos), self.L)
            c[:, :3] -= center
            c[:, :3] = c[:, :3] - np.round(c[:, :3] / self.L) * self.L
            c[:, :3] /= self.L
            obs[:n_valid] = c
            mask[:n_valid] = 1.0

        return obs, mask

    def step(self, action_idx):
        """执行动作"""
        cfg = self.cfg

        if action_idx >= len(self.current_candidates):
            obs, mask = self._get_obs()
            return (obs, mask), 0.0, True

        chosen = self.current_candidates[action_idx]
        pos_new = np.array(chosen[:3])
        r_new = chosen[3]

        # 从候选池中移除
        chosen_feat = np.array(chosen, dtype=np.float32)
        for cid, feat in list(self._candidate_set.items()):
            if np.allclose(feat, chosen_feat, atol=1e-5):
                self._remove_candidate(cid)
                break

        # 添加新粒子
        self.pos.append(pos_new % self.L)
        self.rad.append(r_new)
        self.n += 1
        new_idx = self.n - 1

        # 更新接触对
        for m in range(new_idx):
            dist = np.linalg.norm(pbc_diff(self.pos[m], self.pos[new_idx], self.L))
            r_sum = self.rad[m] + r_new
            if dist < r_sum * (1 + cfg.collision_tol):
                self.contact_pairs.add((min(m, new_idx), max(m, new_idx)))

        # 过滤候选
        n_before_filter = len(self._candidate_set)
        self._filter_candidates(pos_new % self.L, r_new, new_idx)
        n_filtered = n_before_filter - len(self._candidate_set)

        # 为新粒子添加新三元组（O(N) 关键）
        n_before_add = len(self._candidate_set)
        self._add_new_triplets(new_idx)
        n_added = len(self._candidate_set) - n_before_add

        done = self.n >= cfg.max_particles
        obs, mask = self._get_obs()

        if np.sum(mask) == 0 and not done:
            done = True

        reward = self.get_phi() if done else 0.0
        return (obs, mask), reward, done

    def get_phi(self):
        """计算当前体积分数"""
        box_vol = self.L**3
        total_vol = sum((4.0 / 3.0) * np.pi * (r**3) for r in self.rad)
        return total_vol / box_vol

    def get_graph_data(self):
        """返回当前图数据"""
        return (np.array(self.pos, dtype=np.float64),
                np.array(self.rad, dtype=np.float64))


# =====================================================================
# 主程序：生成 2000 粒子
# =====================================================================
def generate_2000_particles(model_path, output_file="packing_2000.conf", target_N=2000):
    """
    使用训练好的模型生成 2000 个粒子
    策略：使用 O(N) 增量候选点维护（只处理接触粒子对的三元组）
    """
    print(f"\n{'='*60}")
    print(f"开始生成 {target_N} 个粒子")
    print(f"模型: {model_path}")
    print(f"{'='*60}\n")

    import os
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return None

    # 创建配置
    class GenConfig(Config):
        target_N = 2000
        max_particles = 3000
        max_candidates = 2000

    cfg = GenConfig()
    device = cfg.device

    # 创建环境
    env = ConstructEnv2000(cfg, target_N=target_N)

    # 加载模型
    policy = PackingPolicy(cfg).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    print(f"模型加载完成")
    print(f"目标粒子数: {target_N}")
    print(f"目标体积分数: {cfg.target_phi}")
    print(f"盒子尺寸 L = {env.L:.4f}")
    print(f"初始粒子数: {env.n}\n")

    # 初始化
    obs, mask = env.reset()
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # 生成循环
    while env.n < target_N:
        # 获取当前图数据
        graph_pos, graph_rad = env.get_graph_data()

        # 模型前向推理
        with torch.no_grad():
            scores = policy(obs_t, graph_pos, graph_rad, env.L, mask_t)

        scores_np = scores[0].cpu().numpy()
        valid_idx = np.where(mask > 0)[0]

        if len(valid_idx) == 0:
            print(f"\n【警告】没有合法的候选点，生成停止 (当前粒子数: {env.n})")
            break

        # 贪婪选择
        action_idx = int(np.argmax(scores_np))

        # 执行步骤
        (next_obs, next_mask), reward, done = env.step(action_idx)

        # 更新观测
        obs = next_obs
        mask = next_mask
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        # 进度输出
        if env.n % 200 == 0:
            current_phi = env.get_phi()
            print(f"[{env.n}/{target_N}] phi = {current_phi:.4f}")

        if done:
            break

    # 计算最终结果
    final_phi = env.get_phi()
    final_pos = np.array(env.pos)
    final_rad = np.array(env.rad)

    print(f"\n{'='*60}")
    print(f"生成完成!")
    print(f"{'='*60}")
    print(f"最终粒子数: {env.n}")
    print(f"最终体积分数 phi: {final_phi:.4f}")
    print(f"盒子尺寸 L: {env.L:.4f}")

    # 计算配位数
    coord_counts = np.zeros(env.n, dtype=int)
    for i in range(env.n):
        for j in range(i + 1, env.n):
            diff = final_pos[j] - final_pos[i]
            diff -= np.round(diff / env.L) * env.L
            dist = np.linalg.norm(diff)
            if dist <= (final_rad[i] + final_rad[j]) * (1 + cfg.collision_tol):
                coord_counts[i] += 1
                coord_counts[j] += 1
    z_mean = coord_counts.mean()
    print(f"平均配位数 Z: {z_mean:.2f}")

    # 保存配置
    with open(output_file, 'w') as f:
        for p, r in zip(final_pos, final_rad):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
        f.write(f"{env.L:.6f} {env.L:.6f} {env.L:.6f}\n")

    print(f"\n结构已保存至: {output_file}")

    return {
        'n': env.n,
        'phi': final_phi,
        'z_mean': z_mean,
        'output_file': output_file
    }


if __name__ == "__main__":
    torch.set_num_threads(1)

    model_path = "construct_v7.0_final.pth"
    result = generate_2000_particles(model_path, output_file="packing_2000.conf", target_N=2000)

    if result:
        print(f"\nDONE! 生成了 {result['n']} 个粒子，phi = {result['phi']:.4f}")
