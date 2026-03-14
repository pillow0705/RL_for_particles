import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import multiprocessing as mp
import csv
import time
from collections import deque
from numba import njit

# 限制 PyTorch 线程数，防止并行争抢 CPU
torch.set_num_threads(1)

# ==========================================
# 1. 核心加速算法 (Numba JIT)
# ==========================================
@njit(inline='always')
def pbc_diff_njit(p1, p2, L):
    """计算 p1 - p2 的 PBC 位移向量"""
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
    """三球求解的 JIT 版本"""
    dp21 = p2 - p1
    d12 = np.sqrt(np.sum(dp21**2))
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, np.zeros(3), np.zeros(3)

    ex = dp21 / d12
    dp31 = p3 - p1
    i = np.dot(ex, dp31)
    ey_vec = dp31 - i * ex
    d_ey = np.sqrt(np.sum(ey_vec**2))
    
    if d_ey < 1e-7: return False, np.zeros(3), np.zeros(3)
    ey = ey_vec / d_ey
    
    # 叉乘手动实现 (numba 对 np.cross 在某些版本支持有限)
    ez = np.array([
        ex[1]*ey[2] - ex[2]*ey[1],
        ex[2]*ey[0] - ex[0]*ey[2],
        ex[0]*ey[1] - ex[1]*ey[0]
    ])

    x = (s1**2 - s2**2 + d12**2) / (2 * d12)
    y = (s1**2 - s3**2 + i**2 + d_ey**2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1**2 - x**2 - y**2
    
    if z_sq < 0: return False, np.zeros(3), np.zeros(3)
    z = np.sqrt(z_sq)

    sol1 = p1 + x * ex + y * ey + z * ez
    sol2 = p1 + x * ex + y * ey - z * ez
    return True, sol1, sol2

@njit
def find_candidates_njit(active_pos, active_rad, all_pos, all_rad, L, diameters, max_cands):
    """
    【核心变更】遍历所有的直径，直接算出绑定了特定直径的完美坐标！
    返回: (M, 5) 数组 [x, y, z, r, coord]
    """
    results = np.zeros((max_cands, 5))
    count = 0
    n_active = len(active_pos)
    n_all = len(all_pos)
    
    # 遍历所有可能的直径
    for d_idx in range(len(diameters)):
        r_new = diameters[d_idx] / 2.0
        
        for i in range(n_active):
            p_i = active_pos[i]
            r_i = active_rad[i]
            
            for j in range(i + 1, n_active):
                p_j = p_i + pbc_diff_njit(active_pos[j], p_i, L)
                r_j = active_rad[j]
                d_ij = np.sqrt(np.sum((p_j - p_i)**2))
                
                if d_ij > (r_i + r_j + 2 * r_new + 0.05): continue
                
                for k in range(j + 1, n_active):
                    p_k = p_i + pbc_diff_njit(active_pos[k], p_i, L)
                    r_k = active_rad[k]
                    
                    # 算出的 s1, s2 是和 r_new 完美相切的坐标
                    valid, s1, s2 = solve_three_spheres_njit(p_i, r_i, p_j, r_j, p_k, r_k, r_new)
                    if not valid: continue
                    
                    solutions = np.stack((s1, s2))
                    for s_idx in range(2):
                        sol = solutions[s_idx]
                        
                        collision = False
                        coordination = 0
                        
                        for m in range(n_all):
                            dm = pbc_diff_njit(all_pos[m], sol, L)
                            dist = np.sqrt(np.sum(dm**2))
                            
                            gap = dist - (all_rad[m] + r_new)
                            
                            # 极其严格的碰撞检测，保证放进去绝对不重叠
                            if gap < -0.05:
                                collision = True
                                break
                            if gap < 0.05:
                                coordination += 1
                        
                        if not collision:
                            results[count, 0:3] = sol % L # PBC
                            results[count, 3] = r_new
                            results[count, 4] = coordination
                            count += 1
                            if count >= max_cands:
                                return results[:count]
                                
    return results[:count]

def get_pbc_center_of_mass(pos_array, L):
    theta = 2 * np.pi * pos_array / L
    cos_sum = np.mean(np.cos(theta), axis=0)
    sin_sum = np.mean(np.sin(theta), axis=0)
    phi = np.arctan2(sin_sum, cos_sum)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    return (phi / (2 * np.pi)) * L

# ==========================================
# 2. 全局配置
# ==========================================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    num_envs = 12           
    batch_size = 512        
    update_epochs = 10      
    total_updates = 2000    
    
    target_N = 100          
    target_phi = 0.72
    
    diameters = np.arange(0.7, 1.4, 0.05)
    
    # 候选点可能很多，因为每种直径都算了一遍，放宽到 1000
    max_candidates = 1000    
    steps_per_env = 150 # 允许超量放置 

# ==========================================
# 3. 构造环境
# ==========================================
class ConstructEnv:
    def __init__(self, rank=0):
        self.rank = rank
        avg_vol = (4.0/3.0) * np.pi * (0.5**3)
        total_vol_needed = Config.target_N * avg_vol / Config.target_phi
        self.L = total_vol_needed ** (1.0/3.0)
        self.box = np.array([self.L, self.L, self.L])
        self.reset()

    def reset(self):
        # 固定的四面体底座
        r0, r1, r2, r3 = 0.5, 0.6, 0.4, 0.5
        p0 = np.array([self.L/2, self.L/2, self.L/2])
        p1 = p0 + np.array([r0+r1, 0, 0])
        
        d01, d02, d12 = r0+r1, r0+r2, r1+r2
        x2 = (d02**2 - d12**2 + d01**2) / (2*d01)
        y2 = np.sqrt(max(0, d02**2 - x2**2))
        p2 = p0 + np.array([x2, y2, 0])
        
        valid, p3, _ = solve_three_spheres_njit(p0, r0, p1, r1, p2, r2, r3)
        if not valid: p3 = p0 + np.array([0, 0, r0+r3])

        raw_pos = [p0, p1, p2, p3]
        self.pos = [p % self.L for p in raw_pos]
        self.rad = [r0, r1, r2, r3]
        self.step_count = 4  
        self.n_particles = 4

        return self._get_observation()

    def _get_candidates(self):
        active_indices = list(range(max(0, self.n_particles - 20), self.n_particles))
        if self.n_particles > 20:
            n_old = self.n_particles - 20
            n_sample = min(10, n_old)
            if n_sample > 0:
                others = np.random.choice(range(n_old), n_sample, replace=False)
                active_indices.extend(others)

        active_pos = np.array(self.pos)[active_indices].astype(np.float64)
        active_rad = np.array(self.rad)[active_indices].astype(np.float64)
        all_pos = np.array(self.pos).astype(np.float64)
        all_rad = np.array(self.rad).astype(np.float64)

        candidates_array = find_candidates_njit(
            active_pos, active_rad, all_pos, all_rad, 
            self.L, Config.diameters, Config.max_candidates
        )
        return candidates_array.tolist()

    def _get_observation(self):
        candidates = self._get_candidates()

        if len(candidates) > 0:
            candidates = np.array(candidates)
            # 排序：优先配位数高，其次半径大
            idx = np.lexsort((candidates[:, 3], candidates[:, 4]))[::-1]
            candidates = candidates[idx]
            self.current_candidates = candidates.tolist()
        else:
            self.current_candidates =[]

        obs = np.zeros((Config.max_candidates, 5), dtype=np.float32)
        mask = np.zeros(Config.max_candidates, dtype=np.float32)

        n = min(len(self.current_candidates), Config.max_candidates)
        if n > 0:
            c_array = np.array(self.current_candidates[:n])
            center = get_pbc_center_of_mass(np.array(self.pos), self.L)
            c_array[:, 0:3] -= center
            c_array[:, 0:3] = c_array[:, 0:3] - np.round(c_array[:, 0:3] / self.L) * self.L
            c_array[:, 0:3] *= 0.5 
            obs[:n] = c_array
            mask[:n] = 1.0

        return obs, mask

    def step(self, action_idx):
        """
        【极致清爽的 step】：只接收选哪个坑。
        """
        # 1. 越界保护 (没坑位了)
        if action_idx >= len(self.current_candidates):
            obs, mask = self._get_observation()
            return (obs, mask), 0.0, True, {} # 不惩罚，正常结束

        chosen_cand = self.current_candidates[action_idx]
        pos_new = np.array(chosen_cand[:3])
        r_new = chosen_cand[3] # 坑位自带的完美相切半径！
        coord = chosen_cand[4]

        # 2. 放置成功 (因为 Numba 里已经做了碰撞检查，这里 100% 成功)
        self.pos.append(pos_new) # 早已在 Numba 取过模
        self.rad.append(r_new)
        self.n_particles += 1
        self.step_count += 1

        # 3. 基础奖励
        vol_added = (4.0/3.0) * np.pi * (r_new**3)
        reward = vol_added * 5.0 + coord * 0.5 

        # 4. 结束条件与终极大奖
        done = False
        if self.n_particles >= int(Config.target_N * 1.5):
            done = True

        next_obs_pair = self._get_observation()

        if np.sum(next_obs_pair[1]) == 0 and not done:
            done = True

        if done:
            box_vol = self.L ** 3
            total_vol = sum((4.0/3.0) * np.pi * (r**3) for r in self.rad)
            phi_final = total_vol / box_vol
            density_bonus = max(0.0, (phi_final - 0.60) * 20.0)
            
            all_diams = np.array(self.rad) * 2.0
            diam_std = np.std(all_diams) if len(all_diams) > 0 else 0
            dispersity_bonus = diam_std * 10.0 
            
            reward += (density_bonus + dispersity_bonus)

        return next_obs_pair, reward, done, {}

# ==========================================
# 4. 单头 PointNet 策略
# ==========================================
class PointNetPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        hid = 128
        self.local_net = nn.Sequential(
            nn.Linear(5, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        
        # 只保留一个位置选择头
        self.pos_head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x, mask):
        local_feat = self.local_net(x)  
        mask_exp = mask.unsqueeze(-1)
        masked_feat = local_feat * mask_exp + (1 - mask_exp) * -1e9
        
        global_feat = torch.max(masked_feat, dim=1)[0]  
        global_feat_exp = global_feat.unsqueeze(1).expand(-1, x.size(1), -1)
        combined = torch.cat([local_feat, global_feat_exp], dim=-1)  

        pos_logits = self.pos_head(combined).squeeze(-1)  
        pos_logits = pos_logits.masked_fill(mask == 0, -1e18)

        value = self.critic(global_feat)

        return pos_logits, value

    def get_action_and_value(self, x, mask, action_pos=None):
        pos_logits, value = self.forward(x, mask)
        
        pos_dist = Categorical(logits=pos_logits)
        if action_pos is None:
            # 防御机制
            if torch.all(mask == 0): action_pos = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else: action_pos = pos_dist.sample()
            
        log_prob = pos_dist.log_prob(action_pos)
        entropy = pos_dist.entropy()
        
        # 只返回 4 个值！
        return action_pos, log_prob, entropy, value

# ==========================================
# 5. 并行逻辑
# ==========================================
def worker(remote, parent_remote, rank):
    parent_remote.close()
    env = ConstructEnv(rank)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                pos_idx = data # 只有 1 个动作
                (obs, mask), reward, done, info = env.step(pos_idx)
                if done:
                    (obs, mask) = env.reset()
                remote.send(((obs, mask), reward, done, info))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'close':
                break
    except KeyboardInterrupt: remote.close()

class VecEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps =[mp.Process(target=worker, args=(w, r, i)) for i, (w, r) in enumerate(zip(self.work_remotes, self.remotes))]
        for p in self.ps: p.daemon = True; p.start()
        for w in self.work_remotes: w.close()

    def step(self, pos_idxs):
        for remote, p_i in zip(self.remotes, pos_idxs):
            remote.send(('step', p_i))
        results =[remote.recv() for remote in self.remotes]
        obs_data, rewards, dones, infos = zip(*results)
        batch_x = np.stack([o[0] for o in obs_data])
        batch_m = np.stack([o[1] for o in obs_data])
        return (torch.FloatTensor(batch_x), torch.FloatTensor(batch_m)), torch.FloatTensor(rewards), torch.FloatTensor(dones), infos

    def reset(self):
        for remote in self.remotes: remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        batch_x = np.stack([o[0] for o in results])
        batch_m = np.stack([o[1] for o in results])
        return torch.FloatTensor(batch_x), torch.FloatTensor(batch_m)

    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()

# ==========================================
# 6. 主训练循环
# ==========================================
def train():
    try: mp.set_start_method('spawn')
    except RuntimeError: pass
    
    envs = VecEnv(Config.num_envs)
    agent = PointNetPolicy().to(Config.device)
    optimizer = optim.Adam(agent.parameters(), lr=Config.lr, eps=1e-5)
    
    print(f"Start Clean Constructive Training... Target N={Config.target_N}")
    csv_file = open("Construct_Clean_log_5.2.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Update", "AvgReward", "AvgLen", "ActorLoss", "ValueLoss"])
    
    ep_len_queue = deque(maxlen=Config.num_envs * 2)
    b_x, b_m, b_pos, b_lp, b_rew, b_val, b_done = [],[],[],[],[],[],[]
    
    state_x, state_m = envs.reset()
    state_x, state_m = state_x.to(Config.device), state_m.to(Config.device)
    current_lens = np.zeros(Config.num_envs)
    
    for update in range(1, Config.total_updates + 1):
        for _ in range(Config.steps_per_env):
            with torch.no_grad():
                # 只有 4 个返回值
                p_idx, lp, _, val = agent.get_action_and_value(state_x, state_m)
            
            (next_x, next_m), reward, done, _ = envs.step(p_idx.cpu().numpy())
            
            b_x.append(state_x); b_m.append(state_m); b_pos.append(p_idx)
            b_lp.append(lp); b_val.append(val.flatten()); b_rew.append(reward.to(Config.device)); b_done.append(done.to(Config.device))
            
            state_x, state_m = next_x.to(Config.device), next_m.to(Config.device)
            current_lens += 1
            done_np = done.numpy()
            for i in range(Config.num_envs):
                if done_np[i]:
                    ep_len_queue.append(current_lens[i])
                    current_lens[i] = 0
        
        with torch.no_grad():
            _, _, _, next_val = agent.get_action_and_value(state_x, state_m)
            b_rew_t = torch.stack(b_rew); b_val_t = torch.stack(b_val); b_done_t = torch.stack(b_done)
            returns = torch.zeros_like(b_rew_t); advantages = torch.zeros_like(returns); lastgae = 0
            for t in reversed(range(Config.steps_per_env)):
                nonterm = 1.0 - (b_done_t[t] if t == Config.steps_per_env-1 else b_done_t[t+1]).float()
                nxt_v = next_val.view(-1) if t == Config.steps_per_env-1 else b_val_t[t+1]
                delta = b_rew_t[t] + Config.gamma * nxt_v * nonterm - b_val_t[t]
                lastgae = delta + Config.gamma * Config.gae_lambda * nonterm * lastgae
                advantages[t] = lastgae
                returns[t] = advantages[t] + b_val_t[t]

        flat_x, flat_m = torch.cat(b_x), torch.cat(b_m)
        flat_pos, flat_lp = torch.cat(b_pos).view(-1), torch.cat(b_lp).view(-1)
        flat_adv, flat_ret = advantages.view(-1), returns.view(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        
        inds = np.arange(flat_x.size(0))
        for _ in range(Config.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, flat_x.size(0), Config.batch_size):
                mb = inds[start:start+Config.batch_size]
                # 只有 4 个返回值
                _, new_lp, ent, new_v = agent.get_action_and_value(flat_x[mb], flat_m[mb], flat_pos[mb])
                
                ratio = (new_lp - flat_lp[mb]).exp()
                mb_adv = flat_adv[mb]
                pg_loss = -torch.max(mb_adv * ratio, mb_adv * torch.clamp(ratio, 1-Config.clip_coef, 1+Config.clip_coef)).mean()
                v_loss = 0.5 * ((new_v.view(-1) - flat_ret[mb])**2).mean()
                loss = pg_loss - Config.ent_coef * ent.mean() + Config.vf_coef * v_loss
                
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); optimizer.step()

        avg_len = np.mean(ep_len_queue) if len(ep_len_queue)>0 else 0.0
        print(f"Upd {update}, Rew: {b_rew_t.mean():.2f}, Len: {avg_len:.1f}")
        writer.writerow([update, b_rew_t.mean().item(), avg_len, pg_loss.item(), v_loss.item()]); csv_file.flush()
        
        b_x, b_m, b_pos, b_lp, b_rew, b_val, b_done = [],[],[],[],[],[],[]
        if update % 50 == 0: torch.save(agent.state_dict(), f"construct_v5.2_{update}.pth")

    csv_file.close(); envs.close()

# ==========================================
# 7. 生成测试
# ==========================================
def generate_packing(model_path, target_n=100, output_file="final_packing.conf"):
    print(f"\nGenerative Mode: Building {target_n} particles...")

    original_target_N = Config.target_N
    Config.target_N = target_n

    env = ConstructEnv(rank=0)
    avg_vol = (4.0/3.0) * np.pi * (0.5**3)
    total_vol = target_n * avg_vol / Config.target_phi
    env.L = total_vol ** (1.0/3.0)
    env.box = np.array([env.L, env.L, env.L])
    
    obs, mask = env.reset()
    
    agent = PointNetPolicy().to(Config.device)
    state_dict = torch.load(model_path, map_location=Config.device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    obs = torch.from_numpy(obs).unsqueeze(0).to(Config.device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(Config.device)
    
    success_count = 4
    print(f"Box L = {env.L:.4f}")
    
    while success_count < int(target_n * 2):
        with torch.no_grad():
            pos_logits, _ = agent.forward(obs, mask)
            pos_idx = torch.argmax(pos_logits, dim=1).item()
            
            if pos_logits[0, pos_idx] < -1e9:
                print(f"Deadlock reached at {success_count}.")
                break
                
            built_in_diam = env.current_candidates[pos_idx][3] * 2.0
            print(f"  Action: Pos={pos_idx}, Diam={built_in_diam:.3f}")

        (next_obs, next_mask), reward, done, _ = env.step(pos_idx)

        success_count += 1
        if np.sum(next_mask) == 0:
            print("No more pockets.")
            break

        obs = torch.from_numpy(next_obs).unsqueeze(0).to(Config.device)
        mask = torch.from_numpy(next_mask).unsqueeze(0).to(Config.device)
        
    print(f"Finished. Placed: {success_count}")
    
    box_vol = env.L ** 3
    actual_vol = sum((4.0/3.0) * np.pi * (r**3) for r in env.rad)
    print(f"==> Target Phi: {Config.target_phi:.4f}, Actual Phi: {actual_vol / box_vol:.4f}")

    Config.target_N = original_target_N

    with open(output_file, 'w') as f:
        for p, r in zip(env.pos, env.rad):
            f.write(f"{p[0]} {p[1]} {p[2]} {r}\n")
        f.write(f"{env.L} {env.L} {env.L}\n")

if __name__ == "__main__":
    train()
    # generate_packing("construct_v5.2_1000.pth", target_n=2000)