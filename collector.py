import os
import pickle
import tempfile
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from config import Config
from model import PackingPolicy
from env import ConstructEnv


def _worker_collect_episode(args):
    """采集一条轨迹，将结果写入临时文件，返回文件路径（避免大数据通过 pipe 传输）。"""
    policy_state_dict, greedy, temperature, seed = args
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = Config()

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
        graph_pos, graph_rad = env.get_graph_data()
        obs_t  = torch.from_numpy(obs).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if policy is None:
            valid_idx = np.where(mask > 0)[0]
            if len(valid_idx) == 0:
                break
            action_idx = int(np.random.choice(valid_idx))
        else:
            with torch.no_grad():
                scores = policy(obs_t, graph_pos, graph_rad, env.L, mask_t)
            scores_np = scores[0].cpu().numpy()
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
            'L'         : env.L,
            'action'    : action_idx,
            'reward'    : reward,
            'cand_stats': dict(env._last_cand_stats),
        })

        obs, mask = next_obs, next_mask

    phi_final = env.get_phi()
    result = {
        'steps'     : traj_steps,
        'phi_final' : phi_final,
        'final_pos' : np.array(env.pos, dtype=np.float64).copy(),
        'final_rad' : np.array(env.rad, dtype=np.float64).copy(),
        'L'         : env.L,
    }

    # 写入临时文件，避免大轨迹数据通过 multiprocessing pipe 传输（BrokenPipeError）
    fd, tmp_path = tempfile.mkstemp(suffix='.pkl', prefix='traj_', dir='/tmp')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tmp_path


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
        trajectories = []
        with ctx.Pool(processes=min(cfg.num_workers, n_samples)) as pool:
            for i, tmp_path in enumerate(pool.imap_unordered(_worker_collect_episode, args_list), 1):
                with open(tmp_path, 'rb') as f:
                    traj = pickle.load(f)
                os.unlink(tmp_path)
                trajectories.append(traj)
                print(f"  [采集] {i}/{n_samples}  phi={traj['phi_final']:.4f}", flush=True)

        phis = [t['phi_final'] for t in trajectories]
        print(f"  [采集完成] {n_samples} 条  "
              f"phi: mean={np.mean(phis):.4f}  "
              f"max={np.max(phis):.4f}  "
              f"min={np.min(phis):.4f}")

        return trajectories


class VectorizedCollector:
    """
    向量化采集器：n_envs 个环境同步步进，每步做一次批量 GPU 推理。
    物理步（numba @njit）释放 GIL，可用线程池真正并行执行。
    """

    def __init__(self, cfg: Config, n_envs: int = None):
        self.cfg    = cfg
        self.n_envs = n_envs if n_envs is not None else cfg.num_workers

    # ------------------------------------------------------------------
    def collect(self, policy, n_samples: int, greedy: bool = False) -> list:
        cfg    = self.cfg
        device = cfg.device
        n_envs = min(self.n_envs, n_samples)

        # PyTorch CPU 线程限为 1，避免与 numba 线程竞争 CPU 核心
        torch.set_num_threads(1)

        # 初始化环境
        envs      = [ConstructEnv(cfg) for _ in range(n_envs)]
        obs_list  = []
        mask_list = []
        for env in envs:
            obs, mask = env.reset()
            obs_list.append(obs)
            mask_list.append(mask)

        traj_steps = [[] for _ in range(n_envs)]
        completed  = []
        active     = list(range(n_envs))

        # 线程池提到循环外复用，避免每步重建的 OS 开销（10万次 → 1次）
        with ThreadPoolExecutor(max_workers=n_envs) as executor:

            while active and len(completed) < n_samples:

                # ── 1. 收集当前步的图数据（step 前） ─────────────────
                gdata = []
                for i in active:
                    gp, gr = envs[i].get_graph_data()
                    gdata.append({
                        'graph_pos': gp.copy(),
                        'graph_rad': gr.copy(),
                        'L'        : envs[i].L,
                    })

                # ── 2. 批量 GPU 推理 ──────────────────────────────────
                obs_b  = torch.from_numpy(
                    np.stack([obs_list[i]  for i in active])).to(device)
                mask_b = torch.from_numpy(
                    np.stack([mask_list[i] for i in active])).to(device)

                with torch.no_grad():
                    if policy is not None:
                        scores_np = policy.batch_forward(
                            obs_b, mask_b, gdata, device).cpu().numpy()
                    else:
                        scores_np = None

                # ── 3. 采样动作 ───────────────────────────────────────
                actions = {}
                for bi, i in enumerate(active):
                    valid = np.where(mask_list[i] > 0)[0]
                    if len(valid) == 0:
                        actions[i] = 0
                        continue
                    if policy is None:
                        actions[i] = int(np.random.choice(valid))
                    elif greedy:
                        actions[i] = int(np.argmax(scores_np[bi]))
                    else:
                        s = scores_np[bi][valid]
                        s = s - s.max()
                        p = np.exp(s / cfg.temperature)
                        p /= p.sum()
                        actions[i] = int(np.random.choice(valid, p=p))

                # ── 4. 并行 env.step（复用线程池，numba 释放 GIL）────
                def _step(i):
                    (no, nm), r, done = envs[i].step(actions[i])
                    return i, no, nm, r, done, dict(envs[i]._last_cand_stats)

                results = list(executor.map(_step, active))

                # ── 5. 处理结果 ───────────────────────────────────────
                next_active = []
                for bi, (i, no, nm, r, done, cs) in enumerate(results):
                    traj_steps[i].append({
                        'obs'       : obs_list[i],
                        'mask'      : mask_list[i],
                        'graph_pos' : gdata[bi]['graph_pos'],
                        'graph_rad' : gdata[bi]['graph_rad'],
                        'L'         : gdata[bi]['L'],
                        'action'    : actions[i],
                        'reward'    : r,
                        'cand_stats': cs,
                    })
                    obs_list[i], mask_list[i] = no, nm

                    if done:
                        phi = envs[i].get_phi()
                        completed.append({
                            'steps'    : traj_steps[i],
                            'phi_final': phi,
                            'final_pos': np.array(envs[i].pos, dtype=np.float64).copy(),
                            'final_rad': np.array(envs[i].rad, dtype=np.float64).copy(),
                            'L'        : envs[i].L,
                        })
                        print(f"  [采集] {len(completed)}/{n_samples}  "
                              f"phi={phi:.4f}", flush=True)

                        if len(completed) < n_samples:
                            obs, mask = envs[i].reset()
                            obs_list[i], mask_list[i] = obs, mask
                            traj_steps[i] = []
                            next_active.append(i)
                    else:
                        next_active.append(i)

                active = next_active

        phis = [t['phi_final'] for t in completed]
        print(f"  [采集完成] {len(completed)} 条  "
              f"phi: mean={np.mean(phis):.4f}  "
              f"max={np.max(phis):.4f}  "
              f"min={np.min(phis):.4f}")
        return completed
