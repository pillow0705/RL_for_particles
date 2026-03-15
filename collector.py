import numpy as np
import torch
import multiprocessing as mp

from config import Config
from model import PackingPolicy
from env import ConstructEnv


def _worker_collect_episode(args):
    policy_state_dict, greedy, temperature, seed = args
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
            'cand_stats': dict(env._last_cand_stats),
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
