import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from model import PackingPolicy


class Trainer:
    def __init__(self, policy: PackingPolicy, cfg: Config):
        self.policy    = policy
        self.cfg       = cfg
        self.optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
        self._ckpt     = None   # 上一轮训练前的权重备份

    def backup(self):
        """训练前备份当前权重。"""
        self._ckpt = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}

    def rollback(self):
        """回滚到上一次 backup() 时的权重，并将 lr 缩小 0.5 倍。"""
        if self._ckpt is None:
            return
        self.policy.load_state_dict(self._ckpt)
        for pg in self.optimizer.param_groups:
            pg['lr'] *= 0.5
        print(f"  [回滚] 模型已恢复，lr 调整为 {self.optimizer.param_groups[0]['lr']:.2e}")

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

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

        # 丢弃归一化 advantage 绝对值最小的一部分（梯度噪声来源）
        if cfg.advantage_filter_ratio > 0:
            norm_adv  = np.abs((all_returns - ret_mean) / ret_std)
            threshold = np.quantile(norm_adv, cfg.advantage_filter_ratio)
            all_samples = [s for s, a in zip(all_samples, norm_adv) if a >= threshold]
            if len(all_samples) == 0:
                return 0.0

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
                log_probs = torch.nn.functional.log_softmax(scores / cfg.temperature, dim=-1)
                log_pa    = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

                batch_loss = -(log_pa * advantages).mean()
                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()
                n_updates  += 1

        return total_loss / max(n_updates, 1)
