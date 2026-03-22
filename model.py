import numpy as np
import torch
import torch.nn as nn

from config import Config
from physics import get_pbc_center_of_mass


class CandidateEncoder(nn.Module):
    """MLP: 候选点特征 (B, max_candidates, 5) → (B, max_candidates, embed_dim)"""
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


class ParticleTransformer(nn.Module):
    """
    已放置粒子 → Transformer Encoder → CLS token → embed_dim。
    粒子特征: [x/L, y/L, z/L, r/r_max]  (4维，无需位置编码)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.transformer_d_model
        self.r_max      = float(cfg.diameters.max() / 2)   # 全局最大半径，归一化用
        self.input_proj = nn.Sequential(nn.Linear(4, d), nn.ReLU())
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d))
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.transformer_nhead,
            dim_feedforward=cfg.transformer_ffn_dim,
            batch_first=True, dropout=0.0
        )
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.output_proj = nn.Linear(d, cfg.embed_dim)

    def _build_node_features(self, pos_np, rad_np, L):
        center = get_pbc_center_of_mass(pos_np, L)
        rel    = pos_np - center
        rel    = rel - np.round(rel / L) * L   # PBC wrap
        rel    = rel / L                        # 归一化到 [-0.5, 0.5]，与候选点坐标缩放一致
        return np.concatenate(
            [rel, (rad_np / self.r_max).reshape(-1, 1)], axis=1
        ).astype(np.float32)   # (N, 4)

    def forward_single(self, pos_np, rad_np, L, device):
        """单样本推理，无 padding。返回 (embed_dim,)。"""
        node = self._build_node_features(pos_np, rad_np, L)
        x    = self.input_proj(torch.from_numpy(node).to(device)).unsqueeze(0)  # (1, N, d)
        x    = torch.cat([self.cls_token, x], dim=1)                            # (1, N+1, d)
        x    = self.encoder(x)                                                  # (1, N+1, d)
        return self.output_proj(x[0, 0])                                        # (embed_dim,)

    def forward_batch(self, samples, device):
        """批量推理，自动 padding。返回 (B, embed_dim)。"""
        B     = len(samples)
        nodes = [self._build_node_features(s['graph_pos'], s['graph_rad'], s['L'])
                 for s in samples]
        max_n = max(n.shape[0] for n in nodes)

        padded   = np.zeros((B, max_n, 4), dtype=np.float32)
        pad_mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        for i, node in enumerate(nodes):
            n = node.shape[0]
            padded[i, :n] = node
            pad_mask[i, n:] = True                       # padding 位置不参与 attention

        x    = self.input_proj(torch.from_numpy(padded).to(device))   # (B, max_n, d)
        cls  = self.cls_token.expand(B, -1, -1)                       # (B, 1, d)
        x    = torch.cat([cls, x], dim=1)                             # (B, max_n+1, d)

        # CLS 位置不屏蔽
        cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)            # (B, max_n+1)

        x = self.encoder(x, src_key_padding_mask=full_mask)           # (B, max_n+1, d)
        return self.output_proj(x[:, 0])                              # (B, embed_dim)


class PackingPolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg                  = cfg
        self.cand_encoder         = CandidateEncoder(cfg)
        self.particle_transformer = ParticleTransformer(cfg)

    def forward(self, cand_feat, graph_pos, graph_rad, L, mask):
        """单步推理（采集时调用）。"""
        device    = cand_feat.device
        cand_emb  = self.cand_encoder(cand_feat)                                          # (1, 1000, 128)
        state_emb = self.particle_transformer.forward_single(graph_pos, graph_rad, L, device)  # (128,)
        scores    = (cand_emb * state_emb).sum(-1)                                        # (1, 1000)
        return scores.masked_fill(mask == 0, float('-inf'))

    def batch_forward(self, obs_batch, mask_batch, samples, device):
        """批量训练前向。"""
        cand_emb  = self.cand_encoder(obs_batch)                              # (B, 1000, 128)
        state_emb = self.particle_transformer.forward_batch(samples, device)  # (B, 128)
        scores    = (cand_emb * state_emb.unsqueeze(1)).sum(-1)               # (B, 1000)
        return scores.masked_fill(mask_batch == 0, float('-inf'))
