import numpy as np
import torch
import torch.nn as nn

from config import Config


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
        deg      = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg      = torch.matmul(adj, node_feat) / deg
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
        self.gnn_layers  = nn.ModuleList([GNNLayer(hid, hid) for _ in range(cfg.gnn_layers)])
        self.output_proj = nn.Linear(hid * 2, cfg.embed_dim)

    def forward(self, pos, rad, L, adj_np, device):
        node_np   = np.concatenate(
            [pos / L, (rad / rad.max()).reshape(-1, 1)], axis=1
        ).astype(np.float32)
        node_feat = torch.from_numpy(node_np).to(device)
        adj       = torch.from_numpy(adj_np).to(device)

        h = self.input_proj(node_feat)
        for layer in self.gnn_layers:
            h = layer(h, adj)

        h_glob = torch.cat([h.max(dim=0)[0], h.mean(dim=0)], dim=-1)
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
        return self.net(combined).squeeze(-1)


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
