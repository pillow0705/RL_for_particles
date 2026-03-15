import numpy as np
import torch


class Config:
    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 物理 / 环境参数 ----
    target_N       = 50
    target_phi     = 0.72
    diam_min       = 0.7
    diam_max       = 1.35
    diam_step      = 0.05
    diameters      = np.arange(0.7, 1.40, 0.05)
    max_candidates = 1000
    collision_tol  = 0.0
    edge_tol       = 0.05
    max_particles  = 200

    # ---- 候选点编码器 (左侧 MLP) ----
    candidate_input_dim  = 5
    candidate_mlp_layers = [32, 64, 128]

    # ---- 图编码器 (右侧 GNN) ----
    graph_input_dim  = 4
    graph_hidden_dim = 128
    gnn_layers       = 3

    # ---- 融合解码器 ----
    embed_dim     = 128
    fusion_layers = [256, 128]

    # ---- 训练超参数 ----
    num_workers      = 8
    num_iterations   = 20
    samples_per_iter = 20
    train_epochs     = 5
    batch_size       = 256
    lr               = 3e-4
    gamma            = 0.99
    temperature      = 1.0

    # ---- 输出 ----
    log_file      = "v7.0_train_log.csv"
    ckpt_prefix   = "construct_v7.0"
    save_interval = 5

    # ---- 评测 ----
    eval_episodes    = 30
    eval_temperature = 1.0
    eval_report_file = "v7.0_eval_report.txt"
    eval_conf_file   = "v7.0_best_packing.conf"
