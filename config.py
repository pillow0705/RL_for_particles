import numpy as np
import torch


class Config:
    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 物理 / 环境参数 ----
    target_N       = 2000
    target_phi     = 0.72
    diameters      = np.arange(0.7, 1.40, 0.05)
    max_candidates = 2000
    collision_tol  = 0.05
    edge_tol       = 0.05   # 评测时计算配位数用
    max_particles  = 3000

    # ---- 候选点编码器 (MLP) ----
    candidate_input_dim  = 5
    candidate_mlp_layers = [64, 128, 256]

    # ---- 粒子 Transformer 编码器 ----
    embed_dim            = 256   # 输出嵌入维度（与候选点编码器对齐，用于点积）
    transformer_d_model  = 256   # Transformer 内部宽度
    transformer_nhead    = 8     # 注意力头数
    transformer_layers   = 4     # Encoder 层数
    transformer_ffn_dim  = 512   # FFN 隐层宽度

    # ---- 训练超参数 ----
    num_workers      = 16
    num_iterations   = 50
    samples_per_iter = 50
    train_epochs     = 2
    advantage_filter_ratio = 0.95   # 丢弃 |advantage| 最小的 95%，只保留最强信号
    batch_size       = 32
    lr               = 5e-4
    gamma            = 0.99
    temperature      = 5.0

    # ---- 输出 ----
    log_file      = "v7.0_train_log.csv"
    ckpt_prefix   = "construct_v7.0"
    save_data     = False
    rollback_tol  = 0.01

    # ---- 评测 ----
    eval_episodes    = 20
    eval_temperature = 1.0
    eval_report_file = "v7.0_eval_report.txt"
    eval_conf_file   = "v7.0_best_packing.conf"
