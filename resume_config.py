from config import Config


class ResumeConfig(Config):
    # ----------------------------------------------------------------
    # 必填：要继续的实验目录
    # ----------------------------------------------------------------
    resume_from = "experiments/20260315_143022"   # 改成你要续训的目录

    # ----------------------------------------------------------------
    # 可调整的训练参数
    # ----------------------------------------------------------------
    num_iterations   = 20     # 再训练多少轮
    samples_per_iter = 20
    train_epochs     = 5
    lr               = 1e-4   # 续训建议用更小的学习率
    temperature      = 1.0
    num_workers      = 8
    save_data        = True

    # ----------------------------------------------------------------
    # 以下参数必须与原实验保持一致，不要修改
    # 修改网络结构参数会导致权重加载失败
    # 修改物理参数会导致模型行为不一致
    # ----------------------------------------------------------------
    # target_N, target_phi, diameters, collision_tol, edge_tol
    # candidate_input_dim, candidate_mlp_layers, embed_dim
    # graph_input_dim, graph_hidden_dim, gnn_layers, fusion_layers
