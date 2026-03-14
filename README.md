# RL for Particle Packing

Using reinforcement learning to construct high-density polydisperse sphere packings.

## Model Architecture

- **Left branch**: Candidate point encoder (MLP, 5D → 128D)
- **Right branch**: Graph encoder (GNN, dynamic graph of placed particles → 128D)
- **Fusion**: Concatenate + MLP → scalar score per candidate

At each step, all candidate positions are scored and the highest-scoring one is selected.

## Training Algorithm

Iterative REINFORCE (no Critic):
1. Round 0: collect packings with random policy
2. Train model on collected data
3. Use trained model to collect better data
4. Repeat

## Files

| File | Description |
|---|---|
| `RL_Construct_v6.0.py` | Main training code (current version) |
| `RL_Construct_test5.2.py` | Previous version (PPO + PointNet) |

## Quick Start

```bash
pip install torch numba numpy

# Run training (uses GPU 0 by default)
python RL_Construct_v6.0.py

# Specify GPU
CUDA_VISIBLE_DEVICES=6 python RL_Construct_v6.0.py
```

## Output Files

| File | Description |
|---|---|
| `v6.0_train_log.csv` | Training metrics per iteration |
| `v6.0_eval_report.txt` | Final evaluation report (phi, Z, PDI) |
| `v6.0_best_packing.conf` | Best packing coordinates |
| `construct_v6.0_final.pth` | Trained model weights |

## Config

All hyperparameters are in the `Config` class at the top of `RL_Construct_v6.0.py`.

Key parameters:
- `target_N`: number of particles (default: 100)
- `embed_dim`: embedding dimension (default: 128)
- `num_iterations`: training iterations (default: 20)
- `samples_per_iter`: trajectories per iteration (default: 30)
