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
| `RL_Construct_v7.0.py` | Main training code (current version) |
| `RL_Construct_test5.2.py` | Previous version (PPO + PointNet) |

## Quick Start

```bash
pip install torch numba numpy

# Run training (auto-selects GPU if available)
python RL_Construct_v7.0.py

# Specify GPU
CUDA_VISIBLE_DEVICES=0 python RL_Construct_v7.0.py
```

## Output Structure

Each run automatically creates a new folder under `experiments/`:

```
experiments/
  run_001/
    config.json       # Full hyperparameter snapshot
    run.log           # Complete training log (all stdout)
    train_log.csv     # Per-iteration metrics (phi, loss, steps)
    eval_report.txt   # Final evaluation report (phi, Z, PDI)
    best_packing.conf # Best packing coordinates
    construct_v7.0_iter5.pth   # Checkpoint (every save_interval)
    construct_v7.0_final.pth   # Final model weights
  run_002/
    ...
```

## Config

All hyperparameters are in the `Config` class at the top of `RL_Construct_v7.0.py`.

Key parameters:
- `target_N`: number of particles (default: 50)
- `embed_dim`: embedding dimension (default: 128)
- `num_iterations`: training iterations (default: 20)
- `samples_per_iter`: trajectories per iteration (default: 20)
- `num_workers`: parallel workers for data collection (default: 8)

## v7.0 Changes vs v6.0

- Incremental candidate set maintenance (no KDTree rebuild per step)
- `_candidate_set` + `_triplet_set` + `_cand_to_triplets` for O(N) updates
- Experiment folder auto-creation with config snapshot and full log
