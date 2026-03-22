# RL for Particle Packing

Using reinforcement learning to construct dense random packings of polydisperse spheres, maximizing volume fraction φ.

---

## Problem Background

Dense random packing of spheres is a fundamental problem in condensed matter physics, materials science, and granular media. For **monodisperse** spheres, random close packing (RCP) reaches φ ≈ 0.64. Introducing **polydispersity** (a mixture of different sizes) can push φ significantly higher, but finding the optimal placement sequence is a combinatorial problem that scales poorly with particle count.

Traditional approaches include:
- **Random sequential addition (RSA)**: place particles randomly until jamming — simple but terminates well below RCP
- **Compression algorithms** (Lubachevsky-Stillinger): start with small spheres and inflate — reaches high φ but is slow and not constructive
- **Event-driven MD**: physically simulate compression — accurate but computationally expensive

This project takes a different angle: frame the packing construction as a **sequential decision problem** and train a policy via reinforcement learning to learn where to place each particle.

---

## Task Definition

A polydisperse sphere packing is built step by step inside a cubic box with **periodic boundary conditions (PBC)**:

- Box size: determined by target φ = 0.72 and N = 2000 particles
- Particle diameters: 14 types uniformly spaced in [0.70, 1.35]
- At each step: the agent picks one position from a set of **candidate placement points**
- Episode ends when N = 2000 particles are placed or no valid candidates remain
- Reward: final volume fraction φ (sparse, only at episode end)

Candidate points are generated analytically using **three-sphere tangency**: given any three already-placed particles A, B, C, a new sphere simultaneously tangent to all three has a closed-form center position. This ensures every candidate is physically contact-consistent from construction.

---

## Model Architecture

```
Placed particles (N × 4)          Candidate positions (≤2000 × 5)
[Δx, Δy, Δz, r/r_max]             [x, y, z, r, coord_number]
        │                                      │
  Linear(4→256) + ReLU              MLP(5→64→128→256) + ReLU
        │                                      │
  prepend [CLS] token                          │
        │                                      │
  Transformer Encoder                          │
  (4 layers, 8 heads, d=256)                   │
        │                                      │
  CLS output (256-dim)  ──── dot product ───── candidate embeddings
                                      │
                              mask invalid positions
                                      │
                            softmax(score / T) → sample
```

**Key design choices:**
- No positional encoding: particle coordinates already carry spatial information, and particles have no meaningful ordering
- Dot-product scoring: CLS vector encodes "what the current packing needs"; candidate embeddings encode "what this position offers" — their dot product measures compatibility
- Incremental candidate maintenance: updating the candidate set after each placement costs O(neighbors²) per step instead of O(N³) for full rebuild

---

## Training Algorithm

**REINFORCE** with advantage filtering:

```
for iteration in range(50):
    1. Collect 50 episodes with current policy (T=5.0 for exploration)
       → iteration 1 uses random policy to establish baseline
    2. Return for every step = final φ of that episode
    3. Normalize advantage = (φ - mean) / std across all steps
    4. Keep only the top 5% steps by |advantage| (filter_ratio=0.95)
    5. Train 2 epochs, batch_size=32
       loss = -mean(log π(a|s) × advantage)
    6. Gradient clipping max_norm=1.0
    7. If phi_max drops > 0.01 from previous best: rollback weights, lr × 0.5
```

The aggressive filtering (keeping only 5% of steps) focuses gradient updates on the most informative transitions, reducing noise from the dense reward signal.

---

## Experimental Results

| Experiment | N | φ_max | Notes |
|-----------|---|-------|-------|
| Random baseline | 2000 | ~0.587 | No policy, random candidate selection |
| RL training (S1) | 2000 | **0.711** | σ=[0.3,0.6], 50 iter, stable plateau |
| RL training (S2) | 2000 | **0.696** | σ=[0.35,0.7], 52 iter, self-recovered |

The trained policy consistently surpasses the random baseline by ~0.12 in φ and approaches known RCP upper bounds for polydisperse systems. Training stabilizes after ~40 iterations with φ_max fluctuating < 0.002.

---

## Repository Structure

```
config.py        hyperparameters (Config class)
physics.py       vectorized numpy physics core (three-sphere solver, collision detection)
env.py           ConstructEnv — incremental candidate set maintenance, O(N) update
model.py         PackingPolicy — CandidateEncoder + Transformer + dot-product scoring
collector.py     DataCollector — multiprocessing collection (16 workers)
trainer.py       REINFORCE Trainer — advantage filtering + rollback
train.py         main entry: train() / evaluate()
utils.py         experiment dir, logging Tee, checkpoint helpers
run.sh           Tencent Cloud GPU server one-click script
requirements_cpu.txt   CPU dependencies (torch CPU + numpy)
requirements_gpu.txt   GPU dependencies (cu121)
loop_log/        per-iteration experiment logs
experiments/     training outputs (auto-created per run)
```

---

## Quick Start

```bash
# CPU (local)
pip install -r requirements_cpu.txt
python train.py

# Monitor training
tail -f /tmp/train.log
```

Output is written to `experiments/<timestamp>/`:
```
config.json          hyperparameter snapshot
run.log              full training log
train_log.csv        per-iteration: phi, loss, adv_var, avg_cands, ...
best_packing.conf    best packing coordinates (x y z r per line, last line: Lx Ly Lz)
construct_v7.0_best.pth
```
