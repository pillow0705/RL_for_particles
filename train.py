"""
train.py — 主入口
用法: python train.py
"""

import csv
import sys
import time
import warnings
import multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

from config import Config
from model import PackingPolicy
from collector import DataCollector
from trainer import Trainer
from utils import _Tee, create_experiment_dir, save_config, save_trajectories, save_best_packing


# =====================================================================
# 训练
# =====================================================================
def train():
    cfg    = Config()
    device = cfg.device

    exp_dir = create_experiment_dir()
    save_config(exp_dir)

    cfg.log_file         = str(exp_dir / "train_log.csv")
    cfg.ckpt_prefix      = str(exp_dir / "construct_v7.0")
    cfg.eval_report_file = str(exp_dir / "eval_report.txt")
    cfg.eval_conf_file   = str(exp_dir / "best_packing.conf")

    log_txt    = open(exp_dir / "run.log", 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_txt)

    try:
        print(f"实验目录: {exp_dir}")
        print(f"使用设备: {device}")
        print(f"目标粒子数: {cfg.target_N}  目标体积分数: {cfg.target_phi}")

        policy    = PackingPolicy(cfg).to(device)
        collector = DataCollector(cfg)
        trainer   = Trainer(policy, cfg)

        log_f  = open(cfg.log_file, 'w', newline='')
        writer = csv.writer(log_f)
        writer.writerow(["Iteration", "PhiMean", "PhiMax", "PhiMin",
                         "AvgSteps", "Loss", "AdvVar",
                         "AvgCandsBefore", "AvgFiltered", "AvgAdded", "AvgCandsAfter"])

        best_phi  = -1.0
        prev_phi_max = -1.0

        for iteration in range(cfg.num_iterations):
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"迭代 {iteration+1}/{cfg.num_iterations}  lr={trainer.current_lr():.2e}")

            use_policy = None if iteration == 0 else policy
            print(f"  采集 {cfg.samples_per_iter} 个样本 "
                  f"({'随机策略' if use_policy is None else '模型策略'})...")
            trajs = collector.collect(use_policy, cfg.samples_per_iter, greedy=False)

            phis      = [t['phi_final'] for t in trajs]
            steps_cnt = [len(t['steps']) for t in trajs]
            phi_mean  = np.mean(phis)
            phi_max   = np.max(phis)
            phi_min   = np.min(phis)
            avg_steps = np.mean(steps_cnt)
            print(f"  phi: mean={phi_mean:.4f}  max={phi_max:.4f}  "
                  f"min={phi_min:.4f}  avg_steps={avg_steps:.1f}")

            if cfg.save_data:
                data_file = save_trajectories(trajs, exp_dir, iteration + 1)
                print(f"  数据已保存: {data_file}")

            all_cs       = [s['cand_stats'] for t in trajs for s in t['steps']]
            avg_before   = np.mean([c['n_before']   for c in all_cs])
            avg_filtered = np.mean([c['n_filtered'] for c in all_cs])
            avg_added    = np.mean([c['n_added']    for c in all_cs])
            avg_after    = np.mean([c['n_after']    for c in all_cs])
            print(f"  候选点: before={avg_before:.1f}  "
                  f"filtered={avg_filtered:.1f}  added={avg_added:.1f}  "
                  f"after={avg_after:.1f}")

            # 退步检测：回滚后跳过本轮训练，用恢复的模型重新采集
            rolled_back = False
            if prev_phi_max > 0 and phi_max < prev_phi_max - cfg.rollback_tol:
                print(f"  [退步] phi_max {phi_max:.4f} < {prev_phi_max:.4f} - {cfg.rollback_tol}")
                trainer.rollback()
                rolled_back = True

            if not rolled_back:
                print(f"  训练 {cfg.train_epochs} epoch ...")
                loss, adv_var = trainer.train(trajs)
                trainer.backup()   # 训练完成后备份，供下轮可能的回滚使用
                print(f"  loss={loss:.4f}  adv_var={adv_var:.4f}  耗时={time.time()-t0:.1f}s")
                prev_phi_max = phi_max   # 只在正常训练后更新基准
            else:
                loss, adv_var = float('nan'), float('nan')
                print(f"  [跳过训练] 耗时={time.time()-t0:.1f}s  下轮用恢复的模型重新采集")

            writer.writerow([iteration + 1, phi_mean, phi_max, phi_min,
                             avg_steps, loss, round(adv_var, 4),
                             round(avg_before, 2), round(avg_filtered, 2),
                             round(avg_added, 2),  round(avg_after, 2)])
            log_f.flush()

            new_best = save_best_packing(trajs, best_phi, exp_dir)
            if new_best > best_phi:
                best_phi = new_best
                ckpt = f"{cfg.ckpt_prefix}_best.pth"
                torch.save(policy.state_dict(), ckpt)
                print(f"  *** 新纪录！phi_max={best_phi:.4f}  已保存: {ckpt} + best_packing.conf ***")

        log_f.close()
        final_ckpt = f"{cfg.ckpt_prefix}_final.pth"
        torch.save(policy.state_dict(), final_ckpt)
        print(f"\n训练完成！最终模型已保存至 {final_ckpt}")

        evaluate(policy, cfg)

    finally:
        sys.stdout = sys.__stdout__
        log_txt.close()


# =====================================================================
# 评测
# =====================================================================
def evaluate(policy: PackingPolicy, cfg: Config):
    device = cfg.device
    policy.eval()

    orig_temp       = cfg.temperature
    cfg.temperature = cfg.eval_temperature
    collector       = DataCollector(cfg)

    print(f"\n{'='*60}")
    print(f"开始评测（采样策略 T={cfg.eval_temperature}，共 {cfg.eval_episodes} 局）...")

    trajs           = collector.collect(policy, n_samples=cfg.eval_episodes, greedy=False)
    cfg.temperature = orig_temp

    records  = []
    best_phi = -1.0

    for traj in trajs:
        pos_arr = traj['final_pos']
        rad_arr = traj['final_rad']
        L_traj  = traj['L']
        n       = len(pos_arr)
        phi     = traj['phi_final']

        coord_counts = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos_arr[j] - pos_arr[i]
                diff -= np.round(diff / L_traj) * L_traj
                dist = np.linalg.norm(diff)
                if dist <= rad_arr[i] + rad_arr[j] + cfg.edge_tol:
                    coord_counts[i] += 1
                    coord_counts[j] += 1
        z_mean = coord_counts.mean()

        diams = rad_arr * 2.0
        pdi   = diams.std() / diams.mean() if diams.mean() > 0 else 0.0

        records.append({'n': n, 'phi': phi, 'z': z_mean, 'pdi': pdi})

        if phi > best_phi:
            best_phi     = phi
            best_pos_arr = pos_arr
            best_rad_arr = rad_arr
            best_L       = L_traj

    phis = [r['phi'] for r in records]
    ns   = [r['n']   for r in records]
    zs   = [r['z']   for r in records]
    pdis = [r['pdi'] for r in records]

    with open(cfg.eval_report_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"评测报告  模型: {cfg.ckpt_prefix}_final.pth\n")
        f.write(f"评测局数: {cfg.eval_episodes}    目标粒子数: {cfg.target_N}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'指标':<20}{'均值':>10}{'最大':>10}{'最小':>10}{'标准差':>10}\n")
        f.write("-" * 60 + "\n")
        for name, vals in [("体积分数 phi", phis), ("粒子数 N", ns),
                            ("平均配位数 Z", zs), ("多分散性 PDI", pdis)]:
            arr = np.array(vals, dtype=float)
            f.write(f"{name:<20}{arr.mean():>10.4f}{arr.max():>10.4f}"
                    f"{arr.min():>10.4f}{arr.std():>10.4f}\n")
        f.write("\n各局明细:\n")
        f.write(f"{'局':<6}{'N':>6}{'phi':>8}{'Z':>8}{'PDI':>8}\n")
        f.write("-" * 40 + "\n")
        for i, r in enumerate(records):
            f.write(f"{i+1:<6}{r['n']:>6}{r['phi']:>8.4f}"
                    f"{r['z']:>8.2f}{r['pdi']:>8.4f}\n")

    print(f"  评测完成  phi: mean={np.mean(phis):.4f}  "
          f"max={np.max(phis):.4f}  min={np.min(phis):.4f}")
    print(f"  报告已保存: {cfg.eval_report_file}")

    if best_phi > 0:
        with open(cfg.eval_conf_file, 'w') as f:
            for p, r in zip(best_pos_arr, best_rad_arr):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r:.6f}\n")
            f.write(f"{best_L:.6f} {best_L:.6f} {best_L:.6f}\n")
        print(f"  最优堆积已保存: {cfg.eval_conf_file}  (phi={best_phi:.4f})")

    policy.train()


# =====================================================================
# 独立生成（加载已有模型）
# =====================================================================
# 从历史数据训练（无需重新采集）
# =====================================================================
def train_from_data():
    """加载 data/ 下所有历史轨迹，直接进行训练。"""
    from utils import load_all_trajectories

    cfg    = Config()
    device = cfg.device

    exp_dir = create_experiment_dir()
    save_config(exp_dir)

    cfg.log_file         = str(exp_dir / "train_log.csv")
    cfg.ckpt_prefix      = str(exp_dir / "construct_v7.0")
    cfg.eval_report_file = str(exp_dir / "eval_report.txt")
    cfg.eval_conf_file   = str(exp_dir / "best_packing.conf")

    log_txt    = open(exp_dir / "run.log", 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_txt)

    try:
        print(f"[train_from_data] 实验目录: {exp_dir}")
        print(f"使用设备: {device}")

        print("\n加载历史轨迹数据...")
        all_trajs = load_all_trajectories()
        if not all_trajs:
            print("没有可用数据，退出。")
            return

        policy  = PackingPolicy(cfg).to(device)
        trainer = Trainer(policy, cfg)

        log_f  = open(cfg.log_file, 'w', newline='')
        writer = csv.writer(log_f)
        writer.writerow(["Epoch", "Loss", "AdvVar"])

        for epoch in range(cfg.train_epochs):
            loss, adv_var = trainer.train(all_trajs)
            print(f"  epoch {epoch+1}/{cfg.train_epochs}  loss={loss:.4f}  adv_var={adv_var:.4f}")
            writer.writerow([epoch + 1, loss, round(adv_var, 4)])
            log_f.flush()

        log_f.close()
        final_ckpt = f"{cfg.ckpt_prefix}_from_data.pth"
        torch.save(policy.state_dict(), final_ckpt)
        print(f"\n训练完成！模型已保存至 {final_ckpt}")

        evaluate(policy, cfg)

    finally:
        sys.stdout = sys.__stdout__
        log_txt.close()


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train()
