"""
resume.py — 从已有实验继续训练
用法: python resume.py
在 resume_config.py 中设置 resume_from 和其他参数。
"""

import csv
import pathlib
import sys
import time
import multiprocessing as mp

import numpy as np
import torch

from resume_config import ResumeConfig
from model import PackingPolicy
from collector import DataCollector
from trainer import Trainer
from utils import _Tee, create_experiment_dir, save_config, save_trajectories


def _find_latest_checkpoint(exp_dir: pathlib.Path) -> pathlib.Path:
    """在实验目录中找最新的 checkpoint（优先 final，其次最大 iter 编号）。"""
    final = exp_dir / "construct_v7.0_final.pth"
    if final.exists():
        return final

    ckpts = sorted(exp_dir.glob("construct_v7.0_iter*.pth"))
    if ckpts:
        return ckpts[-1]   # iter 编号最大的

    raise FileNotFoundError(f"在 {exp_dir} 中没有找到任何 checkpoint。")


def resume():
    cfg    = ResumeConfig()
    device = cfg.device

    resume_dir = pathlib.Path(cfg.resume_from)
    if not resume_dir.exists():
        raise FileNotFoundError(f"续训目录不存在: {resume_dir}")

    ckpt_path = _find_latest_checkpoint(resume_dir)

    # 新建实验目录（续训结果单独存放）
    exp_dir = create_experiment_dir()
    save_config(exp_dir)

    cfg.log_file         = str(exp_dir / "train_log.csv")
    cfg.ckpt_prefix      = str(exp_dir / "construct_v7.0")
    cfg.eval_report_file = str(exp_dir / "eval_report.txt")
    cfg.eval_conf_file   = str(exp_dir / "best_packing.conf")

    log_txt    = open(exp_dir / "run.log", 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_txt)

    try:
        print(f"续训实验目录: {exp_dir}")
        print(f"加载权重: {ckpt_path}")
        print(f"使用设备: {device}")
        print(f"目标粒子数: {cfg.target_N}  目标体积分数: {cfg.target_phi}")

        policy = PackingPolicy(cfg).to(device)
        policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("模型权重加载成功。")

        collector = DataCollector(cfg)
        trainer   = Trainer(policy, cfg)

        log_f  = open(cfg.log_file, 'w', newline='')
        writer = csv.writer(log_f)
        writer.writerow(["Iteration", "PhiMean", "PhiMax", "PhiMin",
                         "AvgSteps", "Loss",
                         "AvgCandsBefore", "AvgFiltered", "AvgAdded", "AvgCandsAfter"])

        for iteration in range(cfg.num_iterations):
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"迭代 {iteration+1}/{cfg.num_iterations}")

            print(f"  采集 {cfg.samples_per_iter} 个样本 (模型策略)...")
            trajs = collector.collect(policy, cfg.samples_per_iter, greedy=False)

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

            print(f"  训练 {cfg.train_epochs} epoch ...")
            loss = trainer.train(trajs)
            print(f"  loss={loss:.4f}  耗时={time.time()-t0:.1f}s")

            writer.writerow([iteration + 1, phi_mean, phi_max, phi_min,
                             avg_steps, loss,
                             round(avg_before, 2), round(avg_filtered, 2),
                             round(avg_added, 2),  round(avg_after, 2)])
            log_f.flush()

            if (iteration + 1) % cfg.save_interval == 0:
                ckpt = f"{cfg.ckpt_prefix}_iter{iteration+1}.pth"
                torch.save(policy.state_dict(), ckpt)
                print(f"  已保存 checkpoint: {ckpt}")

        log_f.close()
        final_ckpt = f"{cfg.ckpt_prefix}_final.pth"
        torch.save(policy.state_dict(), final_ckpt)
        print(f"\n续训完成！最终模型已保存至 {final_ckpt}")

        from train import evaluate
        evaluate(policy, cfg)

    finally:
        sys.stdout = sys.__stdout__
        log_txt.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    resume()
