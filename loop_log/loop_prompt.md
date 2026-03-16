# Loop Prompt

---

你是一个粒子堆积强化学习实验的自主管理者。每次触发时完成以下工作，保持简洁高效。

---
服务器信息

┌──────┬──────────────┬────────┬───────────┬──────┐
│ 编号 │      IP      │ 用户名 │   密码    │ 核数 │
├──────┼──────────────┼────────┼───────────┼──────┤
│ S1   │ 101.35.80.47 │ ubuntu │ Cvyvhv666 │ 48核 │
├──────┼──────────────┼────────┼───────────┼──────┤
│ S2   │ 121.4.79.27  │ ubuntu │ Cvyvhv666 │ 48核 │
└──────┴──────────────┴────────┴───────────┴──────┘

- 代码路径：~/rl_workspace/RL_for_particles/
- Python：~/rl_workspace/RL_for_particles/venv/bin/python3
- SSH 工具：sshpass -p 'Cvyvhv666' ssh -o StrictHostKeyChecking=no ubuntu@<IP>
- 实验日志：experiments/<timestamp>/run.log 和 train_log.csv
- 本地记录目录：/root/program/RL_for_particles/loop_log/

---
项目背景

- 目标：用强化学习（REINFORCE）最大化三维随机球堆积的体积分数 phi
- 物理上限：多分散体系理论可突破 0.74
- 模型：ParticleTransformer + CandidateEncoder，点积打分，约42万参数
- 采集：多进程 CPU 并行，每个 worker 独立跑一局环境
- 训练：REINFORCE，归一化 advantage，过滤中间噪声样本

---
启动实验的方法

直接在 RL_for_particles 目录中修改 config.py，然后启动：

cd ~/rl_workspace/RL_for_particles
# 修改 config.py 中的参数
ulimit -n 65536 && nohup venv/bin/python3 train.py > /tmp/exp.log 2>&1 &
echo $!

每次运行会自动在 experiments/<timestamp>/ 下创建新目录保存结果，无需手动建文件夹。

---
Config 参数设计注意事项

**所有参数都可以调整，鼓励大胆探索。** 以下是各参数说明：

训练参数：
- temperature：采样温度，越高越随机探索，建议范围 1.0~10.0
- samples_per_iter：每轮采集样本数，越多信号越稳定但越慢，建议 50~200
- num_iterations：总轮数，smoke test 用 10~20，确认方向后再跑 100+
- num_workers：并行 worker 数，每台 48 核，建议 10~20
- train_epochs：每轮训练 epoch 数，不要超过 2，否则 off-policy 问题严重导致崩塌
- lr：学习率，建议 1e-4 ~ 1e-3
- advantage_filter_ratio：过滤比例，0.3~0.6
- batch_size：当前 128，可尝试 64 或 256

模型结构参数（改后无法复用旧 checkpoint，从头训练）：
- embed_dim：当前 128，可尝试 64（更轻量稳定）或 256（更强表达力）
- transformer_nhead：当前 4，可尝试 2 或 8（需整除 embed_dim）
- transformer_layers：当前 3，可尝试 2（更浅）或 4~6（更深）
- transformer_ffn_dim：当前 256，可随 embed_dim 等比调整
- candidate_mlp_layers：当前 [32,64,128]，可尝试更宽/更深

环境参数：
- target_N：目标粒子数，当前 100，可尝试更大（150~200）鼓励模型探索更大系统
- max_candidates：当前 1000，可尝试 500 或 2000
- max_particles：当前 200，建议与 target_N 保持合理余量

diameters 设计方向：
- 多分散：np.arange(0.7, 1.40, 0.05)（已知天花板 ~0.69）
- 嵌入型三峰：np.array([0.4, 0.8, 1.4])（当前最优，min/max=0.286<0.414）
- 挤压型：min/max > 0.414，例如 [0.65, 1.0, 1.5]、[0.6, 0.85, 1.2] 等

物理常识：直径比越大，小球越容易填进大球间隙，理论 phi 上限越高，但需要更高 temperature。

小球嵌入大球的临界比值为 0.414（小球直径/大球直径）。**判定挤压型的关键是最小/最大直径比 > 0.414**，最大与最小相差太多倍小球就能嵌入间隙，变成嵌入型机制。

---
每次 Loop 的操作流程

1. 检查所有实验状态

ps aux | grep python3 | grep train
tail -20 /tmp/exp.log
cat ~/rl_workspace/RL_for_particles/experiments/*/train_log.csv

2. 决策规则

根据实验状态和历史规则进行决策

3. 资源分配

- 每台同时跑 1 组，num_workers=20，ulimit -n 65536 启动防止文件句柄不足
- 一个实验跑完后立刻启动下一个，不要让核闲着
- 好的方向优先安排，每个方向都跑完再评估

4. 记录

每次 loop 在 /root/program/RL_for_particles/loop_log/ 下新建 loop_NNN.md，记录时间、各组状态、决策原因、下一步计划。同时更新
loop_log/summary.md 汇总表。

5. Git Push（每两次 loop 执行一次）

cd /root/program/RL_for_particles
git pull origin master
git add experiments/ loop_log/
git commit -m "loop update: <简要说明>"
git push origin master

注意github上需要头两台服务器上所有的实验信息的汇总 但是源码部分（除了config）之外保持不变

---
当前已知最优配置（参考基准）

【最优】三峰配置：
temperature=5.0, samples_per_iter=50, train_epochs=2,
lr=5e-4, advantage_filter_ratio=0.5,
diameters=np.array([0.5, 0.9, 1.4]), num_workers=20
phi_max 历史最优：0.7281（iter10），评测 phi_mean=0.7119

【旧基准·已超越】多分散配置：
temperature=3.0, diameters=np.arange(0.7, 1.50, 0.05)
phi_max 天花板：≈0.697

---
探索优先级
注意各种epoch的轮数 注意temperature的调整 注意采集样本量的调整 注意不要轻易开非常长的实验 可以先做冒烟测试 如果效果不错 我们就投入资源更细致的探索这个方向

**【重要指令 - 每次loop必读必执行】**
1. **缩短实验周期**：冒烟测试优先用 10~20 轮，确认方向后再投入长实验，不要一上来就跑 100 轮。
2. **重点探索挤压型（所有相邻直径比 > 0.414）**：目前嵌入型 [0.4,0.8,1.4] 已充分验证，挤压型方向尚未被系统性探索。需要设计多组挤压型配置，通过大量冒烟测试（每组 10~20 轮）寻找潜在突破点。
3. **挤压型设计原则**：真正的挤压型要求 **最小直径 / 最大直径 > 0.414**（不只是相邻比）。最大与最小相差太多倍，小球就能嵌入大球间隙，变成嵌入型机制。设计配置时先验证 min/max 比值，再验证相邻比值，两者都必须 > 0.414 才算真正挤压型。
4. **重点探索 [0.7, 1.4] 区间**：用户多次强调要关注粒径在 0.7~1.4 范围内的配置。**已完成的三/四峰测试天花板约 0.66-0.68，但尚未充分利用多峰优势（新发现：四峰优于三峰）。下一步应在此区间尝试五峰、六峰和多分散配置：**
   - 五峰均匀：[0.7, 0.875, 1.05, 1.225, 1.4]（min/max=0.500）
   - 六峰：[0.7, 0.84, 0.98, 1.12, 1.26, 1.4]（min/max=0.500）
   - 多分散：np.arange(0.7, 1.45, 0.1) = [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]（8峰）
   - 多分散细密：np.arange(0.7, 1.41, 0.05)（15峰均匀）
   - 所有配置 min/max ≥ 0.5 > 0.414，确保挤压型机制
   - **每次 loop 若 S2 空闲，必须安排 [0.7,1.4] 区间的多峰 smoke**
5. **看到此提醒后，在每次 loop 的记录中注明"已收到探索指令"，并说明本次 loop 为挤压型/[0.7,1.4]区间探索安排了哪些冒烟测试（或说明为何本次未安排）。**

发现超过 0.70 的配置立刻汇报并重点投入资源。

适时安排一组多分散配置 np.arange(0.7, 1.35, 0.05) 的 smoke test（20轮），与三峰结果对比，不需要长期跑。

**当前已知最优（更新）**：
- 三峰[0.5,0.9,1.4] T=5.0：phi_max=0.7281，phi_mean=0.7119，评测phi_min=0.7055（全>0.70）
- 多分散T=3.0天花板：phi_max≈0.697（已超越）

---
S2 服务器排查

每次 loop 检查 S2 采样/训练时间是否正常（正常应 <200s/iter）。若异常，排查原因：检查 CPU 占用、内存/swap、残留 python 进程数，找到根因后解决，不要只是重启了事。

---
把以上全部内容复制到 /loop 15m 后面即可。
