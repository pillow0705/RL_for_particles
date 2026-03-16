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

⚠️ **用户明确指示（最高优先级，不可违背）**：
- **禁止**再做嵌入型实验（[0.4,0.8,1.4] 等 min/max < 0.414 的配置）
- **必须**聚焦 [0.7, 1.4] 区间 + 模型参数扩展两个方向
- 过去 20 轮 smoke 结论"天花板~0.672"是错误结论，没有给过长跑和大模型机会

1. **[0.7,1.4] 区间深度探索（每次 loop 必须有进展）**：
   - 当前最优基线：8峰 arange(0.7,1.41,0.1)，eval phi_mean=0.6722（仅 20 轮 smoke，未充分）
   - 待探索：更高温度（T=8.0, T=10.0）、更多样本（samples=100,200）、长跑（100轮）
   - 待探索：非均匀峰分布（偏密小端/大端）、更细步长 arange(0.025)

2. **模型参数扩展（从未探索，立即展开）**：
   - embed_dim=256（当前128），transformer_ffn_dim=512，candidate_mlp_layers=[64,128,256]
   - transformer_layers=4 或 6（更深）
   - 结合 [0.7,1.4] 配置测试，先 20 轮 smoke，有效则投长跑

3. **资源分配原则**：两台服务器始终保持运行，优先 [0.7,1.4] + 大模型组合，不做嵌入型。

4. **每次 loop 记录**：注明本次为"[0.7,1.4]+模型扩展探索"，说明具体安排。

**当前已知最优（更新）**：
- [0.7,1.4] 8峰 T=5.0 标准模型：eval phi_mean=0.6722（20轮，**未充分探索**）
- 嵌入型历史最高：phi_max=0.7621（已停止探索该方向）

---
S2 服务器排查

每次 loop 检查 S2 采样/训练时间是否正常（正常应 <200s/iter）。若异常，排查原因：检查 CPU 占用、内存/swap、残留 python 进程数，找到根因后解决，不要只是重启了事。

---
把以上全部内容复制到 /loop 15m 后面即可。
