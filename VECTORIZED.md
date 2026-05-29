# Vectorized Self-Play 改造路线图

> Fork 自 [tolerye/cardgame-ai](https://github.com/tolerye/cardgame-ai)，目的是让 NN 训练真正用上 GPU。

## 为什么要改

当前 cardgame-ai 主仓库的训练架构在 GPU 上是**反向收益**：

| 配置 | 实测 game/s | GPU 利用率 |
|---|---|---|
| M4 + workers=6 + cpu | **1.7** | 0% |
| 笔记本 14 核 + workers=14 + cpu | 0.6 | 0% |
| **笔记本 3070 Ti + workers=1 + cuda** | **0.083** | **10%** |

GPU 比 CPU 慢 7-21 倍。原因明确：

```
单 game 决策时：
- GPU 推理 1 batch ≈ 1ms
- CPU 跑 game 逻辑 + 树搜索 ≈ 50ms
=> GPU 50/51 = 98% 时间在等
```

## 改造目标

让 N 个 self-play game **同步推进**，所有 NN 推理凑成 batch=N×32 一起送 GPU。

### 预期吞吐（3070 Ti laptop）

| 配置 | 当前 | 改造后 | 提升 |
|---|---|---|---|
| game/s | 0.6（CPU 14 worker） | **3–5** | 5–8× |
| GPU 利用率 | 0% | 50–70% | — |
| 12h 训练量 | 25000 game | **130000–200000 game** | — |

## 核心架构（方案 A：Vectorized）

```python
def parallel_selfplay(model, n_concurrent=32, n_sims=80, device='cuda'):
    games = [GameState() for _ in range(n_concurrent)]
    while not all(g.done for g in games):
        active = [g for g in games if not g.done]

        # ============ MCTS 同步推进 ============
        for sim in range(n_sims):
            # 1. 每个 game 各自走到 leaf（CPU）
            leaves = [g.mcts_select_leaf() for g in active]

            # 2. 一次性 batch 推理（GPU 这里真用上）
            X = torch.stack([encode(l.state, l.my_idx) for l in leaves]).to(device)
            with torch.no_grad():
                policies, values = model(X)  # batch = N

            # 3. 各自 backup
            for g, p, v in zip(active, policies, values):
                g.mcts_backup(p.cpu(), v.item())

        # ============ 各 game 决策 + 应用 ============
        for g in active:
            g.apply(g.best_action())
```

关键：
- **单进程**（绕开 multiprocessing + CUDA 死锁问题）
- **CPU 跑游戏逻辑，GPU 跑 NN**，两者 pipeline
- **同步推进**简化代码（异步 game-state 也可以但复杂得多）

## 改造步骤

### Step 1: 抽出 BatchedMCTS 的 leaf-eval 逻辑（0.5 天）
- 把 `agents/batched_mcts.py` 里 `_simulate_to_frontier` 和 backup 拆成可独立调用的方法
- 让外部能控制"先收集 N 个 leaf → batch 推理 → 各自 backup"的循环

### Step 2: 实现 ParallelSelfPlay（0.5 天）
- 新文件 `train/parallel_selfplay.py`
- 持有 N 个 GameState + N 个 BatchedMCTS 实例
- 主循环按上述伪代码

### Step 3: 集成 train_overnight.py（0.2 天）
- 加 `--parallel-games N` 参数
- 当 N > 1 时切换到 ParallelSelfPlay 路径

### Step 4: GPU 调优（0.3 天）
- 测不同 N（16/32/64/128）找最优
- 测 mixed precision (fp16) 是否进一步加速
- 测 model.compile() 是否帮助

### Step 5: 验证收敛（持续）
- 训练 1000 game 看 raw policy 胜率是否仍在涨
- 对比主仓库 model_best 看效果是否一致

## 不做的事

- ❌ 多进程 + GPU server（IPC 复杂，容易死锁）
- ❌ 完全 GPU 化 game logic（state 全 tensor 化，工作量太大）
- ❌ 修改 game engine（rules 已在主仓库验证过 6/6 测试）

## 时间估算

完整一天工作（搭建 + 调试 + 验证）。建议**新一轮 fresh session**专注做这件事，不和其他事混。

## 同步策略

- **改造期间** 主仓库继续训练（M4 + 笔记本 CPU），不停
- **完成后** 把训练好的 model 用主仓库 eval_neural.py 评测，确认胜率不退步
- **正式发布** 把 vectorized 代码 PR 回主仓库

## 当前状态

- ✅ Repo 初始化，复制 baseline 代码
- ⏳ 等待新 session 开始实现
