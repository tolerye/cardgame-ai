# AlphaZero 自学习退化诊断与修复方案

> 注：本次调研期间 WebSearch / WebFetch 工具不可用，结论基于已有的 AlphaZero 文献知识与对你代码的实际审查（`train/selfplay.py`、`train/network.py`、`agents/batched_mcts.py`）。引用列表里的链接是公开资源，未在线复核可访问性，请自行确认。

## 结论先行

**最可能根因：value target 信号在 4 人游戏里被严重稀释，value head 几乎学不到东西，导致 batched MCTS 切到"纯 NN value leaf eval"后整条链路崩溃。** v1 之所以好，是因为 leaf eval 用的是 expectimax rollout（一个真实的、信息丰富的信号），value head 烂不烂无所谓。v2/v3 把 leaf eval 换成 NN value 后，烂 value head 直接污染搜索。

数学证据（直接套你 `selfplay.py` 第 194 行的代码）：4 玩家 winner-takes-all，E[z] = 1·(1/4) + (−1)·(3/4) = **−0.5**，Var(z) = 0.75。一个常数预测 −0.5 的"白痴模型" MSE = 0.75。**你 value_loss 卡在 0.6**，说明模型只比常数基线好一丁点 —— 这跟"学不动"基本等价。policy_loss 能从 0.55 降到 0.48 是因为 MCTS 访问分布本身信息更密、且和具体状态强相关。

## 1. 问题诊断（按概率排序）

| # | 假设 | 证据 | 概率 |
|---|---|---|---|
| **A** | **Value target 设计错误**（4 人 binary win/loss 信号噪声压倒信号） | value_loss 0.6 ≈ 常数基线 0.75 的小幅改进；切 NN value leaf 后立刻退化 | **高** |
| **B** | **Bootstrapping 死循环**：弱 value head → 弱 leaf eval → 弱搜索 → 弱训练数据 → 更弱 value head | v1 用 expectimax rollout 兜底所以稳，v2 抽掉兜底就崩 | **高** |
| **C** | **Reward 太稀疏**（一局 200 分要好几个 round 才决出胜者，但每个动作都只用最终胜负打标） | 单 episode 几十个决策点共享同一个 ±1，每个决策点的 credit assignment 信号噪声极大 | **中-高** |
| D | 数据量不够 | 1-2 万 game 在 2-action、62-feature、30K 参数的小模型上其实够，参考 surag 的 Othello impl 几千局就能 work | 低 |
| E | dirichlet/temperature 超参没救回来是因为根本不是探索问题，是 target 问题 | v3 加探索仍退化，与 A 一致 | — |

## 2. 修复方案（3 条可执行路径）

### 修复 1：把 value target 从 binary 改成 placement-based 连续值（强烈推荐先做）

**改什么**：`train/selfplay.py` `play_one_game` 末尾循环。

```python
# 当前（94 行）：z = 1.0 if idx == winner else -1.0
# 改为基于排名的连续 z，并 zero-mean 化：
final_scores = [p.total_score for p in engine.state.players]
ranks = np.argsort(np.argsort(-np.array(final_scores)))  # 0=1st, 3=4th
# 4 人映射：1st=+1, 2nd=+1/3, 3rd=-1/3, 4th=-1  (zero-mean, 均匀分布)
z_table = {0: 1.0, 1: 1/3, 2: -1/3, 3: -1.0}
z = z_table[ranks[idx]]
```

- **工程量**：0.5 小时
- **预期成功率**：**高**。这是 Petosa & Balch 2019 *Multiplayer AlphaZero* 的核心做法，把 value head 从"猜 25% 概率事件"变成"预测一个 zero-mean、有梯度的连续量"。MSE 楼板从 0.75 掉到 ~0.55，且每个 state 的信号方差大幅降低。
- **验证方法**：训练 200 game 后看 value_loss 是否能降到 < 0.4；同时画一张 value head 输出 vs 真实 z 的散点图（见诊断方法 5）。

### 修复 2：把 leaf eval 改回 hybrid（rollout + NN）直到 value head 健康

**改什么**：`agents/batched_mcts.py` 的 leaf evaluation 处。

```python
# 伪代码：在 _evaluate_leaves 里
if self.use_hybrid:
    # 50% leaf 用一个浅 expectimax rollout（depth=1 或 2）
    # 50% leaf 用 NN value
    # 或者：取两者加权平均 v = α * v_nn + (1-α) * v_rollout，α 随训练进度从 0 升到 1
    v = alpha * v_nn + (1 - alpha) * v_rollout
```

`α` 调度：前 1000 game `α=0`（全 rollout），1000-3000 线性升到 0.5，之后到 1.0。

- **工程量**：3-4 小时（要给 batched MCTS 加一个轻量 rollout path；可以复用 `agents/expectimax_agent.py` depth=1）
- **预期成功率**：**中-高**。代价是 self-play 慢一截，但能直接复刻 v1 的稳定性，同时让 NN 慢慢被 bootstrap 起来。这是 Anthony et al. *Thinking Fast and Slow with Deep Learning and Tree Search* (ExIt) 的核心思想：teacher (rollout) → student (NN) 平滑过渡。
- **验证方法**：α=0 时应该立刻能复现 v1 的 30% 胜率基线；α 升到 1.0 后胜率不退化即成功。

### 修复 3：加 round-level 的辅助 value target（多任务）

**改什么**：`network.py` 加第三个 head；`selfplay.py` 在每个 round 结束时记录 round-level outcome。

```python
# network.py:
self.value_head_match = nn.Sequential(nn.Linear(h, 1), nn.Tanh())  # 整局
self.value_head_round = nn.Sequential(nn.Linear(h, 1), nn.Tanh())  # 本 round 净分

# selfplay.py: 训练时
loss = policy_loss + value_loss_match + 0.5 * value_loss_round
```

round-level 信号：本 round 你的 locked_score - 平均其他 player locked_score，归一化到 [-1, 1]。

- **工程量**：4-6 小时（要改 engine 暴露 round 边界、加 example 字段、改 train_step、改 batched_mcts 的 leaf eval 决定用哪个 head）
- **预期成功率**：**中**。理论上很对路（dense reward 是 sparse RL 的标配解药），但你只有 2 个动作 + 60 维 feature，可能修复 1+2 就够了，多任务反而引入复杂度。
- **验证方法**：value_loss_round 应该比 value_loss_match 降得快很多（信号密、噪声小）；最终评估时只看 batched MCTS 胜率。

## 3. value head 健康度诊断方法（先做这个再决定改哪个）

**3 行代码就能跑**，加到 `eval_neural.py` 里：

```python
# 跑 100 局 self-play，收集 (state, true_final_z) 对
# 用 model 预测 v_pred，画 scatter
# 健康指标：
#   - corr(v_pred, true_z) > 0.3
#   - v_pred 的方差 > 0.05（不是塌缩成常数）
#   - 在 "我已经赢了 (total_score >= 200)" 的 state 上 v_pred 应该 > 0.5
#   - 在 "我已经输了 (有人赢了)" 的 state 上 v_pred 应该 < -0.5
```

如果第 3、4 条 sanity check 都不过，value head 就是没学到任何东西，先做修复 1。

## 4. 关键资源

1. **Petosa & Balch 2019, *Multiplayer AlphaZero*** (arXiv:1910.13012) — 直接讨论 N>2 玩家场景下 value target 用 placement / softmax-rank，不是 binary。**最相关**。
2. **Surag Nair 的 alpha-zero-general 教程** (suragnair/alpha-zero-general on GitHub + web.stanford.edu/~surag/posts/alphazero.html) — 小规模 (Othello 6×6) self-play 几千局成功的工程参考实现，其 issue tracker 里有大量"value loss 不降"的真实案例和解法。
3. **Anthony, Tian, Barber 2017, *Thinking Fast and Slow with Deep Learning and Tree Search* (ExIt)** (NeurIPS 2017) — hybrid teacher-student leaf eval 的理论基础，对应修复 2。
4. **Cazenave 等人在 Hex/Polygames 上的工作** — 小规模 AlphaZero 在简单游戏上的实战配方，强调 value head warm-up 阶段必须有非 NN 的兜底信号。
5. **DeepMind, *Player of Games* (2021, arXiv:2112.03178)** — 含随机和不完美信息的扩展，对你"含弃牌堆/抽牌随机"场景的 reward design 有参考。

## 推荐执行顺序

1. **今天**：先做"诊断方法"（30 分钟），确认 value head 是不是真的没在学。
2. **明天**：做"修复 1（placement value target）"，0.5 小时，跑 500 game 看 value_loss 是否破 0.4。
3. **如果还不行**：做"修复 2（hybrid leaf eval）"，恢复 v1 基线，再渐进切换。
4. **修复 3 暂不做**，留作后手。

不要先调超参（lr/n_sims/c_puct），目标信号没修好之前调超参是噪声里捞噪声。
