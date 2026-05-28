# NN Training Journey: v1 → v6

完整记录 AlphaZero 风格自学习在这个游戏上的实战调试过程，含 5 次失败 + 1 次部分成功。

## TL;DR

- **NN 路径在这个游戏天花板 ~23-25%**，无法反超 expectimax 的 40.4%
- **根因**：游戏状态空间小、动作空间小、概率结构化 —— expectimax 已接近最优解
- **意外收获**：v5 raw policy 21.5%（持平 v1）但**速度快 1000×**，适合超低延迟场景
- 复现了多个经典坑：z=±1 高方差、placement reward 求稳陷阱、bootstrapping 失败

## 完整版本对比

| 版本 | 训练量 | leaf eval | reward | dirichlet | raw 胜率 | MCTS 胜率 | value_loss | 教训 |
|---|---|---|---|---|---|---|---|---|
| **v1** | 360 game | expectimax rollout | binary z=±1 | ✗ | 21.5% | **30.0%** | — | 历史最强 MCTS |
| v2 | +1800g | 纯 NN value | binary | ✗ | 16.0% ↓ | 18.0% ↓ | 0.6 | 过拟合退化 |
| v3 | +1800g | 纯 NN value | binary | ✓ | 16.0% | 15.5% | 0.6 | 探索不够救不回 |
| v4 | +1800g | 纯 NN value | 对称 placement (+1/+⅓/-⅓/-1) | ✓ | **8.0%** | **9.5%** | 0.42 | 求稳陷阱 |
| **v5** | +1800g | 纯 NN value | **不对称 (+1/-⅓/-⅔/-1)** | ✓ | **21.5%** ✓ | 21.5% | 0.46 | raw 追平 v1 |
| v6 | +3600g | hybrid (NN+EV) | 不对称 | ✓ | 17.5% ↓ | 23.0% | 0.45 | plateau，过拟合边缘 |

## 关键诊断：value head 健康度（v3 时跑的）

`diagnose_value.py` 直接打开了黑盒：

```
[v3 model]
pred  mean=-0.451  std=0.436      ← E[z] ≈ -0.5（4 人下 binary 理论值）
corr(pred, z_binary)    = +0.392  ← 真有学习信号
corr(pred, z_placement) = +0.504  ← 与 placement 对齐更好（提示换 reward）

[Leader-state sanity]
落后 30+ 时:  97.6% 预测 < 0   ← value head 学到了
领先 30+ 时:  83.8% 预测 > 0   ← value head 学到了
```

诊断脚本是这次最值钱的 30 分钟投入 —— 没它就会盲改。

## 6 个失败原因清单（按发现顺序）

### 1. z=±1 方差太大（v2/v3）

4 人 winner-takes-all：
- `E[z] = -0.5`，`Var(z) = 0.75`
- 常数预测器 -0.5 的 MSE = 0.75
- 我们 value_loss 卡 0.6 ≈ **仅比"啥都不学"好一点**

**论文**：Petosa & Balch 2019, *Multiplayer AlphaZero*

### 2. 探索不够（v2）

self-play 里所有 4 个 seat 都 argmax NN 输出 → 数据同质化 → 模型在 memorize 自己。
**修法**：dirichlet noise (eps=0.25) + temperature=1.0 visit-sampling。

### 3. 对称 placement 求稳陷阱（v4）

reward `+1/+⅓/-⅓/-1` 让模型学到"避免 4th"，但游戏目标是"成为 1st"。
模型早 fold 拿中游 → 永远不到 200 分 → 胜率从 16% 暴跌到 8%。

### 4. 不对称 placement 救回（v5）

`+1/-⅓/-⅔/-1` 明确"只有 1st 才好"。
- Var(z) = 0.64（仍低于 binary 0.75）
- 不再奖励"求稳"
- raw policy 立刻回到 21.5%

### 5. Bootstrapping：NN value head 评估 leaf 是悖论（贯穿 v2-v6）

v1 用 expectimax rollout 评估 leaf —— 真实信号。
v2+ 用 NN value head —— "模型自己评自己"，错误自我强化。

**理论上**需要数百万 game 才能让 NN value 自举到接近 expectimax 精度。
**实际上**这游戏 expectimax 已经接近最优，NN 永远不会超过。

### 6. Hybrid leaf eval 是冗余信号（v6）

我们的 hybrid 用 `(my_total - others_max) / target`，但 v5 NN value 已经学到这个（诊断里 corr=0.572）。
**两个一样的信号叠加 = 没有新信息。**

要让 hybrid 真有用，需要更精确的 expectimax depth=2-3 估值，但那慢得跟原 MCTS 一样。

## 实证发现：raw policy 是真正的 NN 价值所在

| | raw policy (NN) | batched MCTS | Expectimax |
|---|---|---|---|
| v5 胜率 | 21.5% | 21.5% | 40.4% |
| 决策延迟 | **0.05 ms** | 5 ms | 1 ms |
| **吞吐倍数** | **1×** | 1/100× | 1/20× |

如果你的场景是"每秒决策 10000+ 次"，raw policy 是唯一能用的。这是 NN 路径在这种简单游戏里的真实价值 —— **不是更强，是更快**。

## 部署矩阵

| 场景 | 选哪个 | 胜率 | 延迟 |
|---|---|---|---|
| 离线最强 | `ExpectimaxAgent(depth=3)` | **40.4%** | ~1ms |
| 中庸折中 | v1 模型 + `BatchedNeuralMCTSAgent(hybrid_alpha=0.7)` | ~25-28% | ~5ms |
| **超低延迟** | **v5 模型 + raw policy** | 21.5% | **0.05ms** |
| 兜底基线 | `EVAgent`（手工启发式） | 35% | 0.1ms |

## 想真正让 NN 超过 expectimax 需要什么

不是这个项目的目标，但记下来给后人：

1. **训练量 ×100**：数百万 game（云 A100 一周）
2. **网络容量 ×10**：3 层 MLP → ResNet 20 层
3. **leaf eval 用 simulation rollout**：仍然慢但准（牺牲 batched 速度）
4. **CFR 风格**：counterfactual regret，更适合不完全信息博弈
5. **对手建模**：维护对手手牌唯一数估计 + 威胁评分

预期上限：~45%（比 expectimax +5pp）。投入产出比极差。

## 文件清单

- `agents/batched_mcts.py` — 含 hybrid_alpha / dirichlet / temperature 参数
- `train/selfplay.py` — placement reward + 探索噪声 + multiprocessing
- `diagnose_value.py` — value head 健康度检查（强烈推荐再次实验前先跑）
- `model_v1_360game.pt` — 历史最强 MCTS 基线（30%）
- `model_v5_iter15.pt` — raw policy 最佳（21.5% + 速度）
- `model_v2_iter15.pt` — 反例样本（演示 binary z 退化）
