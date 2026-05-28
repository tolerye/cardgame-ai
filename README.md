# cardgame-ai

一个针对你描述的"6翻了"卡牌游戏的自动决策算法实现。四个阶段全部落地：精确 EV、启发式修正、MCTS、神经网络自对弈框架。

## 游戏假设
- 4 玩家，目标分 200
- 牌库 94 张：数字 79 + 加分牌 6（**3 张 +10 / 3 张翻倍当前数字总和**）+ 技能 9
- 6 张不同数字 = "6翻了"，触发者 +15 分奖励，其他人当前牌面强制锁分

## 项目结构
```
game/        # 引擎（卡牌/状态/规则）—— 已通过单元测试
agents/      # 决策算法
  baselines.py     # 随机 / 固定阈值贪心
  ev_agent.py      # 阶段①+② 单步 EV + 风险曲线 + 6翻冲刺 + 技能启发式
  expectimax_agent.py  # 阶段②.5 深度受限递归 EV（实测最强）
  mcts_agent.py    # 阶段③ Information-Set MCTS
  neural_agent.py  # 阶段④ 推理用包装
train/       # 阶段④训练栈
  encoder.py       # State → 62 维特征
  network.py       # PyTorch 双头 (policy + value) MLP
  selfplay.py      # NeuralMCTS + 自我对弈数据收集 + 训练步
simulate.py  # 锦标赛仿真器
tests/       # 引擎规则测试（六翻、爆牌±保险、放逐、三连）
```

## 关键设计决策

### 阶段① 精确 EV
- 牌库**完全可观测**（94 张全公开 + 弃牌堆），所以 EV 用精确剩余分布而非估计
- 每个动作的 EV 包含数字命中、爆牌、加分牌、技能牌、6翻终结的全分支期望

### 阶段② 启发式修正
- **风险曲线**：落后越多，draw EV 乘数越大（最高 1.6）；领先时压低（最低 0.7）
- **6翻冲刺**：手上 5 张不同数字时，绕过 EV 比较直接看 P(命中) ≥ P(爆) 就抽
- **结束博弈守卫**：fold 后总分 ≥ 200 时强制 fold（直接赢比赛）
- **技能目标启发式**：放逐打 round_score 最高者；三连打无保险且手牌多者；强制送保险给威胁最低者

### 阶段② 升级：递归 expectimax
- 单步 EV 把"draw 后"用 cur_score+v 近似，是悲观偏差
- Expectimax 递归到 depth 3：每个 chance 节点展开 ~18 个分支，最优 fold/draw 决策反向汇总
- 复杂度 ~5800 leaves/decision，~1ms，**实测胜率最高**

### 阶段③ MCTS（IS-MCTS）
- 每次模拟用 `clone_for_simulation` 重新 determinize 牌库
- Root UCB1 + my-turn 用 expectimax + 对手 EV 的混合 rollout
- **观察**：在这个游戏里 MCTS 没超过 expectimax。原因：动作空间只有 2 个，结构化随机性让 expectimax 的精确递归非常合算；MCTS 的蒙卡误差 + Python 速度劣势抵消了它的优势。如果想让 MCTS 占优，需要走阶段④（NN 引导）

### 阶段④ 神经网络自对弈
- 62 维特征：自手牌 18 + 三对手 24 + 全局 20
- Policy/Value 双头 MLP（默认 128 隐藏单元）
- AlphaZero loop：NeuralMCTS → 收集 (state, π_visits, z) → 监督训练
- 框架完整可运行，**训练需要时间**：建议 50+ iter × 20 game 才能看到提升

## 快速验证

```bash
# 单元测试
python3 tests/test_engine.py

# 锦标赛（1000 局，~3 分钟）
python3 simulate.py --agents exmax3 ev greedy random -n 1000

# 神经网络训练（需 pip install torch）
python3 -m train.selfplay --iters 30 --games-per-iter 8 --n-sims 60
```

## 1000 局基准

| Agent | 胜率 | 备注 |
|---|---|---|
| **Expectimax depth=3** | **40.4%** | 推荐部署 |
| EV (单步+启发式) | 35.0% | 快 10×，胜率仅低 5pp |
| Greedy fold@28 | 24.5% | 基线 |
| Random | 0.1% | 几乎从不赢 |

均匀基线 25%；Expectimax 相对提升 **+62%**。

## 进一步提升方向

1. **Expectimax 调优**：把 `INSURANCE_GAIN_VALUE` / `EXILE_DRAW_VALUE` 等常数用网格搜索校准
2. **加深递归**：depth=4 / 5 慢但更准（注意 Python 解释器开销，Cython/numba 化收益大）
3. **对手建模**：当前 expectimax 假设"我后续单人决策"，没有把对手放逐/三连的反制建进去；可以维护对手手牌唯一数估计 + 威胁评分
4. **NN 训练**：跑足量 self-play 后 NeuralAgent 应能超过 expectimax，原因是它能学到对手建模和长程价值
5. **CFR 风格**：对这种不完全信息博弈，Counterfactual Regret Minimization 也很合适，可作为阶段⑤
