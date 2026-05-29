#!/bin/bash
# 周末训练循环：大网络 + 对抗微调
# Deadline: 2026-06-01 10:00（周一早 10 点）
# 启动: caffeinate -i bash trainer_weekend.sh & disown

set -u
cd "$HOME/projects/cardgame-ai"

DEADLINE=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-06-01 10:00:00' +%s 2>/dev/null \
        || date -d '2026-06-01 10:00:00' +%s)

LOG="training_weekend.log"
ROUND=1

echo "" >> "$LOG"
echo "[$(date)] ╔════════════════════════════════════════════════════════════╗" >> "$LOG"
echo "[$(date)] ║ 周末训练启动                                                 ║" >> "$LOG"
echo "[$(date)] ║ deadline = $(date -r $DEADLINE)                  ║" >> "$LOG"
echo "[$(date)] ║ 配置: 256h/5L 大网 + adversarial + asym + 400 games × 300 sims  ║" >> "$LOG"
echo "[$(date)] ╚════════════════════════════════════════════════════════════╝" >> "$LOG"

while true; do
    NOW=$(date +%s)
    REMAIN=$(( DEADLINE - NOW ))
    if [ $REMAIN -le 1800 ]; then
        echo "[$(date)] 剩 $((REMAIN / 60)) 分钟，不再启动新轮次" >> "$LOG"
        break
    fi

    REMAIN_H=$(awk "BEGIN { printf \"%.1f\", $REMAIN / 3600 }")
    echo "" >> "$LOG"
    echo "[$(date)] ━━━━━━━━ Round $ROUND  (剩 ${REMAIN_H} hr) ━━━━━━━━" >> "$LOG"

    # 每轮跑 90 iter（每 30 iter 一次 eval = 3 次评测）
    # 90 iter × ~75s = ~2 hr 一轮
    # 剩余时间不到 3 hr 就跑短的
    if [ $REMAIN -lt 10800 ]; then
        ITERS=30
    else
        ITERS=90
    fi

    python3 train_overnight.py \
        --hidden 256 --n-layers 5 \
        --adversarial \
        --reward-shape asym \
        --iters $ITERS --eval-every 30 --games 400 --n-sims 300 \
        --workers 6 \
        --baseline-raw 21.5 --early-stop-drop 8 \
        --ckpt-prefix model_weekend_iter \
        >> "$LOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(date)] Round $ROUND 异常退出 rc=$RC，60s 后重试" >> "$LOG"
        sleep 60
    fi

    ROUND=$((ROUND + 1))
done

echo "[$(date)] ╚══ trainer_weekend.sh 完成 ══╝" >> "$LOG"
