#!/bin/bash
# 持续训练直到 deadline，每轮跑 train_overnight.py，下一轮基于 model.pt 接着训
set -u
cd "$HOME/projects/cardgame-ai"

# 当前正在跑的训练 PID（21:26:54 启动）
CURRENT_PID=${CURRENT_PID:-22545}
# Deadline: 2026-05-29 09:00 (~11.5h from now)
DEADLINE=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-05-29 09:00:00' +%s 2>/dev/null \
        || date -d '2026-05-29 09:00:00' +%s)

LOG="training_overnight.log"

echo "" >> "$LOG"
echo "[$(date)] ╔══ trainer_loop.sh started, deadline = $(date -r $DEADLINE) ══╗" >> "$LOG"

# 阶段 1：等当前训练结束（每 2 分钟检查一次）
while kill -0 $CURRENT_PID 2>/dev/null; do
    NOW=$(date +%s)
    if [ $NOW -ge $DEADLINE ]; then
        echo "[$(date)] deadline reached while waiting; not starting new rounds" >> "$LOG"
        exit 0
    fi
    sleep 120
done

echo "[$(date)] PID $CURRENT_PID finished, entering loop" >> "$LOG"

# 阶段 2：循环训练直到 deadline
ROUND=2
while true; do
    NOW=$(date +%s)
    REMAIN=$(( DEADLINE - NOW ))
    if [ $REMAIN -le 1800 ]; then
        echo "[$(date)] only $((REMAIN / 60)) min left, not starting new round" >> "$LOG"
        break
    fi

    REMAIN_H=$(( REMAIN / 3600 ))
    echo "" >> "$LOG"
    echo "[$(date)] ━━━━━━━━ Round $ROUND start  (${REMAIN_H}h until deadline) ━━━━━━━━" >> "$LOG"

    # 每轮跑 300 iter，如果剩余时间不够 6h 则减半
    if [ $REMAIN -lt 21600 ]; then
        ITERS=150
    else
        ITERS=300
    fi
    GAMES=120

    python3 train_overnight.py --iters $ITERS --games $GAMES --workers 6 >> "$LOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(date)] round $ROUND exited with code $RC" >> "$LOG"
        sleep 60
    fi

    ROUND=$((ROUND + 1))
done

echo "[$(date)] ╚══ trainer_loop.sh finished ($(($(date +%s) >= $DEADLINE && echo deadline || echo done))) ══╝" >> "$LOG"
