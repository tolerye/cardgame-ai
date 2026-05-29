#!/bin/bash
# 等到 18:30 再启动 trainer_weekend.sh

cd "$HOME/projects/cardgame-ai"

TARGET=$(date -j -f '%Y-%m-%d %H:%M:%S' '2026-05-29 18:30:00' +%s 2>/dev/null \
       || date -d '2026-05-29 18:30:00' +%s)

LOG="training_weekend.log"

while true; do
    NOW=$(date +%s)
    REMAIN=$(( TARGET - NOW ))
    if [ $REMAIN -le 0 ]; then
        break
    fi
    if [ $REMAIN -le 300 ]; then
        # 最后 5 分钟每 30s 检查
        sleep 30
    else
        # 早期每 5 分钟一次
        sleep 300
    fi
done

echo "[$(date)] ⏰ 18:30 到，启动训练" >> "$LOG"
exec bash "$HOME/projects/cardgame-ai/trainer_weekend.sh"
