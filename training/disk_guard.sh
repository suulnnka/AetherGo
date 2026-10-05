#!/usr/bin/env bash
# 磁盘哨兵:数据目录超过软限就优雅停掉 katago selfplay(SIGINT 会把在途数据落盘后退出)。
# 自对弈进程不在了(到达 -max-games-total 或被手动停)哨兵也自行退出,不留常驻。
#
# 用法:nohup bash disk_guard.sh [监控目录] [软限GB] [检查间隔分] >/tmp/aether-disk-guard.log 2>&1 &
#   监控目录默认 /home/a/go/aether;软限默认 25(给 30G 硬限留缓冲)

DIR="${1:-/home/a/go/aether}"
LIMIT_GB="${2:-25}"
INTERVAL_MIN="${3:-10}"
LIMIT_BYTES=$((LIMIT_GB * 1000 * 1000 * 1000))

echo "[$(date '+%F %T')] 哨兵启动:监控 $DIR,软限 ${LIMIT_GB}G,每 ${INTERVAL_MIN} 分钟检查"
MISSING=0
while true; do
  sleep $((INTERVAL_MIN * 60))
  SIZE=$(du -sb "$DIR" 2>/dev/null | awk '{print $1}')
  [ -z "$SIZE" ] && continue
  SIZE_GB=$(awk -v s="$SIZE" 'BEGIN{printf "%.2f", s/1e9}')
  if [ "$SIZE" -ge "$LIMIT_BYTES" ]; then
    echo "[$(date '+%F %T')] $DIR = ${SIZE_GB}G ≥ ${LIMIT_GB}G —— 发 SIGINT 停自对弈(数据落盘)"
    pkill -INT -f "katago selfplay"
    sleep 120
    pkill -9 -f "katago selfplay" 2>/dev/null   # 2 分钟还没退就强杀,绝不让它继续写
    exit 0
  fi
  if ! pgrep -f "katago selfplay" > /dev/null; then
    MISSING=$((MISSING + 1))
    if [ "$MISSING" -ge 3 ]; then
      echo "[$(date '+%F %T')] 自对弈已停(当前 ${SIZE_GB}G),哨兵退出"
      exit 0
    fi
  else
    MISSING=0
    echo "[$(date '+%F %T')] ${SIZE_GB}G / ${LIMIT_GB}G — 正常"
  fi
done
