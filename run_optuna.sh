#!/bin/bash
WORKERS=${1:-2}

mkdir -p logs

echo "Starting $WORKERS Optuna workers..."

for i in $(seq 0 $((WORKERS - 1))); do
    python optuna_search.py > logs/worker_$i.log 2>&1 &
    echo "  worker_$i started (pid $!)"
done

echo "Tailing logs (Ctrl+C to stop tailing, workers keep running)..."
tail -f logs/worker_*.log
