#!/usr/bin/env bash
# Monitor autoresearch progress from local machine
# Usage: ./monitor.sh <pod-id>

POD_ID="${1:?Usage: ./monitor.sh <pod-id>}"
CMD="runpodctl ssh cmd $POD_ID"

echo "=== Autoresearch Monitor ==="
echo ""

echo "--- Results (last 15) ---"
$CMD "tail -15 /workspace/offline-tarteel/autoresearch/results.csv" 2>/dev/null
echo ""

echo "--- Best accuracy ---"
$CMD "cat /workspace/offline-tarteel/autoresearch/.best_accuracy" 2>/dev/null
echo ""

echo "--- Git log (last 10) ---"
$CMD "cd /workspace/offline-tarteel && git log --oneline -10" 2>/dev/null
echo ""

echo "--- Artifacts ---"
$CMD "ls -la /workspace/offline-tarteel/autoresearch/artifacts/" 2>/dev/null
echo ""

echo "--- tmux status ---"
$CMD "tmux has-session -t autoresearch 2>/dev/null && echo 'RUNNING' || echo 'NOT RUNNING'" 2>/dev/null
