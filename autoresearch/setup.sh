#!/usr/bin/env bash
set -euo pipefail

# Provision a RunPod A100-80GB for autoresearch
# Prerequisites: runpodctl authenticated, HF_TOKEN set
#
# Claude Code uses subscription auth (not API key).
# After SSH'ing into the pod, run: claude setup-token
#
# Required env vars (will be passed to pod):
#   HF_TOKEN — for HuggingFace dataset access

: "${HF_TOKEN:?Set HF_TOKEN}"

REPO_URL="${1:-}"

echo "=== Autoresearch RunPod Setup ==="

# 1. Create pod
echo "[1/2] Creating RunPod A100-80GB pod..."
POD_JSON=$(runpodctl pod create \
    --name "autoresearch-tarteel" \
    --gpu-id "NVIDIA A100 80GB PCIe" \
    --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
    --container-disk-in-gb 100 \
    --volume-in-gb 200 \
    --ports "22/tcp" \
    --env "{\"HF_TOKEN\":\"${HF_TOKEN}\"}")

POD_ID=$(echo "$POD_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Pod created: $POD_ID"

echo ""
echo "[2/2] Connect and run setup:"
echo "  runpodctl ssh connect $POD_ID"
echo ""
echo "Then run the following on the pod:"
echo "================================================"
cat << REMOTE_SETUP

# === Run on the RunPod pod ===

# System deps
apt-get update && apt-get install -y git ffmpeg tmux

# Node.js 20 (for accuracy eval via validate-streaming.ts)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g tsx

# Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Claude Code auth (subscription-based, not API key)
claude setup-token
# Follow the prompts to authenticate with your Anthropic subscription

# Clone repo
cd /workspace
git clone ${REPO_URL:-<YOUR_REPO_URL>} offline-tarteel
cd offline-tarteel
git checkout -b autoresearch/run-001

# Create Python venv and install ML deps
python3 -m venv .venv
source .venv/bin/activate
pip install 'nemo_toolkit[asr]>=2.7.0' 'datasets>=3.0,<4.0' 'lightning>=2.4,<3.0' \
    'omegaconf>=2.3,<3.0' soundfile librosa onnxruntime

# Install frontend deps (needed for accuracy eval)
cd web/frontend && npm install && cd ../..

# Prepare data (one-time, ~15 min)
cd autoresearch
python prepare.py

# Verify
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
npx tsx ../web/frontend/test/validate-streaming.ts --no-streaming 2>&1 | tail -5

# Start the loop in tmux (venv is activated inside run.sh)
tmux new -d -s autoresearch './run.sh 2>&1 | tee run.log'
echo "Loop started in tmux session 'autoresearch'"
echo "Attach: tmux attach -t autoresearch"

REMOTE_SETUP
echo "================================================"
echo ""
echo "Pod ID: $POD_ID"
