#!/usr/bin/env bash
set -euo pipefail

# Autoresearch loop for offline-tarteel
# Usage: ./run.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_CSV="$SCRIPT_DIR/results.csv"
TRAIN_PY="$SCRIPT_DIR/train.py"
RESULT_JSON="/tmp/autoresearch_result.json"
HYPOTHESIS_FILE="/tmp/autoresearch_hypothesis.txt"
BEST_ACCURACY_FILE="$SCRIPT_DIR/.best_accuracy"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
CHECKPOINT_DIR="/tmp/autoresearch_checkpoints"
PUBLIC_DIR="$REPO_DIR/web/frontend/public"
CLAUDE_TIMEOUT=120  # 2 min max for agent to propose a change

# Activate venv
source "$REPO_DIR/.venv/bin/activate"

# Claude Code auth (must be exported for subprocess)
: "${CLAUDE_CODE_OAUTH_TOKEN:?Set CLAUDE_CODE_OAUTH_TOKEN}"
export CLAUDE_CODE_OAUTH_TOKEN
# Allow --dangerously-skip-permissions as root (Vast.ai/RunPod run as root)
export IS_SANDBOX=1

# Restore best promoted ONNX to public/ after a rejected run.
# evaluate_accuracy() overwrites public/ with the candidate model,
# so we must restore the best known good model after non-promotion.
_restore_best_onnx() {
    # Find the latest promoted artifact dir
    local latest_artifact
    latest_artifact=$(ls -d "$ARTIFACTS_DIR"/run-* 2>/dev/null | sort | tail -1)
    if [[ -n "$latest_artifact" ]] && [[ -f "$latest_artifact/fastconformer_phoneme_q8.onnx" ]]; then
        cp "$latest_artifact/fastconformer_phoneme_q8.onnx" "$PUBLIC_DIR/" 2>/dev/null || true
        cp "$latest_artifact/phoneme_vocab.json" "$PUBLIC_DIR/" 2>/dev/null || true
        echo "Restored best ONNX from $latest_artifact"
    fi
}

# Initialize tracking files
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "run,val_loss,accuracy,best_accuracy,promoted,steps,elapsed,timestamp,description" > "$RESULTS_CSV"
fi
[[ -f "$BEST_ACCURACY_FILE" ]] || echo "0/0" > "$BEST_ACCURACY_FILE"
mkdir -p "$ARTIFACTS_DIR"

RUN_NUM=$(tail -n +2 "$RESULTS_CSV" | wc -l | tr -d ' ')
BEST_ACCURACY=$(cat "$BEST_ACCURACY_FILE")

echo "=== Autoresearch Loop ==="
echo "Starting from run #$RUN_NUM"
echo "Best accuracy: $BEST_ACCURACY"
echo ""

while true; do
    RUN_NUM=$((RUN_NUM + 1))
    echo "========================================"
    echo "RUN #$RUN_NUM — $(date)"
    echo "========================================"

    # 1. Agent proposes ONE change to train.py
    #    Writes hypothesis to file, edits train.py directly.
    #    No max-turns — agent runs until done (edits are fast).
    #    --dangerously-skip-permissions avoids interactive prompts.
    echo "[1/5] Agent proposing change..."
    cp "$TRAIN_PY" "$TRAIN_PY.bak"
    rm -f "$HYPOTHESIS_FILE"

    cd "$SCRIPT_DIR"
    timeout "$CLAUDE_TIMEOUT" claude --print --dangerously-skip-permissions -p "
You are an autonomous ML researcher optimizing a FastConformer phoneme CTC model.

Read program.md for full context and rules. Read results.csv to see prior experiments.
Current best accuracy: $BEST_ACCURACY (run #$RUN_NUM)

Your task:
1. Write your hypothesis (one sentence) to /tmp/autoresearch_hypothesis.txt
2. Make ONE targeted change to train.py

Do not change any other files. Do not run the training yourself.
" > /tmp/autoresearch_claude_output.txt 2>&1 || true

    # Read hypothesis from file (agent writes it), fallback to first line of output
    if [[ -f "$HYPOTHESIS_FILE" ]]; then
        DESCRIPTION=$(cat "$HYPOTHESIS_FILE")
    else
        DESCRIPTION=$(head -1 /tmp/autoresearch_claude_output.txt 2>/dev/null || echo "no-hypothesis")
    fi
    echo "Hypothesis: $DESCRIPTION"

    # Check if train.py was actually modified
    if diff -q "$TRAIN_PY" "$TRAIN_PY.bak" > /dev/null 2>&1; then
        echo "Agent made no changes to train.py, skipping run"
        TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        echo "$RUN_NUM,,,${BEST_ACCURACY},skipped,0,0,$TIMESTAMP,\"no-change: $DESCRIPTION\"" >> "$RESULTS_CSV"
        continue
    fi

    # 2. Run training (5-minute budget + 30s grace)
    echo "[2/5] Training (5 min budget)..."
    rm -f "$RESULT_JSON"
    rm -rf "$CHECKPOINT_DIR"/*

    TRAIN_OK="yes"
    if timeout 360 python "$TRAIN_PY" 2>&1 | tail -30; then  # uses venv python via activate
        true
    else
        TRAIN_OK="no"
    fi

    # 3. Read val_loss result
    VAL_LOSS="999.0"
    STEPS="0"
    ELAPSED="0"
    NEMO_PATH=""
    if [[ "$TRAIN_OK" == "yes" ]] && [[ -f "$RESULT_JSON" ]]; then
        VAL_LOSS=$(python3 -c "import json; print(json.load(open('$RESULT_JSON'))['val_loss'])")
        STEPS=$(python3 -c "import json; print(json.load(open('$RESULT_JSON'))['steps'])")
        ELAPSED=$(python3 -c "import json; print(json.load(open('$RESULT_JSON'))['elapsed_seconds'])")
        NEMO_PATH=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('nemo_path',''))")
    else
        echo "Training failed or timed out"
    fi

    echo "[3/5] val_loss=$VAL_LOSS"

    # 4. Promotion gate: ALWAYS run accuracy eval on successful training
    #    val_loss is logged for analysis but NOT used as a filter.
    #    Only accuracy determines promotion. This avoids suppressing
    #    models that have worse val_loss but better real-world accuracy.
    ACCURACY=""
    PROMOTED="no"

    if [[ "$TRAIN_OK" == "yes" ]] && [[ -n "$NEMO_PATH" ]] && [[ "$NEMO_PATH" != "" ]]; then
        echo "[4/5] Running ONNX export + accuracy eval..."

        # Run accuracy evaluation (export + validate-streaming)
        ACCURACY=$(python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from prepare import evaluate_accuracy
correct, total = evaluate_accuracy('$NEMO_PATH')
print(f'{correct}/{total}')
" 2>&1 | tail -1) || ACCURACY="0/0"

        echo "Accuracy: $ACCURACY (best: $BEST_ACCURACY)"

        # Parse accuracy numerators for comparison
        NEW_CORRECT=$(echo "$ACCURACY" | cut -d/ -f1)
        BEST_CORRECT=$(echo "$BEST_ACCURACY" | cut -d/ -f1)

        if [[ "$NEW_CORRECT" -gt "$BEST_CORRECT" ]] 2>/dev/null; then
            PROMOTED="yes"
            BEST_ACCURACY="$ACCURACY"
            echo "$BEST_ACCURACY" > "$BEST_ACCURACY_FILE"

            # Persist winning artifacts
            RUN_DIR="$ARTIFACTS_DIR/run-$(printf '%04d' $RUN_NUM)"
            mkdir -p "$RUN_DIR"
            cp "$NEMO_PATH" "$RUN_DIR/model.nemo" 2>/dev/null || true
            cp "$REPO_DIR/web/frontend/public/fastconformer_phoneme_q8.onnx" "$RUN_DIR/" 2>/dev/null || true
            cp "$REPO_DIR/web/frontend/public/phoneme_vocab.json" "$RUN_DIR/" 2>/dev/null || true
            cp "$RESULT_JSON" "$RUN_DIR/result.json" 2>/dev/null || true

            # Commit the train.py change
            cd "$REPO_DIR"
            git add autoresearch/train.py
            git commit -m "promoted: accuracy=$ACCURACY val_loss=$VAL_LOSS — $DESCRIPTION"

            echo "PROMOTED: accuracy $ACCURACY > $BEST_CORRECT — committed + artifacts saved"
        else
            echo "NOT PROMOTED: accuracy $ACCURACY <= best $BEST_ACCURACY"
            # Revert train.py and restore best promoted ONNX in public/
            cp "$TRAIN_PY.bak" "$TRAIN_PY"
            _restore_best_onnx
            cd "$REPO_DIR"
            git commit --allow-empty -m "rejected: accuracy=$ACCURACY val_loss=$VAL_LOSS — $DESCRIPTION"
        fi
    else
        echo "[4/5] SKIPPED accuracy eval (training failed)"
        # Revert train.py (public/ unchanged since no eval ran)
        cp "$TRAIN_PY.bak" "$TRAIN_PY"
        cd "$REPO_DIR"
        git commit --allow-empty -m "crashed: val_loss=$VAL_LOSS — $DESCRIPTION"
    fi

    # 5. Log to CSV
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "$RUN_NUM,$VAL_LOSS,$ACCURACY,$BEST_ACCURACY,$PROMOTED,$STEPS,$ELAPSED,$TIMESTAMP,\"$DESCRIPTION\"" >> "$RESULTS_CSV"

    # Clean up checkpoints for next run
    rm -rf "$CHECKPOINT_DIR"/*

    echo "[5/5] Logged. Cleaning up for next run."
    echo ""
done
