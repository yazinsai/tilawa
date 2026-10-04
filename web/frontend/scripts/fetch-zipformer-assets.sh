#!/usr/bin/env bash
# Copy the default a0w-ep1-a0.5 int8 ONNX + I/O manifest + zipformer phoneme
# corpus into the Vite public/ tree, plus the previous default
# interp-gentle-a0.5 (kept for comparison; skip it with ZIPFORMER_SKIP_LEGACY=1).
# The ONNX and lexicon are gitignored (NPL-1.2 derived); the interp io.json is
# committed in-repo.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FRONTEND="$(cd "$HERE/.." && pwd)"
DEST_MODELS="$FRONTEND/public/models"
A0W_ONNX="zipformer_a0w_ep1_a05.int8.onnx"
A0W_IO="zipformer_a0w_ep1_a05.io.json"
A0W_BASE="https://github.com/yazinsai/tilawa/releases/download/zipformer-a0w-ep1-a0.5"
ONNX_NAME="zipformer_interp_gentle_a05.int8.onnx"
IO_NAME="zipformer_interp_gentle_a05.io.json"
CORPUS_NAME="zipformer_quran.json"
RELEASE_BASE="https://github.com/yazinsai/tilawa/releases/download/v0.3.0"

SRC_DIR="${ZIPFORMER_EXPORT:-/tmp/zipformer-interp-gentle-a0.5/interp-gentle-a0.5}"
MAIN_PUBLIC="/Users/rock/ai/projects/offline-tarteel/web/frontend/public"
MAIN_CORPUS="/Users/rock/ai/projects/offline-tarteel/data/zipformer/quran.json"
MAIN_LAB_CORPUS="/Users/rock/ai/projects/offline-tarteel/lab/data/zipformer/quran.json"
REPO_ROOT="$(git -C "$FRONTEND" rev-parse --show-toplevel 2>/dev/null || echo "")"
WORKTREE_CORPUS="${REPO_ROOT:+$REPO_ROOT/../../data/zipformer/quran.json}"

mkdir -p "$DEST_MODELS"

download() {
  local url="$1" dest="$2"
  echo "Downloading $url"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

resolve_onnx() {
  if [[ -f "$SRC_DIR/model.int8.onnx" ]]; then echo "$SRC_DIR/model.int8.onnx"; return; fi
  if [[ -f "$MAIN_PUBLIC/models/$ONNX_NAME" ]]; then echo "$MAIN_PUBLIC/models/$ONNX_NAME"; return; fi
  echo ""
}

resolve_io() {
  if [[ -f "$DEST_MODELS/$IO_NAME" ]]; then echo "$DEST_MODELS/$IO_NAME"; return; fi
  if [[ -f "$SRC_DIR/model.io.json" ]]; then echo "$SRC_DIR/model.io.json"; return; fi
  if [[ -f "$MAIN_PUBLIC/models/$IO_NAME" ]]; then echo "$MAIN_PUBLIC/models/$IO_NAME"; return; fi
  echo ""
}

resolve_corpus() {
  for candidate in "$MAIN_CORPUS" "$MAIN_LAB_CORPUS" ${WORKTREE_CORPUS:+"$WORKTREE_CORPUS"} "${ZIPFORMER_CORPUS:-}" "$MAIN_PUBLIC/$CORPUS_NAME"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}

# Default model: a0w-ep1-a0.5 int8 (with the encoder-frame output), from its own release.
for f in "$A0W_ONNX" "$A0W_IO"; do
  if [[ -n "${ZIPFORMER_A0W_DIR:-}" && -f "$ZIPFORMER_A0W_DIR/$f" ]]; then
    cp "$ZIPFORMER_A0W_DIR/$f" "$DEST_MODELS/$f"
  elif [[ -f "$MAIN_PUBLIC/models/$f" ]]; then
    cp "$MAIN_PUBLIC/models/$f" "$DEST_MODELS/$f"
  elif [[ ! -f "$DEST_MODELS/$f" ]]; then
    download "$A0W_BASE/$f" "$DEST_MODELS/$f"
  fi
done

ONNX_SRC="$(resolve_onnx)"
IO_SRC="$(resolve_io)"
CORPUS_SRC="$(resolve_corpus)"

if [[ "${ZIPFORMER_SKIP_LEGACY:-0}" == "1" ]]; then
  :
elif [[ -n "$ONNX_SRC" ]]; then
  cp "$ONNX_SRC" "$DEST_MODELS/$ONNX_NAME"
else
  download "$RELEASE_BASE/$ONNX_NAME" "$DEST_MODELS/$ONNX_NAME"
fi

if [[ -n "$IO_SRC" && "$IO_SRC" != "$DEST_MODELS/$IO_NAME" ]]; then
  cp "$IO_SRC" "$DEST_MODELS/$IO_NAME"
fi

if [[ -n "$CORPUS_SRC" ]]; then
  cp "$CORPUS_SRC" "$FRONTEND/public/$CORPUS_NAME"
else
  download "$RELEASE_BASE/$CORPUS_NAME" "$FRONTEND/public/$CORPUS_NAME"
fi

if [[ ! -f "$DEST_MODELS/$A0W_ONNX" || ! -f "$DEST_MODELS/$A0W_IO" || ! -f "$FRONTEND/public/$CORPUS_NAME" ]]; then
  echo "Failed to materialize Zipformer browser assets."
  echo "Tried local export ($SRC_DIR), main checkout public/, and $RELEASE_BASE"
  echo "Manual fallback:"
  echo "  modal volume get zipformer-ctc-training /exports/interp-gentle-a0.5 /tmp/zipformer-interp-gentle-a0.5"
  exit 1
fi

echo "Ready:"
echo "  $DEST_MODELS/$A0W_ONNX (default)"
echo "  $DEST_MODELS/$A0W_IO"
echo "  $DEST_MODELS/$ONNX_NAME (previous default)"
echo "  $DEST_MODELS/$IO_NAME"
echo "  $FRONTEND/public/$CORPUS_NAME"
echo "sha256 a0w int8: $(shasum -a 256 "$DEST_MODELS/$A0W_ONNX" | awk '{print $1}')"
