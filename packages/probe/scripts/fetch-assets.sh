#!/usr/bin/env bash
# Downloads the probe's models and clips into public/ (gitignored). Needs curl, tar, ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/../public"
mkdir -p models clips

for m in sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 sherpa-onnx-streaming-zipformer-en-2023-06-26; do
  [ -d "models/$m" ] && continue
  curl -fL "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$m.tar.bz2" | tar xj -C models
done

if [ ! -f clips/quran-60s.wav ]; then
  tmp=$(mktemp -d)
  for a in $(seq -f "%03g" 1 12); do curl -fsL "https://everyayah.com/data/Alafasy_128kbps/002$a.mp3" -o "$tmp/$a.mp3"; done
  ls "$tmp"/*.mp3 | sed "s/^/file '/; s/$/'/" > "$tmp/list.txt"
  ffmpeg -loglevel error -y -f concat -safe 0 -i "$tmp/list.txt" -t 60 -ac 1 -ar 16000 -c:a pcm_s16le clips/quran-60s.wav
  rm -rf "$tmp"
fi

cp -n models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/test_wavs/0.wav clips/english-check.wav || true

# Keep only what the page loads, so the deploy stays small.
rm -rf models/*/test_wavs models/*/*.sh models/*/README.md models/*/bpe.model
rm -f models/sherpa-onnx-streaming-zipformer-en-2023-06-26/*-128.onnx
du -sh models/* clips/*
