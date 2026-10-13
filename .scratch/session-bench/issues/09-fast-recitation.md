# 09 Fast (hadr) recitation support

Status: ready-for-human
Type: task
Blocked by: none

Hadr QUD recordings (afasy-baqarah, badri-juz-tabarak, diban-sad, jaber-nisa) are excluded from session scoring as unsupported (`sessions-cli --unsupported-styles`, default `hadr`). Evidence: `docs/benchmarks/session-bench-v1.md`, "Unsupported: fast (hadr) recitation".

Both listeners delete about 36–39% of reference phonemes on fast speech, mostly held-sound repeats and short sounds. In 24 of the student's 28 false warnings, the teacher misreads the word just as badly.

## To unblock

- Collect fast-recitation samples (several speakers, word-timed) for training and evaluation, kept separate.
- Fine-tune the listener on them.
- Before training, check whether `ref-v2` writes word-joining sounds the way fast connected recitation actually says them (initial `ء`, `ل`). If the reference doesn't match connected speech, fix the reference rather than the model.
- Possibly make the checker tolerant of shortened held sounds at fast tempo.
- Re-enable by running with `--unsupported-styles none` and compare.
