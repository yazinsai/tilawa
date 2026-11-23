# Validation scope

The initial iOS simulator matrix passed 16/17 checks. A deliberately substituted word was missed even though the original TypeScript checker flagged it. Investigation found two native bugs:

1. Counting all phoneme matches let shared vowels hide a different lexical word. The native checker now also requires evidence from consonants and at least two full-cost consonant substitutions; confusable pronunciation pairs alone are excluded from this rule.
2. An earlier confirmed prompt was cleared when the rolling evidence window cut through its first word. Only fully assessable words can now resolve an earlier prompt.

After those fixes, the same matrix passed **17/17** checks in the iOS 27.2 simulator, using native ONNX Runtime 1.24.3 and `spliced-clean-lite@1-int8`:

| Recorded-audio case | Native result |
|---|---|
| Clean Ya-Sin, 44.1 kHz | Ends at 36:5, word 3; no prompts |
| Clean Ya-Sin, 16 kHz | Same final position; no prompts |
| 16 kHz, 100 ms and 600 ms chunks | Same native token sequence as 300 ms chunks |
| Clean declared passage | Same final position; no prompts |
| Silence and quiet synthetic noise | No locked position or prompts |
| Deliberately omitted word | Omission prompt |
| Deliberately substituted word | Wrong-word prompt |
| Deliberately inserted word | Addition prompt |
| Deliberately skipped passage | Skipped-passage prompt |
| Audio outside the declared passage | Native tracker stays inside the declared passage |
| Session reset | Same native token sequence as a fresh take |
| Missing fixture file | Harness reports a recoverable error |

The input was the repository's public `web/frontend/e2e/fixtures/multi_036_001_005.wav`. Error clips were made by deterministic audio splices, not by asking anyone to recite incorrectly. The original SDK detected all four inserted edit classes. No recordings or edited audio are included in this contribution.

Native and Node CTC runs can differ in held vowel length. The original 44.1 kHz clip produced 115 native phoneme runs versus 113 in Node; held-sound-normalized sequences and the final word agreed. This is disclosed rather than called exact inference parity.

## Limits

This is one recitation plus synthetic signals and controlled edits. It is not an independent listening review, a speaker-held-out benchmark, a precision/recall estimate, or a real-device microphone/latency test. Original SDK agreement is correlated evidence, not ground truth for arbitrary recitation. The full TypeScript checker's policies and edge cases have not been ported. No model, reference, label, or published scoreboard value is changed by this contribution.

The checked-in tests cover the reproducible algorithm/lifecycle parts without external runtimes or audio. Broader natural-mistake evaluation, real devices, self-corrections, repeated/parallel passages, and native runtime CI remain follow-up work before claiming production support.
