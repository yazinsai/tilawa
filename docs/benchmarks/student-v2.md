# Student distillation v2 (ticket 08 student)

Ticket 08. Same model and data as `student-v1.md`: streaming Zipformer2-M phoneme CTC (64.6M), 25 Hz, initialised from LibriSpeech streaming Zipformer-M at base LR 0.015, 12 epochs on the tlog-v2 `clean-match` train split, with noise, RIR, gain and SpecAugment. New in v2: the 150 gold PER tlog clips are excluded from train (`docs/benchmarks/gold-per-v1-exclude.txt`; 225,551 → 225,401 clips, the other 3 of the 153 clean-match/train gold clips are over the 30 s cap), and the `--target spliced` CTC target. Machine-readable: `student-v2.json`.

**Winner: `spliced-clean`**, the ticket 08 student.

| | teacher | combo-libri-soft5-full | combo-clean | **spliced-clean** |
|---|---|---|---|---|
| CTC target | — | teacher greedy | teacher greedy | spliced reference, runs collapsed |
| soft top-5 KD weight | — | 1 | 1 | 2 |
| gold clips in train | — | yes | no | no |
| Gold PER, canon (233 clips) | 3.39% | 4.82%* | 4.97% | **4.71%** |
| Gold PER, never in train (62 clips) | 7.27% | 9.19% | 9.45% | **8.80%** |
| Silver PER (211 help clips) | — | 3.22% | 3.37% | **3.09%** |
| Fair PER, clean-match test (canon vs spliced) | 2.44% | 4.05% | 4.14% | **3.87%** |
| Raw PER vs reference | 8.5% | 8.7% | 8.7% | 9.4% |
| Raw PER vs teacher | 0 | 4.3% | 4.4% | 5.6% |
| E1b canonicalised share | 34.5% | 34.2% | 34.0% | 41.7% |
| E1b spliced error in greedy | 54.5% | 52.1% | 54.2% | 43.4% |
| Help: free precision / recall | 76.2% / 50.0% | 83.8% / 48.4% | 75.7% / 43.8% | 81.8% / 42.2% |
| Help: declared precision / recall | 77.0% / 89.1% | 81.8% / 84.4% | 75.0% / 79.7% | **83.6% / 87.5%** |
| Help: false warnings per 5 min | 0.22 | 0.22 | 0.54 | 0.32 |
| GPU hours | — | 3.7 | 3.2 | 3.8 |

\* trained on 150 of the 191 gold tlog clips, so optimistic. Its fair gold number is the 62-clip slice.

Gold PER bar is 3.7% (`listener-target-v1.json`). No student passes it yet.

## Readout

- **Gold exclusion cost nothing measurable.** combo-clean is within noise of the old combo on fair and held-out PER (+0.1 and +0.3 pts). The old combo's help precision lead (83.8% vs 75.7%) comes from 3 clips on 37 flags, so it's noise, not leakage. Help clips were never in train.
- **The spliced target beats the teacher-greedy target on every PER that scores against human or reference labels.** Gold −0.26 pts, held-out −0.65, silver −0.28, fair −0.27. It also scores best on the declared arm: 83.6% / 87.5%, above the teacher's precision. Raw PER rises because the target collapses runs: the reference keeps doubled letters, and the soft KD only partly brings duration back. PER vs teacher rises because the student is no longer copying the teacher's interior mistakes.
- **The cost is canonicalisation.** Spliced splices land on the canonical letter 41.7% of the time, against 34.5% for the teacher, and the greedy decode keeps only 43.4% of them. That's the expected side effect of training the interior on the text. v1's checker has no pronunciation classes, and memorization errors are word-level, so this doesn't touch any v1 bar. The spliced target is the wrong base for post-v1 pronunciation work unless E3, which trains on spliced errors, corrects it.
- Free-arm recall: 42.2% vs 43.8%. That's one clip, so call it equal.

## Reproduce

    modal run --detach modal/student_train.py --name spliced-clean --size full --init librispeech-m --base-lr 0.015 --kd-weight 2 --target spliced --epochs 12
    modal run --detach modal/student_train.py --name combo-clean --size full --init librispeech-m --base-lr 0.015 --kd-weight 1 --epochs 12
    modal run modal/student_decode_cpu.py::main --run <run>          # then tools/per_breakdown.py <run>
    modal run modal/student_decode_cpu.py::gold --runs <run>         # then tools/gold_set.py score
    modal run modal/student_train.py::transcribe --run <run> --src eval/e1b-{spliced,clean}-in.parquet ...   # then tools/e1_report.py --e1b

The spliced run set KD weight 2 against combo's 1, following the spliced-target definition in `teacher-per-breakdown-v1.md`. So the comparison changes two things at once. The PER gains could come partly from the higher KD weight.
