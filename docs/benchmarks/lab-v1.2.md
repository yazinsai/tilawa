# Lab test set v1.2

lab-v1.1 rebuilt with three splice rules from your listens to the 7 cases both triage models rejected (`lab-v1.1/review/triage-confirm.json`). Three of those were bad splices, and four were fine. Everything else is as in `lab-v1.md`: same catalog, seed 20261028, slices, targets, phone bank, checks and loop. Counts are in `lab-v1.2.json` beside this file. lab-v1.1 is untouched on the volume.

- Build: `modal run modal/lab_build_sessions.py --stage loop` (energy, aug, then plan, render, seam gate, teacher decode, automated checks and refusals until nothing fails; then the `spliced-clean-int8` decode and triage). `--stage audit` runs this version's rules over lab-v1.1's splices.
- Logic: `tools/lab_sessions.py`. Test: `uv run --python cpython-3.12-macos-aarch64-none --with numpy --with scipy --with pytest pytest tools/test_lab_sessions.py`.
- Data: `/lab/data/lab-v1.2/` (same layout as lab-v1), plus `energy/<read>/<surah>.npy`: the 10 ms dB frames of every catalog recording, which the planner reads.

## The three bad splices and the rules

| Case | What you heard | Cause | Rule |
| --- | --- | --- | --- |
| e01-x3 (تُرْجَعُونَ added before ٱلدَّارِ) | "…turjaʿūn \| ddāri": the ال missing | The return seam's cut, 0.16 s before the aligned start of ٱلدَّارِ, sat in a voiced gap (never below −15 dB of speech), so it clipped the hamzat al-wasl's weak ʾa | Weak onset after a pause |
| p03-x1 (فِيهَا for مَا) | a bad join | Out-seam linked cut at −3.8 dB after a vowel | Linked cut depth |
| p19-x11 (مَا added) | the word not heard | In-seam linked cut at −4.0 dB after a vowel, and a donor 0.24 s long | Linked cut depth, minimum donor length |

All three are enforced twice: the planner refuses the boundary or donor, and the seam gate re-measures the render's sources and fails the splice (its site and donor go to `refusals-v1.json`, so the next plan rebuilds it).

**Linked cut depth.** A linked cut must be at least 6 dB below the ayah level in its source (`LINK_DB`, was 3), or 8 dB when the word before it ends in a vowel or long vowel (`LINK_TAIL_DB`). The gate measures the source frame at each side's cut against the level of the word before it (`outCutDb`, `inCutDb`). The data, lab-v1.1's 615 linked seam sides: the median is −9.6 dB, the 75th percentile −6.6 dB and the 90th −4.3 dB. The two bad ones are at −3.8 and −4.0, both after vowels. Of the four you passed, the shallowest are e10-x6 at −6.3 (long vowel) and e13-x8 at −8.1 (vowel). At −6 dB for every tail, e10-x6 would pass but vowel cuts between −6 and −8 would too, and those sit close to the two failures. So vowel tails take −8, which costs e10-x6.

**Minimum donor length.** A wrong or added word lasts at least 0.26 s in its source (`MIN_DONOR_S`). The alignment runs on 40 ms frames, so donor lengths come in steps of 0.20, 0.24, 0.28 s. The bad donor is 0.24 s; p01-x4's, which you passed, is 0.28 s. 0.26 s means 7 frames or more. The gate fails a wrong or added token shorter than that in the output.

**Weak onset after a pause.** A cut before a word that starts weak (hamza, including hamzat al-wasl said with its hamza after a pause, ع, ح, ه, and خ, غ, the existing `WEAK_ONSET` set) must have at least 80 ms of measured silence ending 5 ms after the cut (half the crossfade). Silence is a 10 ms frame below the recording's room tone + 3 dB, and never above 30 dB under the ayah level. Room tone is the 2nd-percentile frame of the recording. These are reverberant mosque and studio recordings, so a fixed −30 dB floor would refuse nearly every such pause. When the aligned cut fails, the planner moves it earlier, in 10 ms steps, back to 20 ms before the previous word's end, to the latest point that passes; with none, the boundary is weak and can't start a piece. Alignment alone isn't trusted: the teacher's 40 ms frames can place a word start late. The gate measures `inSilenceS` the same way in the render's in-side source.

At e01-x3's return seam there was 0 ms of silence. Over the whole catalog, 968 weak-onset pauses had their cut moved into silence and 8758 had no silent stretch and were refused as in-sides.

## The rules on lab-v1.1

`modal run modal/lab_build_sessions.py --stage audit` → `lab-v1.2-rules-on-lab-v1.1.json`. Of lab-v1.1's 360 splices:

| Rule | Planner | Gate |
| --- | ---: | ---: |
| Linked cut depth | 149 | 148 |
| Weak onset after a pause | 28 | 30 |
| Donor under 0.26 s | 2 | 2 |
| Any | 172 | 172 |

One more splice would only have had a cut moved. The three planner and gate differences: e13-x3 sits at the −8.0 limit (rounding), and p15-x1 and p19-x3 had cuts the planner would move, which the gate judges where lab-v1.1 put them.

- e01-x3: refused (weak onset, 0 ms of silence).
- p03-x1: refused (out side −3.8 dB after a vowel).
- p19-x11: refused (in side −4.0 dB after a vowel; donor 0.24 s).
- Your four fine ones: e13-x8, e16-x4 and p01-x4 pass. e10-x6 is refused (long vowel at −6.3 dB against −8).

## Mutashabihat swaps with shared words

All 10 of lab-v1.1's mutashabihat swaps had a linked cut between −4.2 and −7.3 dB after a vowel or long vowel. The depth rule refuses every one, and a first plan under the new rules found 1. A twin ayah shares every word but one with its host. So a swap can now take shared words with it (`Donors.twin_spans`, up to `TWIN_SIDE` = 4 either side, ayah-bound). The reader's twin ayah runs from word a to b around the twin word, in place of the host's a..b, so the seams land at the nearest boundaries that join on both sides, often the ayah ends. The shared words must read exactly as the host's (same diacritised text), and they are labelled as recited at the host's positions, `from` the twin's audio. Only the twin word is the mistake. The splice's `donor` gains `span` and `shared`. The shortest span comes first; a=b is the plain one-word swap.

All 25 of lab-v1.2's swaps use spans, with 1 to 5 shared words.

## Counts

| | practice | exam | all |
| --- | --- | --- | --- |
| sessions | 20 | 20 | 40 |
| reciters | 15 | 14 | 29 |
| minutes | 87.5 | 93.4 | 180.9 |
| ayah-1 starts | 4 | 4 | 8 |
| strict starts | 3 | 3 | 6 |
| clean sessions | 5 | 5 | 10 |
| planted mistakes | 168 | 183 | 351 |
| wrong word (plain) | 22 | 23 | 45 |
| mutashabihat swap | 9 | 16 | 25 |
| omitted word | 10 | 20 | 30 |
| added word | 79 | 91 | 170 |
| skipped passage | 48 | 33 | 81 |
| repeats / restarts (not mistakes) | 13 / 7 | 10 / 10 | 23 / 17 |
| events with `forms_other_ayah` | 18 | 22 | 40 |

Sessions run 184–393 s, at 0.35 planted mistakes per ayah. lab-v1.1 had 320 (94 wrong, 10 mutashabihat, 42 omitted, 105 added, 69 skipped).

The mix moved toward added words and skips. An added word needs one host boundary and skips need pauses, while wrong and omitted words need two linked boundaries that are now rarer. The class weights are unchanged.

Seams: 631, of which 271 are pause to pause, 177 pause to link, 172 link to link and 11 link to pause.

## Loop

Converged in 5 iterations. Every iteration rendered only the sessions whose plan changed.

| Iteration | Rendered | Seam gate | Studio mistakes | Non-errors | Clean | Timing | New refusals |
| ---: | ---: | --- | --- | --- | --- | --- | ---: |
| 1 | 40 | 635/636 (e08-x1: linked cut at −7.9 dB after a vowel, the planner read −8.0) | | | | | 1 |
| 2 | 1 | 636/636 | 349/353 | 40/40 | 468/468 | 34/40 | 12 |
| 3 | 12 | 632/632 | 347/351 | 40/40 | 466/466 | 40/40 | 8 |
| 4 | 3 | 631/631 | 349/351 | 40/40 | 466/466 | 39/40 | 4 |
| 5 | 2 | 631/631 | **351/351** | **40/40** | **466/466** | **40/40** | 0 |

Refusal totals: 7 ayahs (timing), 11 sites and 7 donors. All edit failures were added or wrong words that the teacher preferred without, or a skip it heard blurred.

Final gate: 631/631 seams pass every check, including linked cut and onset. 9299/9299 words were found at their labels, 0 of 129 clipped words were over the limit, and 0 of 240 donors were short.

Final checks, studio (phone in brackets):

| | n | pass |
| --- | --- | --- |
| wrong word | 45 | 45 (43) |
| mutashabihat | 25 | 25 (25) |
| omitted word | 30 | 30 (26) |
| added word | 170 | 170 (162) |
| skipped passage | 81 | 81 (80) |
| repeat / restart | 23 / 17 | 23 / 17 (22 / 16) |
| clean stretches | 466 | 466 (431) |
| sessions with timings in 0.3 s | 40 | 40 (35) |
| reciters | 40 | 40 |

Studio clean PER is 0.95% over 61,508 phonemes (phone 2.54%).

## Decode

80 files per listener (40 sessions × studio, phone; 21,711 s of audio). Both listeners ran on both variants.

- `teacher` ran once per iteration, 4 times, on L4 at about 295 container-s each.
- `spliced-clean-int8` (`/lab/data/models/spliced-clean-int8`) ran once after convergence: 2913 container-s on CPU, 137 s wall.

## Known limits

- **Fewer wrong and omitted words.** The depth rule halves the usable linked boundaries. There are still 351 planted mistakes, but 48% of them are added words.
- **Weak-onset pauses are mostly refused.** Few pauses in these recordings get 80 ms near room tone. The rule is conservative by design: a refused site is rebuilt elsewhere, while a clipped onset ships a wrong label.
- **Span swaps reuse twin audio for shared words.** These are real recitation of the same text by the same reader, but from another ayah's occurrence. A listener can't tell, and neither can the scorer, which only reads labels.
