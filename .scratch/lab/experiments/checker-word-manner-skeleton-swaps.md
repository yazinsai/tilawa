# Whole-word manner skeleton, plus the connected madd

Status: not practice-scored. Left for the supervisor.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. Checker −2, overall −2. The two checker wrong-word misses are 9:47:17 (بالمتقين for بالظالمين) and 20:43:1 (اذهب for اذهبا). Upside −3 if 2:284:28 or 12:78:8 also flip. Honest if they do not: the phone trace at 2:284:28 is قَثِۦۦَه against قَدِيرٌ, and those consonants stay inside one manner class.

## Mechanism

Two separate checks, no new threshold.

Manner. The string path prices 9:47:17 as soft substitutions: waqf and the shared -īn take the tail, and the stem consonants land on vowels. The unit alignment the checker already uses for changed phones (`align`) still pairs ظ/ل, ل/ت and م/ق. Manner is the table from the pausal-final near-miss: nasal, liquid (lateral or trill), obstruent (stop or fricative), glide. ن/م are one nasal and ف/ك are one obstruent, so those pairs are not a change. A repeated copy of the same written consonant against the same heard manner is one position. Two or more such positions, and the heard manner sequence equal to a whole attested mushaf pronunciation that is not this word and not a word within ±3, is a wrong word. That match stands in for the lexical doubling and the 0.6 confidence cut. `SUPPORT`, the weak-evidence doubling, and the parallel-phrase path are unchanged.

Connected madd. Landed as measured: a written word that ends in a long vowel, followed with no labelled pause by a hamza word, and whose 4-word phrase with that vowel dropped occurs at another ayah. The frames from the stem consonant to the next hamza have to be shorter than every earlier matched madd munfasil in the session. A pause stays on today's abstention. One vowel does not have to move the forward sum.

## Validation

- Phone phones at `lab-v1.5-p76` 9:47:17, aligned to بِظظَاالِمِۦۦن, change manner at 3 consonant positions (ظ/ل, ل/ت, م/ق). The manner sequence is بالمتقين's skeleton, not the written word or its ±3 neighbours. The kill criterion does not fire.
- Oracle-position trace, shipped policy (`modal-out-7`). Both targets confirm: 9:47:17 observed `بِلمُتتَقِۦۦن`, 20:43:1 observed `ءِذهَبڇ`. With the posterior window off, 9:47:17 stays quiet, so the manner candidate still has to clear `SUPPORT`. `lab-v1.5-p79` has no consonant manner change; the madd rule is what fires. The phone at 2:284:28 (`قَثِۦۦَه` against `قَدِيرٌ`) does not.
- Unit tests: a low-confidence بالمتقين confirms; ظ→ط, a single ن→ل, neutral posteriors, and the written word stay quiet; a neighbour's phones are not called a wrong word. The madd tests from the measured rule pass, including the shorter-than-own-madd flag, the equal madd, the pause, and the non-phrase. The old low-confidence fixture recited الكوثر, which is another word's skeleton, so it now confirms; the 0.6 cut is tested on ش→ص and ف→ب, which stay obstruents.
- Core and sdk suites: `bash tools/autolab_tests.sh`, six shards. Five shards passed. The sixth failed one assertion, the fixture above, and that test passes after the change. Vitest's worker RPC deadline on `checker.test.ts` is the known one.

## Limits

The manner gate needs the checker's unit alignment to substitute two consonants, and the heard sequence to equal some attested skeleton exactly. A blur that drops or inserts a consonant changes the skeleton and abstains. 2:284:28's heard consonants are all obstruents, so this rule does not reach it. The madd threshold is the reciter's own earlier madd; a session with no earlier madd munfasil still abstains. Practice score is the supervisor's.
