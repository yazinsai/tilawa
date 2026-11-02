# Resolve a twin refrain only on the first lock

Status: replayed. No practice score from this worktree; the supervisor scores the branch.

Test set: lab-v1.5 practice. Listener evidence `spliced-clean-int8` (the folded context-free gap decode). Stacked on `tracker-issue-explains-drop`.

## Prediction, before any replay

Tracker −4, overall −4. The continuation resolver already flipped both p25 name warnings. The issue-explains-drop branch under it is the other −2, verified there at zero loss. Limiting the resolver to the first lock, before that lock has heard a location-unique word, should drop the three practice losses and the three studio regressions without giving the p25 wins back. 0 lost.

## Mechanism

`slot-spans.ts` is unchanged from the discarded continuation branch: runs of four or more words whose copies differ in at most one word per ayah. While a lock sits in such a run, accusations on the differing slots are held. The first heard word whose existing word cost fits only one member's in-order continuation, by the unchanged commit margin, emits `relocated` and the checker judges the held slots against that copy.

That resolver also fired on later refrains, after the lock had already heard text that occurs in only one place. Same-copy settlements re-judged the window and dropped planted mistakes; the one cross-copy settlement (p41 phone, 26:140 → 26:122) did the same. The guard is one flag, `arm`. It opens only when the first lock's word lies inside a slot-differing span, and it closes when that lock's position leaves every such span, or on any later `startLocal`. A word outside every span is unique to its location. The open span still gets to score its own continuation word, which is the first unique word, so p25 can still move. After the flag closes, position events carry no `spanHold` and the checker does not watch spans, which is the issue-explains-drop behaviour.

## What the replay showed

Step 0 replayed the discarded branch against this branch's parent (hold-tokens, no issue-explains) on every practice session, phone and studio.

Phone, 4 won and 3 lost:

- won `lab-v1.5-p46/phone/mistake:omitted_word/54:36:4/120.31`
- won `lab-v1.5-p66/phone/mistake:wrong_word/9:3:21/51.84`
- won `lab-v1.5-p25/phone/warning:wrong_word/26:105:3/8.40`
- won `lab-v1.5-p25/phone/warning:wrong_word/26:106:5/9.28`
- lost `lab-v1.5-p24/phone/mistake:omitted_word/26:128:1/234.62`
- lost `lab-v1.5-p41/phone/mistake:wrong_word/26:123:2/185.54`
- lost `lab-v1.5-p74/phone/mistake:wrong_word/83:8:4/31.25`

Studio, 3 lost and nothing won:

- `lab-v1.5-p24/studio/mistake:omitted_word/26:128:1/234.62`
- `lab-v1.5-p74/studio/mistake:wrong_word/83:8:4/31.25`
- `lab-v1.5-p16/studio/mistake:skipped_passage/77:39:1/144.97`

None of those six arise before a location-unique word. Each session's first lock is already outside every slot span (p24 at 26:91:3, 3.12 s; p41 at 26:85:3, 1.84 s; p74 at 83:3:2, 1.68 s; p16 at 77:1:2, 4.56 s). The losses sit on later refrains: p41 phone is the 26:140 → 26:122 move at 204.92 s, and the others are same-copy settlements (p24 at 26:122, p74 at 83:19, p16 at 77:41). p25 phone is the opposite shape: the first lock is 26:106:5, inside the span, and at 35.04 s the continuation resolves 26:104 → 26:159, landing on 26:165.

The same seven sessions under the guard: phone still wins those four and loses none; studio wins none and loses none. The only relocations left are p25's. Phone still ends on 26:165:1. Studio's first lock is already 26:161:5 and its same-copy settlement does not change the incident list.

Core and sdk suites passed on Modal (6 shards), including the new test that a lead-in outside every slot span does not relocate the Noah/Lut refrain, and the existing test that a lock which starts inside the span still moves to Lut.

## Limits

No full practice score from this worktree. The guard only turns the resolver off, so a session the discarded branch left unchanged stays unchanged; the seven sessions above are every phone or studio incident that branch moved. p25 studio still settles its own copy, because that first lock is inside the span. No exam replay.
