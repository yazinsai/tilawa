# 04 Automated test set checks

Status: ready-for-agent
Type: task
Blocked by: 03, 05

Spec: `../spec.md` decision 16.

## Acceptance

- Each planted mistake: the teacher's alignment of the built audio hears the planted word and not the original at that spot.
- Each clean stretch: the teacher matches the text (PER under the gold-set teacher level, with a margin).
- Reciter not in any training source (from 01).
- Session ayah timings consistent with the alignment.
- Failures make the builder rebuild or drop the case and are logged with the reason. No case is hand-patched.
- The same checks run on any test set version, including the old help clips (ticket 13).
