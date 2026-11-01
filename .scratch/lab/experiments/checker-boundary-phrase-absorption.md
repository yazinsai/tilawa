# Boundary phrase absorption

Status: killed. One practice score used. The final branch has no engine change.

## Prediction

Against main `abb918b` on lab-v1.5 practice, the card predicted checker −5 and overall −4. The kill criterion was any new boundary added-word warning on a fully clean session or a regression of the folded ayah-start omissions.

## Mechanism check

The hypothesis was false for this cluster. Oracle traces of all eleven target incidents showed that none of the extra spans was marked `junk`, the checker's representation for a recognized non-Quran phrase. The existing phrase path already requires a fuzzy match against the supplied phrase asset before it marks evidence as `junk`; arbitrary residue cannot enter that path.

The misses instead came from ordinary Quran alignment: the extra speech was represented as a jump to another word or as an initial visit to the boundary word followed by another visit. Changing phrase insertion cost therefore cannot expose these additions.

## Rejected prototype

A prototype inspected all speech between the previous ayah's final intact visit and the next ayah's first intact visit. It recovered additions, but it also classified legitimate repeats and listener residue as additions. A backward-repeat exemption was then added after inspecting a practice incident. Review correctly rejected that exemption as benchmark-derived post-hoc weakening, so both the exemption and the entire boundary-residue heuristic were removed.

The phrase asset additions made only for that prototype were removed as well. They do not address the observed failure mode.

## Measured limit

The rejected prototype at `d357270` used one practice score:

| Phone practice | Baseline | Prototype | Change |
| --- | ---: | ---: | ---: |
| Wrong calls | 89 | 90 | +1 |
| Missed mistakes | 64 | 58 | −6 |
| False warnings | 25 | 32 | +7 |
| Checker | 37 | 27 | −10 |

It introduced seven phone false warnings, including one on a fully clean session, while recovering six target incidents. This directly hit the kill criterion and failed the fold gate. No constants were tuned and no second score was run.

## Validation

The final code matches the experiment's parent for checker logic, checker tests, and the phrase asset. Existing unit coverage still protects the ayah-start omission behavior, real آمين, ordinary additions, self-corrections, and repeats. Modal run `checker-boundary-final-903b602-dirty-20261101T060059Z` passed core shards 0–2 and SDK; shard 3 passed all 149 tests before Vitest's 60-second worker RPC timeout. Its threads rerun, `checker-boundary-final-shard4-903b602-dirty-20261101T060246Z`, exited 0.

The experiment is closed without a checker change because the proposed phrase-path mechanism does not occur in the target traces.
