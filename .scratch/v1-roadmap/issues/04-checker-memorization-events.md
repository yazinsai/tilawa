# 04: Checker, memorization events

**What to build:** The checker as a separate consumer of tracker output. It emits versioned checker events for wrong, omitted and added words and for skips, with the confirm and resolve policy.

**Status:** ready-for-agent

- [ ] Consumes the locked alignment and the evidence; emits `suspected | confirmed | resolved | uncertain` with error class and word location, using the versioned event schema.
- [ ] Self-correction: a wrong word followed by a restart on the right word goes `suspected`, then `resolved`. A policy setting decides whether resolved errors count as mistakes; by default they don't.
- [ ] Repeats produce no checker events.
- [ ] Skip: a suspected omission is `resolved` if the words are recited before the next ayah ends, and `confirmed` otherwise.
- [ ] Confirm delay is a policy setting: 2–3 s, or the next word onset.
- [ ] Unclear evidence (low confidence or tracker not locked) yields `uncertain`, never `confirmed`.
- [ ] Tests on corrupted teacher evidence (substituted, dropped and inserted words) plus clean fixtures. The tlog fixtures emit no confirmed errors.
- [ ] False warnings per 5 minutes reported on the e2e sessions (professional audio, so it should be zero).
