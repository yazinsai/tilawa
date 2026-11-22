# Wrong-word miss inspection v1

Inspection of saved teacher evidence and tracker/checker events; no new listening labels were assigned. Original evaluation remains unchanged.

## Findings

- 29 wrong-word positives: 1 confirmed, 28 missed under the default confirmation policy.
- Cause groups: {'confirmation_delay_short_clip': 2, 'caught': 1, 'alternate_passage_acquisition': 24, 'canonical_output_label_unverified': 2}.
- Every alternate-passage acquisition case is an unverified label. The decoded evidence is closer to the planted sequence than the canonical expected sequence. Several substitutions produce exactly another valid ayah; others differ only in context-sensitive phonemes.
- The evaluation supplies memorization mode but no intended passage prior. Thus the observed bookmark can follow the alternate ayah. This cannot establish failure to hear the substitution, nor can audio alone establish which of identical passages was intended.
- Two unverified misses decode closer to the expected canonical sequence. They need listening review to distinguish a mistake not performed from listener canonicalization.

## Two verified misses

- `munxrow9_d812be38`, 18:65:9, وعلمنه → لكنا: suspected at 13.00 s, uncertain at clip end 15.32 s. Only 2.32 s elapsed, short of the default 2.5 s confirmation delay.
- `munxt1gn_c30a2942`, 80:26:2, شققنا → يزكي: suspected at 6.04 s, uncertain at clip end 6.32 s. Only 0.28 s elapsed after acquisition.
- Replaying both with the existing next-word confirmation policy confirms the correct error in both cases. Confidence stays above 0.93; disabling the confidence threshold adds nothing. This is a diagnostic comparison, not justification for changing the default without checking false positives.

## Per clip

| Clip | Audit | Expected → planted | Canonical / planted edit distance | Locked ayahs | Diagnosis |
|---|---|---|---:|---|---|
| munxrow9_d812be38 | verified | وعلمنه → لكنا | 11 / 8 | 18:65, 18:64 | confirmation_delay_short_clip |
| munxsgj7_fe81f7d1 | verified | المزن → ممنوعه | 12 / 6 | 56:69 | caught |
| munxt1gn_c30a2942 | verified | شققنا → يزكي | 6 / 2 | 80:26 | confirmation_delay_short_clip |
| muo5l3xx_45159333 | provisional | وابعث → وارسل | 6 / 0 | 7:111 | alternate_passage_acquisition |
| muo6d9p9_ab2334ec | provisional | وارسل → وابعث | 8 / 2 | 26:36 | alternate_passage_acquisition |
| muo782ep_5ab47851 | provisional | نوح → هود | 4 / 2 | 26:124 | alternate_passage_acquisition |
| muo7ahel_4cc29975 | provisional | هود → صلح | 7 / 0 | 26:142 | alternate_passage_acquisition |
| muo7bjrc_eda68f67 | provisional | هود → صلح | 7 / 0 | 26:142 | alternate_passage_acquisition |
| muo7c2dl_91b5a145 | provisional | جعلنه → انزلنه | 5 / 0 | 12:2 | alternate_passage_acquisition |
| muo8bzgh_5e67f4d3 | provisional | هود → صلح | 9 / 2 | 26:142 | alternate_passage_acquisition |
| muod1z9i_36eeb67b | provisional | وارسل → وابعث | 9 / 5 | 26:36 | alternate_passage_acquisition |
| muod529w_2666cdfc | provisional | هود → صلح | 7 / 0 | 26:142 | alternate_passage_acquisition |
| muodfjiw_c16401bc | provisional | نوح → هود | 10 / 9 | 26:177 | alternate_passage_acquisition |
| muodhmzj_0639dae3 | provisional | الجحيم → الجنه | 6 / 0 | 79:41 | alternate_passage_acquisition |
| muodkqyr_711bd18d | provisional | وارسل → وابعث | 13 / 9 | 26:36 | alternate_passage_acquisition |
| muoeza4q_647d3a70 | provisional | انزلنه → جعلنه | 7 / 2 | 43:3 | alternate_passage_acquisition |
| muof4w2h_64d8e478 | provisional | نوح → هود | 3 / 1 | 26:124 | alternate_passage_acquisition |
| muof5q4h_a99d2c5d | provisional | الجحيم → الجنه | 6 / 0 | 79:41 | alternate_passage_acquisition |
| muof8ik2_5becb732 | provisional | وابعث → وارسل | 7 / 1 | 7:111 | alternate_passage_acquisition |
| muog21hl_5aeb0f2b | provisional | انزلنه → جعلنه | 8 / 3 | 43:3 | alternate_passage_acquisition |
| muoj4ci1_b3f95786 | provisional | جعلنه → انزلنه | 6 / 1 | 12:2 | alternate_passage_acquisition |
| muoj4vi0_e027d49b | provisional | جعلنه → انزلنه | 7 / 4 | 12:2 | alternate_passage_acquisition |
| muoj7duz_76afba75 | provisional | نوح → هود | 5 / 3 | 26:124 | alternate_passage_acquisition |
| muokludg_246b6bc8 | provisional | وارسل → وابعث | 10 / 4 | 26:36 | alternate_passage_acquisition |
| muokqj9e_95f78f3d | provisional | الجحيم → الجنه | 6 / 0 | 79:41 | alternate_passage_acquisition |
| muoorz0r_5d638d92 | provisional | نوح → هود | 2 / 0 | 26:124 | alternate_passage_acquisition |
| muooux8p_73ec57b9 | provisional | انزلنه → جعلنه | 7 / 2 | 43:3 | alternate_passage_acquisition |
| muouih3u_4d1737e2 | provisional | الجحيم → الجنه | 0 / 6 | 79:39 | canonical_output_label_unverified |
| muoujupy_536ca64c | provisional | وابعث → وارسل | 2 / 8 | 26:36 | canonical_output_label_unverified |

## Next diagnostic runs

Compare free acquisition with an explicitly supplied intended passage, preserving observed and expected bookmarks separately. A weak prior alone may not resolve clips that exactly recite another valid ayah. Report parallel-ayah substitutions separately from unique wrong-word examples.
Compare default-delay and next-word confirmation across all held-out clips, including clean clips, before selecting a policy. Complete the audit of the two canonical-output cases.

Distance comparison inserts the donor word’s reference phonemes into the expected ayah. It is approximate because transitions and stop forms may differ; closer planted text is evidence consistent with preservation, not independent audio ground truth.
