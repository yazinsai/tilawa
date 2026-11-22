---
version: 1
slug: "modal-lab-web-dashboard-html"
primary_target: "modal/lab_web/dashboard.html"
related_targets: ["modal/lab_web/review.html","modal/lab_web/item.js"]
---

Scope: the Tilawa lab web app (modal/lab_web): score overview, component drill-down, case view, listening review, release tab. Mode: Operate.

Audience and job: the founder checks whether the engine got better, which component is the biggest lever, which experiments need his fold/discard call, and listens to failing cases to mark bad data or a component fault. Laptop, short sessions, daylight office.

Constraints: vanilla HTML/CSS/JS; all JSON endpoints, verdict store format, keyboard shortcuts, ?run= / ?ledger= / ?reviewer= params unchanged. Counts not percentages; a change of 2 or fewer is noise.

Assumptions (no interview ran; subagent brief): direction taken unattended from the roll; brief pins "airy, modern, clear hierarchy".

## Direction contract

THESIS: the lab reads like one well-set mushaf page: one number set large with wide margins, everything else quiet beneath it. Refuses the metrics-dashboard default of a header strip, same-size cards and chips everywhere.

OWN-WORLD: near-white ground with a faint green tint, mushaf green (hsl 162 55% 24%) only on things you can press, illumination pigments for the three components (lapis listener, ochre tracker, madder checker), gold tint only for "needs your decision". Onest for Latin UI and figures (Geist was dropped: overused), Amiri Quran for Arabic at 36px / 2.1. Hairline rules, one 8px radius, almost no shadow. A double-hairline frame around recited Arabic, as a mushaf page frame.

STORY: see the overall score and whether it moved; see which component owns most of it; see what needs a decision; open a component, listen case by case, mark data or fault.

FIRST VIEWPORT: thin top bar (wordmark, Score / Release / Listening check, compact score once the hero scrolls away). Left-aligned hero: 96px score, "wrong calls every 5 minutes, lower is better", the missed + false-warnings sentence, change since the last main-line score, quiet test-set and listening-check line, "How this is measured" disclosure. Below: one stacked bar of the split with "= total", then three ruled component rows (count at 36px, one plain health line, biggest lever marked), each opening its drill-down.

FORM: candidate 6 of 7 (the Madina mushaf page), translated to an airy Operate screen. Signature move: the ayah-marker rail, numbered round markers for review items, filled by verdict, the current one ringed in gold; it is the position control on the review and case views. Raises: from the warm consumer surface, green strictly on actionable elements; from the doujin catalog, persistent marked state per cell; from the variable-font specimen, hierarchy by scale contrast with no ornament; from the acetate manual, one body size does the work. seed: ee2dcd0c47641ec6

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
