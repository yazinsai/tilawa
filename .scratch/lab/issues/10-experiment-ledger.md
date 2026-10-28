# 10 Experiment cards, ledger and fold-in

Status: ready-for-agent
Type: task
Blocked by: 08, 09

Spec: `../spec.md` decisions 10–13.

## Acceptance

- Card (in the worktree): component, hypothesis, expected change in counts for the component score and the overall score, listener (default main line).
- `lab new`, `lab run`, `lab status`: each experiment's practice result against the main line, with a verdict of better, worse or noise (a change of 2 or fewer is noise).
- `lab fold <exp>`: rebase onto the main line, re-score practice and exam, and record expected, measured on its branch, and measured after rebase. Merge only on your confirmation. If the gain disappears after rebase, mark it superseded.
- `lab discard <exp>`: archive the card and its results.
- `lab apply-flags`: create the next test set version, then re-score the main line and every open experiment.
- The ledger lives on the lab volume; a summary is committed to `docs/benchmarks/lab-ledger.json` on fold-in.
