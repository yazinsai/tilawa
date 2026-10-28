# 11 Lab dashboard

Status: ready-for-agent
Type: task
Blocked by: 06, 10

Spec: `../spec.md` goal and decisions 1, 8, 9, 14. Modal web endpoint.

## Acceptance

- Header, always visible: the exam overall score on the main line, its missed and false-warning counts, the test set version, and the number of pending flags.
- Three component scores, plus attribution as one bar that adds up to the overall score. The teacher is shown as a ceiling.
- Experiments: component, expected, actual (component and overall, practice; exam after fold-in), verdict, status. Experiments waiting on your decision come first.
- Drill-down per component: failing practice cases, with likely bad data first (decision 14). Clicking a case opens the review item with buttons for bad data or a component fault.
- Release tab: the existing release gates.
- Plain language and counts, no jargon in labels.
