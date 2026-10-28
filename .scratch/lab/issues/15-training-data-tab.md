# 15 Training data tab

Status: needs-triage
Type: task
Blocked by: 07, 11

Spec: `../spec.md` decision 19.

Sample about 30 clips per training bucket and source, using the review UI (with ticket 12's pre-screen if calibrated). Show the estimated bad-label rate with a range per bucket. Flags feed a training data version (an exclusion list). Retraining on it is a listener experiment, judged on test set v1.
