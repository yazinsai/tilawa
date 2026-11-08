# Live page: folded 24-call engine

Status: ready-for-agent

The user authorized deploying the best autolab findings on 2026-11-08 for manual
microphone testing. The latest folded result is already on main: practice phone
24 wrong calls (13 misses, 11 false warnings), listener 23, tracker 1, checker 0.
This is a practice result, with no exam result recorded.

The primary spliced-clean int8 model hash matches the folded bundle exactly.
The folded addition gate also requires the fp32 Muaalem insertion teacher
declared in packages/sdk/src/registry.ts. Browser loadWebModel does not install
that auxiliary asset. Its external weights are 2.4 GB.

Deploy the unchanged teacher ONNX bundle on Modal using the SDK's existing
createInsertionTeacher frontend and decoder. Attach its remote decode callback
on the live page. Primary streaming stays on-device; final seam checks send
cropped audio online when the recording stops. No model retraining, inference
algorithm changes, or new lab experiment. Keep the recorded 24-call result
distinct from live browser smoke testing.

Service: modal/live_teacher.py (tilawa-live-teacher). Bundle:
/lab/data/models/exp-listener-teacher-insert-marginal-gate-ad6eb8c0.
The service verifies both teacher hashes on container start.

Deployment: python3 tools/build_live_review.py, modal deploy modal/live_teacher.py,
then modal deploy modal/lab_review.py. User testing URL:
https://amal-invest--tilawa-lab-review-web.us-east.modal.run/live/.
