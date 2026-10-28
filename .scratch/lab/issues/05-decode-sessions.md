# 05 Decode test set v1 on Modal

Status: ready-for-agent
Type: task
Blocked by: 03

Decode each session (studio and phone-degraded) with the main-line listener `spliced-clean-int8` (streaming SDK, 20 ms pushes, posteriors on) and with the teacher. Evidence goes onto the lab volume, keyed by test set version, listener and model hash. One command decodes any listener checkpoint, so listener experiments can use it later.
