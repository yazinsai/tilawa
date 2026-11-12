# Prefix-supervised phone-head refit

Status: implementation prepared; acoustic premise, training and export blocked
because the supervisor cannot start commands. This is not a measured or killed
experiment. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −1 and overall −1, with no losses predicted. Whole-recording
CTC may permit weak phone occurrences to be emitted after their acoustic source.
The proposed refit freezes the shipped encoder and retains whole-recording CTC,
adding internal-prefix CTC that requires completed occurrences and marginalizes
the partly observed final occurrence. The acoustic premise remains unverified.

`modal/student_prefix.py` implements a differentiable CTC forward lattice and
detached full-transcript forward/backward alignments. At internal streaming chunk
boundaries, a single progress count or two adjacent counts must contain 95% of
the alignment posterior; diffuse boundaries supply no prefix target. This is a
training default, not a cut selected on practice. A prefix accepts the last
required phone, its trailing blank, or the active next phone. It cannot complete
that optional occurrence or advance into later phones. Adjacent uncertain
counts form a normalized probability mixture. Repeated phone IDs retain CTC's
required separating blank, and suffix frames contribute no prefix gradient.

The cache uses the encoder's actual streaming state loop, carrying embedding
state, attention/convolution state and processed lengths across the recording.
It retains lookahead, end padding and the SDK's output count. It does not cold
crop prefixes or substitute a whole-recording chunk mask for deployed state.
`modal/student_prefix_refit.py` reads geometry from the supplied deployed
manifest and checks the existing inventory, blank, frontend dimensions and
40 ms clock. The original encoder is frozen before caching; neither optimizer
contains encoder parameters.

Both arms use the same cached recordings, sampled training prefixes, recording
order, Adam optimizer, update count and disabled head dropout. Both compute the
auxiliary loss, but the ordinary-CTC control discards it. Full-recording CTC and
KL preservation against the initial head remain in both arms, including ordinary
speech and pause frames. This limits drift rather than guaranteeing quiet pauses.
The run has a one-day cache/refit budget.

The input list must contain performed phone-occurrence transcripts, identified
eligible training speakers and explicit `held_out: false`. Accepted sources are
authorized transcribed recordings and help-train none-labelled takes; the latter
also require `label: none`. Scripted mistakes, repeats, unresolved eligibility
and canonical-only transcript guesses are rejected before reading their audio.
Release-speaker exclusion must be established when preparing that list; a
boolean supplied by a caller is not independent proof of exclusion. No real
training list has been prepared or inspected here.

Eligible speakers are deterministically partitioned into training and validation
groups. Only training speakers supply or sample prefixes. Each arm's checkpoint
is selected by average whole-recording CTC on validation speakers, without
practice targets, incident counts or release speakers. The runner writes both
ordinary student checkpoints. Its optional export uses the existing exporter,
checks ONNX against cached streaming features and writes a generated bundle with
its manifest/hash only after parity passes. This export path has not been run.
Tracker, checker, insertion verification and SDK decoding code are unchanged.

## Validation and premise audit

The small CPU behavior suite checks CTC partition sums and alignment uncertainty
against exhaustive path enumeration, prefix probability and gradient agreement,
bounded progress, optional-final-phone marginalization, repeated labels,
unreachable states, eligibility rejection, speaker separation and matching head
updates. A synthetic streaming encoder checks carried state, warm-context masks,
padding and exact output count. All 11 tests passed on CPU with PyTorch 2.15.0
in an ignored worktree-local test environment (8.5 seconds for the final run).
Python syntax checks, CLI argument loading and `git diff --check` also passed.
These tests validate training code, not real acoustic recovery, the pinned
remote PyTorch 2.8 environment, an actual Zipformer export or a trained model.

The earlier ignored acoustic diagnostic remains prepared under `.autolab/`.
It captures full-vocabulary probabilities from the normal SDK backend, retains
phone-occurrence identity and reports first-entry and occupancy distributions
over several surrounding transcript lengths. It would compare missing added
consonants with the addition's acoustic end, a further 160 ms boundary margin
and labelled following speech. Its separate exhaustive-path checks passed for
four synthetic transcripts, including repeated labels.

All five registered incidents remain acoustically unresolved. At least two must
have a missing planted consonant whose conditioned emission mass concentrates
after the addition ends, inside following speech. Weak mass or neighbour-owned
phones alone do not establish that premise. Supplied label boundaries have not
been independently reviewed; no authorized independent audio service is
connected, and conditioned alignment is not independent listening evidence.

Supervisor requests 1–4 all returned `NotFoundError: No such file or directory`,
with zero compute charged and no command output. Requests 3 and 4 were just
`pwd`, establishing that this is a failure to start the environment rather than
failure of the diagnostic. Request 5, the required six-shard core/SDK test run,
failed identically. Request 6 for the new Python behavior suite failed with the
same error and zero compute charged. No alternate Modal route, scorer, lab
tooling, evidence or test data were changed.

## Limits and next step

The previous attempt's docs-only limitation is addressed by an executable head
refit and behavior tests. Completing the experiment still requires remote access
to the shipped source checkpoint, eligible transcribed training audio and the
practice audio. Even `pwd` fails through the only authorized compute route, so
those dependencies cannot be inventoried or used here. No acoustic premise
measurement, real checkpoint, complete exported model, target recovery count
or practice delta exists. The code must not be evaluated by silently decoding
with the main model: there is no `.autolab/model.json` declaration because
there is no verified replacement bundle.

Restore supervisor execution and run the prepared premise audit first. If fewer
than two targets meet the premise, revert these engine additions and record a
killed result. Otherwise, prepare the eligible training list and shipped source
checkpoint, run both arms, verify the complete export and inspect target wins.
Fewer than three built-model target wins requires reverting and killing; at
least three requires an official practice score before deciding. Infrastructure
failure is not zero target wins and does not establish a kill criterion.
All three practice scores remain available. No fold or discard was performed.
