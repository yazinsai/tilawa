# Articulatory waveguide reconstruction judge

Status: premise-probe implementation committed; remote validation blocked by the supervisor environment. The experiment is unfinished. No practice score or target result.

## Prediction

Card `listener-articulatory-waveguide-judge`, listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`, practice set `lab-v1.5u`: listener −1 and overall −1. The registered windows are the false substitutions at 26:106:1, 55:48:1 and 92:13:1. The card requires at least two physical-written/control-rival reversals; the builder instruction additionally requires three target removals before proceeding to practice scoring.

## Mechanism implemented for the premise probe

The speech-production model has six oral tube areas and a nasal side branch. Each tube is a lossy bidirectional fractional digital delay. A nodal admittance solve enforces pressure continuity and volume-flow conservation at every junction; positive areas, attenuated stable delays and passive terminations bound the possible explanations. Velum opening, nasal area scale, tract length and wall loss complete the ten physical coordinates. Glottal excitation and constriction-local turbulent excitation have separate transfer functions.

A small inverse network learns initial coordinates from generated waveguide waveforms. Canonical none-labelled single-ayah takes from help-train training speakers calibrate pronunciation-conditioned physical centers and low-rank ranges. An explicit exclusion list is combined with every speaker whose label metadata has a non-training split. Complete-transcript soft placements replace the equal-duration initialization. Shrinkage is selected on grouped training speakers, then refitted on the authorized training speakers. A ten-coordinate stable lattice/FIR source-filter model with the same inverse-network capacity, training data, timing and range procedure is the unconstrained experimental control.

For each exact string-path wrong-word comparison requested by the existing checker, both complete pronunciations receive one causal channel/noise/voicing estimate from audio strictly preceding the whole comparison interval. Gaussian waveform covariance reconstruction integrates source phase; its Fourier likelihood uses the physical transfer functions, rather than learned spectral emissions. The scorer integrates phone boundaries and holds a latent tract realization throughout each phone, with smooth entry trajectories. Every phone must consume audio. The matched control is diagnostic only: a production veto would require a finite greater physical likelihood for the written pronunciation.

The probe exports the actual checker hypotheses and frame spans, replays comparisons with exact-window matching, and checks that tracker output and other error classes are identical. Internal checker confirmations are retained for subsequent ownership and merge decisions; only the public final string-path verdict is vetoed. A deployable SDK scorer and complete side-asset bundle remain contingent on the premise check.

## Validation so far

Python syntax compilation and `git diff --check` pass. Added numerical tests cover travelling-wave equations, passive delay magnitudes, physical gradients, finite impulse reconstruction, nasal and constriction response changes, complete-pronunciation timing normalization, persistence of a realization across a phone, and exclusion of disputed/future samples from nuisance estimation. Core/SDK tests cover the exact comparison route, final verdict filtering, unavailable acoustic evidence, finite likelihood preferences and exact pronunciation/window/clock matching.

All four supervisor Modal requests returned `NotFoundError: No such file or directory` without command output or numerical/test results: two minimal environment probes, training-image numerical tests (request 3), and the six-shard repository test suite (request 4). The GPU attempt records USD 0.0529; the other requests record USD 0, and the reported remaining GPU budget is USD 249.95. Execution cannot be established from the returned error alone. No authorized recordings or trained assets were obtained, no target comparison made and no model bundle declared. These infrastructure failures do not establish the mechanism's kill criterion. No killed outcome file was written, and no baseline decode is presented as evaluating the proposed side judge.

## Remaining work when the supervisor is restored

Run the numerical invariants and core/SDK tests first. Train both synthetic initializers and physical/control pronunciation ranges using only the authorized training speakers. Export exact engine comparisons for the three registered sessions, freeze the training-selected models, and perform physical/control reconstruction and exact replay. If fewer than three registered targets are removed, or the card's physical/control premise fails, revert the engine changes and record a killed outcome with the measured result. Otherwise implement and verify the deployable SDK scorer and complete hashed bundle, declare it through the untracked model pointer, and take the required practice score before deciding. No practice-score invocation has been used.

## Limits

The likelihood uses a locally stationary Fourier covariance approximation and marginalizes source phase; it does not claim sample-aligned glottal waveform recovery. A low-order tract and sparse none-labelled recordings may produce inaccurate articulation ranges. Causal outside-window source/channel estimates can become stale with changing pitch or level. These are unmeasured risks, including suppression of genuine substitutions. Primary decoding, graph confirmations and insertion decisions are outside the intervention. No gain, loss count, clean-session guarantee or release result is claimed.
