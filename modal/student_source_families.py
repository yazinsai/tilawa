"""Performed-recording families for matched frozen-encoder phone-head refits.

Identity is acoustic, never inferred from reader, transcript or verse. All views
of a recording retain its family; family mass is divided over copies and views.
Grouping settings must be frozen on training-speaker calibration before use.
"""

from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class CopyPolicy:
    sample_rate: int = 16000
    max_delay_seconds: float = .05
    min_coverage: float = .99
    min_correlation: float = .995
    max_relative_error: float = .1
    block_seconds: float = .25

    def __post_init__(self):
        if (self.sample_rate <= 0 or self.max_delay_seconds < 0 or self.block_seconds <= 0 or
                not 0 < self.min_coverage <= 1 or not 0 < self.min_correlation <= 1 or
                self.max_relative_error < 0):
            raise ValueError('Invalid acoustic copy policy')


def waveform_hash(wave, sample_rate):
    """Exact decoded samples, with rate and length included in the identity."""
    import numpy as np

    x = np.asarray(wave, dtype='<f4')
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Waveform must contain finite mono samples')
    h = hashlib.sha256(f'{sample_rate}:{len(x)}:'.encode())
    h.update(x.tobytes())
    return h.hexdigest()


def verify_reencoding(left, right, policy):
    """Verify full performance correspondence, allowing only codec delay/gain.

    Every non-silent block must agree, so matching silence or one shared phrase
    cannot identify a whole recording. No tempo deformation is allowed: that
    would make two performances of the same text easier to conflate.
    """
    import numpy as np
    from scipy.signal import correlate, correlation_lags

    a, b = [np.asarray(w, dtype=np.float64) for w in (left, right)]
    if any(w.ndim != 1 or not len(w) or not np.isfinite(w).all() for w in (a, b)):
        raise ValueError('Waveform must contain finite mono samples')
    if min(len(a), len(b))/max(len(a), len(b)) < policy.min_coverage:
        return None
    a, b = a-a.mean(), b-b.mean()
    if min(np.dot(a, a), np.dot(b, b)) <= 0:
        return None
    limit = round(policy.sample_rate*policy.max_delay_seconds)
    cc = correlate(a, b, mode='full', method='fft')
    lags = correlation_lags(len(a), len(b))
    eligible = np.abs(lags) <= limit
    lag = int(lags[eligible][np.argmax(cc[eligible])])
    start_a, start_b = max(0, lag), max(0, -lag)
    length = min(len(a)-start_a, len(b)-start_b)
    if length/max(len(a), len(b)) < policy.min_coverage:
        return None
    a, b = a[start_a:start_a+length], b[start_b:start_b+length]
    aa, bb, ab = float(a@a), float(b@b), float(a@b)
    correlation = ab/math.sqrt(aa*bb)
    gain = ab/bb
    relative_error = math.sqrt(float(np.sum((a-gain*b)**2))/aa)
    if gain <= 0 or correlation < policy.min_correlation or relative_error > policy.max_relative_error:
        return None
    block = max(1, round(policy.block_seconds*policy.sample_rate))
    # The maximum block energy defines silence without candidate text or labels.
    intervals = [(i, min(i+block, length)) for i in range(0, length, block)]
    energies = [(float(a[i:j]@a[i:j]), float(b[i:j]@b[i:j])) for i, j in intervals]
    floor = max(max(p, q) for p, q in energies)*1e-4
    supported = 0
    minimum_block_correlation = 1.
    for (i, j), (p, q) in zip(intervals, energies):
        if max(p, q) <= floor:
            continue
        if min(p, q) <= floor:
            return None
        c = float(a[i:j]@b[i:j])/math.sqrt(p*q)
        minimum_block_correlation = min(minimum_block_correlation, c)
        if c < policy.min_correlation:
            return None
        supported += 1
    if supported < 2:
        return None
    return dict(lag=lag, gain=gain, coverage=length/max(len(left), len(right)),
                correlation=correlation, relative_error=relative_error,
                minimum_block_correlation=minimum_block_correlation, speech_blocks=supported)


def _signatures(recordings, policy):
    import numpy as np

    rows = sorted(recordings, key=lambda r: r[0])
    if len(set(k for k, _ in rows)) != len(rows):
        raise ValueError('Recording keys must be unique')
    waves, hashes, envelopes = {}, {}, {}
    for key, wave in rows:
        wave = np.asarray(wave)
        hashes[key] = waveform_hash(wave, policy.sample_rate)
        waves[key] = wave
        # A cheap acoustic candidate screen. Interpolated full-recording RMS
        # envelopes reject unrelated takes before the expensive sample-level
        # verification. Passing this screen never supplies an identity witness.
        bin_size = max(1, round(policy.sample_rate*.01))
        powers = np.asarray([np.mean(np.square(wave[i:i+bin_size], dtype=np.float64))
                             for i in range(0, len(wave), bin_size)])
        envelope = np.sqrt(np.interp(np.linspace(0, len(powers)-1, 128), np.arange(len(powers)), powers))
        envelopes[key] = envelope/max(float(np.linalg.norm(envelope)), 1e-12)
    return waves, hashes, envelopes


def _candidate(left, right, waves, hashes, envelopes, policy):
    return (min(len(waves[left]), len(waves[right]))/max(len(waves[left]), len(waves[right])) >= policy.min_coverage
            and (hashes[left] == hashes[right] or float(envelopes[left]@envelopes[right]) >= .98))


def _witness(left, right, waves, hashes, policy):
    import numpy as np

    if hashes[left] == hashes[right] and float(np.std(waves[right])) > 0:
        return dict(kind='exact-decoded', sha256=hashes[right])
    witness = verify_reencoding(waves[left], waves[right], policy)
    return None if witness is None else dict(kind='verified-reencoding', **witness)


def source_families(recordings, policy):
    """Complete-link families prevent weak acoustic matches chaining takes.

    Caller supplies one training-speaker partition, after release exclusions.
    Silence alone is no performance identity. Accepted pairs carry witnesses.
    """
    waves, hashes, envelopes = _signatures(recordings, policy)
    families, certificates = [], []
    for key in waves:
        placed = False
        for family in families:
            first = family[0]
            if not _candidate(first, key, waves, hashes, envelopes, policy):
                continue
            found = []
            for member in family:
                witness = _witness(member, key, waves, hashes, policy)
                if witness is None:
                    break
                found.append(dict(left=member, right=key, **witness))
            else:
                family.append(key)
                certificates.extend(found)
                placed = True
                break
        if not placed:
            families.append([key])
    mapping = {key: hashlib.sha256('\n'.join(family).encode()).hexdigest()
               for family in families for key in family}
    return mapping, certificates


def partition_aliases(recordings, partition_by_recording, policy):
    """Audit every cross-partition candidate pair, independent of clustering.

    A pair may share a performance even when one member fails complete-link
    against a third recording. Such an alias still invalidates held-out loss.
    """
    waves, hashes, envelopes = _signatures(recordings, policy)
    keys = list(waves)
    witnesses = []
    for i, left in enumerate(keys):
        for right in keys[i+1:]:
            if partition_by_recording[left] == partition_by_recording[right]:
                continue
            if not _candidate(left, right, waves, hashes, envelopes, policy):
                continue
            witness = _witness(left, right, waves, hashes, policy)
            if witness is not None:
                witnesses.append(dict(left=left, right=right, **witness))
    return witnesses


def view_weights(views, family_by_recording):
    """Matched original mass versus exactly equal total source-family mass.

    A view has a recording key and the original epoch probability for that view.
    Original weights are preserved in the control. Balanced mass is normalized
    within each family across *all* its copies and cached augmentations, retaining
    their relative proportions. One family with many views still gets one unit.
    """
    mass = {}
    for view in views:
        family = family_by_recording[view['recording']]
        w = view['original_weight']
        if not math.isfinite(w) or w <= 0:
            raise ValueError('Original sampling weights must be positive and finite')
        mass[family] = mass.get(family, 0.) + w
    if not mass:
        raise ValueError('No training views')
    total = sum(mass.values())
    original = [v['original_weight']/total for v in views]
    balanced = [v['original_weight']/mass[family_by_recording[v['recording']]]/len(mass) for v in views]
    return original, balanced
