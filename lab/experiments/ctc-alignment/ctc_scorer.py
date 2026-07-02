"""CTC forced-alignment scorer.

Given frame-level log-probabilities and candidate texts,
scores how well each candidate explains the audio using
the CTC forward algorithm (via torch.nn.functional.ctc_loss).
"""
import torch
import torch.nn.functional as F

# Sentinel score for candidates that can't possibly align
_IMPOSSIBLE = float("inf")


def score_candidates(
    logits: torch.Tensor,
    candidates: list[dict],
    tokenize_fn: callable,
    blank_id: int,
) -> list[tuple[dict, float]]:
    """Score candidate verses against audio frame logits.

    Args:
        logits: (1, T, V) raw model output
        candidates: list of verse dicts with "text_clean" field
        tokenize_fn: text -> list[int] character index mapping
        blank_id: CTC blank token index

    Returns:
        List of (candidate, score) sorted best-first.
        Score is normalized negative log-likelihood (lower = better match).
    """
    if not candidates:
        return []

    # CTC loss expects (T, N, C) for log_probs
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, 1, V)
    T = log_probs.size(0)

    # Tokenize all candidates and filter out those too long for the audio.
    # CTC requires T >= 2*L+1 (alternating blanks and characters).
    encoded = []
    target_lengths = []
    feasible_indices = []
    for i, c in enumerate(candidates):
        ids = tokenize_fn(c["text_clean"])
        if len(ids) == 0:
            ids = [blank_id]
        if len(ids) * 2 + 1 > T:
            # Target too long for this audio — skip CTC scoring
            continue
        encoded.append(ids)
        target_lengths.append(len(ids))
        feasible_indices.append(i)

    # Build results: infeasible candidates get _IMPOSSIBLE score
    results = []

    if not encoded:
        # All candidates are too long for the audio
        for c in candidates:
            results.append((c, _IMPOSSIBLE))
        results.sort(key=lambda x: x[1])
        return results

    N = len(encoded)
    log_probs_batch = log_probs.expand(T, N, -1).contiguous()
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

    all_targets = torch.tensor(
        [idx for seq in encoded for idx in seq], dtype=torch.long
    )

    losses = F.ctc_loss(
        log_probs_batch,
        all_targets,
        input_lengths,
        target_lengths_t,
        blank=blank_id,
        reduction="none",
        zero_infinity=False,
    )  # (N,)

    # Replace any remaining inf/nan with a large penalty
    losses = torch.where(
        torch.isfinite(losses), losses, torch.tensor(1e9)
    )

    # Normalize by target length to avoid bias toward short verses
    scores = (losses / target_lengths_t.float()).tolist()

    # Merge feasible scores with infeasible sentinels
    score_map = {feasible_indices[j]: scores[j] for j in range(N)}
    for i, c in enumerate(candidates):
        results.append((c, score_map.get(i, _IMPOSSIBLE)))

    results.sort(key=lambda x: x[1])
    return results
