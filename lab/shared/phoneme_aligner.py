"""Phoneme-level alignment using edit distance.

Aligns predicted phoneme sequences against reference sequences and
classifies each position as correct, substitution, deletion, or insertion.
"""


def align_phonemes(predicted: list[str], reference: list[str]) -> dict:
    """Align predicted vs reference phoneme sequences using edit distance.

    Uses standard Levenshtein DP with backtrace to produce an optimal
    alignment, then classifies each aligned position.

    Args:
        predicted: List of predicted phoneme tokens (e.g., ["b", "i", "s", "m"])
        reference: List of reference phoneme tokens

    Returns:
        dict with keys:
            errors: list of {type, position, expected, got}
            per: phoneme error rate (edits / reference length)
            correct_rate: fraction of correctly matched phonemes
            alignment: list of (ref_token, pred_token) pairs
    """
    n = len(reference)
    m = len(predicted)

    # Edge cases
    if n == 0 and m == 0:
        return {
            "errors": [],
            "per": 0.0,
            "correct_rate": 1.0,
            "alignment": [],
        }
    if n == 0:
        # All insertions
        errors = [
            {"type": "insertion", "position": 0, "expected": None, "got": predicted[j]}
            for j in range(m)
        ]
        alignment = [(None, p) for p in predicted]
        return {
            "errors": errors,
            "per": float(m),  # m / 0 is undefined; use m as total errors
            "correct_rate": 0.0,
            "alignment": alignment,
        }
    if m == 0:
        # All deletions
        errors = [
            {"type": "deletion", "position": i, "expected": reference[i], "got": None}
            for i in range(n)
        ]
        alignment = [(r, None) for r in reference]
        return {
            "errors": errors,
            "per": 1.0,
            "correct_rate": 0.0,
            "alignment": alignment,
        }

    # --- Build DP matrix ---
    # dp[i][j] = min edit distance between reference[:i] and predicted[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"  # deletion
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"  # insertion

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == predicted[j - 1] else 1
            sub = dp[i - 1][j - 1] + cost
            ins = dp[i][j - 1] + 1      # insertion (extra in predicted)
            delete = dp[i - 1][j] + 1    # deletion (missing from predicted)
            best = min(sub, ins, delete)
            dp[i][j] = best
            # Tie-break: prefer substitution/match > deletion > insertion
            if best == sub:
                bt[i][j] = "S"
            elif best == delete:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"

    # --- Backtrace ---
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            alignment.append((None, predicted[j - 1]))
            j -= 1
        elif j == 0:
            alignment.append((reference[i - 1], None))
            i -= 1
        else:
            move = bt[i][j]
            if move == "S":
                alignment.append((reference[i - 1], predicted[j - 1]))
                i -= 1
                j -= 1
            elif move == "D":
                alignment.append((reference[i - 1], None))
                i -= 1
            else:  # "I"
                alignment.append((None, predicted[j - 1]))
                j -= 1

    alignment.reverse()

    # --- Classify each position ---
    errors = []
    correct = 0
    ref_pos = 0  # tracks position in the reference sequence

    for ref_tok, pred_tok in alignment:
        if ref_tok is not None and pred_tok is not None:
            if ref_tok == pred_tok:
                correct += 1
            else:
                errors.append({
                    "type": "substitution",
                    "position": ref_pos,
                    "expected": ref_tok,
                    "got": pred_tok,
                })
            ref_pos += 1
        elif ref_tok is not None and pred_tok is None:
            errors.append({
                "type": "deletion",
                "position": ref_pos,
                "expected": ref_tok,
                "got": None,
            })
            ref_pos += 1
        else:
            # Insertion: pred_tok present, ref_tok is None
            errors.append({
                "type": "insertion",
                "position": ref_pos,
                "expected": None,
                "got": pred_tok,
            })

    total_edits = len(errors)
    per = total_edits / n
    correct_rate = correct / n

    return {
        "errors": errors,
        "per": per,
        "correct_rate": correct_rate,
        "alignment": alignment,
    }


def align_phoneme_strings(predicted: str, reference: str) -> dict:
    """Convenience wrapper that accepts space-separated phoneme strings."""
    pred_tokens = predicted.strip().split() if predicted.strip() else []
    ref_tokens = reference.strip().split() if reference.strip() else []
    return align_phonemes(pred_tokens, ref_tokens)
