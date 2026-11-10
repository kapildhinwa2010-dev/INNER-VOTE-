"""Measure layer-wise hidden-state agreement across paraphrases.

Given the states of N paraphrases of the same question at every
layer, INNER-VOTE reports the mean and minimum pairwise cosine
agreement per layer, the first layer where the paraphrases
disagree, and the layer with the strongest consensus. Pure NumPy.
"""

import numpy as np


def _mean_states(states):
    s = np.asarray(states, dtype=float)
    if s.ndim == 3:
        s = s.mean(axis=1)
    if s.ndim != 2:
        raise ValueError("states must have shape (N, D) or (N, T, D)")
    if s.shape[0] < 1:
        raise ValueError("states must contain at least one paraphrase")
    return s


def agreement_matrix(states):
    """Pairwise cosine matrix between paraphrase states."""
    s = _mean_states(states)
    norm = np.clip(np.linalg.norm(s, axis=1, keepdims=True), 1e-12, None)
    u = s / norm
    return u @ u.T


def consensus_curve(states_by_layer):
    """Mean and minimum off-diagonal agreement per layer."""
    if not isinstance(states_by_layer, (list, tuple)) or len(states_by_layer) == 0:
        raise ValueError("states_by_layer must be a non-empty list")
    out = []
    for s in states_by_layer:
        m = agreement_matrix(s)
        n = m.shape[0]
        off = m[~np.eye(n, dtype=bool)]
        out.append({
            "mean_agreement": float(np.mean(off)) if off.size else 1.0,
            "min_agreement": float(np.min(off)) if off.size else 1.0,
        })
    return out


def consensus_report(states_by_layer, low=0.7):
    """Full internal consensus report.

    ``first_disagreement_layer`` is the first layer whose mean
    agreement falls below ``low``. ``peak_layer`` is the layer with
    the strongest consensus.
    """
    curve = consensus_curve(states_by_layer)
    means = [c["mean_agreement"] for c in curve]
    first = None
    for i, m in enumerate(means):
        if m < low:
            first = i
            break
    return {
        "curve": curve,
        "first_disagreement_layer": first,
        "peak_layer": int(np.argmax(means)),
        "final_agreement": means[-1],
    }