import numpy as np
import pytest

from inner_vote.vote import agreement_matrix, consensus_curve, consensus_report

N, D = 4, 8
SHARED = np.ones(D)


def shared_layer():
    states = []
    for i in range(N):
        v = SHARED.copy()
        v[i] = 1.01
        states.append(v)
    return np.stack(states)


def orthogonal_layer():
    return np.eye(N, D)


def test_agreement_matrix_shapes():
    m = agreement_matrix(shared_layer())
    assert m.shape == (N, N)
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T)


def test_shared_layer_agrees():
    m = agreement_matrix(shared_layer())
    off = m[~np.eye(N, dtype=bool)]
    assert float(np.min(off)) > 0.99


def test_orthogonal_layer_disagrees():
    m = agreement_matrix(orthogonal_layer())
    off = m[~np.eye(N, dtype=bool)]
    assert float(np.max(np.abs(off))) == pytest.approx(0.0)


def test_consensus_report_curve():
    layers = [shared_layer(), shared_layer(), orthogonal_layer(), orthogonal_layer()]
    r = consensus_report(layers, low=0.7)
    assert len(r["curve"]) == 4
    assert r["curve"][0]["mean_agreement"] > 0.9
    assert r["curve"][2]["mean_agreement"] == pytest.approx(0.0)
    assert r["first_disagreement_layer"] == 2
    assert r["peak_layer"] == 0
    assert r["final_agreement"] == pytest.approx(0.0)


def test_token_mean_path():
    # (N paraphrases, T tokens, D) states
    states = np.stack([shared_layer() for _ in range(3)], axis=1)
    assert states.shape == (N, 3, D)
    m = agreement_matrix(states)
    assert m.shape == (N, N)
    assert float(np.min(m)) > 0.99


def test_single_paraphrase_layer():
    layers = [np.ones((1, D))]
    r = consensus_report(layers)
    assert r["curve"][0]["mean_agreement"] == 1.0
    assert r["first_disagreement_layer"] is None


def test_errors():
    with pytest.raises(ValueError):
        consensus_curve([])
    with pytest.raises(ValueError):
        agreement_matrix(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError):
        agreement_matrix(np.zeros((0, D)))