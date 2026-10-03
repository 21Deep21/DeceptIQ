import numpy as np
from scipy import sparse

from features.feature_builder import URLFeatureBuilder
from model.train_v2 import CONDITIONS, augment_training, fusion_matrix


def _tiny():
    urls = ([f"http://192.0.2.{10+i}:8080/login{i}.php" for i in range(8)]
            + [f"https://docs{i}.example.org/guide{i}" for i in range(8)])
    y = np.array([1] * 8 + [0] * 8)
    E = np.zeros((16, 14))
    E[:8, 0] = 1        # malicious rows: dns_ok
    E[8:, 8] = 1        # benign rows: rdap_ok
    E[:, 3] = 5         # a nonzero numeric everywhere
    return urls, y, E


def test_augment_shapes_labels_and_conditions():
    urls, y, E = _tiny()
    Xurl = URLFeatureBuilder().fit(urls).transform(urls)
    X, yy, cond, orig = augment_training(Xurl, E, y, seed=0)
    assert X.shape[0] == 4 * len(y)
    assert X.shape[1] == Xurl.shape[1] + 14
    assert len(yy) == len(cond) == len(orig)
    for c in CONDITIONS:
        assert int((cond == c).sum()) == len(y)
    # labels preserved per copy
    assert all(yy[i] == y[orig[i]] for i in range(len(yy)))


def test_augment_masked_conditions_zero_evidence_block():
    urls, y, E = _tiny()
    Xurl = URLFeatureBuilder().fit(urls).transform(urls)
    X, yy, cond, orig = augment_training(Xurl, E, y, seed=0)
    ev = np.asarray(X[:, Xurl.shape[1]:].todense())
    none_rows = cond == "none"
    assert np.all(ev[none_rows] == 0)
    full_rows = cond == "full"
    assert np.allclose(ev[full_rows], E[orig[full_rows]])
    dns_only = cond == "dns_only"
    assert np.all(ev[dns_only][:, 8:] == 0)
    assert np.all(np.abs(ev[dns_only][:, :8]).sum(axis=1) > 0)  # each row keeps SOME dns signal


def test_augment_deterministic():
    urls, y, E = _tiny()
    Xurl = URLFeatureBuilder().fit(urls).transform(urls)
    a = augment_training(Xurl, E, y, seed=7)
    b = augment_training(Xurl, E, y, seed=7)
    assert (a[0] != b[0]).nnz == 0
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_fusion_matrix_shape_and_mask():
    urls, y, E = _tiny()
    b = URLFeatureBuilder().fit(urls)
    X = fusion_matrix(b, urls, E)
    assert X.shape == (16, b.transform(urls).shape[1] + 14)
    Xm = fusion_matrix(b, urls, E, mask=("dns", "rdap"))
    ev = np.asarray(Xm[:, b.transform(urls).shape[1]:].todense())
    assert np.all(ev == 0)
