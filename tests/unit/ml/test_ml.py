from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from neurostream_common.preprocessing import PreprocessConfig  # noqa: E402
from neurostream_common.windowing import WindowSpec  # noqa: E402
from neurostream_ml.csp import CSPLDA, window_covariances  # noqa: E402
from neurostream_ml.dataset import pure_windows_for_run  # noqa: E402
from neurostream_ml.eegnet import EEGNet  # noqa: E402
from neurostream_ml.export import export_onnx  # noqa: E402
from neurostream_ml.metrics import bootstrap_ci, per_subject, summarize  # noqa: E402
from neurostream_ml.splits import subject_folds  # noqa: E402
from neurostream_ml.train import TrainConfig, predict_proba, train_eegnet  # noqa: E402


def test_subject_folds_never_share_subjects() -> None:
    subjects = list(range(1, 107))
    folds = subject_folds(subjects, k=5, seed=0)
    tests = [set(f.test) for f in folds]
    assert set().union(*tests) == set(subjects)
    assert sum(len(t) for t in tests) == len(subjects)  # each subject tested exactly once
    for f in folds:
        assert not (set(f.train) & set(f.test)) and not (set(f.val) & set(f.test))
        assert not (set(f.train) & set(f.val))
        assert set(f.train) | set(f.val) | set(f.test) == set(subjects)
    assert subject_folds(subjects, seed=0) == folds  # deterministic


def test_pure_windows_only_and_labelled_by_core() -> None:
    n = 1600
    labels = np.zeros(n, np.int8)
    labels[800:] = 2
    data = np.random.default_rng(0).normal(size=(4, n)).astype(np.float32)
    X, y = pure_windows_for_run(data, labels, WindowSpec(), PreprocessConfig())
    # cores are [start+160, start+480); pure iff entirely below 800 or at/after 800
    starts = range(0, n - 480 + 1, 80)
    expected = [0 if s + 480 <= 800 else 2 for s in starts if s + 480 <= 800 or s + 160 >= 800]
    assert y.tolist() == expected and X.shape == (len(expected), 4, 320)


def synthetic_mi(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Class c has extra power on channel c: what CSP is designed to find."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, 6, 320))
    X[np.arange(n), y] *= 3.0
    return X.astype(np.float32), y


def test_csp_lda_learns_variance_differences() -> None:
    X, y = synthetic_mi()
    Xt, yt = synthetic_mi(seed=1)
    model = CSPLDA(n_filters_per_side=1).fit_covs(window_covariances(X), y)
    acc = (model.predict_covs(window_covariances(Xt)) == yt).mean()
    assert acc > 0.9


@pytest.mark.parametrize("n_channels", [21, 64])
def test_eegnet_shapes_and_max_norm(n_channels: int) -> None:
    m = EEGNet(n_channels, 320)
    assert m(torch.zeros(5, 1, n_channels, 320)).shape == (5, 3)
    with torch.no_grad():
        m.classifier.weight.mul_(100)
    m.apply_max_norm()
    assert m.classifier.weight.norm(dim=1).max() <= 0.25 + 1e-6


def test_training_loop_runs_and_restores_best_model() -> None:
    X, y = synthetic_mi(n=192)
    res = train_eegnet(X, y, X[:64], y[:64], TrainConfig(epochs=3, patience=2, threads=1))
    assert 0 <= res.best_epoch < 3 and len(res.history) >= 1
    proba = predict_proba(res.model, X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_onnx_export_matches_pytorch(tmp_path: Path) -> None:
    model = EEGNet(6, 320).eval()
    assert export_onnx(model, tmp_path / "m.onnx") < 1e-5


def test_metrics() -> None:
    y = np.array([0, 0, 0, 1, 2, 2])
    p = np.array([0, 0, 1, 1, 2, 0])
    m = summarize(y, p)
    assert m["majority_baseline"] == pytest.approx(0.5)
    assert m["confusion"][0] == [2, 1, 0]
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 + 0.5) / 3)
    ps = per_subject(y, p, np.array([1, 1, 1, 2, 2, 2]))
    assert set(ps) == {1, 2}
    lo, hi = bootstrap_ci([0.4, 0.5, 0.6])
    assert lo <= 0.5 <= hi
