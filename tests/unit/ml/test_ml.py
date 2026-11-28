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
    train, val = np.arange(64, 192), np.arange(64)
    res = train_eegnet(X, y, train, val, TrainConfig(epochs=3, patience=2, threads=1))
    assert 0 <= res.best_epoch < 3 and len(res.history) >= 1
    proba = predict_proba(res.model, X, np.arange(10))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # A boolean mask selects the same windows as the equivalent index array.
    mask = np.zeros(len(X), dtype=bool)
    mask[:10] = True
    np.testing.assert_array_equal(predict_proba(res.model, X, mask), proba)
    # Selecting by index must not depend on the windows outside the selection.
    X2 = X.copy()
    X2[10:] = 0
    np.testing.assert_allclose(predict_proba(res.model, X2, np.arange(10)), proba, atol=1e-6)


class _Killed(Exception):
    pass


def test_resumed_training_matches_uninterrupted(tmp_path: Path) -> None:
    """Killed after epoch 1 and resumed: same history, same best epoch, same weights."""
    X, y = synthetic_mi(n=192)
    cfg = TrainConfig(epochs=4, patience=4, threads=1)
    full = train_eegnet(X, y, np.arange(64, 192), np.arange(64), cfg)

    resume = tmp_path / "fold.resume.pt"

    def kill(epoch: int) -> None:
        if epoch == 1:
            raise _Killed

    with pytest.raises(_Killed):
        train_eegnet(X, y, np.arange(64, 192), np.arange(64), cfg, resume, kill)
    assert torch.load(resume, weights_only=False)["epoch"] == 1
    resumed = train_eegnet(X, y, np.arange(64, 192), np.arange(64), cfg, resume)

    assert resumed.history == full.history
    assert resumed.best_epoch == full.best_epoch
    for k, v in full.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[k], v, rtol=0, atol=0)


def test_resume_refuses_a_different_config(tmp_path: Path) -> None:
    X, y = synthetic_mi(n=192)
    resume = tmp_path / "fold.resume.pt"
    train_eegnet(X, y, np.arange(64, 192), np.arange(64), TrainConfig(epochs=1, threads=1), resume)
    with pytest.raises(SystemExit, match="written with"):
        train_eegnet(
            X,
            y,
            np.arange(64, 192),
            np.arange(64),
            TrainConfig(epochs=1, lr=0.01, threads=1),
            resume,
        )


def test_training_with_index_arrays_matches_training_on_copies() -> None:
    """Index-based training (no copies) is the same computation as training on a slice."""
    X, y = synthetic_mi(n=192)
    cfg = TrainConfig(epochs=2, patience=2, threads=1)
    a = train_eegnet(X, y, np.arange(64, 192), np.arange(64), cfg)
    Xc, yc = X[np.r_[64:192, 0:64]], y[np.r_[64:192, 0:64]]
    b = train_eegnet(Xc, yc, np.arange(128), np.arange(128, 192), cfg)
    assert [h["val_bal_acc"] for h in a.history] == [h["val_bal_acc"] for h in b.history]
    np.testing.assert_allclose(
        [h["train_loss"] for h in a.history], [h["train_loss"] for h in b.history], rtol=1e-5
    )


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


def test_parse_history_log_keeps_only_finished_folds(tmp_path: Path) -> None:
    from neurostream_ml.experiments import parse_history_log

    log = tmp_path / "train.log"
    log.write_text(
        "x INFO ml.train: epoch 0 loss 1.0769 val_bal_acc 0.3682 (38s)\n"
        "x INFO ml.train: epoch 1 loss 1.0500 val_bal_acc 0.4000 (75s)\n"
        "x INFO ml.experiments: fold 0 test {'accuracy': 0.4}\n"
        "x INFO ml.train: epoch 0 loss 1.0826 val_bal_acc 0.3915 (37s)\n"
        "x INFO ml.experiments: fold 1 test {'accuracy': 0.5}\n"
        "x INFO ml.train: epoch 0 loss 1.0809 val_bal_acc 0.3902 (43s)\n"  # killed here
    )
    hist = parse_history_log(log)
    assert sorted(hist) == [0, 1]
    assert [h["val_bal_acc"] for h in hist[0]] == [0.3682, 0.4]
    assert hist[0][-1]["elapsed_s"] == 75.0 and len(hist[1]) == 1


def test_load_window_set_is_disk_backed_and_matches_serving_normalization(
    tmp_path: Path,
) -> None:
    import json

    from neurostream_common.preprocessing import normalize_window
    from neurostream_ml.dataset import load_window_set

    rng = np.random.default_rng(0)
    raw = (rng.normal(size=(10, 4, 320)) * 50).astype(np.float16)
    np.save(tmp_path / "X.npy", raw)
    np.save(tmp_path / "y.npy", np.zeros(10, np.int8))
    np.save(tmp_path / "subject.npy", np.ones(10, np.int16))
    np.save(tmp_path / "run.npy", np.full(10, 4, np.int8))
    (tmp_path / "done.json").write_text(json.dumps({"ch_names": ["A", "B", "C", "D"]}))

    ws = load_window_set(tmp_path, ["D", "B"], chunk=3)  # chunk not dividing N
    assert isinstance(ws.X, np.memmap) and ws.ch_names == ["D", "B"]
    expected = np.stack([normalize_window(w[[3, 1]].astype(np.float32)) for w in raw])
    np.testing.assert_array_equal(ws.X, expected.astype(np.float16))
    files = sorted(f.name for f in tmp_path.glob("X_norm_*"))
    assert len(files) == 1 and ".tmp" not in files[0]
    again = load_window_set(tmp_path, ["D", "B"])  # reuses the file
    np.testing.assert_array_equal(again.X, ws.X)
    assert sorted(f.name for f in tmp_path.glob("X_norm_*")) == files
    del ws, again  # release the maps so Windows can delete tmp_path


def test_paired_diff_uses_only_shared_subjects() -> None:
    from neurostream_ml.report_tables import paired_diff

    a = {"1": 0.5, "2": 0.6, "3": 0.7, "9": 0.0}
    b = {"1": 0.4, "2": 0.5, "3": 0.6}
    mean, lo, hi, n = paired_diff(a, b)
    assert n == 3 and mean == pytest.approx(0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
