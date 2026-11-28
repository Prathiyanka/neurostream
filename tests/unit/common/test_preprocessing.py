import numpy as np
import pytest

from neurostream_common.channels import MOTOR_CHANNELS, channel_indices
from neurostream_common.preprocessing import (
    PreprocessConfig,
    WindowPreprocessor,
    causal_filter,
    filter_and_crop,
    normalize_window,
)

CFG = PreprocessConfig()
FS = 160.0


def sine(freq: float, n: int = 1600, amp: float = 1.0) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / FS))[None, :]


def test_filter_is_causal() -> None:
    """Changing future samples must not change any earlier output sample."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 800))
    y1 = causal_filter(x, CFG.sos)
    x2 = x.copy()
    x2[:, 500:] = rng.normal(size=(3, 300)) * 100
    y2 = causal_filter(x2, CFG.sos)
    np.testing.assert_array_equal(y1[:, :500], y2[:, :500])
    assert not np.allclose(y1[:, 500:], y2[:, 500:])


@pytest.mark.parametrize(
    ("freq", "passes"), [(2.0, False), (12.0, True), (20.0, True), (50.0, False)]
)
def test_band_pass_response(freq: float, passes: bool) -> None:
    y = causal_filter(sine(freq), CFG.sos)[0, 800:]  # after settling
    gain = np.abs(y).max()
    assert (gain > 0.8) if passes else (gain < 0.1)


def test_steady_state_init_suppresses_dc_transient() -> None:
    dc = np.full((1, 400), 500.0)  # constant offset in microvolts
    assert np.abs(causal_filter(dc, CFG.sos, "steady")).max() < 1e-6
    assert np.abs(causal_filter(dc, CFG.sos, "zero")).max() > 10


def test_unknown_init_rejected() -> None:
    with pytest.raises(ValueError):
        causal_filter(np.ones((1, 10)), CFG.sos, "warm")


def test_filter_and_crop_drops_pad() -> None:
    span = np.random.default_rng(1).normal(size=(4, 480)).astype(np.float32)
    core = filter_and_crop(span, 160, CFG)
    assert core.shape == (4, 320) and core.dtype == np.float32
    np.testing.assert_allclose(core, causal_filter(span, CFG.sos)[:, 160:], rtol=1e-5, atol=1e-5)


def test_normalization_keeps_relative_channel_power() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 320)) * np.array([[1.0], [3.0]]) + 7.0
    z = normalize_window(x)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-5)
    assert z.std() == pytest.approx(1.0, rel=1e-4)
    ratio_before = x[1].std() / x[0].std()
    assert z[1].std() / z[0].std() == pytest.approx(ratio_before, rel=1e-4)


def test_channel_selection() -> None:
    available = ["Fp1", "C3", "Cz", "C4"]
    assert channel_indices(available, ["C4", "C3"]) == [3, 1]
    assert channel_indices(available, None) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="FC5"):
        channel_indices(available, ["FC5"])


def test_window_preprocessor_selects_motor_channels() -> None:
    names = [f"X{i}" for i in range(43)] + list(MOTOR_CHANNELS)
    prep = WindowPreprocessor(PreprocessConfig(channels=MOTOR_CHANNELS), names)
    span = np.random.default_rng(3).normal(size=(64, 480))
    out = prep(span, 160)
    assert out.shape == (21, 320) and prep.n_channels == 21


def test_eegmmidb_channel_list_is_64_unique_and_contains_the_motor_set() -> None:
    from neurostream_common.channels import EEGMMIDB_CHANNELS, MOTOR_CHANNELS

    assert len(EEGMMIDB_CHANNELS) == len(set(EEGMMIDB_CHANNELS)) == 64
    assert set(MOTOR_CHANNELS) <= set(EEGMMIDB_CHANNELS)
