import numpy as np
import pytest

from neurostream_common.labels import UNKNOWN
from neurostream_common.windowing import StreamingWindower, WindowSpec, extract_windows

SPEC = WindowSpec(sfreq=160.0, window_s=2.0, step_s=0.5, pad_s=1.0)


def recording(n: int = 1600, n_ch: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_ch, n)).astype(np.float32)
    labels = ((np.arange(n) // 400) % 3).astype(np.int8)
    return data, labels


def test_spec_sample_counts() -> None:
    assert (SPEC.window_samples, SPEC.step_samples, SPEC.pad_samples) == (320, 80, 160)
    assert SPEC.span_samples == 480


def test_invalid_spec_rejected() -> None:
    with pytest.raises(ValueError):
        WindowSpec(step_s=0.0)


def test_offline_window_count_and_positions() -> None:
    data, labels = recording()
    windows = list(extract_windows(data, labels, SPEC))
    assert len(windows) == (1600 - 480) // 80 + 1
    assert [w.start_sample for w in windows[:3]] == [0, 80, 160]
    assert windows[0].end_sample == 479
    assert windows[0].data.shape == (4, 480)
    np.testing.assert_array_equal(windows[1].data, data[:, 80:560])


@pytest.mark.parametrize("chunk", [1, 7, 16, 80, 500])
def test_streaming_matches_offline_exactly(chunk: int) -> None:
    """The guarantee training relies on: serving sees exactly the windows training saw."""
    data, labels = recording()
    offline = list(extract_windows(data, labels, SPEC))
    w = StreamingWindower(SPEC, n_channels=4)
    streamed = []
    for start in range(0, data.shape[1], chunk):
        streamed += w.push(start, data[:, start : start + chunk], labels[start : start + chunk])
    assert len(streamed) == len(offline)
    for a, b in zip(streamed, offline, strict=True):
        assert a.start_sample == b.start_sample
        np.testing.assert_array_equal(a.data, b.data)
        np.testing.assert_array_equal(a.labels, b.labels)
        assert a.filled_samples == 0


def test_streaming_buffer_stays_bounded() -> None:
    data, labels = recording(n=16_000)
    w = StreamingWindower(SPEC, n_channels=4)
    for start in range(0, 16_000, 16):
        w.push(start, data[:, start : start + 16], labels[start : start + 16])
        assert w._data.shape[1] <= SPEC.span_samples + 16


def test_label_end_and_purity() -> None:
    data, labels = recording()
    windows = list(extract_windows(data, labels, SPEC))
    w0 = windows[0]  # core = samples 160..479, labels 0 until 400 then 1
    assert w0.label_end == 1 and not w0.label_pure
    w = next(w for w in windows if w.start_sample == 640)  # core 800..1119 -> all label 2
    assert w.label_end == 2 and w.label_pure


def test_small_gap_is_filled_by_holding_last_value() -> None:
    data, labels = recording()
    w = StreamingWindower(SPEC, n_channels=4, max_fill_samples=40)
    out = []
    for start in range(0, 1600, 16):
        if start == 320:  # lose one 16-sample packet
            continue
        out += w.push(start, data[:, start : start + 16], labels[start : start + 16])
    assert w.resets == 0 and w.filled_total == 16
    first = out[0]
    np.testing.assert_array_equal(first.data[:, 320:336], np.repeat(data[:, 319:320], 16, axis=1))
    assert np.all(first.labels[320:336] == UNKNOWN)
    assert first.filled_samples == 16 and not first.label_pure
    # Same number of windows as without the gap: one lost packet costs no windows.
    assert len(out) == (1600 - 480) // 80 + 1


def test_large_gap_resets_window_grid() -> None:
    data, labels = recording()
    w = StreamingWindower(SPEC, n_channels=4, max_fill_samples=40)
    w.push(0, data[:, :400], labels[:400])
    out = w.push(1000, data[:, 1000:1600], labels[1000:1600])
    assert w.resets == 1
    assert out[0].start_sample == 1000  # grid restarts at the new data
    assert all(x.filled_samples == 0 for x in out)
    assert len(out) == (600 - 480) // 80 + 1


def test_rewinding_is_an_error() -> None:
    data, labels = recording()
    w = StreamingWindower(SPEC, n_channels=4)
    w.push(0, data[:, :16], labels[:16])
    with pytest.raises(ValueError):
        w.push(0, data[:, :16], labels[:16])
