import numpy as np

from neurostream_common.codec import (
    RawPacket,
    WindowJob,
    decode_raw_packet,
    decode_window_job,
    encode_raw_packet,
    encode_window_job,
)
from neurostream_common.labels import LEFT, REST, RIGHT, UNKNOWN, annotations_to_labels
from neurostream_common.windowing import Window


def as_redis(fields: dict[str, object]) -> dict[bytes, bytes]:
    """What redis-py returns: bytes keys and values."""
    return {k.encode(): v if isinstance(v, bytes) else str(v).encode() for k, v in fields.items()}


def test_window_job_roundtrip() -> None:
    data = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 7
    w = Window(
        start_sample=80,
        data=data,
        labels=np.array([0, 0, 1, 1, 2, -1], np.int8),
        pad_samples=2,
        filled_samples=1,
    )
    job = WindowJob("sess", w, 160.0, ingest_ts=1.25, emitted_ts=1.5)
    back = decode_window_job(as_redis(encode_window_job(job)))
    assert back.window_id == "sess:85" == job.window_id
    np.testing.assert_array_equal(back.window.data, data)
    np.testing.assert_array_equal(back.window.labels, w.labels)
    assert (back.window.pad_samples, back.window.filled_samples) == (2, 1)
    assert (back.ingest_ts, back.emitted_ts, back.sfreq) == (1.25, 1.5, 160.0)


def test_raw_packet_roundtrip() -> None:
    p = RawPacket("s", 3, 48, 10.0, 11.0, np.ones((2, 16), np.float32), np.zeros(16, np.int8))
    back = decode_raw_packet(as_redis(encode_raw_packet(p)))
    assert (back.seq, back.start_sample, back.timestamp) == (3, 48, 10.0)
    np.testing.assert_array_equal(back.data, p.data)


def test_annotations_to_labels() -> None:
    labels = annotations_to_labels(
        n_samples=100,
        sfreq=10.0,
        onsets_s=[0.0, 4.0, 6.0],
        durations_s=[4.0, 2.0, 2.0],
        descriptions=["T0", "T1", "T2"],
    )
    assert labels[0] == REST and labels[39] == REST
    assert labels[40] == LEFT and labels[60] == RIGHT
    assert labels[80] == UNKNOWN  # after the last annotation


def test_unknown_annotations_ignored() -> None:
    labels = annotations_to_labels(10, 10.0, [0.0], [1.0], ["BAD"])
    assert np.all(labels == UNKNOWN)


def test_prediction_round_trips_through_stream_fields_and_json() -> None:
    import json

    from neurostream_common.codec import Prediction, decode_prediction, encode_prediction

    p = Prediction(
        window_id="s1:399",
        session_id="s1",
        window_end_sample=399,
        window_end_ts=1700000000.25,
        predicted_class=2,
        probabilities=(0.1, 0.2, 0.7),
        true_label=-1,
        filled_samples=3,
        model="eegnet-motor-1.0.1",
        worker_id="w1",
        reclaimed=True,
        batch_size=4,
        ingest_ms=0.5,
        queue_ms=1.5,
        preprocess_ms=0.4,
        inference_ms=1.2,
        latency_ms=3.9,
        publish_ts=1700000000.2539,
    )
    # Redis returns bytes for every field value.
    raw = {k.encode(): str(v).encode() for k, v in encode_prediction(p).items()}
    assert decode_prediction(raw) == p
    assert json.loads(p.to_json())["probabilities"] == [0.1, 0.2, 0.7]
