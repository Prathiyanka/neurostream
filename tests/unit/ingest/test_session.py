import numpy as np
import pytest

from ingest.session import SessionState
from ingest.writer import DropOldestQueue, StreamItem
from neurostream_common.schemas import DataPacket, SessionHello
from neurostream_common.sequence import SeqStatus

CH = ["C3", "Cz", "C4"]


def make_state(chunk: int = 16) -> SessionState:
    hello = SessionHello(session_id="s1", sfreq=160.0, ch_names=CH, chunk_size=chunk)
    return SessionState(hello, window_s=2.0, step_s=0.5, pad_s=1.0, max_fill_s=0.25)


def pkt(seq: int, n_ch: int = 3, n: int = 16, value: float = 0.0) -> DataPacket:
    return DataPacket(
        session_id="s1",
        seq=seq,
        timestamp=0.0,
        samples=[[value + seq] * n for _ in range(n_ch)],
        truth={"labels": [1] * n},  # type: ignore[arg-type]
    )


def test_windows_appear_once_enough_samples_arrived() -> None:
    st = make_state()
    jobs = []
    for seq in range(30):  # 480 samples == exactly one padded window
        jobs += st.process(pkt(seq), ingest_ts=100.0 + seq, now=0.0).jobs
    assert len(jobs) == 1
    assert jobs[0].window.end_sample == 479
    assert jobs[0].ingest_ts == 129.0  # arrival of the packet holding the last sample
    assert jobs[0].window_id == "s1:479"


def test_duplicates_produce_no_output_and_are_counted() -> None:
    st = make_state()
    st.process(pkt(0), 0.0, 0.0)
    r = st.process(pkt(0), 0.0, 0.0)
    assert r.status is SeqStatus.DUPLICATE and r.raw is None and not r.jobs
    assert st.metrics()["duplicates"] == 1


def test_gap_is_filled_in_sample_space() -> None:
    st = make_state()
    st.process(pkt(0), 0.0, 0.0)
    r = st.process(pkt(2), 0.0, 0.0)  # packet 1 lost
    assert r.status is SeqStatus.GAP
    m = st.metrics()
    assert m["missing_packets"] == 1 and m["filled_samples"] == 16 and m["windower_resets"] == 0


def test_long_gap_resets_windowing() -> None:
    st = make_state()
    st.process(pkt(0), 0.0, 0.0)
    st.process(pkt(10), 0.0, 0.0)  # 9 packets = 144 samples > 40-sample fill limit
    assert st.metrics()["windower_resets"] == 1


def test_shape_mismatch_is_invalid() -> None:
    st = make_state()
    r = st.process(pkt(0, n_ch=2), 0.0, 0.0)
    assert r.status is None and r.error and st.counters.invalid == 1
    assert st.tracker.counters.received == 0  # invalid packets don't advance the sequence


def test_reconnect_compatibility() -> None:
    st = make_state()
    same = SessionHello(session_id="s1", sfreq=160.0, ch_names=CH, chunk_size=16)
    other = SessionHello(session_id="s1", sfreq=160.0, ch_names=CH, chunk_size=32)
    assert st.compatible_with(same) and not st.compatible_with(other)


def test_missing_truth_gives_unknown_labels() -> None:
    st = make_state()
    p = DataPacket(session_id="s1", seq=0, timestamp=0.0, samples=[[0.0] * 16] * 3)
    r = st.process(p, 0.0, 0.0)
    assert r.raw is not None and np.all(r.raw.labels == -1)


def item(i: int) -> StreamItem:
    return StreamItem("k", {"i": i}, 10)


async def test_queue_drops_oldest_when_full() -> None:
    q = DropOldestQueue(maxsize=3)
    for i in range(5):
        q.put(item(i))
    assert q.dropped == 2 and len(q) == 3
    batch = await q.get_batch(10)
    assert [b.fields["i"] for b in batch] == [2, 3, 4]


async def test_queue_batches_are_capped() -> None:
    q = DropOldestQueue(maxsize=100)
    for i in range(10):
        q.put(item(i))
    assert len(await q.get_batch(4)) == 4
    assert len(q) == 6


@pytest.mark.parametrize("n", [1, 3])
async def test_queue_get_waits_for_items(n: int) -> None:
    import asyncio

    q = DropOldestQueue(maxsize=10)
    task = asyncio.create_task(q.get_batch(10))
    await asyncio.sleep(0.01)
    assert not task.done()
    for i in range(n):
        q.put(item(i))
    assert len(await asyncio.wait_for(task, 1.0)) == n
