from neurostream_common.sequence import SeqStatus, SequenceTracker


def statuses(seqs: list[int]) -> tuple[list[SeqStatus], SequenceTracker]:
    t = SequenceTracker()
    return [t.observe(s).status for s in seqs], t


def test_in_order_stream_is_all_ok() -> None:
    st, t = statuses([0, 1, 2, 3])
    assert st == [SeqStatus.OK] * 4
    assert t.counters.accepted == 4 and t.counters.gaps == 0


def test_stream_may_start_at_any_seq() -> None:
    st, _ = statuses([41, 42])
    assert st == [SeqStatus.OK, SeqStatus.OK]


def test_gap_reports_number_of_missing_packets() -> None:
    t = SequenceTracker()
    t.observe(0)
    r = t.observe(4)
    assert r.status is SeqStatus.GAP and r.missing == 3 and r.accept
    assert t.counters.gaps == 1 and t.counters.missing_packets == 3
    assert t.expected == 5


def test_duplicate_is_rejected() -> None:
    st, t = statuses([0, 1, 1, 2, 0])
    assert st == [
        SeqStatus.OK,
        SeqStatus.OK,
        SeqStatus.DUPLICATE,
        SeqStatus.OK,
        SeqStatus.DUPLICATE,
    ]
    assert t.counters.duplicates == 2 and t.counters.accepted == 3


def test_reordered_packet_is_late_not_duplicate() -> None:
    # 2 arrives before 1: 2 opens a gap, then 1 fills it late.
    st, t = statuses([0, 2, 1, 3])
    assert st == [SeqStatus.OK, SeqStatus.GAP, SeqStatus.LATE, SeqStatus.OK]
    assert t.counters.late == 1 and t.counters.duplicates == 0


def test_late_packet_counts_only_once() -> None:
    st, _ = statuses([0, 3, 1, 1, 2])
    assert st == [SeqStatus.OK, SeqStatus.GAP, SeqStatus.LATE, SeqStatus.DUPLICATE, SeqStatus.LATE]


def test_missing_ranges_are_bounded() -> None:
    t = SequenceTracker(max_missing_ranges=2)
    for s in [0, 2, 4, 6]:  # three gaps; the oldest range is forgotten
        t.observe(s)
    assert t.observe(1).status is SeqStatus.DUPLICATE  # forgotten -> can't tell it was late
    assert t.observe(5).status is SeqStatus.LATE
