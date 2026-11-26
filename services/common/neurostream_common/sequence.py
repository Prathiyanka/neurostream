"""Sequence-number bookkeeping: detects gaps, duplicates and late (out-of-order) packets.

Policy: no reorder buffer. A packet is either the next expected one (``OK``), ahead of it
(``GAP``: the skipped packets are presumed lost), or behind it. Behind means ``LATE`` if
that seq was one we had marked missing, otherwise ``DUPLICATE``. Late packets are dropped
because their samples were already gap-filled. Holding packets back to reorder them would
add latency to every packet to fix a rare case.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class SeqStatus(Enum):
    OK = "ok"
    GAP = "gap"
    DUPLICATE = "duplicate"
    LATE = "late"


@dataclass(frozen=True)
class SeqResult:
    status: SeqStatus
    missing: int = 0  # packets skipped (only for GAP)

    @property
    def accept(self) -> bool:
        return self.status in (SeqStatus.OK, SeqStatus.GAP)


@dataclass
class SeqCounters:
    received: int = 0
    accepted: int = 0
    duplicates: int = 0
    late: int = 0
    gaps: int = 0
    missing_packets: int = 0


class SequenceTracker:
    def __init__(self, max_missing_ranges: int = 1024) -> None:
        self.expected: int | None = None
        self.counters = SeqCounters()
        # (start, stop) half-open ranges of seqs we skipped; bounded so memory can't grow.
        self._missing: deque[tuple[int, int]] = deque(maxlen=max_missing_ranges)

    def observe(self, seq: int) -> SeqResult:
        c = self.counters
        c.received += 1
        if self.expected is None or seq == self.expected:
            result = SeqResult(SeqStatus.OK)
        elif seq > self.expected:
            missing = seq - self.expected
            self._missing.append((self.expected, seq))
            c.gaps += 1
            c.missing_packets += missing
            result = SeqResult(SeqStatus.GAP, missing)
        elif self._take_missing(seq):
            c.late += 1
            return SeqResult(SeqStatus.LATE)
        else:
            c.duplicates += 1
            return SeqResult(SeqStatus.DUPLICATE)
        c.accepted += 1
        self.expected = seq + 1
        return result

    def _take_missing(self, seq: int) -> bool:
        for i, (start, stop) in enumerate(self._missing):
            if start <= seq < stop:
                del self._missing[i]
                if start < seq:
                    self._missing.append((start, seq))
                if seq + 1 < stop:
                    self._missing.append((seq + 1, stop))
                return True
        return False
