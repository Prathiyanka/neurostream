"""Redis operations of the worker: reading jobs, reclaiming, publishing, dead-lettering.

Loss accounting under the ``ns:windows`` length cap
---------------------------------------------------
Ingest adds jobs with ``XADD ... MAXLEN ~ N``. Trimming removes the oldest entries whether
or not a worker has acknowledged them, so under overload jobs can disappear two ways:

* **never delivered** (trimmed while waiting in the stream). The consumer group's
  ``entries-read`` is a *logical position* in everything ever added (verified on Redis
  7.4): when a read jumps over trimmed entries it advances by more than the number
  delivered. ``READ_SCRIPT`` reads the position, reads the jobs and reads the position
  again in one atomic script, so ``skipped = after - before - delivered`` is exact even
  with many workers reading at once. Doing it as separate commands would let another
  worker's read land in between and make the counter wrong.
* **delivered but unacknowledged** (trimmed while pending). The worker holding it still
  has the data, so this only loses the job if that worker dies. ``XAUTOCLAIM`` then
  reports the entry as deleted, and it is counted.

Both are added to ``ns:metrics:workers`` (``lost_trimmed_undelivered``,
``lost_trimmed_pending``). Then, once the stream is drained, this always holds
(tested in ``test_worker_streams.py``)::

    entries-added = acked + lost_trimmed_undelivered + lost_trimmed_pending
                    + dead_lettered + pending + lag

Assumption: nothing deletes entries from ``ns:windows`` with ``XDEL`` (it would make
Redis's ``entries-read`` unknowable). Nothing in NeuroStream does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from redis import Redis
from redis.exceptions import ResponseError

from neurostream_common import keys
from neurostream_common.codec import Fields

Entry = tuple[str, dict[bytes, bytes]]  # (stream id, fields)

# KEYS: stream, metrics hash. ARGV: group, consumer, count.
# Returns {entries or false, skipped, last-delivered-id}.
READ_SCRIPT = r"""
local function group_info()
  for _, g in ipairs(redis.call('XINFO', 'GROUPS', KEYS[1])) do
    local info = {}
    for i = 1, #g, 2 do info[g[i]] = g[i + 1] end
    if info['name'] == ARGV[1] then return info end
  end
  return redis.error_reply('NOGROUP no group ' .. ARGV[1])
end
local before = group_info()['entries-read'] or 0
local res = redis.call('XREADGROUP', 'GROUP', ARGV[1], ARGV[2], 'COUNT', ARGV[3],
                       'STREAMS', KEYS[1], '>')
local n = 0
if res then n = #res[1][2] end
local after = group_info()
local skipped = (after['entries-read'] or 0) - before - n
if skipped > 0 then
  redis.call('HINCRBY', KEYS[2], 'lost_trimmed_undelivered', skipped)
end
if n > 0 then redis.call('HINCRBY', KEYS[2], 'delivered', n) end
return {res, skipped, after['last-delivered-id']}
"""

# KEYS: stream, metrics hash. ARGV: group, ids... Acks and counts in one step, so `acked`
# counts each job exactly once even if two workers both processed it (only one ack wins).
ACK_SCRIPT = r"""
local n = redis.call('XACK', KEYS[1], ARGV[1], unpack(ARGV, 2))
redis.call('HINCRBY', KEYS[2], 'acked', n)
local missed = #ARGV - 1 - n
if missed > 0 then redis.call('HINCRBY', KEYS[2], 'ack_misses', missed) end
return n
"""


@dataclass(frozen=True)
class ReadResult:
    entries: list[Entry]
    skipped: int  # jobs trimmed before anyone read them, found by this read
    last_delivered_id: str


def _fields(flat: Sequence[bytes]) -> dict[bytes, bytes]:
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


def _text(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


class JobStreams:
    def __init__(
        self,
        redis: Redis,
        consumer: str,
        stream: str = keys.WINDOWS_STREAM,
        group: str = keys.WORKER_GROUP,
        metrics_key: str = keys.WORKER_METRICS,
    ) -> None:
        self.r = redis
        self.consumer = consumer
        self.stream = stream
        self.group = group
        self.metrics_key = metrics_key
        self._read = redis.register_script(READ_SCRIPT)
        self._ack = redis.register_script(ACK_SCRIPT)
        self._claim_cursor = "0-0"

    def ensure_group(self) -> None:
        """Create the group if needed, starting at the beginning of the stream (``0``) so a
        job added before the first worker started is still processed, not skipped."""
        try:
            self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def read(self, count: int) -> ReadResult:
        """Take up to ``count`` jobs that are available *now*. Never waits to fill a batch."""
        try:
            res, skipped, last_id = self._read(
                keys=[self.stream, self.metrics_key], args=[self.group, self.consumer, count]
            )
        except ResponseError as exc:
            if "NOGROUP" not in str(exc) and "no such key" not in str(exc):
                raise
            self.ensure_group()  # stream or group was deleted (e.g. Redis flushed)
            return ReadResult([], 0, "0-0")
        entries = [(_text(i), _fields(f)) for i, f in res[0][1]] if res else []
        return ReadResult(entries, int(skipped), _text(last_id))

    def wait(self, after_id: str, block_ms: int) -> None:
        """Sleep until an entry newer than ``after_id`` exists (or ``block_ms`` passes).

        A plain ``XREAD`` used only as a wake-up: it consumes nothing, so no job can be
        missed between the empty read and the wait."""
        self.r.xread({self.stream: after_id}, count=1, block=block_ms)

    def reclaim(self, min_idle_ms: int, count: int) -> tuple[list[tuple[Entry, int]], int]:
        """Take over jobs pending longer than ``min_idle_ms`` on any consumer.

        Returns ([(entry, times delivered)], number of pending jobs that had been trimmed:
        those are lost and counted)."""
        res: list[Any] = self.r.xautoclaim(  # type: ignore[assignment]
            self.stream, self.group, self.consumer, min_idle_ms, self._claim_cursor, count
        )
        cursor, claimed, deleted = res[0], res[1], res[2]
        self._claim_cursor = _text(cursor)  # resume the scan where it stopped; 0-0 = wrapped
        n_deleted = len(deleted)
        if n_deleted:
            self.r.hincrby(self.metrics_key, "lost_trimmed_pending", n_deleted)
        if not claimed:
            return [], n_deleted
        pipe = self.r.pipeline(transaction=False)
        for entry_id, _ in claimed:
            pipe.xpending_range(self.stream, self.group, entry_id, entry_id, 1)
        pending = pipe.execute()  # type: ignore[no-untyped-call]
        counts = [rows[0]["times_delivered"] if rows else 1 for rows in pending]
        entries = [(_text(i), dict(f)) for i, f in claimed]
        return list(zip(entries, counts, strict=True)), n_deleted

    def publish_and_ack(
        self,
        predictions: Sequence[tuple[Fields, str]],
        ack_ids: Sequence[str],
        predictions_maxlen: int,
        counters: dict[str, int],
    ) -> int:
        """Publish predictions (durable stream + live pub/sub), *then* ack, in one round
        trip. Acking only after publishing means a crash can duplicate a prediction but
        never lose one. Returns how many jobs this ack newly acknowledged."""
        pipe = self.r.pipeline(transaction=False)
        for fields, live_json in predictions:
            pipe.xadd(
                keys.PREDICTIONS_STREAM,
                fields,  # type: ignore[arg-type]
                maxlen=predictions_maxlen,
                approximate=True,
            )
            pipe.publish(keys.PREDICTIONS_CHANNEL, live_json)
        for name, value in counters.items():
            if value:
                pipe.hincrby(self.metrics_key, name, value)
        if ack_ids:
            self._ack(
                keys=[self.stream, self.metrics_key], args=[self.group, *ack_ids], client=pipe
            )
        results = pipe.execute()  # type: ignore[no-untyped-call]
        return int(results[-1]) if ack_ids else 0

    def dead_letter(self, entry: Entry, reason: str, deliveries: int, maxlen: int) -> None:
        entry_id, fields = entry
        record: dict[bytes | str, bytes | str | int] = {k: v for k, v in fields.items()}
        record.update(
            {
                "dead_stream_id": entry_id,
                "dead_reason": reason,
                "dead_consumer": self.consumer,
                "dead_deliveries": deliveries,
            }
        )
        pipe = self.r.pipeline(transaction=False)
        pipe.xadd(keys.WINDOWS_DEAD_STREAM, record, maxlen=maxlen, approximate=True)  # type: ignore[arg-type]
        pipe.hincrby(self.metrics_key, "dead_lettered", 1)
        pipe.xack(self.stream, self.group, entry_id)  # plain ack: not counted as `acked`
        pipe.execute()  # type: ignore[no-untyped-call]
