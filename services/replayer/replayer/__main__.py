"""CLI: ``python -m replayer --subjects 1 2 --speed 1``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from dataclasses import asdict
from pathlib import Path

from neurostream_common.logging_utils import configure_logging
from replayer.faults import FaultConfig
from replayer.recording import load_subject, synthetic_recording
from replayer.streamer import SessionConfig, SessionStreamer


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="replayer", description=__doc__)
    p.add_argument("--url", default="ws://localhost:8001/ws/device")
    p.add_argument(
        "--subjects", type=int, nargs="*", default=[1], help="one concurrent session per subject"
    )
    p.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="stream N synthetic sessions instead of real data (no dataset needed)",
    )
    p.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--drop-prob", type=float, default=0.0)
    p.add_argument("--duplicate-prob", type=float, default=0.0)
    p.add_argument("--reorder-prob", type=float, default=0.0)
    p.add_argument("--jitter-ms", type=float, default=0.0)
    p.add_argument("--disconnect-every-s", type=float, default=0.0)
    p.add_argument("--reconnect-delay-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    if args.synthetic:
        recs = [
            synthetic_recording(n_channels=64, seconds=60, seed=i) for i in range(args.synthetic)
        ]
    else:
        recs = [load_subject(s, args.data_dir, args.runs) for s in args.subjects]
    streamers = []
    for i, rec in enumerate(recs):
        name = f"sub{rec.subject:03d}" if rec.subject is not None else f"synth{i}"
        faults = FaultConfig(
            drop_prob=args.drop_prob,
            duplicate_prob=args.duplicate_prob,
            reorder_prob=args.reorder_prob,
            jitter_ms=args.jitter_ms,
            disconnect_every_s=args.disconnect_every_s,
            reconnect_delay_s=args.reconnect_delay_s,
            seed=args.seed + i,
        )
        cfg = SessionConfig(
            url=args.url,
            session_id=f"{name}-{secrets.token_hex(3)}",
            chunk_size=args.chunk_size,
            speed=args.speed,
            loop=args.loop,
            max_seconds=args.max_seconds,
            faults=faults,
        )
        streamers.append(SessionStreamer(cfg, rec))
    results = await asyncio.gather(*(s.run() for s in streamers))
    print(json.dumps([asdict(r) for r in results], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["serve"]:  # long-running device service, controlled over HTTP
        from replayer.server import serve

        serve(port=int(os.environ.get("REPLAYER_PORT", "8002")))
        return 0
    try:
        return asyncio.run(run(parse_args(argv)))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
