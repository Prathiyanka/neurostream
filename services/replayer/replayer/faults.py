"""Deterministic fault injection, so every faulty run can be reproduced from its seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaultConfig:
    drop_prob: float = 0.0
    duplicate_prob: float = 0.0
    reorder_prob: float = 0.0  # swap this packet with the next one
    jitter_ms: float = 0.0  # extra send delay, uniform in [0, jitter_ms]
    disconnect_every_s: float = 0.0  # 0 disables; measured in recording time
    reconnect_delay_s: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("drop_prob", "duplicate_prob", "reorder_prob"):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {p}")
        if self.jitter_ms < 0 or self.disconnect_every_s < 0 or self.reconnect_delay_s < 0:
            raise ValueError("jitter/disconnect/reconnect values must be >= 0")

    @property
    def any(self) -> bool:
        return bool(
            self.drop_prob
            or self.duplicate_prob
            or self.reorder_prob
            or self.jitter_ms
            or self.disconnect_every_s
        )


@dataclass(frozen=True)
class PacketFate:
    drop: bool = False
    duplicate: bool = False
    reorder: bool = False
    delay_s: float = 0.0


class FaultInjector:
    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def fate(self) -> PacketFate:
        c = self.config
        if not c.any:
            return PacketFate()
        # Always draw all four numbers so a packet's fate doesn't depend on earlier outcomes.
        r = self._rng.random(4)
        drop = bool(r[0] < c.drop_prob)
        return PacketFate(
            drop=drop,
            duplicate=not drop and bool(r[1] < c.duplicate_prob),
            reorder=not drop and bool(r[2] < c.reorder_prob),
            delay_s=float(r[3] * c.jitter_ms / 1000.0),
        )
