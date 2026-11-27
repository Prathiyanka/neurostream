"""Channel sets. A model declares the channels it needs (in its model card)."""

from __future__ import annotations

from collections.abc import Sequence

# Sensorimotor strip over the motor cortex (10-10 system): the frontocentral, central and
# centroparietal rows from column 5 to 6. Hand-motor imagery shows up here as mu/beta
# desynchronization, strongest over C3 (right hand) and C4 (left hand).
MOTOR_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)  # fmt: skip

CHANNEL_SETS: dict[str, tuple[str, ...] | None] = {"all": None, "motor": MOTOR_CHANNELS}


def channel_indices(available: Sequence[str], wanted: Sequence[str] | None) -> list[int]:
    """Indices of ``wanted`` in ``available``, in ``wanted`` order (None = all channels)."""
    if wanted is None:
        return list(range(len(available)))
    lookup = {name: i for i, name in enumerate(available)}
    missing = [c for c in wanted if c not in lookup]
    if missing:
        raise ValueError(f"channels not in recording: {missing}")
    return [lookup[c] for c in wanted]
