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

# The 64 EEGMMIDB channels in recording order (as MNE names them after loading the EDF).
# Used to give synthetic recordings a realistic layout, so models can find their channels.
EEGMMIDB_CHANNELS: tuple[str, ...] = (
    *MOTOR_CHANNELS,
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
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
