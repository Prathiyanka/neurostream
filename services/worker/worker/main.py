"""Entry point: ``python -m worker.main``. Configuration comes from environment variables
(see ``worker.settings``)."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from types import FrameType

from redis import Redis

from neurostream_common.logging_utils import configure_logging
from neurostream_common.model_card import ModelCardError, load_model_card
from worker.loop import Worker
from worker.model import InferenceModel
from worker.settings import Settings

log = logging.getLogger("worker")


def main() -> int:
    configure_logging()
    s = Settings.from_env()
    try:
        card = load_model_card(Path(s.model_dir) / s.model)  # checks the sha256
    except ModelCardError as exc:
        log.error("refusing to start: %s", exc)
        return 2
    worker = Worker(s, Redis.from_url(s.redis_url), InferenceModel(card, s.ort_threads))

    def _stop(signum: int, _frame: FrameType | None) -> None:
        log.info("signal %d: finishing the current batch, then stopping", signum)
        worker.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
