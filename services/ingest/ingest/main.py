"""Entry point: ``python -m ingest.main``."""

from __future__ import annotations

import os

import uvicorn

from ingest.app import create_app
from neurostream_common.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    uvicorn.run(
        create_app(),
        host=os.environ.get("INGEST_HOST", "0.0.0.0"),
        port=int(os.environ.get("INGEST_PORT", "8001")),
        log_config=None,
        ws_max_size=4 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
