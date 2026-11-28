"""Entry point: ``python -m api.main``."""

from __future__ import annotations

import os

import uvicorn

from api.app import create_app
from neurostream_common.logging_utils import configure_logging


def main() -> None:
    configure_logging()
    uvicorn.run(
        create_app(),
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
