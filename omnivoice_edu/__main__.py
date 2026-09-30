"""Entry point: ``python -m omnivoice_edu`` or the ``omnivoice-edu`` command."""

import logging

import uvicorn

from .config import settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logging.getLogger(__name__).info(
        "Starting OmniVoice Edu on %s:%s", settings.host, settings.port
    )
    uvicorn.run(
        "omnivoice_edu.api:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
