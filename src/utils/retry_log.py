"""Counts the model client's own retries (rate limits, server errors, dropped connections).

The OpenAI client retries a failed request after a short wait and logs "Retrying request to <url>
in <seconds> seconds". This handler counts those lines and adds up the waits, so a run's record
shows how much of its time went to waiting on the service.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "openai._base_client"


class RetryCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.count = 0
        self.wait_seconds = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if not message.startswith("Retrying request"):
                return
            self.count += 1
            self.wait_seconds += float(message.rsplit(" in ", 1)[1].split(" ", 1)[0])
        except (IndexError, ValueError):
            self.count += 1

    def install(self) -> "RetryCounter":
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.addHandler(self)
        return self

    def summary(self) -> dict:
        return {"retries": self.count, "wait_seconds": round(self.wait_seconds, 1)}
