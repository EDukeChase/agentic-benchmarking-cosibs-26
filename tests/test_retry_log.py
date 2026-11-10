import logging
import unittest

from src.utils.retry_log import LOGGER_NAME, RetryCounter


class RetryCounterTests(unittest.TestCase):
    def test_counts_retries_and_adds_up_the_waits(self):
        counter = RetryCounter().install()
        self.addCleanup(logging.getLogger(LOGGER_NAME).removeHandler, counter)
        log = logging.getLogger(LOGGER_NAME)
        log.info("Retrying request to %s in %f seconds", "/responses", 0.5)
        log.info("Retrying request to %s in %f seconds", "/responses", 8.0)
        log.info("something else")
        self.assertEqual(counter.summary(), {"retries": 2, "wait_seconds": 8.5})


if __name__ == "__main__":
    unittest.main()
