import time
import logging


class DuplicateCache:
    """
    Final-level deduplication.

    Blocks sending the same finalized plate
    within a configured time window.
    """

    def __init__(self, window_seconds: int):
        self.window = window_seconds
        self.cache = {}

    def is_duplicate(self, plate: str) -> bool:
        now = time.time()

        if plate in self.cache:
            elapsed = now - self.cache[plate]
            if elapsed < self.window:
                logging.info(
                    f"[DUPLICATE BLOCKED] Plate={plate} "
                    f"Elapsed={elapsed:.2f}s"
                )
                return True

        self.cache[plate] = now
        return False
