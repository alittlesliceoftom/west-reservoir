"""
The worker: one long-running process.

Every hour it sweeps every active water; every 15 seconds it recomputes the
waters the web has flagged dirty, and writes a heartbeat. An error in one
cycle is logged and the loop carries on; the web keeps serving the last
snapshot meanwhile.

    python -m worker.main
"""

import logging
import signal
import time
from datetime import datetime, timedelta, timezone

from db.session import session_factory
from forecast.openmeteo import OpenMeteoClient
from worker import cycle

SWEEP_EVERY = timedelta(hours=1)
POLL_SECONDS = 15

log = logging.getLogger("swimtemps.worker")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_once(factory, client, now: datetime, last_sweep: datetime | None) -> datetime | None:
    """One pass of the loop. Returns when the last sweep ran."""
    if last_sweep is None or now - last_sweep >= SWEEP_EVERY:
        done = cycle.sweep(factory, client, now)
        log.info("sweep: %d waters", done)
        last_sweep = now
    else:
        done = cycle.drain_dirty(factory, client, now)
        if done:
            log.info("recomputed %d dirty waters", done)
    cycle.write_heartbeat(factory, now)
    return last_sweep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    factory = session_factory()
    client = OpenMeteoClient()
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    last_sweep = None
    log.info("worker started")
    while not stopping:
        try:
            last_sweep = run_once(factory, client, utcnow(), last_sweep)
        except Exception:
            log.exception("cycle failed")
        for _ in range(POLL_SECONDS):
            if stopping:
                break
            time.sleep(1)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
