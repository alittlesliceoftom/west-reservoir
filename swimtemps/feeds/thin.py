"""Thinning a feed to one reading per clock hour, on arrival."""

from datetime import datetime, timedelta


def _distance_to_the_hour(at: datetime) -> timedelta:
    past = at - at.replace(minute=0, second=0, microsecond=0)
    return min(past, timedelta(hours=1) - past)


def thin_hourly(rows: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """
    One row per clock hour: the one nearest the top of an hour (this one or
    the next), the earlier on a tie, so a re-fetch keeps the same rows.
    Input order does not matter; the output is in time order.
    """
    kept: dict[datetime, tuple[datetime, float]] = {}
    for at, temp in rows:
        hour = at.replace(minute=0, second=0, microsecond=0)
        best = kept.get(hour)
        if best is None or (_distance_to_the_hour(at), at) < (_distance_to_the_hour(best[0]), best[0]):
            kept[hour] = (at, temp)
    return [kept[hour] for hour in sorted(kept)]
