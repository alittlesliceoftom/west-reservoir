import random
from datetime import datetime, timedelta, timezone

from feeds.thin import thin_hourly

T0 = datetime(2026, 10, 18, 0, 0, tzinfo=timezone.utc)


def at(hour, minute=0):
    return T0 + timedelta(hours=hour, minutes=minute)


def test_half_hourly_input_halves_keeping_the_top_of_each_hour():
    rows = [(at(h, m), 13.0 + h / 10) for h in range(6) for m in (0, 30)]
    thinned = thin_hourly(rows)
    assert [t for t, _ in thinned] == [at(h) for h in range(6)]


def test_input_order_does_not_change_the_choice():
    rows = [(at(h, m), 13.0 + m / 100) for h in range(24) for m in (5, 25, 45)]
    shuffled = rows[:]
    random.Random(4).shuffle(shuffled)
    assert thin_hourly(shuffled) == thin_hourly(rows)
    assert thin_hourly(rows) == sorted(thin_hourly(rows))


def test_a_tie_keeps_the_earlier_reading():
    # 07:15 and 07:45 are each 15 minutes from the top of an hour.
    assert thin_hourly([(at(7, 45), 13.2), (at(7, 15), 13.1)]) == [(at(7, 15), 13.1)]


def test_nearest_the_top_of_the_hour_counts_the_next_hour_too():
    assert thin_hourly([(at(7, 20), 13.1), (at(7, 50), 13.2)]) == [(at(7, 50), 13.2)]


def test_hourly_input_is_unchanged():
    rows = [(at(h, 2), 13.0) for h in range(5)]
    assert thin_hourly(rows) == rows


def test_empty():
    assert thin_hourly([]) == []
