from datetime import date, datetime, time, timezone

import pytest

from web.paste import parse_paste

TZ = "Europe/London"
TODAY = date(2026, 10, 18)


def one(line, **kw):
    rows = parse_paste(line, TZ, time(7, 0), TODAY, **kw)
    assert len(rows) == 1
    return rows[0]


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


@pytest.mark.parametrize("line, at, temp", [
    ("18/10/2026 13.5", utc(2026, 10, 18, 6, 0), 13.5),          # default time, BST
    ("18/10/2026, 13.5", utc(2026, 10, 18, 6, 0), 13.5),
    ("18/10/2026\t13.5", utc(2026, 10, 18, 6, 0), 13.5),
    ("18/10/2026;13.5", utc(2026, 10, 18, 6, 0), 13.5),
    ("18/10 08:15, 13.5", utc(2026, 10, 18, 7, 15), 13.5),        # no year: this year
    ("18/10/26 13.5", utc(2026, 10, 18, 6, 0), 13.5),            # two-digit year
    ("18.10.2026 13.5", utc(2026, 10, 18, 6, 0), 13.5),
    ("18-10-2026 13.5°C", utc(2026, 10, 18, 6, 0), 13.5),
    ("18/10/2026 13.5 C", utc(2026, 10, 18, 6, 0), 13.5),
    ("18/10/2026 13,5", utc(2026, 10, 18, 6, 0), 13.5),           # decimal comma
    ("2026-10-18 13.5", utc(2026, 10, 18, 6, 0), 13.5),           # ISO
    ("2026-10-18T19:30 13.5", utc(2026, 10, 18, 18, 30), 13.5),
    ("1/2/2026 4", utc(2026, 2, 1, 7, 0), 4.0),                  # GMT, day first
    ("01/02/2026 7:05pm 4.2", utc(2026, 2, 1, 19, 5), 4.2),
    ("  18/10/2026   -0.5  ", utc(2026, 10, 18, 6, 0), -0.5),
])
def test_accepted_formats(line, at, temp):
    row = one(line)
    assert row.error is None, row.error
    assert (row.measured_at, row.temp) == (at, temp)


def test_a_date_without_a_year_that_would_be_in_the_future_is_last_year():
    assert one("25/12 6.0").measured_at.year == 2025


def test_blank_lines_and_headers_are_skipped_and_line_numbers_kept():
    rows = parse_paste("Date,Temp\n\n17/10/2026 13.9\n18/10/2026 13.5\n", TZ, time(7, 0), TODAY)
    assert [r.line for r in rows] == [3, 4]


@pytest.mark.parametrize("line, message", [
    ("31/02/2026 13.5", "not a real date"),
    ("18/10/2026", "no temperature"),
    ("18/10/2026 13.5 14.0", "more than one temperature"),
    ("13.5", "no date"),
    ("18/10/2026 45", "between -2 and 40"),
    ("18/10/2026 -3", "between -2 and 40"),
    ("20/10/2026 13.5", "in the future"),
    ("18/10/2026 25:00 13.5", "not a real time"),
    ("10/18/2026 13.5", "not a real date"),  # month-first is not guessed at
])
def test_rejected_rows_say_why(line, message):
    row = one(line)
    assert row.measured_at is None or row.error is not None
    assert message in row.error


def test_a_time_later_today_is_in_the_future():
    row = one("18/10/2026 23:00 13.5", now=utc(2026, 10, 18, 12, 0))
    assert "in the future" in row.error


def test_a_reading_time_in_the_spring_gap_is_moved_forward_not_lost():
    # 01:30 on 29 March 2026 does not exist in London.
    row = parse_paste("29/03/2026 13.5", TZ, time(1, 30), TODAY)[0]
    assert row.error is None
    assert row.measured_at == utc(2026, 3, 29, 1, 30)
