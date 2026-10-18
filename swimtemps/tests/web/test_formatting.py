from datetime import datetime, timezone

from web.formatting import ago, localtime, temp

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def test_localtime_converts_to_the_waters_zone():
    assert localtime("2026-10-18T06:00:00+00:00", "Europe/London") == "Sun 18 Oct 07:00"
    assert localtime("2026-10-24T06:00:00+00:00", "Europe/London", "%a %H:%M") == "Sat 07:00"
    assert localtime("2026-10-26T07:00:00+00:00", "Europe/London", "%a %H:%M") == "Mon 07:00"


def test_localtime_does_not_zero_pad_the_day():
    assert localtime("2026-11-03T07:00:00+00:00", "Europe/London") == "Tue 3 Nov 07:00"


def test_ago_reads_like_speech():
    assert ago("2026-10-18T11:59:40+00:00", NOW) == "just now"
    assert ago("2026-10-18T11:46:00+00:00", NOW) == "14 min ago"
    assert ago("2026-10-18T11:00:00+00:00", NOW) == "1 hour ago"
    assert ago("2026-10-18T07:00:00+00:00", NOW) == "5 hours ago"
    assert ago("2026-10-16T12:00:00+00:00", NOW) == "2 days ago"


def test_temp_has_one_decimal_and_the_unit():
    assert temp(13.84) == "13.8 °C"
    assert temp(-0.04) == "0.0 °C"
