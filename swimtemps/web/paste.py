"""
Parse pasted readings, one per line: a date, an optional time, a temperature.

Dates are day-first (`18/10/2026`, `18.10.26`, `18-10`) or ISO
(`2026-10-18`, `2026-10-18T07:00`). A missing year is this year, or last
year if that would put the date in the future. A missing time is the water's
reading time. Separators may be commas, semicolons, tabs or spaces, and the
temperature may carry `°C`, `C` or a decimal comma.

Nothing is guessed: a line that does not read cleanly is reported with its
line number and why. Lines with no digits at all (headers) are skipped.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

MIN_TEMP, MAX_TEMP = -2.0, 40.0

ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2}))?")
# The lookaheads stop a match ending inside a longer number or date: without
# them "18/10/2026" could backtrack to "18/10" and leave "/2026" behind.
DAY_FIRST = re.compile(r"^(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{4}|\d{2}))?(?!\d)(?![/.-]\d)")
CLOCK = re.compile(r"^[\s,;]*(\d{1,2}):(\d{2})\s*(am|pm)?(?![\d])", re.IGNORECASE)
TEMP = re.compile(r"^[\s,;]*(-?\d+(?:[.,]\d+)?)\s*(?:°\s*C?|C)?[\s,;]*$", re.IGNORECASE)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class ParsedRow:
    line: int
    raw: str
    measured_at: datetime | None = None   # UTC
    temp: float | None = None
    error: str | None = None


def _clock(match) -> tuple[int, int] | None:
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem.lower() == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _parse_line(text: str, tz: ZoneInfo, default_time: time, today: date, now: datetime) -> ParsedRow | str:
    rest = text.strip()
    explicit_time = None
    if len(NUMBER.findall(rest)) < 2:
        return "no date (expected day/month/year first)"

    iso = ISO_DATE.match(rest)
    day_first = None if iso else DAY_FIRST.match(rest)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
        if iso.group(4):
            explicit_time = (int(iso.group(4)), int(iso.group(5)))
        rest = rest[iso.end():]
    elif day_first:
        day, month = int(day_first.group(1)), int(day_first.group(2))
        year_text = day_first.group(3)
        year = None if year_text is None else int(year_text) + (2000 if len(year_text) == 2 else 0)
        rest = rest[day_first.end():]
    else:
        return "no date (expected day/month/year first)"

    if explicit_time is None:
        clock = CLOCK.match(rest)
        if clock:
            explicit_time = _clock(clock)
            if explicit_time is None:
                return "not a real time"
            rest = rest[clock.end():]
    elif explicit_time[0] > 23 or explicit_time[1] > 59:
        return "not a real time"

    try:
        if year is None:
            year = today.year
            if date(year, month, day) > today:
                year -= 1
        day_value = date(year, month, day)
    except ValueError:
        return "not a real date"

    numbers = NUMBER.findall(rest)
    if not numbers:
        return "no temperature"
    temp_match = TEMP.match(rest)
    if not temp_match:
        return "more than one temperature" if len(numbers) > 1 else "could not read the temperature"
    temp = float(temp_match.group(1).replace(",", "."))
    if not MIN_TEMP <= temp <= MAX_TEMP:
        return "temperature must be between -2 and 40 °C"

    hour, minute = explicit_time or (default_time.hour, default_time.minute)
    local = datetime.combine(day_value, time(hour, minute), tzinfo=tz)
    measured_at = local.astimezone(timezone.utc)
    if measured_at > now:
        return "in the future"
    return ParsedRow(line=0, raw=text, measured_at=measured_at, temp=temp)


def parse_paste(text: str, tz: str, default_time: time, today: date, now: datetime | None = None) -> list[ParsedRow]:
    """
    One ParsedRow per non-blank, non-header line, in order. Rows that could
    not be read have `error` set and no timestamp or temperature.
    """
    zone = ZoneInfo(tz)
    now = now or datetime.combine(today, time(23, 59, 59), tzinfo=zone).astimezone(timezone.utc)
    rows = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or not re.search(r"\d", raw):
            continue
        parsed = _parse_line(raw, zone, default_time, today, now)
        if isinstance(parsed, str):
            rows.append(ParsedRow(line=number, raw=raw.strip(), error=parsed))
        else:
            parsed.line, parsed.raw = number, raw.strip()
            rows.append(parsed)
    return rows
