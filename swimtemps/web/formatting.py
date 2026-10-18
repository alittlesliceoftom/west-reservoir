"""Jinja filters. Payload timestamps are ISO UTC; display is the water's local time."""

from datetime import datetime
from zoneinfo import ZoneInfo


def _parse(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def localtime(value, tz: str, fmt: str = "%a {day} %b %H:%M") -> str:
    """`{day}` is the unpadded day of the month."""
    local = _parse(value).astimezone(ZoneInfo(tz))
    return local.strftime(fmt.replace("{day}", str(local.day)))


def ago(value, now: datetime) -> str:
    seconds = (now - _parse(value)).total_seconds()
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    return f"{hours // 24} days ago"


def temp(value) -> str:
    return f"{abs(value) if round(value, 1) == 0 else value:.1f} °C"
