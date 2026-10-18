"""
Recipes: turn a feed's JSON into (time in UTC, °C) rows.

`json_table` reads the two shapes venues publish: `rows` (a list of records
or arrays) and `columns` (parallel lists of times and values). A source it
cannot express gets a small adapter function registered in `ADAPTERS`
under its own name, with the same signature.

Shape problems (a path that is missing, not a list, or columns of different
lengths) raise `RecipeError`: the whole fetch is unusable. A bad row is an
entry in `errors` and is skipped.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

_TOKEN = re.compile(r"\.?([^.\[\]]+)|\[(\d+)\]")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_DIRECTIVES = ("%H", "%I", "%M", "%S", "%X", "%c", "%p", "%T", "%R", "%s")


class RecipeError(Exception):
    pass


@dataclass
class RecipeResult:
    rows: list[tuple[datetime, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def resolve_path(payload, path: str):
    """Follow a dotted path with `[n]` indexes, such as `datasets[0].data`. `$` is the root."""
    rest = path.strip()
    rest = rest[1:] if rest.startswith("$") else rest
    node, position = payload, 0
    while position < len(rest):
        match = _TOKEN.match(rest, position)
        if not match:
            raise RecipeError(f"'{path}' is not a path like labels or datasets[0].data")
        key, index = match.group(1), match.group(2)
        try:
            node = node[int(index)] if index is not None else node[key]
        except (KeyError, IndexError, TypeError):
            raise RecipeError(f"'{path}' was not found in the feed: no '{key if index is None else f'[{index}]'}'") from None
        position = match.end()
    return node


def _list_at(payload, path: str) -> list:
    node = resolve_path(payload, path)
    if not isinstance(node, list):
        raise RecipeError(f"'{path}' is not a list in the feed")
    return node


def _number(value) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError
    number = float(value.strip()) if isinstance(value, str) else float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def _localise(naive: datetime, tz: str) -> datetime:
    # fold=0: an ambiguous time at the autumn change is its first occurrence.
    return naive.replace(tzinfo=ZoneInfo(tz), fold=0).astimezone(timezone.utc)


def parse_time(value, time_format: str, tz: str, default_time: time) -> datetime:
    """One feed time as UTC. Raises ValueError when it cannot be read."""
    if time_format in ("epoch_ms", "epoch_s"):
        seconds = _number(value) / (1000 if time_format == "epoch_ms" else 1)
        return datetime.fromtimestamp(seconds, timezone.utc)
    if not isinstance(value, str):
        raise ValueError
    text = value.strip()
    if time_format == "iso":
        if _DATE_ONLY.match(text):
            return _localise(datetime.combine(date.fromisoformat(text), default_time), tz)
        parsed = datetime.fromisoformat(text)
    else:
        parsed = datetime.strptime(text, time_format)
        if not any(d in time_format for d in _TIME_DIRECTIVES):
            parsed = datetime.combine(parsed.date(), default_time)
    if parsed.tzinfo is None:
        return _localise(parsed, tz)
    return parsed.astimezone(timezone.utc)


def _pick(record, key):
    if isinstance(record, dict) and isinstance(key, int):
        key = str(key)
    try:
        return record[key]
    except (KeyError, IndexError, TypeError):
        return None


def read_json_table(payload, config: dict, tz: str, default_time: time) -> RecipeResult:
    time_format = config["time_format"]
    tz = config.get("timezone") or tz
    fahrenheit = config.get("unit", "C") == "F"

    if config["orientation"] == "rows":
        records = _list_at(payload, config.get("rows_path") or "$")
        first = 1 if config.get("header_row") else 0
        pairs = [(i + 1, _pick(r, config["time_field"]), _pick(r, config["value_field"]))
                 for i, r in enumerate(records) if i >= first]
    else:
        times, values = _list_at(payload, config["time_path"]), _list_at(payload, config["value_path"])
        if len(times) != len(values):
            raise RecipeError(
                f"'{config['time_path']}' has {len(times)} times but '{config['value_path']}' has "
                f"{len(values)} temperatures: the lists must be the same length")
        pairs = [(i + 1, t, v) for i, (t, v) in enumerate(zip(times, values))]

    result = RecipeResult()
    for row, raw_time, raw_value in pairs:
        try:
            at = parse_time(raw_time, time_format, tz, default_time)
        except (ValueError, OverflowError, OSError):
            result.errors.append(f"row {row}: '{raw_time}' is not a time in the format {time_format}")
            continue
        try:
            value = _number(raw_value)
        except (ValueError, TypeError):
            result.errors.append(f"row {row}: '{raw_value}' is not a temperature")
            continue
        result.rows.append((at, (value - 32) * 5 / 9 if fahrenheit else value))
    return result


Adapter = Callable[[object, dict, str, time], RecipeResult]
ADAPTERS: dict[str, Adapter] = {"json_table": read_json_table}


def read_feed(recipe: str, payload, config: dict, tz: str, default_time: time) -> RecipeResult:
    """Run the named recipe: `json_table`, or a registered adapter."""
    adapter = ADAPTERS.get(recipe)
    if adapter is None:
        raise RecipeError(f"There is no recipe called '{recipe}'.")
    return adapter(payload, config, tz, default_time)
