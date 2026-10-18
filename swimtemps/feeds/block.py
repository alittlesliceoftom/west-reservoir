"""
The `swimtemps-feed v1` block: the fixed format the setup page's AI prompt
asks for, and the parser that reads it back out of a pasted reply.

The parser takes the lines from `swimtemps-feed v1` to the next `end`,
ignoring chat, code fences and `>` quote marks around them, and reads
`key: value` lines. Blank values and `unknown` count as not given.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

VERSION = "v1"
RECIPE_KEYS = (
    "url", "orientation", "rows_path", "header_row", "time_field", "value_field", "time_path", "value_path",
    "time_format", "timezone", "unit", "source_name", "source_link",
)
KEYS = (*RECIPE_KEYS, "automated", "terms", "notes")
REQUIRED = ("url", "orientation", "time_format", "source_name", "source_link")
REQUIRED_FOR = {"rows": ("time_field", "value_field"), "columns": ("time_path", "value_path")}
TIME_FORMATS = ("epoch_ms", "epoch_s", "iso")
TERMS = ("ok", "forbids automated access")

_HEADER = re.compile(r"^swimtemps-feed\s+(\S+)$", re.IGNORECASE)
_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
_QUOTE = re.compile(r"^(\s*>)+\s?")


@dataclass
class BlockResult:
    config: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    # 'not_automated' | 'terms_forbid' | 'no_block' | 'bad_version', or None
    stop: str | None = None


def _clean_line(line: str) -> str:
    return _QUOTE.sub("", line).strip().strip("`").strip()


def _yes_no(value: str) -> bool | None:
    return {"yes": True, "no": False}.get(value.lower())


def _is_web_url(value: str) -> bool:
    parts = urlparse(value)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _field(value: str) -> int | str:
    """A record key, or a position in an array record."""
    return int(value) if value.isdigit() else value


def check_value(key: str, value: str):
    """The value normalised for the config, or raise ValueError with a plain reason."""
    if key in ("url", "source_link"):
        if not _is_web_url(value):
            raise ValueError(f"{key}: '{value}' is not a web address starting http:// or https://")
        return value
    if key == "orientation":
        if value.lower() not in REQUIRED_FOR:
            raise ValueError(f"orientation: '{value}' is not rows or columns")
        return value.lower()
    if key in ("header_row", "automated"):
        answer = _yes_no(value)
        if answer is None:
            raise ValueError(f"{key}: '{value}' is not yes or no")
        return answer
    if key in ("time_field", "value_field"):
        return _field(value)
    if key == "time_format":
        if value.lower() in TIME_FORMATS:
            return value.lower()
        if "%" not in value:
            raise ValueError(f"time_format: '{value}' is not epoch_ms, epoch_s, iso or a pattern like %d-%b-%Y")
        return value
    if key == "timezone":
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"timezone: '{value}' is not a timezone name like Europe/London") from None
        return value
    if key == "unit":
        unit = value.upper().lstrip("°")
        if unit not in ("C", "F"):
            raise ValueError(f"unit: '{value}' is not C or F")
        return unit
    if key == "terms":
        if value.lower() not in TERMS:
            raise ValueError(f"terms: '{value}' is not ok, forbids automated access, or unknown")
        return value.lower()
    return value


def missing_keys(config: dict) -> list[str]:
    """Required keys not in `config`, including those its orientation needs."""
    required = list(REQUIRED) + list(REQUIRED_FOR.get(config.get("orientation"), ()))
    return [key for key in required if key not in config]


def parse_block(text: str) -> BlockResult:
    lines = [_clean_line(line) for line in text.splitlines()]
    start = next((i for i, line in enumerate(lines) if _HEADER.match(line)), None)
    if start is None:
        return BlockResult(stop="no_block")
    if _HEADER.match(lines[start]).group(1).lower() != VERSION:
        return BlockResult(stop="bad_version")

    result = BlockResult()
    for line in lines[start + 1:]:
        if line.lower() == "end":
            break
        match = _LINE.match(line)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2).strip()
        if key not in KEYS:
            result.unknown.append(key)
            continue
        if not value or value.lower() == "unknown":
            continue
        try:
            result.config[key] = check_value(key, value)
        except ValueError as error:
            result.invalid.append(str(error))

    result.missing = missing_keys(result.config)
    if result.config.get("automated") is False:
        result.stop = "not_automated"
    elif result.config.get("terms") == "forbids automated access":
        result.stop = "terms_forbid"
    return result
