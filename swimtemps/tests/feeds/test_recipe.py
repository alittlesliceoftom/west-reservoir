import json
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from feeds import recipe
from feeds.recipe import ADAPTERS, RecipeError, RecipeResult, read_feed, read_json_table, resolve_path

FIXTURES = Path(__file__).parents[1] / "fixtures" / "feeds"
LONDON = "Europe/London"
SEVEN = time(7, 0)

CROMHALL = {"orientation": "rows", "rows_path": "$", "header_row": True, "time_field": 0, "value_field": 1,
            "time_format": "%d-%b-%Y"}
CLEVEDON = {"orientation": "columns", "time_path": "labels", "value_path": "datasets[0].data",
            "time_format": "epoch_ms"}


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_cromhall_rows_give_299_readings_at_seven_local():
    payload = load("cromhall_lake_2026-10-18.json")
    result = read_json_table(payload, CROMHALL, LONDON, SEVEN)
    assert len(result.rows) == 299
    assert all(at.tzinfo == timezone.utc for at, _ in result.rows)
    assert {at.astimezone(ZoneInfo(LONDON)).time() for at, _ in result.rows} == {SEVEN}
    assert result.rows[0] == (datetime(2024, 9, 12, 6, 0, tzinfo=timezone.utc), 10.6)
    row = next(i for i, r in enumerate(payload) if r[1] == "25..1") + 1
    assert len(result.errors) == 1
    assert "25..1" in result.errors[0] and f"row {row}" in result.errors[0]


def test_clevedon_columns_give_353_readings_in_utc():
    payload = load("clevedon_266367_168h_2026-10-18.json")
    result = read_json_table(payload, CLEVEDON, LONDON, SEVEN)
    assert len(result.rows) == 353 and result.errors == []
    first = payload["labels"][0]
    assert result.rows[0] == (datetime.fromtimestamp(first / 1000, timezone.utc), payload["datasets"][0]["data"][0])


def test_fahrenheit_is_converted():
    payload = [{"t": "2026-10-18T07:00:00Z", "v": 59}]
    config = {"orientation": "rows", "time_field": "t", "value_field": "v", "time_format": "iso", "unit": "F"}
    assert read_json_table(payload, config, LONDON, SEVEN).rows[0][1] == pytest.approx(15.0)


def test_a_naive_time_at_the_dst_fall_back_takes_the_first_occurrence():
    payload = [{"t": "2026-10-25T01:30", "v": 12.0}]
    config = {"orientation": "rows", "time_field": "t", "value_field": "v", "time_format": "iso"}
    at, _ = read_json_table(payload, config, LONDON, SEVEN).rows[0]
    assert at == datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc)


def test_an_iso_date_alone_takes_the_reading_time():
    payload = [{"t": "2026-10-18", "v": 12.0}]
    config = {"orientation": "rows", "time_field": "t", "value_field": "v", "time_format": "iso"}
    assert read_json_table(payload, config, LONDON, SEVEN).rows[0][0] == datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)


def test_epoch_seconds_and_numeric_strings():
    payload = {"times": ["1792281600"], "temps": ["13.4"]}
    config = {"orientation": "columns", "time_path": "times", "value_path": "temps", "time_format": "epoch_s"}
    assert read_json_table(payload, config, LONDON, SEVEN).rows == [(datetime(2026, 10, 18, 0, 0, tzinfo=timezone.utc), 13.4)]


def test_rows_with_no_value_or_a_bad_time_are_errors_and_skipped():
    payload = [{"t": "2026-10-18T07:00Z", "v": None}, {"t": "yesterday", "v": 12}, {"t": "2026-10-18T08:00Z", "v": True},
               {"t": "2026-10-18T09:00Z", "v": 12.5}]
    config = {"orientation": "rows", "time_field": "t", "value_field": "v", "time_format": "iso"}
    result = read_json_table(payload, config, LONDON, SEVEN)
    assert len(result.rows) == 1 and len(result.errors) == 3
    assert result.errors[1].startswith("row 2")


def test_a_missing_path_raises():
    with pytest.raises(RecipeError, match="datasets"):
        read_json_table({"labels": []}, CLEVEDON, LONDON, SEVEN)


def test_a_path_that_is_not_a_list_raises():
    with pytest.raises(RecipeError, match="list"):
        read_json_table({"labels": 5, "datasets": [{"data": []}]}, CLEVEDON, LONDON, SEVEN)


def test_columns_of_different_lengths_raise():
    with pytest.raises(RecipeError, match="length"):
        read_json_table({"labels": [1, 2], "datasets": [{"data": [3.0]}]}, CLEVEDON, LONDON, SEVEN)


def test_resolve_path():
    payload = {"a": {"b": [[1, 2], [3, {"c": 4}]]}}
    assert resolve_path(payload, "$") is payload
    assert resolve_path(payload, "a.b[1][1].c") == 4
    assert resolve_path(payload, "$.a.b[0]") == [1, 2]
    with pytest.raises(RecipeError):
        resolve_path(payload, "a.b[7]")


def test_read_feed_dispatches_to_a_registered_adapter(monkeypatch):
    def fake(payload, config, tz, default_time):
        return RecipeResult(rows=[(datetime(2026, 10, 18, tzinfo=timezone.utc), float(payload["x"]))], errors=[])

    monkeypatch.setitem(ADAPTERS, "fake_lake", fake)
    assert read_feed("fake_lake", {"x": 3}, {}, LONDON, SEVEN).rows[0][1] == 3.0
    assert recipe.ADAPTERS["json_table"] is read_json_table


def test_read_feed_refuses_an_unknown_recipe():
    with pytest.raises(RecipeError, match="no_such"):
        read_feed("no_such", {}, {}, LONDON, SEVEN)


def test_a_numeric_field_also_finds_a_record_key():
    payload = [{"0": "2026-10-18T07:00Z", "1": 12.5}]
    config = {"orientation": "rows", "time_field": 0, "value_field": 1, "time_format": "iso"}
    assert len(read_json_table(payload, config, LONDON, SEVEN).rows) == 1
