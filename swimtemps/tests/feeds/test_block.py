from feeds.block import parse_block

EXAMPLE = """swimtemps-feed v1
url: https://example-lake.org/wp-json/lake/v1/temps?hours=168
orientation: columns
time_path: labels
value_path: datasets[0].data
time_format: epoch_ms
timezone: Europe/London
unit: C
source_name: Example Lake
source_link: https://example-lake.org/live-temperature/
automated: yes
terms: unknown
notes: half-hourly sensor; the API keeps 7 days
end"""

EXPECTED = {
    "url": "https://example-lake.org/wp-json/lake/v1/temps?hours=168",
    "orientation": "columns",
    "time_path": "labels",
    "value_path": "datasets[0].data",
    "time_format": "epoch_ms",
    "timezone": "Europe/London",
    "unit": "C",
    "source_name": "Example Lake",
    "source_link": "https://example-lake.org/live-temperature/",
    "automated": True,
    "notes": "half-hourly sensor; the API keeps 7 days",
}


def test_the_prompts_own_example_parses_to_exactly_its_fields():
    result = parse_block(EXAMPLE)
    assert result.config == EXPECTED
    assert (result.missing, result.unknown, result.invalid, result.stop) == ([], [], [], None)


def test_a_reply_wrapped_in_chat_a_fence_and_quote_marks_parses_the_same():
    quoted = "\n".join("> " + line for line in EXAMPLE.splitlines())
    reply = f"Sure! I found the API behind the chart.\n\n> ```\n{quoted}\n> ```\n\nLet me know if you need anything else."
    assert parse_block(reply).config == EXPECTED


def test_a_fenced_reply_parses_the_same():
    assert parse_block(f"Here you go:\n```text\n{EXAMPLE}\n```\n").config == EXPECTED


def test_keys_are_case_insensitive():
    assert parse_block(EXAMPLE.replace("source_name:", "Source_Name:")).config["source_name"] == "Example Lake"


def test_unknown_and_blank_values_count_as_missing():
    text = EXAMPLE.replace("time_format: epoch_ms", "time_format: unknown").replace("source_name: Example Lake", "source_name:")
    result = parse_block(text)
    assert "time_format" not in result.config and "source_name" not in result.config
    assert result.missing == ["time_format", "source_name"]


def test_a_value_may_contain_colons():
    text = EXAMPLE.replace("url: https://example-lake.org/wp-json/lake/v1/temps?hours=168",
                           "url: https://example-lake.org:8443/temps?at=12:00")
    assert parse_block(text).config["url"] == "https://example-lake.org:8443/temps?at=12:00"


def test_an_orientation_that_is_neither_rows_nor_columns_is_invalid():
    result = parse_block(EXAMPLE.replace("orientation: columns", "orientation: sideways"))
    assert any("orientation" in message and "sideways" in message for message in result.invalid)
    assert "orientation" not in result.config


def test_an_unrecognised_key_is_listed():
    result = parse_block(EXAMPLE.replace("notes:", "sensor_depth: 1m\nnotes:"))
    assert result.unknown == ["sensor_depth"]


def test_lines_that_are_not_key_value_are_ignored():
    result = parse_block(EXAMPLE.replace("unit: C", "unit: C\nThe rows key is not needed here.\n"))
    assert result.config == EXPECTED and result.unknown == []


def test_rows_keys_are_required_for_rows():
    text = """swimtemps-feed v1
url: https://script.example.com/exec
orientation: rows
rows_path: $
header_row: yes
time_field: 0
time_format: %d-%b-%Y
source_name: The Lake
source_link: https://lake.example/temperature/
end"""
    result = parse_block(text)
    assert result.missing == ["value_field"]
    assert result.config["header_row"] is True and result.config["time_field"] == 0


def test_columns_keys_are_required_for_columns():
    result = parse_block(EXAMPLE.replace("value_path: datasets[0].data\n", ""))
    assert result.missing == ["value_path"]


def test_a_source_kept_up_to_date_by_hand_is_accepted():
    result = parse_block(EXAMPLE.replace("automated: yes", "automated: no"))
    assert result.stop is None and result.config["automated"] is False


def test_terms_that_forbid_automated_access_stop_setup():
    assert parse_block(EXAMPLE.replace("terms: unknown", "terms: forbids automated access")).stop == "terms_forbid"


def test_no_block_is_rejected():
    assert parse_block("I could not find an API on that page, sorry.").stop == "no_block"


def test_another_version_is_rejected():
    assert parse_block(EXAMPLE.replace("swimtemps-feed v1", "swimtemps-feed v2")).stop == "bad_version"


def test_bad_values_are_invalid():
    text = (EXAMPLE.replace("unit: C", "unit: K").replace("timezone: Europe/London", "timezone: Mars/Olympus")
            .replace("time_format: epoch_ms", "time_format: sometimes")
            .replace("url: https://example-lake.org/wp-json/lake/v1/temps?hours=168", "url: ftp://example-lake.org/x"))
    invalid = " ".join(parse_block(text).invalid)
    for word in ("unit", "timezone", "time_format", "url"):
        assert word in invalid


def test_the_prompts_example_is_the_example_tested_here():
    from feeds.block import PROMPT
    example = PROMPT[PROMPT.index("would give:"):]
    assert parse_block(example).config == {k: v for k, v in EXPECTED.items() if k != "automated"}


def test_the_prompt_no_longer_asks_whether_a_sensor_produces_the_readings():
    from feeds.block import PROMPT
    assert "automated:" not in PROMPT
