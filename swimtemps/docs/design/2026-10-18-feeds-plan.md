# swimtemps feeds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a water take its readings automatically from a venue's JSON feed instead of manual entry, with visible cleaning and quiet-feed handling.

**Architecture:** A new pure package `feeds/` (paste-block parser, `json_table` recipe, hourly thinning, cleaning, a guarded fetcher) beside `forecast/`. The worker polls feeds in its sweep and stores readings with a source and flag state. The model pairs feed readings into day-long legs. The web gains feed setup (paste the AI block, test, attest, turn on), feed requests, flag overrules, and quiet/paused states on the water page and `/healthz`.

**Tech Stack:** unchanged: Python 3.14, FastAPI, SQLAlchemy 2.1, Alembic, httpx, pandas, pytest.

**Spec:** `docs/design/2026-10-18-feeds-design.md`

## Global Constraints

- One measurement point per water. Mode is `manual` or `feed`, exclusive, switchable both ways, history kept.
- Feeds only for automated, machine-readable JSON sources; `automated: no` or `terms: forbids automated access` in a pasted block stops setup with an explanation.
- Outbound requests send user-agent `swimtemps/0.1` and nothing that identifies a person.
- Pages still never fetch: only the setup page's Test POST and the worker contact a feed.
- Readings are never destroyed. A venue's changed value becomes a correction (soft-delete + new row + `replaced_by`).
- Fetched rows pass the manual checks: -2 to 40 °C, not in the future (10-minute allowance), not before 2000.
- Sub-hourly sources are thinned to one reading per clock hour on arrival (nearest the top of the hour, earlier on a tie).
- Cleaning defaults: jump > 2.0 °C + 0.1 °C/h from the last good reading; > 2.5 °C from the 24 h median of good readings; flatline > 24 h. Comparisons use good readings only. "Matches the air temperature" added to the reason when within 1 °C of stored air.
- Quiet clocks measure time since the last **good** feed reading: banner at 12 h; `/healthz` fails at 3 days; forecast paused at 5 days. A paused feed is not polled and does not fail `/healthz`.
- Feed readings pair into legs with the feed reading closest to 24 h later within 21-27 h. Manual readings pair as now.
- Attestation checkbox ticked by default with the full statement: "I have the venue's permission to share this data, or it is publicly published." Store who, when, and the statement text.
- The `swimtemps-feed v1` block (spec §6) is the contract between the prompt and the parser.

## Review Focus

1. **Redirect to an internal address**: a feed URL on a public host that 302s to `http://169.254.169.254/` or `localhost` must be refused, not followed. Test in Task 4.
2. **Huge or slow responses**: a feed returning 200 MB, or trickling bytes, must not hang the sweep: 5 MB cap, 20 s timeout. Test in Task 4.
3. **The venue changes its JSON shape**: a missing key or a non-list must record `last_error` on that feed and leave every other feed and water untouched. Test in Task 5.
4. **Naive local times at the DST fall-back**: `2026-10-25 01:30` with no offset happens twice in London; parse with fold=0 (first occurrence) and never raise. Test in Task 2.
5. **Switching mode with feed history**: feed→manual must keep feed readings in the fit and on the chart, stop polling, and reopen add-reading; manual→feed must hide add-reading and refuse its POST. Test in Task 7.

---

### Task 1: Schema for feeds

**Files:**
- Modify: `db/models.py`
- Create: `db/migrations/versions/0002_feeds.py`
- Modify: `db/queries.py` (`active_readings`, `own_errors`, new `chart_readings`)
- Modify: `web/routes/admin.py:56` (outer join for readings with no submitter), `web/routes/readings.py` `save` (sets `source='manual'`)
- Test: `tests/db/test_schema.py`, `tests/db/test_feed_schema.py`

**Interfaces:**
- Produces:
  - `Water.feed_mode: str` (`'manual'|'feed'`, default `'manual'`).
  - `Feed` model: `id, water_id (unique FK), recipe ('json_table' or an adapter name), config JSONB, source_name, source_link, thresholds JSONB, state ('on'|'paused'|'off'), attested_by FK users, attested_at, attested_statement, last_polled_at, last_success_at, last_good_reading_at, last_error, created_by, created_at`.
  - `FeedRequest` model: `id, page_url, notes, ai_block, requested_by FK users, water_id FK nullable, status ('open'|'done'|'declined'), created_at, updated_at`.
  - `Reading`: `submitted_by` nullable; `source ('manual'|'feed')` default `'manual'`; `feed_id` FK nullable; `flagged bool` default false; `flag_reason text`; `flag_override ('accept'|'reject')` nullable; `override_by`, `override_at`; partial unique index `readings_feed_time` on (`feed_id`, `measured_at`) `WHERE feed_id IS NOT NULL AND deleted_at IS NULL`.
  - `USABLE_SQL = "(r.flag_override = 'accept' OR (r.flag_override IS NULL AND NOT r.flagged))"`. `active_readings` returns only usable, standing readings, with an added `source` column. `chart_readings(session, water_id, since) -> DataFrame[measured_at, temp_c, usable, reason]` returns all standing readings. `own_errors` scores only usable readings.

- [ ] Test: migration matches models (existing test); partial unique index refuses a second standing row for the same (feed, time) and allows one after the first is soft-deleted; `active_readings` excludes a flagged row and includes it once overruled `accept`; excludes an unflagged row overruled `reject`; `chart_readings` returns both with `usable` and the reason; the admin page lists a reading with `submitted_by = NULL`.
- [ ] Autogenerate `0002`, tidy it, run `pytest -q`, commit.

### Task 2: The paste-in block and the `json_table` recipe

**Files:**
- Create: `feeds/__init__.py`, `feeds/block.py`, `feeds/recipe.py`
- Test: `tests/feeds/test_block.py`, `tests/feeds/test_recipe.py` (fixtures from `tests/fixtures/feeds/`)

**Interfaces:**
- Produces:
  - `parse_block(text: str) -> BlockResult(config: dict, missing: list[str], unknown: list[str], invalid: list[str], stop: str | None)`. `stop` is `'not_automated'`, `'terms_forbid'`, `'no_block'` or `'bad_version'`. Required keys: `url, orientation, time_format, source_name, source_link`, plus `time_field, value_field` (rows) or `time_path, value_path` (columns).
  - `RecipeError(Exception)`.
  - `read_json_table(payload, config: dict, tz: str, default_time: time) -> RecipeResult(rows: list[tuple[datetime_utc, float]], errors: list[str])`. Raises `RecipeError` for shape problems (path missing, not a list, mismatched column lengths). Per-row problems go in `errors` and the row is skipped.
  - `resolve_path(payload, path: str)`: dotted keys plus `[n]`; `$` is the root.
  - `ADAPTERS: dict[str, Callable[[object, dict, str, time], RecipeResult]]` with `"json_table": read_json_table` registered, and `read_feed(recipe: str, payload, config, tz, default_time) -> RecipeResult` dispatching on it. An unknown recipe raises `RecipeError`. This is the spec's escape hatch: a hand-written adapter is one function registered here.
- [ ] Tests, `block`: the spec §6 example parses to exactly its fields; the same block wrapped in chat text, a ``` fence and `> ` quote marks parses identically; `unknown` and blank values count as missing; a URL with `:` survives; `orientation: sideways` is invalid; an unknown key is listed; `automated: no` gives stop `not_automated`; `terms: forbids automated access` gives `terms_forbid`; no block gives `no_block`; `swimtemps-feed v2` gives `bad_version`.
- [ ] Tests, `recipe`: `read_feed` dispatches to a test-registered adapter and refuses an unknown name.
- [ ] Tests, `recipe` (json_table):
  - The Cromhall fixture with `orientation: rows, rows_path: $, header_row: yes, time_field: 0, value_field: 1, time_format: %d-%b-%Y` gives 299 rows at 07:00 London time. `25..1` is an error naming its row.
  - The Clevedon fixture with `orientation: columns, time_path: labels, value_path: datasets[0].data, time_format: epoch_ms` gives 353 rows in UTC.
  - `unit: F` converts.
  - A naive `2026-10-25T01:30` (fall-back) parses to 00:30 UTC.
  - A missing path raises `RecipeError`.
- [ ] Implement, run `pytest tests/feeds -q`, commit.

### Task 3: Hourly thinning and cleaning

**Files:**
- Create: `feeds/thin.py`, `feeds/clean.py`
- Test: `tests/feeds/test_thin.py`, `tests/feeds/test_clean.py`

**Interfaces:**
- Produces:
  - `thin_hourly(rows: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]`: one row per clock hour, nearest the top of the hour, the earlier on a tie; input order irrelevant; output sorted.
  - `DEFAULT_THRESHOLDS = {"jump_base": 2.0, "jump_per_hour": 0.1, "day_limit": 2.5, "flatline_hours": 24, "air_margin": 1.0}`.
  - `clean(new_rows, good_history, air: pd.Series, thresholds=None) -> list[str | None]`. `new_rows` and `good_history` are sorted `(at, temp)`. Returns one reason or `None` per new row, processed in order, each accepted row joining the good history.
- [ ] Tests, `thin`: half-hourly input halves; a re-run on the same input picks the same rows; a 07:30 tie keeps 07:00's neighbour rule (earlier).
- [ ] Tests, `clean`, on the recorded Clevedon fixtures thinned to hourly, with an air series recorded alongside them (save the 2026-10-11..18 Open-Meteo hourly air for 51.435,-2.87 as `tests/fixtures/feeds/clevedon_air_2026-10-18.json` in this task):
  - zero flags before 2026-10-18 on both sensors;
  - every hour 04:00-08:00 flagged on the deep sensor and 04:00-09:00 on the shallow;
  - the deep sensor's 09:00 is not flagged;
  - at least one reason contains "air temperature";
  - a value repeated for 25 hours is flagged "flatline";
  - with no good reading in 24 h, a genuine 3 °C step is accepted once the allowance passes it.
- [ ] Implement, run, commit.

### Task 4: Guarded fetch

**Files:**
- Create: `feeds/fetch.py`
- Test: `tests/feeds/test_fetch.py`

**Interfaces:**
- Produces:
  - `FetchError(Exception)`.
  - `fetch_json(url: str, client: httpx.Client | None = None, resolver=socket.getaddrinfo) -> object`: GET with user-agent `swimtemps/0.1`, 20 s timeout, at most 3 redirects followed by hand, and each hop's host resolved and refused if any address is private, loopback, link-local, multicast or reserved. Only `http`/`https`. Streams with a 5 MB cap. Non-JSON or a non-200 response raises `FetchError` with a plain reason.
- [ ] Tests with `httpx.MockTransport` and a fake resolver: public host OK; a `localhost` or `10.0.0.5` URL refused before any request; a redirect to `http://169.254.169.254/` refused (Review Focus 1); four redirects refused; a 6 MB body refused (Review Focus 2); HTML with a 200 refused; the user-agent is exactly `swimtemps/0.1`.
- [ ] Implement, run, commit.

### Task 5: The worker polls feeds

**Files:**
- Create: `worker/feeds.py`
- Modify: `worker/cycle.py` (`sweep` polls first; `compute_water` handles the quiet/paused states), `forecast/model.py` (`build_legs` pairs feed rows day-long), `worker/snapshot.py` (payload additions)
- Test: `tests/worker/test_feeds.py`, `tests/forecast/test_model.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces:
  - `poll_feed(session, feed: Feed, now, fetch=fetch_json) -> PollResult(new: int, corrected: int, flagged: int, errors: list[str])`. Steps: fetch, recipe, the manual checks, thin, then match on standing rows by `measured_at`: insert, skip if equal, or correct. A stored standing feed row already in the same clock hour wins: a new row for that hour at a different minute is skipped, so thinning stays stable across polls. Clean new and corrected rows against good history from 48 h before the first of them, using stored air. Then update `last_polled_at`, `last_success_at`, `last_good_reading_at` and `last_error` (cleared on success, set on `FetchError`/`RecipeError`), and mark the water dirty when anything changed.
  - `poll_feeds(factory, now, fetch=fetch_json) -> int`: every feed with `state='on'` on a non-archived water in `feed_mode='feed'`, one transaction each; one failure never stops the others.
  - `build_legs` accepts an optional `source` column. Rows with `source == 'feed'` pair to the feed row closest to +24 h within 21-27 h. Other rows pair as now. Legs from both are concatenated.
  - Payload additions:
    - `readings` items gain a third element: `null` for a usable reading, the reason string for a flagged one;
    - `feed: {"source_name", "source_link", "state", "last_good_reading_at", "quiet": bool, "paused_forecast": bool}`, or `null` on a manual water.
  - Quiet/paused rules for a feed water: quiet when `now - last_good_reading_at > 12 h`. When `> 5 days` or `state == 'paused'`, no forecast run is written and the payload has `estimate_now = null` and `forecast = []`.
- [ ] Tests (fake fetch returning the fixtures; no network):
  - first poll stores thinned rows with flags;
  - a later poll carrying a different minute in an hour already stored adds nothing;
  - a second identical poll adds nothing;
  - a changed value becomes a correction with `replaced_by`;
  - a `RecipeError` sets `last_error`, and a second feed in the same sweep still polls (Review Focus 3);
  - a feed with only flagged new rows does not advance `last_good_reading_at`;
  - quiet at 13 h; forecast paused at 5 days and after `state='paused'`;
  - a paused feed is not fetched;
  - `build_legs`: hourly feed rows give legs about 24 h long, none shorter than 21 h; a 30-hour gap gives no leg across it; a mixed water gives manual legs plus feed legs.
- [ ] Implement, run the full suite, commit.

### Task 6: Water page and `/healthz`

**Files:**
- Modify: `web/templates/water.html`, `web/static/chart.js`, `web/routes/pages.py`, `web/routes/health.py`, `db/queries.py` (`health` adds quiet feeds)
- Test: `tests/web/test_pages.py`, `tests/web/test_health.py`

**Interfaces:**
- Consumes: Task 5 payload.
- Produces:
  - Feed water page:
    - "Readings from <source_name>" linking to `source_link`;
    - "Feed quiet since <local time>" when quiet;
    - "Forecast paused: no data from <source_name> since <date>" when paused by age;
    - "Feed paused by the venue" when `state='paused'`;
    - no Add-a-reading button.
  - Chart: a grey scatter series of flagged readings whose tooltip shows the reason.
  - `/healthz` fails with "feed for <water name> has had no good reading for 3 days" for an `on` feed whose `last_good_reading_at` is older than 3 days, and ignores paused feeds.
- [ ] Tests: each banner's copy; credit link; no add-reading link in feed mode; chart JSON carries the flag reasons; healthz 503 at 3 days 1 minute, 200 for the same feed paused.
- [ ] Implement, browser-check with a seeded feed water, commit.

### Task 7: Feed setup, mode switching, overrules

**Files:**
- Create: `web/routes/feeds.py`, `web/templates/feed_setup.html`, `web/templates/_feed_preview.html`
- Modify: `web/routes/readings.py` (refuse add/paste on a feed water), `web/routes/admin.py` (flag overrule; no delete/correct for feed rows), `web/templates/admin.html`, `web/templates/me.html`, `web/app.py`
- Test: `tests/web/test_feed_setup.py`

**Interfaces:**
- Consumes: Tasks 1-5. The Test fetch reuses `feeds.fetch.fetch_json`, injectable as `app.state.feed_fetch`.
- Produces, all admin-only (`can_admin`) and CSRF-protected:
  - `GET /w/{slug}/feed`: the setup page with the prompt, a copy button, the paste box and the fields.
  - `POST /w/{slug}/feed/paste`: parses the block and re-renders the fields filled in, listing missing, unknown and invalid fields. Stops with the plain explanation and a request-form link on `stop`.
  - `POST /w/{slug}/feed/test`: fetches once, runs recipe, thinning and cleaning in memory, and renders `_feed_preview.html` (counts, the first and last rows, errors, flags). Stores nothing.
  - `POST /w/{slug}/feed/enable`: requires a successful test of the same config in this request. Saves `Feed` with the attestation, sets `feed_mode='feed'` and `state='on'`, and marks the water dirty.
  - `POST /w/{slug}/feed/pause`, `/resume`, `/disable`. `/disable` sets `feed_mode='manual'`, `state='off'` and keeps the config.
  - `POST /w/{slug}/admin/readings/{id}/override` with `decision=accept|reject|clear`: stores it with who and when, and marks the water dirty.
  - `/me` lists the user's feed waters with their state and quiet age, and a "Switch to manual" button.
  - `POST /w/{slug}/readings` and `/readings/paste` return 409 "This water's readings come from <source_name>" in feed mode.
- [ ] Tests:
  - contributor 403 on every route;
  - pasting the spec example fills the fields;
  - pasting `automated: no` shows the manual-water explanation;
  - Test with a fake fetch shows the counts and stores nothing;
  - enable without a test is refused;
  - enable stores the attestation, its statement text and its timestamp, and the checkbox is rendered `checked` by default;
  - a URL resolving to a private address shows the refusal message;
  - an overrule flips usability in `active_readings`;
  - delete and correct are not offered for feed rows;
  - Review Focus 5, both directions.
- [ ] Implement, browser-check the setup flow on 375px, commit.

### Task 8: Feed requests, docs

**Files:**
- Create: `web/templates/feed_request.html`, `web/templates/feed_requests.html`
- Modify: `web/routes/feeds.py`, `README.md`, `CLAUDE.md`
- Test: `tests/web/test_feed_requests.py`

**Interfaces:**
- Produces: `GET/POST /feeds/request` (any signed-in user; page URL required, notes and pasted block optional; optional `water` query param links it); `GET /admin/feed-requests` and `POST /admin/feed-requests/{id}` with `status=done|declined` (superusers only).
- [ ] Tests: anonymous redirected to sign-in; a request stores its fields; non-superuser 403 on the list; status change.
- [ ] README: the feeds section (what qualifies, setup, the block format, quiet and paused behaviour). CLAUDE.md: add feeds to the rules list ("fetches only in the worker and the Test POST", "never destroy feed readings").
- [ ] Full suite, ruff, commit.
