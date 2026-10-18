# CLAUDE.md

swimtemps: a multi-water water-temperature site. See README.md for the
architecture, local setup and launch checklist, and the design spec it links.

## Rules that are easy to break

- **The web app never computes a forecast and never calls Open-Meteo on a page
  load.** Pages read `snapshots`. The create wizard's `/api/geocode` and
  `/api/point` are the only web-side weather calls. A test enforces this.
- **Readings are never destroyed.** Delete sets `deleted_at`; a correction also
  sets `replaced_by`. Every read filters `deleted_at IS NULL`. That holds for
  feed readings too: a venue's changed value is a correction. An admin's
  correction or deletion of a feed reading is held: polls skip feed times a
  person deleted and standing feed rows a person submitted.
- **Only usable readings reach the model.** Fits, estimates and scoring use
  `queries.active_readings` (filtered by `USABLE_SQL`); the chart uses
  `chart_readings`, which keeps flagged ones with their reason.
- **Feeds are fetched only by the worker and the feed setup's Test and Turn-on
  POSTs**, always through `feeds.fetch.fetch_json` (public addresses only,
  5 MB, 20 s, user-agent `swimtemps/0.1` and nothing that identifies anyone).
  Pages never fetch a feed.
- **Timestamps are UTC everywhere they are stored**; weather is fetched with
  `timezone=UTC`. A water's `timezone` is for display and reading-time maths.
- **The worker clears `needs_recompute` before it reads data**, and sets it back
  on failure. Do not move the clear to after the compute: a reading saved
  mid-compute would be lost.
- **Model constants live in `forecast/params.json`** with their provenance.
  Change them by rerunning the replay (`pytest -m slow`), not by hand.
- No emojis in the UI. Dates on pages always say which day.
- Docstrings describe what the code does now: no history, no issue numbers.

## Commands

```bash
.venv/bin/pytest                 # full suite, own Postgres, no network
.venv/bin/pytest -m slow         # cold-start replay over West Res
.venv/bin/ruff check .
.venv/bin/alembic revision --autogenerate -m "..."   # then read and tidy it
```

`tests/db/test_schema.py::test_the_migration_matches_the_models` fails if a
model changes without a migration.

## Layout

- `forecast/` pure model (no DB, no web): `physics`, `model` (anchors, targets,
  legs, staged fit), `band`, `openmeteo`, `replay`
- `feeds/` pure feed handling: `block` (the `swimtemps-feed v1` format and its
  AI prompt), `recipe` (`json_table`, `ADAPTERS`), `thin`, `clean`, `fetch`
- `db/` `models`, `queries` (raw SQL via `text()`), `session`, `cli`, `migrations/`
- `worker/` `cycle` (refresh, compute, sweep, drain), `feeds` (polling), `snapshot` (page payload), `main`, `seed`
- `web/` `app` factory, `auth` (Supabase PKCE, JWT, CSRF middleware), `routes/`, `templates/`, `static/`
