# CLAUDE.md

swimtemps: a multi-water water-temperature site. See README.md for the
architecture, local setup and launch checklist, and the design spec it links.

## Rules that are easy to break

- **The web app never computes a forecast and never calls Open-Meteo on a page
  load.** Pages read `snapshots`. The create wizard's `/api/geocode` and
  `/api/point` are the only web-side weather calls. A test enforces this.
- **Readings are never destroyed.** Delete sets `deleted_at`; a correction also
  sets `replaced_by`. Every read filters `deleted_at IS NULL`.
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
- `db/` `models`, `queries` (raw SQL via `text()`), `session`, `cli`, `migrations/`
- `worker/` `cycle` (refresh, compute, sweep, drain), `snapshot` (page payload), `main`, `seed`
- `web/` `app` factory, `auth` (Supabase PKCE, JWT, CSRF middleware), `routes/`, `templates/`, `static/`
