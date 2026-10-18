# swimtemps platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-hosted site where anyone can add a body of water, enter readings, and see a 5-day water-temperature forecast with an 80% band.

**Architecture:** Pure `forecast/` package (physics, staged fit, band, Open-Meteo client) with no DB or web; a sync FastAPI + Jinja + HTMX `web/` that only reads snapshots and writes readings; a `worker/` loop that refreshes weather, fits, forecasts and writes one `snapshots` row per water. Postgres (Supabase in prod) via SQLAlchemy 2.0 + Alembic, raw `text()` SQL for analytics.

**Tech Stack:** Python 3.14, FastAPI 0.143, SQLAlchemy 2.1, psycopg 3.3, Alembic 1.20, Jinja2, HTMX 2, ECharts 5, Leaflet 1.9, scipy/numpy/pandas, PyJWT, supabase-auth 2.33, pytest, ruff. Docker Compose + Caddy for deploy.

**Spec:** `docs/design/2026-10-18-platform-design.md`

## Global Constraints

- Lives in `swimtemps/` on this branch, self-contained (own requirements, tests, Dockerfile, `.github/`), so `git subtree split --prefix=swimtemps` makes it the new repo. West Res code is untouched.
- No page load calls a weather API or runs the optimiser. The wizard's geocode/point-preview endpoints are the only web-side Open-Meteo calls.
- All temperatures stored in Celsius. All timestamps stored UTC (`timestamptz`); Open-Meteo is requested with `timezone=UTC` so `weather_hourly` never collides at DST fall-back. A water's IANA timezone is stored separately and used for display and for reading-time arithmetic.
- Every read of `readings` filters `deleted_at IS NULL`. Corrections soft-delete and set `replaced_by`.
- Hard reading limit -2 to 40 °C; soft warning beyond 3 °C from the current estimate.
- Rate limit: 30 readings per user per water per rolling hour; a paste counts per row; a paste that would exceed it is refused whole with the remaining allowance stated.
- Horizon 5 days. Forecast points at the water's `default_reading_time` on each of the next 5 local days.
- Depth prior s: shallow 2.0, lake 1.0, deep 0.5, unknown 1.0. Fit method multipliers prior x1.5, scale x1.25, full x1.0; river/harbour x1.5. Own errors replace the base band once a water has >= 20 scored points.
- Footer credits "Weather data by Open-Meteo.com". No emojis in UI. Mobile-first.
- POST-only state changes, SameSite=Lax httpOnly cookies, CSRF double-submit token in forms and the HTMX header.
- Ruff `F`,`E9` only. Docstrings describe what the code does now: no history narration, no issue numbers.

## Review Focus

1. **Dirty flag race** — a reading saved while the worker computes that water must still trigger a recompute. Worker clears `needs_recompute` *before* reading data and re-sets it on failure. Test in Task 5.
2. **Soft warning with no estimate** — a brand-new water has no estimate; the soft check must be skipped, not crash. Test in Task 7.
3. **Paste parser ambiguity** — `dd/mm/yyyy temp`, `dd/mm hh:mm, temp`, `yyyy-mm-dd temp`, comma/tab/space separators, `°C`/`C` suffix; unparseable rows reported with line numbers, never guessed; duplicates (same water, same local hour, in DB or within the paste) flagged. Test in Task 7.
4. **Stale anchor** — latest reading weeks old: estimate-now and forecast still simulate across stored weather; if weather coverage has a hole, the estimate is withheld (shown as unavailable) rather than fabricated. Test in Task 2.
5. **Slug collision / archived waters** — two waters named "Lido" get distinct slugs; archived waters vanish from `/` and from the 300 m proximity check. Test in Task 8.

---

### Task 1: Scaffold + physics core

**Files:**
- Create: `swimtemps/requirements.txt`, `swimtemps/pyproject.toml` (pytest + ruff config, `pythonpath = ["."]`, marker `slow`), `swimtemps/.gitignore`
- Create: `swimtemps/forecast/__init__.py`, `swimtemps/forecast/physics.py`, `swimtemps/forecast/params.json`
- Test: `swimtemps/tests/forecast/test_physics.py`

**Interfaces:**
- Produces: `Coefficients(k_air: float, k_solar: float, k_cool: float)` frozen dataclass with `.scaled(s) -> Coefficients`; `step(t, air, solar, cloud, c) -> float`; `simulate(t0: float, weather: pd.DataFrame, c) -> np.ndarray` (len(weather)+1 values, row 0 = t0); `leg_end_temps(start_temps, A, I, C, mask, n_hours, c) -> np.ndarray` (vectorised closed form for fitting); `load_params() -> dict` reading `params.json` (`west_res` coefficients, `depth_prior`, `lambda`, `scale_min_legs`, `full_min_legs`, `max_leg_days`, `band_base` by horizon, `method_multiplier`, `type_multiplier`, provenance).

- [ ] Test: `step` matches the spec equation on hand numbers; `simulate` of 24 constant hours equals 24 chained `step`s; `leg_end_temps` equals per-leg `simulate` end for ragged legs (mask).
- [ ] Implement. Closed form: with a = 1 - k_air and forcing x_n = k_air*A_n + k_solar*I_n - k_cool*(1 - C_n/100), T_N = a^N T_0 + sum a^(N-1-n) x_n.
- [ ] Run `pytest tests/forecast -q`, commit.

### Task 2: Weather frame, anchors, targets, staged fit, band

**Files:**
- Create: `swimtemps/forecast/model.py`, `swimtemps/forecast/band.py`
- Test: `swimtemps/tests/forecast/test_model.py`, `test_band.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `prepare_weather(df) -> pd.DataFrame` — index `valid_at` (UTC hourly), cols `air_temp, shortwave, cloud_cover`; fills missing solar 0 / cloud 100; ffill air up to 3 h; reindexes to a full hourly range so gaps are NaN.
  - `anchor_hour(ts) -> pd.Timestamp` — UTC, rounded to nearest hour.
  - `forecast_targets(now_utc, tz: str, reading_time: time, days=5) -> list[pd.Timestamp]` — next 5 local days at reading time (today excluded if reading time has passed... always days 1..5 after today's local date), returned UTC.
  - `build_legs(readings: pd.DataFrame[measured_at, temp_c], weather, max_leg_days) -> Legs` (start temps, end temps, padded A/I/C matrices, mask, n_hours); a leg needs >= 80% of its hours with weather.
  - `FitResult(coefficients, scale, method: 'prior'|'scale'|'full', n_legs, train_mae)`; `fit_water(readings, weather, depth_class, params=None, method=None) -> FitResult` — method chosen by leg count unless forced (replay uses forcing).
  - `run_forward(anchor_at, anchor_temp, weather, coeffs, until) -> pd.Series` hourly trajectory; NaN from the first missing hour on.
  - `estimate_at(...)`, `forecast_points(anchor_at, anchor_temp, weather, coeffs, targets) -> list[(valid_at, temp|None)]`.
  - `band.halfwidth(horizon_days: float, method, water_type, own=None, params=None) -> float`; `band.horizon_days(anchor_at, valid_at) -> int` (ceil, clamped >= 1).
- [ ] Tests: reading at 19:00 simulates from 19:00 (not rewound); 19:40 rounds to 20:00; targets for `Europe/London` across the 2026-10-25 DST change land at 07:00 local (06:00 UTC before, 07:00 UTC after); <3 legs -> `prior` with s from depth; 3-29 -> `scale`, s pulled toward truth on synthetic data generated with s=2; >=30 -> `full` recovers synthetic k within 15%; weather hole -> trajectory NaN after it (Review Focus 4); band multipliers multiply; `own` replaces base.
- [ ] Implement; scale fit is bounded `minimize_scalar` over log s; full fit is L-BFGS-B over log k, both with `lambda*(log - log prior)^2` penalties.
- [ ] Run, commit.

### Task 3: Open-Meteo client + West Res fixtures + cold-start replay

**Files:**
- Create: `swimtemps/forecast/openmeteo.py`, `swimtemps/forecast/replay.py`, `swimtemps/forecast/data/westres_readings.csv`, `swimtemps/forecast/data/westres_weather.csv`, `swimtemps/docs/cold-start-replay.md` (generated report)
- Create: `swimtemps/tests/fixtures/openmeteo_forecast.json`, `openmeteo_archive.json`, `openmeteo_geocode.json` (recorded once)
- Test: `swimtemps/tests/forecast/test_openmeteo.py`, `test_replay.py` (`@pytest.mark.slow` full run)

**Interfaces:**
- Produces: `fetch_forecast(lat, lon) -> pd.DataFrame` (`past_days=7, forecast_days=6`, UTC, retried once), `fetch_archive(lat, lon, start: date, end: date) -> pd.DataFrame`, `geocode(name) -> list[dict]`, `point_preview(lat, lon) -> dict(air_temp, elevation, timezone)`, `parse_hourly(payload) -> DataFrame`; `OpenMeteoError`. `replay.run(readings, weather, lambdas, counts, s_priors) -> pd.DataFrame` and `replay.report(df) -> str`.
- [ ] Tests: `parse_hourly` on recorded payloads -> UTC index, column names, NaN preserved; HTTP failure raises `OpenMeteoError` after one retry (httpx MockTransport).
- [ ] Run replay over West Res for lambda in {0.1, 0.3, 1, 3, 10}, counts {3,5,10,20,30,50}, s_prior {0.5, 1, 2} (wrong depth guesses), plus the mature model (fit on everything before the anchor) for the base band (80th percentile |error| by horizon 1-5). Write report, bake chosen values into `params.json` with provenance.
- [ ] Commit.

### Task 4: Schema + Alembic + test Postgres

**Files:**
- Create: `swimtemps/db/{__init__,models,session,queries}.py`, `swimtemps/alembic.ini`, `swimtemps/db/migrations/{env.py,script.py.mako,versions/0001_initial.py}`
- Test: `swimtemps/tests/conftest.py` (session fixture: `initdb` into tmp, `pg_ctl start` on a free port, `alembic upgrade head`, truncate between tests), `swimtemps/tests/db/test_schema.py`

**Interfaces:**
- Produces: ORM classes `User, Water, WaterMember, Reading, WeatherHourly, ModelFit, ForecastRun, ForecastPoint, Snapshot, Heartbeat` exactly per spec §3; `get_engine(url=None)`, `SessionLocal`, `session_scope()`; `queries.upsert_weather(session, water_id, df, source)` (archive overwrites forecast, forecast never overwrites archive), `queries.active_readings(session, water_id) -> DataFrame`.
- [ ] Tests: migration applies on empty DB; `upsert_weather` precedence both directions; active readings exclude soft-deleted.
- [ ] Commit.

### Task 5: Worker

**Files:**
- Create: `swimtemps/worker/{__init__,cycle,snapshot,main,seed}.py`
- Test: `swimtemps/tests/worker/test_cycle.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces: `refresh_weather(session, waters, client)` (groups by `round(lat,2), round(lon,2)`, one forecast call per group; archive backfill when the oldest reading lacking weather is > 7 days old); `compute_water(session, water, now) -> Snapshot` (refit when readings changed since last fit, no fit exists, or the fit is > 24 h old; always new forecast run + points + snapshot); `build_payload(...) -> dict`; `sweep(session_factory, client, now)`; `drain_dirty(session_factory, client, now)`; `main()` loop (sweep hourly, dirty every 15 s, heartbeat each cycle); `seed.py` dev command seeding West Reservoir from the bundled CSV.
- [ ] Tests (recorded responses, no network): full single-water cycle produces fit, run, 5 points, snapshot with headline; dirty water processed and flag cleared; flag re-set mid-compute survives (Review Focus 1); failure re-sets flag; heartbeat row written; coordinate dedupe = one call for two waters 100 m apart.
- [ ] Commit.

### Task 6: Read-only web: home, water page, healthz

**Files:**
- Create: `swimtemps/web/{__init__,app,config,deps,formatting}.py`, `swimtemps/web/routes/{pages,health}.py`, `swimtemps/web/templates/{base,home,water,_status}.html`, `swimtemps/web/static/{app.css,chart.js}`
- Test: `swimtemps/tests/web/test_pages.py`, `test_health.py`

**Interfaces:**
- Produces: `create_app(settings=None) -> FastAPI`; `get_db` dependency; Jinja filters `localtime(iso, tz, fmt)`, `ago(iso)`, `temp(x)`; `GET /`, `GET /w/{slug}`, `GET /w/{slug}/status?since=` (HTMX poll; `HX-Refresh: true` once `built_at` > since), `GET /healthz` (fails: any active water snapshot > 3 h old, max `weather_hourly.fetched_at` > 3 h old when waters exist, heartbeat > 3 h old).
- [ ] Tests: page renders headline "About 13.8 °C now" and "Last reading 13.6 °C, Tue 18 Oct 07:00" from a stored payload; strip line format "Sat 07:00 — 13.1 °C (12.4–13.8)"; prior-method copy; river note; footer credit; water without snapshot shows "Computing your first forecast..."; 404 for unknown slug; archived water absent from `/`; healthz 200/503 cases.
- [ ] Browser check via `preview_start` on the seeded West Res. Commit.

### Task 7: Auth + readings

**Files:**
- Create: `swimtemps/web/auth.py`, `swimtemps/web/csrf.py`, `swimtemps/web/paste.py`, `swimtemps/web/routes/{auth,readings}.py`, templates `add_reading.html`, `_paste_preview.html`, `dev_login.html`
- Test: `swimtemps/tests/web/test_auth.py`, `test_paste.py`, `test_readings.py`

**Interfaces:**
- Produces: `Settings.supabase_url/anon_key/jwt_secret/dev_login/env`; `TokenVerifier.verify(token) -> dict` (HS256 secret if configured, else JWKS via `PyJWKClient`, audience `authenticated`); auth middleware populating `request.state.claims` and refreshing an expired token once via `SyncGoTrueClient.refresh_session`; `current_user(request, db) -> User | None` (upserts users row from claims), `require_user`; `/auth/login` (PKCE, verifier cookie), `/auth/callback`, `POST /auth/logout`, `/auth/dev-login` (only when `DEV_LOGIN=1` and `ENV != production`); `csrf_token(request)` + `verify_csrf` dependency; `parse_paste(text, tz, default_time, today) -> list[ParsedRow(line, measured_at|None, temp|None, error|None)]`; `GET/POST /w/{slug}/readings`, `POST /w/{slug}/readings/preview`, `POST /w/{slug}/readings/paste`.
- [ ] Tests: real HS256 tokens through the middleware; expired + no refresh -> signed out; dev login refused in production; POST without CSRF -> 403; anonymous POST -> redirect to login; parser cases (Review Focus 3); -2/40 limits; soft warning text and none when no estimate (Review Focus 2); rate limit 30 and whole-paste refusal; save flags `needs_recompute`.
- [ ] Commit.

### Task 8: Create-a-water wizard

**Files:**
- Create: `swimtemps/web/routes/create.py`, `swimtemps/web/slugs.py`, templates `new_location.html`, `new_details.html`, `new_readings.html`, `swimtemps/web/static/map.js`
- Test: `swimtemps/tests/web/test_create.py`

**Interfaces:**
- Produces: `slugify_unique(session, name) -> str`; `queries.nearby_waters(session, lat, lon, metres=300)` (haversine, non-archived); `GET /new` (requires sign-in), `GET /api/geocode?q=`, `GET /api/point?lat=&lon=` (preview + nearby), `POST /new/details`, `POST /new` (creates water + admin member, flags dirty, redirects to first readings), `GET /w/{slug}/first-readings`.
- [ ] Tests: create flow with mocked Open-Meteo; creator is admin; slug collision `lido`, `lido-2` (Review Focus 5); nearby excludes archived; validation (lat/lon range, timezone valid, type/depth enums).
- [ ] Browser check of the wizard on localhost with dev login. Commit.

### Task 9: Admin + /me

**Files:**
- Create: `swimtemps/web/routes/admin.py`, `swimtemps/web/permissions.py`, templates `admin.html`, `me.html`
- Test: `swimtemps/tests/web/test_admin.py`

**Interfaces:**
- Produces: `can_admin(session, user, water) -> bool` (admin member or superuser); `GET /w/{slug}/admin`, `POST /w/{slug}/admin/readings/{id}/correct` (soft-delete + replacement, `replaced_by` set), `POST .../delete`, `POST /w/{slug}/admin/details`, `POST /w/{slug}/admin/members` (promote by email), `POST /w/{slug}/admin/archive`; `GET /me`; `python -m db.cli make-superuser EMAIL`.
- [ ] Tests: contributor gets 403; superuser allowed; correction preserves old row; delete hides from page data; promote unknown email -> error message; every write flags dirty.
- [ ] Commit.

### Task 10: Deploy + docs

**Files:**
- Create: `swimtemps/Dockerfile`, `swimtemps/deploy/{docker-compose.yml,Caddyfile,.env.example}`, `swimtemps/.github/workflows/{ci.yml,deploy.yml}`, `swimtemps/README.md` (incl. morning/launch checklist), `swimtemps/CLAUDE.md`, `swimtemps/hooks/pre-commit`
- [ ] `ruff check .` clean, full suite green, `python -c "import web.app, worker.main"`.
- [ ] Commit.
