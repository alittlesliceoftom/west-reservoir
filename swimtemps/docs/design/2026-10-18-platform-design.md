# swimtemps: multi-water forecasting platform — design

Status: draft for review. Project name: `swimtemps`.
Tracking: Linear project "swimtemps" (P-CRA-20).

## 1. Goal

A simple, self-hosted website where anyone can add a body of water, enter
temperature readings, and get a water-temperature forecast driven by weather
APIs. It replaces nothing yet: the West Res Streamlit app stays live and
untouched until the new site is running well, then West Res migrates as a
separate piece of work.

### Success criteria

- A water page renders from stored data in well under a second. No page load
  ever calls a weather API or runs the optimiser.
- A new water with a handful of pasted readings shows a forecast within about
  30 seconds of creation.
- Forecasts for a new water visibly improve (narrower band, "calibrated from N
  readings") as readings accumulate.
- The Bristol test user can create their water and enter readings on a phone
  without help.

### Decisions taken

| Topic | Decision |
|---|---|
| Contribution model | Crowdsourced: any signed-in user may add readings to any water. Schema leaves room for owner-only and commercial venues later |
| Sign-in | Google only, via Supabase Auth |
| Water admin | The creator of a water is its admin and corrects readings. The site owner is a superuser but not in the day-to-day loop |
| West Res | Not migrated in v1. Supplies the default model coefficients |
| Hosting | Hetzner VPS (Docker Compose) + Supabase free tier (Postgres + Auth). Roughly EUR 6/month plus domain |
| Frontend | Server-rendered Python: FastAPI + Jinja + HTMX, ECharts for charts |
| DB access | Hybrid: SQLAlchemy 2.0 ORM for tables and writes, Alembic migrations, raw SQL via `text()` for analytical queries |
| Repo | New repo. The forecast model is a package inside it |
| Water types | Still water, rivers and harbours. Rivers and harbours carry a "less reliable" warning. Sea is out of scope |
| Location | A map pin, not a weather station. Open-Meteo is gridded, so any lat/lon works |
| Reading time | Each water has a default reading time set at creation; each reading may override it |
| Depth | Optional rough depth class at creation, used as the model prior |
| Horizon | 5 days |

## 2. Architecture

```
                  +------------- VPS (Docker Compose) -------------+
 browser --HTTPS--> Caddy --> web (FastAPI + Jinja + HTMX)          |
                  |              | reads snapshots, writes readings  |
                  |              v                                   |
                  |           worker (Python loop)                   |
                  |              | weather -> fit -> forecast -> snapshot
                  +--------------+-----------------------------------+
                                 v
                Supabase: Postgres + Auth (Google)      Open-Meteo (weather)
```

### Repo layout

```
forecast/        pure model package: physics, fitting, Open-Meteo client.
                 No DB, no web. Heaviest test coverage.
web/             FastAPI app: routes, templates, auth. Never computes forecasts.
worker/          long-running loop: hourly sweep + dirty-queue polling.
db/              SQLAlchemy models, Alembic migrations, analytical SQL.
deploy/          docker-compose.yml, Caddyfile, GitHub Actions workflow.
tests/
```

### Compute model

The worker is one Python process with two duties:

1. **Hourly sweep.** For every active water: refresh weather, fit if new data
   since the last fit, forecast, write snapshot.
2. **Dirty queue.** Every ~15 seconds, pick up waters with
   `needs_recompute = true` (set by the web on any reading insert, correction or
   water edit), recompute them, clear the flag.

No Celery or Redis: a flag column in Postgres is the queue. The web never waits
on the worker. If the worker or Open-Meteo is down, pages keep serving the last
snapshot with an explicit "Forecast updated N hours ago" label.

## 3. Data model

```
users            id uuid PK (= Supabase auth user id), email, google_sub,
                 display_name, is_superuser bool, created_at

waters           id, slug (URL /w/<slug>), name, lat, lon, timezone,
                 water_type ('still'|'river'|'harbour'),
                 depth_class ('shallow'|'lake'|'deep'|'unknown'),
                 default_reading_time time,
                 needs_recompute bool, created_by -> users,
                 created_at, archived_at

water_members    water_id, user_id, role ('admin'|'contributor')
                 PK (water_id, user_id)

readings         id, water_id, measured_at timestamptz, temp_c numeric(4,1),
                 submitted_by -> users, created_at,
                 deleted_at, deleted_by -> users, replaced_by -> readings

weather_hourly   water_id, valid_at timestamptz, source ('archive'|'forecast'),
                 air_temp, shortwave, cloud_cover, fetched_at
                 PK (water_id, valid_at)

model_fits       id, water_id, fitted_at, k_air, k_solar, k_cool, scale,
                 n_legs, method ('prior'|'scale'|'full'), train_mae

forecast_runs    id, water_id, created_at, fit_id -> model_fits,
                 anchor_reading_id -> readings, anchor_at, anchor_temp

forecast_points  run_id, valid_at, water_temp, lower, upper
                 PK (run_id, valid_at)

snapshots        water_id PK, built_at, payload jsonb

heartbeat        id, at   (one row written per worker cycle)
```

Rules:

- `snapshots.payload` is exactly what the water page renders: headline,
  latest reading, chart series, 5-day strip, calibration status, built_at.
  The page reads one row.
- Every forecast run is kept, so a later accuracy page is a join against
  `readings` with history from day one.
- Weather is stored per water. The worker deduplicates Open-Meteo calls by
  coordinates rounded to two decimal places. For past hours, an archive value
  overwrites a forecast value.
- Corrections never destroy data. A correction soft-deletes the old reading
  and points `replaced_by` at the new one. All reads filter
  `deleted_at IS NULL`.
- The creator gets a `water_members` row with role `admin`. Contributors need
  no row while the water is crowdsourced. Owner-only or commercial later means
  a `contribution_policy` column on `waters` and new role values, not a
  restructure.
- All temperatures are stored in Celsius.
- `users.google_sub` is kept so identity survives a move off Supabase Auth.

## 4. Forecast model (`forecast/` package)

The physics is unchanged from West Res: per hour,

```
clearness(t) = 1 - cloud_cover(t) / 100
T(t+1h) = T(t) + k_air*(T_air(t) - T(t)) + k_solar*I(t) - k_cool*clearness(t)
```

What changes:

### 4.1 Timestamped anchors

The current code hardcodes a 07:00 measurement hour (`MEASUREMENT_HOUR` in
`forecaster.py`) and discards time-of-day in `predict_forward`. The new package
works from `measured_at` timestamps:

- Simulation starts at the reading's hour (rounded to the nearest hour) and
  steps hourly.
- Training legs run reading to reading at whatever spacing they have, capped at
  3 days elapsed, as now.
- Forecast points are emitted at the water's `default_reading_time` on each of
  the next 5 local days.
- **Estimated now:** when the latest reading is older than the current hour,
  simulate from it to now. This drives the page headline.

### 4.2 Depth prior

Every term is a heat flux divided by the water's thermal mass, so all three
coefficients scale roughly with 1/depth. One scale factor `s` per water
multiplies the West Res coefficients:

| depth_class | prior s |
|---|---|
| shallow (< 2 m) | 2.0 |
| lake (2-5 m), West Res reference | 1.0 |
| deep (> 5 m) | 0.5 |
| unknown | 1.0 |

West Res's absolute depth is not needed; it is the reference.

### 4.3 Staged fitting

By number of usable legs (consecutive reading pairs with weather coverage):

| Legs | Method | Fitted |
|---|---|---|
| < 3 | `prior` | nothing; prior `s` |
| 3-29 | `scale` | `s` only, with penalty `lambda * (log s - log s_prior)^2` |
| >= 30 | `full` | all three k, each penalised in log space towards `s_prior * k_WestRes` |

`lambda` is set so the prior carries roughly the weight of 5 legs. The exact
value and the 3 / 30 thresholds are fixed by the cold-start replay (4.5), not
guessed.

### 4.4 Uncertainty band

An 80% band on each forecast point:

- Base: per-horizon error measured by the cold-start replay on West Res.
- Multiplied by fit method: `prior` x1.5, `scale` x1.25, `full` x1.0.
- Multiplied by water type: river or harbour x1.5.
- Once a water has at least 20 scored forecast points of its own, its own
  per-horizon errors replace the West Res base.

### 4.5 Cold-start replay (the key validation)

Replay West Res history as if it were a new water: for each start point, give
the model only the first 3, 5, 10, 20 readings, fit, forecast 1-5 days, score
against measured values. Output: error by horizon by reading count, for each
candidate `lambda`. This selects `lambda`, the thresholds, and the base band,
and tells us what Bristol's first week will look like.

### 4.6 Weather

- Hourly sweep: one Open-Meteo forecast call per coordinate group with
  `past_days=7&forecast_days=6`, hourly air temperature, shortwave radiation,
  cloud cover, `timezone=auto`.
- Backfill: readings older than 7 days trigger an Open-Meteo archive call for
  the missing range.
- Missing solar/cloud is filled with zero solar and 100% cloud, as now, which
  collapses to air conduction alone.
- Open-Meteo free tier is non-commercial, under 10,000 calls/day, CC BY 4.0.
  Footer credits "Weather data by Open-Meteo.com". At ~24 calls per coordinate
  group per day this supports ~400 groups.

## 5. Pages and flows

Mobile-first throughout: most readings are typed at the water's edge.

### Public

- **`/`**: list of waters with name, estimated temperature now, last reading
  date. One call to action: "Add your water".
- **`/w/<slug>`**:
  - Headline: "About 13.8 °C now", then "Last reading 13.6 °C, Tue 18 Oct 07:00".
  - Chart: readings as points, air temperature as a faint line, 5-day forecast
    with 80% band, a today marker.
  - Next-5-days strip: "Sat 07:00 — 13.1 °C (12.4–13.8)".
  - Calibration status: "Calibrated from 12 readings" or "Using default
    parameters — forecasts tighten as readings are added".
  - River/harbour note where applicable.
  - Footer: "Forecast updated 14 min ago", Open-Meteo credit.

### Add a reading (signed in)

- Single entry: large numeric field; time defaults to today at the water's
  reading time, with "Edit time" to override.
  - Hard limit -2 to 40 °C.
  - Soft warning beyond 3 °C from the current estimate.
- Paste several: textarea accepting `dd/mm/yyyy temp`, `dd/mm hh:mm, temp` and
  similar, day-first, time optional. Shows a preview table flagging
  duplicates (same water, same hour) and outliers, then a confirm button.
- On save the water is flagged dirty; the page shows "Updating forecast..." and
  HTMX polls until the snapshot's `built_at` advances.
- Rate limit: 30 readings per user per water per hour (a paste counts per row).

### Create a water (`/new`), four steps

1. Sign in with Google.
2. Where is it: place search (Open-Meteo geocoding) then a draggable pin on an
   OpenStreetMap (Leaflet) map. Below the map, "Weather for this spot: 11.2 °C
   air now, elevation 45 m" so the user confirms the weather looks right; this
   replaces choosing a weather station. Timezone is taken from Open-Meteo.
   If an existing water is within 300 m, ask "Is this <name>?" before
   continuing.
3. About it: name, type (lake/reservoir, river, harbour), rough depth,
   usual reading time (default 07:00).
4. First readings: paste or type a few, or skip. Lands on the water page with
   "Computing your first forecast...".

### Admin

- `/w/<slug>/admin` (water admins and superusers): readings table with
  correct (soft-delete + replacement) and delete (soft); edit water details;
  promote other users to admin.
- `/me`: my waters and my recent readings.

## 6. Authentication

- Supabase Auth with the Google provider. The Google OAuth client is created
  once in Google Cloud and configured in Supabase.
- Server-side PKCE flow with supabase-py (`flow_type="pkce"`): `/auth/login`
  starts the flow with the code verifier stored in a short-lived httpOnly
  cookie; `/auth/callback` exchanges the code for a session.
- Access and refresh tokens are stored in httpOnly, Secure, SameSite=Lax
  cookies. Each request verifies the access token locally against the
  project's JWKS (PyJWT `PyJWKClient`, cached); an expired token is refreshed
  once via Supabase, otherwise the user is signed out.
- On first sign-in, upsert a `users` row with the Supabase user id, email and
  Google `sub`.
- The browser never talks to Supabase directly; row-level security is not
  used. Authorisation lives in the web app.
- State-changing requests are POST only, behind SameSite=Lax cookies plus a
  CSRF token in forms and HTMX headers.
- The exact supabase-py method signatures differ between its reference docs and
  examples; the first auth task pins the installed version and verifies them.

## 7. Deployment and operations

- Hetzner CX23, Docker Compose: `caddy` (HTTPS via Let's Encrypt), `web`
  (uvicorn), `worker`.
- Supabase free project, London region. Direct connection is IPv6-only, which
  Hetzner supports; the Supabase pooler is the IPv4 fallback.
- Deploy: push to `main` -> GitHub Action builds images, SSHes to the VPS,
  runs `alembic upgrade head` then `docker compose up -d`.
- Secrets in a `.env` on the box, never committed.
- Supabase pause protection: the worker writes a `heartbeat` row each cycle
  (pausing is judged on real queries, not open connections).
- Freshness: `/healthz` fails if any active water's snapshot is older than 3
  hours or Open-Meteo has not answered in 3 hours. A free UptimeRobot monitor
  emails on failure.
- Backups are deferred to post-release (see Linear project description).

## 8. Testing

- `forecast/`: unit tests for the physics step, timestamped anchors, staged
  fitting and the band; the cold-start replay as a slow, marked test that
  writes its report.
- `web/`: FastAPI TestClient against a local Postgres in Docker. Covers
  permissions, the paste parser, soft-delete and corrections, snapshot render.
  Auth is mocked by injecting verified claims.
- `worker/`: a full single-water cycle against recorded Open-Meteo responses,
  no network.
- Ruff pre-commit hook (`F`, `E9`), as in the current repo.

## 9. Build order

Each step ends deployable.

1. `forecast/` package: port physics, timestamped anchors, depth prior, staged
   fitting, band; cold-start replay on West Res.
2. Schema and migrations; worker producing snapshots for a hand-seeded water.
3. Read-only water page and home page, deployed on the VPS. First thing the
   Bristol user can see.
4. Supabase Google sign-in; add-a-reading and paste-several.
5. Create-a-water wizard.
6. Admin: corrections, members.
7. Bristol onboarding.

## 10. Out of scope for v1

- Forecast accuracy page (data is stored from day one).
- Map home page.
- Notifications, unit toggle.
- PostHog analytics.
- Off-box backups (post-release, Linear).
- Sea water and marine SST.
- Owner-only and commercial contribution policies (schema-ready only).
- Migrating West Res.
- Commercial use, which would need a paid Open-Meteo plan.

## 11. Open questions

- Domain.
