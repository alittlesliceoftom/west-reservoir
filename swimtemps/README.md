# swimtemps

Water temperatures for open-water swimming spots. Anyone can add a body of
water, swimmers add readings, and every water gets a 5-day forecast with an
80% range, driven by Open-Meteo weather and the West Reservoir heat model.

Design: [`docs/design/2026-10-18-platform-design.md`](docs/design/2026-10-18-platform-design.md);
implementation plan alongside it.

## How it works

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

- **The web app never computes.** A water page reads one `snapshots` row. It
  never calls a weather API or runs the optimiser.
- **The worker** sweeps every water hourly (one Open-Meteo call per ~1 km grid
  cell), and every 15 seconds recomputes waters the web has flagged with
  `needs_recompute`. A new water with a few readings has a forecast within
  about 20 seconds.
- **The model** (`forecast/`) is the West Res hourly physics, scaled per water
  by a depth prior and fitted in stages as readings accumulate: the prior alone
  below 2 legs, a single scale factor up to 30, all three coefficients after.
  The constants were chosen by replaying West Res's history as if it were a new
  water: [`docs/cold-start-replay.md`](docs/cold-start-replay.md).

| Directory | What it is |
|---|---|
| `forecast/` | Pure model: physics, staged fit, band, Open-Meteo client, cold-start replay. No DB, no web |
| `feeds/` | Pure feed handling: the `swimtemps-feed v1` block, the `json_table` recipe, hourly thinning, cleaning, the guarded fetch |
| `db/` | SQLAlchemy models, Alembic migrations, analytical SQL, operator CLI |
| `worker/` | The worker loop, one cycle's units of work, the snapshot payload, a dev seed |
| `web/` | FastAPI app: pages, auth, readings, create wizard, admin. Templates and static files |
| `deploy/` | Compose stack, Caddyfile, `.env.example` |
| `tests/` | pytest; runs against a throwaway Postgres, no network |

## Feeds

A few venues publish water temperatures automatically from a sensor or
logger. Such a water can take its readings from the venue's feed instead of
from people typing them in. Design: [`docs/design/2026-10-18-feeds-design.md`](docs/design/2026-10-18-feeds-design.md).

- **What qualifies.** Any public, machine-readable source: a sensor or
  logger's API, or a sheet that staff keep up to date, so long as it is
  published as JSON. One measurement point per water.
- **Mode.** Each water is `manual` or `feed`, never both. Its admins switch
  either way from Admin, then "set up a feed" (`/w/<slug>/feed`); history is
  kept both ways. A feed water refuses typed readings with 409.
- **Setup.** The setup page offers a prompt to give an AI with the venue's
  page. The AI answers with a fixed block, which the page reads back into the
  fields:

  ```
  swimtemps-feed v1
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
  end
  ```

  `orientation: rows` uses `rows_path`, `header_row`, `time_field` and
  `value_field` instead. `terms: forbids automated access`
  stops setup and points to the request form. **Test** fetches once and shows
  what would be stored and flagged, storing nothing. **Turn on** needs that
  test, and the permission box (ticked by default): "I have the venue's
  permission to share this data, or it is publicly published." Who ticked it,
  when, and the statement are stored. The water page credits the venue with a
  link.
- **Requests.** Anything the recipe cannot read goes to `/feeds/request`
  (any signed-in user); superusers handle them at `/admin/feed-requests`. A
  source that needs code gets a small adapter registered in
  `feeds.recipe.ADAPTERS`.
- **Polling.** The worker polls each feed that is on at the start of its
  hourly sweep (and a just-enabled feed within seconds). Fetches go only to
  public addresses, at most 5 MB and 20 s, with the user-agent `swimtemps/0.1`.
  A failure is recorded on the feed and never stops other feeds.
- **Storing.** Rows pass the manual checks (-2 to 40 °C, not in the future,
  not before 2000), are thinned to one per clock hour, and are matched on
  their time: new rows are inserted, unchanged ones skipped, and a changed
  value is stored as the venue's correction. Nothing is destroyed.
- **Cleaning.** A reading is flagged, with its reason, when it jumps more than
  2.0 °C + 0.1 °C an hour from the last good reading, sits more than 2.5 °C
  from the previous day's good readings, or repeats one value through more than
  a day of hourly readings in a row. Flagged readings are stored and shown in grey on the chart, and kept
  out of the fit, the estimate and scoring. Admins can overrule a flag either
  way from the admin page, and correct or delete any feed reading by hand:
  later polls leave that reading as the admin left it.
- **Model.** Feed readings pair into day-long legs: each to the feed reading
  closest to 24 hours later, within 21 to 27 hours. Typed readings pair as
  before.
- **Quiet and paused.** Clocks run from the feed's last good reading: the
  page says "Feed quiet since ..." after 12 hours (or 1.5 times the feed's
  usual gap between readings, if longer: a daily sheet is quiet after a day
  and a half); `/healthz` fails after 3
  days; the forecast stops after 5 days and resumes with the next good
  reading. Admins can pause a feed (a sensor out for winter): a paused feed is
  not polled, shows no forecast, and does not fail `/healthz`. `/me` lists a
  user's feeds with a switch to manual.

## Local development

Python 3.14 and Postgres 17 (`brew install postgresql@17`).

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
git config core.hooksPath hooks   # ruff on commit (bugs only: F, E9)
```

A local database, migrated and seeded with West Reservoir's 416 readings:

```bash
LC_ALL=en_US.UTF-8 /opt/homebrew/opt/postgresql@17/bin/initdb -D .devdb -U postgres -A trust
LC_ALL=en_US.UTF-8 /opt/homebrew/opt/postgresql@17/bin/pg_ctl -D .devdb -o "-p 54329" -l .devdb/log start
/opt/homebrew/opt/postgresql@17/bin/createdb -h localhost -p 54329 -U postgres swimtemps
export DATABASE_URL=postgresql://postgres@localhost:54329/swimtemps
.venv/bin/alembic upgrade head
.venv/bin/python -m worker.seed
```

Run the worker and the site (two terminals, same `DATABASE_URL`):

```bash
.venv/bin/python -m worker.main
```

```bash
ENV=development DEV_LOGIN=1 SUPABASE_JWT_SECRET=any-long-local-string-of-32-chars-or-more \
  .venv/bin/uvicorn web.app:app --reload
```

`DEV_LOGIN=1` swaps Google for a local form that signs you in as any email
(refused when `ENV=production`). Make yourself a superuser with
`python -m db.cli make-superuser you@example.com`.

## Tests

```bash
.venv/bin/pytest            # ~245 tests, ~10 s, no network
.venv/bin/pytest -m slow    # the full West Res cold-start replay (~1 min), rewrites docs/cold-start-replay.md
.venv/bin/ruff check .
```

The suite starts its own Postgres cluster (via `initdb`/`pg_ctl` on PATH or
Homebrew's) and applies the Alembic migrations, so the migrations are tested
every run. Set `TEST_DATABASE_URL` to use an existing server instead, as CI does.
Open-Meteo responses are recorded in `tests/fixtures/`.

## Launch checklist

Everything below needs an account or a decision, so none of it is done yet.
Each step names the setting it produces.

1. **Make this its own repo.** From the West Res repo root:
   `git subtree split --prefix=swimtemps -b swimtemps-main`, then push that
   branch as `main` of a new `swimtemps` repo. The workflows in `.github/` start
   working once they are at the repo root.
2. **Domain** (open question in the spec). Point `A`/`AAAA` records at the VPS.
   Sets `DOMAIN`, `SITE_URL`.
3. **Supabase** project, London region.
   - Database -> Connect -> **Session pooler** string -> `DATABASE_URL`. The
     direct connection is IPv6-only, and containers on Docker's default
     network have no IPv6, so the pooler is the default here. To use the
     direct connection, enable IPv6 in the Docker daemon and add
     `networks: {default: {enable_ipv6: true}}` to the compose file first.
   - Project URL and anon key -> `SUPABASE_URL`, `SUPABASE_ANON_KEY`.
   - Keep the default asymmetric JWT signing keys and leave
     `SUPABASE_JWT_SECRET` empty: tokens are verified against the project's JWKS.
4. **Google sign-in.** In Google Cloud, create an OAuth client (web
   application) with authorised redirect URI
   `https://PROJECTREF.supabase.co/auth/v1/callback`. In Supabase, Auth ->
   Providers -> Google: paste its client ID and secret. Auth -> URL
   configuration: Site URL = `SITE_URL`; add `SITE_URL/auth/callback` to the
   redirect allow-list. **Disable the Email and Anonymous providers**: the app
   accepts any token the project signs, so a provider left on is another way in.
5. **Hetzner CX23**, Ubuntu 24.04: install Docker, create `/opt/swimtemps`,
   put `.env` there (from `deploy/.env.example`), and a deploy user whose SSH
   key goes in GitHub. If the image is private, `docker login ghcr.io` on the
   box with a read-only token.
6. **GitHub**: secrets `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`,
   `VPS_KNOWN_HOSTS`; variable `DEPLOY_ENABLED=true`. A push to `main` then
   tests, builds, pushes, migrates and restarts.
7. **First run**: sign in once, then on the box
   `docker compose run --rm web python -m db.cli make-superuser YOU@EMAIL`.
8. **Monitoring**: an UptimeRobot HTTP monitor on `SITE_URL/healthz`. It returns
   503 when a snapshot is over 3 hours old, Open-Meteo has not answered in 3
   hours, or the worker has stopped writing heartbeats. The heartbeat rows also
   keep the free Supabase project from pausing.
9. **Bristol onboarding**: send them `SITE_URL/new`.

## Notes

- Weather is requested in UTC (`timezone=UTC`) so hourly rows are unique across
  DST changes; each water's own timezone is only used for display and reading
  times. The forecast call asks for 7 days ahead, not 6: Open-Meteo counts UTC
  days, and the fifth local reading time can fall on the seventh UTC day.
- Open-Meteo's free tier is non-commercial and under 10,000 calls a day; the
  footer carries the CC BY 4.0 credit. The grid-cell dedupe keeps the sweep
  to about 24 calls per cell per day.
- Backups are deferred to after release (Linear).
