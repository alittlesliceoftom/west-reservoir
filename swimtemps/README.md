# swimtemps

Water temperatures for open-water swimming spots. Anyone can add a body of
water, swimmers add readings, and every water gets a 5-day forecast with an
80% range, driven by Open-Meteo weather and the West Reservoir heat model.

Design: [`docs/superpowers/specs/2026-10-18-swimtemps-platform-design.md`](../docs/superpowers/specs/2026-10-18-swimtemps-platform-design.md)
(in the West Res repo until this directory becomes its own repo).

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
| `db/` | SQLAlchemy models, Alembic migrations, analytical SQL, operator CLI |
| `worker/` | The worker loop, one cycle's units of work, the snapshot payload, a dev seed |
| `web/` | FastAPI app: pages, auth, readings, create wizard, admin. Templates and static files |
| `deploy/` | Compose stack, Caddyfile, `.env.example` |
| `tests/` | pytest; runs against a throwaway Postgres, no network |

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
.venv/bin/pytest            # ~230 tests, ~10 s, no network
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
   - Database connection string -> `DATABASE_URL` (direct connection is
     IPv6-only, fine on Hetzner; the session pooler is the IPv4 fallback).
   - Project URL and anon key -> `SUPABASE_URL`, `SUPABASE_ANON_KEY`.
   - Keep the default asymmetric JWT signing keys and leave
     `SUPABASE_JWT_SECRET` empty: tokens are verified against the project's JWKS.
4. **Google sign-in.** In Google Cloud, create an OAuth client (web
   application) with authorised redirect URI
   `https://PROJECTREF.supabase.co/auth/v1/callback`. In Supabase, Auth ->
   Providers -> Google: paste its client ID and secret. Auth -> URL
   configuration: Site URL = `SITE_URL`; add `SITE_URL/auth/callback` to the
   redirect allow-list.
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
