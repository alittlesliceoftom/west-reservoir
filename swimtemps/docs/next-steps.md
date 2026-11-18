# swimtemps: next steps (2026-11-18)

Feeds are built, reviewed, fixed and published to `alittlesliceoftom/swimtemps` main.
Code lives in the `swimtemps/` subtree of west-res branch `claude/swimtemps-platform-impl-f0c93e`.

## 1. Before launch: small fixes (Claude, about an hour, TDD)

1. Cache-busting for static assets: add a content-hash `?v=` to `app.css`, `app.js`,
   `chart.js` and `map.js` in templates, so deploys don't serve stale CSS/JS.
2. Rate limits: `/api/geocode` and `/api/point` (new-water wizard) and the feed setup's
   Test POST, per user.
3. Feed quirks:
   - an overrule refreshes `feed.last_good_reading_at`;
   - `/me` shows "Switch to manual" only to the water's admins, and says "quiet";
   - bad Test input (port `:99999`, deeply nested JSON) gives a message, not a 500.

## 2. Accounts (the user; README "Launch checklist" has each step)

1. Supabase project (free): database plus sign-in. Gives `SUPABASE_URL`,
   `SUPABASE_ANON_KEY`, the JWT secret/JWKS, and the pooler `DATABASE_URL`.
2. Google Cloud OAuth client (free, ~10 min) for "Sign in with Google"; paste the
   client ID and secret into Supabase Auth > Providers > Google.
   - Alternative: email magic links, which need about an hour of app work plus an
     SMTP service (e.g. Resend).
3. Hetzner VPS (~€4/month), with SSH access for the deploy job.
4. A domain, pointed at the VPS (Caddy gets the TLS certificate).

## 3. Deploy

1. Set the GitHub repo secrets/vars from the checklist, then `DEPLOY_ENABLED=true`.
2. Push main; check CI and the deploy run; run the migration profile (`alembic upgrade head`).
3. `make-superuser` for the first account; point an uptime monitor at `/healthz`.

## 4. After launch (backlog)

- Reconfiguring a feed (new URL, timezone fix) reuses its history: start a fresh
  history or warn on the setup page.
- The admin page lists only the newest 200 readings (~8 days of an hourly feed):
  paginate it.
- DNS-rebinding window in `feeds/fetch.py`: pin the checked IP (httpx `sni_hostname`).
- `forecast_points` retention against the 500 MB free tier.
- Per-feed cleaning thresholds have no UI; recheck the defaults on summer data.
- `/healthz` (3 days) and the forecast pause (5 days) are fixed, so a weekly-updated
  sheet keeps `/healthz` red.
- Older platform minors: the auth middleware running on static files, SRI hashes,
  refresh-token revocation on logout.

## Dev environment

- Postgres: `swimtemps/.devdb`, port 54329. If it isn't running, start it with
  `LC_ALL=en_US.UTF-8 pg_ctl -D .devdb -o "-p 54329" start`.
- Web: preview "swimtemps-web" on port 8765 (`.claude/launch.json`); dev login, with
  `tester@example.com` as superuser.
- Worker: `DATABASE_URL=postgresql://postgres@127.0.0.1:54329/swimtemps .venv/bin/python -m worker.main`.
- Tests: `.venv/bin/pytest -q` (456 pass), `.venv/bin/ruff check .`.
- Publish: `git subtree split --prefix=swimtemps -b swimtemps-main`, then
  `git push git@github.com:alittlesliceoftom/swimtemps.git swimtemps-main:main` over SSH.
- Outbound requests: user-agent `swimtemps/0.1`, nothing that identifies the user.
