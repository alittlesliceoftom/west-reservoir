# swimtemps: automated feeds — design

Status: draft for review. Extends `2026-10-18-platform-design.md`.

## 1. Intent

A few venues run a sensor or logger that publishes water temperatures
automatically, in a machine-readable form (JSON or an API). For those, a water
should be able to take its readings from the venue's feed instead of from
people typing them in, with as little per-venue code as possible. Everything
after "fetch and map to (time, temperature)" is shared with manual readings.

Manual entry remains the primary use case. Feeds are a convenience for the
handful of venues whose data is already automated.

### Decisions taken (by the product owner)

| Topic | Decision |
|---|---|
| Measurement points | One per water. Venues with several sensors are out of scope for now; they would be modelled later as separate waters. Clevedon Marine Lake (two sensors) is parked, not a launch example |
| Mode | Each water is either **manual** or **feed**, never both at once. The water's admin chooses at creation or later in admin, and can switch either way without losing history |
| What qualifies | Only automated, reliable, machine-readable sources: a sensor or logger publishing JSON or an API. Sources staff update by hand (a Google Form, a CMS sentence, a vendor's hand-entered field) stay manual waters, even if they are machine-readable |
| Who sets it up | Anyone who can configure it can. Where the source fits the generic recipe, the person fills in the recipe themselves. Otherwise they submit a feed request and we set it up |
| Helping people configure | The setup page explains, in plain words, how to get a recipe: "If your venue's website shows live temperatures, give this prompt and the page's address to an AI, then paste its answer here" |
| Permission | A public feed is fair game provided we credit the venue and link back. Setup includes a checkbox, ticked by default, with the full statement beside it: "I have the venue's permission to share this data, or it is publicly published." Who confirmed it and when is stored |
| Cleaning | Every fetched reading is stored as received. Readings our checks flag are shown greyed out on the chart with the reason, and are excluded from the fit, the estimate and scoring. Never hidden: people should be able to see when cleaning got it wrong, or appreciate that it was done |
| Quiet feed | The water page says "Feed quiet since <date>" and keeps showing the last good data, the band widening as the gap grows. The water's admins see it on `/me` with a one-click switch to manual. After 3 days without a new reading the feed fails `/healthz`. After 5 days without data, forecasting stops |

### Non-goals

- Multiple sensors or depths per water.
- Scraping HTML pages. A feed must be JSON or an API.
- Push ingestion (venues posting to us).
- Getting past signed APIs, bot checks or terms that forbid automated
  access. Such sources are recorded as "needs the venue's or vendor's
  agreement" and not fetched. Example: My Water Live (Tooting Bec, Brockwell),
  whose terms forbid scraping and whose API is signed.

### Anti-goals

- A cleaning step that silently drops data.
- A feed water that looks current when its feed has stopped.
- Requests that carry anyone's name, email or handle. The user-agent is
  `swimtemps/0.1` and nothing else.

## 2. Reference sources

| Source | Shape | Use |
|---|---|---|
| The Lake, Cromhall (Bristol) | Google Apps Script URL returning `[["What date…","What was the temperature"],["12-Sep-2024","10.60"],…]`: rows with a header, `d-Mon-YYYY` dates, no time, full history every call | **Not a feed by the rule above** (staff type it into a Google Form), but it is the clearest example of the row-shaped JSON the recipe must read, so use it as a test fixture |
| Clevedon Marine Lake | `GET /wp-json/cml/v1/temps?sensor=<id>&hours=168`: `{"labels":[epoch ms…],"datasets":[{"data":[°C…]}]}`, half-hourly, 7 days of history | Parked (two sensors). Still the example of the column-shaped JSON (parallel arrays) the recipe must read, and of real sensor faults: on 2026-10-18 both sensors dropped ~4 °C in 30 minutes to the overnight air temperature |

Recorded responses are in `tests/fixtures/feeds/` (see its README for the URLs and sensor ids). They are the only copies of the Clevedon fault, because that API keeps 7 days.

## 3. The recipe

There is one generic recipe, `json_table`, plus a code escape hatch.

### `json_table` configuration

| Field | Meaning |
|---|---|
| `url` | GET endpoint returning JSON |
| `orientation` | `rows` (a list of records or arrays) or `columns` (parallel lists of times and values) |
| `rows_path` | for `rows`: where the list is (`$` for the root) |
| `header_row` | for `rows` of arrays: whether the first row is a header to skip |
| `time_field`, `value_field` | for `rows`: the index or key of the time and the temperature in each record |
| `time_path`, `value_path` | for `columns`: where the two parallel lists are, e.g. `labels` and `datasets[0].data` |
| `time_format` | `epoch_ms`, `epoch_s`, `iso`, or a strptime pattern such as `%d-%b-%Y` |
| `timezone` | for times without an offset; defaults to the water's timezone |
| `unit` | `C` (default) or `F` |
| `source_name`, `source_link` | for the credit line: "Readings from <source_name>" linking to `source_link` (the venue's own page, not the API) |

Paths are a small dotted syntax with `[n]` indexes, enough for both
reference shapes. The builder may adopt a JSONPath library instead.

Date-only times take the water's `default_reading_time`. Values arrive as
numbers or numeric strings; anything else is a parse error for that row.

### The escape hatch

A feed request we accept but `json_table` cannot express is implemented as a
small Python adapter registered by name, returning the same (time, °C) rows.
The recipe field then names the adapter, and its configuration is whatever
that adapter needs.

### Setup flow

1. In the water's admin (or the last step of creating a water), choose
   "Readings come from the venue's feed".
2. Fill in the recipe. Above the form, the explainer and the AI prompt (§6).
3. **Test**: the server fetches once and shows the result in the same preview
   used for pasted readings: rows ready to store, rows that failed to parse
   (with why), and rows the cleaning would flag (with why).
4. The permission checkbox, ticked by default, with its full statement.
5. **Turn on**. The water switches to feed mode and polling starts.

For anything that does not fit, a **Request a feed** form takes the venue
page URL, notes, and optionally the AI's summary pasted in. Requests are
listed for superusers to handle.

## 4. Behaviour

### Polling

The worker polls each enabled feed hourly as part of its sweep, one GET per
feed with a timeout. A fetch failure is recorded on the feed (`last_error`)
and never stops other feeds or waters.

### Storing

Fetched rows pass the same checks as manual readings (−2 to 40 °C, not in
the future, not before 2000). A row that fails is not stored but counts in
the feed's last-fetch summary.

Valid rows are matched to stored ones on (feed, measured time) among rows
that stand (`deleted_at IS NULL`):

- No match: insert.
- Same value: nothing happens, so re-fetching a full history adds nothing.
- Different value: the venue corrected it. Store it as a correction, the
  same way an admin correction works now: the old row is soft-deleted, a new
  row is inserted, and `replaced_by` points to it. Nothing is destroyed, and
  refit detection keeps working off `created_at` and `deleted_at`.

New or corrected rows flag the water for recompute.

*(Design call.)* On a feed water, admins do not delete or correct feed
readings, because the next poll would bring them back. The tool for a bad
feed reading is the flag overrule. An overrule belongs to the value it was
made on: if the venue later corrects that reading, the new row is cleaned
afresh.

### Cleaning

Applied to feed readings on arrival. Each rule sets `flagged` with a
plain-English reason:

| Rule | Default | Catches |
|---|---|---|
| Rate of change | more than 1.0 °C per 30 minutes, scaled by the actual interval | step faults |
| Tracking air | within 1 °C of the stored air temperature for that hour **and** more than 2 °C from the reading's own trailing 6-hour median | a sensor out of the water |
| Flatline | the same value for more than 24 hours | a stuck sensor |

Thresholds are per-feed settings, with these defaults.

The water's admins can overrule any flag, either way ("this was real" /
"this is wrong"). The overrule is stored with who and when, and the chart
shows it.

### What the model uses

The chart shows every stored reading. The fit, the estimate and scoring use
unflagged readings only (an overrule decides for the row it is on).

*(Design call.)* Readings with `source = feed` are thinned to one per local
day, the one nearest the water's reading time, before fitting; half-hourly
sensor data would otherwise make training legs dominated by sensor noise.
Manual readings are never thinned, whatever the water's mode, so manual
waters fit exactly as now. A day with both keeps both; the fit already
merges readings in the same hour.

The anchor for "now" is the latest unflagged reading of either source,
unthinned.

### Quiet feeds

*(Design call.)* Every clock here measures time since the feed's latest
**unflagged** reading, its last good reading. A sensor lying out of the
water still publishes, but everything it sends is flagged, so it counts as
quiet. Without this the page would keep forecasting from an ever-older
anchor while looking live.

- **12 hours without a good reading** *(design call: long enough that
  hourly polling of a half-hourly feed never shows it)*: the page shows
  "Feed quiet since <time of last good reading>" and keeps forecasting from
  that reading, the band widening with the anchor's age as it already does.
- **3 days without a new reading:** `/healthz` fails, naming the water.
  The water's admins see the feed as quiet on `/me`, with a button to switch
  the water to manual.
- **5 days without a good reading:** no forecast and no estimate. The page
  shows the readings and "Forecast paused: no data from <source_name> since
  <date>". Forecasting resumes on the next good reading.
- **Paused by an admin** *(design call)*: a water's admins can pause its
  feed, for example while the sensor is out for winter. A paused feed is
  not polled and does not fail `/healthz`; the page says "Feed paused by the
  venue" over the last readings, with no forecast. Without this, one
  seasonal sensor would keep the site's only uptime monitor red for months
  and hide real outages.

Manual waters are unchanged: they keep estimating from an old reading.

### Switching modes

Manual to feed: existing manual readings stay and are used by the fit; the
add-reading button is replaced by the credit line. Feed to manual: polling
stops, feed readings stay, the add-reading button returns. The feed's
configuration is kept so the water can switch back.

## 5. Data model changes

- `waters.feed_mode`: `manual` (default) or `feed`.
- `feeds`: one per water. Recipe name, configuration (JSONB), source name and
  link, cleaning thresholds, state (`on`, `paused`, `off`), `attested_by`,
  `attested_at`, the statement text attested, `last_polled_at`,
  `last_success_at`, `last_good_reading_at`, `last_error`, `created_by`,
  `created_at`.
- `feed_requests`: page URL, notes, pasted AI summary, requester, water (if
  one exists yet), status (`open`, `done`, `declined`), timestamps.
- `readings`:
  - `submitted_by` becomes nullable;
  - `source` is `manual` or `feed`, plus `feed_id`;
  - a unique key on (`feed_id`, `measured_at`) among standing feed rows
    (a partial index, `WHERE deleted_at IS NULL`);
  - `flagged` and `flag_reason`;
  - `flag_override` (`accept` or `reject`), with `override_by` and
    `override_at`.

  Soft delete and corrections are unchanged.
- The snapshot payload gains each chart reading's flag state and reason, the
  feed credit, and the quiet/paused state.

## 6. The AI prompt (draft copy for the setup page)

> I'm setting up an automatic water-temperature feed on swimtemps.
> Here is my venue's page: <URL>. Please find out where the temperatures
> on that page come from: open the page and look for a JSON or API request
> behind any chart or live reading. If there is one, give me:
> the endpoint URL; whether the data is a list of records (rows) or separate
> lists of times and values (columns); where the list or lists sit in the
> JSON; which field is the time and which the temperature; the time format
> (epoch milliseconds, ISO, or a pattern like %d-%b-%Y); and the unit.
> If the readings are typed in by staff rather than produced by a sensor or
> logger, or there is no JSON or API, say so. If the site's terms forbid
> automated access, say that too.

## 7. Left to the builder

- Path syntax implementation, the recipe validation errors' wording, and
  the preview's layout beyond "same as paste".
- How the chart greys flagged points and shows overrules.
- Whether quiet-feed and paused states live in the snapshot or are computed
  at render time from `feeds` (the page must still never fetch).
- Request-handling UI for superusers: a plain list is enough.
- Tests: the recipe against both recorded fixtures; the cleaning rules
  against the recorded Clevedon fault; a venue correction becoming a
  soft-delete plus a new row; the 12-hour, 3-day and 5-day transitions,
  including a feed that keeps sending only flagged rows; pausing; switching
  modes both ways.

Also settled, as the builder proposed: setting up a feed is for the water's
admins and superusers, and any signed-in user can submit a feed request.
Feed URLs are refused if they resolve to private or internal addresses. The
setup page's Test button fetches on a POST, and the test that pages never
call the network is extended to cover it. Date-only feed rows are matched on
local date. Cleaning compares against unflagged neighbours.
