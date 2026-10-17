"""
Cold-start replay: West Res history replayed as if it were a new water.

For each start point, the model sees only the first N readings from there,
fits, forecasts 1-5 days from the last of them, and is scored against what
was measured. Repeated for each fit method, each penalty weight `lambda`, and
each prior scale (1.0 is the right depth guess for West Res; 0.5 and 2.0 are
a user picking the wrong depth class).

A "mature" pass fits on every reading before each anchor, which is what a
long-established water gets; its 80th-percentile error by horizon is the
base of the uncertainty band.

    python -m forecast.replay            # writes docs/cold-start-replay.md
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.model import (
    build_legs,
    choose_method,
    fit_legs,
    forecast_points,
    prepare_weather,
)
from forecast.physics import load_params

DATA = Path(__file__).with_name("data")
REPORT = Path(__file__).parents[1] / "docs" / "cold-start-replay.md"
HORIZONS = (1, 2, 3, 4, 5)
MATCH_TOLERANCE = pd.Timedelta(minutes=90)

# The replay drives the model with measured weather, so its errors leave out
# the error in the weather forecast itself. West Res's live forecasts, scored
# against its readings, have these MAEs (°C) by horizon; the band is scaled to
# them, with the horizons between interpolated linearly.
PRODUCTION_MAE = {1: 0.28, 5: 1.31}

# Lambdas whose cold-start error is within this of the best count as tied;
# the largest tied value is chosen, as the most robust to a mistyped reading.
LAMBDA_TIE = 0.01


def load_westres() -> tuple[pd.DataFrame, pd.DataFrame]:
    readings = pd.read_csv(DATA / "westres_readings.csv", parse_dates=["measured_at"])
    weather = pd.read_csv(DATA / "westres_weather.csv", parse_dates=["valid_at"])
    return readings, prepare_weather(weather)


def _actuals(readings: pd.DataFrame, anchor_at: pd.Timestamp) -> dict[int, float]:
    """Measured value at each horizon's target time, where a reading lies within tolerance."""
    out = {}
    for h in HORIZONS:
        target = anchor_at + pd.Timedelta(days=h)
        near = readings[(readings.measured_at - target).abs() <= MATCH_TOLERANCE]
        if not near.empty:
            out[h] = float(near.iloc[0].temp_c)
    return out


def _errors(fit, anchor, readings, weather) -> dict[int, float]:
    actual = _actuals(readings, anchor.measured_at)
    targets = [anchor.measured_at + pd.Timedelta(days=h) for h in actual]
    points = forecast_points(anchor.measured_at, anchor.temp_c, weather, fit.coefficients, targets)
    return {
        h: value - actual[h]
        for h, (_, value) in zip(actual, points)
        if value is not None
    }


def run(
    readings: pd.DataFrame,
    weather: pd.DataFrame,
    lambdas=(0.1, 0.3, 1.0, 3.0, 10.0),
    counts=(3, 5, 10, 20, 30, 50, 100),
    s_priors=(0.5, 1.0, 2.0),
    methods=("prior", "scale", "full"),
    stride: int = 12,
    params: dict | None = None,
) -> pd.DataFrame:
    """One row per (start, count, method, lambda, s_prior, horizon) with the signed error."""
    params = params or load_params()
    readings = readings.sort_values("measured_at").reset_index(drop=True)
    rows = []
    last_start = len(readings) - max(counts) - max(HORIZONS)
    for start in range(0, max(last_start, 1), stride):
        for n in counts:
            train = readings.iloc[start:start + n]
            anchor = train.iloc[-1]
            legs = build_legs(train, weather, params["max_leg_days"], params["min_leg_coverage"])
            for method, lam, s_prior in product(methods, lambdas, s_priors):
                if method == "prior" and lam != lambdas[0]:
                    continue  # the prior does not depend on lambda
                fit = fit_legs(legs, s_prior, {**params, "lambda": lam}, method=method)
                for h, error in _errors(fit, anchor, readings, weather).items():
                    rows.append({
                        "start": start, "count": n, "legs": legs.count, "method": method,
                        "lambda": lam if method != "prior" else np.nan,
                        "s_prior": s_prior, "horizon": h, "error": error,
                    })
    return pd.DataFrame(rows)


def run_mature(readings: pd.DataFrame, weather: pd.DataFrame, min_readings=60, stride=3, params=None) -> pd.DataFrame:
    """Errors of a model fitted on all readings before each anchor, auto method."""
    params = params or load_params()
    readings = readings.sort_values("measured_at").reset_index(drop=True)
    rows = []
    for end in range(min_readings, len(readings) - 1, stride):
        train = readings.iloc[:end]
        legs = build_legs(train, weather, params["max_leg_days"], params["min_leg_coverage"])
        fit = fit_legs(legs, 1.0, params)
        for h, error in _errors(fit, train.iloc[-1], readings, weather).items():
            rows.append({"end": end, "method": fit.method, "horizon": h, "error": error})
    return pd.DataFrame(rows)


def production_mae() -> pd.Series:
    known = pd.Series(PRODUCTION_MAE, dtype=float)
    return known.reindex(HORIZONS).interpolate()


def calibrate_band(p80_by_h: pd.Series, mae_by_h: pd.Series) -> pd.Series:
    """Keep the replay's spread (p80 / MAE) but at production's error level."""
    return p80_by_h / mae_by_h * production_mae()


def p80(errors: pd.Series) -> float:
    return float(np.percentile(np.abs(errors), 80))


def _table(frame: pd.DataFrame, fmt="{:.2f}") -> str:
    header = "| " + " | ".join(str(c) for c in [frame.index.name or ""] + list(frame.columns)) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    body = [
        "| " + " | ".join([str(idx)] + [fmt.format(v) if pd.notna(v) else "" for v in row]) + " |"
        for idx, row in zip(frame.index, frame.to_numpy())
    ]
    return "\n".join([header, rule, *body])


def choose(cold: pd.DataFrame, params: dict) -> dict:
    """
    Pick lambda and the method thresholds from the replay.

    lambda: lowest mean |error| for the scale method over the cold-start
    counts, averaged over right and wrong depth guesses, ties going to the
    larger value. Thresholds: the
    smallest reading count from which a method beats the one before it on
    mean |error| at that count and every larger one, expressed in legs (the
    median legs at that count). With no sustained win the current value stands.
    """
    cold = cold.assign(abs_error=cold.error.abs())
    scale = cold[cold.method == "scale"]
    by_lambda = scale.groupby("lambda").abs_error.mean()
    lam = float(by_lambda[by_lambda <= by_lambda.min() + LAMBDA_TIE].index.max())
    tuned = cold[(cold.method == "prior") | (cold["lambda"] == lam)]
    by = tuned.groupby(["count", "method"]).abs_error.mean().unstack()
    legs_at = cold.groupby("count").legs.median()

    def first_count(better, worse):
        """Smallest count from which `better` wins at that count and every larger one."""
        wins = (by[better] < by[worse]).sort_index()
        sustained = wins[::-1].cummin()[::-1]
        return int(sustained.index[sustained].min()) if sustained.any() else None

    scale_from = first_count("scale", "prior")
    full_from = first_count("full", "scale")
    return {
        "lambda": lam,
        "scale_min_legs": max(1, int(legs_at[scale_from])) if scale_from else params["scale_min_legs"],
        "full_min_legs": int(legs_at[full_from]) if full_from else params["full_min_legs"],
        "full_from_replay": full_from is not None,
        "by_count_method": by,
    }


def report(cold: pd.DataFrame, mature: pd.DataFrame, params: dict) -> tuple[str, dict]:
    chosen = choose(cold, params)
    lam = chosen["lambda"]
    cold = cold.assign(abs_error=cold.error.abs())
    tuned = cold[(cold.method == "prior") | (cold["lambda"] == lam)]

    base = mature.groupby("horizon").error.apply(p80)
    mature_mae = mature.assign(a=mature.error.abs()).groupby("horizon").a.mean()
    calibrated = calibrate_band(base, mature_mae)

    lam_table = cold[cold.method == "scale"].pivot_table(
        index="lambda", columns="count", values="abs_error", aggfunc="mean")
    lam_table.columns = [f"n={c}" for c in lam_table.columns]

    by_count = chosen["by_count_method"].copy()
    by_count.index = [f"{c} ({int(cold[cold['count'] == c].legs.median())} legs)" for c in by_count.index]
    by_count.index.name = "readings"

    wrong = tuned[tuned.method != "prior"].pivot_table(
        index="s_prior", columns="count", values="abs_error", aggfunc="mean")
    wrong.columns = [f"n={c}" for c in wrong.columns]

    def auto_method(n_legs):
        return choose_method(n_legs, {**params, **{k: chosen[k] for k in ("scale_min_legs", "full_min_legs")}})

    auto = tuned[tuned.apply(lambda r: r.method == auto_method(r.legs), axis=1)]
    auto_p80 = auto.groupby(["count", "horizon"]).error.apply(p80).unstack()
    auto_p80.columns = [f"{h}d" for h in auto_p80.columns]

    ratios = {}
    for method in ("prior", "scale"):
        sub = auto[auto.method == method]
        if not sub.empty:
            ratios[method] = float(np.mean([
                p80(sub[sub.horizon == h].error) / base[h] for h in HORIZONS if (sub.horizon == h).any()
            ]))

    base_table = pd.DataFrame({
        "p80 |error|": base, "MAE": mature_mae,
        "production MAE": production_mae(), "band_base": calibrated,
    })
    base_table.index.name = "horizon (days)"

    lines = [
        "# Cold-start replay",
        "",
        "Generated by `python -m forecast.replay` over the bundled West Res record "
        f"({cold.start.nunique()} start points; mature pass over {mature.end.nunique()} anchors).",
        "Errors are forecast minus measured, °C. A forecast is scored only where a reading",
        "exists within 90 minutes of its target.",
        "",
        "## Chosen values",
        "",
        f"- `lambda` = {lam}",
        f"- `scale_min_legs` = {chosen['scale_min_legs']}",
        f"- `full_min_legs` = {chosen['full_min_legs']}"
        + ("" if chosen["full_from_replay"] else " (kept: the full fit never sustained a win over the scale fit)"),
        "- `band_base` = " + ", ".join(f"{h}d {calibrated[h]:.2f}" for h in HORIZONS),
        "",
        "## Mean |error| by readings available and method (chosen lambda)",
        "",
        _table(by_count),
        "",
        "## Scale fit: mean |error| by lambda and readings available",
        "",
        _table(lam_table),
        "",
        "## Wrong depth guesses: mean |error| of the fitted methods by prior scale",
        "",
        "West Res is the reference, so s_prior = 1.0 is the right guess. 0.5 and 2.0 are",
        "a user choosing the wrong depth class; the fit should recover as readings grow.",
        "",
        _table(wrong),
        "",
        "## What a new water sees: 80% |error| by readings and horizon (auto method)",
        "",
        _table(auto_p80),
        "",
        "## Band base: mature model by horizon",
        "",
        "The replay runs on measured weather, so it misses weather-forecast error. Its",
        "ratio of 80th-percentile to mean error is kept and applied to the MAE West Res",
        "actually achieves on forecast weather: band_base = p80 / MAE x production MAE.",
        "",
        _table(base_table),
        "",
        "## Method multipliers",
        "",
        "Spec multipliers are prior x1.5, scale x1.25. Measured ratio of the cold-start",
        "80% error to the mature base, averaged over horizons:",
        "",
        *[f"- {m}: x{r:.2f}" for m, r in ratios.items()],
        "",
        "West Res's prior is West Res's own fit, so the s_prior = 1.0 rows flatter",
        "the prior: a new water's real prior error is closer to the wrong-guess rows.",
        "",
    ]
    band = {str(h): round(float(calibrated[h]), 2) for h in HORIZONS}
    return "\n".join(lines), {**chosen, "band_base": band, "ratios": ratios}


def main(argv=None) -> int:
    readings, weather = load_westres()
    params = load_params()
    print(f"replaying {len(readings)} readings", file=sys.stderr)
    cold = run(readings, weather, params=params)
    mature = run_mature(readings, weather, params=params)
    text, chosen = report(cold, mature, params)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(text)
    print(text)
    print({k: v for k, v in chosen.items() if k != "by_count_method"}, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
