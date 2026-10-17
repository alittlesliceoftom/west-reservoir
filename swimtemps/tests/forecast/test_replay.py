import pandas as pd
import pytest

from forecast import replay
from forecast.model import prepare_weather
from forecast.physics import load_params, west_res_coefficients
from tests.forecast.synthetic import synthetic_readings, synthetic_weather


@pytest.fixture(scope="module")
def synthetic():
    weather = prepare_weather(synthetic_weather(60).reset_index())
    readings = synthetic_readings(weather, west_res_coefficients().scaled(1.5), noise=0.05)
    return readings, weather


def test_run_scores_every_method_at_every_horizon(synthetic):
    readings, weather = synthetic
    cold = replay.run(readings, weather, lambdas=(0.3, 3.0), counts=(3, 10), s_priors=(1.0,), stride=20)
    assert set(cold.method) == {"prior", "scale", "full"}
    assert set(cold.horizon) == {1, 2, 3, 4, 5}
    assert cold.error.notna().all()
    # The prior does not depend on lambda, so it is run once per start and count.
    prior = cold[cold.method == "prior"]
    assert prior.groupby(["start", "count", "horizon"]).size().max() == 1


def test_fitting_beats_a_wrong_prior_on_data_from_a_different_water(synthetic):
    readings, weather = synthetic
    cold = replay.run(readings, weather, lambdas=(0.3,), counts=(10,), s_priors=(1.0,), stride=10)
    mae = cold.assign(a=cold.error.abs()).groupby("method").a.mean()
    assert mae["scale"] < mae["prior"]


def test_report_names_the_chosen_values(synthetic):
    readings, weather = synthetic
    params = load_params()
    cold = replay.run(readings, weather, lambdas=(0.3, 3.0), counts=(3, 5, 10), s_priors=(1.0, 2.0), stride=20)
    mature = replay.run_mature(readings, weather, min_readings=20, stride=10)
    text, chosen = replay.report(cold, mature, params)
    assert "# Cold-start replay" in text
    assert chosen["lambda"] in (0.3, 3.0)
    assert set(chosen["band_base"]) == {"1", "2", "3", "4", "5"}
    assert all(v > 0 for v in chosen["band_base"].values())


def test_actuals_match_within_ninety_minutes():
    anchor = pd.Timestamp("2026-06-01 07:00", tz="UTC")
    readings = pd.DataFrame({
        "measured_at": [anchor + pd.Timedelta(days=1, minutes=80), anchor + pd.Timedelta(days=2, hours=3)],
        "temp_c": [12.0, 13.0],
    })
    assert replay._actuals(readings, anchor) == {1: 12.0}


@pytest.mark.slow
def test_full_west_res_replay_writes_its_report():
    assert replay.main() == 0
    assert replay.REPORT.exists()


def test_lambda_ties_go_to_the_larger_value():
    rows = []
    for lam, err in [(0.1, 0.400), (0.3, 0.405), (1.0, 0.45)]:
        for count in (3, 5):
            for method in ("scale", "full"):
                rows.append({"start": 0, "count": count, "legs": count - 1, "method": method,
                             "lambda": lam, "s_prior": 1.0, "horizon": 1, "error": err})
            rows.append({"start": 0, "count": count, "legs": count - 1, "method": "prior",
                         "lambda": float("nan"), "s_prior": 1.0, "horizon": 1, "error": 0.5})
    assert replay.choose(pd.DataFrame(rows), load_params())["lambda"] == 0.3


def test_band_is_calibrated_to_production_error_keeping_the_replay_spread():
    p80 = pd.Series({1: 0.5, 2: 0.6, 3: 0.6, 4: 0.8, 5: 1.0})
    mae = pd.Series({1: 0.25, 2: 0.4, 3: 0.4, 4: 0.5, 5: 0.5})
    band = replay.calibrate_band(p80, mae)
    assert band[1] == pytest.approx(2.0 * 0.28)
    assert band[5] == pytest.approx(2.0 * 1.31)
    assert band[3] == pytest.approx(1.5 * (0.28 + 1.31) / 2)


def test_a_single_lucky_count_does_not_move_a_threshold():
    rows = []
    table = {3: (0.5, 0.6, 0.4), 10: (0.5, 0.6, 0.6), 30: (0.5, 0.6, 0.55)}  # scale, prior, full
    for count, (scale, prior, full) in table.items():
        for method, err in (("scale", scale), ("prior", prior), ("full", full)):
            rows.append({"start": 0, "count": count, "legs": count - 1, "method": method,
                         "lambda": float("nan") if method == "prior" else 0.3,
                         "s_prior": 1.0, "horizon": 1, "error": err})
    params = load_params()
    chosen = replay.choose(pd.DataFrame(rows), params)
    assert chosen["scale_min_legs"] == 2           # scale wins from 3 readings onward
    assert chosen["full_min_legs"] == params["full_min_legs"]  # full won only at 3
    assert chosen["full_from_replay"] is False
