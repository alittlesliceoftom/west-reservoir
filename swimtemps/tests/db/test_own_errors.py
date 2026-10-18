import pandas as pd
import pytest

from db import queries
from db.models import ForecastPoint, ForecastRun, ModelFit, Reading
from tests.factories import add_reading, make_user, make_water

DAY0 = pd.Timestamp("2026-06-01 06:00", tz="UTC")


def day(n, hours=0):
    return (DAY0 + pd.Timedelta(days=n, hours=hours)).to_pydatetime()


def run(db, water, fit, created, anchor, points):
    r = ForecastRun(water_id=water.id, created_at=created, fit_id=fit.id,
                    anchor_reading_id=anchor.id, anchor_at=anchor.measured_at, anchor_temp=anchor.temp_c)
    db.add(r)
    db.flush()
    for valid_at, temp in points:
        db.add(ForecastPoint(run_id=r.id, valid_at=valid_at, water_temp=temp, lower=temp - 1, upper=temp + 1))


def scored_water(db, days=25):
    """
    Daily readings. After each one, the forecast as first issued misses the
    next day by 0.5 and the day after by 1.0; hourly re-runs made later, with
    weather that has since happened, miss by only 0.1.
    """
    user = make_user(db)
    water = make_water(db, user)
    fit = ModelFit(water_id=water.id, k_air=0.01, k_solar=1e-4, k_cool=0.01, scale=1, n_legs=0, method="prior")
    db.add(fit)
    db.flush()
    temps = {d: 10.0 + 0.1 * d for d in range(days + 1)}
    readings = {d: add_reading(db, water, user, day(d) + pd.Timedelta(minutes=10), t) for d, t in temps.items()}
    for d in range(days):
        ahead = [n for n in (d + 1, d + 2) if n in temps]
        issued = [(day(n), temps[n] + (0.5 if n == d + 1 else -1.0)) for n in ahead]
        run(db, water, fit, day(d, hours=1), readings[d], issued)
        run(db, water, fit, day(d, hours=20), readings[d], [(day(n), temps[n] + 0.1) for n in ahead])
    db.commit()
    return water


def test_each_forecast_is_scored_as_first_issued(db):
    water = scored_water(db)
    own = queries.own_errors(db, water.id, min_points=20)
    assert own == {1: pytest.approx(0.5), 2: pytest.approx(1.0)}


def test_too_few_scored_points_gives_none(db):
    water = scored_water(db)
    assert queries.own_errors(db, water.id, min_points=1000) is None


def test_a_horizon_with_ten_points_is_used(db):
    water = scored_water(db, days=11)   # 11 points one day ahead, 10 two days ahead
    assert set(queries.own_errors(db, water.id, min_points=20)) == {1, 2}


def test_a_horizon_with_fewer_than_ten_points_falls_back_to_the_base(db):
    water = scored_water(db, days=10)   # 10 and 9
    assert set(queries.own_errors(db, water.id, min_points=19)) == {1}


def test_a_water_with_no_runs_gives_none(db):
    water = make_water(db, make_user(db), name="Empty")
    assert queries.own_errors(db, water.id, min_points=20) is None


def test_flagged_readings_are_not_scored(db):
    water = scored_water(db)
    db.query(Reading).filter(Reading.water_id == water.id).update({"flagged": True})
    db.commit()
    assert queries.own_errors(db, water.id, min_points=1) is None
