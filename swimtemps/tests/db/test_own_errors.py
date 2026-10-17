import pandas as pd
import pytest

from db import queries
from db.models import ForecastPoint, ForecastRun, ModelFit
from tests.factories import add_reading, make_user, make_water

DAY0 = pd.Timestamp("2026-06-01 06:00", tz="UTC")


def day(n, hours=0):
    return (DAY0 + pd.Timedelta(days=n, hours=hours)).to_pydatetime()


def run(db, water, fit, created, anchor, points):
    r = ForecastRun(water_id=water.id, created_at=created, fit_id=fit.id, anchor_at=anchor, anchor_temp=10.0)
    db.add(r)
    db.flush()
    for valid_at, temp in points:
        db.add(ForecastPoint(run_id=r.id, valid_at=valid_at, water_temp=temp, lower=temp - 1, upper=temp + 1))


@pytest.fixture
def scored_water(db):
    user = make_user(db)
    water = make_water(db, user)
    fit = ModelFit(water_id=water.id, k_air=0.01, k_solar=1e-4, k_cool=0.01, scale=1, n_legs=0, method="prior")
    db.add(fit)
    db.flush()
    temps = {d: 10.0 + 0.1 * d for d in range(1, 26)}
    for d, t in temps.items():
        add_reading(db, water, user, day(d, hours=0) + pd.Timedelta(minutes=10), t)
    for d in range(1, 26):
        # A stale run from earlier that morning, way off: superseded by the next one.
        run(db, water, fit, day(d, hours=-3), day(d - 1), [(day(d), temps[d] + 5.0)])
        points = [(day(d), temps[d] + 0.5)]
        if d + 1 in temps:
            points.append((day(d + 1), temps[d + 1] - 1.0))
        run(db, water, fit, day(d, hours=-1), day(d - 1), points)
    db.commit()
    return water


def test_each_reading_is_scored_against_the_latest_run_before_it(db, scored_water):
    own = queries.own_errors(db, scored_water.id, min_points=20)
    assert own == {1: pytest.approx(0.5), 2: pytest.approx(1.0)}


def test_too_few_scored_points_gives_none(db, scored_water):
    assert queries.own_errors(db, scored_water.id, min_points=1000) is None


def test_a_water_with_no_runs_gives_none(db):
    water = make_water(db, make_user(db))
    assert queries.own_errors(db, water.id, min_points=20) is None
