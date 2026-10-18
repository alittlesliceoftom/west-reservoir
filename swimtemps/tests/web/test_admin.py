from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from db import queries
from db.models import Reading, User, Water, WaterMember
from tests.factories import add_reading, make_user, make_water
from tests.web.helpers import post, sign_in


@pytest.fixture
def admin(db):
    return make_user(db, email="admin@example.com", name="Admin")


@pytest.fixture
def swimmer(db):
    return make_user(db, email="swimmer@example.com", name="Swimmer")


@pytest.fixture
def water(db, admin):
    return make_water(db, admin, needs_recompute=False)


@pytest.fixture
def reading(db, water, swimmer):
    return add_reading(db, water, swimmer, "2026-10-17 06:00", 31.4)


def fresh(db, model, key):
    db.expire_all()
    return db.get(model, key)


# --- who may -----------------------------------------------------------------

def test_signed_out_visitors_are_sent_to_sign_in(client, water):
    assert client.get("/w/test-lake/admin", follow_redirects=False).headers["location"] == "/auth/login?next=/w/test-lake/admin"


def test_contributors_are_refused(client, db, water, swimmer, reading):
    sign_in(client, swimmer)
    assert client.get("/w/test-lake/admin").status_code == 403
    assert post(client, f"/w/test-lake/admin/readings/{reading.id}/delete").status_code == 403
    assert fresh(db, Reading, reading.id).deleted_at is None


def test_the_waters_admin_sees_its_readings(client, db, water, admin, reading):
    sign_in(client, admin)
    page = client.get("/w/test-lake/admin").text
    assert "31.4" in page and "Swimmer" in page and "Sat 17 Oct 07:00" in page


def test_superusers_may_administer_any_water(client, db, water, reading):
    boss = make_user(db, email="boss@example.com", superuser=True)
    sign_in(client, boss)
    assert client.get("/w/test-lake/admin").status_code == 200


def test_the_water_page_links_to_admin_only_for_admins(client, db, water, admin, swimmer):
    sign_in(client, swimmer)
    assert "/w/test-lake/admin" not in client.get("/w/test-lake").text
    sign_in(client, admin)
    assert "/w/test-lake/admin" in client.get("/w/test-lake").text


# --- corrections -------------------------------------------------------------

def test_a_correction_keeps_the_old_reading_and_points_it_at_the_new_one(client, db, water, admin, reading):
    sign_in(client, admin)
    response = post(client, f"/w/test-lake/admin/readings/{reading.id}/correct",
                    {"temp": "13.4", "measured_at": "2026-10-17T07:00"})
    assert response.status_code == 303
    old = fresh(db, Reading, reading.id)
    new = db.get(Reading, old.replaced_by)
    assert old.deleted_at is not None and old.deleted_by == admin.id
    assert (new.temp_c, new.measured_at, new.deleted_at) == (13.4, datetime(2026, 10, 17, 6, tzinfo=timezone.utc), None)
    assert new.submitted_by == admin.id
    assert queries.active_readings(db, water.id).temp_c.tolist() == [13.4]
    assert fresh(db, Water, water.id).needs_recompute is True


def test_a_correction_can_move_the_time(client, db, water, admin, reading):
    sign_in(client, admin)
    post(client, f"/w/test-lake/admin/readings/{reading.id}/correct", {"temp": "31.4", "measured_at": "2026-10-16T07:00"})
    assert queries.active_readings(db, water.id).measured_at.tolist() == [datetime(2026, 10, 16, 6, tzinfo=timezone.utc)]


@pytest.mark.parametrize("temp", ["41", "abc"])
def test_a_bad_correction_changes_nothing(client, db, water, admin, reading, temp):
    sign_in(client, admin)
    response = post(client, f"/w/test-lake/admin/readings/{reading.id}/correct", {"temp": temp, "measured_at": "2026-10-17T07:00"})
    assert response.status_code == 422
    assert fresh(db, Reading, reading.id).deleted_at is None
    assert db.scalar(select(Reading).where(Reading.id != reading.id)) is None


def test_a_deleted_reading_cannot_be_corrected_again(client, db, water, admin, reading):
    sign_in(client, admin)
    post(client, f"/w/test-lake/admin/readings/{reading.id}/delete")
    assert post(client, f"/w/test-lake/admin/readings/{reading.id}/correct",
                {"temp": "13.0", "measured_at": "2026-10-17T07:00"}).status_code == 404


def test_another_waters_reading_is_not_reachable(client, db, water, admin, swimmer):
    other = make_water(db, swimmer, name="Other")
    theirs = add_reading(db, other, swimmer, "2026-10-17 06:00", 12.0)
    sign_in(client, admin)
    assert post(client, f"/w/test-lake/admin/readings/{theirs.id}/delete").status_code == 404
    assert fresh(db, Reading, theirs.id).deleted_at is None


# --- deletion ----------------------------------------------------------------

def test_delete_is_soft_and_flags_the_water(client, db, water, admin, reading):
    sign_in(client, admin)
    assert post(client, f"/w/test-lake/admin/readings/{reading.id}/delete").status_code == 303
    row = fresh(db, Reading, reading.id)
    assert row is not None and row.deleted_at is not None and row.deleted_by == admin.id
    assert queries.active_readings(db, water.id).empty
    assert fresh(db, Water, water.id).needs_recompute is True
    assert "31.4" not in client.get("/w/test-lake/admin").text


# --- details, members, archive ----------------------------------------------

def test_editing_details_keeps_the_slug_and_flags_the_water(client, db, water, admin):
    sign_in(client, admin)
    response = post(client, "/w/test-lake/admin/details", {
        "name": "Test Lake (north)", "water_type": "river", "depth_class": "deep", "reading_time": "06:30"})
    assert response.status_code == 303
    row = fresh(db, Water, water.id)
    assert (row.name, row.slug, row.water_type, row.depth_class) == ("Test Lake (north)", "test-lake", "river", "deep")
    assert row.default_reading_time.strftime("%H:%M") == "06:30"
    assert row.needs_recompute is True


def test_bad_details_are_refused(client, db, water, admin):
    sign_in(client, admin)
    response = post(client, "/w/test-lake/admin/details",
                    {"name": "", "water_type": "sea", "depth_class": "deep", "reading_time": "06:30"})
    assert response.status_code == 422
    assert fresh(db, Water, water.id).name == "Test Lake"


def test_promoting_a_user_by_email_makes_them_an_admin(client, db, water, admin, swimmer):
    sign_in(client, admin)
    assert post(client, "/w/test-lake/admin/members", {"email": " Swimmer@Example.com "}).status_code == 303
    assert fresh(db, WaterMember, (water.id, swimmer.id)).role == "admin"
    sign_in(client, swimmer)
    assert client.get("/w/test-lake/admin").status_code == 200


def test_promoting_an_unknown_email_explains(client, db, water, admin):
    sign_in(client, admin)
    response = post(client, "/w/test-lake/admin/members", {"email": "nobody@example.com"})
    assert response.status_code == 422
    assert "has not signed in to swimtemps yet" in response.text


def test_promoting_twice_is_harmless(client, db, water, admin, swimmer):
    sign_in(client, admin)
    post(client, "/w/test-lake/admin/members", {"email": "swimmer@example.com"})
    assert post(client, "/w/test-lake/admin/members", {"email": "swimmer@example.com"}).status_code == 303


def test_archiving_hides_the_water(client, db, water, admin):
    sign_in(client, admin)
    assert post(client, "/w/test-lake/admin/archive").headers["location"] == "/"
    assert fresh(db, Water, water.id).archived_at is not None
    assert client.get("/w/test-lake").status_code == 404
    assert "Test Lake" not in client.get("/").text


# --- /me -------------------------------------------------------------------

def test_me_lists_my_waters_and_my_recent_readings(client, db, water, admin, swimmer, reading):
    other = make_water(db, swimmer, name="Swimmer's Pond")
    add_reading(db, other, swimmer, "2026-10-16 06:00", 12.5)
    sign_in(client, swimmer)
    page = client.get("/me").text
    assert "Swimmer&#39;s Pond" in page or "Swimmer's Pond" in page
    assert "31.4" in page and "12.5" in page and "Test Lake" in page


def test_me_needs_sign_in(client):
    assert client.get("/me", follow_redirects=False).status_code == 303


# --- superuser CLI -----------------------------------------------------------

def test_make_superuser_cli(db, session_factory, swimmer):
    from db import cli
    assert cli.make_superuser(session_factory, "SWIMMER@example.com") == "swimmer@example.com is now a superuser"
    assert fresh(db, User, swimmer.id).is_superuser is True
    assert "no user" in cli.make_superuser(session_factory, "ghost@example.com")


def test_a_water_name_cannot_inject_script_into_the_admin_page(client, db, admin):
    import html
    make_water(db, admin, name="Lido'+alert(document.domain)+'", slug="lido")
    sign_in(client, admin)
    page = html.unescape(client.get("/w/lido/admin").text)
    assert "confirm('Archive Lido'+alert" not in page
    assert "onsubmit" not in page
    assert 'data-confirm="Archive Lido' in client.get("/w/lido/admin").text.replace("&#39;", "'")
