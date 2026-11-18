import hashlib
import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from tests.factories import make_user, make_water
from tests.web.conftest import NOW, store_snapshot
from web.app import create_app

STATIC = Path(__file__).parents[2] / "web" / "static"
TEMPLATES = Path(__file__).parents[2] / "web" / "templates"


def short_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def asset_urls(html: str) -> dict[str, str]:
    """Each /static/ CSS or JS file the page references, mapped to its version."""
    return dict(re.findall(r'/static/([\w.-]+\.(?:css|js))\?v=([0-9a-f]+)', html))


def test_water_page_assets_carry_the_hash_of_their_contents(client, db):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    urls = asset_urls(client.get("/w/test-lake").text)
    for name in ("app.css", "app.js", "chart.js"):
        assert urls[name] == short_hash(STATIC / name)


def test_the_hash_changes_when_the_file_changes(settings, session_factory, db, tmp_path):
    water = make_water(db, make_user(db), needs_recompute=False)
    store_snapshot(db, water)
    static = tmp_path / "static"
    shutil.copytree(STATIC, static)

    def versions():
        app = create_app(settings=settings, session_factory=session_factory, clock=lambda: NOW, static_dir=static)
        with TestClient(app) as c:
            return asset_urls(c.get("/w/test-lake").text)

    before = versions()
    (static / "app.css").write_text((static / "app.css").read_text() + "\n/* changed */\n")
    after = versions()
    assert after["app.css"] != before["app.css"]
    assert after["app.css"] == short_hash(static / "app.css")
    assert after["app.js"] == before["app.js"]


def test_no_template_references_an_unversioned_css_or_js_file():
    for template in TEMPLATES.rglob("*.html"):
        assert not re.search(r'/static/[\w.-]+\.(?:css|js)"', template.read_text()), template.name
