import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_the_image_copies_every_package():
    copied = set(re.findall(r"^COPY (\S+) \./\1$", (ROOT / "Dockerfile").read_text(), re.M))
    packages = {p.parent.name for p in ROOT.glob("*/__init__.py")} - {"tests"}
    assert packages - copied == set()
