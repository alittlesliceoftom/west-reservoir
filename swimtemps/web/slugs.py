"""URL slugs for waters: /w/<slug>."""

import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Water

# Letters NFKD does not decompose into ASCII.
FOLD = str.maketrans({"æ": "ae", "Æ": "ae", "ø": "o", "Ø": "o", "ß": "ss", "œ": "oe", "Œ": "oe",
                      "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "þ": "th", "Þ": "th"})
RESERVED = {"new", "admin", "api", "auth", "me", "static"}
MAX_LENGTH = 60


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name.translate(FOLD)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:MAX_LENGTH].strip("-")
    return slug or "water"


def slugify_unique(db: Session, name: str) -> str:
    """A slug no water (archived ones included) already has: `lido`, then `lido-2`, `lido-3`..."""
    base = slugify(name)
    taken = set(db.scalars(select(Water.slug).where(Water.slug.like(f"{base}%"))))
    taken |= RESERVED
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"
