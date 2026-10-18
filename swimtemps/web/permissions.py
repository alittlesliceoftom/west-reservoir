"""Who may do what. Any signed-in user may add readings; admins also correct them."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import User, Water, WaterMember


def can_admin(db: Session, user: User | None, water: Water) -> bool:
    """The water's admins, and the site's superusers."""
    if user is None:
        return False
    if user.is_superuser:
        return True
    role = db.scalar(select(WaterMember.role).where(WaterMember.water_id == water.id, WaterMember.user_id == user.id))
    return role == "admin"
