"""
Operator commands.

    python -m db.cli make-superuser EMAIL
"""

import sys

from sqlalchemy import func, select

from db.models import User
from db.session import session_factory


def make_superuser(factory, email: str) -> str:
    email = email.strip().lower()
    with factory() as session:
        user = session.scalar(select(User).where(func.lower(User.email) == email))
        if user is None:
            return f"no user with email {email}: they must sign in once first"
        user.is_superuser = True
        session.commit()
        return f"{email} is now a superuser"


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "make-superuser":
        print(__doc__.strip())
        return 2
    print(make_superuser(session_factory(), argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
