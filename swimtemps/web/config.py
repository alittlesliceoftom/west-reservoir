"""Web settings, from the environment. See deploy/.env.example."""

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    env: str = "development"            # development | test | production
    site_url: str = "http://localhost:8000"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    # Set only for a Supabase project on legacy HS256 JWT secrets; otherwise
    # tokens are verified against the project's JWKS.
    supabase_jwt_secret: str = ""
    # A local sign-in form that skips Google. Refused in production.
    dev_login: bool = False

    @property
    def production(self) -> bool:
        return self.env == "production"

    @property
    def secure_cookies(self) -> bool:
        return self.site_url.startswith("https://")

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            env=os.environ.get("ENV", "development"),
            site_url=os.environ.get("SITE_URL", "http://localhost:8000").rstrip("/"),
            supabase_url=os.environ.get("SUPABASE_URL", "").rstrip("/"),
            supabase_anon_key=os.environ.get("SUPABASE_ANON_KEY", ""),
            supabase_jwt_secret=os.environ.get("SUPABASE_JWT_SECRET", ""),
            dev_login=os.environ.get("DEV_LOGIN", "") == "1",
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.production and self.dev_login:
            raise ConfigError("DEV_LOGIN=1 is refused when ENV=production.")
        if self.production and not (self.supabase_url and self.supabase_anon_key):
            raise ConfigError("SUPABASE_URL and SUPABASE_ANON_KEY are required in production.")
