"""Jinja environment and the one render helper every route uses."""

import json
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from web import formatting

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))
TEMPLATES.env.filters["localtime"] = formatting.localtime
TEMPLATES.env.filters["temp"] = formatting.temp
TEMPLATES.env.globals["WATER_TYPES"] = {"still": "Lake or reservoir", "river": "River", "harbour": "Harbour"}
TEMPLATES.env.globals["DEPTH_CLASSES"] = {
    "shallow": "Shallow (under 2 m)", "lake": "Medium (2-5 m)", "deep": "Deep (over 5 m)", "unknown": "Not sure",
}


def json_script(data) -> Markup:
    """JSON safe to place inside a <script> element."""
    return Markup(json.dumps(data).replace("</", "<\\/"))


TEMPLATES.env.filters["json_script"] = json_script


def render(request: Request, name: str, context: dict | None = None, status_code: int = 200, headers=None):
    now = request.app.state.clock()
    base = {
        "now": now,
        "ago": lambda value: formatting.ago(value, now),
        "user": getattr(request.state, "user", None),
        "csrf_token": getattr(request.state, "csrf_token", ""),
        "settings": request.app.state.settings,
    }
    return TEMPLATES.TemplateResponse(request, name, {**base, **(context or {})},
                                      status_code=status_code, headers=headers)
