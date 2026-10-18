# One image for both the web app and the worker; compose picks the command.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini ./
COPY db ./db
COPY feeds ./feeds
COPY forecast ./forecast
COPY web ./web
COPY worker ./worker

RUN useradd --system --uid 10001 swimtemps
USER swimtemps

EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
