FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG INSTALL_DEV=0
COPY pyproject.toml README.md LICENSE ./
COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$INSTALL_DEV" = "1" ]; then pip install --no-cache-dir ".[dev]"; else pip install --no-cache-dir .; fi

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" || exit 1

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
