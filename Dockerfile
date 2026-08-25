FROM python:3.11-slim@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY finger_impedance/ ./finger_impedance/

RUN uv sync --frozen --no-dev --no-editable --extra all

ENV PATH="/app/.venv/bin:$PATH"

COPY scripts/ ./scripts/
COPY examples/ ./examples/

CMD ["python"]
