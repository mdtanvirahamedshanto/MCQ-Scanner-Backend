# ---- Builder Stage ----
FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies and pyinstaller
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pyinstaller

# Copy application source code
COPY . .

# Run PyInstaller to bundle the application
# We use run.py as the entry point
# --add-data "alembic:alembic" -> Include alembic migrations folder
# --add-data "alembic.ini:." -> Include alembic config
# --hidden-import uvicorn
# --hidden-import app.main
RUN pyinstaller --noconfirm \
    --name mcq_scanner \
    --paths /app \
    --add-data "alembic:alembic" \
    --add-data "alembic.ini:." \
    --copy-metadata fastapi \
    --copy-metadata pydantic \
    --copy-metadata starlette \
    --copy-metadata uvicorn \
    --hidden-import "asyncpg" \
    --hidden-import "asyncpg.pgproto" \
    --hidden-import "passlib.handlers.bcrypt" \
    --hidden-import "jose.backends.cryptography_backend" \
    --hidden-import "cv2" \
    --hidden-import "easyocr" \
    --hidden-import "sqlalchemy.dialects.postgresql.asyncpg" \
    --hidden-import "sqlalchemy.sql.default_comparator" \
    --hidden-import "app.api.v1.endpoints" \
    run.py


# ---- Production Stage ----
FROM debian:bullseye-slim AS runner

WORKDIR /app

# Install standard OS runtime dependencies required by the compiled binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary directory from the builder stage
# PyInstaller outputs to dist/mcq_scanner
COPY --from=builder /app/dist/mcq_scanner /app/

# Expose port
EXPOSE 8000

# Set environment paths and run
ENV PATH="/app:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["./mcq_scanner"]
