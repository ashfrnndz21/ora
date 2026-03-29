FROM python:3.12-slim

WORKDIR /app

# Install system deps needed for native extensions (psycopg2, mysqlclient, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first (layer-cache friendly)
COPY pyproject.toml README.md ./
COPY sqlagent/ sqlagent/

# Install all connector extras so the image can talk to any supported database.
# Use --no-cache-dir to keep image size down.
# Snowflake and BigQuery add ~200 MB — use build args to opt out for smaller images.
ARG EXTRAS="postgres,mysql"
RUN pip install --no-cache-dir -e ".[$EXTRAS]"

# Non-root user for security
RUN useradd -m -u 1000 sqlagent && chown -R sqlagent:sqlagent /app
USER sqlagent

# Data directory for SQLite checkpoints and JWT secret
RUN mkdir -p /home/sqlagent/.sqlagent

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "sqlagent.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
