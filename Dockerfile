# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# System deps (keep minimal). gcc only if your deps need compilation; remove if not needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir gunicorn==21.2.0

# Copy app source
COPY . /app

# Cloud Run uses 8080 by default
EXPOSE 8080

# Default command = API service.
# Cloud Run Jobs will override this with: python worker.py process|mine
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "2", "--threads", "8", "--timeout", "0", "app:create_app()"]