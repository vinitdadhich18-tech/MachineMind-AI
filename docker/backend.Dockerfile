# MachineMind AI - Flask Application Layer Image
FROM python:3.11-slim

# Prevent bytecode generation and configure unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal system utilities for healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend and ml-service modules (for ML inference engine)
COPY backend /app/backend
COPY ml-service /app/ml-service

# Expose Flask API default port
EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:5000/api/health || exit 1

CMD ["python", "backend/app.py"]
