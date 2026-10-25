# MachineMind AI - Industrial MQTT & Telemetry Services Image
FROM python:3.11-slim

# Prevent bytecode generation and configure unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/ml-service:/app

WORKDIR /app

# Install minimal system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MQTT & signal processing Python dependencies
COPY docker/requirements-mqtt.txt /app/docker/requirements-mqtt.txt
RUN pip install --no-cache-dir -r /app/docker/requirements-mqtt.txt

# Copy application source code
COPY ml-service /app/ml-service
COPY frontend /app/frontend

# Ensure runtime telemetry data directory exists
RUN mkdir -p /app/frontend/data

CMD ["python", "ml-service/src/mqtt_dashboard_bridge.py"]
