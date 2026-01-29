# Dockerfile for ECS Worker
# Production-grade container for running analysis jobs

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/worker/requirements.txt /app/requirements.txt
COPY backend/shared/ /app/backend/shared/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY backend/worker/ /app/backend/worker/

# Copy business logic modules (existing analysis code)
COPY dsaitekika.py /app/
COPY linear_analysis_module.py /app/
COPY 01_model_builder.py /app/
COPY 02_prediction.py /app/
COPY 03_pareto_analyzer.py /app/
# Add other necessary modules

# Create temp directory
RUN mkdir -p /tmp/zerosec

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Run worker
CMD ["python", "-m", "backend.worker.worker"]
