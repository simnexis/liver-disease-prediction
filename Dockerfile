# ==============================================================================
# Multi-stage production Dockerfile for Liver Risk ML Service & Web Application
# ==============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, trained models, results, and configuration
COPY src/ /app/src/
COPY models/ /app/models/
COPY results/ /app/results/
COPY reports/ /app/reports/
COPY data/ /app/data/
COPY api.py app.py predict.py ./

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Default command: run FastAPI production server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
