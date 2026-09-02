# Use slim Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for C compilation (e.g. rapidfuzz/faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy project files
COPY backend /app/backend
COPY data /app/data
COPY frontend /app/frontend

# Set Python path to backend directory
ENV PYTHONPATH=/app/backend

# Expose FastAPI backend port
EXPOSE 8000

# Set default working directory to backend for execution
WORKDIR /app/backend

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
