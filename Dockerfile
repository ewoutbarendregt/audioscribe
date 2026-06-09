# Use Python slim image
FROM python:3.11-slim

# Install ffmpeg for audio processing, curl for the container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (explicit list avoids accidentally baking in secrets)
COPY main.py transcriber.py ./
COPY static/ static/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
