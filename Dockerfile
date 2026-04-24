FROM python:3.12-slim

# Security: run as non-root user
RUN groupadd -r tradeflow && useradd -r -g tradeflow tradeflow

WORKDIR /app

# Install system-level build dependencies required for C-extension packages
# (pandas, numpy, lxml all require gcc on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.yaml .
COPY .streamlit/ ./.streamlit/
COPY app/ ./app/

# Create data directory for SQLite persistence
RUN mkdir -p /app/data && chown -R tradeflow:tradeflow /app

# Switch to non-root user
USER tradeflow

# Expose FastAPI port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/api/health')" || exit 1

# Start FastAPI
CMD ["uvicorn", "app.webui.server:app", "--host", "0.0.0.0", "--port", "8501"]
