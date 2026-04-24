FROM python:3.11-slim

# Security: run as non-root user
RUN groupadd -r tradeflow && useradd -r -g tradeflow tradeflow

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.yaml .
COPY app/ ./app/

# Create data directory for SQLite persistence
RUN mkdir -p /app/data && chown -R tradeflow:tradeflow /app

# Switch to non-root user
USER tradeflow

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Start Streamlit
CMD ["streamlit", "run", "app/webui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
