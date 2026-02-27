FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
# Build context is repo root, so path is backend/requirements.txt
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from backend/
COPY backend/ .

# Create __init__.py files for packages
RUN touch agents/__init__.py router/__init__.py memory/__init__.py \
         api/__init__.py db/__init__.py core/__init__.py

# Run as non-root for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Railway injects $PORT — falls back to 8000 for local dev
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
