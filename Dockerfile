# grid_agent application image.
# Small, boring, reproducible: slim Python base, deps cached in their own
# layer, no build tools needed (psycopg ships binary wheels).
FROM python:3.13-slim

WORKDIR /app

# Dependency layer — invalidated only when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and assets.
COPY src/ src/
COPY scripts/ scripts/
COPY frontend/ frontend/

# The dataset is generated at container start (deterministic seed: every
# container produces the identical file), then the API starts. Sessions
# themselves live in PostgreSQL, so containers are disposable.
EXPOSE 8000
CMD ["sh", "-c", "python scripts/generate_data.py && \
     uvicorn grid_agent.api:app --app-dir src --host 0.0.0.0 --port 8000"]
