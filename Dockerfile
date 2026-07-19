# grid_agent application image.
# Small, boring, reproducible: slim Python base, deps cached in their own
# layer, no build tools needed (psycopg ships binary wheels).
FROM python:3.13-slim

WORKDIR /app

# Dependency layer — invalidated only when pyproject.toml changes.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY frontend/ frontend/

# The package is installed in site-packages, so config.py cannot infer the
# root that holds data/, frontend/ and traces/ — point it at /app.
ENV GRID_AGENT_ROOT=/app

# The dataset is generated at container start (deterministic seed: every
# container produces the identical file), then the API starts. Sessions
# themselves live in PostgreSQL, so containers are disposable.
EXPOSE 8000
CMD ["sh", "-c", "python -m grid_agent.datagen && \
     uvicorn grid_agent.api:app --host 0.0.0.0 --port 8000"]
