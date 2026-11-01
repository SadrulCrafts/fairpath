# syntax=docker/dockerfile:1.7
# ---- 1. build the dashboard (static export) ----
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
ARG NEXT_PUBLIC_REPO_URL=""
ENV NEXT_PUBLIC_REPO_URL=${NEXT_PUBLIC_REPO_URL} NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- 2. Python runtime: API + static files ----
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 FAIRPATH_LOG_LEVEL=INFO
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY reports/ reports/
COPY data/synthetic/ data/synthetic/
COPY --from=web /web/out web/out
ENV FAIRPATH_WEB_DIR=/app/web/out FAIRPATH_ARTIFACTS=/app/artifacts FAIRPATH_REPORTS=/app/reports
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health').status==200 else 1)"
CMD ["uvicorn", "fairpath.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers"]
