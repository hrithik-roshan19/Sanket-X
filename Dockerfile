# syntax=docker/dockerfile:1
#
# Sanket - one image, one process: FastAPI serves both the API and the built dashboard.
#
# The alternative (two Render services, or a static site calling a separate API) needs CORS
# and burns two free instances for one product; one origin needs neither.
#
# What this image deliberately does NOT do is train. A full retrain peaks around 2 GB and
# the free tier gives 512 MB, so the model is built on a 16 GB GitHub Actions runner and
# pulled in below as a release asset. See .github/workflows/refresh-data.yml.

# ---------------------------------------------------------------- stage 1: the dashboard
FROM node:20-alpine AS web
WORKDIR /web

# package files first so a source-only edit does not re-resolve the dependency tree
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Both are compile-time constants in the bundle, not runtime lookups:
#  - no API base => same-origin, because this image serves the API itself
#  - upload off  => the dropzone would trigger a retrain this box cannot survive
ENV VITE_ENABLE_UPLOAD=false
RUN npm run build


# ------------------------------------------------------------------- stage 2: the server
FROM python:3.12-slim AS app
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl for the release asset below; ca-certificates so HTTPS works at all
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Core requirements only. requirements-live.txt (eccodes/cfgrib/xarray) is for decoding
# fresh GRIB2 and is CI's job, not this container's - leaving it out keeps the image small
# and avoids the eccodes-on-slim mess entirely.
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
# main.py mounts this directory only if it exists, so local dev (where it does not) is
# untouched and keeps using the Vite dev server.
COPY --from=web /web/dist ./app/static

# The trained model, the canonical parquet store and the metadata db are build inputs
# rather than source: they are regenerated every cycle and would otherwise be a 17 MB
# commit each time. CI publishes them to a fixed release tag.
#
# The fetch itself happens in the ENTRYPOINT, not here. Docker caches a RUN layer on its
# command string, and this URL is deliberately constant - so a rebuild triggered by a
# *data* refresh (the 6-hourly job, which changes no code) would reuse the cached download
# and redeploy the same stale artifact while reporting success. Fetching at startup also
# means a plain restart picks up fresh data with no rebuild.
#
# ENV rather than ARG: the entrypoint needs it at runtime, and ARG does not survive
# into the running container.
ENV DATA_ASSET_URL="https://github.com/Bhushan2318/sih-main/releases/download/data-latest/sanket-data.tar.gz"

# A build-time copy as a genuine fallback, so a GitHub outage at boot degrades the site
# to stale data rather than to no data. This layer IS cached and so may be old - that is
# fine for a fallback, and the entrypoint overwrites it with the current artifact on every
# start. Never fatal: an image that will not build is worse than one serving last week's
# cycle, and the app reports having no model honestly if both fetches fail.
RUN curl -fsSL --retry 3 --retry-delay 2 -o /tmp/data.tar.gz "$DATA_ASSET_URL" \
      && tar -xzf /tmp/data.tar.gz -C /app && rm /tmp/data.tar.gz \
      && echo "baked fallback data into the image" \
    || echo "WARNING: no fallback data baked in; the entrypoint fetch will have to work"

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
