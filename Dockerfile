# tilawa.dev: the V1 demo (/), /help, the server, and the immutable /v1/ assets.
#
#   docker build --build-arg TILAWA_ASSETS_URL=<source> -t tilawa-web .
#   docker run -p 5000:5000 -v tilawa-storage:/app/storage tilawa-web
#
# TILAWA_ASSETS_URL is where web/frontend/scripts/prepare-v1-assets.mjs downloads the models (see that script):
#   - Default: the immutable GitHub release models-v1, i.e.
#       https://github.com/yazinsai/tilawa/releases/download/models-v1/{id}-{version}.onnx
#     ({id}/{version} are filled in per registry tier: spliced-clean 1-int8, spliced-clean-lite 1-int8).
#   - Or any mirror of the /v1/ layout, e.g. the live https://tilawa.dev/v1/ or a local
#       (cd web/frontend && npm run build && npm run assets:v1 -- --models-dir <exports>
#        && python3 -m http.server 8000 -d dist/v1)
#     with --build-arg TILAWA_ASSETS_URL=http://host.docker.internal:8000/
#   - Or "none": the image then has no models. Mount a prepared tree and point ASSETS_V1_DIR at it, e.g.
#       docker run -v $PWD/web/frontend/dist/v1:/srv/v1:ro -e ASSETS_V1_DIR=/srv/v1 ...
# Every model is checked against the sha256 in packages/core's registry; a mismatch fails the build.

# Stage 1: build
FROM node:22-slim AS builder

# 1. @tilawa/core first: the frontend depends on its built dist (file:../../packages/core), so the repo layout
#    is mirrored under /app.
WORKDIR /app/packages/core
COPY packages/core/package.json packages/core/package-lock.json ./
RUN npm ci
COPY packages/core/ ./
RUN npm run build

# 2. The frontend (demo + /help) against it, then the server bundle.
WORKDIR /app/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ ./
RUN npm run build && npm run build:server

# 3. The immutable /v1/ tree (models, reference, onnxruntime-web runtime for the locked version, manifest.json).
#    After `vite build`, which empties dist/.
ARG TILAWA_ASSETS_URL=https://github.com/yazinsai/tilawa/releases/download/models-v1/{id}-{version}.onnx
RUN if [ -z "$TILAWA_ASSETS_URL" ]; then \
      echo "Set --build-arg TILAWA_ASSETS_URL=<url> (or =none to mount the assets at runtime)" >&2; exit 1; \
    elif [ "$TILAWA_ASSETS_URL" = "none" ]; then \
      echo "TILAWA_ASSETS_URL=none: no /v1 assets in the image"; \
    else \
      node scripts/prepare-v1-assets.mjs --models-url "$TILAWA_ASSETS_URL"; \
    fi

# 4. Production dependencies only, for the runtime stage.
RUN npm prune --omit=dev && rm -rf node_modules/.cache

# The 0.x demo's model files (zipformer_*.onnx, zipformer_quran.json) used to be curled into dist/ here. The V1
# demo doesn't use them, so they're gone. Old @tilawa/core 0.x installs download them from GitHub releases
# directly, and those releases stay up.

# Stage 2: runtime
FROM node:22-slim
WORKDIR /app

# The server bundle's only runtime dependencies (hono, @hono/node-server); everything else is bundled. They are
# pruned in the builder: `npm ci` here would fail on the lockfile's file:../../packages/core link (a dev dependency).
COPY web/frontend/package.json ./
COPY --from=builder /app/web/frontend/node_modules ./node_modules

COPY --from=builder /app/web/frontend/dist ./dist
COPY --from=builder /app/web/frontend/dist-server ./dist-server

# Static, immutable assets live under /app/dist (dist/v1 is served only by the /v1 route). User data (reports,
# /help submissions) lives under /app/storage, which is never served statically: mount a volume there.
ENV ASSETS_V1_DIR=/app/dist/v1
RUN mkdir -p /app/storage/reports /app/storage/help

EXPOSE 5000
CMD ["node", "dist-server/index.mjs"]
