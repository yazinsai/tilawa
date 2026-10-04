# Stage 1: Build frontend + server
FROM node:22-slim AS builder
# Mirror the repo layout so the demo's `@tilawa/core` alias
# (../../packages/core/src/index.ts, resolved by Vite + tsconfig paths)
# points at real files. Core is pure TS with no runtime deps of its own.
WORKDIR /app/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY packages/core /app/packages/core
COPY web/frontend/ .
RUN npm run build
RUN npm run build:server

# Stage 2: Runtime
FROM node:22-slim
WORKDIR /app

# Copy built frontend
COPY --from=builder /app/web/frontend/dist ./dist
COPY --from=builder /app/web/frontend/dist-server ./dist-server

# Copy package files for production deps only
COPY --from=builder /app/web/frontend/package.json /app/web/frontend/package-lock.json ./
RUN npm ci --omit=dev

# Models are gitignored / LFS-skipped on Dokku, so materialize them from
# immutable GitHub release assets.
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p dist/models \
 && curl -fL -o dist/models/zipformer_a0w_ep1_a05.int8.onnx \
    https://github.com/yazinsai/tilawa/releases/download/zipformer-a0w-ep1-a0.5/zipformer_a0w_ep1_a05.int8.onnx \
 && curl -fL -o dist/models/zipformer_a0w_ep1_a05.io.json \
    https://github.com/yazinsai/tilawa/releases/download/zipformer-a0w-ep1-a0.5/zipformer_a0w_ep1_a05.io.json \
 && curl -fL -o dist/zipformer_quran.json \
    https://github.com/yazinsai/tilawa/releases/download/v0.3.0/zipformer_quran.json

# Create storage directory
RUN mkdir -p /app/storage/reports

EXPOSE 5000
CMD ["node", "dist-server/index.mjs"]
