# Stage 1: Build frontend + server
FROM node:22-slim AS builder
WORKDIR /app
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ .
RUN npm run build
RUN npm run build:server

# Stage 2: Runtime
FROM node:22-slim
WORKDIR /app

# Copy built frontend
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/dist-server ./dist-server

# Copy package files for production deps only
COPY --from=builder /app/package.json /app/package-lock.json ./
RUN npm ci --omit=dev

# Download ONNX model
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -L -o dist/fastconformer_phoneme_q8.onnx \
    https://github.com/yazinsai/offline-tarteel/releases/download/v0.1.0/fastconformer_phoneme_q8.onnx

# Create storage directory
RUN mkdir -p /app/storage/reports

EXPOSE 5000
CMD ["node", "dist-server/index.mjs"]
