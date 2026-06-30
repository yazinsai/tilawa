# Stage 1: Build frontend
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
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
RUN curl -fL -o dist/fastconformer_full_mixed.onnx \
    https://github.com/yazinsai/tilawa/releases/download/v0.2.0/fastconformer_full_mixed.onnx

# Create storage directory
RUN mkdir -p /app/storage/reports

EXPOSE 5000
CMD ["node", "dist-server/index.mjs"]
