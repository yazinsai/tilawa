# Build from the repo root: docker build -f collect/Dockerfile -t tilawa-collect .
FROM node:22-alpine
WORKDIR /app
COPY collect/package.json collect/prompts.mjs collect/server.mjs ./
COPY collect/public ./public
COPY lab/data/quran.json ./quran.json
ENV PORT=8787 DATA_DIR=/data QURAN_JSON=/app/quran.json
VOLUME /data
EXPOSE 8787
USER node
CMD ["node", "server.mjs"]
