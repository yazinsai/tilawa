/** Nearest-rank percentile. */
export const percentile = (xs: number[], p: number): number => {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)] ?? NaN;
};

export interface RunTimings {
  audioSec: number;
  chunkMs: number[];
  loadMs: number;
}

export interface RunSummary {
  rtf: number;
  realTime: boolean;
  p50ChunkMs: number;
  p95ChunkMs: number;
  maxChunkMs: number;
  chunks: number;
  loadMs: number;
}

export const summarise = ({ audioSec, chunkMs, loadMs }: RunTimings): RunSummary => {
  const rtf = chunkMs.reduce((a, b) => a + b, 0) / 1000 / audioSec;
  return {
    rtf: Math.round(rtf * 1e4) / 1e4,
    realTime: rtf < 1,
    p50ChunkMs: percentile(chunkMs, 50),
    p95ChunkMs: percentile(chunkMs, 95),
    maxChunkMs: Math.max(...chunkMs),
    chunks: chunkMs.length,
    loadMs,
  };
};
