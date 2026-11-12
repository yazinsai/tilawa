import { readFileSync } from "node:fs";
import { createTilawa } from "../src/node.js";
import type { NodeTilawaOptions } from "../src/node.js";
import { readWav } from "../src/runtime/node.js";

export const tinyDir = new URL("./fixtures/tiny", import.meta.url).pathname;
export const wavPath = new URL("./fixtures/tlog-clean-00010-630.wav", import.meta.url).pathname;
export const wav = () => readWav(wavPath);
export const asset = <T>(p: string): T => JSON.parse(readFileSync(new URL(`../assets/${p}`, import.meta.url), "utf8")) as T;

/** The public Node entry on the tiny fixture model; nothing is downloaded. */
export const tinyTilawa = (o: NodeTilawaOptions = {}) => createTilawa({ modelDir: tinyDir, ...o });
