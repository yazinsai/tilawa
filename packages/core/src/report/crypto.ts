/** WebCrypto helpers: browsers (secure contexts), Node 20+ and React Native with a WebCrypto polyfill. */

function webCrypto(): Crypto {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (!c) throw new Error("@tilawa/core/report needs WebCrypto (globalThis.crypto)");
  return c;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle = webCrypto().subtle;
  if (!subtle) throw new Error("@tilawa/core/report needs crypto.subtle (a secure context in browsers)");
  const digest = new Uint8Array(await subtle.digest("SHA-256", bytes as unknown as ArrayBuffer));
  let hex = "";
  for (const b of digest) hex += b.toString(16).padStart(2, "0");
  return hex;
}

export function uuidV4(): string {
  const c = webCrypto();
  if (typeof c.randomUUID === "function") return c.randomUUID();
  const b = c.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}
