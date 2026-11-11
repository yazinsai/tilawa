/** RFC 4122 v4. `crypto.randomUUID` is missing in insecure browser contexts and some React Native runtimes. */
export function uuid(): string {
  const c = globalThis.crypto;
  if (typeof c?.randomUUID === "function") {
    try {
      return c.randomUUID();
    } catch {
      // Insecure context: fall through.
    }
  }
  const b = new Uint8Array(16);
  if (typeof c?.getRandomValues === "function") c.getRandomValues(b);
  else for (let i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

/** Hex sha256, or null where WebCrypto is unavailable (plain-http pages, React Native). */
export async function sha256Hex(bytes: Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const d = new Uint8Array(await subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>));
  return Array.from(d, (x) => x.toString(16).padStart(2, "0")).join("");
}
