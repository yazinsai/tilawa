// A fetch for sendReport that reports upload progress (fetch itself cannot). XHR underneath.
export function progressFetch(onProgress: (fraction: number) => void): typeof fetch {
  return ((input: RequestInfo | URL, init: RequestInit = {}) =>
    new Promise<Response>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open(init.method ?? "GET", typeof input === "string" || input instanceof URL ? String(input) : input.url);
      const headers = new Headers(init.headers);
      headers.forEach((v, k) => xhr.setRequestHeader(k, v));
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        const status = xhr.status;
        const nullBody = status === 204 || status === 205 || status === 304;
        resolve(new Response(nullBody ? null : xhr.responseText, { status: status || 500, statusText: xhr.statusText }));
      };
      xhr.onerror = () => reject(new TypeError("Network request failed"));
      xhr.ontimeout = () => reject(new TypeError("Network request timed out"));
      xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
      if (init.signal) {
        if (init.signal.aborted) return xhr.abort();
        init.signal.addEventListener("abort", () => xhr.abort(), { once: true });
      }
      xhr.send((init.body ?? null) as XMLHttpRequestBodyInit | null);
    })) as typeof fetch;
}
