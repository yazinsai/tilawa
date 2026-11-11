/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where the V1 assets (models, reference, ORT runtime) are served. Default: same-origin `/v1/`. */
  readonly VITE_TILAWA_ASSET_BASE?: string;
}
