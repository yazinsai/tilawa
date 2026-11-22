/**
 * The engine commit this package ships (scripts/sync-engine.mjs) and is parity-tested against: a commit on this
 * repo's main (packages/core, packages/sdk). Bumping it is the deliberate step that promotes engine changes into
 * the SDK. REPO is recorded in src/engine/SOURCE.json (shipped in dist); it names where the pinned commit was
 * first published and becomes "github.com/yazinsai/tilawa" at the next promotion.
 */
export const REPO = "github.com/yazinsai/tilawa-reborn";
export const COMMIT = "8cb52cb4af3a6697d600a6f0292c86e17956a246";
