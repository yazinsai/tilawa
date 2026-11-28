/**
 * The engine commit this package ships (scripts/sync-engine.mjs) and is parity-tested against: a commit on this
 * repo's main (packages/core, packages/sdk). Bumping it is the deliberate step that promotes engine changes into
 * the SDK. REPO is recorded in src/engine/SOURCE.json (shipped in dist); it names where the pinned commit was
 * first published and becomes "github.com/yazinsai/tilawa" at the next promotion.
 */
export const REPO = "github.com/yazinsai/tilawa";
export const COMMIT = "f6a841bc8d74fc1c346cdba584d6c76c2dc2515d";
