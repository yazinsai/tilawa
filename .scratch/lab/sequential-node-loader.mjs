/** Resolve source .js imports to their TypeScript sources for dependency-free smoke assertions only. */
import { existsSync } from 'node:fs';
export function resolve(specifier, context, next) {
  if (context.parentURL?.startsWith('file:') && specifier.startsWith('.') && specifier.endsWith('.js')) {
    const source = new URL(specifier.slice(0, -3) + '.ts', context.parentURL);
    if (existsSync(source)) return { url: source.href, shortCircuit: true };
  }
  return next(specifier, context);
}
