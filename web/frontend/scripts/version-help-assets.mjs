import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';

const root = new URL('../dist/help/', import.meta.url);
const names = ['app.js', 'i18n.js', 'style.css', 'icons.svg', 'social-preview.png'];
const hash = createHash('sha256');
for (const name of names) hash.update(await readFile(new URL(name, root)));
const version = hash.digest('hex').slice(0, 12);
for (const name of ['index.html', 'app.js']) {
  let text = await readFile(new URL(name, root), 'utf8');
  for (const asset of names) text = text.replaceAll(asset, `${asset}?v=${version}`);
  await writeFile(new URL(name, root), text);
}
console.log(`Versioned /help assets: ${version}`);
