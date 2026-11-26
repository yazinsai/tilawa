import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { STATIC_AR, tr } from "../src/lib/i18n";

it("has Arabic for every key index.html marks for translation", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const keys = [...html.matchAll(/data-i18n(?:-label)?="([^"]+)"/g)].map((m) => m[1]);
  expect(keys.length).toBeGreaterThan(40);
  expect(keys.filter((k) => !STATIC_AR[k])).toEqual([]);
});

it("fills the placeholders in both languages", () => {
  expect(tr("sumMany", { n: 3 }, "en")).toBe("3 words are worth a second look.");
  expect(tr("strayHead", { ref: "36:2" }, "ar")).toBe("خارج المقطع المحدّد · 36:2");
  expect(tr("movedBody", { from: "A", to: "B" }, "ar")).not.toContain("{");
});
